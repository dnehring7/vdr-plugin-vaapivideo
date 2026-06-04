// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file audio.h
 * @brief ALSA audio sink: IEC61937 passthrough and decoded-PCM fallback
 */

#ifndef VDR_VAAPIVIDEO_AUDIO_H
#define VDR_VAAPIVIDEO_AUDIO_H

#include "caps.h"
#include "common.h"
#include "stream.h"

// Platform
#include <alsa/asoundlib.h>
#include <climits>

// VDR
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <vdr/thread.h>
#include <vdr/tools.h>
#pragma GCC diagnostic pop

// FFmpeg (forward declarations for IEC61937 spdif muxer)
struct AVFormatContext;
struct AVIOContext;

// ============================================================================
// === STRUCTURES ===
// ============================================================================

// AudioStreamInfo (stream.h) is the single source of truth for stream identity
// and format parameters, shared by the PES path and future mediaplayer path.
// AudioStreamParams is a local alias kept for call-site readability in audio.cpp.
using AudioStreamParams = AudioStreamInfo;

// ============================================================================
// === AUDIO PROCESSOR CLASS ===
// ============================================================================

/**
 * @class cAudioProcessor
 * @brief Threaded ALSA audio renderer with IEC61937 passthrough and FFmpeg PCM decode fallback.
 *
 * A background thread (Action()) dequeues compressed audio packets, decodes them to
 * S16LE PCM via FFmpeg, and writes the result to an ALSA PCM device. When the HDMI
 * sink advertises support for the codec's IEC61937 framing (via ELD or PassthroughMode::On),
 * the compressed bitstream is passed through directly without decoding.
 *
 * Thread safety: all public methods are safe to call from any thread.
 * See audio.cpp file header for the full threading model and synchronization design.
 */
class cAudioProcessor : public cThread {
  public:
    cAudioProcessor();
    ~cAudioProcessor() noexcept override;
    cAudioProcessor(const cAudioProcessor &) = delete;
    cAudioProcessor(cAudioProcessor &&) noexcept = delete;
    auto operator=(const cAudioProcessor &) -> cAudioProcessor & = delete;
    auto operator=(cAudioProcessor &&) noexcept -> cAudioProcessor & = delete;

    // ========================================================================
    // === PUBLIC API ===
    // ========================================================================
    auto Clear() -> void; ///< Drops buffered audio and resets the playback clock; call on seek or channel change
    auto DropOutput(bool pauseClock = false)
        -> void; ///< Silence playback fast: snd_pcm_drop + decode-queue flush, but DO NOT reset the
                 ///< playback clock. Use from Mute/Freeze/SetTrickSpeed -- those are stream-preserving
                 ///< events; a full Clear() would null GetClock(), force the decoder into freerun, and
                 ///< cause a display underrun downstream. On position changes (FF/REW resume) the
                 ///< Action()-thread >5s-jump guard re-anchors the clock automatically.
                 ///< pauseClock=true pins GetClock() at the current playbackPts (no wall-clock
                 ///< extrapolation) until the next ALSA write re-anchors it. Used by Freeze() so a
                 ///< short pause does not advance the master clock through silence and silently shift
                 ///< the preserved jitterBuf head on resume.
    auto Decode(const uint8_t *data, size_t size, int64_t pts)
        -> void; ///< Parses raw PES payload into access units and enqueues them for decoding/passthrough
    [[nodiscard]] auto GetClock() const noexcept
        -> int64_t; ///< Estimated PTS at the DAC output in 90 kHz ticks; AV_NOPTS_VALUE when stale or uninitialized
    [[nodiscard]] auto Initialize(std::string_view alsaDevice)
        -> bool; ///< Opens the ALSA device and starts the processing thread; idempotent for the same device name
    [[nodiscard]] auto IsInitialized() const noexcept
        -> bool; ///< True after Initialize() succeeds and before Shutdown()
    [[nodiscard]] auto IsPassthrough() const noexcept
        -> bool;                                    ///< True when the device is currently in IEC61937 passthrough mode
    [[nodiscard]] auto IsQueueFull() const -> bool; ///< True when the packet queue has reached AUDIO_QUEUE_CAPACITY
    [[nodiscard]] auto GetQueueSize() const -> size_t; ///< Current number of packets in the decode queue
    [[nodiscard]] auto GetPendingWorkSize() const
        -> size_t; ///< Queue + consumer-held packet + unplayed ALSA tail. The mediaplayer EOS drain needs this, not
                   ///< GetQueueSize(): the queue hits 0 while the last packet is still in flight / queued in ALSA.
    [[nodiscard]] auto OpenCodec(AVCodecID codecId, int sampleRate, int channels)
        -> bool; ///< Convenience wrapper around SetStreamParams() for simple codec+rate+channels changes
    [[nodiscard]] auto OpenCodecWithInfo(const AudioStreamInfo &info)
        -> bool; ///< Mediaplayer path: forwards AudioStreamInfo (carrying extradata) to SetStreamParams().
                 ///< Caller's extradata buffer is deep-copied inside AdoptStreamParams; safe to free after return.
    [[nodiscard]] auto EnqueuePacket(const AVPacket *packet)
        -> bool; ///< Mediaplayer path: clones a pre-demuxed AU onto the audio queue. Returns false on
                 ///< capacity overflow; caller is expected to back off and retry.
    auto SetVolume(int vol)
        -> void;             ///< PCM volume in range [0, 255]; 0 = mute, 255 = full scale; skipped for passthrough
    auto Shutdown() -> void; ///< Stops the processing thread and closes ALSA + decoder + parser. Idempotent;
                             ///< called by the destructor. Initialize()'s device-swap path calls CloseDevice()
                             ///< directly instead, keeping the thread alive across the swap.

  protected:
    // ========================================================================
    // === THREAD ===
    // ========================================================================
    auto Action() -> void override; ///< Processing thread: dequeues packets and calls DecodeToPcm() or WrapIec61937()

  private:
    // ========================================================================
    // === INTERNAL METHODS ===
    // ========================================================================
    [[nodiscard]] auto CanPassthrough(AVCodecID codecId) const
        -> bool; ///< True when IEC61937 passthrough should be used; honors PassthroughMode
                 ///< (Auto = ELD-driven, On = any wrappable codec regardless of ELD, Off = never)
    [[nodiscard]] static auto CodecWrappable(AVCodecID codecId)
        -> bool; ///< True iff the spdif muxer can frame this codec as IEC61937; delegates to IsPassthroughCapable()
                 ///< (stream.h). AudioSinkCaps::Supports() must mirror this set for Auto mode.
    [[nodiscard]] auto SinkSupports(AVCodecID codecId) const
        -> bool;                ///< True iff the HDMI sink's ELD advertises IEC61937 support for this codec
    auto CloseDevice() -> void; ///< Closes ALSA + decoder + spdif muxer and resets all device state.
                                ///< Does NOT touch the processing thread; called from Shutdown() (thread exited)
                                ///< and from Initialize() during a device swap (thread kept alive).
    [[nodiscard]] auto SetStreamParams(const AudioStreamParams &params)
        -> bool; ///< Reconfigures ALSA and the FFmpeg decoder when codec, rate, or passthrough mode changes.
                 ///< Returns false if the pipeline could not be established.
    auto CloseDecoder() -> void; ///< Spins until in-flight DecodeToPcm() callers finish, then frees decoder + parser
    auto DrainPacketQueue() -> void;   ///< Pops and frees every queued packet. Caller must hold mutex.
    auto FlushDecoderState() -> void;  ///< avcodec_flush_buffers + swr teardown + error counter reset
    auto RecreateParser() -> void;     ///< Close + re-init parser for the current codec; caller holds mutex.
    auto ResetPlaybackClock() -> void; ///< Zeroes playbackPts, lastClockUpdateMs, pcmNextPts under the seqlock.
                                       ///< Caller must hold mutex (single-writer invariant).
    [[nodiscard]] auto ComputeAlsaRate(AVCodecID codecId, unsigned streamRate, bool passthrough) const
        -> unsigned; ///< Returns ALSA carrier rate: 4x streamRate for DD+/AC-4/MPEG-H passthrough, 1x otherwise
    [[nodiscard]] auto ConfigureAlsaParams(snd_pcm_t *handle, snd_pcm_format_t format, unsigned channels, unsigned rate,
                                           bool allowResample)
        -> bool; ///< Applies hardware and software ALSA parameters to an open PCM handle;
                 ///< allowResample=false is required for IEC61937 passthrough (rate must be exact)
    [[nodiscard]] auto DecodeToPcm(std::span<const uint8_t> data, int64_t pts, uint32_t expectedGeneration)
        -> bool; ///< Decodes one compressed packet and forwards S16LE PCM to WritePcmToAlsa().
                 ///< expectedGeneration is clearGeneration captured at dequeue; mismatch means the
                 ///< packet is from a previous codec era and is dropped to avoid an INVALIDDATA cascade.
    [[nodiscard]] auto OpenAlsaDevice()
        -> bool; ///< Opens alsaDeviceName in passthrough or PCM mode; falls back to PCM if passthrough fails
    auto OpenDecoder() -> void; ///< Allocates and opens the FFmpeg decoder + parser for streamParams
    [[nodiscard]] auto OpenSpdifMuxer(AVCodecID codecId, int sampleRate)
        -> bool; ///< Opens the FFmpeg spdif muxer for IEC61937 burst framing; called on passthrough open
    auto CloseSpdifMuxer() -> void; ///< Closes the spdif muxer and frees the AVIO buffer
    [[nodiscard]] auto WrapIec61937(const uint8_t *data, int size)
        -> std::span<const uint8_t>; ///< Wraps one compressed frame into an IEC61937 burst; result valid until next
                                     ///< call
    auto ProbeSinkCaps()
        -> void; ///< Reads the HDMI ELD via ALSA control interface and populates sinkCaps; cached per device name
    auto SetIec958NonAudio(bool enable) const
        -> void; ///< Sets/clears IEC 60958-3 AES0 bit 1 ("non-audio") on the HDMI mixer control.
                 ///< Must be set before passthrough writes; cleared on PCM open to prevent noise on the receiver.
    [[nodiscard]] auto WritePcmToAlsa(std::span<const uint8_t> data, int64_t startPts90k, unsigned frames,
                                      uint32_t expectedGeneration)
        -> bool; ///< Writes PCM/burst to ALSA and publishes the updated playbackPts under the seqlock.
                 ///< Gated by expectedGeneration: old-era writes are silently skipped so they cannot
                 ///< resurrect a stale playbackPts after a Clear()/codec swap.
    [[nodiscard]] auto WriteToAlsa(std::span<const uint8_t> data)
        -> bool; ///< Raw ALSA write loop: volume scaling, frame alignment, four-tier error recovery

    // ========================================================================
    // === ALSA DEVICE ===
    // ========================================================================
    int alsaCardId{-1};                             ///< ALSA card number; cached by ProbeSinkCaps()
    std::atomic<unsigned> alsaChannels{0};          ///< Negotiated channel count
    std::string alsaDeviceName;                     ///< ALSA PCM device name (e.g. "plughw:0,3")
    std::atomic<int> alsaErrorCount{0};             ///< Consecutive snd_pcm_writei failures
    std::atomic<size_t> alsaFrameBytes{0};          ///< Bytes per interleaved frame
    snd_pcm_t *alsaHandle{nullptr};                 ///< Open PCM device handle; nullptr when closed
    unsigned alsaIec958CtlIndex{UINT_MAX};          ///< "IEC958 Playback Default" control index; UINT_MAX = unresolved
    std::atomic<bool> alsaPassthroughActive{false}; ///< True in IEC61937 passthrough mode
    std::atomic<unsigned> alsaSampleRate{0};        ///< Negotiated sample rate (Hz)

    // ========================================================================
    // === IEC61937 SPDIF MUXER ===
    // ========================================================================
    AVFormatContext *spdifMuxCtx{nullptr}; ///< FFmpeg spdif output context
    std::vector<uint8_t> spdifOutputBuf;   ///< IEC61937 burst bytes; valid until next WrapIec61937()

    // ========================================================================
    // === DECODER ===
    // ========================================================================
    int consecutiveDecodeErrors{};                               ///< Consecutive avcodec_send_packet failures
    std::unique_ptr<AVCodecContext, FreeAVCodecContext> decoder; ///< FFmpeg decoder context
    int decoderGracePackets{0};                                  ///< Packets to silently discard after (re)init
    std::atomic<int> decoderRefCount{0};                         ///< In-flight DecodeToPcm() callers
    std::atomic<bool> needsFlush{false};                         ///< Set by Clear(), consumed by DecodeToPcm()
    std::atomic<bool> parserNeedsReset{false}; ///< Action() flags INVALIDDATA cascade; Decode() recreates parserCtx
                                               ///< (cannot be done in Action() -- would deadlock CloseDecoder()).
    std::unique_ptr<AVCodecParserContext, FreeAVCodecParserContext> parserCtx; ///< AU-framing parser
    int swrChannels{};                                                         ///< swrCtx channel count
    SwrContext *swrCtx{nullptr};                                               ///< Conversion context to S16LE
    AVSampleFormat swrFormat{AV_SAMPLE_FMT_NONE};                              ///< swrCtx sample format

    // ========================================================================
    // === PACKET QUEUE ===
    // ========================================================================
    cCondVar packetCondition;                ///< Wakes Action() on enqueue
    std::atomic<bool> packetInFlight{false}; ///< Action() popped a packet but hasn't finished handing it to ALSA.
                                             ///< Closes the EOS-drain false-zero gap between pop and ALSA write.
    std::queue<AVPacket *> packetQueue;      ///< Compressed packets awaiting decode

    // ========================================================================
    // === PCM CLOCK ===
    // ========================================================================
    std::atomic<uint32_t> clearGeneration{0};          ///< Bumped on Clear()/codec swap; tags packet era
    std::atomic<uint32_t> clockSequence{0};            ///< Seqlock for (playbackPts, lastClockUpdateMs); single-writer
    std::atomic<uint64_t> lastClockUpdateMs{0};        ///< cTimeMs::Now() at last publish; 0 = never written
    std::atomic<int64_t> pcmNextPts{AV_NOPTS_VALUE};   ///< DVB-anchored 90 kHz PTS for next ALSA write
    std::atomic<int64_t> playbackPts{AV_NOPTS_VALUE};  ///< Estimated PTS at DAC output
    std::atomic<bool> outputDropped{false};            ///< snd_pcm_drop emptied ALSA but pcmNextPts/playbackPts were
                                                       ///< kept (Mute/Freeze/trick). Stops GetPendingWorkSize() from
                                                       ///< reading that preserved clock as a phantom ALSA tail.
                                                       ///< Cleared by ResetPlaybackClock() / next WritePcmToAlsa().
    mutable std::atomic<bool> clockStaleLogged{false}; ///< Edge-trigger flag for the GetClock() stale-age diagnostic;
                                                       ///< set when GetClock() first returns NOPTS due to age, cleared
                                                       ///< on the next valid read. Prevents log spam at 50 Hz polling.
    std::atomic<bool> clockPaused{false};              ///< Set by DropOutput(pauseClock=true) (Freeze()); cleared by
                                                       ///< Clear() / ResetPlaybackClock() / the next WritePcmToAlsa.
                                                       ///< While set, GetClock() returns playbackPts verbatim instead
                                                       ///< of extrapolating against wall-clock through ALSA silence.

    // ========================================================================
    // === SINK CAPABILITIES ===
    // ========================================================================
    AudioSinkCaps sinkCaps;     ///< Compressed-audio formats the HDMI sink advertises (from ELD)
    bool sinkCapsCached{false}; ///< True when sinkCaps was probed for sinkCapsDevice
    std::string sinkCapsDevice; ///< Device name for which sinkCaps was last probed; invalidated on device change

    // ========================================================================
    // === STREAM ===
    // ========================================================================
    AudioStreamParams streamParams;       ///< Active codec and format; extradata pointer (when non-null) always points
                                          ///< into storedExtradata -- never at caller memory (see AdoptStreamParams())
    std::vector<uint8_t> storedExtradata; ///< Owning copy of the active stream's extradata. AudioStreamInfo::extradata
                                          ///< is non-owning (see src/stream.h); the bytes are deep-copied here to avoid
                                          ///< a use-after-free when the caller frees its buffer between calls.

    auto AdoptStreamParams(const AudioStreamParams &params)
        -> void; ///< Deep-copies params.extradata into storedExtradata and rewires streamParams.extradata
                 ///< to the owned buffer. Caller must hold mutex.

    // ========================================================================
    // === THREAD CONTROL ===
    // ========================================================================
    std::atomic<bool> hasExited{false};    ///< Set by Action() just before it returns; Shutdown() polls this
    std::atomic<bool> initialized{false};  ///< True after Initialize() succeeds and before Shutdown()/CloseDevice()
    mutable std::unique_ptr<cMutex> mutex; ///< Serializes ALSA handle, packet queue, decoder state, and seqlock writes
    std::atomic<bool> stopping{
        false};                   ///< Signals Action() to exit; also set on fatal ALSA error to mark processor unusable
    std::atomic<int> volume{255}; ///< PCM volume scale applied in WriteToAlsa(); 0 = mute, 255 = unity

    // ========================================================================
    // === TIMING ===
    // ========================================================================
    cTimeMs lastDecodeErrorLog; ///< Rate-limits decode-error syslog to once per AUDIO_ERROR_LOG_INTERVAL_MS
    cTimeMs lastQueueWarn;      ///< Rate-limits "queue full" syslog to once per 500 ms
    cTimeMs lastReopenAttempt;  ///< Rate-limits ALSA device-reopen attempts to once per 1000 ms
};

#endif // VDR_VAAPIVIDEO_AUDIO_H
