// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file audio.h
 * @brief ALSA output with IEC61937 passthrough and PCM fallback
 */

#ifndef VDR_VAAPIVIDEO_AUDIO_H
#define VDR_VAAPIVIDEO_AUDIO_H

#include "common.h"

// Platform
#include <alsa/asoundlib.h>

// FFmpeg (forward declarations for IEC61937 spdif muxer)
struct AVFormatContext;
struct AVIOContext;

// ============================================================================
// === STRUCTURES ===
// ============================================================================

/// HDMI sink compressed-audio format capabilities decoded from the ELD Short Audio Descriptors.
struct HdmiSinkCaps {
    bool ac3{false};     ///< IEC61937 AC-3 (Dolby Digital) passthrough supported
    bool ac4{false};     ///< IEC61937 AC-4 passthrough supported
    bool dts{false};     ///< IEC61937 DTS passthrough supported
    bool dtshd{false};   ///< IEC61937 DTS-HD passthrough supported
    bool eac3{false};    ///< IEC61937 E-AC-3 (Dolby Digital Plus) passthrough supported
    bool mpegh3d{false}; ///< IEC61937 MPEG-H 3D Audio passthrough supported
    bool truehd{false};  ///< IEC61937 Dolby TrueHD passthrough supported
};

/// Codec identity and format parameters for a single audio elementary stream.
struct AudioStreamParams {
    int channels{};                      ///< Decoded output channel count
    AVCodecID codecId{AV_CODEC_ID_NONE}; ///< FFmpeg codec identifier
    const uint8_t *extradata{};          ///< Out-of-band codec configuration (e.g. AAC AudioSpecificConfig)
    int extradataSize{};                 ///< Byte length of extradata
    int sampleRate{};                    ///< Audio sample rate in Hz
};

// ============================================================================
// === AUDIO PROCESSOR CLASS ===
// ============================================================================

/**
 * @class cAudioProcessor
 * @brief Threaded ALSA audio renderer with IEC61937 passthrough and FFmpeg PCM decode fallback.
 *
 * A background thread dequeues compressed audio packets (enqueued via Decode()), decodes
 * them to S16LE stereo PCM via FFmpeg, and writes the result to an ALSA PCM device.
 * When the HDMI sink advertises support for the codec's IEC61937 framing, the compressed
 * bitstream is passed through directly without decoding.
 *
 * Thread safety: all public methods are safe to call from any thread.
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
    auto Decode(const uint8_t *data, size_t size, int64_t pts)
        -> void; ///< Parses raw PES payload into access units and enqueues them for decoding
    [[nodiscard]] auto GetClock() const noexcept
        -> int64_t; ///< Returns the PTS of the sample at the DAC output in 90 kHz ticks, or AV_NOPTS_VALUE
    [[nodiscard]] auto Initialize(std::string_view alsaDevice)
        -> bool; ///< Opens the ALSA device and starts the processing thread; idempotent for the same device
    [[nodiscard]] auto IsInitialized() const noexcept
        -> bool; ///< Returns true if the device is open and the processing thread is running
    [[nodiscard]] auto IsPassthrough() const noexcept
        -> bool; ///< Returns true if the device is currently in IEC61937 passthrough mode
    [[nodiscard]] auto IsQueueFull() const
        -> bool; ///< Returns true if the packet queue has reached AUDIO_QUEUE_CAPACITY
    [[nodiscard]] auto OpenCodec(AVCodecID codecId, int sampleRate, int channels)
        -> bool; ///< Convenience wrapper: configures codec, sample rate, and channel count in one call
    auto SetVolume(int vol) -> void; ///< Sets PCM playback volume in the range 0 (mute) to 255 (full scale)
    auto Stop() -> void;             ///< Signals the processing thread to exit, waits for it, then calls Shutdown()

  protected:
    // ========================================================================
    // === THREAD ===
    // ========================================================================
    auto Action() -> void override; ///< Processing thread main loop: dequeues and renders audio packets

  private:
    // ========================================================================
    // === INTERNAL METHODS ===
    // ========================================================================
    [[nodiscard]] auto CanPassthrough(AVCodecID codecId) const
        -> bool; ///< Returns true if sinkCaps advertises IEC61937 support for this codec
    auto SetStreamParams(const AudioStreamParams &params)
        -> void;                 ///< Reconfigures ALSA and the FFmpeg decoder when the codec or sample format changes
    auto Shutdown() -> void;     ///< Closes ALSA and frees codec resources; the processing thread continues running
    auto CloseDecoder() -> void; ///< Waits for in-flight decode calls, then frees the decoder and parser contexts
    auto FlushDecoderState() -> void; ///< Flushes FFmpeg decoder, resets swr context, clears error/grace counters
    [[nodiscard]] auto ComputeAlsaRate(AVCodecID codecId, unsigned streamRate, bool passthrough) const
        -> unsigned; ///< Returns the ALSA sample rate required for the given codec and passthrough mode
    [[nodiscard]] auto ConfigureAlsaParams(snd_pcm_t *handle, snd_pcm_format_t format, unsigned channels, unsigned rate,
                                           bool allowResample)
        -> bool; ///< Applies hardware and software ALSA parameters to an open PCM handle
    [[nodiscard]] auto DecodeToPcm(std::span<const uint8_t> data, int64_t pts)
        -> bool; ///< Decodes a compressed packet via FFmpeg and forwards the resulting S16LE PCM to WritePcmToAlsa()
    [[nodiscard]] auto EnqueuePacket(const AVPacket *packet)
        -> bool; ///< Clones the packet and appends it to the processing queue; returns false if full
    [[nodiscard]] auto OpenAlsaDevice()
        -> bool; ///< Opens the PCM device with passthrough or PCM parameters; falls back to PCM on failure
    auto OpenDecoder() -> void; ///< Allocates and opens the FFmpeg decoder and parser for the current stream parameters
    [[nodiscard]] auto OpenSpdifMuxer(AVCodecID codecId, int sampleRate)
        -> bool; ///< Opens the FFmpeg spdif muxer for IEC61937 burst framing; call when entering passthrough mode
    auto CloseSpdifMuxer() -> void; ///< Closes the spdif muxer and frees associated resources
    [[nodiscard]] auto WrapIec61937(const uint8_t *data, int size)
        -> std::span<const uint8_t>; ///< Wraps a compressed audio frame in an IEC61937 burst via the spdif muxer
    auto ProbeSinkCaps()
        -> void; ///< Reads the HDMI ELD via ALSA control interface and populates sinkCaps; result is cached per device
    [[nodiscard]] auto WritePcmToAlsa(std::span<const uint8_t> data, int64_t startPts90k, unsigned frames)
        -> bool; ///< Writes PCM to ALSA and advances the 90 kHz playback clock by the number of written frames
    [[nodiscard]] auto WriteToAlsa(std::span<const uint8_t> data)
        -> bool; ///< Raw ALSA write loop with volume scaling, frame alignment, and three-tier error recovery

    // ========================================================================
    // === ALSA DEVICE ===
    // ========================================================================
    unsigned alsaChannels{0};              ///< Configured hardware channel count
    std::string alsaDeviceName;            ///< ALSA PCM device name (e.g. "plughw:0,3")
    std::atomic<int> alsaErrorCount{0};    ///< Consecutive ALSA write failures since the last successful write
    std::atomic<size_t> alsaFrameBytes{0}; ///< Bytes per interleaved audio frame (channels x sample_size); atomic so
                                           ///<   WriteToAlsa() can read it without holding the mutex
    snd_pcm_t *alsaHandle{nullptr};        ///< ALSA PCM device handle; nullptr when closed
    bool alsaPassthroughActive{false};     ///< True when the device is open in IEC61937 passthrough mode
    unsigned alsaSampleRate{0};            ///< Hardware sample rate negotiated with ALSA (Hz)

    // ========================================================================
    // === IEC61937 SPDIF MUXER ===
    // ========================================================================
    AVFormatContext *spdifMuxCtx{nullptr}; ///< FFmpeg spdif output context for IEC61937 burst creation
    std::vector<uint8_t> spdifOutputBuf;   ///< Accumulates IEC61937 burst bytes from the spdif muxer write callback

    // ========================================================================
    // === DECODER ===
    // ========================================================================
    int consecutiveDecodeErrors{}; ///< Consecutive avcodec_send_packet failures for error recovery
    std::unique_ptr<AVCodecContext, FreeAVCodecContext> decoder{
        nullptr};                        ///< FFmpeg decoder context; nullptr when closed
    int decoderGracePackets{0};          ///< Remaining packets to silently discard after decoder reinit
    std::atomic<int> decoderRefCount{0}; ///< Count of threads currently inside DecodeToPcm()
    std::atomic<bool> needsFlush{false}; ///< Signals DecodeToPcm() to flush the decoder on next entry
    std::unique_ptr<AVCodecParserContext, FreeAVCodecParserContext>
        parserCtx;                                ///< FFmpeg bitstream parser for access-unit framing
    int swrChannels{};                            ///< Channel count for which swrCtx was last initialized
    SwrContext *swrCtx{nullptr};                  ///< libswresample context for format/channel conversion to S16LE
    AVSampleFormat swrFormat{AV_SAMPLE_FMT_NONE}; ///< Sample format for which swrCtx was last initialized

    // ========================================================================
    // === PACKET QUEUE ===
    // ========================================================================
    cCondVar packetCondition;           ///< Signals the processing thread when a new packet is enqueued
    std::queue<AVPacket *> packetQueue; ///< Owned compressed audio packets awaiting decode (freed via RAII on dequeue)

    // ========================================================================
    // === PCM CLOCK ===
    // ========================================================================
    std::atomic<uint32_t> clearGeneration{0}; ///< Bumped on Clear(); stale DecodeToPcm calls skip clock writes
    int64_t pcmNextPts{AV_NOPTS_VALUE};       ///< DVB-anchored 90 kHz PTS for the next ALSA write (PCM & passthrough)
    int64_t pcmQueueEndPts{AV_NOPTS_VALUE};   ///< 90 kHz PTS of the last sample written into the ALSA ring buffer
    std::atomic<int64_t> playbackPts{AV_NOPTS_VALUE}; ///< Cached DAC-output PTS (endPts minus ALSA delay at write time)

    // ========================================================================
    // === SINK CAPABILITIES ===
    // ========================================================================
    HdmiSinkCaps sinkCaps;      ///< Compressed-audio formats the HDMI sink can decode (from ELD)
    bool sinkCapsCached{false}; ///< True when sinkCaps has been read for sinkCapsDevice
    std::string sinkCapsDevice; ///< Device name for which sinkCaps was last probed

    // ========================================================================
    // === STREAM ===
    // ========================================================================
    AudioStreamParams streamParams; ///< Active codec and format parameters for the current stream

    // ========================================================================
    // === THREAD CONTROL ===
    // ========================================================================
    std::atomic<bool> hasExited{false};    ///< Set by the processing thread just before it returns
    std::atomic<bool> initialized{false};  ///< True after Initialize() succeeds and before Shutdown()
    mutable std::unique_ptr<cMutex> mutex; ///< Serializes access to ALSA handle, queue, and decoder state
    std::atomic<bool> stopping{false};     ///< Signals the processing thread to stop
    std::atomic<int> volume{255};          ///< PCM volume scale: 0 = mute, 255 = full scale

    // ========================================================================
    // === TIMING ===
    // ========================================================================
    cTimeMs lastDecodeErrorLog; ///< Rate-limits the decode-error warning to once per AUDIO_ERROR_LOG_INTERVAL_MS
    cTimeMs lastQueueWarn;      ///< Rate-limits the "queue full" warning to once per 500 ms
    cTimeMs lastReopenAttempt;  ///< Rate-limits device-reopen attempts to once per 1000 ms
};

#endif // VDR_VAAPIVIDEO_AUDIO_H
