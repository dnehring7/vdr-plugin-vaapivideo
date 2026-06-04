// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file audio.cpp
 * @brief ALSA audio sink: IEC61937 compressed passthrough with decoded-PCM fallback.
 *
 * Threading model:
 *   - Producer (VDR PES thread): Decode() -> parser -> EnqueuePacket() under `mutex`.
 *   - Consumer Action(): PCM path is lock-free via `decoderRefCount`+`clearGeneration`;
 *     passthrough holds `mutex` across WrapIec61937 + WritePcmToAlsa to gate against
 *     CloseSpdifMuxer (no refcount equivalent for the spdif muxer).
 *   - GetClock() is lock-free, seqlock-paired with WritePcmToAlsa().
 */

#include "audio.h"
#include "common.h"
#include "config.h"
#include "stream.h"

// ALSA
#include <alsa/asoundlib.h>

// C++ Standard Library
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <format>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// FFmpeg
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/codec.h>
#include <libavcodec/codec_id.h>
#include <libavcodec/defs.h>
#include <libavcodec/packet.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavutil/avutil.h>
#include <libavutil/channel_layout.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/mem.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
}
#pragma GCC diagnostic pop

// VDR
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <vdr/remux.h>
#include <vdr/thread.h>
#include <vdr/tools.h>
#pragma GCC diagnostic pop

// Pin the derived constant to VDR's authoritative definition.
static_assert(PTS_TICKS_PER_MS == PTSTICKS / 1000, "PTS_TICKS_PER_MS must equal VDR PTSTICKS / 1000");

// ============================================================================
// === CONSTANTS ===
// ============================================================================

// Shared AV cushion: the ring stays near-full in steady state, so the audio clock lags
// real time by ~this much. The video due-gate accumulates the matching frame count in
// jitterBuf so audio and video share one cushion and lip-sync is preserved. This is a
// floor; ALSA may negotiate slightly larger. See AVSYNC.md.
constexpr int AUDIO_ALSA_BUFFER_MS = 400;

constexpr int AUDIO_ALSA_ERROR_LIMIT = 5; ///< Consecutive snd_pcm_writei failures before device reopen

// After this age GetClock() returns AV_NOPTS_VALUE to force video freerun instead of
// drifting against a frozen audio clock (channel switch, dead ALSA device).
constexpr uint64_t AUDIO_CLOCK_STALE_MS = 1000;

constexpr int AUDIO_DECODER_DRAIN_TIMEOUT_MS =
    200; ///< CloseDecoder() spin-wait ceiling for in-flight DecodeToPcm() callers (ms)
constexpr int AUDIO_DECODER_ERROR_LIMIT = 50; ///< Consecutive avcodec_send_packet failures before flush + parser reset
constexpr int AUDIO_DECODER_GRACE_PACKETS =
    3; ///< Error logs suppressed after decoder (re)init; absorbs parser priming garbage on the first few frames
constexpr int AUDIO_ERROR_LOG_INTERVAL_MS = 2000; ///< Minimum interval between repeated decode-error log messages (ms)

// Sized to absorb the OSD-load + codec-prime burst at channel switch or VDR start
// without dropping (~10 s at AC-3 ~32 ms framing).
constexpr size_t AUDIO_QUEUE_CAPACITY = 300;

// ============================================================================
// === AUDIO PROCESSOR CLASS ===
// ============================================================================

cAudioProcessor::cAudioProcessor() : cThread("vaapivideo/audio"), mutex(std::make_unique<cMutex>()) {}

cAudioProcessor::~cAudioProcessor() noexcept {
    dsyslog("vaapivideo/audio: destroying (stopping=%d)", stopping.load(std::memory_order_relaxed));
    Shutdown();
}

// ============================================================================
// === PUBLIC API ===
// ============================================================================

auto cAudioProcessor::Clear() -> void {
    const cMutexLock lock(mutex.get());

    if (alsaHandle) {
        (void)snd_pcm_drop(alsaHandle);
        (void)snd_pcm_prepare(alsaHandle);
    }

    alsaErrorCount.store(0, std::memory_order_relaxed);
    ResetPlaybackClock();
    clearGeneration.fetch_add(1, std::memory_order_release);
    DrainPacketQueue();
    // The parser recreate below subsumes any pending reset requested by Action().
    parserNeedsReset.store(false, std::memory_order_relaxed);

    // avcodec_flush_buffers() must run on the Action thread: calling it here races
    // against an in-flight DecodeToPcm() that already passed the refcount gate.
    if (decoder) {
        needsFlush.store(true, std::memory_order_release);
    }

    RecreateParser();
}

auto cAudioProcessor::Decode(const uint8_t *data, size_t size, int64_t pts) -> void {
    if (!data || size == 0) [[unlikely]] {
        return;
    }

    const cMutexLock lock(mutex.get());

    if (!parserCtx || !decoder) [[unlikely]] {
        return;
    }

    // Cascade-recovery handoff: Action() set parserNeedsReset after AUDIO_DECODER_ERROR_LIMIT
    // but cannot recreate the parser itself (would deadlock CloseDecoder's refcount spin).
    // We hold the mutex here, so reset the parser, drain packets it already emitted, and
    // bump clearGeneration so an Action() packet already dequeued from that era is dropped.
    if (parserNeedsReset.exchange(false, std::memory_order_acquire)) {
        dsyslog("vaapivideo/audio: cascade recovery -- recreating parser for %s",
                avcodec_get_name(streamParams.codecId));
        RecreateParser();
        DrainPacketQueue();
        clearGeneration.fetch_add(1, std::memory_order_release);
        if (!parserCtx) {
            return;
        }
    }

    const uint8_t *currentData = data;
    int remainingSize = static_cast<int>(size);

    while (remainingSize > 0) {
        uint8_t *parsedData = nullptr;
        int parsedSize = 0;

        const int consumed = av_parser_parse2(parserCtx.get(), decoder.get(), &parsedData, &parsedSize, currentData,
                                              remainingSize, pts, AV_NOPTS_VALUE, 0);
        if (consumed < 0) {
            break;
        }

        currentData += consumed;
        remainingSize -= consumed;
        pts = AV_NOPTS_VALUE; // only the first chunk of a PES unit carries the PTS

        if (parsedSize > 0 && parsedData) {
            AVPacket pkt{};
            pkt.data = parsedData;
            pkt.size = parsedSize;
            pkt.pts = parserCtx->pts;
            pkt.dts = parserCtx->dts;
            (void)EnqueuePacket(&pkt);
        }
    }
}

[[nodiscard]] auto cAudioProcessor::GetClock() const noexcept -> int64_t {
    // Extrapolate the DAC position forward from the last playbackPts snapshot.
    // WritePcmToAlsa() snapshots once per ALSA period (~25 ms); reading verbatim would
    // introduce up to 25 ms of staleness. Because audio plays in real time, the DAC
    // position advances by exactly ageMs * 90 ticks between snapshots.
    //
    // If WritePcmToAlsa() stops firing (channel switch, codec swap), extrapolation
    // balloons. Return AV_NOPTS_VALUE after AUDIO_CLOCK_STALE_MS to force video freerun;
    // lipsync re-anchors at the next valid write.
    //
    // Seqlock read of the (playbackPts, lastClockUpdateMs) pair: retry until two even sequence loads
    // match. Both fences are load-bearing -- seq1's acquire stops the data loads from hoisting above
    // it, and the acquire fence below stops them from sinking past seq2. Without that fence the
    // compiler (even on x86) may move the relaxed loads after seq2, letting a torn snapshot pass the
    // seq1==seq2 check and bias GetClock() by a full ALSA period. (Boehm's seqlock hazard.)
    uint64_t lastMs = 0;
    int64_t pts = AV_NOPTS_VALUE;
    while (true) {
        const uint32_t seq1 = clockSequence.load(std::memory_order_acquire);
        if ((seq1 & 1U) != 0U) {
            continue; // writer mid-update
        }
        lastMs = lastClockUpdateMs.load(std::memory_order_relaxed);
        pts = playbackPts.load(std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_acquire); // orders the data loads before seq2

        const uint32_t seq2 = clockSequence.load(std::memory_order_relaxed);
        if (seq1 == seq2) {
            break;
        }
    }
    if (lastMs == 0 || pts == AV_NOPTS_VALUE) {
        // Post-reset / pre-first-write: caller is expected to enter freerun; not a diagnostic event.
        return AV_NOPTS_VALUE;
    }
    if (clockPaused.load(std::memory_order_acquire)) {
        // Freeze() pinned the clock so it cannot extrapolate or age-out through ALSA silence.
        // Returning pts verbatim keeps the decoder's view of the master clock stable across the
        // pause and resume window. The flag clears on the next WritePcmToAlsa() / Clear() /
        // ResetPlaybackClock(); clear clockStaleLogged here so a long pause followed by resume
        // does not suppress a real staleness diagnostic later.
        clockStaleLogged.store(false, std::memory_order_relaxed);
        return pts;
    }
    const uint64_t nowMs = cTimeMs::Now();
    // Unsigned subtraction: wrap on clock skew / atomic race makes ageMs huge -> stale check fires safely.
    const uint64_t ageMs = nowMs - lastMs;
    if (ageMs > AUDIO_CLOCK_STALE_MS) {
        // Stale path: writer thread stopped publishing without ResetPlaybackClock(). Edge-triggered
        // log so a 50 Hz polling decoder doesn't spam syslog while the stall persists. Flag clears
        // on the next valid read below.
        if (!clockStaleLogged.exchange(true, std::memory_order_relaxed)) {
            dsyslog("vaapivideo/audio: GetClock=NOPTS (stale: age=%lums, threshold=%lums)",
                    static_cast<unsigned long>(ageMs), static_cast<unsigned long>(AUDIO_CLOCK_STALE_MS));
        }
        return AV_NOPTS_VALUE;
    }
    clockStaleLogged.store(false, std::memory_order_relaxed);
    return pts + (static_cast<int64_t>(ageMs) * PTS_TICKS_PER_MS);
}

[[nodiscard]] auto cAudioProcessor::Initialize(std::string_view alsaDevice) -> bool {
    if (alsaDevice.empty()) {
        esyslog("vaapivideo/audio: empty device name");
        return false;
    }

    const cMutexLock lock(mutex.get());

    if (initialized.load(std::memory_order_relaxed)) {
        if (alsaDeviceName == alsaDevice) {
            return true;
        }
        // Different device: close the current one but keep the processing thread alive
        // to avoid the create/destroy round-trip cost.
        CloseDevice();
    }

    alsaDeviceName = std::string(alsaDevice);

    ProbeSinkCaps();

    if (!OpenAlsaDevice()) {
        CloseDevice();
        alsaDeviceName.clear();
        return false;
    }

    initialized.store(true, std::memory_order_release);
    stopping.store(false, std::memory_order_release);
    hasExited.store(false, std::memory_order_relaxed);
    // Start() must follow the flag writes above: a stale stopping=true from a prior
    // Shutdown() causes Action() to exit on the first iteration.
    Start();

    isyslog("vaapivideo/audio: initialized on '%.*s'", static_cast<int>(alsaDevice.size()), alsaDevice.data());
    return true;
}

[[nodiscard]] auto cAudioProcessor::IsInitialized() const noexcept -> bool {
    return initialized.load(std::memory_order_relaxed);
}

[[nodiscard]] auto cAudioProcessor::IsPassthrough() const noexcept -> bool {
    return alsaPassthroughActive.load(std::memory_order_acquire);
}

[[nodiscard]] auto cAudioProcessor::GetQueueSize() const -> size_t {
    const cMutexLock lock(mutex.get());
    return packetQueue.size();
}

[[nodiscard]] auto cAudioProcessor::GetPendingWorkSize() const -> size_t {
    size_t depth = 0;
    {
        const cMutexLock lock(mutex.get());
        depth = packetQueue.size();
    }
    // Queue reads 0 while Action() still holds a popped packet mid-handoff to ALSA.
    if (packetInFlight.load(std::memory_order_acquire)) {
        ++depth;
    }
    // After DropOutput() (Mute/Freeze/trick) ALSA is empty but pcmNextPts/clock are preserved, so the
    // clock < pcmNextPts tail estimate would report a phantom tail that never drains.
    if (!outputDropped.load(std::memory_order_acquire)) {
        // clock (DAC position) < end-of-queued-ALSA PTS => samples still unplayed.
        const int64_t endPts = pcmNextPts.load(std::memory_order_acquire);
        if (endPts != AV_NOPTS_VALUE) {
            const int64_t clock = GetClock();
            if (clock != AV_NOPTS_VALUE && clock < endPts) {
                ++depth;
            }
        }
    }
    return depth;
}

[[nodiscard]] auto cAudioProcessor::IsQueueFull() const -> bool {
    const cMutexLock lock(mutex.get());
    return packetQueue.size() >= AUDIO_QUEUE_CAPACITY;
}

[[nodiscard]] auto cAudioProcessor::OpenCodec(AVCodecID codecId, int sampleRate, int channels) -> bool {
    if (sampleRate <= 0 || channels <= 0) [[unlikely]] {
        esyslog("vaapivideo/audio: invalid parameters (%dHz %dch)", sampleRate, channels);
        return false;
    }

    return SetStreamParams({.channels = channels, .codecId = codecId, .sampleRate = sampleRate});
}

[[nodiscard]] auto cAudioProcessor::OpenCodecWithInfo(const AudioStreamInfo &info) -> bool {
    // Mediaplayer path: AudioStreamInfo carries extradata (e.g. AAC raw, MP4 ESDS payload).
    // AdoptStreamParams inside SetStreamParams() deep-copies the bytes into storedExtradata,
    // so the caller can free/reuse its buffer immediately after this returns.
    return SetStreamParams(info);
}

[[nodiscard]] auto cAudioProcessor::SetStreamParams(const AudioStreamParams &params) -> bool {
    if (params.sampleRate <= 0 || params.channels <= 0) [[unlikely]] {
        esyslog("vaapivideo/audio: invalid stream parameters");
        return false;
    }

    const cMutexLock lock(mutex.get());

    // Probe before computing wantPassthrough so CanPassthrough() reads fresh sinkCaps
    // and the fast-path bailout below sees the current ELD-derived support set.
    ProbeSinkCaps();

    const bool wantPassthrough = CanPassthrough(params.codecId);
    const bool currentlyPassthrough = alsaPassthroughActive.load(std::memory_order_relaxed);

    // streamParams.extradata (when non-null) points into storedExtradata (owned copy),
    // so this memcmp is safe regardless of what the caller did with its buffer.
    // AudioStreamInfo::extradata is non-owning by contract (see src/stream.h).
    const auto storedSize = static_cast<int>(storedExtradata.size());
    bool extradataMatches = storedSize == params.extradataSize;
    if (extradataMatches && params.extradataSize > 0) {
        extradataMatches = params.extradata != nullptr && std::memcmp(storedExtradata.data(), params.extradata,
                                                                      static_cast<size_t>(params.extradataSize)) == 0;
    }
    const bool hasActivePipeline = alsaHandle != nullptr && decoder != nullptr && parserCtx != nullptr;

    // Fast path: all params and passthrough decision unchanged, pipeline live.
    // PassthroughMode comparison is load-bearing: the user can flip Auto/On/Off from
    // the setup menu between identical-codec calls, which must still trigger a reopen.
    // hasActivePipeline prevents reporting success when a prior open left a broken state.
    if (streamParams.codecId == params.codecId && streamParams.sampleRate == params.sampleRate &&
        streamParams.channels == params.channels && extradataMatches && wantPassthrough == currentlyPassthrough &&
        hasActivePipeline) {
        return true;
    }

    // PassthroughMode::On intentionally ignores the ELD. Log once per codec change so a
    // "no audio with On" report points at this line rather than requiring trace logging.
    if (wantPassthrough && vaapiConfig.passthroughMode.load(std::memory_order_relaxed) == PassthroughMode::On &&
        sinkCaps.elded && !SinkSupports(params.codecId)) {
        isyslog("vaapivideo/audio: PassthroughMode=on overriding ELD for %s (sink advertises no support); "
                "expect silence if the downstream device cannot actually decode IEC61937",
                avcodec_get_name(params.codecId));
    }

    const AVCodecID oldCodecId = streamParams.codecId;
    const bool decoderConfigChanged = oldCodecId != params.codecId || !extradataMatches;

    DrainPacketQueue();

    AdoptStreamParams(params);

    const auto targetRate = ComputeAlsaRate(params.codecId, static_cast<unsigned>(params.sampleRate), wantPassthrough);

    // ALSA reopen is needed when: handle missing, passthrough mode changed, or the
    // hardware rate/channel count changed. A missing decoder/parser alone is NOT a
    // reason to reopen; the decoderConfigChanged branch below handles that and is
    // the common first-Decode-after-Initialize path (decoder/parserCtx still null).
    const unsigned currentAlsaRate = alsaSampleRate.load(std::memory_order_relaxed);
    const unsigned currentAlsaChannels = alsaChannels.load(std::memory_order_relaxed);
    const bool needsReconfig =
        !initialized.load(std::memory_order_relaxed) || alsaHandle == nullptr ||
        currentlyPassthrough != wantPassthrough ||
        (wantPassthrough ? targetRate != currentAlsaRate
                         : (targetRate != currentAlsaRate || params.channels != static_cast<int>(currentAlsaChannels)));

    // Bump clearGeneration BEFORE tearing down decoder/parser. The Action thread may
    // hold an already-dequeued packet from the old codec era; the bump marks it stale
    // so DecodeToPcm()/passthrough drops it instead of feeding the fresh decoder.
    // Clear() also bumps; this covers callers that bypass Clear().
    if (needsReconfig || decoderConfigChanged) {
        clearGeneration.fetch_add(1, std::memory_order_release);
    }

    if (needsReconfig) {
        if (oldCodecId != AV_CODEC_ID_NONE && !currentlyPassthrough) {
            CloseDecoder();
        }
        if (alsaHandle) {
            (void)snd_pcm_drop(alsaHandle);
            snd_pcm_close(alsaHandle);
            alsaHandle = nullptr;
        }

        if (!OpenAlsaDevice()) {
            esyslog("vaapivideo/audio: reconfiguration failed");
            return false;
        }
        OpenDecoder();
        return alsaHandle != nullptr && decoder != nullptr && parserCtx != nullptr;
    }

    if (decoderConfigChanged || !hasActivePipeline) {
        // (a) Same HW config but codec or extradata changed, or (b) pipeline torn down
        // with ALSA still valid (first Decode() after Initialize(), or recovery from a
        // prior OpenDecoder() failure). Drop in-flight PCM so GetClock() doesn't report
        // delay from the previous codec's queued samples, then rebuild the decoder side.
        if (alsaHandle) {
            (void)snd_pcm_drop(alsaHandle);
            (void)snd_pcm_prepare(alsaHandle);
        }
        ResetPlaybackClock();
        CloseDecoder();
        OpenDecoder();
        return alsaHandle != nullptr && decoder != nullptr && parserCtx != nullptr;
    }
    return true;
}

auto cAudioProcessor::AdoptStreamParams(const AudioStreamParams &params) -> void {
    // Caller must hold mutex. Deep-copies extradata into storedExtradata and rewires
    // streamParams.extradata to the owned buffer; later reads (memcmp fast path,
    // OpenDecoder()) never touch caller-owned memory.
    streamParams = params;
    if (params.extradata != nullptr && params.extradataSize > 0) {
        const auto size = static_cast<size_t>(params.extradataSize);
        storedExtradata.assign(params.extradata, params.extradata + size);
        streamParams.extradata = storedExtradata.data();
    } else {
        storedExtradata.clear();
        streamParams.extradata = nullptr;
        streamParams.extradataSize = 0;
    }
}

auto cAudioProcessor::SetVolume(int vol) -> void {
    const int clampedVol = std::clamp(vol, 0, 255);
    const int oldVol = volume.exchange(clampedVol, std::memory_order_relaxed);
    if (oldVol != clampedVol) {
        dsyslog("vaapivideo/audio: SetVolume(%d) %s", clampedVol, clampedVol == 0 ? "[muted]" : "");
    }
}

auto cAudioProcessor::Shutdown() -> void {
    // Idempotent via stopping exchange. First caller (stopping 0->1) signals the thread;
    // subsequent callers skip the signal/cancel but still run the wait + CloseDevice()
    // tail so a destructor-after-Shutdown() doesn't leave handles dangling.
    // Initialize()'s device-swap path calls CloseDevice() directly instead, keeping the
    // processing thread alive across the swap.
    const bool wasStopping = stopping.exchange(true, std::memory_order_release);
    if (!wasStopping) {
        dsyslog("vaapivideo/audio: shutting down");
        packetCondition.Broadcast();
        Cancel(3);
    }

    const cTimeMs timeout(SHUTDOWN_TIMEOUT_MS);
    while (!hasExited.load(std::memory_order_acquire) && !timeout.TimedOut()) {
        packetCondition.Broadcast();
        cCondWait::SleepMs(10);
    }

    CloseDevice();
}

auto cAudioProcessor::CloseDevice() -> void {
    // Caller is responsible for ensuring the processing thread has exited (Shutdown)
    // or will safely re-discover config on the next Decode() (Initialize device-swap).
    const cMutexLock lock(mutex.get());

    // HDMI codec drivers persist the IEC958 non-audio bit across snd_pcm_close(); leaving it
    // set strands the next process in IEC61937 mode. Reset unconditionally before close.
    SetIec958NonAudio(false);

    CloseSpdifMuxer();
    CloseDecoder();
    DrainPacketQueue();

    if (alsaHandle) {
        (void)snd_pcm_drop(alsaHandle);
        snd_pcm_close(alsaHandle);
        alsaHandle = nullptr;
    }

    alsaErrorCount.store(0, std::memory_order_relaxed);
    ResetPlaybackClock();
    parserNeedsReset.store(false, std::memory_order_relaxed); // CloseDecoder() above already destroyed the parser
    alsaPassthroughActive.store(false, std::memory_order_release);
    alsaFrameBytes.store(0, std::memory_order_release);
    alsaChannels.store(0, std::memory_order_relaxed);
    alsaSampleRate.store(0, std::memory_order_relaxed);
    initialized.store(false, std::memory_order_release);
}

auto cAudioProcessor::DrainPacketQueue() -> void {
    // Caller must hold mutex. FreeAVPacket deleter runs on each packet as the unique_ptr leaves scope.
    while (!packetQueue.empty()) {
        const std::unique_ptr<AVPacket, FreeAVPacket> dropped{packetQueue.front()};
        packetQueue.pop();
    }
}

auto cAudioProcessor::RecreateParser() -> void {
    // Caller holds mutex. Parser has no flush API; close (nullptr-safe) + re-init.
    av_parser_close(parserCtx.release());
    if (streamParams.codecId != AV_CODEC_ID_NONE) {
        parserCtx.reset(av_parser_init(streamParams.codecId));
    }
}

auto cAudioProcessor::DropOutput(bool pauseClock) -> void {
    // Clock-preserving variant of Clear(): silence playback NOW (snd_pcm_drop drains the ~200 ms
    // already queued in the ALSA sink), bump clearGeneration so in-flight decoder packets from the
    // previous era are dropped silently, but leave (playbackPts, lastClockUpdateMs, pcmNextPts)
    // intact. GetClock() keeps returning valid timestamps, the decoder stays paced, the display
    // queue does not underrun. Used by Mute/Freeze/SetTrickSpeed -- all stream-preserving events.
    //
    // If the resumed audio is from a different timeline (FF/REW exit), WritePcmToAlsa()'s
    // >5s-PTS-jump guard auto-detects and calls ResetPlaybackClock().
    //
    // pauseClock=true (Freeze() path): pin GetClock() at the current playbackPts so it cannot
    // extrapolate against wall-clock through ALSA silence (a short pause < AUDIO_CLOCK_STALE_MS
    // would otherwise produce a fake-advanced clock on resume, which then drops the preserved
    // jitterBuf head via SkipStaleJitterFrames -- silent playback-position skip on un-pause).
    // Cleared by ResetPlaybackClock() / Clear() / the next WritePcmToAlsa() that re-anchors it.
    const cMutexLock lock(mutex.get());
    if (alsaHandle) {
        (void)snd_pcm_drop(alsaHandle);
        (void)snd_pcm_prepare(alsaHandle);
    }
    alsaErrorCount.store(0, std::memory_order_relaxed);
    // pcmNextPts/playbackPts survive the drop, so flag it: else GetPendingWorkSize() reads the
    // preserved clock as a still-draining tail. Cleared by the next WritePcmToAlsa/ResetPlaybackClock.
    outputDropped.store(true, std::memory_order_release);
    clearGeneration.fetch_add(1, std::memory_order_release);
    DrainPacketQueue();
    if (pauseClock) {
        clockPaused.store(true, std::memory_order_release);
    }
}

auto cAudioProcessor::ResetPlaybackClock() -> void {
    // lastClockUpdateMs == 0 is GetClock()'s "no clock yet" sentinel.
    // Clears pcmNextPts so the next ALSA write re-anchors on the new timeline.
    // Brackets the pair with clockSequence bumps so GetClock()'s seqlock retries
    // past the transient odd sequence.
    //
    // Caller MUST hold `mutex` (Clear(), SetStreamParams(), CloseDevice(), Action()'s
    // 5-s jump all do). The mutex serializes this writer against WritePcmToAlsa(),
    // preserving the seqlock's single-writer invariant.
    clockSequence.fetch_add(1, std::memory_order_acq_rel);
    playbackPts.store(AV_NOPTS_VALUE, std::memory_order_relaxed);
    lastClockUpdateMs.store(0, std::memory_order_relaxed);
    clockSequence.fetch_add(1, std::memory_order_release);
    outputDropped.store(false, std::memory_order_release);
    pcmNextPts.store(AV_NOPTS_VALUE, std::memory_order_relaxed);
    // Reset content boundary: clear any pause pin so the new timeline doesn't inherit it.
    clockPaused.store(false, std::memory_order_release);
}

// ============================================================================
// === THREAD ===
// ============================================================================

auto cAudioProcessor::Action() -> void {
    dsyslog("vaapivideo/audio: processing thread started");

    while (!stopping.load(std::memory_order_acquire)) {
        std::unique_ptr<AVPacket, FreeAVPacket> packet;
        bool passthrough = false;
        uint32_t generationAtDequeue = 0;

        {
            const cMutexLock lock(mutex.get());
            while (packetQueue.empty() && !stopping.load(std::memory_order_acquire)) {
                packetCondition.TimedWait(*mutex, 100);
            }

            if (stopping.load(std::memory_order_acquire)) {
                break;
            }

            packet.reset(packetQueue.front());
            packetQueue.pop();
            // Latch under the mutex so GetPendingWorkSize() never sees empty-queue + no-in-flight
            // while this iteration is still draining the packet to ALSA.
            packetInFlight.store(true, std::memory_order_release);
            passthrough = alsaPassthroughActive.load(std::memory_order_relaxed);
            // Snapshot clearGeneration under the mutex so it pairs atomically with the pop.
            // DecodeToPcm()/passthrough re-check it later to drop packets whose bytes
            // belong to a codec era swapped out between dequeue and decode.
            generationAtDequeue = clearGeneration.load(std::memory_order_relaxed);
        }

        // Drop stale-era packets before they touch pcmNextPts or the 5-s jump path.
        // An old-era PTS would silently anchor the new-era timeline, causing
        // WritePcmToAlsa() to publish a bogus playbackPts on the next valid packet.
        if (clearGeneration.load(std::memory_order_acquire) != generationAtDequeue) {
            packetInFlight.store(false, std::memory_order_release);
            continue;
        }

        const int64_t pts = packet->pts;
        if (pts != AV_NOPTS_VALUE) {
            // Single-writer contract: pcmNextPts has multiple writers (this site,
            // ResetPlaybackClock, WritePcmToAlsa) but every writer must hold `mutex`
            // so the value is well-defined to readers. Holding the mutex across
            // read+decision+store here keeps the contract; the alternative (store
            // outside the lock) lets ResetPlaybackClock race the store and leave a
            // stale pre-jump value visible to the next iteration's prevNextPts.
            //
            // Cost: one extra mutex acquisition per packet with a valid PTS. At DVB
            // packet rates (~30/s for AC-3, ~40/s for E-AC-3) this is negligible.
            //
            // The mutex also fences the 5 s jump path: flush ALSA + ResetPlaybackClock
            // (which zeroes pcmNextPts to NOPTS) must complete BEFORE the final store
            // of pts, or the new-timeline anchor races the reset.
            //
            // If you add new pcmNextPts readers outside the mutex (currently only
            // generation-gated readers exist), re-verify the ordering here.
            const cMutexLock lock(mutex.get());
            const int64_t prevNextPts = pcmNextPts.load(std::memory_order_relaxed);
            if (prevNextPts != AV_NOPTS_VALUE) {
                const int64_t diff = (pts > prevNextPts) ? (pts - prevNextPts) : (prevNextPts - pts);
                if (diff > (5 * PTSTICKS)) {
                    // >5 s PTS jump (channel switch, seek, wrap): flush ALSA so
                    // WritePcmToAlsa() publishes a clean playbackPts on the new timeline.
                    if (alsaHandle) {
                        (void)snd_pcm_drop(alsaHandle);
                        (void)snd_pcm_prepare(alsaHandle);
                    }
                    ResetPlaybackClock();
                }
            }
            pcmNextPts.store(pts, std::memory_order_relaxed);
        }

        if (passthrough) {
            // Mutex serializes WrapIec61937 with CloseSpdifMuxer; without it spdifMuxCtx
            // can be freed mid-write. WritePcmToAlsa nests recursively.
            const cMutexLock lock(mutex.get());
            if (clearGeneration.load(std::memory_order_acquire) != generationAtDequeue) {
                packetInFlight.store(false, std::memory_order_release);
                continue;
            }
            const auto burst = WrapIec61937(packet->data, packet->size);
            if (!burst.empty()) {
                const size_t bpf = alsaFrameBytes.load(std::memory_order_relaxed);
                const unsigned burstFrames = (bpf > 0) ? static_cast<unsigned>(burst.size() / bpf) : 0;
                const int64_t startPts90k = pcmNextPts.load(std::memory_order_relaxed);
                (void)WritePcmToAlsa(burst, startPts90k, burstFrames, generationAtDequeue);
            }
        } else {
            (void)DecodeToPcm(std::span(packet->data, static_cast<size_t>(packet->size)), pts, generationAtDequeue);
        }
        packetInFlight.store(false, std::memory_order_release);
    }

    hasExited.store(true, std::memory_order_release);
    dsyslog("vaapivideo/audio: processing thread stopped");
}

// ============================================================================
// === INTERNAL METHODS ===
// ============================================================================

[[nodiscard]] auto cAudioProcessor::CanPassthrough(AVCodecID codecId) const -> bool {
    // Non-wrappable codecs (AAC, MP2, ...) return false from both CodecWrappable() and
    // SinkSupports() and always take the PCM path. See README for mode semantics.
    switch (vaapiConfig.passthroughMode.load(std::memory_order_relaxed)) {
        case PassthroughMode::Off:
            return false;
        case PassthroughMode::On:
            return CodecWrappable(codecId);
        case PassthroughMode::Auto:
            break;
    }
    return SinkSupports(codecId);
}

[[nodiscard]] auto cAudioProcessor::CodecWrappable(AVCodecID codecId) -> bool {
    // Delegates to IsPassthroughCapable() in stream.h, which owns the wrappable codec
    // list (kAudioPassthroughTable). AudioSinkCaps::Supports() must mirror this set so
    // Auto mode can enable passthrough when the ELD confirms support.
    return IsPassthroughCapable(codecId);
}

[[nodiscard]] auto cAudioProcessor::SinkSupports(AVCodecID codecId) const -> bool {
    // DTS-HD is only a sink capability in this plugin. We wrap AV_CODEC_ID_DTS core,
    // not DTS-HD/HBR; a DTS-HD-capable sink is still valid for DTS core passthrough.
    return sinkCaps.Supports(codecId);
}

auto cAudioProcessor::CloseDecoder() -> void {
    // Spin until all in-flight DecodeToPcm() callers release the refcount before freeing
    // the decoder and parser. AUDIO_DECODER_DRAIN_TIMEOUT_MS caps the wait.
    for (const cTimeMs start;
         decoderRefCount.load(std::memory_order_acquire) > 0 && start.Elapsed() < AUDIO_DECODER_DRAIN_TIMEOUT_MS;) {
        cCondWait::SleepMs(1);
    }

    decoder.reset();
    parserCtx.reset();
    if (swrCtx) {
        swr_free(&swrCtx);
    }
    swrChannels = 0;
    swrFormat = AV_SAMPLE_FMT_NONE;
}

auto cAudioProcessor::FlushDecoderState() -> void {
    avcodec_flush_buffers(decoder.get());
    // swrCtx intentionally retained: a seek inside the same stream keeps the same sample
    // format / channel layout, so reinitializing swresample is wasted work (and emits a
    // log line per seek). The format-change guard in the conversion path drops swrCtx
    // automatically when the post-flush frame's format actually differs.
    consecutiveDecodeErrors = 0;
    decoderGracePackets = AUDIO_DECODER_GRACE_PACKETS;
}

[[nodiscard]] auto cAudioProcessor::ComputeAlsaRate(AVCodecID codecId, unsigned streamRate, bool passthrough) const
    -> unsigned {
    if (!passthrough) {
        return streamRate;
    }
    // IEC61937 carrier rate: AC-3 and DTS run at 1x the stream rate. DD+, AC-4, and
    // MPEG-H require 4x so the HDMI receiver can reassemble the burst correctly.
    const bool quadRate =
        (codecId == AV_CODEC_ID_EAC3 || codecId == AV_CODEC_ID_AC4 || codecId == AV_CODEC_ID_MPEGH_3D_AUDIO);
    return quadRate ? streamRate * 4 : streamRate;
}

[[nodiscard]] auto cAudioProcessor::ConfigureAlsaParams(snd_pcm_t *handle, snd_pcm_format_t format, unsigned channels,
                                                        unsigned rate, bool allowResample) -> bool {
    if (!handle) {
        return false;
    }

    snd_pcm_hw_params_t *hwParams = nullptr;
    snd_pcm_hw_params_alloca(&hwParams);

    if (const int err = snd_pcm_nonblock(handle, 1); err < 0) {
        dsyslog("vaapivideo/audio: cannot set non-blocking mode: %s", snd_strerror(err));
        return false;
    }

    if (const int err = snd_pcm_hw_params_any(handle, hwParams); err < 0) {
        dsyslog("vaapivideo/audio: cannot initialize hardware parameters: %s", snd_strerror(err));
        return false;
    }

    if (const int err = snd_pcm_hw_params_set_access(handle, hwParams, SND_PCM_ACCESS_RW_INTERLEAVED); err < 0) {
        dsyslog("vaapivideo/audio: interleaved access not supported: %s", snd_strerror(err));
        return false;
    }

    if (const int err = snd_pcm_hw_params_set_format(handle, hwParams, format); err < 0) {
        dsyslog("vaapivideo/audio: format %s not supported: %s", snd_pcm_format_name(format), snd_strerror(err));
        return false;
    }

    if (const int err = snd_pcm_hw_params_set_channels(handle, hwParams, channels); err < 0) {
        dsyslog("vaapivideo/audio: %uch configuration failed: %s", channels, snd_strerror(err));
        return false;
    }

    if (const int err = snd_pcm_hw_params_set_rate_resample(handle, hwParams, allowResample ? 1 : 0); err < 0) {
        dsyslog("vaapivideo/audio: cannot set ALSA rate-resample mode: %s", snd_strerror(err));
        return false;
    }

    // set_rate_near picks the exact rate when supported and the closest match otherwise; set_rate
    // is strict. Selecting one or the other up front by allowResample keeps the success path linear.
    // strict set_rate cannot return success with actualRate != rate, so no post-check needed.
    unsigned actualRate = rate;
    if (const int err = allowResample ? snd_pcm_hw_params_set_rate_near(handle, hwParams, &actualRate, nullptr)
                                      : snd_pcm_hw_params_set_rate(handle, hwParams, actualRate, 0);
        err < 0) {
        dsyslog("vaapivideo/audio: %uHz sample rate not supported: %s", rate, snd_strerror(err));
        return false;
    }

    // Buffer sized to AUDIO_ALSA_BUFFER_MS; the video due-gate mirrors the same cushion
    // (see AVSYNC.md). Period at 25 ms keeps interrupt overhead low while providing
    // enough granularity for the seqlock clock update.
    auto bufferSize = static_cast<snd_pcm_uframes_t>(static_cast<uint64_t>(actualRate) * AUDIO_ALSA_BUFFER_MS / 1000);
    auto periodSize = static_cast<snd_pcm_uframes_t>(actualRate / 40); // 25 ms per interrupt

    if (const int err = snd_pcm_hw_params_set_buffer_size_near(handle, hwParams, &bufferSize); err < 0) {
        dsyslog("vaapivideo/audio: set_buffer_size_near failed: %s", snd_strerror(err));
        return false;
    }
    if (const int err = snd_pcm_hw_params_set_period_size_near(handle, hwParams, &periodSize, nullptr); err < 0) {
        dsyslog("vaapivideo/audio: set_period_size_near failed: %s", snd_strerror(err));
        return false;
    }
    if (const int err = snd_pcm_hw_params(handle, hwParams); err < 0) {
        dsyslog("vaapivideo/audio: snd_pcm_hw_params failed: %s", snd_strerror(err));
        return false;
    }
    if (const int err = snd_pcm_prepare(handle); err < 0) {
        dsyslog("vaapivideo/audio: snd_pcm_prepare failed: %s", snd_strerror(err));
        return false;
    }

    snd_pcm_sw_params_t *swParams = nullptr;
    snd_pcm_sw_params_alloca(&swParams);

    if (const int err = snd_pcm_sw_params_current(handle, swParams); err < 0) {
        dsyslog("vaapivideo/audio: cannot get software parameters: %s", snd_strerror(err));
        return false;
    }

    // Start threshold at 1/3 of the ring: caps channel-switch-to-first-audio latency
    // at ~133 ms while giving the decode ramp-up time to prebuffer.
    const snd_pcm_uframes_t startThreshold = bufferSize / 3;
    if (const int err = snd_pcm_sw_params_set_start_threshold(handle, swParams, startThreshold); err < 0) {
        dsyslog("vaapivideo/audio: set_start_threshold failed: %s", snd_strerror(err));
        return false;
    }
    if (const int err = snd_pcm_sw_params_set_avail_min(handle, swParams, periodSize); err < 0) {
        dsyslog("vaapivideo/audio: set_avail_min failed: %s", snd_strerror(err));
        return false;
    }
    if (const int err = snd_pcm_sw_params(handle, swParams); err < 0) {
        dsyslog("vaapivideo/audio: snd_pcm_sw_params failed: %s", snd_strerror(err));
        return false;
    }

    // HW may round buffer/period; log negotiated values for diagnostics.
    snd_pcm_uframes_t actualBuffer = 0;
    snd_pcm_uframes_t actualPeriod = 0;
    (void)snd_pcm_hw_params_get_buffer_size(hwParams, &actualBuffer);
    (void)snd_pcm_hw_params_get_period_size(hwParams, &actualPeriod, nullptr);
    const unsigned bufMs = (actualRate > 0) ? static_cast<unsigned>(actualBuffer * 1000 / actualRate) : 0;
    const unsigned periodMs = (actualRate > 0) ? static_cast<unsigned>(actualPeriod * 1000 / actualRate) : 0;
    dsyslog("vaapivideo/audio: ALSA buffer=%ums period=%ums start=%ums (target=%dms)", bufMs, periodMs, bufMs / 3,
            AUDIO_ALSA_BUFFER_MS);

    unsigned configuredChannels = 0;
    unsigned configuredRate = 0;
    if (const int err = snd_pcm_hw_params_get_channels(hwParams, &configuredChannels); err < 0) [[unlikely]] {
        dsyslog("vaapivideo/audio: failed to read negotiated channels: %s", snd_strerror(err));
        return false;
    }
    if (const int err = snd_pcm_hw_params_get_rate(hwParams, &configuredRate, nullptr); err < 0) [[unlikely]] {
        dsyslog("vaapivideo/audio: failed to read negotiated rate: %s", snd_strerror(err));
        return false;
    }
    alsaChannels.store(configuredChannels, std::memory_order_release);
    alsaSampleRate.store(configuredRate, std::memory_order_release);

    const auto frameBytes = snd_pcm_frames_to_bytes(handle, 1);
    if (frameBytes <= 0) {
        dsyslog("vaapivideo/audio: invalid frame size");
        return false;
    }

    alsaFrameBytes.store(static_cast<size_t>(frameBytes), std::memory_order_release);

    return true;
}

[[nodiscard]] auto cAudioProcessor::DecodeToPcm(std::span<const uint8_t> data, int64_t pts, uint32_t expectedGeneration)
    -> bool {
    if (!decoder) {
        return false;
    }

    // Increment refcount so a concurrent CloseDecoder() spins until this caller releases it.
    // Acquire pairs with the generation re-check immediately below.
    decoderRefCount.fetch_add(1, std::memory_order_acquire);

    // Re-check decoder pointer AND generation. The refcount only blocks *concurrent*
    // teardown; a completed teardown + fresh OpenDecoder() before this increment slips
    // through. The generation marker (bumped before any swap) catches old-era packets:
    // feeding them to the new decoder would trigger an INVALIDDATA cascade.
    if (!decoder || clearGeneration.load(std::memory_order_acquire) != expectedGeneration) {
        decoderRefCount.fetch_sub(1, std::memory_order_release);
        return false;
    }

    if (needsFlush.exchange(false, std::memory_order_acquire)) {
        FlushDecoderState();
    }

    const std::unique_ptr<AVPacket, FreeAVPacket> packet{av_packet_alloc()};
    if (!packet || av_new_packet(packet.get(), static_cast<int>(data.size())) < 0) {
        decoderRefCount.fetch_sub(1, std::memory_order_release);
        return false;
    }

    std::memcpy(packet->data, data.data(), data.size());
    packet->pts = pts;

    const int sendRet = avcodec_send_packet(decoder.get(), packet.get());

    if (sendRet < 0 && sendRet != AVERROR(EAGAIN)) [[unlikely]] {
        if (decoderGracePackets > 0) {
            --decoderGracePackets;
        } else if (lastDecodeErrorLog.Elapsed() > AUDIO_ERROR_LOG_INTERVAL_MS) {
            dsyslog("vaapivideo/audio: avcodec_send_packet failed: %s (consecutive=%d)", AvErr(sendRet).data(),
                    consecutiveDecodeErrors);
            lastDecodeErrorLog.Set();
        }

        ++consecutiveDecodeErrors;
        if (consecutiveDecodeErrors >= AUDIO_DECODER_ERROR_LIMIT) {
            dsyslog("vaapivideo/audio: %d consecutive decode failures, flushing", consecutiveDecodeErrors);
            FlushDecoderState();
            // Cascade self-recovery: sustained INVALIDDATA means the FFmpeg parser
            // (owned by the producer thread in Decode()) is stuck on a corrupt partial
            // frame from a PES discontinuity. Flushing the decoder alone is insufficient;
            // the parser keeps producing the same garbage. Defer the parser recreate to
            // Decode()'s next entry, where it runs safely under the producer mutex.
            // Cannot do it here: decoderRefCount > 0 would deadlock CloseDecoder().
            parserNeedsReset.store(true, std::memory_order_release);
        }

        decoderRefCount.fetch_sub(1, std::memory_order_release);
        return true; // soft failure: decoder may recover on a later packet
    }

    consecutiveDecodeErrors = 0;

    const std::unique_ptr<AVFrame, FreeAVFrame> frame{av_frame_alloc()};
    if (!frame) {
        decoderRefCount.fetch_sub(1, std::memory_order_release);
        return false;
    }

    while (avcodec_receive_frame(decoder.get(), frame.get()) == 0) {
        if (frame->nb_samples <= 0) [[unlikely]] {
            av_frame_unref(frame.get());
            continue;
        }

        const unsigned configuredChannels = alsaChannels.load(std::memory_order_relaxed);
        const unsigned outCh = configuredChannels > 0 ? configuredChannels : 2;
        const auto frameFmt = static_cast<AVSampleFormat>(frame->format);
        const int frameCh = frame->ch_layout.nb_channels;

        if (swrCtx && (frameFmt != swrFormat || frameCh != swrChannels)) {
            swr_free(&swrCtx);
        }

        if (!swrCtx) {
            AVChannelLayout outLayout{};
            av_channel_layout_default(&outLayout, static_cast<int>(outCh));

            const int ret = swr_alloc_set_opts2(&swrCtx, &outLayout, AV_SAMPLE_FMT_S16, frame->sample_rate,
                                                &frame->ch_layout, frameFmt, frame->sample_rate, 0, nullptr);

            if (ret < 0 || !swrCtx || swr_init(swrCtx) < 0) {
                esyslog("vaapivideo/audio: swr_alloc_set_opts2 failed for %s %dch -> S16 %uch",
                        av_get_sample_fmt_name(frameFmt), frameCh, outCh);
                if (swrCtx) {
                    swr_free(&swrCtx);
                }
                swrChannels = 0;
                swrFormat = AV_SAMPLE_FMT_NONE;
                decoderRefCount.fetch_sub(1, std::memory_order_release);
                return false;
            }

            swrFormat = frameFmt;
            swrChannels = frameCh;
            dsyslog("vaapivideo/audio: initialized swresample for %s %dch -> S16 %uch",
                    av_get_sample_fmt_name(frameFmt), frameCh, outCh);
        }

        const uint8_t *pcmData = nullptr;
        size_t pcmSize = 0;
        std::vector<uint8_t> convertedBuffer;

        {
            const int estimatedOut = swr_get_out_samples(swrCtx, frame->nb_samples);
            const int maxOutSamples = std::max(estimatedOut, frame->nb_samples) + 128;
            const size_t bufferSize = static_cast<size_t>(maxOutSamples) * outCh * 2;
            convertedBuffer.resize(bufferSize);

            // outPtr cannot be const: swr_convert writes through &outPtr.
            uint8_t *outPtr = convertedBuffer.data(); // NOLINT(misc-const-correctness)
            const int converted = swr_convert(swrCtx, &outPtr, maxOutSamples, frame->data, frame->nb_samples);

            if (converted < 0) [[unlikely]] {
                esyslog("vaapivideo/audio: swr_convert failed");
                decoderRefCount.fetch_sub(1, std::memory_order_release);
                return false;
            }

            pcmSize = static_cast<size_t>(converted) * outCh * 2;
            pcmData = convertedBuffer.data();
        }

        const auto sampleCount = static_cast<unsigned>(frame->nb_samples);

        // Re-check generation: the receive_frame loop can run for many ms (large packets,
        // swr conversion), giving SetStreamParams()/Clear() a window to race.
        // Drop the frame so stale PCM never reaches ALSA or corrupts pcmNextPts.
        if (clearGeneration.load(std::memory_order_acquire) != expectedGeneration) [[unlikely]] {
            decoderRefCount.fetch_sub(1, std::memory_order_release);
            return true;
        }

        const int64_t startPts90k = pcmNextPts.load(std::memory_order_relaxed);
        const bool writeOk = WritePcmToAlsa(std::span(pcmData, pcmSize), startPts90k, sampleCount, expectedGeneration);
        av_frame_unref(frame.get());

        if (!writeOk) {
            decoderRefCount.fetch_sub(1, std::memory_order_release);
            return false;
        }
    }

    decoderRefCount.fetch_sub(1, std::memory_order_release);
    return true;
}

[[nodiscard]] auto cAudioProcessor::EnqueuePacket(const AVPacket *rawPacket) -> bool {
    if (!rawPacket || !rawPacket->data || rawPacket->size <= 0 || !initialized.load(std::memory_order_relaxed))
        [[unlikely]] {
        return false;
    }

    auto packet = std::unique_ptr<AVPacket, FreeAVPacket>{av_packet_clone(rawPacket)};
    if (!packet) [[unlikely]] {
        esyslog("vaapivideo/audio: failed to clone packet");
        return false;
    }

    {
        const cMutexLock lock(mutex.get());

        if (packetQueue.size() >= AUDIO_QUEUE_CAPACITY) [[unlikely]] {
            // Throttle to one log per 500 ms to avoid flooding syslog under sustained overload.
            if (lastQueueWarn.Elapsed() > 500) {
                esyslog("vaapivideo/audio: queue full (%zu packets), dropping (%s @ %dHz %dch passthrough=%s)",
                        packetQueue.size(), avcodec_get_name(streamParams.codecId), streamParams.sampleRate,
                        streamParams.channels, alsaPassthroughActive.load(std::memory_order_relaxed) ? "yes" : "no");
                lastQueueWarn.Set();
            }
            return false;
        }

        packetQueue.push(packet.release());
    }

    packetCondition.Broadcast();
    return true;
}

/// AVIO write callback for the spdif muxer: appends IEC61937 burst bytes to spdifOutputBuf.
static auto SpdifWriteCallback(void *opaque, const uint8_t *buf, int bufSize) -> int {
    auto &output = *static_cast<std::vector<uint8_t> *>(opaque);
    output.insert(output.end(), buf, buf + bufSize);
    return bufSize;
}

auto cAudioProcessor::OpenSpdifMuxer(AVCodecID codecId, int sampleRate) -> bool {
    CloseSpdifMuxer();

    if (avformat_alloc_output_context2(&spdifMuxCtx, nullptr, "spdif", nullptr) < 0 || !spdifMuxCtx) {
        dsyslog("vaapivideo/audio: failed to allocate spdif muxer context");
        return false;
    }

    constexpr int kIoBufSize = 32768; // AVIO scratch buffer; SpdifWriteCallback drains it on every av_write_frame()
    auto *ioBuffer = static_cast<uint8_t *>(av_malloc(kIoBufSize));
    if (!ioBuffer) {
        avformat_free_context(spdifMuxCtx);
        spdifMuxCtx = nullptr;
        return false;
    }

    spdifMuxCtx->pb =
        avio_alloc_context(ioBuffer, kIoBufSize, 1, &spdifOutputBuf, nullptr, SpdifWriteCallback, nullptr);
    if (!spdifMuxCtx->pb) {
        av_free(ioBuffer);
        avformat_free_context(spdifMuxCtx);
        spdifMuxCtx = nullptr;
        return false;
    }

    // stream pointee cannot be const: codecpar fields are mutated below.
    AVStream *const stream = avformat_new_stream(spdifMuxCtx, nullptr); // NOLINT(misc-const-correctness)
    if (!stream) {
        CloseSpdifMuxer();
        return false;
    }

    stream->codecpar->codec_type = AVMEDIA_TYPE_AUDIO;
    stream->codecpar->codec_id = codecId;
    stream->codecpar->sample_rate = sampleRate;
    av_channel_layout_default(&stream->codecpar->ch_layout, 2);

    if (avformat_write_header(spdifMuxCtx, nullptr) < 0) {
        dsyslog("vaapivideo/audio: spdif muxer write_header failed for %s", avcodec_get_name(codecId));
        CloseSpdifMuxer();
        return false;
    }

    dsyslog("vaapivideo/audio: spdif muxer opened for %s @ %dHz", avcodec_get_name(codecId), sampleRate);
    return true;
}

auto cAudioProcessor::CloseSpdifMuxer() -> void {
    if (spdifMuxCtx) {
        if (spdifMuxCtx->pb) {
            av_freep(static_cast<void *>(&spdifMuxCtx->pb->buffer));
            avio_context_free(&spdifMuxCtx->pb);
        }
        avformat_free_context(spdifMuxCtx);
        spdifMuxCtx = nullptr;
    }
    spdifOutputBuf.clear();
}

[[nodiscard]] auto cAudioProcessor::WrapIec61937(const uint8_t *data, int size) -> std::span<const uint8_t> {
    if (!spdifMuxCtx || !data || size <= 0) {
        return {};
    }

    spdifOutputBuf.clear();

    AVPacket *pkt = av_packet_alloc();
    if (!pkt) {
        return {};
    }

    // av_write_frame() only reads pkt->data; const_cast is safe at this API boundary.
    pkt->data = const_cast<uint8_t *>(data); // NOLINT(cppcoreguidelines-pro-type-const-cast)
    pkt->size = size;
    pkt->stream_index = 0;

    const int ret = av_write_frame(spdifMuxCtx, pkt);
    av_packet_free(&pkt);

    if (ret < 0) {
        return {};
    }

    return {spdifOutputBuf.data(), spdifOutputBuf.size()};
}

auto cAudioProcessor::SetIec958NonAudio(bool enable) const -> void {
    if (alsaCardId < 0 || alsaIec958CtlIndex == UINT_MAX) {
        return; // probe never ran or IEC958 control not exposed (e.g. ALSA 'default' via dmix)
    }

    // Fixed-size buffer: ~cAudioProcessor -> Shutdown -> CloseDevice reaches here noexcept,
    // and both std::format and std::string can throw bad_alloc. At the C-API boundary a stack
    // buffer is the cleanest fix.
    std::array<char, 16> ctlName{};
    (void)std::snprintf(ctlName.data(), ctlName.size(), "hw:%d", alsaCardId);
    snd_ctl_t *ctl = nullptr;
    if (const int err = snd_ctl_open(&ctl, ctlName.data(), 0); err < 0) {
        dsyslog("vaapivideo/audio: snd_ctl_open(%s) failed: %s", ctlName.data(), snd_strerror(err));
        return;
    }

    snd_ctl_elem_id_t *id = nullptr;
    snd_ctl_elem_id_alloca(&id);
    snd_ctl_elem_id_set_interface(id, SND_CTL_ELEM_IFACE_PCM);
    snd_ctl_elem_id_set_name(id, "IEC958 Playback Default");
    snd_ctl_elem_id_set_index(id, alsaIec958CtlIndex);

    snd_ctl_elem_value_t *val = nullptr;
    snd_ctl_elem_value_alloca(&val);
    snd_ctl_elem_value_set_id(val, id);

    // IEC 60958-3 AES0 bit 1 ("non-audio") gates compressed bitstreams. Always write rather than
    // compare-and-skip: the kernel cache drifts from the actual link state across AVR power-cycle,
    // hotplug, or another process touching IEC958.
    if (const int err = snd_ctl_elem_read(ctl, val); err < 0) {
        dsyslog("vaapivideo/audio: IEC958 read failed (cardId=%d ctlIndex=%u): %s", alsaCardId, alsaIec958CtlIndex,
                snd_strerror(err));
        snd_ctl_close(ctl);
        return;
    }
    const auto aes0 = snd_ctl_elem_value_get_byte(val, 0);
    const auto newAes0 = enable ? static_cast<unsigned char>(aes0 | 0x02U) : static_cast<unsigned char>(aes0 & ~0x02U);
    snd_ctl_elem_value_set_byte(val, 0, newAes0);
    if (const int err = snd_ctl_elem_write(ctl, val); err < 0) {
        dsyslog("vaapivideo/audio: IEC958 write failed (cardId=%d aes0=0x%02x): %s", alsaCardId, aes0,
                snd_strerror(err));
    } else {
        dsyslog("vaapivideo/audio: IEC958 AES0 0x%02x -> 0x%02x (%s)", aes0, newAes0, enable ? "non-audio" : "audio");
    }
    snd_ctl_close(ctl);
}

[[nodiscard]] auto cAudioProcessor::OpenAlsaDevice() -> bool {
    if (alsaDeviceName.empty()) {
        esyslog("vaapivideo/audio: empty device name");
        return false;
    }

    if (alsaHandle) {
        (void)snd_pcm_drop(alsaHandle);
        snd_pcm_close(alsaHandle);
        alsaHandle = nullptr;
    }

    alsaErrorCount.store(0, std::memory_order_relaxed);

    CloseSpdifMuxer();

    const bool wantPassthrough = CanPassthrough(streamParams.codecId);
    const unsigned streamRate = streamParams.sampleRate > 0 ? static_cast<unsigned>(streamParams.sampleRate) : 48000;
    const unsigned alsaRate = ComputeAlsaRate(streamParams.codecId, streamRate, wantPassthrough);
    const unsigned channels =
        (!wantPassthrough && streamParams.channels > 0) ? static_cast<unsigned>(streamParams.channels) : 2;

    constexpr snd_pcm_format_t format = SND_PCM_FORMAT_S16_LE; // fixed; WriteToAlsa() reinterprets as int16_t

    snd_pcm_t *handle = nullptr;
    if (const int err = snd_pcm_open(&handle, alsaDeviceName.c_str(), SND_PCM_STREAM_PLAYBACK, SND_PCM_NONBLOCK);
        err < 0) {
        esyslog("vaapivideo/audio: snd_pcm_open failed for '%s': %s", alsaDeviceName.c_str(), snd_strerror(err));
        return false;
    }

    // Attempt passthrough first; fall back to decoded PCM on any failure (hw config or
    // spdif muxer init). A fresh handle is needed for the PCM fallback because
    // ConfigureAlsaParams() leaves the failed handle in an undefined state.
    if (wantPassthrough) {
        SetIec958NonAudio(true);
        if (ConfigureAlsaParams(handle, format, 2, alsaRate, false)) {
            if (OpenSpdifMuxer(streamParams.codecId, streamParams.sampleRate)) {
                alsaHandle = handle;
                alsaPassthroughActive.store(true, std::memory_order_release);
                isyslog("vaapivideo/audio: opened passthrough @ %uHz on '%s'",
                        alsaSampleRate.load(std::memory_order_relaxed), alsaDeviceName.c_str());
                return true;
            }
            dsyslog("vaapivideo/audio: spdif muxer failed, trying PCM fallback");
        } else {
            dsyslog("vaapivideo/audio: passthrough hw config failed, trying PCM fallback");
        }

        SetIec958NonAudio(false);
        snd_pcm_close(handle);
        handle = nullptr;

        if (const int err = snd_pcm_open(&handle, alsaDeviceName.c_str(), SND_PCM_STREAM_PLAYBACK, SND_PCM_NONBLOCK);
            err < 0) {
            esyslog("vaapivideo/audio: snd_pcm_open failed for PCM fallback: %s", snd_strerror(err));
            return false;
        }

        const unsigned pcmChannels = streamParams.channels > 0 ? static_cast<unsigned>(streamParams.channels) : 2;
        if (ConfigureAlsaParams(handle, format, pcmChannels, streamRate, true)) {
            alsaHandle = handle;
            alsaPassthroughActive.store(false, std::memory_order_release);
            isyslog("vaapivideo/audio: PCM fallback @ %uHz", alsaSampleRate.load(std::memory_order_relaxed));
            return true;
        }
    } else {
        SetIec958NonAudio(false);
        if (ConfigureAlsaParams(handle, format, channels, alsaRate, true)) {
            alsaHandle = handle;
            alsaPassthroughActive.store(false, std::memory_order_release);
            isyslog("vaapivideo/audio: opened PCM @ %uHz on '%s'", alsaSampleRate.load(std::memory_order_relaxed),
                    alsaDeviceName.c_str());
            return true;
        }
    }

    esyslog("vaapivideo/audio: all configuration attempts failed");
    if (handle) {
        snd_pcm_close(handle);
    }
    return false;
}

auto cAudioProcessor::OpenDecoder() -> void {
    CloseDecoder();

    if (streamParams.codecId == AV_CODEC_ID_NONE) {
        return;
    }

    const AVCodec *codec = avcodec_find_decoder(streamParams.codecId);
    if (!codec) {
        esyslog("vaapivideo/audio: no decoder found for %s", avcodec_get_name(streamParams.codecId));
        return;
    }

    AVCodecContext *ctx = avcodec_alloc_context3(codec);
    if (!ctx) {
        esyslog("vaapivideo/audio: avcodec_alloc_context3 failed");
        return;
    }

    decoder.reset(ctx);

    ctx->request_sample_fmt = AV_SAMPLE_FMT_S16;
    ctx->sample_rate = std::max(streamParams.sampleRate, 0);
    // DVB broadcasts often ship marginally non-conforming AC-3/E-AC-3 frames; relax
    // strict compliance gates so the decoder accepts them instead of dropping packets.
    ctx->err_recognition = AV_EF_CAREFUL;
    ctx->strict_std_compliance = FF_COMPLIANCE_UNOFFICIAL;

    if (streamParams.channels > 0) {
        av_channel_layout_default(&ctx->ch_layout, streamParams.channels);
    }

    // FFmpeg requires AV_INPUT_BUFFER_PADDING_SIZE zero bytes after extradata
    // (e.g. AAC AudioSpecificConfig) to allow SIMD overread without a fault.
    if (streamParams.extradata && streamParams.extradataSize > 0) {
        const size_t allocSize = static_cast<size_t>(streamParams.extradataSize) + AV_INPUT_BUFFER_PADDING_SIZE;
        ctx->extradata = static_cast<uint8_t *>(av_malloc(allocSize));
        if (!ctx->extradata) [[unlikely]] {
            esyslog("vaapivideo/audio: av_malloc failed for extradata");
            decoder.reset();
            return;
        }
        std::memcpy(ctx->extradata, streamParams.extradata, static_cast<size_t>(streamParams.extradataSize));
        std::memset(ctx->extradata + streamParams.extradataSize, 0, AV_INPUT_BUFFER_PADDING_SIZE);
        ctx->extradata_size = streamParams.extradataSize;
    }

    if (const int ret = avcodec_open2(ctx, codec, nullptr); ret < 0) {
        esyslog("vaapivideo/audio: avcodec_open2(%s) failed: %s", codec->name, AvErr(ret).data());
        decoder.reset();
        return;
    }

    parserCtx.reset(av_parser_init(streamParams.codecId));
    if (!parserCtx) {
        esyslog("vaapivideo/audio: av_parser_init failed");
    }

    decoderGracePackets = AUDIO_DECODER_GRACE_PACKETS;
    isyslog("vaapivideo/audio: opened %s @ %dHz %dch (%s)", codec->name, ctx->sample_rate, ctx->ch_layout.nb_channels,
            alsaPassthroughActive.load(std::memory_order_relaxed) ? "passthrough" : "PCM");
}

auto cAudioProcessor::ProbeSinkCaps() -> void {
    if (sinkCapsCached && sinkCapsDevice == alsaDeviceName) {
        return;
    }

    sinkCaps = AudioSinkCaps{};
    sinkCapsDevice = alsaDeviceName;
    sinkCapsCached = true;
    alsaCardId = -1;
    alsaIec958CtlIndex = UINT_MAX;

    snd_pcm_t *handle = nullptr;

    // Retry on EBUSY: the PCM device may still be held by a previous open during a fast
    // device-swap. Linear backoff: 100 ms, 200 ms.
    for (int retry = 0; retry < 3; ++retry) {
        const int ret = snd_pcm_open(&handle, alsaDeviceName.c_str(), SND_PCM_STREAM_PLAYBACK, SND_PCM_NONBLOCK);

        if (ret == 0) {
            break;
        }

        if (ret == -EBUSY && retry < 2) {
            dsyslog("vaapivideo/audio: device busy, retry %d/3", retry + 1);
            cCondWait::SleepMs(100 * (retry + 1));
            continue;
        }

        dsyslog("vaapivideo/audio: cannot probe capabilities, PCM-only mode: %s", snd_strerror(ret));
        return;
    }

    snd_pcm_info_t *info = nullptr;
    snd_pcm_info_alloca(&info);

    if (const int err = snd_pcm_info(handle, info); err < 0) {
        dsyslog("vaapivideo/audio: snd_pcm_info failed: %s", snd_strerror(err));
        snd_pcm_close(handle);
        return;
    }

    const int cardId = snd_pcm_info_get_card(info);
    const int deviceId = static_cast<int>(snd_pcm_info_get_device(info));

    alsaCardId = cardId;

    snd_pcm_close(handle);

    const auto ctlName = std::format("hw:{}", cardId);

    snd_ctl_t *ctlRaw = nullptr;
    if (const int err = snd_ctl_open(&ctlRaw, ctlName.c_str(), SND_CTL_READONLY); err < 0) {
        dsyslog("vaapivideo/audio: snd_ctl_open failed for hw:%d: %s", cardId, snd_strerror(err));
    } else {
        const std::unique_ptr<snd_ctl_t, decltype(&snd_ctl_close)> ctl{ctlRaw, snd_ctl_close};

        snd_ctl_elem_id_t *elemId = nullptr;
        snd_ctl_elem_id_alloca(&elemId);
        snd_ctl_elem_id_set_interface(elemId, SND_CTL_ELEM_IFACE_PCM);
        snd_ctl_elem_id_set_name(elemId, "ELD");
        snd_ctl_elem_id_set_device(elemId, static_cast<unsigned>(deviceId));

        snd_ctl_elem_value_t *elemValue = nullptr;
        snd_ctl_elem_value_alloca(&elemValue);

        bool foundValidEld = false;

        // Multi-port HDMI cards expose one ELD per physical port; scan all indices.
        for (unsigned index = 0; index < 8 && !foundValidEld; ++index) {
            snd_ctl_elem_id_set_index(elemId, index);
            snd_ctl_elem_value_set_id(elemValue, elemId);

            if (snd_ctl_elem_read(ctl.get(), elemValue) < 0) {
                continue;
            }

            snd_ctl_elem_info_t *elemInfo = nullptr;
            snd_ctl_elem_info_alloca(&elemInfo);
            snd_ctl_elem_info_set_id(elemInfo, elemId);

            if (snd_ctl_elem_info(ctl.get(), elemInfo) < 0) {
                continue;
            }

            const unsigned eldSize = snd_ctl_elem_info_get_count(elemInfo);
            constexpr unsigned kEldFixedHeader = 20; ///< ELD fixed header (bytes 0-19); see kernel sound/hda/hda_eld.c
            if (eldSize < kEldFixedHeader) {
                dsyslog("vaapivideo/audio: ELD too small (%u bytes) at index %u", eldSize, index);
                continue;
            }

            std::vector<uint8_t> eldBuffer(eldSize);
            for (unsigned i = 0; i < eldSize; ++i) {
                eldBuffer.at(i) = static_cast<uint8_t>(snd_ctl_elem_value_get_byte(elemValue, i));
            }

            // ELD layout (kernel sound/hda/hda_eld.c):
            //   Byte 4 [4:0] = MNL (Monitor Name Length)
            //   Byte 5 [7:4] = SAD count
            //   Bytes 20 .. 20+MNL-1 = monitor name string
            //   Bytes 20+MNL .. = Short Audio Descriptors (3 bytes each)
            const unsigned mnl = eldBuffer.at(4) & 0x1FU;
            const unsigned sadCount = (eldBuffer.at(5) >> 4) & 0x0FU;
            const unsigned sadOffset = kEldFixedHeader + mnl;
            constexpr unsigned kSadSize = 3; ///< CEA-861 SAD is always 3 bytes

            if (sadCount == 0) {
                dsyslog("vaapivideo/audio: ELD found but no SADs (PCM-only sink)");
                foundValidEld = true;
                break;
            }

            if (sadOffset + (sadCount * kSadSize) > eldSize) {
                dsyslog("vaapivideo/audio: truncated ELD (%u SADs @ offset %u need %u bytes, have %u)", sadCount,
                        sadOffset, sadOffset + (sadCount * kSadSize), eldSize);
                continue;
            }

            // Audio Format Codes (AFC): CEA-861-D Table 37 / CTA-861-H Table 38.
            // SAD byte 0 bits [6:3] = AFC. 0x0F = Extended; actual format in byte 2 [7:3] (EAFC).
            constexpr uint8_t kCeaAc3 = 0x02;        ///< Dolby Digital (AC-3)
            constexpr uint8_t kCeaDts = 0x07;        ///< DTS Coherent Acoustics
            constexpr uint8_t kCeaEac3 = 0x0A;       ///< Dolby Digital Plus (E-AC-3)
            constexpr uint8_t kCeaDtshd = 0x0B;      ///< DTS-HD Master Audio
            constexpr uint8_t kCeaTruehd = 0x0C;     ///< Dolby TrueHD (Atmos is a metadata layer on top)
            constexpr uint8_t kCeaExtended = 0x0F;   ///< Extended format escape -- read EAFC from byte 2
            constexpr uint8_t kCeaExtMpegh3d = 0x0B; ///< EAFC: MPEG-H 3D Audio
            constexpr uint8_t kCeaExtAc4 = 0x0C;     ///< EAFC: Dolby AC-4

            for (unsigned i = 0; i < sadCount; ++i) {
                const size_t offset = sadOffset + (i * kSadSize);
                const uint8_t formatCode = (eldBuffer.at(offset) >> 3) & 0x0F;

                switch (formatCode) {
                    case kCeaAc3:
                        sinkCaps.ac3 = true;
                        break;
                    case kCeaDts:
                        sinkCaps.dts = true;
                        break;
                    case kCeaEac3:
                        sinkCaps.eac3 = true;
                        break;
                    case kCeaDtshd:
                        sinkCaps.dtshd = true;
                        break;
                    case kCeaTruehd:
                        sinkCaps.truehd = true;
                        break;
                    case kCeaExtended: {
                        const uint8_t extCode = (eldBuffer.at(offset + 2) >> 3) & 0x1F;
                        if (extCode == kCeaExtMpegh3d) {
                            sinkCaps.mpegh = true;
                        } else if (extCode == kCeaExtAc4) {
                            sinkCaps.ac4 = true;
                        }
                        break;
                    }
                    default:
                        break;
                }
            }

            foundValidEld = true;
        }

        sinkCaps.elded = foundValidEld;
        if (!foundValidEld) {
            dsyslog("vaapivideo/audio: no valid ELD found across all indices");
        }

        snd_ctl_elem_id_t *iecId = nullptr;
        snd_ctl_elem_id_alloca(&iecId);
        snd_ctl_elem_id_set_interface(iecId, SND_CTL_ELEM_IFACE_PCM);
        snd_ctl_elem_id_set_name(iecId, "IEC958 Playback Default");

        snd_ctl_elem_value_t *iecVal = nullptr;
        snd_ctl_elem_value_alloca(&iecVal);

        // Multi-port HDMI cards expose one IEC958 control per port; find ours by scanning indices.
        for (unsigned idx = 0; idx < 16; ++idx) {
            snd_ctl_elem_id_set_index(iecId, idx);
            snd_ctl_elem_value_set_id(iecVal, iecId);
            if (snd_ctl_elem_read(ctl.get(), iecVal) == 0) {
                alsaIec958CtlIndex = idx;
                dsyslog("vaapivideo/audio: IEC958 Playback Default found at index %u", idx);
                break;
            }
        }
        if (alsaIec958CtlIndex == UINT_MAX) {
            dsyslog("vaapivideo/audio: IEC958 Playback Default not found on hw:%d -- non-audio bit will not be managed",
                    cardId);
        }
    }

    constexpr std::array<std::pair<bool AudioSinkCaps::*, std::string_view>, 7> kFormats{
        {{&AudioSinkCaps::ac3, "AC-3"},
         {&AudioSinkCaps::eac3, "E-AC-3"},
         {&AudioSinkCaps::truehd, "TrueHD"},
         {&AudioSinkCaps::dts, "DTS"},
         {&AudioSinkCaps::dtshd, "DTS-HD"},
         {&AudioSinkCaps::ac4, "AC-4"},
         {&AudioSinkCaps::mpegh, "MPEG-H"}}};

    std::string msg = "sink capabilities:";
    bool hasAny = false;
    for (const auto &[member, name] : kFormats) {
        if (sinkCaps.*member) {
            msg += std::format(" {}", name);
            hasAny = true;
        }
    }
    isyslog("vaapivideo/audio: %s%s", msg.c_str(), hasAny ? "" : " PCM-only");
}

[[nodiscard]] auto cAudioProcessor::WritePcmToAlsa(std::span<const uint8_t> data, int64_t startPts90k, unsigned frames,
                                                   uint32_t expectedGeneration) -> bool {
    // Serializes ALSA write + seqlock publish + pcmNextPts advance against CloseDevice() and
    // ResetPlaybackClock(). Recursive: WriteToAlsa's error-recovery re-acquisition nests safely.
    // pcmNextPts advances here (not at call sites) so a gen bump or write failure leaves it intact.
    const cMutexLock lock(mutex.get());

    // Pre-write gate: don't write old-era bytes to a freshly re-prepared ALSA handle.
    if (clearGeneration.load(std::memory_order_acquire) != expectedGeneration) [[unlikely]] {
        return true;
    }
    if (!WriteToAlsa(data)) {
        return false;
    }
    // Bytes queued again (even on the NOPTS/zero-frame early-out below): the ALSA tail is real now.
    outputDropped.store(false, std::memory_order_release);

    const unsigned rate = alsaSampleRate.load(std::memory_order_relaxed);
    if (startPts90k == AV_NOPTS_VALUE || frames == 0U || rate == 0U) {
        return true;
    }

    const int64_t endPts = startPts90k + static_cast<int64_t>((static_cast<uint64_t>(frames) * PTSTICKS) / rate);

    // playbackPts = PTS leaving the DAC = endPts minus the live ring-buffer backlog.
    // If snd_pcm_delay() ever returns suspiciously small values for IEC61937 passthrough
    // (some driver/sink combos don't track the queued IEC frames), playbackPts pins near
    // endPts and GetClock() biases forward, surfacing as a positive rawDelta EMA offset.
    int64_t currentPlaybackPts = endPts;
    snd_pcm_sframes_t delayFrames = 0;
    if (alsaHandle && snd_pcm_delay(alsaHandle, &delayFrames) == 0) {
        delayFrames = std::max<snd_pcm_sframes_t>(delayFrames, 0);
        // Upward clamp: some drivers return stale (multi-second) delays for a few writes
        // after snd_pcm_drop+prepare, pinning playbackPts in the past and freezing video
        // startup until the clock recovers. 2x the configured buffer is ample headroom.
        const auto maxSaneDelay =
            static_cast<snd_pcm_sframes_t>(static_cast<uint64_t>(rate) * AUDIO_ALSA_BUFFER_MS * 2 / 1000);
        if (delayFrames > maxSaneDelay) [[unlikely]] {
            dsyslog("vaapivideo/audio: snd_pcm_delay implausible (%ld frames @ %uHz, cap=%ld) -- clamping",
                    static_cast<long>(delayFrames), rate, static_cast<long>(maxSaneDelay));
            delayFrames = maxSaneDelay;
        }
        const auto delay90k = static_cast<int64_t>((static_cast<uint64_t>(delayFrames) * PTSTICKS) / rate);
        currentPlaybackPts = endPts - delay90k;
    }

    // Single-writer seqlock: mutex already excludes ResetPlaybackClock(). GetClock()
    // retries past the odd sequence so no concurrent reader sees a torn pair.
    const uint64_t nowMs = cTimeMs::Now();
    clockSequence.fetch_add(1, std::memory_order_acq_rel);
    playbackPts.store(currentPlaybackPts, std::memory_order_relaxed);
    lastClockUpdateMs.store(nowMs, std::memory_order_relaxed);
    clockSequence.fetch_add(1, std::memory_order_release);
    // First write after Freeze() re-anchors the clock: lift the pin so GetClock() resumes
    // wall-clock extrapolation on top of the new (now-advancing) playbackPts. Outside the
    // seqlock pair on purpose -- the pin is its own atomic, not part of the (pts, lastMs) tuple.
    clockPaused.store(false, std::memory_order_release);

    // NOPTS continuations of a multi-frame PES inherit endPts as their startPts90k.
    pcmNextPts.store(endPts, std::memory_order_relaxed);

    return true;
}

[[nodiscard]] auto cAudioProcessor::WriteToAlsa(std::span<const uint8_t> data) -> bool {
    if (!alsaHandle || data.empty()) {
        return false;
    }

    const int currentVolume = volume.load(std::memory_order_relaxed);
    const bool passthrough = alsaPassthroughActive.load(std::memory_order_relaxed);
    std::vector<uint8_t> scaledBuffer;

    if (!passthrough && currentVolume != 255) {
        if (currentVolume == 0) {
            scaledBuffer.assign(data.size(), 0);
        } else {
            // Format is always S16LE (ConfigureAlsaParams() locks it), so the int16_t
            // reinterpret is safe. Volume scaling is skipped for passthrough bursts.
            scaledBuffer.resize(data.size());
            const auto *src =
                reinterpret_cast<const int16_t *>(data.data()); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
            auto *dst =
                reinterpret_cast<int16_t *>(scaledBuffer.data()); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
            const size_t samples = data.size() / sizeof(int16_t);

            for (size_t i = 0; i < samples; ++i) {
                dst[i] = static_cast<int16_t>((src[i] * currentVolume) / 255);
            }
        }
        data = std::span<const uint8_t>(scaledBuffer.data(), scaledBuffer.size());
    }

    size_t bpf = alsaFrameBytes.load(std::memory_order_relaxed);
    if (bpf == 0) {
        return false;
    }

    size_t size = data.size();
    const uint8_t *ptr = data.data();
    std::vector<uint8_t> alignedBuffer;

    // ALSA accepts only whole frames. PCM: drop trailing sub-frame bytes (rounding
    // loss is imperceptible at 25 ms periods). Passthrough: zero-pad up -- IEC61937
    // bursts often end mid-frame and truncating would corrupt the bitstream.
    if (size % bpf != 0) {
        if (!passthrough) {
            size -= size % bpf;
            if (size == 0) {
                return true;
            }
        } else {
            const size_t aligned = ((size + bpf - 1) / bpf) * bpf;
            alignedBuffer.assign(ptr, ptr + size);
            alignedBuffer.resize(aligned, 0);
            ptr = alignedBuffer.data();
            size = aligned;
        }
    }

    size_t offset = 0;

    // Four-tier ALSA error recovery:
    //   1. EAGAIN               -> snd_pcm_wait 5 ms (ring full, normal under load)
    //   2. EINTR/EPIPE/ESTRPIPE -> snd_pcm_recover (xruns, suspend)
    //   3. repeated failures    -> close + OpenAlsaDevice (rate-limited to 1 s)
    //   4. reopen failed        -> inline teardown (cannot call Shutdown() from this
    //      thread: Cancel(3) would self-join and deadlock)
    while (offset < size) {
        const auto frames = static_cast<snd_pcm_sframes_t>((size - offset) / bpf);
        const snd_pcm_sframes_t written =
            snd_pcm_writei(alsaHandle, ptr + offset, static_cast<snd_pcm_uframes_t>(frames));

        if (written >= 0) {
            offset += static_cast<size_t>(written) * bpf;
            alsaErrorCount.store(0, std::memory_order_relaxed);
            continue;
        }

        const int err = static_cast<int>(written);

        if (err == -EAGAIN) {
            snd_pcm_wait(alsaHandle, 5);
            continue;
        }

        // ESTRPIPE arrives via <alsa/asoundlib.h>; clang-tidy's IWYU misses that path.
        if ((err == -EINTR || err == -EPIPE || err == -ESTRPIPE) && // NOLINT(misc-include-cleaner)
            snd_pcm_recover(alsaHandle, err, 1) >= 0) {
            continue;
        }

        const int errorCount = alsaErrorCount.fetch_add(1, std::memory_order_relaxed) + 1;
        if (errorCount >= AUDIO_ALSA_ERROR_LIMIT) {
            if (lastReopenAttempt.Elapsed() > 1000) {
                isyslog("vaapivideo/audio: attempting device reopen after %s", snd_strerror(err));
                lastReopenAttempt.Set();
                alsaErrorCount.store(0, std::memory_order_relaxed);

                {
                    const cMutexLock lock(mutex.get());
                    snd_pcm_close(alsaHandle);
                    alsaHandle = nullptr;

                    if (OpenAlsaDevice()) {
                        bpf = alsaFrameBytes.load(std::memory_order_relaxed);
                        if (bpf == 0) {
                            esyslog("vaapivideo/audio: invalid frame size after reopen");
                            // Mark unusable and let the destructor / outer Shutdown() free
                            // handles via CloseDevice() once this thread has returned.
                            ResetPlaybackClock();
                            initialized.store(false, std::memory_order_release);
                            stopping.store(true, std::memory_order_release);
                            packetCondition.Broadcast();
                            return false;
                        }
                        continue;
                    }
                }

                esyslog("vaapivideo/audio: device reopen failed");
            }
            break;
        }

        if (alsaHandle) {
            const cMutexLock lock(mutex.get());
            if (snd_pcm_drop(alsaHandle) == 0 && snd_pcm_prepare(alsaHandle) == 0) {
                continue;
            }
        }

        {
            const cMutexLock lock(mutex.get());
            ResetPlaybackClock();
            initialized.store(false, std::memory_order_release);
        }
        stopping.store(true, std::memory_order_release);
        packetCondition.Broadcast();

        esyslog("vaapivideo/audio: unrecoverable error, device closed: %s", snd_strerror(err));
        return false;
    }

    return true;
}
