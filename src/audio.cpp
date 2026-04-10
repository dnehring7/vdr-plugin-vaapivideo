// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file audio.cpp
 * @brief ALSA audio sink: IEC61937 compressed passthrough with decoded-PCM fallback.
 *
 * Threading model:
 *   - Producer (VDR PES thread): Decode() -> parser -> EnqueuePacket() under `mutex`.
 *   - Consumer (cThread Action()): pops packets, runs DecodeToPcm() or WrapIec61937(),
 *     then WritePcmToAlsa(). The `decoderRefCount` + `clearGeneration` pair lets the
 *     consumer survive concurrent reconfiguration without dragging the producer mutex
 *     into the decode hot path.
 *   - GetClock() is called from the video sync controller on yet another thread; it is
 *     lock-free and relies on release/acquire ordering against WritePcmToAlsa().
 */

#include "audio.h"
#include "common.h"

// ALSA
#include <alsa/asoundlib.h>

// C++ Standard Library
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
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
#include <vdr/thread.h>
#include <vdr/tools.h>
#pragma GCC diagnostic pop

// ============================================================================
// === CONSTANTS ===
// ============================================================================

constexpr int AUDIO_ALSA_ERROR_LIMIT = 5;       ///< Consecutive ALSA write failures that trigger a full device reopen
constexpr uint64_t AUDIO_CLOCK_STALE_MS = 1000; ///< If WritePcmToAlsa() has not refreshed playbackPts within this many
                                                ///< ms, GetClock() returns AV_NOPTS_VALUE so the video sync controller
                                                ///< bypasses the (now stale) audio clock instead of drifting against it
constexpr int AUDIO_DECODER_DRAIN_TIMEOUT_MS =
    200; ///< CloseDecoder() spin-wait ceiling for in-flight DecodeToPcm() callers (ms)
constexpr int AUDIO_DECODER_ERROR_LIMIT = 50;     ///< Consecutive decode failures before flush + parser-reset request
constexpr int AUDIO_DECODER_GRACE_PACKETS = 3;    ///< Decode-error logs suppressed right after (re)init to swallow
                                                  ///< priming garbage from the parser's first frames
constexpr int AUDIO_ERROR_LOG_INTERVAL_MS = 2000; ///< Throttle for repeated decode-error log messages
constexpr size_t AUDIO_QUEUE_CAPACITY = 100;      ///< Audio packet queue depth (~3 s at AC-3 ~32 ms framing)

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
    parserNeedsReset.store(false, std::memory_order_relaxed); // The parser recreate below subsumes any pending request.

    // Defer the actual avcodec_flush_buffers() to the Action() thread; running it here
    // would race against an in-flight DecodeToPcm() that already passed the refcount gate.
    if (decoder) {
        needsFlush.store(true, std::memory_order_release);
    }

    // AVCodecParserContext exposes no flush; recreate from scratch.
    if (parserCtx) {
        av_parser_close(parserCtx.release());
        if (streamParams.codecId != AV_CODEC_ID_NONE) {
            parserCtx.reset(av_parser_init(streamParams.codecId));
        }
    }
}

auto cAudioProcessor::Decode(const uint8_t *data, size_t size, int64_t pts) -> void {
    if (!data || size == 0) [[unlikely]] {
        return;
    }

    const cMutexLock lock(mutex.get());

    if (!parserCtx || !decoder) [[unlikely]] {
        return;
    }

    // Cascade-recovery handoff: the Action thread set parserNeedsReset after the INVALIDDATA
    // limit but cannot recreate the parser itself (would deadlock against CloseDecoder's
    // refcount spin). We are on the producer thread under the mutex, so it is safe here.
    // Bump clearGeneration so queued packets emitted by the corrupt parser fragment are
    // dropped by the Action thread instead of fed to the still-fine decoder.
    if (parserNeedsReset.exchange(false, std::memory_order_acquire)) {
        dsyslog("vaapivideo/audio: cascade recovery -- recreating parser for %s",
                avcodec_get_name(streamParams.codecId));
        av_parser_close(parserCtx.release());
        if (streamParams.codecId != AV_CODEC_ID_NONE) {
            parserCtx.reset(av_parser_init(streamParams.codecId));
        }
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
        pts = AV_NOPTS_VALUE;

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
    // PTS at the DAC output: endPts - snd_pcm_delay, refreshed on each ALSA write (~24-32 ms).
    //
    // Stale-clock guard: if WritePcmToAlsa() stops firing (decoder error after a channel
    // switch with stale TS, codec swap mid-stream) playbackPts freezes. The video sync
    // controller would treat the frozen value as authoritative and let drift balloon to
    // multi-second levels. Returning AV_NOPTS_VALUE forces the freerun branch -- video
    // keeps playing unpaced and lipsync re-anchors at the next valid clock.
    //
    // Load order matters: WritePcmToAlsa() stores playbackPts THEN lastClockUpdateMs
    // (release on each). Loading lastMs first guarantees a fresh stamp pairs with the
    // matching pts on the second load -- the opposite order would race.
    const uint64_t lastMs = lastClockUpdateMs.load(std::memory_order_acquire);
    const int64_t pts = playbackPts.load(std::memory_order_acquire);
    if (lastMs == 0) {
        // Never written (post-construct or post-Clear()). No usable clock yet.
        return AV_NOPTS_VALUE;
    }
    if (pts == AV_NOPTS_VALUE) {
        return AV_NOPTS_VALUE;
    }
    if (cTimeMs::Now() - lastMs > AUDIO_CLOCK_STALE_MS) {
        return AV_NOPTS_VALUE;
    }
    return pts;
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
        // Different device requested: close current device but keep the processing thread
        // alive to skip the create/destroy round-trip.
        CloseDevice();
    }

    alsaDeviceName = std::string(alsaDevice);

    ProbeSinkCaps();

    if (!OpenAlsaDevice()) {
        esyslog("vaapivideo/audio: failed to open ALSA device");
        CloseDevice();
        alsaDeviceName.clear();
        return false;
    }

    initialized.store(true, std::memory_order_release);
    stopping.store(false, std::memory_order_release);
    hasExited.store(false, std::memory_order_relaxed);
    // Start() must come after stopping=false / hasExited=false above; a stale stopping flag
    // from a prior Shutdown() would otherwise cause Action() to exit immediately.
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

[[nodiscard]] auto cAudioProcessor::IsQueueFull() const -> bool {
    const cMutexLock lock(mutex.get());
    return packetQueue.size() >= AUDIO_QUEUE_CAPACITY;
}

[[nodiscard]] auto cAudioProcessor::OpenCodec(AVCodecID codecId, int sampleRate, int channels) -> bool {
    if (sampleRate <= 0 || channels <= 0) [[unlikely]] {
        esyslog("vaapivideo/audio: invalid parameters (%dHz %dch)", sampleRate, channels);
        return false;
    }

    SetStreamParams({.channels = channels, .codecId = codecId, .sampleRate = sampleRate});
    return true;
}

auto cAudioProcessor::SetStreamParams(const AudioStreamParams &params) -> void {
    if (params.sampleRate <= 0 || params.channels <= 0) [[unlikely]] {
        esyslog("vaapivideo/audio: invalid stream parameters");
        return;
    }

    const cMutexLock lock(mutex.get());

    if (streamParams.codecId == params.codecId && streamParams.sampleRate == params.sampleRate &&
        streamParams.channels == params.channels) {
        return;
    }

    const AVCodecID oldCodecId = streamParams.codecId;

    ProbeSinkCaps();

    while (!packetQueue.empty()) {
        const std::unique_ptr<AVPacket, FreeAVPacket> dropped{packetQueue.front()};
        packetQueue.pop();
    }

    streamParams = params;

    const bool wantPassthrough = CanPassthrough(params.codecId);
    const auto targetRate = ComputeAlsaRate(params.codecId, static_cast<unsigned>(params.sampleRate), wantPassthrough);

    // Reopen ALSA only when the hardware format changes. Passthrough is always 2 ch.
    const bool currentlyPassthrough = alsaPassthroughActive.load(std::memory_order_relaxed);
    const bool needsReconfig =
        !initialized.load(std::memory_order_relaxed) || currentlyPassthrough != wantPassthrough ||
        (wantPassthrough ? targetRate != alsaSampleRate
                         : (targetRate != alsaSampleRate || params.channels != static_cast<int>(alsaChannels)));

    // Bump clearGeneration BEFORE tearing down the old decoder/parser. The Action thread
    // may be holding an already-dequeued packet whose bytes belong to the old codec era;
    // the bump marks it stale so DecodeToPcm()/passthrough drops it instead of feeding the
    // freshly opened decoder. Clear() also bumps; this covers callers that bypass Clear().
    if (needsReconfig || oldCodecId != params.codecId) {
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
            return;
        }
        OpenDecoder();
    } else if (oldCodecId != params.codecId) {
        // Same hw config, codec changed only: drop in-flight PCM so GetClock() doesn't
        // report delay belonging to the previous codec's queued samples.
        if (alsaHandle) {
            (void)snd_pcm_drop(alsaHandle);
            (void)snd_pcm_prepare(alsaHandle);
        }
        ResetPlaybackClock();
        CloseDecoder();
        OpenDecoder();
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
    // Idempotent: stop the processing thread, then close the device. The wasStopping
    // early-return funnels destructor / explicit caller / re-init through one path with
    // no double-stop or double-close. (Initialize's reopen path skips this and calls
    // CloseDevice() directly so the thread keeps running.)
    const bool wasStopping = stopping.exchange(true, std::memory_order_release);
    dsyslog("vaapivideo/audio: shutting down (wasStopping=%d)", wasStopping);
    if (wasStopping) {
        return;
    }

    packetCondition.Broadcast();
    Cancel(3);

    const cTimeMs timeout(SHUTDOWN_TIMEOUT_MS);
    while (!hasExited.load(std::memory_order_acquire) && !timeout.TimedOut()) {
        packetCondition.Broadcast();
        cCondWait::SleepMs(10);
    }

    CloseDevice();
}

auto cAudioProcessor::CloseDevice() -> void {
    // Releases ALSA + decoder + spdif muxer; resets device-side fields to post-construct
    // state. Does NOT touch the processing thread -- caller ensures the thread is either
    // stopped (Shutdown) or will safely re-discover the new config on the next Decode()
    // call (Initialize re-open).
    const cMutexLock lock(mutex.get());

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
    parserNeedsReset.store(false, std::memory_order_relaxed); // CloseDecoder() above already destroyed the parser.
    alsaPassthroughActive.store(false, std::memory_order_release);
    alsaFrameBytes.store(0, std::memory_order_release);
    alsaChannels = 0;
    alsaSampleRate = 0;
    initialized.store(false, std::memory_order_release);
}

auto cAudioProcessor::DrainPacketQueue() -> void {
    // Caller must hold mutex. FreeAVPacket frees each packet on scope exit.
    while (!packetQueue.empty()) {
        const std::unique_ptr<AVPacket, FreeAVPacket> dropped{packetQueue.front()};
        packetQueue.pop();
    }
}

auto cAudioProcessor::ResetPlaybackClock() -> void {
    // GetClock()'s stale-clock guard treats lastClockUpdateMs == 0 as "no clock yet", so
    // it must be cleared alongside playbackPts on every flush/reopen. pcm{Next,QueueEnd}Pts
    // re-anchor the next ALSA write onto the new timeline.
    playbackPts.store(AV_NOPTS_VALUE, std::memory_order_relaxed);
    lastClockUpdateMs.store(0, std::memory_order_release);
    pcmNextPts = AV_NOPTS_VALUE;
    pcmQueueEndPts = AV_NOPTS_VALUE;
}

// ============================================================================
// === THREAD ===
// ============================================================================

auto cAudioProcessor::Action() -> void {
    isyslog("vaapivideo/audio: processing thread started");

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
            passthrough = alsaPassthroughActive.load(std::memory_order_relaxed);
            // Snapshot clearGeneration under the mutex so it pairs atomically with the pop.
            // DecodeToPcm()/passthrough re-check it later to drop packets whose bytes belong
            // to a parser/codec era that was swapped out between dequeue and decode.
            generationAtDequeue = clearGeneration.load(std::memory_order_relaxed);
        }

        const int64_t pts = packet->pts;
        if (pts != AV_NOPTS_VALUE) {
            // >5 s PTS jump (channel switch, seek, wrap): drop queued audio so the next
            // WritePcmToAlsa() re-anchors playbackPts cleanly on the new timeline.
            if (pcmNextPts != AV_NOPTS_VALUE) {
                const int64_t diff = (pts > pcmNextPts) ? (pts - pcmNextPts) : (pcmNextPts - pts);
                if (diff > (5 * 90000)) {
                    const cMutexLock lock(mutex.get());
                    if (alsaHandle) {
                        (void)snd_pcm_drop(alsaHandle);
                        (void)snd_pcm_prepare(alsaHandle);
                    }
                    pcmQueueEndPts = AV_NOPTS_VALUE;
                    playbackPts.store(AV_NOPTS_VALUE, std::memory_order_relaxed);
                    lastClockUpdateMs.store(0, std::memory_order_release);
                }
            }
            pcmNextPts = pts;
        }

        if (passthrough) {
            // Same staleness check as the PCM path: spdifMuxCtx may have been torn down
            // by an OpenAlsaDevice() reconfiguration between dequeue and now.
            if (clearGeneration.load(std::memory_order_acquire) != generationAtDequeue) {
                continue;
            }
            const auto burst = WrapIec61937(packet->data, packet->size);
            if (!burst.empty()) {
                const size_t bpf = alsaFrameBytes.load(std::memory_order_relaxed);
                const unsigned burstFrames = (bpf > 0) ? static_cast<unsigned>(burst.size() / bpf) : 0;
                (void)WritePcmToAlsa(burst, pcmNextPts, burstFrames);
            }
        } else {
            (void)DecodeToPcm(std::span(packet->data, static_cast<size_t>(packet->size)), pts, generationAtDequeue);
        }
    }

    hasExited.store(true, std::memory_order_release);
    isyslog("vaapivideo/audio: processing thread stopped");
}

// ============================================================================
// === INTERNAL METHODS ===
// ============================================================================

[[nodiscard]] auto cAudioProcessor::CanPassthrough(AVCodecID codecId) const -> bool {
    switch (codecId) {
        case AV_CODEC_ID_AC3:
            return sinkCaps.ac3;
        case AV_CODEC_ID_EAC3:
            return sinkCaps.eac3;
        case AV_CODEC_ID_TRUEHD:
            return sinkCaps.truehd;
        case AV_CODEC_ID_DTS:
            return sinkCaps.dts || sinkCaps.dtshd; // DTS-HD-capable sinks always decode the DTS core layer.
        case AV_CODEC_ID_AC4:
            return sinkCaps.ac4;
        case AV_CODEC_ID_MPEGH_3D_AUDIO:
            return sinkCaps.mpegh3d;
        default:
            return false;
    }
}

auto cAudioProcessor::CloseDecoder() -> void {
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
    if (swrCtx) {
        swr_free(&swrCtx);
    }
    swrChannels = 0;
    swrFormat = AV_SAMPLE_FMT_NONE;
    consecutiveDecodeErrors = 0;
    decoderGracePackets = AUDIO_DECODER_GRACE_PACKETS;
}

[[nodiscard]] auto cAudioProcessor::ComputeAlsaRate(AVCodecID codecId, unsigned streamRate, bool passthrough) const
    -> unsigned {
    if (!passthrough) {
        return streamRate;
    }
    // IEC61937 carrier rate: AC-3/DTS go 1x, but DD+/AC-4/MPEG-H bursts must be clocked
    // at 4x the underlying sample rate so the sink reassembles them correctly.
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

    if (snd_pcm_nonblock(handle, 1) < 0) {
        esyslog("vaapivideo/audio: cannot set non-blocking mode");
        return false;
    }

    if (snd_pcm_hw_params_any(handle, hwParams) < 0) {
        esyslog("vaapivideo/audio: cannot initialize hardware parameters");
        return false;
    }

    if (snd_pcm_hw_params_set_access(handle, hwParams, SND_PCM_ACCESS_RW_INTERLEAVED) < 0) {
        esyslog("vaapivideo/audio: interleaved access not supported");
        return false;
    }

    if (snd_pcm_hw_params_set_format(handle, hwParams, format) < 0) {
        esyslog("vaapivideo/audio: format %s not supported", snd_pcm_format_name(format));
        return false;
    }

    if (snd_pcm_hw_params_set_channels(handle, hwParams, channels) < 0) {
        esyslog("vaapivideo/audio: %uch configuration failed", channels);
        return false;
    }

    snd_pcm_hw_params_set_rate_resample(handle, hwParams, allowResample ? 1 : 0);

    unsigned actualRate = rate;
    if (snd_pcm_hw_params_set_rate(handle, hwParams, actualRate, 0) < 0) {
        if (!allowResample || snd_pcm_hw_params_set_rate_near(handle, hwParams, &actualRate, nullptr) < 0) {
            esyslog("vaapivideo/audio: %uHz sample rate not supported", rate);
            return false;
        }
    }

    if (actualRate != rate && !allowResample) {
        esyslog("vaapivideo/audio: rate mismatch (%u requested, %u actual)", rate, actualRate);
        return false;
    }

    // 25 ms period keeps WritePcmToAlsa() refreshes frequent enough for the A/V sync clock;
    // 200 ms buffer leaves headroom against scheduler jitter.
    auto bufferSize = static_cast<snd_pcm_uframes_t>(actualRate / 5);  // 200 ms
    auto periodSize = static_cast<snd_pcm_uframes_t>(actualRate / 40); // 25 ms

    if (snd_pcm_hw_params_set_buffer_size_near(handle, hwParams, &bufferSize) < 0 ||
        snd_pcm_hw_params_set_period_size_near(handle, hwParams, &periodSize, nullptr) < 0 ||
        snd_pcm_hw_params(handle, hwParams) < 0 || snd_pcm_prepare(handle) < 0) {
        esyslog("vaapivideo/audio: hardware parameter configuration failed");
        return false;
    }

    snd_pcm_sw_params_t *swParams = nullptr;
    snd_pcm_sw_params_alloca(&swParams);

    if (snd_pcm_sw_params_current(handle, swParams) < 0) {
        esyslog("vaapivideo/audio: cannot get software parameters");
        return false;
    }

    // Start playback once half the buffer is filled: trades ~100 ms of initial latency
    // for headroom against a startup underrun.
    const snd_pcm_uframes_t startThreshold = bufferSize / 2;
    if (snd_pcm_sw_params_set_start_threshold(handle, swParams, startThreshold) < 0 ||
        snd_pcm_sw_params_set_avail_min(handle, swParams, periodSize) < 0 || snd_pcm_sw_params(handle, swParams) < 0) {
        esyslog("vaapivideo/audio: software parameter configuration failed");
        return false;
    }

    snd_pcm_hw_params_get_channels(hwParams, &alsaChannels);
    snd_pcm_hw_params_get_rate(hwParams, &alsaSampleRate, nullptr);

    const auto frameBytes = snd_pcm_frames_to_bytes(handle, 1);
    if (frameBytes <= 0) {
        esyslog("vaapivideo/audio: invalid frame size");
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

    // Bumping decoderRefCount makes a subsequent CloseDecoder() spin until we release it.
    // Acquire pairs with the generation re-check below.
    decoderRefCount.fetch_add(1, std::memory_order_acquire);

    // Re-check decoder pointer AND generation. The refcount only blocks *concurrent*
    // teardown -- a teardown that already completed (with a fresh OpenDecoder afterwards)
    // before we incremented would slip through. The generation marker, bumped by Clear()
    // and SetStreamParams() before any swap, catches the case where our packet was
    // produced by the previous codec era's parser; sending it to the new decoder would
    // trigger an INVALIDDATA cascade. Drop instead.
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
            // Cascade self-recovery: sustained INVALIDDATA means the FFmpeg parser (owned
            // by the producer thread in Decode(), not by us) is stuck on a corrupt partial
            // frame from an upstream PES discontinuity, and every emitted packet is bad.
            // Flushing the decoder alone is insufficient -- the parser keeps producing the
            // same garbage. Defer the parser recreate to Decode()'s next entry, where it
            // runs safely under the producer mutex. We cannot do it here: we hold
            // decoderRefCount > 0 and would deadlock CloseDecoder()'s spin-wait.
            parserNeedsReset.store(true, std::memory_order_release);
        }

        decoderRefCount.fetch_sub(1, std::memory_order_release);
        return true; // Soft failure: decoder may recover on a later packet.
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

        const unsigned outCh = alsaChannels > 0 ? alsaChannels : 2;
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

            uint8_t *outPtr = convertedBuffer.data(); // NOLINT(misc-const-correctness)
            const int converted =
                swr_convert(swrCtx, &outPtr, maxOutSamples,
                            const_cast<const uint8_t **>(frame->data), // NOLINT(cppcoreguidelines-pro-type-const-cast)
                            frame->nb_samples);

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
        // swr conversion), giving SetStreamParams()/Clear() a window to race in. Drop the
        // frame so stale PCM never reaches ALSA.
        if (clearGeneration.load(std::memory_order_acquire) != expectedGeneration) [[unlikely]] {
            decoderRefCount.fetch_sub(1, std::memory_order_release);
            return true;
        }

        const int64_t startPts90k = pcmNextPts;
        const bool writeOk = WritePcmToAlsa(std::span(pcmData, pcmSize), startPts90k, sampleCount);
        av_frame_unref(frame.get());

        if (!writeOk) {
            decoderRefCount.fetch_sub(1, std::memory_order_release);
            return false;
        }

        if (startPts90k != AV_NOPTS_VALUE && alsaSampleRate > 0U) {
            const auto added90k =
                static_cast<int64_t>((static_cast<uint64_t>(sampleCount) * 90000ULL) / alsaSampleRate);
            pcmNextPts = startPts90k + added90k;
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
            // Throttle to one log per 500 ms; sustained overload would otherwise flood syslog.
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

/// AVIO sink for the FFmpeg spdif muxer: appends IEC61937 burst bytes into spdifOutputBuf.
static auto SpdifWriteCallback(void *opaque, const uint8_t *buf, int bufSize) -> int {
    auto &output = *static_cast<std::vector<uint8_t> *>(opaque);
    output.insert(output.end(), buf, buf + bufSize);
    return bufSize;
}

auto cAudioProcessor::OpenSpdifMuxer(AVCodecID codecId, int sampleRate) -> bool {
    CloseSpdifMuxer();

    if (avformat_alloc_output_context2(&spdifMuxCtx, nullptr, "spdif", nullptr) < 0 || !spdifMuxCtx) {
        esyslog("vaapivideo/audio: failed to allocate spdif muxer context");
        return false;
    }

    constexpr int kIoBufSize = 32768; // AVIO scratch buffer; size not burst-critical (SpdifWriteCallback drains it).
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

    AVStream *const stream = avformat_new_stream(spdifMuxCtx, nullptr); // NOLINT(misc-const-correctness)
    if (!stream) {
        CloseSpdifMuxer();
        return false;
    }

    stream->codecpar->codec_type = AVMEDIA_TYPE_AUDIO;
    stream->codecpar->codec_id = codecId;
    stream->codecpar->sample_rate = sampleRate;
    stream->codecpar->ch_layout.nb_channels = 2;

    if (avformat_write_header(spdifMuxCtx, nullptr) < 0) {
        esyslog("vaapivideo/audio: spdif muxer write_header failed for %s", avcodec_get_name(codecId));
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

auto cAudioProcessor::SetIec958NonAudio(bool enable) -> void {
    if (alsaCardId < 0) {
        return;
    }

    const auto ctlName = std::format("hw:{}", alsaCardId);
    snd_ctl_t *ctl = nullptr;
    if (snd_ctl_open(&ctl, ctlName.c_str(), 0) < 0) {
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

    if (snd_ctl_elem_read(ctl, val) == 0) {
        auto aes0 = snd_ctl_elem_value_get_byte(val, 0);
        // IEC 60958-3 channel-status bit 1 ("non-audio") in AES0 is mandatory for compressed
        // bitstreams; without it the receiver attempts PCM playback and emits noise.
        const unsigned char newAes0 =
            enable ? static_cast<unsigned char>(aes0 | 0x02U) : static_cast<unsigned char>(aes0 & ~0x02U);
        if (newAes0 != aes0) {
            snd_ctl_elem_value_set_byte(val, 0, newAes0);
            snd_ctl_elem_write(ctl, val);
            dsyslog("vaapivideo/audio: IEC958 AES0 0x%02x -> 0x%02x (%s)", aes0, newAes0,
                    enable ? "non-audio" : "audio");
        }
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

    constexpr snd_pcm_format_t format = SND_PCM_FORMAT_S16_LE;

    snd_pcm_t *handle = nullptr;
    if (snd_pcm_open(&handle, alsaDeviceName.c_str(), SND_PCM_STREAM_PLAYBACK, SND_PCM_NONBLOCK) < 0) {
        esyslog("vaapivideo/audio: snd_pcm_open failed for '%s'", alsaDeviceName.c_str());
        return false;
    }

    // Try IEC61937 passthrough first when the sink advertises support; on any failure
    // (hw config, spdif muxer init), fall back to a decoded PCM open on a fresh handle.
    if (wantPassthrough) {
        SetIec958NonAudio(true);
        if (ConfigureAlsaParams(handle, format, 2, alsaRate, false)) {
            if (OpenSpdifMuxer(streamParams.codecId, streamParams.sampleRate)) {
                alsaHandle = handle;
                alsaPassthroughActive.store(true, std::memory_order_release);
                isyslog("vaapivideo/audio: opened passthrough @ %uHz on '%s'", alsaSampleRate, alsaDeviceName.c_str());
                return true;
            }
            dsyslog("vaapivideo/audio: spdif muxer failed, trying PCM fallback");
        } else {
            dsyslog("vaapivideo/audio: passthrough hw config failed, trying PCM fallback");
        }

        SetIec958NonAudio(false);
        snd_pcm_close(handle);
        handle = nullptr;

        if (snd_pcm_open(&handle, alsaDeviceName.c_str(), SND_PCM_STREAM_PLAYBACK, SND_PCM_NONBLOCK) < 0) {
            esyslog("vaapivideo/audio: snd_pcm_open failed for PCM fallback");
            return false;
        }

        const unsigned pcmChannels = streamParams.channels > 0 ? static_cast<unsigned>(streamParams.channels) : 2;
        if (ConfigureAlsaParams(handle, format, pcmChannels, streamRate, true)) {
            alsaHandle = handle;
            alsaPassthroughActive.store(false, std::memory_order_release);
            isyslog("vaapivideo/audio: PCM fallback @ %uHz", alsaSampleRate);
            return true;
        }
    } else {
        SetIec958NonAudio(false);
        if (ConfigureAlsaParams(handle, format, channels, alsaRate, true)) {
            alsaHandle = handle;
            alsaPassthroughActive.store(false, std::memory_order_release);
            isyslog("vaapivideo/audio: opened PCM @ %uHz on '%s'", alsaSampleRate, alsaDeviceName.c_str());
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
    // DVB broadcasts ship marginally non-conforming AC-3/E-AC-3 frames; loosen the strict
    // gates so the decoder accepts them instead of dropping the whole packet.
    ctx->err_recognition = AV_EF_CAREFUL;
    ctx->strict_std_compliance = FF_COMPLIANCE_UNOFFICIAL;

    if (streamParams.channels > 0) {
        av_channel_layout_default(&ctx->ch_layout, streamParams.channels);
    }

    // FFmpeg requires AV_INPUT_BUFFER_PADDING_SIZE zero bytes after extradata
    // (e.g. AAC AudioSpecificConfig) to satisfy SIMD overread.
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

    if (avcodec_open2(ctx, codec, nullptr) < 0) {
        esyslog("vaapivideo/audio: avcodec_open2() failed");
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

    sinkCaps = HdmiSinkCaps{};
    sinkCapsDevice = alsaDeviceName;
    sinkCapsCached = true;

    snd_pcm_t *handle = nullptr;

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

        dsyslog("vaapivideo/audio: cannot probe capabilities, PCM-only mode");
        return;
    }

    snd_pcm_info_t *info = nullptr;
    snd_pcm_info_alloca(&info);

    if (snd_pcm_info(handle, info) < 0) {
        dsyslog("vaapivideo/audio: snd_pcm_info failed");
        snd_pcm_close(handle);
        return;
    }

    const int cardId = snd_pcm_info_get_card(info);
    const int deviceId = static_cast<int>(snd_pcm_info_get_device(info));

    alsaCardId = cardId;

    snd_pcm_close(handle);

    const auto ctlName = std::format("hw:{}", cardId);

    snd_ctl_t *ctlRaw = nullptr;
    if (snd_ctl_open(&ctlRaw, ctlName.c_str(), SND_CTL_READONLY) < 0) {
        dsyslog("vaapivideo/audio: snd_ctl_open failed for hw:%d", cardId);
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

        // Scan ELD indices: multi-port HDMI cards expose one ELD per port.
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
            constexpr unsigned kEldFixedHeader = 20; ///< Fixed ELD header size (bytes 0-19)
            if (eldSize < kEldFixedHeader) {
                dsyslog("vaapivideo/audio: ELD too small (%u bytes) at index %u", eldSize, index);
                continue;
            }

            std::vector<uint8_t> eldBuffer(eldSize);
            for (unsigned i = 0; i < eldSize; ++i) {
                eldBuffer.at(i) = static_cast<uint8_t>(snd_ctl_elem_value_get_byte(elemValue, i));
            }

            // ELD layout (see sound/hda/hda_eld.c in the kernel):
            //   Byte 4 [4:0] = MNL (Monitor Name Length)
            //   Byte 5 [7:4] = SAD count
            //   Bytes 20 .. 20+MNL-1 = monitor name
            //   Bytes 20+MNL .. = Short Audio Descriptors (3 bytes each)
            const unsigned mnl = eldBuffer.at(4) & 0x1FU;
            const unsigned sadCount = (eldBuffer.at(5) >> 4) & 0x0FU;
            const unsigned sadOffset = kEldFixedHeader + mnl;
            constexpr unsigned kSadSize = 3;

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

            // Audio Format Codes from CEA-861-D Table 37 / CTA-861-H Table 38.
            // SAD byte 0 bits [6:3] = Audio Format Code (AFC).
            // AFC 0x0F is the "Extended" escape; the actual format is in SAD byte 2 [7:3] (EAFC).
            constexpr uint8_t kCeaAc3 = 0x02;        ///< Dolby Digital (AC-3)
            constexpr uint8_t kCeaDts = 0x07;        ///< DTS Coherent Acoustics
            constexpr uint8_t kCeaEac3 = 0x0A;       ///< Dolby Digital Plus (E-AC-3)
            constexpr uint8_t kCeaDtshd = 0x0B;      ///< DTS-HD Master Audio
            constexpr uint8_t kCeaTruehd = 0x0C;     ///< Dolby TrueHD (Atmos rides on top as a metadata layer)
            constexpr uint8_t kCeaExtended = 0x0F;   ///< Extended format -- read EAFC
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
                            sinkCaps.mpegh3d = true;
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

        if (!foundValidEld) {
            dsyslog("vaapivideo/audio: no valid ELD found across all indices");
        }

        snd_ctl_elem_id_t *iecId = nullptr;
        snd_ctl_elem_id_alloca(&iecId);
        snd_ctl_elem_id_set_interface(iecId, SND_CTL_ELEM_IFACE_PCM);
        snd_ctl_elem_id_set_name(iecId, "IEC958 Playback Default");

        snd_ctl_elem_value_t *iecVal = nullptr;
        snd_ctl_elem_value_alloca(&iecVal);

        // Multi-port HDMI cards expose one IEC958 control per port; find ours by index.
        for (unsigned idx = 0; idx < 16; ++idx) {
            snd_ctl_elem_id_set_index(iecId, idx);
            snd_ctl_elem_value_set_id(iecVal, iecId);
            if (snd_ctl_elem_read(ctl.get(), iecVal) == 0) {
                alsaIec958CtlIndex = idx;
                dsyslog("vaapivideo/audio: IEC958 Playback Default found at index %u", idx);
                break;
            }
        }
    }

    constexpr std::array<std::pair<bool HdmiSinkCaps::*, std::string_view>, 7> kFormats{
        {{&HdmiSinkCaps::ac3, "AC-3"},
         {&HdmiSinkCaps::eac3, "E-AC-3"},
         {&HdmiSinkCaps::truehd, "TrueHD"},
         {&HdmiSinkCaps::dts, "DTS"},
         {&HdmiSinkCaps::dtshd, "DTS-HD"},
         {&HdmiSinkCaps::ac4, "AC-4"},
         {&HdmiSinkCaps::mpegh3d, "MPEG-H"}}};

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

[[nodiscard]] auto cAudioProcessor::WritePcmToAlsa(std::span<const uint8_t> data, int64_t startPts90k, unsigned frames)
    -> bool {
    if (!WriteToAlsa(data)) {
        return false;
    }

    const unsigned rate = alsaSampleRate;
    if (startPts90k == AV_NOPTS_VALUE || frames == 0U || rate == 0U) {
        return true;
    }

    const int64_t endPts = startPts90k + static_cast<int64_t>((static_cast<uint64_t>(frames) * 90000ULL) / rate);
    pcmQueueEndPts = endPts;

    // playbackPts represents the PTS leaving the DAC: endPts minus the live ring-buffer
    // backlog reported by snd_pcm_delay.
    snd_pcm_sframes_t delayFrames = 0;
    if (alsaHandle && snd_pcm_delay(alsaHandle, &delayFrames) == 0) {
        delayFrames = std::max<snd_pcm_sframes_t>(delayFrames, 0);
        const auto delay90k = static_cast<int64_t>((static_cast<uint64_t>(delayFrames) * 90000ULL) / rate);
        playbackPts.store(endPts - delay90k, std::memory_order_release);
    } else {
        playbackPts.store(endPts, std::memory_order_release);
    }
    // Stamp lastClockUpdateMs *after* playbackPts: GetClock() loads the timestamp first,
    // so a fresh stamp here guarantees a matching pts on its second load.
    lastClockUpdateMs.store(cTimeMs::Now(), std::memory_order_release);

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
            // S16LE only: ConfigureAlsaParams() locks the format, so the int16_t reinterpret
            // is safe. Volume scaling is skipped entirely in passthrough (compressed bursts).
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

    // ALSA writes whole frames only. PCM: drop trailing sub-frame bytes (rounding loss is
    // imperceptible). Passthrough: zero-pad up -- IEC61937 bursts often end mid-frame but
    // truncating would corrupt the bitstream, so the next-frame slack is benign.
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
    //   1. EAGAIN              -> snd_pcm_wait 5 ms (buffer full, normal under load)
    //   2. EINTR/EPIPE/ESTRPIPE -> snd_pcm_recover (xruns, suspends)
    //   3. repeated failures   -> close + OpenAlsaDevice (rate-limited to 1 s)
    //   4. reopen failed       -> Shutdown(): the audio path is unusable.
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

        if ((err == -EINTR || err == -EPIPE || err == -ESTRPIPE) && // NOLINT(misc-include-cleaner)
            snd_pcm_recover(alsaHandle, err, 1) >= 0) {
            continue;
        }

        const int errorCount = alsaErrorCount.fetch_add(1, std::memory_order_relaxed) + 1;
        if (errorCount >= AUDIO_ALSA_ERROR_LIMIT) {
            if (lastReopenAttempt.Elapsed() > 1000) {
                dsyslog("vaapivideo/audio: attempting device reopen");
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
                            Shutdown();
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
            Shutdown();
        }

        esyslog("vaapivideo/audio: unrecoverable error, device closed");
        return false;
    }

    return true;
}
