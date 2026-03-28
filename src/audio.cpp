// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file audio.cpp
 * @brief ALSA output with IEC61937 passthrough and PCM fallback
 */

#include "audio.h"
#include "common.h"

// Platform
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

constexpr int AUDIO_ALSA_ERROR_LIMIT = 5; ///< Consecutive ALSA write failures that trigger a full device reopen
constexpr int AUDIO_DECODER_DRAIN_TIMEOUT_MS =
    200;                                       ///< Max wait for in-flight decode calls before destroying context (ms)
constexpr int AUDIO_DECODER_ERROR_LIMIT = 50;  ///< Consecutive decode failures before automatic decoder flush
constexpr int AUDIO_DECODER_GRACE_PACKETS = 3; ///< Packets silently discarded after decoder (re)init
constexpr int AUDIO_ERROR_LOG_INTERVAL_MS = 2000; ///< Minimum interval between repeated decode-error log messages
constexpr size_t AUDIO_QUEUE_CAPACITY = 500;      ///< Maximum audio packet queue depth (~10 s at 50 Hz delivery rate)

// ============================================================================
// === AUDIO PROCESSOR CLASS ===
// ============================================================================

cAudioProcessor::cAudioProcessor() : cThread("vaapivideo/audio"), mutex(std::make_unique<cMutex>()) {}

cAudioProcessor::~cAudioProcessor() noexcept {
    dsyslog("vaapivideo/audio: destructor called (stopping=%d)", stopping.load(std::memory_order_relaxed));
    Stop();
}

// ============================================================================
// === PUBLIC API ===
// ============================================================================

auto cAudioProcessor::Clear() -> void {
    // snd_pcm_drop() + prepare() discards pending audio and resets the ALSA state machine.
    const cMutexLock lock(mutex.get());

    if (alsaHandle) {
        (void)snd_pcm_drop(alsaHandle);
        (void)snd_pcm_prepare(alsaHandle);
    }

    alsaErrorCount.store(0, std::memory_order_relaxed);
    playbackPts.store(AV_NOPTS_VALUE, std::memory_order_relaxed);
    pcmQueueEndPts = AV_NOPTS_VALUE;
    pcmNextPts = AV_NOPTS_VALUE;

    // Bump generation so in-flight DecodeToPcm() calls discard stale clock writes.
    clearGeneration.fetch_add(1, std::memory_order_release);

    while (!packetQueue.empty()) {
        const std::unique_ptr<AVPacket, FreeAVPacket> dropped{packetQueue.front()};
        packetQueue.pop();
    }

    // Deferred flush via needsFlush flag: the actual avcodec_flush_buffers() runs on the Action() thread to avoid
    // racing with DecodeToPcm(). Residual frames after seek can crash swr_convert() if not flushed.
    if (decoder) {
        needsFlush.store(true, std::memory_order_release);
    }

    if (parserCtx) {
        av_parser_close(parserCtx.release());
        if (streamParams.codecId != AV_CODEC_ID_NONE) {
            parserCtx.reset(av_parser_init(streamParams.codecId));
        }
    }
}

auto cAudioProcessor::Decode(const uint8_t *data, size_t size, int64_t pts) -> void {
    // av_parser_parse2() reassembles into complete access units. PTS is supplied only on the first call.
    if (!data || size == 0) [[unlikely]] {
        return;
    }

    // Protects parserCtx/decoder against concurrent Clear()/SetStreamParams(). VDR cMutex is recursive.
    const cMutexLock lock(mutex.get());

    if (!parserCtx || !decoder) [[unlikely]] {
        return;
    }

    const uint8_t *currentData = data;
    int remainingSize = static_cast<int>(size);

    while (remainingSize > 0) {
        uint8_t *parsedData = nullptr;
        int parsedSize = 0;

        // consumed = bytes eaten from input; parsedSize = bytes in complete access unit (zero if more input needed).
        const int consumed = av_parser_parse2(parserCtx.get(), decoder.get(), &parsedData, &parsedSize, currentData,
                                              remainingSize, pts, AV_NOPTS_VALUE, 0);

        if (consumed < 0) {
            break; // Malformed data -- drop remainder
        }

        currentData += consumed;
        remainingSize -= consumed;
        pts = AV_NOPTS_VALUE; // Parser owns PTS from here; only supply it at stream entry

        if (parsedSize > 0 && parsedData) {
            // parsedData points into the parser's internal buffer -- EnqueuePacket() clones it.
            AVPacket pkt{};
            pkt.data = parsedData;
            pkt.size = parsedSize;
            pkt.pts = parserCtx->pts;
            pkt.dts = parserCtx->dts;
            (void)EnqueuePacket(&pkt); // Drop-safe: parser state unaffected if queue is full
        }
    }
}

[[nodiscard]] auto cAudioProcessor::GetClock() const noexcept -> int64_t {
    // clock = pcmQueueEndPts - ALSA ring-buffer delay (in 90 kHz ticks).
    // Falls back to cached playbackPts when the device is closed or no PTS has been anchored.
    const unsigned rate = alsaSampleRate;

    if (!alsaHandle || rate == 0) {
        return playbackPts.load(std::memory_order_relaxed);
    }

    // Read delay BEFORE endPts: if the audio thread writes between the two reads, the clock appears slightly
    // ahead (harmless). The reverse order caused backward jumps triggering sync overshoot.
    snd_pcm_sframes_t delayFrames = 0;
    if (snd_pcm_delay(alsaHandle, &delayFrames) != 0 || delayFrames < 0) {
        return playbackPts.load(std::memory_order_relaxed);
    }

    const int64_t endPts = pcmQueueEndPts;
    if (endPts == AV_NOPTS_VALUE) {
        return playbackPts.load(std::memory_order_relaxed);
    }

    const auto delay90k = static_cast<int64_t>((static_cast<uint64_t>(delayFrames) * 90000ULL) / rate);
    return endPts - delay90k;
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
        Shutdown();
    }

    alsaDeviceName = std::string(alsaDevice);

    ProbeSinkCaps();

    if (!OpenAlsaDevice()) {
        esyslog("vaapivideo/audio: failed to open ALSA device");
        Shutdown();
        alsaDeviceName.clear();
        return false;
    }

    initialized.store(true, std::memory_order_release);
    stopping.store(false, std::memory_order_release);
    hasExited.store(false, std::memory_order_relaxed);
    // Must start after 'initialized' is set; otherwise the thread may exit before any packets arrive.
    Start();

    isyslog("vaapivideo/audio: initialized on '%.*s'", static_cast<int>(alsaDevice.size()), alsaDevice.data());
    return true;
}

[[nodiscard]] auto cAudioProcessor::IsInitialized() const noexcept -> bool {
    return initialized.load(std::memory_order_relaxed);
}

[[nodiscard]] auto cAudioProcessor::IsQueueFull() const -> bool {
    const cMutexLock lock(mutex.get());
    return packetQueue.size() >= AUDIO_QUEUE_CAPACITY;
}

[[nodiscard]] auto cAudioProcessor::OpenCodec(AVCodecID codecId, int sampleRate, int channels) -> bool {
    if (sampleRate <= 0 || channels <= 0) [[unlikely]] {
        esyslog("vaapivideo/audio: invalid parameters (rate=%d, ch=%d)", sampleRate, channels);
        return false;
    }

    SetStreamParams({.channels = channels, .codecId = codecId, .sampleRate = sampleRate});
    return true;
}

auto cAudioProcessor::SetStreamParams(const AudioStreamParams &params) -> void {
    // Entire reconfiguration runs under the mutex to prevent half-configured state.
    if (params.sampleRate <= 0 || params.channels <= 0) [[unlikely]] {
        esyslog("vaapivideo/audio: invalid stream parameters");
        return;
    }

    dsyslog("vaapivideo/audio: stream params - codec=%s, rate=%d, ch=%d", avcodec_get_name(params.codecId),
            params.sampleRate, params.channels);

    const cMutexLock lock(mutex.get());

    // Skip when nothing changed to avoid unnecessary ALSA teardowns.
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

    dsyslog("vaapivideo/audio: codec %s -> %s mode (rate=%u)", avcodec_get_name(params.codecId),
            wantPassthrough ? "passthrough" : "PCM", targetRate);

    // Reopen ALSA only when the hardware format changes. Passthrough always uses 2 IEC958 channels.
    const bool needsReconfig =
        !initialized.load(std::memory_order_relaxed) || alsaPassthroughActive != wantPassthrough ||
        (wantPassthrough ? targetRate != alsaSampleRate
                         : (targetRate != alsaSampleRate || params.channels != static_cast<int>(alsaChannels)));

    if (needsReconfig) {
        // Shut down the decoder before closing ALSA -- may still be mid-frame.
        if (oldCodecId != AV_CODEC_ID_NONE && !alsaPassthroughActive) {
            CloseDecoder();
        }

        // Close handle before reopening. VDR cMutex is recursive, so OpenAlsaDevice() is safe to call here.
        if (alsaHandle) {
            (void)snd_pcm_drop(alsaHandle);
            snd_pcm_close(alsaHandle);
            alsaHandle = nullptr;
        }

        if (!OpenAlsaDevice()) {
            esyslog("vaapivideo/audio: reconfiguration failed");
            return;
        }

        // Decoder/parser needed for framing even in IEC61937 passthrough mode.
        OpenDecoder();
    } else if (oldCodecId != params.codecId) {
        dsyslog("vaapivideo/audio: codec changed, reinitializing decoder");

        // Flush stale PCM so GetClock() does not report an inflated delay from the previous codec's samples.
        if (alsaHandle) {
            (void)snd_pcm_drop(alsaHandle);
            (void)snd_pcm_prepare(alsaHandle);
        }
        playbackPts.store(AV_NOPTS_VALUE, std::memory_order_relaxed);
        pcmQueueEndPts = AV_NOPTS_VALUE;
        pcmNextPts = AV_NOPTS_VALUE;

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
    const cMutexLock lock(mutex.get());

    CloseDecoder();

    while (!packetQueue.empty()) {
        const std::unique_ptr<AVPacket, FreeAVPacket> dropped{packetQueue.front()};
        packetQueue.pop();
    }

    if (alsaHandle) {
        (void)snd_pcm_drop(alsaHandle);
        snd_pcm_close(alsaHandle);
        alsaHandle = nullptr;
    }

    alsaErrorCount.store(0, std::memory_order_relaxed);
    playbackPts.store(AV_NOPTS_VALUE, std::memory_order_relaxed);
    pcmQueueEndPts = AV_NOPTS_VALUE;
    pcmNextPts = AV_NOPTS_VALUE;
    alsaPassthroughActive = false;
    alsaFrameBytes = 0;
    alsaChannels = 0;
    alsaSampleRate = 0;
    initialized.store(false, std::memory_order_release);
}

auto cAudioProcessor::Stop() -> void {
    // Signal + wake + cancel. Shutdown() alone would not terminate the Action() thread.
    const bool wasStopping = stopping.exchange(true, std::memory_order_release);
    dsyslog("vaapivideo/audio: Stop() (wasStopping=%d)", wasStopping);

    if (wasStopping) {
        dsyslog("vaapivideo/audio: already stopped, skipping");
        return;
    }

    packetCondition.Broadcast();
    Cancel(3);

    const cTimeMs timeout(SHUTDOWN_TIMEOUT_MS);
    while (!hasExited.load(std::memory_order_acquire) && !timeout.TimedOut()) {
        packetCondition.Broadcast();
        cCondWait::SleepMs(10);
    }

    Shutdown();

    dsyslog("vaapivideo/audio: stopped");
}

// ============================================================================
// === THREAD ===
// ============================================================================

auto cAudioProcessor::Action() -> void {
    isyslog("vaapivideo/audio: processing thread started");

    while (!stopping.load(std::memory_order_acquire)) {
        std::unique_ptr<AVPacket, FreeAVPacket> packet;
        bool passthrough = false;

        {
            const cMutexLock lock(mutex.get());

            // 100ms backstop against missed Broadcast() wakeups.
            while (packetQueue.empty() && !stopping.load(std::memory_order_acquire)) {
                packetCondition.TimedWait(*mutex, 100);
            }

            if (stopping.load(std::memory_order_acquire)) {
                break;
            }

            if (!packetQueue.empty()) {
                packet.reset(packetQueue.front());
                packetQueue.pop();
            }

            // Snapshot under lock; streamParams may be written by another thread.
            passthrough = CanPassthrough(streamParams.codecId);
        }
        // Lock released; EnqueuePacket() not blocked during slow I/O.

        if (!packet || !packet->data || packet->size <= 0) [[unlikely]] {
            continue;
        }

        // WriteToAlsa() handles error tracking internally.
        (void)(passthrough ? WriteToAlsa(std::span(packet->data, static_cast<size_t>(packet->size)))
                           : DecodeToPcm(std::span(packet->data, static_cast<size_t>(packet->size)), packet->pts));
    }

    hasExited.store(true, std::memory_order_release);
    isyslog("vaapivideo/audio: processing thread stopped");
}

// ============================================================================
// === INTERNAL METHODS ===
// ============================================================================

[[nodiscard]] auto cAudioProcessor::CanPassthrough(AVCodecID codecId) const -> bool {
    // AAC, HE-AAC, and AAC-LATM are never passed through -- IEC61937 has no standardized framing for them.
    switch (codecId) {
        case AV_CODEC_ID_AC3:
            return sinkCaps.ac3;
        case AV_CODEC_ID_EAC3:
            return sinkCaps.eac3;
        case AV_CODEC_ID_TRUEHD:
            return sinkCaps.truehd;
        case AV_CODEC_ID_DTS:
            return sinkCaps.dts || sinkCaps.dtshd; // DTS-HD sinks also decode the DTS core layer
        case AV_CODEC_ID_AC4:
            return sinkCaps.ac4;
        case AV_CODEC_ID_MPEGH_3D_AUDIO:
            return sinkCaps.mpegh3d;
        default:
            return false;
    }
}

auto cAudioProcessor::CloseDecoder() -> void {
    // Wait for in-flight DecodeToPcm() calls to finish before destroying the context.
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

    // DD+/AC-4/MPEG-H require 4x IEC61937 clock (192 kHz for 48 kHz stream); AC-3/DTS use 1x.
    const bool needsQuadRate =
        (codecId == AV_CODEC_ID_EAC3 || codecId == AV_CODEC_ID_AC4 || codecId == AV_CODEC_ID_MPEGH_3D_AUDIO);
    return needsQuadRate ? streamRate * 4 : streamRate;
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

    // 200ms buffer / 25ms period: keeps A/V sync within one video frame at 25 fps.
    auto bufferSize = static_cast<snd_pcm_uframes_t>(actualRate / 5);  // 200ms
    auto periodSize = static_cast<snd_pcm_uframes_t>(actualRate / 40); // 25ms

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

    // Auto-start at half-full (100ms pre-fill) to prevent first-period underrun.
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

    alsaFrameBytes = static_cast<size_t>(frameBytes);
    alsaPassthroughActive = (format == SND_PCM_FORMAT_IEC958_SUBFRAME_LE);

    return true;
}

[[nodiscard]] auto cAudioProcessor::DecodeToPcm(std::span<const uint8_t> data, int64_t pts) -> bool {
    if (!decoder) {
        return false;
    }

    // Snapshot generation to detect concurrent Clear() and suppress stale clock writes.
    const uint32_t myGeneration = clearGeneration.load(std::memory_order_acquire);

    // Guard decoder lifetime; CloseDecoder() waits for refcount to drop before destroying the context.
    decoderRefCount.fetch_add(1, std::memory_order_relaxed);

    // Re-check: CloseDecoder() may have completed between the null check and the fetch_add above.
    if (!decoder) {
        decoderRefCount.fetch_sub(1, std::memory_order_release);
        return false;
    }

    // Deferred flush from Clear() -- must run on this thread to avoid racing with avcodec_send_packet.
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
        // Grace period: the first few packets after codec reinit are often partial and fail to decode.
        if (decoderGracePackets > 0) {
            --decoderGracePackets;
        } else if (lastDecodeErrorLog.Elapsed() > AUDIO_ERROR_LOG_INTERVAL_MS) {
            dsyslog("vaapivideo/audio: avcodec_send_packet failed: %s (consecutive=%d)", AvErr(sendRet).data(),
                    consecutiveDecodeErrors);
            lastDecodeErrorLog.Set();
        }

        // Sustained failures: flush decoder to recover from corrupt state.
        ++consecutiveDecodeErrors;
        if (consecutiveDecodeErrors >= AUDIO_DECODER_ERROR_LIMIT) {
            dsyslog("vaapivideo/audio: %d consecutive decode failures, flushing decoder", consecutiveDecodeErrors);
            FlushDecoderState();
        }

        decoderRefCount.fetch_sub(1, std::memory_order_release);
        return true; // Soft failure -- decoder may recover on next packet
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

        // Re-create swrCtx if format/channels change (e.g. after decoder flush on seek).
        if (frame->format != AV_SAMPLE_FMT_S16) {
            const auto frameFmt = static_cast<AVSampleFormat>(frame->format);
            const int frameCh = frame->ch_layout.nb_channels;

            if (swrCtx && (frameFmt != swrFormat || frameCh != swrChannels)) {
                dsyslog("vaapivideo/audio: frame format changed (%s %dch -> %s %dch), recreating swresample",
                        av_get_sample_fmt_name(swrFormat), swrChannels, av_get_sample_fmt_name(frameFmt), frameCh);
                swr_free(&swrCtx);
            }

            if (!swrCtx) {
                AVChannelLayout outLayout{};
                av_channel_layout_default(&outLayout, static_cast<int>(outCh));

                const int ret = swr_alloc_set_opts2(&swrCtx, &outLayout, AV_SAMPLE_FMT_S16, frame->sample_rate,
                                                    &frame->ch_layout, frameFmt, frame->sample_rate, 0, nullptr);

                if (ret < 0 || !swrCtx || swr_init(swrCtx) < 0) {
                    esyslog("vaapivideo/audio: swr_alloc_set_opts2 failed for %s %dch -> S16 %uch conversion",
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
                dsyslog("vaapivideo/audio: initialized swresample for %s %dch -> S16 %uch conversion",
                        av_get_sample_fmt_name(frameFmt), frameCh, outCh);
            }
        }

        const uint8_t *pcmData = nullptr;
        size_t pcmSize = 0;
        std::vector<uint8_t> convertedBuffer;

        if (frame->format != AV_SAMPLE_FMT_S16 && swrCtx) {
            const int outSamples = frame->nb_samples;
            const size_t bufferSize = static_cast<size_t>(outSamples) * outCh * 2;
            convertedBuffer.resize(bufferSize);

            uint8_t *outPtr = convertedBuffer.data(); // NOLINT(misc-const-correctness)
            const int converted =
                swr_convert(swrCtx, &outPtr, outSamples,
                            const_cast<const uint8_t **>(frame->data), // NOLINT(cppcoreguidelines-pro-type-const-cast)
                            frame->nb_samples);

            if (converted < 0) [[unlikely]] {
                esyslog("vaapivideo/audio: swr_convert failed");
                decoderRefCount.fetch_sub(1, std::memory_order_release);
                return false;
            }

            pcmSize = static_cast<size_t>(converted) * outCh * 2;
            pcmData = convertedBuffer.data();
        } else {
            pcmSize = static_cast<size_t>(frame->nb_samples) * static_cast<size_t>(frame->ch_layout.nb_channels) * 2;
            pcmData = frame->data[0];
        }

        const unsigned framesOut = (frame->format != AV_SAMPLE_FMT_S16 && swrCtx)
                                       ? static_cast<unsigned>(pcmSize / (outCh * 2U))
                                       : static_cast<unsigned>(frame->nb_samples);

        // Maintain a 90 kHz PCM timeline from parser PTS (FFmpeg frame->pts uses a codec-dependent timebase).
        if (pts != AV_NOPTS_VALUE) {
            if (pcmNextPts == AV_NOPTS_VALUE) {
                pcmNextPts = pts;
            } else {
                const int64_t diff = (pts > pcmNextPts) ? (pts - pcmNextPts) : (pcmNextPts - pts);
                if (diff > (5 * 90000)) {
                    // Discontinuity (channel change, stream reset): flush queued audio and re-anchor.
                    {
                        const cMutexLock lock(mutex.get());
                        if (alsaHandle) {
                            (void)snd_pcm_drop(alsaHandle);
                            (void)snd_pcm_prepare(alsaHandle);
                        }
                    }

                    pcmQueueEndPts = AV_NOPTS_VALUE;
                    playbackPts.store(AV_NOPTS_VALUE, std::memory_order_relaxed);
                    pcmNextPts = pts;
                }
            }
        }

        // Skip ALSA write if Clear() ran since dequeue to prevent stale PTS from poisoning the clock.
        if (clearGeneration.load(std::memory_order_acquire) != myGeneration) [[unlikely]] {
            decoderRefCount.fetch_sub(1, std::memory_order_release);
            return true;
        }

        const int64_t startPts90k = pcmNextPts;
        const bool writeOk = WritePcmToAlsa(std::span(pcmData, pcmSize), startPts90k, framesOut);
        av_frame_unref(frame.get());

        if (!writeOk) {
            decoderRefCount.fetch_sub(1, std::memory_order_release);
            return false;
        }

        // Advance PCM timeline by the duration queued, using the ALSA sample rate for delay conversion.
        if (!alsaPassthroughActive && startPts90k != AV_NOPTS_VALUE && alsaSampleRate > 0U) {
            const auto added90k = static_cast<int64_t>((static_cast<uint64_t>(framesOut) * 90000ULL) /
                                                       static_cast<uint64_t>(alsaSampleRate));
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
            // Throttle syslog to once per 500ms during sustained overload.
            if (lastQueueWarn.Elapsed() > 500) {
                esyslog("vaapivideo/audio: queue full (%zu packets), dropping (codec=%s rate=%d ch=%d passthrough=%s)",
                        packetQueue.size(), avcodec_get_name(streamParams.codecId), streamParams.sampleRate,
                        streamParams.channels, alsaPassthroughActive ? "yes" : "no");
                lastQueueWarn.Set();
            }
            return false;
        }

        packetQueue.push(packet.release());
    }

    packetCondition.Broadcast();
    return true;
}

[[nodiscard]] auto cAudioProcessor::OpenAlsaDevice() -> bool {
    // Passthrough: IEC958_SUBFRAME_LE 2ch. PCM: S16_LE with decoded channel count. Falls back to PCM on failure.
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

    const bool wantPassthrough = CanPassthrough(streamParams.codecId);
    const unsigned streamRate = streamParams.sampleRate > 0 ? static_cast<unsigned>(streamParams.sampleRate) : 48000;
    const unsigned alsaRate = ComputeAlsaRate(streamParams.codecId, streamRate, wantPassthrough);
    const unsigned channels =
        (!wantPassthrough && streamParams.channels > 0) ? static_cast<unsigned>(streamParams.channels) : 2;
    const snd_pcm_format_t format = wantPassthrough ? SND_PCM_FORMAT_IEC958_SUBFRAME_LE : SND_PCM_FORMAT_S16_LE;

    snd_pcm_t *handle = nullptr;
    if (snd_pcm_open(&handle, alsaDeviceName.c_str(), SND_PCM_STREAM_PLAYBACK, SND_PCM_NONBLOCK) < 0) {
        esyslog("vaapivideo/audio: snd_pcm_open failed for '%s'", alsaDeviceName.c_str());
        return false;
    }

    if (ConfigureAlsaParams(handle, format, channels, alsaRate, !wantPassthrough)) {
        alsaHandle = handle;
        isyslog("vaapivideo/audio: opened %s @ %uHz on '%s'", alsaPassthroughActive ? "passthrough" : "PCM",
                alsaSampleRate, alsaDeviceName.c_str());
        return true;
    }

    // Passthrough failed -- try PCM fallback.
    if (wantPassthrough) {
        dsyslog("vaapivideo/audio: passthrough failed, trying PCM fallback");
        snd_pcm_close(handle);
        handle = nullptr;

        if (snd_pcm_open(&handle, alsaDeviceName.c_str(), SND_PCM_STREAM_PLAYBACK, SND_PCM_NONBLOCK) >= 0) {
            const unsigned pcmChannels = streamParams.channels > 0 ? static_cast<unsigned>(streamParams.channels) : 2;
            if (ConfigureAlsaParams(handle, SND_PCM_FORMAT_S16_LE, pcmChannels, streamRate, true)) {
                alsaHandle = handle;
                isyslog("vaapivideo/audio: PCM fallback @ %uHz", alsaSampleRate);
                OpenDecoder();
                return true;
            }
        }
    }

    esyslog("vaapivideo/audio: all configuration attempts failed");
    if (handle) {
        snd_pcm_close(handle);
    }
    return false;
}

auto cAudioProcessor::OpenDecoder() -> void {
    // Wait for in-flight DecodeToPcm() calls before replacing the decoder context.
    for (const cTimeMs start;
         decoderRefCount.load(std::memory_order_acquire) > 0 && start.Elapsed() < AUDIO_DECODER_DRAIN_TIMEOUT_MS;) {
        cCondWait::SleepMs(1);
    }

    decoder.reset();
    if (swrCtx) {
        swr_free(&swrCtx);
    }
    swrChannels = 0;
    swrFormat = AV_SAMPLE_FMT_NONE;

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
    // Accept unofficial / slightly non-conforming EAC3 / AC3 bitstreams from DVB broadcasts.
    ctx->err_recognition = AV_EF_CAREFUL;
    ctx->strict_std_compliance = FF_COMPLIANCE_UNOFFICIAL;

    if (streamParams.channels > 0) {
        av_channel_layout_default(&ctx->ch_layout, streamParams.channels);
    }

    // Some codecs (AAC, HE-AAC) carry out-of-band config ("extradata") that must be provided before any frames
    // arrive. AV_INPUT_BUFFER_PADDING_SIZE zero bytes are appended to prevent overreads in FFmpeg's bitstream reader.
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

    avcodec_flush_buffers(ctx);
    // Grace period: the decoder needs at least one complete access unit before producing output.
    decoderGracePackets = AUDIO_DECODER_GRACE_PACKETS;
    isyslog("vaapivideo/audio: decoder initialized - %s @ %dHz %dch", codec->name, ctx->sample_rate,
            ctx->ch_layout.nb_channels);
}

auto cAudioProcessor::ProbeSinkCaps() -> void {
    // Probe HDMI ELD (EDID-Like Data) for CEA-861 Short Audio Descriptors. Falls back to PCM-only when unavailable.
    if (sinkCapsCached && sinkCapsDevice == alsaDeviceName) {
        return; // Already probed; cached result is still valid
    }

    sinkCaps = HdmiSinkCaps{};
    sinkCapsDevice = alsaDeviceName;
    sinkCapsCached = true;

    snd_pcm_t *handle = nullptr;

    // Retry on EBUSY: the device may be transiently in use during plugin start-up.
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

    snd_pcm_hw_params_t *params = nullptr;
    snd_pcm_hw_params_alloca(&params);

    // No point reading ELD if IEC958_SUBFRAME_LE is not supported at the hardware level.
    const bool iec958Supported = snd_pcm_hw_params_any(handle, params) >= 0 &&
                                 snd_pcm_hw_params_test_format(handle, params, SND_PCM_FORMAT_IEC958_SUBFRAME_LE) >= 0;

    snd_pcm_close(handle);

    if (!iec958Supported) {
        dsyslog("vaapivideo/audio: IEC958 not supported on hw:%d,%d", cardId, deviceId);
        return;
    }

    // Read ELD via the ALSA control interface (not the PCM device).
    const auto ctlName = std::format("hw:{}", cardId);

    snd_ctl_t *ctlRaw = nullptr;
    if (snd_ctl_open(&ctlRaw, ctlName.c_str(), SND_CTL_READONLY) < 0) {
        dsyslog("vaapivideo/audio: snd_ctl_open failed for hw:%d", cardId);
    } else {
        const std::unique_ptr<snd_ctl_t, decltype(&snd_ctl_close)> ctl{ctlRaw, snd_ctl_close};

        // Set up ELD element ID.
        snd_ctl_elem_id_t *elemId = nullptr;
        snd_ctl_elem_id_alloca(&elemId);
        snd_ctl_elem_id_set_interface(elemId, SND_CTL_ELEM_IFACE_PCM);
        snd_ctl_elem_id_set_name(elemId, "ELD");
        snd_ctl_elem_id_set_device(elemId, static_cast<unsigned>(deviceId));

        snd_ctl_elem_value_t *elemValue = nullptr;
        snd_ctl_elem_value_alloca(&elemValue);

        bool foundValidEld = false;

        // Scan ELD indices (drivers may expose multiple instances).
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
            if (eldSize < 21) {
                dsyslog("vaapivideo/audio: ELD too small (%u bytes) at index %u", eldSize, index);
                continue;
            }

            std::vector<uint8_t> eldBuffer(eldSize);
            for (unsigned i = 0; i < eldSize; ++i) {
                eldBuffer.at(i) = static_cast<uint8_t>(snd_ctl_elem_value_get_byte(elemValue, i));
            }

            // Parse CEA-861 Short Audio Descriptors (SADs).
            // ELD byte 20 bits [7:4] = SAD count; each SAD is 3 bytes starting at offset 84.
            // SAD byte 0 bits [6:3] = Audio Format Code (AFC).
            const unsigned sadCount = (eldBuffer.at(20) >> 4) & 0x0F;
            constexpr unsigned kSadOffset = 84; ///< Byte offset of first SAD in the ELD
            constexpr unsigned kSadSize = 3;    ///< Bytes per Short Audio Descriptor

            if (sadCount == 0) {
                dsyslog("vaapivideo/audio: ELD found but no SADs (PCM-only sink)");
                foundValidEld = true;
                break;
            }

            if (kSadOffset + (sadCount * kSadSize) > eldSize) {
                dsyslog("vaapivideo/audio: truncated ELD (%u SADs need %u bytes, have %u)", sadCount,
                        kSadOffset + (sadCount * kSadSize), eldSize);
                continue;
            }

            // Audio Format Codes from CEA-861-D Table 37 / CTA-861-H Table 38. AFC 0x0F is the "Extended" escape;
            // the actual format is given by the EAFC in SAD byte 2 [7:3].
            constexpr uint8_t kCeaAc3 = 0x02;        ///< Dolby Digital (AC-3)
            constexpr uint8_t kCeaDts = 0x07;        ///< DTS Coherent Acoustics
            constexpr uint8_t kCeaEac3 = 0x0A;       ///< Dolby Digital Plus (E-AC-3)
            constexpr uint8_t kCeaDtshd = 0x0B;      ///< DTS-HD Master Audio
            constexpr uint8_t kCeaTruehd = 0x0C;     ///< Dolby TrueHD / Atmos
            constexpr uint8_t kCeaExtended = 0x0F;   ///< Extended format -- read EAFC
            constexpr uint8_t kCeaExtMpegh3d = 0x0B; ///< EAFC: MPEG-H 3D Audio
            constexpr uint8_t kCeaExtAc4 = 0x0C;     ///< EAFC: Dolby AC-4

            for (unsigned i = 0; i < sadCount; ++i) {
                const size_t offset = kSadOffset + (i * kSadSize);
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

    // Passthrough: ALSA "frame" is an IEC958 sub-frame, not an audio sample; clock stays at AV_NOPTS_VALUE.
    if (alsaPassthroughActive) {
        return true;
    }

    const unsigned rate = alsaSampleRate;
    if (startPts90k == AV_NOPTS_VALUE || frames == 0U || rate == 0U) {
        return true;
    }

    // Single-writer (Action thread); GetClock() reads atomically.
    int64_t endPts = pcmQueueEndPts;

    // PTS jump > 5 s indicates a discontinuity -- re-anchor the timeline.
    if (endPts != AV_NOPTS_VALUE) {
        const int64_t diff = (startPts90k > endPts) ? (startPts90k - endPts) : (endPts - startPts90k);
        if (diff > (5 * 90000)) {
            endPts = startPts90k;
        }
    } else {
        endPts = startPts90k; // First write: anchor the PCM timeline
    }

    // Advance end-of-queue PTS by the duration written: frames / sampleRate [s] * 90000 [ticks/s].
    endPts += static_cast<int64_t>((static_cast<uint64_t>(frames) * 90000ULL) / rate);
    pcmQueueEndPts = endPts;

    // Compute the playback clock: endPts minus the ALSA ring-buffer delay.
    snd_pcm_sframes_t delayFrames = 0;
    if (alsaHandle && snd_pcm_delay(alsaHandle, &delayFrames) == 0) {
        delayFrames = std::max<snd_pcm_sframes_t>(delayFrames, 0);
        const auto delay90k = static_cast<int64_t>((static_cast<uint64_t>(delayFrames) * 90000ULL) / rate);
        playbackPts.store(endPts - delay90k, std::memory_order_relaxed);
    } else {
        playbackPts.store(endPts, std::memory_order_relaxed);
    }

    return true;
}

[[nodiscard]] auto cAudioProcessor::WriteToAlsa(std::span<const uint8_t> data) -> bool {
    // Software volume scaling, frame alignment, and three-tier error recovery (EAGAIN -> recover -> reopen).
    if (!alsaHandle || data.empty()) {
        return false;
    }

    const int currentVolume = volume.load(std::memory_order_relaxed);
    std::vector<uint8_t> scaledBuffer;

    // Passthrough is untouched -- modifying compressed bitstream would corrupt IEC61937 sync words.
    if (!alsaPassthroughActive && currentVolume != 255) {
        if (currentVolume == 0) {
            scaledBuffer.assign(data.size(), 0); // Muted: replace with silence
        } else {
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
        data = scaledBuffer;
    }

    size_t bpf = 0;
    {
        const cMutexLock lock(mutex.get());
        bpf = alsaFrameBytes;
        if (bpf == 0) {
            return false;
        }
    }

    size_t size = data.size();
    const uint8_t *ptr = data.data();
    std::vector<uint8_t> alignedBuffer;

    if (size % bpf != 0) {
        if (!alsaPassthroughActive) {
            // PCM: truncate partial frames -- partial samples are meaningless.
            size -= size % bpf;
            if (size == 0) {
                return true;
            }
        } else {
            // Passthrough: zero-pad to frame boundary -- receivers expect aligned IEC61937 bursts.
            const size_t aligned = ((size + bpf - 1) / bpf) * bpf;
            alignedBuffer.assign(ptr, ptr + size);
            alignedBuffer.resize(aligned, 0);
            ptr = alignedBuffer.data();
            size = aligned;
        }
    }

    size_t offset = 0;

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
            // Ring buffer full -- normal backpressure, not an error.
            snd_pcm_wait(alsaHandle, 5);
            continue;
        }

        if ((err == -EINTR || err == -EPIPE || err == -ESTRPIPE) && // NOLINT(misc-include-cleaner)
            snd_pcm_recover(alsaHandle, err, 1) >= 0) {
            // -EPIPE: underrun (too slow). -ESTRPIPE: suspended (e.g. HDMI CEC standby).
            continue;
        }

        // Only count truly unrecoverable errors (not EAGAIN or recovered underruns).
        const int errorCount = alsaErrorCount.fetch_add(1, std::memory_order_relaxed) + 1;

        // Tier 3: repeated failures -- close and reopen the PCM device entirely.
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
                        bpf = alsaFrameBytes;
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

        // Drop and re-prepare as a final recovery step before giving up.
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
