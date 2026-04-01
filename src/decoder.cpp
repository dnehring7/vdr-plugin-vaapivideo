// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file decoder.cpp
 * @brief Threaded VAAPI decoder with filter graph and A/V sync
 *
 * Pipeline:  EnqueueData() -> packet queue -> VAAPI decode -> filter graph -> A/V sync -> display
 * Filters:   [bwdif|deinterlace] -> [hqdn3d|denoise] -> scale (BT.709 NV12) -> [sharpen] (probed per GPU)
 * Sync:      Audio-clock master; poll-wait alignment, +-500 ms dead zone, 1500 ms re-sync
 */

#include "decoder.h"
#include "audio.h"
#include "common.h"
#include "config.h"
#include "device.h"
#include "display.h"

// C++ Standard Library
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <memory>
#include <string>
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
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/avutil.h>
#include <libavutil/buffer.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/mem.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
}
#pragma GCC diagnostic pop

// VAAPI
#include <va/va.h>

// VDR
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <vdr/thread.h>
#include <vdr/tools.h>
#pragma GCC diagnostic pop

// ============================================================================
// === CONSTANTS ===
// ============================================================================

constexpr size_t DECODER_QUEUE_CAPACITY = 500;     ///< Video packet queue depth (~10 s at 50 fps)
constexpr int DECODER_SUBMIT_TIMEOUT_MS = 100;     ///< Timeout when submitting a frame to the display (ms)
constexpr int64_t DECODER_SYNC_RESYNC = 1500 * 90; ///< Full Phase-4 re-sync threshold (PTS ticks, ~1500 ms)
constexpr int64_t DECODER_SYNC_CORRECT = 100 * 90; ///< Phase-3 gentle correction threshold (~100 ms)
constexpr int64_t DECODER_SYNC_CONVERGE = 50 * 90; ///< Phase-4 convergence band (~50 ms)
constexpr int DECODER_SYNC_GRACE_MS = 500;         ///< Suppress Phase-3 corrections while ALSA clock stabilizes
constexpr int DECODER_SYNC_FREERUN_FRAMES = 1;     ///< First frame after Clear() shown immediately for responsiveness
constexpr int DECODER_JITTER_BUFFER_MS =
    500; ///< Jitter buffer depth: frames are held until PTS span reaches this threshold

// ============================================================================
// === STRUCTURES ===
// ============================================================================

VaapiFrame::VaapiFrame(VaapiFrame &&other) noexcept
    : avFrame(other.avFrame), ownsFrame(other.ownsFrame), pts(other.pts), vaSurfaceId(other.vaSurfaceId) {
    other.avFrame = nullptr;
    other.vaSurfaceId = VA_INVALID_SURFACE;
    other.ownsFrame = false;
}

VaapiFrame::~VaapiFrame() noexcept {
    if (avFrame && ownsFrame) {
        av_frame_free(&avFrame);
    }
}

auto VaapiFrame::operator=(VaapiFrame &&other) noexcept -> VaapiFrame & {
    if (this != &other) {
        if (avFrame && ownsFrame) {
            av_frame_free(&avFrame);
        }
        avFrame = other.avFrame;
        vaSurfaceId = other.vaSurfaceId;
        pts = other.pts;
        ownsFrame = other.ownsFrame;

        other.avFrame = nullptr;
        other.vaSurfaceId = VA_INVALID_SURFACE;
        other.ownsFrame = false;
    }
    return *this;
}

// ============================================================================
// === DECODER CLASS ===
// ============================================================================

cVaapiDecoder::cVaapiDecoder(cVaapiDisplay *displayPtr, VaapiContext *vaapiCtxPtr)
    : cThread("vaapivideo/decoder"), display(displayPtr), vaapiContext(vaapiCtxPtr) {}

cVaapiDecoder::~cVaapiDecoder() noexcept {
    dsyslog("vaapivideo/decoder: destroying (stopping=%d)", stopping.load(std::memory_order_relaxed));
    Shutdown();
    DrainQueue();
}

// ============================================================================
// === PUBLIC API ===
// ============================================================================

auto cVaapiDecoder::Clear() -> void {
    // Lock ordering: codecMutex -> packetMutex prevents a deadlock with EnqueueData().
    const cMutexLock decodeLock(&codecMutex);

    DrainQueue();

    // Flush decoder; keeps the codec context open so the next I-frame can continue.
    if (codecCtx) {
        avcodec_flush_buffers(codecCtx.get());
    }

    // Filter graph may hold frames with stale PTS; rebuilt lazily on next decoded frame.
    ResetFilterGraph();
    hasLoggedFirstFrame.store(false, std::memory_order_relaxed);

    // AVCodecParserContext has no flush API, so recreate it from scratch.
    if (currentCodecId != AV_CODEC_ID_NONE) {
        parserCtx.reset(av_parser_init(currentCodecId));
    } else {
        parserCtx.reset();
    }

    if (decodedFrame) {
        av_frame_unref(decodedFrame.get());
    }
    if (filteredFrame) {
        av_frame_unref(filteredFrame.get());
    }

    lastPts.store(AV_NOPTS_VALUE, std::memory_order_relaxed);

    freerunFrames.store(DECODER_SYNC_FREERUN_FRAMES, std::memory_order_relaxed);
    syncAcquired.store(false, std::memory_order_release);
    syncLogPending.store(true, std::memory_order_relaxed);
}

auto cVaapiDecoder::DrainQueue() -> void {
    const cMutexLock lock(&packetMutex);
    while (!packetQueue.empty()) {
        const std::unique_ptr<AVPacket, FreeAVPacket> dropped{packetQueue.front()};
        packetQueue.pop();
    }
    packetCondition.Broadcast();
}

auto cVaapiDecoder::EnqueueData(const uint8_t *data, size_t size, int64_t pts) -> void {
    if (!data || size == 0 || stopping.load(std::memory_order_relaxed)) {
        return;
    }

    // Lock ordering: codecMutex -> packetMutex prevents a deadlock with Clear().
    const cMutexLock decodeLock(&codecMutex);

    if (!codecCtx || !parserCtx) {
        return;
    }

    // Walk the input data, feeding the parser until it is consumed.
    const uint8_t *parseData = data;
    if (size > static_cast<size_t>(INT_MAX)) [[unlikely]] {
        return;
    }
    int parseSize = static_cast<int>(size);
    int64_t currentPts = pts; // PTS is only used on the first av_parser_parse2() call

    while (parseSize > 0) {
        uint8_t *parsedData = nullptr; // NOLINT(misc-const-correctness)
        int parsedSize = 0;

        const int parsed = av_parser_parse2(parserCtx.get(), codecCtx.get(), &parsedData, &parsedSize, parseData,
                                            parseSize, currentPts, AV_NOPTS_VALUE, 0);

        if (parsed < 0) [[unlikely]] {
            break;
        }

        // parsed==0: frame boundary at input start (MPEG-2 PES alignment). Output queued, re-enter parse2 for actual
        // data. Break only if no output.
        if (parsed == 0 && parsedSize == 0) {
            break;
        }

        parseData += parsed;
        parseSize -= parsed;
        currentPts = AV_NOPTS_VALUE; // the parser tracks PTS across calls internally

        if (parsedSize > 0) {
            // In FF and reverse mode only keyframes are useful; skip confirmed non-keyframes (key_frame == 0). Unknown
            // status (-1) passes through because we cannot confirm it is not a keyframe. Slow-forward sends all frames
            // to preserve smooth motion.
            if (trickSpeed.load(std::memory_order_acquire) != 0 &&
                (isTrickFastForward.load(std::memory_order_relaxed) ||
                 isTrickReverse.load(std::memory_order_relaxed)) &&
                parserCtx->key_frame == 0) {
                continue;
            }

            AVPacket *pkt = av_packet_alloc();
            if (!pkt) [[unlikely]] {
                break;
            }

            if (av_new_packet(pkt, parsedSize) < 0) [[unlikely]] {
                av_packet_free(&pkt);
                continue;
            }
            std::memcpy(pkt->data, parsedData, static_cast<size_t>(parsedSize));

            pkt->pts = parserCtx->pts; // parser extracted these from the bitstream
            pkt->dts = parserCtx->dts;

            // Trick mode uses a depth-1 queue: if full, discard the new packet and rely on PlayVideo()/Poll()
            // backpressure to pace the source. Normal playback uses a larger queue and drops the oldest when full to
            // keep latency bounded and prefer fresh frames over stale ones.
            const cMutexLock lock(&packetMutex);
            const bool isTrickMode = trickSpeed.load(std::memory_order_relaxed) != 0;
            const size_t maxDepth = isTrickMode ? DECODER_TRICK_QUEUE_DEPTH : DECODER_QUEUE_CAPACITY;
            if (packetQueue.size() >= maxDepth) {
                if (isTrickMode) {
                    av_packet_free(&pkt);
                    continue;
                }
                av_packet_free(&packetQueue.front());
                packetQueue.pop();
            }
            packetQueue.push(pkt);
            packetCondition.Broadcast();
        }
    }

    // H.264 access-unit boundaries are detected implicitly when the next NAL arrives, so no explicit flush is needed.
    // Backward trick I-frames are self-contained and flush the parser state naturally.
}

[[nodiscard]] auto cVaapiDecoder::GetLastPts() const noexcept -> int64_t {
    return lastPts.load(std::memory_order_acquire);
}

[[nodiscard]] auto cVaapiDecoder::GetQueueSize() const -> size_t {
    const cMutexLock lock(&packetMutex);
    return packetQueue.size();
}

[[nodiscard]] auto cVaapiDecoder::GetStreamAspect() const -> double {
    const cMutexLock lock(&codecMutex);
    if (!codecCtx || codecCtx->width == 0 || codecCtx->height == 0) [[unlikely]] {
        return 0.0;
    }
    const int sarNum = codecCtx->sample_aspect_ratio.num > 0 ? codecCtx->sample_aspect_ratio.num : 1;
    const int sarDen = codecCtx->sample_aspect_ratio.den > 0 ? codecCtx->sample_aspect_ratio.den : 1;
    return (static_cast<double>(codecCtx->width) * sarNum) / (static_cast<double>(codecCtx->height) * sarDen);
}

[[nodiscard]] auto cVaapiDecoder::GetStreamHeight() const -> int {
    const cMutexLock lock(&codecMutex);
    return codecCtx ? codecCtx->height : 0;
}

[[nodiscard]] auto cVaapiDecoder::GetStreamWidth() const -> int {
    const cMutexLock lock(&codecMutex);
    return codecCtx ? codecCtx->width : 0;
}

[[nodiscard]] auto cVaapiDecoder::Initialize() -> bool {
    if (!display || !vaapiContext || !vaapiContext->hwDeviceRef) [[unlikely]] {
        esyslog("vaapivideo/decoder: missing display or VAAPI context");
        return false;
    }

    decodedFrame.reset(av_frame_alloc());
    filteredFrame.reset(av_frame_alloc());

    if (!decodedFrame || !filteredFrame) [[unlikely]] {
        esyslog("vaapivideo/decoder: failed to allocate frames");
        return false;
    }

    ready.store(true, std::memory_order_release);
    Start();

    isyslog("vaapivideo/decoder: initialized (packet queue size=%zu)", DECODER_QUEUE_CAPACITY);
    return true;
}

[[nodiscard]] auto cVaapiDecoder::IsQueueEmpty() const -> bool {
    // Used by VDR's Poll() mechanism for backpressure on the PES feed.
    const cMutexLock lock(&packetMutex);
    return packetQueue.empty();
}

[[nodiscard]] auto cVaapiDecoder::IsQueueFull() const -> bool {
    const cMutexLock lock(&packetMutex);
    return packetQueue.size() >= DECODER_QUEUE_CAPACITY;
}

[[nodiscard]] auto cVaapiDecoder::IsReady() const noexcept -> bool { return ready.load(std::memory_order_acquire); }

[[nodiscard]] auto cVaapiDecoder::IsReadyForNextTrickFrame() const noexcept -> bool {
    if (trickSpeed.load(std::memory_order_relaxed) == 0) {
        return true;
    }
    const uint64_t dueTime = nextTrickFrameDue.load(std::memory_order_relaxed);
    return cTimeMs::Now() >= dueTime;
}

auto cVaapiDecoder::RequestCodecReopen() -> void {
    const cMutexLock lock(&codecMutex);
    forceCodecReopen = true;
}

[[nodiscard]] auto cVaapiDecoder::OpenCodec(AVCodecID codecId) -> bool {
    const cMutexLock decodeLock(&codecMutex);

    if (!ready.load(std::memory_order_acquire)) [[unlikely]] {
        esyslog("vaapivideo/decoder: not initialized");
        return false;
    }

    if (codecCtx && currentCodecId == codecId && !forceCodecReopen) {
        return true;
    }
    forceCodecReopen = false;

    // Tear down all codec-dependent state; the filter graph holds hw_frames_ctx references that become invalid.
    parserCtx.reset();
    codecCtx.reset();
    ResetFilterGraph();
    currentCodecId = AV_CODEC_ID_NONE;
    hasLoggedFirstFrame.store(false, std::memory_order_relaxed);

    const AVCodec *decoder = avcodec_find_decoder(codecId);
    if (!decoder) [[unlikely]] {
        esyslog("vaapivideo/decoder: codec %d not found", static_cast<int>(codecId));
        return false;
    }

    // VA profile flags probed once at device init by ProbeVppCapabilities().
    bool useHwDecode = false;
    switch (codecId) {
        case AV_CODEC_ID_MPEG2VIDEO:
            useHwDecode = vaapiContext->hwMpeg2;
            break;
        case AV_CODEC_ID_H264:
            useHwDecode = vaapiContext->hwH264;
            break;
        case AV_CODEC_ID_HEVC:
            useHwDecode = vaapiContext->hwHevc;
            break;
        default:
            break;
    }

    // For HW decode, also verify FFmpeg has VAAPI hw_config for this codec.
    if (useHwDecode) {
        bool hasHwConfig = false;
        for (int i = 0;; ++i) {
            const AVCodecHWConfig *hwCfg = avcodec_get_hw_config(decoder, i);
            if (!hwCfg) {
                break;
            }
            if (hwCfg->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
                hwCfg->device_type == AV_HWDEVICE_TYPE_VAAPI) {
                hasHwConfig = true;
                break;
            }
        }
        useHwDecode = hasHwConfig;
    }

    parserCtx.reset(av_parser_init(codecId));
    if (!parserCtx) [[unlikely]] {
        esyslog("vaapivideo/decoder: failed to create parser for %s", decoder->name);
        return false;
    }

    std::unique_ptr<AVCodecContext, FreeAVCodecContext> decoderCtx{avcodec_alloc_context3(decoder)};
    if (!decoderCtx) [[unlikely]] {
        esyslog("vaapivideo/decoder: failed to allocate context for %s", decoder->name);
        return false;
    }

    // DVB stream tolerance: coarse frame-level checks only (no CRC / bitstream-spec abort),
    // accept slightly non-conforming bitstreams common in HEVC / MPEG-2 broadcasts.
    decoderCtx->err_recognition = AV_EF_CAREFUL;
    decoderCtx->strict_std_compliance = FF_COMPLIANCE_UNOFFICIAL;

    if (useHwDecode) {
        // VAAPI hardware decode -- single-threaded; GPU handles its own parallelism.
        decoderCtx->thread_count = 1;

        decoderCtx->hw_device_ctx = av_buffer_ref(vaapiContext->hwDeviceRef);
        if (!decoderCtx->hw_device_ctx) [[unlikely]] {
            esyslog("vaapivideo/decoder: failed to reference VAAPI device for %s", decoder->name);
            return false;
        }

        // get_format callback: tell FFmpeg to use VAAPI surfaces for all output frames.
        decoderCtx->get_format = [](AVCodecContext *, const AVPixelFormat *formats) -> AVPixelFormat {
            for (const AVPixelFormat *fmt = formats; *fmt != AV_PIX_FMT_NONE; ++fmt) {
                if (*fmt == AV_PIX_FMT_VAAPI) {
                    return AV_PIX_FMT_VAAPI;
                }
            }
            return AV_PIX_FMT_NONE;
        };
    }
    // else: software decode -- no hw_device_ctx; InitFilterGraph() will hwupload frames.

    if (const int ret = avcodec_open2(decoderCtx.get(), decoder, nullptr); ret < 0) [[unlikely]] {
        esyslog("vaapivideo/decoder: failed to open %s: %s", decoder->name, AvErr(ret).data());
        return false;
    }

    codecCtx = std::move(decoderCtx);
    currentCodecId = codecId;
    isyslog("vaapivideo/decoder: opened %s (%s)", decoder->name, useHwDecode ? "hardware" : "software");

    return true;
}

auto cVaapiDecoder::NotifyAudioChange() -> void {
    syncAcquired.store(false, std::memory_order_release);
    syncLogPending.store(true, std::memory_order_relaxed);
}

auto cVaapiDecoder::SetAudioProcessor(cAudioProcessor *audio) -> void { audioProcessor = audio; }

auto cVaapiDecoder::SetLiveMode(bool live) -> void { liveMode.store(live, std::memory_order_relaxed); }

auto cVaapiDecoder::RequestTrickExit() -> void { trickExitPending.store(true, std::memory_order_release); }

auto cVaapiDecoder::SetTrickSpeed(int speed, bool forward, bool fast) -> void {
    // VDR trick-speed convention:
    //   speed==0: normal play, fast+speed>0: FF/REW (6->2x, 3->4x, 1->8x), !fast+speed>0: slow 1/speed.

    // A new TrickSpeed() call cancels any pending exit from Play().
    trickExitPending.store(false, std::memory_order_relaxed);

    // VDR skips DeviceClear() for FF; flush everything ourselves. Lock ordering: codecMutex -> packetMutex.
    if (fast && speed > 0) {
        const cMutexLock decodeLock(&codecMutex);
        DrainQueue();
        if (codecCtx) {
            avcodec_flush_buffers(codecCtx.get());
        }
        // AVCodecParserContext has no flush API; stale buffered data causes garbled NALs.
        if (currentCodecId != AV_CODEC_ID_NONE) {
            parserCtx.reset(av_parser_init(currentCodecId));
        }
        // Flush may reallocate hw_frames_ctx, invalidating the filter graph. Rebuild on next decoded frame.
        ResetFilterGraph();
        if (decodedFrame) {
            av_frame_unref(decodedFrame.get());
        }
        if (filteredFrame) {
            av_frame_unref(filteredFrame.get());
        }
    }

    // Write flags before trickSpeed release-store so the decode thread sees a consistent set.
    isTrickFastForward.store(forward && fast, std::memory_order_relaxed);
    isTrickReverse.store(!forward, std::memory_order_relaxed);
    prevTrickPts.store(AV_NOPTS_VALUE, std::memory_order_relaxed);

    // Map VDR speed to pacing multipliers; slow mode uses fixed per-frame hold.
    if (fast && speed > 0) {
        if (speed >= 6) {
            trickMultiplier.store(2, std::memory_order_relaxed);
        } else if (speed >= 3) {
            trickMultiplier.store(4, std::memory_order_relaxed);
        } else {
            trickMultiplier.store(8, std::memory_order_relaxed);
        }
        trickHoldMs.store(DECODER_TRICK_HOLD_MS, std::memory_order_relaxed);
    } else {
        trickMultiplier.store(0, std::memory_order_relaxed);
        trickHoldMs.store(static_cast<uint64_t>(speed) * DECODER_TRICK_HOLD_MS, std::memory_order_relaxed);
    }

    // Arm the pacing timer so the first trick frame is displayed immediately.
    nextTrickFrameDue.store(cTimeMs::Now(), std::memory_order_relaxed);

    if (speed == 0) {
        lastPts.store(AV_NOPTS_VALUE, std::memory_order_relaxed);
        syncAcquired.store(false, std::memory_order_release);
        syncLogPending.store(true, std::memory_order_relaxed);
    }

    trickSpeed.store(speed, std::memory_order_release);
}

auto cVaapiDecoder::Shutdown() -> void {
    const bool wasStopping = stopping.exchange(true, std::memory_order_acq_rel);
    dsyslog("vaapivideo/decoder: shutting down (wasStopping=%d)", wasStopping);

    if (wasStopping) {
        dsyslog("vaapivideo/decoder: already shut down, skipping");
        return;
    }

    {
        const cMutexLock lock(&packetMutex);
        packetCondition.Broadcast();
    }

    // VDR's Running() is not thread-safe; use hasExited instead.
    if (!hasExited.load(std::memory_order_acquire)) {
        Cancel(3);
    }

    // Spin until hasExited for proper happens-before.
    const cTimeMs timeout(SHUTDOWN_TIMEOUT_MS);
    while (!hasExited.load(std::memory_order_acquire) && !timeout.TimedOut()) {
        {
            const cMutexLock lock(&packetMutex);
            packetCondition.Broadcast();
        }
        cCondWait::SleepMs(10);
    }
}

// ============================================================================
// === THREAD ===
// ============================================================================

auto cVaapiDecoder::Action() -> void {
    isyslog("vaapivideo/decoder: thread started");

    AVPacket *workPacket = av_packet_alloc();
    if (!workPacket) [[unlikely]] {
        esyslog("vaapivideo/decoder: failed to allocate packet");
        hasExited.store(true, std::memory_order_release);
        return;
    }

    while (!stopping.load(std::memory_order_acquire)) {
        std::unique_ptr<AVPacket, FreeAVPacket> queuedPacket;

        // --- Dequeue: timed wait paces jitter drain at frame rate ---
        {
            const cMutexLock lock(&packetMutex);
            if (packetQueue.empty() && !stopping.load(std::memory_order_acquire)) {
                const int waitMs = (jitterPrimed && !jitterBuf.empty()) ? outputFrameDurationMs : 10;
                packetCondition.TimedWait(packetMutex, waitMs);
            }

            if (stopping.load(std::memory_order_acquire)) {
                break;
            }

            if (!packetQueue.empty()) {
                queuedPacket.reset(packetQueue.front());
                packetQueue.pop();
            }
        }

        // --- Decode ---
        std::vector<std::unique_ptr<VaapiFrame>> pendingFrames;

        if (queuedPacket) {
            av_packet_unref(workPacket);
            // Copy into workPacket so queuedPacket can be freed before the decode call,
            // reducing packetMutex contention for EnqueueData().
            if (av_packet_ref(workPacket, queuedPacket.get()) == 0) {
                queuedPacket.reset();

                // codecMutex held only during decode; released before frame delivery.
                if (!stopping.load(std::memory_order_acquire)) {
                    const cMutexLock decodeLock(&codecMutex);
                    if (codecCtx) {
                        (void)DecodeOnePacket(workPacket, pendingFrames);
                    }
                }
            }
        }

        // --- Frame delivery ---
        if (!liveMode.load(std::memory_order_relaxed)) {
            // Replay: no jitter buffering; submit directly to A/V sync.
            for (auto &frame : pendingFrames) {
                if (stopping.load(std::memory_order_relaxed)) [[unlikely]] {
                    break;
                }
                (void)SyncAndSubmitFrame(std::move(frame));
            }
        } else {
            // Live TV: jitter buffer absorbs DVB-over-IP timing jitter.
            auto frameIt = pendingFrames.begin();

            // Clear() / channel switch / seek: flush stale frames, show the first
            // decoded frame immediately (freerun), then re-prime the jitter buffer.
            // SyncAndSubmitFrame() consumes freerunFrames via its Phase-2 path.
            if (freerunFrames.load(std::memory_order_relaxed) > 0) {
                jitterBuf.clear();
                jitterPrimed = false;
                if (frameIt != pendingFrames.end() && !stopping.load(std::memory_order_relaxed)) {
                    (void)SyncAndSubmitFrame(std::move(*frameIt));
                    ++frameIt;
                }
            }

            for (; frameIt != pendingFrames.end(); ++frameIt) {
                jitterBuf.push_back(std::move(*frameIt));
            }

            // Prime: accumulate jitterTarget frames before draining.  Count-based
            // so rate=field deinterlaced content is measured correctly.
            if (!jitterPrimed && jitterTarget > 0 && static_cast<int>(jitterBuf.size()) >= jitterTarget) {
                jitterPrimed = true;

                // Skip frames whose PTS is behind the audio clock so the first
                // displayed frame aligns with what the viewer hears.
                if (audioProcessor) {
                    const int64_t clock = audioProcessor->GetClock();
                    if (clock != AV_NOPTS_VALUE) {
                        const int64_t lat = static_cast<int64_t>(vaapiConfig.audioLatency) * 90;
                        while (jitterBuf.size() > 1) {
                            const int64_t pts = jitterBuf.front()->pts;
                            if (pts != AV_NOPTS_VALUE && (pts - clock - lat) < 0) {
                                jitterBuf.pop_front();
                            } else {
                                break;
                            }
                        }
                    }
                }
            }

            // Drain one frame per iteration; timed wait above paces at frame rate.
            // Extra drain when above target to recover from buffer inflation
            // caused by Phase 4 poll-wait stalls blocking the decode loop.
            if (jitterPrimed && !jitterBuf.empty() && !stopping.load(std::memory_order_relaxed)) {
                auto drainFrame = std::move(jitterBuf.front());
                jitterBuf.pop_front();
                (void)SyncAndSubmitFrame(std::move(drainFrame));

                if (!jitterBuf.empty() && static_cast<int>(jitterBuf.size()) > jitterTarget &&
                    !stopping.load(std::memory_order_relaxed)) {
                    auto extraFrame = std::move(jitterBuf.front());
                    jitterBuf.pop_front();
                    (void)SyncAndSubmitFrame(std::move(extraFrame));
                }
            }
        }
    }

    av_packet_free(&workPacket);

    hasExited.store(true, std::memory_order_release);
}

// ============================================================================
// === INTERNAL METHODS ===
// ============================================================================

[[nodiscard]] auto cVaapiDecoder::CreateVaapiFrame(AVFrame *src) const -> std::unique_ptr<VaapiFrame> {
    if (!src || src->format != AV_PIX_FMT_VAAPI) [[unlikely]] {
        return nullptr;
    }

    auto vaapiFrame = std::make_unique<VaapiFrame>();

    // av_frame_clone() increments the VAAPI surface refcount; surface stays alive while this VaapiFrame is live.
    vaapiFrame->avFrame = av_frame_clone(src);
    if (!vaapiFrame->avFrame) [[unlikely]] {
        return nullptr;
    }

    // data[3] holds the VASurfaceID cast to a pointer -- the FFmpeg VAAPI convention.
    vaapiFrame->vaSurfaceId =
        static_cast<VASurfaceID>(reinterpret_cast<uintptr_t>( // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
            vaapiFrame->avFrame->data[3]));
    vaapiFrame->pts = src->pts;

    return vaapiFrame;
}

[[nodiscard]] auto cVaapiDecoder::DecodeOnePacket(AVPacket *pkt, std::vector<std::unique_ptr<VaapiFrame>> &outFrames)
    -> bool {
    if (!codecCtx) [[unlikely]] {
        return false;
    }

    bool anyFrameDecoded = false;
    bool packetSent = false;

    // MPEG-2: the codec context does not know the frame size until the sequence header has been parsed, so skip all
    // packets until width and height are known.
    if (codecCtx->codec_id == AV_CODEC_ID_MPEG2VIDEO && (codecCtx->width == 0 || codecCtx->height == 0)) {
        return false;
    }

    // Send-drain loop: on EAGAIN (HW output queue full), drain frames and retry. Normally runs once.
    while (!packetSent) {
        const int ret = avcodec_send_packet(codecCtx.get(), pkt);
        if (ret == AVERROR(EAGAIN)) {
            // VAAPI decoder is full; drain frames below and retry the send.
        } else {
            if (ret < 0 && ret != AVERROR_EOF) [[unlikely]] {
                // A hard send failure usually means the decoder has entered an error state
                // (e.g. corrupt NAL on HEVC). Flush and rebuild the filter graph so the next
                // IDR / I-frame can recover cleanly without leaving stale VAAPI surfaces.
                dsyslog("vaapivideo/decoder: send_packet failed: %s -- flushing for recovery", AvErr(ret).data());
                avcodec_flush_buffers(codecCtx.get());
                ResetFilterGraph();
                return anyFrameDecoded;
            }
            packetSent = true; // success or EOF -- don't retry
        }

        // Drain all decoded frames through the filter graph. No flush between trick frames; IDRs drain naturally.
        bool receivedThisIteration = false;
        while (true) {
            av_frame_unref(decodedFrame.get());
            const int recvRet = avcodec_receive_frame(codecCtx.get(), decodedFrame.get());
            if (recvRet == AVERROR(EAGAIN) || recvRet == AVERROR_EOF) {
                break;
            }
            if (recvRet < 0) [[unlikely]] {
                break;
            }

            receivedThisIteration = true;

            if (!hasLoggedFirstFrame.exchange(true, std::memory_order_relaxed)) {
                const char *fmtName = av_get_pix_fmt_name(static_cast<AVPixelFormat>(decodedFrame->format));
                isyslog("vaapivideo/decoder: first frame %dx%d %s%s", decodedFrame->width, decodedFrame->height,
                        fmtName ? fmtName : "unknown",
                        (decodedFrame->flags & AV_FRAME_FLAG_INTERLACED) ? " interlaced" : "");
            }

            // DVB MPEG-2 often omits color_description; force BT.470BG for correct SD color conversion.
            if (decodedFrame->colorspace == AVCOL_SPC_UNSPECIFIED && decodedFrame->height <= 576) {
                decodedFrame->colorspace = AVCOL_SPC_BT470BG;
            }

            // Build the filter graph lazily on the first decoded frame (or after Clear()).
            if (!filterGraph) {
                (void)InitFilterGraph(decodedFrame.get());
            }

            // Keep source PTS for both deinterlace outputs; interpolated field PTS would break our sync model.
            const int64_t sourcePts = decodedFrame->pts;

            if (filterGraph) {
                if (av_buffersrc_add_frame_flags(bufferSrcCtx, decodedFrame.get(), AV_BUFFERSRC_FLAG_KEEP_REF) < 0)
                    [[unlikely]] {
                    // Unexpected rejection -- rebuild the graph on the next frame.
                    ResetFilterGraph();
                    continue;
                }

                // With rate=field deinterlacing each source frame produces two outputs.
                while (true) {
                    av_frame_unref(filteredFrame.get());
                    const int filterRet = av_buffersink_get_frame(bufferSinkCtx, filteredFrame.get());
                    if (filterRet == AVERROR(EAGAIN) || filterRet == AVERROR_EOF) {
                        break;
                    }
                    if (filterRet < 0) [[unlikely]] {
                        break;
                    }

                    const bool isTrickMode = trickSpeed.load(std::memory_order_acquire) != 0;
                    filteredFrame->pts = sourcePts;

                    if (auto vaapiFrame = CreateVaapiFrame(filteredFrame.get())) {
                        // In FF/reverse trick mode, rate=field emits two outputs per source frame. The first has a
                        // cross-frame blend ghost; only keep the second.
                        if (isTrickMode &&
                            (isTrickFastForward.load(std::memory_order_relaxed) ||
                             isTrickReverse.load(std::memory_order_relaxed)) &&
                            !outFrames.empty() && outFrames.back()->pts == sourcePts) {
                            outFrames.back() = std::move(vaapiFrame);
                        } else {
                            outFrames.push_back(std::move(vaapiFrame));
                        }
                        anyFrameDecoded = true;
                    }
                }
            } else {
                // Filter graph is not available; pass the raw VAAPI frame directly.
                if (auto vaapiFrame = CreateVaapiFrame(decodedFrame.get())) {
                    outFrames.push_back(std::move(vaapiFrame));
                    anyFrameDecoded = true;
                }
            }
        }

        if (!packetSent && !receivedThisIteration) {
            // Prevent infinite loop: EAGAIN with no drainable frames -> drop the packet.
            esyslog("vaapivideo/decoder: EAGAIN deadlock guard fired (dropping packet)");
            break;
        }
    }

    return anyFrameDecoded;
}

[[nodiscard]] auto cVaapiDecoder::InitFilterGraph(AVFrame *firstFrame) -> bool {
    if (filterGraph) {
        return true;
    }

    if (!display) [[unlikely]] {
        esyslog("vaapivideo/decoder: no display for filter setup");
        return false;
    }

    if (!codecCtx) [[unlikely]] {
        esyslog("vaapivideo/decoder: no codec context for filter setup");
        return false;
    }

    // Source frame properties.
    const int srcWidth = firstFrame->width;
    const int srcHeight = firstFrame->height;
    const bool isInterlaced = (firstFrame->flags & AV_FRAME_FLAG_INTERLACED) != 0;
    const auto srcPixFmt = static_cast<AVPixelFormat>(firstFrame->format);

    const bool isUhd = (srcWidth > 1920 || srcHeight > 1088);
    const bool isSoftwareDecode = (srcPixFmt != AV_PIX_FMT_VAAPI);

    const uint32_t dstWidth = display->GetOutputWidth();
    const uint32_t dstHeight = display->GetOutputHeight();

    // Compute DAR from SAR, then fit into display bounds (letterbox/pillarbox). DRM has no HW scaler.
    const int sarNum = firstFrame->sample_aspect_ratio.num > 0 ? firstFrame->sample_aspect_ratio.num : 1;
    const int sarDen = firstFrame->sample_aspect_ratio.den > 0 ? firstFrame->sample_aspect_ratio.den : 1;

    const uint64_t darNum = static_cast<uint64_t>(srcWidth) * static_cast<uint64_t>(sarNum);
    const uint64_t darDen = static_cast<uint64_t>(srcHeight) * static_cast<uint64_t>(sarDen);

    uint32_t filterWidth = dstWidth;
    uint32_t filterHeight = dstHeight;

    // Integer cross-multiply: darNum/darDen vs dstWidth/dstHeight (no floating point).
    if (darNum * dstHeight > darDen * static_cast<uint64_t>(dstWidth)) {
        // Wider -> letterbox.
        filterWidth = dstWidth;
        filterHeight = static_cast<uint32_t>(static_cast<uint64_t>(dstWidth) * darDen / darNum);
    } else if (darNum * dstHeight < darDen * static_cast<uint64_t>(dstWidth)) {
        // Narrower -> pillarbox.
        filterHeight = dstHeight;
        filterWidth = static_cast<uint32_t>(static_cast<uint64_t>(dstHeight) * darNum / darDen);
    }

    // Even dimensions for NV12 chroma alignment, minimum 2.
    filterWidth = std::max(filterWidth & ~1U, 2U);
    filterHeight = std::max(filterHeight & ~1U, 2U);

    // Denoise/sharpen levels: UHD = off (GPU-bound), MPEG-2 SD = heavy, H.264/HEVC HD = moderate.
    int denoiseLevel = 0;
    int sharpnessLevel = 0;

    if (!isUhd) {
        if (currentCodecId == AV_CODEC_ID_MPEG2VIDEO) {
            denoiseLevel = 12;
            sharpnessLevel = 44;
        } else {
            denoiseLevel = 4;
            sharpnessLevel = 32;
        }
    }

    const bool needsResize =
        (filterWidth != static_cast<uint32_t>(srcWidth) || filterHeight != static_cast<uint32_t>(srcHeight));

    // Build the filter chain as a list of stages joined by commas. All paths normalize to NV12 BT.709.
    // SW path:  [bwdif] -> [hqdn3d] -> format=nv12 -> hwupload -> [scale_vaapi] -> [sharpness_vaapi]
    // HW path:  [deinterlace_vaapi] -> [denoise_vaapi] -> [scale_vaapi] -> [sharpness_vaapi]
    std::vector<std::string> filters;

    if (isSoftwareDecode) {
        // SW filters before hwupload for superior quality, then VAAPI VPP for scale/sharpen.
        if (isInterlaced) {
            // bwdif: w3fdif + yadif with cubic interpolation; better than VAAPI motion_adaptive on AMD/Mesa.
            filters.emplace_back("bwdif=mode=send_field:parity=auto:deint=all");
        }
        if (denoiseLevel > 0) {
            // hqdn3d: spatial + temporal denoise. SD MPEG-2 -> 4 (heavier), HD -> 2 (light).
            const int hqdn3dStrength = (currentCodecId == AV_CODEC_ID_MPEG2VIDEO) ? 4 : 2;
            filters.push_back(std::format("hqdn3d={}", hqdn3dStrength));
        }
        filters.emplace_back("format=nv12");
        filters.emplace_back("hwupload");
    } else {
        // VAAPI VPP filters before scale.
        if (isInterlaced && !vaapiContext->deinterlaceMode.empty()) {
            filters.push_back(std::format("deinterlace_vaapi=mode={}:rate=field", vaapiContext->deinterlaceMode));
        }
        if (denoiseLevel > 0 && vaapiContext->hasDenoise) {
            filters.push_back(std::format("denoise_vaapi=denoise={}", denoiseLevel));
        }
    }

    // scale_vaapi: always present for NV12/BT.709 normalization. Skip resize + HQ mode when src == dst geometry.
    if (needsResize) {
        const char *scaleMode = isUhd ? "" : ":mode=hq"; // HQ too expensive at 4K
        filters.push_back(std::format("scale_vaapi=w={}:h={}{}:format=nv12:out_color_matrix=bt709:out_range=tv",
                                      filterWidth, filterHeight, scaleMode));
    } else {
        filters.emplace_back("scale_vaapi=format=nv12:out_color_matrix=bt709:out_range=tv");
    }

    if (sharpnessLevel > 0 && vaapiContext->hasSharpness) {
        filters.push_back(std::format("sharpness_vaapi=sharpness={}", sharpnessLevel));
    }

    // Join stages into a comma-separated filter chain.
    std::string filterChain;
    for (const auto &filter : filters) {
        if (!filterChain.empty()) {
            filterChain += ',';
        }
        filterChain += filter;
    }

    // ---- Create FFmpeg filter graph ----

    filterGraph.reset(avfilter_graph_alloc());
    if (!filterGraph) [[unlikely]] {
        esyslog("vaapivideo/decoder: failed to allocate filter graph");
        return false;
    }

    // Fall back to 50 fps (DVB default) for VBR streams with unknown framerate.
    const int fpsNum = codecCtx->framerate.num > 0 ? codecCtx->framerate.num : 50;
    const int fpsDen = codecCtx->framerate.den > 0 ? codecCtx->framerate.den : 1;

    // Numeric pix_fmt avoids name-lookup issues across FFmpeg versions with HW format aliases.
    const std::string bufferSrcArgs =
        std::format("video_size={}x{}:pix_fmt={}:time_base=1/90000:pixel_aspect={}/{}:frame_rate={}/{}", srcWidth,
                    srcHeight, static_cast<int>(srcPixFmt), sarNum, sarDen, fpsNum, fpsDen);

    dsyslog("vaapivideo/decoder: buffer source args='%s'", bufferSrcArgs.c_str());

    // Alloc without init: hw_frames_ctx must be attached before init (FFmpeg 7.x validates in init_video()).
    bufferSrcCtx = avfilter_graph_alloc_filter(filterGraph.get(), avfilter_get_by_name("buffer"), "in");
    if (!bufferSrcCtx) [[unlikely]] {
        esyslog("vaapivideo/decoder: failed to allocate buffer source filter");
        ResetFilterGraph();
        return false;
    }

    // HW decode: attach hw_frames_ctx before init so the buffer source knows the format is VAAPI-backed.
    if (!isSoftwareDecode) {
        AVBufferSrcParameters *hwFramesParams = av_buffersrc_parameters_alloc();
        if (!hwFramesParams) [[unlikely]] {
            esyslog("vaapivideo/decoder: failed to allocate buffer source parameters");
            ResetFilterGraph();
            return false;
        }

        hwFramesParams->hw_frames_ctx = av_buffer_ref(codecCtx->hw_frames_ctx);
        if (!hwFramesParams->hw_frames_ctx) [[unlikely]] {
            esyslog("vaapivideo/decoder: av_buffer_ref(hw_frames_ctx) failed");
            av_free(hwFramesParams);
            ResetFilterGraph();
            return false;
        }
        if (const int setRet = av_buffersrc_parameters_set(bufferSrcCtx, hwFramesParams); setRet < 0) [[unlikely]] {
            esyslog("vaapivideo/decoder: av_buffersrc_parameters_set failed: %s", AvErr(setRet).data());
            av_free(hwFramesParams);
            ResetFilterGraph();
            return false;
        }
        // hw_frames_ctx now owned by the filter node; plain C struct requires av_free().
        av_free(hwFramesParams);
    }

    // Init buffer source: parses args and validates hw_frames_ctx.
    int ret = avfilter_init_str(bufferSrcCtx, bufferSrcArgs.c_str());
    if (ret < 0) [[unlikely]] {
        esyslog("vaapivideo/decoder: failed to init buffer source '%s': %s", bufferSrcArgs.c_str(), AvErr(ret).data());
        ResetFilterGraph();
        return false;
    }

    ret = avfilter_graph_create_filter(&bufferSinkCtx, avfilter_get_by_name("buffersink"), "out", nullptr, nullptr,
                                       filterGraph.get());
    if (ret < 0) [[unlikely]] {
        esyslog("vaapivideo/decoder: failed to create buffer sink: %s", AvErr(ret).data());
        ResetFilterGraph();
        return false;
    }

    // FFmpeg convention: "inputs" = sink end, "outputs" = source end.
    AVFilterInOut *graphInputs = avfilter_inout_alloc();
    AVFilterInOut *graphOutputs = avfilter_inout_alloc();
    if (!graphInputs || !graphOutputs) [[unlikely]] {
        avfilter_inout_free(&graphInputs);
        avfilter_inout_free(&graphOutputs);
        ResetFilterGraph();
        return false;
    }

    graphOutputs->name = av_strdup("in");
    graphOutputs->filter_ctx = bufferSrcCtx;

    graphInputs->name = av_strdup("out");
    graphInputs->filter_ctx = bufferSinkCtx;

    if (!graphOutputs->name || !graphInputs->name) [[unlikely]] {
        avfilter_inout_free(&graphInputs);
        avfilter_inout_free(&graphOutputs);
        ResetFilterGraph();
        return false;
    }

    ret = avfilter_graph_parse_ptr(filterGraph.get(), filterChain.c_str(), &graphInputs, &graphOutputs, nullptr);
    avfilter_inout_free(&graphInputs);
    avfilter_inout_free(&graphOutputs);

    if (ret < 0) [[unlikely]] {
        esyslog("vaapivideo/decoder: failed to parse filter chain '%s': %s", filterChain.c_str(), AvErr(ret).data());
        ResetFilterGraph();
        return false;
    }

    // SW decode: hwupload needs a target VAAPI device on every filter node.
    if (isSoftwareDecode) {
        for (unsigned int i = 0; i < filterGraph->nb_filters; ++i) {
            filterGraph->filters[i]->hw_device_ctx = av_buffer_ref(vaapiContext->hwDeviceRef);
            if (!filterGraph->filters[i]->hw_device_ctx) [[unlikely]] {
                esyslog("vaapivideo/decoder: av_buffer_ref(hwDeviceRef) failed for filter %u", i);
                ResetFilterGraph();
                return false;
            }
        }
    }

    ret = avfilter_graph_config(filterGraph.get(), nullptr);
    if (ret < 0) [[unlikely]] {
        esyslog("vaapivideo/decoder: failed to configure filter graph '%s': %s", filterChain.c_str(),
                AvErr(ret).data());
        ResetFilterGraph();
        return false;
    }

    // Compute output frame rate (rate=field doubles it for interlaced content).
    const int outputFps = (fpsNum / std::max(fpsDen, 1)) * (isInterlaced ? 2 : 1);
    outputFrameDurationMs = outputFps > 0 ? 1000 / outputFps : 20;
    jitterTarget = outputFps > 0 ? ((outputFps * DECODER_JITTER_BUFFER_MS) + 500) / 1000 : 25;

    isyslog("vaapivideo/decoder: VAAPI filter initialized (%dx%d -> %ux%u%s)", srcWidth, srcHeight, filterWidth,
            filterHeight, isInterlaced ? ", deinterlaced" : "");
    dsyslog("vaapivideo/decoder: filter chain='%s'", filterChain.c_str());
    return true;
}

auto cVaapiDecoder::ResetFilterGraph() -> void {
    bufferSrcCtx = nullptr;
    bufferSinkCtx = nullptr;
    filterGraph.reset();
}

[[nodiscard]] auto cVaapiDecoder::SyncAndSubmitFrame(std::unique_ptr<VaapiFrame> frame) -> bool {
    // A/V sync -- audio clock is master.  delta = vPTS - audioClock - latency
    //
    //   Phase 3 (steady state, +-100 ms dead zone):
    //     |delta| <= 100 ms   pass-through (page-flip pacing absorbs jitter)
    //     delta > +100 ms     video ahead   -> poll-wait for audio
    //     delta < -100 ms     video behind  -> drop frame
    //     |delta| > 1500 ms   escalate to Phase 4
    //
    //   Phase 4 (initial sync / re-sync, +-50 ms convergence):
    //     delta > +50 ms      video ahead   -> poll-wait for audio
    //     delta < -50 ms      video behind  -> drop frame

    if (!frame || !display) [[unlikely]] {
        return false;
    }

    const auto submit = [&]() -> bool { return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS); };
    const int64_t originalPts = frame->pts;

    // --- Phase 1: trick mode (pacing timer, no A/V sync) ---
    if (trickSpeed.load(std::memory_order_acquire) != 0) {
        // Deferred exit: Play() was called but TrickSpeed() did not follow.
        if (trickExitPending.exchange(false, std::memory_order_acquire)) {
            trickSpeed.store(0, std::memory_order_relaxed);
            isTrickReverse.store(false, std::memory_order_relaxed);
            isTrickFastForward.store(false, std::memory_order_relaxed);
            lastPts.store(AV_NOPTS_VALUE, std::memory_order_relaxed);
            freerunFrames.store(DECODER_SYNC_FREERUN_FRAMES, std::memory_order_relaxed);
            syncAcquired.store(false, std::memory_order_release);
            syncLogPending.store(true, std::memory_order_relaxed);
            // Fall through to normal A/V sync.
        } else {
            const int64_t savedPrevPts = prevTrickPts.load(std::memory_order_relaxed);
            const bool newSource = (originalPts != savedPrevPts);

            // Reverse: enforce monotonically decreasing PTS.
            if (isTrickReverse.load(std::memory_order_relaxed) && newSource && originalPts != AV_NOPTS_VALUE &&
                savedPrevPts != AV_NOPTS_VALUE && originalPts > savedPrevPts) {
                return true;
            }

            if (newSource) {
                prevTrickPts.store(originalPts, std::memory_order_relaxed);
                if (originalPts != AV_NOPTS_VALUE) {
                    lastPts.store(originalPts, std::memory_order_release);
                }

                // Wait for pacing timer, then arm the next one.
                const uint64_t due = nextTrickFrameDue.load(std::memory_order_relaxed);
                while (cTimeMs::Now() < due && !stopping.load(std::memory_order_relaxed) &&
                       trickSpeed.load(std::memory_order_relaxed) != 0) {
                    cCondWait::SleepMs(10);
                }

                // Fast: PTS-derived hold; slow: fixed trickHoldMs.
                const uint64_t mult = trickMultiplier.load(std::memory_order_relaxed);
                if (mult > 0 && originalPts != AV_NOPTS_VALUE && savedPrevPts != AV_NOPTS_VALUE) {
                    const auto ptsDelta = static_cast<uint64_t>(std::abs(originalPts - savedPrevPts));
                    const uint64_t holdMs = std::clamp(ptsDelta / (uint64_t{90} * mult), uint64_t{10}, uint64_t{2000});
                    nextTrickFrameDue.store(cTimeMs::Now() + holdMs, std::memory_order_relaxed);
                } else {
                    nextTrickFrameDue.store(cTimeMs::Now() + trickHoldMs.load(std::memory_order_relaxed),
                                            std::memory_order_relaxed);
                }
            }

            return submit();
        }
    }

    if (originalPts != AV_NOPTS_VALUE) {
        lastPts.store(originalPts, std::memory_order_release);
    }

    // --- Phase 2: freerun (no audio clock or no PTS) ---
    if (!audioProcessor || originalPts == AV_NOPTS_VALUE) {
        return submit();
    }

    // First frame after Clear(): submit immediately for responsive channel switch.
    if (freerunFrames.load(std::memory_order_relaxed) > 0) {
        freerunFrames.fetch_sub(1, std::memory_order_relaxed);
        return submit();
    }

    const int64_t latency = static_cast<int64_t>(vaapiConfig.audioLatency) * 90;
    int64_t clock = audioProcessor->GetClock();

    // --- Phase 3: steady state ---
    //   |delta| <= 100 ms   pass-through (page-flip pacing absorbs jitter)
    //   100-1500 ms          correct: drop (behind) or poll-wait (ahead)
    //   > 1500 ms           escalate to Phase 4
    //   Grace period (500 ms) suppresses corrections after sync and after each correction.
    if (syncAcquired.load(std::memory_order_acquire)) {
        if (clock == AV_NOPTS_VALUE) {
            return submit();
        }

        int64_t delta = originalPts - clock - latency;

        if (syncLogPending.exchange(false, std::memory_order_relaxed) || nextSyncLog.TimedOut()) {
            dsyslog("vaapivideo/decoder: sync status d=%+lldms buf=%zu", static_cast<long long>(delta / 90),
                    jitterBuf.size());
            nextSyncLog.Set(30000);
        }

        // Re-sync: |delta| > 1500 ms -- escalate to Phase 4.
        if (delta < -DECODER_SYNC_RESYNC || delta > DECODER_SYNC_RESYNC) {
            dsyslog("vaapivideo/decoder: sync lost d=%+lldms", static_cast<long long>(delta / 90));
            correctDrops = 0;
            syncAcquired.store(false, std::memory_order_relaxed);
            syncLogPending.store(true, std::memory_order_relaxed);
            if (delta < 0) {
                return true; // late: drop; Phase 4 catches up
            }
            // Early: fall through to Phase 4 poll-wait.
        } else {
            // Grace period: suppress corrections while ALSA clock stabilizes.
            if (!syncGrace.TimedOut()) {
                return submit();
            }

            // Behind > 100 ms: drop to catch up.
            if (delta < -DECODER_SYNC_CORRECT) {
                ++correctDrops;
                return true;
            }

            // Log when a drop run ends.
            if (correctDrops > 0) {
                dsyslog("vaapivideo/decoder: sync correct %d dropped d=%+lldms", correctDrops,
                        static_cast<long long>(delta / 90));
                correctDrops = 0;
                syncGrace.Set(DECODER_SYNC_GRACE_MS);
            }

            // Ahead > 100 ms: poll-wait until audio catches up.
            if (delta > DECODER_SYNC_CORRECT) {
                const int64_t entryDelta = delta;
                const cTimeMs pollDeadline(static_cast<int>(std::min((delta / 90) + 100, int64_t{3000})));
                while (delta > 0 && !pollDeadline.TimedOut() && !stopping.load(std::memory_order_relaxed) &&
                       syncAcquired.load(std::memory_order_acquire)) {
                    cCondWait::SleepMs(5);
                    clock = audioProcessor->GetClock();
                    if (clock == AV_NOPTS_VALUE) {
                        return submit();
                    }
                    delta = originalPts - clock - latency;
                }
                if (!syncAcquired.load(std::memory_order_acquire)) {
                    return submit();
                }
                dsyslog("vaapivideo/decoder: sync correct d=%+lldms -> %+lldms",
                        static_cast<long long>(entryDelta / 90), static_cast<long long>(delta / 90));
                syncGrace.Set(DECODER_SYNC_GRACE_MS);
            }

            // Pass-through: |delta| <= 100 ms -- page-flip pacing is sufficient.
            return submit();
        }
    }

    // --- Phase 4: initial sync or re-sync ---
    // Polls until audio clock arrives, converges to +-50 ms via poll-wait or drops.
    clock = audioProcessor->GetClock();

    // Wait up to 3 s for audio clock.
    if (clock == AV_NOPTS_VALUE) {
        const cTimeMs waitTimeout(3000);
        while (clock == AV_NOPTS_VALUE && !stopping.load(std::memory_order_relaxed) &&
               trickSpeed.load(std::memory_order_relaxed) == 0 && !waitTimeout.TimedOut()) {
            cCondWait::SleepMs(5);
            clock = audioProcessor->GetClock();
        }
    }
    if (clock == AV_NOPTS_VALUE) {
        dsyslog("vaapivideo/decoder: sync deferred (no audio clock)");
        syncAcquired.store(true, std::memory_order_relaxed);
        return submit();
    }

    int64_t delta = originalPts - clock - latency;

    if (syncLogPending.exchange(false, std::memory_order_relaxed)) {
        dsyslog("vaapivideo/decoder: sync start d=%+lldms buf=%zu", static_cast<long long>(delta / 90),
                jitterBuf.size());
        catchupDrops = 0;
    }

    // Video ahead: poll-wait with frozen-clock safety net (< 10 ms advance in 100 ms).
    if (delta > 0) {
        const int64_t clockAtWaitStart = clock;
        const cTimeMs frozenCheck(100);
        const cTimeMs waitTimeout(static_cast<int>(std::min((delta / 90) + 10, int64_t{3000})));
        while (delta > -10 && clock != AV_NOPTS_VALUE && !stopping.load(std::memory_order_relaxed) &&
               trickSpeed.load(std::memory_order_relaxed) == 0 && !waitTimeout.TimedOut()) {
            cCondWait::SleepMs(5);
            clock = audioProcessor->GetClock();
            if (clock != AV_NOPTS_VALUE) {
                delta = originalPts - clock - latency;
            }
            if (frozenCheck.TimedOut() && clock != AV_NOPTS_VALUE && clock - clockAtWaitStart < 10 * 90) {
                dsyslog("vaapivideo/decoder: sync frozen-clock break d=%+lldms", static_cast<long long>(delta / 90));
                break;
            }
        }
    }

    if (clock == AV_NOPTS_VALUE) {
        dsyslog("vaapivideo/decoder: sync deferred (clock lost)");
        return submit();
    }

    // Behind: drop until convergence.
    if (delta < -DECODER_SYNC_CONVERGE) {
        ++catchupDrops;
        return true;
    }

    // Still ahead: submit and re-enter next iteration.
    if (delta > DECODER_SYNC_CONVERGE) {
        return submit();
    }

    // Converged: |delta| <= 50 ms.
    if (catchupDrops > 0) {
        dsyslog("vaapivideo/decoder: sync acquired d=%+lldms (dropped %d)", static_cast<long long>(delta / 90),
                catchupDrops);
    } else {
        dsyslog("vaapivideo/decoder: sync acquired d=%+lldms", static_cast<long long>(delta / 90));
    }
    catchupDrops = 0;
    correctDrops = 0;
    syncGrace.Set(DECODER_SYNC_GRACE_MS);
    syncAcquired.store(true, std::memory_order_relaxed);
    nextSyncLog.Set(30000);
    return submit();
}
