// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file decoder.cpp
 * @brief Threaded VAAPI decoder with filter graph and A/V sync
 *
 * Pipeline:  EnqueueData() -> packet queue -> VAAPI decode -> filter graph -> A/V sync -> display
 * Filters:   [bwdif|deinterlace] -> [hqdn3d|denoise] -> scale (BT.709 NV12) -> [sharpen] (probed per GPU)
 * Sync:      Audio-clock master; unified threshold: drop-behind (both modes), wait-ahead (replay only)
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

constexpr size_t DECODER_QUEUE_CAPACITY = 500; ///< Video packet queue depth (~10 s at 50 fps)
constexpr int DECODER_SUBMIT_TIMEOUT_MS = 100; ///< SubmitFrame VSync backpressure timeout (ms)
constexpr int64_t DECODER_SYNC_HARD_THRESHOLD =
    200 * 90; ///< Beyond this raw delta A/V sync drops (behind) or blocks (ahead) (~200 ms)
constexpr int64_t DECODER_SYNC_CORRIDOR = 50 * 90;  ///< Soft target corridor: |EMA| > +/-50 ms triggers correction
constexpr int DECODER_SYNC_COOLDOWN_MS = 5000;      ///< Min interval between soft corrections (~1 EMA time constant)
constexpr int DECODER_SYNC_MAX_CORRECTION_MS = 100; ///< Cap on a single correction event (sleep ms or drop burst ms)
constexpr int DECODER_SYNC_FREERUN_FRAMES =
    1; ///< Frames submitted without sync after Clear() to show first picture fast
constexpr int DECODER_JITTER_BUFFER_MS = 500; ///< Jitter buffer target: must absorb DVB-over-IP timing variance (ms)
constexpr int DECODER_SYNC_EMA_SAMPLES = 250; ///< EMA divisor (alpha = 1/N); ~5 s time constant @ 50 fps
constexpr int DECODER_SYNC_WARMUP_SAMPLES =
    50; ///< Samples averaged into a simple mean before the EMA is seeded (post-reset bias guard)
constexpr int DECODER_SYNC_LOG_INTERVAL_MS = 30000; ///< Periodic sync diagnostic interval (ms)

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

    // Signal the decode thread to flush stale jitter-buffer frames at the top of its next delivery
    // cycle.  Doing it here (under codecMutex) would race with the delivery section, which holds no
    // lock while iterating over jitterBuf.  The decode thread owns those fields and applies the flush
    // itself when it sees jitterFlushPending.  Stale frames from a preceding live-TV session would
    // otherwise hold VAAPI surface references across the codec teardown boundary, and jitterPrimed=true
    // would cause the loop to pace at outputFrameDurationMs instead of 10 ms even in replay mode.
    jitterFlushPending.store(true, std::memory_order_release);

    freerunFrames.store(DECODER_SYNC_FREERUN_FRAMES, std::memory_order_relaxed);
    ResetSmoothedDelta();

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

    // av_parser_parse2() consumes incrementally; PTS applies to the first call only -- the parser
    // propagates it internally across subsequent calls for the same access unit.
    const uint8_t *parseData = data;
    if (size > static_cast<size_t>(INT_MAX)) [[unlikely]] {
        return;
    }
    int parseSize = static_cast<int>(size);
    int64_t currentPts = pts;

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
        currentPts = AV_NOPTS_VALUE;

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

            pkt->pts = parserCtx->pts;
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
    // VDR's Poll() calls this; returning true tells VDR to deliver the next PES packet.
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

    // Full teardown: the filter graph caches hw_frames_ctx from the old codec and cannot be reused.
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

    // VA profile flags alone aren't enough: FFmpeg must also have a VAAPI hw_config entry for this codec.
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
        // VAAPI decode: single-threaded; the GPU parallelizes internally.
        // Multi-threaded SW decode (the else path) uses FFmpeg's default thread pool.
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
    // Audio codec changed: the audio clock will reset. Grant a freerun window so the first frames
    // after the change pass through unsync'd, then the grace period lets the new clock stabilize.
    freerunFrames.store(DECODER_SYNC_FREERUN_FRAMES, std::memory_order_relaxed);
    syncLogPending.store(true, std::memory_order_relaxed);
}

auto cVaapiDecoder::SetAudioProcessor(cAudioProcessor *audio) -> void {
    audioProcessor.store(audio, std::memory_order_release);
}

auto cVaapiDecoder::SetLiveMode(bool live) -> void { liveMode.store(live, std::memory_order_relaxed); }

auto cVaapiDecoder::RequestTrickExit() -> void { trickExitPending.store(true, std::memory_order_release); }

auto cVaapiDecoder::SetTrickSpeed(int speed, bool forward, bool fast) -> void {
    // VDR trick-speed convention:
    //   speed==0: normal play, fast+speed>0: FF/REW (6->2x, 3->4x, 1->8x), !fast+speed>0: slow 1/speed.

    // A new TrickSpeed() call cancels any pending exit from Play().
    trickExitPending.store(false, std::memory_order_relaxed);

    // VDR does not call DeviceClear() when entering FF/REW; we must flush stale state ourselves.
    // Lock ordering: codecMutex -> packetMutex (matches Clear()/EnqueueData()).
    if (fast && speed > 0) {
        const cMutexLock decodeLock(&codecMutex);
        DrainQueue();
        if (codecCtx) {
            avcodec_flush_buffers(codecCtx.get());
        }
        // AVCodecParserContext has no flush API; stale partial NALs produce garbled output.
        if (currentCodecId != AV_CODEC_ID_NONE) {
            parserCtx.reset(av_parser_init(currentCodecId));
        }
        // avcodec_flush_buffers() may reallocate hw_frames_ctx, invalidating cached surface pools.
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

    // VDR's cThread::Running() is not thread-safe (no memory fence); use our atomic hasExited.
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

    const std::unique_ptr<AVPacket, FreeAVPacket> workPacket{av_packet_alloc()};
    if (!workPacket) [[unlikely]] {
        esyslog("vaapivideo/decoder: failed to allocate packet");
        hasExited.store(true, std::memory_order_release);
        return;
    }

    std::vector<std::unique_ptr<VaapiFrame>> pendingFrames;
    bool primeSyncPending{false};
    uint64_t lastDrainMs{0};

    while (!stopping.load(std::memory_order_acquire)) {
        std::unique_ptr<AVPacket, FreeAVPacket> queuedPacket;

        // Wait timeout: 18 ms during jitter drain keeps the loop paced by SubmitFrame's VSync
        // backpressure rather than the timer; 20 ms would compound with VSync and degrade 50 fps
        // rate=field output to ~33 fps.  10 ms idle-wait when no jitter buffer is active.
        {
            const cMutexLock lock(&packetMutex);
            if (packetQueue.empty() && !stopping.load(std::memory_order_acquire)) {
                const int waitMs = (jitterPrimed && !jitterBuf.empty()) ? 18 : 10;
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
        pendingFrames.clear();

        if (queuedPacket) {
            av_packet_unref(workPacket.get());
            // Ref-copy into workPacket, then release queuedPacket early -- avoids holding
            // packetMutex during the potentially slow avcodec_send_packet() GPU submission.
            if (av_packet_ref(workPacket.get(), queuedPacket.get()) == 0) {
                queuedPacket.reset();

                // codecMutex held only during decode; released before frame delivery.
                if (!stopping.load(std::memory_order_acquire)) {
                    const cMutexLock decodeLock(&codecMutex);
                    if (codecCtx) {
                        (void)DecodeOnePacket(workPacket.get(), pendingFrames);
                    }
                }
            }
        }

        // --- Frame delivery ---
        // Deferred jitter flush from Clear().  Jitter fields are decoder-thread-owned;
        // Clear() signals via atomic, we apply here without lock contention.
        if (jitterFlushPending.exchange(false, std::memory_order_acquire)) {
            jitterBuf.clear();
            jitterPrimed = false;
            pendingDrops = 0;
        }

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
            // SyncAndSubmitFrame() consumes the freerunFrames counter.
            if (freerunFrames.load(std::memory_order_relaxed) > 0) {
                jitterBuf.clear();
                jitterPrimed = false;
                if (frameIt != pendingFrames.end() && !stopping.load(std::memory_order_relaxed)) {
                    (void)SyncAndSubmitFrame(std::move(*frameIt));
                    ++frameIt;
                }
            }

            // No filter graph yet (jitterTarget == 0): submit directly.
            if (jitterTarget == 0) {
                for (; frameIt != pendingFrames.end(); ++frameIt) {
                    if (stopping.load(std::memory_order_relaxed)) [[unlikely]] {
                        break;
                    }
                    (void)SyncAndSubmitFrame(std::move(*frameIt));
                }
            } else {
                for (; frameIt != pendingFrames.end(); ++frameIt) {
                    jitterBuf.push_back(std::move(*frameIt));
                }

                // Prime: accumulate jitterTarget frames before first drain.
                if (!jitterPrimed && static_cast<int>(jitterBuf.size()) >= jitterTarget) {
                    jitterPrimed = true;
                    primeSyncPending = true; // request one-shot settle before first drain
                    ResetSmoothedDelta();
                    syncCooldown.Set(DECODER_SYNC_COOLDOWN_MS);
                    nextSyncLog.Set(DECODER_SYNC_LOG_INTERVAL_MS);
                    lastDrainMs = 0;
                    drainMissCount = 0;
                    dsyslog("vaapivideo/decoder: jitter buffer primed (buf=%zu target=%d)", jitterBuf.size(),
                            jitterTarget);
                }

                // Underrun: buffer was primed but drained empty. Re-prime to rebuild the cushion.
                if (jitterPrimed && jitterBuf.empty()) {
                    dsyslog("vaapivideo/decoder: jitter buffer underrun -- re-priming (target=%d)", jitterTarget);
                    jitterPrimed = false;
                }

                // Drain one frame per iteration; VSync backpressure paces the rate.
                if (jitterPrimed && !jitterBuf.empty() && !stopping.load(std::memory_order_relaxed)) {
                    auto *const ap = audioProcessor.load(std::memory_order_acquire);
                    if (ap) {
                        if (primeSyncPending) {
                            primeSyncPending = false;
                            RunJitterPrimeSync(ap);
                        }
                        SkipStaleJitterFrames(ap);
                    }
                    if (!jitterBuf.empty() && !stopping.load(std::memory_order_relaxed)) {
                        const uint64_t nowMs = cTimeMs::Now();
                        if (lastDrainMs > 0 && static_cast<int>(nowMs - lastDrainMs) > outputFrameDurationMs * 2) {
                            ++drainMissCount;
                        }
                        lastDrainMs = nowMs;
                        auto drainFrame = std::move(jitterBuf.front());
                        jitterBuf.pop_front();
                        (void)SyncAndSubmitFrame(std::move(drainFrame));
                    }
                }
            }
        }
    }

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

    // FFmpeg VAAPI convention: data[3] is a VASurfaceID cast to a pointer (not a real allocation).
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

    // MPEG-2 sequence header provides frame dimensions; filter graph cannot be built without them.
    if (codecCtx->codec_id == AV_CODEC_ID_MPEG2VIDEO && (codecCtx->width == 0 || codecCtx->height == 0)) {
        return false;
    }

    // Send-drain loop: EAGAIN means the GPU output queue is full; drain frames then retry.
    // Normally single-iteration; loops only when the VAAPI surface pool is saturated.
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

            // DVB MPEG-2 broadcasts rarely set color_description. Without this, FFmpeg assumes
            // BT.709 for SD content, producing washed-out colors after the scale_vaapi BT.709 conversion.
            if (decodedFrame->colorspace == AVCOL_SPC_UNSPECIFIED && decodedFrame->height <= 576) {
                decodedFrame->colorspace = AVCOL_SPC_BT470BG;
            }

            // Build the filter graph lazily on the first decoded frame (or after Clear()).
            // VA driver calls (filter push/pull + surface sync) are not thread-safe per-VADisplay;
            // vaDriverMutex serializes with the display thread's DRM export path.
            const int64_t sourcePts = decodedFrame->pts;
            {
                const cMutexLock vaLock(&display->GetVaDriverMutex());

                if (!filterGraph) {
                    (void)InitFilterGraph(decodedFrame.get());
                }

                if (filterGraph) {
                    if (av_buffersrc_add_frame_flags(bufferSrcCtx, decodedFrame.get(), AV_BUFFERSRC_FLAG_KEEP_REF) < 0)
                        [[unlikely]] {
                        ResetFilterGraph();
                        continue;
                    }

                    while (true) {
                        av_frame_unref(filteredFrame.get());
                        const int filterRet = av_buffersink_get_frame(bufferSinkCtx, filteredFrame.get());
                        if (filterRet == AVERROR(EAGAIN) || filterRet == AVERROR_EOF) {
                            break;
                        }
                        if (filterRet < 0) [[unlikely]] {
                            break;
                        }

                        if (auto vaapiFrame = CreateVaapiFrame(filteredFrame.get())) {
                            outFrames.push_back(std::move(vaapiFrame));
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

            // rate=field deinterlacing doubles frame count: assign monotonic PTS so A/V sync
            // sees a smooth timeline (first field = source PTS, second = +1 field period).
            // Done outside VA driver lock since it's pure arithmetic.
            for (size_t i = 0; i < outFrames.size(); ++i) {
                outFrames.at(i)->pts =
                    (sourcePts != AV_NOPTS_VALUE && i > 0)
                        ? sourcePts + (static_cast<int64_t>(outputFrameDurationMs) * 90 * static_cast<int64_t>(i))
                        : sourcePts;
            }

            // rate=field's first output blends adjacent source fields, visible as ghosting in
            // trick mode where adjacent fields may be far apart temporally. Keep only the clean second.
            if (trickSpeed.load(std::memory_order_relaxed) != 0 &&
                (isTrickFastForward.load(std::memory_order_relaxed) ||
                 isTrickReverse.load(std::memory_order_relaxed)) &&
                outFrames.size() > 1 && outFrames.front()->pts == sourcePts) {
                outFrames.erase(outFrames.begin());
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

    // Compute DAR from SAR, then letterbox/pillarbox into display bounds.
    // DRM atomic modesetting has no hardware scaler; all scaling must happen in the VPP filter.
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

    // Denoise/sharpen tuning: UHD = off (GPU-bound at 4K), MPEG-2 SD = heavy (noisy analog-era
    // sources), H.264/HEVC HD = moderate.  Values tuned empirically on AMD/Intel VAAPI.
    int denoiseLevel = 0;
    int sharpnessLevel = 0;

    if (!isUhd) {
        if (currentCodecId == AV_CODEC_ID_MPEG2VIDEO) {
            denoiseLevel = 12;   ///< MPEG-2 SD: heavy denoise for analog-era source noise
            sharpnessLevel = 44; ///< MPEG-2 SD: compensate post-denoise softening
        } else {
            denoiseLevel = 4;    ///< HD: light denoise preserving detail
            sharpnessLevel = 32; ///< HD: mild sharpening for upscale clarity
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
            // bwdif: w3fdif + yadif hybrid; superior to VAAPI motion_adaptive on AMD/Mesa drivers.
            filters.emplace_back("bwdif=mode=send_field:parity=auto:deint=all");
        }
        if (denoiseLevel > 0) {
            // hqdn3d strength maps: MPEG-2 SD gets 4 (heavier temporal), HD gets 2 (spatial only).
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

    // scale_vaapi always present: normalizes to NV12 BT.709 TV range even without resize.
    if (needsResize) {
        const char *scaleMode = isUhd ? "" : ":mode=hq"; // HQ uses bicubic; too expensive at 4K on most GPUs
        filters.push_back(std::format("scale_vaapi=w={}:h={}{}:format=nv12:out_color_matrix=bt709:out_range=tv",
                                      filterWidth, filterHeight, scaleMode));
    } else {
        filters.emplace_back("scale_vaapi=format=nv12:out_color_matrix=bt709:out_range=tv");
    }

    if (sharpnessLevel > 0 && vaapiContext->hasSharpness) {
        filters.push_back(std::format("sharpness_vaapi=sharpness={}", sharpnessLevel));
    }

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

    // DVB default: 50 fps (25 fps interlaced = 50 fields/s). VBR streams may report 0.
    const int fpsNum = codecCtx->framerate.num > 0 ? codecCtx->framerate.num : 50;
    const int fpsDen = codecCtx->framerate.den > 0 ? codecCtx->framerate.den : 1;

    // Numeric pix_fmt avoids name-lookup issues across FFmpeg versions with HW format aliases.
    const std::string bufferSrcArgs =
        std::format("video_size={}x{}:pix_fmt={}:time_base=1/90000:pixel_aspect={}/{}:frame_rate={}/{}", srcWidth,
                    srcHeight, static_cast<int>(srcPixFmt), sarNum, sarDen, fpsNum, fpsDen);

    dsyslog("vaapivideo/decoder: buffer source args='%s'", bufferSrcArgs.c_str());

    // alloc_filter (not create_filter): hw_frames_ctx must be set BEFORE init; FFmpeg 7.x
    // validates in init_video() and rejects VAAPI format without a valid frames context.
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
        // hwFramesParams is a plain C struct allocated by av_malloc; the hw_frames_ctx ref
        // inside was transferred to the filter node by av_buffersrc_parameters_set().
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

    // FFmpeg naming is counterintuitive: "outputs" connects to the source, "inputs" to the sink.
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

    // rate=field deinterlacing doubles the output frame rate (e.g. 25i -> 50p).
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

[[nodiscard]] auto cVaapiDecoder::SubmitTrickFrame(std::unique_ptr<VaapiFrame> frame) -> bool {
    const int64_t pts = frame->pts;
    const int64_t prevPts = prevTrickPts.load(std::memory_order_relaxed);

    // Reverse trick mode: VDR sends GOPs in reverse order but frames within each GOP arrive
    // in forward (decode) order. Drop non-decreasing PTS to show only one frame per GOP.
    if (isTrickReverse.load(std::memory_order_relaxed) && pts != AV_NOPTS_VALUE && prevPts != AV_NOPTS_VALUE &&
        pts > prevPts) {
        return true;
    }

    // Deinterlaced field pairs share the same source PTS; only pace and re-arm on new source frames.
    if (pts != prevPts) {
        prevTrickPts.store(pts, std::memory_order_relaxed);
        if (pts != AV_NOPTS_VALUE) {
            lastPts.store(pts, std::memory_order_release);
        }

        // Wait for pacing timer, then arm the next one.
        const uint64_t due = nextTrickFrameDue.load(std::memory_order_relaxed);
        while (cTimeMs::Now() < due && !stopping.load(std::memory_order_relaxed) &&
               trickSpeed.load(std::memory_order_relaxed) != 0) {
            cCondWait::SleepMs(10);
        }

        // Fast: PTS-derived hold; slow: fixed trickHoldMs.
        const uint64_t mult = trickMultiplier.load(std::memory_order_relaxed);
        if (mult > 0 && pts != AV_NOPTS_VALUE && prevPts != AV_NOPTS_VALUE) {
            const auto ptsDelta = static_cast<uint64_t>(std::abs(pts - prevPts));
            const uint64_t holdMs = std::clamp(ptsDelta / (uint64_t{90} * mult), uint64_t{10}, uint64_t{2000});
            nextTrickFrameDue.store(cTimeMs::Now() + holdMs, std::memory_order_relaxed);
        } else {
            nextTrickFrameDue.store(cTimeMs::Now() + trickHoldMs.load(std::memory_order_relaxed),
                                    std::memory_order_relaxed);
        }
    }

    return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
}

auto cVaapiDecoder::WaitForAudioCatchUp(cAudioProcessor *ap, int64_t pts, int64_t latency, int64_t delta) -> void {
    dsyslog("vaapivideo/decoder: sync ahead d=%+lldms -- waiting for audio", static_cast<long long>(delta / 90));

    // Hard cap prevents indefinite blocking on broken streams or dead audio pipeline.
    // Wait at most "delta + 1 s safety margin", clamped to 5 s.
    const int64_t maxWaitMs = std::min<int64_t>((delta / 90) + 1000, 5000LL);
    const cTimeMs deadline(static_cast<int>(maxWaitMs));

    while (!deadline.TimedOut() && !stopping.load(std::memory_order_relaxed)) {
        // Clear() arms freerunFrames: the frame is stale, stop blocking.
        if (freerunFrames.load(std::memory_order_relaxed) > 0) {
            break;
        }
        const int64_t freshClock = ap->GetClock();
        if (freshClock == AV_NOPTS_VALUE || (pts - freshClock - latency) <= 0) {
            break;
        }
        cCondWait::SleepMs(10);
    }

    ResetSmoothedDelta();
    syncCooldown.Set(DECODER_SYNC_COOLDOWN_MS);
    syncLogPending.store(true, std::memory_order_relaxed);
}

[[nodiscard]] auto cVaapiDecoder::SyncLatency90k() const noexcept -> int64_t {
    // External audio delay (AV receiver) + one frame period for DRM scanout pipeline.
    return (static_cast<int64_t>(vaapiConfig.audioLatency.load(std::memory_order_relaxed)) +
            static_cast<int64_t>(outputFrameDurationMs)) *
           90;
}

auto cVaapiDecoder::ResetSmoothedDelta() noexcept -> void {
    smoothedDeltaValid = false;
    smoothedDelta90k = 0;
    warmupSampleCount = 0;
    warmupSampleSum90k = 0;
}

auto cVaapiDecoder::UpdateSmoothedDelta(int64_t rawDelta90k) noexcept -> void {
    // Two-phase smoother:
    //   1. Warmup: collect WARMUP_SAMPLES raw samples into a simple mean. Seeding
    //      the EMA from a single noisy sample would bias it for the next ~5 s
    //      (the raw delta on deinterlaced 50p oscillates by ~150 ms between
    //      alternating output frames). A 50-sample mean cuts the seed error
    //      by sqrt(50), so the EMA starts close to the true mean.
    //   2. Steady state: low-pass with alpha = 1/EMA_SAMPLES (~5 s time
    //      constant) to reject jitter while tracking drift.
    // The soft-correction path checks smoothedDeltaValid, so corrections are
    // suppressed during warmup -- another reason not to act on noisy seed data.
    if (!smoothedDeltaValid) {
        warmupSampleSum90k += rawDelta90k;
        ++warmupSampleCount;
        if (warmupSampleCount < DECODER_SYNC_WARMUP_SAMPLES) {
            return;
        }
        smoothedDelta90k = warmupSampleSum90k / warmupSampleCount;
        smoothedDeltaValid = true;
        warmupSampleCount = 0;
        warmupSampleSum90k = 0;
        return;
    }
    smoothedDelta90k += (rawDelta90k - smoothedDelta90k) / DECODER_SYNC_EMA_SAMPLES;
}

auto cVaapiDecoder::LogSyncStats(int64_t rawDelta90k, const cAudioProcessor *ap) -> void {
    if (!(syncLogPending.exchange(false, std::memory_order_relaxed) || nextSyncLog.TimedOut())) {
        return;
    }
    // Suppress logging while the EMA is still warming up: a partial mean of
    // 1-49 samples is noisy and would mislead the operator into thinking the
    // avg "jumps" once warmup completes. We do NOT reschedule the timer here
    // so the next call (immediately after warmup finishes) emits the line.
    if (!smoothedDeltaValid) {
        return;
    }
    const auto avgTenths = static_cast<long long>(smoothedDelta90k * 10 / 90);
    dsyslog("vaapivideo/decoder: sync d=%+lldms avg=%+lld.%01lldms buf=%zu aq=%zu miss=%d drop=%d skip=%d",
            static_cast<long long>(rawDelta90k / 90), avgTenths / 10, std::abs(avgTenths % 10), jitterBuf.size(),
            ap->GetQueueSize(), drainMissCount, syncDropSinceLog, syncSkipSinceLog);
    drainMissCount = 0;
    syncDropSinceLog = 0;
    syncSkipSinceLog = 0;
    nextSyncLog.Set(DECODER_SYNC_LOG_INTERVAL_MS);
}

auto cVaapiDecoder::RunJitterPrimeSync(cAudioProcessor *ap) -> void {
    // One-shot coarse alignment performed exactly once after the jitter buffer
    // first reaches its prime fill level. The goal is to land the head-of-queue
    // frame close to the audio clock so that the steady-state soft sync in
    // SyncAndSubmitFrame() only has to handle small residuals.
    //
    //   behind: head PTS already trails audio -> drop stale frames until the
    //           head is at or past the audio clock. If the buffer empties we
    //           simply return and the caller will re-prime on the next refill.
    //   ahead:  head PTS leads the audio clock -> busy-wait (10 ms granularity)
    //           until audio catches up, then drop any frames that overshot
    //           during the wait so playback resumes exactly on the boundary.
    if (jitterBuf.empty()) {
        return;
    }
    // Snapshot PTS and clock once for the initial "behind vs ahead" decision.
    // The loops below re-read the clock so they react to audio progress.
    const int64_t fpts0 = jitterBuf.front()->pts;
    const int64_t clock0 = ap->GetClock();
    if (fpts0 == AV_NOPTS_VALUE || clock0 == AV_NOPTS_VALUE) {
        // Without a valid PTS or audio clock there is nothing to align against;
        // SyncAndSubmitFrame() will fall through to its freerun path.
        return;
    }
    const int64_t latency = SyncLatency90k();
    // Positive delta = video ahead of audio; negative = video behind.
    // Latency accounts for the audio sink's pending PCM that has not yet been
    // played out, so the comparison is "PTS vs the sample currently audible".
    const int64_t initDelta = fpts0 - clock0 - latency;

    if (initDelta < 0) {
        // --- Video behind audio: drop stale frames at the head ---
        int drops = 0;
        while (!jitterBuf.empty() && !stopping.load(std::memory_order_relaxed)) {
            if (jitterBuf.front()->pts == AV_NOPTS_VALUE) {
                // Unknown PTS -- cannot decide, stop dropping and let the
                // steady-state path handle it (likely as freerun).
                break;
            }
            const int64_t clk = ap->GetClock();
            if (clk == AV_NOPTS_VALUE || jitterBuf.front()->pts >= clk + latency) {
                // Head frame has caught up to (or passed) the audio clock.
                break;
            }
            jitterBuf.pop_front();
            ++drops;
        }
        dsyslog("vaapivideo/decoder: prime-sync dropped %d frames (initDelta=%+lldms)", drops,
                static_cast<long long>(initDelta / 90));
    } else if (initDelta > 0) {
        // --- Video ahead of audio: hold the buffer until audio catches up ---
        dsyslog("vaapivideo/decoder: prime-sync ahead d=%+lldms -- waiting for audio",
                static_cast<long long>(initDelta / 90));
        // Block in 10 ms slices. We re-read the clock each iteration so we
        // wake up promptly once the audio thread has consumed enough PCM.
        while (!stopping.load(std::memory_order_relaxed)) {
            const int64_t clk = ap->GetClock();
            if (clk == AV_NOPTS_VALUE || clk >= fpts0 - latency) {
                break;
            }
            cCondWait::SleepMs(10);
        }
        // Audio may have advanced past fpts0 while we were sleeping (we wake
        // on a 10 ms tick, not on an edge). Trim any frames whose PTS is now
        // behind the freshly observed clock so the next submit lands cleanly.
        // We always keep at least one frame so the display is not starved.
        int drops = 0;
        while (jitterBuf.size() > 1 && !stopping.load(std::memory_order_relaxed)) {
            if (jitterBuf.front()->pts == AV_NOPTS_VALUE) {
                break;
            }
            const int64_t clk = ap->GetClock();
            if (clk == AV_NOPTS_VALUE || jitterBuf.front()->pts - clk - latency >= 0) {
                break;
            }
            jitterBuf.pop_front();
            ++drops;
        }
        if (drops > 0) {
            dsyslog("vaapivideo/decoder: prime-sync overshoot correction: dropped %d frames", drops);
        }
    }

    // Reseed the EMA and arm the cooldown: the steady-state controller must
    // not react to the (possibly large) raw deltas seen immediately after a
    // coarse jump. nextSyncLog=0 forces an immediate stat line on the next
    // submit so the log shows the post-prime baseline.
    ResetSmoothedDelta();
    syncCooldown.Set(DECODER_SYNC_COOLDOWN_MS);
    nextSyncLog.Set(0);
}

auto cVaapiDecoder::SkipStaleJitterFrames(cAudioProcessor *ap) -> void {
    // Transient catch-up helper, called from the decode loop after events that
    // can leave a backlog of unplayable frames in the jitter buffer (decode
    // stall, scheduling hiccup, USB audio re-sync, ...). Anything more than
    // HARD_THRESHOLD behind the audio clock is unrecoverable by the soft
    // controller, so we drop it here in one shot rather than letting
    // SyncAndSubmitFrame() drop frames one-by-one (which would also incur a
    // grace window per drop).
    //
    // We always keep at least one frame so the display pipeline is not left
    // empty, and we abort on AV_NOPTS_VALUE because we can't compare an
    // unknown PTS against the clock.
    const int64_t latency = SyncLatency90k();
    while (jitterBuf.size() > 1 && !stopping.load(std::memory_order_relaxed)) {
        if (jitterBuf.front()->pts == AV_NOPTS_VALUE) {
            break;
        }
        const int64_t clock = ap->GetClock();
        // Stop as soon as the head is within the hard window; the soft path
        // will handle the residual.
        if (clock == AV_NOPTS_VALUE || jitterBuf.front()->pts - clock - latency >= -DECODER_SYNC_HARD_THRESHOLD) {
            break;
        }
        jitterBuf.pop_front();
        ++syncDropSinceLog;
    }
}

[[nodiscard]] auto cVaapiDecoder::SyncAndSubmitFrame(std::unique_ptr<VaapiFrame> frame) -> bool {
    // Steady-state A/V sync gate. Every decoded frame passes through here on
    // its way to the display. The audio clock is the master timebase; video
    // is paced to it. See AVSYNC.md for the full design rationale.
    //
    //   bypass: trick mode, freerun, no clock, no PTS -> submit immediately
    //   hard:   |raw delta| > HARD -> always fire (not gated by cooldown):
    //              behind -> drop frame (channel switch / stream gap flush)
    //              ahead  -> block until audio catches up (replay only)
    //   soft:   |EMA| > CORRIDOR AND cooldown expired -> proportional correction:
    //              behind -> drop ceil(|EMA|/frameDur) frames, capped at MAX_CORRECTION
    //                        (one frame this call, remainder via pendingDrops)
    //              ahead  -> sleep min(EMA, MAX_CORRECTION) ms in one shot
    //
    // Both soft paths adjust smoothedDelta90k by the *exact* known correction
    // amount instead of resetting the EMA. This preserves the smoother's
    // history so the next sample doesn't reseed from a single noisy raw value
    // (the raw delta on deinterlaced 50p output oscillates by ~150 ms between
    // alternating output frames). Math:
    //   sleep N ms : audio advances N while video sits still -> EMA -= N*90
    //   drop frame : next display PTS is +frameDur with no clock advance ->
    //                EMA += frameDur*90
    //
    // Rate-limited to one event per COOLDOWN (5 s, = one EMA time constant).
    // With MAX_CORRECTION=100 ms the controller can absorb up to ~20 ms/s
    // sustained drift -- far above any realistic stream. Real-world correction
    // rate is governed by the corridor + actual drift, not the cooldown.

    if (!frame || !display) [[unlikely]] {
        return false;
    }

    const int64_t pts = frame->pts;

    // --- Trick mode: VDR is in fast-forward / slow-motion / reverse ---
    // Trick playback uses its own pacing timer (SubmitTrickFrame) and
    // ignores the audio clock entirely, because audio is muted in this mode.
    if (trickSpeed.load(std::memory_order_acquire) != 0) {
        if (trickExitPending.exchange(false, std::memory_order_acquire)) {
            // VDR called Play() without a matching TrickSpeed(0) -- this is a
            // legal transition out of trick mode. Reset the trick state and
            // arm a freerun window so the first few normal-speed frames are
            // submitted without sync corrections (the audio clock has not yet
            // re-anchored to the new PTS stream).
            trickSpeed.store(0, std::memory_order_relaxed);
            isTrickReverse.store(false, std::memory_order_relaxed);
            isTrickFastForward.store(false, std::memory_order_relaxed);
            freerunFrames.store(DECODER_SYNC_FREERUN_FRAMES, std::memory_order_relaxed);
            syncLogPending.store(true, std::memory_order_relaxed);
            // Fall through to normal A/V sync below.
        } else {
            return SubmitTrickFrame(std::move(frame));
        }
    }

    // Publish the latest valid PTS so external observers (still-image
    // detection, position queries, ...) can read it lock-free.
    if (pts != AV_NOPTS_VALUE) {
        lastPts.store(pts, std::memory_order_release);
    }

    // --- Freerun: no master clock available, submit unpaced ---
    // Cases that hit this branch:
    //   * audio processor not attached (radio mode / no audio stream)
    //   * frame has no PTS (e.g. some MPEG-2 B-frames)
    //   * we are inside the post-Clear() / post-trick-exit freerun window,
    //     during which the audio clock is unreliable
    auto *const ap = audioProcessor.load(std::memory_order_acquire);
    if (!ap || pts == AV_NOPTS_VALUE) {
        return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
    }
    if (freerunFrames.load(std::memory_order_relaxed) > 0) {
        freerunFrames.fetch_sub(1, std::memory_order_relaxed);
        return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
    }

    // --- Audio pipeline attached but not yet running (no PCM written yet) ---
    // GetClock() returns AV_NOPTS_VALUE until the audio thread has actually
    // pushed samples to the sink. We submit freely in the meantime so the
    // display does not stall waiting for audio to start.
    const int64_t latency = SyncLatency90k();
    const int64_t clock = ap->GetClock();
    if (clock == AV_NOPTS_VALUE) {
        if (syncLogPending.exchange(false, std::memory_order_relaxed) || nextSyncLog.TimedOut()) {
            dsyslog("vaapivideo/decoder: sync freerun (no clock) buf=%zu", jitterBuf.size());
            nextSyncLog.Set(DECODER_SYNC_LOG_INTERVAL_MS);
        }
        return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
    }

    // --- Continue an in-progress drop burst from a previous correction ---
    // Multi-frame drops are spread across consecutive calls so we never burst
    // through the jitter buffer all at once. Each step adjusts the EMA by the
    // exact known correction amount.
    if (pendingDrops > 0) {
        --pendingDrops;
        smoothedDelta90k += static_cast<int64_t>(outputFrameDurationMs) * 90;
        ++syncDropSinceLog;
        return true;
    }

    // --- Measure raw delta and feed the EMA smoother ---
    // rawDelta is the instantaneous error: positive = video ahead, negative =
    // video behind. The EMA (smoothedDelta90k) filters jitter so the soft
    // controller reacts to sustained drift, not to per-frame noise.
    const int64_t rawDelta = pts - clock - latency;
    UpdateSmoothedDelta(rawDelta);
    LogSyncStats(rawDelta, ap);

    // --- Hard transient drop: video far behind audio ---
    // Channel switch, stream gap, decode stall recovery. NOT gated by the
    // cooldown -- a burst of stale frames must be flushed immediately. Resets
    // the EMA because the post-burst state is unrelated to pre-burst.
    if (rawDelta < -DECODER_SYNC_HARD_THRESHOLD) [[unlikely]] {
        ++syncDropSinceLog;
        ResetSmoothedDelta();
        pendingDrops = 0;
        syncCooldown.Set(DECODER_SYNC_COOLDOWN_MS);
        return true;
    }

    // --- Hard transient hold (replay only): video far ahead of audio ---
    // Post-seek: video lands in the future relative to audio. Block until
    // audio catches up. Disabled in live mode -- a live source can't be
    // paused server-side, so blocking would just grow the upstream buffer.
    if (!liveMode.load(std::memory_order_relaxed) && rawDelta > DECODER_SYNC_HARD_THRESHOLD) [[unlikely]] {
        WaitForAudioCatchUp(ap, pts, latency, rawDelta);
        ++syncSkipSinceLog;
        pendingDrops = 0;
        return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
    }

    // --- Soft corridor correction (rate-limited, proportional) ---
    // Triggered when |EMA| > CORRIDOR and the cooldown has elapsed. The
    // correction amount is "enough to bring the EMA back to ~0", capped at
    // MAX_CORRECTION per event. We adjust the EMA by the exact correction
    // amount instead of resetting it -- the smoother's history is preserved
    // so the next sample doesn't reseed from a single noisy raw value.
    if (syncCooldown.TimedOut() && smoothedDeltaValid) {
        const int64_t absDelta90k = smoothedDelta90k < 0 ? -smoothedDelta90k : smoothedDelta90k;
        if (absDelta90k > DECODER_SYNC_CORRIDOR) {
            const int correctMs = std::min(static_cast<int>(absDelta90k / 90), DECODER_SYNC_MAX_CORRECTION_MS);
            syncCooldown.Set(DECODER_SYNC_COOLDOWN_MS);

            if (smoothedDelta90k < 0) {
                // Video lagging: schedule N drops, execute one now, defer rest.
                const int totalDrops = std::max(1, correctMs / outputFrameDurationMs);
                pendingDrops = totalDrops - 1;
                smoothedDelta90k += static_cast<int64_t>(outputFrameDurationMs) * 90;
                ++syncDropSinceLog;
                return true;
            }
            // Video leading: sleep once -- audio catches up while video sits still.
            cCondWait::SleepMs(correctMs);
            smoothedDelta90k -= static_cast<int64_t>(correctMs) * 90;
            ++syncSkipSinceLog;
        }
    }

    return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
}
