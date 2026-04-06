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

constexpr size_t DECODER_QUEUE_CAPACITY = 500;       ///< Video packet queue depth (~10 s at 50 fps)
constexpr int DECODER_SUBMIT_TIMEOUT_MS = 100;       ///< Timeout when submitting a frame to the display (ms)
constexpr int64_t DECODER_SYNC_THRESHOLD = 200 * 90; ///< A/V sync threshold in 90 kHz ticks (~200 ms):
                                                     ///<   drop when behind, wait-before-submit when ahead (replay)
constexpr int DECODER_SYNC_GRACE_MS = 500;           ///< Grace period after sync correction (ms)
constexpr int DECODER_SYNC_FREERUN_FRAMES = 1;       ///< Frames submitted immediately after Clear() without sync
constexpr int DECODER_JITTER_BUFFER_MS = 500;        ///< Jitter buffer target depth (ms)
constexpr int DECODER_SYNC_LOG_INTERVAL_MS = 3ß000;  ///< Interval between periodic A/V sync log lines (ms)

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
    smoothedDeltaValid = false;
    // driftIntegral90k intentionally NOT reset: it represents hardware drift, not channel state.

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

    const std::unique_ptr<AVPacket, FreeAVPacket> workPacket{av_packet_alloc()};
    if (!workPacket) [[unlikely]] {
        esyslog("vaapivideo/decoder: failed to allocate packet");
        hasExited.store(true, std::memory_order_release);
        return;
    }

    std::vector<std::unique_ptr<VaapiFrame>> pendingFrames;
    bool primeSyncPending{false};             // one-shot settle requested at jitter-buffer prime
    uint64_t drainFrameCount{0};              // frames drained since last fps reset
    uint64_t drainFpsStartMs{cTimeMs::Now()}; // timestamp of last fps calculation reset
    uint64_t lastDrainMs{0};                  // timestamp of most recent drain for gap detection

    while (!stopping.load(std::memory_order_acquire)) {
        std::unique_ptr<AVPacket, FreeAVPacket> queuedPacket;

        // --- Dequeue: timed wait avoids busy-looping when no packets are pending ---
        //
        // When the jitter buffer is primed and has frames to drain, use a short
        // timeout (15 ms) so the loop iterates well before the next VSync.  The
        // Phase-2 ahead gate (audio-clock PTS check) decides whether to actually
        // drain a frame, and SubmitFrame's single-slot backpressure throttles us
        // if the display thread hasn't consumed the previous frame yet.  This
        // makes the effective drain cadence track VSync rather than an imprecise
        // wall-clock timer: the decoder delivers early, the display thread
        // presents at exactly the VSync rate.
        //
        // Without this, TimedWait(20 ms) stacks on top of any SubmitFrame
        // blocking, degrading 50 fps rate=field content to ~33 fps and causing
        // the jitter buffer to grow.
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
            // Copy into workPacket so queuedPacket can be freed before the decode call,
            // reducing packetMutex contention for EnqueueData().
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
        // Apply any pending jitter flush requested by Clear().  Checked here rather than in Clear()
        // because the jitter fields are decoder-thread-owned: Clear() cannot safely modify them
        // while the delivery section may be iterating over them (no lock is held at this point).
        if (jitterFlushPending.exchange(false, std::memory_order_acquire)) {
            jitterBuf.clear();
            jitterPrimed = false;
            correctDrops = 0;
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
            // SyncAndSubmitFrame() consumes freerunFrames via its Phase-2 path.
            if (freerunFrames.load(std::memory_order_relaxed) > 0) {
                jitterBuf.clear();
                jitterPrimed = false;
                if (frameIt != pendingFrames.end() && !stopping.load(std::memory_order_relaxed)) {
                    (void)SyncAndSubmitFrame(std::move(*frameIt));
                    ++frameIt;
                }
            }

            // Guard: if jitterTarget is still 0 (InitFilterGraph not yet run or failed),
            // submit directly to avoid unbounded accumulation in jitterBuf.
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

                // Prime: accumulate jitterTarget frames before draining.  Count-based
                // so rate=field deinterlaced content is measured correctly.
                if (!jitterPrimed && static_cast<int>(jitterBuf.size()) >= jitterTarget) {
                    jitterPrimed = true;
                    primeSyncPending = true; // request one-shot settle before first drain
                    smoothedDeltaValid = false;
                    syncGrace.Set(DECODER_SYNC_GRACE_MS);
                    nextSyncLog.Set(DECODER_SYNC_LOG_INTERVAL_MS);
                    // Reset drain tracking for this channel.
                    drainFrameCount = 0;
                    drainFpsStartMs = cTimeMs::Now();
                    lastDrainMs = 0;
                    drainFps = 0;
                    drainMissCount = 0;
                    dsyslog("vaapivideo/decoder: jitter buffer primed (buf=%zu target=%d)", jitterBuf.size(),
                            jitterTarget);
                }

                // Underrun: buffer was primed but drained empty. Re-prime to rebuild the cushion.
                if (jitterPrimed && jitterBuf.empty()) {
                    dsyslog("vaapivideo/decoder: jitter buffer underrun -- re-priming (target=%d)", jitterTarget);
                    jitterPrimed = false;
                }

                // Two-phase drain: fast-skip then single VSync-paced frame.
                //
                // Phase 1 skips frames hopelessly behind audio at CPU speed (no
                // SubmitFrame).  Phase 2 drains at most one frame per iteration
                // with an ahead gate for one-shot startup sync.
                //
                // A while-SubmitFrame loop would cascade: SubmitFrame blocks to
                // VSync (~20 ms), advancing the audio clock just enough for the
                // NEXT frame to appear "due", draining the entire buffer in one
                // pass.  Single drain avoids this; the main loop naturally
                // iterates at the output frame rate (TimedWait fires at
                // outputFrameDurationMs), so one drain per iteration matches the
                // display refresh.  Rate=field 25→50 fps is paced by extra
                // TimedWait timeout iterations between source packets.
                if (jitterPrimed && !jitterBuf.empty() && !stopping.load(std::memory_order_relaxed)) {
                    if (auto *const ap = audioProcessor.load(std::memory_order_acquire); ap) {
                        // Prime-time one-shot sync: align video front to the audio clock
                        // before the first drain so the PI controller starts from delta ≈ 0.
                        //
                        //  initDelta < 0  video is BEHIND audio: fast-drop stale jitter frames
                        //                 until the front PTS meets the clock.
                        //
                        //  initDelta > 0  video is AHEAD of audio: block until the audio clock
                        //                 advances to meet the video front PTS.
                        //
                        // EMA is reset so the P-term starts fresh.  The I-term (learned HW
                        // drift rate) is preserved across channel switches for instant relock.
                        if (primeSyncPending && !jitterBuf.empty()) {
                            primeSyncPending = false;
                            const int64_t fpts0 = jitterBuf.front()->pts;
                            const int64_t clock0 = ap->GetClock();
                            if (fpts0 != AV_NOPTS_VALUE && clock0 != AV_NOPTS_VALUE) {
                                const int64_t primeLatency = SyncLatency90k();
                                const int64_t initDelta = fpts0 - clock0 - primeLatency;
                                if (initDelta < 0) {
                                    // Video is behind: drop frames until front ≥ audio clock.
                                    // Allow draining the entire buffer: when the deficit exceeds
                                    // one buffer's worth, all frames are stale.  The buffer then
                                    // empties, underrun triggers re-prime, and the cycle repeats
                                    // at CPU speed until the deficit fits within one buffer.
                                    int primeDrops = 0;
                                    while (!jitterBuf.empty() && !stopping.load(std::memory_order_relaxed)) {
                                        if (jitterBuf.front()->pts == AV_NOPTS_VALUE) {
                                            break;
                                        }
                                        const int64_t clk = ap->GetClock();
                                        if (clk == AV_NOPTS_VALUE || jitterBuf.front()->pts >= clk + primeLatency) {
                                            break;
                                        }
                                        jitterBuf.pop_front();
                                        ++primeDrops;
                                    }
                                    dsyslog("vaapivideo/decoder: prime-sync dropped %d frames (initDelta=%+lldms)",
                                            primeDrops, static_cast<long long>(initDelta / 90));
                                } else if (initDelta > 0) {
                                    // Video is ahead: wait for the audio clock to catch up.
                                    dsyslog("vaapivideo/decoder: prime-sync ahead d=%+lldms -- waiting for audio",
                                            static_cast<long long>(initDelta / 90));
                                    while (!stopping.load(std::memory_order_relaxed)) {
                                        const int64_t clk = ap->GetClock();
                                        if (clk == AV_NOPTS_VALUE || clk >= fpts0 - primeLatency) {
                                            break;
                                        }
                                        cCondWait::SleepMs(10);
                                    }
                                    // Correct clock overshoot: the audio clock is updated in
                                    // ~24 ms steps so it can jump past the target.  Drop any
                                    // jitter frames that are now behind the audio position.
                                    int postDrops = 0;
                                    while (jitterBuf.size() > 1 && !stopping.load(std::memory_order_relaxed)) {
                                        if (jitterBuf.front()->pts == AV_NOPTS_VALUE) {
                                            break;
                                        }
                                        const int64_t clk = ap->GetClock();
                                        if (clk == AV_NOPTS_VALUE || jitterBuf.front()->pts - clk - primeLatency >= 0) {
                                            break;
                                        }
                                        jitterBuf.pop_front();
                                        ++postDrops;
                                    }
                                    if (postDrops > 0) {
                                        dsyslog("vaapivideo/decoder: prime-sync overshoot correction: "
                                                "dropped %d frames",
                                                postDrops);
                                    }
                                }
                                // Reset EMA so P-term starts fresh; keep I-term (HW drift).
                                smoothedDeltaValid = false;
                                syncGrace.Set(DECODER_SYNC_GRACE_MS);
                                nextSyncLog.Set(0); // log immediately to confirm
                            }
                        }

                        // Phase 1: Fast-skip frames hopelessly behind audio at CPU speed.
                        while (jitterBuf.size() > 1 && !stopping.load(std::memory_order_relaxed)) {
                            if (jitterBuf.front()->pts == AV_NOPTS_VALUE) {
                                break;
                            }
                            const int64_t clock = ap->GetClock();
                            if (clock == AV_NOPTS_VALUE) {
                                break;
                            }
                            const int64_t frontDelta = jitterBuf.front()->pts - clock - SyncLatency90k();
                            if (frontDelta >= -DECODER_SYNC_THRESHOLD) {
                                break;
                            }
                            jitterBuf.pop_front();
                            ++correctDrops;
                        }

                        // Buffer regulation: cap jitter buffer to prevent unbounded
                        // growth.  Drop from the BACK (newest/furthest-in-future) to
                        // keep the front aligned with the audio clock. Dropping from
                        // the front would jump the front PTS far ahead of audio,
                        // stalling drain and causing a runaway feedback loop.
                        constexpr int JITTER_MAX_FACTOR = 3;
                        const int maxJitter = jitterTarget * JITTER_MAX_FACTOR;
                        if (static_cast<int>(jitterBuf.size()) > maxJitter) {
                            int bufDrops = 0;
                            while (static_cast<int>(jitterBuf.size()) > jitterTarget * 2 && jitterBuf.size() > 1) {
                                jitterBuf.pop_back();
                                ++bufDrops;
                            }
                            if (bufDrops > 0) {
                                dsyslog("vaapivideo/decoder: buffer regulation: dropped %d "
                                        "(buf=%zu target=%d)",
                                        bufDrops, jitterBuf.size(), jitterTarget);
                                smoothedDeltaValid = false;
                                syncGrace.Set(DECODER_SYNC_GRACE_MS);
                                nextSyncLog.Set(0);
                            }
                        }

                        // Audio overflow safety net: one-shot flush when the audio
                        // queue grows beyond the emergency threshold.  With buffer
                        // regulation the PI delta stays accurate and comp converges
                        // to the true drift, so this should rarely fire.
                        constexpr int AQ_EMERGENCY = 50;
                        constexpr int AQ_TARGET = 10;
                        if (static_cast<int>(ap->GetQueueSize()) >= AQ_EMERGENCY && syncGrace.TimedOut()) {
                            const int excess = static_cast<int>(ap->GetQueueSize()) - AQ_TARGET;
                            if (excess > 0) {
                                const int aDropped = ap->DropPackets(excess);
                                dsyslog("vaapivideo/decoder: audio overflow: "
                                        "dropped %d packets (aq=%zu)",
                                        aDropped, ap->GetQueueSize());
                            }
                        }

                        // Phase 2: Drain a single frame.
                        // VSync backpressure from SubmitFrame's single-slot naturally
                        // limits drain to the display refresh rate.  Prime-sync handles
                        // initial alignment; the PI controller handles steady-state.
                        // No ahead gate: it was starving drain rate (fps < source rate)
                        // and causing buffer growth that triggered regulation, whose
                        // front-drops then jumped the PTS far ahead of audio, blocking
                        // all further drains in a runaway feedback loop.
                        if (!jitterBuf.empty() && !stopping.load(std::memory_order_relaxed)) {
                            ++drainFrameCount;
                            const uint64_t nowMs = cTimeMs::Now();
                            if (lastDrainMs > 0) {
                                const auto gap = static_cast<int>(nowMs - lastDrainMs);
                                if (gap > outputFrameDurationMs * 2) {
                                    ++drainMissCount;
                                }
                            }
                            lastDrainMs = nowMs;
                            const uint64_t elapsed = nowMs - drainFpsStartMs;
                            if (elapsed >= 2000) {
                                drainFps = static_cast<double>(drainFrameCount * 1000) / static_cast<double>(elapsed);
                                drainFrameCount = 0;
                                drainFpsStartMs = nowMs;
                            }
                            auto drainFrame = std::move(jitterBuf.front());
                            jitterBuf.pop_front();
                            (void)SyncAndSubmitFrame(std::move(drainFrame));
                        }
                    } else {
                        // No audio processor: drain one frame (freerun).
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
            // vaDriverMutex serializes all VA driver calls (filter graph + surface sync) between
            // the decode and display threads; concurrent VA operations on the same VADisplay are not thread-safe.
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

            // PTS assignment and trick-mode filtering happen outside the VA driver lock.
            // With rate=field deinterlacing each source frame produces two outputs.
            // Assign monotonically increasing PTS: first field keeps the source PTS,
            // second field advances by one field period so A/V sync sees a smooth timeline.
            for (size_t i = 0; i < outFrames.size(); ++i) {
                outFrames.at(i)->pts =
                    (sourcePts != AV_NOPTS_VALUE && i > 0)
                        ? sourcePts + (static_cast<int64_t>(outputFrameDurationMs) * 90 * static_cast<int64_t>(i))
                        : sourcePts;
            }

            // In FF/reverse trick mode, rate=field emits two outputs per source frame. The first has a
            // cross-frame blend ghost; only keep the second.
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

[[nodiscard]] auto cVaapiDecoder::SubmitTrickFrame(std::unique_ptr<VaapiFrame> frame) -> bool {
    const int64_t pts = frame->pts;
    const int64_t prevPts = prevTrickPts.load(std::memory_order_relaxed);

    // Reverse: enforce monotonically decreasing PTS.
    if (isTrickReverse.load(std::memory_order_relaxed) && pts != AV_NOPTS_VALUE && prevPts != AV_NOPTS_VALUE &&
        pts > prevPts) {
        return true; // drop out-of-order frame
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

    // Cap at 5 s to avoid indefinite blocking on broken streams.
    const int64_t maxWaitMs = std::min<int64_t>((delta / 90) + DECODER_SYNC_GRACE_MS, 5000LL);
    const cTimeMs deadline(static_cast<int>(maxWaitMs));

    while (!deadline.TimedOut() && !stopping.load(std::memory_order_relaxed)) {
        // Clear() sets freerunFrames to signal that a seek or channel switch happened;
        // the frame we're waiting to submit is now stale - stop blocking immediately.
        if (freerunFrames.load(std::memory_order_relaxed) > 0) {
            break;
        }
        const int64_t freshClock = ap->GetClock();
        if (freshClock == AV_NOPTS_VALUE || (pts - freshClock - latency) <= 0) {
            break;
        }
        cCondWait::SleepMs(10);
    }

    syncGrace.Set(DECODER_SYNC_GRACE_MS);
    syncLogPending.store(true, std::memory_order_relaxed);
}

[[nodiscard]] auto cVaapiDecoder::SyncLatency90k() const noexcept -> int64_t {
    // audioLatency compensates for external delays (AV receiver, etc.).
    // videoPipelineDelay accounts for the display path: after SubmitFrame() the frame is committed via
    // drmModeAtomicCommit() and scanned out at the next vblank - roughly one frame period end-to-end.
    return (static_cast<int64_t>(vaapiConfig.audioLatency.load(std::memory_order_relaxed)) +
            static_cast<int64_t>(outputFrameDurationMs)) *
           90;
}

[[nodiscard]] auto cVaapiDecoder::SyncAndSubmitFrame(std::unique_ptr<VaapiFrame> frame) -> bool {
    // -----------------------------------------------------------------------
    // A/V sync -- audio clock is master.  delta = vPTS - audioClock - latency
    //
    //   Trick mode:
    //     Pacing timer only; no A/V sync.
    //
    //   Live (jitter buffer active):
    //     delta >= -threshold  submit  (PI controller handles positive drift)
    //     delta <  -threshold  drop    (video behind audio)
    //
    //   Replay:
    //     |delta| <= threshold  submit  (VSync pacing)
    //     delta  <  -threshold  drop    (video behind audio)
    //     delta  >  +threshold  wait    (video ahead; hold until audio catches up)
    //
    // Called exclusively from the decode thread (Action).
    // -----------------------------------------------------------------------

    if (!frame || !display) [[unlikely]] {
        return false;
    }

    const int64_t pts = frame->pts;

    // --- Trick mode: pacing timer, no A/V sync ---

    if (trickSpeed.load(std::memory_order_acquire) != 0) {
        if (trickExitPending.exchange(false, std::memory_order_acquire)) {
            // Play() was called but TrickSpeed() did not follow -- leave trick mode.
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

    if (pts != AV_NOPTS_VALUE) {
        lastPts.store(pts, std::memory_order_release);
    }

    // --- Freerun: no audio processor, no PTS, or first frame after Clear() ---

    auto *const ap = audioProcessor.load(std::memory_order_acquire);
    if (!ap || pts == AV_NOPTS_VALUE) {
        return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
    }

    if (freerunFrames.load(std::memory_order_relaxed) > 0) {
        freerunFrames.fetch_sub(1, std::memory_order_relaxed);
        return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
    }

    // --- A/V sync ---

    const int64_t latency = SyncLatency90k();
    const int64_t clock = ap->GetClock();

    // Audio pipeline not yet started (PCM and passthrough before first write).
    if (clock == AV_NOPTS_VALUE) {
        if (syncLogPending.exchange(false, std::memory_order_relaxed) || nextSyncLog.TimedOut()) {
            dsyslog("vaapivideo/decoder: sync freerun (no clock) buf=%zu", jitterBuf.size());
            nextSyncLog.Set(DECODER_SYNC_LOG_INTERVAL_MS);
        }
        return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
    }

    const int64_t delta = pts - clock - latency;

    // --- Drift tracking (EMA, ~2 s time constant at 50 fps) ---
    if (!smoothedDeltaValid) {
        smoothedDelta90k = delta;
        smoothedDeltaValid = true;
    } else {
        smoothedDelta90k += (delta - smoothedDelta90k) / 100; // alpha ~= 0.01
    }

    // --- PI drift compensation for PCM audio ---
    //
    // Published to the audio processor which applies swr_set_compensation() to micro-adjust
    // the resampler output rate.  Units: samples per 10000 output samples.
    //
    // P-term (smoothedDelta90k/90): drives the current A/V offset toward zero.
    //   Fast response (~10 s), but drops to zero when avg=0, letting drift resume.
    //
    // I-term (driftIntegral90k): accumulates the error to learn the steady-state
    //   hardware drift rate.  When P has driven avg to 0, the I-term alone sustains the
    //   exact compensation needed to keep it there.  Accumulates raw smoothedDelta90k to
    //   avoid integer truncation (small deltas like 7200 / 10000 = 0 in integer math).
    //   The /I_DIVISOR is applied only on the output.  Time constant ~5 s at 50 fps.
    //   Clamped to prevent integral windup on transients.
    //
    // Together they form a PI controller: P handles convergence, I handles steady-state.
    //
    // Integral clamp: DVB source clocks can drift up to ~2% from HDMI/ALSA sinks;
    // ±200 in comp units (2%) covers pathological SAT>IP setups.
    constexpr int I_DIVISOR = 250;                             // I-term output divisor; ~5 s time constant at 50 fps
    constexpr int64_t INTEGRAL_CLAMP = 200LL * 90 * I_DIVISOR; // ±200 comp equivalent
    const int64_t newIntegral = std::clamp(driftIntegral90k + smoothedDelta90k, -INTEGRAL_CLAMP, INTEGRAL_CLAMP);
    const int rawComp = static_cast<int>(-(smoothedDelta90k + newIntegral / I_DIVISOR) / 90);
    // Anti-windup: accumulate the integral only when the output is not saturated.
    // When saturated, back-calculate the integral to the value that produces exactly
    // the clamped output.  Pure freeze (the previous approach) pins the integral at a
    // deep value from transients that takes minutes to unwind, holding comp at ±200
    // indefinitely and overflowing the audio queue.
    if (rawComp > -200 && rawComp < 200) {
        driftIntegral90k = newIntegral;
    } else {
        const auto clampedComp = static_cast<int64_t>(std::clamp(rawComp, -200, 200));
        driftIntegral90k =
            std::clamp((-clampedComp * 90 - smoothedDelta90k) * I_DIVISOR, -INTEGRAL_CLAMP, INTEGRAL_CLAMP);
    }
    const int comp = std::clamp(rawComp, -200, 200);
    ap->SetDriftCompensation(comp);

    if (syncLogPending.exchange(false, std::memory_order_relaxed) || nextSyncLog.TimedOut()) {
        dsyslog("vaapivideo/decoder: sync d=%+lldms avg=%+lld.%01lldms comp=%d i=%+lld.%01lldms "
                "buf=%zu aq=%zu fps=%.1f miss=%d",
                static_cast<long long>(delta / 90), static_cast<long long>(smoothedDelta90k / 90),
                std::abs(static_cast<long long>((smoothedDelta90k * 10 / 90) % 10)), comp,
                static_cast<long long>(driftIntegral90k / I_DIVISOR / 90),
                std::abs(static_cast<long long>((driftIntegral90k / I_DIVISOR * 10 / 90) % 10)), jitterBuf.size(),
                ap->GetQueueSize(), drainFps, drainMissCount);
        drainMissCount = 0;
        nextSyncLog.Set(DECODER_SYNC_LOG_INTERVAL_MS);
    }

    // Grace period: suppress corrections while the ALSA clock stabilizes.
    if (!syncGrace.TimedOut()) {
        return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
    }

    // Hard drop: frame far behind audio (safety net for transients/channel switch).
    if (delta < -DECODER_SYNC_THRESHOLD) {
        ++correctDrops;
        return true; // frame dropped
    }

    // After a hard-drop run: reset the EMA to the current (post-correction) delta so the
    // smoothed value doesn't carry a stale deficit that would trigger endless soft drops.
    if (correctDrops > 0) {
        dsyslog("vaapivideo/decoder: sync corrected %d dropped d=%+lldms", correctDrops,
                static_cast<long long>(delta / 90));
        correctDrops = 0;
        smoothedDelta90k = delta; // re-seed EMA from post-correction position
        driftIntegral90k = 0;     // reset: anti-windup value from transient ≠ drift
        syncGrace.Set(DECODER_SYNC_GRACE_MS);
    }

    // Ahead (replay only): hold the decode thread until audio catches up.
    // Live ahead correction is handled by the jitter-buffer ahead gate in Action().
    // Safe to block in replay: the file-reader pre-fills packet queues deep enough
    // that audio never starves.
    if (!liveMode.load(std::memory_order_relaxed) && delta > DECODER_SYNC_THRESHOLD) {
        WaitForAudioCatchUp(ap, pts, latency, delta);
    }

    return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
}
