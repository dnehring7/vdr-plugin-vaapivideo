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

constexpr size_t DECODER_QUEUE_CAPACITY = 500;            ///< Video packet queue depth (~10 s at 50 fps)
constexpr int DECODER_SUBMIT_TIMEOUT_MS = 100;            ///< Timeout when submitting a frame to the display (ms)
constexpr int64_t DECODER_SYNC_HARD_THRESHOLD = 200 * 90; ///< Raw-delta safety net (~200 ms): drop when behind,
                                                          ///<   block until catch-up when ahead (replay only).
                                                          ///<   Used for transients (channel switch, network gap).
constexpr int64_t DECODER_SYNC_TARGET_BAND = 15 * 90;     ///< Steady-state EMA dead band (+/-15 ms).  When the EMA
                                                          ///<   exceeds the band on the AHEAD side, the decode thread
                                                          ///<   sleeps the excess so the audio clock can catch up.
                                                          ///<   15 ms sits clear of the EMA noise floor (~5-10 ms at
                                                          ///<   5 s time constant) and well inside the imperceptible
                                                          ///<   lip-sync window (~+/-40 ms).  No corrective action is
                                                          ///<   taken on the BEHIND side -- the audio clock derives
                                                          ///<   from the source PTS via endPts-snd_pcm_delay, so it
                                                          ///<   cannot drift permanently faster than the DVB stream;
                                                          ///<   any sustained negative delta is a transient handled
                                                          ///<   by HARD_THRESHOLD below.
constexpr int DECODER_SYNC_MAX_WAIT_MS = 80;              ///< Cap on a single sleep (ms).  Limits the slow-down on a
                                                          ///<   large calibration shift to ~5x normal frame time.
constexpr int DECODER_SYNC_GRACE_MS = 1000;               ///< Grace period after a sync correction or state change
                                                          ///<   (prime-sync, hard transient, soft wait): suppresses
                                                          ///<   further corrections while the EMA reseeds.
constexpr int DECODER_SYNC_FREERUN_FRAMES = 1;            ///< Frames submitted immediately after Clear() without sync
constexpr int DECODER_JITTER_BUFFER_MS = 500;             ///< Jitter buffer target depth (ms)
constexpr int DECODER_SYNC_EMA_SAMPLES = 250;             ///< EMA divisor (alpha = 1/N).  ~5 s time constant @ 50 fps
constexpr int DECODER_SYNC_LOG_INTERVAL_MS = 30000;       ///< Interval between periodic A/V sync log lines (ms)

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
    bool primeSyncPending{false}; // one-shot settle requested at jitter-buffer prime
    uint64_t lastDrainMs{0};      // wall-clock timestamp of the previous drain, for miss detection

    while (!stopping.load(std::memory_order_acquire)) {
        std::unique_ptr<AVPacket, FreeAVPacket> queuedPacket;

        // --- Dequeue: timed wait avoids busy-looping when no packets are pending ---
        //
        // When the jitter buffer has frames to drain, use a slightly shorter
        // wait (18 ms) than one frame period: the loop iterates often enough
        // that SubmitFrame's single-slot backpressure -- not the wall-clock
        // timer -- paces drain to the display VSync.  Stacking TimedWait(20)
        // on top of any SubmitFrame blocking would degrade 50 fps rate=field
        // content to ~33 fps and grow the jitter buffer unbounded.
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
            // SyncAndSubmitFrame() consumes the freerunFrames counter.
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

                // Drain one frame per loop iteration.  SubmitFrame's single-slot
                // backpressure paces this to the display VSync (rate=field 25->50
                // is sustained because TimedWait above fires between source packets).
                // No buffer regulation: audio and video share the source clock, so
                // the jitter level tracks the audio queue -- any sustained growth is
                // an audio-path issue, not ours.
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

    smoothedDeltaValid = false;
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

auto cVaapiDecoder::UpdateSmoothedDelta(int64_t rawDelta90k) noexcept -> void {
    // EMA, alpha = 1/N (~N frames time constant).  Long enough to suppress ALSA
    // clock-period aliasing (~30 ms steps), short enough to converge within 1 s.
    if (!smoothedDeltaValid) {
        smoothedDelta90k = rawDelta90k;
        smoothedDeltaValid = true;
        return;
    }
    smoothedDelta90k += (rawDelta90k - smoothedDelta90k) / DECODER_SYNC_EMA_SAMPLES;
}

auto cVaapiDecoder::LogSyncStats(int64_t rawDelta90k, const cAudioProcessor *ap) -> void {
    if (!(syncLogPending.exchange(false, std::memory_order_relaxed) || nextSyncLog.TimedOut())) {
        return;
    }
    dsyslog("vaapivideo/decoder: sync d=%+lldms avg=%+lld.%01lldms buf=%zu aq=%zu miss=%d",
            static_cast<long long>(rawDelta90k / 90), static_cast<long long>(smoothedDelta90k / 90),
            std::abs(static_cast<long long>((smoothedDelta90k * 10 / 90) % 10)), jitterBuf.size(), ap->GetQueueSize(),
            drainMissCount);
    drainMissCount = 0;
    nextSyncLog.Set(DECODER_SYNC_LOG_INTERVAL_MS);
}

auto cVaapiDecoder::RunJitterPrimeSync(cAudioProcessor *ap) -> void {
    // One-shot pre-drain alignment: bring the front of the jitter buffer to delta ~= 0
    // before we start submitting frames, so the EMA starts from a clean state.
    //   initDelta < 0  video is BEHIND audio: drop stale frames until front >= clock
    //                  (if the deficit exceeds one buffer's worth, the buffer empties,
    //                   underrun re-prime triggers, and the cycle repeats until aligned)
    //   initDelta > 0  video is AHEAD: block until the audio clock catches up, then drop
    //                  any frames the ~24 ms ALSA clock step overshot past
    if (jitterBuf.empty()) {
        return;
    }
    const int64_t fpts0 = jitterBuf.front()->pts;
    const int64_t clock0 = ap->GetClock();
    if (fpts0 == AV_NOPTS_VALUE || clock0 == AV_NOPTS_VALUE) {
        return;
    }
    const int64_t latency = SyncLatency90k();
    const int64_t initDelta = fpts0 - clock0 - latency;

    if (initDelta < 0) {
        int drops = 0;
        while (!jitterBuf.empty() && !stopping.load(std::memory_order_relaxed)) {
            if (jitterBuf.front()->pts == AV_NOPTS_VALUE) {
                break;
            }
            const int64_t clk = ap->GetClock();
            if (clk == AV_NOPTS_VALUE || jitterBuf.front()->pts >= clk + latency) {
                break;
            }
            jitterBuf.pop_front();
            ++drops;
        }
        dsyslog("vaapivideo/decoder: prime-sync dropped %d frames (initDelta=%+lldms)", drops,
                static_cast<long long>(initDelta / 90));
    } else if (initDelta > 0) {
        dsyslog("vaapivideo/decoder: prime-sync ahead d=%+lldms -- waiting for audio",
                static_cast<long long>(initDelta / 90));
        while (!stopping.load(std::memory_order_relaxed)) {
            const int64_t clk = ap->GetClock();
            if (clk == AV_NOPTS_VALUE || clk >= fpts0 - latency) {
                break;
            }
            cCondWait::SleepMs(10);
        }
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

    smoothedDeltaValid = false;
    syncGrace.Set(DECODER_SYNC_GRACE_MS);
    nextSyncLog.Set(0);
}

auto cVaapiDecoder::SkipStaleJitterFrames(cAudioProcessor *ap) -> void {
    // Transient catch-up: drop jitter-buffer frames whose PTS is more than HARD_THRESHOLD
    // (~200 ms) behind the audio clock at CPU speed.  Per-iteration draining via
    // SyncAndSubmitFrame can only drop one such frame per loop turn, which is too slow
    // to recover from a multi-frame stale burst (e.g. after a network gap).
    while (jitterBuf.size() > 1 && !stopping.load(std::memory_order_relaxed)) {
        if (jitterBuf.front()->pts == AV_NOPTS_VALUE) {
            break;
        }
        const int64_t clock = ap->GetClock();
        if (clock == AV_NOPTS_VALUE) {
            break;
        }
        const int64_t frontDelta = jitterBuf.front()->pts - clock - SyncLatency90k();
        if (frontDelta >= -DECODER_SYNC_HARD_THRESHOLD) {
            break;
        }
        jitterBuf.pop_front();
        ++correctDrops;
    }
}

[[nodiscard]] auto cVaapiDecoder::SyncAndSubmitFrame(std::unique_ptr<VaapiFrame> frame) -> bool {
    // -----------------------------------------------------------------------
    // A/V sync -- audio clock is master.  delta = vPTS - audioClock - latency
    //
    // Audio plays at the native rate; no resampler compensation.  Decisions use
    // the EMA-smoothed delta, never the raw value: GetClock() is piecewise-
    // constant between ALSA writes (~24-32 ms), so the raw delta carries +/-50 ms
    // aliasing noise that is not real drift.
    //
    // Convergence strategy: drive the smoothed delta to a small dead band around
    // zero by *scaled* corrections (sleep the excess) rather than threshold-edge
    // corrections (sleep one frame).  A threshold-edge controller parks at the
    // threshold; a scaled controller parks at the band.
    //
    //   Trick mode / freerun / no-clock / no-PTS: submit immediately (unchanged).
    //   Grace after correction:                    submit (let the EMA settle).
    //
    //   Hard transient safety nets (raw delta, large excursion):
    //     delta < -HARD              drop  (channel switch / network gap)
    //     delta > +HARD (replay)     block until audio catches up
    //
    //   Steady state (smoothed EMA delta):
    //     smoothed > +TARGET_BAND    sleep (smoothed - 0) capped at MAX_WAIT,
    //                                then submit.  The wait pauses video for
    //                                the excess so the audio clock catches up;
    //                                source frames pile up in the jitter buffer
    //                                naturally (~4 frames for an 80 ms shift).
    //
    //   No correction is applied on the behind side.  The audio clock derives
    //   from the source PTS (endPts - ALSA delay), so it cannot run permanently
    //   faster than the DVB stream -- any sustained negative delta is a real
    //   transient that the HARD safety net handles.  Soft drops on the behind
    //   side would only fire on EMA noise and would inject visible artifacts.
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

    // --- Audio pipeline not yet running (no PCM written yet) ---
    const int64_t latency = SyncLatency90k();
    const int64_t clock = ap->GetClock();
    if (clock == AV_NOPTS_VALUE) {
        if (syncLogPending.exchange(false, std::memory_order_relaxed) || nextSyncLog.TimedOut()) {
            dsyslog("vaapivideo/decoder: sync freerun (no clock) buf=%zu", jitterBuf.size());
            nextSyncLog.Set(DECODER_SYNC_LOG_INTERVAL_MS);
        }
        return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
    }

    // --- Measure and smooth ---
    const int64_t rawDelta = pts - clock - latency;
    UpdateSmoothedDelta(rawDelta);
    LogSyncStats(rawDelta, ap);

    // Grace: suppress all corrections while the EMA reseeds after a prior correction.
    if (!syncGrace.TimedOut()) {
        return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
    }

    // --- Hard transient drop: frame far behind audio (channel switch / gap) ---
    if (rawDelta < -DECODER_SYNC_HARD_THRESHOLD) [[unlikely]] {
        ++correctDrops;
        smoothedDeltaValid = false;
        syncGrace.Set(DECODER_SYNC_GRACE_MS);
        return true;
    }
    if (correctDrops > 0) {
        dsyslog("vaapivideo/decoder: sync corrected %d dropped d=%+lldms", correctDrops,
                static_cast<long long>(rawDelta / 90));
        correctDrops = 0;
    }

    // --- Hard transient hold (replay only): video far ahead, e.g. post-seek ---
    if (!liveMode.load(std::memory_order_relaxed) && rawDelta > DECODER_SYNC_HARD_THRESHOLD) [[unlikely]] {
        WaitForAudioCatchUp(ap, pts, latency, rawDelta);
        return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
    }

    // --- Soft wait: video sustainably ahead -- sleep the excess to converge to 0 ---
    // The wait length is the smoothed delta itself (cap at MAX_WAIT_MS), driving
    // the smoothed value to ~0 in a single shot.  Reseed the EMA so the next
    // sample re-anchors instantly (the 5 s EMA otherwise has too much inertia
    // to recognize the post-wait state) and arm grace as a belt-and-braces.
    if (smoothedDelta90k > DECODER_SYNC_TARGET_BAND) {
        const auto excessMs = static_cast<int>(smoothedDelta90k / 90);
        const int waitMs = std::min(excessMs, DECODER_SYNC_MAX_WAIT_MS);
        cCondWait::SleepMs(waitMs);
        smoothedDeltaValid = false;
        syncGrace.Set(DECODER_SYNC_GRACE_MS);
    }

    return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
}
