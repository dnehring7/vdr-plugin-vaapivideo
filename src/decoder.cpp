// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file decoder.cpp
 * @brief Threaded VAAPI decoder with filter graph and A/V sync
 *
 * Pipeline:  EnqueueData() -> packet queue -> VAAPI decode -> filter graph -> A/V sync -> display
 * Filters:   [bwdif|deinterlace] -> [hqdn3d|denoise] -> scale (BT.709 NV12) -> [sharpen] (probed per GPU)
 * Sync:      Audio-clock master, threshold submit + first-frame alignment
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

constexpr size_t DECODER_QUEUE_CAPACITY = 50;               ///< Video packet queue depth (~2 s at 25 fps)
constexpr int DECODER_SUBMIT_TIMEOUT_MS = 100;              ///< Timeout when submitting a frame to the display (ms)
constexpr int64_t DECODER_SYNC_CATCHUP_THRESHOLD = 30 * 90; ///< Drop sync frames when video still >30 ms behind
constexpr int64_t DECODER_SYNC_DRIFT_THRESHOLD = 1500 * 90; ///< Re-sync when drift exceeds this (PTS ticks, ~1500 ms)
constexpr int DECODER_SYNC_FREERUN_FRAMES = 10; ///< Frames to submit without sync after Clear() (~200 ms at 50 Hz)

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
    // ownsFrame=true: this VaapiFrame is sole owner of the AVFrame and must free it. ownsFrame=false: the frame is
    // borrowed from another owner (e.g. the filter output buffer still held by FFmpeg) and must not be freed here. Move
    // ops transfer ownership by setting the source ownsFrame=false and vaSurfaceId to VA_INVALID_SURFACE so double-free
    // and stale surface access are both impossible.

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

    // Destroy the filter graph: it may hold buffered decoded frames with stale PTS values. InitFilterGraph() rebuilds
    // it lazily on the next decoded frame. bufferSrcCtx and bufferSinkCtx are owned by filterGraph and become dangling
    // after reset().
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

    freerunFramesLeft.store(DECODER_SYNC_FREERUN_FRAMES, std::memory_order_relaxed);
    syncCorrectionDone.store(false, std::memory_order_release);
    logSyncNow.store(true, std::memory_order_relaxed);
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
            if (trickSpeed.load(std::memory_order_acquire) != 0 && (isTrickFastForward || isTrickReverse) &&
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
    // IsQueueEmpty() / IsQueueFull() are called by VDR's Poll() mechanism to apply backpressure on the PES feed. Poll()
    // must return false (stall) when the queue is full and true (ready) when there is room for at least one packet.
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
    const uint64_t dueTime = nextTrickFrameDue.load(std::memory_order_acquire);
    return cTimeMs::Now() >= dueTime;
}

auto cVaapiDecoder::RequestCodecReopen() -> void {
    const cMutexLock lock(&codecMutex);
    forceCodecReopen = true;
}

[[nodiscard]] auto cVaapiDecoder::OpenCodec(AVCodecID codecId) -> bool {
    if (!ready.load(std::memory_order_acquire)) [[unlikely]] {
        esyslog("vaapivideo/decoder: not initialized");
        return false;
    }

    if (codecCtx && currentCodecId == codecId && !forceCodecReopen) {
        return true;
    }
    forceCodecReopen = false;

    // Codec is changing (or being opened for the first time): tear down all codec-dependent state. The filter graph and
    // parser are both tightly coupled to the codec context -- the filter graph holds VAAPI hw_frames_ctx references
    // that become invalid when the codec context is destroyed.
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

    // Determine if this GPU supports hardware decode for this codec. The VA profile flags were probed once at device
    // init by ProbeVppCapabilities().
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
    dsyslog("vaapivideo/decoder: audio codec changed, resetting A/V sync");
    syncCorrectionDone.store(false, std::memory_order_release);
    logSyncNow.store(true, std::memory_order_relaxed);
}

auto cVaapiDecoder::SetAudioProcessor(cAudioProcessor *audio) -> void { audioProcessor = audio; }

auto cVaapiDecoder::SetTrickSpeed(int speed, bool forward, bool fast) -> void {
    // VDR trick-speed convention used throughout this function:
    //   speed == 0            : normal play (exit trick mode)
    //   fast == true, speed>0 : fast-forward (forward=true) or reverse (forward=false)
    //                           speed encodes the multiplier: 6->2x, 3->4x, 1->8x
    //   fast == false, speed>0: slow-forward at 1/speed fractional rate
    // isTrickFastForward and isTrickReverse are set atomically-safe below.

    // Fast trick entry requires a clean slate: VDR skips DeviceClear() for FF so we must explicitly flush packets,
    // codec, parser, and filter ourselves. Lock ordering: codecMutex -> packetMutex.
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
        // Flush may reallocate hw_frames_ctx inside the codec, invalidating the existing filter graph. Rebuild it on
        // the next decoded frame to avoid green output caused by mismatched VAAPI surface pool references.
        ResetFilterGraph();
        if (decodedFrame) {
            av_frame_unref(decodedFrame.get());
        }
        if (filteredFrame) {
            av_frame_unref(filteredFrame.get());
        }
    }

    // Non-atomic state must be written before the trickSpeed release-store so the decode thread reads a consistent set
    // of flags when it sees the new speed.
    isTrickFastForward = forward && fast;
    isTrickReverse = !forward;
    prevTrickPts = AV_NOPTS_VALUE;

    // Map VDR speed values to pacing multipliers: 6 -> 2x, 3 -> 4x, 1 -> 8x. Slow mode ignores the multiplier and uses
    // a fixed per-frame hold.
    if (fast && speed > 0) {
        if (speed >= 6) {
            trickMultiplier = 2;
        } else if (speed >= 3) {
            trickMultiplier = 4;
        } else {
            trickMultiplier = 8;
        }
        trickHoldMs = DECODER_TRICK_HOLD_MS;
    } else {
        trickMultiplier = 0;
        trickHoldMs = static_cast<uint64_t>(speed) * DECODER_TRICK_HOLD_MS;
    }

    // Arm the pacing timer so the first trick frame is displayed immediately.
    nextTrickFrameDue.store(cTimeMs::Now(), std::memory_order_release);

    if (speed == 0) {
        lastPts.store(AV_NOPTS_VALUE, std::memory_order_relaxed);
        syncCorrectionDone.store(false, std::memory_order_release);
        logSyncNow.store(true, std::memory_order_relaxed);
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

    // Broadcast to unblock the decode thread so it can observe the stopping flag.
    {
        const cMutexLock lock(&packetMutex);
        packetCondition.Broadcast();
    }

    // Running() is not thread-safe in VDR -- use the atomic hasExited flag instead.
    if (!hasExited.load(std::memory_order_acquire)) {
        Cancel(3);
    }

    // Wait for the thread to set hasExited before we return (proper happens-before).
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

        // Wait up to 10 ms so we also wake on a stop request if Broadcast() was missed between the queue-empty check
        // and this wait.
        {
            const cMutexLock lock(&packetMutex);
            while (packetQueue.empty() && !stopping.load(std::memory_order_acquire)) {
                packetCondition.TimedWait(packetMutex, 10);
            }

            if (stopping.load(std::memory_order_acquire)) {
                break;
            }

            if (!packetQueue.empty()) {
                queuedPacket.reset(packetQueue.front());
                packetQueue.pop();
            }
        }

        std::vector<std::unique_ptr<VaapiFrame>> pendingFrames;

        if (queuedPacket) {
            av_packet_unref(workPacket);
            // Copy packet data into the pre-allocated workPacket so we can free queuedPacket immediately. This is not
            // redundant: it lets us release packetMutex before the (potentially long) VAAPI decode call so that
            // EnqueueData() is not blocked for the entire decode duration.
            if (av_packet_ref(workPacket, queuedPacket.get()) == 0) {
                queuedPacket.reset(); // free early; workPacket now owns the data

                // Hold codecMutex only while decoding; release it before SyncAndSubmitFrame() so EnqueueData() is not
                // blocked during display wait.
                if (!stopping.load(std::memory_order_acquire) && codecCtx) {
                    const cMutexLock decodeLock(&codecMutex);
                    if (codecCtx) { // recheck: OpenCodec() may reset it between the two loads
                        (void)DecodeOnePacket(workPacket, pendingFrames);
                    }
                }
            }
            // else: queuedPacket auto-frees at end of scope.
        }

        // Submit decoded frames to the display; no mutex held so EnqueueData() can run in parallel.
        for (auto &frame : pendingFrames) {
            if (stopping.load(std::memory_order_relaxed)) [[unlikely]] {
                break;
            }
            (void)SyncAndSubmitFrame(std::move(frame));
        }
    }

    av_packet_free(&workPacket);

    // Store true before the final log so Shutdown()'s spin-wait sees it immediately.
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

    // av_frame_clone() increments the refcount on the VAAPI surface buffer; the surface stays alive as long as this
    // VaapiFrame is live.
    vaapiFrame->avFrame = av_frame_clone(src);
    if (!vaapiFrame->avFrame) [[unlikely]] {
        return nullptr;
    }

    // data[3] holds the VASurfaceID cast to a pointer -- the FFmpeg VAAPI convention.
    vaapiFrame->vaSurfaceId =
        static_cast<VASurfaceID>(reinterpret_cast<uintptr_t>( // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
            vaapiFrame->avFrame->data[3]));
    vaapiFrame->pts = src->pts;
    vaapiFrame->ownsFrame = true;

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

    // VAAPI send-drain loop: avcodec_send_packet() may return EAGAIN when the hardware output queue is full. In that
    // case we drain all available frames first and then retry the send. In normal operation this loop runs once; on
    // EAGAIN it runs twice.
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

        // Drain all available decoded frames and push them through the filter graph. We do not flush between trick-mode
        // frames because VAAPI has a 1-2 frame internal latency and flushing destroys the pipeline state; IDRs drain it
        // naturally.
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

            // DVB MPEG-2 streams often omit the color_description syntax element; force BT.470BG so downstream
            // color-space conversion is correct for SD.
            if (decodedFrame->colorspace == AVCOL_SPC_UNSPECIFIED && decodedFrame->height <= 576) {
                decodedFrame->colorspace = AVCOL_SPC_BT470BG;
            }

            // Build the filter graph lazily on the first decoded frame (or after Clear()).
            if (!filterGraph) {
                (void)InitFilterGraph(decodedFrame.get());
            }

            // Stash the source PTS before handing the frame to the filter. The deinterlace filter interpolates PTS for
            // the second output field which is not suitable for our sync model here, so both outputs keep the source
            // PTS. Trick mode relies on that too for field-pair detection.
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
                        if (isTrickMode && (isTrickFastForward || isTrickReverse) && !outFrames.empty() &&
                            outFrames.back()->pts == sourcePts) {
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
            // Guard against an infinite loop: if send returns EAGAIN but the decoder produces no frames to drain, we'd
            // spin forever. Drop the packet instead.
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

    // Compute DAR = (srcWidth * sarNum) : (srcHeight * sarDen), then fit into the display bounds to get the final
    // output size (letterbox or pillarbox). DRM presents at 1:1 pixel mapping (no HW scaler). Normalize SAR: treat
    // 0/0 and 0/N as 1:1 square pixels. Both DAR and the buffer source args must use the same SAR values.
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

    // Allocate buffer source without init -- hw_frames_ctx must be attached first (FFmpeg 7.x validates it in
    // init_video()).
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

    // Wire the filter chain between the buffer source ("in") and buffersink ("out") endpoints.
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
    // A/V sync -- audio clock master.  delta = vPTS - audioClock - latency
    //   positive delta = video ahead (early)    -> block until audio catches up
    //   negative delta = video behind (late)    -> drop frame
    //
    // Phases:
    //   1. Trick mode    -- pacing timer, no sync
    //   2. Freerun       -- no clock or first N frames after Clear()
    //   3. Steady state  -- submit if |delta| < 1500 ms, else trigger re-sync
    //   4. Sync align    -- one-shot (initial) or drift re-sync (fallthrough from phase 3)
    //                       wait for clock, wait for delta <= 0, catchup-drop if still behind

    if (!frame || !display) [[unlikely]] {
        return false;
    }

    const int64_t originalPts = frame->pts;

    // --- Trick mode: pacing timer, no A/V sync ---
    if (trickSpeed.load(std::memory_order_acquire) != 0) {
        const bool newSource = (originalPts != prevTrickPts);
        const int64_t savedPrevPts = prevTrickPts;

        // Reverse playback: PTS must decrease monotonically.
        if (isTrickReverse && newSource && originalPts != AV_NOPTS_VALUE && savedPrevPts != AV_NOPTS_VALUE &&
            originalPts > savedPrevPts) {
            return true;
        }
        if (newSource) {
            prevTrickPts = originalPts;
        }
        if (newSource && originalPts != AV_NOPTS_VALUE) {
            lastPts.store(originalPts, std::memory_order_release);
        }

        // Fast-forward/reverse: block until the pacing timer expires.
        if (newSource && trickMultiplier > 0) {
            const uint64_t due = nextTrickFrameDue.load(std::memory_order_acquire);
            while (cTimeMs::Now() < due && !stopping.load(std::memory_order_relaxed) &&
                   trickSpeed.load(std::memory_order_relaxed) != 0) {
                cCondWait::SleepMs(10);
            }
        }

        // Arm the pacing timer for the next source frame.
        if (newSource) {
            if (trickMultiplier > 0 && originalPts != AV_NOPTS_VALUE && savedPrevPts != AV_NOPTS_VALUE) {
                const auto ptsDelta = static_cast<uint64_t>(std::abs(originalPts - savedPrevPts));
                const uint64_t holdMs =
                    std::clamp(ptsDelta / (static_cast<uint64_t>(90) * trickMultiplier), uint64_t{10}, uint64_t{2000});
                nextTrickFrameDue.store(cTimeMs::Now() + holdMs, std::memory_order_release);
            } else {
                nextTrickFrameDue.store(cTimeMs::Now() + trickHoldMs, std::memory_order_release);
            }
        }
        return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
    }

    if (originalPts != AV_NOPTS_VALUE) {
        lastPts.store(originalPts, std::memory_order_release);
    }

    // --- Freerun: no audio clock or no PTS available ---
    if (!audioProcessor || originalPts == AV_NOPTS_VALUE) {
        return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
    }

    // --- Freerun: first N frames after Clear() -- show picture immediately ---
    if (freerunFramesLeft.load(std::memory_order_acquire) > 0) {
        freerunFramesLeft.fetch_sub(1, std::memory_order_relaxed);
        return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
    }

    const int64_t latency = static_cast<int64_t>(vaapiConfig.audioLatency) * 90;
    int64_t clock = audioProcessor->GetClock();
    const bool synced = syncCorrectionDone.load(std::memory_order_acquire);

    // --- Steady state: submit in-range frames, trigger re-sync on drift ---
    if (synced) {
        if (clock == AV_NOPTS_VALUE) {
            return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
        }

        const int64_t delta = originalPts - clock - latency;

        if (logSyncNow.exchange(false, std::memory_order_relaxed) || nextSyncLog.TimedOut()) {
            dsyslog("vaapivideo/decoder: sync status d=%lldms vPTS=%lld aPTS=%lld", static_cast<long long>(delta / 90),
                    static_cast<long long>(originalPts / 90), static_cast<long long>(clock / 90));
            nextSyncLog.Set(30000);
        }

        // Video too far behind: drop frame, trigger re-sync on the next one.
        if (delta < -DECODER_SYNC_DRIFT_THRESHOLD) {
            dsyslog("vaapivideo/decoder: sync drop d=%+lldms, triggering re-sync", static_cast<long long>(delta / 90));
            syncCorrectionDone.store(false, std::memory_order_release);
            logSyncNow.store(true, std::memory_order_relaxed);
            return true;
        }

        // Within tolerance: submit normally.
        if (delta <= DECODER_SYNC_DRIFT_THRESHOLD) {
            return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
        }

        // Video too far ahead (e.g. DVB PTS discontinuity): fall through to sync alignment.
        dsyslog("vaapivideo/decoder: drift re-sync d=%+lldms vPTS=%lld aPTS=%lld", static_cast<long long>(delta / 90),
                static_cast<long long>(originalPts / 90), static_cast<long long>(clock / 90));
    }

    // --- Sync alignment (one-shot after Clear() or drift re-sync from above) ---

    // Step 1: wait up to 3 s for audio clock to become available.
    if (clock == AV_NOPTS_VALUE) {
        const cTimeMs waitTimeout(3000);
        while (clock == AV_NOPTS_VALUE && !stopping.load(std::memory_order_relaxed) &&
               trickSpeed.load(std::memory_order_relaxed) == 0 && !waitTimeout.TimedOut()) {
            cCondWait::SleepMs(5);
            clock = audioProcessor->GetClock();
        }
    }
    if (clock == AV_NOPTS_VALUE) {
        // No clock after 3 s: mark synced to avoid blocking every subsequent frame.
        syncCorrectionDone.store(true, std::memory_order_release);
        return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
    }

    int64_t delta = originalPts - clock - latency;

    if (!synced) {
        dsyslog("vaapivideo/decoder: one-shot sync d=%+lldms vPTS=%lld aPTS=%lld", static_cast<long long>(delta / 90),
                static_cast<long long>(originalPts / 90), static_cast<long long>(clock / 90));
    }

    // Step 2: if video is ahead, block until audio catches up (or timeout).
    // In replay mode the video packet queue may be full, blocking VDR's Poll() loop and
    // preventing audio delivery. Exit early if the audio clock is not advancing after 100 ms
    // so that returning here lets the decode thread drain one more packet from the queue,
    // freeing Poll() to resume audio.
    if (delta > 0) {
        const int64_t clockAtWaitStart = clock;
        const cTimeMs frozenCheckTimeout(100);
        const cTimeMs waitTimeout(static_cast<int>(std::min((delta / 90) + 10, int64_t{3000})));
        while (delta > 0 && clock != AV_NOPTS_VALUE && !stopping.load(std::memory_order_relaxed) &&
               trickSpeed.load(std::memory_order_relaxed) == 0 && !waitTimeout.TimedOut()) {
            cCondWait::SleepMs(5);
            clock = audioProcessor->GetClock();
            if (clock != AV_NOPTS_VALUE) {
                delta = originalPts - clock - latency;
            }
            // Exit early if clock hasn't advanced by at least 10 ms after 100 ms of waiting.
            // Less than 10 ms of advance in 100 ms means audio is stalled (ALSA not yet started
            // or VDR Poll() blocked, preventing audio packet delivery).
            if (frozenCheckTimeout.TimedOut() && clock != AV_NOPTS_VALUE && clock - clockAtWaitStart < 10 * 90) {
                break;
            }
        }
    }

    // Step 3: audio clock lost during wait (codec reinit). Submit frame without marking synced;
    // NotifyAudioChange() already reset syncCorrectionDone -- the next frame retries.
    if (clock == AV_NOPTS_VALUE) {
        dsyslog("vaapivideo/decoder: %ssync deferred (audio clock lost during wait)", synced ? "drift re-" : "");
        return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
    }

    // Step 4: audio codec changed during a drift re-sync wait (NotifyAudioChange() cleared
    // syncCorrectionDone). Submit this frame, let the next frame do a fresh one-shot.
    if (synced && !syncCorrectionDone.load(std::memory_order_acquire)) {
        dsyslog("vaapivideo/decoder: drift re-sync aborted (audio changed during wait) d=%+lldms",
                static_cast<long long>(delta / 90));
        return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
    }

    // Step 5: video still behind -- drop frame and retry on the next one until delta converges to zero.
    if (delta < -DECODER_SYNC_CATCHUP_THRESHOLD) {
        dsyslog("vaapivideo/decoder: %ssync catchup drop d=%+lldms", synced ? "drift re-" : "",
                static_cast<long long>(delta / 90));
        return true;
    }

    // Step 6: delta within tolerance -- sync acquired.
    // Guard: if delta still exceeds the drift threshold, acquiring here would immediately
    // re-trigger drift re-sync on every subsequent frame (oscillation loop). Submit without
    // marking synced so the video queue can drain and VDR can resume audio delivery.
    if (delta > DECODER_SYNC_DRIFT_THRESHOLD) {
        dsyslog("vaapivideo/decoder: %ssync deferred (audio stalled) d=%+lldms", synced ? "drift re-" : "",
                static_cast<long long>(delta / 90));
        return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
    }
    dsyslog("vaapivideo/decoder: %ssync acquired d=%+lldms", synced ? "drift re-" : "",
            static_cast<long long>(delta / 90));
    syncCorrectionDone.store(true, std::memory_order_release);
    nextSyncLog.Set(30000);
    return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
}
