// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file decoder.cpp
 * @brief Threaded VAAPI decoder with VPP filter graph and audio-mastered A/V sync.
 *
 * Pipeline:  EnqueueData() -> packetQueue -> Action() -> VAAPI decode -> filter graph
 *            -> [live: jitterBuf] -> SyncAndSubmitFrame -> display->SubmitFrame
 * Filters:   [bwdif|deinterlace_vaapi rate=field] -> [hqdn3d|denoise_vaapi]
 *            -> scale_vaapi (NV12 BT.709 TV-range) -> [sharpness_vaapi]
 *            VPP nodes are probed per GPU at device init.
 * Sync:      Audio clock is master. Three regimes:
 *              freerun (no clock / post-event window) -> submit unpaced
 *              hard    (|raw| > 200 ms)               -> replay: block ahead / drop behind
 *                                                       live:   1 big sleep ahead / drop behind
 *              soft    (|EMA| > 20 ms, cooldown OK)   -> round(EMA/frameDur) drops or one
 *                                                       sleep of (correctMs + frameDur);
 *                                                       EMA adjusted by measured effect
 *
 * Threading: codecMutex protects codec/parser state; packetMutex protects packetQueue.
 *            Lock order is ALWAYS codecMutex -> packetMutex (Clear / EnqueueData / SetTrickSpeed
 *            all match this). Jitter buffer is decoder-thread-owned (no lock); cross-thread
 *            requests use atomics + a deferred apply at the top of Action().
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

constexpr int DECODER_JITTER_BUFFER_MS =
    1000; ///< Live jitter buffer fill target; 1 s = 25-50 frames, absorbs burst gaps on weak transponders.
constexpr size_t DECODER_QUEUE_CAPACITY =
    200; ///< Packet queue (~4 s @ 50 fps / ~8 s @ 25 fps); drops oldest when full, sized to match jitter buffer.
constexpr int DECODER_SUBMIT_TIMEOUT_MS = 100; ///< Max VSync backpressure wait inside display->SubmitFrame() (ms).
constexpr int DECODER_SYNC_COOLDOWN_MS = 5000; ///< Min interval between soft corrections (5x EMA time constant).
constexpr int64_t DECODER_SYNC_CORRIDOR =
    20 * 90; ///< Soft corridor half-width (20 ms in 90 kHz ticks); minimum sleep that crosses one VSync period.
constexpr int DECODER_SYNC_EMA_SAMPLES =
    50; ///< EMA window (alpha = 1/N, ~1 s @ 50 fps); residual accumulator prevents integer-truncation stall.
constexpr int DECODER_SYNC_FREERUN_FRAMES =
    1; ///< Unpaced frames after sync-disrupting events (Clear / track switch / trick exit).
constexpr int64_t DECODER_SYNC_HARD_THRESHOLD =
    200 * 90; ///< Raw delta past which we hard-drop (behind) or hard-block (ahead), bypassing the soft corridor.
constexpr int DECODER_SYNC_LOG_INTERVAL_MS = 30000; ///< Periodic sync-stats dsyslog cadence (ms).
constexpr int DECODER_SYNC_MAX_CORRECTION_MS =
    100; ///< Per-event correction cap (ms); bounds worst-case glitch for both sleep and drop-burst paths.
constexpr int DECODER_SYNC_WARMUP_SAMPLES =
    50; ///< Samples averaged to seed the EMA; sqrt(50) suppresses the 150 ms deinterlace oscillation bias.

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
}

// ============================================================================
// === PUBLIC API ===
// ============================================================================

auto cVaapiDecoder::Clear() -> void {
    // Lock ordering: codecMutex -> packetMutex prevents a deadlock with EnqueueData().
    const cMutexLock decodeLock(&codecMutex);

    DrainQueue();

    {
        // vaDriverMutex: flush and graph teardown make VA API calls that
        // would otherwise race with the display thread's PRIME export.
        const cMutexLock vaLock(&display->GetVaDriverMutex());

        // Flush decoder; the codec context stays open so the next I-frame can continue.
        if (codecCtx) {
            avcodec_flush_buffers(codecCtx.get());
        }

        // Filter graph caches hw_frames_ctx and may hold stale-PTS frames; rebuilt lazily.
        ResetFilterGraph();
        if (decodedFrame) {
            av_frame_unref(decodedFrame.get());
        }
        if (filteredFrame) {
            av_frame_unref(filteredFrame.get());
        }
    }
    // hasLoggedFirstFrame NOT reset: only OpenCodec() resets it to avoid duplicate logs.

    // AVCodecParserContext has no flush API; recreate.
    if (currentCodecId != AV_CODEC_ID_NONE) {
        parserCtx.reset(av_parser_init(currentCodecId));
    } else {
        parserCtx.reset();
    }

    lastPts.store(AV_NOPTS_VALUE, std::memory_order_relaxed);
    codecDrainPending.store(false, std::memory_order_relaxed);
    stillPictureMode.store(false, std::memory_order_relaxed);

    // Deferred to the decode thread (which owns jitterBuf without locking). Stale frames
    // would otherwise pin VAAPI surfaces across codec teardown.
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

    // pts applies only to the first call per AU; the parser propagates it internally.
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

        // No progress (typical at MPEG-2 PES boundaries); break to avoid infinite loop.
        if (parsed == 0 && parsedSize == 0) {
            break;
        }

        parseData += parsed;
        parseSize -= parsed;
        currentPts = AV_NOPTS_VALUE;

        if (parsedSize > 0) {
            // Fast-forward only: drop non-keyframes (no refs). Reverse must keep
            // key_frame=0 for PAFF interlaced second fields. Slow keeps all for smoothness.
            if (trickSpeed.load(std::memory_order_acquire) != 0 && isTrickFastForward.load(std::memory_order_relaxed) &&
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
            if (parserCtx->key_frame == 1) {
                pkt->flags |= AV_PKT_FLAG_KEY;
            }

            // Trick: depth-1, drop incoming (Poll throttles). Normal: drop oldest on overflow.
            const cMutexLock lock(&packetMutex);
            const bool isTrickMode = trickSpeed.load(std::memory_order_relaxed) != 0;
            const size_t maxDepth = isTrickMode ? DECODER_TRICK_QUEUE_DEPTH : DECODER_QUEUE_CAPACITY;
            if (packetQueue.size() >= maxDepth) {
                if (isTrickMode) {
                    dsyslog("vaapivideo/decoder: trick enqueue: queue full (depth=%zu) -- dropping incoming",
                            packetQueue.size());
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

    // Do NOT drain here: the parser withholds the final AU until the next start code.
    // FlushParser() drains after all PES chunks are delivered; draining per-call would
    // fragment multi-PES I-frames into partial AUs.
}

auto cVaapiDecoder::FlushParser() -> void {
    // Still-picture / Goto(Still): drain the parser so the single I-frame surfaces.
    const cMutexLock decodeLock(&codecMutex);
    DrainPendingParserAU();
}

auto cVaapiDecoder::DrainPendingParserAU() -> void {
    // Bitstream parsers withhold an AU until the next start code. NULL/0 input is the
    // documented EOS-flush. Callers: still-picture and reverse trick (isolated I-frames).
    // Caller holds codecMutex; packetMutex taken inside.

    if (!codecCtx || !parserCtx) {
        return;
    }

    uint8_t *parsedData = nullptr; // NOLINT(misc-const-correctness)
    int parsedSize = 0;
    const int parsed = av_parser_parse2(parserCtx.get(), codecCtx.get(), &parsedData, &parsedSize, nullptr, 0,
                                        AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);

    if (parsed < 0 || parsedSize <= 0 || !parsedData) {
        return;
    }

    // Fast-forward keyframe filter (mirrors EnqueueData). Reverse must not filter: PAFF needs key_frame=0.
    if (trickSpeed.load(std::memory_order_relaxed) != 0 && isTrickFastForward.load(std::memory_order_relaxed) &&
        parserCtx->key_frame == 0) {
        return;
    }

    AVPacket *pkt = av_packet_alloc();
    if (!pkt) [[unlikely]] {
        return;
    }
    if (av_new_packet(pkt, parsedSize) < 0) [[unlikely]] {
        av_packet_free(&pkt);
        return;
    }
    std::memcpy(pkt->data, parsedData, static_cast<size_t>(parsedSize));
    pkt->pts = parserCtx->pts;
    pkt->dts = parserCtx->dts;
    if (parserCtx->key_frame == 1) {
        pkt->flags |= AV_PKT_FLAG_KEY;
    }

    {
        const cMutexLock lock(&packetMutex);
        // Drop-oldest: in trick mode the freshly drained AU must not be rejected.
        const bool isTrickMode = trickSpeed.load(std::memory_order_relaxed) != 0;
        const size_t maxDepth = isTrickMode ? DECODER_TRICK_QUEUE_DEPTH : DECODER_QUEUE_CAPACITY;
        if (packetQueue.size() >= maxDepth) {
            av_packet_free(&packetQueue.front());
            packetQueue.pop();
        }
        packetQueue.push(pkt);
        packetCondition.Broadcast();
    }
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
    // VDR's Poll() asks this -- "true" means "send the next PES packet". Returning false
    // throttles delivery and is how we apply backpressure on the dvbplayer thread.
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

auto cVaapiDecoder::RequestCodecDrain() -> void { codecDrainPending.store(true, std::memory_order_release); }

auto cVaapiDecoder::SetStillPictureMode(bool mode) -> void { stillPictureMode.store(mode, std::memory_order_release); }

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

    // Full teardown: the filter graph holds a ref to the old codec's hw_frames_ctx and is
    // not safe to reuse. hasLoggedFirstFrame is reset here (and only here) so the per-codec
    // "first frame" line still fires once per real reopen.
    parserCtx.reset();
    {
        // vaDriverMutex: codec context destruction (vaDestroyContext for VAAPI hwaccel)
        // and graph teardown (VPP pipeline teardown) make VA API calls.
        const cMutexLock vaLock(&display->GetVaDriverMutex());
        codecCtx.reset();
        ResetFilterGraph();
    }
    currentCodecId = AV_CODEC_ID_NONE;
    hasLoggedFirstFrame.store(false, std::memory_order_relaxed);

    const AVCodec *decoder = avcodec_find_decoder(codecId);
    if (!decoder) [[unlikely]] {
        esyslog("vaapivideo/decoder: codec %d not found", static_cast<int>(codecId));
        return false;
    }

    // hwMpeg2 / hwH264 / hwHevc are populated once at device init by ProbeVppCapabilities().
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

    // The VA driver advertising a profile is necessary but not sufficient -- FFmpeg also
    // needs a VAAPI hw_config entry. Walk the codec's configs and require both.
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

    // DVB MPEG-2/HEVC streams are routinely marginally non-conforming; loosen the strict
    // gates so the decoder accepts them instead of bailing on the whole packet.
    decoderCtx->err_recognition = AV_EF_CAREFUL;
    decoderCtx->strict_std_compliance = FF_COMPLIANCE_UNOFFICIAL;

    if (useHwDecode) {
        // VAAPI decode is single-threaded by design -- the GPU parallelizes internally and
        // FFmpeg's slice threading would just contend on the VA driver. SW decode (else
        // branch) uses FFmpeg's default thread pool.
        decoderCtx->thread_count = 1;

        decoderCtx->hw_device_ctx = av_buffer_ref(vaapiContext->hwDeviceRef);
        if (!decoderCtx->hw_device_ctx) [[unlikely]] {
            esyslog("vaapivideo/decoder: failed to reference VAAPI device for %s", decoder->name);
            return false;
        }

        // get_format hook: pin output to VAAPI surfaces. FFmpeg falls back to SW frames
        // here if we return NONE -- the explicit pick keeps decoding on the GPU.
        decoderCtx->get_format = [](AVCodecContext *, const AVPixelFormat *formats) -> AVPixelFormat {
            for (const AVPixelFormat *fmt = formats; *fmt != AV_PIX_FMT_NONE; ++fmt) {
                if (*fmt == AV_PIX_FMT_VAAPI) {
                    return AV_PIX_FMT_VAAPI;
                }
            }
            return AV_PIX_FMT_NONE;
        };
    }
    // SW decode: leave hw_device_ctx unset; InitFilterGraph() inserts hwupload + attaches
    // hwDeviceRef on every filter node so VPP scaling still runs on the GPU.

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
    // Audio codec / track switch: the audio clock will be NOPTS for ~100-500 ms while the
    // new pipeline primes. Arm the freerun window so the first frame surfaces immediately
    // and Action() will issue a one-shot prime-sync against the *new* clock once it arrives
    // (RunJitterPrimeSync defers when the clock is unavailable).
    freerunFrames.store(DECODER_SYNC_FREERUN_FRAMES, std::memory_order_relaxed);
    syncLogPending.store(true, std::memory_order_relaxed);
}

auto cVaapiDecoder::SetAudioProcessor(cAudioProcessor *audio) -> void {
    audioProcessor.store(audio, std::memory_order_release);
}

auto cVaapiDecoder::SetLiveMode(bool live) -> void { liveMode.store(live, std::memory_order_relaxed); }

auto cVaapiDecoder::RequestTrickExit() -> void { trickExitPending.store(true, std::memory_order_release); }

auto cVaapiDecoder::SetTrickSpeed(int speed, bool forward, bool fast) -> void {
    // speed==0: normal. fast: FF/REW (>=6->2x, >=3->4x, else 8x). !fast: slow at 1/speed.
    trickExitPending.store(false, std::memory_order_relaxed); // supersedes pending trick-exit

    // VDR skips DeviceClear() on trick entry. FF/REW/slow-reverse are non-contiguous;
    // without flush, stale NALs cause the parser to swallow I-frames before resyncing.
    // Only slow-forward (frame-by-frame) keeps warm state. Lock: codecMutex -> packetMutex.
    const bool needsFlush = (speed > 0) && (fast || !forward);
    dsyslog("vaapivideo/decoder: SetTrickSpeed speed=%d forward=%d fast=%d needsFlush=%d", speed, forward, fast,
            needsFlush);
    {
        // codecMutex pairs flag updates atomically with EnqueueData's parse path.
        const cMutexLock decodeLock(&codecMutex);

        if (needsFlush) {
            DrainQueue();
            {
                // vaDriverMutex: avcodec_flush_buffers (VAAPI hwaccel) and
                // ResetFilterGraph (avfilter_graph_free -> VPP teardown) both
                // make VA API calls. Without the mutex the display thread's
                // av_hwframe_map (PRIME export) races on the same VADisplay,
                // crashing iHD. Lock order: codecMutex -> vaDriverMutex.
                const cMutexLock vaLock(&display->GetVaDriverMutex());
                if (codecCtx) {
                    avcodec_flush_buffers(codecCtx.get());
                }
                // flush_buffers may rebuild hw_frames_ctx; graph holds the old ref.
                ResetFilterGraph();
                if (decodedFrame) {
                    av_frame_unref(decodedFrame.get());
                }
                if (filteredFrame) {
                    av_frame_unref(filteredFrame.get());
                }
            }
            // No parser flush API; recreate to discard stale partial NALs.
            if (currentCodecId != AV_CODEC_ID_NONE) {
                parserCtx.reset(av_parser_init(currentCodecId));
            }
        }

        // Flags before trickSpeed release-store: reader sees consistent (mode, speed).
        isTrickFastForward.store(forward && fast, std::memory_order_relaxed);
        isTrickReverse.store(!forward, std::memory_order_relaxed);
        prevTrickPts.store(AV_NOPTS_VALUE, std::memory_order_relaxed);
    }

    // Fast: PTS-derived hold via multiplier. Slow: fixed hold = speed * TRICK_HOLD_MS.
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

    nextTrickFrameDue.store(cTimeMs::Now(), std::memory_order_relaxed); // no initial hold

    if (speed == 0) {
        lastPts.store(AV_NOPTS_VALUE, std::memory_order_relaxed);

        syncLogPending.store(true, std::memory_order_relaxed);
    }

    trickSpeed.store(speed, std::memory_order_release);
}

auto cVaapiDecoder::Shutdown() -> void {
    // Idempotent. Postcondition: decode thread joined, packetQueue drained. Callers may
    // then safely tear down anything the decoder was sharing with the display thread
    // (codec context, hw_frames_ctx, VAAPI surfaces).
    const bool wasStopping = stopping.exchange(true, std::memory_order_acq_rel);
    dsyslog("vaapivideo/decoder: shutting down (wasStopping=%d)", wasStopping);
    if (wasStopping) {
        return;
    }

    {
        const cMutexLock lock(&packetMutex);
        packetCondition.Broadcast();
    }

    // VDR's cThread::Running() is unfenced (CLAUDE.md "Thread-Safety Pitfall"); read our
    // own release/acquire hasExited instead.
    if (!hasExited.load(std::memory_order_acquire)) {
        Cancel(3);
    }

    // Wait for the decode thread to publish hasExited so we have a real happens-before edge
    // with the work it just finished -- Cancel() alone gives us a join, not a fence.
    const cTimeMs timeout(SHUTDOWN_TIMEOUT_MS);
    while (!hasExited.load(std::memory_order_acquire) && !timeout.TimedOut()) {
        {
            const cMutexLock lock(&packetMutex);
            packetCondition.Broadcast();
        }
        cCondWait::SleepMs(10);
    }

    // Drain after the thread exits: any packet enqueued between hasExited and now would
    // leak otherwise (std::queue<AVPacket*> does not own its contents).
    DrainQueue();
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

        // 18 ms while draining a primed buffer so VSync backpressure paces the loop, not
        // this timer. 10 ms otherwise. 18 ms (not 20 ms) avoids a beat against 50 Hz VSync
        // that dropped 50p output to ~33 fps; re-verify on a 50p source before changing.
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
            // Ref-copy, then drop queuedPacket before the GPU submission to avoid holding
            // packetMutex across the slow avcodec_send_packet() call.
            if (av_packet_ref(workPacket.get(), queuedPacket.get()) == 0) {
                queuedPacket.reset();

                // codecMutex is released before frame delivery so Clear() / SetTrickSpeed()
                // on other threads don't stall behind a VSync-blocked SubmitFrame.
                if (!stopping.load(std::memory_order_acquire)) {
                    const cMutexLock decodeLock(&codecMutex);
                    if (codecCtx) {
                        (void)DecodeOnePacket(workPacket.get(), pendingFrames);
                    }
                }
            }
        }

        // --- Still-picture codec drain ---
        // Race: the decode thread may consume the packet before RequestCodecDrain sets
        // the flag. Running the drain here catches both orderings.
        if (codecDrainPending.exchange(false, std::memory_order_acquire)) {
            const cMutexLock decodeLock(&codecMutex);
            DrainCodecAtEos(pendingFrames);

            // EOS-flush the filter graph: temporal filters hold frames internally.
            if (filterGraph && bufferSrcCtx) {
                const cMutexLock vaLock(&display->GetVaDriverMutex());
                if (av_buffersrc_add_frame_flags(bufferSrcCtx, nullptr, 0) >= 0) {
                    while (true) {
                        av_frame_unref(filteredFrame.get());
                        const int filterRet = av_buffersink_get_frame(bufferSinkCtx, filteredFrame.get());
                        if (filterRet < 0) {
                            break;
                        }
                        if (auto vaapiFrame = CreateVaapiFrame(filteredFrame.get())) {
                            pendingFrames.push_back(std::move(vaapiFrame));
                        }
                    }
                }
                ResetFilterGraph(); // graph is now in EOF state
            }

            // Leave still mode: the next real packet rebuilds the graph with full filters.
            stillPictureMode.store(false, std::memory_order_release);
        }

        // --- Frame delivery ---
        // Deferred jitter flush from Clear(); applied here (decoder thread owns jitterBuf).
        if (jitterFlushPending.exchange(false, std::memory_order_acquire)) {
            jitterBuf.clear();
            jitterPrimed = false;
            pendingDrops = 0;
        }

        if (!liveMode.load(std::memory_order_relaxed)) {
            // Replay: no jitter buffer needed.
            for (auto &frame : pendingFrames) {
                if (stopping.load(std::memory_order_relaxed)) [[unlikely]] {
                    break;
                }
                (void)SyncAndSubmitFrame(std::move(frame));
            }
        } else {
            // Live TV: jitter buffer absorbs DVB-over-IP arrival jitter.
            auto frameIt = pendingFrames.begin();

            // Freerun: Clear() flushes jitterBuf; NotifyAudioChange() keeps it but
            // re-arms primeSyncPending for realignment once the new audio clock arrives.
            if (freerunFrames.load(std::memory_order_relaxed) > 0) {
                if (jitterPrimed) {
                    primeSyncPending = true;
                }
                if (frameIt != pendingFrames.end() && !stopping.load(std::memory_order_relaxed)) {
                    (void)SyncAndSubmitFrame(std::move(*frameIt));
                    ++frameIt;
                }
            }

            // No target yet (InitFilterGraph hasn't run): pass through unpaced.
            if (jitterTarget == 0) {
                for (; frameIt != pendingFrames.end(); ++frameIt) {
                    if (stopping.load(std::memory_order_relaxed)) [[unlikely]] {
                        break;
                    }
                    (void)SyncAndSubmitFrame(std::move(*frameIt));
                }
            } else {
                // Drain exactly pushedCount (1 progressive / 2 deinterlaced) to match
                // input rate and hold the buffer near jitterTarget.
                const int pushedCount = static_cast<int>(pendingFrames.end() - frameIt);
                for (; frameIt != pendingFrames.end(); ++frameIt) {
                    jitterBuf.push_back(std::move(*frameIt));
                }

                // Prime once jitterTarget frames queued; reset sync to avoid EMA noise.
                if (!jitterPrimed && static_cast<int>(jitterBuf.size()) >= jitterTarget) {
                    jitterPrimed = true;
                    primeSyncPending = true; // one-shot coarse alignment before first drain
                    ResetSmoothedDelta();
                    pendingDrops = 0;
                    syncCooldown.Set(DECODER_SYNC_COOLDOWN_MS);
                    nextSyncLog.Set(DECODER_SYNC_LOG_INTERVAL_MS);
                    lastDrainMs = 0;
                    drainMissCount = 0;
                    dsyslog("vaapivideo/decoder: jitter buffer primed (buf=%zu target=%d)", jitterBuf.size(),
                            jitterTarget);
                }

                // Underrun: re-prime to rebuild cushion before resuming drain.
                if (jitterPrimed && jitterBuf.empty()) {
                    dsyslog("vaapivideo/decoder: jitter buffer underrun -- re-priming (target=%d)", jitterTarget);
                    jitterPrimed = false;
                    pendingDrops = 0;
                }

                // Drain pushedCount fresh frames; pendingDrops (soft-corridor burst)
                // are extra and don't count against toDrainFresh. VSync paces the loop.
                if (jitterPrimed && !jitterBuf.empty() && !stopping.load(std::memory_order_relaxed)) {
                    auto *const ap = audioProcessor.load(std::memory_order_acquire);
                    if (ap) {
                        if (primeSyncPending) {
                            // Defers until clock valid to avoid post-track-switch EMA skew.
                            if (RunJitterPrimeSync(ap)) {
                                primeSyncPending = false;
                            }
                        }
                        SkipStaleJitterFrames(ap);
                    }
                    int toDrainFresh = pushedCount;
                    while (!jitterBuf.empty() && !stopping.load(std::memory_order_relaxed)) {
                        if (pendingDrops == 0 && toDrainFresh <= 0) {
                            break;
                        }
                        // >2x frameDur gap = starvation (stall, display blocked, audio underrun).
                        const uint64_t nowMs = cTimeMs::Now();
                        if (lastDrainMs > 0 && static_cast<int>(nowMs - lastDrainMs) > outputFrameDurationMs * 2) {
                            ++drainMissCount;
                        }
                        lastDrainMs = nowMs;

                        const bool consumingDrop = (pendingDrops > 0);
                        auto drainFrame = std::move(jitterBuf.front());
                        jitterBuf.pop_front();
                        (void)SyncAndSubmitFrame(std::move(drainFrame));
                        if (!consumingDrop) {
                            --toDrainFresh;
                        }
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

    // av_frame_clone() refs the VAAPI surface; it stays alive for VaapiFrame's lifetime.
    // The display thread later releases it after the next pageflip retires the surface.
    vaapiFrame->avFrame = av_frame_clone(src);
    if (!vaapiFrame->avFrame) [[unlikely]] {
        return nullptr;
    }

    // FFmpeg VAAPI ABI: data[3] holds the VASurfaceID itself, cast through uintptr_t -- it
    // is NOT a pointer to memory, never dereference it. Surface ownership is via the AVFrame.
    vaapiFrame->vaSurfaceId =
        static_cast<VASurfaceID>(reinterpret_cast<uintptr_t>( // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
            vaapiFrame->avFrame->data[3]));
    vaapiFrame->pts = src->pts;

    return vaapiFrame;
}

auto cVaapiDecoder::FilterAndAppendDecodedFrame(std::vector<std::unique_ptr<VaapiFrame>> &outFrames) -> void {
    // SD DVB streams omit color_description; default to BT.470BG (PAL).
    if (decodedFrame->colorspace == AVCOL_SPC_UNSPECIFIED && decodedFrame->height <= 576) {
        decodedFrame->colorspace = AVCOL_SPC_BT470BG;
    }

    const int64_t sourcePts = decodedFrame->pts;
    const size_t prevOutCount = outFrames.size();
    {
        const cMutexLock vaLock(&display->GetVaDriverMutex());
        if (!filterGraph) {
            (void)InitFilterGraph(decodedFrame.get());
        }

        if (filterGraph && bufferSrcCtx &&
            av_buffersrc_add_frame_flags(bufferSrcCtx, decodedFrame.get(), AV_BUFFERSRC_FLAG_KEEP_REF) >= 0) {
            while (true) {
                av_frame_unref(filteredFrame.get());
                const int filterRet = av_buffersink_get_frame(bufferSinkCtx, filteredFrame.get());
                if (filterRet < 0) {
                    break;
                }
                if (auto vaapiFrame = CreateVaapiFrame(filteredFrame.get())) {
                    outFrames.push_back(std::move(vaapiFrame));
                }
            }
        } else if (!filterGraph) {
            if (auto vaapiFrame = CreateVaapiFrame(decodedFrame.get())) {
                outFrames.push_back(std::move(vaapiFrame));
            }
        }
    }

    // rate=field doubles frame count; stamp extra fields at sourcePts + i*frameDur.
    const size_t newOutCount = outFrames.size() - prevOutCount;
    for (size_t i = 0; i < newOutCount; ++i) {
        outFrames.at(prevOutCount + i)->pts =
            (sourcePts != AV_NOPTS_VALUE && i > 0)
                ? sourcePts + (static_cast<int64_t>(outputFrameDurationMs) * 90 * static_cast<int64_t>(i))
                : sourcePts;
    }
}

auto cVaapiDecoder::DrainCodecAtEos(std::vector<std::unique_ptr<VaapiFrame>> &outFrames) -> void {
    if (!codecCtx) {
        return;
    }
    // NULL-packet drain is the documented EOS idiom; flush_buffers re-arms the codec
    // so the next real packet can be sent without a spurious EOF error.
    (void)avcodec_send_packet(codecCtx.get(), nullptr);
    while (true) {
        av_frame_unref(decodedFrame.get());
        const int drainRet = avcodec_receive_frame(codecCtx.get(), decodedFrame.get());
        if (drainRet < 0) {
            break;
        }
        FilterAndAppendDecodedFrame(outFrames);
    }
    // vaDriverMutex: avcodec_flush_buffers touches the VA driver (VAAPI hwaccel
    // reset); without it the display thread's av_hwframe_map races on iHD.
    {
        const cMutexLock vaLock(&display->GetVaDriverMutex());
        avcodec_flush_buffers(codecCtx.get());
    }
}

[[nodiscard]] auto cVaapiDecoder::DecodeOnePacket(AVPacket *pkt, std::vector<std::unique_ptr<VaapiFrame>> &outFrames)
    -> bool {
    if (!codecCtx) [[unlikely]] {
        return false;
    }

    bool anyFrameDecoded = false;
    bool packetSent = false;

    // MPEG-2: width/height arrive only with the sequence header. Drop until then -- the
    // filter graph cannot be sized without dimensions.
    if (codecCtx->codec_id == AV_CODEC_ID_MPEG2VIDEO && (codecCtx->width == 0 || codecCtx->height == 0)) {
        return false;
    }

    // EAGAIN = VAAPI surface pool full; drain frames then retry. Single iteration normally.
    while (!packetSent) {
        const int ret = avcodec_send_packet(codecCtx.get(), pkt);
        if (ret == AVERROR(EAGAIN)) {
            // Fall through to drain; next iteration retries the send.
        } else {
            if (ret < 0 && ret != AVERROR_EOF) [[unlikely]] {
                // Hard failure (corrupt HEVC NAL etc.): flush and drop the filter graph
                // so the next IDR can recover without stale VAAPI surfaces.
                // vaDriverMutex: avcodec_flush_buffers touches the VA driver;
                // without it the display thread's av_hwframe_map races on iHD.
                dsyslog("vaapivideo/decoder: send_packet failed: %s -- flushing for recovery", AvErr(ret).data());
                const cMutexLock vaLock(&display->GetVaDriverMutex());
                avcodec_flush_buffers(codecCtx.get());
                ResetFilterGraph();
                return anyFrameDecoded;
            }
            packetSent = true; // success or EOF; don't retry
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

            // SD DVB streams omit color_description; default to BT.470BG (PAL) so
            // scale_vaapi's BT.709 conversion doesn't desaturate the output.
            if (decodedFrame->colorspace == AVCOL_SPC_UNSPECIFIED && decodedFrame->height <= 576) {
                decodedFrame->colorspace = AVCOL_SPC_BT470BG;
            }

            // Filter graph is built lazily on the first frame and after each Clear().
            // vaDriverMutex serializes VAAPI access against the display thread's DRM export.
            const int64_t sourcePts = decodedFrame->pts;
            const size_t prevOutCount = outFrames.size();
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
                    // Filter graph unavailable: pass the raw frame through unscaled.
                    if (auto vaapiFrame = CreateVaapiFrame(decodedFrame.get())) {
                        outFrames.push_back(std::move(vaapiFrame));
                        anyFrameDecoded = true;
                    }
                }
            }

            // rate=field doubles frame count; assign monotonic PTS to the extra fields
            // (source + i*frameDur). Restricted to [prevOutCount, end) to avoid re-stamping
            // outputs from earlier receive iterations within the same packet.
            const size_t newOutCount = outFrames.size() - prevOutCount;
            for (size_t i = 0; i < newOutCount; ++i) {
                outFrames.at(prevOutCount + i)->pts =
                    (sourcePts != AV_NOPTS_VALUE && i > 0)
                        ? sourcePts + (static_cast<int64_t>(outputFrameDurationMs) * 90 * static_cast<int64_t>(i))
                        : sourcePts;
            }

            // In trick mode, rate=field's first output blends temporally distant fields
            // (visible green ghosting); drop it and keep the clean second field only.
            if (trickSpeed.load(std::memory_order_relaxed) != 0 &&
                (isTrickFastForward.load(std::memory_order_relaxed) ||
                 isTrickReverse.load(std::memory_order_relaxed)) &&
                newOutCount > 1 && outFrames.at(prevOutCount)->pts == sourcePts) {
                outFrames.erase(outFrames.begin() + static_cast<std::ptrdiff_t>(prevOutCount));
            }
        }

        if (!packetSent && !receivedThisIteration) {
            // EAGAIN with nothing to drain would loop forever; drop and move on.
            esyslog("vaapivideo/decoder: EAGAIN deadlock guard fired (dropping packet)");
            break;
        }
    }

    // Reverse trick: drain the DPB after each field pair so non-IDR I-frames (broadcast
    // H.264) produce immediate output instead of accumulating in the reorder buffer.
    // PAFF: drain only after the second field (non-key) so the pair is complete.
    if (trickSpeed.load(std::memory_order_relaxed) != 0 && isTrickReverse.load(std::memory_order_relaxed) &&
        !(pkt->flags & AV_PKT_FLAG_KEY)) {
        const size_t preDrainSize = outFrames.size();
        DrainCodecAtEos(outFrames);
        if (outFrames.size() > preDrainSize) {
            anyFrameDecoded = true;
        }
    }

    // Do NOT reset the filter graph here: bob is spatial-only (no state leak), and
    // destroying it would invalidate in-flight VPP surfaces (EIO on PRIME export).
    // codecDrainPending (still-picture) is handled in Action(), not here -- the drain
    // must run even when no packet was queued (race with RequestCodecDrain).

    return anyFrameDecoded;
}

[[nodiscard]] auto cVaapiDecoder::InitFilterGraph(AVFrame *firstFrame) -> bool {
    if (filterGraph) {
        return true;
    }

    // Previous graph (via ResetFilterGraph) stays alive until the next non-null overwrite
    // so the display thread can finish mapping outstanding VPP surfaces.

    if (!display) [[unlikely]] {
        esyslog("vaapivideo/decoder: no display for filter setup");
        return false;
    }

    if (!codecCtx) [[unlikely]] {
        esyslog("vaapivideo/decoder: no codec context for filter setup");
        return false;
    }

    const int srcWidth = firstFrame->width;
    const int srcHeight = firstFrame->height;
    const bool isInterlaced = (firstFrame->flags & AV_FRAME_FLAG_INTERLACED) != 0;
    const auto srcPixFmt = static_cast<AVPixelFormat>(firstFrame->format);

    // 1088 (not 1080) catches HD streams that pad height to a 16-pixel macroblock multiple.
    const bool isUhd = (srcWidth > 1920 || srcHeight > 1088);
    const bool isSoftwareDecode = (srcPixFmt != AV_PIX_FMT_VAAPI);

    const uint32_t dstWidth = display->GetOutputWidth();
    const uint32_t dstHeight = display->GetOutputHeight();

    // DRM atomic planes have no HW scaler; compute DAR from SAR for letterbox/pillarbox.
    const int sarNum = firstFrame->sample_aspect_ratio.num > 0 ? firstFrame->sample_aspect_ratio.num : 1;
    const int sarDen = firstFrame->sample_aspect_ratio.den > 0 ? firstFrame->sample_aspect_ratio.den : 1;

    const uint64_t darNum = static_cast<uint64_t>(srcWidth) * static_cast<uint64_t>(sarNum);
    const uint64_t darDen = static_cast<uint64_t>(srcHeight) * static_cast<uint64_t>(sarDen);

    uint32_t filterWidth = dstWidth;
    uint32_t filterHeight = dstHeight;

    // Integer cross-multiply (no FP): darNum/darDen vs dstWidth/dstHeight.
    if (darNum * dstHeight > darDen * static_cast<uint64_t>(dstWidth)) {
        // Source wider than display -> letterbox.
        filterWidth = dstWidth;
        filterHeight = static_cast<uint32_t>(static_cast<uint64_t>(dstWidth) * darDen / darNum);
    } else if (darNum * dstHeight < darDen * static_cast<uint64_t>(dstWidth)) {
        // Source narrower than display -> pillarbox.
        filterHeight = dstHeight;
        filterWidth = static_cast<uint32_t>(static_cast<uint64_t>(dstHeight) * darNum / darDen);
    }

    // NV12 chroma is 2x2-subsampled; dimensions must be even.
    filterWidth = std::max(filterWidth & ~1U, 2U);
    filterHeight = std::max(filterHeight & ~1U, 2U);

    // Denoise/sharpen: off at UHD (GPU-bound), heavy for MPEG-2 SD, light for HD.
    int denoiseLevel = 0;
    int sharpnessLevel = 0;

    if (!isUhd) {
        if (currentCodecId == AV_CODEC_ID_MPEG2VIDEO) {
            denoiseLevel = 16;
            sharpnessLevel = 36;
        } else {
            denoiseLevel = 6;
            sharpnessLevel = 30;
        }
    }

    const bool needsResize =
        (filterWidth != static_cast<uint32_t>(srcWidth) || filterHeight != static_cast<uint32_t>(srcHeight));

    // VBR DVB streams may report framerate==0; default to 50 fps (DVB-S/T baseline, = 25i).
    const int fpsNum = codecCtx->framerate.num > 0 ? codecCtx->framerate.num : 50;
    const int fpsDen = codecCtx->framerate.den > 0 ? codecCtx->framerate.den : 1;

    // rate=field doubles interlaced output (25i -> 50p). Progressive below display refresh
    // is upconverted via fps (metadata-only, zero-copy). Never downconvert.
    const int naturalOutputFps = (fpsNum / std::max(fpsDen, 1)) * (isInterlaced ? 2 : 1);
    const int displayFps = static_cast<int>(display->GetOutputRefreshRate());
    const bool upconvertProgressive =
        (!isInterlaced && naturalOutputFps > 0 && displayFps > 0 && naturalOutputFps < displayFps);
    const int outputFps = upconvertProgressive ? displayFps : naturalOutputFps;

    // Filter chain (comma-joined). Both paths converge on NV12 BT.709 TV-range.
    //   SW: [bwdif] -> [hqdn3d] -> format=nv12 -> hwupload -> [scale_vaapi] -> [sharpness_vaapi] -> [fps]
    //   HW: [deinterlace_vaapi] -> [denoise_vaapi] -> [scale_vaapi] -> [sharpness_vaapi] -> [fps]
    std::vector<std::string> filters;

    // Trick / still: minimal pipeline (no denoise/sharpen).
    // Still: skip deinterlacer -- VAAPI VPP holds one frame for timestamp interpolation
    // and never flushes on EOS, so a single I-frame would never emerge. Minor combing OK.
    // Trick: bob (spatial-only); the steady cadence absorbs the one-frame delay.
    const bool isTrickMode = trickSpeed.load(std::memory_order_relaxed) != 0;
    const bool isStill = stillPictureMode.load(std::memory_order_relaxed);
    const bool useSimpleDeinterlace = isTrickMode || isStill;

    if (isSoftwareDecode) {
        if (isInterlaced && !isStill) {
            if (useSimpleDeinterlace) {
                // yadif: spatial-only, no temporal priming (bwdif would absorb first I-frame).
                filters.emplace_back("yadif=mode=send_frame:parity=auto:deint=all");
            } else {
                // bwdif (w3fdif+yadif hybrid) beats VAAPI motion_adaptive on Mesa.
                filters.emplace_back("bwdif=mode=send_field:parity=auto:deint=all");
            }
        }
        if (!useSimpleDeinterlace && denoiseLevel > 0) {
            // hqdn3d strength: 5 for MPEG-2 (temporal-heavy), 3 for HD (spatial-dominant).
            const int hqdn3dStrength = (currentCodecId == AV_CODEC_ID_MPEG2VIDEO) ? 5 : 3;
            filters.push_back(std::format("hqdn3d={}", hqdn3dStrength));
        }
        filters.emplace_back("format=nv12");
        filters.emplace_back("hwupload");
    } else {
        if (isInterlaced && !isStill && !vaapiContext->deinterlaceMode.empty()) {
            if (isTrickMode) {
                // bob: spatial-only, immediate output from a single input frame.
                filters.emplace_back("deinterlace_vaapi=mode=bob:rate=frame");
            } else {
                filters.push_back(std::format("deinterlace_vaapi=mode={}:rate=field", vaapiContext->deinterlaceMode));
            }
        }
        if (!useSimpleDeinterlace && denoiseLevel > 0 && vaapiContext->hasDenoise) {
            filters.push_back(std::format("denoise_vaapi=denoise={}", denoiseLevel));
        }
    }

    // scale_vaapi unconditional: normalizes to NV12 BT.709 TV-range for the DRM plane.
    if (needsResize) {
        const char *scaleMode = isUhd ? "" : ":mode=hq"; // hq (bicubic) too expensive at UHD
        filters.push_back(std::format("scale_vaapi=w={}:h={}{}:format=nv12:out_color_matrix=bt709:out_range=tv",
                                      filterWidth, filterHeight, scaleMode));
    } else {
        filters.emplace_back("scale_vaapi=format=nv12:out_color_matrix=bt709:out_range=tv");
    }

    if (!useSimpleDeinterlace && sharpnessLevel > 0 && vaapiContext->hasSharpness) {
        filters.push_back(std::format("sharpness_vaapi=sharpness={}", sharpnessLevel));
    }

    if (!useSimpleDeinterlace && upconvertProgressive) {
        // fps last: scale/sharpen run once per input frame; only metadata is duplicated.
        filters.push_back(std::format("fps={}", displayFps));
    }

    std::string filterChain;
    for (const auto &filter : filters) {
        if (!filterChain.empty()) {
            filterChain += ',';
        }
        filterChain += filter;
    }

    filterGraph.reset(avfilter_graph_alloc());
    if (!filterGraph) [[unlikely]] {
        esyslog("vaapivideo/decoder: failed to allocate filter graph");
        return false;
    }

    // Numeric pix_fmt: symbolic aliases vary across FFmpeg versions for HW formats.
    const std::string bufferSrcArgs =
        std::format("video_size={}x{}:pix_fmt={}:time_base=1/90000:pixel_aspect={}/{}:frame_rate={}/{}", srcWidth,
                    srcHeight, static_cast<int>(srcPixFmt), sarNum, sarDen, fpsNum, fpsDen);

    dsyslog("vaapivideo/decoder: buffer source args='%s'", bufferSrcArgs.c_str());

    // alloc_filter: hw_frames_ctx must be attached before init (FFmpeg 7.x rejects without).
    bufferSrcCtx = avfilter_graph_alloc_filter(filterGraph.get(), avfilter_get_by_name("buffer"), "in");
    if (!bufferSrcCtx) [[unlikely]] {
        esyslog("vaapivideo/decoder: failed to allocate buffer source filter");
        ResetFilterGraph();
        return false;
    }

    // HW decode: buffer source needs hw_frames_ctx to recognise VAAPI surfaces.
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
        // parameters_set transfers the hw_frames_ctx ref; only free the struct.
        av_free(hwFramesParams);
    }

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

    // FFmpeg I/O naming is from the graph string's POV: outputs -> src, inputs -> sink.
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

    // SW decode: all nodes need hwDeviceRef, not just hwupload (scale/sharpen resolve via it).
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

    // Sync parameters: frame duration for the A/V controller, jitter buffer depth at post-filter rate.
    outputFrameDurationMs = outputFps > 0 ? std::max(1, 1000 / outputFps) : 20;
    jitterTarget = outputFps > 0 ? ((outputFps * DECODER_JITTER_BUFFER_MS) + 500) / 1000 : 50;

    isyslog("vaapivideo/decoder: VAAPI filter initialized (%dx%d -> %ux%u%s%s)", srcWidth, srcHeight, filterWidth,
            filterHeight, isInterlaced ? ", deinterlaced" : "",
            upconvertProgressive ? (isInterlaced ? "" : ", upconverted") : "");
    dsyslog("vaapivideo/decoder: filter chain='%s'", filterChain.c_str());
    return true;
}

auto cVaapiDecoder::ResetFilterGraph() -> void {
    bufferSrcCtx = nullptr;
    bufferSinkCtx = nullptr;
    // Keep the old graph alive: its hw_frames_ctx keeps VPP output surfaces PRIME-exportable
    // until the display thread finishes mapping them. Destroying now causes -EIO on iHD.
    // Guard: double-reset (Clear -> drain -> EOS) must not null-out the saved graph.
    if (filterGraph) {
        previousFilterGraph = std::move(filterGraph);
    }
}

[[nodiscard]] auto cVaapiDecoder::SubmitTrickFrame(std::unique_ptr<VaapiFrame> frame) -> bool {
    const int64_t pts = frame->pts;
    const int64_t prevPts = prevTrickPts.load(std::memory_order_relaxed);

    // Reverse trick: GOPs arrive backwards but frames within each GOP are in decode order.
    // Show only the first (lowest-PTS) frame per GOP; drop the rest.
    if (isTrickReverse.load(std::memory_order_relaxed) && pts != AV_NOPTS_VALUE && prevPts != AV_NOPTS_VALUE &&
        pts > prevPts) {
        return true;
    }

    // Deinterlaced field pairs share PTS; pace once per source frame, pass the second field.
    if (pts != prevPts) {
        prevTrickPts.store(pts, std::memory_order_relaxed);
        if (pts != AV_NOPTS_VALUE) {
            lastPts.store(pts, std::memory_order_release);
        }

        // Wait for the pacing deadline, then arm the next one.
        const uint64_t due = nextTrickFrameDue.load(std::memory_order_relaxed);
        while (cTimeMs::Now() < due && !stopping.load(std::memory_order_relaxed) &&
               trickSpeed.load(std::memory_order_relaxed) != 0) {
            cCondWait::SleepMs(10);
        }

        // Fast: hold = ptsDelta/90/mult clamped to [10,2000] ms. Slow: precomputed trickHoldMs.
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

    // Hard cap: delta + 1 s headroom, max 5 s, so a dead audio path can't block forever.
    const int64_t maxWaitMs = std::min<int64_t>((delta / 90) + 1000, 5000LL);
    const cTimeMs deadline(static_cast<int>(maxWaitMs));

    while (!deadline.TimedOut() && !stopping.load(std::memory_order_relaxed)) {
        // Clear() / channel-switch arms freerunFrames; our held frame is now stale.
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

[[nodiscard]] auto cVaapiDecoder::SyncLatency90k(const cAudioProcessor *ap) const noexcept -> int64_t {
    // PCM vs passthrough latency knob + constant 2-frame tail (1 VSync pipeline + 1 HDMI
    // link lag). The 2-frame model zeros the default bias; operators only need non-zero
    // knobs for unusual TV input lag (gaming-mode: +ms; movie mode ~50 ms: -ms).
    const int latencyMs = (ap && ap->IsPassthrough()) ? vaapiConfig.passthroughLatency.load(std::memory_order_relaxed)
                                                      : vaapiConfig.pcmLatency.load(std::memory_order_relaxed);
    return (static_cast<int64_t>(latencyMs) + (2 * static_cast<int64_t>(outputFrameDurationMs))) * 90;
}

auto cVaapiDecoder::ResetSmoothedDelta() noexcept -> void {
    smoothedDeltaValid = false;
    smoothedDelta90k = 0;
    emaResidual90k = 0;
    warmupSampleCount = 0;
    warmupSampleSum90k = 0;
    rawDeltaSumSinceLog90k = 0;
    rawDeltaCountSinceLog = 0;
}

auto cVaapiDecoder::UpdateSmoothedDelta(int64_t rawDelta90k) noexcept -> void {
    // Two-phase: warmup N-sample mean seeds the EMA (cuts deinterlaced-50p oscillation bias);
    // then residual-accumulator EMA, alpha=1/EMA_SAMPLES.
    // INVARIANT: call exactly once per output frame; a second call site alters reaction time.

    // Feed the log-interval mean ("d=" in LogSyncStats).
    rawDeltaSumSinceLog90k += rawDelta90k;
    ++rawDeltaCountSinceLog;

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
        emaResidual90k = 0;
        return;
    }
    // Residual-accumulator EMA: accumulate into emaResidual90k and apply whole-tick steps
    // when the residual crosses N; avoids integer truncation stall on small diffs.
    const int64_t diff = rawDelta90k - smoothedDelta90k;
    emaResidual90k += diff;
    const int64_t step = emaResidual90k / DECODER_SYNC_EMA_SAMPLES;
    smoothedDelta90k += step;
    emaResidual90k -= step * DECODER_SYNC_EMA_SAMPLES;
}

auto cVaapiDecoder::LogSyncStats(int64_t rawDelta90k, int64_t latency90k, const cAudioProcessor *ap) -> void {
    if (!(syncLogPending.exchange(false, std::memory_order_relaxed) || nextSyncLog.TimedOut())) {
        return;
    }
    // Suppress during warmup; timer not rearmed so first post-warmup call fires at once.
    if (!smoothedDeltaValid) {
        return;
    }
    // d= interval mean (not point sample); lat= SyncLatency90k (2*frameDur + user knob).
    // If bumping the knob doesn't move lat=, the stream uses the other path (PCM vs PT).
    const int64_t meanRawDelta90k =
        (rawDeltaCountSinceLog > 0) ? (rawDeltaSumSinceLog90k / rawDeltaCountSinceLog) : rawDelta90k;
    const auto meanTenths = static_cast<long long>(meanRawDelta90k * 10 / 90);
    const auto avgTenths = static_cast<long long>(smoothedDelta90k * 10 / 90);
    dsyslog("vaapivideo/decoder: sync d=%+lld.%01lldms avg=%+lld.%01lldms lat=%lldms buf=%zu aq=%zu miss=%d "
            "drop=%d skip=%d",
            meanTenths / 10, std::abs(meanTenths % 10), avgTenths / 10, std::abs(avgTenths % 10),
            static_cast<long long>(latency90k / 90), jitterBuf.size(), ap->GetQueueSize(), drainMissCount,
            syncDropSinceLog, syncSkipSinceLog);
    rawDeltaSumSinceLog90k = 0;
    rawDeltaCountSinceLog = 0;
    drainMissCount = 0;
    syncDropSinceLog = 0;
    syncSkipSinceLog = 0;
    nextSyncLog.Set(DECODER_SYNC_LOG_INTERVAL_MS);
}

auto cVaapiDecoder::RunJitterPrimeSync(cAudioProcessor *ap) -> bool {
    // One-shot coarse alignment after prime fill: behind -> drop stale heads; ahead ->
    // busy-wait + trim overshoot. Returns false if clock is NOPTS (post track/channel
    // switch) to defer and avoid seeding the EMA from a transient skew.
    if (jitterBuf.empty()) {
        return true; // nothing to do; "consumed" so the caller stops retrying
    }
    // Snapshot for initial decision; per-loop checks re-read the clock.
    const int64_t fpts0 = jitterBuf.front()->pts;
    const int64_t clock0 = ap->GetClock();
    if (clock0 == AV_NOPTS_VALUE) {
        // Defer; see the function header for the rationale.
        return false;
    }
    if (fpts0 == AV_NOPTS_VALUE) {
        // No PTS on head (e.g. MPEG-2 B-frame after teardown); let steady-state handle it.
        return true;
    }
    // >0: video ahead; <0: behind. latency accounts for pending PCM in the sink.
    const int64_t latency = SyncLatency90k(ap);
    const int64_t initDelta = fpts0 - clock0 - latency;

    if (initDelta < 0) {
        int drops = 0;
        while (!jitterBuf.empty() && !stopping.load(std::memory_order_relaxed)) {
            if (jitterBuf.front()->pts == AV_NOPTS_VALUE) {
                // Unknown PTS, can't compare; let the steady-state path handle it.
                break;
            }
            const int64_t clk = ap->GetClock();
            if (clk == AV_NOPTS_VALUE || jitterBuf.front()->pts >= clk + latency) {
                break; // caught up
            }
            jitterBuf.pop_front();
            ++drops;
        }
        dsyslog("vaapivideo/decoder: prime-sync dropped %d frames (initDelta=%+lldms)", drops,
                static_cast<long long>(initDelta / 90));
    } else if (initDelta > 0) {
        dsyslog("vaapivideo/decoder: prime-sync ahead d=%+lldms -- waiting for audio",
                static_cast<long long>(initDelta / 90));
        // 10 ms slices; one-shot prime, not hot path. Freerun drains during prolonged waits.
        while (!stopping.load(std::memory_order_relaxed)) {
            const int64_t clk = ap->GetClock();
            if (clk == AV_NOPTS_VALUE || clk >= fpts0 - latency) {
                break;
            }
            cCondWait::SleepMs(10);
        }
        // Overshoot trim: 10 ms tick granularity may leave audio past fpts0.
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

    // Reseed EMA and arm cooldown; nextSyncLog=0 forces an immediate post-prime stat line.
    ResetSmoothedDelta();
    syncCooldown.Set(DECODER_SYNC_COOLDOWN_MS);
    nextSyncLog.Set(0);
    return true;
}

auto cVaapiDecoder::SkipStaleJitterFrames(cAudioProcessor *ap) -> void {
    // Bulk-drop heads more than HARD_THRESHOLD behind the clock. Faster than SyncAndSubmitFrame's
    // one-by-one path after backlogs (decode stall, USB audio re-sync). Keeps >= 1 frame.
    const int64_t latency = SyncLatency90k(ap);
    while (jitterBuf.size() > 1 && !stopping.load(std::memory_order_relaxed)) {
        if (jitterBuf.front()->pts == AV_NOPTS_VALUE) {
            break;
        }
        const int64_t clock = ap->GetClock();
        // Inside the hard window; soft corridor handles the residual.
        if (clock == AV_NOPTS_VALUE || jitterBuf.front()->pts - clock - latency >= -DECODER_SYNC_HARD_THRESHOLD) {
            break;
        }
        jitterBuf.pop_front();
        ++syncDropSinceLog;
    }
}

[[nodiscard]] auto cVaapiDecoder::SyncAndSubmitFrame(std::unique_ptr<VaapiFrame> frame) -> bool {
    // A/V sync gate (audio-master). See AVSYNC.md. Three regimes:
    //   bypass: trick / freerun / NOPTS -> submit immediately
    //   hard (|raw| > 200 ms): behind -> drop; ahead -> replay: block, live: big sleep
    //   soft (|EMA| > 20 ms, cooldown OK, cap MAX_CORRECTION):
    //     behind -> drop round(ms/frameDur); ahead -> sleep(correctMs + frameDur)
    // Soft adjusts EMA by *measured* effect (drop: +=frameDur*90; sleep: -=(elapsed-frameDur)*90).
    // One event per COOLDOWN (5 s = 5 tau). MAX_CORRECTION=100 ms tracks ~20 ms/s drift.

    if (!frame || !display) [[unlikely]] {
        return false;
    }

    const int64_t pts = frame->pts;

    // --- Trick mode: SubmitTrickFrame() owns pacing; audio is muted. ---
    if (trickSpeed.load(std::memory_order_acquire) != 0) {
        if (trickExitPending.exchange(false, std::memory_order_acquire)) {
            // Play()-without-TrickSpeed(0): clear flags, arm freerun while audio re-anchors.
            trickSpeed.store(0, std::memory_order_relaxed);
            isTrickReverse.store(false, std::memory_order_relaxed);
            isTrickFastForward.store(false, std::memory_order_relaxed);
            freerunFrames.store(DECODER_SYNC_FREERUN_FRAMES, std::memory_order_relaxed);
            syncLogPending.store(true, std::memory_order_relaxed);
            // Fall through to the normal sync path.
        } else {
            return SubmitTrickFrame(std::move(frame));
        }
    }

    // Still picture: no audio to sync against; stale clock would drop/delay the frame.
    if (stillPictureMode.load(std::memory_order_relaxed)) {
        if (pts != AV_NOPTS_VALUE) {
            lastPts.store(pts, std::memory_order_release);
        }
        return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
    }

    // Publish latest valid PTS for lock-free observers (still detection, position query).
    if (pts != AV_NOPTS_VALUE) {
        lastPts.store(pts, std::memory_order_release);
    }

    // Freerun: no audio processor, no PTS, or inside post-Clear/trick-exit window.
    auto *const ap = audioProcessor.load(std::memory_order_acquire);
    if (!ap || pts == AV_NOPTS_VALUE) {
        return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
    }
    if (freerunFrames.load(std::memory_order_relaxed) > 0) {
        freerunFrames.fetch_sub(1, std::memory_order_relaxed);
        return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
    }

    // Snapshot latency once; a mid-frame PCM<->passthrough flip must not change it.
    const int64_t latency = SyncLatency90k(ap);
    const int64_t clock = ap->GetClock();
    if (clock == AV_NOPTS_VALUE) {
        // No clock yet (stall / codec swap / device reset): freerun, discard stale drops.
        pendingDrops = 0;
        if (syncLogPending.exchange(false, std::memory_order_relaxed) || nextSyncLog.TimedOut()) {
            dsyslog("vaapivideo/decoder: sync freerun (no clock) buf=%zu", jitterBuf.size());
            nextSyncLog.Set(DECODER_SYNC_LOG_INTERVAL_MS);
        }
        return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
    }

    // Prior correction burst: one drop per call; pts advances by frameDur, clock steady.
    if (pendingDrops > 0) {
        --pendingDrops;
        smoothedDelta90k += static_cast<int64_t>(outputFrameDurationMs) * 90;
        ++syncDropSinceLog;
        return true;
    }

    // rawDelta: instantaneous A/V error (>0 = video ahead); EMA filters jitter for soft corridor.
    const int64_t rawDelta = pts - clock - latency;
    UpdateSmoothedDelta(rawDelta);
    LogSyncStats(rawDelta, latency, ap);

    // Hard drop (channel switch / stream gap): bypass cooldown, reset discontinuous EMA.
    if (rawDelta < -DECODER_SYNC_HARD_THRESHOLD) [[unlikely]] {
        ++syncDropSinceLog;
        ResetSmoothedDelta();
        pendingDrops = 0;
        syncCooldown.Set(DECODER_SYNC_COOLDOWN_MS);
        return true;
    }

    // Pre-sleep timestamp for the post-submit EMA adjustment; 0 = no sleep happened.
    uint64_t preSleepMs = 0;

    // Hard-ahead: replay blocks until audio catches up; live fires one big sleep
    // (<= HARD_AHEAD_MAX_MS) to collapse the bias into the soft corridor's range.
    if (rawDelta > DECODER_SYNC_HARD_THRESHOLD) [[unlikely]] {
        if (!liveMode.load(std::memory_order_relaxed)) {
            WaitForAudioCatchUp(ap, pts, latency, rawDelta);
            ++syncSkipSinceLog;
            pendingDrops = 0;
            syncCooldown.Set(DECODER_SYNC_COOLDOWN_MS);
            return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
        }
        constexpr int HARD_AHEAD_MAX_MS = 500;
        const int bigSleepMs = std::min(static_cast<int>(rawDelta / 90), HARD_AHEAD_MAX_MS);
        preSleepMs = cTimeMs::Now();
        cCondWait::SleepMs(bigSleepMs + outputFrameDurationMs);
        ++syncSkipSinceLog;
        pendingDrops = 0;
        syncCooldown.Set(DECODER_SYNC_COOLDOWN_MS);
    }

    // Soft corridor: proportional correction capped at MAX_CORRECTION, rate-limited by cooldown.
    // Ahead sleeps (correctMs + frameDur): the extra frameDur compensates for the normal
    // iteration period so the net shift equals correctMs. Measured post-hoc via elapsed.
    if (syncCooldown.TimedOut() && smoothedDeltaValid) {
        const int64_t absDelta90k = smoothedDelta90k < 0 ? -smoothedDelta90k : smoothedDelta90k;
        if (absDelta90k > DECODER_SYNC_CORRIDOR) {
            const int correctMs = std::min(static_cast<int>(absDelta90k / 90), DECODER_SYNC_MAX_CORRECTION_MS);
            syncCooldown.Set(DECODER_SYNC_COOLDOWN_MS);

            if (smoothedDelta90k < 0) {
                // Behind: drop N frames (round-to-nearest); one now, rest via pendingDrops.
                const int totalDrops = std::max(1, (correctMs + (outputFrameDurationMs / 2)) / outputFrameDurationMs);
                pendingDrops = totalDrops - 1;
                smoothedDelta90k += static_cast<int64_t>(outputFrameDurationMs) * 90;
                ++syncDropSinceLog;
                return true;
            }
            // Ahead: sleep once; EMA updated from measured elapsed after SubmitFrame.
            preSleepMs = cTimeMs::Now();
            cCondWait::SleepMs(correctMs + outputFrameDurationMs);
            ++syncSkipSinceLog;
        }
    }

    const bool submitted = display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
    if (preSleepMs != 0) {
        // Effective shift = elapsed - frameDur; std::max guards unsigned underflow.
        const auto elapsedMs = static_cast<int>(cTimeMs::Now() - preSleepMs);
        const int extraMs = std::max(0, elapsedMs - outputFrameDurationMs);
        smoothedDelta90k -= static_cast<int64_t>(extraMs) * 90;
    }
    return submitted;
}
