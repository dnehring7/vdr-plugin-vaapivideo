// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file decoder.cpp
 * @brief Threaded VAAPI decoder with VPP filter graph and audio-mastered A/V sync.
 *
 * Pipeline:
 *   EnqueueData() -> packetQueue -> Action() -> VAAPI decode -> filter graph
 *   -> jitterBuf -> SyncAndSubmitFrame -> display->SubmitFrame
 *   (Unified for live and replay. liveMode selects hard-ahead policy in SyncAndSubmitFrame.)
 *
 * Filters:
 *   [bwdif|deinterlace_vaapi rate=field] -> [hqdn3d|denoise_vaapi]
 *   -> scale_vaapi (NV12 BT.709 TV-range) -> [sharpness_vaapi]
 *   VPP nodes probed per GPU at device init (caps.cpp).
 *
 * Sync (audio-master, four regimes):
 *   freerun  -- no clock / post-event window             -> submit unpaced
 *   catch-up -- raw < -2xHARD_THRESHOLD                 -> silent bulk drop until in-range
 *   hard     -- |raw| > HARD_THRESHOLD                   -> replay: block ahead / drop behind
 *                                                           live: one big sleep ahead / drop behind
 *   soft     -- |EMA| > CORRIDOR, cooldown OK            -> round(EMA/frameDur) drops or one
 *                                                           sleep of (correctMs + frameDur);
 *                                                           EMA adjusted by measured effect
 *
 * Threading: codecMutex guards codec context + filter graph (decode thread vs reopen/clear).
 *   parserMutex guards the AVCodecParserContext (EnqueueData vs reopen/clear). Separate mutexes
 *   so the dvbplayer/receiver feeding EnqueueData does not serialize the decode thread on its
 *   way to VAAPI. av_parser_parse2 only reads immutable codecCtx fields. packetMutex guards
 *   packetQueue (also the condvar futex).
 *   Lock order: ALWAYS codecMutex -> parserMutex -> packetMutex. Writers of codecCtx existence
 *   (OpenCodecWithInfo / Clear / SetTrickSpeed flush) take BOTH codecMutex and parserMutex;
 *   the decode thread takes only codecMutex; EnqueueData takes only parserMutex.
 *   jitterBuf is decode-thread-owned (no lock needed); cross-thread requests use atomics +
 *   a deferred apply at the top of Action().
 */

#include "decoder.h"
#include "audio.h"
#include "common.h"
#include "config.h"
#include "device.h"
#include "display.h"
#include "filter.h"
#include "stream.h"

// C++ Standard Library
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
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
#include <libavutil/buffer.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/mastering_display_metadata.h>
#include <libavutil/mem.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
#include <libavutil/rational.h>
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

constexpr size_t DECODER_QUEUE_CAPACITY =
    200; ///< ~4 s @ 50 fps. Overflow drops oldest; trick mode limits to DECODER_TRICK_QUEUE_DEPTH.
constexpr int DECODER_SUBMIT_TIMEOUT_MS = 100; ///< VSync backpressure budget inside display->SubmitFrame().
constexpr int DECODER_SYNC_COOLDOWN_MS = 5000; ///< Min interval between soft corrections; equals 5x EMA time constant.
constexpr int64_t DECODER_SYNC_CORRIDOR_90K =
    50 * PTS_TICKS_PER_MS; ///< Soft half-width: 50 ms in 90k ticks. Below the ~80 ms lipsync percept threshold.
constexpr int DECODER_SYNC_EMA_SAMPLES =
    50; ///< EMA alpha = 1/N (~= 1 s @ 50 fps). Residual accumulator avoids truncation stall.
constexpr int DECODER_SYNC_FREERUN_FRAMES = 1; ///< Unpaced frames after Clear() / track switch / trick-exit.
constexpr int64_t DECODER_SYNC_HARD_THRESHOLD_90K =
    200 * PTS_TICKS_PER_MS; ///< 200 ms in 90k ticks. Beyond this the soft corridor cannot recover in one event.
constexpr int DECODER_SYNC_LOG_INTERVAL_MS = 2000; ///< Periodic sync-stats dsyslog cadence.
constexpr int DECODER_SYNC_MAX_CORRECTION_MS =
    200; ///< Per-event cap = HARD_THRESHOLD; one event clears any in-corridor offset.
constexpr int DECODER_SYNC_WARMUP_SAMPLES =
    50; ///< Mean-seed samples before EMA starts; sqrt(N) cuts 50p deinterlace bias.
constexpr int DECODER_SYNC_HARD_AHEAD_MAX_MS =
    500; ///< Live hard-ahead sleep cap (ms); prevents indefinite chase + upstream queue overflow.
constexpr size_t DECODER_JITTERBUF_HARD_CAP =
    150; ///< ~3 s @ 50 fps. Caps GPU surface retention when a stuck-but-valid audio clock holds the
         ///< drain's due-gate closed (SkipStaleJitterFrames only drops BEHIND-clock heads).
constexpr int64_t DECODER_DRAIN_FUTURE_MAX_MS =
    3000; ///< Drop any head sitting more than this far ahead of audio_clock. Such an offset is a real
          ///< PTS discontinuity (post-ATTA anchor mismatch, broadcast PCR break) -- waiting for natural
          ///< sync wastes seconds and looks like a startup freeze. Dropping immediately lets the next
          ///< frame proceed and SyncAndSubmitFrame's catch-up path cleans up the residual.

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
    // codecMutex first, parserMutex second (file-level lock order). Together they exclude both
    // the decode thread (codecMutex) and any concurrent EnqueueData parse (parserMutex), so the
    // codec + parser can be reset as one atomic operation.
    const cMutexLock decodeLock(&codecMutex);
    const cMutexLock parseLock(&parserMutex);

    DrainQueue();

    {
        // vaDriverMutex: avcodec_flush_buffers (VAAPI hwaccel reset) and filterChain.Reset()
        // (VPP teardown) make VA API calls that race the display thread's PRIME export.
        const cMutexLock vaLock(&display->GetVaDriverMutex());

        if (codecCtx) {
            avcodec_flush_buffers(codecCtx.get()); // codec stays open; next I-frame continues without reopen.
        }

        // Graph caches hw_frames_ctx and may hold stale-PTS frames; rebuilt lazily on next frame.
        filterChain.Reset();
        if (decodedFrame) {
            av_frame_unref(decodedFrame.get());
        }
        if (filteredFrame) {
            av_frame_unref(filteredFrame.get());
        }
    }
    // hasLoggedFirstFrame NOT reset here: only OpenCodecWithInfo() resets it, avoiding duplicate logs per reopen.

    // AVCodecParserContext has no flush API; recreate it. Null parserCtx means the mediaplayer
    // path (extradata present); reintroducing a parser mid-stream would corrupt AU boundaries.
    if (parserCtx && currentCodecId != AV_CODEC_ID_NONE) {
        parserCtx.reset(av_parser_init(currentCodecId));
    } else if (currentCodecId == AV_CODEC_ID_NONE) {
        parserCtx.reset();
    }

    // Epoch BEFORE NOPTS: any in-flight decoder publish observes the new epoch on its check or
    // recheck and aborts/undoes. Reversed order would let a stale publish overwrite NOPTS.
    clearEpoch.fetch_add(1, std::memory_order_release);
    lastPts.store(AV_NOPTS_VALUE, std::memory_order_release);
    codecDrainPending.store(false, std::memory_order_relaxed);
    stillPictureMode.store(false, std::memory_order_relaxed);

    // jitterBuf and EMA state are decode-thread-owned; reset is deferred via atomic flag.
    // freerunFrames MUST store before jitterFlushPending: if the order is reversed the
    // decode thread could drain through the due-gate on a still-NOPTS clock and black the screen.
    freerunFrames.store(DECODER_SYNC_FREERUN_FRAMES, std::memory_order_relaxed);
    jitterFlushPending.store(true, std::memory_order_release);

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

    // parserMutex only: av_parser_parse2 reads codecCtx's codec_id (set once at open) and writes
    // parserCtx state. The decode thread modifies codecCtx under codecMutex without touching the
    // parser, so the two paths run concurrently. Writers of codecCtx existence
    // (OpenCodecWithInfo / Clear / SetTrickSpeed) take BOTH mutexes (codecMutex -> parserMutex).
    const cMutexLock parseLock(&parserMutex);

    if (!codecCtx || !parserCtx) {
        return;
    }

    // pts is valid only for the first segment of each AU; the parser propagates it internally.
    const uint8_t *parseData = data;
    if (size > static_cast<size_t>(INT_MAX)) [[unlikely]] {
        return;
    }
    int parseSize = static_cast<int>(size);
    int64_t currentPts = pts;

    while (parseSize > 0) {
        uint8_t *parsedData = nullptr; // NOLINT(misc-const-correctness) -- av_parser_parse2 out-param
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
            // FF: drop non-keyframes (inter frames are undisplayable without reference frames).
            // Reverse must not filter: PAFF second fields carry key_frame=0 but are needed.
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

            PushPacketToQueue(pkt);
        }
    }

    // Do NOT drain here: the parser withholds the final AU until the next start code.
    // FlushParser() drains after all PES chunks are delivered; draining per-call would
    // fragment multi-PES I-frames into partial AUs.
}

auto cVaapiDecoder::EnqueuePacket(const AVPacket *packet) -> void {
    // Container demuxers (av_read_frame) yield whole AUs, so no parser step is needed.
    // Clone the caller's packet and push through the same trick/queue-depth policy as EnqueueData.
    if (!packet || stopping.load(std::memory_order_relaxed)) {
        return;
    }

    const cMutexLock decodeLock(&codecMutex);
    if (!codecCtx) {
        return;
    }

    // Same FF keyframe filter as EnqueueData; AV_PKT_FLAG_KEY is the demuxer's equivalent of parserCtx->key_frame.
    if (trickSpeed.load(std::memory_order_acquire) != 0 && isTrickFastForward.load(std::memory_order_relaxed) &&
        (packet->flags & AV_PKT_FLAG_KEY) == 0) {
        return;
    }

    AVPacket *pkt = av_packet_clone(packet);
    if (!pkt) [[unlikely]] {
        return;
    }
    PushPacketToQueue(pkt);
}

auto cVaapiDecoder::PushPacketToQueue(AVPacket *pkt) -> void {
    // Takes ownership of pkt regardless of outcome: freed on overflow/trick drop, queued otherwise.
    const cMutexLock lock(&packetMutex);
    const bool isTrickMode = trickSpeed.load(std::memory_order_relaxed) != 0;
    const size_t maxDepth = isTrickMode ? DECODER_TRICK_QUEUE_DEPTH : DECODER_QUEUE_CAPACITY;
    if (packetQueue.size() >= maxDepth) {
        if (isTrickMode) {
            // Trick-queue depth is 1, and during a TrickSpeed transition VDR can bulk-feed dozens
            // of stale packets in milliseconds. Throttle to one log per 500 ms with a drop count so
            // syslog isn't flooded; cf. audio.cpp "queue full" rate-limit.
            ++trickDropsSinceWarn;
            if (lastTrickDropWarn.Elapsed() > 500) {
                dsyslog("vaapivideo/decoder: trick enqueue: queue full (depth=%zu) -- %zu drops since last log",
                        packetQueue.size(), trickDropsSinceWarn);
                trickDropsSinceWarn = 0;
                lastTrickDropWarn.Set();
            }
            av_packet_free(&pkt);
            return;
        }
        const std::unique_ptr<AVPacket, FreeAVPacket> dropped{packetQueue.front()};
        packetQueue.pop();
    }
    packetQueue.push(pkt);
    packetCondition.Broadcast();
}

auto cVaapiDecoder::NoteStarvationTick(const AVPacket *pkt) noexcept -> void {
    // Single relaxed load on the fast path once first frame lands; no steady-state cost.
    if (hasLoggedFirstFrame.load(std::memory_order_relaxed)) {
        return;
    }

    const size_t sent = packetsSinceOpen.fetch_add(1, std::memory_order_relaxed) + 1;
    if (pkt != nullptr && (pkt->flags & AV_PKT_FLAG_KEY) != 0) {
        keyPacketsSinceOpen.fetch_add(1, std::memory_order_relaxed);
    }

    const uint64_t openedMs = codecOpenTimeMs.load(std::memory_order_relaxed);
    if (openedMs == 0) [[unlikely]] {
        return;
    }
    const uint64_t nowMs = cTimeMs::Now();
    const uint64_t elapsed = nowMs - openedMs;

    // Tier 1 (~3 s): typical for long-GOP streams where mid-entry must wait for the next IDR.
    if (elapsed > 3000 && !starvationWarned.exchange(true, std::memory_order_relaxed)) {
        isyslog("vaapivideo/decoder: no frame %llu ms after open (packets sent=%zu, "
                "of which keyframes=%zu) -- FFmpeg is waiting for a random-access point; "
                "likely mid-GOP entry on a long-GOP stream",
                static_cast<unsigned long long>(elapsed), sent, keyPacketsSinceOpen.load(std::memory_order_relaxed));
    }
    // Tier 2 (~15 s): kf==0 -> upstream silent/off-air; kf>=1 -> HW decoder stall.
    if (elapsed > 15000 && !starvationWarnedSustained.exchange(true, std::memory_order_relaxed)) {
        const size_t kf = keyPacketsSinceOpen.load(std::memory_order_relaxed);
        isyslog("vaapivideo/decoder: still no frame %llu ms after open (packets sent=%zu, "
                "keyframes=%zu) -- %s",
                static_cast<unsigned long long>(elapsed), sent, kf,
                kf == 0 ? "no keyframe on the wire (upstream silent / off-air?)"
                        : "keyframe arrived but decoder is not producing output (HW decoder stall)");
    }
}

auto cVaapiDecoder::FlushParser() -> void {
    // Still-picture / Goto(Still): the parser withholds the last AU until a start code follows.
    // NULL-input flush surfaces the single I-frame immediately.
    const cMutexLock parseLock(&parserMutex);
    DrainPendingParserAU();
}

auto cVaapiDecoder::DrainPendingParserAU() -> void {
    // NULL/0 input is the documented EOS-flush idiom for av_parser_parse2.
    // Callers: FlushParser (still-picture) and SetTrickSpeed (reverse isolated I-frames).
    // Caller holds parserMutex; packetMutex taken internally. codecCtx is read-only here.
    if (!codecCtx || !parserCtx) {
        return;
    }

    uint8_t *parsedData = nullptr; // NOLINT(misc-const-correctness) -- av_parser_parse2 out-param
    int parsedSize = 0;
    const int parsed = av_parser_parse2(parserCtx.get(), codecCtx.get(), &parsedData, &parsedSize, nullptr, 0,
                                        AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);

    if (parsed < 0 || parsedSize <= 0 || !parsedData) {
        return;
    }

    // Same FF keyframe gate as EnqueueData; reverse must not filter (PAFF key_frame=0).
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
        // Drop-oldest: the drained AU is the only one in trick mode and must not be silently lost.
        const bool isTrickMode = trickSpeed.load(std::memory_order_relaxed) != 0;
        const size_t maxDepth = isTrickMode ? DECODER_TRICK_QUEUE_DEPTH : DECODER_QUEUE_CAPACITY;
        if (packetQueue.size() >= maxDepth) {
            const std::unique_ptr<AVPacket, FreeAVPacket> dropped{packetQueue.front()};
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

    // Reset stopping/hasExited before Start() so the spawned thread observes a clean state.
    stopping.store(false, std::memory_order_release);
    hasExited.store(false, std::memory_order_release);
    ready.store(true, std::memory_order_release);
    Start();

    isyslog("vaapivideo/decoder: initialized (packet queue size=%zu)", DECODER_QUEUE_CAPACITY);
    return true;
}

[[nodiscard]] auto cVaapiDecoder::IsQueueEmpty() const -> bool {
    // VDR Poll(): true -> dvbplayer sends the next PES packet; false -> backpressure.
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

auto cVaapiDecoder::RequestFilterRebuild() -> void { videoRectDirty.store(true, std::memory_order_release); }

[[nodiscard]] auto cVaapiDecoder::OpenCodec(AVCodecID codecId) -> bool {
    // PES path: no extradata; SPS/PPS/VPS are inline, and FFmpeg infers profile on first frame.
    // Backend table defaults to 8-bit row; bit-depth update requires a full reopen via OpenCodecWithInfo.
    VideoStreamInfo info;
    info.codecId = codecId;
    return OpenCodecWithInfo(info);
}

[[nodiscard]] auto cVaapiDecoder::OpenCodecWithInfo(const VideoStreamInfo &info) -> bool {
    // codecMutex + parserMutex: reopens both contexts atomically; lock order is fixed
    // (codec -> parser) and matches Clear() / SetTrickSpeed().
    const cMutexLock decodeLock(&codecMutex);
    const cMutexLock parseLock(&parserMutex);

    if (!ready.load(std::memory_order_acquire) || stopping.load(std::memory_order_acquire)) [[unlikely]] {
        esyslog("vaapivideo/decoder: not initialized or shutting down");
        return false;
    }

    const AVCodec *decoder = avcodec_find_decoder(info.codecId);
    if (!decoder) [[unlikely]] {
        esyslog("vaapivideo/decoder: codec %d not found", static_cast<int>(info.codecId));
        return false;
    }

    // Table-driven (kVideoBackendTable in stream.h): returns the GpuCaps member pointer for this
    // codec/profile/depth, or nullptr for SW-only. Adding a codec = one row in stream.h + one probe in caps.cpp.
    bool useHwDecode = false;
    if (auto capFlag = SelectVideoBackendCap(info); capFlag != nullptr) {
        useHwDecode = vaapiContext->caps.*capFlag;
    }

    // VA driver advertising a profile is necessary but not sufficient; FFmpeg also needs
    // a VAAPI AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX entry. Require both.
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

    // Reuse only when codec ID, HW/SW choice, AND extradata all match.
    // Codec ID alone is insufficient: a same-codec stream change can switch bit-depth
    // (8-bit Main->10-bit Main10 flips the kVideoBackendTable row) or replace extradata (seek).
    bool extradataMatches = true;
    if (codecCtx) {
        if (codecCtx->extradata_size != info.extradataSize) {
            extradataMatches = false;
        } else if (info.extradataSize > 0) {
            extradataMatches =
                codecCtx->extradata != nullptr && info.extradata != nullptr &&
                std::memcmp(codecCtx->extradata, info.extradata, static_cast<size_t>(info.extradataSize)) == 0;
        }
    }

    if (codecCtx && currentCodecId == info.codecId && !forceCodecReopen &&
        ((codecCtx->hw_device_ctx != nullptr) == useHwDecode) && extradataMatches) {
        return true;
    }
    forceCodecReopen = false;

    // Full teardown required: filter graph holds a hw_frames_ctx ref tied to the old context.
    // hasLoggedFirstFrame reset here and ONLY here so the "first frame" line fires once per reopen.
    parserCtx.reset();
    {
        // vaDriverMutex: vaDestroyContext (VAAPI hwaccel) + VPP teardown make VA API calls.
        const cMutexLock vaLock(&display->GetVaDriverMutex());
        codecCtx.reset();
        filterChain.Reset();
    }
    currentCodecId = AV_CODEC_ID_NONE;
    hasLoggedFirstFrame.store(false, std::memory_order_relaxed);
    // Reset so the 3 s and 15 s starvation tiers fire at most once per codec open.
    starvationWarned.store(false, std::memory_order_relaxed);
    starvationWarnedSustained.store(false, std::memory_order_relaxed);
    packetsSinceOpen.store(0, std::memory_order_relaxed);
    keyPacketsSinceOpen.store(0, std::memory_order_relaxed);
    codecOpenTimeMs.store(cTimeMs::Now(), std::memory_order_relaxed);

    // PES path: NAL stream -> parser mandatory. Mediaplayer path: pre-demuxed AUs + extradata -> no parser.
    // info.extradata == nullptr is the canonical PES-path marker.
    parserCtx.reset();
    const bool needsParser = (info.extradata == nullptr) || (info.extradataSize <= 0);
    if (needsParser) {
        parserCtx.reset(av_parser_init(info.codecId));
        if (!parserCtx) [[unlikely]] {
            esyslog("vaapivideo/decoder: failed to create parser for %s", decoder->name);
            return false;
        }
    }

    std::unique_ptr<AVCodecContext, FreeAVCodecContext> decoderCtx{avcodec_alloc_context3(decoder)};
    if (!decoderCtx) [[unlikely]] {
        esyslog("vaapivideo/decoder: failed to allocate context for %s", decoder->name);
        return false;
    }

    // DVB streams are routinely marginally non-conforming; CAREFUL avoids hard failures on minor violations.
    decoderCtx->err_recognition = AV_EF_CAREFUL;
    decoderCtx->strict_std_compliance = FF_COMPLIANCE_UNOFFICIAL;

    // Mediaplayer path: container demuxers (mkv/mp4) store parameter sets in the track header,
    // not in every AU like DVB. FFmpeg requires them here before avcodec_open2.
    if (info.extradata && info.extradataSize > 0) {
        const size_t padded = static_cast<size_t>(info.extradataSize) + AV_INPUT_BUFFER_PADDING_SIZE;
        decoderCtx->extradata = static_cast<uint8_t *>(av_mallocz(padded));
        if (!decoderCtx->extradata) [[unlikely]] {
            esyslog("vaapivideo/decoder: extradata alloc failed (%d bytes)", info.extradataSize);
            return false;
        }
        std::memcpy(decoderCtx->extradata, info.extradata, static_cast<size_t>(info.extradataSize));
        decoderCtx->extradata_size = info.extradataSize;
    }

    if (useHwDecode) {
        // GPU parallelizes decode internally; FFmpeg slice-threading would only contend on the VA driver.
        decoderCtx->thread_count = 1;

        decoderCtx->hw_device_ctx = av_buffer_ref(vaapiContext->hwDeviceRef);
        if (!decoderCtx->hw_device_ctx) [[unlikely]] {
            esyslog("vaapivideo/decoder: failed to reference VAAPI device for %s", decoder->name);
            return false;
        }

        // Pin output to VAAPI surfaces. Without this FFmpeg may silently fall back to SW frames.
        decoderCtx->get_format = [](AVCodecContext *, const AVPixelFormat *formats) -> AVPixelFormat {
            for (const AVPixelFormat *fmt = formats; *fmt != AV_PIX_FMT_NONE; ++fmt) {
                if (*fmt == AV_PIX_FMT_VAAPI) {
                    return AV_PIX_FMT_VAAPI;
                }
            }
            return AV_PIX_FMT_NONE;
        };
    }
    // SW decode: hw_device_ctx left unset; filter chain inserts hwupload so VPP still runs on the GPU.

    if (const int ret = avcodec_open2(decoderCtx.get(), decoder, nullptr); ret < 0) [[unlikely]] {
        esyslog("vaapivideo/decoder: failed to open %s: %s", decoder->name, AvErr(ret).data());
        return false;
    }

    codecCtx = std::move(decoderCtx);
    currentCodecId = info.codecId;
    // codecCtx->profile is AV_PROFILE_UNKNOWN until FFmpeg parses the first SPS/VPS.
    // The authoritative value is logged in the first-frame block inside DecodeOnePacket.
    isyslog("vaapivideo/decoder: opened %s (%s%s)", decoder->name, useHwDecode ? "hardware" : "software",
            info.extradataSize > 0 ? ", extradata" : "");

    return true;
}

auto cVaapiDecoder::NotifyAudioChange() -> void {
    // Audio clock is NOPTS for ~100--500 ms while the new pipeline primes.
    // Freerun surfaces the next frame immediately; catch-up realigns once the clock re-anchors.
    freerunFrames.store(DECODER_SYNC_FREERUN_FRAMES, std::memory_order_relaxed);
    syncLogPending.store(true, std::memory_order_relaxed);
}

auto cVaapiDecoder::SetAudioProcessor(cAudioProcessor *audio) -> void {
    audioProcessor.store(audio, std::memory_order_release);
}

auto cVaapiDecoder::SetLiveMode(bool live) -> void { liveMode.store(live, std::memory_order_relaxed); }

auto cVaapiDecoder::RequestTrickExit() -> void { trickExitPending.store(true, std::memory_order_release); }

auto cVaapiDecoder::SetTrickSpeed(int speed, bool forward, bool fast) -> void {
    // speed=0: normal. fast: FF/REW (speed>=6->2x, >=3->4x, else 8x). !fast: slow at 1/speed hold.
    trickExitPending.store(false, std::memory_order_relaxed); // supersedes any pending deferred trick-exit

    // Notify the display thread that trick pacing is (about to be) active BEFORE the codec/filter
    // teardown below. The teardown can hold vaDriverMutex for several VSyncs, during which the
    // display would otherwise see (trickActive=false, pendingFrames=empty) and increment its
    // underrun counter. Setting trickActive up front closes that race; clearing it on speed==0
    // happens at the end of this function once the new pacing state is fully published.
    if (display && speed != 0) {
        display->SetTrickActive(true);
    }

    // VDR skips DeviceClear() on trick entry. FF/REW/slow-reverse are non-contiguous streams,
    // so stale partial NALs would corrupt the parser. Slow-forward stays contiguous; no flush.
    const bool needsFlush = (speed > 0) && (fast || !forward);
    dsyslog("vaapivideo/decoder: SetTrickSpeed speed=%d forward=%d fast=%d needsFlush=%d", speed, forward, fast,
            needsFlush);
    {
        // codecMutex + parserMutex: same atomicity guarantee as Clear() during the flush path
        // (codec flush + parser recreate must not race a concurrent EnqueueData parse).
        const cMutexLock decodeLock(&codecMutex);
        const cMutexLock parseLock(&parserMutex);

        if (needsFlush) {
            DrainQueue();
            {
                // vaDriverMutex: avcodec_flush_buffers + filterChain.Reset() make VA API calls
                // that race av_hwframe_map on the same VADisplay (iHD driver crash).
                const cMutexLock vaLock(&display->GetVaDriverMutex());
                if (codecCtx) {
                    avcodec_flush_buffers(codecCtx.get());
                }
                // flush_buffers may rebuild hw_frames_ctx; graph holds the old ref and must be torn down.
                filterChain.Reset();
                if (decodedFrame) {
                    av_frame_unref(decodedFrame.get());
                }
                if (filteredFrame) {
                    av_frame_unref(filteredFrame.get());
                }
            }
            // No parser flush API; recreate to discard stale partial NALs.
            // Null parserCtx = mediaplayer path; must not introduce one mid-stream.
            if (parserCtx && currentCodecId != AV_CODEC_ID_NONE) {
                parserCtx.reset(av_parser_init(currentCodecId));
            }
        }

        // Flags before trickSpeed release-store: reader sees consistent (mode, speed).
        isTrickFastForward.store(forward && fast, std::memory_order_relaxed);
        isTrickReverse.store(!forward, std::memory_order_relaxed);
        prevTrickPts.store(AV_NOPTS_VALUE, std::memory_order_relaxed);
    }

    // Fast: PTS-derived hold via multiplier. Slow: fixed hold = speed * DECODER_TRICK_HOLD_MS.
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
        // Same Clear-race guard as Clear(): epoch BEFORE NOPTS.
        clearEpoch.fetch_add(1, std::memory_order_release);
        lastPts.store(AV_NOPTS_VALUE, std::memory_order_release);

        syncLogPending.store(true, std::memory_order_relaxed);
    }

    trickSpeed.store(speed, std::memory_order_release);

    // speed==0 path: clear the display's trick flag once pacing state is fully published.
    // The speed!=0 case was set up front, before the teardown above.
    if (display && speed == 0) {
        display->SetTrickActive(false);
    }
}

auto cVaapiDecoder::Shutdown() -> void {
    // Idempotent: the first caller signals/cancels; later callers still wait and drain.
    // Postcondition: decode thread joined, packetQueue drained -- callers may then safely
    // tear down codec context, hw_frames_ctx, and VAAPI surfaces.
    const bool wasStopping = stopping.exchange(true, std::memory_order_acq_rel);
    // Close the public gate so concurrent OpenCodec()/EnqueueData() bail out.
    ready.store(false, std::memory_order_release);
    if (!wasStopping) {
        dsyslog("vaapivideo/decoder: shutting down");
    }

    if (!wasStopping) {
        {
            const cMutexLock lock(&packetMutex);
            packetCondition.Broadcast();
        }

        // cThread::Running() is unfenced; hasExited defaults to true (no thread
        // started), so Cancel only when Action() is actually running.
        if (!hasExited.load(std::memory_order_acquire)) {
            Cancel(3);
        }
    }

    // Cancel() gives a join but not a happens-before fence. Spin on hasExited for the
    // release/acquire edge with the thread's last stores. Second caller observes hasExited=true
    // (set by first caller's wait) and returns immediately.
    const cTimeMs timeout(SHUTDOWN_TIMEOUT_MS);
    while (!hasExited.load(std::memory_order_acquire) && !timeout.TimedOut()) {
        {
            const cMutexLock lock(&packetMutex);
            packetCondition.Broadcast();
        }
        cCondWait::SleepMs(10);
    }

    // Drain after thread exits: packets enqueued after the thread checked stopping would
    // leak otherwise (std::queue<AVPacket*> does not own its elements).
    DrainQueue();
}

// ============================================================================
// === THREAD ===
// ============================================================================

auto cVaapiDecoder::Action() -> void {
    isyslog("vaapivideo/decoder: thread started");

    std::vector<std::unique_ptr<VaapiFrame>> pendingFrames;
    uint64_t lastDrainMs{0};
    int trickEmptyDecodes{0}; ///< Consecutive reverse-trick packets that yielded no frame; arms force-drain.

    while (!stopping.load(std::memory_order_acquire)) {
        // Snapshot for this iteration. Drain loops and PublishLastPts() abort on mismatch with
        // clearEpoch so a Clear() / SetTrickSpeed(0) racing this iteration doesn't submit stale
        // frames or publish stale pts.
        iterationEpoch = clearEpoch.load(std::memory_order_acquire);

        // Consume a Clear() that already fired before we got scheduled. Without this, the
        // post-decode flush below would discard the first valid post-Clear frame -- Clear()
        // drains the packet queue and calls avcodec_flush_buffers(), so the next decoded frame
        // is genuinely new material that belongs in the jitter buffer.
        if (jitterFlushPending.exchange(false, std::memory_order_acquire)) {
            ApplyDeferredJitterFlush(lastDrainMs);
        }

        std::unique_ptr<AVPacket, FreeAVPacket> queuedPacket;

        // Wait clamped to [1, 18] ms: upper bound prevents 50 Hz VSync beating that drops 50p to ~33
        // fps; dynamic value sleeps just until the head jitterBuf frame is due (no halfFrame overshoot).
        {
            const cMutexLock lock(&packetMutex);
            if (packetQueue.empty() && !stopping.load(std::memory_order_acquire)) {
                int waitMs = 10; // jitterBuf empty: deep sleep, woken by next EnqueueData broadcast.
                if (!jitterBuf.empty()) {
                    auto *const ap = audioProcessor.load(std::memory_order_acquire);
                    const bool consumingDrop = pendingDrops > 0;
                    const bool canFreerun = freerunFrames.load(std::memory_order_relaxed) > 0;
                    if (consumingDrop || canFreerun || !ap) {
                        waitMs = 1;
                    } else {
                        const int64_t clock = ap->GetClock();
                        const int64_t headPts = jitterBuf.front()->pts;
                        if (clock == AV_NOPTS_VALUE || headPts == AV_NOPTS_VALUE) {
                            waitMs = 18;
                        } else {
                            const int64_t dueIn90k = headPts - clock - SyncLatency90k(ap);
                            const int64_t halfFrame =
                                (static_cast<int64_t>(outputFrameDurationMs) * PTS_TICKS_PER_MS) / 2;
                            const int64_t frameDur = static_cast<int64_t>(outputFrameDurationMs) * PTS_TICKS_PER_MS;
                            // Pre-fill wake: depth==0 means the next VSync would re-present. Wake
                            // one frame earlier so the drain (below) reaches the pre-submit window.
                            const int64_t wakeThreshold =
                                (display && display->PendingDepth() == 0) ? halfFrame + frameDur : halfFrame;
                            waitMs = (dueIn90k <= wakeThreshold)
                                         ? 1
                                         : std::clamp(static_cast<int>((dueIn90k - wakeThreshold) / PTS_TICKS_PER_MS),
                                                      1, 18);
                        }
                    }
                }
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

        // codecMutex scoped to decode only; released before SubmitFrame so Clear() / SetTrickSpeed()
        // on other threads don't stall behind VSync-blocked frame delivery.
        if (queuedPacket && !stopping.load(std::memory_order_acquire)) {
            const cMutexLock decodeLock(&codecMutex);
            if (codecCtx) {
                (void)DecodeOnePacket(queuedPacket.get(), pendingFrames);
            }
        }
        // Trick reverse: H.264 holds IDRs in the reorder buffer until later inputs push them
        // out. With reverse-ordered isolated IDRs the buffer never fills usefully, pendingFrames
        // stays empty, SubmitTrickFrame (the only nextTrickFrameDue updater) never runs, and
        // dvbplayer race-feeds past the recording boundary. Force-drain to surface the held
        // IDR; flush_buffers rebuilds hw_frames_ctx so the filter graph must reset too. Fallback
        // pace covers the no-frame case (corrupt packet) so VDR's gate still throttles.
        // Reverse-only: forward trick has a normal B-frame pipeline where 1 packet -> 0 frames
        // is routine, and tearing the filter graph down on every such packet thrashes VAAPI.
        // Require two consecutive empty decodes: PAFF interlaced reverse buffers field 1 (no output)
        // and produces a frame on field 2; force-draining after field 1 would discard the pair.
        const bool reverseTrick =
            trickSpeed.load(std::memory_order_relaxed) != 0 && isTrickReverse.load(std::memory_order_relaxed);
        if (queuedPacket && reverseTrick) {
            trickEmptyDecodes = pendingFrames.empty() ? trickEmptyDecodes + 1 : 0;
        } else {
            trickEmptyDecodes = 0;
        }
        if (queuedPacket && reverseTrick && pendingFrames.empty() && trickEmptyDecodes >= 2) {
            const cMutexLock decodeLock(&codecMutex);
            const cMutexLock vaLock(&display->GetVaDriverMutex());
            if (codecCtx) {
                DrainCodecAtEos(pendingFrames);
                if (filterChain.IsBuilt()) {
                    if (filterChain.SendFrame(nullptr) >= 0) {
                        while (true) {
                            av_frame_unref(filteredFrame.get());
                            if (filterChain.ReceiveFrame(filteredFrame.get()) < 0) {
                                break;
                            }
                            if (auto vaapiFrame = CreateVaapiFrame(filteredFrame.get())) {
                                pendingFrames.push_back(std::move(vaapiFrame));
                            }
                        }
                    }
                    filterChain.Reset();
                }
            }
            if (pendingFrames.empty()) {
                const uint64_t holdMs = trickHoldMs.load(std::memory_order_relaxed);
                nextTrickFrameDue.store(cTimeMs::Now() + holdMs, std::memory_order_relaxed);
            }
            trickEmptyDecodes = 0; // codec was just flushed; restart the consecutive-empty counter
        }

        // --- Still-picture codec drain ---
        // The decode thread may consume the packet before RequestCodecDrain sets the flag;
        // checking here (after the decode step) catches both orderings.
        if (codecDrainPending.exchange(false, std::memory_order_acquire)) {
            const cMutexLock decodeLock(&codecMutex);
            DrainCodecAtEos(pendingFrames);

            // Temporal filters (bwdif) hold frames internally; EOS-flush surfaces them.
            if (filterChain.IsBuilt()) {
                const cMutexLock vaLock(&display->GetVaDriverMutex());
                if (filterChain.SendFrame(nullptr) >= 0) {
                    while (true) {
                        av_frame_unref(filteredFrame.get());
                        if (filterChain.ReceiveFrame(filteredFrame.get()) < 0) {
                            break;
                        }
                        if (auto vaapiFrame = CreateVaapiFrame(filteredFrame.get())) {
                            pendingFrames.push_back(std::move(vaapiFrame));
                        }
                    }
                }
                filterChain.Reset(); // graph in EOF state after drain; rebuild on next packet.
            }

            // Exit still mode: next packet rebuilds the graph with full temporal filters.
            stillPictureMode.store(false, std::memory_order_release);
        }

        // --- Frame delivery ---
        // Catches a Clear() that fired BETWEEN the top-of-iteration consume and now -- iow
        // mid-decode, while we held codecMutex. pendingFrames in that case carries pre-Clear
        // PTS and must be discarded along with the jitter buffer.
        if (jitterFlushPending.exchange(false, std::memory_order_acquire)) {
            ApplyDeferredJitterFlush(lastDrainMs);
            pendingFrames.clear();
        }

        // Unified drain for live and replay: push to jitterBuf, pop when due. SkipStaleJitterFrames
        // and SyncAndSubmitFrame's pre-warmup catch-up handle pre-roll/trick-exit backlogs; liveMode
        // still selects hard-ahead policy inside SyncAndSubmitFrame.
        auto *const ap = audioProcessor.load(std::memory_order_acquire);
        for (auto &frame : pendingFrames) {
            jitterBuf.push_back(std::move(frame));
        }

        // Runaway guard: SkipStaleJitterFrames only drops BEHIND-clock heads, so a stuck-but-valid
        // clock with PTS marching ahead would grow jitterBuf forever. Drop oldest to bound memory.
        if (jitterBuf.size() > DECODER_JITTERBUF_HARD_CAP) [[unlikely]] {
            const size_t dropCount = jitterBuf.size() - DECODER_JITTERBUF_HARD_CAP;
            for (size_t i = 0; i < dropCount; ++i) {
                jitterBuf.pop_front();
            }
            syncDropSinceLog += static_cast<int>(dropCount);
            dsyslog("vaapivideo/decoder: jitterBuf overflow -- dropped %zu frame(s), cap=%zu", dropCount,
                    DECODER_JITTERBUF_HARD_CAP);
        }

        // Bulk-drop catastrophically stale heads (post-seek, long decode stall, trick-exit backlog).
        if (ap && !jitterBuf.empty()) {
            SkipStaleJitterFrames(ap);
        }

        // Freerun and pendingDrops (soft-behind burst) bypass the due-gate.
        while (!jitterBuf.empty() && !stopping.load(std::memory_order_relaxed)) {
            // Clear() / SetTrickSpeed(0) raced this iteration: abandon the stale jitterBuf to
            // avoid scanning out pre-Clear frames or burning the new freerun budget on them.
            // The next iteration's jitterFlushPending consumption empties jitterBuf.
            if (clearEpoch.load(std::memory_order_acquire) != iterationEpoch) [[unlikely]] {
                break;
            }
            const bool consumingDrop = pendingDrops > 0;
            const bool canFreerun = freerunFrames.load(std::memory_order_relaxed) > 0;

            if (!consumingDrop && !canFreerun && ap) {
                // Normal path: with a valid clock, hold until head PTS is due.
                // No clock / no audio processor falls through to SyncAndSubmitFrame()'s
                // freerun-on-no-clock path so video doesn't freeze when the audio clock
                // is missing (track switch, ALSA reset, video-only stream).
                const int64_t clock = ap->GetClock();
                if (clock != AV_NOPTS_VALUE) {
                    const int64_t headPts = jitterBuf.front()->pts;
                    if (headPts != AV_NOPTS_VALUE) {
                        const int64_t latency = SyncLatency90k(ap);
                        const int64_t dueIn = headPts - clock - latency;
                        const int64_t halfFrame = (static_cast<int64_t>(outputFrameDurationMs) * PTS_TICKS_PER_MS) / 2;
                        if (dueIn > halfFrame) {
                            // Magnitude guard: head sitting >DECODER_DRAIN_FUTURE_MAX_MS ahead of
                            // audio_clock is a real PTS discontinuity (post-ATTA anchor mismatch,
                            // broadcast PCR break). Drop immediately; catch-up handles the residual.
                            if (dueIn > DECODER_DRAIN_FUTURE_MAX_MS * PTS_TICKS_PER_MS) {
                                dsyslog(
                                    "vaapivideo/decoder: head too far in future (dueIn=%+lldms buf=%zu) -- dropping",
                                    static_cast<long long>(dueIn / PTS_TICKS_PER_MS), jitterBuf.size());
                                jitterBuf.pop_front();
                                continue; // re-check new head
                            }
                            // Pre-fill one frame ahead when display queue is empty: keeps depth
                            // at 1-2 instead of 0-1 and absorbs audio-clock vs VSync phase drift
                            // that would otherwise tick the underrun counter on a healthy stream.
                            // Bounded to one frameDur to never run further ahead.
                            const int64_t frameDur = static_cast<int64_t>(outputFrameDurationMs) * PTS_TICKS_PER_MS;
                            const bool prefill =
                                display && display->PendingDepth() == 0 && dueIn <= halfFrame + frameDur;
                            if (!prefill) {
                                break;
                            }
                        }
                    }
                }
            }

            // Trick play paces frames at the trick hold (60-2000 ms), well above the 2*frameDur
            // miss threshold. Don't count those as drain misses and reset lastDrainMs so the first
            // post-trick drain isn't flagged either. A sync-correction sleep (hard-ahead / soft-ahead
            // / WaitForAudioCatchUp) inside the previous SyncAndSubmitFrame causes the same big gap,
            // so consume sleptInLastSubmit to skip exactly one miss.
            const bool inTrick = trickSpeed.load(std::memory_order_relaxed) != 0;
            const bool consumedSleep = std::exchange(sleptInLastSubmit, false);
            const uint64_t nowMs = cTimeMs::Now();
            if (!inTrick && !consumedSleep && lastDrainMs > 0 &&
                static_cast<int>(nowMs - lastDrainMs) > outputFrameDurationMs * 2) {
                ++drainMissCount;
            }
            lastDrainMs = inTrick ? 0 : nowMs;

            auto drainFrame = std::move(jitterBuf.front());
            jitterBuf.pop_front();
            (void)SyncAndSubmitFrame(std::move(drainFrame));
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

    // av_frame_clone() increments the VAAPI surface refcount; surface stays alive until VaapiFrame is destroyed.
    vaapiFrame->avFrame = av_frame_clone(src);
    if (!vaapiFrame->avFrame) [[unlikely]] {
        return nullptr;
    }

    // FFmpeg VAAPI ABI: data[3] encodes the VASurfaceID directly as a uintptr_t, not a pointer.
    // Never dereference it; surface lifetime is governed by the AVFrame refcount.
    vaapiFrame->vaSurfaceId =
        static_cast<VASurfaceID>(reinterpret_cast<uintptr_t>( // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
            vaapiFrame->avFrame->data[3]));
    vaapiFrame->pts = src->pts;

    return vaapiFrame;
}

auto cVaapiDecoder::FilterAndAppendDecodedFrame(std::vector<std::unique_ptr<VaapiFrame>> &outFrames) -> void {
    // SD DVB streams routinely omit color_description; BT.470BG (PAL) prevents desaturation by scale_vaapi.
    if (decodedFrame->colorspace == AVCOL_SPC_UNSPECIFIED && decodedFrame->height <= 576) {
        decodedFrame->colorspace = AVCOL_SPC_BT470BG;
    }

    const int64_t sourcePts = decodedFrame->pts;
    const size_t prevOutCount = outFrames.size();
    {
        const cMutexLock vaLock(&display->GetVaDriverMutex());
        if (!filterChain.IsBuilt()) {
            (void)InitFilterGraph(decodedFrame.get());
        }

        if (filterChain.IsBuilt() && filterChain.SendFrame(decodedFrame.get()) >= 0) {
            while (true) {
                av_frame_unref(filteredFrame.get());
                if (filterChain.ReceiveFrame(filteredFrame.get()) < 0) {
                    break;
                }
                if (auto vaapiFrame = CreateVaapiFrame(filteredFrame.get())) {
                    outFrames.push_back(std::move(vaapiFrame));
                }
            }
        } else if (!filterChain.IsBuilt()) {
            // Filter graph failed to build (rare: GPU OOM, driver bug); pass raw frame through unscaled.
            if (auto vaapiFrame = CreateVaapiFrame(decodedFrame.get())) {
                outFrames.push_back(std::move(vaapiFrame));
            }
        }
    }

    // bwdif rate=field doubles the frame count; stamp extra output fields at sourcePts + i*frameDur.
    const size_t newOutCount = outFrames.size() - prevOutCount;
    for (size_t i = 0; i < newOutCount; ++i) {
        outFrames.at(prevOutCount + i)->pts =
            (sourcePts != AV_NOPTS_VALUE && i > 0)
                ? sourcePts + (static_cast<int64_t>(outputFrameDurationMs) * PTS_TICKS_PER_MS * static_cast<int64_t>(i))
                : sourcePts;
    }
}

auto cVaapiDecoder::DrainCodecAtEos(std::vector<std::unique_ptr<VaapiFrame>> &outFrames) -> void {
    if (!codecCtx) {
        return;
    }
    // NULL-packet EOS idiom. avcodec_flush_buffers re-arms the codec so the next packet
    // doesn't receive a spurious AVERROR_EOF.
    // EAGAIN from send_packet(NULL) means the codec has output to give first; drain receive,
    // then retry send. Loop terminates on send accept (>=0 / AVERROR_EOF) or hard send error.
    while (true) {
        const int sendRet = avcodec_send_packet(codecCtx.get(), nullptr);
        if (sendRet == AVERROR(EAGAIN)) {
            bool drainedAny = false;
            while (true) {
                av_frame_unref(decodedFrame.get());
                const int recvRet = avcodec_receive_frame(codecCtx.get(), decodedFrame.get());
                if (recvRet == AVERROR(EAGAIN) || recvRet == AVERROR_EOF) {
                    break;
                }
                if (recvRet < 0) [[unlikely]] {
                    dsyslog("vaapivideo/decoder: receive_frame during drain failed: %s", AvErr(recvRet).data());
                    break;
                }
                drainedAny = true;
                FilterAndAppendDecodedFrame(outFrames);
            }
            if (!drainedAny) {
                // EAGAIN with nothing to drain would loop forever; codec is wedged.
                break;
            }
            continue;
        }
        if (sendRet < 0 && sendRet != AVERROR_EOF) [[unlikely]] {
            dsyslog("vaapivideo/decoder: send_packet(NULL) during drain failed: %s", AvErr(sendRet).data());
            break;
        }
        // sendRet >= 0 or AVERROR_EOF: drain remaining frames to AVERROR_EOF.
        while (true) {
            av_frame_unref(decodedFrame.get());
            const int recvRet = avcodec_receive_frame(codecCtx.get(), decodedFrame.get());
            if (recvRet == AVERROR(EAGAIN) || recvRet == AVERROR_EOF) {
                break;
            }
            if (recvRet < 0) [[unlikely]] {
                dsyslog("vaapivideo/decoder: receive_frame during drain failed: %s", AvErr(recvRet).data());
                break;
            }
            FilterAndAppendDecodedFrame(outFrames);
        }
        break;
    }
    {
        // vaDriverMutex: avcodec_flush_buffers touches the VA driver (VAAPI hwaccel reset);
        // without it the display thread's av_hwframe_map races on iHD.
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

    // EAGAIN = VAAPI surface pool full; drain available frames then retry. Normally one iteration.
    while (!packetSent) {
        const int ret = avcodec_send_packet(codecCtx.get(), pkt);
        if (ret == AVERROR(EAGAIN)) {
            // Fall through to drain loop; next outer iteration retries the send.
        } else {
            if (ret < 0 && ret != AVERROR_EOF) [[unlikely]] {
                // Hard failure (corrupt HEVC NAL etc.): flush + reset graph so the next IDR recovers cleanly.
                // vaDriverMutex: avcodec_flush_buffers touches the VA driver; races av_hwframe_map on iHD.
                dsyslog("vaapivideo/decoder: send_packet failed: %s -- flushing for recovery", AvErr(ret).data());
                const cMutexLock vaLock(&display->GetVaDriverMutex());
                avcodec_flush_buffers(codecCtx.get());
                filterChain.Reset();
                return anyFrameDecoded;
            }
            packetSent = true; // success or EOF; don't retry
            NoteStarvationTick(pkt);
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
                const char *swFmtName = av_get_pix_fmt_name(ResolveSwPixFmt(decodedFrame.get()));
                const HdrStreamInfo hdr = ExtractHdrInfo(decodedFrame.get());
                // codecCtx->profile is populated by FFmpeg during SPS/VPS parsing, which
                // is guaranteed to have happened before the first frame surfaces. At
                // OpenCodec() time this field is still AV_PROFILE_UNKNOWN and logging it
                // there is meaningless -- the authoritative value lands here.
                const AVCodec *openedDecoder = codecCtx ? codecCtx->codec : nullptr;
                const char *profileName =
                    openedDecoder ? av_get_profile_name(openedDecoder, codecCtx->profile) : nullptr;
                isyslog("vaapivideo/decoder: first frame %dx%d %s (sw=%s, profile=%s)%s", decodedFrame->width,
                        decodedFrame->height, fmtName ? fmtName : "unknown", swFmtName ? swFmtName : "unknown",
                        profileName ? profileName : "unknown",
                        (decodedFrame->flags & AV_FRAME_FLAG_INTERLACED) ? " interlaced" : "");
                isyslog("vaapivideo/decoder: colorimetry -- primaries=%d trc=%d space=%d range=%d kind=%s",
                        static_cast<int>(decodedFrame->color_primaries), static_cast<int>(decodedFrame->color_trc),
                        static_cast<int>(decodedFrame->colorspace), static_cast<int>(decodedFrame->color_range),
                        StreamHdrKindName(hdr.kind));
                if (hdr.hasMasteringDisplay) {
                    // Luminance fields are in 1/10000 cd/m^2 per AVMasteringDisplayMetadata.
                    const double maxLuma = hdr.mastering.has_luminance ? av_q2d(hdr.mastering.max_luminance) : 0.0;
                    const double minLuma = hdr.mastering.has_luminance ? av_q2d(hdr.mastering.min_luminance) : 0.0;
                    isyslog("vaapivideo/decoder: mastering display -- max=%.0f cd/m^2 min=%.4f cd/m^2", maxLuma,
                            minLuma);
                }
                if (hdr.hasContentLight) {
                    isyslog("vaapivideo/decoder: content light -- MaxCLL=%u MaxFALL=%u", hdr.contentLight.MaxCLL,
                            hdr.contentLight.MaxFALL);
                }
            }

            // SD DVB streams omit color_description; default to BT.470BG (PAL) so
            // scale_vaapi's BT.709 conversion doesn't desaturate the output.
            if (decodedFrame->colorspace == AVCOL_SPC_UNSPECIFIED && decodedFrame->height <= 576) {
                decodedFrame->colorspace = AVCOL_SPC_BT470BG;
            }

            // Catch-up fast path: skip the VPP chain for frames SyncAndSubmitFrame would drop
            // anyway. The filter (MC deinterlace + denoise + scale + sharpness) dominates
            // per-frame cost on 4K; running it only to discard the output pins recovery near
            // 1x wall speed. Bypassing it lets the decoder drain a seek backlog many times
            // faster. Trick/still paths use their own pacing and must not be shortcut.
            if (catchingUp && trickSpeed.load(std::memory_order_relaxed) == 0 &&
                !stillPictureMode.load(std::memory_order_relaxed) && decodedFrame->pts != AV_NOPTS_VALUE) {
                if (auto *const ap = audioProcessor.load(std::memory_order_acquire)) {
                    const int64_t clock = ap->GetClock();
                    if (clock != AV_NOPTS_VALUE &&
                        decodedFrame->pts - clock - SyncLatency90k(ap) < -DECODER_SYNC_CORRIDOR_90K) {
                        PublishLastPts(decodedFrame->pts);
                        ++catchUpDrops;
                        ++syncDropSinceLog;
                        continue;
                    }
                }
            }

            // vaDriverMutex serializes VAAPI access against the display thread's DRM PRIME export.
            // Filter graph is built lazily on first frame and after each Clear() or ScaleVideo() change.
            const int64_t sourcePts = decodedFrame->pts;
            const size_t prevOutCount = outFrames.size();
            {
                const cMutexLock vaLock(&display->GetVaDriverMutex());

                if (videoRectDirty.exchange(false, std::memory_order_acq_rel)) {
                    dsyslog("vaapivideo/decoder: video rect changed, rebuilding filter graph");
                    jitterBuf.clear(); // drop old-sized frames before they reach PresentBuffer
                    filterChain.Reset();
                }
                if (!filterChain.IsBuilt()) {
                    (void)InitFilterGraph(decodedFrame.get());
                }

                if (filterChain.IsBuilt()) {
                    if (filterChain.SendFrame(decodedFrame.get()) < 0) [[unlikely]] {
                        filterChain.Reset();
                        continue;
                    }

                    while (true) {
                        av_frame_unref(filteredFrame.get());
                        const int filterRet = filterChain.ReceiveFrame(filteredFrame.get());
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

            // bwdif rate=field doubles frame count; assign monotonic PTS to extra fields (source + i*frameDur).
            // Only stamp [prevOutCount, end) to avoid re-stamping outputs from earlier receive iterations.
            const size_t newOutCount = outFrames.size() - prevOutCount;
            for (size_t i = 0; i < newOutCount; ++i) {
                outFrames.at(prevOutCount + i)->pts = (sourcePts != AV_NOPTS_VALUE && i > 0)
                                                          ? sourcePts + (static_cast<int64_t>(outputFrameDurationMs) *
                                                                         PTS_TICKS_PER_MS * static_cast<int64_t>(i))
                                                          : sourcePts;
            }

            // Trick mode: bwdif rate=field's first output blends temporally distant fields
            // (visible green ghosting on FF/REW); drop it and keep only the clean second field.
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

    // Reverse: drain DPB after each field pair so non-IDR I-frames (broadcast H.264) surface
    // immediately instead of accumulating in the reorder buffer.
    // PAFF: trigger only on second field (non-key) so the pair is complete before drain.
    if (trickSpeed.load(std::memory_order_relaxed) != 0 && isTrickReverse.load(std::memory_order_relaxed) &&
        !(pkt->flags & AV_PKT_FLAG_KEY)) {
        const size_t preDrainSize = outFrames.size();
        DrainCodecAtEos(outFrames);
        if (outFrames.size() > preDrainSize) {
            anyFrameDecoded = true;
        }
    }

    // Do NOT reset the filter graph here: bob is spatial-only (no temporal state), and
    // destroying it invalidates in-flight VPP surfaces (EIO on DRM PRIME export).
    // codecDrainPending is handled in Action() so the drain runs even when no packet was queued.

    return anyFrameDecoded;
}

[[nodiscard]] auto cVaapiDecoder::InitFilterGraph(AVFrame *firstFrame) -> bool {
    // Fills BuildParams from decoder/display/config/caps and delegates to cVideoFilterChain::Build().
    // The filter chain itself has no decoder/display dependencies, so a future mediaplayer
    // orchestrator can reuse it by supplying different BuildParams.
    if (!display) [[unlikely]] {
        esyslog("vaapivideo/decoder: no display for filter setup");
        return false;
    }
    if (!codecCtx) [[unlikely]] {
        esyslog("vaapivideo/decoder: no codec context for filter setup");
        return false;
    }

    // Classify HDR and stage the passthrough decision on the display BEFORE building the chain
    // so the next DRM atomic commit carries the correct HDR metadata (or clears it for SDR).
    const HdrStreamInfo hdrInfo = ExtractHdrInfo(firstFrame);
    const bool hdrPassthrough = ShouldUseHdrPassthrough(hdrInfo);
    display->SetHdrOutputState(hdrPassthrough ? hdrInfo : HdrStreamInfo{});

    cVideoFilterChain::BuildParams params;
    params.codecId = currentCodecId;
    params.fpsNum = codecCtx->framerate.num;
    params.fpsDen = codecCtx->framerate.den;
    params.hwFramesCtx = codecCtx->hw_frames_ctx;
    params.hwDeviceRef = vaapiContext->hwDeviceRef;
    params.outputWidth = display->GetVideoRectWidth();
    params.outputHeight = display->GetVideoRectHeight();
    params.outputRefreshHz = display->GetOutputRefreshRate();
    params.hdrPassthrough = hdrPassthrough;
    params.hdrInfo = hdrInfo;
    params.hasDenoise = vaapiContext->caps.vppDenoise;
    params.hasSharpness = vaapiContext->caps.vppSharpness;
    params.deinterlaceMode = vaapiContext->caps.deinterlaceMode;
    params.trickMode = trickSpeed.load(std::memory_order_relaxed) != 0;
    params.stillPicture = stillPictureMode.load(std::memory_order_relaxed);

    if (!filterChain.Build(firstFrame, params)) {
        return false;
    }

    outputFrameDurationMs = filterChain.GetOutputFrameDurationMs(); // drives pacing in SyncAndSubmitFrame.
    return true;
}

[[nodiscard]] auto cVaapiDecoder::ShouldUseHdrPassthrough(const HdrStreamInfo &info) const noexcept -> bool {
    if (info.kind == StreamHdrKind::Sdr) {
        return false;
    }
    const HdrMode userMode = vaapiConfig.hdrMode.load(std::memory_order_relaxed);
    if (userMode == HdrMode::Off) {
        return false;
    }
    // vppP010 is the GPU-side minimum: SW HEVC Main10 decode uploads via `format=p010le; hwupload`.
    if (!vaapiContext->caps.vppP010 || !display) {
        return false;
    }
    // On: skip EDID gate, but still refuse if CanDriveHdrPlane() would produce black (partial HDR signals).
    // Auto: additionally checks EDID EOTF for the specific kind (PQ/HLG).
    return (userMode == HdrMode::On) ? display->CanDriveHdrPlane() : display->SupportsHdrPassthrough(info.kind);
}

[[nodiscard]] auto cVaapiDecoder::SubmitTrickFrame(std::unique_ptr<VaapiFrame> frame) -> bool {
    const int64_t pts = frame->pts;
    const int64_t prevPts = prevTrickPts.load(std::memory_order_relaxed);

    // Reverse: GOPs arrive backward; frames within a GOP are in decode (ascending PTS) order.
    // Show only the first (lowest-PTS) frame per GOP; skip the rest.
    if (isTrickReverse.load(std::memory_order_relaxed) && pts != AV_NOPTS_VALUE && prevPts != AV_NOPTS_VALUE &&
        pts > prevPts) {
        return true;
    }

    // Deinterlaced field pairs share source PTS; pace once per source frame, pass both fields.
    if (pts != prevPts) {
        prevTrickPts.store(pts, std::memory_order_relaxed);
        if (pts != AV_NOPTS_VALUE) {
            PublishLastPts(pts);
        }

        // Block until pacing deadline, then arm the next one.
        const uint64_t due = nextTrickFrameDue.load(std::memory_order_relaxed);
        while (cTimeMs::Now() < due && !stopping.load(std::memory_order_relaxed) &&
               trickSpeed.load(std::memory_order_relaxed) != 0) {
            cCondWait::SleepMs(10);
        }

        // Fast: hold = |ptsDelta| / PTS_TICKS_PER_MS / mult, clamped to [10, 2000] ms.
        // Slow: precomputed trickHoldMs = speed * DECODER_TRICK_HOLD_MS.
        const uint64_t mult = trickMultiplier.load(std::memory_order_relaxed);
        if (mult > 0 && pts != AV_NOPTS_VALUE && prevPts != AV_NOPTS_VALUE) {
            const auto ptsDelta = static_cast<uint64_t>(std::abs(pts - prevPts));
            const uint64_t holdMs =
                std::clamp(ptsDelta / (static_cast<uint64_t>(PTS_TICKS_PER_MS) * mult), uint64_t{10}, uint64_t{2000});
            nextTrickFrameDue.store(cTimeMs::Now() + holdMs, std::memory_order_relaxed);
        } else {
            nextTrickFrameDue.store(cTimeMs::Now() + trickHoldMs.load(std::memory_order_relaxed),
                                    std::memory_order_relaxed);
        }
    }

    // Clear-race guard: pacing wait above may have been raced by SetTrickSpeed(0) / Clear().
    if (clearEpoch.load(std::memory_order_acquire) != iterationEpoch) [[unlikely]] {
        return true;
    }
    return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
}

auto cVaapiDecoder::WaitForAudioCatchUp(cAudioProcessor *ap, int64_t pts, int64_t latency, int64_t delta) -> void {
    dsyslog("vaapivideo/decoder: sync ahead d=%+lldms -- waiting for audio",
            static_cast<long long>(delta / PTS_TICKS_PER_MS));

    // Cap = delta + 1 s headroom, max 5 s: prevents a dead audio path from blocking indefinitely.
    const int64_t maxWaitMs = std::min<int64_t>((delta / PTS_TICKS_PER_MS) + 1000, 5000LL);
    const cTimeMs deadline(static_cast<int>(maxWaitMs));

    // Notify the display thread of the deliberate pause so its underrun detector doesn't
    // count the re-presents during this loop.
    if (display) {
        display->SetSyncSleeping(true);
    }
    while (!deadline.TimedOut() && !stopping.load(std::memory_order_relaxed)) {
        // A Clear() / channel-switch arms freerunFrames; bail so the new clock domain starts fresh.
        if (freerunFrames.load(std::memory_order_relaxed) > 0) {
            break;
        }
        const int64_t freshClock = ap->GetClock();
        if (freshClock == AV_NOPTS_VALUE || (pts - freshClock - latency) <= 0) {
            break;
        }
        cCondWait::SleepMs(10);
    }
    if (display) {
        display->SetSyncSleeping(false);
    }

    // Suppress the next drain-loop miss check: the long gap is from the deliberate wait, not
    // upstream starvation.
    sleptInLastSubmit = true;
    ResetSmoothedDelta();
    syncCooldown.Set(DECODER_SYNC_COOLDOWN_MS);
    syncLogPending.store(true, std::memory_order_relaxed);
}

auto cVaapiDecoder::PublishLastPts(int64_t pts) noexcept -> void {
    // Decode-thread chokepoint for lastPts. iterationEpoch (Action() iter top) compared against
    // clearEpoch (bumped by Clear() / SetTrickSpeed(0)) gates the publish; recheck-undo collapses
    // the ns-scale window where Clear() fires between check and store. Without it a stale
    // pre-Clear PTS would clobber Clear()'s NOPTS reset and surface via cVaapiDevice::GetSTC().
    if (clearEpoch.load(std::memory_order_acquire) != iterationEpoch) [[unlikely]] {
        return;
    }
    lastPts.store(pts, std::memory_order_release);
    if (clearEpoch.load(std::memory_order_acquire) != iterationEpoch) [[unlikely]] {
        lastPts.store(AV_NOPTS_VALUE, std::memory_order_release);
    }
}

[[nodiscard]] auto cVaapiDecoder::SyncLatency90k(const cAudioProcessor *ap) const noexcept -> int64_t {
    // User knob (PCM or passthrough) + 1-frame pipeline constant.
    // The 1-frame tail models the dominant scanout delay (commit + page flip) for an empty
    // prerender cache, which is the throughput-bound regime where the EMA used to settle at
    // -2*frameDur with the older 2-frame model. With 1*frameDur the throughput-bound baseline
    // sits at -1*frameDur (~-20 ms @ 50 fps), well inside CORRIDOR (50 ms), so 10-15 ms/s
    // crystal/pipeline drift takes ~3x longer to breach. Gate-bound (live TV, replay backlog)
    // is unaffected: the gate releases at dueIn <= halfFrame, so the EMA still settles at
    // +halfFrame regardless of latency. Operator knob still shifts the bias on top.
    const int latencyMs = (ap && ap->IsPassthrough()) ? vaapiConfig.passthroughLatency.load(std::memory_order_relaxed)
                                                      : vaapiConfig.pcmLatency.load(std::memory_order_relaxed);
    return (static_cast<int64_t>(latencyMs) + static_cast<int64_t>(outputFrameDurationMs)) * PTS_TICKS_PER_MS;
}

auto cVaapiDecoder::ResetSmoothedDelta() noexcept -> void {
    smoothedDeltaValid = false;
    smoothedDelta90k = 0;
    emaResidual90k = 0;
    warmupSampleCount = 0;
    warmupSampleSum90k = 0;
    rawDeltaSumSinceLog90k = 0;
    rawDeltaCountSinceLog = 0;
    hardAheadDebounce = 0;
    hardBehindDebounce = 0;
    catchingUp = false;
    catchUpDrops = 0;
}

auto cVaapiDecoder::ApplyDeferredJitterFlush(uint64_t &lastDrainMs) noexcept -> void {
    ResetSmoothedDelta();
    jitterBuf.clear();
    pendingDrops = 0;
    lastDrainMs = 0;
}

[[nodiscard]] auto cVaapiDecoder::SubmitIfCurrent(std::unique_ptr<VaapiFrame> frame) -> bool {
    // Clear-race guard for paths that sleep before submit: a Clear() during the sleep makes
    // the frame's PTS belong to the old epoch. Drop silently (returning true so callers don't
    // count it as a submit failure).
    if (clearEpoch.load(std::memory_order_acquire) != iterationEpoch) [[unlikely]] {
        return true;
    }
    return display->SubmitFrame(std::move(frame), DECODER_SUBMIT_TIMEOUT_MS);
}

auto cVaapiDecoder::UpdateSmoothedDelta(int64_t rawDelta90k) noexcept -> void {
    // Phase 1: N-sample mean seeds the EMA. Cuts the oscillation bias from deinterlaced 50p
    //          where alternating field timestamps skew early samples.
    // Phase 2: residual-accumulator EMA, alpha=1/EMA_SAMPLES.
    // INVARIANT: call exactly once per output frame; a second call site changes the reaction time.
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
    // Residual accumulator: avoids integer truncation stall when |diff| < EMA_SAMPLES ticks.
    const int64_t diff = rawDelta90k - smoothedDelta90k;
    emaResidual90k += diff;
    const int64_t step = emaResidual90k / DECODER_SYNC_EMA_SAMPLES;
    smoothedDelta90k += step;
    emaResidual90k -= step * DECODER_SYNC_EMA_SAMPLES;
}

auto cVaapiDecoder::LogSyncStats(int64_t rawDelta90k, int64_t latency90k, const cAudioProcessor *ap) -> void {
    // Suppress during EMA warmup. syncLogPending must be checked AFTER the warmup guard so
    // a Clear()-armed request survives warmup and fires immediately on the first post-warmup frame.
    if (!smoothedDeltaValid) {
        return;
    }
    if (!(syncLogPending.exchange(false, std::memory_order_relaxed) || nextSyncLog.TimedOut())) {
        return;
    }
    // d= interval mean (not point sample). lat= SyncLatency90k (frameDur + user knob).
    // If bumping the knob leaves lat= unchanged, the stream is using the other path (PCM vs PT).
    const int64_t meanRawDelta90k =
        (rawDeltaCountSinceLog > 0) ? (rawDeltaSumSinceLog90k / rawDeltaCountSinceLog) : rawDelta90k;
    const auto meanTenths = static_cast<long long>(meanRawDelta90k * 10 / PTS_TICKS_PER_MS);
    const auto avgTenths = static_cast<long long>(smoothedDelta90k * 10 / PTS_TICKS_PER_MS);
    dsyslog("vaapivideo/decoder: sync d=%+lld.%01lldms avg=%+lld.%01lldms lat=%lldms buf=%zu aq=%zu miss=%d "
            "drop=%d skip=%d",
            meanTenths / 10, std::abs(meanTenths % 10), avgTenths / 10, std::abs(avgTenths % 10),
            static_cast<long long>(latency90k / PTS_TICKS_PER_MS), jitterBuf.size(), ap->GetQueueSize(), drainMissCount,
            syncDropSinceLog, syncSkipSinceLog);
    rawDeltaSumSinceLog90k = 0;
    rawDeltaCountSinceLog = 0;
    drainMissCount = 0;
    syncDropSinceLog = 0;
    syncSkipSinceLog = 0;
    nextSyncLog.Set(DECODER_SYNC_LOG_INTERVAL_MS);
}

auto cVaapiDecoder::SkipStaleJitterFrames(cAudioProcessor *ap) -> void {
    // Bulk-drop heads more than HARD_THRESHOLD behind the clock; faster than SyncAndSubmitFrame's
    // one-by-one path after a decode stall or USB audio re-sync. Always keeps >= 1 frame.
    const int64_t latency = SyncLatency90k(ap);
    int dropped = 0;
    int64_t firstDelta90k = 0;
    // During catch-up: drop the entire buffer silently (summary logged by SyncAndSubmitFrame on exit).
    // Normal mode: keep >=1 frame to avoid starving the display.
    const bool silent = catchingUp;
    while ((silent ? !jitterBuf.empty() : jitterBuf.size() > 1) && !stopping.load(std::memory_order_relaxed)) {
        if (jitterBuf.front()->pts == AV_NOPTS_VALUE) {
            break;
        }
        const int64_t clock = ap->GetClock();
        if (clock == AV_NOPTS_VALUE) {
            break;
        }
        // Stop dropping once inside the hard window; soft corridor handles the residual.
        const int64_t currentDelta = jitterBuf.front()->pts - clock - latency;
        if (currentDelta >= -DECODER_SYNC_HARD_THRESHOLD_90K) {
            break;
        }
        if (dropped == 0) {
            firstDelta90k = currentDelta;
        }
        jitterBuf.pop_front();
        ++syncDropSinceLog;
        ++dropped;
    }
    if (dropped > 0) {
        if (silent) {
            catchUpDrops += dropped;
        } else {
            dsyslog("vaapivideo/decoder: sync drop (stale-jitter bulk) count=%d firstRaw=%+lldms buf=%zu", dropped,
                    static_cast<long long>(firstDelta90k / PTS_TICKS_PER_MS), jitterBuf.size());
        }
    }
}

[[nodiscard]] auto cVaapiDecoder::SyncAndSubmitFrame(std::unique_ptr<VaapiFrame> frame) -> bool {
    // Audio-master A/V sync gate. See AVSYNC.md for the full state diagram.

    if (!frame || !display) [[unlikely]] {
        return false;
    }

    const int64_t pts = frame->pts;

    // --- Trick mode ---
    if (trickSpeed.load(std::memory_order_acquire) != 0) {
        if (trickExitPending.exchange(false, std::memory_order_acquire)) {
            // Play()-without-TrickSpeed(0): clear trick flags, arm freerun while audio re-anchors.
            trickSpeed.store(0, std::memory_order_relaxed);
            isTrickReverse.store(false, std::memory_order_relaxed);
            isTrickFastForward.store(false, std::memory_order_relaxed);
            display->SetTrickActive(false); // re-enable underrun detector now that pacing is normal
            freerunFrames.store(DECODER_SYNC_FREERUN_FRAMES, std::memory_order_relaxed);
            syncLogPending.store(true, std::memory_order_relaxed);
            // Fall through to the normal sync path.
        } else {
            return SubmitTrickFrame(std::move(frame));
        }
    }

    // Still picture: no audio clock to sync against; sync logic would erroneously drop/delay it.
    if (stillPictureMode.load(std::memory_order_relaxed)) {
        if (pts != AV_NOPTS_VALUE) {
            PublishLastPts(pts);
        }
        return SubmitIfCurrent(std::move(frame));
    }

    // Publish PTS for lock-free readers (still detection, position query via GetLastPts).
    if (pts != AV_NOPTS_VALUE) {
        PublishLastPts(pts);
    }

    // Freerun bypass: no audio processor, no PTS, or inside the post-Clear / trick-exit window.
    auto *const ap = audioProcessor.load(std::memory_order_acquire);
    if (!ap || pts == AV_NOPTS_VALUE) {
        return SubmitIfCurrent(std::move(frame));
    }
    if (freerunFrames.load(std::memory_order_relaxed) > 0) {
        freerunFrames.fetch_sub(1, std::memory_order_relaxed);
        // Reset all controller state (EMA + warmup + debounce counters + catch-up + pendingDrops)
        // so the old clock domain doesn't bleed into the new one.
        pendingDrops = 0;
        ResetSmoothedDelta();
        return SubmitIfCurrent(std::move(frame));
    }

    // Snapshot latency once per frame; a mid-frame PCM<->passthrough flip must not change it.
    const int64_t latency = SyncLatency90k(ap);
    const int64_t clock = ap->GetClock();
    if (clock == AV_NOPTS_VALUE) {
        // No clock (ALSA reset / codec swap / device reset): freerun + reset all controller state.
        // EMA reset is essential -- a stale-domain EMA would fire a bogus correction once the clock
        // re-anchors. Discontinuous sources that skip Clear()/NotifyAudioChange() rely on this path.
        pendingDrops = 0;
        ResetSmoothedDelta();
        if (syncLogPending.exchange(false, std::memory_order_relaxed) || nextSyncLog.TimedOut()) {
            dsyslog("vaapivideo/decoder: sync freerun (no clock) buf=%zu", jitterBuf.size());
            nextSyncLog.Set(DECODER_SYNC_LOG_INTERVAL_MS);
        }
        return SubmitIfCurrent(std::move(frame));
    }

    // Consume one pending drop from a previous soft- or hard-behind burst. The triggering
    // event already logged the count; the EMA was reset there, so it is not bumped here.
    if (pendingDrops > 0) {
        --pendingDrops;
        ++syncDropSinceLog;
        return true;
    }

    // rawDelta > 0 = video ahead of audio. EMA is updated only outside catch-up regime
    // so spike-driven drops don't poison the smoother (see AVSYNC.md).
    const int64_t rawDelta = pts - clock - latency;

    // Catch-up: filter-bypass bulk drop until delta returns inside the soft corridor.
    //   Spike entry:    rawDelta < -2*HARD_THRESHOLD (~-400 ms). 2x guards against a lone outlier.
    //   Sustained entry: smoothedDelta < -2*CORRIDOR (~-100 ms): replay queue lag soft-behind cannot
    //     clear within its cooldown. EMA + warmup gate this trip so it survives a single bad frame.
    //   Exit: rawDelta > -CORRIDOR. Hysteresis = CORRIDOR; ResetSmoothedDelta forces a fresh EMA
    //   warmup before re-entry, preventing ping-pong across the gap.
    if (catchingUp) {
        // Exit at -CORRIDOR (not at +halfFrame): catch-up only progresses rawDelta when frames are
        // already cached in jitterBuf (drop iter is then a fast pop, audio barely advances). Once
        // the cache is drained each subsequent drop has to wait one VPP cycle for the next frame,
        // and on marginal-VPP hardware (UHD upscale, ~50 fps == audio rate) the per-iter audio
        // advance ~= PTS advance, so rawDelta no longer climbs. Targeting +halfFrame would then
        // make catch-up hang forever, silently dropping every newly decoded frame. -CORRIDOR is
        // the highest threshold that's guaranteed reachable from any typical entry point with
        // ordinary jitterBuf depth -- the small permanent negative offset that may remain is well
        // below the lipsync percept threshold.
        if (rawDelta > -DECODER_SYNC_CORRIDOR_90K) {
            dsyslog("vaapivideo/decoder: catch-up complete dropped=%d wall=%llums exit-raw=%+lldms", catchUpDrops,
                    static_cast<unsigned long long>(cTimeMs::Now() - catchUpStartMs),
                    static_cast<long long>(rawDelta / PTS_TICKS_PER_MS));
            catchingUp = false;
            catchUpDrops = 0;
            ResetSmoothedDelta(); // warmup alone gates soft until EMA reseeds; no follow-up event needed.
            pendingDrops = 0;
        } else {
            ++catchUpDrops;
            ++syncDropSinceLog;
            return true;
        }
    } else if (rawDelta < -2 * DECODER_SYNC_HARD_THRESHOLD_90K) {
        catchingUp = true;
        catchUpStartMs = cTimeMs::Now();
        catchUpDrops = 1;
        ++syncDropSinceLog;
        dsyslog("vaapivideo/decoder: catch-up entered (spike) raw=%+lldms",
                static_cast<long long>(rawDelta / PTS_TICKS_PER_MS));
        return true;
    } else if (!smoothedDeltaValid && rawDelta < -2 * DECODER_SYNC_CORRIDOR_90K) {
        // Pre-warmup raw-based entry: drains a stale pre-roll backlog before it poisons the EMA.
        catchingUp = true;
        catchUpStartMs = cTimeMs::Now();
        catchUpDrops = 1;
        ++syncDropSinceLog;
        dsyslog("vaapivideo/decoder: catch-up entered (warmup) raw=%+lldms",
                static_cast<long long>(rawDelta / PTS_TICKS_PER_MS));
        return true;
    } else if (smoothedDeltaValid && smoothedDelta90k < -2 * DECODER_SYNC_CORRIDOR_90K) {
        catchingUp = true;
        catchUpStartMs = cTimeMs::Now();
        catchUpDrops = 1;
        ++syncDropSinceLog;
        dsyslog("vaapivideo/decoder: catch-up entered (sustained) avg=%+lldms raw=%+lldms",
                static_cast<long long>(smoothedDelta90k / PTS_TICKS_PER_MS),
                static_cast<long long>(rawDelta / PTS_TICKS_PER_MS));
        return true;
    }

    UpdateSmoothedDelta(rawDelta);
    LogSyncStats(rawDelta, latency, ap);

    // Hard-behind: batch-drop to close the gap; reset EMA. 2-sample debounce guards against lone
    // GetClock/snd_pcm_delay outliers that would wipe ~1 s of EMA. Batch (vs single frameDur) because
    // a single drop leaves ~HARD_THRESHOLD residual pinned for the 5 s cooldown; pendingDrops + warmup
    // serialize follow-up events and warmup reseeds inside the corridor so soft stays a no-op.
    if (rawDelta < -DECODER_SYNC_HARD_THRESHOLD_90K) [[unlikely]] {
        if (++hardBehindDebounce < 2) {
            return SubmitIfCurrent(std::move(frame));
        }
        hardBehindDebounce = 0;
        const int correctMs = static_cast<int>(-rawDelta / PTS_TICKS_PER_MS);
        const int totalDrops = std::max(1, (correctMs + (outputFrameDurationMs / 2)) / outputFrameDurationMs);
        pendingDrops = totalDrops - 1;
        ++syncDropSinceLog;
        dsyslog("vaapivideo/decoder: sync drop (hard-behind) pts=%lld raw=%+lldms thr=%lldms drops=%d",
                static_cast<long long>(pts), static_cast<long long>(rawDelta / PTS_TICKS_PER_MS),
                static_cast<long long>(DECODER_SYNC_HARD_THRESHOLD_90K / PTS_TICKS_PER_MS), totalDrops);
        ResetSmoothedDelta();
        return true;
    }
    hardBehindDebounce = 0;

    uint64_t preSleepMs = 0; // non-zero if a sleep occurred; used for post-submit EMA correction.

    // Hard-ahead: replay blocks via WaitForAudioCatchUp; live sleeps <= HARD_AHEAD_MAX_MS.
    // 2-sample debounce same as hard-behind: avoids a 500 ms freeze on a lone snd_pcm_delay spike.
    if (rawDelta > DECODER_SYNC_HARD_THRESHOLD_90K) {
        if (++hardAheadDebounce < 2) {
            return SubmitIfCurrent(std::move(frame));
        }
        hardAheadDebounce = 0;
        if (!liveMode.load(std::memory_order_relaxed)) {
            WaitForAudioCatchUp(ap, pts, latency, rawDelta);
            ++syncSkipSinceLog;
            dsyslog("vaapivideo/decoder: sync skip (hard-ahead replay, post audio-catchup) pts=%lld raw=%+lldms",
                    static_cast<long long>(pts), static_cast<long long>(rawDelta / PTS_TICKS_PER_MS));
            pendingDrops = 0;
            syncCooldown.Set(DECODER_SYNC_COOLDOWN_MS);
            return SubmitIfCurrent(std::move(frame));
        }
        const int rawDeltaMs = static_cast<int>(rawDelta / PTS_TICKS_PER_MS);
        const int bigSleepMs = std::min(rawDeltaMs, DECODER_SYNC_HARD_AHEAD_MAX_MS);
        preSleepMs = cTimeMs::Now();
        if (display) {
            display->SetSyncSleeping(true);
        }
        cCondWait::SleepMs(bigSleepMs + outputFrameDurationMs);
        if (display) {
            display->SetSyncSleeping(false);
        }
        ++syncSkipSinceLog;
        dsyslog("vaapivideo/decoder: sync skip (hard-ahead live) pts=%lld raw=%+lldms slept=%dms",
                static_cast<long long>(pts), static_cast<long long>(rawDelta / PTS_TICKS_PER_MS),
                bigSleepMs + outputFrameDurationMs);
        pendingDrops = 0;
        syncCooldown.Set(DECODER_SYNC_COOLDOWN_MS);
    } else {
        hardAheadDebounce = 0;
    }

    // Soft corridor: |EMA| crossed CORRIDOR -> the drift is real, not noise. Trigger uses the
    // smoothed value (debounce); the correction size uses the instantaneous rawDelta (close
    // the actual gap, not the lagging average). Behind bursts drops + resets the EMA, just
    // like hard-behind, only smaller. Ahead sleeps. Cooldown rate-limits both directions.
    if (syncCooldown.TimedOut() && smoothedDeltaValid) {
        const int64_t absDelta90k = smoothedDelta90k < 0 ? -smoothedDelta90k : smoothedDelta90k;
        if (absDelta90k > DECODER_SYNC_CORRIDOR_90K) {
            const int correctMs =
                std::min(static_cast<int>(std::abs(rawDelta) / PTS_TICKS_PER_MS), DECODER_SYNC_MAX_CORRECTION_MS);
            syncCooldown.Set(DECODER_SYNC_COOLDOWN_MS);

            if (smoothedDelta90k < 0) {
                const int totalDrops = std::max(1, (correctMs + (outputFrameDurationMs / 2)) / outputFrameDurationMs);
                pendingDrops = totalDrops - 1;
                const auto triggerAvgMs = static_cast<long long>(smoothedDelta90k / PTS_TICKS_PER_MS);
                ++syncDropSinceLog;
                dsyslog("vaapivideo/decoder: sync drop (soft-behind) pts=%lld raw=%+lldms avg=%+lldms corr=%dms "
                        "drops=%d",
                        static_cast<long long>(pts), static_cast<long long>(rawDelta / PTS_TICKS_PER_MS), triggerAvgMs,
                        correctMs, totalDrops);
                ResetSmoothedDelta();
                return true;
            }
            // Ahead: sleep once; EMA feedback applied from measured elapsed after SubmitFrame.
            preSleepMs = cTimeMs::Now();
            if (display) {
                display->SetSyncSleeping(true);
            }
            cCondWait::SleepMs(correctMs + outputFrameDurationMs);
            if (display) {
                display->SetSyncSleeping(false);
            }
            ++syncSkipSinceLog;
            dsyslog("vaapivideo/decoder: sync skip (soft-ahead) pts=%lld raw=%+lldms avg=%+lldms slept=%dms",
                    static_cast<long long>(pts), static_cast<long long>(rawDelta / PTS_TICKS_PER_MS),
                    static_cast<long long>(smoothedDelta90k / PTS_TICKS_PER_MS), correctMs + outputFrameDurationMs);
        }
    }

    const bool submitted = SubmitIfCurrent(std::move(frame));
    if (preSleepMs != 0) {
        // EMA feedback: subtract measured shift (elapsed - frameDur); std::max guards underflow.
        const auto elapsedMs = static_cast<int>(cTimeMs::Now() - preSleepMs);
        const int extraMs = std::max(0, elapsedMs - outputFrameDurationMs);
        smoothedDelta90k -= static_cast<int64_t>(extraMs) * PTS_TICKS_PER_MS;
        // Suppress the next drain-loop miss check: the gap straddling this submit was a
        // deliberate hard-ahead / soft-ahead sleep, not upstream starvation.
        sleptInLastSubmit = true;
    }
    return submitted;
}
