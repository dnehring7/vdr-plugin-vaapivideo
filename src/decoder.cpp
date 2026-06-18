// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file decoder.cpp
 * @brief Threaded VAAPI decoder with VPP filter graph and audio-mastered A/V sync.
 *
 * Pipeline (two threads so a slow 4K VPP step never stalls presentation):
 *   EnqueueData() -> packetQueue -> Action() [DECODE thread]: VAAPI decode -> filter graph
 *      -> handoffQueue (epoch-stamped)
 *   handoffQueue -> PresentAction() [PRESENT thread]: splice -> jitterBuf -> due-gated drain
 *      -> SyncAndSubmitFrame -> display->SubmitFrame
 *   The present thread drains the decoded reserve at the audio-synced cadence while the decode
 *   thread is still filtering the next frame. (Unified for live and replay; liveMode selects the
 *   hard-ahead policy in SyncAndSubmitFrame.)
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
 *   handoffMutex is a strict leaf guarding handoffQueue. jitterBuf and the sync controller are
 *   presentation-thread-owned (no lock needed there); cross-thread requests use atomics + a
 *   deferred apply at the top of PresentAction().
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
#include <functional>
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
#include <vdr/osd.h>
#include <vdr/thread.h>
#include <vdr/tools.h>
#pragma GCC diagnostic pop

// ============================================================================
// === CONSTANTS ===
// ============================================================================

constexpr size_t DECODER_QUEUE_CAPACITY =
    200; ///< ~4 s @ 50 fps. Overflow drops oldest; trick mode limits to DECODER_TRICK_QUEUE_DEPTH.
constexpr int DECODER_SUBMIT_TIMEOUT_MS = 100; ///< VSync backpressure budget inside display->SubmitFrame().
constexpr int DECODER_CATCHUP_LOG_THROTTLE_MS =
    2000; ///< Min interval between catch-up entry/exit log pairs. Suppresses log flood during sustained
          ///< slow-decode cycling (e.g. VVC SW decode on hardware without VVC HW support); the next
          ///< non-throttled entry emits an aggregated summary of the suppressed cycles.
constexpr int DECODER_CATCHUP_SUMMARY_INTERVAL_MS =
    10000; ///< Periodic summary cadence while catch-up keeps cycling under throttle.
constexpr int DECODER_SYNC_COOLDOWN_MS = 5000; ///< Min interval between soft corrections; equals 5x EMA time constant.
constexpr int64_t DECODER_SYNC_CORRIDOR_90K =
    50 * PTS_TICKS_PER_MS; ///< Soft half-width: 50 ms in 90k ticks. Below the ~80 ms lipsync percept threshold.
constexpr int DECODER_SYNC_HINT_MAX_AGE_MS =
    DECODER_SYNC_COOLDOWN_MS; ///< Max age of a pre-correction stableDelta snapshot before it counts as stale and the
                              ///< hint falls back to the current smoothedDelta. Mirrors the soft-correction cooldown:
                              ///< if no correction has fired in this window, smoothedDelta is in its own steady-state
                              ///< and is a better seed than an older snapshot from a different operating point.
constexpr int DECODER_SYNC_EMA_SAMPLES =
    50; ///< EMA alpha = 1/N (~= 1 s @ 50 fps). Residual accumulator avoids truncation stall.
constexpr int DECODER_SYNC_FREERUN_FRAMES = 1; ///< Unpaced frames after Clear() / track switch / trick-exit.
constexpr int64_t DECODER_SYNC_HARD_THRESHOLD_90K =
    200 * PTS_TICKS_PER_MS; ///< 200 ms in 90k ticks. Beyond this the soft corridor cannot recover in one event.
constexpr int DECODER_SYNC_LOG_INTERVAL_MS = 2000; ///< Periodic sync-stats dsyslog cadence.
constexpr int DECODER_SYNC_MAX_CORRECTION_MS = static_cast<int>(DECODER_SYNC_HARD_THRESHOLD_90K / PTS_TICKS_PER_MS);
///< Per-event cap = HARD_THRESHOLD (200 ms); one event clears any in-corridor offset.
constexpr int DECODER_SYNC_WARMUP_SAMPLES =
    50; ///< Mean-seed samples before EMA starts; sqrt(N) cuts 50p deinterlace bias.
constexpr int DECODER_SYNC_HARD_AHEAD_MAX_MS =
    500; ///< Live hard-ahead sleep cap (ms); prevents indefinite chase + upstream queue overflow.
// DECODER_RESERVE_HARD_CAP lives in decoder.h (statically coupled to the mediaplayer backpressure gate).
constexpr int64_t DECODER_DRAIN_FUTURE_MAX_MS =
    3000; ///< Drop any head sitting more than this far ahead of audio_clock. Such an offset is a real
          ///< PTS discontinuity (ATTA anchor swap, broadcast PCR break, stale snd_pcm_delay clamp).
          ///< Legitimate offsets must stay below this: TS live recordings can have video leading audio
          ///< by ~1500 ms after channel switches or recording start -- those should converge naturally
          ///< as the audio clock advances, not be dropped. Sync alignment of mediaplayer streams is
          ///< handled at the source (cVaapiMediaSource::PopulateStreamInfo picks ptsOrigin = max of
          ///< per-stream start_times so both streams begin at rebased PTS 0 together), so the
          ///< threshold only needs to catch genuine discontinuities, not normal startup offsets.
constexpr uint64_t DECODER_NO_CLOCK_HOLD_MS =
    1500; ///< Walltime the drain holds the first frame still (jitterBuf non-empty, ap non-null,
          ///< GetClock()=NOPTS) waiting for the audio clock to anchor before falling back to no-clock
          ///< freerun. Keeps post-seek frames buffered so the head doesn't run ahead of the clock at
          ///< VSync rate. Bounded so a video-only stream (ap non-null, no audio anchor ever) does not
          ///< freeze forever; the nearCap escape additionally bypasses it if jitterBuf hits the cap.
constexpr int VDR_SLOW_REVERSE_SPEED_MULT =
    12; ///< VDR dvbplayer.c SPEED_MULT. Slow FORWARD passes the bare slowdown factor (2/4/8); slow REVERSE
        ///< passes slowdown * this, clamped to MAX_VIDEO_SLOWMOTION (24/48/96 -> 63). Divided back out in
        ///< SetTrickSpeed to recover the 1..3 slowdown level (2/4/5 after the clamp) for reverse pacing.
constexpr uint64_t DECODER_TRICK_SLOW_HOLD_MAX_MS =
    200; ///< Safety cap on the slow-FORWARD per-frame hold (>= 5 fps): keeps pacing responsive even if VDR sends
         ///< an unexpected speed scaling. The slowest intended slow forward (/8 = 160 ms) stays under it, so this
         ///< only clamps pathological values. Slow reverse has its own larger cap (see below).
constexpr uint64_t DECODER_SLOW_REVERSE_HOLD_BASE_MS =
    260; ///< Slow-REVERSE per-frame hold floor. Reverse delivers I-frames stepping a FIXED ~0.4 s of content each
         ///< (dvbplayer.c d=0.4*fps, independent of speed -- identical stepping to fast reverse, only the hold
         ///< differs), so the hold maps to reverse CONTENT speed, not to a per-source-frame slowdown like slow
         ///< forward; it must be far larger than the forward 20 ms base or content flies backward at 4-10x.
constexpr uint64_t DECODER_SLOW_REVERSE_HOLD_STEP_MS =
    70; ///< Added per slowdown unit: slowdown 2/4/5 -> 400/540/610 ms hold (~1.1x/0.8x/0.7x reverse), every level
        ///< clearly under fast reverse's 2-8x without a multi-second still-frame slideshow.
constexpr uint64_t DECODER_SLOW_REVERSE_HOLD_MAX_MS =
    700; ///< Cap on the slow-reverse hold: keeps the slowest level off the seconds-long slideshow that a genuinely
         ///< 0.25x reverse (~1.6 s/frame) would reintroduce.

// ============================================================================
// === STRUCTURES ===
// ============================================================================

VaapiFrame::VaapiFrame(VaapiFrame &&other) noexcept
    : avFrame(std::exchange(other.avFrame, nullptr)), ownsFrame(std::exchange(other.ownsFrame, false)),
      producedEpoch(std::exchange(other.producedEpoch, 0)), pts(std::exchange(other.pts, AV_NOPTS_VALUE)),
      vaSurfaceId(std::exchange(other.vaSurfaceId, VA_INVALID_SURFACE)) {}

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
        avFrame = std::exchange(other.avFrame, nullptr);
        ownsFrame = std::exchange(other.ownsFrame, false);
        producedEpoch = std::exchange(other.producedEpoch, 0);
        pts = std::exchange(other.pts, AV_NOPTS_VALUE);
        vaSurfaceId = std::exchange(other.vaSurfaceId, VA_INVALID_SURFACE);
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
    // Content-boundary flush (channel switch, playlist advance, mediaplayer close, ...): the new
    // pipeline may have an entirely different GPU-vs-audio offset, so the EMA must warm up from
    // scratch. preserveSeekHint=false is bound to this request directly so a coalesced flush
    // can't accidentally preserve a hint from an unrelated FlushForSeek that happened in between.
    ClearInternal(/*resetFilter=*/true, /*preserveSeekHint=*/false);
}

auto cVaapiDecoder::FlushForSeek() -> void {
    // Filter reset is required only when the active chain contains a temporal filter that
    // cannot survive a seek-sized PTS jump. Today that's just `fps=N` (added for source rates
    // below the display rate, e.g. 25 fps source on a 50 Hz display): it bridges the gap
    // between previous_output_pts and the first post-seek input by emitting hundreds of
    // duplicate frames at stale PTS -- observable in syslog as a long stale-jitter /
    // catch-up cascade with raw advancing ~40 ms per outer iter (one source frame).
    //
    // bwdif / yadif hold at most 1-2 pre-seek fields and self-clear inside one filter window,
    // so for chains without `fps=N` we keep the filter graph alive across the seek and save
    // the ~100 ms rebuild + the 3-line filter-init log spam per seek. The atomic load is
    // safe: cVideoFilterChain methods only mutate hasFpsFilter_ from the decode thread under
    // codecMutex, and FlushForSeek already serializes with that mutex inside ClearInternal.
    const bool needFilterReset = filterChain.HasFpsFilter();
    filterCompactRebuildPending.store(needFilterReset, std::memory_order_release);
    // Carry the converged smoothedDelta across the flush as a "fast start" hint: the GPU vs.
    // audio offset is a property of the pipeline, unchanged by the playback position. The
    // preserve policy travels *with* the flush request (jitterFlushRequest=2), not as a separate
    // atomic, so a stale flush observed by the decode thread always carries its originator's
    // policy -- never a later issuer's.
    ClearInternal(needFilterReset, /*preserveSeekHint=*/true);
}

auto cVaapiDecoder::ClearInternal(bool resetFilter, bool preserveSeekHint) -> void {
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

        if (resetFilter) {
            // Graph caches hw_frames_ctx and may hold stale-PTS frames; rebuilt lazily on next frame.
            filterChain.Reset();
        }
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
    trickAwaitSecondField = false; // parser reset: forget any pending PAFF field-pair (parserMutex held)

    // Epoch BEFORE NOPTS: any in-flight decoder publish observes the new epoch on its check or
    // recheck and aborts/undoes. Reversed order would let a stale publish overwrite NOPTS.
    clearEpoch.fetch_add(1, std::memory_order_release);
    lastPts.store(AV_NOPTS_VALUE, std::memory_order_release);
    codecDrainPending.store(false, std::memory_order_relaxed);
    stillPictureMode.store(false, std::memory_order_relaxed);

    // jitterBuf and EMA state are presentation-thread-owned; reset is deferred via atomic request.
    // freerunFrames MUST store before jitterFlushRequest: if the order is reversed the decode
    // thread could drain through the due-gate on a still-NOPTS clock and black the screen.
    // Request encodes the preserve-hint policy *with* the flush request so a stale flush in
    // the decode-thread's queue can never observe a later issuer's policy by accident.
    freerunFrames.store(DECODER_SYNC_FREERUN_FRAMES, std::memory_order_relaxed);
    jitterFlushRequest.store(preserveSeekHint ? 2 : 1, std::memory_order_release);

    syncLogPending.store(true, std::memory_order_relaxed);

    // Wake the presenter so it consumes the flush promptly (it owns jitterBuf + the controller now).
    // Safe while holding codecMutex/parserMutex: handoffMutex is a strict leaf.
    WakePresenter();
}

auto cVaapiDecoder::WakePresenter() noexcept -> void {
    const cMutexLock hl(&handoffMutex);
    handoffCondition.Broadcast();
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
            // FF: keep only keyframes (inter frames are undisplayable here). PAFF exception: an I-frame
            // is two field pictures whose second field carries key_frame==0 -- keep it too, or the
            // decoder gets unpaired top fields and rejects them (send_packet EINVAL). Reverse skips this
            // filter entirely (same reason). One-shot: a kept keyframe field arms the next field.
            if (trickSpeed.load(std::memory_order_acquire) != 0 && isTrickFastForward.load(std::memory_order_relaxed)) {
                const bool isField = parserCtx->picture_structure == AV_PICTURE_STRUCTURE_TOP_FIELD ||
                                     parserCtx->picture_structure == AV_PICTURE_STRUCTURE_BOTTOM_FIELD;
                if (parserCtx->key_frame != 0) {
                    trickAwaitSecondField = isField; // I-frame first field -> keep its pair's second field
                } else if (trickAwaitSecondField && isField) {
                    trickAwaitSecondField = false; // second field completes the keyframe pair
                } else {
                    trickAwaitSecondField = false;
                    continue; // inter frame
                }
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
        dsyslog("vaapivideo/decoder: no frame %llu ms after open (packets sent=%zu, "
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
    if (trickSpeed.load(std::memory_order_acquire) != 0 && isTrickFastForward.load(std::memory_order_relaxed) &&
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

    // Reset stopping + both exit flags before Start() so the spawned threads observe a clean state.
    stopping.store(false, std::memory_order_release);
    hasExited.store(false, std::memory_order_release);
    presentExited.store(false, std::memory_order_release);
    ready.store(true, std::memory_order_release);
    // Start the presenter FIRST so it is parked on handoffCondition before the decode thread hands
    // off the first frame (avoids first-frame latency). On failure, do NOT start the decode thread,
    // and mark both exit flags so a subsequent Shutdown() returns immediately.
    if (!presenter.Start()) {
        esyslog("vaapivideo/decoder: failed to start presentation thread");
        hasExited.store(true, std::memory_order_release);
        presentExited.store(true, std::memory_order_release);
        ready.store(false, std::memory_order_release);
        return false;
    }
    // Symmetric with the presenter check: cThread::Start() returns false on pthread_create failure
    // and Action() never runs, so hasExited would stay false forever and Shutdown()'s decode spin
    // would busy-wait the full SHUTDOWN_TIMEOUT_MS. On failure, mark the decode side exited and stop
    // the already-running presenter cleanly (signal stopping, wake it, join) before returning false.
    if (!Start()) { // decode thread
        esyslog("vaapivideo/decoder: failed to start decode thread");
        hasExited.store(true, std::memory_order_release);
        ready.store(false, std::memory_order_release);
        stopping.store(true, std::memory_order_release);
        {
            const cMutexLock hl(&handoffMutex);
            handoffCondition.Broadcast();
        }
        presenter.Stop(3);
        presentExited.store(true, std::memory_order_release);
        return false;
    }

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

/// True iff @p codec exposes a VAAPI hw_device_ctx-style HW config entry. Used to filter
/// sibling decoders for the same codec ID when avcodec_find_decoder()'s default pick is
/// software-only (libdav1d / libvpx in typical FFmpeg builds).
[[nodiscard]] static auto HasVaapiHwConfig(const AVCodec *codec) noexcept -> bool {
    if (codec == nullptr) {
        return false;
    }
    for (int i = 0;; ++i) {
        const AVCodecHWConfig *hwCfg = avcodec_get_hw_config(codec, i);
        if (hwCfg == nullptr) {
            return false;
        }
        if ((hwCfg->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) != 0 &&
            hwCfg->device_type == AV_HWDEVICE_TYPE_VAAPI) {
            return true;
        }
    }
}

/// Walk every registered decoder for @p codecId and return the first one whose HW config
/// matches HasVaapiHwConfig(). Returns nullptr if FFmpeg has no VAAPI-capable decoder for
/// this codec ID at all (caller must keep its avcodec_find_decoder() result as the SW
/// fallback and clear useHwDecode).
[[nodiscard]] static auto FindVaapiDecoder(AVCodecID codecId) noexcept -> const AVCodec * {
    void *iter = nullptr;
    while (const AVCodec *codec = av_codec_iterate(&iter)) {
        if (av_codec_is_decoder(codec) != 0 && codec->id == codecId && HasVaapiHwConfig(codec)) {
            return codec;
        }
    }
    return nullptr;
}

// AVCodecContext::get_format callback: pin decode output to VAAPI surfaces. Without this FFmpeg may
// silently fall back to SW frames. Free function (not a lambda) so it reads as the C callback it is.
[[nodiscard]] static auto SelectVaapiPixelFormat(AVCodecContext * /*ctx*/, const AVPixelFormat *formats)
    -> AVPixelFormat {
    for (const AVPixelFormat *fmt = formats; *fmt != AV_PIX_FMT_NONE; ++fmt) {
        if (*fmt == AV_PIX_FMT_VAAPI) {
            return AV_PIX_FMT_VAAPI;
        }
    }
    return AV_PIX_FMT_NONE;
}

[[nodiscard]] auto cVaapiDecoder::OpenCodecWithInfo(const VideoStreamInfo &info) -> bool {
    // codecMutex + parserMutex: reopens both contexts atomically; lock order is fixed
    // (codec -> parser) and matches Clear() / SetTrickSpeed().
    const cMutexLock decodeLock(&codecMutex);
    const cMutexLock parseLock(&parserMutex);

    if (!ready.load(std::memory_order_acquire) || stopping.load(std::memory_order_acquire)) [[unlikely]] {
        dsyslog("vaapivideo/decoder: not initialized or shutting down");
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

    // VA driver advertising a profile is necessary but not sufficient; FFmpeg also needs a
    // decoder for this codec ID that exposes a VAAPI AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX
    // entry. avcodec_find_decoder() above returns the highest-priority registered decoder,
    // which for AV1 / VP9 in most distro FFmpeg builds is libdav1d / libvpx -- both
    // software-only, so the VAAPI path would be silently bypassed even though hwAv1 / hwVp9
    // is true. Switch to a sibling decoder that does expose VAAPI; fall back to SW (keep the
    // original decoder pointer) if none exists, since the original is a valid SW codec.
    if (useHwDecode) {
        if (const AVCodec *hwDecoder = FindVaapiDecoder(info.codecId); hwDecoder != nullptr) {
            decoder = hwDecoder;
        } else {
            useHwDecode = false;
        }
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
        // Context is shared across an MPEG-2 -> MPEG-2 switch, so refresh the hint here; the
        // preceding Clear() reset the graph, which rebuilds with this value on the next frame.
        streamInterlaced = info.streamInterlaced;
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
    streamInterlaced = false;
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

    // Mediaplayer hint: VP9/AV1 carry no bitstream framerate; without this seed
    // codecCtx->framerate stays 0/0 and the filter chain falls back to the DVB 50/1
    // default. The decoder may overwrite this from VUI for h.264/HEVC -- the hint is
    // strictly a fallback for codecs that don't signal it.
    if (info.fpsNum > 0 && info.fpsDen > 0) {
        decoderCtx->framerate = AVRational{.num = info.fpsNum, .den = info.fpsDen};
    }

    // VP9/AV1 signal HDR transfer/primaries only in the container, not the bitstream. Seed the codec
    // color fields so FFmpeg stamps them onto every frame; UNSPECIFIED on the PES path leaves the
    // in-bitstream VUI authoritative.
    decoderCtx->color_primaries = info.primaries;
    decoderCtx->color_trc = info.transfer;
    decoderCtx->colorspace = info.colorSpace;
    decoderCtx->color_range = info.range;

    // Post-decode net for the seed: a decoder may rewrite the frame's color from an UNSPECIFIED VUI
    // (HEVC/DV base layers), and mastering/content-light have no avctx field at all.
    hintColorPrimaries = info.primaries;
    hintColorTransfer = info.transfer;
    hintColorSpace = info.colorSpace;
    hintColorRange = info.range;
    hintHasMasteringDisplay = info.hasMasteringDisplay;
    hintMasteringDisplay = info.masteringDisplay;
    hintHasContentLight = info.hasContentLight;
    hintContentLight = info.contentLight;

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
        decoderCtx->get_format = SelectVaapiPixelFormat;
    }
    // SW decode: thread_count left at lavc auto (libdav1d honors it via FF_CODEC_CAP_AUTO_THREADS);
    // hw_device_ctx left unset, filter chain's hwupload still runs VPP on the GPU.

    if (const int ret = avcodec_open2(decoderCtx.get(), decoder, nullptr); ret < 0) [[unlikely]] {
        esyslog("vaapivideo/decoder: failed to open %s: %s", decoder->name, AvErr(ret).data());
        return false;
    }

    codecCtx = std::move(decoderCtx);
    currentCodecId = info.codecId;
    streamInterlaced = info.streamInterlaced;
    // codecCtx->profile is AV_PROFILE_UNKNOWN until FFmpeg parses the first SPS/VPS.
    // The authoritative value is logged in the first-frame block inside DecodeOnePacket.
    isyslog("vaapivideo/decoder: opened %s (%s%s)", decoder->name, useHwDecode ? "hardware" : "software",
            info.extradataSize > 0 ? ", extradata" : "");

    // Software-fallback warning: heavy codecs without a hardware decoder are likely to
    // miss real-time on consumer iGPUs. VVC has no widely-deployed HW decoder yet (requires
    // Intel Lunar Lake / Battlemage or newer); AV1 SW pegs the CPU at 1080p+; HEVC SW is
    // borderline at 4K. The warning is purely diagnostic -- playback still proceeds -- and
    // gives the user a clear hint when the symptom is sustained catch-up cycling.
    if (!useHwDecode) {
        const bool tooHeavy = info.codecId == AV_CODEC_ID_VVC ||
                              (info.codecId == AV_CODEC_ID_AV1 && info.codedHeight >= 1080) ||
                              (info.codecId == AV_CODEC_ID_HEVC && info.codedHeight >= 2160);
        if (tooHeavy) {
            esyslog("vaapivideo/decoder: warning: %s software decode at %dx%d may not sustain real-time on this "
                    "hardware; expect dropped frames if catch-up cycling appears in the log",
                    decoder->name, info.codedWidth, info.codedHeight);
        }
    }

    return true;
}

auto cVaapiDecoder::NotifyAudioChange() -> void {
    // Audio clock is NOPTS for ~100--500 ms while the new pipeline primes.
    // Freerun surfaces the next frame immediately; catch-up realigns once the clock re-anchors.
    freerunFrames.store(DECODER_SYNC_FREERUN_FRAMES, std::memory_order_relaxed);
    syncLogPending.store(true, std::memory_order_relaxed);
    WakePresenter(); // present thread acts on freerunFrames; wake it so it doesn't wait out the poll.
}

auto cVaapiDecoder::SetAudioProcessor(cAudioProcessor *audio) -> void {
    audioProcessor.store(audio, std::memory_order_release);
}

auto cVaapiDecoder::SetLoopTickCallback(std::function<void()> callback) -> void {
    // Set once before Initialize() starts the thread, so the decode loop reads it without a lock.
    loopTickCallback = std::move(callback);
}

auto cVaapiDecoder::SetDevicePaused(bool paused) noexcept -> void {
    // Flips the drain-loop hold gate. See decoder.h declaration for the rationale; the read
    // site lives at the top of the drain `while` in PresentAction().
    devicePaused.store(paused, std::memory_order_release);
    WakePresenter(); // resume/pause observed within one present iteration instead of waiting the poll.
}

auto cVaapiDecoder::SetLiveMode(bool live) -> void { liveMode.store(live, std::memory_order_relaxed); }

auto cVaapiDecoder::RequestTrickExit() -> void {
    // One-frame cancellation grace: VDR fires Play() as the middle leg of REW->PLAY->REW, so a
    // SetTrickSpeed() arriving within a frame period supersedes this. After the grace the present
    // thread resolves the exit unconditionally (ResolvePendingTrickExit) -- queue-independent, so FF
    // can't hang waiting for a keyframe that the FF gate would have to let through first.
    const auto graceMs = static_cast<uint64_t>(std::max(1, outputFrameDurationMs.load(std::memory_order_relaxed)));
    deferredTrickExitDueMs.store(cTimeMs::Now() + graceMs, std::memory_order_relaxed);
    deferredTrickExitPending.store(true, std::memory_order_release);
    WakePresenter(); // trick-exit is resolved on the present thread; wake it to start the grace timer.
}

auto cVaapiDecoder::SetTrickSpeed(int speed, bool forward, bool fast) -> void {
    // speed=0: normal. fast: FF/REW (speed>=6->2x, >=3->4x, else 8x). !fast: slow at 1/speed hold.
    deferredTrickExitPending.store(false, std::memory_order_relaxed); // supersedes any pending deferred trick-exit
    deferredTrickExitDueMs.store(0, std::memory_order_relaxed);

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

    // Decide the new pacing + whether this transition is a generation boundary BEFORE taking the
    // locks, so the publish below is a tight sequence under codecMutex->parserMutex. A generation
    // boundary is entry/exit (0<->N) or an FF<->REW direction flip; an in-mode speed change is not.
    const int previousSpeed = trickSpeed.load(std::memory_order_acquire);
    const bool newFastForward = forward && fast;
    const bool newReverse = !forward;
    const bool oldFastForward = isTrickFastForward.load(std::memory_order_relaxed);
    const bool oldReverse = isTrickReverse.load(std::memory_order_relaxed);
    const bool generationBoundary = (previousSpeed == 0) != (speed == 0) ||
                                    (speed != 0 && (oldFastForward != newFastForward || oldReverse != newReverse));

    // Compute this transition's pacing. Fast (FF/REW): PTS-derived hold via trickMultiplier (set below).
    // Slow forward and slow reverse use DIFFERENT hold models -- see each branch.
    uint64_t newMultiplier = 0;
    uint64_t newHoldMs = 0;
    if (speed > 0 && !forward) {
        // Slow reverse paces by reverse CONTENT speed, not a per-source-frame slowdown: VDR delivers
        // I-frames stepping a fixed ~0.4 s of content each (dvbplayer.c -- same stepping as fast reverse,
        // only the hold differs), so a slow-forward-style slowdown*20 ms hold (40-100 ms) ran content
        // backward at 4-10x: the "still too fast" report. Recover the slowdown level (sp/12 = 2/4/5 after
        // VDR's 96->63 clamp) and hold each I-frame BASE + slowdown*STEP ms (400/540/610), capped.
        const uint64_t slowdown = std::max<uint64_t>(1, static_cast<uint64_t>(speed) / VDR_SLOW_REVERSE_SPEED_MULT);
        newHoldMs =
            std::min<uint64_t>(DECODER_SLOW_REVERSE_HOLD_BASE_MS + (slowdown * DECODER_SLOW_REVERSE_HOLD_STEP_MS),
                               DECODER_SLOW_REVERSE_HOLD_MAX_MS);
    } else {
        // Slow forward (every source frame delivered) and speed==0: hold = slowdown * base, capped.
        const uint64_t slowdown = static_cast<uint64_t>(std::max(0, speed));
        newHoldMs = std::min<uint64_t>(slowdown * DECODER_TRICK_HOLD_MS, DECODER_TRICK_SLOW_HOLD_MAX_MS);
    }
    if (fast && speed > 0) {
        if (speed >= 6) {
            newMultiplier = 2;
        } else if (speed >= 3) {
            newMultiplier = 4;
        } else {
            newMultiplier = 8;
        }
        newHoldMs = DECODER_TRICK_HOLD_MS;
    }
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
        } else if (generationBoundary && display && filterChain.IsBuilt()) {
            // Slow-forward entry, or any exit to normal play: the codec stream stays contiguous (no flush),
            // but the FILTER must still switch -- a very light bob chain while pacing/FF/RW, the full quality
            // chain in normal play. Reset so the next decoded frame rebuilds with the current trickMode; the
            // keep-alive slot protects in-flight display surfaces. Without this, slow forward kept painting
            // through the heavy normal-play chain (sluggish), and exits left normal play on the light chain.
            const cMutexLock vaLock(&display->GetVaDriverMutex());
            filterChain.Reset();
        }

        // Flags, pacing, the generation-boundary purge, and trickSpeed are ALL published while
        // parserMutex is held, so a concurrent EnqueueData() never parses non-keyframes against a
        // stale trickSpeed in the window just after the flush. Flags before the trickSpeed release-
        // store: an acquire reader that sees the new speed sees a consistent (mode, direction).
        isTrickFastForward.store(newFastForward, std::memory_order_relaxed);
        isTrickReverse.store(newReverse, std::memory_order_relaxed);
        prevTrickPts.store(AV_NOPTS_VALUE, std::memory_order_relaxed);
        trickMultiplier.store(newMultiplier, std::memory_order_relaxed);
        trickHoldMs.store(newHoldMs, std::memory_order_relaxed);
        nextTrickFrameDue.store(cTimeMs::Now(), std::memory_order_relaxed); // no initial hold

        // Generation boundary (entry/exit, or FF<->REW flip): purge the presenter's pre-transition
        // decoded reserve via clearEpoch + a deferred flush. The decouple decodes ~2 s ahead in normal
        // play, so without the entry purge slow-forward would crawl through that stale reserve for
        // ~15 s instead of pacing the current position (the symptom the seeking modes avoid because
        // VDR's own Clear() bursts already purge). An in-mode speed change (slow 8->4) keeps the tiny
        // trick reserve and just re-paces it. Epoch BEFORE NOPTS (Clear-race guard).
        if (generationBoundary) {
            clearEpoch.fetch_add(1, std::memory_order_release);
            lastPts.store(AV_NOPTS_VALUE, std::memory_order_release);
            jitterFlushRequest.store(1, std::memory_order_release);
            if (speed == 0) {
                freerunFrames.store(DECODER_SYNC_FREERUN_FRAMES, std::memory_order_relaxed);
            }
            syncLogPending.store(true, std::memory_order_relaxed);
        }

        trickSpeed.store(speed, std::memory_order_release);
    }

    // speed==0 path: clear the display's trick flag once pacing state is fully published.
    // The speed!=0 case was set up front, before the teardown above.
    if (display && speed == 0) {
        display->SetTrickActive(false);
    }

    // Wake the presenter so it observes the clearEpoch bump (stale-frame purge) and the new pacing.
    WakePresenter();
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
        // Decode-FIRST teardown: stop the producer before joining the consumer so the present
        // thread can drain any remaining handed-off frames cleanly. Wake both loops, then Cancel.
        {
            const cMutexLock lock(&packetMutex);
            packetCondition.Broadcast();
        }
        {
            const cMutexLock lock(&handoffMutex);
            handoffCondition.Broadcast(); // present thread parked on the due-gate wait
            handoffNotFull.Broadcast();   // decode thread parked on backpressure
        }
        // cThread::Running() is unfenced; hasExited/presentExited default to true (no thread
        // started), so Cancel only when the loop is actually running.
        if (!hasExited.load(std::memory_order_acquire)) {
            Cancel(3);
        }
        if (!presentExited.load(std::memory_order_acquire)) {
            presenter.Stop(3);
        }
    }

    // Both spins run UNCONDITIONALLY (every caller waits) so a second Shutdown() can never reach the
    // handoffQueue.clear() below while a thread is still draining handed-off frames. Separate
    // deadlines so a slow decode join doesn't consume the presenter's budget; re-broadcast each
    // loop's condvar every 10 ms in case a wakeup was lost to a race. The decode thread may itself be
    // parked on handoffNotFull (backpressure), so broadcast that too while waiting on it.
    const cTimeMs decodeTimeout(SHUTDOWN_TIMEOUT_MS);
    while (!hasExited.load(std::memory_order_acquire) && !decodeTimeout.TimedOut()) {
        {
            const cMutexLock lock(&packetMutex);
            packetCondition.Broadcast();
        }
        {
            const cMutexLock lock(&handoffMutex);
            handoffNotFull.Broadcast();
        }
        cCondWait::SleepMs(10);
    }
    const cTimeMs presentTimeout(SHUTDOWN_TIMEOUT_MS);
    while (!presentExited.load(std::memory_order_acquire) && !presentTimeout.TimedOut()) {
        {
            const cMutexLock lock(&handoffMutex);
            handoffCondition.Broadcast();
        }
        cCondWait::SleepMs(10);
    }

    // Drain after both threads exit: packets enqueued after the decode thread checked stopping would
    // leak otherwise (std::queue<AVPacket*> does not own its elements), and the handoff holds
    // decoded VaapiFrames whose VAAPI surfaces must be released. Safe only once both threads are
    // joined -- the unconditional spins above guarantee that for every caller.
    DrainQueue();
    {
        const cMutexLock lock(&handoffMutex);
        handoffQueue.clear();
    }
}

// ============================================================================
// === THREAD ===
// ============================================================================

auto cVaapiDecoder::Action() -> void {
    dsyslog("vaapivideo/decoder: thread started");

    std::vector<std::unique_ptr<VaapiFrame>> pendingFrames;
    int trickEmptyDecodes{0};        ///< Consecutive reverse-trick packets that yielded no frame; arms force-drain.
    cTimeMs jitterOverflowLogGate;   ///< Rate-limits the per-iter "handoff overflow" syslog spam.
    size_t jitterOverflowSinceLog{}; ///< Sum of frames dropped since the last overflow log line.

    // Action-local tiny transform: stamp the frames a producing op appended ([firstIndex, size)) with
    // the epoch the caller snapshotted WHILE HOLDING codecMutex (clearEpoch is bumped under codecMutex by
    // Clear()/SetTrickSpeed(0)), so a Clear() landing between two producing ops can't stamp the later
    // op's post-flush frames with a stale epoch. The epoch correctness is the caller's locked snapshot;
    // this is just the field write, so it stays a local lambda by its only user rather than a file helper.
    const auto stampProducedEpoch = [](std::vector<std::unique_ptr<VaapiFrame>> &frames, size_t firstIndex,
                                       uint64_t epoch) -> void {
        for (size_t index = firstIndex; index < frames.size(); ++index) {
            frames.at(index)->producedEpoch = epoch;
        }
    };

    while (!stopping.load(std::memory_order_acquire)) {
        // Device hook ticked every iteration -- including the ~10 ms idle waits when no packets
        // arrive -- so the device gets a reliable tick even on a scrambled channel that delivers no
        // PES. No decoder lock is held here; the callback is a cheap no-op unless armed.
        if (loopTickCallback) {
            loopTickCallback();
        }

        // Handoff backpressure: if the presentation thread has not drained the reserve below the cap
        // (e.g. it is inside a hard-ahead audio-catch-up sleep), stop decoding ahead instead of
        // overflowing the handoff and dropping already-decoded reserve frames. packetQueue then fills
        // and the existing upstream gate (IsQueueFull / AUDIO_QUEUE_HIGHWATER) throttles the feed.
        // Held only on handoffMutex (a strict leaf) so this never blocks Clear()/EnqueueData().
        //
        // Trick play caps the in-flight depth at DECODER_TRICK_QUEUE_DEPTH: SubmitTrickFrame's pacing
        // runs on the PRESENT thread now, so the decode thread must re-throttle itself (pre-decouple it
        // blocked in SubmitTrickFrame, which made the device's IsReadyForNextTrickFrame gate the feed).
        // Without this the decode thread drains packetQueue into the handoff far faster than the trick
        // pacer consumes it -- the device floods frames ahead and trick play breaks (over-buffered,
        // wrong speed, overflow drops).
        const bool inTrick = trickSpeed.load(std::memory_order_acquire) != 0;
        const size_t handoffCap = inTrick ? DECODER_TRICK_QUEUE_DEPTH : DECODER_RESERVE_HARD_CAP;
        {
            const cMutexLock hl(&handoffMutex);
            // Bound the producer to the TOTAL decoded reserve (handoffQueue + jitterBuf): the present
            // thread splices handoffQueue into jitterBuf, so checking handoffQueue alone would let
            // jitterBuf grow under a due-gate hold until the consumer-side guard had to drop frames.
            // publishedDecodedReserveSize carries that total. In trick mode the depth is already bounded to
            // DECODER_TRICK_QUEUE_DEPTH by the handoffQueue check, and publishedDecodedReserveSize lags one
            // (long) present iteration -- so skip the total check there to avoid stalling fast FF/REW
            // on a stale value (steady playback republishes every ~frame, so the lag is harmless).
            while ((handoffQueue.size() >= handoffCap ||
                    (!inTrick && publishedDecodedReserveSize.load(std::memory_order_relaxed) >= handoffCap)) &&
                   !stopping.load(std::memory_order_acquire)) {
                handoffNotFull.TimedWait(handoffMutex, 50);
            }
        }
        if (stopping.load(std::memory_order_acquire)) {
            break;
        }

        std::unique_ptr<AVPacket, FreeAVPacket> queuedPacket;

        // The presentation thread owns the due-gate pacing now, so the decode thread only needs to
        // wake promptly on new input. 10 ms cap keeps the loopTickCallback watchdog cadence on a
        // scrambled no-PES channel.
        {
            const cMutexLock lock(&packetMutex);
            if (packetQueue.empty() && !stopping.load(std::memory_order_acquire)) {
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

        // --- Decode ---
        pendingFrames.clear();
        bool drainedCodecAtEof = false;

        // codecMutex scoped to decode only; released before the handoff push so Clear() /
        // SetTrickSpeed() on other threads don't stall behind it. The producedEpoch is snapshotted
        // INSIDE the codecMutex critical section (see stampProducedEpoch above) so a racing flush
        // can't stamp these frames stale-then-current.
        if (queuedPacket && !stopping.load(std::memory_order_acquire)) {
            const cMutexLock decodeLock(&codecMutex);
            if (codecCtx) {
                const uint64_t producedEpoch = clearEpoch.load(std::memory_order_acquire);
                const size_t firstProducedIndex = pendingFrames.size();
                (void)DecodeOnePacket(queuedPacket.get(), pendingFrames);
                stampProducedEpoch(pendingFrames, firstProducedIndex, producedEpoch);
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
            trickSpeed.load(std::memory_order_acquire) != 0 && isTrickReverse.load(std::memory_order_relaxed);
        if (queuedPacket && reverseTrick) {
            trickEmptyDecodes = pendingFrames.empty() ? trickEmptyDecodes + 1 : 0;
        } else {
            trickEmptyDecodes = 0;
        }
        if (queuedPacket && reverseTrick && pendingFrames.empty() && trickEmptyDecodes >= 2) {
            const cMutexLock decodeLock(&codecMutex);
            if (codecCtx) {
                const uint64_t producedEpoch = clearEpoch.load(std::memory_order_acquire);
                const size_t firstProducedIndex = pendingFrames.size();
                DrainCodecAtEos(pendingFrames); // takes vaDriverMutex internally only where needed
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
                    filterChain.Reset();
                }
                stampProducedEpoch(pendingFrames, firstProducedIndex, producedEpoch);
            }
            if (pendingFrames.empty()) {
                const uint64_t holdMs = trickHoldMs.load(std::memory_order_relaxed);
                nextTrickFrameDue.store(cTimeMs::Now() + holdMs, std::memory_order_relaxed);
            }
            trickEmptyDecodes = 0; // codec was just flushed; restart the consecutive-empty counter
        }

        // --- Still-picture codec drain ---
        // Drain only when the queue is empty. A PAFF field pair is two packets (one decoded per
        // iteration); draining mid-pair sends EOS holding just the top field, which is discarded ->
        // frozen 1080i still. RequestCodecDrain() is published after FlushParser() queues any
        // trailing field, so sample the flag first and only then snapshot the queue state.
        const bool drainPending = codecDrainPending.load(std::memory_order_acquire);
        bool packetQueueEmpty = false;
        if (drainPending) {
            const cMutexLock lock(&packetMutex);
            packetQueueEmpty = packetQueue.empty();
        }
        if (drainPending && packetQueueEmpty && codecDrainPending.exchange(false, std::memory_order_acquire)) {
            drainedCodecAtEof = true;
            const cMutexLock decodeLock(&codecMutex);
            const uint64_t producedEpoch = clearEpoch.load(std::memory_order_acquire);
            const size_t firstProducedIndex = pendingFrames.size();
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
            stampProducedEpoch(pendingFrames, firstProducedIndex, producedEpoch);
        }

        // --- Hand off to the presentation thread ---
        // Push the batch across the handoff. Each frame is already stamped with the epoch it was
        // produced under (under codecMutex, see the producing operations above), so a Clear()/
        // SetTrickSpeed(0) that raced this decode is reflected per-frame and the presenter's
        // splice/purge discards superseded frames -- no separate recheck is needed here.
        // handoffMutex is a strict leaf, so this never blocks Clear()/EnqueueData().
        if (!pendingFrames.empty()) {
            const cMutexLock hl(&handoffMutex);
            size_t retainedNewFrames = pendingFrames.size();
            for (auto &frame : pendingFrames) {
                handoffQueue.push_back(std::move(frame));
            }
            // Final safety only: the backpressure wait at the loop top already bounds the producer;
            // drop-oldest if a wedged presenter somehow let the handoff exceed the cap so memory
            // stays bounded. (jitterOverflowSinceLog/Gate are decode-thread locals.)
            if (handoffQueue.size() > DECODER_RESERVE_HARD_CAP) [[unlikely]] {
                const size_t dropCount = handoffQueue.size() - DECODER_RESERVE_HARD_CAP;
                // Worst-case for the EOS bridge below: assume drops reach the just-pushed frames.
                // (Unreachable during the EOF drain, which keeps the reserve below the cap.)
                retainedNewFrames = (dropCount >= retainedNewFrames) ? 0 : retainedNewFrames - dropCount;
                for (size_t i = 0; i < dropCount; ++i) {
                    handoffQueue.pop_front();
                }
                jitterOverflowSinceLog += dropCount;
                if (jitterOverflowLogGate.Elapsed() >= 500) {
                    dsyslog("vaapivideo/decoder: handoff overflow -- dropped %zu frame(s), cap=%zu",
                            jitterOverflowSinceLog, DECODER_RESERVE_HARD_CAP);
                    jitterOverflowSinceLog = 0;
                    jitterOverflowLogGate.Set();
                }
            }
            // EOS-drain bridge: codecDrainPending is already cleared but the present thread only
            // republishes the reserve total next iteration, so a mediaplayer drain poll in that gap
            // could read a premature 0 and tear down before these final frames present. Bump the
            // count now (under handoffMutex, serialized with that store; transient over-count is safe).
            if (drainedCodecAtEof && retainedNewFrames > 0) {
                publishedDecodedReserveSize.fetch_add(retainedNewFrames, std::memory_order_relaxed);
            }
            handoffCondition.Broadcast();
        }
        pendingFrames.clear();
    }

    hasExited.store(true, std::memory_order_release);
}

// ============================================================================
// === PRESENTATION THREAD ===
// ============================================================================

auto cVaapiDecoder::PresentAction() -> void {
    dsyslog("vaapivideo/present: thread started");

    uint64_t noClockBlockedSinceMs{0}; ///< Walltime of the first no-clock hold; 0 = not holding.
    uint64_t lastDrainMs{0};
    cTimeMs jitterOverflowLogGate;   ///< Rate-limits the "jitterBuf overflow" syslog spam.
    size_t jitterOverflowSinceLog{}; ///< Frames dropped by the runaway guard since the last overflow log.
    cTimeMs futureDropLogGate;       ///< Rate-limits the "head too far in future" drop log (it can fire per frame).
    size_t futureDropSinceLog{};     ///< Future-head frames dropped since the last log line.

    while (!stopping.load(std::memory_order_acquire)) {
        // Consume a deferred Clear()/FlushForSeek() (this is the single consumer -- one exchange(0)
        // per iteration): reset the EMA/controller and drop the private jitterBuf. presentEpoch is
        // snapshotted AFTER the exchange so the splice below uses the freshest epoch and a
        // just-superseded handoff batch is dropped at the splice, not merely caught later by the
        // drain-loop epoch break.
        if (const int flushRequest = jitterFlushRequest.exchange(0, std::memory_order_acquire); flushRequest != 0) {
            ApplyDeferredJitterFlush(lastDrainMs, /*preserveSeekHint=*/flushRequest == 2);
            noClockBlockedSinceMs = 0;
        }
        presentEpoch = clearEpoch.load(std::memory_order_acquire);

        // A Clear()/FlushForSeek() that armed jitterFlushRequest *after* the exchange above would
        // otherwise let this iteration splice + drain using stale EMA/drop state for the first new
        // frame. Loop back to consume it before touching the queues. (Bounded: a real flush is
        // consumed on the next iteration; back-to-back Clears are user-paced, not a spin.)
        if (jitterFlushRequest.load(std::memory_order_acquire) != 0) {
            continue;
        }

        // Resolve a deferred Play()-out-of-trick once its cancellation grace expires. Queue-independent
        // (it fires even while the FF keyframe gate is starving the present thread of frames, which a
        // frame-driven exit could hang on), and ordered BEFORE the splice so the clearEpoch bump it
        // performs refreshes presentEpoch and purges the trick-era reserve in the splice below.
        (void)ResolvePendingTrickExit();

        // Splice the handoff into the private jitterBuf, dropping any frame the decode thread
        // produced under a SUPERSEDED (strictly older) epoch. clearEpoch is monotonic and
        // producedEpoch is always a past snapshot of it, so producedEpoch < presentEpoch means
        // pre-Clear material (discard), while producedEpoch > presentEpoch is a frame newer than this
        // iteration's stale snapshot (e.g. a SetTrickSpeed(0)/Clear bumped clearEpoch after the
        // snapshot at line above): KEEP it -- the drain-loop epoch guard re-validates it next
        // iteration; dropping it would silently lose valid current material. Broadcast handoffNotFull
        // so a backpressured decode thread resumes producing.
        {
            const cMutexLock hl(&handoffMutex);
            while (!handoffQueue.empty()) {
                if (handoffQueue.front()->producedEpoch < presentEpoch) {
                    handoffQueue.pop_front(); // superseded (pre-Clear/pre-trick-exit): discard
                    continue;
                }
                // Stop splicing when jitterBuf is full: leave the remainder in handoffQueue so the
                // producer backpressure (handoffQueue >= cap) throttles the decode thread instead of
                // the consumer-side runaway guard having to drop already-decoded frames.
                if (jitterBuf.size() >= DECODER_RESERVE_HARD_CAP) {
                    break;
                }
                jitterBuf.push_back(std::move(handoffQueue.front()));
                handoffQueue.pop_front();
            }
            handoffNotFull.Broadcast();
        }

        // Deterministic stale purge: SetTrickSpeed(0) bumps clearEpoch WITHOUT arming a flush, so
        // pre-trick-exit frames already in jitterBuf (strictly older epoch, spliced on a prior
        // iteration) must be dropped here rather than relying on the drain-loop break being reached.
        // clearEpoch only increases, so every stale frame is at the front. Use `<` (not `!=`) so a
        // kept newer-than-snapshot frame is not destroyed -- mirrors the splice retain rule above.
        while (!jitterBuf.empty() && jitterBuf.front()->producedEpoch < presentEpoch) {
            jitterBuf.pop_front();
        }

        // Runaway guard (consumer side): SkipStaleJitterFrames + the due-gate only drop BEHIND-clock
        // heads, so a stuck-but-valid audio clock with PTS marching ahead holds the due-gate closed
        // while the decode thread keeps producing -- jitterBuf would grow past the documented cap
        // (the handoff backpressure only bounds handoffQueue, which the splice above keeps draining).
        // Drop oldest to bound GPU surface retention; runs every iteration regardless of due-gate state.
        if (jitterBuf.size() > DECODER_RESERVE_HARD_CAP) [[unlikely]] {
            const size_t dropCount = jitterBuf.size() - DECODER_RESERVE_HARD_CAP;
            for (size_t i = 0; i < dropCount; ++i) {
                jitterBuf.pop_front();
            }
            jitterOverflowSinceLog += dropCount;
            if (jitterOverflowLogGate.Elapsed() >= 500) {
                dsyslog("vaapivideo/decoder: jitterBuf overflow -- dropped %zu frame(s), cap=%zu",
                        jitterOverflowSinceLog, DECODER_RESERVE_HARD_CAP);
                jitterOverflowSinceLog = 0;
                jitterOverflowLogGate.Set();
            }
        }

        auto *const ap = audioProcessor.load(std::memory_order_acquire);

        // Bulk-drop catastrophically stale heads (post-seek, long decode stall, trick-exit backlog).
        // Skip while devicePaused: ALSA is dropped, GetClock() extrapolates forward for
        // AUDIO_CLOCK_STALE_MS (1 s) before returning NOPTS, so the frozen head_pts appears
        // ~200 ms "behind" each iteration -- the bulk-drop would then drain the entire pre-pause
        // jitterBuf one frame at a time over the staleness window. The inner drain loop already
        // holds for devicePaused (no submits), but only Bulk-drop runs outside that hold.
        if (ap && !jitterBuf.empty() && !devicePaused.load(std::memory_order_acquire)) {
            SkipStaleJitterFrames(ap);
        }

        // Freerun and pendingDrops (soft-behind burst) bypass the due-gate.
        while (!jitterBuf.empty() && !stopping.load(std::memory_order_relaxed)) {
            // Clear() / SetTrickSpeed(0) raced this iteration: abandon the stale jitterBuf to
            // avoid scanning out pre-Clear frames or burning the new freerun budget on them.
            // The next iteration's jitterFlushRequest consumption empties jitterBuf.
            if (PresentEpochStale()) [[unlikely]] {
                break;
            }
            // Trick frames are paced by SubmitTrickFrame (nextTrickFrameDue), not the audio clock, so
            // skip the clock due-gate for them: a stale/absent trick-time clock (audio is not played at
            // trick speeds, so GetClock() goes NOPTS after ~1 s) would otherwise hold them in the
            // no-clock window or drop them via the future-head guard.
            const bool inTrick = trickSpeed.load(std::memory_order_acquire) != 0;

            // Device paused (cVaapiDevice::Freeze): hold the drain so the head's PTS doesn't drift
            // while ALSA is dropped. Without this hold, GetClock() goes stale ~1 s into the pause,
            // the no-clock-freerun path below fires, frames are submitted at vsync rate, and on
            // resume the head sits hundreds of ms ahead of the re-anchored audio clock -- a
            // video-ahead drain-stall loop, here from pause rather than a mux offset. Reset the
            // stall/miss trackers so the pause duration doesn't surface as a spurious drain stall or
            // miss spike when playback resumes.
            // NOT during slow trick: VDR calls Freeze() before slow FF/REW (so devicePaused is set),
            // but those frames must still be paced out by SubmitTrickFrame, so holding here would
            // freeze slow motion (fast trick has no preceding Freeze, so it is unaffected).
            if (devicePaused.load(std::memory_order_acquire) && !inTrick) [[unlikely]] {
                noClockBlockedSinceMs = 0;
                lastDrainMs = 0;
                break;
            }
            const bool consumingDrop = pendingDrops > 0;
            const bool canFreerun = freerunFrames.load(std::memory_order_relaxed) > 0;

            if (!inTrick && !consumingDrop && !canFreerun && ap) {
                // With a valid clock, hold until the head PTS is due. A missing clock (track switch,
                // ALSA reset, video-only stream) falls through to SyncAndSubmitFrame()'s no-clock freerun.
                const int64_t clock = ap->GetClock();
                if (clock == AV_NOPTS_VALUE) {
                    // Post-Clear/seek window, audio anchor incoming: hold so freerun does not submit
                    // pre-anchor video at VSync rate and leave the head far ahead of the clock once audio
                    // anchors. After DECODER_NO_CLOCK_HOLD_MS -- or once jitterBuf nears the cap, which a
                    // fast HW decoder would otherwise overflow during the hold -- fall through to no-clock
                    // submit WITHOUT resetting noClockBlockedSinceMs, so a stuck-NOPTS stream doesn't
                    // re-hold and stutter; jitterFlushRequest / the clock-valid else-branch re-arm it.
                    const uint64_t nowMs = cTimeMs::Now();
                    if (noClockBlockedSinceMs == 0) {
                        noClockBlockedSinceMs = nowMs;
                    }
                    const bool nearCap = jitterBuf.size() >= DECODER_RESERVE_HARD_CAP;
                    if (!nearCap && nowMs - noClockBlockedSinceMs < DECODER_NO_CLOCK_HOLD_MS) {
                        break;
                    }
                } else {
                    noClockBlockedSinceMs = 0;
                    const int64_t headPts = jitterBuf.front()->pts;
                    if (headPts != AV_NOPTS_VALUE) {
                        const int64_t dueIn = headPts - clock - SyncLatency90k(ap);
                        // Release the head once it is within the wake threshold: strict-due, or half a
                        // frame early to pre-fill an empty display queue (keeps depth at 1 and absorbs
                        // audio-clock vs VSync phase drift). PresentWakeThreshold90k() is the single
                        // source of truth shared with the waitMs sleep below, so the two can't disagree.
                        if (dueIn > PresentWakeThreshold90k()) {
                            // Magnitude guard: head >DECODER_DRAIN_FUTURE_MAX_MS ahead of the clock is a
                            // real PTS discontinuity (post-ATTA anchor mismatch, broadcast PCR break), not
                            // a normal startup offset -- drop it; catch-up handles the residual.
                            if (dueIn > DECODER_DRAIN_FUTURE_MAX_MS * PTS_TICKS_PER_MS) {
                                // Throttled: this drops one frame and re-checks the next, so a real
                                // discontinuity fires it per frame -- an unthrottled dsyslog here would
                                // both flood the journal and block this (present) hot thread on it.
                                ++futureDropSinceLog;
                                if (futureDropLogGate.Elapsed() >= 500) {
                                    dsyslog("vaapivideo/decoder: head too far in future (dueIn=%+lldms buf=%zu) -- "
                                            "dropping (%zu since last)",
                                            static_cast<long long>(dueIn / PTS_TICKS_PER_MS), jitterBuf.size(),
                                            futureDropSinceLog);
                                    futureDropSinceLog = 0;
                                    futureDropLogGate.Set();
                                }
                                jitterBuf.pop_front();
                                continue; // re-check new head
                            }
                            // Head ahead of the clock: hold (still frame) until it is due, no crawl --
                            // the one post-Clear freerun frame is already shown, so this freezes on it
                            // until audio catches up. A long hold (outside the corridor) zeroes
                            // lastDrainMs so the resume drain isn't flagged a starvation miss.
                            if (dueIn > DECODER_SYNC_CORRIDOR_90K) {
                                lastDrainMs = 0;
                            }
                            break;
                        }
                    }
                }
            }

            // Trick play paces frames at the trick hold (60-2000 ms), well above the 2*frameDur miss
            // threshold -- don't count those as drain misses, and reset lastDrainMs so the first
            // post-trick drain isn't flagged either. A sync-correction sleep (hard/soft-ahead,
            // WaitForAudioCatchUp) in the previous SyncAndSubmitFrame causes the same big gap, so
            // consume sleptInLastSubmit to skip exactly one miss. A still-frame hold resume also
            // carries lastDrainMs==0 (zeroed by the hold above), so the `> 0` guard skips it too.
            const bool consumedSleep = std::exchange(sleptInLastSubmit, false);
            const uint64_t nowMs = cTimeMs::Now();
            if (!inTrick && !consumedSleep && lastDrainMs > 0 &&
                static_cast<int>(nowMs - lastDrainMs) > outputFrameDurationMs.load(std::memory_order_relaxed) * 2) {
                ++drainMissCount;
            }
            lastDrainMs = inTrick ? 0 : nowMs;

            auto drainFrame = std::move(jitterBuf.front());
            jitterBuf.pop_front();
            (void)SyncAndSubmitFrame(std::move(drainFrame));
        }

        // Sleep until the head frame is due, or a new handoff batch / control-path change wakes us.
        // waitMs is clamped to [1,18] ms: the upper bound avoids 50 Hz vsync beating; the lower bound
        // (>=1) bounds CPU even when the head is perpetually "due". Do NOT gate the sleep on "!due" --
        // that is exactly what would spin at 100%.
        int waitMs = 10; // jitterBuf empty: deep-ish sleep, woken by the next handoff Broadcast.
        if (devicePaused.load(std::memory_order_acquire)) {
            // Freeze pins GetClock() to a valid (non-NOPTS) PTS and the drain loop holds (no submits),
            // so the clock path below would clamp waitMs to 1 and wake ~1 kHz for the whole pause.
            // Poll at the ceiling instead; SetDevicePaused(false) Broadcasts handoffCondition, so
            // resume latency stays bounded to one tick.
            waitMs = 18;
        } else if (!jitterBuf.empty()) {
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
                    const int64_t wakeThreshold = PresentWakeThreshold90k();
                    waitMs = (dueIn90k <= wakeThreshold)
                                 ? 1
                                 : std::clamp(static_cast<int>((dueIn90k - wakeThreshold) / PTS_TICKS_PER_MS), 1, 18);
                }
            }
        }

        // Don't sleep past a pending trick-exit deadline. The cancellation grace is ~one frame, so
        // clamp the wait to it (>=1 ms): the next iteration's loop-top ResolvePendingTrickExit then
        // fires on time even when no frames are flowing to wake us through the handoff Broadcast.
        if (deferredTrickExitPending.load(std::memory_order_acquire)) {
            const uint64_t dueMs = deferredTrickExitDueMs.load(std::memory_order_relaxed);
            const uint64_t nowMs = cTimeMs::Now();
            const int exitWaitMs = (dueMs > nowMs) ? std::clamp(static_cast<int>(dueMs - nowMs), 1, 18) : 1;
            waitMs = std::min(waitMs, exitWaitMs);
        }

        // Classic condvar wait under handoffMutex so a batch handed off (or a control-path Broadcast)
        // between the splice and here is never missed -- the decode push and Clear()/SetTrickSpeed()/
        // NotifyAudioChange()/SetDevicePaused() all Broadcast handoffCondition. Publish the total
        // decoded reserve (jitterBuf + live handoff) here, under the lock, for mediaplayer backpressure.
        {
            const cMutexLock hl(&handoffMutex);
            const size_t decodedReserve = jitterBuf.size() + handoffQueue.size();
            publishedDecodedReserveSize.store(decodedReserve, std::memory_order_relaxed);
            // Wake a backpressured decode thread now that the reserve total is republished: the splice
            // only broadcasts when it empties handoffQueue, so after a hard-ahead sleep / no-clock hold
            // drained jitterBuf below the cap, the producer would otherwise sleep out its 50 ms slice.
            const size_t reserveCap = (trickSpeed.load(std::memory_order_acquire) != 0) ? DECODER_TRICK_QUEUE_DEPTH
                                                                                        : DECODER_RESERVE_HARD_CAP;
            if (decodedReserve < reserveCap) {
                handoffNotFull.Broadcast();
            }
            if (handoffQueue.empty() && !stopping.load(std::memory_order_acquire)) {
                handoffCondition.TimedWait(handoffMutex, waitMs);
            }
        }
    }

    presentExited.store(true, std::memory_order_release);
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

// SD DVB streams routinely omit color_description: default BT.470BG (PAL) primaries AND limited (MPEG)
// range so scale_vaapi's BT.709/tv conversion neither desaturates the colour nor washes out the levels.
// SD video -- MPEG-2 and SD H.264 alike -- is BT.601 limited-range by convention.
static auto ApplySdColorDefaults(AVFrame *frame) noexcept -> void {
    if (frame->height > 576) {
        return;
    }
    if (frame->colorspace == AVCOL_SPC_UNSPECIFIED) {
        frame->colorspace = AVCOL_SPC_BT470BG;
    }
    if (frame->color_range == AVCOL_RANGE_UNSPECIFIED) {
        frame->color_range = AVCOL_RANGE_MPEG;
    }
}

auto cVaapiDecoder::ApplyContainerColorHints(AVFrame *frame) const noexcept -> void {
    // Fill only UNSPECIFIED fields so an explicit in-bitstream VUI always wins over the container.
    if (frame->color_primaries == AVCOL_PRI_UNSPECIFIED && hintColorPrimaries != AVCOL_PRI_UNSPECIFIED) {
        frame->color_primaries = hintColorPrimaries;
    }
    if (frame->color_trc == AVCOL_TRC_UNSPECIFIED && hintColorTransfer != AVCOL_TRC_UNSPECIFIED) {
        frame->color_trc = hintColorTransfer;
    }
    if (frame->colorspace == AVCOL_SPC_UNSPECIFIED && hintColorSpace != AVCOL_SPC_UNSPECIFIED) {
        frame->colorspace = hintColorSpace;
    }
    if (frame->color_range == AVCOL_RANGE_UNSPECIFIED && hintColorRange != AVCOL_RANGE_UNSPECIFIED) {
        frame->color_range = hintColorRange;
    }
}

auto cVaapiDecoder::ResolveHdrInfo(const AVFrame *frame) const noexcept -> HdrStreamInfo {
    HdrStreamInfo info = ExtractHdrInfo(frame);
    // Backfill mastering/content-light from the container only when the frame lacked it (VP9/AV1
    // carry it in the MKV/MP4 header, not the bitstream); frame SEI wins.
    if (info.kind != StreamHdrKind::Sdr) {
        if (!info.hasMasteringDisplay && hintHasMasteringDisplay) {
            info.hasMasteringDisplay = true;
            info.mastering = hintMasteringDisplay;
        }
        if (!info.hasContentLight && hintHasContentLight) {
            info.hasContentLight = true;
            info.contentLight = hintContentLight;
        }
    }
    return info;
}

auto cVaapiDecoder::FilterAndAppendDecodedFrame(std::vector<std::unique_ptr<VaapiFrame>> &outFrames) -> void {
    ApplyContainerColorHints(decodedFrame.get());
    ApplySdColorDefaults(decodedFrame.get());

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
    const auto frameDurMs = static_cast<int64_t>(outputFrameDurationMs.load(std::memory_order_relaxed));
    for (size_t i = 0; i < newOutCount; ++i) {
        outFrames.at(prevOutCount + i)->pts =
            (sourcePts != AV_NOPTS_VALUE && i > 0)
                ? sourcePts + (frameDurMs * PTS_TICKS_PER_MS * static_cast<int64_t>(i))
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
                // Mark the rebuild compact: the chain is unchanged across a recovery, so a corruption
                // burst logs one "filter rebuilt" line per reset, not the full 3-line diagnostic.
                dsyslog("vaapivideo/decoder: send_packet failed: %s -- flushing for recovery", AvErr(ret).data());
                const cMutexLock vaLock(&display->GetVaDriverMutex());
                avcodec_flush_buffers(codecCtx.get());
                filterChain.Reset();
                filterCompactRebuildPending.store(true, std::memory_order_release);
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

            // Drop concealment frames: on a corrupt stream the AMD VAAPI h264 decoder emits a
            // full-green frame (zeroed chroma) that the still-frame hold would freeze on. Skip it to
            // keep the last good picture; clean streams never set these flags.
            if (decodedFrame->decode_error_flags != 0 || (decodedFrame->flags & AV_FRAME_FLAG_CORRUPT) != 0)
                [[unlikely]] {
                continue; // re-check the receive queue for the next (hopefully clean) frame
            }

            // Before the log + filter/HDR build below, so all of them see the corrected colorimetry.
            ApplyContainerColorHints(decodedFrame.get());

            if (!hasLoggedFirstFrame.exchange(true, std::memory_order_relaxed)) {
                const char *fmtName = av_get_pix_fmt_name(static_cast<AVPixelFormat>(decodedFrame->format));
                const char *swFmtName = av_get_pix_fmt_name(ResolveSwPixFmt(decodedFrame.get()));
                const HdrStreamInfo hdr = ResolveHdrInfo(decodedFrame.get());
                // codecCtx->profile is populated by FFmpeg during SPS/VPS parsing, which
                // is guaranteed to have happened before the first frame surfaces. At
                // OpenCodec() time this field is still AV_PROFILE_UNKNOWN and logging it
                // there is meaningless -- the authoritative value lands here.
                const AVCodec *openedDecoder = codecCtx ? codecCtx->codec : nullptr;
                const char *profileName =
                    openedDecoder ? av_get_profile_name(openedDecoder, codecCtx->profile) : nullptr;
                // Match the filter chain's verdict (stream hint folded in) so the log can't read
                // "progressive" while the chain deinterlaces. See FrameNeedsDeinterlace().
                isyslog("vaapivideo/decoder: first frame %dx%d %s (sw=%s, profile=%s)%s", decodedFrame->width,
                        decodedFrame->height, fmtName ? fmtName : "unknown", swFmtName ? swFmtName : "unknown",
                        profileName ? profileName : "unknown",
                        FrameNeedsDeinterlace(decodedFrame.get(), streamInterlaced) ? " interlaced" : "");
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

            ApplySdColorDefaults(decodedFrame.get());

            // No decode-thread catch-up fast path: frames always flow through the handoff and the
            // present-thread controller drops what it must (catchingUp/catchUpDrops are present-thread
            // state). The reserve absorbs VPP cost, so the lost seek-backlog speedup is acceptable.

            // vaDriverMutex serializes VAAPI access against the display thread's DRM PRIME export.
            // Filter graph is built lazily on first frame and after each Clear() or ScaleVideo() change.
            const int64_t sourcePts = decodedFrame->pts;
            const size_t prevOutCount = outFrames.size();
            {
                const cMutexLock vaLock(&display->GetVaDriverMutex());

                // ScaleVideo() target changed: rebuild the VPP graph for the new size. jitterBuf
                // is kept; old-sized frames keep painting at the old scanout rect until a matching
                // fb arrives (PresentBuffer promotes videoRect there). compactLog distinguishes
                // this rebuild from a Clear/channel-switch rebuild so logging stays informative.
                bool compactLog = false;
                if (videoRectDirty.exchange(false, std::memory_order_acq_rel)) {
                    filterChain.Reset();
                    compactLog = true;
                }
                // FlushForSeek requested a compact log on this rebuild (the chain parameters
                // are unchanged across a seek so the full diagnostic is just noise; the new
                // "filter rebuilt -> WxH" one-liner is enough to confirm the reset happened).
                if (filterCompactRebuildPending.exchange(false, std::memory_order_acq_rel)) {
                    compactLog = true;
                }
                if (!filterChain.IsBuilt()) {
                    (void)InitFilterGraph(decodedFrame.get(), compactLog);
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
            const auto frameDurMs = static_cast<int64_t>(outputFrameDurationMs.load(std::memory_order_relaxed));
            for (size_t i = 0; i < newOutCount; ++i) {
                outFrames.at(prevOutCount + i)->pts =
                    (sourcePts != AV_NOPTS_VALUE && i > 0)
                        ? sourcePts + (frameDurMs * PTS_TICKS_PER_MS * static_cast<int64_t>(i))
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
    if (trickSpeed.load(std::memory_order_acquire) != 0 && isTrickReverse.load(std::memory_order_relaxed) &&
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

[[nodiscard]] auto cVaapiDecoder::InitFilterGraph(AVFrame *firstFrame, bool compactLog) -> bool {
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
    const HdrStreamInfo hdrInfo = ResolveHdrInfo(firstFrame);
    const bool hdrPassthrough = ShouldUseHdrPassthrough(hdrInfo);
    display->SetHdrOutputState(hdrPassthrough ? hdrInfo : HdrStreamInfo{});

    cVideoFilterChain::BuildParams params;
    params.codecId = currentCodecId;
    params.streamInterlaced = streamInterlaced;
    params.fpsNum = codecCtx->framerate.num;
    params.fpsDen = codecCtx->framerate.den;
    params.hwFramesCtx = codecCtx->hw_frames_ctx;
    params.hwDeviceRef = vaapiContext->hwDeviceRef;
    // Target the staged ScaleVideo() rect (not the active one) so the next fb already fits and
    // KMS scanout stays 1:1.
    const cRect targetRect = display->GetTargetVideoRect();
    params.outputWidth = static_cast<uint32_t>(targetRect.Width());
    params.outputHeight = static_cast<uint32_t>(targetRect.Height());
    params.outputRefreshHz = display->GetOutputRefreshRate();
    params.hdrPassthrough = hdrPassthrough;
    params.hdrInfo = hdrInfo;
    params.hasDenoise = vaapiContext->caps.vppDenoise;
    params.hasSharpness = vaapiContext->caps.vppSharpness;
    params.deinterlaceMode = vaapiContext->caps.deinterlaceMode;
    params.deinterlaceModeMask = vaapiContext->caps.deinterlaceModeMask;
    // Post-processing policies; re-read on every filter-graph rebuild so a setup change applies live.
    params.userDeint = vaapiConfig.deinterlaceMode.load(std::memory_order_relaxed);
    params.denoise = vaapiConfig.denoiseMode.load(std::memory_order_relaxed);
    params.scale = vaapiConfig.scaleMode.load(std::memory_order_relaxed);
    params.sharpen = vaapiConfig.sharpenMode.load(std::memory_order_relaxed);
    params.trickMode = trickSpeed.load(std::memory_order_relaxed) != 0;
    params.stillPicture = stillPictureMode.load(std::memory_order_relaxed);
    params.compactLog = compactLog;
    // Manual zoom: resolve the active cycle stop (0 = Off) to an equal per-side crop fraction. The
    // level is a zoom-in factor in tenths-of-% (344 = +34.4% = magnification z=1.344); the per-side
    // crop that yields that magnification once the kept region refills the screen is (1 - 1/z)/2.
    const int zoomStop = vaapiConfig.zoomActive.load(std::memory_order_relaxed);
    if (zoomStop >= 1 && zoomStop <= CONFIG_ZOOM_PRESET_COUNT) {
        const int level = vaapiConfig.zoomLevel[zoomStop - 1].load(std::memory_order_relaxed);
        if (level > 0) {
            const double z = 1.0 + (static_cast<double>(level) / 1000.0);
            const double cropFraction = (1.0 - (1.0 / z)) / 2.0;
            params.cropH = cropFraction;
            params.cropV = cropFraction;
        }
    }

    if (!filterChain.Build(firstFrame, params)) {
        return false;
    }

    // Relaxed: standalone scalar read by the present-thread sync math; no happens-before role.
    outputFrameDurationMs.store(filterChain.GetOutputFrameDurationMs(), std::memory_order_relaxed);
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

[[nodiscard]] auto cVaapiDecoder::ResolvePendingTrickExit() -> bool {
    // Present thread only. RequestTrickExit() (VDR fires Play() as the middle leg of REW->PLAY->REW)
    // arms a deferred exit with a one-frame cancellation grace; this resolves it once that grace
    // expires WITHOUT waiting for a frame to reach SyncAndSubmitFrame -- in FF the keyframe gate can
    // starve the present thread of frames, so a frame-driven exit could hang indefinitely. Called
    // every present iteration (loop top, queue-independent) and again from SyncAndSubmitFrame as a
    // frame arrives; the exchange below makes the two callers race to a single winner.
    if (!deferredTrickExitPending.load(std::memory_order_acquire)) {
        return false;
    }
    const uint64_t dueMs = deferredTrickExitDueMs.load(std::memory_order_relaxed);
    if (dueMs != 0 && cTimeMs::Now() < dueMs) {
        return false; // cancellation grace not yet expired -- a racing SetTrickSpeed() may supersede.
    }
    if (!deferredTrickExitPending.exchange(false, std::memory_order_acq_rel)) {
        return false; // the other caller resolved it first this iteration.
    }
    deferredTrickExitDueMs.store(0, std::memory_order_relaxed);

    {
        // codecMutex -> parserMutex: same atomicity as SetTrickSpeed(), so the codec flush, parser
        // recreate, clearEpoch bump, and trickSpeed publication never race a concurrent EnqueueData().
        const cMutexLock decodeLock(&codecMutex);
        const cMutexLock parseLock(&parserMutex);
        if (trickSpeed.load(std::memory_order_acquire) == 0) {
            return false; // a SetTrickSpeed(0) won the race and already normalized; nothing to do.
        }

        // FF/REW are non-contiguous (keyframe-only / backward GOPs): flush codec+parser so stale partial
        // NALs from the trick stream can't corrupt the first normal-play frames. Slow modes stayed
        // contiguous and need no flush -- mirrors SetTrickSpeed()'s needsFlush entry rule.
        const bool flushCodec =
            isTrickFastForward.load(std::memory_order_relaxed) || isTrickReverse.load(std::memory_order_relaxed);
        if (flushCodec) {
            DrainQueue();
            {
                // vaDriverMutex: avcodec_flush_buffers + filterChain.Reset() issue VA API calls that
                // race av_hwframe_map on the same VADisplay (iHD driver crash).
                const cMutexLock vaLock(&display->GetVaDriverMutex());
                if (codecCtx) {
                    avcodec_flush_buffers(codecCtx.get());
                }
                filterChain.Reset();
                if (decodedFrame) {
                    av_frame_unref(decodedFrame.get());
                }
                if (filteredFrame) {
                    av_frame_unref(filteredFrame.get());
                }
            }
            if (parserCtx && currentCodecId != AV_CODEC_ID_NONE) {
                parserCtx.reset(av_parser_init(currentCodecId));
            }
        } else if (display && filterChain.IsBuilt()) {
            // Slow-forward deferred exit (contiguous, no codec flush): still rebuild the filter so normal
            // play restores the full-quality chain instead of staying on the light trick/bob chain.
            const cMutexLock vaLock(&display->GetVaDriverMutex());
            filterChain.Reset();
        }

        // Publish normal-play state. Flags before the trickSpeed release-store so an acquire reader
        // that observes speed==0 sees a consistent (mode, direction). Epoch BEFORE NOPTS (Clear-race
        // guard), exactly like SetTrickSpeed()'s generation-boundary publish.
        isTrickFastForward.store(false, std::memory_order_relaxed);
        isTrickReverse.store(false, std::memory_order_relaxed);
        prevTrickPts.store(AV_NOPTS_VALUE, std::memory_order_relaxed);
        trickMultiplier.store(0, std::memory_order_relaxed);
        trickHoldMs.store(0, std::memory_order_relaxed);
        nextTrickFrameDue.store(cTimeMs::Now(), std::memory_order_relaxed);
        clearEpoch.fetch_add(1, std::memory_order_release);
        lastPts.store(AV_NOPTS_VALUE, std::memory_order_release);
        trickSpeed.store(0, std::memory_order_release);
    }

    // Present-private controller reset (we ARE the present thread): refresh the epoch snapshot, drop
    // the trick-era jitterBuf, reset the EMA, and arm freerun while the audio clock re-anchors.
    presentEpoch = clearEpoch.load(std::memory_order_acquire);
    jitterBuf.clear();
    pendingDrops = 0;
    ResetSmoothedDelta();
    freerunFrames.store(DECODER_SYNC_FREERUN_FRAMES, std::memory_order_relaxed);
    syncLogPending.store(true, std::memory_order_relaxed);

    // Purge frames the decode thread already handed off under the old epoch, then wake it (its
    // backpressure cap shrinks back to the normal reserve) and republish the reserve total.
    {
        const cMutexLock hl(&handoffMutex);
        while (!handoffQueue.empty() && handoffQueue.front()->producedEpoch < presentEpoch) {
            handoffQueue.pop_front();
        }
        publishedDecodedReserveSize.store(jitterBuf.size() + handoffQueue.size(), std::memory_order_relaxed);
        handoffNotFull.Broadcast();
    }

    if (display) {
        display->SetTrickActive(false); // re-enable the underrun detector now that pacing is normal.
    }
    return true;
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
    if (PresentEpochStale()) [[unlikely]] {
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
    // Present-thread chokepoint for lastPts (single writer now that the catch-up VPP bypass is gone).
    // presentEpoch (present-loop iter top) compared against clearEpoch (bumped by Clear() /
    // SetTrickSpeed(0)) gates the publish; recheck-undo collapses the ns-scale window where Clear()
    // fires between check and store. Without it a stale pre-Clear PTS would clobber Clear()'s NOPTS
    // reset and surface via cVaapiDevice::GetSTC().
    if (PresentEpochStale()) [[unlikely]] {
        return;
    }
    lastPts.store(pts, std::memory_order_release);
    if (PresentEpochStale()) [[unlikely]] {
        lastPts.store(AV_NOPTS_VALUE, std::memory_order_release);
    }
}

[[nodiscard]] auto cVaapiDecoder::SyncLatency90k(const cAudioProcessor *ap) const noexcept -> int64_t {
    // User knob (PCM or passthrough) + 1-frame pipeline constant. The 1-frame tail models the
    // dominant scanout delay (commit + page flip) for an empty prerender cache (throughput-bound
    // regime): the baseline settles at -1*frameDur (~-20 ms @ 50 fps), well inside CORRIDOR (50 ms),
    // so 10-15 ms/s crystal/pipeline drift takes a long time to breach. Gate-bound (live TV, replay
    // backlog) is unaffected: the gate releases at dueIn <= halfFrame, so the EMA still settles at
    // +halfFrame regardless of latency. Operator knob still shifts the bias on top.
    const int latencyMs = (ap && ap->IsPassthrough()) ? vaapiConfig.passthroughLatency.load(std::memory_order_relaxed)
                                                      : vaapiConfig.pcmLatency.load(std::memory_order_relaxed);
    return (static_cast<int64_t>(latencyMs) +
            static_cast<int64_t>(outputFrameDurationMs.load(std::memory_order_relaxed))) *
           PTS_TICKS_PER_MS;
}

auto cVaapiDecoder::PresentEpochStale() const noexcept -> bool {
    return clearEpoch.load(std::memory_order_acquire) != presentEpoch;
}

auto cVaapiDecoder::PresentWakeThreshold90k() const noexcept -> int64_t {
    // Pre-fill when the display prerender queue is empty: release the head half a frame early (=
    // one full frameDur above strict-due) to keep depth at 1 and absorb audio-clock vs VSync phase
    // drift; otherwise wake at strict-due (halfFrame). The drain loop and the waitMs sleep both call
    // this so their thresholds can never silently disagree.
    const int64_t frameDur =
        static_cast<int64_t>(outputFrameDurationMs.load(std::memory_order_relaxed)) * PTS_TICKS_PER_MS;
    const int64_t halfFrame = frameDur / 2;
    return (display && display->PendingDepth() == 0) ? frameDur : halfFrame;
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
    stableDelta90k = AV_NOPTS_VALUE;
    stableDeltaCapturedMs = 0;
    // NOTE: lastCatchUpExitMs is intentionally NOT reset here. This function fires from many
    // paths (no-clock freerun, hard-behind, catch-up entry, ...) and clearing the marker on
    // each call would make the "sinceCatchUp" diagnostic always report 0. Cleared explicitly
    // by Clear() / FlushForSeek (via the cVaapiDecoder member-state reset path) instead.
}

auto cVaapiDecoder::ApplyDeferredJitterFlush(uint64_t &lastDrainMs, bool preserveSeekHint) noexcept -> void {
    // Fast-start hint: FlushForSeek requests carrying the converged smoothedDelta across the
    // flush. The GPU-vs-audio-clock offset is a property of the pipeline (decode + VPP + KMS
    // queue latency vs ALSA hw_ptr), unaffected by playback position, so the pre-seek
    // steady-state is the right catch-up exit target AND the right EMA seed -- saving the
    // ~50-frame warmup on every seek. Cleared by plain Clear() (channel switch / new file
    // / device reset): different content can have different decode latency characteristics.
    //
    // preserveSeekHint travels with the flush request via jitterFlushRequest (0/1/2 encoding),
    // not a separate atomic, so racing back-to-back Clear() / FlushForSeek() can't cross-
    // pollinate the policy. Earlier design with a separate `preserveSeekHintOnFlush` atomic
    // had a race: an in-flight older flush could consume the *next* FlushForSeek's preserve
    // store before that FlushForSeek armed its own request, then the new request would run
    // with preserve=false and drop the hint.
    //
    // Source priority: prefer stableDelta90k (pre-correction snapshot) over smoothedDelta90k.
    // The soft-/hard-ahead path applies a predictive `-=` to smoothedDelta after every sleep,
    // making it a transient value during the EMA recovery window (~3 s on low-fps content).
    // A seek that lands inside that window would otherwise grab the mid-recovery value as the
    // hint, biasing the post-seek EMA seed toward a temporary low. stableDelta is captured
    // at the trigger site *before* the `-=`, so it always reflects the long-term offset that
    // the controller had measured. Falls back to smoothedDelta when no correction has fired
    // yet (no snapshot exists) -- in that case smoothedDelta itself is clean.
    //
    // Invalid-EMA preservation: when smoothedDeltaValid=false, we PRESERVE the existing
    // seekHintDelta90k rather than overwriting it. Motivating case: a rapid back-to-back seek
    // burst where flush N+1 fires before any frame reseeds the EMA from flush N's capture.
    // Flush N captured a fresh valid hint, then its own ResetSmoothedDelta cleared the EMA --
    // so at flush N+1 smoothedDeltaValid is false even though the stored hint is current.
    // Without this guard, flush N+1 would overwrite the hint with AV_NOPTS_VALUE; the
    // subsequent catch-up then ran without /hint and fell back to +halfFrame. Plain Clear()
    // (content boundary) still drops the hint because preserveSeekHint=false on that request
    // takes the outer `else` branch.
    if (preserveSeekHint) {
        if (smoothedDeltaValid) {
            // Source selection. Prefer stableDelta90k only when it's *fresh* -- a snapshot
            // older than DECODER_SYNC_HINT_MAX_AGE_MS predates the soft-correction cooldown
            // window, which means no further correction has refreshed it. In that case the
            // EMA has been quietly running on its own and smoothedDelta90k is the better
            // signal; a single old snapshot from a different operating point shouldn't
            // dominate every future seek.
            const uint64_t nowMs = cTimeMs::Now();
            const bool stableHintFresh = stableDelta90k != AV_NOPTS_VALUE && stableDeltaCapturedMs != 0 &&
                                         nowMs - stableDeltaCapturedMs <= DECODER_SYNC_HINT_MAX_AGE_MS;
            const int64_t rawHint90k = stableHintFresh ? stableDelta90k : smoothedDelta90k;
            // Clamp at capture so catch-up target AND EMA seed see the same bounded value.
            // If the unclamped hint exceeded +CORRIDOR, the EMA seed would make
            // smoothedDeltaValid=true with a value that immediately trips the soft-ahead
            // check on the very next frame -- effectively a double correction. Clamping
            // here lands the seed at the corridor edge; natural pacing handles any further
            // drift toward the true offset.
            seekHintDelta90k =
                std::clamp<int64_t>(rawHint90k, -DECODER_SYNC_CORRIDOR_90K + 1, DECODER_SYNC_CORRIDOR_90K);
        }
        // else: leave seekHintDelta90k untouched -- any prior valid hint is still current
        // (rapid back-to-back seek burst where the EMA was reset before reseed).
    } else {
        seekHintDelta90k = AV_NOPTS_VALUE;
        // Plain Clear() (content boundary): cancel any pending compact-log request that a
        // preceding FlushForSeek may have left armed. Same race shape as the seek hint -- a
        // FlushForSeek sets filterCompactRebuildPending=true, then a Clear() overrides the flush
        // (jitterFlushRequest 2->1) before the decode thread rebuilds. Without this the
        // channel-switch / new-content rebuild would emit the terse one-line "filter rebuilt"
        // diagnostic instead of the full graph dump. Binding it to the flush policy here keeps
        // the verbose diagnostic for real content boundaries.
        filterCompactRebuildPending.store(false, std::memory_order_release);
    }
    ResetSmoothedDelta();
    jitterBuf.clear();
    pendingDrops = 0;
    lastDrainMs = 0;
    // "sinceCatchUp" diagnostic baseline: only meaningful within one playback session.
    lastCatchUpExitMs = 0;
    catchUpLogThisCycle = false;
    // Catch-up log-throttle run state. Reset it ONLY on a content boundary (plain Clear / trick-exit,
    // preserveSeekHint=false): pre-boundary cycles describe a different pipeline, and zeroing the
    // aggregates keeps the run-boundary "cycling settled" summary from misattributing them to the new
    // session. Across a same-content SEEK (preserveSeekHint=true) KEEP the run state: a rapid
    // forward/back-seek burst then registers as ONE cycling run and is throttled. Otherwise every
    // seek's post-flush warmup catch-up logs a full entry+exit pair on this (present) hot thread --
    // a journal flood that also blocks the drain on syslog I/O exactly while the user is scrubbing.
    if (!preserveSeekHint) {
        lastCatchUpEntryMs = 0;
        suppressedCatchUpCycles = 0;
        suppressedCatchUpDrops = 0;
        suppressedCatchUpWallMs = 0;
        nextCatchUpSummaryMs = 0;
        // Sync-stats accumulators: zero them so the new session's first post-warmup log can't report
        // drops/skips/misses carried over from before this content boundary. They survive a Clear()
        // because LogSyncStats only resets them after warmup completes, which a content switch
        // restarts -- hence a stale pre-switch count would otherwise surface as a phantom first-log
        // drop spike. A seek keeps them: its post-flush catch-up drops belong to this session.
        drainMissCount = 0;
        syncDropSinceLog = 0;
        syncSkipSinceLog = 0;
        rawDeltaSumSinceLog90k = 0;
        rawDeltaCountSinceLog = 0;
    }
}

[[nodiscard]] auto cVaapiDecoder::BeginCatchUpLogCycle() noexcept -> bool {
    const uint64_t nowMs = cTimeMs::Now();
    // Cycling run detection keys off the gap to the *previous entry* (logged or suppressed),
    // not the previous logged entry. Keying off the logged entry (the earlier design) forced a
    // log every THROTTLE_MS and made the periodic "sustained" summary unreachable, because
    // suppression reset before the longer summary interval could elapse.
    const bool runActive = lastCatchUpEntryMs != 0 && (nowMs - lastCatchUpEntryMs) < DECODER_CATCHUP_LOG_THROTTLE_MS;
    lastCatchUpEntryMs = nowMs;
    if (runActive) {
        // Inside a cycling run: suppress this entry (and its exit). The exit path aggregates
        // the cycle and emits the periodic "sustained" summary.
        catchUpLogThisCycle = false;
        return false;
    }
    // Run boundary: either the first catch-up ever, or cycling stopped long enough that this is a
    // fresh isolated event. If a run just ended, emit its final "settled" summary before logging
    // this entry so the entry line marks the resumption of normal cadence.
    if (suppressedCatchUpCycles > 0) {
        isyslog("vaapivideo/decoder: catch-up cycling settled: %d additional cycles drops=%d wallSum=%llums",
                suppressedCatchUpCycles, suppressedCatchUpDrops,
                static_cast<unsigned long long>(suppressedCatchUpWallMs));
        suppressedCatchUpCycles = 0;
        suppressedCatchUpDrops = 0;
        suppressedCatchUpWallMs = 0;
    }
    nextCatchUpSummaryMs = 0; // re-armed by the next run's first suppressed exit
    catchUpLogThisCycle = true;
    return true;
}

[[nodiscard]] auto cVaapiDecoder::SubmitIfCurrent(std::unique_ptr<VaapiFrame> frame) -> bool {
    // Clear-race guard for paths that sleep before submit: a Clear() during the sleep makes
    // the frame's PTS belong to the old epoch. Drop silently (returning true so callers don't
    // count it as a submit failure).
    if (PresentEpochStale()) [[unlikely]] {
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
        // Fast-start: a FlushForSeek (same session, same pipeline) has handed us the
        // pre-seek converged offset. Seed the EMA directly and skip the 50-sample warmup --
        // the catch-up controller can target the steady-state from the first frame.
        //
        // One-shot: clear the hint here so subsequent ResetSmoothedDelta paths (no-clock
        // freerun, catch-up exit much later, hard-behind, etc.) warm up from real samples
        // instead of re-applying the same stale pre-seek value forever. The catch-up
        // *exit* before this point may have read the hint as its drop target, which is
        // fine -- that's the same one shot, just a step earlier in the sequence.
        if (seekHintDelta90k != AV_NOPTS_VALUE) {
            smoothedDelta90k = seekHintDelta90k;
            seekHintDelta90k = AV_NOPTS_VALUE;
            smoothedDeltaValid = true;
            warmupSampleCount = 0;
            warmupSampleSum90k = 0;
            emaResidual90k = 0;
            return;
        }
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
    // aq via the lock-free GetQueueSizeRelaxed(): the mutexed GetQueueSize() would park this (present)
    // thread behind WritePcmToAlsa's EAGAIN spin (audio mutex held while the ALSA ring is full under a
    // bursty replay feed) -- a hot-path stall that surfaced as periodic soft/hard-behind drops.
    dsyslog("vaapivideo/decoder: sync d=%+lld.%01lldms avg=%+lld.%01lldms lat=%lldms buf=%zu aq=%zu miss=%d "
            "drop=%d skip=%d",
            meanTenths / 10, std::abs(meanTenths % 10), avgTenths / 10, std::abs(avgTenths % 10),
            static_cast<long long>(latency90k / PTS_TICKS_PER_MS), jitterBuf.size(), ap->GetQueueSizeRelaxed(),
            drainMissCount, syncDropSinceLog, syncSkipSinceLog);
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
            // Time-since-catch-up-exit surfaces the cascade pattern: when stale-jitter fires
            // within ~2 s of a catch-up exit, the GPU/audio drift is exceeding the catch-up
            // follow-up margin and the system is stuck in a recovery loop. ms=0 means
            // catch-up has not run yet this session. Throttled: a rapid-seek burst triggers this
            // per seek, and an unthrottled dsyslog on this (present) hot thread would flood the
            // journal and block the drain; suppressed drops fold into the next line's tail count.
            staleJitterDropsSinceLog += dropped;
            if (staleJitterLogGate.Elapsed() >= 500) {
                const uint64_t sinceCatchUpMs = (lastCatchUpExitMs != 0) ? (cTimeMs::Now() - lastCatchUpExitMs) : 0;
                dsyslog("vaapivideo/decoder: sync drop (stale-jitter bulk) count=%d firstRaw=%+lldms buf=%zu "
                        "sinceCatchUp=%llums (%d since last)",
                        dropped, static_cast<long long>(firstDelta90k / PTS_TICKS_PER_MS), jitterBuf.size(),
                        static_cast<unsigned long long>(sinceCatchUpMs), staleJitterDropsSinceLog);
                staleJitterDropsSinceLog = 0;
                staleJitterLogGate.Set();
            }
        }
    }
}

[[nodiscard]] auto cVaapiDecoder::SyncAndSubmitFrame(std::unique_ptr<VaapiFrame> frame) -> bool {
    // Audio-master A/V sync gate. See AVSYNC.md for the full state diagram.

    if (!frame || !display) [[unlikely]] {
        return false;
    }

    const int64_t pts = frame->pts;
    // Snapshot once per frame: the decode thread may rewrite outputFrameDurationMs on a concurrent
    // filter rebuild. A one-frame-stale value only perturbs this frame's pacing; the EMA reconverges.
    const int frameDurMs = outputFrameDurationMs.load(std::memory_order_relaxed);

    // --- Trick mode ---
    if (trickSpeed.load(std::memory_order_acquire) != 0) {
        // A deferred Play()-out-of-trick whose grace has expired resolves here too (the loop-top call
        // may not have fired since this frame became due): ResolvePendingTrickExit() runs the full
        // generation-boundary teardown and leaves trickSpeed==0, so this frame falls through as the
        // first normal-play re-anchor. Until the grace expires it returns false -- keep pacing trick.
        if (!ResolvePendingTrickExit()) {
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
            // Target selection:
            //   - With a fast-start hint (post-seek, same pipeline): use the converged pre-seek
            //     offset directly. The GPU/audio offset doesn't change with playback position,
            //     so landing right at the steady-state avoids any post-catch-up settling drift.
            //   - Without a hint (channel switch / first play): aim head one halfFrame *past*
            //     clock. Without the positive margin, any GPU/audio drift (drain rate slightly
            //     < audio rate) immediately erodes the catch-up's progress: head drifts back
            //     into the -HARD_THRESHOLD zone within a couple of seconds, SkipStaleJitterFrames
            //     drops it, catch-up re-enters, repeat. Observed in the log as ~1 s cycles of
            //     "stale-jitter / catch-up entered / catch-up complete" for 13 s after a seek,
            //     with firstRaw improving only ~40 ms per second (= one 25 fps source frame).
            //     Pushing head positive past clock buys ~30 ms of drift headroom per cycle.
            const int64_t frameDur90k = static_cast<int64_t>(frameDurMs) * PTS_TICKS_PER_MS;
            const int64_t halfFrame90k = frameDur90k / 2;
            const bool haveHint = (seekHintDelta90k != AV_NOPTS_VALUE);
            // Clamp the hint inside (-CORRIDOR, +CORRIDOR] -- the soft-corridor trip threshold.
            // Lower bound stops a degenerate negative hint from re-entering the catch-up zone.
            // Upper bound is CORRIDOR (not halfFrame): on low-fps content (e.g. 15 fps uneven
            // cadence on 50 Hz refresh) the natural steady-state offset is well above halfFrame
            // (~+60 ms with halfFrame=33 ms) because soft-ahead skipping is part of the
            // pipeline's normal pacing. Clamping to halfFrame would land the post-catch-up
            // state below the natural offset, forcing the EMA to drift back up over several
            // seconds; CORRIDOR lets the hint reach the soft-ahead trip threshold where natural
            // pacing resumes immediately. The cold-start fallback (no hint) still uses
            // +halfFrame: without a measured offset we don't know the pipeline's true steady-state.
            const int64_t hintClamped = haveHint ? std::clamp<int64_t>(seekHintDelta90k, -DECODER_SYNC_CORRIDOR_90K + 1,
                                                                       DECODER_SYNC_CORRIDOR_90K)
                                                 : halfFrame90k;
            const int64_t targetRaw = haveHint ? hintClamped : halfFrame90k;
            const int64_t shift = std::max<int64_t>(0, targetRaw - rawDelta);
            // Bound the follow-up so a marginal-HW catch-up that can't reach the target doesn't
            // burn dozens of frames at the corridor edge. 8 frames = 160 ms at 50 fps, comfortably
            // covers the -CORRIDOR..+CORRIDOR swing (the clamp range above) without entering
            // trick-mode territory.
            const int followUpDrops = std::min<int>(8, static_cast<int>((shift + frameDur90k - 1) / frameDur90k));
            const auto targetMs = static_cast<long long>(targetRaw / PTS_TICKS_PER_MS);
            const uint64_t exitNowMs = cTimeMs::Now();
            const uint64_t cycleWallMs = exitNowMs - catchUpStartMs;
            if (catchUpLogThisCycle) {
                dsyslog("vaapivideo/decoder: catch-up complete dropped=%d wall=%llums exit-raw=%+lldms "
                        "follow-up=%d (target=%+lldms%s)",
                        catchUpDrops, static_cast<unsigned long long>(cycleWallMs),
                        static_cast<long long>(rawDelta / PTS_TICKS_PER_MS), followUpDrops, targetMs,
                        haveHint ? "/hint" : "");
            } else {
                ++suppressedCatchUpCycles;
                suppressedCatchUpDrops += catchUpDrops;
                suppressedCatchUpWallMs += cycleWallMs;
                if (nextCatchUpSummaryMs == 0) {
                    // First suppression of this run -- schedule the periodic summary.
                    nextCatchUpSummaryMs = exitNowMs + DECODER_CATCHUP_SUMMARY_INTERVAL_MS;
                } else if (exitNowMs >= nextCatchUpSummaryMs) {
                    // Run still going at the summary interval: emit a periodic "sustained" line and
                    // reset the aggregate so the count reflects one interval, not the whole run.
                    isyslog("vaapivideo/decoder: catch-up cycling sustained: %d cycles drops=%d wallSum=%llums "
                            "in last %dms (decoder unable to keep up)",
                            suppressedCatchUpCycles, suppressedCatchUpDrops,
                            static_cast<unsigned long long>(suppressedCatchUpWallMs),
                            DECODER_CATCHUP_SUMMARY_INTERVAL_MS);
                    suppressedCatchUpCycles = 0;
                    suppressedCatchUpDrops = 0;
                    suppressedCatchUpWallMs = 0;
                    nextCatchUpSummaryMs = exitNowMs + DECODER_CATCHUP_SUMMARY_INTERVAL_MS;
                }
            }
            catchingUp = false;
            catchUpDrops = 0;
            lastCatchUpExitMs = exitNowMs;
            ResetSmoothedDelta();
            // Only arm the follow-up burst when jitterBuf actually holds enough cached frames to
            // satisfy it. A cached drop is a fast pop: head PTS jumps a frame while the audio clock
            // barely advances, so rawDelta climbs toward the target. On a saturated/drained pipeline
            // (UHD upscale, where catch-up just bulk-drained jitterBuf via SkipStaleJitterFrames)
            // each follow-up drop instead waits a full VPP cycle, during which audio advances
            // ~one frame -- rawDelta never reaches the target, so the burst discards good frames for
            // no sync gain and pins the queue at depth 1 (the sustained buf=1 catch-up cascade).
            // Skip it there; the next decoded frame is submitted normally and the cushion rebuilds.
            pendingDrops = (jitterBuf.size() > static_cast<size_t>(followUpDrops)) ? followUpDrops : 0;
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
        if (BeginCatchUpLogCycle()) {
            dsyslog("vaapivideo/decoder: catch-up entered (spike) raw=%+lldms",
                    static_cast<long long>(rawDelta / PTS_TICKS_PER_MS));
        }
        return true;
    } else if (!smoothedDeltaValid && rawDelta < -2 * DECODER_SYNC_CORRIDOR_90K) {
        // Pre-warmup raw-based entry: drains a stale pre-roll backlog before it poisons the EMA.
        catchingUp = true;
        catchUpStartMs = cTimeMs::Now();
        catchUpDrops = 1;
        ++syncDropSinceLog;
        if (BeginCatchUpLogCycle()) {
            dsyslog("vaapivideo/decoder: catch-up entered (warmup) raw=%+lldms",
                    static_cast<long long>(rawDelta / PTS_TICKS_PER_MS));
        }
        return true;
    } else if (smoothedDeltaValid && smoothedDelta90k < -2 * DECODER_SYNC_CORRIDOR_90K) {
        catchingUp = true;
        catchUpStartMs = cTimeMs::Now();
        catchUpDrops = 1;
        ++syncDropSinceLog;
        if (BeginCatchUpLogCycle()) {
            dsyslog("vaapivideo/decoder: catch-up entered (sustained) avg=%+lldms raw=%+lldms",
                    static_cast<long long>(smoothedDelta90k / PTS_TICKS_PER_MS),
                    static_cast<long long>(rawDelta / PTS_TICKS_PER_MS));
        }
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
        const int totalDrops = std::max(1, (correctMs + (frameDurMs / 2)) / frameDurMs);
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
        cCondWait::SleepMs(bigSleepMs + frameDurMs);
        if (display) {
            display->SetSyncSleeping(false);
        }
        ++syncSkipSinceLog;
        dsyslog("vaapivideo/decoder: sync skip (hard-ahead live) pts=%lld raw=%+lldms slept=%dms",
                static_cast<long long>(pts), static_cast<long long>(rawDelta / PTS_TICKS_PER_MS),
                bigSleepMs + frameDurMs);
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
                const int totalDrops = std::max(1, (correctMs + (frameDurMs / 2)) / frameDurMs);
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
            cCondWait::SleepMs(correctMs + frameDurMs);
            if (display) {
                display->SetSyncSleeping(false);
            }
            ++syncSkipSinceLog;
            dsyslog("vaapivideo/decoder: sync skip (soft-ahead) pts=%lld raw=%+lldms avg=%+lldms slept=%dms",
                    static_cast<long long>(pts), static_cast<long long>(rawDelta / PTS_TICKS_PER_MS),
                    static_cast<long long>(smoothedDelta90k / PTS_TICKS_PER_MS), correctMs + frameDurMs);
        }
    }

    const bool submitted = SubmitIfCurrent(std::move(frame));
    if (preSleepMs != 0) {
        // EMA feedback: subtract measured shift (elapsed - frameDur); std::max guards underflow.
        const auto elapsedMs = static_cast<int>(cTimeMs::Now() - preSleepMs);
        const int extraMs = std::max(0, elapsedMs - frameDurMs);
        // Snapshot the *pre-correction* smoothedDelta as the canonical steady-state offset for
        // seek-hint capture. The `-=` below predictively models the post-sleep raw delta, which
        // makes smoothedDelta90k a transient value during the EMA recovery window (~3 s for a
        // 15 fps source). A FlushForSeek that lands inside that window would otherwise grab the
        // mid-recovery value as the hint and bias the post-seek EMA seed; capturing here
        // preserves the steady-state value the controller had just acted on. The timestamp
        // pairs with DECODER_SYNC_HINT_MAX_AGE_MS so the snapshot ages out if no further
        // correction refreshes it -- a stale snapshot from a past operating point shouldn't
        // override a currently-valid smoothedDelta.
        if (smoothedDeltaValid) {
            stableDelta90k = smoothedDelta90k;
            stableDeltaCapturedMs = cTimeMs::Now();
        }
        smoothedDelta90k -= static_cast<int64_t>(extraMs) * PTS_TICKS_PER_MS;
        // Suppress the next drain-loop miss check: the gap straddling this submit was a
        // deliberate hard-ahead / soft-ahead sleep, not upstream starvation.
        sleptInLastSubmit = true;
    }
    return submitted;
}
