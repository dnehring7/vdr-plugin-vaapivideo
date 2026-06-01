// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file decoder.h
 * @brief Threaded VAAPI decoder with VPP filter graph and audio-mastered A/V sync
 */

#ifndef VDR_VAAPIVIDEO_DECODER_H
#define VDR_VAAPIVIDEO_DECODER_H

#include "common.h"
#include "filter.h"
#include "stream.h"

#include <deque>
#include <functional>

// VDR
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <vdr/thread.h>
#include <vdr/tools.h>
#pragma GCC diagnostic pop

class cAudioProcessor;
class cVaapiDisplay;
struct VaapiContext;

// ============================================================================
// === CONSTANTS ===
// ============================================================================

inline constexpr size_t DECODER_TRICK_QUEUE_DEPTH =
    1;                                           ///< Depth-1: Poll() throttles the producer; overflow drops incoming.
inline constexpr int DECODER_TRICK_HOLD_MS = 20; ///< Base hold per frame for slow trick (~= one field period @ 50 Hz).

// ============================================================================
// === STRUCTURES ===
// ============================================================================

/// Decoded output frame. Owns the AVFrame reference that keeps the VAAPI surface alive.
/// The display thread releases it after the DRM pageflip retires the surface.
struct VaapiFrame {
    VaapiFrame() = default;
    ~VaapiFrame() noexcept;
    VaapiFrame(const VaapiFrame &) = delete;
    VaapiFrame(VaapiFrame &&other) noexcept;
    auto operator=(const VaapiFrame &) -> VaapiFrame & = delete;
    auto operator=(VaapiFrame &&other) noexcept -> VaapiFrame &;

    // ========================================================================
    // === DATA ===
    // ========================================================================
    AVFrame *avFrame{};          ///< Holds the VAAPI surface buffer ref; keep alive until DRM retires it.
    bool ownsFrame{true};        ///< False after a move; move-out nulls avFrame so dtor is a no-op.
    int64_t pts{AV_NOPTS_VALUE}; ///< Presentation timestamp in 90 kHz units.
    VASurfaceID vaSurfaceId{VA_INVALID_SURFACE}; ///< Cached from avFrame->data[3]; used for zero-copy DRM PRIME export.
};

// ============================================================================
// === DECODER CLASS ===
// ============================================================================

/// Threaded VAAPI decoder. Pipeline overview and lock ordering are documented in decoder.cpp's
/// file-scope comment. Public API is called from VDR's dvbplayer/device thread; Action() runs
/// on its own cThread. All cross-thread state uses atomics or one of the two mutexes.
class cVaapiDecoder : public cThread {
  public:
    cVaapiDecoder(cVaapiDisplay *display, VaapiContext *vaapiCtx);
    ~cVaapiDecoder() noexcept override;
    cVaapiDecoder(const cVaapiDecoder &) = delete;
    cVaapiDecoder(cVaapiDecoder &&) noexcept = delete;
    auto operator=(const cVaapiDecoder &) -> cVaapiDecoder & = delete;
    auto operator=(cVaapiDecoder &&) noexcept -> cVaapiDecoder & = delete;

    // ========================================================================
    // === PUBLIC API ===
    // ========================================================================
    auto Clear() -> void;        ///< Flush queued packets, codec buffers, and filter graph; resets A/V sync state.
    auto DrainQueue() -> void;   ///< Discard all queued packets without touching codec or filter state.
    auto FlushForSeek() -> void; ///< Same as Clear() but keeps the filter graph alive (mediaplayer seek path).
    auto EnqueueData(const uint8_t *data, size_t size, int64_t pts)
        -> void; ///< PES path: parse raw NAL bytes via av_parser_parse2 and push complete AUs onto the queue.
    // [MEDIAPLAYER-SEAM] Currently unused: reserved for the libavformat-based mediaplayer path.
    auto EnqueuePacket(const AVPacket *packet)
        -> void; ///< Mediaplayer path: clone a pre-demuxed AU (whole access unit) onto the decode queue.
    auto FlushParser()
        -> void; ///< Force-drain the parser's held-back AU. Required for still-picture (single I-frame delivery).
    [[nodiscard]] auto GetLastPts() const noexcept
        -> int64_t; ///< PTS of the most recently decoded frame in 90 kHz ticks, or AV_NOPTS_VALUE.
                    ///< Includes catch-up-dropped frames (see PublishLastPts site in DecodeOnePacket).
    [[nodiscard]] auto GetJitterBufSize() const noexcept -> size_t {
        return publishedJitterBufSize.load(std::memory_order_relaxed);
    } ///< Cross-thread snapshot of post-decode buffered frames.
    [[nodiscard]] auto GetQueueSize() const -> size_t;    ///< Packets waiting in the decode queue.
    [[nodiscard]] auto GetStreamAspect() const -> double; ///< Stream DAR (width x SAR), or 0.0 when closed.
    [[nodiscard]] auto GetStreamHeight() const -> int;    ///< Coded stream height, or 0 when closed.
    [[nodiscard]] auto GetStreamWidth() const -> int;     ///< Coded stream width, or 0 when closed.
    [[nodiscard]] auto Initialize() -> bool;              ///< Allocate staging frames, set ready, start thread.
    [[nodiscard]] auto IsQueueEmpty() const -> bool;      ///< VDR Poll(): true -> accept next PES packet.
    [[nodiscard]] auto IsQueueFull() const -> bool;       ///< True when queue has reached DECODER_QUEUE_CAPACITY.
    [[nodiscard]] auto IsReady() const noexcept -> bool;  ///< True after Initialize() succeeds.
    [[nodiscard]] auto IsReadyForNextTrickFrame() const noexcept
        -> bool; ///< True when trick-mode pacing timer has expired.
    [[nodiscard]] auto OpenCodec(AVCodecID codecId)
        -> bool; ///< PES path wrapper: no extradata, 8-bit profile assumed; delegates to OpenCodecWithInfo().
    [[nodiscard]] auto OpenCodecWithInfo(const VideoStreamInfo &info)
        -> bool; ///< Open (or reuse) a decoder. HW vs SW selected via SelectVideoBackendCap() + GpuCaps.
                 ///< Reuse requires matching codec ID, HW/SW choice, and extradata; anything else triggers teardown.
    auto NotifyAudioChange() -> void; ///< Arm freerun after an audio codec/track switch; audio clock is NOPTS briefly.
    auto SetAudioProcessor(cAudioProcessor *audio)
        -> void; ///< Attach the A/V sync master clock. Stored as atomic pointer.
    auto SetLoopTickCallback(std::function<void()> callback)
        -> void; ///< Called once per decode-loop iteration (incl. the ~10 ms idle ticks when no packets arrive),
                 ///< giving the device a thread that ticks even when a scrambled channel delivers no PES. Must be
                 ///< set before Initialize() starts the thread.
    auto SetDevicePaused(bool paused) noexcept
        -> void; ///< Mirror cVaapiDevice::Freeze() / Play() into the drain loop. While paused the drain HOLDS
                 ///< the jitterBuf (no submit, no stall-watchdog re-arm) so the head's PTS doesn't drift while
                 ///< the audio master clock is genuinely frozen (ALSA dropped). Without this the decoder's
                 ///< no-clock-freerun fires when GetClock() goes stale and submits frames at vsync rate during
                 ///< pause, leaving the head hundreds of ms ahead of the audio clock on resume -> persistent
                 ///< video-ahead drain-stall loop that never recovers.
    auto SetLiveMode(bool live) -> void; ///< true = live TV (jitter buffer active); false = replay.
    auto RequestCodecDrain() -> void;    ///< Ask decode thread to drain B-frame reorder buffer (e.g. before still).
    auto SetStillPictureMode(bool mode) -> void; ///< Spatial-only deinterlace for single-frame output; clears on drain.
    auto RequestCodecReopen() -> void;           ///< Force full codec teardown on next OpenCodec() even for same ID.
    auto RequestFilterRebuild()
        -> void; ///< Schedule filter graph rebuild on next decoded frame (e.g. after ScaleVideo dim change).
    auto RequestTrickExit() -> void; ///< Deferred Play()-without-TrickSpeed(0); cleared if SetTrickSpeed() follows.
    auto SetTrickSpeed(int speed, bool forward = true, bool fast = false)
        -> void;             ///< Configure trick-play pacing. speed=0 returns to normal. fast=true -> key-frames only.
    auto Shutdown() -> void; ///< Stop decode thread and release all resources. Idempotent via stopping flag.

  protected:
    // ========================================================================
    // === THREAD ===
    // ========================================================================
    auto Action() -> void override; ///< Decode thread: dequeue -> VAAPI decode -> filter -> A/V sync -> display.

  private:
    // ========================================================================
    // === INTERNAL METHODS ===
    // ========================================================================
    auto ClearInternal(bool resetFilter, bool preserveSeekHint)
        -> void; ///< Shared body of Clear() / FlushForSeek(). preserveSeekHint=true binds the seek-hint
                 ///< preservation request directly to the jitter-flush request so a coalesced flush
                 ///< observed by the decode thread always sees the matching policy (race-free).
    [[nodiscard]] auto CreateVaapiFrame(AVFrame *src) const
        -> std::unique_ptr<VaapiFrame>; ///< av_frame_clone() the filtered surface; extracts VASurfaceID from data[3].
    [[nodiscard]] auto DecodeOnePacket(AVPacket *pkt, std::vector<std::unique_ptr<VaapiFrame>> &outFrames)
        -> bool;                         ///< avcodec_send_packet + drain loop. Returns true if any frame was appended.
    auto DrainPendingParserAU() -> void; ///< NULL-input flush of av_parser_parse2. Caller holds parserMutex.
    auto FilterAndAppendDecodedFrame(std::vector<std::unique_ptr<VaapiFrame>> &outFrames)
        -> void; ///< Push decodedFrame through the filter graph (lazily built) and append with monotonic PTS.
                 ///< Caller holds codecMutex and must have populated decodedFrame.
    auto DrainCodecAtEos(std::vector<std::unique_ptr<VaapiFrame>> &outFrames)
        -> void; ///< NULL-packet EOS drain, then avcodec_flush_buffers to re-arm. Caller holds codecMutex.
    [[nodiscard]] auto InitFilterGraph(AVFrame *firstFrame, bool compactLog = false)
        -> bool; ///< Fill BuildParams and delegate to filterChain_.Build(). compactLog=true for
                 ///< ScaleVideo-driven rebuilds (one-line dsyslog); false for first build / channel switch.
    [[nodiscard]] auto ShouldUseHdrPassthrough(const HdrStreamInfo &info) const noexcept
        -> bool; ///< True when stream + GPU (vppP010) + display (EDID) + user config all permit HDR passthrough.
    [[nodiscard]] auto SubmitIfCurrent(std::unique_ptr<VaapiFrame> frame)
        -> bool; ///< Submit unless clearEpoch raced this iteration; stale-epoch frames are dropped silently
                 ///< (returns true so callers don't count it as a submit failure).
    [[nodiscard]] auto SubmitTrickFrame(std::unique_ptr<VaapiFrame> frame)
        -> bool; ///< Pacing: wait deadline, skip reverse-GOP duplicates, arm next deadline; then submit.
    [[nodiscard]] auto SyncAndSubmitFrame(std::unique_ptr<VaapiFrame> frame)
        -> bool; ///< Audio-master A/V sync gate (four regimes; see decoder.cpp file comment and AVSYNC.md).
    [[nodiscard]] auto SyncLatency90k(const cAudioProcessor *ap) const noexcept
        -> int64_t; ///< User latency knob (PCM or passthrough) + 1-frame pipeline constant (dominant scanout delay:
                    ///< commit + page flip). Pass nullptr to use PCM knob (safe default before audio processor is
                    ///< attached).
    auto UpdateSmoothedDelta(int64_t rawDelta90k) noexcept
        -> void;                                   ///< Residual-accumulator EMA; call once per output frame.
    auto ResetSmoothedDelta() noexcept -> void;    ///< Invalidate EMA, clear warmup, zero debounce counters.
    auto PushPacketToQueue(AVPacket *pkt) -> void; ///< Takes ownership of pkt. Trick mode: drops incoming on overflow.
                                                   ///< Normal mode: drops oldest. Shared by PES and mediaplayer paths.
    auto NoteStarvationTick(const AVPacket *pkt) noexcept
        -> void; ///< Starvation diagnostic: counts packets/keyframes until first frame lands. No-op after that (one
                 ///< load).
    auto LogSyncStats(int64_t rawDelta90k, int64_t latency90k, const cAudioProcessor *ap)
        -> void; ///< Periodic dsyslog; suppressed during EMA warmup.
    auto SkipStaleJitterFrames(cAudioProcessor *ap)
        -> void; ///< Bulk-pop heads > HARD_THRESHOLD behind clock; keeps >=1 frame.
    auto WaitForAudioCatchUp(cAudioProcessor *ap, int64_t pts, int64_t latency, int64_t delta)
        -> void; ///< Replay hard-ahead: block until audio clock reaches video PTS. Capped at delta/90 + 1 s, max 5 s.
    auto PublishLastPts(int64_t pts) noexcept
        -> void; ///< Decode thread only. Stores pts iff iterationEpoch matches clearEpoch (Clear-race guard).
    auto ApplyDeferredJitterFlush(uint64_t &lastDrainMs, bool preserveSeekHint) noexcept
        -> void; ///< Consume a pending Clear() / FlushForSeek(): reset EMA, drop jitterBuf, zero pendingDrops
                 ///< + lastDrainMs. preserveSeekHint comes from the flush request itself (not a separate
                 ///< atomic) so racing flushes can't cross-pollinate the preserve policy.
    [[nodiscard]] auto BeginCatchUpLogCycle() noexcept
        -> bool; ///< Catch-up log throttle gate; flushes any pending "cycling settled" summary on the first
                 ///< logged entry after a suppressed run. Returns true if this entry should be logged.

    // ========================================================================
    // === SYNCHRONIZATION ===
    // ========================================================================
    // Lock order: ALWAYS codecMutex -> parserMutex -> packetMutex. DrainQueue takes only packetMutex.
    //
    // codecMutex and parserMutex are deliberately separate: the dvbplayer / receiver feeds the
    // parser via EnqueueData() while the decode thread is busy submitting work to VAAPI. Sharing
    // one mutex would serialize the two -- in replay that costs ~25 ms/s of GPU submission time
    // on UHD upscale and shows up as sustained negative drift. av_parser_parse2() only reads
    // immutable codecCtx fields (codec_id + descriptor); writers of codecCtx existence
    // (OpenCodecWithInfo / Clear / SetTrickSpeed) take BOTH mutexes in fixed order.
    mutable cMutex codecMutex;  ///< Guards codec context + filter graph (decode thread vs reopen/clear).
    mutable cMutex parserMutex; ///< Guards parser context (EnqueueData vs reopen/clear).
    mutable cMutex packetMutex; ///< Guards packetQueue; also used as condvar futex.
    cCondVar packetCondition;   ///< Wakes the decode thread on enqueue or shutdown.

    // ========================================================================
    // === REFERENCES ===
    // ========================================================================
    std::atomic<cAudioProcessor *> audioProcessor{
        nullptr};                           ///< A/V sync master clock. Written by main thread, read by decode thread.
    cVaapiDisplay *display;                 ///< Receives completed VaapiFrames via SubmitFrame().
    VaapiContext *vaapiContext;             ///< Shared VAAPI hw_device_ctx and GpuCaps.
    std::function<void()> loopTickCallback; ///< Per-iteration device hook; set before Initialize(), then read-only.

    // ========================================================================
    // === FFMPEG STATE ===
    // ========================================================================
    std::unique_ptr<AVCodecContext, FreeAVCodecContext>
        codecCtx;                               ///< Active decoder context (HW or SW). Null before OpenCodec().
    AVCodecID currentCodecId{AV_CODEC_ID_NONE}; ///< Codec ID currently open; used for reuse check and parser recreate.
    bool forceCodecReopen{};                    ///< Set by RequestCodecReopen(); cleared by OpenCodecWithInfo().
    std::unique_ptr<AVFrame, FreeAVFrame>
        decodedFrame;              ///< Staging for avcodec_receive_frame(); unref'd each iteration.
    cVideoFilterChain filterChain; ///< VPP graph (bwdif/deinterlace -> scale_vaapi -> optional denoise/sharpness).
    std::unique_ptr<AVFrame, FreeAVFrame>
        filteredFrame; ///< Staging for filterChain.ReceiveFrame(); unref'd each iteration.
    std::unique_ptr<AVCodecParserContext, FreeAVCodecParserContext>
        parserCtx; ///< Null on mediaplayer path (extradata present). Slices PES NAL bytes into whole AUs.

    // ========================================================================
    // === PACKET QUEUE ===
    // ========================================================================
    std::queue<AVPacket *> packetQueue; ///< FIFO of parsed packets awaiting HW decode. Owned by packetMutex.
    cTimeMs lastTrickDropWarn;          ///< Rate-limits the trick-enqueue drop log. Held under packetMutex.
    size_t trickDropsSinceWarn{0};      ///< Drops accumulated since the last emitted trick-drop log line.
    std::atomic<bool> stopping; ///< Shutdown signal. Set by Shutdown(); read by encode thread and enqueue paths.

    // ========================================================================
    // === PLAYBACK STATE ===
    // ========================================================================
    std::atomic<bool> codecDrainPending;   ///< Decode thread drains codec (NULL packet) then clears this.
    std::atomic<bool> stillPictureMode;    ///< Selects spatial-only (bob) deinterlace; cleared after drain.
    std::atomic<bool> hasExited{true};     ///< False only while Action() is running; checked by Shutdown().
    std::atomic<bool> hasLoggedFirstFrame; ///< One-time first-frame info log guard; reset only in OpenCodecWithInfo().
    std::atomic<bool> starvationWarned;    ///< One-time "no frame 3 s after open" warning; reset per codec open.
    std::atomic<bool>
        starvationWarnedSustained;         ///< One-time "still no frame 15 s after open" warning; reset per codec open.
    std::atomic<uint64_t> codecOpenTimeMs; ///< cTimeMs::Now() at last OpenCodecWithInfo(); used by starvation tiers.
    std::atomic<size_t> packetsSinceOpen;  ///< avcodec_send_packet calls since last open; starvation counters.
    std::atomic<size_t> keyPacketsSinceOpen; ///< Subset with AV_PKT_FLAG_KEY; distinguishes silent feed vs HW stall.
    std::atomic<int64_t> lastPts{
        AV_NOPTS_VALUE};                     ///< Last decoded PTS in 90 kHz ticks. Read by GetLastPts() / device STC.
    std::atomic<uint64_t> clearEpoch{0};     ///< Generation tag for lastPts; bumped by Clear() / SetTrickSpeed(0).
    uint64_t iterationEpoch{0};              ///< Decode thread only. Snapshot of clearEpoch at each Action() iter top.
    std::atomic<bool> liveMode;              ///< Hard-ahead policy: replay blocks via WaitForAudioCatchUp, live sleeps.
    std::atomic<bool> devicePaused{false};   ///< Mirrors cVaapiDevice::Freeze()/Play(). When true the drain loop holds
                                             ///< (no submit, no stall-watchdog re-arm) so the head's PTS doesn't drift
                                             ///< while ALSA is dropped and the audio master clock is genuinely frozen.
    std::atomic<bool> ready{false};          ///< Set by Initialize(); gate for OpenCodec() and EnqueueData().
    std::atomic<int> trickSpeed;             ///< 0 = normal; >0 = trick mode (speed value mirrors VDR TrickSpeed).
    std::atomic<bool> videoRectDirty{false}; ///< Triggers filterChain.Reset() on next frame; set by
                                             ///< RequestFilterRebuild when ScaleVideo() changes the target dimensions.
    std::atomic<bool> filterCompactRebuildPending{
        false}; ///< Set by FlushForSeek to request a one-line "filter rebuilt" diagnostic on the
                ///< next InitFilterGraph call instead of the full 3-line graph init dump.
                ///< Consumed (exchanged to false) by the decode-thread filter-build path.

    // ========================================================================
    // === TRICK MODE ===
    // ========================================================================
    std::atomic<bool> trickExitPending;      ///< Play() without TrickSpeed(0); clear flags and freerun on next frame.
    std::atomic<bool> isTrickFastForward;    ///< FF mode: only keyframes enqueued; first field of each pair dropped.
    std::atomic<bool> isTrickReverse;        ///< REW: GOPs arrive backward; skip frames with rising PTS within a GOP.
    std::atomic<uint64_t> nextTrickFrameDue; ///< cTimeMs::Now() deadline for next submission; enforces pacing.
    std::atomic<int64_t> prevTrickPts{AV_NOPTS_VALUE}; ///< Source PTS of previous trick frame; detects field pairs.
    std::atomic<uint64_t> trickHoldMs{
        DECODER_TRICK_HOLD_MS};            ///< Hold per frame in slow mode = speed * DECODER_TRICK_HOLD_MS.
    std::atomic<uint64_t> trickMultiplier; ///< Fast-mode PTS-derived hold divisor (2/4/8x). 0 = slow mode.

    // ========================================================================
    // === A/V SYNC ===
    // ========================================================================
    // Default 1 so the very first frame after construction is submitted immediately.
    // Without it the due-gate holds until the audio clock is anchored and the screen stays black.
    std::atomic<int> freerunFrames{
        1}; ///< Bypass A/V sync for N frames. Set by Clear() / trick-exit / NotifyAudioChange().
    std::atomic<int> jitterFlushRequest{
        0}; ///< Deferred Clear() / FlushForSeek() request consumed by the decode thread:
            ///<   0 = no flush pending,
            ///<   1 = plain Clear()  -- drop seek-hint, content boundary,
            ///<   2 = FlushForSeek() -- preserve seek-hint across the flush.
            ///< The preserve policy is encoded *in* the request so back-to-back FlushForSeek /
            ///< Clear() can't cross-pollinate (a stale flush observed by the decode thread always
            ///< carries its originator's policy, never a later issuer's). Last writer wins, which
            ///< is correct: Clear() after FlushForSeek dropping the hint = content boundary win;
            ///< FlushForSeek after FlushForSeek = coalesced preserve.
    std::atomic<size_t> publishedJitterBufSize{0}; ///< Cross-thread snapshot of jitterBuf.size() for backpressure.
                                                   ///< Written by decode thread once per Action() iteration,
                                                   ///< read by the mediaplayer demux thread.
    std::atomic<bool> syncLogPending;              ///< Force sync log on next frame regardless of timer.
    cTimeMs nextSyncLog;              ///< Decode thread only. Deadline for the periodic sync-stats dsyslog.
    int drainMissCount{};             ///< Drain gaps > 2xframeDur since last sync log; indicates upstream starvation.
    int syncDropSinceLog{};           ///< Frames dropped (video behind) since last sync log. Decode thread only.
    int syncSkipSinceLog{};           ///< Frames delayed (video ahead) since last sync log. Decode thread only.
    int pendingDrops{};               ///< Remaining frames to drop in current soft- or hard-behind burst.
                                      ///< Consumed one-per-iteration in SyncAndSubmitFrame. Decode thread only.
    bool sleptInLastSubmit{};         ///< Set by SyncAndSubmitFrame / WaitForAudioCatchUp when a sync-correction
                                      ///< sleep ran. Consumed (cleared) by the drain loop's miss check so the
                                      ///< self-inflicted gap doesn't inflate drainMissCount. Decode thread only.
    int64_t rawDeltaSumSinceLog90k{}; ///< Accumulator for interval-mean rawDelta in the sync log.
    int rawDeltaCountSinceLog{};      ///< Sample count for rawDeltaSumSinceLog90k.
    int warmupSampleCount{};          ///< Samples accumulated during EMA warmup (post-reset). Decode thread only.
    int64_t warmupSampleSum90k{};     ///< Warmup accumulator in 90 kHz ticks. Decode thread only.
    int64_t smoothedDelta90k{};       ///< EMA-smoothed A/V delta in 90 kHz ticks. Decode thread only.
    int64_t emaResidual90k{};         ///< Integer EMA remainder: carries sub-sample rounding so the filter converges
                                      ///< exactly to the mean rather than stalling when |diff| < EMA_SAMPLES ticks.
    bool smoothedDeltaValid{false};   ///< True after warmup completes. Gates soft-corridor and catch-up (sustained).
    int64_t seekHintDelta90k{AV_NOPTS_VALUE}; ///< One-shot fast-start hint: last converged smoothedDelta carried across
                                              ///< a FlushForSeek. Used as the catch-up exit target AND as the EMA seed
                                              ///< on the first valid post-seek sample, so playback resumes at the
                                              ///< steady-state offset within milliseconds instead of waiting out the
                                              ///< 50-sample warmup. Consumed (set back to AV_NOPTS_VALUE) once the EMA
                                              ///< is seeded so subsequent controller resets warm up from real samples
                                              ///< rather than re-applying the same stale value. Decode thread only.
    int64_t stableDelta90k{
        AV_NOPTS_VALUE};              ///< Pre-correction snapshot of smoothedDelta taken at each hard-/soft-ahead
                                      ///< trigger, just before the post-sleep `smoothedDelta -= extraMs` feedback.
                                      ///< Represents the long-term GPU-vs-audio offset that the EMA had converged
                                      ///< to before the in-flight correction perturbed it. Used as the seek-hint
                                      ///< source so the captured hint survives a FlushForSeek that lands inside
                                      ///< the post-correction recovery window. AV_NOPTS_VALUE until the first
                                      ///< trigger fires; cleared by ResetSmoothedDelta. Decode thread only.
    uint64_t stableDeltaCapturedMs{}; ///< cTimeMs::Now() of the most recent stableDelta90k capture. Paired with
                                      ///< DECODER_SYNC_HINT_MAX_AGE_MS so an old snapshot from a single past
                                      ///< correction cannot dominate seeks long after the pipeline has settled at
                                      ///< a different offset. 0 = no capture yet; reset by ResetSmoothedDelta.
    int hardAheadDebounce{};          ///< Consecutive rawDelta > HARD_THRESHOLD; 2-sample debounce before action.
    int hardBehindDebounce{};         ///< Consecutive rawDelta < -HARD_THRESHOLD; 2-sample debounce before action.
    bool catchingUp{};                ///< Bulk-dropping a catastrophic backlog (seek / startup stall).
    int catchUpDrops{};               ///< Frames silently dropped in the current catch-up pass.
    uint64_t catchUpStartMs{};        ///< cTimeMs::Now() at catch-up entry; reported at exit.
    uint64_t lastCatchUpExitMs{};     ///< cTimeMs::Now() at catch-up exit; diagnostic for cascade detection.
    cTimeMs syncCooldown;             ///< Rate-limits soft corrections to once per DECODER_SYNC_COOLDOWN_MS.

    // --- Catch-up log throttling ---
    // Sustained catch-up cycling (e.g. VVC SW decode, or a marginal HW decoder dropping ~5%) emits one
    // entry+exit pair per cycle, flooding syslog. Cycling is detected by inter-entry gap: two entries
    // within DECODER_CATCHUP_LOG_THROTTLE_MS of each other are a "run". The first entry of a run logs
    // normally; subsequent entries in the run are suppressed and aggregated into a periodic
    // (DECODER_CATCHUP_SUMMARY_INTERVAL_MS) "sustained" summary while the run continues, plus a final
    // "settled" summary when the gap exceeds the window (run ended). Controller behavior is unchanged.
    uint64_t lastCatchUpEntryMs{};      ///< cTimeMs::Now() of the most recent catch-up entry (logged OR suppressed).
                                        ///< Inter-entry gap vs DECODER_CATCHUP_LOG_THROTTLE_MS detects an active run.
    bool catchUpLogThisCycle{false};    ///< Set at entry, consulted at exit so suppressed entries don't log their exit.
    int suppressedCatchUpCycles{};      ///< Cycles aggregated since the run started (or last periodic summary).
    int suppressedCatchUpDrops{};       ///< Cumulative dropped-frame count of those cycles.
    uint64_t suppressedCatchUpWallMs{}; ///< Cumulative wall time spent inside those cycles.
    uint64_t nextCatchUpSummaryMs{};    ///< cTimeMs::Now() at which the next periodic-during-run summary should fire.

    // ========================================================================
    // === JITTER BUFFER ===
    // ========================================================================
    std::deque<std::unique_ptr<VaapiFrame>>
        jitterBuf;                 ///< Decoded frames pending display. Decode thread only; see AVSYNC.md.
    int outputFrameDurationMs{20}; ///< Updated by filterChain after build; 20 ms = 50fps field rate, 40 ms = 25fps.
};

#endif // VDR_VAAPIVIDEO_DECODER_H
