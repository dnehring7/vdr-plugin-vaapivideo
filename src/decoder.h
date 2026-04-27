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
    auto Clear() -> void;      ///< Flush queued packets, codec buffers, and filter graph; resets A/V sync state.
    auto DrainQueue() -> void; ///< Discard all queued packets without touching codec or filter state.
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
        -> void;                         ///< Attach the A/V sync master clock. Stored as atomic pointer.
    auto SetLiveMode(bool live) -> void; ///< true = live TV (jitter buffer active); false = replay.
    auto RequestCodecDrain() -> void;    ///< Ask decode thread to drain B-frame reorder buffer (e.g. before still).
    auto SetStillPictureMode(bool mode) -> void; ///< Spatial-only deinterlace for single-frame output; clears on drain.
    auto RequestCodecReopen() -> void;           ///< Force full codec teardown on next OpenCodec() even for same ID.
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
    [[nodiscard]] auto CreateVaapiFrame(AVFrame *src) const
        -> std::unique_ptr<VaapiFrame>; ///< av_frame_clone() the filtered surface; extracts VASurfaceID from data[3].
    [[nodiscard]] auto DecodeOnePacket(AVPacket *pkt, std::vector<std::unique_ptr<VaapiFrame>> &outFrames)
        -> bool;                         ///< avcodec_send_packet + drain loop. Returns true if any frame was appended.
    auto DrainPendingParserAU() -> void; ///< NULL-input flush of av_parser_parse2. Caller holds codecMutex.
    auto FilterAndAppendDecodedFrame(std::vector<std::unique_ptr<VaapiFrame>> &outFrames)
        -> void; ///< Push decodedFrame through the filter graph (lazily built) and append with monotonic PTS.
                 ///< Caller holds codecMutex and must have populated decodedFrame.
    auto DrainCodecAtEos(std::vector<std::unique_ptr<VaapiFrame>> &outFrames)
        -> void; ///< NULL-packet EOS drain, then avcodec_flush_buffers to re-arm. Caller holds codecMutex.
    [[nodiscard]] auto InitFilterGraph(AVFrame *firstFrame)
        -> bool; ///< Fill BuildParams and delegate to filterChain_.Build().
    [[nodiscard]] auto ShouldUseHdrPassthrough(const HdrStreamInfo &info) const noexcept
        -> bool; ///< True when stream + GPU (vppP010) + display (EDID) + user config all permit HDR passthrough.
    [[nodiscard]] auto SubmitTrickFrame(std::unique_ptr<VaapiFrame> frame)
        -> bool; ///< Pacing: wait deadline, skip reverse-GOP duplicates, arm next deadline; then submit.
    [[nodiscard]] auto SyncAndSubmitFrame(std::unique_ptr<VaapiFrame> frame)
        -> bool; ///< Audio-master A/V sync gate (four regimes; see decoder.cpp file comment and AVSYNC.md).
    [[nodiscard]] auto SyncLatency90k(const cAudioProcessor *ap) const noexcept
        -> int64_t; ///< User latency knob (PCM or passthrough) + 2-frame pipeline constant (decoder->scanout + HDMI
                    ///< link). Pass nullptr to use PCM knob (safe default before audio processor is attached).
    auto UpdateSmoothedDelta(int64_t rawDelta90k) noexcept
        -> void;                                   ///< Residual-accumulator EMA; call once per output frame.
    auto ResetSmoothedDelta() noexcept -> void;    ///< Invalidate EMA, clear warmup, zero streak counters.
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

    // ========================================================================
    // === SYNCHRONIZATION ===
    // ========================================================================
    // Lock order: ALWAYS codecMutex -> packetMutex. DrainQueue takes only packetMutex.
    mutable cMutex codecMutex;  ///< Guards codec context, parser, and filter graph (EnqueueData <-> decode thread).
    mutable cMutex packetMutex; ///< Guards packetQueue; also used as condvar futex.
    cCondVar packetCondition;   ///< Wakes the decode thread on enqueue or shutdown.

    // ========================================================================
    // === REFERENCES ===
    // ========================================================================
    std::atomic<cAudioProcessor *> audioProcessor{
        nullptr};               ///< A/V sync master clock. Written by main thread, read by decode thread.
    cVaapiDisplay *display;     ///< Receives completed VaapiFrames via SubmitFrame().
    VaapiContext *vaapiContext; ///< Shared VAAPI hw_device_ctx and GpuCaps.

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
        AV_NOPTS_VALUE};                 ///< Last decoded PTS in 90 kHz ticks. Read by GetLastPts() / device STC.
    std::atomic<uint64_t> clearEpoch{0}; ///< Generation tag for lastPts; bumped by Clear() / SetTrickSpeed(0).
    uint64_t iterationEpoch{0};          ///< Decode thread only. Snapshot of clearEpoch at each Action() iter top.
    std::atomic<bool> liveMode;          ///< Live TV -> jitter-buffer drain; replay -> sleep-pace to audio clock.
    std::atomic<bool> ready{false};      ///< Set by Initialize(); gate for OpenCodec() and EnqueueData().
    std::atomic<int> trickSpeed;         ///< 0 = normal; >0 = trick mode (speed value mirrors VDR TrickSpeed).

    // ========================================================================
    // === TRICK MODE ===
    // ========================================================================
    std::atomic<bool> trickExitPending;      ///< Play() without TrickSpeed(0); clear flags and freerun on next frame.
    std::atomic<bool> isTrickFastForward;    ///< FF mode: only keyframes enqueued; first field of each pair dropped.
    std::atomic<bool> isTrickReverse;        ///< REW: GOPs arrive backward; skip frames with rising PTS within a GOP.
    std::atomic<uint64_t> nextTrickFrameDue; ///< cTimeMs::Now() deadline for next submission; enforces pacing.
    std::atomic<int64_t> prevTrickPts{AV_NOPTS_VALUE}; ///< Source PTS of previous trick frame; detects field pairs.
    std::atomic<uint64_t> trickHoldMs{DECODER_TRICK_HOLD_MS}; ///< Hold per frame in slow mode = speed * TRICK_HOLD_MS.
    std::atomic<uint64_t> trickMultiplier; ///< Fast-mode PTS-derived hold divisor (2/4/8x). 0 = slow mode.

    // ========================================================================
    // === A/V SYNC ===
    // ========================================================================
    // Default 1 so the very first frame after construction is submitted immediately.
    // Without it the due-gate holds until the audio clock is anchored and the screen stays black.
    std::atomic<int> freerunFrames{
        1}; ///< Bypass A/V sync for N frames. Set by Clear() / trick-exit / NotifyAudioChange().
    std::atomic<bool> jitterFlushPending{
        false};                       ///< Deferred Clear(): decode thread resets jitterBuf + EMA on next cycle.
    std::atomic<bool> syncLogPending; ///< Force sync log on next frame regardless of timer.
    cTimeMs nextSyncLog;              ///< Decode thread only. Deadline for the periodic sync-stats dsyslog.
    int drainMissCount{};             ///< Drain gaps > 2xframeDur since last sync log; indicates upstream starvation.
    int syncDropSinceLog{};           ///< Frames dropped (video behind) since last sync log. Decode thread only.
    int syncSkipSinceLog{};           ///< Frames delayed (video ahead) since last sync log. Decode thread only.
    int pendingDrops{};               ///< Remaining frames to drop in current soft-behind burst. Decode thread only.
    int64_t rawDeltaSumSinceLog90k{}; ///< Accumulator for interval-mean rawDelta in the sync log.
    int rawDeltaCountSinceLog{};      ///< Sample count for rawDeltaSumSinceLog90k.
    int warmupSampleCount{};          ///< Samples accumulated during EMA warmup (post-reset). Decode thread only.
    int64_t warmupSampleSum90k{};     ///< Warmup accumulator in 90 kHz ticks. Decode thread only.
    int64_t smoothedDelta90k{};       ///< EMA-smoothed A/V delta in 90 kHz ticks. Decode thread only.
    int64_t emaResidual90k{};         ///< Integer EMA remainder: carries sub-sample rounding so the filter converges
                                      ///< exactly to the mean rather than stalling when |diff| < EMA_SAMPLES ticks.
    bool smoothedDeltaValid{false};   ///< True after warmup completes. Gates soft-corridor and catch-up (sustained).
    int hardAheadStreak{};            ///< Consecutive rawDelta > HARD_THRESHOLD; 2-sample debounce before action.
    int hardBehindStreak{};           ///< Consecutive rawDelta < -HARD_THRESHOLD; 2-sample debounce before action.
    bool catchingUp{};                ///< Bulk-dropping a catastrophic backlog (seek / startup stall).
    int catchUpDrops{};               ///< Frames silently dropped in the current catch-up pass.
    uint64_t catchUpStartMs{};        ///< cTimeMs::Now() at catch-up entry; reported at exit.
    cTimeMs syncCooldown;             ///< Rate-limits soft corrections to once per DECODER_SYNC_COOLDOWN_MS.

    // ========================================================================
    // === JITTER BUFFER (live TV only) ===
    // ========================================================================
    std::deque<std::unique_ptr<VaapiFrame>>
        jitterBuf;                 ///< Decoded frames pending display. Decode thread only; see AVSYNC.md.
    int outputFrameDurationMs{20}; ///< Updated by filterChain after build; 20 ms = 50fps field rate, 40 ms = 25fps.
};

#endif // VDR_VAAPIVIDEO_DECODER_H
