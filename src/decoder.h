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

/// Depth-1: Poll() throttles the producer; overflow drops incoming.
inline constexpr size_t DECODER_TRICK_QUEUE_DEPTH = 1;
inline constexpr int DECODER_TRICK_HOLD_MS = 20; ///< Base hold per frame for slow trick (~= one field period @ 50 Hz).
/// Cap on the decode-ahead reserve (handoffQueue + jitterBuf, ~1.3 s @ 50 fps total): the decode thread backpressures
/// when the published total reaches it (or handoffQueue alone does), and each stage also drop-oldest-trims past it as a
/// runaway guard. Caps GPU surface retention (~64 4K NV12 surfaces ~= 0.8 GB GTT) while still dwarfing the <40 ms VPP
/// variance and the 8-slot display prerender. In decoder.h (not the .cpp) so the mediaplayer's backpressure gate can be
/// statically checked against it (see device.cpp).
inline constexpr size_t DECODER_RESERVE_HARD_CAP = 64;

// ============================================================================
// === STRUCTURES ===
// ============================================================================

/// Decoded output frame. Owns the AVFrame reference that keeps the VAAPI surface alive.
/// The display thread releases it after the DRM pageflip retires the surface.
struct VaapiFrame {
    VaapiFrame() = default;
    ~VaapiFrame() noexcept;
    VaapiFrame(const VaapiFrame &) = delete;
    VaapiFrame(VaapiFrame &&other) noexcept; ///< Steals the AVFrame and DRM handles; @p other is left empty
    auto operator=(const VaapiFrame &) -> VaapiFrame & = delete;
    /// Releases our own handles first, then steals @p other's.
    auto operator=(VaapiFrame &&other) noexcept -> VaapiFrame &;

    // ========================================================================
    // === DATA ===
    // ========================================================================
    AVFrame *avFrame{}; ///< Holds the VAAPI surface buffer ref; keep alive until DRM retires it.
    /// Keeps the producing VPP graph alive while this surface is in flight (see FilterGraphToken);
    /// null for unfiltered frames. Must be released after avFrame -- the dtor body frees avFrame
    /// first, and moves hand both to cVaapiDisplay::DrmFramebuffer together.
    FilterGraphToken graphToken;
    bool ownsFrame{true};        ///< False after a move; move-out nulls avFrame so dtor is a no-op.
    uint64_t producedEpoch{0};   ///< clearEpoch at production (decode thread); the presentation thread drops frames
                                 ///< older than the current clearEpoch, so pre-Clear material self-discards regardless
                                 ///< of the race timing between decode, present, and Clear().
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
    /// Both pointers are borrowed and must outlive the decoder; neither thread is started until Initialize().
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
    auto ReleasePendingAccessUnit()
        -> void; ///< FlushParser() + parser recreate: a flushed AVCodecParser keeps a stale frame_start_found
                 ///< that would cut the next AU at its first NAL. Once per live codec open, when the first
                 ///< keyframe's PES is complete, so it decodes without waiting for the next PES.
    [[nodiscard]] auto GetLastPts() const noexcept
        -> int64_t; ///< PTS of the most recently decoded frame in 90 kHz ticks, or AV_NOPTS_VALUE.
                    ///< Includes catch-up-dropped frames (see PublishLastPts site in DecodeOnePacket).
    [[nodiscard]] auto GetDecodedReserveSize() const noexcept -> size_t {
        return publishedDecodedReserveSize.load(std::memory_order_relaxed);
    } ///< Cross-thread snapshot of the total decode-ahead reserve (jitterBuf + handoff); mediaplayer backpressure.
    [[nodiscard]] auto GetQueueSize() const -> size_t;    ///< Packets waiting in the decode queue.
    [[nodiscard]] auto GetStreamAspect() const -> double; ///< Stream DAR (width x SAR), or 0.0 when closed.
    [[nodiscard]] auto GetStreamHeight() const -> int;    ///< Coded stream height, or 0 when closed.
    [[nodiscard]] auto GetStreamWidth() const -> int;     ///< Coded stream width, or 0 when closed.
    [[nodiscard]] auto Initialize() -> bool;              ///< Allocate staging frames, set ready, start thread.
    [[nodiscard]] auto IsQueueEmpty() const -> bool;      ///< VDR Poll(): true -> accept next PES packet.
    [[nodiscard]] auto IsQueueFull() const -> bool;       ///< True when queue has reached DECODER_QUEUE_CAPACITY.
    [[nodiscard]] auto IsInTrickMode() const noexcept -> bool {
        return trickSpeed.load(std::memory_order_acquire) != 0;
    } ///< True while the decoder routes packets through the depth-1 trick queue -- including the short
      ///< window after DevicePlay() where the exit is still resolving on the present thread
      ///< (RequestTrickExit's cancellation grace). Feed gates hold normal-play packets during it.
    [[nodiscard]] auto IsReady() const noexcept -> bool; ///< True after Initialize() succeeds.
    [[nodiscard]] auto IsReadyForNextTrickFrame() const noexcept
        -> bool; ///< True when trick-mode pacing timer has expired.
    [[nodiscard]] auto TakeTrickStep(int64_t pts, int64_t prevPts)
        -> bool; ///< Claim the next paced trick step from outside the present thread; an audio-only replay
                 ///< never reaches SubmitTrickFrame() and drives this pacing state from the feed instead.
                 ///< Non-blocking (a feed thread must not sleep): false = hold still running, true = due,
                 ///< next deadline armed from the pts/prevPts content distance (fast: divided by the rate,
                 ///< slow forward: stretched by the slowdown). @p prevPts is the caller's own previous
                 ///< step (AV_NOPTS_VALUE on entry).
    [[nodiscard]] auto OpenCodec(AVCodecID codecId)
        -> bool; ///< PES path wrapper: no extradata, 8-bit profile assumed; delegates to OpenCodecWithInfo().
    [[nodiscard]] auto OpenCodecWithInfo(const VideoStreamInfo &info)
        -> bool; ///< Open (or reuse) a decoder. HW vs SW selected via SelectVideoBackendCap() + GpuCaps.
                 ///< Reuse requires matching codec ID, HW/SW choice, and extradata; anything else triggers teardown.
    auto NotifyAudioChange() -> void; ///< Arm freerun after an audio codec/track switch; audio clock is NOPTS briefly.
    auto ArmStartTrace(uint64_t epochMs) noexcept
        -> void; ///< Stream-start trace: first decoded / presented / clock-paced frame vs the switch epoch.
    auto DisarmStartTrace() noexcept -> void; ///< Drop pending stream-start milestones.
    auto SetAudioProcessor(cAudioProcessor *audio)
        -> void; ///< Attach the A/V sync master clock. Stored as atomic pointer.
    auto SetLoopTickCallback(std::function<void()> callback)
        -> void; ///< Called once per decode-loop iteration (incl. the ~100 ms idle ticks when no packets arrive),
                 ///< giving the device a thread that ticks even when a scrambled channel delivers no PES. Must be
                 ///< set before Initialize() starts the thread.
    auto SetStreamFormatCallback(std::function<void(uint32_t width, uint32_t height, uint32_t rateMilliHz)> callback)
        -> void; ///< Called from the decode thread after every successful filter-graph build with the coded size and
                 ///< the chain's pre-fps-filter output rate (field rate when deinterlacing, so 1080i25 reports
                 ///< 50000 mHz). The device runs it through the display-mode policy. This is the reactive path for
                 ///< live TV and recordings, where no frame rate is known until FFmpeg has parsed the VUI. Must be
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
    [[nodiscard]] auto IsCodecDrainPending() const noexcept -> bool {
        return codecDrainPending.load(std::memory_order_acquire);
    } ///< True until the decode thread consumes the drain. Keeps the mediaplayer EOS wait alive until
      ///< the reorder-buffer tail has been pushed to the reserve.
    auto SetStillPictureMode(bool mode) -> void; ///< Spatial-only deinterlace for single-frame output; clears on drain.
    auto RequestCodecReopen() -> void;           ///< Force full codec teardown on next OpenCodec() even for same ID.
    auto RequestFilterRebuild()
        -> void; ///< Schedule a debounced filter graph rebuild (e.g. after a ScaleVideo dim change).
    auto RequestTrickExit() -> void; ///< Deferred Play()-without-TrickSpeed(0); cleared if SetTrickSpeed() follows.
    auto SetTrickSpeed(int speed, bool forward = true, bool fast = false)
        -> void;             ///< Configure trick-play pacing. speed=0 returns to normal. fast=true -> key-frames only.
    auto Shutdown() -> void; ///< Stop decode thread and release all resources. Idempotent via stopping flag.

  protected:
    // ========================================================================
    // === THREAD ===
    // ========================================================================
    auto Action() -> void override; ///< Decode thread: dequeue -> VAAPI decode -> filter -> hand off to presenter.

  private:
    // ========================================================================
    // === PRESENTATION THREAD ===
    // ========================================================================
    /// The drain + A/V-sync controller runs on its OWN thread so a slow 4K VPP step on the decode
    /// thread never stalls frame presentation: the presenter drains the decoded reserve at the
    /// audio-synced cadence while the decode thread is still filtering the next frame. cThread runs
    /// exactly one Action() per instance, so the second loop needs a second cThread instance.
    class cPresenter : public cThread {
      public:
        /// @p owner must outlive the thread; cVaapiDecoder::Shutdown() joins before tearing down members.
        explicit cPresenter(cVaapiDecoder *owner) noexcept : cThread("vaapivideo/present"), owner_(owner) {}
        ~cPresenter() noexcept override = default;
        cPresenter(const cPresenter &) = delete;
        cPresenter(cPresenter &&) noexcept = delete;
        auto operator=(const cPresenter &) -> cPresenter & = delete;
        auto operator=(cPresenter &&) noexcept -> cPresenter & = delete;

        auto Stop(int waitSeconds) -> void {
            Cancel(waitSeconds);
        } ///< Public join wrapper (cThread::Cancel is protected).

      protected:
        /// The loop lives in cVaapiDecoder::PresentAction(), next to the state it drives.
        auto Action() -> void override { owner_->PresentAction(); }

      private:
        cVaapiDecoder *owner_; ///< Stable for the presenter's whole life: constructed before Start(), joined in
                               ///< Shutdown() before any member tears down.
    };

    auto PresentAction() -> void; ///< Presentation thread body: splice handoff -> private jitterBuf, due-gated
                                  ///< drain -> SyncAndSubmitFrame. Owns the entire A/V-sync controller state.

    // ========================================================================
    // === INTERNAL METHODS ===
    // ========================================================================
    auto ClearInternal(bool resetFilter, bool preserveSeekHint)
        -> void; ///< Shared body of Clear() / FlushForSeek(). preserveSeekHint=true binds the seek-hint
                 ///< preservation request directly to the jitter-flush request so a coalesced flush
                 ///< observed by the decode thread always sees the matching policy (race-free).
    [[nodiscard]] auto CreateVaapiFrame(AVFrame *src) const
        -> std::unique_ptr<VaapiFrame>; ///< av_frame_clone() the filtered surface; extracts VASurfaceID from data[3].
    [[nodiscard]] auto TakeFilterRebuild() noexcept
        -> bool; ///< Consume a debounced RequestFilterRebuild(); true = rebuild now. Decode thread only.
    auto ClearPendingFilterRebuild() noexcept
        -> void; ///< Cancel a debounced rebuild that an explicit filter reset just subsumed.
    [[nodiscard]] auto DecodeOnePacket(AVPacket *pkt, std::vector<std::unique_ptr<VaapiFrame>> &outFrames)
        -> bool;                         ///< avcodec_send_packet + drain loop. Returns true if any frame was appended.
    auto DrainPendingParserAU() -> void; ///< NULL-input flush of av_parser_parse2. Caller holds parserMutex.
    auto FilterAndAppendDecodedFrame(std::vector<std::unique_ptr<VaapiFrame>> &outFrames)
        -> void; ///< Push decodedFrame through the filter graph (lazily built) and append with monotonic PTS.
                 ///< Caller holds codecMutex and must have populated decodedFrame.
    auto DrainCodecAtEos(std::vector<std::unique_ptr<VaapiFrame>> &outFrames)
        -> void; ///< NULL-packet EOS drain, then avcodec_flush_buffers to re-arm. Caller holds codecMutex.
    auto ApplyContainerColorHints(AVFrame *frame) const noexcept
        -> void; ///< Fill UNSPECIFIED frame color from container hints so VP9/DV streams classify as HDR.
                 ///< Caller holds codecMutex.
    [[nodiscard]] auto ResolveHdrInfo(const AVFrame *frame) const noexcept
        -> HdrStreamInfo; ///< ExtractHdrInfo + container mastering/content-light when the frame lacked it.
                          ///< Caller holds codecMutex.
    [[nodiscard]] auto InitFilterGraph(AVFrame *firstFrame, bool compactLog = false)
        -> bool; ///< Fill BuildParams and delegate to filterChain_.Build(). compactLog=true for
                 ///< ScaleVideo-driven rebuilds (chain line only); false for first build / channel switch.
    [[nodiscard]] auto ShouldUseHdrPassthrough(const HdrStreamInfo &info) const noexcept
        -> bool; ///< True when stream + GPU (vppP010) + display (EDID) + user config all permit HDR passthrough.
    auto TracePresent(int64_t pts, const cAudioProcessor *ap, const char *path, bool paced,
                      int64_t rawDelta90k) noexcept
        -> void; ///< Stream-start trace: first frame handed to the display, and first clock-paced frame.
    [[nodiscard]] auto SubmitIfCurrent(std::unique_ptr<VaapiFrame> frame)
        -> bool; ///< Submit unless clearEpoch raced this iteration; stale-epoch frames are dropped silently
                 ///< (returns true so callers don't count it as a submit failure).
    [[nodiscard]] auto SubmitTrickFrame(std::unique_ptr<VaapiFrame> frame)
        -> bool; ///< Pacing: wait deadline, skip reverse-GOP duplicates, arm next deadline; then submit.
    [[nodiscard]] auto TrickHoldMsFor(int64_t pts, int64_t prevPts) const noexcept
        -> uint64_t; ///< Per-step hold for the current trick mode. Fast: |pts - prevPts| / trickMultiplier,
                     ///< clamped. Slow (multiplier 0) or unusable PTS: the precomputed trickHoldMs.
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
        -> void; ///< Presentation thread only. Stores pts iff presentEpoch matches clearEpoch (Clear-race guard).
    auto ApplyDeferredJitterFlush(uint64_t &lastDrainMs, bool preserveSeekHint) noexcept
        -> void; ///< Consume a pending Clear() / FlushForSeek(): reset EMA, drop jitterBuf, zero pendingDrops
                 ///< + lastDrainMs. preserveSeekHint comes from the flush request itself (not a separate
                 ///< atomic) so racing flushes can't cross-pollinate the preserve policy.
    auto WakePresenter() noexcept -> void; ///< Broadcast handoffCondition (leaf lock) so the present thread promptly
                                           ///< observes a control change it must act on (clearEpoch bump from Clear()/
                                           ///< SetTrickSpeed(0), freerun arm from NotifyAudioChange(), trick-exit, or a
                                           ///< pause change) instead of waiting out the bounded due-gate poll.
    [[nodiscard]] auto ResolvePendingTrickExit() -> bool; ///< Present thread only. Resolves a deferred Play()-out-of-
                                                          ///< trick once the cancellation grace expires: flushes the
                                                          ///< codec for FF/REW, transitions to normal under
                                                          ///< codecMutex->parserMutex, purges the reserve. Returns true
                                                          ///< if it just left trick mode.
    [[nodiscard]] auto PresentEpochStale() const noexcept
        -> bool; ///< Present thread only. True once a Clear()/SetTrickSpeed(0) bumped clearEpoch past the
                 ///< presentEpoch snapshotted this iteration -- the in-flight frame/PTS is a superseded generation.
    [[nodiscard]] auto PresentWakeThreshold90k() const noexcept
        -> int64_t; ///< Present thread only. dueIn (= headPts - clock - latency) threshold below which the head is
                    ///< released: half a frame early when the display queue is empty (pre-fill), else strict-due
                    ///< (halfFrame). Single source of truth shared by the drain loop and the waitMs sleep.
    [[nodiscard]] auto BeginCatchUpLogCycle() noexcept
        -> bool; ///< Catch-up log throttle gate; flushes any pending "cycling settled" summary on the first
                 ///< logged entry after a suppressed run. Returns true if this entry should be logged.

    // ========================================================================
    // === SYNCHRONIZATION ===
    // ========================================================================
    // Lock order: ALWAYS codecMutex -> parserMutex -> packetMutex. DrainQueue takes only packetMutex.
    // handoffMutex is a near-LEAF: no thread acquires codecMutex/parserMutex/packetMutex/display
    // bufferMutex while holding it, and it is never held across display->SubmitFrame() or any
    // cCondWait::SleepMs. Its single outgoing edge is handoffMutex -> vaDriverMutex: destroying a
    // VaapiFrame under it may drop the last FilterGraphToken, whose deleter locks vaDriverMutex.
    // Safe because vaDriverMutex has no path back to any decoder-side lock (it precedes only the
    // display leaf mutexes -- see the display.cpp lock-order comment).
    // (ClearInternal/SetTrickSpeed/etc. take handoffMutex briefly to Broadcast handoffCondition;
    // that codecMutex->handoffMutex nesting relies on the above.)
    //
    // codecMutex and parserMutex are deliberately separate: the dvbplayer / receiver feeds the
    // parser via EnqueueData() while the decode thread is busy submitting work to VAAPI. Sharing
    // one mutex would serialize the two -- in replay that costs ~25 ms/s of GPU submission time
    // on UHD upscale and shows up as sustained negative drift. av_parser_parse2() only reads
    // immutable codecCtx fields (codec_id + descriptor); writers of codecCtx existence
    // (OpenCodecWithInfo / Clear / SetTrickSpeed) take BOTH mutexes in fixed order.
    mutable cMutex codecMutex;   ///< Guards codec context + filter graph (decode thread vs reopen/clear).
    mutable cMutex parserMutex;  ///< Guards parser context (EnqueueData vs reopen/clear).
    mutable cMutex packetMutex;  ///< Guards packetQueue; also used as condvar futex.
    cCondVar packetCondition;    ///< Wakes the decode thread on enqueue or shutdown.
    mutable cMutex handoffMutex; ///< Guards handoffQueue (decode producer -> present consumer). Near-leaf:
                                 ///< dropping the last frame token under it may take vaDriverMutex (see above).
    cCondVar handoffCondition;   ///< Wakes the presentation thread: new handed-off batch, Clear()/FlushForSeek()/
                                 ///< SetTrickSpeed()/NotifyAudioChange()/SetDevicePaused(), or shutdown.
    cCondVar handoffNotFull;     ///< Wakes the decode thread when the presenter drains a full handoffQueue below the
                                 ///< cap (producer-side backpressure so a present-thread sync sleep can't overflow it).

    // ========================================================================
    // === REFERENCES ===
    // ========================================================================
    /// A/V sync master clock. Written by main thread, read by decode thread.
    std::atomic<cAudioProcessor *> audioProcessor{nullptr};
    cVaapiDisplay *display;                 ///< Receives completed VaapiFrames via SubmitFrame().
    VaapiContext *vaapiContext;             ///< Shared VAAPI hw_device_ctx and GpuCaps.
    std::function<void()> loopTickCallback; ///< Per-iteration device hook; set before Initialize(), then read-only.
    std::function<void(uint32_t, uint32_t, uint32_t)>
        streamFormatCallback; ///< Post-filter-build device hook (w, h, rate in mHz) for display-mode matching;
                              ///< set before Initialize(), then read-only.

    // ========================================================================
    // === FFMPEG STATE ===
    // ========================================================================
    /// Active decoder context (HW or SW). Null before OpenCodec().
    std::unique_ptr<AVCodecContext, FreeAVCodecContext> codecCtx;
    AVCodecID currentCodecId{AV_CODEC_ID_NONE}; ///< Codec ID currently open; used for reuse check and parser recreate.
    bool forceCodecReopen{};                    ///< Set by RequestCodecReopen(); cleared by OpenCodecWithInfo().
    bool streamInterlaced{false};               ///< Positive sequence-level hint; forces deinterlace at graph build.
    // Container HDR hints from VideoStreamInfo; set + read under codecMutex. UNSPECIFIED on the PES path.
    AVColorPrimaries hintColorPrimaries{AVCOL_PRI_UNSPECIFIED};             ///< Container colour primaries
    AVColorTransferCharacteristic hintColorTransfer{AVCOL_TRC_UNSPECIFIED}; ///< Container transfer function
    AVColorSpace hintColorSpace{AVCOL_SPC_UNSPECIFIED};                     ///< Container matrix coefficients
    AVColorRange hintColorRange{AVCOL_RANGE_UNSPECIFIED};                   ///< Container range (tv/pc)
    bool hintHasMasteringDisplay{false};               ///< Whether hintMasteringDisplay carries a payload
    AVMasteringDisplayMetadata hintMasteringDisplay{}; ///< HDR10 mastering display volume, valid per the flag above
    bool hintHasContentLight{false};                   ///< Whether hintContentLight carries a payload
    AVContentLightMetadata hintContentLight{};         ///< HDR10 MaxCLL/MaxFALL, valid per the flag above
    /// Staging for avcodec_receive_frame(); unref'd each iteration.
    std::unique_ptr<AVFrame, FreeAVFrame> decodedFrame;
    cVideoFilterChain filterChain; ///< VPP graph (bwdif/deinterlace -> scale_vaapi -> optional denoise/sharpness).
    /// Staging for filterChain.ReceiveFrame(); unref'd each iteration.
    std::unique_ptr<AVFrame, FreeAVFrame> filteredFrame;
    /// Null on mediaplayer path (extradata present). Slices PES NAL bytes into whole AUs.
    std::unique_ptr<AVCodecParserContext, FreeAVCodecParserContext> parserCtx;
    bool trickAwaitSecondField{false}; ///< FF keyframe filter: keep the PAFF I-frame's 2nd field (parserMutex).

    // ========================================================================
    // === PACKET QUEUE ===
    // ========================================================================
    std::queue<AVPacket *> packetQueue; ///< FIFO of parsed packets awaiting HW decode. Owned by packetMutex.
    cTimeMs lastTrickDropWarn;          ///< Rate-limits the trick-enqueue drop log. Held under packetMutex.
    size_t trickDropsSinceWarn{0};      ///< Drops accumulated since the last emitted trick-drop log line.
    std::atomic<bool> stopping{false}; ///< Shutdown signal. Set by Shutdown(); read by encode thread and enqueue paths.

    // ========================================================================
    // === PLAYBACK STATE ===
    // ========================================================================
    std::atomic<bool> codecDrainPending{false};   ///< Decode thread drains codec (NULL packet) then clears this.
    std::atomic<bool> stillPictureMode{false};    ///< Selects spatial-only (bob) deinterlace; cleared after drain.
    std::atomic<bool> hasExited{true};            ///< False only while Action() (decode) runs; checked by Shutdown().
    std::atomic<bool> presentExited{true};        ///< False only while PresentAction() runs; checked by Shutdown().
    std::atomic<bool> hasLoggedFirstFrame{false}; ///< One-time first-frame log guard; reset in OpenCodecWithInfo().
    std::atomic<bool> starvationWarned{false};    ///< One-time "no frame 3 s after open" warning; reset per open.
    /// One-time "still no frame 15 s after open" warning; reset per codec open.
    std::atomic<bool> starvationWarnedSustained{false};
    std::atomic<uint64_t> codecOpenTimeMs{0};   ///< cTimeMs::Now() at last OpenCodecWithInfo(); starvation tiers.
    StreamStartTrace startTrace;                ///< Stream-start milestones (decoded / presented / paced)
    std::atomic<size_t> tracePacketsSent{0};    ///< Packets fed since Arm; shows the reorder delay at first frame
    std::atomic<size_t> packetsSinceOpen{0};    ///< avcodec_send_packet calls since last open; starvation counters.
    std::atomic<size_t> keyPacketsSinceOpen{0}; ///< Subset with AV_PKT_FLAG_KEY; silent feed vs HW stall.
    /// Last decoded PTS in 90 kHz ticks. Read by GetLastPts() / device STC.
    std::atomic<int64_t> lastPts{AV_NOPTS_VALUE};
    std::atomic<uint64_t> clearEpoch{0};   ///< Generation tag for lastPts; bumped by Clear() / SetTrickSpeed(0).
    uint64_t presentEpoch{0};              ///< Presentation thread only. Snapshot of clearEpoch at each present-loop
                                           ///< iteration; gates submit (SubmitIfCurrent), PublishLastPts, and the
                                           ///< drain-loop stale-frame discard. The decode thread needs no iteration
                                           ///< epoch of its own: it stamps producedEpoch from clearEpoch while holding
                                           ///< codecMutex for the producing decode/drain operation.
    std::atomic<bool> liveMode{false};     ///< Hard-ahead policy: replay blocks via WaitForAudioCatchUp, live sleeps.
    std::atomic<bool> devicePaused{false}; ///< Mirrors cVaapiDevice::Freeze()/Play(). When true the drain loop holds
                                           ///< (no submit, no stall-watchdog re-arm) so the head's PTS doesn't drift
                                           ///< while ALSA is dropped and the audio master clock is genuinely frozen.
    std::atomic<bool> ready{false};        ///< Set by Initialize(); gate for OpenCodec() and EnqueueData().
    std::atomic<int> trickSpeed{0};        ///< 0 = normal; >0 = trick mode (speed value mirrors VDR TrickSpeed).
    // Debounced rebuild request (ScaleVideo / zoom), written by RequestFilterRebuild(), consumed
    // by TakeFilterRebuild() on the decode thread. Timestamps are published before the flag.
    std::atomic<uint64_t> filterRebuildFirstRequestMs{0}; ///< Burst start; bounds the total deferral.
    std::atomic<uint64_t> filterRebuildLastRequestMs{0};  ///< Most recent request; restarts the quiet window.
    std::atomic<bool> filterRebuildPending{false};        ///< Set last; TakeFilterRebuild() consumes.
    /// Set by FlushForSeek to request the compact chain-line-only diagnostic on the next InitFilterGraph call instead
    /// of the full 3-line graph init dump. Consumed (exchanged to false) by the decode-thread filter-build path.
    std::atomic<bool> filterCompactRebuildPending{false};

    // ========================================================================
    // === TRICK MODE ===
    // ========================================================================
    std::atomic<bool> deferredTrickExitPending{false}; ///< Play() without TrickSpeed(0); resolved on the present thread
                                                       ///< once the cancellation grace expires (queue-independent, so
                                                       ///< FF can't hang waiting for a keyframe that never arrives).
    /// cTimeMs::Now() deadline after which the deferred exit resolves.
    std::atomic<uint64_t> deferredTrickExitDueMs{0};
    std::atomic<bool> isTrickFastForward{false}; ///< FF mode: only keyframes enqueued; first field of a pair dropped.
    std::atomic<bool> isTrickReverse{false};     ///< REW: GOPs arrive backward; skip frames with rising PTS in a GOP.
    std::atomic<uint64_t> nextTrickFrameDue{0};  ///< cTimeMs::Now() deadline for next submission; enforces pacing.
    std::atomic<int64_t> prevTrickPts{AV_NOPTS_VALUE}; ///< Source PTS of previous trick frame; detects field pairs.
    /// Hold per frame in slow mode = speed * DECODER_TRICK_HOLD_MS. Zero-init to match the normal-play
    /// state both trick-exit paths publish, so a SetTrickSpeed(0) no-op is not mistaken for a change.
    std::atomic<uint64_t> trickHoldMs{0};
    std::atomic<uint64_t> trickMultiplier{0}; ///< Fast-mode PTS-derived hold divisor (2/4/8x). 0 = slow mode.

    // ========================================================================
    // === A/V SYNC ===
    // ========================================================================
    // Default 1 so the very first frame after construction is submitted immediately.
    // Without it the due-gate holds until the audio clock is anchored and the screen stays black.
    /// Bypass A/V sync for N frames. Set by Clear() / trick-exit / NotifyAudioChange().
    std::atomic<int> freerunFrames{1};
    /// Deferred Clear() / FlushForSeek() request consumed by the PRESENTATION thread (single consumer, one
    /// exchange(0) per present iteration):
    ///   - 0 = no flush pending
    ///   - 1 = plain Clear() -- drop seek-hint, content boundary
    ///   - 2 = FlushForSeek() -- preserve seek-hint across the flush
    /// The preserve policy is encoded *in* the request so back-to-back FlushForSeek / Clear() can't cross-pollinate
    /// (a stale flush observed by the present thread always carries its originator's policy, never a later issuer's).
    /// Last writer wins, which is correct: Clear() after FlushForSeek dropping the hint = content boundary win;
    /// FlushForSeek after FlushForSeek = coalesced preserve.
    std::atomic<int> jitterFlushRequest{0};
    std::atomic<size_t> publishedDecodedReserveSize{0}; ///< Cross-thread snapshot of the total decoded reserve
                                                        ///< (jitterBuf.size() + handoffQueue.size()) for backpressure.
                                                        ///< Written by the PRESENT thread once per present iteration,
                                                        ///< read by the mediaplayer demux thread.
    std::atomic<bool> syncLogPending{false};            ///< Force sync log on next frame regardless of timer.
    cTimeMs nextSyncLog;              ///< Presentation thread only. Deadline for the periodic sync-stats evaluation.
    cTimeMs syncLogHeartbeat;         ///< Presentation thread only. Max-silence deadline: forces a sync line even
                                      ///< when nothing changed, so a quiet log still proves the loop is alive.
    int64_t lastLoggedAvg90k{};       ///< Presentation thread only. smoothedDelta90k at the last emitted sync line;
                                      ///< a new line fires when the EMA drifts >= SYNC_LOG_AVG_STEP from it.
    int drainMissCount{};             ///< Drain gaps > 2xframeDur since last sync log = upstream starvation.
                                      ///< Excludes controller-driven pacing (trick, sync sleep, still-frame hold).
    int syncDropSinceLog{};           ///< Frames dropped (video behind) since last sync log. Presentation thread only.
    int syncSkipSinceLog{};           ///< Frames delayed (video ahead) since last sync log. Presentation thread only.
    int pendingDrops{};               ///< Remaining frames to drop in current soft- or hard-behind burst.
                                      ///< Consumed one-per-iteration in SyncAndSubmitFrame. Presentation thread only.
    bool sleptInLastSubmit{};         ///< Set by SyncAndSubmitFrame / WaitForAudioCatchUp when a sync-correction
                                      ///< sleep ran. Consumed (cleared) by the drain loop's miss check so the
                                      ///< self-inflicted gap doesn't inflate drainMissCount. Presentation thread only.
    int64_t rawDeltaSumSinceLog90k{}; ///< Accumulator for interval-mean rawDelta in the sync log.
    int rawDeltaCountSinceLog{};      ///< Sample count for rawDeltaSumSinceLog90k.
    int warmupSampleCount{};          ///< Samples accumulated during EMA warmup (post-reset). Presentation thread only.
    int64_t warmupSampleSum90k{};     ///< Warmup accumulator in 90 kHz ticks. Presentation thread only.
    int64_t smoothedDelta90k{};       ///< EMA-smoothed A/V delta in 90 kHz ticks. Presentation thread only.
    int64_t emaResidual90k{};         ///< Integer EMA remainder: carries sub-sample rounding so the filter converges
                                      ///< exactly to the mean rather than stalling when |diff| < EMA_SAMPLES ticks.
    bool smoothedDeltaValid{false};   ///< True after warmup completes. Gates soft-corridor and catch-up (sustained).
    /// One-shot fast-start hint: last converged smoothedDelta carried across a FlushForSeek. Used as the catch-up exit
    /// target AND as the EMA seed on the first valid post-seek sample, so playback resumes at the steady-state offset
    /// within milliseconds instead of waiting out the 50-sample warmup. Consumed (set back to AV_NOPTS_VALUE) once the
    /// EMA is seeded so subsequent controller resets warm up from real samples rather than re-applying the same stale
    /// value. Presentation thread only.
    int64_t seekHintDelta90k{AV_NOPTS_VALUE};
    /// Pre-correction snapshot of smoothedDelta taken at each hard-/soft-ahead trigger, just before the post-sleep
    /// `smoothedDelta -= extraMs` feedback. Represents the long-term GPU-vs-audio offset that the EMA had converged to
    /// before the in-flight correction perturbed it. Used as the seek-hint source so the captured hint survives a
    /// FlushForSeek that lands inside the post-correction recovery window. AV_NOPTS_VALUE until the first trigger
    /// fires; cleared by ResetSmoothedDelta. Presentation thread only.
    int64_t stableDelta90k{AV_NOPTS_VALUE};
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
    // within DECODER_SYNC_CATCHUP_LOG_INTERVAL_MS of each other are a "run". The first entry of a run logs
    // normally; subsequent entries in the run are suppressed and aggregated into a periodic
    // (DECODER_SYNC_CATCHUP_SUMMARY_INTERVAL_MS) "sustained" summary while the run continues, plus a final
    // "settled" summary when the gap exceeds the window (run ended). Controller behavior is unchanged.
    uint64_t lastCatchUpEntryMs{};   ///< cTimeMs::Now() of the most recent catch-up entry (logged OR suppressed).
                                     ///< Inter-entry gap vs DECODER_SYNC_CATCHUP_LOG_INTERVAL_MS detects an active run.
    bool catchUpLogThisCycle{false}; ///< Set at entry, consulted at exit so suppressed entries don't log their exit.
    int suppressedCatchUpCycles{};   ///< Cycles aggregated since the run started (or last periodic summary).
    int suppressedCatchUpDrops{};    ///< Cumulative dropped-frame count of those cycles.
    uint64_t suppressedCatchUpWallMs{}; ///< Cumulative wall time spent inside those cycles.
    uint64_t nextCatchUpSummaryMs{};    ///< cTimeMs::Now() at which the next periodic-during-run summary should fire.
    cTimeMs staleJitterLogGate;         ///< Rate-limits the "stale-jitter bulk" drop log (present thread); a rapid-seek
                                ///< burst fires it per seek and would otherwise flood/block the drain on syslog.
    int staleJitterDropsSinceLog{}; ///< Stale-jitter frames dropped while the gate suppressed the log.

    // ========================================================================
    // === JITTER BUFFER ===
    // ========================================================================
    /// Decode->present handoff. Producer: decode thread (push under handoffMutex). Consumer: present thread
    /// (splice under handoffMutex). FIFO; bounded by DECODER_RESERVE_HARD_CAP with producer backpressure
    /// via handoffNotFull.
    std::deque<std::unique_ptr<VaapiFrame>> handoffQueue;
    /// Decoded frames pending display. Presentation thread only (spliced from handoffQueue each present
    /// iteration); see AVSYNC.md.
    std::deque<std::unique_ptr<VaapiFrame>> jitterBuf;
    std::atomic<int> outputFrameDurationMs{20}; ///< Field/frame duration ms. Written by the decode thread after the
                                                ///< filter graph is (re)built; read by the present-thread sync math and
                                                ///< the decode-thread PTS stamping. Standalone scalar: relaxed both
                                                ///< ways (no happens-before role; a one-frame-stale read across a graph
                                                ///< rebuild only perturbs pacing for one frame and the EMA
                                                ///< reconverges). 20 ms = 50fps field rate, 40 ms = 25fps.

    // Declared LAST so it is destroyed FIRST: the presenter thread (whose PresentAction body touches
    // handoffMutex/handoffQueue/jitterBuf and the controller scalars above) must be torn down before
    // those members. The dtor->Shutdown() join is the primary guarantee; this ordering is defense-in-depth.
    /// Presentation (drain + A/V-sync) thread; started/stopped alongside the decode thread.
    cPresenter presenter{this};
};

#endif // VDR_VAAPIVIDEO_DECODER_H
