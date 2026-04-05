// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file decoder.h
 * @brief Threaded VAAPI decoder with filter graphs and A/V sync
 */

#ifndef VDR_VAAPIVIDEO_DECODER_H
#define VDR_VAAPIVIDEO_DECODER_H

#include "common.h"

#include <deque>

class cAudioProcessor;
class cVaapiDisplay;
struct VaapiContext;

// ============================================================================
// === CONSTANTS ===
// ============================================================================

// Queue and timing
inline constexpr size_t DECODER_TRICK_QUEUE_DEPTH = 1; ///< Trick-mode queue depth (single-packet handoff)
inline constexpr int DECODER_TRICK_HOLD_MS = 20;       ///< Per-frame hold time for slow trick mode (one field @ 50 Hz)

// ============================================================================
// === STRUCTURES ===
// ============================================================================

/**
 * @struct VaapiFrame
 * @brief Decoded output frame carrying a VAAPI surface reference and presentation timestamp
 */
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
    AVFrame *avFrame{};                          ///< AVFrame that holds the VAAPI surface buffer reference
    bool ownsFrame{true};                        ///< True when this instance is responsible for freeing avFrame
    int64_t pts{AV_NOPTS_VALUE};                 ///< Presentation timestamp in 90 kHz units
    VASurfaceID vaSurfaceId{VA_INVALID_SURFACE}; ///< VAAPI surface ID used for zero-copy DRM export
};

// ============================================================================
// === DECODER CLASS ===
// ============================================================================

/**
 * @class cVaapiDecoder
 * @brief Threaded VAAPI decoder: parses PES data, decodes (HW or SW), post-processes via filter graph, and submits
 * frames
 */
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
    auto Clear() -> void;      ///< Flush queued packets, codec buffers, and filter graph; reset A/V sync
    auto DrainQueue() -> void; ///< Discard all queued packets without touching the codec or filter state
    auto EnqueueData(const uint8_t *data, size_t size, int64_t pts)
        -> void; ///< Parse raw PES payload and push resulting packets onto the decode queue
    [[nodiscard]] auto GetLastPts() const noexcept
        -> int64_t; ///< Return the PTS of the most recently submitted frame (90 kHz), or AV_NOPTS_VALUE
    [[nodiscard]] auto GetQueueSize() const
        -> size_t; ///< Return the current number of packets waiting in the decode queue
    [[nodiscard]] auto GetStreamAspect() const
        -> double; ///< Return the stream DAR (width x SAR), or 0.0 when no codec is open
    [[nodiscard]] auto GetStreamHeight() const -> int; ///< Return the coded stream height, or 0 when no codec is open
    [[nodiscard]] auto GetStreamWidth() const -> int;  ///< Return the coded stream width, or 0 when no codec is open
    [[nodiscard]] auto Initialize()
        -> bool; ///< Allocate working frames, mark decoder ready, and start the decode thread
    [[nodiscard]] auto IsQueueEmpty() const -> bool; ///< Return true when the decode queue contains no packets
    [[nodiscard]] auto IsQueueFull() const
        -> bool; ///< Return true when the decode queue has reached DECODER_QUEUE_CAPACITY
    [[nodiscard]] auto IsReady() const noexcept -> bool; ///< Return true after Initialize() has completed successfully
    [[nodiscard]] auto IsReadyForNextTrickFrame() const noexcept
        -> bool; ///< Return true when the trick-mode pacing timer has expired
    [[nodiscard]] auto OpenCodec(AVCodecID codecId)
        -> bool; ///< Open (or reuse) a decoder for the given codec (VAAPI HW if available, else SW fallback)
    auto NotifyAudioChange() -> void; ///< Reset A/V sync after an audio codec change
    auto SetAudioProcessor(cAudioProcessor *audio)
        -> void;                         ///< Attach the audio processor used as the A/V sync master clock
    auto SetLiveMode(bool live) -> void; ///< Enable jitter buffering for live TV; disable for replay
    auto RequestCodecReopen()
        -> void; ///< Force the next OpenCodec() call to do a full teardown/reopen even for the same codec
    auto RequestTrickExit()
        -> void; ///< Deferred trick-mode exit from Play(); cleared if TrickSpeed() follows before the next frame
    auto SetTrickSpeed(int speed, bool forward = true, bool fast = false)
        -> void;             ///< Configure trick-play mode; speed 0 returns to normal playback
    auto Shutdown() -> void; ///< Stop the decode thread and release all codec resources

  protected:
    // ========================================================================
    // === THREAD ===
    // ========================================================================
    auto Action() -> void override; ///< Decode-thread body: dequeues packets, decodes, filters, and submits frames

  private:
    // ========================================================================
    // === INTERNAL METHODS ===
    // ========================================================================
    [[nodiscard]] auto CreateVaapiFrame(AVFrame *src) const
        -> std::unique_ptr<VaapiFrame>; ///< Wrap a filtered AVFrame in a VaapiFrame (increments VAAPI surface refcount)
    [[nodiscard]] auto DecodeOnePacket(AVPacket *pkt,
                                       std::vector<std::unique_ptr<VaapiFrame>> &outFrames)
        -> bool; ///< Send one packet to the VAAPI decoder and drain all resulting filtered frames into outFrames
    [[nodiscard]] auto InitFilterGraph(AVFrame *firstFrame)
        -> bool; ///< Build VPP filter graph: HW path or SW (bwdif/hqdn3d -> hwupload) + scale/sharpen
    auto ResetFilterGraph() -> void; ///< Null filter pointers and destroy the graph (idempotent)
    [[nodiscard]] auto SubmitTrickFrame(std::unique_ptr<VaapiFrame> frame)
        -> bool; ///< Trick-mode pacing: enforce timing, reverse-PTS filtering, then submit
    [[nodiscard]] auto SyncAndSubmitFrame(std::unique_ptr<VaapiFrame> frame)
        -> bool; ///< Apply A/V sync policy (wait / drop / pass) and forward the frame to the display
    [[nodiscard]] auto SyncLatency90k() const noexcept
        -> int64_t; ///< Combined A/V sync latency: audioLatency + one-frame pipeline delay (90 kHz ticks)
    auto WaitForAudioCatchUp(cAudioProcessor *ap, int64_t pts, int64_t latency, int64_t delta)
        -> void; ///< Block decode thread until audio clock reaches video PTS (replay ahead correction)

    // ========================================================================
    // === SYNCHRONIZATION ===
    // ========================================================================
    mutable cMutex codecMutex;  ///< Serializes codec context access between EnqueueData() and the decode thread
    mutable cMutex packetMutex; ///< Protects packetQueue and coordinates producer/consumer wake-ups
    cCondVar packetCondition;   ///< Signals the decode thread when a new packet has been enqueued

    // ========================================================================
    // === REFERENCES ===
    // ========================================================================
    std::atomic<cAudioProcessor *> audioProcessor{nullptr}; ///< Audio processor providing the master clock for A/V sync
    cVaapiDisplay *display;                                 ///< Display that receives completed VaapiFrames
    VaapiContext *vaapiContext;                             ///< Shared VAAPI hardware device context

    // ========================================================================
    // === FFMPEG STATE ===
    // ========================================================================
    AVFilterContext *bufferSinkCtx{}; ///< Output node of the VAAPI filter graph (buffersink)
    AVFilterContext *bufferSrcCtx{};  ///< Input node of the VAAPI filter graph (buffer source)
    std::unique_ptr<AVCodecContext, FreeAVCodecContext> codecCtx; ///< Active decoder context (HW or SW)
    AVCodecID currentCodecId{AV_CODEC_ID_NONE};                   ///< Codec ID currently open in codecCtx
    bool forceCodecReopen{};                                      ///< Force full teardown on next OpenCodec()
    std::unique_ptr<AVFrame, FreeAVFrame> decodedFrame; ///< Reusable staging frame for avcodec_receive_frame() output
    std::unique_ptr<AVFilterGraph, FreeAVFilterGraph>
        filterGraph; ///< VAAPI post-processing filter graph; null until first frame
    std::unique_ptr<AVFrame, FreeAVFrame>
        filteredFrame; ///< Reusable staging frame for av_buffersink_get_frame() output
    std::unique_ptr<AVCodecParserContext, FreeAVCodecParserContext>
        parserCtx; ///< Bitstream parser that extracts complete access units from PES payloads

    // ========================================================================
    // === PACKET QUEUE ===
    // ========================================================================
    std::queue<AVPacket *> packetQueue; ///< FIFO of parsed packets awaiting hardware decode
    std::atomic<bool> stopping;         ///< Set to true when the decoder is shutting down

    // ========================================================================
    // === PLAYBACK STATE ===
    // ========================================================================
    std::atomic<bool> hasExited;                  ///< Set to true by the decode thread just before it returns
    std::atomic<bool> hasLoggedFirstFrame;        ///< Guards the one-time first-frame info log
    std::atomic<int64_t> lastPts{AV_NOPTS_VALUE}; ///< PTS of the last submitted frame, read by GetLastPts() / GetSTC()
    std::atomic<bool> liveMode;                   ///< True for live TV (jitter buffer active); false for replay
    std::atomic<bool> ready;                      ///< True once Initialize() has succeeded
    std::atomic<int> trickSpeed;                  ///< Current trick speed: 0 = normal play, >0 = trick mode

    // ========================================================================
    // === TRICK MODE ===
    // ========================================================================
    std::atomic<bool> trickExitPending;   ///< Deferred exit requested by Play(); cleared by SetTrickSpeed()
    std::atomic<bool> isTrickFastForward; ///< True in fast-forward mode (forward=true, fast=true)
    std::atomic<bool> isTrickReverse;     ///< True in fast-reverse trick mode; enforces monotonically decreasing PTS
    std::atomic<uint64_t>
        nextTrickFrameDue; ///< Absolute time (cTimeMs::Now()) when the next trick frame may be submitted
    std::atomic<int64_t> prevTrickPts{
        AV_NOPTS_VALUE}; ///< PTS of the previous trick-mode source frame, used for field-pair detection
    std::atomic<uint64_t> trickHoldMs{
        DECODER_TRICK_HOLD_MS};            ///< Per-frame hold duration for slow trick mode (ms), set by SetTrickSpeed()
    std::atomic<uint64_t> trickMultiplier; ///< Fast-mode speed multiplier (2 / 4 / 8x); 0 means slow mode

    // ========================================================================
    // === A/V SYNC ===
    // ========================================================================
    std::atomic<int> freerunFrames; ///< Frames to submit without sync after Clear() (written by main thread)
    std::atomic<bool> jitterFlushPending{
        false};                       ///< Set by Clear(); decode thread applies jitter reset on next delivery cycle
    std::atomic<bool> syncLogPending; ///< Triggers sync log on next frame
    cTimeMs nextSyncLog;              ///< Deadline for the periodic "sync status" log (decoder thread only)
    int correctDrops{};               ///< Correction drops, summarized when run ends (decoder thread only)
    cTimeMs syncGrace;                ///< Suppresses Phase-3 corrections after sync and after each correction

    // ========================================================================
    // === JITTER BUFFER (live TV only) ===
    // ========================================================================
    std::deque<std::unique_ptr<VaapiFrame>> jitterBuf; ///< Decoded frames awaiting display (decoder thread only)
    bool jitterPrimed{};                               ///< True once jitterBuf has accumulated enough frames
    int jitterTarget{};            ///< Frame count needed to fill DECODER_JITTER_BUFFER_MS (set by InitFilterGraph)
    int outputFrameDurationMs{20}; ///< Duration per output frame in ms (e.g. 20 for 50fps, 40 for 25fps)
    uint64_t jitterDrainTimeMs{0}; ///< Wall-clock timestamp (ms) of the last jitter-buffer drain
    uint64_t lastSeenVSyncMs{0};   ///< VSync timestamp at last jitter-buffer drain, for VSync-aligned pacing
};

#endif // VDR_VAAPIVIDEO_DECODER_H
