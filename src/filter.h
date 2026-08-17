// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file filter.h
 * @brief VAAPI VPP filter chain: deinterlace, denoise, scale, sharpness, and HDR classification.
 *
 * cVideoFilterChain is standalone and codec-agnostic: Build() takes all decisions through
 * BuildParams so this class has no dependency on cVaapiDecoder, cVaapiDisplay, or
 * VaapiContext. Every frame a graph produces holds a FilterGraphToken on it, so a retired
 * graph outlives its in-flight output surfaces (see FilterGraphToken -- destroying it
 * earlier is a use-after-free in the VA driver). HDR classification helpers (ClassifyStream,
 * ExtractHdrInfo, ...) are co-located here because they operate on decoded AVFrames and
 * drive the scale_vaapi color directives.
 */

#ifndef VDR_VAAPIVIDEO_FILTER_H
#define VDR_VAAPIVIDEO_FILTER_H

#include "common.h"
#include "config.h"
#include "stream.h"

#include <string>

// VDR
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <vdr/thread.h>
#pragma GCC diagnostic pop

// ============================================================================
// === FRAME CLASSIFICATION HELPERS ===
// ============================================================================

/// Returns the software pixel format for a decoded frame regardless of whether it was
/// decoded in hardware (AV_PIX_FMT_VAAPI) or software. Needed to determine bit depth
/// without pulling a VAAPI surface back to system memory via av_hwframe_transfer_data().
[[nodiscard]] auto ResolveSwPixFmt(const AVFrame *frame) noexcept -> AVPixelFormat;

/// True if the frame's luma samples are at least @p minBits wide. Uses the pixel-format
/// descriptor rather than a hard-coded format list so driver-chosen surfaces (P010, NV20,
/// YUV420P10LE) all resolve correctly.
[[nodiscard]] auto FrameBitDepthAtLeast(const AVFrame *frame, int minBits) noexcept -> bool;

/// Classify HDR kind from color_primaries + color_trc + bit depth. Codec profile is not
/// a gate: HEVC Main10 can carry BT.709 SDR, and ad-insertion frames inside an HDR
/// program correctly fall back to Sdr.
[[nodiscard]] auto ClassifyStream(const AVFrame *frame) noexcept -> StreamHdrKind;

/// Extract HDR10 static metadata (mastering display + content light level) from
/// AVFrame side-data. Absence of either blob is non-fatal; check hasMasteringDisplay /
/// hasContentLight before reading the payload fields.
[[nodiscard]] auto ExtractHdrInfo(const AVFrame *frame) noexcept -> HdrStreamInfo;

// ============================================================================
// === FILTER GRAPH TOKEN ===
// ============================================================================

/// Deleter that frees a filter graph under the VA driver mutex. The last FilterGraphToken can be
/// dropped on any thread (decode, present, display, or main during Clear()), and
/// avfilter_graph_free() issues vaDestroyContext/vaDestroyConfig -- unserialized VA-driver entry.
/// Re-entry from a thread already holding the lock (a Reset() that drops the last reference) is
/// safe on every supported VDR: cMutex::Lock() short-circuits same-thread relock (thread.c 5.7,
/// "Improved recursive locking in cMutex"), and older releases get the same net effect because
/// the ERRORCHECK mutex returns EDEADLK without blocking while the `locked` counter balances.
/// No other lock is taken while freeing.
struct FreeFilterGraphLocked {
    cMutex *vaDriverMutex{nullptr}; ///< Borrowed from cVaapiDisplay; outlives every token (device tears
                                    ///< the decoder -- and all frames -- down before the display)
    /// Free @p graph, holding vaDriverMutex across the teardown when one was supplied.
    auto operator()(AVFilterGraph *graph) const noexcept -> void;
};

/// Keep-alive handle every produced frame holds on the graph that produced it.
///
/// Why the graph and not just the surface: a VPP surface stays tied to the VA context that
/// rendered it, and syncing the surface -- which av_hwframe_map() does on export -- reaches back
/// into that context. FFmpeg refcounts the surface through the AVFrame, but nothing refcounts the
/// context, so a graph freed while its frames are still queued (up to ~1.3 s of them:
/// DECODER_RESERVE_HARD_CAP + display prerender) crashes the display thread on its next export.
/// Rapid ScaleVideo() rebuilds (skin menu open/close) hit exactly that window.
/// Const: token holders extend the graph's lifetime, never touch it.
using FilterGraphToken = std::shared_ptr<const AVFilterGraph>;

// ============================================================================
// === VIDEO FILTER CHAIN ===
// ============================================================================

/// VAAPI post-processing filter graph with frame-lifetime-tied retirement.
///
/// Lifecycle: Build() on the first decoded frame -> SendFrame/ReceiveFrame in a loop ->
/// Reset() on format change or stream end (retires the graph; in-flight FilterGraphTokens may
/// keep it alive) -> Build() again. Destructor drops the chain's own reference.
///
/// Thread safety: not thread-safe. The decoder thread owns the instance and must hold the codec
/// mutex (and VA driver mutex) around Build() and Reset(). A retired graph may outlive the chain;
/// whichever thread drops the last token frees it, serialized by FreeFilterGraphLocked.
class cVideoFilterChain {
  public:
    cVideoFilterChain() = default;
    ~cVideoFilterChain() noexcept = default;
    cVideoFilterChain(const cVideoFilterChain &) = delete;
    cVideoFilterChain(cVideoFilterChain &&) noexcept = delete;
    auto operator=(const cVideoFilterChain &) -> cVideoFilterChain & = delete;
    auto operator=(cVideoFilterChain &&) noexcept -> cVideoFilterChain & = delete;

    /// All inputs Build() needs. Filled once per Build() call; no retained reference after return.
    struct BuildParams {
        // --- Source stream ---
        AVCodecID codecId{AV_CODEC_ID_NONE}; ///< Used to tune denoise/sharpen levels (MPEG-2 vs. H.264/H.265)
        int fpsNum{0};                       ///< codecCtx->framerate.num; 0 = unknown (defaults to 50)
        int fpsDen{1};                       ///< codecCtx->framerate.den
        AVBufferRef *hwFramesCtx{nullptr};   ///< codecCtx->hw_frames_ctx; nullptr for SW decode
        AVBufferRef *hwDeviceRef{nullptr};   ///< VAAPI device (required, borrowed -- Build() refs internally)
        bool streamInterlaced{false};        ///< Positive sequence/container hint; forces deinterlace when set
        cMutex *vaDriverMutex{nullptr};      ///< Stored in the graph's deleter -- see FreeFilterGraphLocked

        // --- Target surface ---
        uint32_t outputWidth{0};  ///< Target video rect width; VPP output is DAR-fitted for 1:1 KMS scanout
        uint32_t outputHeight{0}; ///< Target video rect height
        uint32_t outputParNum{1}; ///< Scanout pixel aspect (num:den), from ModePixelAspectRatio(). 1:1 on every
        uint32_t outputParDen{1}; ///< square-pixel timing; 64:45 on a 720x576 flagged 16:9. The DAR fit below has
                                  ///< to account for it -- KMS scanout is 1:1, so an anamorphic raster can only be
                                  ///< compensated here, and the TV stretches the result back out on its own.
        uint32_t outputRefreshMilliHz{0}; ///< Display refresh in millihertz; decides whether an fps re-timing filter
                                          ///< is needed. Millihertz, not Hz: on a display running a genuine 59.94 Hz
                                          ///< mode an integer 60 would look like a mismatch and insert an fps=60
                                          ///< filter that duplicates a frame roughly every 1000.

        // --- HDR decisions (resolved by caller before Build) ---
        bool hdrPassthrough{false}; ///< true -> emit P010 + BT.2020 color directives; false -> NV12 BT.709
        HdrStreamInfo hdrInfo{};    ///< Determines PQ vs. HLG transfer function in scale_vaapi args

        // --- GPU capabilities (queried from GpuCaps by caller) ---
        bool hasDenoise{false};           ///< denoise_vaapi is available on this device
        bool hasSharpness{false};         ///< sharpness_vaapi is available on this device
        std::string_view deinterlaceMode; ///< Best advertised mode "motion_adaptive"/"bob"/...; empty = skip HW deint
        /// Bit (1u<<VppDeintMode) per driver-supported mode; bounds ClampDeinterlaceMode
        unsigned deinterlaceModeMask{};

        // --- Post-processing policies (resolved from config by caller) ---
        DeinterlaceMode userDeint{DeinterlaceMode::Auto}; ///< Deinterlacer selection (GPU clamp vs. SW block)
        DenoiseMode denoise{DenoiseMode::Auto};           ///< Denoise strength (denoise_vaapi or hqdn3d)
        ScaleMode scale{ScaleMode::Auto};                 ///< Scaler (scale_vaapi hq/fast or swscale lanczos)
        SharpenMode sharpen{SharpenMode::Auto};           ///< Sharpening (sharpness_vaapi or unsharp)

        // --- Manual zoom (symmetric crop before scale) ---
        double cropH{0.0}; ///< Per-side horizontal crop fraction (decoder derives it from the zoom-in factor); 0 = none
        double cropV{0.0}; ///< Per-side vertical crop fraction (decoder derives it from the zoom-in factor); 0 = none

        // --- Playback mode flags ---
        bool trickMode{false};    ///< Trick speed: use minimal chain + bob deint (no priming delay)
        bool stillPicture{false}; ///< Single I-frame: skip temporal deinterlace (would never flush)
        bool compactLog{false};   ///< true for ScaleVideo-only rebuilds: one-line dsyslog instead of full diagnostic
    };

    /// Build the filter graph. Source geometry (size, SAR, interlaced flag, pixel format) is
    /// read from @p firstFrame; all policy decisions come from @p params. Idempotent: returns
    /// true immediately if already built. Returns false and leaves the chain unbuilt on error.
    [[nodiscard]] auto Build(AVFrame *firstFrame, const BuildParams &params) -> bool;

    /// Feed a decoded frame into the graph. Pass nullptr to signal EOS and flush.
    /// Returns 0 on success, negative FFmpeg error otherwise. Returns AVERROR(EINVAL)
    /// if called before Build() succeeds.
    [[nodiscard]] auto SendFrame(AVFrame *frame) noexcept -> int;

    /// Pull one filtered frame. Returns 0 on success (caller owns @p out and must unref),
    /// AVERROR(EAGAIN) if no frame is ready yet, AVERROR_EOF when the graph is drained.
    [[nodiscard]] auto ReceiveFrame(AVFrame *out) noexcept -> int;

    /// True after Build() succeeds and until Reset() is called.
    [[nodiscard]] auto IsBuilt() const noexcept -> bool { return graph_ != nullptr; }

    /// Token for the graph currently producing frames (null while unbuilt). Every frame pulled from
    /// ReceiveFrame() must carry one until its surface is retired -- see FilterGraphToken.
    [[nodiscard]] auto CurrentGraphToken() const noexcept -> FilterGraphToken { return graph_; }

    /// Retire the active graph: drop the chain's reference and unbuild. In-flight tokens keep the
    /// graph alive; the last one dropped frees it. Idempotent.
    auto Reset() noexcept -> void;

    /// Approximate output frame duration in milliseconds (1000 / outputFps), computed in
    /// Build() from framerate, interlaced flag, and upconvert decision. Returns the 20 ms
    /// (50 fps) fallback before the first successful Build(), after Reset(), and after a
    /// failed Build(). Used by the A/V sync controller.
    [[nodiscard]] auto GetOutputFrameDurationMs() const noexcept -> int { return outputFrameDurationMs_; }

    /// True iff the active chain contains a temporal filter whose internal state survives a
    /// seek-sized PTS jump destructively. Currently only `fps=N` qualifies: with a source
    /// rate below the display rate, that filter bridges the gap between previous_output_pts
    /// and the first post-seek input by emitting many duplicate frames at stale PTS,
    /// triggering long stale-jitter / catch-up cascades. bwdif / yadif retain at most 1-2
    /// fields of pre-seek content and self-clear inside a filter window -- not flagged.
    /// FlushForSeek consults this to decide whether to pay the ~100 ms filter-rebuild cost.
    /// Atomic (relaxed): FlushForSeek reads it from the mediaplayer thread before taking
    /// codecMutex, while Build()/Reset() write it on the decode thread.
    [[nodiscard]] auto HasFpsFilter() const noexcept -> bool { return hasFpsFilter_.load(std::memory_order_relaxed); }

    /// Rate the chain emits in millihertz, before any fps re-timing: source rate times the
    /// field-rate factor (2x when a field-rate deinterlacer is active, so 1080i25 reports 50000).
    /// Display-independent by construction, which is what makes it usable as the mode-matcher's
    /// input -- feeding back the post-fps-filter rate would just re-assert the current mode.
    /// Returns 0 before the first successful Build() and after Reset().
    [[nodiscard]] auto NaturalOutputRateMilliHz() const noexcept -> uint32_t {
        return naturalOutputRateMilliHz_.load(std::memory_order_relaxed);
    }

  private:
    /// Reset the derived state (hasFpsFilter_ / outputFrameDurationMs_ / ctx pointers) to the
    /// unbuilt defaults and return false. Used on every Build() failure path (the half-built
    /// graph is a Build() local and frees itself); never touches graph_.
    [[nodiscard]] auto FailBuild() noexcept -> bool;

    /// Active graph; null until the first Build() and after every Reset(). Produced frames hold
    /// FilterGraphTokens on it, which is what keeps a retired graph's VA context alive.
    std::shared_ptr<AVFilterGraph> graph_;
    /// Source args + chain + geometry of the last *logged* build. Compact rebuilds matching it
    /// byte-for-byte skip the chain line (reverse trick rebuilds an identical graph once per
    /// keyframe step). Deliberately survives Reset() -- it tracks the log, not the graph.
    std::string lastChainLogKey_;
    AVFilterContext *bufferSrcCtx_{};       ///< owned by graph_; raw pointer valid only while graph_ is live
    AVFilterContext *bufferSinkCtx_{};      ///< owned by graph_; same lifetime constraint
    int outputFrameDurationMs_{20};         ///< 20 = 50 fps fallback; updated by Build()
    std::atomic<bool> hasFpsFilter_{false}; ///< true iff the active chain ends with `fps=N`; Build()/Reset() write it
    std::atomic<uint32_t> naturalOutputRateMilliHz_{0}; ///< Pre-fps-filter output rate; read by the decode thread to
                                                        ///< publish the stream format for display-mode matching
};

#endif // VDR_VAAPIVIDEO_FILTER_H
