// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file filter.h
 * @brief VAAPI VPP filter chain: deinterlace, denoise, scale, sharpness, and HDR classification.
 *
 * cVideoFilterChain is standalone and codec-agnostic: Build() takes all decisions through
 * BuildParams so this class has no dependency on cVaapiDecoder, cVaapiDisplay, or
 * VaapiContext. A keep-alive slot holds the previous graph until the display thread
 * finishes DMA-BUF mapping its VPP output surfaces (destroying earlier causes -EIO on iHD).
 * HDR classification helpers (ClassifyStream, ExtractHdrInfo, ...) are co-located here
 * because they operate on decoded AVFrames and drive the scale_vaapi colour directives.
 */

#ifndef VDR_VAAPIVIDEO_FILTER_H
#define VDR_VAAPIVIDEO_FILTER_H

#include "common.h"
#include "stream.h"

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
/// programme correctly fall back to Sdr.
[[nodiscard]] auto ClassifyStream(const AVFrame *frame) noexcept -> StreamHdrKind;

/// Extract HDR10 static metadata (mastering display + content light level) from
/// AVFrame side-data. Absence of either blob is non-fatal; check hasMasteringDisplay /
/// hasContentLight before reading the payload fields.
[[nodiscard]] auto ExtractHdrInfo(const AVFrame *frame) noexcept -> HdrStreamInfo;

// ============================================================================
// === VIDEO FILTER CHAIN ===
// ============================================================================

// VAAPI post-processing filter graph with keep-alive reset.
//
// Lifecycle: Build() on the first decoded frame -> SendFrame/ReceiveFrame in a loop ->
// Reset() on format change or stream end (keeps old graph alive) -> Build() again.
// Destructor releases both the active and keep-alive graphs.
//
// Thread safety: not thread-safe. The decoder thread owns the instance and must
// hold the codec mutex (and VA driver mutex) around Build() and Reset(), matching
// the pre-refactor InitFilterGraph / ResetFilterGraph call sites.
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

        // --- Target surface ---
        uint32_t outputWidth{0};     ///< DRM plane width; VPP output is scaled to this (no HW scaler downstream)
        uint32_t outputHeight{0};    ///< DRM plane height
        uint32_t outputRefreshHz{0}; ///< Used to decide whether to insert an fps upconvert filter

        // --- HDR decisions (resolved by caller before Build) ---
        bool hdrPassthrough{false}; ///< true -> emit P010 + BT.2020 colour directives; false -> NV12 BT.709
        HdrStreamInfo hdrInfo{};    ///< Determines PQ vs. HLG transfer function in scale_vaapi args

        // --- GPU capabilities (queried from GpuCaps by caller) ---
        bool hasDenoise{false};           ///< denoise_vaapi is available on this device
        bool hasSharpness{false};         ///< sharpness_vaapi is available on this device
        std::string_view deinterlaceMode; ///< "motion_adaptive" / "bob" / ...; empty = skip HW deint

        // --- Playback mode flags ---
        bool trickMode{false};    ///< Trick speed: use minimal chain + bob deint (no priming delay)
        bool stillPicture{false}; ///< Single I-frame: skip temporal deinterlace (would never flush)
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
    [[nodiscard]] auto IsBuilt() const noexcept -> bool { return filterGraph_ != nullptr; }

    /// Move the active graph to the keep-alive slot without destroying it. The old graph
    /// is released on the next Build() or destructor, giving the display thread time to
    /// finish DMA-BUF mapping. Idempotent; never overwrites a saved graph with null.
    auto Reset() noexcept -> void;

    /// Approximate output frame duration in milliseconds (1000 / outputFps), computed in
    /// Build() from framerate, interlaced flag, and upconvert decision. Returns 20 (50 fps)
    /// before Build() succeeds. Used by the A/V sync controller.
    [[nodiscard]] auto GetOutputFrameDurationMs() const noexcept -> int { return outputFrameDurationMs_; }

  private:
    std::unique_ptr<AVFilterGraph, FreeAVFilterGraph> filterGraph_;
    std::unique_ptr<AVFilterGraph, FreeAVFilterGraph>
        previousFilterGraph_;          ///< keep-alive: released after display maps its surfaces
    AVFilterContext *bufferSrcCtx_{};  ///< owned by filterGraph_; raw pointer valid only while filterGraph_ is live
    AVFilterContext *bufferSinkCtx_{}; ///< owned by filterGraph_; same lifetime constraint
    int outputFrameDurationMs_{20};    ///< 20 = 50 fps fallback; updated by Build()
};

#endif // VDR_VAAPIVIDEO_FILTER_H
