// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file filter.cpp
 * @brief VAAPI VPP filter chain implementation: graph construction, frame routing, and HDR helpers.
 */

#include "filter.h"

#include "common.h"
#include "config.h"

// C++ Standard Library
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <format>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// FFmpeg
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
extern "C" {
#include <libavcodec/codec_id.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/avutil.h>
#include <libavutil/buffer.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/mastering_display_metadata.h>
#include <libavutil/mathematics.h>
#include <libavutil/mem.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
#include <libavutil/rational.h>
}
#pragma GCC diagnostic pop

// VDR
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <vdr/remux.h>
#include <vdr/thread.h>
#include <vdr/tools.h>
#pragma GCC diagnostic pop

// ============================================================================
// === CONSTANTS ===
// ============================================================================

namespace {
constexpr uint64_t FILTER_RATE_MATCH_TOLERANCE_PPM =
    2000; ///< 0.2%: source and display rates closer than this count as matched, so no fps re-timing
          ///< filter is emitted. Sized to swallow the 1000-ppm NTSC pull-down gap (59.94 vs 60) --
          ///< the one case where an exact comparison would insert a filter that duplicates a frame
          ///< every ~1000 for no benefit. The residual drift is far below one frame per minute and
          ///< the A/V sync controller corrects it; it must stay well under the smallest genuine
          ///< cadence step (50 vs 60 Hz = 200000 ppm) so real mismatches still get the filter.
} // namespace

// ============================================================================
// === FRAME CLASSIFICATION HELPERS ===
// ============================================================================

auto ResolveSwPixFmt(const AVFrame *frame) noexcept -> AVPixelFormat {
    if (!frame) [[unlikely]] {
        return AV_PIX_FMT_NONE;
    }
    if (frame->format != AV_PIX_FMT_VAAPI) {
        return static_cast<AVPixelFormat>(frame->format);
    }
    if (!frame->hw_frames_ctx) {
        return AV_PIX_FMT_NONE;
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- FFmpeg ABI
    const auto *framesCtx = reinterpret_cast<const AVHWFramesContext *>(frame->hw_frames_ctx->data);
    return framesCtx->sw_format;
}

auto FrameBitDepthAtLeast(const AVFrame *frame, int minBits) noexcept -> bool {
    const AVPixelFormat swFmt = ResolveSwPixFmt(frame);
    if (swFmt == AV_PIX_FMT_NONE) [[unlikely]] {
        return false;
    }
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(swFmt);
    if (!desc || desc->nb_components == 0) [[unlikely]] {
        return false;
    }
    return desc->comp[0].depth >= minBits;
}

auto ClassifyStream(const AVFrame *frame) noexcept -> StreamHdrKind {
    if (!frame) [[unlikely]] {
        return StreamHdrKind::Sdr;
    }
    if (frame->color_primaries != AVCOL_PRI_BT2020) {
        return StreamHdrKind::Sdr;
    }
    if (!FrameBitDepthAtLeast(frame, 10)) {
        return StreamHdrKind::Sdr;
    }
    switch (frame->color_trc) {
        case AVCOL_TRC_SMPTE2084:
            return StreamHdrKind::Hdr10;
        case AVCOL_TRC_ARIB_STD_B67:
            return StreamHdrKind::Hlg;
        default:
            return StreamHdrKind::Sdr;
    }
}

auto ExtractHdrInfo(const AVFrame *frame) noexcept -> HdrStreamInfo {
    HdrStreamInfo info{};
    info.kind = ClassifyStream(frame);
    if (info.kind == StreamHdrKind::Sdr) {
        return info; // null-frame case already resolved to Sdr inside ClassifyStream
    }
    // Side-data size check guards against header/runtime ABI skew: FFmpeg's contract
    // promises sizeof(AVMastering...), but >= tolerates a larger future layout.
    if (const AVFrameSideData *sd = av_frame_get_side_data(frame, AV_FRAME_DATA_MASTERING_DISPLAY_METADATA);
        sd != nullptr && sd->size >= sizeof(AVMasteringDisplayMetadata)) {
        info.hasMasteringDisplay = true;
        std::memcpy(&info.mastering, sd->data, sizeof(info.mastering));
    }
    if (const AVFrameSideData *sd = av_frame_get_side_data(frame, AV_FRAME_DATA_CONTENT_LIGHT_LEVEL);
        sd != nullptr && sd->size >= sizeof(AVContentLightMetadata)) {
        info.hasContentLight = true;
        std::memcpy(&info.contentLight, sd->data, sizeof(info.contentLight));
    }
    return info;
}

// ============================================================================
// === BUILD ===
// ============================================================================

namespace {

/// Resolve the GPU deinterlacer for a requested VppDeintMode rank, choosing ONLY modes the driver
/// advertises in @p supportedMask -- emitting an unadvertised mode makes VAAPI reject the whole graph
/// (a GPU may expose just one mode). The VppDeintMode value is its rank (lower = better);
/// rank 0 (MCDI) means "best advertised". See config.h.
[[nodiscard]] auto ClampDeinterlaceMode(std::string_view gpuMode, int capRank, unsigned supportedMask)
    -> std::string_view {
    const int cap = std::clamp(capRank, 0, CONFIG_VPP_DEINT_MODE_COUNT - 1);
    if (cap == 0 || supportedMask == 0) {
        return gpuMode; // MCDI never limits; an empty mask means we can't safely pick a mode
    }
    const auto isSupported = [supportedMask](int rank) -> bool {
        return (supportedMask & (1U << static_cast<unsigned>(rank))) != 0;
    };
    for (int rank = cap; rank < CONFIG_VPP_DEINT_MODE_COUNT; ++rank) { // best supported mode no better than the cap
        if (isSupported(rank)) {
            return VppDeintModeArg(static_cast<VppDeintMode>(rank));
        }
    }
    for (int rank = CONFIG_VPP_DEINT_MODE_COUNT - 1; rank >= 0; --rank) { // cap unreachable: cheapest mode on offer
        if (isSupported(rank)) {
            return VppDeintModeArg(static_cast<VppDeintMode>(rank));
        }
    }
    return gpuMode;
}

/// Map a user Deinterlace HW selection to its VppDeintMode rank for ClampDeinterlaceMode. Auto resolves
/// to rank 0 (best advertised); Sw* never reach here.
[[nodiscard]] auto UserDeintRank(DeinterlaceMode mode) noexcept -> int {
    switch (mode) {
        case DeinterlaceMode::HwMotionAdaptive:
            return static_cast<int>(VppDeintMode::MotionAdaptive);
        case DeinterlaceMode::HwWeave:
            return static_cast<int>(VppDeintMode::Weave);
        case DeinterlaceMode::HwBob:
            return static_cast<int>(VppDeintMode::Bob);
        default: // Auto and (defensively) the Sw* values
            return static_cast<int>(VppDeintMode::MotionCompensated);
    }
}

// SW deinterlacers, field-rate (2x, matching deinterlace_vaapi=rate=field). deint=interlaced acts per
// frame on the picture's flag: MPEG-2 toggles progressive_frame per-GOP, so a fixed verdict would comb
// progressive frames or needlessly double-rate them. mode/parity stay at their field-rate defaults;
// w3fdif filter=simple (3-tap) is lighter than the default complex (9-tap) with little loss on broadcast.
inline constexpr const char *SW_DEINT_BWDIF = "bwdif=deint=interlaced";
inline constexpr const char *SW_DEINT_W3FDIF = "w3fdif=filter=simple:deint=interlaced";

// Software denoise (hqdn3d luma_spatial:chroma_spatial:luma_tmp:chroma_tmp) presets.
inline constexpr const char *SW_DENOISE_MINIMAL = "hqdn3d=1.0:1.0:2.0:2.0";
inline constexpr const char *SW_DENOISE_ENHANCED = "hqdn3d=2.0:1.5:4.0:3.5";

// Software sharpen (unsharp) presets.
inline constexpr const char *SW_SHARPEN_MILD = "unsharp=3:3:0.25:3:3:0.0";
inline constexpr const char *SW_SHARPEN_MEDIUM = "unsharp=5:5:0.5:5:5:0.0";

} // namespace

// ============================================================================
// === FILTER GRAPH TOKEN ===
// ============================================================================

auto FreeFilterGraphLocked::operator()(AVFilterGraph *graph) const noexcept -> void {
    // Why locked: teardown issues vaDestroyContext/vaDestroyConfig, and the last token can be
    // dropped on any thread (see the struct doc). Null mutex = graph never touched a display.
    if (vaDriverMutex == nullptr) [[unlikely]] {
        avfilter_graph_free(&graph);
        return;
    }
    const cMutexLock vaLock(vaDriverMutex);
    avfilter_graph_free(&graph);
}

// ============================================================================
// === BUILD ===
// ============================================================================

auto cVideoFilterChain::FailBuild() noexcept -> bool {
    bufferSrcCtx_ = nullptr;
    bufferSinkCtx_ = nullptr;
    hasFpsFilter_.store(false, std::memory_order_relaxed);
    outputFrameDurationMs_ = 20;
    naturalOutputRateMilliHz_.store(0, std::memory_order_relaxed);
    return false;
}

auto cVideoFilterChain::Build(AVFrame *firstFrame, const BuildParams &params) -> bool {
    if (graph_) {
        return true;
    }

    if (!firstFrame) [[unlikely]] {
        esyslog("vaapivideo/filter: no first frame for filter setup");
        return false;
    }
    if (!params.hwDeviceRef) [[unlikely]] {
        esyslog("vaapivideo/filter: no hw_device_ref for filter setup");
        return false;
    }
    if (params.outputWidth == 0 || params.outputHeight == 0) [[unlikely]] {
        esyslog("vaapivideo/filter: zero output dimensions");
        return false;
    }
    // Compact dsyslog only when explicitly requested by the caller (ScaleVideo-driven rebuild).
    // A Clear() / channel-switch rebuild still emits the full chain diagnostic.
    const bool compactLog = params.compactLog;

    const int srcWidth = firstFrame->width;
    const int srcHeight = firstFrame->height;
    // streamInterlaced overrides the unreliable per-frame flag (see FrameNeedsDeinterlace).
    const bool isInterlaced = FrameNeedsDeinterlace(firstFrame, params.streamInterlaced);
    const auto srcPixFmt = static_cast<AVPixelFormat>(firstFrame->format);

    // 1088, not 1080: MPEG-2/H.264 pad height to a 16-px macroblock boundary.
    const bool isUhd = (srcWidth > 1920 || srcHeight > 1088);
    const bool isSoftwareDecode = (srcPixFmt != AV_PIX_FMT_VAAPI);

    const uint32_t dstWidth = params.outputWidth;
    const uint32_t dstHeight = params.outputHeight;

    // VPP outputs DAR-fitted to the active video rect; KMS scanout stays 1:1 (no plane scaler).
    // Guard against streams that report SAR 0/0 (treated as square).
    const int sarNum = firstFrame->sample_aspect_ratio.num > 0 ? firstFrame->sample_aspect_ratio.num : 1;
    const int sarDen = firstFrame->sample_aspect_ratio.den > 0 ? firstFrame->sample_aspect_ratio.den : 1;

    // Manual zoom: symmetrically crop the source (cropH/cropV per side) before scaling so baked-in
    // black bars can be removed. Every dimension fed to crop/scale_vaapi must be even -- NV12/P010 are
    // 4:2:0, so an odd width/height/offset would split a 2x2-subsampled chroma sample. Offsets are
    // even-aligned and clamped so the kept region stays >= 2 px; the cropped dimensions are then
    // masked even too (a no-op for the always-even 4:2:0 source, but it removes the dependency on it).
    // SAR is unchanged by cropping, so the DAR below uses the cropped luma dimensions and the existing
    // letterbox/pillarbox fit fills the rect.
    // cropH/cropV arrive as per-side crop fractions (the decoder derives them from the zoom-in
    // factor). Clamp to a hard geometric safety: 0..0.499, just shy of the degenerate 0.5. The config
    // ceiling (+49.9% zoom-in) maps to only ~0.167, so this never alters a configured value.
    const double cropFracH = std::clamp(params.cropH, 0.0, 0.499);
    const double cropFracV = std::clamp(params.cropV, 0.0, 0.499);
    bool wantCrop = cropFracH > 0.0 || cropFracV > 0.0;
    auto croppedW = static_cast<uint32_t>(srcWidth);
    auto croppedH = static_cast<uint32_t>(srcHeight);
    uint32_t cropOffX = 0;
    uint32_t cropOffY = 0;
    if (wantCrop) {
        const auto sourceW = static_cast<uint32_t>(srcWidth);
        const auto sourceH = static_cast<uint32_t>(srcHeight);
        const uint32_t maxCropX = sourceW > 2U ? (((sourceW - 2U) / 2U) & ~1U) : 0U;
        const uint32_t maxCropY = sourceH > 2U ? (((sourceH - 2U) / 2U) & ~1U) : 0U;
        cropOffX = std::min(static_cast<uint32_t>(static_cast<double>(sourceW) * cropFracH) & ~1U, maxCropX);
        cropOffY = std::min(static_cast<uint32_t>(static_cast<double>(sourceH) * cropFracV) & ~1U, maxCropY);
        croppedW = std::max((sourceW - (2U * cropOffX)) & ~1U, 2U);
        croppedH = std::max((sourceH - (2U * cropOffY)) & ~1U, 2U);
        // Sub-px crop that rounded down to nothing: treat as no crop so we skip the round trip below.
        wantCrop = cropOffX > 0U || cropOffY > 0U;
    }

    // Source display aspect, divided by the scanout pixel aspect: the result is the shape the
    // picture must have in RASTER pixels to look right on the panel, so the fit below is the plain
    // square-pixel one. On an anamorphic CEA timing (720x576 flagged 16:9) the two differ by 42%,
    // and fitting without the divide letterboxes a 16:9 source into 720x405 that the TV stretches
    // back to full width -- ~30% short vertically with black bars. par is 1:1 on every other mode,
    // where this collapses to the source DAR exactly.
    const uint64_t parNum = params.outputParNum > 0 ? params.outputParNum : 1;
    const uint64_t parDen = params.outputParDen > 0 ? params.outputParDen : 1;
    const uint64_t fitNum = static_cast<uint64_t>(croppedW) * static_cast<uint64_t>(sarNum) * parDen;
    const uint64_t fitDen = static_cast<uint64_t>(croppedH) * static_cast<uint64_t>(sarDen) * parNum;

    uint32_t filterWidth = dstWidth;
    uint32_t filterHeight = dstHeight;

    // Integer cross-multiply avoids FP rounding: compare fitNum/fitDen vs dstWidth/dstHeight.
    if (fitNum * dstHeight > fitDen * static_cast<uint64_t>(dstWidth)) {
        filterWidth = dstWidth; // source wider -> letterbox
        filterHeight = static_cast<uint32_t>(static_cast<uint64_t>(dstWidth) * fitDen / fitNum);
    } else if (fitNum * dstHeight < fitDen * static_cast<uint64_t>(dstWidth)) {
        filterHeight = dstHeight; // source narrower -> pillarbox
        filterWidth = static_cast<uint32_t>(static_cast<uint64_t>(dstHeight) * fitNum / fitDen);
    }

    // NV12/P010 chroma is 4:2:0 (2x2-subsampled); odd dimensions produce artifacts.
    // Minimum 2 so scale_vaapi never receives a 0-size surface.
    const auto evenAtLeastTwo = [](uint32_t value) noexcept -> uint32_t { return std::max(value & ~1U, 2U); };
    filterWidth = evenAtLeastTwo(filterWidth);
    filterHeight = evenAtLeastTwo(filterHeight);

    // Snap to the exact rect when the fit lands within ~1%: integer crop-offset / aspect rounding
    // otherwise leaves a 2-6 px black sliver on an image that should fill the screen (most visibly a
    // zoomed 16:9 source -> 1920x1076 instead of 1920x1080). The implied <=1% stretch is invisible;
    // genuine letterbox/pillarbox (4:3, scope, ...) is far larger than 1% and stays untouched.
    // The <= dst guard makes the no-underflow precondition explicit (the min-2 clamp above can push a
    // dimension past a degenerate 1-px rect); snapping keeps the 4:2:0 evenness/minimum invariant.
    if (filterWidth <= dstWidth && dstWidth - filterWidth <= dstWidth / 100) {
        filterWidth = evenAtLeastTwo(dstWidth);
    }
    if (filterHeight <= dstHeight && dstHeight - filterHeight <= dstHeight / 100) {
        filterHeight = evenAtLeastTwo(dstHeight);
    }

    // Compare against the cropped dimensions: those are what scale_vaapi actually receives.
    const bool needsResize = (filterWidth != croppedW || filterHeight != croppedH);
    // Upscale specifically, not merely "resized": magnifying is the only case where the VPP itself
    // softens edges, and that softening is what the SD sharpen level below exists to undo.
    const bool upscales = (filterWidth > croppedW || filterHeight > croppedH);

    // UHD: GPU already saturated by 4K decode/scale; skip denoise/sharpen to avoid stutter.
    // MPEG-2 SD: DCT-block and analog-tape artifacts warrant heavier processing.
    // H.264/H.265 HD: lighter touch preserves encoder-intended detail.
    int denoiseLevel = 0;
    int sharpnessLevel = 0;

    if (!isUhd) {
        if (params.codecId == AV_CODEC_ID_MPEG2VIDEO) {
            // Denoise regardless of the scale factor: the blocking is in the source and gets
            // magnified either way -- by us when we upscale, by the TV's scaler when we hand it a
            // native 576p raster. Higher smears motion.
            denoiseLevel = 12;
            // Sharpen only when WE magnify. Resolution matching can now put an SD stream out 1:1 on
            // a 576p/480p mode, where the VPP softens nothing and the TV's own scaler applies its
            // own edge enhancement downstream -- pre-sharpening would stack on top of that and
            // halate titles. 26 restores upscale-softened edges without ringing.
            sharpnessLevel = upscales ? 26 : 0;
        } else {
            denoiseLevel = 4;    // 1080i is near-native; just tame ringing at bitrate-starved edges
            sharpnessLevel = 20; // mild -- stronger values halate bright HD content
        }
    }

    // HDR path: P010 (10-bit packed) preserves bit depth through VPP; NV12 would clip to 8-bit.
    // scaleColorArgs is appended after ':' (resize) or after '=' (no-resize); no leading separator.
    const char *pixFmt = params.hdrPassthrough ? "p010le" : "nv12";
    std::string scaleColorArgs;
    if (params.hdrPassthrough) {
        // Pin colorimetry explicitly: scale_vaapi's "preserve input" default is driver-dependent
        // and has been seen to silently downgrade HDR output to BT.709.
        const char *transfer = (params.hdrInfo.kind == StreamHdrKind::Hlg) ? "arib-std-b67" : "smpte2084";
        scaleColorArgs = std::format(
            "format={}:out_color_matrix=bt2020nc:out_color_primaries=bt2020:out_color_transfer={}:out_range=tv", pixFmt,
            transfer);
    } else {
        scaleColorArgs = std::format("format={}:out_color_matrix=bt709:out_range=tv", pixFmt);
    }

    // Trick/still drop denoise/sharpness (minimal chain). Deinterlacing differs (see the GPU VPP
    // block): trick keeps a spatial deinterlacer (1x, frame rate) so interlaced FF/RW/slow doesn't
    // comb; still drops it -- a lone frame has no field-pair partner and the temporal VPP delay
    // swallows it.
    const bool minimalChain = params.trickMode || params.stillPicture;
    const bool useSpatialDeinterlace = params.trickMode;

    // SW post-processing block decision. Any effective sw-* post-process choice pulls the needed prefix
    // into one hwdownload..hwupload block. Forced off for HDR (no P010/BT.2020 in the SW chain), UHD
    // (SW filters stutter at 4K), and trick/still (minimal chain).
    const bool swDeintRequested =
        params.userDeint == DeinterlaceMode::SwBwdif || params.userDeint == DeinterlaceMode::SwW3fdif;
    const bool swDenoiseRequested =
        params.denoise == DenoiseMode::SwMinimal || params.denoise == DenoiseMode::SwEnhanced;
    const bool swSharpenRequested = params.sharpen == SharpenMode::SwMild || params.sharpen == SharpenMode::SwMedium;
    const bool swScaleRequested = params.scale == ScaleMode::SwQuality || params.scale == ScaleMode::SwFast;
    const bool useSwPost =
        !minimalChain && !params.hdrPassthrough && !isUhd &&
        ((swDeintRequested && isInterlaced) || swDenoiseRequested || swSharpenRequested || swScaleRequested);

    // VBR DVB streams (and some cable muxes) omit framerate; 50/1 is the DVB-S/T baseline (= 25i).
    // Note the fallback is already a FIELD rate, so for an interlaced stream the fieldRateFactor
    // below doubles it again -- harmless for the fps-filter decision (which then always inserts
    // fps=display and pins the real output rate) but NOT a number the display-mode matcher may
    // act on. fpsDeclared gates that; see naturalOutputRateMilliHz_ below.
    const bool fpsDeclared = params.fpsNum > 0;
    const int fpsNum = fpsDeclared ? params.fpsNum : 50;
    const int fpsDen = params.fpsDen > 0 ? params.fpsDen : 1;

    // rate=field doubles interlaced pairs (25i -> 50p); auto=1/deint=interlaced pass progressive frames
    // 1:1, so this is the output-rate upper bound. The doubling only applies when a field-rate (2x)
    // deinterlacer is actually emitted: trick/still use a spatial 1x one (or none at all), and the
    // HW-decode GPU path skips the deinterlacer entirely when the driver advertises no mode. Round
    // naturalOutputFps so NTSC rates (30000/1001, 60000/1001) resolve to 30/60 rather than truncating
    // to 29/59.
    const bool hasHwDeinterlace = !params.deinterlaceMode.empty() && params.deinterlaceModeMask != 0U;
    const bool fieldRateDeint = isInterlaced && !minimalChain && (useSwPost || isSoftwareDecode || hasHwDeinterlace);
    const int fieldRateFactor = fieldRateDeint ? 2 : 1;
    const int64_t outputRateNum = static_cast<int64_t>(fpsNum) * fieldRateFactor;
    const int64_t outputRateDen = std::max<int64_t>(fpsDen, 1);
    const int naturalOutputFps = static_cast<int>((outputRateNum + (outputRateDen / 2)) / outputRateDen);
    // Exact pre-fps-filter output rate in millihertz, for both the rate comparison below and the
    // stream format the decoder publishes to the display-mode matcher.
    const auto naturalOutputMilliHz =
        static_cast<uint32_t>(((outputRateNum * 1000) + (outputRateDen / 2)) / outputRateDen);
    const uint32_t displayMilliHz = params.outputRefreshMilliHz;
    const int displayFps = static_cast<int>((displayMilliHz + 500U) / 1000U);
    // Insert fps=display whenever the post-deinterlace output rate differs from the display rate.
    // The fps filter paces the decoder at source rate by buffering its output to the target rate;
    // without it, the decoder thread is paced by SubmitFrame's vsync backpressure (= display rate)
    // and source-rate consumption drifts off real time:
    //   60 fps -> 50 Hz, no fps filter: decoder pulls 50/sec, source advances 50/60 = 83% (slow)
    //   24 fps -> 50 Hz, no fps filter: decoder pulls 50/sec, source advances 50/24 = 208% (fast)
    // With audio anchored the due-gate corrects this (catch-up drops / re-presents), but video-only
    // playback (HDR demo files etc.) depends entirely on the fps filter for correct pacing. The
    // filter nearest-neighbor (drops for source>display, duplicates for source<display) -- same
    // visual cadence the display would produce anyway -- but the producer-side pacing is what makes
    // the decoder consume source at its actual rate. Adding it on audio-clocked paths is safe (and
    // eliminates "catch-up cycling sustained" log spam from the routine source>display drop work).
    //
    // Compared with a tolerance rather than for equality: with display-mode matching enabled the
    // output can sit on a genuine 59.94 Hz (or 23.976 Hz) mode, and an exact test would call that a
    // mismatch against a 59.94 fps source and insert a pointless fps=60 that duplicates a frame
    // every ~1000. Within the tolerance the residual drift is well under a frame per minute and the
    // A/V sync controller absorbs it -- strictly better than resampling the cadence.
    const uint64_t rateDeltaMilliHz = naturalOutputMilliHz > displayMilliHz
                                          ? naturalOutputMilliHz - displayMilliHz
                                          : static_cast<uint64_t>(displayMilliHz) - naturalOutputMilliHz;
    const bool ratesDiffer =
        naturalOutputMilliHz > 0 && displayMilliHz > 0 &&
        rateDeltaMilliHz * 1000000ULL > static_cast<uint64_t>(displayMilliHz) * FILTER_RATE_MATCH_TOLERANCE_PPM;
    // Trick/still never get the fps filter (minimal chain), so their output stays at the natural rate.
    const bool insertFpsFilter = !minimalChain && ratesDiffer;
    const int outputFps = insertFpsFilter ? displayFps : naturalOutputFps;

    // Filter chain (comma-joined, two domains chosen by useSwPost below):
    //   GPU VPP: [deinterlace_vaapi|bwdif(SW-decode)] -> [denoise_vaapi] -> [crop] -> scale_vaapi
    //            -> [sharpness_vaapi] -> [fps]
    //   Hybrid:  hwdownload -> [bwdif|w3fdif] -> [SW-only denoise/scale/sharpen] -> format=nv12,hwupload
    //            -> [denoise_vaapi] -> [crop] -> [scale_vaapi] -> [sharpness_vaapi] -> [fps]
    // The hybrid keeps exactly ONE hwdownload and ONE hwupload but runs only the software-mandatory
    // filters (the SW deinterlacer + any explicit sw-* node) in system memory; everything else stays on
    // the GPU after the upload. It is taken only when the user picks a sw-* deint/scale/sharpen and the
    // content is SDR, non-UHD, non-trick/still (see useSwPost). [crop] is the manual-zoom node; the
    // final surface is always VAAPI for KMS scanout.
    std::vector<std::string> filters;
    // Local glue: append the manual-zoom crop node (even-aligned dims/offsets computed above). Capturing
    // the crop geometry lets the call sites below read as a plain `appendCropFilter()`.
    const auto appendCropFilter = [&filters, croppedW, croppedH, cropOffX, cropOffY]() -> void {
        filters.push_back(std::format("crop={}:{}:{}:{}", croppedW, croppedH, cropOffX, cropOffY));
    };

    // Effective GPU VPP levels for the active policy (0 = skip). Only Auto applies HW denoise/sharpen;
    // Off skips and the Sw* values never reach the GPU domain (they force the SW block). UHD leaves the
    // base at 0.
    const int gpuDenoiseLevel = (params.denoise == DenoiseMode::Auto) ? denoiseLevel : 0;
    const int gpuSharpenLevel = (params.sharpen == SharpenMode::Auto) ? sharpnessLevel : 0;

    if (useSwPost) {
        // --- Hybrid SW/HW post-processing: ONE hwdownload, ONE hwupload ---
        // Only the filters that MUST run in software go in the SW segment (the deinterlacer, plus any
        // explicit sw-* denoise/scale/sharpen); everything else stays on the GPU after the single
        // upload (denoise_vaapi / scale_vaapi / sharpness_vaapi). The expensive scale in particular is
        // kept on the GPU unless the user explicitly asked for software scaling -- the previous all-SW
        // block needlessly ran swscale (and HW-capable denoise/sharpen) on the CPU.
        //
        // The SW segment is the pipeline prefix deinterlace->denoise->scale->sharpen up to the LAST
        // sw-* stage (one upload => no going back to the GPU and returning). Pipeline order forces the
        // boundary: sw sharpen (post-scale) pulls scale into SW; sw scale pulls denoise's slot into SW.
        // An "auto" denoise/sharpen that lands inside the SW segment is dropped (auto = HW-only); scale
        // inside the segment runs as swscale because it is mandatory.
        const bool scaleInSw = swScaleRequested || swSharpenRequested; // sharpen follows the scale slot; keep boundary
        const bool denoiseInSw = swDenoiseRequested || scaleInSw;      // denoise precedes scale; keep SW contiguous

        // -- single hwdownload (HW decode only; an already-SW-decoded frame is in system memory) --
        if (!isSoftwareDecode) {
            filters.emplace_back("hwdownload,format=nv12");
        }
        // Deinterlace (software). bwdif unless the user explicitly picked w3fdif; a HW-leaning
        // selection forced into the SW domain by a sw scale/sharpen lands on bwdif.
        if (isInterlaced) {
            filters.emplace_back(params.userDeint == DeinterlaceMode::SwW3fdif ? SW_DEINT_W3FDIF : SW_DEINT_BWDIF);
        }
        // Denoise (software hqdn3d) -- only the explicit sw-* presets.
        if (swDenoiseRequested) {
            filters.emplace_back(params.denoise == DenoiseMode::SwEnhanced ? SW_DENOISE_ENHANCED : SW_DENOISE_MINIMAL);
        }
        // Scale (software swscale) -- only when it is pulled into the SW segment. Emitted only when it
        // does real work: a resize, a manual-zoom crop, or BT.601->BT.709 / range normalization.
        if (scaleInSw) {
            if (wantCrop) {
                appendCropFilter();
            }
            const bool swNeedsScale = needsResize || wantCrop || firstFrame->colorspace != AVCOL_SPC_BT709 ||
                                      firstFrame->color_range == AVCOL_RANGE_JPEG;
            if (swNeedsScale) {
                // Fast (bilinear) for SwFast / HwFast; HQ (lanczos) otherwise. HwFast can only appear
                // here when a sw sharpen pulled scale into the SW segment -- honor the "fast" intent.
                const bool fastScale = params.scale == ScaleMode::SwFast || params.scale == ScaleMode::HwFast;
                const char *swScaleFlags = fastScale ? "bilinear" : "lanczos+full_chroma_int+accurate_rnd";
                filters.push_back(
                    std::format("scale=w={}:h={}:flags={}:in_color_matrix=auto:out_color_matrix=bt709:out_range=tv",
                                filterWidth, filterHeight, swScaleFlags));
            }
        }
        // Sharpen (software unsharp) -- only the explicit sw-* presets.
        if (swSharpenRequested) {
            filters.emplace_back(params.sharpen == SharpenMode::SwMedium ? SW_SHARPEN_MEDIUM : SW_SHARPEN_MILD);
        }
        // -- single hwupload: back onto a VAAPI surface --
        filters.emplace_back("format=nv12,hwupload");

        // -- HW VPP tail: everything not done in software, on the GPU --
        if (!denoiseInSw && gpuDenoiseLevel > 0 && params.hasDenoise) {
            filters.push_back(std::format("denoise_vaapi=denoise={}", gpuDenoiseLevel));
        }
        if (!scaleInSw) {
            // Crop is HW here (scale_vaapi consumes the crop_* metadata in the same pass).
            if (wantCrop) {
                appendCropFilter();
            }
            const bool needsExplicitScaleSize = needsResize || wantCrop;
            if (needsExplicitScaleSize) {
                const char *scaleHq = (isUhd || params.scale == ScaleMode::HwFast) ? "" : ":mode=hq";
                filters.push_back(
                    std::format("scale_vaapi=w={}:h={}{}:{}", filterWidth, filterHeight, scaleHq, scaleColorArgs));
            } else {
                filters.push_back(std::format("scale_vaapi={}", scaleColorArgs));
            }
        }
        if (!swSharpenRequested && gpuSharpenLevel > 0 && params.hasSharpness) {
            filters.push_back(std::format("sharpness_vaapi=sharpness={}", gpuSharpenLevel));
        }
    } else {
        // --- GPU VPP domain (zero-copy where possible) ---
        // Still drops the deinterlacer (lone frame, no field pair -> swallowed). Trick keeps a spatial
        // one (bob/yadif, 1x): no temporal buffering, and it stops interlaced FF/RW/slow from combing.
        if (isInterlaced && !params.stillPicture) {
            if (isSoftwareDecode) {
                // SW-decoded frame in system memory: VAAPI can't deinterlace it. Normal play: bwdif
                // (temporal, 2x). Trick: yadif spatial 1x, but only on frames marked interlaced.
                filters.emplace_back(useSpatialDeinterlace ? "yadif=deint=interlaced" : SW_DEINT_BWDIF);
            } else if (useSpatialDeinterlace && hasHwDeinterlace) {
                // Trick: bob spatial deinterlace at frame rate (1x); auto=1 leaves progressive frames
                // untouched inside mixed streams. Clamp to an advertised mode -- a GPU may expose only
                // one, and an unadvertised bob would fail the whole graph (no video in trick mode).
                // The temporal fallback costs 1-2 frames of latency, which trick's continuous frame
                // flow absorbs; rate=frame keeps the 1x cadence either way.
                const std::string_view deintMode =
                    ClampDeinterlaceMode(VppDeintModeArg(VppDeintMode::Bob), static_cast<int>(VppDeintMode::Bob),
                                         params.deinterlaceModeMask);
                filters.push_back(std::format("deinterlace_vaapi=mode={}:rate=frame:auto=1", deintMode));
            } else if (!useSpatialDeinterlace && hasHwDeinterlace) {
                // User HW selection clamped to a driver-advertised mode (Auto = best advertised).
                // auto=1 passes progressive frames through, so per-GOP progressive_frame toggles need no rebuild.
                const std::string_view deintMode = ClampDeinterlaceMode(
                    params.deinterlaceMode, UserDeintRank(params.userDeint), params.deinterlaceModeMask);
                filters.push_back(std::format("deinterlace_vaapi=mode={}:rate=field:auto=1", deintMode));
            }
        }

        const bool wantDenoise = !minimalChain && gpuDenoiseLevel > 0;
        if (isSoftwareDecode) {
            // SW-decode sysmem tail. MPEG-2 hqdn3d fall-back only when the GPU lacks denoise_vaapi:
            // SW decode is cheap and block-artifact removal is worth the per-frame CPU cost (~5 ms
            // @ 1080p25). Crop in system memory before uploading so zoom adds no GPU readback.
            if (wantDenoise && !params.hasDenoise && params.codecId == AV_CODEC_ID_MPEG2VIDEO) {
                filters.emplace_back("hqdn3d=5");
            }
            filters.push_back(std::format("format={}", pixFmt));
            if (wantCrop) {
                appendCropFilter();
            }
            filters.emplace_back("hwupload");
        }

        // HW denoise (native for HW decode / post-hwupload for SW). Skipped on GPUs without it.
        if (wantDenoise && params.hasDenoise) {
            filters.push_back(std::format("denoise_vaapi=denoise={}", gpuDenoiseLevel));
        }

        // HW decode: crop only stores AVFrame::crop_* metadata on a VAAPI surface. The scale_vaapi below
        // consumes it as the VPP surface_region in the same GPU pass -- as long as that pass actually runs
        // VPP and isn't elided to a passthrough. The format/range options in scaleColorArgs keep it on the
        // VPP path (verified even when the conversion is a runtime no-op), and needsExplicitScaleSize keeps
        // explicit w/h; a bare scale_vaapi (no options, output == surface size) would drop the crop, so we
        // never emit one while cropping. SW decode already cropped in system memory before hwupload.
        if (wantCrop && !isSoftwareDecode) {
            appendCropFilter();
        }

        // Hardware crop does not change the input link dimensions, so force an explicit-size scale even
        // when the kept region already matches the target (keeps VPP doing real work that reads the crop).
        const bool needsExplicitScaleSize = needsResize || (wantCrop && !isSoftwareDecode);
        if (needsExplicitScaleSize) {
            // bicubic (hq) too expensive at 4K; ScaleMode::HwFast drops it on non-UHD too.
            const char *scaleHq = (isUhd || params.scale == ScaleMode::HwFast) ? "" : ":mode=hq";
            filters.push_back(
                std::format("scale_vaapi=w={}:h={}{}:{}", filterWidth, filterHeight, scaleHq, scaleColorArgs));
        } else {
            filters.push_back(std::format("scale_vaapi={}", scaleColorArgs));
        }

        if (!minimalChain && gpuSharpenLevel > 0 && params.hasSharpness) {
            filters.push_back(std::format("sharpness_vaapi=sharpness={}", gpuSharpenLevel));
        }
    }

    if (insertFpsFilter) {
        // Nearest-neighbor sample/duplicate to the display rate. Drops for source>display, dupes
        // for source<display (exact 2x for 25->50/24->48, or uneven cadence for inexact ratios).
        // No pixel work on the duplicated frame.
        filters.push_back(std::format("fps={}", displayFps));
    }

    std::string filterChain;
    for (const auto &filter : filters) {
        if (!filterChain.empty()) {
            filterChain += ',';
        }
        filterChain += filter;
    }

    // Build()-local until fully configured: every failure path frees it on return, and graph_
    // stays null so IsBuilt() never sees a half-built chain.
    std::unique_ptr<AVFilterGraph, FreeAVFilterGraph> graph{avfilter_graph_alloc()};
    if (!graph) [[unlikely]] {
        esyslog("vaapivideo/filter: failed to allocate filter graph");
        return false;
    }
    // Note: the SW filters (bwdif/w3fdif/hqdn3d/swscale/unsharp) slice-thread across cores already --
    // avfilter_graph_alloc() defaults nb_threads=0 (auto = all cores) and thread_type=allow-all. We
    // deliberately do NOT override either: forcing a count only ever caps parallelism below the auto.

    // Use numeric pix_fmt: symbolic aliases for HW formats differ across FFmpeg versions.
    // time_base = 1/PTSTICKS matches the 90 kHz domain that stream.cpp rescales every packet into.
    // colorspace/range only when tagged: an "unknown" buffersrc trips FFmpeg's "changing properties on
    // the fly" warning on the first tagged frame and leaves scale_vaapi no input matrix. Unspecified is
    // already the default, so omit it (matches the device.cpp software-scaler).
    std::string bufferSrcArgs =
        std::format("video_size={}x{}:pix_fmt={}:time_base=1/{}:pixel_aspect={}/{}:frame_rate={}/{}", srcWidth,
                    srcHeight, static_cast<int>(srcPixFmt), PTSTICKS, sarNum, sarDen, fpsNum, fpsDen);
    if (firstFrame->colorspace != AVCOL_SPC_UNSPECIFIED) {
        bufferSrcArgs += std::format(":colorspace={}", static_cast<int>(firstFrame->colorspace));
    }
    if (firstFrame->color_range != AVCOL_RANGE_UNSPECIFIED) {
        bufferSrcArgs += std::format(":range={}", static_cast<int>(firstFrame->color_range));
    }

    // compactLog is the sole gate here -- no dedup on this line, or a same-format channel switch
    // (always non-compact) would never surface its settings. Compact rebuilds are deduped at the
    // chain line below instead.
    if (!compactLog) {
        dsyslog("vaapivideo/filter: buffer source args='%s'", bufferSrcArgs.c_str());
    }

    // hw_frames_ctx must be attached to the buffer source before avfilter_init_str();
    // FFmpeg 7.x rejects initialization of a HW-format source without it.
    bufferSrcCtx_ = avfilter_graph_alloc_filter(graph.get(), avfilter_get_by_name("buffer"), "in");
    if (!bufferSrcCtx_) [[unlikely]] {
        esyslog("vaapivideo/filter: failed to allocate buffer source filter");
        return FailBuild();
    }

    if (!isSoftwareDecode) {
        if (!params.hwFramesCtx) [[unlikely]] {
            esyslog("vaapivideo/filter: hw decode requires hw_frames_ctx");
            return FailBuild();
        }
        AVBufferSrcParameters *hwFramesParams = av_buffersrc_parameters_alloc();
        if (!hwFramesParams) [[unlikely]] {
            esyslog("vaapivideo/filter: failed to allocate buffer source parameters");
            return FailBuild();
        }

        hwFramesParams->hw_frames_ctx = av_buffer_ref(params.hwFramesCtx);
        if (!hwFramesParams->hw_frames_ctx) [[unlikely]] {
            esyslog("vaapivideo/filter: av_buffer_ref(hw_frames_ctx) failed");
            av_free(hwFramesParams);
            return FailBuild();
        }
        const int setRet = av_buffersrc_parameters_set(bufferSrcCtx_, hwFramesParams);
        // av_buffersrc_parameters_set makes its own internal ref; caller must unref unconditionally
        // (both success and failure) to avoid leaking the AVHWFramesContext ref on every Build().
        av_buffer_unref(&hwFramesParams->hw_frames_ctx);
        av_free(hwFramesParams);
        if (setRet < 0) [[unlikely]] {
            esyslog("vaapivideo/filter: av_buffersrc_parameters_set failed: %s", AvErr(setRet).data());
            return FailBuild();
        }
    }

    int ret = avfilter_init_str(bufferSrcCtx_, bufferSrcArgs.c_str());
    if (ret < 0) [[unlikely]] {
        esyslog("vaapivideo/filter: failed to init buffer source '%s': %s", bufferSrcArgs.c_str(), AvErr(ret).data());
        return FailBuild();
    }

    ret = avfilter_graph_create_filter(&bufferSinkCtx_, avfilter_get_by_name("buffersink"), "out", nullptr, nullptr,
                                       graph.get());
    if (ret < 0) [[unlikely]] {
        esyslog("vaapivideo/filter: failed to create buffer sink: %s", AvErr(ret).data());
        return FailBuild();
    }

    // Segment API (not parse_ptr) because FFmpeg 8.x hwupload_init() rejects a filter with
    // no hw_device_ctx and parse_ptr inits filters as part of parsing -- too early to attach
    // the device. Segment splits parse / create / init so we can set hw_device_ctx between
    // create and init. (HW decode doesn't strictly need hw_device_ctx on the VAAPI filters --
    // they pick it up from hw_frames_ctx via the link -- but setting it is harmless.)
    AVFilterGraphSegment *segment = nullptr;
    ret = avfilter_graph_segment_parse(graph.get(), filterChain.c_str(), 0, &segment);
    if (ret < 0) [[unlikely]] {
        esyslog("vaapivideo/filter: failed to parse filter chain '%s': %s", filterChain.c_str(), AvErr(ret).data());
        return FailBuild();
    }

    ret = avfilter_graph_segment_create_filters(segment, 0);
    if (ret < 0) [[unlikely]] {
        esyslog("vaapivideo/filter: failed to create segment filters '%s': %s", filterChain.c_str(), AvErr(ret).data());
        avfilter_graph_segment_free(&segment);
        return FailBuild();
    }

    // Attach hw_device_ctx to every newly created filter (skip the externally allocated
    // buffer source/sink, which were already initialized above). Without this, hwupload's
    // init returns EINVAL and scale_vaapi/sharpness_vaapi fail at graph config time.
    for (unsigned int i = 0; i < graph->nb_filters; ++i) {
        AVFilterContext *filterCtx = graph->filters[i];
        if (filterCtx == bufferSrcCtx_ || filterCtx == bufferSinkCtx_) {
            continue;
        }
        if (filterCtx->hw_device_ctx) {
            continue;
        }
        filterCtx->hw_device_ctx = av_buffer_ref(params.hwDeviceRef);
        if (!filterCtx->hw_device_ctx) [[unlikely]] {
            esyslog("vaapivideo/filter: av_buffer_ref(hwDeviceRef) failed for '%s'", filterCtx->name);
            avfilter_graph_segment_free(&segment);
            return FailBuild();
        }
    }

    AVFilterInOut *segmentInputs = nullptr;
    AVFilterInOut *segmentOutputs = nullptr;
    ret = avfilter_graph_segment_apply(segment, 0, &segmentInputs, &segmentOutputs);
    if (ret < 0) [[unlikely]] {
        esyslog("vaapivideo/filter: failed to apply segment '%s': %s", filterChain.c_str(), AvErr(ret).data());
        avfilter_inout_free(&segmentInputs);
        avfilter_inout_free(&segmentOutputs);
        avfilter_graph_segment_free(&segment);
        return FailBuild();
    }

    // segmentInputs  = unlinked input pads of the chain head  -> wire buffersrc into them.
    // segmentOutputs = unlinked output pads of the chain tail -> wire them into buffersink.
    if (!segmentInputs || !segmentOutputs) [[unlikely]] {
        esyslog("vaapivideo/filter: segment has no free in/out pads (chain='%s')", filterChain.c_str());
        avfilter_inout_free(&segmentInputs);
        avfilter_inout_free(&segmentOutputs);
        avfilter_graph_segment_free(&segment);
        return FailBuild();
    }

    // avfilter_link takes pad indices as unsigned; AVFilterInOut stores them as int. Cast
    // explicitly to keep -Wsign-conversion happy; libavfilter only ever emits non-negative
    // pad indices here so the cast is safe.
    ret = avfilter_link(bufferSrcCtx_, 0, segmentInputs->filter_ctx, static_cast<unsigned>(segmentInputs->pad_idx));
    if (ret >= 0) {
        ret = avfilter_link(segmentOutputs->filter_ctx, static_cast<unsigned>(segmentOutputs->pad_idx), bufferSinkCtx_,
                            0);
    }
    avfilter_inout_free(&segmentInputs);
    avfilter_inout_free(&segmentOutputs);
    avfilter_graph_segment_free(&segment);
    if (ret < 0) [[unlikely]] {
        esyslog("vaapivideo/filter: failed to link buffersrc/buffersink to chain '%s': %s", filterChain.c_str(),
                AvErr(ret).data());
        return FailBuild();
    }

    ret = avfilter_graph_config(graph.get(), nullptr);
    if (ret < 0) [[unlikely]] {
        esyslog("vaapivideo/filter: failed to configure filter graph '%s': %s", filterChain.c_str(), AvErr(ret).data());
        return FailBuild();
    }

    // Commit derived state only after the graph is fully configured: a failed Build() must not leave
    // hasFpsFilter_ describing a chain that doesn't exist (FlushForSeek consults it for the rebuild
    // decision).
    hasFpsFilter_.store(insertFpsFilter, std::memory_order_relaxed);
    outputFrameDurationMs_ = outputFps > 0 ? std::max(1, 1000 / outputFps) : 20; // 20 ms = 50 fps fallback
    // Published only when the stream actually declared a frame rate. A guess must never reach the
    // display-mode matcher: the 50/1 fallback is a field rate, so an interlaced stream without VUI
    // timing would report 100 Hz and could drive the CRTC to a 100 Hz mode -- above the operator's
    // refresh cap, since the matcher deliberately exempts the k==1 multiple from it. Zero makes
    // the decoder skip the notification and StreamModeRequest::IsValid() keep the matcher inert.
    naturalOutputRateMilliHz_.store(fpsDeclared ? naturalOutputMilliHz : 0, std::memory_order_relaxed);

    // Publish: from here the graph is owned jointly by the chain and by every frame it produces
    // (via FilterGraphToken), so it survives Reset() for as long as any of them is in flight.
    graph_ = std::shared_ptr<AVFilterGraph>(graph.release(), FreeFilterGraphLocked{params.vaDriverMutex});

    if (!compactLog) {
        // Non-compact => Clear / channel switch (trick/zoom/seek rebuilds set compactLog). Log it
        // even when the chain is byte-for-byte identical, so every switch surfaces its settings.
        const char *cadenceTag = "";
        if (insertFpsFilter) {
            if (naturalOutputMilliHz < displayMilliHz) {
                // "duplicated" only when the display rate is a whole multiple of the source rate
                // (every frame shown the same number of times); anything else beats out a 3:2-style
                // uneven pattern. Compared with the same tolerance as the match test above so a
                // 23.976-into-59.94 pull-down is not mislabeled over a rounding remainder.
                const uint64_t multiple =
                    (static_cast<uint64_t>(displayMilliHz) + (naturalOutputMilliHz / 2)) / naturalOutputMilliHz;
                const uint64_t ideal = multiple * naturalOutputMilliHz;
                const uint64_t deviation = ideal > displayMilliHz ? ideal - displayMilliHz : displayMilliHz - ideal;
                const bool wholeMultiple =
                    multiple >= 1 &&
                    deviation * 1000000ULL <= static_cast<uint64_t>(displayMilliHz) * FILTER_RATE_MATCH_TOLERANCE_PPM;
                cadenceTag = wholeMultiple ? ", duplicated" : ", uneven cadence";
            } else {
                cadenceTag = ", decimated";
            }
        }
        // "deinterlaced" reflects the chain, not the source: still drops the deinterlacer, and a
        // HW-decode path without an advertised VPP mode has none to emit.
        const bool chainDeinterlaces =
            isInterlaced && !params.stillPicture && (useSwPost || isSoftwareDecode || hasHwDeinterlace);
        isyslog("vaapivideo/filter: VAAPI filter initialized (%dx%d -> %ux%u%s%s, out=%s %s)", srcWidth, srcHeight,
                filterWidth, filterHeight, chainDeinterlaces ? ", deinterlaced" : "", cadenceTag, pixFmt,
                params.hdrPassthrough ? StreamHdrKindName(params.hdrInfo.kind) : "SDR");
    }
    // On compact rebuilds this is the ONLY line, and a byte-identical repeat is skipped outright:
    // trick reverse rebuilds the same graph once per keyframe step. Non-compact builds (Clear /
    // channel switch) always log, so a same-format switch still surfaces its settings. The output
    // size is appended because a no-resize/no-crop chain emits a bare "scale_vaapi=..." element
    // with no w=/h= args.
    std::string chainLogKey = std::format("{}|{}|{}x{}", bufferSrcArgs, filterChain, filterWidth, filterHeight);
    if (!compactLog || chainLogKey != lastChainLogKey_) {
        dsyslog("vaapivideo/filter: filter chain='%s' (out=%ux%u)", filterChain.c_str(), filterWidth, filterHeight);
    }
    lastChainLogKey_ = std::move(chainLogKey);

    return true;
}

// ============================================================================
// === SEND / RECEIVE / RESET ===
// ============================================================================

auto cVideoFilterChain::SendFrame(AVFrame *frame) noexcept -> int {
    if (!graph_ || !bufferSrcCtx_) [[unlikely]] {
        return AVERROR(EINVAL);
    }
    // KEEP_REF: the filter graph makes its own ref; the decoder retains ownership of the
    // underlying AVFrame data. For a null frame (EOS flush signal) pass 0 explicitly --
    // flags are undefined on the EOS path in FFmpeg's internal buffersrc code.
    const int flags = (frame != nullptr) ? AV_BUFFERSRC_FLAG_KEEP_REF : 0;
    return av_buffersrc_add_frame_flags(bufferSrcCtx_, frame, flags);
}

auto cVideoFilterChain::ReceiveFrame(AVFrame *out) noexcept -> int {
    if (!graph_ || !bufferSinkCtx_) [[unlikely]] {
        return AVERROR(EINVAL);
    }
    const int ret = av_buffersink_get_frame(bufferSinkCtx_, out);
    if (ret < 0) {
        return ret;
    }
    // The sink's time base differs from the source's 1/90k (halved by a field-rate deinterlacer, 1/rate
    // after fps=). Rescale rather than relabel: only a temporal filter knows which frame it emitted.
    const AVRational sinkTimeBase = av_buffersink_get_time_base(bufferSinkCtx_);
    constexpr AVRational kPtsTimeBase{.num = 1, .den = PTSTICKS};
    if (out->pts != AV_NOPTS_VALUE) {
        out->pts = av_rescale_q(out->pts, sinkTimeBase, kPtsTimeBase);
    }
    if (out->duration > 0) {
        out->duration = av_rescale_q(out->duration, sinkTimeBase, kPtsTimeBase);
    }
    out->time_base = kPtsTimeBase;
    return ret;
}

auto cVideoFilterChain::Reset() noexcept -> void {
    bufferSrcCtx_ = nullptr;
    bufferSinkCtx_ = nullptr;
    hasFpsFilter_.store(false, std::memory_order_relaxed);
    outputFrameDurationMs_ = 20;
    naturalOutputRateMilliHz_.store(0, std::memory_order_relaxed);
    // Drop only the chain's own reference: in-flight frames hold FilterGraphTokens, and destroying
    // the graph under them is a driver use-after-free (see FilterGraphToken). With nothing in
    // flight this IS the last reference and frees the graph right here -- the recursive
    // vaDriverMutex re-entry is fine since every Reset() call site already holds it.
    graph_.reset();
}
