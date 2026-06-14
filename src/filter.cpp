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
#include <array>
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
#include <libavutil/buffer.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/mastering_display_metadata.h>
#include <libavutil/mem.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
}
#pragma GCC diagnostic pop

// VDR
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <vdr/remux.h>
#include <vdr/tools.h>
#pragma GCC diagnostic pop

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

/// Stack-allocated FFmpeg error string; avoids heap allocation on every error path.
[[nodiscard]] auto FmtErr(int ret) noexcept -> std::array<char, AV_ERROR_MAX_STRING_SIZE> {
    std::array<char, AV_ERROR_MAX_STRING_SIZE> buf{};
    av_make_error_string(buf.data(), buf.size(), ret);
    return buf;
}

/// Resolve the deinterlacer under the low-perf cap, choosing ONLY modes the driver advertises in
/// @p supportedMask -- emitting an unadvertised mode makes VAAPI reject the whole graph (some iHD GPUs
/// expose just motion_compensated). The DeintMode value is its rank (lower = better); see config.h.
[[nodiscard]] auto ClampDeinterlaceMode(std::string_view gpuMode, int capRank, unsigned supportedMask)
    -> std::string_view {
    const int cap = std::clamp(capRank, 0, CONFIG_DEINT_MODE_COUNT - 1);
    if (cap == 0 || supportedMask == 0) {
        return gpuMode; // MCDI never limits; an empty mask means we can't safely pick a mode
    }
    const auto isSupported = [supportedMask](int rank) -> bool {
        return (supportedMask & (1U << static_cast<unsigned>(rank))) != 0;
    };
    for (int rank = cap; rank < CONFIG_DEINT_MODE_COUNT; ++rank) { // best supported mode no better than the cap
        if (isSupported(rank)) {
            return DeintModeName(static_cast<DeintMode>(rank));
        }
    }
    for (int rank = CONFIG_DEINT_MODE_COUNT - 1; rank >= 0; --rank) { // cap unreachable: cheapest mode on offer
        if (isSupported(rank)) {
            return DeintModeName(static_cast<DeintMode>(rank));
        }
    }
    return gpuMode;
}

} // namespace

auto cVideoFilterChain::FailBuild() noexcept -> bool {
    bufferSrcCtx_ = nullptr;
    bufferSinkCtx_ = nullptr;
    filterGraph_.reset();
    return false;
}

auto cVideoFilterChain::Build(AVFrame *firstFrame, const BuildParams &params) -> bool {
    if (filterGraph_) {
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

    const uint64_t darNum = static_cast<uint64_t>(croppedW) * static_cast<uint64_t>(sarNum);
    const uint64_t darDen = static_cast<uint64_t>(croppedH) * static_cast<uint64_t>(sarDen);

    uint32_t filterWidth = dstWidth;
    uint32_t filterHeight = dstHeight;

    // Integer cross-multiply avoids FP rounding: compare darNum/darDen vs dstWidth/dstHeight.
    if (darNum * dstHeight > darDen * static_cast<uint64_t>(dstWidth)) {
        filterWidth = dstWidth; // source wider -> letterbox
        filterHeight = static_cast<uint32_t>(static_cast<uint64_t>(dstWidth) * darDen / darNum);
    } else if (darNum * dstHeight < darDen * static_cast<uint64_t>(dstWidth)) {
        filterHeight = dstHeight; // source narrower -> pillarbox
        filterWidth = static_cast<uint32_t>(static_cast<uint64_t>(dstHeight) * darNum / darDen);
    }

    // NV12/P010 chroma is 4:2:0 (2x2-subsampled); odd dimensions produce artifacts.
    // Minimum 2 so scale_vaapi never receives a 0-size surface.
    filterWidth = std::max(filterWidth & ~1U, 2U);
    filterHeight = std::max(filterHeight & ~1U, 2U);

    // Snap to the exact rect when the fit lands within ~1%: integer crop-offset / aspect rounding
    // otherwise leaves a 2-6 px black sliver on an image that should fill the screen (most visibly a
    // zoomed 16:9 source -> 1920x1076 instead of 1920x1080). The implied <=1% stretch is invisible;
    // genuine letterbox/pillarbox (4:3, scope, ...) is far larger than 1% and stays untouched.
    if (dstWidth - filterWidth <= dstWidth / 100) {
        filterWidth = dstWidth;
    }
    if (dstHeight - filterHeight <= dstHeight / 100) {
        filterHeight = dstHeight;
    }

    // UHD: GPU already saturated by 4K decode/scale; skip denoise/sharpen to avoid stutter.
    // MPEG-2 SD: DCT-block and analog-tape artefacts warrant heavier processing.
    // H.264/H.265 HD: lighter touch preserves encoder-intended detail.
    int denoiseLevel = 0;
    int sharpnessLevel = 0;

    if (!isUhd) {
        if (params.codecId == AV_CODEC_ID_MPEG2VIDEO) {
            denoiseLevel = 16;   // empirical: removes MPEG-2 blocking without smearing motion
            sharpnessLevel = 36; // empirical: compensates for heavy chroma subsampling
        } else {
            denoiseLevel = 6;    // subtle: reduces H.264/H.265 ringing at bitrate-starved edges
            sharpnessLevel = 30; // mild enhancement; strong values halate bright HD content
        }
    }

    // Compare against the cropped dimensions: those are what scale_vaapi actually receives.
    const bool needsResize = (filterWidth != croppedW || filterHeight != croppedH);

    // HDR path: P010 (10-bit packed) preserves bit depth through VPP; NV12 would clip to 8-bit.
    // scaleColorArgs is appended after ':' (resize) or after '=' (no-resize); no leading separator.
    const char *pixFmt = params.hdrPassthrough ? "p010le" : "nv12";
    std::string scaleColorArgs;
    if (params.hdrPassthrough) {
        // Pin colorimetry explicitly: scale_vaapi's "preserve input" default is driver-dependent
        // and iHD has been observed to silently downgrade to BT.709 on some frame sizes.
        const char *transfer = (params.hdrInfo.kind == StreamHdrKind::Hlg) ? "arib-std-b67" : "smpte2084";
        scaleColorArgs = std::format(
            "format={}:out_color_matrix=bt2020nc:out_color_primaries=bt2020:out_color_transfer={}:out_range=tv", pixFmt,
            transfer);
    } else {
        scaleColorArgs = std::format("format={}:out_color_matrix=bt709:out_range=tv", pixFmt);
    }

    // VBR DVB streams (and some cable muxes) omit framerate; 50/1 is the DVB-S/T baseline (= 25i).
    const int fpsNum = params.fpsNum > 0 ? params.fpsNum : 50;
    const int fpsDen = params.fpsDen > 0 ? params.fpsDen : 1;

    // deinterlace_vaapi rate=field emits two frames per field pair (25i -> 50p).
    // Round when computing naturalOutputFps so NTSC fractional rates (30000/1001,
    // 60000/1001) resolve to 30/60 rather than truncating to 29/59.
    const int fieldRateFactor = isInterlaced ? 2 : 1;
    const int64_t outputRateNum = static_cast<int64_t>(fpsNum) * fieldRateFactor;
    const int64_t outputRateDen = std::max<int64_t>(fpsDen, 1);
    const int naturalOutputFps = static_cast<int>((outputRateNum + (outputRateDen / 2)) / outputRateDen);
    const int displayFps = static_cast<int>(params.outputRefreshHz);
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
    const int64_t displayRateInSourceDen = static_cast<int64_t>(displayFps) * outputRateDen;
    const bool ratesDiffer = outputRateNum > 0 && displayFps > 0 && outputRateNum != displayRateInSourceDen;
    const int outputFps = ratesDiffer ? displayFps : naturalOutputFps;

    // Filter chain (comma-joined, built dynamically from flags above):
    //   SW decode: [bwdif|yadif] -> [hqdn3d for MPEG-2 w/o HW denoise] -> format -> [crop] -> hwupload ->
    //              [denoise_vaapi] -> scale_vaapi -> [sharpness_vaapi] -> [fps]
    //   HW decode: [deinterlace_vaapi] -> [denoise_vaapi] -> [crop] -> scale_vaapi -> [sharpness_vaapi] -> [fps]
    // [crop] only when a manual-zoom preset is active. HW decode passes crop metadata to scale_vaapi
    // (one GPU pass, zero-copy); SW decode crops in system memory before hwupload.
    // scale_vaapi is always present: it normalizes pixel format and colorimetry regardless of resize.
    // Denoise / sharpness run on the GPU (denoise_vaapi / sharpness_vaapi) whenever VPP exposes them,
    // so they never compete with a CPU-bound SW decoder (libdav1d 1080p50 etc). CPU hqdn3d is only
    // retained as a fall-back for MPEG-2 SW decode on GPUs that lack denoise_vaapi (e.g. some
    // Radeon iGPUs): MPEG-2 SW decode is cheap and the block-artefact removal materially improves
    // perceived quality. For modern codecs (H.264/HEVC/AV1) without HW denoise we skip denoise
    // entirely -- not worth the CPU cost on an already-saturated decoder thread.
    std::vector<std::string> filters;
    // Local glue: append the manual-zoom crop node (even-aligned dims/offsets computed above). Capturing
    // the crop geometry lets the two call sites below read as a plain `appendCropFilter()`.
    const auto appendCropFilter = [&filters, croppedW, croppedH, cropOffX, cropOffY]() -> void {
        filters.push_back(std::format("crop={}:{}:{}:{}", croppedW, croppedH, cropOffX, cropOffY));
    };

    // Still picture: skip the temporal deinterlacer entirely -- VAAPI VPP buffers one field pair
    // for motion estimation and never flushes it on EOS, so a lone I-frame would be swallowed.
    // Minor combing artefacts on a still frame are acceptable.
    // Trick mode: bob (spatial-only) avoids that one-frame priming delay; steady cadence absorbs it.
    // Both modes also skip denoise/sharpness: quality irrelevant at trick speeds, and still frames
    // are typically JPEG-style I-frames where spatial filters add no value.
    const bool useSimpleDeinterlace = params.trickMode || params.stillPicture;
    const bool wantDenoise = !useSimpleDeinterlace && denoiseLevel > 0 && !params.disableDenoise;

    if (isSoftwareDecode) {
        if (isInterlaced && !params.stillPicture) {
            if (useSimpleDeinterlace) {
                // yadif send_frame: spatial-only, no temporal priming delay (bwdif needs two fields).
                filters.emplace_back("yadif=mode=send_frame:parity=auto:deint=all");
            } else {
                // bwdif (w3fdif+yadif hybrid) produces better motion at broadcast bitrates than
                // VAAPI motion_adaptive on Mesa; HW deint path is taken for iHD/VA-API capable GPUs.
                filters.emplace_back("bwdif=mode=send_field:parity=auto:deint=all");
            }
        }
        // MPEG-2 SW fall-back: only when the GPU lacks denoise_vaapi. SW decode is cheap and the
        // block-artefact removal is worth the per-frame CPU cost here (~5 ms @ 1080p25).
        if (wantDenoise && !params.hasDenoise && params.codecId == AV_CODEC_ID_MPEG2VIDEO) {
            filters.emplace_back("hqdn3d=5");
        }
        filters.push_back(std::format("format={}", pixFmt));
        // SW decode: crop the system-memory frame before uploading so only the kept region is
        // uploaded and zoom adds no GPU readback.
        if (wantCrop) {
            appendCropFilter();
        }
        filters.emplace_back("hwupload");
    } else {
        if (isInterlaced && !params.stillPicture && !params.deinterlaceMode.empty()) {
            if (params.trickMode) {
                // bob rate=frame: spatial-only, single-frame latency; avoids the one-frame
                // priming buffer that motion_adaptive requires and never drains at trick speeds.
                filters.emplace_back("deinterlace_vaapi=mode=bob:rate=frame");
            } else {
                // Low-perf cap (no-op at default); only ever emits a driver-supported mode.
                const std::string_view deintMode =
                    ClampDeinterlaceMode(params.deinterlaceMode, params.deinterlaceMaxRank, params.deinterlaceModeMask);
                filters.push_back(std::format("deinterlace_vaapi=mode={}:rate=field", deintMode));
            }
        }
    }

    // HW denoise (post-hwupload for SW / native for HW). Skipped on GPUs that don't expose it;
    // the MPEG-2 SW fall-back above covers the one case where that materially hurts quality.
    if (wantDenoise && params.hasDenoise) {
        filters.push_back(std::format("denoise_vaapi=denoise={}", denoiseLevel));
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
        // bicubic (hq) too expensive at 4K; the low-perf knob drops it on non-UHD too.
        const char *scaleMode = (isUhd || params.disableHqScaling) ? "" : ":mode=hq";
        filters.push_back(
            std::format("scale_vaapi=w={}:h={}{}:{}", filterWidth, filterHeight, scaleMode, scaleColorArgs));
    } else {
        filters.push_back(std::format("scale_vaapi={}", scaleColorArgs));
    }

    if (!useSimpleDeinterlace && sharpnessLevel > 0 && params.hasSharpness && !params.disableSharpness) {
        filters.push_back(std::format("sharpness_vaapi=sharpness={}", sharpnessLevel));
    }

    hasFpsFilter_ = !useSimpleDeinterlace && ratesDiffer;
    if (hasFpsFilter_) {
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

    filterGraph_.reset(avfilter_graph_alloc());
    if (!filterGraph_) [[unlikely]] {
        esyslog("vaapivideo/filter: failed to allocate filter graph");
        return false;
    }

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

    // compactLog is the sole gate -- don't also dedup identical args, or a same-format channel
    // switch (trick/zoom set compactLog) would never surface its settings.
    if (!compactLog) {
        dsyslog("vaapivideo/filter: buffer source args='%s'", bufferSrcArgs.c_str());
    }

    // hw_frames_ctx must be attached to the buffer source before avfilter_init_str();
    // FFmpeg 7.x rejects initialization of a HW-format source without it.
    bufferSrcCtx_ = avfilter_graph_alloc_filter(filterGraph_.get(), avfilter_get_by_name("buffer"), "in");
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
            esyslog("vaapivideo/filter: av_buffersrc_parameters_set failed: %s", FmtErr(setRet).data());
            return FailBuild();
        }
    }

    int ret = avfilter_init_str(bufferSrcCtx_, bufferSrcArgs.c_str());
    if (ret < 0) [[unlikely]] {
        esyslog("vaapivideo/filter: failed to init buffer source '%s': %s", bufferSrcArgs.c_str(), FmtErr(ret).data());
        return FailBuild();
    }

    ret = avfilter_graph_create_filter(&bufferSinkCtx_, avfilter_get_by_name("buffersink"), "out", nullptr, nullptr,
                                       filterGraph_.get());
    if (ret < 0) [[unlikely]] {
        esyslog("vaapivideo/filter: failed to create buffer sink: %s", FmtErr(ret).data());
        return FailBuild();
    }

    // Segment API (not parse_ptr) because FFmpeg 8.x hwupload_init() rejects a filter with
    // no hw_device_ctx and parse_ptr inits filters as part of parsing -- too early to attach
    // the device. Segment splits parse / create / init so we can set hw_device_ctx between
    // create and init. (HW decode doesn't strictly need hw_device_ctx on the VAAPI filters --
    // they pick it up from hw_frames_ctx via the link -- but setting it is harmless.)
    AVFilterGraphSegment *segment = nullptr;
    ret = avfilter_graph_segment_parse(filterGraph_.get(), filterChain.c_str(), 0, &segment);
    if (ret < 0) [[unlikely]] {
        esyslog("vaapivideo/filter: failed to parse filter chain '%s': %s", filterChain.c_str(), FmtErr(ret).data());
        return FailBuild();
    }

    ret = avfilter_graph_segment_create_filters(segment, 0);
    if (ret < 0) [[unlikely]] {
        esyslog("vaapivideo/filter: failed to create segment filters '%s': %s", filterChain.c_str(),
                FmtErr(ret).data());
        avfilter_graph_segment_free(&segment);
        return FailBuild();
    }

    // Attach hw_device_ctx to every newly created filter (skip the externally allocated
    // buffer source/sink, which were already initialized above). Without this, hwupload's
    // init returns EINVAL and scale_vaapi/sharpness_vaapi fail at graph config time.
    for (unsigned int i = 0; i < filterGraph_->nb_filters; ++i) {
        AVFilterContext *filterCtx = filterGraph_->filters[i];
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
        esyslog("vaapivideo/filter: failed to apply segment '%s': %s", filterChain.c_str(), FmtErr(ret).data());
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
                FmtErr(ret).data());
        return FailBuild();
    }

    ret = avfilter_graph_config(filterGraph_.get(), nullptr);
    if (ret < 0) [[unlikely]] {
        esyslog("vaapivideo/filter: failed to configure filter graph '%s': %s", filterChain.c_str(),
                FmtErr(ret).data());
        return FailBuild();
    }

    outputFrameDurationMs_ = outputFps > 0 ? std::max(1, 1000 / outputFps) : 20; // 20 ms = 50 fps fallback

    if (compactLog) {
        if (wantCrop) {
            // Zoom-driven rebuild: show the crop so the active zoom is verifiable in the log.
            dsyslog("vaapivideo/filter: rebuilt -> %ux%u (zoom crop=%u:%u:%u:%u)", filterWidth, filterHeight, croppedW,
                    croppedH, cropOffX, cropOffY);
        } else {
            dsyslog("vaapivideo/filter: rebuilt -> %ux%u", filterWidth, filterHeight);
        }
    } else {
        // Non-compact => Clear / channel switch (trick/zoom set compactLog). Log it even when the
        // chain is byte-for-byte identical, so every switch surfaces its settings.
        const char *cadenceTag = "";
        if (ratesDiffer) {
            if (outputRateNum < displayRateInSourceDen) {
                cadenceTag = (displayRateInSourceDen % outputRateNum) == 0 ? ", duplicated" : ", uneven cadence";
            } else {
                cadenceTag = ", decimated";
            }
        }
        isyslog("vaapivideo/filter: VAAPI filter initialized (%dx%d -> %ux%u%s%s, out=%s %s)", srcWidth, srcHeight,
                filterWidth, filterHeight, isInterlaced ? ", deinterlaced" : "", cadenceTag, pixFmt,
                params.hdrPassthrough ? StreamHdrKindName(params.hdrInfo.kind) : "SDR");
        dsyslog("vaapivideo/filter: filter chain='%s'", filterChain.c_str());
    }

    return true;
}

// ============================================================================
// === SEND / RECEIVE / RESET ===
// ============================================================================

auto cVideoFilterChain::SendFrame(AVFrame *frame) noexcept -> int {
    if (!filterGraph_ || !bufferSrcCtx_) [[unlikely]] {
        return AVERROR(EINVAL);
    }
    // KEEP_REF: the filter graph makes its own ref; the decoder retains ownership of the
    // underlying AVFrame data. For a null frame (EOS flush signal) pass 0 explicitly --
    // flags are undefined on the EOS path in FFmpeg's internal buffersrc code.
    const int flags = (frame != nullptr) ? AV_BUFFERSRC_FLAG_KEEP_REF : 0;
    return av_buffersrc_add_frame_flags(bufferSrcCtx_, frame, flags);
}

auto cVideoFilterChain::ReceiveFrame(AVFrame *out) noexcept -> int {
    if (!filterGraph_ || !bufferSinkCtx_) [[unlikely]] {
        return AVERROR(EINVAL);
    }
    return av_buffersink_get_frame(bufferSinkCtx_, out);
}

auto cVideoFilterChain::Reset() noexcept -> void {
    bufferSrcCtx_ = nullptr;
    bufferSinkCtx_ = nullptr;
    hasFpsFilter_ = false;
    // Keep the old graph alive in previousFilterGraph_: destroying it immediately causes
    // -EIO on iHD because the VPP output surfaces are still DMA-BUF mapped by the display
    // thread. The saved graph (and its hw_frames_ctx) is released on the next Build() or
    // destructor, by which time the display thread has finished mapping.
    // Guard: a double-reset (Clear -> drain -> EOS) must not overwrite the saved graph with null.
    if (filterGraph_) {
        previousFilterGraph_ = std::move(filterGraph_);
    }
}
