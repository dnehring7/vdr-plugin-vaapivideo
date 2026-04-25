// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file filter.cpp
 * @brief VAAPI VPP filter chain implementation: graph construction, frame routing, and HDR helpers.
 */

#include "filter.h"

#include "common.h"

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
    // promises sizeof(AVMastering...), but >= lets us tolerate a larger future layout.
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

} // namespace

auto cVideoFilterChain::Build(AVFrame *firstFrame, const BuildParams &params) -> bool {
    if (filterGraph_) {
        return true;
    }

    // Drop a partially-built graph without touching previousFilterGraph_: that slot
    // owns the last good graph whose hw_frames_ctx keeps in-flight VPP surfaces
    // PRIME-exportable. Routing through Reset() would clobber it and break export.
    const auto failBuild = [&]() noexcept -> bool {
        bufferSrcCtx_ = nullptr;
        bufferSinkCtx_ = nullptr;
        filterGraph_.reset();
        return false;
    };

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

    const int srcWidth = firstFrame->width;
    const int srcHeight = firstFrame->height;
    const bool isInterlaced = (firstFrame->flags & AV_FRAME_FLAG_INTERLACED) != 0;
    const auto srcPixFmt = static_cast<AVPixelFormat>(firstFrame->format);

    // 1088, not 1080: MPEG-2/H.264 pad height to a 16-px macroblock boundary.
    const bool isUhd = (srcWidth > 1920 || srcHeight > 1088);
    const bool isSoftwareDecode = (srcPixFmt != AV_PIX_FMT_VAAPI);

    const uint32_t dstWidth = params.outputWidth;
    const uint32_t dstHeight = params.outputHeight;

    // DRM atomic planes have no hardware scaler, so VPP must output the letterboxed/pillarboxed
    // size directly. Guard against streams that report SAR 0/0 (treated as square).
    const int sarNum = firstFrame->sample_aspect_ratio.num > 0 ? firstFrame->sample_aspect_ratio.num : 1;
    const int sarDen = firstFrame->sample_aspect_ratio.den > 0 ? firstFrame->sample_aspect_ratio.den : 1;

    const uint64_t darNum = static_cast<uint64_t>(srcWidth) * static_cast<uint64_t>(sarNum);
    const uint64_t darDen = static_cast<uint64_t>(srcHeight) * static_cast<uint64_t>(sarDen);

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

    // UHD: GPU already saturated by 4K decode/scale; skip denoise/sharpen to avoid stutter.
    // MPEG-2 SD: DCT-block and analog-tape artefacts warrant heavier processing.
    // H.264/H.265 HD: lighter touch preserves encoder-intended detail.
    int denoiseLevel = 0;
    int sharpnessLevel = 0;

    if (!isUhd) {
        if (params.codecId == AV_CODEC_ID_MPEG2VIDEO) {
            denoiseLevel = 16;   ///< empirical: removes MPEG-2 blocking without smearing motion
            sharpnessLevel = 36; ///< empirical: compensates for heavy chroma subsampling
        } else {
            denoiseLevel = 6;    ///< subtle: reduces H.264/H.265 ringing at bitrate-starved edges
            sharpnessLevel = 30; ///< mild enhancement; strong values halate bright HD content
        }
    }

    const bool needsResize =
        (filterWidth != static_cast<uint32_t>(srcWidth) || filterHeight != static_cast<uint32_t>(srcHeight));

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
    const bool upconvertProgressive =
        (!isInterlaced && naturalOutputFps > 0 && displayFps > 0 && naturalOutputFps < displayFps);
    const int outputFps = upconvertProgressive ? displayFps : naturalOutputFps;

    // Filter chain (comma-joined, built dynamically from flags above):
    //   SW decode: [bwdif|yadif] -> [hqdn3d] -> format -> hwupload -> scale_vaapi -> [sharpness_vaapi] -> [fps]
    //   HW decode: [deinterlace_vaapi] -> [denoise_vaapi] -> scale_vaapi -> [sharpness_vaapi] -> [fps]
    // scale_vaapi is always present: it normalises pixel format and colorimetry regardless of resize.
    std::vector<std::string> filters;

    // Still picture: skip the temporal deinterlacer entirely -- VAAPI VPP buffers one field pair
    // for motion estimation and never flushes it on EOS, so a lone I-frame would be swallowed.
    // Minor combing artefacts on a still frame are acceptable.
    // Trick mode: bob (spatial-only) avoids that one-frame priming delay; steady cadence absorbs it.
    // Both modes also skip denoise/sharpness: quality irrelevant at trick speeds, and still frames
    // are typically JPEG-style I-frames where spatial filters add no value.
    const bool useSimpleDeinterlace = params.trickMode || params.stillPicture;

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
        if (!useSimpleDeinterlace && denoiseLevel > 0) {
            // hqdn3d: temporal weight 5 for MPEG-2 (heavy grain), 3 for H.264/H.265 (spatial artefacts).
            const int hqdn3dStrength = (params.codecId == AV_CODEC_ID_MPEG2VIDEO) ? 5 : 3;
            filters.push_back(std::format("hqdn3d={}", hqdn3dStrength));
        }
        filters.push_back(std::format("format={}", pixFmt));
        filters.emplace_back("hwupload");
    } else {
        if (isInterlaced && !params.stillPicture && !params.deinterlaceMode.empty()) {
            if (params.trickMode) {
                // bob rate=frame: spatial-only, single-frame latency; avoids the one-frame
                // priming buffer that motion_adaptive requires and never drains at trick speeds.
                filters.emplace_back("deinterlace_vaapi=mode=bob:rate=frame");
            } else {
                filters.push_back(std::format("deinterlace_vaapi=mode={}:rate=field", params.deinterlaceMode));
            }
        }
        if (!useSimpleDeinterlace && denoiseLevel > 0 && params.hasDenoise) {
            filters.push_back(std::format("denoise_vaapi=denoise={}", denoiseLevel));
        }
    }

    if (needsResize) {
        const char *scaleMode = isUhd ? "" : ":mode=hq"; // bicubic (hq) too expensive at 4K
        filters.push_back(
            std::format("scale_vaapi=w={}:h={}{}:{}", filterWidth, filterHeight, scaleMode, scaleColorArgs));
    } else {
        filters.push_back(std::format("scale_vaapi={}", scaleColorArgs));
    }

    if (!useSimpleDeinterlace && sharpnessLevel > 0 && params.hasSharpness) {
        filters.push_back(std::format("sharpness_vaapi=sharpness={}", sharpnessLevel));
    }

    if (!useSimpleDeinterlace && upconvertProgressive) {
        // fps placed last: scale and sharpness run once per source frame; fps only duplicates
        // timestamps and presentation timing metadata -- no pixel work on duplicated frames.
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
    const std::string bufferSrcArgs =
        std::format("video_size={}x{}:pix_fmt={}:time_base=1/90000:pixel_aspect={}/{}:frame_rate={}/{}", srcWidth,
                    srcHeight, static_cast<int>(srcPixFmt), sarNum, sarDen, fpsNum, fpsDen);

    dsyslog("vaapivideo/filter: buffer source args='%s'", bufferSrcArgs.c_str());

    // hw_frames_ctx must be attached to the buffer source before avfilter_init_str();
    // FFmpeg 7.x rejects initialisation of a HW-format source without it.
    bufferSrcCtx_ = avfilter_graph_alloc_filter(filterGraph_.get(), avfilter_get_by_name("buffer"), "in");
    if (!bufferSrcCtx_) [[unlikely]] {
        esyslog("vaapivideo/filter: failed to allocate buffer source filter");
        return failBuild();
    }

    if (!isSoftwareDecode) {
        if (!params.hwFramesCtx) [[unlikely]] {
            esyslog("vaapivideo/filter: hw decode requires hw_frames_ctx");
            return failBuild();
        }
        AVBufferSrcParameters *hwFramesParams = av_buffersrc_parameters_alloc();
        if (!hwFramesParams) [[unlikely]] {
            esyslog("vaapivideo/filter: failed to allocate buffer source parameters");
            return failBuild();
        }

        hwFramesParams->hw_frames_ctx = av_buffer_ref(params.hwFramesCtx);
        if (!hwFramesParams->hw_frames_ctx) [[unlikely]] {
            esyslog("vaapivideo/filter: av_buffer_ref(hw_frames_ctx) failed");
            av_free(hwFramesParams);
            return failBuild();
        }
        const int setRet = av_buffersrc_parameters_set(bufferSrcCtx_, hwFramesParams);
        // av_buffersrc_parameters_set makes its own internal ref; caller must unref unconditionally
        // (both success and failure) to avoid leaking the AVHWFramesContext ref on every Build().
        av_buffer_unref(&hwFramesParams->hw_frames_ctx);
        av_free(hwFramesParams);
        if (setRet < 0) [[unlikely]] {
            esyslog("vaapivideo/filter: av_buffersrc_parameters_set failed: %s", FmtErr(setRet).data());
            return failBuild();
        }
    }

    int ret = avfilter_init_str(bufferSrcCtx_, bufferSrcArgs.c_str());
    if (ret < 0) [[unlikely]] {
        esyslog("vaapivideo/filter: failed to init buffer source '%s': %s", bufferSrcArgs.c_str(), FmtErr(ret).data());
        return failBuild();
    }

    ret = avfilter_graph_create_filter(&bufferSinkCtx_, avfilter_get_by_name("buffersink"), "out", nullptr, nullptr,
                                       filterGraph_.get());
    if (ret < 0) [[unlikely]] {
        esyslog("vaapivideo/filter: failed to create buffer sink: %s", FmtErr(ret).data());
        return failBuild();
    }

    // FFmpeg names these from the filter-graph string's perspective, not the caller's:
    // "outputs" connects to our buffer source (data flows out of the graph string input),
    // "inputs" connects to our buffer sink (data flows into the graph string output).
    AVFilterInOut *graphInputs = avfilter_inout_alloc();
    AVFilterInOut *graphOutputs = avfilter_inout_alloc();
    if (!graphInputs || !graphOutputs) [[unlikely]] {
        avfilter_inout_free(&graphInputs);
        avfilter_inout_free(&graphOutputs);
        return failBuild();
    }

    graphOutputs->name = av_strdup("in");
    graphOutputs->filter_ctx = bufferSrcCtx_;

    graphInputs->name = av_strdup("out");
    graphInputs->filter_ctx = bufferSinkCtx_;

    if (!graphOutputs->name || !graphInputs->name) [[unlikely]] {
        avfilter_inout_free(&graphInputs);
        avfilter_inout_free(&graphOutputs);
        return failBuild();
    }

    ret = avfilter_graph_parse_ptr(filterGraph_.get(), filterChain.c_str(), &graphInputs, &graphOutputs, nullptr);
    avfilter_inout_free(&graphInputs);
    avfilter_inout_free(&graphOutputs);

    if (ret < 0) [[unlikely]] {
        esyslog("vaapivideo/filter: failed to parse filter chain '%s': %s", filterChain.c_str(), FmtErr(ret).data());
        return failBuild();
    }

    // SW decode: every filter node must have hw_device_ctx, not just hwupload.
    // scale_vaapi and sharpness_vaapi resolve their VAAPI device through this context; omitting
    // it on those nodes causes AVERROR(EINVAL) at graph config time on FFmpeg 6+.
    if (isSoftwareDecode) {
        for (unsigned int i = 0; i < filterGraph_->nb_filters; ++i) {
            filterGraph_->filters[i]->hw_device_ctx = av_buffer_ref(params.hwDeviceRef);
            if (!filterGraph_->filters[i]->hw_device_ctx) [[unlikely]] {
                esyslog("vaapivideo/filter: av_buffer_ref(hwDeviceRef) failed for filter %u", i);
                return failBuild();
            }
        }
    }

    ret = avfilter_graph_config(filterGraph_.get(), nullptr);
    if (ret < 0) [[unlikely]] {
        esyslog("vaapivideo/filter: failed to configure filter graph '%s': %s", filterChain.c_str(),
                FmtErr(ret).data());
        return failBuild();
    }

    outputFrameDurationMs_ = outputFps > 0 ? std::max(1, 1000 / outputFps) : 20; // 20 ms = 50 fps fallback

    isyslog("vaapivideo/filter: VAAPI filter initialized (%dx%d -> %ux%u%s%s, out=%s %s)", srcWidth, srcHeight,
            filterWidth, filterHeight, isInterlaced ? ", deinterlaced" : "",
            upconvertProgressive ? (isInterlaced ? "" : ", upconverted") : "", pixFmt,
            params.hdrPassthrough ? StreamHdrKindName(params.hdrInfo.kind) : "SDR");
    dsyslog("vaapivideo/filter: filter chain='%s'", filterChain.c_str());
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
    // Keep the old graph alive in previousFilterGraph_: destroying it immediately causes
    // -EIO on iHD because the VPP output surfaces are still DMA-BUF mapped by the display
    // thread. The saved graph (and its hw_frames_ctx) is released on the next Build() or
    // destructor, by which time the display thread has finished mapping.
    // Guard: a double-reset (Clear -> drain -> EOS) must not overwrite the saved graph with null.
    if (filterGraph_) {
        previousFilterGraph_ = std::move(filterGraph_);
    }
}
