// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file common.h
 * @brief RAII deleters, version guards, and shared definitions
 */

#ifndef VDR_VAAPIVIDEO_COMMON_H
#define VDR_VAAPIVIDEO_COMMON_H

// ============================================================================
// === SYSTEM HEADERS ===
// ============================================================================

// ============================================================================
// === C++ STANDARD LIBRARY ===
// ============================================================================
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <memory>
#include <queue>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// ============================================================================
// === DRM/KMS ===
// ============================================================================
#include <libdrm/drm.h>
#include <libdrm/drm_fourcc.h>
#include <libdrm/drm_mode.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

// ============================================================================
// === FFMPEG HEADERS ===
// ============================================================================

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/codec.h>
#include <libavcodec/codec_id.h>
#include <libavcodec/defs.h>
#include <libavcodec/packet.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/avutil.h>
#include <libavutil/buffer.h>
#include <libavutil/channel_layout.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_drm.h>
#include <libavutil/hwcontext_vaapi.h>
#include <libavutil/intreadwrite.h>
#include <libavutil/log.h>
#include <libavutil/mem.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
}
#pragma GCC diagnostic pop

// ============================================================================
// === VDR HEADERS ===
// ============================================================================

#include <vdr/plugin.h>
#include <vdr/thread.h>
#include <vdr/tools.h>

// ============================================================================
// === VERSION CHECKS ===
// ============================================================================

#if APIVERSNUM < 30011
#error "VDR 2.7.9+ required (APIVERSNUM >= 30011)"
#endif

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(61, 3, 100)
#error "FFmpeg 7.0+ required (libavcodec 61.3.100+)"
#endif

// ============================================================================
// === PLUGIN METADATA ===
// ============================================================================

inline constexpr const char *PLUGIN_DESCRIPTION = "Hardware-accelerated video playback with VAAPI";
inline constexpr const char *PLUGIN_VERSION = "1.0.0";

// ============================================================================
// === CONSTANTS ===
// ============================================================================

inline constexpr int SHUTDOWN_TIMEOUT_MS = 5000; ///< Thread shutdown timeout (ms)

// ============================================================================
// === FFMPEG UTILITIES ===
// ============================================================================

/// Convert an FFmpeg AVERROR code to a human-readable error string.
/// The returned array is a temporary whose .data() is valid for the full expression.
/// Usage: esyslog("failed: %s", AvErr(ret).data());
[[nodiscard]] inline auto AvErr(int errnum) noexcept -> std::array<char, AV_ERROR_MAX_STRING_SIZE> {
    std::array<char, AV_ERROR_MAX_STRING_SIZE> buf{};
    av_make_error_string(buf.data(), buf.size(), errnum);
    return buf;
}

// ============================================================================
// === RAII CUSTOM DELETERS ===
// ============================================================================

// --- FFmpeg Deleters ---

/// Deleter for AVCodecContext (avcodec_free_context)
struct FreeAVCodecContext {
    auto operator()(AVCodecContext *ctx) const noexcept -> void { avcodec_free_context(&ctx); }
};

/// Deleter for AVCodecParserContext (av_parser_close)
struct FreeAVCodecParserContext {
    auto operator()(AVCodecParserContext *ctx) const noexcept -> void { av_parser_close(ctx); }
};

/// Deleter for AVFilterGraph (avfilter_graph_free)
struct FreeAVFilterGraph {
    auto operator()(AVFilterGraph *graph) const noexcept -> void { avfilter_graph_free(&graph); }
};

/// Deleter for AVFrame (av_frame_free)
struct FreeAVFrame {
    auto operator()(AVFrame *frame) const noexcept -> void { av_frame_free(&frame); }
};

/// Deleter for AVPacket (av_packet_free)
struct FreeAVPacket {
    auto operator()(AVPacket *pkt) const noexcept -> void { av_packet_free(&pkt); }
};

// --- DRM Deleters ---

/// Deleter for drmModeConnector (drmModeFreeConnector)
struct FreeDrmConnector {
    auto operator()(drmModeConnector *conn) const noexcept -> void { drmModeFreeConnector(conn); }
};

/// Deleter for drmDevice (drmFreeDevice)
struct FreeDrmDevice {
    auto operator()(drmDevice *dev) const noexcept -> void { drmFreeDevice(&dev); }
};

/// Deleter for drmModeObjectProperties (drmModeFreeObjectProperties)
struct FreeDrmObjectProperties {
    auto operator()(drmModeObjectProperties *props) const noexcept -> void { drmModeFreeObjectProperties(props); }
};

/// Deleter for drmModePlaneRes (drmModeFreePlaneResources)
struct FreeDrmPlaneResources {
    auto operator()(drmModePlaneRes *res) const noexcept -> void { drmModeFreePlaneResources(res); }
};

/// Deleter for drmModePropertyRes (drmModeFreeProperty)
struct FreeDrmProperty {
    auto operator()(drmModePropertyRes *prop) const noexcept -> void { drmModeFreeProperty(prop); }
};

/// Deleter for drmModeRes (drmModeFreeResources)
struct FreeDrmResources {
    auto operator()(drmModeRes *res) const noexcept -> void { drmModeFreeResources(res); }
};

#endif // VDR_VAAPIVIDEO_COMMON_H
