// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file common.h
 * @brief RAII deleters, version guards, and shared definitions
 */

#ifndef VDR_VAAPIVIDEO_COMMON_H
#define VDR_VAAPIVIDEO_COMMON_H

// config.h is a leaf (it includes nothing of ours), so this stays acyclic. Needed for the display
// defaults the DRM helpers below fall back to, which must keep a single definition.
#include "config.h"

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
#include <numeric>
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
#include <libavformat/avformat.h>
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
#include <libavutil/mastering_display_metadata.h>
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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <vdr/config.h>
#pragma GCC diagnostic pop

// ============================================================================
// === VERSION CHECKS ===
// ============================================================================

#if APIVERSNUM < 20606
#error "VDR 2.6.6+ required (APIVERSNUM >= 20606)"
#endif

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(61, 3, 100)
#error "FFmpeg 7.0+ required (libavcodec 61.3.100+)"
#endif

// ============================================================================
// === PLUGIN METADATA ===
// ============================================================================

/// Shown by "vdr -h" and in VDR's plugin list.
inline constexpr const char *PLUGIN_DESCRIPTION = "Hardware-accelerated video playback with VAAPI";
inline constexpr const char *PLUGIN_NAME = "vaapivideo"; ///< VDR plugin name; cRemote::CallPlugin arg.
inline constexpr const char *PLUGIN_VERSION = "1.8.3";   ///< Reported to VDR; "make dist" greps this line.

// ============================================================================
// === TRACE LOGGING ===
// ============================================================================

/// Debug log line emitted only while tracing is on (-t / --trace, SVDRP TRACE on|off).
/// Shaped like VDR's own dsyslog: the arguments are not evaluated when the gate is closed,
/// so it is cheap enough for the A/V-sync hot paths whose per-frame narration it carries
/// (see AVSYNC.md "Tracing"). Everything a user needs without -t stays on dsyslog/isyslog.
#define tsyslog(...) void(TraceEnabled() && (dsyslog(__VA_ARGS__), true)) // NOLINT(cppcoreguidelines-macro-usage)

// ============================================================================
// === CONSTANTS ===
// ============================================================================

inline constexpr int SHUTDOWN_TIMEOUT_MS = 5000; ///< Thread shutdown timeout (ms)

// ============================================================================
// === HDR STREAM CLASSIFICATION ===
// ============================================================================

/// Coarse HDR classification of an incoming video stream. Produced by the decoder on the
/// first decoded frame (from codec profile + AVFrame color_* fields) and consumed by the
/// display to decide the DRM/KMS output path (SDR BT.709 NV12 vs HDR10/HLG P010 BT.2020).
enum class StreamHdrKind : uint8_t {
    Sdr = 0,   ///< Not HDR -- use the existing SDR BT.709 pipeline
    Hdr10 = 1, ///< BT.2020 primaries + SMPTE ST 2084 (PQ) transfer, 10-bit
    Hlg = 2,   ///< BT.2020 primaries + ARIB STD-B67 (HLG) transfer, 10-bit
};

/// HDR metadata extracted from the first frame of an HDR stream. `kind == Sdr` means the
/// `mastering*` / `contentLight*` fields are meaningless and must be ignored. For HLG,
/// mastering-display metadata is uncommon; absence is non-fatal (emit all-zero bits per
/// HDMI 2.1 section 7.6.1 -- the sink treats that as "unknown").
struct HdrStreamInfo {
    StreamHdrKind kind{StreamHdrKind::Sdr}; ///< Classification result
    bool hasMasteringDisplay{};             ///< True iff AV_FRAME_DATA_MASTERING_DISPLAY_METADATA was present
    bool hasContentLight{};                 ///< True iff AV_FRAME_DATA_CONTENT_LIGHT_LEVEL was present
    AVMasteringDisplayMetadata mastering{}; ///< Display primaries + luminance; valid iff hasMasteringDisplay
    AVContentLightMetadata contentLight{};  ///< MaxCLL / MaxFALL; valid iff hasContentLight
};

/// Human-readable name for a StreamHdrKind; shared by decoder logs and display logs.
[[nodiscard]] constexpr auto StreamHdrKindName(StreamHdrKind kind) noexcept -> const char * {
    switch (kind) {
        case StreamHdrKind::Sdr:
            return "SDR";
        case StreamHdrKind::Hdr10:
            return "HDR10";
        case StreamHdrKind::Hlg:
            return "HLG";
    }
    return "?";
}

// ============================================================================
// === STREAM-START TRACE ===
// ============================================================================

/// One-shot "time to first ..." trace for a stream start. While tracing is on, SetPlayMode() arms every
/// component with one common epoch; each reports its first post-start milestones exactly once as
/// "+N ms" after it, so a single journal excerpt shows where channel-switch latency went (see AVSYNC.md
/// "Stream-start trace"). The milestones are emitted with tsyslog(), so turning tracing off mid-stream
/// drops the ones still pending.
/// Lock-free so it may sit on hot paths: once drained, Fire() is a single acquire load.
struct StreamStartTrace {
    std::atomic<uint64_t> epochMs{0}; ///< cTimeMs::Now() at the switch; read only when a milestone fires
    std::atomic<uint32_t> pending{0}; ///< Bit per milestone still to report; 0 = disarmed / drained

    /// Arm with the switch epoch and the set of milestones to report (component-defined bit mask).
    auto Arm(uint64_t epoch, uint32_t mask) noexcept -> void {
        epochMs.store(epoch, std::memory_order_relaxed);
        pending.store(mask, std::memory_order_release); // publishes epochMs to every Fire() that sees the mask
    }
    /// Disarm: every remaining milestone is dropped silently.
    auto Disarm() noexcept -> void { pending.store(0, std::memory_order_relaxed); }
    /// True while @p bit has not fired yet -- for callers that must probe state before deciding to fire.
    [[nodiscard]] auto Pending(uint32_t bit) const noexcept -> bool {
        return (pending.load(std::memory_order_acquire) & bit) != 0;
    }
    /// Report milestone @p bit: ms since the epoch on its first firing, -1 when it already fired,
    /// the trace is disarmed, or another thread won the race for the bit.
    [[nodiscard]] auto Fire(uint32_t bit) noexcept -> int64_t {
        if ((pending.load(std::memory_order_acquire) & bit) == 0) [[likely]] {
            return -1;
        }
        if ((pending.fetch_and(~bit, std::memory_order_acq_rel) & bit) == 0) [[unlikely]] {
            return -1;
        }
        return static_cast<int64_t>(cTimeMs::Now() - epochMs.load(std::memory_order_relaxed));
    }
};

// ============================================================================
// === DRM/KMS UTILITIES ===
// ============================================================================

/// Exact refresh rate of a KMS mode, in millihertz; 0 for a degenerate mode.
///
/// drmModeModeInfo::vrefresh is rounded to whole Hz, so 59.94 and 60 (and 23.976 and 24) are
/// indistinguishable there -- fatal for cadence matching, where the wrong one costs a duplicated
/// frame every ~1000. Mirrors the kernel's drm_mode_vrefresh() so the rounded value always agrees
/// with what the driver reports.
[[nodiscard]] inline auto ModeRefreshMilliHz(const drmModeModeInfo &mode) noexcept -> uint32_t {
    if (mode.htotal == 0 || mode.vtotal == 0 || mode.clock == 0) [[unlikely]] {
        return 0;
    }
    // Interlaced modes are quoted at their FIELD rate; doublescan and vscan divide it.
    uint64_t num = static_cast<uint64_t>(mode.clock) * 1000000ULL;
    if ((mode.flags & DRM_MODE_FLAG_INTERLACE) != 0) {
        num *= 2;
    }
    uint64_t den = static_cast<uint64_t>(mode.htotal) * static_cast<uint64_t>(mode.vtotal);
    if ((mode.flags & DRM_MODE_FLAG_DBLSCAN) != 0) {
        den *= 2;
    }
    if (mode.vscan > 1) {
        den *= mode.vscan;
    }
    return static_cast<uint32_t>((num + (den / 2)) / den); // round to nearest
}

/// An exact width:height ratio. Rational rather than a double so the VPP fit can cross-multiply in
/// integers (see cVideoFilterChain::Build).
struct AspectRatio {
    uint32_t den{1}; ///< Height side of the ratio
    uint32_t num{1}; ///< Width side of the ratio
};

/// Picture aspect a KMS mode is meant to be shown at, as an exact ratio.
///
/// Not hdisplay/vdisplay: the CEA SD timings (720x576, 720x480) are anamorphic and exist in both a
/// 4:3 and a 16:9 variant differing only in the picture-aspect flag. Falls back to the pixel ratio
/// for VESA/GTF timings, which are square-pixel by construction.
///
/// The flag survives drm_mode_getconnector() only once DRM_CLIENT_CAP_ATOMIC (or
/// DRM_CLIENT_CAP_ASPECT_RATIO) is taken on the fd. It also round-trips into our mode blob, which
/// is what makes the driver emit the CEA VIC / AVI-InfoFrame aspect that tells the TV to stretch
/// an anamorphic raster back out.
[[nodiscard]] inline auto ModePictureAspectRatio(const drmModeModeInfo &mode) noexcept -> AspectRatio {
    switch (mode.flags & DRM_MODE_FLAG_PIC_AR_MASK) {
        case DRM_MODE_FLAG_PIC_AR_4_3:
            return {.den = 3, .num = 4};
        case DRM_MODE_FLAG_PIC_AR_16_9:
            return {.den = 9, .num = 16};
        case DRM_MODE_FLAG_PIC_AR_64_27:
            return {.den = 27, .num = 64};
        case DRM_MODE_FLAG_PIC_AR_256_135:
            return {.den = 135, .num = 256};
        default:
            break;
    }
    if (mode.hdisplay == 0 || mode.vdisplay == 0) [[unlikely]] {
        return {.den = 1, .num = 1};
    }
    return {.den = mode.vdisplay, .num = mode.hdisplay};
}

/// Double form of ModePictureAspectRatio(), for logging and the aspect gate's tolerance compare.
[[nodiscard]] inline auto ModePictureAspect(const drmModeModeInfo &mode) noexcept -> double {
    const AspectRatio aspect = ModePictureAspectRatio(mode);
    if (aspect.den == 0) [[unlikely]] {
        return DISPLAY_DEFAULT_ASPECT_RATIO;
    }
    return static_cast<double>(aspect.num) / static_cast<double>(aspect.den);
}

/// Pixel aspect of a mode's scanout raster: how much wider than tall one framebuffer pixel lands on
/// the panel. 1:1 on every square-pixel timing, 64:45 on a 720x576 flagged 16:9, 32:27 on a 720x480.
///
/// KMS scanout is 1:1 -- there is no plane scaler in the path -- so the VPP-fitted framebuffer is
/// the only place the anamorphic stretch can be compensated. Get it wrong and a 16:9 source is
/// letterboxed into a 1.25:1 raster the TV then stretches back out: ~30% short vertically.
[[nodiscard]] inline auto ModePixelAspectRatio(const drmModeModeInfo &mode) noexcept -> AspectRatio {
    const AspectRatio picture = ModePictureAspectRatio(mode);
    if (mode.hdisplay == 0 || mode.vdisplay == 0 || picture.num == 0 || picture.den == 0) [[unlikely]] {
        return {.den = 1, .num = 1};
    }
    // par = picture / (hdisplay/vdisplay). Reduced so the filter's cross-multiply stays far from
    // overflow and an equality test against 1:1 is exact.
    uint32_t num = picture.num * mode.vdisplay;
    uint32_t den = picture.den * mode.hdisplay;
    if (const uint32_t divisor = std::gcd(num, den); divisor > 1) {
        num /= divisor;
        den /= divisor;
    }
    return {.den = den, .num = num};
}

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

/// Deinterlace verdict for a decoded frame; single source of truth.
///
/// @p streamInterlaced is a positive sequence/container hint (MPEG-2 progressive_sequence, or the
/// container field_order) that forces deinterlace when the per-frame flag is unreliable: FFmpeg
/// clears AV_FRAME_FLAG_INTERLACED on a progressive-coded I-picture even inside an interlaced
/// MPEG-2 sequence -- exactly the picture the lazy first-frame filter build samples. It only ever
/// forces deinterlace ON; it is left false for H.264/HEVC, whose per-frame flag is reliable.
[[nodiscard]] inline auto FrameNeedsDeinterlace(const AVFrame *frame, bool streamInterlaced) noexcept -> bool {
    if (frame == nullptr) [[unlikely]] {
        return false;
    }
    return streamInterlaced || (frame->flags & AV_FRAME_FLAG_INTERLACED) != 0;
}

// ============================================================================
// === RAII CUSTOM DELETERS ===
// ============================================================================

// --- FFmpeg Deleters ---
// These free functions take a double pointer and null it, hence the address-of below (av_parser_close excepted).

/// Deleter for AVBufferRef (av_buffer_unref)
struct FreeAVBufferRef {
    /// Drops one reference; the buffer dies with the last one.
    auto operator()(AVBufferRef *ref) const noexcept -> void { av_buffer_unref(&ref); }
};

/// Deleter for AVCodecContext (avcodec_free_context)
struct FreeAVCodecContext {
    /// Closes the codec before freeing the context.
    auto operator()(AVCodecContext *ctx) const noexcept -> void { avcodec_free_context(&ctx); }
};

/// Deleter for AVCodecParserContext (av_parser_close)
struct FreeAVCodecParserContext {
    /// The exception: av_parser_close() takes the pointer itself.
    auto operator()(AVCodecParserContext *ctx) const noexcept -> void { av_parser_close(ctx); }
};

/// Deleter for AVFilterGraph (avfilter_graph_free)
struct FreeAVFilterGraph {
    /// Frees the graph and every filter in it.
    auto operator()(AVFilterGraph *graph) const noexcept -> void { avfilter_graph_free(&graph); }
};

/// Deleter for AVFormatContext (avformat_close_input). Use with std::unique_ptr<AVFormatContext, FreeAVFormatContext>
/// for the libavformat-based mediaplayer path. avformat_close_input is the canonical pairing for
/// avformat_open_input; it nulls the local pointer too, hence the address-of.
struct FreeAVFormatContext {
    /// Closes the input before freeing the context.
    auto operator()(AVFormatContext *ctx) const noexcept -> void { avformat_close_input(&ctx); }
};

/// Deleter for AVFrame (av_frame_free)
struct FreeAVFrame {
    /// Unreferences the frame's buffers, then frees it.
    auto operator()(AVFrame *frame) const noexcept -> void { av_frame_free(&frame); }
};

/// Deleter for AVPacket (av_packet_free)
struct FreeAVPacket {
    /// Unreferences the payload, then frees the packet.
    auto operator()(AVPacket *pkt) const noexcept -> void { av_packet_free(&pkt); }
};

// --- DRM Deleters ---
// Unlike FFmpeg's, these take the pointer itself -- drmFreeDevice() being the one exception.

/// Deleter for drmModeConnector (drmModeFreeConnector)
struct FreeDrmConnector {
    /// Also frees its mode and property arrays -- anything pointing into them dangles.
    auto operator()(drmModeConnector *conn) const noexcept -> void { drmModeFreeConnector(conn); }
};

/// Deleter for drmDevice (drmFreeDevice)
struct FreeDrmDevice {
    /// The exception: takes a double pointer.
    auto operator()(drmDevice *dev) const noexcept -> void { drmFreeDevice(&dev); }
};

/// Deleter for drmModeObjectProperties (drmModeFreeObjectProperties)
struct FreeDrmObjectProperties {
    /// Frees the id/value arrays from drmModeObjectGetProperties().
    auto operator()(drmModeObjectProperties *props) const noexcept -> void { drmModeFreeObjectProperties(props); }
};

/// Deleter for drmModePlaneRes (drmModeFreePlaneResources)
struct FreeDrmPlaneResources {
    /// Frees the plane id list from drmModeGetPlaneResources().
    auto operator()(drmModePlaneRes *res) const noexcept -> void { drmModeFreePlaneResources(res); }
};

/// Deleter for drmModePropertyRes (drmModeFreeProperty)
struct FreeDrmProperty {
    /// Frees one property descriptor, including its enum-name table.
    auto operator()(drmModePropertyRes *prop) const noexcept -> void { drmModeFreeProperty(prop); }
};

/// Deleter for drmModeRes (drmModeFreeResources)
struct FreeDrmResources {
    /// Frees the connector/encoder/CRTC id lists from drmModeGetResources().
    auto operator()(drmModeRes *res) const noexcept -> void { drmModeFreeResources(res); }
};

#endif // VDR_VAAPIVIDEO_COMMON_H
