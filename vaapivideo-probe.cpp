// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file vaapivideo-probe.cpp
 * @brief Standalone VAAPI capability prober for vdr-plugin-vaapivideo.
 *
 * Mirrors the probe matrix of the plugin runtime (src/caps.cpp ProbeGpuCaps +
 * src/stream.h STREAM_VIDEO_BACKEND_TABLE) so an operator can predict which
 * codec/profile/bit-depth combinations will hardware-decode on a given GPU.
 *
 * Probed codecs:
 *   MPEG-2  (Simple, Main)                          -- SD broadcast
 *   H.264   (Constrained Baseline, Main, High, High 10) -- HD broadcast
 *   HEVC    (Main 8-bit, Main10 10-bit)             -- UHD + HDR
 *   AV1     (Profile 0, 8-bit + 10-bit)             -- streaming / mediaplayer
 *   VP9     (Profile 0, Profile 2)                  -- streaming / mediaplayer
 *   VVC     (H.266 Main 10, 8-bit + 10-bit)         -- next-gen broadcast
 *
 * Each profile is tested against the exact surface format it requires
 * (YUV420 for 8-bit, YUV420_10 for 10-bit). VPP filters, HDR tone mapping
 * (both HDR->SDR and the H2H path that re-masters HLG to HDR10), and
 * deinterlacing modes are also reported. Per connector, the sink EDID is
 * parsed for its advertised HDR10 (PQ) / HLG / BT.2020 / HDR10+ support,
 * Dolby Vision VSVDB -- the display half of the HDR decision (DV is reported
 * for information only; VAAPI cannot decode it) -- and the advertised VRR
 * ranges (HDMI 2.1 game-VRR, AMD FreeSync, VESA range limits), alongside the
 * kernel's vrr_capable / VRR_ENABLED properties.
 *
 * Usage:   ./vaapivideo-probe [/dev/dri/cardN]   (default: /dev/dri/card0)
 * Compile: g++ -std=c++20 $(pkg-config --cflags --libs libdrm libva-drm) -o vaapivideo-probe vaapivideo-probe.cpp
 * Lint:    clang-tidy vaapivideo-probe.cpp -- -std=c++20 $(pkg-config --cflags libdrm libva-drm)
 */

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <va/va.h>
#include <va/va_drm.h>
#include <va/va_vpp.h>

#include <libdrm/drm.h>
#include <libdrm/drm_fourcc.h>
#include <libdrm/drm_mode.h>
#include <libdrm/i915_drm.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

// Small enough to avoid wasting VRAM; large enough that no driver rejects it as
// below its minimum surface alignment (typically 16 px).
constexpr int PROBE_SURFACE_SIZE = 64;

namespace {
struct DrmDeviceDeleter {
    auto operator()(drmDevice *dev) const noexcept -> void { drmFreeDevice(&dev); }
};
struct DrmBlobDeleter {
    auto operator()(drmModePropertyBlobRes *blob) const noexcept -> void { drmModeFreePropertyBlob(blob); }
};
} // namespace

// Every (profile, required RT surface format) combination worth reporting. The
// pipeline is 4:2:0 only; non-4:2:0 profiles (VP9 Profile 1/3, AV1 Profile 1/2,
// HEVC range extensions, JPEG, ...) are out of scope and not listed. The
// plugin-consumed subset is determined by ProbeDecodeProfiles' switch below
// (mirrors GpuCaps in src/caps.cpp + STREAM_VIDEO_BACKEND_TABLE in src/stream.h).
//
// Naming convention: where the same VAProfile shows up at two bit-depths
// (AV1 Profile 0, VVC Main 10), the rows share the profile name and the
// right column ("8-bit 4:2:0" vs "10-bit 4:2:0") disambiguates them. Where
// the bit-depth is part of the official VAProfile name (H.264 High 10, HEVC
// Main 10/12), it stays in the name.
namespace {
struct DecodeProbe {
    VAProfile profile;
    unsigned int rtFormat;
    const char *name;
};
} // namespace
constexpr DecodeProbe DECODE_PROBES[] = {
    // MPEG-2
    {.profile = VAProfileMPEG2Simple, .rtFormat = VA_RT_FORMAT_YUV420, .name = "MPEG-2 Simple"},
    {.profile = VAProfileMPEG2Main, .rtFormat = VA_RT_FORMAT_YUV420, .name = "MPEG-2 Main"},
    // H.264
    {.profile = VAProfileH264ConstrainedBaseline,
     .rtFormat = VA_RT_FORMAT_YUV420,
     .name = "H.264 Constrained Baseline"},
    {.profile = VAProfileH264Main, .rtFormat = VA_RT_FORMAT_YUV420, .name = "H.264 Main"},
    {.profile = VAProfileH264High, .rtFormat = VA_RT_FORMAT_YUV420, .name = "H.264 High"},
    {.profile = VAProfileH264High10, .rtFormat = VA_RT_FORMAT_YUV420_10, .name = "H.264 High 10"},
    // HEVC -- Main 12 is informational only (12-bit not in NV12/P010 pipeline).
    {.profile = VAProfileHEVCMain, .rtFormat = VA_RT_FORMAT_YUV420, .name = "HEVC Main"},
    {.profile = VAProfileHEVCMain10, .rtFormat = VA_RT_FORMAT_YUV420_10, .name = "HEVC Main 10"},
    {.profile = VAProfileHEVCMain12, .rtFormat = VA_RT_FORMAT_YUV420_12, .name = "HEVC Main 12"},
    // AV1 -- Profile 1 / Profile 2 omitted: non-4:2:0 chroma.
    {.profile = VAProfileAV1Profile0, .rtFormat = VA_RT_FORMAT_YUV420, .name = "AV1 Profile 0"},
    {.profile = VAProfileAV1Profile0, .rtFormat = VA_RT_FORMAT_YUV420_10, .name = "AV1 Profile 0"},
    // VP9 -- Profile 1/3 are non-4:2:0 per spec; omitted.
    {.profile = VAProfileVP9Profile0, .rtFormat = VA_RT_FORMAT_YUV420, .name = "VP9 Profile 0"},
    {.profile = VAProfileVP9Profile2, .rtFormat = VA_RT_FORMAT_YUV420_10, .name = "VP9 Profile 2"},
    // VVC / H.266 -- Main 10 profile decodes both 8-bit and 10-bit per spec.
    {.profile = VAProfileVVCMain10, .rtFormat = VA_RT_FORMAT_YUV420, .name = "VVC / H.266 Main 10"},
    {.profile = VAProfileVVCMain10, .rtFormat = VA_RT_FORMAT_YUV420_10, .name = "VVC / H.266 Main 10"},
};

// General VPP filter types. Deinterlacing and HDR tone mapping are probed
// separately because they require capability structs beyond a simple yes/no.
constexpr struct {
    VAProcFilterType type;
    const char *name;
} kVppFilterTypes[] = {
    {.type = VAProcFilterNoiseReduction, .name = "Noise Reduction  (Denoise)"},
    {.type = VAProcFilterSharpening, .name = "Sharpening"},
    {.type = VAProcFilterColorBalance, .name = "Color Balance"},
    {.type = VAProcFilterSkinToneEnhancement, .name = "Skin Tone Enhancement"},
    {.type = VAProcFilterTotalColorCorrection, .name = "Total Color Correction"},
    {.type = VAProcFilterHVSNoiseReduction, .name = "HVS Noise Reduction"},
};

// Listed highest-quality first so the first supported entry is the preferred choice.
constexpr struct {
    VAProcDeinterlacingType type;
    const char *name;
} kDeintAlgorithms[] = {
    {.type = VAProcDeinterlacingMotionCompensated, .name = "Motion Compensated"},
    {.type = VAProcDeinterlacingMotionAdaptive, .name = "Motion Adaptive"},
    {.type = VAProcDeinterlacingWeave, .name = "Weave"},
    {.type = VAProcDeinterlacingBob, .name = "Bob"},
};

namespace {

[[nodiscard]] auto SupportsVldEntrypoint(VADisplay display, VAProfile profile) -> bool {
    const int maxEntrypoints = vaMaxNumEntrypoints(display);
    if (maxEntrypoints <= 0) {
        return false;
    }

    std::vector<VAEntrypoint> entrypoints(static_cast<size_t>(maxEntrypoints));
    int entrypointCount = 0;
    if (vaQueryConfigEntrypoints(display, profile, entrypoints.data(), &entrypointCount) != VA_STATUS_SUCCESS) {
        return false;
    }

    // Some drivers report a count larger than the allocated buffer; clamp defensively.
    const auto validCount =
        (entrypointCount > 0) ? std::min(static_cast<size_t>(entrypointCount), entrypoints.size()) : size_t{0};
    const std::span<const VAEntrypoint> valid{entrypoints.data(), validCount};
    return std::ranges::find(valid, VAEntrypointVLD) != valid.end();
}

[[nodiscard]] auto SupportsRtFormat(VADisplay display, VAProfile profile, unsigned int rtFormat) -> bool {
    VAConfigAttrib attrib{};
    attrib.type = VAConfigAttribRTFormat;
    if (vaGetConfigAttributes(display, profile, VAEntrypointVLD, &attrib, 1) != VA_STATUS_SUCCESS) {
        return false;
    }
    return (attrib.value & rtFormat) != 0;
}

[[nodiscard]] auto ResolveRenderNode(int drmFd) -> std::string {
    ::drmDevicePtr rawDev = nullptr;
    if (drmGetDevice2(drmFd, 0, &rawDev) != 0 || !rawDev) {
        return {};
    }
    std::unique_ptr<::drmDevice, DrmDeviceDeleter> dev{rawDev};

    if (!(dev->available_nodes & (1 << DRM_NODE_RENDER)) || !dev->nodes[DRM_NODE_RENDER]) {
        return {};
    }
    return dev->nodes[DRM_NODE_RENDER];
}

auto PrintCapability(const char *label, bool supported) -> void {
    std::printf("  %-44s %s\n", label, supported ? "\033[32myes\033[0m" : "\033[31mno\033[0m");
}

// Human-readable label for each VA_RT_FORMAT class, used to annotate probe
// rows with the surface layout required by each codec/profile.
[[nodiscard]] auto RtFormatLabel(unsigned int rtFormat) -> const char * {
    switch (rtFormat) {
        case VA_RT_FORMAT_YUV420:
            return "8-bit  4:2:0";
        case VA_RT_FORMAT_YUV420_10:
            return "10-bit 4:2:0";
        case VA_RT_FORMAT_YUV420_12:
            return "12-bit 4:2:0";
        default:
            return "?";
    }
}

// Build a fixed-width "Name  (N-bit 4:X:X)" label for one probe row.
// Thread-local buffer avoids heap allocation inside the probe loop.
[[nodiscard]] auto FormatProbeLabel(const DecodeProbe &row) -> const char * {
    static thread_local std::array<char, 64> buf{};
    (void)std::snprintf(buf.data(), buf.size(), "%-29s (%s)", row.name, RtFormatLabel(row.rtFormat));
    return buf.data();
}

[[nodiscard]] auto CanCreateSurface(VADisplay display, unsigned int rtFormat) -> bool {
    VASurfaceID surface = VA_INVALID_SURFACE;
    if (vaCreateSurfaces(display, rtFormat, PROBE_SURFACE_SIZE, PROBE_SURFACE_SIZE, &surface, 1, nullptr, 0) !=
        VA_STATUS_SUCCESS) {
        return false;
    }
    vaDestroySurfaces(display, &surface, 1);
    return true;
}

// Probe a specific FourCC, not just an RT_FORMAT class.  VA_RT_FORMAT_YUV420
// covers any 8-bit 4:2:0 layout (NV12, YV12, IYUV, ...); a class-only probe can
// succeed while the driver allocates a FourCC the pipeline cannot consume.
// The plugin's VPP (scale_vaapi) output and the DRM video plane both require
// NV12 for 8-bit and P010 for 10-bit.  Some older Intel drivers accept the
// class but reject the explicit FourCC -- this catches that regression.
[[nodiscard]] auto CanCreateSurfaceFourcc(VADisplay display, unsigned int rtFormat, uint32_t fourcc) -> bool {
    // NOLINTBEGIN(bugprone-invalid-enum-default-initialization, cppcoreguidelines-pro-type-union-access)
    // VAGenericValue has no zero-valued enumerator, so brace-init triggers a
    // clang-tidy warning; the union access is mandated by the libva ABI.
    VASurfaceAttrib attrib{};
    attrib.type = VASurfaceAttribPixelFormat;
    attrib.flags = VA_SURFACE_ATTRIB_SETTABLE;
    attrib.value.type = VAGenericValueTypeInteger;
    attrib.value.value.i = static_cast<int>(fourcc);
    // NOLINTEND(bugprone-invalid-enum-default-initialization, cppcoreguidelines-pro-type-union-access)

    VASurfaceID surface = VA_INVALID_SURFACE;
    if (vaCreateSurfaces(display, rtFormat, PROBE_SURFACE_SIZE, PROBE_SURFACE_SIZE, &surface, 1, &attrib, 1) !=
        VA_STATUS_SUCCESS) {
        return false;
    }
    vaDestroySurfaces(display, &surface, 1);
    return true;
}

// Mirrors GpuCaps (src/caps.h): one flag per codec/bit-depth the plugin uses.
// Populated by ProbeDecodeProfiles; consumed by PrintColorConversions.
struct DecodeSupport {
    bool mpeg2 = false;       ///< MPEG-2 Simple or Main (8-bit)
    bool h264 = false;        ///< H.264 CBP / Main / High (8-bit)
    bool h264High10 = false;  ///< H.264 High 10 (10-bit)
    bool hevc = false;        ///< HEVC Main / SccMain (8-bit 4:2:0)
    bool hevcMain10 = false;  ///< HEVC Main 10 / SccMain 10 (10-bit 4:2:0)
    bool av1 = false;         ///< AV1 Profile 0 (8-bit)
    bool av1Main10 = false;   ///< AV1 Profile 0 (10-bit)
    bool vp9 = false;         ///< VP9 Profile 0 (8-bit)
    bool vp9Profile2 = false; ///< VP9 Profile 2 (10-bit)
    bool vvc = false;         ///< VVC Main 10 at YUV420 (8-bit)
    bool vvcMain10 = false;   ///< VVC Main 10 at YUV420_10 (10-bit)
};

[[nodiscard]] auto ProbeDecodeProfiles(VADisplay display) -> DecodeSupport {
    std::printf("\n--- Hardware Decode (VLD + required RT format) ---\n");

    const int maxProfiles = vaMaxNumProfiles(display);
    if (maxProfiles <= 0) {
        for (const auto &row : DECODE_PROBES) {
            PrintCapability(row.name, false);
        }
        return {};
    }

    std::vector<VAProfile> profileBuf(static_cast<size_t>(maxProfiles));
    int profileCount = 0;
    const bool queryOk = vaQueryConfigProfiles(display, profileBuf.data(), &profileCount) == VA_STATUS_SUCCESS;
    // Some drivers report a count larger than the allocated buffer; clamp defensively.
    const auto validCount =
        (queryOk && profileCount > 0) ? std::min(static_cast<size_t>(profileCount), profileBuf.size()) : size_t{0};
    const std::span<const VAProfile> profiles{profileBuf.data(), validCount};

    DecodeSupport result;
    for (const auto &row : DECODE_PROBES) {
        const bool hasProfile = std::ranges::find(profiles, row.profile) != profiles.end();
        const bool hasVld = hasProfile && SupportsVldEntrypoint(display, row.profile);
        const bool hasRt = hasVld && SupportsRtFormat(display, row.profile, row.rtFormat);

        PrintCapability(FormatProbeLabel(row), hasRt);

        if (hasRt) {
            // Mirrors GpuCaps flag-setting in ProbeGpuCaps (src/caps.cpp).
            // Profiles not in this switch (HEVC 12-bit) are informational only.
            switch (row.profile) {
                case VAProfileMPEG2Simple:
                case VAProfileMPEG2Main:
                    result.mpeg2 = true;
                    break;
                case VAProfileH264ConstrainedBaseline:
                case VAProfileH264Main:
                case VAProfileH264High:
                    result.h264 = true;
                    break;
                case VAProfileH264High10:
                    result.h264High10 = true;
                    // A High10-capable VLD typically also decodes 8-bit streams, but only
                    // when the driver advertises YUV420 on the High10 profile itself.  This
                    // keeps h264 in sync with GpuCaps::hwH264 on drivers that list High10
                    // without separately listing Main/High (mirrors ProbeGpuCaps).
                    if (SupportsRtFormat(display, row.profile, VA_RT_FORMAT_YUV420)) {
                        result.h264 = true;
                    }
                    break;
                case VAProfileHEVCMain:
                    result.hevc = true;
                    break;
                case VAProfileHEVCMain10:
                    result.hevcMain10 = true;
                    // Same fallback as H264High10: Main10 VLDs often handle 8-bit too.
                    if (SupportsRtFormat(display, row.profile, VA_RT_FORMAT_YUV420)) {
                        result.hevc = true;
                    }
                    break;
                case VAProfileAV1Profile0:
                    if (row.rtFormat == VA_RT_FORMAT_YUV420_10) {
                        result.av1Main10 = true;
                    } else {
                        result.av1 = true;
                    }
                    break;
                case VAProfileVP9Profile0:
                    result.vp9 = true;
                    break;
                case VAProfileVP9Profile2:
                    result.vp9Profile2 = true;
                    break;
                case VAProfileVVCMain10:
                    if (row.rtFormat == VA_RT_FORMAT_YUV420_10) {
                        result.vvcMain10 = true;
                    } else {
                        result.vvc = true;
                    }
                    break;
                default:
                    break;
            }
        } else if (hasProfile && !hasVld) {
            std::printf("  %-44s (profile present but no VLD entrypoint)\n", "");
        }
        // hasProfile && hasVld && !hasRt: the top-level "no" is sufficient;
        // a second diagnostic line would add noise without actionable info.
    }
    return result;
}

// Per-direction tone-mapping support for the directions a broadcast pipeline uses
// (SDR->HDR is deliberately excluded: no broadcast source requires it).  Not printed
// directly; the results are surfaced through PrintColorConversions so the operator
// sees them alongside the codec/surface context that makes them actionable.  The VPP
// filter list already shows whether VAProcFilterHighDynamicRangeToneMapping is
// advertised at all.
struct HdrToneMapCaps {
    bool toHdr10{}; ///< VA_TONE_MAPPING_HDR_TO_HDR: HDR input (PQ or HLG) re-mastered to HDR10 output
    bool toSdr{};   ///< VA_TONE_MAPPING_HDR_TO_SDR: HDR input rendered down to SDR
};

[[nodiscard]] auto ProbeHdrToneMapping(VADisplay display, VAContextID vppContext,
                                       std::span<const VAProcFilterType> filters) -> HdrToneMapCaps {
    HdrToneMapCaps result;
    if (std::ranges::find(filters, VAProcFilterHighDynamicRangeToneMapping) == filters.end()) {
        return result;
    }

    VAProcFilterCapHighDynamicRange hdrCaps[VAProcHighDynamicRangeMetadataTypeCount];
    auto hdrCapCount = static_cast<unsigned int>(VAProcHighDynamicRangeMetadataTypeCount);
    if (vaQueryVideoProcFilterCaps(display, vppContext, VAProcFilterHighDynamicRangeToneMapping, hdrCaps,
                                   &hdrCapCount) != VA_STATUS_SUCCESS) {
        return result;
    }

    // Only HDR10-metadata rows count: both directions need the driver to understand
    // ST 2086 / CTA-861.3 metadata (mastering data on the input for HDR->SDR, metadata
    // attach on the output for HDR->HDR10).
    for (unsigned int i = 0; i < hdrCapCount; ++i) {
        if (hdrCaps[i].metadata_type != VAProcHighDynamicRangeMetadataHDR10) {
            continue;
        }
        result.toHdr10 = result.toHdr10 || (hdrCaps[i].caps_flag & VA_TONE_MAPPING_HDR_TO_HDR) != 0;
        result.toSdr = result.toSdr || (hdrCaps[i].caps_flag & VA_TONE_MAPPING_HDR_TO_SDR) != 0;
    }
    return result;
}

auto ProbeDeinterlacing(VADisplay display, VAContextID vppContext) -> void {
    std::printf("\n--- Deinterlacing Algorithms ---\n");

    VAProcFilterCapDeinterlacing deintCaps[VAProcDeinterlacingCount];
    auto deintCapCount = static_cast<unsigned int>(VAProcDeinterlacingCount);
    if (vaQueryVideoProcFilterCaps(display, vppContext, VAProcFilterDeinterlacing, deintCaps, &deintCapCount) !=
        VA_STATUS_SUCCESS) {
        deintCapCount = 0;
    }

    const std::span<const VAProcFilterCapDeinterlacing> caps{deintCaps, static_cast<size_t>(deintCapCount)};
    for (const auto &[deintType, deintName] : kDeintAlgorithms) {
        const bool supported =
            std::ranges::any_of(caps, [deintType](const auto &c) -> bool { return c.type == deintType; });
        PrintCapability(deintName, supported);
    }
}

auto PrintColorConversions(const DecodeSupport &dec, bool hasP010, bool hasNV12, const HdrToneMapCaps &toneMap)
    -> void {
    std::printf("\n--- Color Conversion Paths ---\n");

    const bool any10Bit = dec.hevcMain10 || dec.h264High10 || dec.av1Main10;
    // Both surfaces are needed: P010 for the decoded frame, NV12 for VPP output.
    const bool tenToEight = any10Bit && hasP010 && hasNV12;

    // BT.601 SD broadcast upscaled to BT.709 HD/UHD display via VPP.
    PrintCapability("BT.601 -> BT.709   (MPEG-2 SD)", dec.mpeg2);
    // BT.709 8-bit: passthrough, no colorspace conversion required.
    PrintCapability("BT.709 passthrough  (8-bit HD)", dec.h264 || dec.hevc || dec.av1);
    // BT.2020 10-bit HDR decode surface allocation.
    PrintCapability("BT.2020/P010      (HDR decode)", any10Bit && hasP010);
    // VPP P010->NV12 downconvert for SDR display.
    PrintCapability("P010 -> NV12       (10->8 bit)", tenToEight);
    // HLG is backward-compatible with BT.709 -- display handles it without VPP TM.
    PrintCapability("HLG -> SDR    (no TM required)", any10Bit);
    // PQ/HDR10: requires VAProcFilterHighDynamicRangeToneMapping with HDR->SDR.
    PrintCapability("PQ/HDR10 -> SDR     (tone map)", any10Bit && toneMap.toSdr);
    // HLG for PQ-only sinks: VPP H2H re-masters ARIB STD-B67 input to ST 2084 output
    // with HDR10 metadata attached. Both sides ride P010 (decode surface in, HDR out).
    PrintCapability("HLG -> HDR10    (H2H tone map)", any10Bit && hasP010 && toneMap.toHdr10);
}

} // namespace

// ============================================================================
// === DRM PROBING ===
// ============================================================================

namespace {

[[nodiscard]] auto ConnectorTypeName(uint32_t t) -> const char * {
    switch (t) {
        case DRM_MODE_CONNECTOR_HDMIA:
            return "HDMI-A";
        case DRM_MODE_CONNECTOR_HDMIB:
            return "HDMI-B";
        case DRM_MODE_CONNECTOR_DisplayPort:
            return "DisplayPort";
        case DRM_MODE_CONNECTOR_eDP:
            return "eDP";
        case DRM_MODE_CONNECTOR_DVII:
            return "DVI-I";
        case DRM_MODE_CONNECTOR_DVID:
            return "DVI-D";
        case DRM_MODE_CONNECTOR_DVIA:
            return "DVI-A";
        case DRM_MODE_CONNECTOR_VGA:
            return "VGA";
        case DRM_MODE_CONNECTOR_LVDS:
            return "LVDS";
        case DRM_MODE_CONNECTOR_Composite:
            return "Composite";
        case DRM_MODE_CONNECTOR_SVIDEO:
            return "S-Video";
        case DRM_MODE_CONNECTOR_Component:
            return "Component";
        case DRM_MODE_CONNECTOR_DSI:
            return "DSI";
        default:
            return "?";
    }
}

[[nodiscard]] auto PlaneTypeName(uint64_t t) -> const char * {
    switch (t) {
        case DRM_PLANE_TYPE_PRIMARY:
            return "PRIMARY";
        case DRM_PLANE_TYPE_OVERLAY:
            return "OVERLAY";
        case DRM_PLANE_TYPE_CURSOR:
            return "CURSOR";
        default:
            return "?";
    }
}

auto PrintDrmCap(int fd, const char *name, uint64_t cap) -> void {
    uint64_t value = 0;
    const bool hasCap = drmGetCap(fd, cap, &value) == 0;
    // For boolean / bitmask caps a successful query returning 0 means "not enabled".
    // Treat (hasCap && value == 0) as a "no" so the report doesn't lie about disabled caps.
    const bool enabled = hasCap && value != 0;
    const std::string detail = hasCap ? " = " + std::to_string(value) : "";
    std::printf("  %-44s %s%s\n", name, enabled ? "\033[32myes\033[0m" : "\033[31mno\033[0m", detail.c_str());
}

auto PrintDrmClientCap(int fd, const char *name, uint64_t cap) -> void {
    // Probe is non-destructive: setting these caps for the duration of the process is fine,
    // the runtime sets the same caps itself.
    const bool ok = drmSetClientCap(fd, cap, 1) == 0;
    std::printf("  %-44s %s\n", name, ok ? "\033[32myes\033[0m" : "\033[31mno\033[0m");
}

// Friendly vendor name for the common GPU PCI vendor IDs.
[[nodiscard]] auto PciVendorName(uint16_t vendorId) -> const char * {
    switch (vendorId) {
        case 0x8086:
            return "Intel";
        case 0x1002:
            return "AMD/ATI";
        case 0x10de:
            return "NVIDIA";
        case 0x1af4:
            return "Virtio";
        default:
            return "unknown vendor";
    }
}

// First parsable unsigned value among candidate sysfs paths. Kernels moved the i915
// frequency knobs from card<N>/gt_*_freq_mhz to card<N>/gt/gt0/rps_*_freq_mhz, so both
// layouts are tried in order. from_chars keeps the parse locale-independent.
[[nodiscard]] auto ReadSysfsUint(std::span<const std::string> paths) -> std::optional<uint64_t> {
    for (const auto &path : paths) {
        std::ifstream file(path);
        std::string line;
        if (!std::getline(file, line)) {
            continue;
        }
        const char *const base = line.c_str();
        uint64_t value = 0;
        if (const auto parsed = std::from_chars(base, base + line.size(), value);
            parsed.ec == std::errc{} && parsed.ptr != base) {
            return value;
        }
    }
    return std::nullopt;
}

// Resolve the sysfs class directory (/sys/class/drm/cardN) from a DRM device's libdrm
// primary-node path -- robust across primary/render nodes and symlinks, unlike deriving
// it from the opened fd's minor number. Empty if no primary node is advertised.
[[nodiscard]] auto DrmSysfsCardDir(const ::drmDevice &dev) -> std::string {
    if ((dev.available_nodes & (1 << DRM_NODE_PRIMARY)) == 0 || dev.nodes[DRM_NODE_PRIMARY] == nullptr) {
        return {};
    }
    const std::string nodePath{dev.nodes[DRM_NODE_PRIMARY]};
    const size_t slashPos = nodePath.find_last_of('/');
    return "/sys/class/drm/" + nodePath.substr(slashPos == std::string::npos ? 0 : slashPos + 1);
}

// One i915 GETPARAM integer (e.g. EU / subslice topology). nullopt on a non-i915
// driver or an unsupported parameter.
[[nodiscard]] auto I915GetParam(int fd, int param) -> std::optional<int> {
    int value = 0;
    drm_i915_getparam_t gp{};
    gp.param = param;
    gp.value = &value;
    if (drmIoctl(fd, DRM_IOCTL_I915_GETPARAM, &gp) != 0) {
        return std::nullopt;
    }
    return value;
}

// Parse a leading base-16 id from a null-terminated line span [first, last). On success
// returns the value and the past-the-digits pointer; nullopt if no hex digits lead.
[[nodiscard]] auto ParseHexPrefix(const char *first, const char *last)
    -> std::optional<std::pair<unsigned, const char *>> {
    unsigned value = 0;
    const auto [ptr, ec] = std::from_chars(first, last, value, 16);
    // pci.ids vendor/device ids are exactly four hex digits; rejecting other lengths stops a
    // class line like "C 00" from being read as vendor 0x000c.
    if (ec != std::errc{} || ptr - first != 4) {
        return std::nullopt;
    }
    return std::make_pair(value, ptr);
}

// Marketing name for a PCI vendor:device pair from the system pci.ids database (the
// source lspci uses), e.g. 1002:1681 -> "Rembrandt [Radeon 680M]". Empty if the file
// or the entry is absent. Layout: vendor lines start in column 0, device lines are
// indented one tab, both "<hex-id>  <name>"; entries are sorted by id.
[[nodiscard]] auto LookupPciDeviceName(uint16_t vendorId, uint16_t deviceId) -> std::string {
    static constexpr std::array<const char *, 3> kPciIdsPaths{"/usr/share/hwdata/pci.ids", "/usr/share/misc/pci.ids",
                                                              "/usr/share/pci.ids"};

    for (const char *path : kPciIdsPaths) {
        std::ifstream file(path);
        if (!file) {
            continue;
        }
        bool inVendor = false;
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line.starts_with('#')) {
                continue;
            }
            const char *const base = line.c_str();
            const char *const end = base + line.size();
            if (!line.starts_with('\t')) { // vendor (or class) line
                if (inVendor) {
                    break; // sorted file: left our vendor's block without a match -- try the next database
                }
                const auto parsed = ParseHexPrefix(base, end);
                inVendor = parsed && parsed->first == vendorId;
            } else if (inVendor && !line.starts_with("\t\t")) { // device line (single tab)
                const auto parsed = ParseHexPrefix(base + 1, end);
                if (parsed && parsed->first == deviceId) {
                    const auto namePos = line.find_first_not_of(" \t", static_cast<size_t>(parsed->second - base));
                    return namePos == std::string::npos ? std::string{} : line.substr(namePos);
                }
            }
        }
        // No match in this database; it may be stale or partial, so fall through to the next path.
    }
    return {};
}

// amdgpu engine-clock (sclk) range from the DPM state table device/pp_dpm_sclk, whose
// lines read "<idx>: <freq>Mhz [*]". Returns (min, max) in MHz, nullopt if unreadable.
[[nodiscard]] auto ReadAmdgpuSclkRange(const std::string &cardDir) -> std::optional<std::pair<uint64_t, uint64_t>> {
    std::ifstream file(cardDir + "/device/pp_dpm_sclk");
    if (!file) {
        return std::nullopt;
    }
    uint64_t lo = std::numeric_limits<uint64_t>::max();
    uint64_t hi = 0;
    std::string line;
    while (std::getline(file, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        const auto firstDigit = line.find_first_of("0123456789", colon + 1);
        if (firstDigit == std::string::npos) {
            continue;
        }
        const char *const base = line.c_str();
        uint64_t mhz = 0; // from_chars stops at the 'M' of "Mhz", so the unit's case is irrelevant
        const auto parsed = std::from_chars(base + firstDigit, base + line.size(), mhz);
        if (parsed.ec == std::errc{} && mhz > 0) {
            lo = std::min(lo, mhz);
            hi = std::max(hi, mhz);
        }
    }
    if (hi == 0) {
        return std::nullopt;
    }
    return std::make_pair(lo, hi);
}

// PCI identity, execution-unit topology, and GPU clock range. PCI fields come from
// libdrm and work on any PCI GPU; EU/subslice counts and the clock range are read via
// Intel i915 GETPARAM + sysfs (the plugin's target) and silently skipped elsewhere.
auto PrintGpuHardware(int fd, const char *driverName) -> void {
    std::printf("\n--- GPU Hardware ---\n");

    ::drmDevicePtr rawDev = nullptr;
    std::string cardDir;
    // DRM_DEVICE_GET_PCI_REVISION is required to populate revision_id (left 0xff otherwise).
    if (drmGetDevice2(fd, DRM_DEVICE_GET_PCI_REVISION, &rawDev) == 0 && rawDev != nullptr) {
        std::unique_ptr<::drmDevice, DrmDeviceDeleter> dev{rawDev};
        cardDir = DrmSysfsCardDir(*dev);
        if (dev->bustype == DRM_BUS_PCI) {
            // libdrm exposes PCI identity/bus through C unions keyed by bustype (checked above).
            const auto *pci = dev->deviceinfo.pci; // NOLINT(cppcoreguidelines-pro-type-union-access)
            const auto *bus = dev->businfo.pci;    // NOLINT(cppcoreguidelines-pro-type-union-access)
            if (pci != nullptr) {
                std::printf("  PCI ID:      %04x:%04x (rev 0x%02x)  %s\n", static_cast<unsigned>(pci->vendor_id),
                            static_cast<unsigned>(pci->device_id), static_cast<unsigned>(pci->revision_id),
                            PciVendorName(pci->vendor_id));
                if (const std::string name = LookupPciDeviceName(pci->vendor_id, pci->device_id); !name.empty()) {
                    std::printf("  Device:      %s\n", name.c_str());
                }
                if (pci->subvendor_id != 0 || pci->subdevice_id != 0) {
                    std::printf("  Subsystem:   %04x:%04x\n", static_cast<unsigned>(pci->subvendor_id),
                                static_cast<unsigned>(pci->subdevice_id));
                }
            }
            if (bus != nullptr) {
                std::printf("  PCI bus:     %04x:%02x:%02x.%x\n", static_cast<unsigned>(bus->domain),
                            static_cast<unsigned>(bus->bus), static_cast<unsigned>(bus->dev),
                            static_cast<unsigned>(bus->func));
            }
        } else {
            std::printf("  Bus:         non-PCI DRM bus (%d)\n", dev->bustype);
        }
    } else {
        std::printf("  PCI ID:      (drmGetDevice2 failed)\n");
    }

    if (driverName != nullptr && std::strcmp(driverName, "i915") == 0) {
        if (const auto euTotal = I915GetParam(fd, I915_PARAM_EU_TOTAL); euTotal && *euTotal > 0) {
            if (const auto subslices = I915GetParam(fd, I915_PARAM_SUBSLICE_TOTAL); subslices && *subslices > 0) {
                std::printf("  Render:      %d EUs across %d subslices\n", *euTotal, *subslices);
            } else {
                std::printf("  Render:      %d EUs\n", *euTotal);
            }
        }
    }

    // GPU engine-clock range. i915 exposes RPn(min)/RP0(max) under the card's sysfs node
    // (the path moved across kernels); amdgpu lists DPM states in device/pp_dpm_sclk.
    if (!cardDir.empty()) {
        const std::array<std::string, 2> minPaths{cardDir + "/gt_RPn_freq_mhz", cardDir + "/gt/gt0/rps_RPn_freq_mhz"};
        const std::array<std::string, 2> maxPaths{cardDir + "/gt_RP0_freq_mhz", cardDir + "/gt/gt0/rps_RP0_freq_mhz"};
        if (const auto freqMin = ReadSysfsUint(minPaths), freqMax = ReadSysfsUint(maxPaths); freqMin && freqMax) {
            std::printf("  GPU clock:   %llu - %llu MHz\n", static_cast<unsigned long long>(*freqMin),
                        static_cast<unsigned long long>(*freqMax));
        } else if (const auto amdClock = ReadAmdgpuSclkRange(cardDir); amdClock) {
            std::printf("  GPU clock:   %llu - %llu MHz\n", static_cast<unsigned long long>(amdClock->first),
                        static_cast<unsigned long long>(amdClock->second));
        }
    }
}

auto ProbeDrmDeviceCaps(int fd) -> void {
    std::printf("\n--- DRM Driver ---\n");
    std::string driverName;
    if (drmVersionPtr v = drmGetVersion(fd); v != nullptr) {
        driverName.assign(v->name ? v->name : "", v->name ? static_cast<size_t>(v->name_len) : 0);
        std::printf("  Driver:      %.*s %d.%d.%d (%.*s)\n", v->name_len, v->name ? v->name : "?", v->version_major,
                    v->version_minor, v->version_patchlevel, v->desc_len, v->desc ? v->desc : "");
        drmFreeVersion(v);
    } else {
        std::printf("  Driver: (drmGetVersion failed)\n");
    }

    PrintGpuHardware(fd, driverName.c_str());

    std::printf("\n--- DRM Device Caps ---\n");
    PrintDrmCap(fd, "DUMB_BUFFER          (OSD framebuffer)", DRM_CAP_DUMB_BUFFER);
    PrintDrmCap(fd, "PRIME (export/import VAAPI surfaces)", DRM_CAP_PRIME);
    PrintDrmCap(fd, "ADDFB2_MODIFIERS (tiled framebuffers)", DRM_CAP_ADDFB2_MODIFIERS);
    PrintDrmCap(fd, "CRTC_IN_VBLANK_EVENT (atomic flip ev.)", DRM_CAP_CRTC_IN_VBLANK_EVENT);
    PrintDrmCap(fd, "ASYNC_PAGE_FLIP", DRM_CAP_ASYNC_PAGE_FLIP);
    PrintDrmCap(fd, "TIMESTAMP_MONOTONIC", DRM_CAP_TIMESTAMP_MONOTONIC);

    std::printf("\n--- DRM Client Caps ---\n");
    PrintDrmClientCap(fd, "UNIVERSAL_PLANES (overlay enumeration)", DRM_CLIENT_CAP_UNIVERSAL_PLANES);
    PrintDrmClientCap(fd, "ATOMIC           (atomic modeset)", DRM_CLIENT_CAP_ATOMIC);
    PrintDrmClientCap(fd, "ASPECT_RATIO     (mode aspect)", DRM_CLIENT_CAP_ASPECT_RATIO);
}

struct PropEntry {
    uint64_t value;
    std::string name;
};

[[nodiscard]] auto LoadObjectProps(int fd, uint32_t objectId, uint32_t objectType) -> std::vector<PropEntry> {
    std::vector<PropEntry> out;
    drmModeObjectProperties *props = drmModeObjectGetProperties(fd, objectId, objectType);
    if (props == nullptr) {
        return out;
    }
    for (uint32_t i = 0; i < props->count_props; ++i) {
        drmModePropertyRes *p = drmModeGetProperty(fd, props->props[i]);
        if (p == nullptr) {
            continue;
        }
        out.push_back({.value = props->prop_values[i], .name = p->name});
        drmModeFreeProperty(p);
    }
    drmModeFreeObjectProperties(props);
    return out;
}

[[nodiscard]] auto FindProp(const std::vector<PropEntry> &props, const char *name) -> const PropEntry * {
    for (const auto &p : props) {
        if (p.name == name) {
            return &p;
        }
    }
    return nullptr;
}

// What the sink advertises in its EDID CTA-861 blocks -- the display half of the HDR decision (the
// plugin's auto gate needs PQ + BT.2020 Y'CbCr). Mirrors src/display.cpp plus luminance, the
// dynamic-HDR systems DVB broadcasts use (HDR10+, SL-HDR, Dolby Vision), and the HDMI 2.1
// game-VRR range from the HDMI Forum VSDB.
struct SinkHdrCaps {
    bool eotfSdr{};        ///< HDR Static Metadata EOTF bit 0 (traditional gamma SDR)
    bool eotfHdrGamma{};   ///< EOTF bit 1 (traditional gamma HDR)
    bool hdr10Pq{};        ///< EOTF bit 2 (SMPTE ST 2084 / PQ -> HDR10)
    bool hlg{};            ///< EOTF bit 3 (ARIB STD-B67 / HLG)
    bool bt2020Ycc{};      ///< Colorimetry bit 6 (BT.2020 Y'CbCr)
    bool bt2020Rgb{};      ///< Colorimetry bit 7 (BT.2020 RGB)
    bool hdr10Plus{};      ///< HDR Dynamic Metadata type 4 (SMPTE ST 2094-40)
    bool dynMetaType1{};   ///< HDR Dynamic Metadata type 1 (SMPTE ST 2094-10; Dolby Vision / SL-HDR2 carrier)
    bool slHdr1{};         ///< HDR Dynamic Metadata type 2, bit 4: SL-HDR1 (ETSI TS 103 433-1)
    bool slHdr2{};         ///< HDR Dynamic Metadata type 2, bit 5: SL-HDR2 (ETSI TS 103 433-2)
    bool slHdr3{};         ///< HDR Dynamic Metadata type 2, bit 6: SL-HDR3 (ETSI TS 103 433-3)
    bool dolbyVision{};    ///< Vendor-Specific Video Data Block carries the Dolby OUI (00-D0-46)
    uint8_t dvBlockVer{};  ///< Dolby VSVDB layout version 0..2 -- a block format, not the DV generation
    uint8_t vrrMin{};      ///< HF-VSDB VRRmin in Hz; 0 = HDMI game-VRR not advertised
    uint16_t vrrMax{};     ///< HF-VSDB VRRmax in Hz (10-bit field)
    uint8_t fsMin{};       ///< AMD FreeSync VSDB minimum refresh in Hz; 0 = block absent
    uint8_t fsMax{};       ///< AMD FreeSync VSDB maximum refresh in Hz
    uint16_t rangeMinHz{}; ///< Base-block range-limits descriptor min vertical rate (offset applied)
    uint16_t rangeMaxHz{}; ///< Base-block range-limits descriptor max vertical rate (offset applied)
    bool continuousFreq{}; ///< Base-block feature bit 0: display supports continuous frequencies
    bool hasLuminance{};   ///< The HDR Static Metadata block carried the optional luminance bytes
    uint8_t maxLumaCode{}; ///< Desired content max luminance (CTA code; nits = 50 * 2^(code/32))
};

constexpr size_t EDID_BLOCK_BYTES = 128;
constexpr size_t EDID_EXTENSION_COUNT_OFFSET = 126;
constexpr size_t EDID_CHECKSUM_OFFSET = 127;

/// Desired-content-luminance EDID code -> cd/m^2 (CTA-861-G sec.7.5.13).
[[nodiscard]] auto LuminanceCodeToNits(uint8_t code) -> double { return 50.0 * std::pow(2.0, code / 32.0); }

/// Parse one 128-byte CTA-861 extension block (tag 0x02). OR results into @p caps so multiple
/// extension blocks accumulate. Mirrors src/display.cpp ParseCtaExtension plus the dynamic-HDR
/// blocks and luminance.
auto ParseCtaExtensionForSink(std::span<const uint8_t> ext, SinkHdrCaps &caps) -> void {
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- spans are size-checked
    if (ext.size() < 4 || ext[0] != 0x02) {
        return;
    }
    const size_t dtdOffset = ext[2];
    const size_t dataBlockEnd = (dtdOffset == 0) ? EDID_CHECKSUM_OFFSET : dtdOffset;
    if (dataBlockEnd < 4 || dataBlockEnd > ext.size()) {
        return;
    }
    size_t offset = 4;
    while (offset < dataBlockEnd) {
        const uint8_t header = ext[offset];
        const uint8_t blockTag = header >> 5;
        const uint8_t payloadLen = header & 0x1F;
        // Bound payloads by the checksum byte, not the DTD offset: some real panels declare a DTD
        // offset that lands inside the last data block (seen on an AUO eDP panel whose AMD FreeSync
        // VSDB overruns it by 3 bytes; amdgpu byte-scans around the same bug, edid-decode is equally
        // lenient). Headers still stop at the DTD offset via the loop condition.
        if (offset + 1 + payloadLen > EDID_CHECKSUM_OFFSET) {
            break;
        }
        const std::span<const uint8_t> payload = ext.subspan(offset + 1, payloadLen);
        if (blockTag == 7 && !payload.empty()) { // Extended Tag block; payload[0] = extended tag
            const uint8_t extTag = payload[0];
            if (extTag == 0x06 && payload.size() >= 2) { // HDR Static Metadata (sec.7.5.13)
                caps.eotfSdr = caps.eotfSdr || (payload[1] & (1U << 0)) != 0;
                caps.eotfHdrGamma = caps.eotfHdrGamma || (payload[1] & (1U << 1)) != 0;
                caps.hdr10Pq = caps.hdr10Pq || (payload[1] & (1U << 2)) != 0;
                caps.hlg = caps.hlg || (payload[1] & (1U << 3)) != 0;
                if (payload.size() >= 4) { // optional: [2]=descriptor, [3]=max luminance
                    caps.hasLuminance = true;
                    caps.maxLumaCode = payload[3];
                }
            } else if (extTag == 0x05 && payload.size() >= 2) { // Colorimetry (sec.7.5.6)
                caps.bt2020Ycc = caps.bt2020Ycc || (payload[1] & (1U << 6)) != 0;
                caps.bt2020Rgb = caps.bt2020Rgb || (payload[1] & (1U << 7)) != 0;
            } else if (extTag == 0x07 && payload.size() >= 3) { // HDR Dynamic Metadata (sec.7.5.14)
                // Sub-blocks are (length, type-LSB, type-MSB, data...). Type 1 = SMPTE ST 2094-10,
                // type 2 = ETSI TS 103 433 (SL-HDR), type 4 = SMPTE ST 2094-40 (HDR10+).
                size_t p = 1;
                while (p + 2 < payload.size()) {
                    const uint8_t subLen = payload[p];
                    if (subLen == 0 || p + 1 + subLen > payload.size()) {
                        break;
                    }
                    const uint16_t type =
                        static_cast<uint16_t>(payload[p + 1]) | (static_cast<uint16_t>(payload[p + 2]) << 8U);
                    caps.dynMetaType1 = caps.dynMetaType1 || type == 0x0001;
                    caps.hdr10Plus = caps.hdr10Plus || type == 0x0004;
                    if (type == 0x0002 && subLen > 2) {
                        // First data byte: low nibble = sub-block version, bits 4/5/6 = SL-HDR1/2/3.
                        // The flag bits only exist from version 1 on.
                        const uint8_t flags = payload[p + 3];
                        if ((flags & 0x0FU) >= 1) {
                            caps.slHdr1 = caps.slHdr1 || (flags & (1U << 4)) != 0;
                            caps.slHdr2 = caps.slHdr2 || (flags & (1U << 5)) != 0;
                            caps.slHdr3 = caps.slHdr3 || (flags & (1U << 6)) != 0;
                        }
                    }
                    p += 1 + subLen;
                }
            } else if (extTag == 0x01 && payload.size() >= 4) { // Vendor-Specific Video Data Block
                // IEEE OUI stored LSB-first; Dolby = 00-D0-46 -> bytes 46 D0 00.
                if (payload[1] == 0x46 && payload[2] == 0xD0 && payload[3] == 0x00) {
                    caps.dolbyVision = true;
                    if (payload.size() >= 5) { // first data byte: bits 7..5 = VSVDB layout version
                        caps.dvBlockVer = static_cast<uint8_t>((payload[4] >> 5U) & 0x07U);
                    }
                }
            }
        } else if (blockTag == 3 && payload.size() >= 3) { // Vendor-Specific Data Block
            if (payload[0] == 0xD8 && payload[1] == 0x5D && payload[2] == 0xC4 && payload.size() >= 10) {
                // HDMI Forum VSDB (OUI C4-5D-D8 LSB-first), HDMI 2.1 game-VRR range:
                // payload[8] bits 5:0 = VRRmin, bits 7:6 = VRRmax[9:8], payload[9] = VRRmax[7:0]
                // (offsets validated against edid-decode). Both zero = VRR not supported.
                caps.vrrMin = static_cast<uint8_t>(payload[8] & 0x3FU);
                caps.vrrMax = static_cast<uint16_t>(((payload[8] & 0xC0U) << 2U) | payload[9]);
            } else if (payload[0] == 0x1A && payload[1] == 0x00 && payload[2] == 0x00 && payload.size() >= 7) {
                // AMD FreeSync VSDB (OUI 00-00-1A LSB-first): payload[3] = version, payload[5]/[6] =
                // min/max refresh in Hz -- how FreeSync TVs and DP/eDP panels advertise VRR
                // (offsets validated against edid-decode on a live panel EDID).
                caps.fsMin = payload[5];
                caps.fsMax = payload[6];
            }
        }
        offset += 1 + payloadLen;
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
}

/// Parse the EDID base block for the VRR-relevant bits: the continuous-frequency feature flag and
/// the range-limits descriptor's vertical rate span (VESA EDID 1.4 sec.3.10.3.3). On DP/eDP these
/// are exactly what the kernel folds into the connector's vrr_capable property.
auto ParseBaseBlockForSink(std::span<const uint8_t> base, SinkHdrCaps &caps) -> void {
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- span is size-checked
    if (base.size() < EDID_BLOCK_BYTES) {
        return;
    }
    constexpr size_t kFeatureOffset = 24;   // feature support byte; bit 0 = continuous frequency
    constexpr size_t kDescriptorStart = 54; // four 18-byte descriptors at 54/72/90/108
    constexpr size_t kDescriptorEnd = 126;
    caps.continuousFreq = (base[kFeatureOffset] & 0x01U) != 0;
    for (size_t off = kDescriptorStart; off + 18 <= kDescriptorEnd; off += 18) {
        // Display descriptor (not a timing): bytes 0-2 zero, byte 3 = tag 0xFD (range limits).
        if (base[off] != 0 || base[off + 1] != 0 || base[off + 2] != 0 || base[off + 3] != 0xFD) {
            continue;
        }
        // Byte 4 offset flags: bit 0 = min vfreq +255, bit 1 = max vfreq +255; bytes 5/6 = min/max.
        const uint8_t flags = base[off + 4];
        caps.rangeMinHz = static_cast<uint16_t>(base[off + 5] + (((flags & 0x01U) != 0) ? 255U : 0U));
        caps.rangeMaxHz = static_cast<uint16_t>(base[off + 6] + (((flags & 0x02U) != 0) ? 255U : 0U));
        break;
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
}

/// Read and parse a connector's EDID blob into @p caps. Returns false if no usable EDID was found.
[[nodiscard]] auto ParseConnectorEdid(int fd, const std::vector<PropEntry> &props, SinkHdrCaps &caps) -> bool {
    const PropEntry *edidProp = FindProp(props, "EDID");
    if (edidProp == nullptr || edidProp->value == 0) {
        return false;
    }
    const std::unique_ptr<drmModePropertyBlobRes, DrmBlobDeleter> blob{
        drmModeGetPropertyBlob(fd, static_cast<uint32_t>(edidProp->value))};
    if (!blob || blob->data == nullptr || blob->length < EDID_BLOCK_BYTES) {
        return false;
    }
    const std::span<const uint8_t> edid{static_cast<const uint8_t *>(blob->data), blob->length};
    ParseBaseBlockForSink(edid.first(EDID_BLOCK_BYTES), caps);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- bounded by the size check
    const size_t extCount = edid[EDID_EXTENSION_COUNT_OFFSET];
    for (size_t i = 0; i < extCount; ++i) {
        const size_t extOffset = EDID_BLOCK_BYTES * (i + 1);
        if (extOffset + EDID_BLOCK_BYTES > edid.size()) {
            break;
        }
        ParseCtaExtensionForSink(edid.subspan(extOffset, EDID_BLOCK_BYTES), caps);
    }
    return true;
}

auto PrintSinkHdrCaps(int fd, const std::vector<PropEntry> &props) -> void {
    SinkHdrCaps caps;
    if (!ParseConnectorEdid(fd, props, caps)) {
        std::printf("    Sink HDR (EDID):          (no readable EDID)\n");
        return;
    }
    const auto yn = [](bool b) -> const char * { return b ? "yes" : "no"; };
    std::printf("    Sink HDR (EDID CTA-861):\n");
    std::printf("      HDR10 / PQ (ST 2084)    %s\n", yn(caps.hdr10Pq));
    std::printf("      HLG (ARIB STD-B67)      %s\n", yn(caps.hlg));
    std::printf("      BT.2020 Y'CbCr          %s\n", yn(caps.bt2020Ycc));
    std::printf("      HDR10+ (ST 2094-40)     %s\n", yn(caps.hdr10Plus));
    std::printf("      SL-HDR1 (TS 103 433-1)  %s\n", yn(caps.slHdr1));
    std::printf("      SL-HDR2 (TS 103 433-2)  %s\n", yn(caps.slHdr2));
    std::printf("      SL-HDR3 (TS 103 433-3)  %s\n", yn(caps.slHdr3));
    std::printf("      ST 2094-10 dyn. meta    %s\n", yn(caps.dynMetaType1));
    if (caps.hasLuminance) {
        std::printf("      desired max luminance   ~%.0f cd/m^2 (code %u)\n", LuminanceCodeToNits(caps.maxLumaCode),
                    caps.maxLumaCode);
    }
    // DV is sink capability only. The version is the EDID block layout (0..2), not the Dolby Vision
    // generation -- "Dolby Vision 2" has no published EDID signaling, so it cannot be detected here.
    if (caps.dolbyVision) {
        std::printf("      Dolby Vision (sink)     yes (VSVDB v%u)\n", caps.dvBlockVer);
    } else {
        std::printf("      Dolby Vision (sink)     no\n");
    }
    // VRR: sinks signal it three ways -- HDMI 2.1 game-VRR (HF-VSDB), AMD FreeSync (AMD VSDB;
    // FreeSync TVs and DP/eDP panels), and the VESA range-limits descriptor combined with the
    // continuous-frequency flag (what the kernel folds into vrr_capable on DP/eDP).
    std::printf("    Sink VRR (EDID):\n");
    const auto printRange = [](const char *label, unsigned int lo, unsigned int hi) -> void {
        if (lo != 0 && hi != 0) {
            std::printf("      %-24s%u-%u Hz\n", label, lo, hi);
        } else {
            std::printf("      %-24s(not advertised)\n", label);
        }
    };
    printRange("HDMI 2.1 game-VRR", caps.vrrMin, caps.vrrMax);
    printRange("AMD FreeSync", caps.fsMin, caps.fsMax);
    if (caps.rangeMinHz != 0 && caps.rangeMaxHz != 0) {
        std::printf("      %-24s%u-%u Hz (%s)\n", "VESA range limits", caps.rangeMinHz, caps.rangeMaxHz,
                    caps.continuousFreq ? "continuous frequency" : "fixed modes only");
    } else {
        std::printf("      %-24s(no range descriptor)\n", "VESA range limits");
    }
}

// The connector only carries its mode *list*; the mode actually scanning out lives on the
// CRTC reached through the connector's active encoder.
auto PrintCurrentConnectorMode(int fd, const drmModeConnector *c) -> void {
    uint32_t crtcId = 0;
    if (c->encoder_id != 0) {
        if (drmModeEncoder *enc = drmModeGetEncoder(fd, c->encoder_id); enc != nullptr) {
            crtcId = enc->crtc_id;
            drmModeFreeEncoder(enc);
        }
    }
    if (crtcId == 0) {
        std::printf("    current mode:             (no CRTC active)\n");
        return;
    }
    drmModeCrtc *crtc = drmModeGetCrtc(fd, crtcId);
    if (crtc == nullptr) {
        return;
    }
    if (crtc->mode_valid != 0) {
        std::printf("    current mode:             %ux%u@%uHz (crtc %u)\n", crtc->mode.hdisplay, crtc->mode.vdisplay,
                    crtc->mode.vrefresh, crtcId);
    } else {
        std::printf("    current mode:             (crtc %u active, no mode programmed)\n", crtcId);
    }
    drmModeFreeCrtc(crtc);
    // Live VRR state on the CRTC actually scanning this connector out (only meaningful
    // when the connector's vrr_capable says the sink+driver pair can do it at all).
    const auto crtcProps = LoadObjectProps(fd, crtcId, DRM_MODE_OBJECT_CRTC);
    if (const auto *p = FindProp(crtcProps, "VRR_ENABLED"); p != nullptr) {
        std::printf("    VRR_ENABLED (crtc)        %s\n", p->value != 0 ? "on" : "off");
    }
}

auto ProbeDrmConnectors(int fd, drmModeRes *res) -> void {
    std::printf("\n--- DRM Connectors ---\n");
    uint32_t disconnected = 0;
    for (int i = 0; i < res->count_connectors; ++i) {
        drmModeConnector *c = drmModeGetConnector(fd, res->connectors[i]);
        if (c == nullptr) {
            continue;
        }
        if (c->connection != DRM_MODE_CONNECTED) {
            ++disconnected;
            drmModeFreeConnector(c);
            continue;
        }
        std::printf("  Connector %u: %s-%u  connected  modes=%d\n", c->connector_id,
                    ConnectorTypeName(c->connector_type), c->connector_type_id, c->count_modes);
        if (c->count_modes > 0) {
            const drmModeModeInfo &m = c->modes[0];
            std::printf("    preferred mode:           %ux%u@%uHz\n", m.hdisplay, m.vdisplay, m.vrefresh);
        }
        PrintCurrentConnectorMode(fd, c);
        const auto props = LoadObjectProps(fd, c->connector_id, DRM_MODE_OBJECT_CONNECTOR);
        std::printf("    HDR_OUTPUT_METADATA       %s\n",
                    FindProp(props, "HDR_OUTPUT_METADATA") != nullptr ? "yes" : "no");
        if (const auto *p = FindProp(props, "max bpc"); p != nullptr) {
            std::printf("    max bpc                   %u\n", static_cast<uint32_t>(p->value));
        }
        if (const auto *p = FindProp(props, "Colorspace"); p != nullptr) {
            std::printf("    Colorspace (current)      %u\n", static_cast<uint32_t>(p->value));
        }
        // Kernel-side VRR verdict: the driver folds the sink's EDID signaling (HDMI HF-VSDB
        // game-VRR, DP Adaptive-Sync range) and its own scanout support into this property.
        if (const auto *p = FindProp(props, "vrr_capable"); p != nullptr) {
            std::printf("    vrr_capable               %s\n", p->value != 0 ? "yes" : "no");
        } else {
            std::printf("    vrr_capable               no (property absent)\n");
        }
        PrintSinkHdrCaps(fd, props);
        drmModeFreeConnector(c);
    }
    if (disconnected != 0) {
        std::printf("  (+%u disconnected, not listed)\n", disconnected);
    }
}

struct PlaneSummary {
    uint32_t id;            ///< DRM plane object ID
    uint64_t type;          ///< DRM_PLANE_TYPE_* or ~0 if absent
    uint32_t possibleCrtcs; ///< CRTC bitmask the plane can attach to
    bool hasNV12;           ///< accepts NV12 (8-bit video scanout, Gen9.5+)
    bool hasP010;           ///< accepts P010 (10-bit / HDR scanout, Gen12+)
    bool hasARGB8888;       ///< accepts ARGB8888 (OSD candidate)
    uint64_t colorEncoding; ///< current value, ~0 if property absent
};

// Collapse one plane's IN_FORMATS / properties to a few yes/no flags.
[[nodiscard]] auto SummarizePlane(int fd, drmModePlane *plane, const std::vector<PropEntry> &props) -> PlaneSummary {
    PlaneSummary s{
        .id = plane->plane_id,
        .type = ~uint64_t{0},
        .possibleCrtcs = plane->possible_crtcs,
        .hasNV12 = false,
        .hasP010 = false,
        .hasARGB8888 = false,
        .colorEncoding = ~uint64_t{0},
    };
    if (const auto *p = FindProp(props, "type"); p != nullptr) {
        s.type = p->value;
    }
    if (const auto *p = FindProp(props, "COLOR_ENCODING"); p != nullptr) {
        s.colorEncoding = p->value;
    }

    // Prefer IN_FORMATS (modifier-aware), fall back to the raw format list.
    const auto *inFormats = FindProp(props, "IN_FORMATS");
    drmModePropertyBlobRes *blob =
        (inFormats != nullptr) ? drmModeGetPropertyBlob(fd, static_cast<uint32_t>(inFormats->value)) : nullptr;
    std::span<const uint32_t> formats;
    // Bounds-check the blob before constructing a span over it: a malformed kernel blob
    // (or one allocated with a future ABI extension we don't know about) could otherwise
    // produce an out-of-bounds read. Falls back to plane->formats on any validation failure.
    if (blob != nullptr && blob->data != nullptr && blob->length >= sizeof(drm_format_modifier_blob)) {
        const auto *formatsHeader = static_cast<const drm_format_modifier_blob *>(blob->data);
        const auto formatsOffset = static_cast<size_t>(formatsHeader->formats_offset);
        const auto formatsBytes = static_cast<size_t>(formatsHeader->count_formats) * sizeof(uint32_t);
        if (formatsOffset <= blob->length && formatsBytes <= blob->length - formatsOffset) {
            const auto *blobBytes = static_cast<const uint8_t *>(blob->data) + formatsOffset;
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- mandated by DRM blob layout
            const auto *formatData = reinterpret_cast<const uint32_t *>(blobBytes);
            formats = {formatData, formatsHeader->count_formats};
        }
    }
    if (formats.empty()) {
        formats = {plane->formats, plane->count_formats};
    }
    for (const uint32_t fc : formats) {
        if (fc == DRM_FORMAT_NV12) {
            s.hasNV12 = true;
        } else if (fc == DRM_FORMAT_P010) {
            s.hasP010 = true;
        } else if (fc == DRM_FORMAT_ARGB8888) {
            s.hasARGB8888 = true;
        }
    }
    if (blob != nullptr) {
        drmModeFreePropertyBlob(blob);
    }
    return s;
}

auto ProbeDrmPlanes(int fd) -> void {
    std::printf("\n--- DRM Planes ---\n");
    drmModePlaneRes *planeRes = drmModeGetPlaneResources(fd);
    if (planeRes == nullptr) {
        std::printf("  (drmModeGetPlaneResources failed -- ensure UNIVERSAL_PLANES is set)\n");
        return;
    }

    std::vector<PlaneSummary> all;
    all.reserve(planeRes->count_planes);
    for (uint32_t i = 0; i < planeRes->count_planes; ++i) {
        drmModePlane *plane = drmModeGetPlane(fd, planeRes->planes[i]);
        if (plane == nullptr) {
            continue;
        }
        const auto props = LoadObjectProps(fd, plane->plane_id, DRM_MODE_OBJECT_PLANE);
        all.push_back(SummarizePlane(fd, plane, props));
        drmModeFreePlane(plane);
    }

    uint32_t nPrim = 0;
    uint32_t nOver = 0;
    uint32_t nCurs = 0;
    uint32_t nNV12 = 0;
    uint32_t nP010 = 0;
    uint32_t nOsd = 0;
    for (const auto &s : all) {
        if (s.type == DRM_PLANE_TYPE_PRIMARY) {
            ++nPrim;
        } else if (s.type == DRM_PLANE_TYPE_OVERLAY) {
            ++nOver;
        } else if (s.type == DRM_PLANE_TYPE_CURSOR) {
            ++nCurs;
        }
        if (s.hasNV12) {
            ++nNV12;
        }
        if (s.hasP010) {
            ++nP010;
        }
        if (s.hasARGB8888) {
            ++nOsd;
        }
    }
    std::printf("  Total: %u planes  (PRIMARY=%u OVERLAY=%u CURSOR=%u)\n", planeRes->count_planes, nPrim, nOver, nCurs);
    std::printf("  NV12     planes (8-bit video):  %u\n", nNV12);
    std::printf("  P010     planes (10-bit / HDR): %u\n", nP010);
    std::printf("  ARGB8888 planes (OSD):          %u\n", nOsd);

    // Flag non-default COLOR_ENCODING. i915 defaults to 1 (BT.709); a leftover 2 (BT.2020)
    // from a prior HDR session on the same plane can reject the next plain SDR commit on Gen12+.
    bool printedHeader = false;
    for (const auto &s : all) {
        if (s.colorEncoding == ~uint64_t{0} || s.colorEncoding == 1) {
            continue;
        }
        if (!printedHeader) {
            std::printf("  Stale plane state (non-default COLOR_ENCODING, may block atomic commits):\n");
            printedHeader = true;
        }
        const char *enc = "?";
        if (s.colorEncoding == 0) {
            enc = "BT.601";
        } else if (s.colorEncoding == 2) {
            enc = "BT.2020";
        }
        std::printf("    plane %u (%s, crtcs=0x%x): COLOR_ENCODING=%lu (%s)\n", s.id, PlaneTypeName(s.type),
                    s.possibleCrtcs, static_cast<unsigned long>(s.colorEncoding), enc);
    }

    drmModeFreePlaneResources(planeRes);
}

auto ProbeDrm(const char *devicePath) -> void {
    std::printf("\n================================================\n"
                "DRM Capability Trace (%s)\n"
                "================================================\n",
                devicePath);
    const int fd = open(devicePath, O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        std::printf("  (cannot open %s: %s)\n", devicePath, std::strerror(errno));
        return;
    }
    ProbeDrmDeviceCaps(fd);
    drmModeRes *res = drmModeGetResources(fd);
    if (res == nullptr) {
        std::printf("\n  (drmModeGetResources failed -- not a KMS device?)\n");
        close(fd);
        return;
    }
    std::printf("\n--- DRM Resources ---\n");
    std::printf("  CRTCs: %d  Connectors: %d  Encoders: %d\n", res->count_crtcs, res->count_connectors,
                res->count_encoders);
    ProbeDrmConnectors(fd, res);
    ProbeDrmPlanes(fd);
    drmModeFreeResources(res);
    close(fd);
}

} // namespace

auto main(int argc, char *argv[]) -> int {
    if (argc > 1 && (std::strcmp(argv[1], "-h") == 0 || std::strcmp(argv[1], "--help") == 0)) {
        std::printf("Usage: %s [/dev/dri/cardN]  (default: /dev/dri/card0)\n", argv[0]);
        return EXIT_SUCCESS;
    }

    const char *devicePath = (argc > 1) ? argv[1] : "/dev/dri/card0";

    std::printf("VAAPI Capability Prober (vdr-plugin-vaapivideo)\n"
                "================================================\n");

    if (struct utsname uts{}; uname(&uts) == 0) {
        std::printf("Kernel:      %s %s (%s)\n", uts.sysname, uts.release, uts.machine);
    }

    // --- Open DRM device ---
    const int drmFd = open(devicePath, O_RDWR | O_CLOEXEC);
    if (drmFd < 0) [[unlikely]] {
        (void)std::fprintf(stderr, "ERROR: cannot open '%s': %s\n", devicePath, std::strerror(errno));
        (void)std::fprintf(stderr, "       Ensure user is in 'video' or 'render' group.\n");
        return EXIT_FAILURE;
    }

    std::printf("DRM device:  %s\n", devicePath);

    const auto renderNode = ResolveRenderNode(drmFd);
    close(drmFd);

    if (renderNode.empty()) [[unlikely]] {
        (void)std::fprintf(stderr, "ERROR: no render node found for '%s'\n", devicePath);
        return EXIT_FAILURE;
    }
    std::printf("Render node: %s\n", renderNode.c_str());

    // --- Open render node and initialize VAAPI ---
    const int renderFd = open(renderNode.c_str(), O_RDWR | O_CLOEXEC);
    if (renderFd < 0) [[unlikely]] {
        (void)std::fprintf(stderr, "ERROR: cannot open render node '%s': %s\n", renderNode.c_str(),
                           std::strerror(errno));
        return EXIT_FAILURE;
    }

    VADisplay vaDisplay = vaGetDisplayDRM(renderFd);
    if (!vaDisplay) [[unlikely]] {
        (void)std::fprintf(stderr, "ERROR: vaGetDisplayDRM failed\n");
        close(renderFd);
        return EXIT_FAILURE;
    }

    int vaMajor = 0;
    int vaMinor = 0;
    if (const VAStatus st = vaInitialize(vaDisplay, &vaMajor, &vaMinor); st != VA_STATUS_SUCCESS) [[unlikely]] {
        (void)std::fprintf(stderr, "ERROR: vaInitialize failed: %s\n", vaErrorStr(st));
        close(renderFd);
        return EXIT_FAILURE;
    }

    std::printf("VA-API:      %d.%d\n", vaMajor, vaMinor);

    const char *vendor = vaQueryVendorString(vaDisplay);
    std::printf("Driver:      %s\n", vendor ? vendor : "(unknown)");

    const auto decodeSupport = ProbeDecodeProfiles(vaDisplay);

    std::printf("\n--- Video Processing Pipeline (VPP) ---\n");

    VAConfigID vppConfig = VA_INVALID_ID;
    const bool vppAvailable =
        vaCreateConfig(vaDisplay, VAProfileNone, VAEntrypointVideoProc, nullptr, 0, &vppConfig) == VA_STATUS_SUCCESS;

    PrintCapability("General (VideoProc)", vppAvailable);
    PrintCapability("Scaling", vppAvailable); // Implicit in any VPP context; no separate capability bit.

    if (!vppAvailable) {
        std::printf("\n  VPP unavailable -- remaining queries skipped.\n");
        vaTerminate(vaDisplay);
        close(renderFd);
        return EXIT_SUCCESS;
    }

    // VA_RT_FORMAT_YUV420/YUV420_10 are format classes, not specific layouts.
    // The RT-format probe answers "does the driver support this bit depth?";
    // the FourCC probe answers "can the plugin pipeline actually consume it?"
    // (NV12 for 8-bit, P010 for 10-bit -- both required by scale_vaapi and DRM plane).
    const bool hasRtNV12 = CanCreateSurface(vaDisplay, VA_RT_FORMAT_YUV420);
    const bool hasRtP010 = CanCreateSurface(vaDisplay, VA_RT_FORMAT_YUV420_10);
    const bool hasNV12 = CanCreateSurfaceFourcc(vaDisplay, VA_RT_FORMAT_YUV420, VA_FOURCC_NV12);
    const bool hasP010 = CanCreateSurfaceFourcc(vaDisplay, VA_RT_FORMAT_YUV420_10, VA_FOURCC_P010);

    PrintCapability("YUV420    class  (any 8-bit  4:2:0)", hasRtNV12);
    PrintCapability("YUV420_10 class  (any 10-bit 4:2:0)", hasRtP010);
    PrintCapability("NV12 FourCC      (plugin 8-bit path)", hasNV12);
    PrintCapability("P010 FourCC      (plugin 10-bit path)", hasP010);
    if (hasRtNV12 && !hasNV12) {
        std::printf("  %-44s \033[33mYUV420 class works but NV12 FourCC rejected -- buggy driver\033[0m\n", "");
    }
    if (hasRtP010 && !hasP010) {
        std::printf("  %-44s \033[33mYUV420_10 class works but P010 FourCC rejected -- buggy driver\033[0m\n", "");
    }
    PrintCapability("P010 -> NV12     (VPP 10->8 bit)", hasP010 && hasNV12);

    // vaQueryVideoProcFilterCaps requires a context; the context requires at least
    // one surface.  Minimal 64x64 YUV420 surface satisfies the driver constraint.
    VASurfaceID probeSurface = VA_INVALID_SURFACE;
    if (const VAStatus st = vaCreateSurfaces(vaDisplay, VA_RT_FORMAT_YUV420, PROBE_SURFACE_SIZE, PROBE_SURFACE_SIZE,
                                             &probeSurface, 1, nullptr, 0);
        st != VA_STATUS_SUCCESS) [[unlikely]] {
        (void)std::fprintf(stderr, "  WARNING: vaCreateSurfaces failed (%s) -- cannot query filters\n", vaErrorStr(st));
        vaDestroyConfig(vaDisplay, vppConfig);
        vaTerminate(vaDisplay);
        close(renderFd);
        return EXIT_FAILURE;
    }

    VAContextID vppContext = VA_INVALID_ID;
    if (const VAStatus st = vaCreateContext(vaDisplay, vppConfig, PROBE_SURFACE_SIZE, PROBE_SURFACE_SIZE, 0,
                                            &probeSurface, 1, &vppContext);
        st != VA_STATUS_SUCCESS) [[unlikely]] {
        (void)std::fprintf(stderr, "  WARNING: vaCreateContext failed (%s) -- cannot query filters\n", vaErrorStr(st));
        vaDestroySurfaces(vaDisplay, &probeSurface, 1);
        vaDestroyConfig(vaDisplay, vppConfig);
        vaTerminate(vaDisplay);
        close(renderFd);
        return EXIT_FAILURE;
    }

    VAProcFilterType filterBuf[VAProcFilterCount];
    auto filterCount = static_cast<unsigned int>(VAProcFilterCount);
    if (vaQueryVideoProcFilters(vaDisplay, vppContext, filterBuf, &filterCount) != VA_STATUS_SUCCESS) {
        filterCount = 0;
    }

    const std::span<const VAProcFilterType> filters{filterBuf, static_cast<size_t>(filterCount)};

    for (const auto &[filterType, filterName] : kVppFilterTypes) {
        PrintCapability(filterName, std::ranges::find(filters, filterType) != filters.end());
    }

    const HdrToneMapCaps toneMap = ProbeHdrToneMapping(vaDisplay, vppContext, filters);

    PrintColorConversions(decodeSupport, hasP010, hasNV12, toneMap);
    ProbeDeinterlacing(vaDisplay, vppContext);

    vaDestroyContext(vaDisplay, vppContext);
    vaDestroySurfaces(vaDisplay, &probeSurface, 1);
    vaDestroyConfig(vaDisplay, vppConfig);
    vaTerminate(vaDisplay);
    close(renderFd);

    ProbeDrm(devicePath);

    return EXIT_SUCCESS;
}
