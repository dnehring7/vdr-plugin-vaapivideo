// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * @file vaapivideo-probe.cpp
 * @brief Standalone VAAPI capability prober for vdr-plugin-vaapivideo.
 *
 * Mirrors the probe matrix of the plugin runtime (src/caps.cpp ProbeGpuCaps +
 * src/stream.h kVideoBackendTable) so an operator can predict which
 * codec/profile/bit-depth combinations will hardware-decode on a given GPU.
 *
 * Probed codecs:
 *   MPEG-2  (Simple, Main)                          -- SD broadcast
 *   H.264   (Constrained Baseline, Main, High, High 10) -- HD broadcast
 *   HEVC    (Main 8-bit, Main10 10-bit)             -- UHD + HDR
 *   AV1     (Profile 0, 8-bit + 10-bit)             -- streaming / mediaplayer
 *
 * Each profile is tested against the exact surface format it requires
 * (YUV420 for 8-bit, YUV420_10 for 10-bit). VPP filters, HDR tone mapping,
 * and deinterlacing modes are also reported.
 *
 * Usage:   ./vaapivideo-probe [/dev/dri/cardN]   (default: /dev/dri/card0)
 * Compile: g++ -std=c++20 $(pkg-config --cflags --libs libdrm libva-drm) -o vaapivideo-probe vaapivideo-probe.cpp
 * Lint:    clang-tidy vaapivideo-probe.cpp -- -std=c++20 $(pkg-config --cflags libdrm libva-drm)
 */

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include <va/va.h>
#include <va/va_drm.h>
#include <va/va_vpp.h>

#include <xf86drm.h>

// Small enough to avoid wasting VRAM; large enough that no driver rejects it as
// below its minimum surface alignment (typically 16 px).
constexpr int kProbeSurfaceSize = 64;

namespace {
struct DrmDeviceDeleter {
    auto operator()(drmDevice *dev) const noexcept -> void { drmFreeDevice(&dev); }
};
} // namespace

// Every (profile, required RT surface format) combination worth reporting.
// Plugin-consumed rows mirror kVideoBackendTable (src/stream.h) and
// ProbeGpuCaps (src/caps.cpp) so the output predicts runtime behaviour exactly.
// Informational rows cover profiles modern iGPUs advertise but the plugin does
// not decode: VP9, HEVC range extensions (12-bit, 4:2:2, 4:4:4), and JPEG.
namespace {
struct DecodeProbe {
    VAProfile profile;
    unsigned int rtFormat;
    const char *name;
};
} // namespace
static constexpr DecodeProbe kDecodeProbes[] = {
    // --- Consumed by the plugin (must match kVideoBackendTable) ---
    {.profile = VAProfileMPEG2Simple, .rtFormat = VA_RT_FORMAT_YUV420, .name = "MPEG-2 Simple"},
    {.profile = VAProfileMPEG2Main, .rtFormat = VA_RT_FORMAT_YUV420, .name = "MPEG-2 Main"},
    {.profile = VAProfileH264ConstrainedBaseline,
     .rtFormat = VA_RT_FORMAT_YUV420,
     .name = "H.264 Constrained Baseline"},
    {.profile = VAProfileH264Main, .rtFormat = VA_RT_FORMAT_YUV420, .name = "H.264 Main"},
    {.profile = VAProfileH264High, .rtFormat = VA_RT_FORMAT_YUV420, .name = "H.264 High"},
    {.profile = VAProfileH264High10, .rtFormat = VA_RT_FORMAT_YUV420_10, .name = "H.264 High 10"},
    {.profile = VAProfileHEVCMain, .rtFormat = VA_RT_FORMAT_YUV420, .name = "HEVC Main"},
    {.profile = VAProfileHEVCMain10, .rtFormat = VA_RT_FORMAT_YUV420_10, .name = "HEVC Main 10"},
    {.profile = VAProfileAV1Profile0, .rtFormat = VA_RT_FORMAT_YUV420, .name = "AV1 Profile 0 (8-bit)"},
    {.profile = VAProfileAV1Profile0, .rtFormat = VA_RT_FORMAT_YUV420_10, .name = "AV1 Profile 0 (10-bit)"},
    // --- Informational: iGPU can decode, plugin does not consume today ---
    {.profile = VAProfileHEVCMain12, .rtFormat = VA_RT_FORMAT_YUV420_12, .name = "HEVC Main 12"},
    {.profile = VAProfileHEVCMain422_10, .rtFormat = VA_RT_FORMAT_YUV422_10, .name = "HEVC Main 4:2:2 10"},
    {.profile = VAProfileHEVCMain422_12, .rtFormat = VA_RT_FORMAT_YUV422_12, .name = "HEVC Main 4:2:2 12"},
    {.profile = VAProfileHEVCMain444, .rtFormat = VA_RT_FORMAT_YUV444, .name = "HEVC Main 4:4:4"},
    {.profile = VAProfileHEVCMain444_10, .rtFormat = VA_RT_FORMAT_YUV444_10, .name = "HEVC Main 4:4:4 10"},
    {.profile = VAProfileVP9Profile0, .rtFormat = VA_RT_FORMAT_YUV420, .name = "VP9 Profile 0"},
    {.profile = VAProfileVP9Profile2, .rtFormat = VA_RT_FORMAT_YUV420_10, .name = "VP9 Profile 2"},
    {.profile = VAProfileJPEGBaseline, .rtFormat = VA_RT_FORMAT_YUV420, .name = "JPEG Baseline"},
};

// General VPP filter types. Deinterlacing and HDR tone mapping are probed
// separately because they require capability structs beyond a simple yes/no.
static constexpr struct {
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

// Directions relevant to a broadcast playback pipeline.
// SDR->HDR is deliberately excluded: no broadcast source requires it.
static constexpr struct {
    uint16_t flag;
    const char *name;
} kToneMappingFlags[] = {
    {.flag = VA_TONE_MAPPING_HDR_TO_HDR, .name = "HDR->HDR"},
    {.flag = VA_TONE_MAPPING_HDR_TO_SDR, .name = "HDR->SDR"},
};

// Listed highest-quality first so the first supported entry is the preferred choice.
static constexpr struct {
    VAProcDeinterlacingType type;
    const char *name;
} kDeintAlgorithms[] = {
    {.type = VAProcDeinterlacingMotionCompensated, .name = "Motion Compensated"},
    {.type = VAProcDeinterlacingMotionAdaptive, .name = "Motion Adaptive"},
    {.type = VAProcDeinterlacingWeave, .name = "Weave"},
    {.type = VAProcDeinterlacingBob, .name = "Bob"},
};

[[nodiscard]] static auto SupportsVldEntrypoint(VADisplay display, VAProfile profile) -> bool {
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

[[nodiscard]] static auto SupportsRtFormat(VADisplay display, VAProfile profile, unsigned int rtFormat) -> bool {
    VAConfigAttrib attrib{};
    attrib.type = VAConfigAttribRTFormat;
    if (vaGetConfigAttributes(display, profile, VAEntrypointVLD, &attrib, 1) != VA_STATUS_SUCCESS) {
        return false;
    }
    return (attrib.value & rtFormat) != 0;
}

[[nodiscard]] static auto ResolveRenderNode(int drmFd) -> std::string {
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

static auto PrintCapability(const char *label, bool supported) -> void {
    std::printf("  %-44s %s\n", label, supported ? "\033[32myes\033[0m" : "\033[31mno\033[0m");
}

// Human-readable label for each VA_RT_FORMAT class, used to annotate probe
// rows with the surface layout required by each codec/profile.
[[nodiscard]] static auto RtFormatLabel(unsigned int rtFormat) -> const char * {
    switch (rtFormat) {
        case VA_RT_FORMAT_YUV420:
            return "8-bit  4:2:0";
        case VA_RT_FORMAT_YUV420_10:
            return "10-bit 4:2:0";
        case VA_RT_FORMAT_YUV420_12:
            return "12-bit 4:2:0";
        case VA_RT_FORMAT_YUV422:
            return "8-bit  4:2:2";
        case VA_RT_FORMAT_YUV422_10:
            return "10-bit 4:2:2";
        case VA_RT_FORMAT_YUV422_12:
            return "12-bit 4:2:2";
        case VA_RT_FORMAT_YUV444:
            return "8-bit  4:4:4";
        case VA_RT_FORMAT_YUV444_10:
            return "10-bit 4:4:4";
        case VA_RT_FORMAT_YUV444_12:
            return "12-bit 4:4:4";
        default:
            return "?";
    }
}

// Build a fixed-width "Name  (N-bit 4:X:X)" label for one probe row.
// Thread-local buffer avoids heap allocation inside the probe loop.
[[nodiscard]] static auto FormatProbeLabel(const DecodeProbe &row) -> const char * {
    static thread_local std::array<char, 64> buf{};
    (void)std::snprintf(buf.data(), buf.size(), "%-28s (%s)", row.name, RtFormatLabel(row.rtFormat));
    return buf.data();
}

[[nodiscard]] static auto CanCreateSurface(VADisplay display, unsigned int rtFormat) -> bool {
    VASurfaceID surface = VA_INVALID_SURFACE;
    if (vaCreateSurfaces(display, rtFormat, kProbeSurfaceSize, kProbeSurfaceSize, &surface, 1, nullptr, 0) !=
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
[[nodiscard]] static auto CanCreateSurfaceFourcc(VADisplay display, unsigned int rtFormat, uint32_t fourcc) -> bool {
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
    if (vaCreateSurfaces(display, rtFormat, kProbeSurfaceSize, kProbeSurfaceSize, &surface, 1, &attrib, 1) !=
        VA_STATUS_SUCCESS) {
        return false;
    }
    vaDestroySurfaces(display, &surface, 1);
    return true;
}

namespace {
// Mirrors GpuCaps (src/caps.h): one flag per codec/bit-depth the plugin uses.
// Populated by ProbeDecodeProfiles; consumed by PrintColorConversions.
struct DecodeSupport {
    bool mpeg2 = false;      ///< MPEG-2 Simple or Main (8-bit)
    bool h264 = false;       ///< H.264 CBP / Main / High (8-bit)
    bool h264High10 = false; ///< H.264 High 10 (10-bit)
    bool hevc = false;       ///< HEVC Main (8-bit)
    bool hevcMain10 = false; ///< HEVC Main 10 (10-bit)
    bool av1 = false;        ///< AV1 Profile 0 (8-bit)
    bool av1Main10 = false;  ///< AV1 Profile 0 (10-bit)
};
} // namespace

[[nodiscard]] static auto ProbeDecodeProfiles(VADisplay display) -> DecodeSupport {
    std::printf("\n--- Hardware Decode (VLD + required RT format) ---\n");

    const int maxProfiles = vaMaxNumProfiles(display);
    if (maxProfiles <= 0) {
        for (const auto &row : kDecodeProbes) {
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
    for (const auto &row : kDecodeProbes) {
        const bool hasProfile = std::ranges::find(profiles, row.profile) != profiles.end();
        const bool hasVld = hasProfile && SupportsVldEntrypoint(display, row.profile);
        const bool hasRt = hasVld && SupportsRtFormat(display, row.profile, row.rtFormat);

        PrintCapability(FormatProbeLabel(row), hasRt);

        if (hasRt) {
            // Mirrors GpuCaps flag-setting in ProbeGpuCaps (src/caps.cpp).
            // Profiles not in this switch (VP9, HEVC RExt, JPEG) are informational.
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

// Returns true if the driver supports tone mapping in at least one direction the
// plugin uses (HDR->HDR or HDR->SDR).  Does not print directly; the result is
// surfaced through PrintColorConversions so the operator sees it alongside the
// codec/surface context that makes it actionable.  The VPP filter list already
// shows whether VAProcFilterHighDynamicRangeToneMapping is advertised at all.
[[nodiscard]] static auto HasHdrToneMapping(VADisplay display, VAContextID vppContext,
                                            std::span<const VAProcFilterType> filters) -> bool {
    if (std::ranges::find(filters, VAProcFilterHighDynamicRangeToneMapping) == filters.end()) {
        return false;
    }

    VAProcFilterCapHighDynamicRange hdrCaps[VAProcHighDynamicRangeMetadataTypeCount];
    auto hdrCapCount = static_cast<unsigned int>(VAProcHighDynamicRangeMetadataTypeCount);
    if (vaQueryVideoProcFilterCaps(display, vppContext, VAProcFilterHighDynamicRangeToneMapping, hdrCaps,
                                   &hdrCapCount) != VA_STATUS_SUCCESS) {
        return false;
    }

    // Skip the None metadata row; require at least one direction the plugin uses.
    constexpr uint16_t kUsableMask = VA_TONE_MAPPING_HDR_TO_HDR | VA_TONE_MAPPING_HDR_TO_SDR;
    for (unsigned int i = 0; i < hdrCapCount; ++i) {
        if (hdrCaps[i].metadata_type != VAProcHighDynamicRangeMetadataNone &&
            (hdrCaps[i].caps_flag & kUsableMask) != 0) {
            return true;
        }
    }
    return false;
}

static auto ProbeDeinterlacing(VADisplay display, VAContextID vppContext) -> void {
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

static auto PrintColorConversions(const DecodeSupport &dec, bool hasP010, bool hasNV12, bool hasHdrToneMapping)
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
    PrintCapability("PQ/HDR10 -> SDR     (tone map)", any10Bit && hasHdrToneMapping);
}

auto main(int argc, char *argv[]) -> int {
    if (argc > 1 && (std::strcmp(argv[1], "-h") == 0 || std::strcmp(argv[1], "--help") == 0)) {
        std::printf("Usage: %s [/dev/dri/cardN]  (default: /dev/dri/card0)\n", argv[0]);
        return EXIT_SUCCESS;
    }

    const char *devicePath = (argc > 1) ? argv[1] : "/dev/dri/card0";

    std::printf("VAAPI Capability Prober (vdr-plugin-vaapivideo)\n"
                "================================================\n");

    // --- Open DRM device ---
    const int drmFd = open(devicePath, O_RDWR | O_CLOEXEC); // NOLINT(cppcoreguidelines-pro-type-vararg) -- POSIX open() is variadic
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
    const int renderFd = open(renderNode.c_str(), O_RDWR | O_CLOEXEC); // NOLINT(cppcoreguidelines-pro-type-vararg) -- POSIX open() is variadic
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
    if (const VAStatus st = vaCreateSurfaces(vaDisplay, VA_RT_FORMAT_YUV420, kProbeSurfaceSize, kProbeSurfaceSize,
                                             &probeSurface, 1, nullptr, 0);
        st != VA_STATUS_SUCCESS) [[unlikely]] {
        (void)std::fprintf(stderr, "  WARNING: vaCreateSurfaces failed (%s) -- cannot query filters\n", vaErrorStr(st));
        vaDestroyConfig(vaDisplay, vppConfig);
        vaTerminate(vaDisplay);
        close(renderFd);
        return EXIT_FAILURE;
    }

    VAContextID vppContext = VA_INVALID_ID;
    if (const VAStatus st = vaCreateContext(vaDisplay, vppConfig, kProbeSurfaceSize, kProbeSurfaceSize, 0,
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

    const bool hasHdrToneMapping = HasHdrToneMapping(vaDisplay, vppContext, filters);

    PrintColorConversions(decodeSupport, hasP010, hasNV12, hasHdrToneMapping);
    ProbeDeinterlacing(vaDisplay, vppContext);

    vaDestroyContext(vaDisplay, vppContext);
    vaDestroySurfaces(vaDisplay, &probeSurface, 1);
    vaDestroyConfig(vaDisplay, vppConfig);
    vaTerminate(vaDisplay);
    close(renderFd);

    return EXIT_SUCCESS;
}
