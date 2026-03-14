// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Compile:
//   g++ -std=c++20 $(pkg-config --cflags --libs libdrm libva-drm) -o vaapivideo-probe vaapivideo-probe.cpp
//
// Lint:
//   clang-tidy vaapivideo-probe.cpp -- -std=c++20 $(pkg-config --cflags libdrm libva-drm)
//
// Standalone VAAPI capability prober for DVB-S2 decode pipelines.
// Probes decode profiles, VPP filters, HDR tone mapping, and deinterlacing.
//
// Usage: ./vaapivideo-probe [/dev/dri/cardN]   (default: /dev/dri/card0)

#include <algorithm>
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

// ============================================================================
// === CONSTANTS ===
// ============================================================================

constexpr int kProbeSurfaceSize = 64;

// ============================================================================
// === RAII HELPERS ===
// ============================================================================

namespace {
struct DrmDeviceDeleter {
    auto operator()(drmDevice *dev) const noexcept -> void { drmFreeDevice(&dev); }
};
} // namespace

// ============================================================================
// === LOOKUP TABLES ===
// ============================================================================

/// DVB-S2 decode profiles: MPEG-2 Main (SD), H.264 Main/High (HD), HEVC Main 10 (UHD).
static constexpr struct {
    VAProfile profile;
    const char *name;
} kDecodeProfiles[] = {
    {.profile = VAProfileMPEG2Main, .name = "MPEG-2 Main"},
    {.profile = VAProfileH264Main, .name = "H.264 Main"},
    {.profile = VAProfileH264High, .name = "H.264 High"},
    {.profile = VAProfileHEVCMain10, .name = "HEVC Main 10"},
};

/// VPP filter types (deinterlacing and HDR are probed separately).
static constexpr struct {
    VAProcFilterType type;
    const char *name;
} kVppFilterTypes[] = {
    {.type = VAProcFilterNoiseReduction, .name = "Noise Reduction (Denoise)"},
    {.type = VAProcFilterSharpening, .name = "Sharpening"},
    {.type = VAProcFilterColorBalance, .name = "Color Balance"},
    {.type = VAProcFilterSkinToneEnhancement, .name = "Skin Tone Enhancement"},
    {.type = VAProcFilterTotalColorCorrection, .name = "Total Color Correction"},
    {.type = VAProcFilterHVSNoiseReduction, .name = "HVS Noise Reduction"},
};

/// Tone mapping direction flags.
static constexpr struct {
    uint16_t flag;
    const char *name;
} kToneMappingFlags[] = {
    {.flag = VA_TONE_MAPPING_HDR_TO_HDR, .name = "HDR->HDR"},
    {.flag = VA_TONE_MAPPING_HDR_TO_SDR, .name = "HDR->SDR"},
    {.flag = VA_TONE_MAPPING_SDR_TO_HDR, .name = "SDR->HDR"},
};

/// Deinterlacing algorithms, ordered best to worst.
static constexpr struct {
    VAProcDeinterlacingType type;
    const char *name;
} kDeintAlgorithms[] = {
    {.type = VAProcDeinterlacingMotionCompensated, .name = "Motion Compensated"},
    {.type = VAProcDeinterlacingMotionAdaptive, .name = "Motion Adaptive"},
    {.type = VAProcDeinterlacingWeave, .name = "Weave"},
    {.type = VAProcDeinterlacingBob, .name = "Bob"},
};

// ============================================================================
// === HELPERS ===
// ============================================================================

/// Check whether a profile has the VLD (Variable Length Decoding) entrypoint.
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

    const std::span<const VAEntrypoint> valid{entrypoints.data(), static_cast<size_t>(entrypointCount)};
    return std::find(valid.begin(), valid.end(), VAEntrypointVLD) != valid.end();
}

/// Check whether a profile+VLD config supports VA_RT_FORMAT_YUV420.
[[nodiscard]] static auto SupportsYuv420RtFormat(VADisplay display, VAProfile profile) -> bool {
    VAConfigAttrib attrib{};
    attrib.type = VAConfigAttribRTFormat;
    if (vaGetConfigAttributes(display, profile, VAEntrypointVLD, &attrib, 1) != VA_STATUS_SUCCESS) {
        return false;
    }
    return (attrib.value & VA_RT_FORMAT_YUV420) != 0;
}

/// Resolve the render node path (e.g. /dev/dri/renderD128) from an open DRM fd.
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

/// Print a labelled yes/no capability line with ANSI color.
static auto PrintCapability(const char *label, bool supported) -> void {
    std::printf("  %-34s %s\n", label, supported ? "\033[32myes\033[0m" : "\033[31mno\033[0m");
}

/// Test whether the driver can create a surface with the given RT format.
[[nodiscard]] static auto CanCreateSurface(VADisplay display, unsigned int rtFormat) -> bool {
    VASurfaceID surface = VA_INVALID_SURFACE;
    const bool ok = vaCreateSurfaces(display, rtFormat, kProbeSurfaceSize, kProbeSurfaceSize, &surface, 1, nullptr,
                                     0) == VA_STATUS_SUCCESS;
    if (ok) {
        vaDestroySurfaces(display, &surface, 1);
    }
    return ok;
}

// ============================================================================
// === PROBE FUNCTIONS ===
// ============================================================================

/// Probe and print hardware decode profiles.
static auto ProbeDecodeProfiles(VADisplay display) -> void {
    std::printf("\n--- Hardware Decode (VLD + YUV420) ---\n");

    const int maxProfiles = vaMaxNumProfiles(display);
    std::vector<VAProfile> profileBuf(static_cast<size_t>(maxProfiles));
    int profileCount = 0;
    const bool queryOk = vaQueryConfigProfiles(display, profileBuf.data(), &profileCount) == VA_STATUS_SUCCESS;
    const std::span<const VAProfile> profiles{profileBuf.data(), queryOk ? static_cast<size_t>(profileCount) : 0};

    for (const auto &[profile, name] : kDecodeProfiles) {
        const bool hasProfile = std::find(profiles.begin(), profiles.end(), profile) != profiles.end();
        const bool hasVld = hasProfile && SupportsVldEntrypoint(display, profile);
        const bool supported = hasVld && SupportsYuv420RtFormat(display, profile);

        PrintCapability(name, supported);

        if (hasProfile && !supported) {
            if (!hasVld) {
                std::printf("  %-34s (profile present but no VLD entrypoint)\n", "");
            } else {
                std::printf("  %-34s (VLD present but no YUV420 RT format)\n", "");
            }
        }
    }
}

/// Probe HDR tone mapping metadata types and direction flags.
[[nodiscard]] static auto ProbeHdrToneMapping(VADisplay display, VAContextID vppContext,
                                              std::span<const VAProcFilterType> filters) -> bool {
    if (std::find(filters.begin(), filters.end(), VAProcFilterHighDynamicRangeToneMapping) == filters.end()) {
        PrintCapability("HDR Tone Mapping", false);
        return false;
    }

    VAProcFilterCapHighDynamicRange hdrCaps[VAProcHighDynamicRangeMetadataTypeCount];
    auto hdrCapCount = static_cast<unsigned int>(VAProcHighDynamicRangeMetadataTypeCount);
    if (vaQueryVideoProcFilterCaps(display, vppContext, VAProcFilterHighDynamicRangeToneMapping, hdrCaps,
                                   &hdrCapCount) != VA_STATUS_SUCCESS ||
        hdrCapCount == 0) {
        std::printf("  %-34s \033[33mlisted but no usable caps\033[0m\n", "HDR Tone Mapping");
        return false;
    }

    bool hasUsableCaps = false;
    for (unsigned int i = 0; i < hdrCapCount; ++i) {
        if (hdrCaps[i].metadata_type == VAProcHighDynamicRangeMetadataNone) {
            continue;
        }
        if (hdrCaps[i].caps_flag != 0) {
            hasUsableCaps = true;
        }

        const char *hdrLabel = (hdrCaps[i].metadata_type == VAProcHighDynamicRangeMetadataHDR10)
                                   ? "HDR Tone Mapping (HDR10)"
                                   : "HDR Tone Mapping (unknown)";

        if (hdrCaps[i].caps_flag == 0) {
            std::printf("  %-34s \033[31mno usable flags\033[0m\n", hdrLabel);
        } else {
            std::printf("  %-34s ", hdrLabel);
            bool first = true;
            for (const auto &[flag, flagName] : kToneMappingFlags) {
                if (hdrCaps[i].caps_flag & flag) {
                    std::printf("%s%s", first ? "" : ", ", flagName);
                    first = false;
                }
            }
            std::printf("\n");
        }
    }

    if (!hasUsableCaps) {
        std::printf("  %-34s \033[33mlisted but no usable caps\033[0m\n", "HDR Tone Mapping");
    }
    return hasUsableCaps;
}

/// Probe and print supported deinterlacing algorithms.
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
            std::any_of(caps.begin(), caps.end(), [deintType](const auto &c) { return c.type == deintType; });
        PrintCapability(deintName, supported);
    }
}

/// Print DVB-S2 color conversion paths based on discovered capabilities.
static auto PrintColorConversions(bool vppAvailable, bool hasP010, bool hasNV12, bool hasHdrToneMapping) -> void {
    std::printf("\n--- DVB-S2 Color Conversion Paths ---\n");

    // MPEG-2 SD PAL to HD/UHD display.
    PrintCapability("BT.601 -> BT.709  (MPEG-2 SD)", vppAvailable);

    // H.264 HD -- passthrough, always works.
    PrintCapability("BT.709 passthrough (H.264 HD)", true);

    // HEVC UHD wide color gamut to SDR.
    PrintCapability("BT.2020 -> BT.709  (HEVC UHD)", vppAvailable && hasP010);

    // HEVC Main 10 decode to 8-bit output.
    PrintCapability("P010 -> NV12      (10->8 bit)", hasP010 && hasNV12 && vppAvailable);

    // HLG is backwards-compatible, no tone mapping needed.
    PrintCapability("HLG -> SDR   (no TM required)", true);

    // Requires VPP tone mapping with HDR->SDR support.
    PrintCapability("PQ/HDR10 -> SDR    (tone map)", hasHdrToneMapping);
}

// ============================================================================
// === MAIN ===
// ============================================================================

auto main(int argc, char *argv[]) -> int {
    if (argc > 1 && (std::strcmp(argv[1], "-h") == 0 || std::strcmp(argv[1], "--help") == 0)) {
        std::printf("Usage: %s [/dev/dri/cardN]  (default: /dev/dri/card0)\n", argv[0]);
        return EXIT_SUCCESS;
    }

    const char *devicePath = (argc > 1) ? argv[1] : "/dev/dri/card0";

    std::printf("VAAPI Capability Prober (DVB-S2 / YUV420-NV12)\n");
    std::printf("==============================================\n");

    // --- Open DRM device ---
    const int drmFd = open(devicePath, O_RDWR | O_CLOEXEC); // NOLINT(cppcoreguidelines-pro-type-vararg)
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

    // --- Open render node and initialise VAAPI ---
    const int renderFd = open(renderNode.c_str(), O_RDWR | O_CLOEXEC); // NOLINT(cppcoreguidelines-pro-type-vararg)
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
    const VAStatus initStatus = vaInitialize(vaDisplay, &vaMajor, &vaMinor);
    if (initStatus != VA_STATUS_SUCCESS) [[unlikely]] {
        (void)std::fprintf(stderr, "ERROR: vaInitialize failed: %s\n", vaErrorStr(initStatus));
        close(renderFd);
        return EXIT_FAILURE;
    }

    std::printf("VA-API:      %d.%d\n", vaMajor, vaMinor);

    const char *vendor = vaQueryVendorString(vaDisplay);
    std::printf("Driver:      %s\n", vendor ? vendor : "(unknown)");

    // --- Decode profiles ---
    ProbeDecodeProfiles(vaDisplay);

    // --- VPP support ---
    std::printf("\n--- Video Processing Pipeline (VPP) ---\n");

    VAConfigID vppConfig = VA_INVALID_ID;
    const bool vppAvailable =
        vaCreateConfig(vaDisplay, VAProfileNone, VAEntrypointVideoProc, nullptr, 0, &vppConfig) == VA_STATUS_SUCCESS;

    PrintCapability("General (VideoProc)", vppAvailable);
    PrintCapability("Scaling", vppAvailable); // Implicit in any VPP pipeline.

    if (!vppAvailable) {
        std::printf("\n  VPP unavailable -- remaining queries skipped.\n");
        vaTerminate(vaDisplay);
        close(renderFd);
        return EXIT_SUCCESS;
    }

    // --- Surface format support ---
    const bool hasP010 = CanCreateSurface(vaDisplay, VA_RT_FORMAT_YUV420_10);
    const bool hasNV12 = CanCreateSurface(vaDisplay, VA_RT_FORMAT_YUV420);

    PrintCapability("P010 (10-bit) surfaces", hasP010);
    PrintCapability("NV12 (8-bit) surfaces", hasNV12);
    PrintCapability("10-bit -> 8-bit conversion", hasP010 && hasNV12 && vppAvailable);

    // --- Create VPP context for filter queries ---
    VASurfaceID probeSurface = VA_INVALID_SURFACE;
    const VAStatus surfStatus = vaCreateSurfaces(vaDisplay, VA_RT_FORMAT_YUV420, kProbeSurfaceSize, kProbeSurfaceSize,
                                                 &probeSurface, 1, nullptr, 0);

    if (surfStatus != VA_STATUS_SUCCESS) [[unlikely]] {
        (void)std::fprintf(stderr, "  WARNING: vaCreateSurfaces failed (%s) -- cannot query filters\n",
                           vaErrorStr(surfStatus));
        vaDestroyConfig(vaDisplay, vppConfig);
        vaTerminate(vaDisplay);
        close(renderFd);
        return EXIT_FAILURE;
    }

    VAContextID vppContext = VA_INVALID_ID;
    const VAStatus ctxStatus =
        vaCreateContext(vaDisplay, vppConfig, kProbeSurfaceSize, kProbeSurfaceSize, 0, &probeSurface, 1, &vppContext);

    if (ctxStatus != VA_STATUS_SUCCESS) [[unlikely]] {
        (void)std::fprintf(stderr, "  WARNING: vaCreateContext failed (%s) -- cannot query filters\n",
                           vaErrorStr(ctxStatus));
        vaDestroySurfaces(vaDisplay, &probeSurface, 1);
        vaDestroyConfig(vaDisplay, vppConfig);
        vaTerminate(vaDisplay);
        close(renderFd);
        return EXIT_FAILURE;
    }

    // --- VPP filter types ---
    VAProcFilterType filterBuf[VAProcFilterCount];
    auto filterCount = static_cast<unsigned int>(VAProcFilterCount);
    if (vaQueryVideoProcFilters(vaDisplay, vppContext, filterBuf, &filterCount) != VA_STATUS_SUCCESS) {
        filterCount = 0;
    }

    const std::span<const VAProcFilterType> filters{filterBuf, static_cast<size_t>(filterCount)};

    for (const auto &[filterType, filterName] : kVppFilterTypes) {
        PrintCapability(filterName, std::find(filters.begin(), filters.end(), filterType) != filters.end());
    }

    const bool hasHdrToneMapping = ProbeHdrToneMapping(vaDisplay, vppContext, filters);

    PrintColorConversions(vppAvailable, hasP010, hasNV12, hasHdrToneMapping);
    ProbeDeinterlacing(vaDisplay, vppContext);

    // --- Cleanup ---
    vaDestroyContext(vaDisplay, vppContext);
    vaDestroySurfaces(vaDisplay, &probeSurface, 1);
    vaDestroyConfig(vaDisplay, vppConfig);
    vaTerminate(vaDisplay);
    close(renderFd);

    return EXIT_SUCCESS;
}
