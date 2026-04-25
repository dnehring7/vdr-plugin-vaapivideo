// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file caps.cpp
 * @brief Hardware capability probes (stateless)
 */

#include "caps.h"

#include "common.h"

// C++ standard library
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// Platform
#include <fcntl.h>
#include <unistd.h>

// FFmpeg (codec id enumeration used by AudioSinkCaps::Supports)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
extern "C" {
#include <libavcodec/codec_id.h>
}
#pragma GCC diagnostic pop

// VAAPI
#include <va/va.h>
#include <va/va_drm.h>
#include <va/va_vpp.h>

// VDR
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <vdr/tools.h>
#pragma GCC diagnostic pop

// ============================================================================
// === INTERNAL HELPERS ===
// ============================================================================

namespace {

/// Returns true iff @p profile supports VLD (bitstream decode) with @p rtFormat.
/// Both are required: VLD without the RT format means the driver can decode to a
/// surface class the filter chain cannot consume. `rtFormat` is VA_RT_FORMAT_YUV420
/// (8-bit) or VA_RT_FORMAT_YUV420_10 (10-bit / HDR).
[[nodiscard]] auto HasVldDecode(VADisplay display, VAProfile profile, unsigned int rtFormat) noexcept -> bool {
    const int maxEp = vaMaxNumEntrypoints(display);
    if (maxEp <= 0) [[unlikely]] {
        return false;
    }

    std::vector<VAEntrypoint> entrypoints(static_cast<size_t>(maxEp));
    int epCount = 0;
    if (vaQueryConfigEntrypoints(display, profile, entrypoints.data(), &epCount) != VA_STATUS_SUCCESS) {
        return false;
    }

    // Driver bug guard: epCount > maxEp would walk past the allocated buffer.
    const size_t validCount = (epCount > 0) ? std::min(static_cast<size_t>(epCount), entrypoints.size()) : size_t{0};
    const std::span<const VAEntrypoint> valid{entrypoints.data(), validCount};
    if (std::ranges::find(valid, VAEntrypointVLD) == valid.end()) {
        return false;
    }

    VAConfigAttrib attrib{};
    attrib.type = VAConfigAttribRTFormat;
    if (vaGetConfigAttributes(display, profile, VAEntrypointVLD, &attrib, 1) != VA_STATUS_SUCCESS) {
        return false;
    }

    return (attrib.value & rtFormat) != 0U;
}

/// Probes whether the driver will allocate a surface with exactly this FourCC.
/// RT-format flags (YUV420, YUV420_10) are class masks -- a driver may satisfy
/// the class but substitute YV12/I010 for NV12/P010. scale_vaapi and the DRM
/// plane require the exact FourCC, so the RT-class probe alone is insufficient.
[[nodiscard]] auto CanCreateSurfaceFourcc(VADisplay display, unsigned int rtFormat, uint32_t fourcc) noexcept -> bool {
    // NOLINTBEGIN(bugprone-invalid-enum-default-initialization, cppcoreguidelines-pro-type-union-access)
    // VASurfaceAttrib::value is a tagged union (VAGenericValue) with no zero enumerator;
    // brace-init zeroes it, then we overwrite. Union access is the only libva ABI to set a FourCC.
    VASurfaceAttrib attrib{};
    attrib.type = VASurfaceAttribPixelFormat;
    attrib.flags = VA_SURFACE_ATTRIB_SETTABLE;
    attrib.value.type = VAGenericValueTypeInteger;
    attrib.value.value.i = static_cast<int>(fourcc);
    // NOLINTEND(bugprone-invalid-enum-default-initialization, cppcoreguidelines-pro-type-union-access)

    VASurfaceID surface = VA_INVALID_SURFACE;
    if (vaCreateSurfaces(display, rtFormat, 64, 64, &surface, 1, &attrib, 1) != VA_STATUS_SUCCESS) {
        return false;
    }
    vaDestroySurfaces(display, &surface, 1);
    return true;
}

/// RAII guard for the probe's throwaway VADisplay + render fd.
/// Destruction order: vaTerminate before close(fd), as libva requires.
class ProbeVaDisplay {
  public:
    ProbeVaDisplay(VADisplay display, int fd) noexcept : display_{display}, fd_{fd} {}
    ~ProbeVaDisplay() noexcept {
        if (display_) {
            vaTerminate(display_);
        }
        if (fd_ >= 0) {
            close(fd_);
        }
    }
    ProbeVaDisplay(const ProbeVaDisplay &) = delete;
    ProbeVaDisplay(ProbeVaDisplay &&) = delete;
    auto operator=(const ProbeVaDisplay &) -> ProbeVaDisplay & = delete;
    auto operator=(ProbeVaDisplay &&) -> ProbeVaDisplay & = delete;

  private:
    VADisplay display_;
    int fd_;
};

} // namespace

// ============================================================================
// === DisplayCaps METHODS ===
// ============================================================================

auto DisplayCaps::CanDriveHdrPlane() const noexcept -> bool {
    // All seven KMS conditions must hold before we attempt an HDR atomic commit.
    // Missing any one causes the commit to silently fall back or EINVAL at the kernel.
    // Sink EDID flags are NOT checked here; HdrMode::On bypasses the sink check.
    return hasHdrOutputMetadata && hasColorspaceEnum && colorspaceBt2020Ycc && hasMaxBpc && maxBpcSupported >= 10 &&
           planeSupportsP010 && planeColorEncodingValid && planeColorEncodingBt2020;
}

auto DisplayCaps::SupportsHdrKind(StreamHdrKind kind) const noexcept -> bool {
    if (!CanDriveHdrPlane()) {
        return false;
    }
    switch (kind) {
        case StreamHdrKind::Sdr:
            return true;
        case StreamHdrKind::Hdr10:
            return sinkHdr10Pq && sinkBt2020Ycc;
        case StreamHdrKind::Hlg:
            return sinkHlg && sinkBt2020Ycc;
    }
    return false;
}

// ============================================================================
// === AudioSinkCaps METHODS ===
// ============================================================================

auto AudioSinkCaps::Supports(AVCodecID codec) const noexcept -> bool {
    switch (codec) {
        case AV_CODEC_ID_AC3:
            return ac3;
        case AV_CODEC_ID_EAC3:
            return eac3;
        case AV_CODEC_ID_DTS:
            return dts || dtshd; // DTS-HD decoder handles core DTS streams too
        case AV_CODEC_ID_TRUEHD:
            return truehd;
        case AV_CODEC_ID_AC4:
            return ac4;
        case AV_CODEC_ID_MPEGH_3D_AUDIO:
            return mpegh;
        default:
            return false;
    }
}

// ============================================================================
// === ProbeGpuCaps ===
// ============================================================================

auto ProbeGpuCaps(std::string_view renderNode) noexcept -> std::optional<GpuCaps> {
    GpuCaps caps{};

    // Open a dedicated render fd for probing. Reusing FFmpeg's VADisplay tickles an
    // iHD 24.x bug where vaCreateContext starts failing intermittently once the probe
    // context is destroyed (likely a refcount issue in the driver's internal state).
    const std::string renderPath{renderNode};
    const int renderFd = open(renderPath.c_str(), O_RDWR | O_CLOEXEC);
    if (renderFd < 0) [[unlikely]] {
        esyslog("vaapivideo/caps: VPP probe failed -- cannot open render node: %s", std::strerror(errno));
        return std::nullopt;
    }

    VADisplay vaDisplay = vaGetDisplayDRM(renderFd);
    if (!vaDisplay) [[unlikely]] {
        esyslog("vaapivideo/caps: VPP probe failed -- vaGetDisplayDRM error");
        close(renderFd);
        return std::nullopt;
    }

    int vaMajor = 0;
    int vaMinor = 0;
    if (vaInitialize(vaDisplay, &vaMajor, &vaMinor) != VA_STATUS_SUCCESS) [[unlikely]] {
        esyslog("vaapivideo/caps: VPP probe failed -- vaInitialize error");
        close(renderFd);
        return std::nullopt;
    }

    const ProbeVaDisplay vaRaii{vaDisplay, renderFd}; // vaTerminate + close on every exit path

    if (const char *vendorStr = vaQueryVendorString(vaDisplay); vendorStr != nullptr) {
        caps.vendorName = vendorStr;
    }
    isyslog("vaapivideo/caps: VA-API driver -- %s", caps.vendorName.empty() ? "(unknown)" : caps.vendorName.c_str());

    // Probe 8-bit and 10-bit independently per codec: a driver may list Main10 without
    // Main (e.g. HEVC Main10-only on some iHD builds), so both bits need independent
    // confirmation. Only broadcast codecs relevant to VDR are probed.
    const int maxProfiles = vaMaxNumProfiles(vaDisplay);
    if (maxProfiles <= 0) [[unlikely]] {
        esyslog("vaapivideo/caps: vaMaxNumProfiles failed");
        return std::nullopt;
    }
    std::vector<VAProfile> profiles(static_cast<size_t>(maxProfiles));
    int numProfiles = 0;
    if (vaQueryConfigProfiles(vaDisplay, profiles.data(), &numProfiles) == VA_STATUS_SUCCESS) {
        // Driver bug guard: numProfiles > maxProfiles would walk past the buffer.
        const size_t validProfiles =
            (numProfiles > 0) ? std::min(static_cast<size_t>(numProfiles), profiles.size()) : size_t{0};
        for (size_t i = 0; i < validProfiles; ++i) {
            const VAProfile profile =
                profiles[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                             // -- bounded by validProfiles (clamped against profiles.size())
            switch (profile) {
                case VAProfileMPEG2Simple:
                case VAProfileMPEG2Main:
                    if (!caps.hwMpeg2 && HasVldDecode(vaDisplay, profile, VA_RT_FORMAT_YUV420)) {
                        caps.hwMpeg2 = true;
                    }
                    break;
                case VAProfileH264ConstrainedBaseline:
                case VAProfileH264Main:
                case VAProfileH264High:
                    if (!caps.hwH264 && HasVldDecode(vaDisplay, profile, VA_RT_FORMAT_YUV420)) {
                        caps.hwH264 = true;
                    }
                    break;
                case VAProfileH264High10:
                    // High10 decoders can usually decode 8-bit streams too; probe both.
                    if (!caps.hwH264 && HasVldDecode(vaDisplay, profile, VA_RT_FORMAT_YUV420)) {
                        caps.hwH264 = true;
                    }
                    if (!caps.hwH264High10 && HasVldDecode(vaDisplay, profile, VA_RT_FORMAT_YUV420_10)) {
                        caps.hwH264High10 = true;
                    }
                    break;
                case VAProfileHEVCMain:
                    if (!caps.hwHevc && HasVldDecode(vaDisplay, profile, VA_RT_FORMAT_YUV420)) {
                        caps.hwHevc = true;
                    }
                    break;
                case VAProfileHEVCMain10:
                    // Main10 decoders can handle 8-bit Main streams; probe both in case
                    // the driver lists only Main10 (observed on some iHD configurations).
                    if (!caps.hwHevc && HasVldDecode(vaDisplay, profile, VA_RT_FORMAT_YUV420)) {
                        caps.hwHevc = true;
                    }
                    if (!caps.hwHevcMain10 && HasVldDecode(vaDisplay, profile, VA_RT_FORMAT_YUV420_10)) {
                        caps.hwHevcMain10 = true;
                    }
                    break;
                case VAProfileAV1Profile0:
                    if (!caps.hwAv1 && HasVldDecode(vaDisplay, profile, VA_RT_FORMAT_YUV420)) {
                        caps.hwAv1 = true;
                    }
                    if (!caps.hwAv1Main10 && HasVldDecode(vaDisplay, profile, VA_RT_FORMAT_YUV420_10)) {
                        caps.hwAv1Main10 = true;
                    }
                    break;
                default:
                    break;
            }
        }
    }

    isyslog("vaapivideo/caps: decode -- mpeg2=%s h264=%s/%s hevc=%s/%s av1=%s/%s", caps.hwMpeg2 ? "hw" : "sw",
            caps.hwH264 ? "hw" : "sw", caps.hwH264High10 ? "hw10" : "sw10", caps.hwHevc ? "hw" : "sw",
            caps.hwHevcMain10 ? "hw10" : "sw10", caps.hwAv1 ? "hw" : "sw", caps.hwAv1Main10 ? "hw10" : "sw10");

    // vaQueryVideoProcFilters requires a live context (config + surface + context).
    // Build a minimal 64x64 one for the probe, destroyed explicitly before vaRaii runs.
    VAConfigID configId = VA_INVALID_ID;
    if (vaCreateConfig(vaDisplay, VAProfileNone, VAEntrypointVideoProc, nullptr, 0, &configId) != VA_STATUS_SUCCESS)
        [[unlikely]] {
        esyslog("vaapivideo/caps: VAEntrypointVideoProc unavailable -- cannot initialize");
        return std::nullopt;
    }

    // 64x64: iHD SIGSEGVs on zero-size dimensions; some VPP entrypoints reject
    // sub-macroblock sizes. NV12 is required: scale_vaapi outputs NV12 on the SDR
    // path and the DRM video plane consumes it -- no fallback exists if absent.
    if (!CanCreateSurfaceFourcc(vaDisplay, VA_RT_FORMAT_YUV420, VA_FOURCC_NV12)) [[unlikely]] {
        vaDestroyConfig(vaDisplay, configId);
        esyslog("vaapivideo/caps: VPP probe failed -- driver does not allocate NV12 surfaces "
                "(YUV420 class may still work via other FourCC, but the plugin needs NV12)");
        return std::nullopt;
    }
    VASurfaceID surface = VA_INVALID_SURFACE;
    if (vaCreateSurfaces(vaDisplay, VA_RT_FORMAT_YUV420, 64, 64, &surface, 1, nullptr, 0) != VA_STATUS_SUCCESS)
        [[unlikely]] {
        vaDestroyConfig(vaDisplay, configId);
        esyslog("vaapivideo/caps: VPP probe failed -- vaCreateSurfaces error");
        return std::nullopt;
    }

    // P010 probe is non-fatal: absence disables HDR passthrough but SDR playback
    // is unaffected (10-bit streams are downconverted to NV12 via scale_vaapi).
    // Probe the FourCC explicitly -- a driver accepting YUV420_10 class may return
    // I010 rather than P010, which the filter chain cannot consume.
    caps.vppP010 = CanCreateSurfaceFourcc(vaDisplay, VA_RT_FORMAT_YUV420_10, VA_FOURCC_P010);

    VAContextID contextId = VA_INVALID_ID;
    if (vaCreateContext(vaDisplay, configId, 64, 64, 0, &surface, 1, &contextId) != VA_STATUS_SUCCESS) [[unlikely]] {
        vaDestroySurfaces(vaDisplay, &surface, 1);
        vaDestroyConfig(vaDisplay, configId);
        esyslog("vaapivideo/caps: VPP probe failed -- vaCreateContext error");
        return std::nullopt;
    }

    std::array<VAProcFilterType, VAProcFilterCount> filters{};
    auto numFilters = static_cast<unsigned int>(filters.size());
    if (vaQueryVideoProcFilters(vaDisplay, contextId, filters.data(), &numFilters) != VA_STATUS_SUCCESS) {
        numFilters = 0;
    }

    // Driver bug guard: same overflow risk as numProfiles / epCount above.
    const std::span<const VAProcFilterType> validFilters{filters.data(),
                                                         std::min(static_cast<size_t>(numFilters), filters.size())};
    for (const auto filter : validFilters) {
        if (filter == VAProcFilterNoiseReduction) {
            caps.vppDenoise = true;
        } else if (filter == VAProcFilterSharpening) {
            caps.vppSharpness = true;
        }
    }

    // Quality order: motion_compensated > motion_adaptive > weave > bob.
    // The decoder passes this string to deinterlace_vaapi on the HW-decode path.
    // Empty string means no HW deinterlacer; SW-decode path always uses bwdif.
    std::array<VAProcFilterCapDeinterlacing, VAProcDeinterlacingCount> deintCaps{};
    auto numDeintCaps = static_cast<unsigned int>(deintCaps.size());
    if (vaQueryVideoProcFilterCaps(vaDisplay, contextId, VAProcFilterDeinterlacing, deintCaps.data(), &numDeintCaps) ==
        VA_STATUS_SUCCESS) {
        static constexpr struct {
            VAProcDeinterlacingType type;
            const char *name;
        } kDeintModes[] = {
            {.type = VAProcDeinterlacingMotionCompensated, .name = "motion_compensated"},
            {.type = VAProcDeinterlacingMotionAdaptive, .name = "motion_adaptive"},
            {.type = VAProcDeinterlacingWeave, .name = "weave"},
            {.type = VAProcDeinterlacingBob, .name = "bob"},
        };

        // Driver bug guard: same overflow risk as numFilters / numProfiles.
        const std::span<const VAProcFilterCapDeinterlacing> validDeintCaps{
            deintCaps.data(), std::min(static_cast<size_t>(numDeintCaps), deintCaps.size())};
        for (const auto &[type, name] : kDeintModes) {
            if (std::ranges::any_of(validDeintCaps, [type](const auto &c) -> bool { return c.type == type; })) {
                caps.deinterlaceMode = name;
                break;
            }
        }
    }

    vaDestroyContext(vaDisplay, contextId);
    vaDestroySurfaces(vaDisplay, &surface, 1);
    vaDestroyConfig(vaDisplay, configId);
    // vaRaii destructor runs vaTerminate + close(fd) here.

    isyslog("vaapivideo/caps: VPP -- denoise=%s sharpen=%s deinterlace=%s p010=%s", caps.vppDenoise ? "yes" : "no",
            caps.vppSharpness ? "yes" : "no", caps.deinterlaceMode.empty() ? "none" : caps.deinterlaceMode.c_str(),
            caps.vppP010 ? "yes" : "no");
    return caps;
}
