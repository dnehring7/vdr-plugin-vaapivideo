// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file caps.cpp
 * @brief Hardware capability derivation (GPU, display, audio sink); stateless
 *
 * One section per capability domain, in caps.h struct order (GPU, display, audio).
 * Each section co-locates its file-local helpers (anonymous namespace) with the one
 * public entry point that consumes them, so a domain reads top-to-bottom in isolation.
 * The pure byte-blob parsers (EDID, ELD) live here; the live-handle I/O that feeds them
 * stays with the resource owner (cVaapiDisplay's DRM walk, cAudioProcessor's ALSA read).
 */

#include "caps.h"

#include "common.h"
#include "config.h"

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
#include <utility>
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
// === GPU CAPABILITIES (VAAPI decode + VPP) ===
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
    // brace-init zeroes it; the union is then overwritten. Union access is the only libva ABI to set a FourCC.
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

    // fd-only RAII guard; released to ProbeVaDisplay below once vaInitialize succeeds.
    struct FdGuard {
        int fd;
        explicit FdGuard(int f) noexcept : fd{f} {}
        ~FdGuard() noexcept {
            if (fd >= 0) {
                close(fd);
            }
        }
        FdGuard(const FdGuard &) = delete;
        FdGuard(FdGuard &&) = delete;
        auto operator=(const FdGuard &) -> FdGuard & = delete;
        auto operator=(FdGuard &&) -> FdGuard & = delete;
        [[nodiscard]] auto Release() noexcept -> int {
            const int r = fd;
            fd = -1;
            return r;
        }
    };
    FdGuard fdGuard{renderFd};

    // -Wanalyzer-fd-leak false positive on libva calls; FdGuard above is the real guard.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wanalyzer-fd-leak"
#endif
    VADisplay vaDisplay = vaGetDisplayDRM(renderFd);
    if (!vaDisplay) [[unlikely]] {
        esyslog("vaapivideo/caps: VPP probe failed -- vaGetDisplayDRM error");
        return std::nullopt;
    }

    int vaMajor = 0;
    int vaMinor = 0;
    if (vaInitialize(vaDisplay, &vaMajor, &vaMinor) != VA_STATUS_SUCCESS) [[unlikely]] {
        esyslog("vaapivideo/caps: VPP probe failed -- vaInitialize error");
        return std::nullopt;
    }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

    // Transfer ownership: ProbeVaDisplay now closes fd in vaTerminate-then-close order.
    const ProbeVaDisplay vaRaii{vaDisplay, fdGuard.Release()};

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
                case VAProfileVP9Profile0:
                    if (!caps.hwVp9 && HasVldDecode(vaDisplay, profile, VA_RT_FORMAT_YUV420)) {
                        caps.hwVp9 = true;
                    }
                    break;
                case VAProfileVP9Profile2:
                    // Profile 2 is the 10/12-bit YUV 4:2:0 profile (HDR-capable VP9).
                    if (!caps.hwVp9Profile2 && HasVldDecode(vaDisplay, profile, VA_RT_FORMAT_YUV420_10)) {
                        caps.hwVp9Profile2 = true;
                    }
                    break;
                    // HEVC range extensions / SCC / 12-bit are out of scope: the pipeline is
                    // 4:2:0 only and the scanout path doesn't ingest 12-bit P012 surfaces.
                case VAProfileVVCMain10:
                    // VVC / H.266 successor to HEVC. Per ITU-T H.266 the Main 10 profile
                    // also accepts 8-bit streams, so probe YUV420 in addition to YUV420_10
                    // and route via separate caps (mirrors AV1's hwAv1 / hwAv1Main10 split).
                    // lavc gates the actual HW open via avcodec_get_hw_config; if the lavc
                    // build doesn't ship vvc_vaapi yet, hwVvc/hwVvcMain10=true are harmless
                    // (HW open returns false, decoder falls back to SW).
                    if (!caps.hwVvc && HasVldDecode(vaDisplay, profile, VA_RT_FORMAT_YUV420)) {
                        caps.hwVvc = true;
                    }
                    if (!caps.hwVvcMain10 && HasVldDecode(vaDisplay, profile, VA_RT_FORMAT_YUV420_10)) {
                        caps.hwVvcMain10 = true;
                    }
                    break;
                default:
                    break;
            }
        }
    }

    isyslog("vaapivideo/caps: decode -- mpeg2=%s h264=%s/%s hevc=%s/%s av1=%s/%s vp9=%s/%s vvc=%s/%s",
            caps.hwMpeg2 ? "hw" : "sw", caps.hwH264 ? "hw" : "sw", caps.hwH264High10 ? "hw10" : "sw10",
            caps.hwHevc ? "hw" : "sw", caps.hwHevcMain10 ? "hw10" : "sw10", caps.hwAv1 ? "hw" : "sw",
            caps.hwAv1Main10 ? "hw10" : "sw10", caps.hwVp9 ? "hw" : "sw", caps.hwVp9Profile2 ? "hw10" : "sw10",
            caps.hwVvc ? "hw" : "sw", caps.hwVvcMain10 ? "hw10" : "sw10");

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
        // Mode names come from config.h's VppDeintModeArg() so the probe, ClampDeinterlaceMode
        // (filter.cpp), and the diagnostic log all share one source of truth for the quality order.
        static constexpr struct {
            VAProcDeinterlacingType type;
            VppDeintMode mode;
        } kDeintModes[] = {
            {.type = VAProcDeinterlacingMotionCompensated, .mode = VppDeintMode::MotionCompensated},
            {.type = VAProcDeinterlacingMotionAdaptive, .mode = VppDeintMode::MotionAdaptive},
            {.type = VAProcDeinterlacingWeave, .mode = VppDeintMode::Weave},
            {.type = VAProcDeinterlacingBob, .mode = VppDeintMode::Bob},
        };

        // Driver bug guard: same overflow risk as numFilters / numProfiles.
        const std::span<const VAProcFilterCapDeinterlacing> validDeintCaps{
            deintCaps.data(), std::min(static_cast<size_t>(numDeintCaps), deintCaps.size())};
        // Mask EVERY advertised mode, not just the best: ClampDeinterlaceMode needs to know which lower
        // modes actually exist (some iHD GPUs advertise only motion_compensated). Descending quality
        // order means the first hit is also the default best mode.
        for (const auto &[type, mode] : kDeintModes) {
            if (std::ranges::any_of(validDeintCaps, [type](const auto &c) -> bool { return c.type == type; })) {
                caps.deinterlaceModeMask |= 1U << static_cast<unsigned>(mode);
                if (caps.deinterlaceMode.empty()) {
                    caps.deinterlaceMode = VppDeintModeArg(mode);
                }
            }
        }
    }

    vaDestroyContext(vaDisplay, contextId);
    vaDestroySurfaces(vaDisplay, &surface, 1);
    vaDestroyConfig(vaDisplay, configId);
    // vaRaii destructor runs vaTerminate + close(fd) here.

    // Log the full advertised set so ClampDeinterlaceMode's choice on limited drivers is explainable.
    std::string deintModes;
    for (int rank = 0; rank < CONFIG_VPP_DEINT_MODE_COUNT; ++rank) {
        if ((caps.deinterlaceModeMask & (1U << static_cast<unsigned>(rank))) != 0) {
            if (!deintModes.empty()) {
                deintModes += ',';
            }
            deintModes += VppDeintModeArg(static_cast<VppDeintMode>(rank));
        }
    }
    isyslog("vaapivideo/caps: VPP -- denoise=%s sharpen=%s deinterlace=%s (best=%s) p010=%s",
            caps.vppDenoise ? "yes" : "no", caps.vppSharpness ? "yes" : "no",
            deintModes.empty() ? "none" : deintModes.c_str(),
            caps.deinterlaceMode.empty() ? "none" : caps.deinterlaceMode.c_str(), caps.vppP010 ? "yes" : "no");
    return caps;
}

// ============================================================================
// === DISPLAY CAPABILITIES (KMS + EDID HDR) ===
// ============================================================================

auto DisplayCaps::CanDriveHdrPlane() const noexcept -> bool {
    // All seven KMS conditions must hold before attempting an HDR atomic commit.
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

namespace {

// CTA-861-G sec.7.5.13 HDR Static Metadata Data Block (extended tag 0x06):
//   byte 1 = supported EOTFs bitmap: bit 0=SDR, bit 1=HDR gamma, bit 2=PQ, bit 3=HLG.
// CTA-861-G sec.7.5.6 Colorimetry Data Block (extended tag 0x05):
//   byte 1 = colorimetry flags: bit 6=BT.2020 YCC.
constexpr uint8_t EDID_EOTF_PQ = 1U << 2;
constexpr uint8_t EDID_EOTF_HLG = 1U << 3;
constexpr uint8_t EDID_COLORIMETRY_BT2020_YCC = 1U << 6;
constexpr size_t EDID_BLOCK_SIZE = 128;             ///< Every EDID block is exactly 128 bytes
constexpr size_t EDID_CHECKSUM_OFFSET = 127;        ///< Byte 127 is the block checksum (not a data block)
constexpr size_t EDID_EXTENSION_COUNT_OFFSET = 126; ///< Byte 126 of base block: number of extension blocks

/// Parse one 128-byte CTA-861 extension block (tag byte 0x02) for HDR EOTF and BT.2020 YCC.
/// Per CTA-861-G sec.7.3, a DTD offset of 0 means "no DTDs"; data blocks span [4, checksum).
/// Malformed blocks are silently ignored. Results are OR'd into @p caps so multiple extension
/// blocks accumulate rather than overwrite.
auto ParseCtaExtension(std::span<const uint8_t> ext, DisplayCaps &caps) noexcept -> void {
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
        if (offset + 1 + payloadLen > dataBlockEnd) {
            break;
        }
        const std::span<const uint8_t> payload = ext.subspan(offset + 1, payloadLen);
        // CTA-861-G sec.7.5: blockTag==7 = "Extended Tag" block; payload[0] is the extended tag.
        if (blockTag == 7 && !payload.empty()) {
            const uint8_t extTag = payload[0];
            if (extTag == 0x06 && payload.size() >= 3) {
                // HDR Static Metadata Data Block (sec.7.5.13): payload[1] = EOTF bitmap.
                caps.sinkHdr10Pq = caps.sinkHdr10Pq || (payload[1] & EDID_EOTF_PQ) != 0;
                caps.sinkHlg = caps.sinkHlg || (payload[1] & EDID_EOTF_HLG) != 0;
            } else if (extTag == 0x05 && payload.size() >= 2) {
                // Colorimetry Data Block (sec.7.5.6): payload[1] = colorimetry bitmap.
                caps.sinkBt2020Ycc = caps.sinkBt2020Ycc || (payload[1] & EDID_COLORIMETRY_BT2020_YCC) != 0;
            }
        }
        offset += 1 + payloadLen;
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
}

} // namespace

auto ParseEdidHdrCaps(std::span<const uint8_t> edid, DisplayCaps &caps) noexcept -> void {
    if (edid.size() < EDID_BLOCK_SIZE) {
        return;
    }
    // Bounded by edid.size() >= EDID_BLOCK_SIZE (128) check above; the offset is < 128.
    const size_t extCount =
        edid[EDID_EXTENSION_COUNT_OFFSET]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
    for (size_t i = 0; i < extCount; ++i) {
        const size_t extOffset = EDID_BLOCK_SIZE * (i + 1);
        if (extOffset + EDID_BLOCK_SIZE > edid.size()) {
            break;
        }
        ParseCtaExtension(edid.subspan(extOffset, EDID_BLOCK_SIZE), caps);
    }
}

// ============================================================================
// === AUDIO SINK CAPABILITIES (CEA-861 ELD) ===
// ============================================================================

auto AudioSinkCaps::Supports(AVCodecID codec) const noexcept -> bool {
    switch (codec) {
        case AV_CODEC_ID_AC3:
            return ac3;
        case AV_CODEC_ID_EAC3:
            return eac3;
        case AV_CODEC_ID_DTS:
            return dts || dtshd; // DTS-HD-capable sinks are valid for DTS core
        case AV_CODEC_ID_TRUEHD:
            return truehd;
        case AV_CODEC_ID_AC4:
            return ac4;
        case AV_CODEC_ID_MPEGH_3D_AUDIO:
            return mpegh;
        // AV_CODEC_ID_AAC / AV_CODEC_ID_AAC_LATM deliberately fall through: the `aac` cap is
        // probed for diagnostics only, and current routing keeps both codecs on the PCM path.
        default:
            return false;
    }
}

namespace {

// CEA-861 ELD fixed-header length (bytes 0-19); see kernel sound/hda/hda_eld.c.
constexpr unsigned ELD_FIXED_HEADER = 20;
constexpr unsigned SAD_SIZE = 3; ///< Each CEA-861 Short Audio Descriptor is 3 bytes.

// Audio Format Codes (AFC): CEA-861-D Table 37 / CTA-861-H Table 38. SAD byte 0 bits [6:3].
constexpr uint8_t CEA_LPCM = 0x01;     ///< Linear PCM (carries the sink's PCM channel/rate caps)
constexpr uint8_t CEA_AC3 = 0x02;      ///< Dolby Digital (AC-3)
constexpr uint8_t CEA_AAC = 0x06;      ///< AAC (0x05 = MPEG-2 multichannel, 0x08 = ATRAC)
constexpr uint8_t CEA_DTS = 0x07;      ///< DTS Coherent Acoustics
constexpr uint8_t CEA_EAC3 = 0x0A;     ///< Dolby Digital Plus (E-AC-3)
constexpr uint8_t CEA_DTSHD = 0x0B;    ///< DTS-HD Master Audio
constexpr uint8_t CEA_TRUEHD = 0x0C;   ///< Dolby TrueHD (Atmos is a metadata layer on top)
constexpr uint8_t CEA_EXTENDED = 0x0F; ///< Extended format escape -- read EAFC from byte 2 [7:3]

// EAFC AAC-family advertisements (CTA-861 Audio Coding Extension Type). 0x01/0x02 are the legacy
// CEA-861-E HE-AAC codes; 0x04-0x0A are the MPEG-4 variants. All map to AudioSinkCaps::aac.
constexpr uint8_t CEA_EXT_HE_AAC = 0x01;        ///< HE-AAC (legacy)
constexpr uint8_t CEA_EXT_HE_AAC_V2 = 0x02;     ///< HE-AAC v2 (legacy)
constexpr uint8_t CEA_EXT_MP4_HE_AAC = 0x04;    ///< MPEG-4 HE-AAC
constexpr uint8_t CEA_EXT_MP4_HE_AAC_V2 = 0x05; ///< MPEG-4 HE-AAC v2
constexpr uint8_t CEA_EXT_MP4_AAC_LC = 0x06;    ///< MPEG-4 AAC LC
constexpr uint8_t CEA_EXT_MP4_HE_AAC_MS = 0x08; ///< MPEG-4 HE-AAC + MPEG Surround
constexpr uint8_t CEA_EXT_MP4_AAC_LC_MS = 0x0A; ///< MPEG-4 AAC LC + MPEG Surround
constexpr uint8_t CEA_EXT_MPEGH_3D = 0x0B;      ///< MPEG-H 3D Audio
constexpr uint8_t CEA_EXT_AC4 = 0x0C;           ///< Dolby AC-4

// CEA-861 LPCM SAD byte 1: sample-rate support bitmap (bit -> Hz), ascending.
constexpr std::array<std::pair<uint8_t, int>, 7> LPCM_RATE_BITS{{
    {0x01, 32000},
    {0x02, 44100},
    {0x04, 48000},
    {0x08, 88200},
    {0x10, 96000},
    {0x20, 176400},
    {0x40, 192000},
}};

} // namespace

auto ParseEldSinkCaps(std::span<const uint8_t> eld) noexcept -> std::optional<AudioSinkCaps> {
    if (eld.size() < ELD_FIXED_HEADER) {
        return std::nullopt;
    }
    // An all-zero blob is a transient/disconnected ELD, not a real PCM-only sink (a valid ELD has a
    // non-zero version/length header). Reject it so ProbeSinkCaps keeps scanning other indices and
    // Multichannel mode isn't pinned to stereo by a bogus pcmMaxChannels=2.
    if (std::ranges::all_of(eld, [](uint8_t byte) noexcept -> bool { return byte == 0; })) {
        return std::nullopt;
    }

    AudioSinkCaps caps{};
    caps.elded = true;

    // Every eld[] index below is bounds-checked: the size >= ELD_FIXED_HEADER gate above covers bytes
    // 4/5/7; the sadOffset + sadCount*SAD_SIZE gate covers each SAD's 3 bytes.
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

    // ELD layout (kernel sound/hda/hda_eld.c):
    //   Byte 4 [4:0] = MNL (Monitor Name Length)
    //   Byte 5 [7:4] = SAD count
    //   Byte 7 [6:0] = CEA-861 Speaker Allocation
    //   Bytes 20 .. 20+MNL-1 = monitor name string
    //   Bytes 20+MNL .. = Short Audio Descriptors (3 bytes each)
    caps.speakerAlloc = static_cast<uint8_t>(eld[7] & 0x7FU);

    const unsigned mnl = eld[4] & 0x1FU;
    const unsigned sadCount = (eld[5] >> 4) & 0x0FU;
    const unsigned sadOffset = ELD_FIXED_HEADER + mnl;

    if (sadCount == 0) {
        return caps; // PCM-only sink: valid ELD, no compressed formats; baseline defaults stand.
    }
    if (sadOffset + (sadCount * SAD_SIZE) > eld.size()) {
        return std::nullopt; // truncated ELD; let the caller fall back to defaults
    }

    for (unsigned i = 0; i < sadCount; ++i) {
        const size_t sadAt = sadOffset + (i * SAD_SIZE);
        const uint8_t sadByte0 = eld[sadAt];     // [6:3] = format code (AFC), [2:0] = max channels - 1
        const uint8_t sadByte1 = eld[sadAt + 1]; // LPCM: sample-rate support bitmap
        const uint8_t formatCode = (sadByte0 >> 3) & 0x0FU;
        const unsigned channels = (sadByte0 & 0x07U) + 1U;

        switch (formatCode) {
            case CEA_LPCM: {
                // Stays in the documented [2, 8] range by construction: the field starts at 2 and
                // channels = (byte0 & 0x07) + 1 is at most 8, so no explicit clamp is needed.
                caps.pcmMaxChannels = static_cast<uint8_t>(std::max<unsigned>(caps.pcmMaxChannels, channels));
                for (const auto &[bit, hz] : LPCM_RATE_BITS) {
                    if ((sadByte1 & bit) != 0) {
                        caps.pcmRates.push_back(hz);
                    }
                }
                break;
            }
            case CEA_AC3:
                caps.ac3 = true;
                break;
            case CEA_AAC:
                // Diagnostic only: recorded but never enables passthrough (AAC stays on the PCM path).
                caps.aac = true;
                break;
            case CEA_DTS:
                caps.dts = true;
                break;
            case CEA_EAC3:
                caps.eac3 = true;
                break;
            case CEA_DTSHD:
                caps.dtshd = true;
                break;
            case CEA_TRUEHD:
                caps.truehd = true;
                break;
            case CEA_EXTENDED: {
                const uint8_t extCode = (eld[sadAt + 2] >> 3) & 0x1FU; // SAD byte 2 [7:3] = EAFC
                switch (extCode) {
                    case CEA_EXT_HE_AAC:
                    case CEA_EXT_HE_AAC_V2:
                    case CEA_EXT_MP4_HE_AAC:
                    case CEA_EXT_MP4_HE_AAC_V2:
                    case CEA_EXT_MP4_AAC_LC:
                    case CEA_EXT_MP4_HE_AAC_MS:
                    case CEA_EXT_MP4_AAC_LC_MS:
                        caps.aac = true;
                        break;
                    case CEA_EXT_MPEGH_3D:
                        caps.mpegh = true;
                        break;
                    case CEA_EXT_AC4:
                        caps.ac4 = true;
                        break;
                    default:
                        break;
                }
                break;
            }
            default:
                break;
        }
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

    // A sink may advertise several LPCM SADs (e.g. one per bit depth), repeating rates. Keep
    // pcmRates a sorted, duplicate-free set so it logs cleanly and tests deterministically.
    std::ranges::sort(caps.pcmRates);
    const auto dup = std::ranges::unique(caps.pcmRates);
    caps.pcmRates.erase(dup.begin(), dup.end());

    return caps;
}

auto DescribeSpeakerAlloc(uint8_t alloc) -> std::string {
    // CEA-861 / ELD speaker-allocation bitmask (ELD byte 7, bits 6:0): one bit per speaker group present.
    static constexpr std::array<std::string_view, 7> kSpeakerNames{"FL/FR", "LFE",     "FC",     "RL/RR",
                                                                   "RC",    "FLC/FRC", "RLC/RRC"};
    std::string out;
    for (unsigned bit = 0; bit < kSpeakerNames.size(); ++bit) {
        if ((alloc & (1U << bit)) != 0U) {
            if (!out.empty()) {
                out += ' ';
            }
            out.append(kSpeakerNames.at(bit));
        }
    }
    return out.empty() ? std::string{"none"} : out;
}
