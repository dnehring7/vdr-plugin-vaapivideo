// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file caps.h
 * @brief Consolidated hardware capability snapshots (GPU, display, audio sink)
 *
 * Populated once at hardware attach by the Probe...() functions in caps.cpp,
 * then read-only for the session lifetime. On hotplug the structs are replaced
 * wholesale -- no partial updates. Each struct is a plain value type; callers
 * own their copy.
 *
 * Design invariants:
 *  - DisplayCaps carries capability bits only. DRM property IDs needed for
 *    atomic commits (DrmPlaneProps, HdrConnectorProps, ModesetProps) are
 *    mutable runtime state and stay inside cVaapiDisplay.
 *  - AudioSinkCaps provides a single source of truth for both passthrough
 *    eligibility and the PCM fallback path (rates, channel count).
 *  - GpuCaps replaces the flat hwXxx/hasXxx fields that were previously on VaapiContext.
 */

#ifndef VDR_VAAPIVIDEO_CAPS_H
#define VDR_VAAPIVIDEO_CAPS_H

#include "common.h"

#include <optional>

// ============================================================================
// === GPU CAPABILITIES ===
// ============================================================================

/// VAAPI decode + VPP capabilities probed once from the render node.
/// Separate from the FFmpeg hardware handle (AVBufferRef *hwDeviceRef),
/// which stays on VaapiContext and must not be touched here.
struct GpuCaps {
    // === DECODE (codec x profile x bit-depth) ===
    // Each flag means: driver advertises VLD entrypoint AND the required RT surface
    // format for that profile. Both conditions are necessary; VLD alone is insufficient
    // if the driver won't allocate NV12/P010 surfaces for the filter chain.
    bool hwMpeg2{};      ///< VAProfileMPEG2Simple or Main, VLD + YUV420
    bool hwH264{};       ///< VAProfileH264High (or lower), VLD + YUV420 (8-bit)
    bool hwH264High10{}; ///< VAProfileH264High10, VLD + YUV420_10 (10-bit)
    bool hwHevc{};       ///< VAProfileHEVCMain, VLD + YUV420 (8-bit)
    bool hwHevcMain10{}; ///< VAProfileHEVCMain10, VLD + YUV420_10 (10-bit / HDR prerequisite)
    bool hwAv1{};        ///< VAProfileAV1Profile0, VLD + YUV420 (8-bit)
    bool hwAv1Main10{};  ///< VAProfileAV1Profile0, VLD + YUV420_10 (10-bit)

    // === VPP FILTERS ===
    bool vppP010{};              ///< Driver allocates P010 (YUV420_10) surfaces; HDR passthrough prerequisite
    bool vppDenoise{};           ///< VAProcFilterNoiseReduction available
    bool vppSharpness{};         ///< VAProcFilterSharpening available
    std::string deinterlaceMode; ///< Best VAProcDeinterlacing mode name for deinterlace_vaapi; "" = none

    // === DIAGNOSTICS ===
    std::string vendorName; ///< vaQueryVendorString output; logged at startup only
};

// ============================================================================
// === DISPLAY CAPABILITIES ===
// ============================================================================

/// DRM/KMS output + HDMI sink capability bits. Populated from the DRM plane
/// IN_FORMATS blob, connector properties, and the CTA-861 EDID data block.
/// Does NOT carry DRM property IDs -- those live in cVaapiDisplay.
struct DisplayCaps {
    // === DRM PLANE ===
    bool planeSupportsP010{};        ///< DRM_FORMAT_P010 in video plane IN_FORMATS blob
    bool planeColorEncodingValid{};  ///< COLOR_ENCODING enum contains BT.709 (SDR baseline; required even for HDR)
    bool planeColorEncodingBt2020{}; ///< COLOR_ENCODING enum contains BT.2020 (HDR flip target)

    // === CONNECTOR PROPERTIES (presence) ===
    bool hasHdrOutputMetadata{}; ///< HDR_OUTPUT_METADATA blob property exists on connector
    bool hasColorspaceEnum{};    ///< Colorspace enum property exists on connector
    bool hasMaxBpc{};            ///< "max bpc" range property exists and has >= 2 values
    uint8_t maxBpcSupported{8};  ///< Max value accepted by "max bpc"; clamped to [8, 16] at probe time

    // === CONNECTOR COLORSPACE ENUM ===
    bool colorspaceBt2020Ycc{}; ///< BT2020_YCC enum value exists and differs from Default

    // === EDID CTA-861 SINK FLAGS ===
    bool sinkHdr10Pq{};   ///< HDR Static Metadata block advertises EOTF SMPTE ST 2084 (HDR10)
    bool sinkHlg{};       ///< HDR Static Metadata block advertises EOTF ARIB STD-B67 (HLG)
    bool sinkBt2020Ycc{}; ///< Colorimetry Data Block advertises BT.2020 Y'CbCr

    /// True iff all KMS prerequisites for an HDR atomic commit are present.
    /// Does NOT consult sink EDID flags; HdrMode::On intentionally bypasses the sink check.
    [[nodiscard]] auto CanDriveHdrPlane() const noexcept -> bool;

    /// True iff both the display hardware and the sink declare support for @p kind.
    /// Used by HdrMode::Auto; HdrMode::On skips this and relies on CanDriveHdrPlane().
    [[nodiscard]] auto SupportsHdrKind(StreamHdrKind kind) const noexcept -> bool;
};

// ============================================================================
// === AUDIO SINK CAPABILITIES ===
// ============================================================================

/// HDMI sink audio capabilities from the ALSA ELD (CEA-861 Short Audio Descriptors).
/// Struct has safe defaults when @c elded is false (PCM stereo 48 kHz), so callers
/// can always make a routing decision. The flag distinguishes "sink reports no
/// compressed formats" from "ELD unreadable".
struct AudioSinkCaps {
    bool elded{false}; ///< True when ProbeAudioSinkCaps successfully parsed an ELD

    // === PCM BASELINE ===
    std::vector<int> pcmRates; ///< Supported PCM sample rates (Hz), ascending; empty => treat as {48000}
    uint8_t pcmMaxChannels{2}; ///< Max linear-PCM channel count; range 2..8

    // === IEC61937 PASSTHROUGH FORMATS ===
    bool ac3{};    ///< Dolby Digital (AC-3)
    bool eac3{};   ///< Dolby Digital Plus (E-AC-3)
    bool dts{};    ///< DTS core
    bool dtshd{};  ///< DTS-HD (superset; decoder also handles DTS core)
    bool truehd{}; ///< Dolby TrueHD
    bool ac4{};    ///< AC-4
    bool mpegh{};  ///< MPEG-H 3D Audio

    /// True iff the sink advertises IEC61937 passthrough for @p codec.
    /// Returns false for codecs not listed above (PCM, AAC, MP2, etc.) -- those
    /// are handled by the PCM decode path, never via passthrough.
    [[nodiscard]] auto Supports(AVCodecID codec) const noexcept -> bool;
};

// ============================================================================
// === PROBE FUNCTIONS ===
// ============================================================================

/// Probe VAAPI decode profiles and VPP filter capabilities from @p renderNode.
/// Opens a dedicated throwaway VADisplay -- reusing FFmpeg's VADisplay tickles
/// an iHD 24.x bug where vaCreateContext fails intermittently after the first
/// probe context is destroyed. Returns std::nullopt on hard failures (render
/// node unreadable, vaInitialize error, VPP entrypoint missing); the caller
/// must abort hardware attach. A returned struct with all hw* flags false is
/// valid: VPP is operational but no codec was confirmed usable for HW decode.
[[nodiscard]] auto ProbeGpuCaps(std::string_view renderNode) noexcept -> std::optional<GpuCaps>;

#endif // VDR_VAAPIVIDEO_CAPS_H
