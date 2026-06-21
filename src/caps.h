// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file caps.h
 * @brief Consolidated hardware capability snapshots (GPU, display, audio sink)
 *
 * Populated once at hardware attach, then read-only for the session lifetime. On
 * hotplug the structs are replaced wholesale -- no partial updates. Each struct is
 * a plain value type; callers own their copy.
 *
 * Capability *derivation* lives here (this is the single home for "what can the
 * hardware/sink do"); capability *consumption* lives in the feature modules
 * (decoder, filter, display, audio). The pure byte-blob parsers are here and
 * unit-testable; the live-handle I/O that feeds them stays with the resource owner:
 *  - GpuCaps     -- ProbeGpuCaps() (opens its own throwaway VADisplay; fully here).
 *  - DisplayCaps -- ParseEdidHdrCaps() (pure EDID) here; the DRM connector/plane walk
 *                   that supplies the EDID blob + plane formats is in cVaapiDisplay
 *                   (it is interleaved with DRM property-ID capture, which must stay).
 *  - AudioSinkCaps -- ParseEldSinkCaps() (pure ELD) here; the ALSA control read that
 *                   supplies the ELD bytes is in cAudioProcessor.
 *
 * Design invariants:
 *  - DisplayCaps carries capability bits only. DRM property IDs needed for
 *    atomic commits (DrmPlaneProps, HdrConnectorProps, ModesetProps) are
 *    mutable runtime state and stay inside cVaapiDisplay.
 *  - AudioSinkCaps provides a single source of truth for both passthrough
 *    eligibility and the PCM fallback path (rates, channel count).
 *  - GpuCaps is the single source of truth for hardware codec/VPP capability flags.
 */

#ifndef VDR_VAAPIVIDEO_CAPS_H
#define VDR_VAAPIVIDEO_CAPS_H

#include "common.h"

#include <optional>
#include <span>
#include <string>

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
    bool hwMpeg2{};       ///< VAProfileMPEG2Simple or Main, VLD + YUV420
    bool hwH264{};        ///< VAProfileH264High (or lower), VLD + YUV420 (8-bit)
    bool hwH264High10{};  ///< VAProfileH264High10, VLD + YUV420_10 (10-bit)
    bool hwHevc{};        ///< VAProfileHEVCMain, VLD + YUV420 (8-bit)
    bool hwHevcMain10{};  ///< VAProfileHEVCMain10, VLD + YUV420_10 (10-bit / HDR prerequisite)
    bool hwAv1{};         ///< VAProfileAV1Profile0, VLD + YUV420 (8-bit)
    bool hwAv1Main10{};   ///< VAProfileAV1Profile0, VLD + YUV420_10 (10-bit)
    bool hwVp9{};         ///< VAProfileVP9Profile0, VLD + YUV420 (8-bit)
    bool hwVp9Profile2{}; ///< VAProfileVP9Profile2, VLD + YUV420_10 (10-bit)
    bool hwVvc{};         ///< VAProfileVVCMain10, VLD + YUV420 (8-bit)
    bool hwVvcMain10{};   ///< VAProfileVVCMain10, VLD + YUV420_10 (10-bit)

    // === VPP FILTERS ===
    bool vppP010{};                 ///< Driver allocates P010 (YUV420_10) surfaces; HDR passthrough prerequisite
    bool vppDenoise{};              ///< VAProcFilterNoiseReduction available
    bool vppSharpness{};            ///< VAProcFilterSharpening available
    std::string deinterlaceMode;    ///< Best VAProcDeinterlacing mode name for deinterlace_vaapi; "" = none
    unsigned deinterlaceModeMask{}; ///< Bit (1u<<VppDeintMode) set per supported mode; lets ClampDeinterlaceMode pick
                                    ///< only modes the driver actually advertises (some iHD GPUs expose just one)

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
    std::vector<int> pcmRates; ///< Supported LPCM sample rates (Hz), ascending; empty => treat as {48000}
    uint8_t pcmMaxChannels{2}; ///< Max linear-PCM channel count from the LPCM SAD; range 2..8
    uint8_t speakerAlloc{0};   ///< CEA-861 Speaker Allocation mask (ELD byte 7, bits 6:0): bit0 FL/FR, bit1 LFE,
                               ///< bit2 FC, bit3 RL/RR, bit4 RC, bit5 FLC/FRC, bit6 RLC/RRC. 0 when absent.
                               ///< Stored for diagnostics; the PCM channel cap is driven by pcmMaxChannels.

    // === IEC61937 PASSTHROUGH FORMATS ===
    bool ac3{};    ///< Dolby Digital (AC-3)
    bool eac3{};   ///< Dolby Digital Plus (E-AC-3)
    bool dts{};    ///< DTS core
    bool dtshd{};  ///< Sink advertises DTS-HD; used only as proof DTS core is accepted
    bool truehd{}; ///< Dolby TrueHD
    bool ac4{};    ///< AC-4
    bool mpegh{};  ///< MPEG-H 3D Audio
    bool aac{};    ///< Sink advertises AAC-family support (base CEA AFC 0x06 or AAC-family EAFC).
                   ///< DIAGNOSTIC ONLY: current routing keeps AV_CODEC_ID_AAC and AV_CODEC_ID_AAC_LATM
                   ///< (standard DVB LOAS/LATM) on the PCM path. Deliberately NOT consulted by Supports().

    /// True iff the sink advertises IEC61937 passthrough for @p codec.
    /// Returns false for codecs not listed above (PCM, MP2, etc.) -- those are
    /// handled by the PCM decode path, never via passthrough. AAC is probed into
    /// `aac` for logging but intentionally absent here, so AAC/AAC-LATM stay on PCM.
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

/// OR the sink HDR capability bits from a raw EDID blob into @p caps (sinkHdr10Pq, sinkHlg,
/// sinkBt2020Ycc). Pure: no DRM/system dependency, so it is unit-testable and shared by the display
/// probe. Walks every CTA-861 extension block for the HDR Static Metadata (EOTF) and Colorimetry
/// data blocks. The remaining DisplayCaps fields (DRM plane formats, connector properties) come from
/// the live DRM walk in cVaapiDisplay, so this is a contributor, not the sole producer -- it only
/// touches the EDID-derived bits and leaves the rest of @p caps untouched.
auto ParseEdidHdrCaps(std::span<const uint8_t> edid, DisplayCaps &caps) noexcept -> void;

/// Parse a raw HDMI ELD (the byte blob from the ALSA "ELD" control) into AudioSinkCaps.
/// Pure: no ALSA/system dependency, so it is unit-testable and shared by the audio sink probe.
/// Decodes the CEA-861 Short Audio Descriptors (passthrough format flags + the LPCM SAD's max
/// channels / sample rates) and the ELD byte-7 speaker-allocation mask. Returns std::nullopt when
/// @p eld is too short to hold the fixed header or its SAD block is truncated; the returned struct
/// has @c elded == true (a PCM-only sink with zero SADs is still a valid ELD).
[[nodiscard]] auto ParseEldSinkCaps(std::span<const uint8_t> eld) noexcept -> std::optional<AudioSinkCaps>;

/// Render a CEA-861 speaker-allocation mask (AudioSinkCaps::speakerAlloc) as a human-readable list of the
/// speaker groups it advertises, e.g. "FL/FR LFE FC RL/RR RC RLC/RRC"; "none" when the mask is zero. Pure;
/// used only for the sink-capabilities diagnostic log.
[[nodiscard]] auto DescribeSpeakerAlloc(uint8_t alloc) -> std::string;

#endif // VDR_VAAPIVIDEO_CAPS_H
