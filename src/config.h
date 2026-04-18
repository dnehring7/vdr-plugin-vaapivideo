// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file config.h
 * @brief Plugin configuration and setup storage
 */

#ifndef VDR_VAAPIVIDEO_CONFIG_H
#define VDR_VAAPIVIDEO_CONFIG_H

#include <atomic>
#include <cstdint>
#include <string>

// ============================================================================
// === CONSTANTS ===
// ============================================================================

inline constexpr double DISPLAY_DEFAULT_ASPECT_RATIO =
    16.0 / 9.0;                                              ///< Fallback aspect ratio when display height is zero
inline constexpr uint32_t DISPLAY_DEFAULT_HEIGHT = 1080;     ///< Default display height before a mode is selected (px)
inline constexpr uint32_t DISPLAY_DEFAULT_WIDTH = 1920;      ///< Default display width before a mode is selected (px)
inline constexpr uint32_t DISPLAY_DEFAULT_REFRESH_RATE = 50; ///< Default refresh rate before a mode is selected (Hz)

// ============================================================================
// === DISPLAY CONFIGURATION ===
// ============================================================================

/// Desired display output parameters; populated from the --resolution command-line argument.
struct DisplayConfig {
    // ========================================================================
    // === DATA MEMBERS ===
    // ========================================================================
    uint32_t outputHeight{DISPLAY_DEFAULT_HEIGHT};      ///< Active display height (px)
    uint32_t outputWidth{DISPLAY_DEFAULT_WIDTH};        ///< Active display width (px)
    uint32_t refreshRate{DISPLAY_DEFAULT_REFRESH_RATE}; ///< Active refresh rate (Hz)

    // ========================================================================
    // === ACCESSORS ===
    // ========================================================================
    [[nodiscard]] auto GetAspectRatio() const noexcept
        -> double; ///< Return width/height ratio; falls back to DISPLAY_DEFAULT_ASPECT_RATIO when height is zero
    [[nodiscard]] auto GetHeight() const noexcept -> uint32_t {
        return outputHeight;
    } ///< Return active display height (px)
    [[nodiscard]] auto GetRefreshRate() const noexcept -> uint32_t {
        return refreshRate;
    } ///< Return active refresh rate (Hz)
    [[nodiscard]] auto GetWidth() const noexcept -> uint32_t {
        return outputWidth;
    } ///< Return active display width (px)

    // ========================================================================
    // === MUTATORS ===
    // ========================================================================
    [[nodiscard]] auto ParseResolution(const char *resolutionStr)
        -> bool; ///< Parse and apply a "WIDTHxHEIGHT@RATE" string; logs and returns false on error
};

// ============================================================================
// === AUDIO PASSTHROUGH MODE ===
// ============================================================================

/// User policy for IEC61937 audio passthrough. Numeric values are part of the setup.conf
/// wire format -- do not renumber. See README for the full user-facing description.
enum class PassthroughMode : std::uint8_t {
    Auto = 0, ///< Passthrough iff the sink advertises support in the ELD
    On = 1,   ///< Force passthrough for every IEC61937-wrappable codec; ignore the ELD
    Off = 2,  ///< Never passthrough; always decode to PCM
};

/// Lowercase wire-format label for a PassthroughMode. Single source of truth shared by
/// config.cpp (setup.conf parse/log) and vaapivideo.cpp (setup-menu labels).
[[nodiscard]] constexpr auto PassthroughModeName(PassthroughMode mode) noexcept -> const char * {
    switch (mode) {
        case PassthroughMode::Auto:
            return "auto";
        case PassthroughMode::On:
            return "on";
        case PassthroughMode::Off:
            return "off";
    }
    return "?"; // unreachable for a valid enum value; silences control-reaches-end warning
}

// ============================================================================
// === HDR PASSTHROUGH MODE ===
// ============================================================================

/// User policy for HDR10 / HLG output passthrough. Numeric values are part of the
/// setup.conf wire format -- do not renumber. See README for the full description.
enum class HdrMode : std::uint8_t {
    Auto = 0, ///< Passthrough iff stream is HDR AND GPU and sink both advertise HDR support
    On = 1,   ///< Force HDR output when the stream is HDR; skip the sink-capability gate
    Off = 2,  ///< Never passthrough; always use the existing SDR BT.709 output path
};

/// Lowercase wire-format label for an HdrMode. Single source of truth shared by
/// config.cpp (setup.conf parse/log) and vaapivideo.cpp (setup-menu labels).
[[nodiscard]] constexpr auto HdrModeName(HdrMode mode) noexcept -> const char * {
    switch (mode) {
        case HdrMode::Auto:
            return "auto";
        case HdrMode::On:
            return "on";
        case HdrMode::Off:
            return "off";
    }
    return "?"; // unreachable for a valid enum value; silences control-reaches-end warning
}

// ============================================================================
// === PLUGIN CONFIGURATION ===
// ============================================================================

/// Top-level plugin configuration; populated from VDR setup.conf and command-line arguments.
struct VaapiConfig {
    // ========================================================================
    // === DATA MEMBERS ===
    // ========================================================================
    std::atomic<bool> clearOnChannelSwitch{false}; ///< Paint a black frame on channel switch (pmNone teardown) instead
                                                   ///< of leaving the previous channel's last frame on screen
    DisplayConfig display;                         ///< Desired display output parameters
    std::atomic<HdrMode> hdrMode{HdrMode::Auto};   ///< User policy for HDR10/HLG passthrough; re-read on every codec
                                                   ///< change / filter-graph rebuild
    std::atomic<int> passthroughLatency{0}; ///< A/V offset (ms, signed) when audio is in IEC61937 passthrough; positive
                                            ///< delays audio relative to video, negative shifts audio earlier (read by
                                            ///< decode thread)
    std::atomic<PassthroughMode> passthroughMode{
        PassthroughMode::Auto};     ///< User policy for IEC61937 passthrough; re-read on every codec change
    std::atomic<int> pcmLatency{0}; ///< A/V offset (ms, signed) when audio is decoded to PCM; positive delays audio
                                    ///< relative to video, negative shifts audio earlier (read by decode thread)

    // ========================================================================
    // === METHODS ===
    // ========================================================================
    [[nodiscard]] auto GetSummary() const
        -> std::string; ///< Return a human-readable one-line summary of the current configuration
    [[nodiscard]] auto SetupParse(const char *name, const char *value)
        -> bool; ///< Handle a VDR setup.conf key/value pair; returns true when the key is recognized
};

// ============================================================================
// === LATENCY BOUNDS ===
// ============================================================================

inline constexpr int CONFIG_AUDIO_LATENCY_MIN_MS = -200; ///< Lower bound for PCM/passthrough latency compensation (ms)
inline constexpr int CONFIG_AUDIO_LATENCY_MAX_MS = 200;  ///< Upper bound for PCM/passthrough latency compensation (ms)

// ============================================================================
// === GLOBAL INSTANCE ===
// ============================================================================

extern VaapiConfig vaapiConfig; ///< Singleton plugin configuration; written once at startup, then read-only

#endif // VDR_VAAPIVIDEO_CONFIG_H
