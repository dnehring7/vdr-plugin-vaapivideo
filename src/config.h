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
// === PLUGIN CONFIGURATION ===
// ============================================================================

/// Top-level plugin configuration; populated from VDR setup.conf and command-line arguments.
struct VaapiConfig {
    // ========================================================================
    // === DATA MEMBERS ===
    // ========================================================================
    std::atomic<int> audioLatency{0}; ///< Extra A/V synchronization offset (ms); positive shifts video earlier to
                                      ///< compensate for external audio delay (read by decode thread)
    DisplayConfig display;            ///< Desired display output parameters

    // ========================================================================
    // === METHODS ===
    // ========================================================================
    [[nodiscard]] auto GetSummary() const
        -> std::string; ///< Return a human-readable one-line summary of the current configuration
    [[nodiscard]] auto SetupParse(const char *name, const char *value)
        -> bool; ///< Handle a VDR setup.conf key/value pair; returns true when the key is recognized
};

// ============================================================================
// === GLOBAL INSTANCE ===
// ============================================================================

extern VaapiConfig vaapiConfig; ///< Singleton plugin configuration; written once at startup, then read-only

#endif // VDR_VAAPIVIDEO_CONFIG_H
