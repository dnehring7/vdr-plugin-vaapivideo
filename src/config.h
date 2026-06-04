// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file config.h
 * @brief Plugin configuration: resolution parsing + setup.conf load/store.
 */

#ifndef VDR_VAAPIVIDEO_CONFIG_H
#define VDR_VAAPIVIDEO_CONFIG_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

// ============================================================================
// === CONSTANTS ===
// ============================================================================

inline constexpr int64_t PTS_TICKS_PER_MS =
    90; ///< DVB PTS clock: 90 ticks per ms. Equals VDR's PTSTICKS / 1000 (remux.h); a static_assert
        ///< in audio.cpp pins the relationship. Kept literal here so config.h stays free of vdr/remux.h.
inline constexpr double DISPLAY_DEFAULT_ASPECT_RATIO =
    16.0 / 9.0;                                              ///< Fallback aspect ratio when display height is zero
inline constexpr uint32_t DISPLAY_DEFAULT_HEIGHT = 1080;     ///< Default display height before a mode is selected (px)
inline constexpr uint32_t DISPLAY_DEFAULT_WIDTH = 1920;      ///< Default display width before a mode is selected (px)
inline constexpr uint32_t DISPLAY_DEFAULT_REFRESH_RATE = 50; ///< Default refresh rate before a mode is selected (Hz)
inline constexpr size_t DISPLAY_PRERENDER_SLOTS =
    8; ///< Decoder->display handoff queue depth (= 160 ms tolerance @ 50 fps). Sized to absorb a
       ///< single UHD VPP/memory-bandwidth spike (observed ~80 ms in replay) AND the per-frame
       ///< variance of CPU-side SW decoders (libdav1d 1080p50 spikes 30-40 ms on complex frames)
       ///< without draining the cache and forcing a re-present. SubmitFrame blocks when all slots
       ///< are full so audio clock stays in lipsync (the whole pipeline is delayed in lockstep, not
       ///< just video). FHD HW paths never fill past 1-2 slots; the extra depth is a no-op there.

// ============================================================================
// === DISPLAY CONFIGURATION ===
// ============================================================================

/// Desired display output parameters; populated once from the --resolution CLI argument.
/// Not thread-safe after init -- all writes happen before any thread reads these fields.
struct DisplayConfig {
    uint32_t outputHeight{DISPLAY_DEFAULT_HEIGHT};      ///< Active display height (px)
    uint32_t outputWidth{DISPLAY_DEFAULT_WIDTH};        ///< Active display width (px)
    uint32_t refreshRate{DISPLAY_DEFAULT_REFRESH_RATE}; ///< Active refresh rate (Hz)

    [[nodiscard]] auto GetAspectRatio() const noexcept
        -> double; ///< width/height ratio; falls back to DISPLAY_DEFAULT_ASPECT_RATIO when height is zero
    [[nodiscard]] auto GetHeight() const noexcept -> uint32_t { return outputHeight; }
    [[nodiscard]] auto GetRefreshRate() const noexcept -> uint32_t { return refreshRate; }
    [[nodiscard]] auto GetWidth() const noexcept -> uint32_t { return outputWidth; }

    /// Parse and apply "WIDTHxHEIGHT@RATE"; logs via esyslog and returns false on any error.
    [[nodiscard]] auto ParseResolution(const char *resolutionStr) -> bool;
};

// ============================================================================
// === AUDIO PASSTHROUGH MODE ===
// ============================================================================

/// User policy for IEC61937 audio passthrough. Numeric values are part of the setup.conf
/// wire format -- do not renumber. See README for the full user-facing description.
enum class PassthroughMode : uint8_t {
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
enum class HdrMode : uint8_t {
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
// === ZOOM BOUNDS ===
// ============================================================================

inline constexpr int CONFIG_ZOOM_PRESET_COUNT = 5; ///< Editable zoom levels; cycling skips 0 and adds an Off stop
inline constexpr int CONFIG_ZOOM_LEVEL_MIN = 0;   ///< Min zoom-in factor (tenths-of-%, 0 = disabled / skipped in cycle)
inline constexpr int CONFIG_ZOOM_LEVEL_MAX = 499; ///< Max zoom-in factor (tenths-of-%, = +49.9% / 1.499x)

// ============================================================================
// === PLUGIN CONFIGURATION ===
// ============================================================================

/// Top-level plugin configuration; populated from VDR setup.conf and --resolution CLI arg.
/// `display` is written once at startup and then read-only. The atomic fields may be
/// re-written from the VDR main thread at any time via the setup menu; consumers on other
/// threads use relaxed loads (scalar tunables on slow paths, no ordering dependency).
struct VaapiConfig {
    std::atomic<bool> clearOnChannelSwitch{false}; ///< Black frame on channel switch instead of leaving the last frame
    DisplayConfig display;                         ///< Display geometry; init-time only, not thread-safe after that
    std::atomic<HdrMode> hdrMode{HdrMode::Auto};   ///< Re-read on every codec change / filter-graph rebuild
    std::atomic<int> passthroughLatency{0};        ///< A/V offset (ms, signed) for IEC61937 passthrough; + delays audio
    std::atomic<PassthroughMode> passthroughMode{PassthroughMode::Auto}; ///< Re-read on every codec change
    std::atomic<int> pcmLatency{0}; ///< A/V offset (ms, signed) for PCM decode path; + delays audio
    std::atomic<int> zoomActive{
        0}; ///< Runtime cycle stop (0=Off, 1..ZOOM_PRESET_COUNT=level); transient, never persisted
    // A zoom level is a zoom-in factor in tenths-of-% (344 = +34.4%, the picture enlarged 1.344x);
    // the equal per-side crop that yields it is derived in the decoder, and the kept region refills
    // the screen (aspect preserved). 0 disables a level and skips it while cycling. Defaults fill the
    // two common theatrical ratios on a 16:9 screen -- 1 = 2.39:1 scope (+34.4%), 2 = 2.00:1 (+12.5%);
    // 3-5 off.
    std::atomic<int> zoomLevel[CONFIG_ZOOM_PRESET_COUNT]{344, 125, 0, 0, 0};

    [[nodiscard]] auto GetSummary() const -> std::string; ///< One-line human-readable snapshot for logging
    [[nodiscard]] auto SetupParse(const char *name, const char *value)
        -> bool; ///< Called by VDR for each key in setup.conf; returns true when the key is recognized
};

// ============================================================================
// === LATENCY BOUNDS ===
// ============================================================================

inline constexpr int CONFIG_AUDIO_LATENCY_MIN_MS = -200; ///< Lower bound for PCM/passthrough latency compensation (ms)
inline constexpr int CONFIG_AUDIO_LATENCY_MAX_MS = 200;  ///< Upper bound for PCM/passthrough latency compensation (ms)

// ============================================================================
// === GLOBAL INSTANCE ===
// ============================================================================

extern VaapiConfig vaapiConfig; ///< Singleton plugin configuration; see VaapiConfig for thread-safety contract

#endif // VDR_VAAPIVIDEO_CONFIG_H
