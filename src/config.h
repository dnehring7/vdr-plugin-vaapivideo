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
       ///< COUPLED to display.cpp's UNDERRUN_THRESHOLD_VSYNCS (= SLOTS + 2); revisit that margin if you
       ///< change this (the relationship is not linear -- see the note at that definition).

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
// === VAAPI VPP DEINTERLACE MODES (internal, NOT a user option) ===
// ============================================================================
// This is the hardware (deinterlace_vaapi) mode enum and its ffmpeg argument token. It is distinct
// from the user-facing DeinterlaceMode policy below: caps.cpp probes which VppDeintMode values the
// driver advertises, and filter.cpp's ClampDeinterlaceMode maps the user's Hw* choice onto the best
// advertised one. VppDeintModeArg() returns the bare token fed to "deinterlace_vaapi=mode=..." --
// human labels live in DeinterlaceModeName(), never here.

/// VAAPI VPP deinterlace modes, numbered by descending quality so the numeric value IS the rank
/// (lower = better). Quality order matches the caps probe.
enum class VppDeintMode : uint8_t {
    MotionCompensated = 0, ///< MCDI -- highest quality
    MotionAdaptive = 1,    ///< MADI
    Weave = 2,             ///< field weave
    Bob = 3,               ///< line doubling -- lowest cost
};

inline constexpr int CONFIG_VPP_DEINT_MODE_COUNT = 4; ///< Number of VppDeintMode values; bounds the clamp loop

/// Bare ffmpeg "deinterlace_vaapi=mode=" argument token for a VppDeintMode (MUST stay a bare token --
/// it is concatenated into the filter string). Single source shared by filter.cpp (clamp + emit) and
/// caps.cpp (probe + diagnostic log).
[[nodiscard]] constexpr auto VppDeintModeArg(VppDeintMode mode) noexcept -> const char * {
    switch (mode) {
        case VppDeintMode::MotionCompensated:
            return "motion_compensated";
        case VppDeintMode::MotionAdaptive:
            return "motion_adaptive";
        case VppDeintMode::Weave:
            return "weave";
        case VppDeintMode::Bob:
            return "bob";
    }
    return "?"; // unreachable for a valid enum value; silences control-reaches-end warning
}

// ============================================================================
// === POST-PROCESSING OPTIONS (deinterlace / denoise / sharpen / scale) ===
// ============================================================================
// Four independent user policies. Each "sw-*" / "SwQuality" value routes the whole post-process
// through one hwdownload->[SW filters]->hwupload block (filter.cpp); "Auto"/"Hw*" stay on the
// VAAPI VPP path. Numeric values are part of the setup.conf wire format -- do not renumber.

/// Deinterlacer selection. Auto/Hw* run on the GPU (deinterlace_vaapi, clamped to an advertised
/// VppDeintMode); Sw* force the software block (bwdif / w3fdif). No explicit MCDI entry: Auto already
/// selects the best advertised HW mode (MCDI when present).
enum class DeinterlaceMode : uint8_t {
    Auto = 0,             ///< Best HW mode the driver advertises (MCDI when present)
    HwMotionAdaptive = 1, ///< Request MADI (clamped to an advertised mode)
    HwWeave = 2,          ///< Request weave
    HwBob = 3,            ///< Request bob (cheapest HW)
    SwBwdif = 4,          ///< Software bwdif -> forces SW block
    SwW3fdif = 5,         ///< Software w3fdif -> forces SW block
};

inline constexpr int CONFIG_DEINTERLACE_MODE_COUNT = 6; ///< Number of DeinterlaceMode values

/// Human label for a DeinterlaceMode -- single source for the setup menu AND the log summary.
[[nodiscard]] constexpr auto DeinterlaceModeName(DeinterlaceMode mode) noexcept -> const char * {
    switch (mode) {
        case DeinterlaceMode::Auto:
            return "auto (best available)";
        case DeinterlaceMode::HwMotionAdaptive:
            return "hardware: motion adaptive";
        case DeinterlaceMode::HwWeave:
            return "hardware: weave (fast)";
        case DeinterlaceMode::HwBob:
            return "hardware: bob (fastest)";
        case DeinterlaceMode::SwBwdif:
            return "software: bwdif (best)";
        case DeinterlaceMode::SwW3fdif:
            return "software: w3fdif (faster)";
    }
    return "?";
}

/// Denoise. Auto uses HW denoise_vaapi (codec-tuned) on the GPU path and adds nothing when the chain
/// is already in the SW block. Sw* are software hqdn3d presets that force the SW block. Off emits no
/// node.
enum class DenoiseMode : uint8_t {
    Auto = 0,       ///< HW denoise_vaapi (codec-tuned); no SW fallback
    Off = 1,        ///< No denoise
    SwMinimal = 2,  ///< Software hqdn3d (light) -> forces SW block
    SwEnhanced = 3, ///< Software hqdn3d (strong) -> forces SW block
};

inline constexpr int CONFIG_DENOISE_MODE_COUNT = 4; ///< Number of DenoiseMode values

[[nodiscard]] constexpr auto DenoiseModeName(DenoiseMode mode) noexcept -> const char * {
    switch (mode) {
        case DenoiseMode::Auto:
            return "auto (hardware)";
        case DenoiseMode::Off:
            return "off";
        case DenoiseMode::SwMinimal:
            return "software: light";
        case DenoiseMode::SwEnhanced:
            return "software: strong";
    }
    return "?";
}

/// Sharpening. Auto uses HW sharpness_vaapi; Sw* force the SW block (unsharp). Off emits no node.
enum class SharpenMode : uint8_t {
    Auto = 0,     ///< Codec-tuned HW sharpness
    Off = 1,      ///< No sharpening
    SwMild = 2,   ///< Software unsharp (mild) -> forces SW block
    SwMedium = 3, ///< Software unsharp (medium) -> forces SW block
};

inline constexpr int CONFIG_SHARPEN_MODE_COUNT = 4; ///< Number of SharpenMode values

[[nodiscard]] constexpr auto SharpenModeName(SharpenMode mode) noexcept -> const char * {
    switch (mode) {
        case SharpenMode::Auto:
            return "auto (hardware)";
        case SharpenMode::Off:
            return "off";
        case SharpenMode::SwMild:
            return "software: mild";
        case SharpenMode::SwMedium:
            return "software: medium";
    }
    return "?";
}

/// Scaler. Auto/HwFast run scale_vaapi (with/without :mode=hq); Sw* force the SW block (swscale,
/// lanczos for HQ or bilinear for fast).
enum class ScaleMode : uint8_t {
    Auto = 0,      ///< scale_vaapi :mode=hq (best HW)
    HwFast = 1,    ///< scale_vaapi without :mode=hq
    SwQuality = 2, ///< swscale lanczos -> forces SW block
    SwFast = 3,    ///< swscale bilinear -> forces SW block
};

inline constexpr int CONFIG_SCALE_MODE_COUNT = 4; ///< Number of ScaleMode values

[[nodiscard]] constexpr auto ScaleModeName(ScaleMode mode) noexcept -> const char * {
    switch (mode) {
        case ScaleMode::Auto:
            return "auto (hardware, HQ)";
        case ScaleMode::HwFast:
            return "hardware: fast";
        case ScaleMode::SwQuality:
            return "software: HQ (lanczos)";
        case ScaleMode::SwFast:
            return "software: fast (bilinear)";
    }
    return "?";
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
    // Post-processing policies: re-read on every filter-graph rebuild (alphabetical within group).
    std::atomic<DeinterlaceMode> deinterlaceMode{DeinterlaceMode::Auto}; ///< Deinterlacer selection
    std::atomic<DenoiseMode> denoiseMode{DenoiseMode::Auto};             ///< Denoise strength
    std::atomic<ScaleMode> scaleMode{ScaleMode::Auto};                   ///< Scaler selection
    std::atomic<SharpenMode> sharpenMode{SharpenMode::Auto};             ///< Sharpening selection
    std::atomic<int> passthroughLatency{0}; ///< A/V offset (ms, signed) for IEC61937 passthrough; + delays audio
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
