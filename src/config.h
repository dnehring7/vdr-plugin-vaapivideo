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

/// DVB PTS clock: 90 ticks per ms. Equals VDR's PTSTICKS / 1000 (remux.h); a static_assert in audio.cpp pins the
/// relationship. Kept literal here so config.h stays free of vdr/remux.h.
inline constexpr int64_t PTS_TICKS_PER_MS = 90;
/// Fallback aspect ratio when display height is zero
inline constexpr double DISPLAY_DEFAULT_ASPECT_RATIO = 16.0 / 9.0;
inline constexpr uint32_t DISPLAY_DEFAULT_HEIGHT = 1080;     ///< Default display height before a mode is selected (px)
inline constexpr uint32_t DISPLAY_DEFAULT_WIDTH = 1920;      ///< Default display width before a mode is selected (px)
inline constexpr uint32_t DISPLAY_DEFAULT_REFRESH_RATE = 50; ///< Default refresh rate before a mode is selected (Hz)
// DISPLAY_PRERENDER_SLOTS (decoder->display handoff queue depth) lives in display.cpp, its only user,
// next to the DISPLAY_UNDERRUN_THRESHOLD_VSYNCS margin that is derived from it.

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
    [[nodiscard]] auto GetHeight() const noexcept -> uint32_t { return outputHeight; }     ///< Height (px)
    [[nodiscard]] auto GetRefreshRate() const noexcept -> uint32_t { return refreshRate; } ///< Refresh rate (Hz)
    [[nodiscard]] auto GetWidth() const noexcept -> uint32_t { return outputWidth; }       ///< Width (px)

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

/// Derived from the last enumerator so it cannot drift
inline constexpr int CONFIG_PASSTHROUGH_MODE_COUNT = static_cast<int>(PassthroughMode::Off) + 1;

/// Lowercase wire-format label for a PassthroughMode. These *ModeName() functions are the single source
/// of truth shared by config.cpp (setup.conf parse/log) and vaapivideo.cpp (setup-menu labels).
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
// === PCM CHANNEL MODE ===
// ============================================================================

/// User policy for decoded-PCM channel output (the non-passthrough path). Numeric values are part
/// of the setup.conf wire format -- do not renumber. Never fabricates surround from stereo (output is
/// capped by the decoded stream's channel count); mono is carried as stereo for HDMI/ALSA compatibility.
enum class PcmChannelMode : uint8_t {
    Auto = 0,         ///< Sink-driven: native multichannel up to the ELD's PCM channel cap, else stereo
    Stereo = 1,       ///< Always downmix decoded PCM to 2.0
    Multichannel = 2, ///< Force native multichannel; when no ELD is readable, trust the stream's layout
};

/// Derived from the last enumerator so it cannot drift
inline constexpr int CONFIG_PCM_CHANNEL_MODE_COUNT = static_cast<int>(PcmChannelMode::Multichannel) + 1;

/// Lowercase wire-format label for a PcmChannelMode; same contract as PassthroughModeName().
[[nodiscard]] constexpr auto PcmChannelModeName(PcmChannelMode mode) noexcept -> const char * {
    switch (mode) {
        case PcmChannelMode::Auto:
            return "auto";
        case PcmChannelMode::Stereo:
            return "stereo";
        case PcmChannelMode::Multichannel:
            return "multichannel";
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

/// Derived from the last enumerator so it cannot drift
inline constexpr int CONFIG_HDR_MODE_COUNT = static_cast<int>(HdrMode::Off) + 1;

/// Lowercase wire-format label for an HdrMode; same contract as PassthroughModeName().
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

/// Derived from the last enumerator; bounds the clamp loop
inline constexpr int CONFIG_VPP_DEINT_MODE_COUNT = static_cast<int>(VppDeintMode::Bob) + 1;

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

/// Derived from the last enumerator so it cannot drift
inline constexpr int CONFIG_DEINTERLACE_MODE_COUNT = static_cast<int>(DeinterlaceMode::SwW3fdif) + 1;

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

/// Derived from the last enumerator so it cannot drift
inline constexpr int CONFIG_DENOISE_MODE_COUNT = static_cast<int>(DenoiseMode::SwEnhanced) + 1;

/// Lowercase wire-format label for a DenoiseMode; same contract as PassthroughModeName().
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

/// Derived from the last enumerator so it cannot drift
inline constexpr int CONFIG_SHARPEN_MODE_COUNT = static_cast<int>(SharpenMode::SwMedium) + 1;

/// Lowercase wire-format label for a SharpenMode; same contract as PassthroughModeName().
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

/// Derived from the last enumerator so it cannot drift
inline constexpr int CONFIG_SCALE_MODE_COUNT = static_cast<int>(ScaleMode::SwFast) + 1;

/// Lowercase wire-format label for a ScaleMode; same contract as PassthroughModeName().
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
// === DISPLAY MODE SWITCHING ===
// ============================================================================
// Runtime CRTC mode switching: pick the connector mode that best matches the stream currently
// playing instead of staying on the one --resolution mode forever. Refresh and resolution are
// two independent policies, and each playback source (live TV / recordings / mediaplayer) has
// its own enable switch -- everything defaults to off, so an untouched install behaves exactly
// as before. See cVaapiDevice::EvaluateDisplayMode() for the matcher.

/// Lower bound for the dynamic resolution search, as a display height. Setting this to the
/// panel's native height effectively pins the resolution and leaves only refresh matching
/// active. Numeric values are part of the setup.conf wire format -- do not renumber.
enum class MinResolutionMode : uint8_t {
    P576 = 0,  ///< 576p -- PAL SD; allows SD streams to drive an SD mode
    P720 = 1,  ///< 720p -- default floor; SD content is upscaled to 720p
    P1080 = 2, ///< 1080p -- never drop below FHD
    P2160 = 3, ///< 2160p -- UHD only (pins the resolution on a UHD panel)
};

/// Derived from the last enumerator so it cannot drift
inline constexpr int CONFIG_MIN_RESOLUTION_MODE_COUNT = static_cast<int>(MinResolutionMode::P2160) + 1;

/// Human label for a MinResolutionMode -- single source for the setup menu AND the log summary.
[[nodiscard]] constexpr auto MinResolutionModeName(MinResolutionMode mode) noexcept -> const char * {
    switch (mode) {
        case MinResolutionMode::P576:
            return "576p";
        case MinResolutionMode::P720:
            return "720p";
        case MinResolutionMode::P1080:
            return "1080p";
        case MinResolutionMode::P2160:
            return "2160p";
    }
    return "?"; // unreachable for a valid enum value; silences control-reaches-end warning
}

/// Display height in pixels for a MinResolutionMode; the matcher compares this against each
/// candidate mode's vdisplay.
[[nodiscard]] constexpr auto MinResolutionModeHeight(MinResolutionMode mode) noexcept -> uint32_t {
    switch (mode) {
        case MinResolutionMode::P576:
            return 576U;
        case MinResolutionMode::P720:
            return 720U;
        case MinResolutionMode::P1080:
            return 1080U;
        case MinResolutionMode::P2160:
            return 2160U;
    }
    return 720U; // unreachable for a valid enum value
}

/// Upper bound for the refresh-multiple search. The matcher prefers the HIGHEST exact multiple
/// k*source that stays at or below this cap (25p -> 50 Hz, 29.97 -> 59.94 Hz); k==1 is always
/// allowed even above the cap so a 120 fps source is never forced down. Numeric values are part
/// of the setup.conf wire format -- do not renumber.
enum class MaxRefreshMode : uint8_t {
    Hz50 = 0,      ///< Cap at 50 Hz
    Hz60 = 1,      ///< Cap at 60 Hz (default; the usual CEA ceiling)
    Hz100 = 2,     ///< Cap at 100 Hz
    Hz120 = 3,     ///< Cap at 120 Hz
    Unlimited = 4, ///< No cap -- every frequency the connector offers is a candidate
};

/// Derived from the last enumerator so it cannot drift
inline constexpr int CONFIG_MAX_REFRESH_MODE_COUNT = static_cast<int>(MaxRefreshMode::Unlimited) + 1;

/// Human label for a MaxRefreshMode -- single source for the setup menu AND the log summary.
[[nodiscard]] constexpr auto MaxRefreshModeName(MaxRefreshMode mode) noexcept -> const char * {
    switch (mode) {
        case MaxRefreshMode::Hz50:
            return "50 Hz";
        case MaxRefreshMode::Hz60:
            return "60 Hz";
        case MaxRefreshMode::Hz100:
            return "100 Hz";
        case MaxRefreshMode::Hz120:
            return "120 Hz";
        case MaxRefreshMode::Unlimited:
            return "unlimited";
    }
    return "?"; // unreachable for a valid enum value; silences control-reaches-end warning
}

/// Refresh cap in millihertz for a MaxRefreshMode. Millihertz throughout the mode-matching path:
/// drmModeModeInfo::vrefresh is integer-truncated, so 59.94 and 60 are indistinguishable there.
[[nodiscard]] constexpr auto MaxRefreshModeMilliHz(MaxRefreshMode mode) noexcept -> uint32_t {
    switch (mode) {
        case MaxRefreshMode::Hz50:
            return 50000U;
        case MaxRefreshMode::Hz60:
            return 60000U;
        case MaxRefreshMode::Hz100:
            return 100000U;
        case MaxRefreshMode::Hz120:
            return 120000U;
        case MaxRefreshMode::Unlimited:
            return UINT32_MAX;
    }
    return 60000U; // unreachable for a valid enum value
}

// The DISPLAY_MODE_* runtime mode-switch tunables (stability window, rate-limit, idle restore,
// matcher tolerances) live in device.cpp, their only user.

// ============================================================================
// === ZOOM BOUNDS ===
// ============================================================================

inline constexpr int CONFIG_ZOOM_PRESET_COUNT = 5; ///< Editable zoom levels; cycling skips 0 and adds an Off stop
inline constexpr int CONFIG_ZOOM_LEVEL_MIN = 0;   ///< Min zoom-in factor (tenths-of-%, 0 = disabled / skipped in cycle)
inline constexpr int CONFIG_ZOOM_LEVEL_MAX = 499; ///< Max zoom-in factor (tenths-of-%, = +49.9% / 1.499x)

// ============================================================================
// === MEDIA BOOKMARK ===
// ============================================================================

/// Single-slot resume bookmark, persisted in setup.conf as vaapivideo.BookmarkUri /
/// vaapivideo.BookmarkPositionMs. Direct field access is safe only during startup SetupParse()
/// (single-threaded); once playback is live, go through the mutex-serialized accessors below and in
/// mediaplayer.cpp.
struct MediaBookmark {
    std::string uri;   ///< Origin URI the user selected (file, .m3u path, or URL); empty = none
    int positionMs{0}; ///< Resume position; 0 = from the start (playlist / stream / EOF / no bookmark)
};

/// Thread-safe snapshot of the current bookmark; defined in mediaplayer.cpp with the persistence machinery.
[[nodiscard]] auto LoadBookmark() -> MediaBookmark;

// ============================================================================
// === PLUGIN CONFIGURATION ===
// ============================================================================

/// Top-level plugin configuration; populated from VDR setup.conf and --resolution CLI arg.
/// `display` is written once at startup and then read-only. The atomic fields may be
/// re-written from the VDR main thread at any time via the setup menu; consumers on other
/// threads use relaxed loads (scalar tunables on slow paths, no ordering dependency).
struct VaapiConfig {
    MediaBookmark bookmark; ///< Resume bookmark; access serialized in mediaplayer.cpp -- see MediaBookmark
    std::atomic<bool> clearOnChannelSwitch{false}; ///< Black frame on channel switch instead of leaving the last frame
    DisplayConfig display;                         ///< Display geometry; init-time only, not thread-safe after that
    std::atomic<HdrMode> hdrMode{HdrMode::Auto};   ///< Re-read on every codec change / filter-graph rebuild
    // Post-processing policies: re-read on every filter-graph rebuild (alphabetical within group).
    std::atomic<DeinterlaceMode> deinterlaceMode{DeinterlaceMode::Auto}; ///< Deinterlacer selection
    std::atomic<DenoiseMode> denoiseMode{DenoiseMode::Auto};             ///< Denoise strength
    std::atomic<ScaleMode> scaleMode{ScaleMode::Auto};                   ///< Scaler selection
    std::atomic<SharpenMode> sharpenMode{SharpenMode::Auto};             ///< Sharpening selection
    // Display mode switching: two match policies plus one enable switch per playback source.
    // Read from the decode and player threads on every filter-graph rebuild / entry open.
    std::atomic<bool> matchRefreshRate{false}; ///< Track the stream's frame rate with the CRTC refresh rate
    std::atomic<bool> matchResolution{false};  ///< Track the stream's coded size with the CRTC resolution
    std::atomic<MaxRefreshMode> maxRefreshRate{MaxRefreshMode::Hz60};      ///< Ceiling for the k*source rate search
    std::atomic<MinResolutionMode> minResolution{MinResolutionMode::P720}; ///< Floor for the resolution search
    std::atomic<bool> modeSwitchLiveTv{false};      ///< Allow mode switching while watching live TV
    std::atomic<bool> modeSwitchMediaplayer{false}; ///< Allow mode switching in the plugin's mediaplayer
    std::atomic<bool> modeSwitchReplay{false};      ///< Allow mode switching while replaying recordings
    std::atomic<int> passthroughLatency{0}; ///< A/V offset (ms, signed) for IEC61937 passthrough; + delays audio
    std::atomic<PassthroughMode> passthroughMode{PassthroughMode::Auto}; ///< Re-read on every codec change
    /// Decoded-PCM channel policy; re-read per decoded frame (decode-driven)
    std::atomic<PcmChannelMode> pcmChannelMode{PcmChannelMode::Auto};
    std::atomic<int> pcmLatency{0}; ///< A/V offset (ms, signed) for PCM decode path; + delays audio
    /// Diagnostic tracing (-t / --trace, SVDRP TRACE on|off); never persisted to setup.conf.
    /// Read from the present/decode hot paths via TraceEnabled() -- relaxed, no ordering dependency.
    std::atomic<bool> trace{false};
    /// Runtime cycle stop (0=Off, 1..ZOOM_PRESET_COUNT=level); transient, never persisted
    std::atomic<int> zoomActive{0};
    /// Zoom-in factor per preset, in tenths-of-% (344 = +34.4%, the picture enlarged 1.344x); the equal per-side
    /// crop that yields it is derived in the decoder, and the kept region refills the screen (aspect preserved).
    /// 0 disables a level and skips it while cycling. Defaults fill the two common theatrical ratios on a 16:9
    /// screen -- 1 = 2.39:1 scope (+34.4%), 2 = 2.00:1 (+12.5%); 3-5 off.
    std::atomic<int> zoomLevel[CONFIG_ZOOM_PRESET_COUNT]{344, 125, 0, 0, 0};

    [[nodiscard]] auto GetSummary() const -> std::string; ///< One-line human-readable snapshot for logging
    [[nodiscard]] auto SetupParse(const char *name, const char *value)
        -> bool; ///< Called by VDR per setup.conf key; a key we own returns true even for a bad value,
                 ///< because false is what makes VDR log an unknown parameter. Bad values are logged
                 ///< here and leave the target as it was
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

/// True while diagnostic tracing is on; the gate behind every tsyslog() (see common.h).
[[nodiscard]] inline auto TraceEnabled() noexcept -> bool { return vaapiConfig.trace.load(std::memory_order_relaxed); }

#endif // VDR_VAAPIVIDEO_CONFIG_H
