// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file config.cpp
 * @brief DisplayConfig and VaapiConfig: resolution parsing and setup.conf load/store.
 */

#include "config.h"

// POSIX
#include <strings.h>

// C++ Standard Library
#include <atomic>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>

// VDR
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <vdr/tools.h>
#pragma GCC diagnostic pop

// ============================================================================
// === CONSTANTS ===
// ============================================================================

namespace {

// Audio latency bounds live in config.h (CONFIG_AUDIO_LATENCY_{MIN,MAX}_MS) so the parse path
// here and the setup-menu UI share one source of truth -- keep them in lockstep.
constexpr uint32_t CONFIG_MAX_VIDEO_HEIGHT = 2160U; ///< 4K UHD ceiling for ParseResolution() (px)
constexpr uint32_t CONFIG_MAX_VIDEO_WIDTH = 3840U;  ///< 4K UHD ceiling for ParseResolution() (px)

} // namespace

// ============================================================================
// === DISPLAY CONFIGURATION ===
// ============================================================================

[[nodiscard]] auto DisplayConfig::GetAspectRatio() const noexcept -> double {
    if (outputHeight == 0) [[unlikely]] {
        return DISPLAY_DEFAULT_ASPECT_RATIO;
    }
    return static_cast<double>(outputWidth) / static_cast<double>(outputHeight);
}

[[nodiscard]] auto DisplayConfig::ParseResolution(const char *resolutionStr) -> bool {
    if (!resolutionStr || resolutionStr[0] == '\0') [[unlikely]] {
        esyslog("vaapivideo/config: empty resolution string");
        return false;
    }

    // Required format: WIDTHxHEIGHT@RATE (e.g. "1920x1080@50"). No optional fields, no
    // whitespace tolerance -- callers feed values straight from setup.conf / CLI args.
    const char *widthStart = resolutionStr;
    const char *xPos = std::strchr(widthStart, 'x');
    if (!xPos) [[unlikely]] {
        esyslog("vaapivideo/config: invalid resolution format '%s' (missing 'x')", resolutionStr);
        return false;
    }

    const char *heightStart = xPos + 1;
    const char *atPos = std::strchr(heightStart, '@');
    if (!atPos) [[unlikely]] {
        esyslog("vaapivideo/config: invalid resolution format '%s' (missing '@')", resolutionStr);
        return false;
    }

    const char *rateStart = atPos + 1;
    const char *rateEnd = rateStart + std::strlen(rateStart);

    // For each field, require std::from_chars to consume *exactly* up to the delimiter:
    // ec == errc{} alone would accept "1920abc" by stopping at 'a'.
    uint32_t width{};
    auto [ptrW, ecW] = std::from_chars(widthStart, xPos, width);
    if (ecW != std::errc{} || ptrW != xPos) [[unlikely]] {
        esyslog("vaapivideo/config: invalid width in '%s'", resolutionStr);
        return false;
    }

    uint32_t height{};
    auto [ptrH, ecH] = std::from_chars(heightStart, atPos, height);
    if (ecH != std::errc{} || ptrH != atPos) [[unlikely]] {
        esyslog("vaapivideo/config: invalid height in '%s'", resolutionStr);
        return false;
    }

    uint32_t rate{};
    auto [ptrR, ecR] = std::from_chars(rateStart, rateEnd, rate);
    if (ecR != std::errc{} || ptrR != rateEnd) [[unlikely]] {
        esyslog("vaapivideo/config: invalid refresh rate in '%s'", resolutionStr);
        return false;
    }

    // Sanity bounds, not hardware limits: 640x480 / 23 Hz catches 24p content with rounding
    // slack; 4K / 120 Hz is the ceiling the VAAPI/DRM stack is exercised against. Anything
    // outside is almost certainly a typo and would just propagate to a confusing modeset
    // failure later.
    if (width < 640 || width > CONFIG_MAX_VIDEO_WIDTH) [[unlikely]] {
        esyslog("vaapivideo/config: width %u outside valid range [640, %u]", width, CONFIG_MAX_VIDEO_WIDTH);
        return false;
    }
    if (height < 480 || height > CONFIG_MAX_VIDEO_HEIGHT) [[unlikely]] {
        esyslog("vaapivideo/config: height %u outside valid range [480, %u]", height, CONFIG_MAX_VIDEO_HEIGHT);
        return false;
    }
    if (rate < 23 || rate > 120) [[unlikely]] {
        esyslog("vaapivideo/config: refresh rate %u outside valid range [23, 120]", rate);
        return false;
    }

    outputWidth = width;
    outputHeight = height;
    refreshRate = rate;
    isyslog("vaapivideo/config: resolution set to %ux%u@%u (aspect %.3f:1)", outputWidth, outputHeight, refreshRate,
            GetAspectRatio());
    return true;
}

// ============================================================================
// === PLUGIN CONFIGURATION ===
// ============================================================================

[[nodiscard]] auto VaapiConfig::GetSummary() const -> std::string {
    const MediaBookmark bookmarkSnapshot = LoadBookmark(); // locked: a teardown thread may be writing it
    // Zoom levels are stored in tenths-of-% zoom-in factor; render as a human-readable list.
    std::string zoom;
    for (int i = 0; i < CONFIG_ZOOM_PRESET_COUNT; ++i) {
        const int level = zoomLevel[i].load(std::memory_order_relaxed);
        if (level <= 0) {
            zoom += std::format("{}{}=off", i == 0 ? "" : " ", i + 1);
        } else {
            zoom += std::format("{}{}=+{:.1f}%", i == 0 ? "" : " ", i + 1, static_cast<double>(level) / 10.0);
        }
    }
    // Post-processing policies.
    const std::string postProc = std::format("deint={} denoise={} sharpen={} scale={}",
                                             DeinterlaceModeName(deinterlaceMode.load(std::memory_order_relaxed)),
                                             DenoiseModeName(denoiseMode.load(std::memory_order_relaxed)),
                                             SharpenModeName(sharpenMode.load(std::memory_order_relaxed)),
                                             ScaleModeName(scaleMode.load(std::memory_order_relaxed)));
    const std::string mark = bookmarkSnapshot.uri.empty()
                                 ? std::string{"none"}
                                 : std::format("{} @ {}ms", bookmarkSnapshot.uri, bookmarkSnapshot.positionMs);
    // Display mode switching: render the scope switches as a compact source list so an all-off
    // (i.e. legacy) configuration is obvious at a glance.
    std::string modeScope;
    const auto appendScope = [&modeScope](bool enabled, std::string_view label) -> void {
        if (!enabled) {
            return;
        }
        if (!modeScope.empty()) {
            modeScope += '+';
        }
        modeScope += label;
    };
    appendScope(modeSwitchLiveTv.load(std::memory_order_relaxed), "live");
    appendScope(modeSwitchReplay.load(std::memory_order_relaxed), "replay");
    appendScope(modeSwitchMediaplayer.load(std::memory_order_relaxed), "media");
    const std::string modeSwitch = std::format(
        "rate={} res={} min={} max={} scope={}", matchRefreshRate.load(std::memory_order_relaxed) ? "on" : "off",
        matchResolution.load(std::memory_order_relaxed) ? "on" : "off",
        MinResolutionModeName(minResolution.load(std::memory_order_relaxed)),
        MaxRefreshModeName(maxRefreshRate.load(std::memory_order_relaxed)), modeScope.empty() ? "none" : modeScope);
    return std::format("PCM Latency: {}ms, Passthrough Latency: {}ms, Passthrough: {}, PCM channels: {}, HDR: {}, "
                       "Clear on channel switch: {}, Trace: {}, Post-proc: {}, Mode switch: {}, "
                       "Zoom levels (0=off): {}, Bookmark: {}",
                       pcmLatency.load(std::memory_order_relaxed), passthroughLatency.load(std::memory_order_relaxed),
                       PassthroughModeName(passthroughMode.load(std::memory_order_relaxed)),
                       PcmChannelModeName(pcmChannelMode.load(std::memory_order_relaxed)),
                       HdrModeName(hdrMode.load(std::memory_order_relaxed)),
                       clearOnChannelSwitch.load(std::memory_order_relaxed) ? "on" : "off",
                       trace.load(std::memory_order_relaxed) ? "on" : "off", postProc, modeSwitch, zoom, mark);
}

namespace {

/// Don't promise "the default": the target may already hold an earlier duplicate that parsed. The
/// line stays on disk until that key is stored again, and with duplicates SetupStore() drops only
/// the first of them, so more than one save cycle can be needed.
auto RejectValue(const char *key, const char *value, std::string_view why) -> void {
    esyslog("vaapivideo/config: %s value '%s' %.*s -- keeping the current value", key, value,
            static_cast<int>(why.size()), why.data());
}

/// from_chars over the WHOLE string -- stopping at the first bad character would accept "50x" as 50.
[[nodiscard]] auto ParseWholeInt(const char *value) -> std::optional<int> {
    int parsed{};
    const auto *end = value + std::strlen(value);
    const auto [ptr, ec] = std::from_chars(value, end, parsed);
    if (ec != std::errc{} || ptr != end) [[unlikely]] {
        return std::nullopt;
    }
    return parsed;
}

/// Relaxed store: every consumer re-reads on its own cadence (audio latency per packet, zoom per
/// filter rebuild).
auto ParseBoundedIntValue(const char *key, const char *value, std::atomic<int> &target, int min, int max) -> void {
    const auto parsed = ParseWholeInt(value);
    if (!parsed) [[unlikely]] {
        RejectValue(key, value, "is not a number");
        return;
    }
    if (*parsed < min || *parsed > max) [[unlikely]] {
        RejectValue(key, value, std::format("is outside [{},{}]", min, max));
        return;
    }
    target.store(*parsed, std::memory_order_relaxed);
}

/// Only VDR's canonical 0/1 encoding: anything else is a corrupt line, not an implicit false.
auto ParseBoolValue(const char *key, const char *value, std::atomic<bool> &target) -> void {
    const std::string_view v{value};
    if (v != "0" && v != "1") [[unlikely]] {
        RejectValue(key, value, "is not 0 or 1");
        return;
    }
    target.store(v == "1", std::memory_order_relaxed);
}

/// Plain int, not atomic: the bookmark position is touched only here at startup and through the
/// serialized accessors in mediaplayer.cpp.
auto ParseNonNegativeIntValue(const char *key, const char *value, int &target) -> void {
    const auto parsed = ParseWholeInt(value);
    if (!parsed || *parsed < 0) [[unlikely]] {
        RejectValue(key, value, "is not a non-negative number");
        return;
    }
    target = *parsed;
}

/// Enum indices are contiguous from zero -- that is what cMenuEditStraItem writes -- so @p count
/// alone bounds them.
template <typename EnumT>
auto ParseEnumValue(const char *key, const char *value, std::atomic<EnumT> &target, int count) -> void {
    const auto parsed = ParseWholeInt(value);
    if (!parsed) [[unlikely]] {
        RejectValue(key, value, "is not a number");
        return;
    }
    if (*parsed < 0 || *parsed >= count) [[unlikely]] {
        RejectValue(key, value, std::format("is outside [0,{}]", count - 1));
        return;
    }
    target.store(static_cast<EnumT>(*parsed), std::memory_order_relaxed);
}

} // namespace

[[nodiscard]] auto VaapiConfig::SetupParse(const char *name, const char *value) -> bool {
    if (!name || !value) [[unlikely]] {
        return false;
    }

    // Matched with strcasecmp like the PLUGINS.html reference implementation, and because VDR's own
    // lookup (cSetupLine::Compare) is: exact matching here would ignore a hand-edited
    // "vaapivideo.maxrefreshrate" that Store() then happily updates. Key spellings must track the
    // SetupStore() calls in vaapivideo.cpp -- a typo drops the setting silently.
    if (strcasecmp(name, "PcmLatency") == 0) {
        ParseBoundedIntValue("PcmLatency", value, pcmLatency, CONFIG_AUDIO_LATENCY_MIN_MS, CONFIG_AUDIO_LATENCY_MAX_MS);
        return true;
    }
    if (strcasecmp(name, "PassthroughLatency") == 0) {
        ParseBoundedIntValue("PassthroughLatency", value, passthroughLatency, CONFIG_AUDIO_LATENCY_MIN_MS,
                             CONFIG_AUDIO_LATENCY_MAX_MS);
        return true;
    }
    if (strcasecmp(name, "PassthroughMode") == 0) {
        ParseEnumValue("PassthroughMode", value, passthroughMode, CONFIG_PASSTHROUGH_MODE_COUNT);
        return true;
    }
    if (strcasecmp(name, "HdrMode") == 0) {
        ParseEnumValue("HdrMode", value, hdrMode, CONFIG_HDR_MODE_COUNT);
        return true;
    }
    if (strcasecmp(name, "PcmChannelMode") == 0) {
        ParseEnumValue("PcmChannelMode", value, pcmChannelMode, CONFIG_PCM_CHANNEL_MODE_COUNT);
        return true;
    }
    if (strcasecmp(name, "ClearOnChannelSwitch") == 0) {
        ParseBoolValue("ClearOnChannelSwitch", value, clearOnChannelSwitch);
        return true;
    }
    if (strcasecmp(name, "BookmarkUri") == 0) {
        bookmark.uri = value; // free-form path/URL; validated at use (browser open / StartPlayback)
        return true;
    }
    if (strcasecmp(name, "BookmarkPositionMs") == 0) {
        ParseNonNegativeIntValue("BookmarkPositionMs", value, bookmark.positionMs);
        return true;
    }
    if (strcasecmp(name, "DeinterlaceMode") == 0) {
        ParseEnumValue("DeinterlaceMode", value, deinterlaceMode, CONFIG_DEINTERLACE_MODE_COUNT);
        return true;
    }
    if (strcasecmp(name, "DenoiseMode") == 0) {
        ParseEnumValue("DenoiseMode", value, denoiseMode, CONFIG_DENOISE_MODE_COUNT);
        return true;
    }
    if (strcasecmp(name, "SharpenMode") == 0) {
        ParseEnumValue("SharpenMode", value, sharpenMode, CONFIG_SHARPEN_MODE_COUNT);
        return true;
    }
    if (strcasecmp(name, "ScaleMode") == 0) {
        ParseEnumValue("ScaleMode", value, scaleMode, CONFIG_SCALE_MODE_COUNT);
        return true;
    }
    // Display mode switching: both match policies and all three scopes default to off, so a
    // setup.conf predating the feature keeps the legacy fixed-mode behaviour.
    if (strcasecmp(name, "MatchRefreshRate") == 0) {
        ParseBoolValue("MatchRefreshRate", value, matchRefreshRate);
        return true;
    }
    if (strcasecmp(name, "MatchResolution") == 0) {
        ParseBoolValue("MatchResolution", value, matchResolution);
        return true;
    }
    if (strcasecmp(name, "MinResolution") == 0) {
        ParseEnumValue("MinResolution", value, minResolution, CONFIG_MIN_RESOLUTION_MODE_COUNT);
        return true;
    }
    if (strcasecmp(name, "MaxRefreshRate") == 0) {
        ParseEnumValue("MaxRefreshRate", value, maxRefreshRate, CONFIG_MAX_REFRESH_MODE_COUNT);
        return true;
    }
    if (strcasecmp(name, "ModeSwitchLiveTv") == 0) {
        ParseBoolValue("ModeSwitchLiveTv", value, modeSwitchLiveTv);
        return true;
    }
    if (strcasecmp(name, "ModeSwitchMediaplayer") == 0) {
        ParseBoolValue("ModeSwitchMediaplayer", value, modeSwitchMediaplayer);
        return true;
    }
    if (strcasecmp(name, "ModeSwitchReplay") == 0) {
        ParseBoolValue("ModeSwitchReplay", value, modeSwitchReplay);
        return true;
    }
    // Keys Zoom1..Zoom<N>, written by the SetupStore() loop in vaapivideo.cpp. zoomActive is
    // deliberately not persisted: zoom is transient and must reset to Off on every restart.
    static_assert(CONFIG_ZOOM_PRESET_COUNT <= 9, "key decode below reads a single preset digit");
    if (strncasecmp(name, "Zoom", 4) == 0 && name[4] != '\0' && name[5] == '\0') {
        const int preset = name[4] - '1';
        if (preset >= 0 && preset < CONFIG_ZOOM_PRESET_COUNT) {
            ParseBoundedIntValue(name, value, zoomLevel[preset], CONFIG_ZOOM_LEVEL_MIN, CONFIG_ZOOM_LEVEL_MAX);
            return true;
        }
    }

    // Not ours -- a key a newer build wrote, or an older one dropped. VDR keeps the line, so a
    // build that owns it again still finds its value.
    return false;
}

// ============================================================================
// === GLOBAL INSTANCE ===
// ============================================================================

VaapiConfig vaapiConfig; // process-wide singleton declared extern in config.h
