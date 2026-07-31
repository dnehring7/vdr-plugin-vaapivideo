// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file config.cpp
 * @brief DisplayConfig and VaapiConfig: resolution parsing and setup.conf load/store.
 */

#include "config.h"

// C++ Standard Library
#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

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
                       "Clear on channel switch: {}, Post-proc: {}, Mode switch: {}, Zoom levels (0=off): {}, "
                       "Bookmark: {}",
                       pcmLatency.load(std::memory_order_relaxed), passthroughLatency.load(std::memory_order_relaxed),
                       PassthroughModeName(passthroughMode.load(std::memory_order_relaxed)),
                       PcmChannelModeName(pcmChannelMode.load(std::memory_order_relaxed)),
                       HdrModeName(hdrMode.load(std::memory_order_relaxed)),
                       clearOnChannelSwitch.load(std::memory_order_relaxed) ? "on" : "off", postProc, modeSwitch, zoom,
                       mark);
}

namespace {

/// The queued repair for @p key, or nullptr. setup.conf can carry a key twice, so neither
/// queuing nor refreshing may assume it is new.
[[nodiscard]] auto FindRepair(const char *key, std::vector<SetupRepair> &repairs) -> SetupRepair * {
    const auto it = std::ranges::find(repairs, key, &SetupRepair::key);
    return it == repairs.end() ? nullptr : &*it;
}

/// Shared failure path: log, keep @p fallback, queue the line for a rewrite -- VDR re-saves
/// rejected lines verbatim (see SetupRepair). One entry per key; the rewrite replaces one line.
auto RejectValue(const char *key, const char *value, std::string_view why, int fallback,
                 std::vector<SetupRepair> &repairs) -> void {
    esyslog("vaapivideo/config: %s value '%s' %.*s -- keeping %d and rewriting setup.conf", key, value,
            static_cast<int>(why.size()), why.data(), fallback);
    if (auto *queued = FindRepair(key, repairs); queued != nullptr) {
        queued->value = fallback;
    } else {
        repairs.push_back({.key = key, .value = fallback});
    }
}

/// A clean line for @p key: retarget any repair queued by an earlier rejected duplicate, or the
/// rewrite would undo the good line that won. Never queues -- a key that parsed needs no repair.
auto AcceptValue(const char *key, int accepted, std::vector<SetupRepair> &repairs) -> void {
    if (auto *queued = FindRepair(key, repairs); queued != nullptr) [[unlikely]] {
        queued->value = accepted;
    }
}

/// from_chars over the WHOLE string -- stopping at the first bad character would accept "50x" as
/// 50. nullopt on garbage; shared by every numeric parser below.
[[nodiscard]] auto ParseWholeInt(const char *value) -> std::optional<int> {
    int parsed{};
    const auto *end = value + std::strlen(value);
    const auto [ptr, ec] = std::from_chars(value, end, parsed);
    if (ec != std::errc{} || ptr != end) [[unlikely]] {
        return std::nullopt;
    }
    return parsed;
}

/// Parse an integer into @p target after range-checking [@p min, @p max]. Relaxed store: every
/// consumer re-reads on its own cadence (audio latency per packet, zoom per filter rebuild).
auto ParseBoundedIntValue(const char *key, const char *value, std::atomic<int> &target, int min, int max,
                          std::vector<SetupRepair> &repairs) -> void {
    const int fallback = target.load(std::memory_order_relaxed);
    const auto parsed = ParseWholeInt(value);
    if (!parsed) [[unlikely]] {
        RejectValue(key, value, "is not a number", fallback, repairs);
        return;
    }
    if (*parsed < min || *parsed > max) [[unlikely]] {
        RejectValue(key, value, std::format("is outside [{},{}]", min, max), fallback, repairs);
        return;
    }
    target.store(*parsed, std::memory_order_relaxed);
    AcceptValue(key, *parsed, repairs);
}

/// Parse VDR's canonical 0/1 boolean encoding into @p target. Relaxed store, matching the other
/// parsers.
auto ParseBoolValue(const char *key, const char *value, std::atomic<bool> &target, std::vector<SetupRepair> &repairs)
    -> void {
    const std::string_view v{value};
    if (v != "0" && v != "1") [[unlikely]] {
        RejectValue(key, value, "is not 0 or 1", target.load(std::memory_order_relaxed) ? 1 : 0, repairs);
        return;
    }
    const bool parsed = v == "1";
    target.store(parsed, std::memory_order_relaxed);
    AcceptValue(key, parsed ? 1 : 0, repairs);
}

/// Parse a non-negative integer (e.g. a bookmark position in ms) into @p target. Plain int, not
/// atomic: the bookmark is only touched at startup (here) and via the serialized accessors in
/// mediaplayer.cpp.
auto ParseNonNegativeIntValue(const char *key, const char *value, int &target, std::vector<SetupRepair> &repairs)
    -> void {
    const auto parsed = ParseWholeInt(value);
    if (!parsed || *parsed < 0) [[unlikely]] {
        RejectValue(key, value, "is not a non-negative number", target, repairs);
        return;
    }
    target = *parsed;
    AcceptValue(key, *parsed, repairs);
}

/// Parse a contiguous-from-zero enum index (written by cMenuEditStraItem) into @p target after
/// bounds-checking against @p count. Relaxed store, matching the other parsers.
template <typename EnumT>
auto ParseEnumValue(const char *key, const char *value, std::atomic<EnumT> &target, int count,
                    std::vector<SetupRepair> &repairs) -> void {
    const int fallback = static_cast<int>(target.load(std::memory_order_relaxed));
    const auto parsed = ParseWholeInt(value);
    if (!parsed) [[unlikely]] {
        RejectValue(key, value, "is not a number", fallback, repairs);
        return;
    }
    if (*parsed < 0 || *parsed >= count) [[unlikely]] {
        RejectValue(key, value, std::format("is outside [0,{}]", count - 1), fallback, repairs);
        return;
    }
    target.store(static_cast<EnumT>(*parsed), std::memory_order_relaxed);
    AcceptValue(key, *parsed, repairs);
}

} // namespace

[[nodiscard]] auto VaapiConfig::SetupParse(const char *name, const char *value) -> bool {
    if (!name || !value) [[unlikely]] {
        return false;
    }

    // Every branch returns true: VDR asks "is this key yours?", not "was the value any good". A bad
    // value keeps the default and queues a repair, because a rejected line lives forever (SetupRepair).
    //
    // Key strings must stay in sync with the SetupStore() calls in vaapivideo.cpp -- VDR
    // round-trips these verbatim through setup.conf, so a typo silently drops the setting.
    const std::string_view key{name};
    if (key == "PcmLatency") {
        ParseBoundedIntValue("PcmLatency", value, pcmLatency, CONFIG_AUDIO_LATENCY_MIN_MS, CONFIG_AUDIO_LATENCY_MAX_MS,
                             setupRepairs);
        return true;
    }
    if (key == "PassthroughLatency") {
        ParseBoundedIntValue("PassthroughLatency", value, passthroughLatency, CONFIG_AUDIO_LATENCY_MIN_MS,
                             CONFIG_AUDIO_LATENCY_MAX_MS, setupRepairs);
        return true;
    }
    if (key == "PassthroughMode") {
        ParseEnumValue("PassthroughMode", value, passthroughMode, CONFIG_PASSTHROUGH_MODE_COUNT, setupRepairs);
        return true;
    }
    if (key == "HdrMode") {
        ParseEnumValue("HdrMode", value, hdrMode, CONFIG_HDR_MODE_COUNT, setupRepairs);
        return true;
    }
    if (key == "PcmChannelMode") {
        ParseEnumValue("PcmChannelMode", value, pcmChannelMode, CONFIG_PCM_CHANNEL_MODE_COUNT, setupRepairs);
        return true;
    }
    if (key == "ClearOnChannelSwitch") {
        ParseBoolValue("ClearOnChannelSwitch", value, clearOnChannelSwitch, setupRepairs);
        return true;
    }
    if (key == "BookmarkUri") {
        bookmark.uri = value; // free-form path/URL; validated at use (browser open / StartPlayback)
        return true;
    }
    if (key == "BookmarkPositionMs") {
        ParseNonNegativeIntValue("BookmarkPositionMs", value, bookmark.positionMs, setupRepairs);
        return true;
    }
    if (key == "DeinterlaceMode") {
        ParseEnumValue("DeinterlaceMode", value, deinterlaceMode, CONFIG_DEINTERLACE_MODE_COUNT, setupRepairs);
        return true;
    }
    if (key == "DenoiseMode") {
        ParseEnumValue("DenoiseMode", value, denoiseMode, CONFIG_DENOISE_MODE_COUNT, setupRepairs);
        return true;
    }
    if (key == "SharpenMode") {
        ParseEnumValue("SharpenMode", value, sharpenMode, CONFIG_SHARPEN_MODE_COUNT, setupRepairs);
        return true;
    }
    if (key == "ScaleMode") {
        ParseEnumValue("ScaleMode", value, scaleMode, CONFIG_SCALE_MODE_COUNT, setupRepairs);
        return true;
    }
    // Display mode switching. Both match policies and all three scope switches default to off,
    // so a setup.conf written before this feature existed keeps the legacy fixed-mode behaviour.
    if (key == "MatchRefreshRate") {
        ParseBoolValue("MatchRefreshRate", value, matchRefreshRate, setupRepairs);
        return true;
    }
    if (key == "MatchResolution") {
        ParseBoolValue("MatchResolution", value, matchResolution, setupRepairs);
        return true;
    }
    if (key == "MinResolution") {
        ParseEnumValue("MinResolution", value, minResolution, CONFIG_MIN_RESOLUTION_MODE_COUNT, setupRepairs);
        return true;
    }
    if (key == "MaxRefreshRate") {
        ParseEnumValue("MaxRefreshRate", value, maxRefreshRate, CONFIG_MAX_REFRESH_MODE_COUNT, setupRepairs);
        return true;
    }
    if (key == "ModeSwitchLiveTv") {
        ParseBoolValue("ModeSwitchLiveTv", value, modeSwitchLiveTv, setupRepairs);
        return true;
    }
    if (key == "ModeSwitchMediaplayer") {
        ParseBoolValue("ModeSwitchMediaplayer", value, modeSwitchMediaplayer, setupRepairs);
        return true;
    }
    if (key == "ModeSwitchReplay") {
        ParseBoolValue("ModeSwitchReplay", value, modeSwitchReplay, setupRepairs);
        return true;
    }
    // Per-preset zoom level (tenths-of-% zoom-in factor), keys Zoom1..Zoom<N> as written by the
    // SetupStore() loop in vaapivideo.cpp. The active cycle stop (zoomActive) is intentionally NOT
    // parsed/stored: zoom is transient and must reset to Off on every restart.
    static_assert(CONFIG_ZOOM_PRESET_COUNT <= 9, "key decode below reads a single preset digit");
    if (key.size() == 5 && key.starts_with("Zoom")) {
        const int preset = key.back() - '1';
        if (preset >= 0 && preset < CONFIG_ZOOM_PRESET_COUNT) {
            ParseBoundedIntValue(name, value, zoomLevel[preset], CONFIG_ZOOM_LEVEL_MIN, CONFIG_ZOOM_LEVEL_MAX,
                                 setupRepairs);
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
