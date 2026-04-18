// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file config.cpp
 * @brief Plugin configuration: resolution parsing + setup.conf load/store.
 *
 * The global `vaapiConfig` is shared across the VDR main thread (setup menu),
 * the audio thread, and the device thread. Mutable fields (`pcmLatency`,
 * `passthroughLatency`) are `std::atomic<int>` with relaxed ordering -- they
 * are scalar tunables read on slow paths, so no acquire/release pairing is
 * needed and we deliberately avoid a mutex here.
 */

#include "config.h"

// C++ Standard Library
#include <atomic>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <format>
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

// Audio latency bounds live in config.h (CONFIG_AUDIO_LATENCY_{MIN,MAX}_MS) so the parse path
// here and the setup-menu UI share one source of truth -- keep them in lockstep.
constexpr uint32_t CONFIG_MAX_VIDEO_HEIGHT = 2160U; ///< 4K UHD ceiling for ParseResolution() (px)
constexpr uint32_t CONFIG_MAX_VIDEO_WIDTH = 3840U;  ///< 4K UHD ceiling for ParseResolution() (px)

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
    return std::format(
        "PCM Latency: {}ms, Passthrough Latency: {}ms, Passthrough: {}, HDR: {}, Clear on channel switch: {}",
        pcmLatency.load(std::memory_order_relaxed), passthroughLatency.load(std::memory_order_relaxed),
        PassthroughModeName(passthroughMode.load(std::memory_order_relaxed)),
        HdrModeName(hdrMode.load(std::memory_order_relaxed)),
        clearOnChannelSwitch.load(std::memory_order_relaxed) ? "on" : "off");
}

namespace {

/// Parse a latency value (ms) into @p target after range-clamping to
/// [CONFIG_AUDIO_LATENCY_MIN_MS, CONFIG_AUDIO_LATENCY_MAX_MS]. Stores with relaxed ordering;
/// the audio path re-reads on every packet so torn writes don't matter here.
[[nodiscard]] auto ParseLatencyValue(const char *key, const char *value, std::atomic<int> &target) -> bool {
    int parsed{};
    const auto *end = value + std::strlen(value);
    const auto [ptr, ec] = std::from_chars(value, end, parsed);

    if (ec != std::errc{} || ptr != end) {
        esyslog("vaapivideo/config: invalid %s value '%s'", key, value);
        return false;
    }
    if (parsed < CONFIG_AUDIO_LATENCY_MIN_MS || parsed > CONFIG_AUDIO_LATENCY_MAX_MS) {
        esyslog("vaapivideo/config: %s %d outside valid range [%d,%d]", key, parsed, CONFIG_AUDIO_LATENCY_MIN_MS,
                CONFIG_AUDIO_LATENCY_MAX_MS);
        return false;
    }

    dsyslog("vaapivideo/config: %s updated from %d to %d ms", key, target.load(std::memory_order_relaxed), parsed);
    target.store(parsed, std::memory_order_relaxed);
    return true;
}

} // namespace

[[nodiscard]] auto VaapiConfig::SetupParse(const char *name, const char *value) -> bool {
    if (!name || !value) [[unlikely]] {
        return false;
    }

    // Key strings must stay in sync with the SetupStore() calls in vaapivideo.cpp -- VDR
    // round-trips these verbatim through setup.conf, so a typo silently drops the setting.
    const std::string_view key{name};
    if (key == "PcmLatency") {
        return ParseLatencyValue("PcmLatency", value, pcmLatency);
    }
    if (key == "PassthroughLatency") {
        return ParseLatencyValue("PassthroughLatency", value, passthroughLatency);
    }
    if (key == "PassthroughMode") {
        // Encoded as the integer value of PassthroughMode -- written by cMenuEditStraItem,
        // read back here. Range [Auto..Off] is contiguous so a single bounds check suffices.
        int parsed{};
        const auto *end = value + std::strlen(value);
        const auto [ptr, ec] = std::from_chars(value, end, parsed);
        if (ec != std::errc{} || ptr != end || parsed < static_cast<int>(PassthroughMode::Auto) ||
            parsed > static_cast<int>(PassthroughMode::Off)) {
            esyslog("vaapivideo/config: invalid PassthroughMode value '%s'", value);
            return false;
        }
        const auto mode = static_cast<PassthroughMode>(parsed);
        const auto previous = passthroughMode.exchange(mode, std::memory_order_relaxed);
        dsyslog("vaapivideo/config: PassthroughMode updated from %s to %s", PassthroughModeName(previous),
                PassthroughModeName(mode));
        return true;
    }
    if (key == "HdrMode") {
        // Encoded as the integer value of HdrMode -- written by cMenuEditStraItem, read
        // back here. Range [Auto..Off] is contiguous so a single bounds check suffices.
        int parsed{};
        const auto *end = value + std::strlen(value);
        const auto [ptr, ec] = std::from_chars(value, end, parsed);
        if (ec != std::errc{} || ptr != end || parsed < static_cast<int>(HdrMode::Auto) ||
            parsed > static_cast<int>(HdrMode::Off)) {
            esyslog("vaapivideo/config: invalid HdrMode value '%s'", value);
            return false;
        }
        const auto mode = static_cast<HdrMode>(parsed);
        const auto previous = hdrMode.exchange(mode, std::memory_order_relaxed);
        dsyslog("vaapivideo/config: HdrMode updated from %s to %s", HdrModeName(previous), HdrModeName(mode));
        return true;
    }
    if (key == "ClearOnChannelSwitch") {
        // Accept VDR's canonical 0/1 encoding for booleans; anything else is a malformed entry.
        const std::string_view v{value};
        bool parsed{};
        if (v == "0") {
            parsed = false;
        } else if (v == "1") {
            parsed = true;
        } else {
            esyslog("vaapivideo/config: invalid ClearOnChannelSwitch value '%s'", value);
            return false;
        }
        const bool previous = clearOnChannelSwitch.load(std::memory_order_relaxed);
        clearOnChannelSwitch.store(parsed, std::memory_order_relaxed);
        dsyslog("vaapivideo/config: ClearOnChannelSwitch updated from %d to %d", previous ? 1 : 0, parsed ? 1 : 0);
        return true;
    }

    return false; // Unknown key: ignore so older/newer setup.conf entries don't break load.
}

// ----------------------------------------------------------------------------

// Process-wide singleton (declared extern in config.h). Display fields are written once
// during plugin init; latency atomics may be re-written from the VDR main thread whenever
// the user edits the setup menu, hence the std::atomic types.
VaapiConfig vaapiConfig;
