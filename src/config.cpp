// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file config.cpp
 * @brief Plugin configuration and setup storage
 */

#include "config.h"

// C++ Standard Library
#include <atomic>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <format>
#include <string>
#include <system_error>

// VDR
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <vdr/tools.h>
#pragma GCC diagnostic pop

// ============================================================================
// === CONSTANTS ===
// ============================================================================

constexpr int CONFIG_AUDIO_LATENCY_MAX_MS = 200; ///< Upper bound for audio-latency compensation (ms)
constexpr int CONFIG_AUDIO_LATENCY_MIN_MS = 0;   ///< Lower bound for audio-latency compensation (ms)
constexpr uint32_t CONFIG_MAX_VIDEO_HEIGHT =
    2160U; ///< Maximum display height accepted by ParseResolution() (4K UHD, px)
constexpr uint32_t CONFIG_MAX_VIDEO_WIDTH = 3840U; ///< Maximum display width accepted by ParseResolution() (4K UHD, px)

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

    // Expected format: WIDTHxHEIGHT@RATE (e.g. "1920x1080@50"). All three fields are mandatory.
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

    // Verify end pointer matches delimiter to reject trailing garbage (e.g. "1920abc").
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

    // Lower bounds: 640x480 is the smallest meaningful SD resolution; 23 fps covers 24p content with a small rounding
    // margin. Upper bounds: 3840x2160 and 120 Hz are the maximums the VAAPI/DRM stack is expected to handle; anything
    // higher is almost certainly a typo.
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
    return std::format("Audio Latency: {}ms", audioLatency.load(std::memory_order_relaxed));
}

[[nodiscard]] auto VaapiConfig::SetupParse(const char *name, const char *value) -> bool {
    if (!name || !value) [[unlikely]] {
        return false;
    }

    // Key must match the string passed to SetupStore() in vaapivideo.cpp; VDR persists these in setup.conf.
    if (std::string_view{name} == "AudioLatency") {
        int parsed{};
        const auto *end = value + std::strlen(value);
        const auto [ptr, ec] = std::from_chars(value, end, parsed);

        if (ec != std::errc{} || ptr != end) {
            esyslog("vaapivideo/config: invalid AudioLatency value '%s'", value);
            return false;
        }
        if (parsed < CONFIG_AUDIO_LATENCY_MIN_MS || parsed > CONFIG_AUDIO_LATENCY_MAX_MS) {
            esyslog("vaapivideo/config: AudioLatency %d outside valid range [%d,%d]", parsed,
                    CONFIG_AUDIO_LATENCY_MIN_MS, CONFIG_AUDIO_LATENCY_MAX_MS);
            return false;
        }

        dsyslog("vaapivideo/config: audio latency updated from %d to %d ms",
                audioLatency.load(std::memory_order_relaxed), parsed);
        audioLatency.store(parsed, std::memory_order_relaxed);
        return true;
    }

    return false; // Unknown key: VDR will offer it to other plugins.
}

// ----------------------------------------------------------------------------

// Singleton; written once during plugin init, then effectively read-only. Declared extern in config.h.
VaapiConfig vaapiConfig;
