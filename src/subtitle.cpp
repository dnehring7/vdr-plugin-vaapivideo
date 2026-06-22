// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file subtitle.cpp
 * @brief cSubtitleConverter: decode text-subtitle cues and pace them onto the OSD.
 *
 * See subtitle.h for the threading model. Rendering uses the VDR core OSD exactly like the core
 * DVB subtitle path: a cOsd at OSD_LEVEL_SUBTITLES with cOsd::DrawText. The FFmpeg subtitle decoder
 * turns SubRip/ASS/mov_text into plain text (VDR core has no text-subtitle decoder).
 */

#include "subtitle.h"
#include "common.h"
#include "config.h"
#include "device.h"

// C++ Standard Library
#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// FFmpeg
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/codec.h>
#include <libavcodec/codec_id.h>
#include <libavcodec/codec_par.h>
#include <libavcodec/packet.h>
#include <libavutil/avutil.h>
#include <libavutil/rational.h>
}
#pragma GCC diagnostic pop

// VDR
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <vdr/config.h>
#include <vdr/font.h>
#include <vdr/osd.h>
#include <vdr/thread.h>
#include <vdr/tools.h>
#pragma GCC diagnostic pop

namespace {

// ============================================================================
// === LOCAL CONSTANTS ===
// ============================================================================

constexpr int64_t DEFAULT_CUE_DURATION_90K = 270000; ///< 3 s fallback when a cue carries no duration.
constexpr size_t SUBTITLE_QUEUE_CAPACITY = 256;      ///< Bound future cues from a malformed / front-loaded stream.
constexpr int SUBTITLE_SHUTDOWN_TIMEOUT_S = 2;       ///< Action() join timeout in Shutdown().
constexpr int SUBTITLE_TICK_MS = 50;                 ///< Pacing cadence: how often Action() re-checks the cue vs clock.

// ============================================================================
// === TEXT PARSING ===
// ============================================================================

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- indices are bounded
// by the explicit `< .size()` loop conditions.

/// Resolve a few HTML/teletext color names common in broadcaster SRTs to 0xRRGGBB; -1 if unknown.
[[nodiscard]] auto NamedColor(std::string_view name) -> int32_t {
    struct NamedRgb {
        std::string_view name;
        uint32_t rgb;
    };
    static constexpr std::array<NamedRgb, 8> kColors{{
        {.name = "white", .rgb = 0xFFFFFF},
        {.name = "yellow", .rgb = 0xFFFF00},
        {.name = "green", .rgb = 0x00FF00},
        {.name = "cyan", .rgb = 0x00FFFF},
        {.name = "red", .rgb = 0xFF0000},
        {.name = "magenta", .rgb = 0xFF00FF},
        {.name = "blue", .rgb = 0x0000FF},
        {.name = "black", .rgb = 0x000000},
    }};
    const auto ieq = [](std::string_view a, std::string_view b) -> bool {
        return std::ranges::equal(a, b, [](char l, char r) -> bool {
            return std::tolower(static_cast<unsigned char>(l)) == std::tolower(static_cast<unsigned char>(r));
        });
    };
    for (const auto &c : kColors) {
        if (ieq(name, c.name)) {
            return static_cast<int32_t>(c.rgb);
        }
    }
    return -1;
}

/// Parse @p hex (no leading '#'/'&H') as up to @p maxDigits hex digits. Returns the value, or -1
/// if no digits. (8 digits for ASS &HAABBGGRR, 6 for #RRGGBB.)
[[nodiscard]] auto ParseHex(std::string_view hex, int maxDigits = 6) -> int64_t {
    int64_t value = 0;
    int digits = 0;
    for (const char c : hex) {
        if (std::isxdigit(static_cast<unsigned char>(c)) == 0) {
            break;
        }
        const int d = (c <= '9') ? (c - '0') : ((std::tolower(static_cast<unsigned char>(c)) - 'a') + 10);
        value = (value << 4) | d;
        if (++digits == maxDigits) {
            break;
        }
    }
    return digits > 0 ? value : -1;
}

/// Color from an ASS override block (content between {}), e.g. `\c&Hbbggrr&` / `\1c&Hbbggrr&` (the
/// ffmpeg subrip decoder emits `<font color>` in this form). ASS stores B,G,R; returns 0xRRGGBB or -1.
[[nodiscard]] auto ParseAssColor(std::string_view block) -> int32_t {
    for (size_t p = block.find("c&H"); p != std::string_view::npos; p = block.find("c&H", p + 1)) {
        const char prev = p > 0 ? block[p - 1] : '\0';
        if (prev != '\\' && prev != '1') {
            continue; // skip \3c (outline) / \4c (shadow); we only want the primary fill color
        }
        // ASS color is &Hbbggrr& (6) or &Haabbggrr& (8, alpha in the high byte); take the low 24
        // bits (BBGGRR) and reorder to 0xRRGGBB.
        if (const int64_t v = ParseHex(block.substr(p + 3), 8); v >= 0) {
            const auto bgr = static_cast<uint32_t>(v) & 0xFFFFFFU;
            const uint32_t blue = (bgr >> 16) & 0xFF;
            const uint32_t green = (bgr >> 8) & 0xFF;
            const uint32_t red = bgr & 0xFF;
            return static_cast<int32_t>((red << 16) | (green << 8) | blue);
        }
    }
    return -1;
}

/// Color from an HTML-ish tag (content between <>), e.g. `font color="yellow"` / `font color="#rrggbb"`.
/// Returns 0xRRGGBB or -1 when the tag carries no recognizable color.
[[nodiscard]] auto ParseHtmlColor(std::string_view tag) -> int32_t {
    const size_t key = tag.find("color");
    if (key == std::string_view::npos) {
        return -1;
    }
    size_t i = tag.find('=', key);
    if (i == std::string_view::npos) {
        return -1;
    }
    ++i;
    while (i < tag.size() && (tag[i] == ' ' || tag[i] == '"' || tag[i] == '\'')) {
        ++i;
    }
    size_t j = i;
    while (j < tag.size() && tag[j] != '"' && tag[j] != '\'' && tag[j] != ' ' && tag[j] != '>') {
        ++j;
    }
    const std::string_view value = tag.substr(i, j - i);
    if (value.empty()) {
        return -1;
    }
    if (value.front() == '#') {
        const int64_t v = ParseHex(value.substr(1));
        return v >= 0 ? static_cast<int32_t>(v) : -1; // #rrggbb is already 0xRRGGBB
    }
    return NamedColor(value);
}

/// Append @p text (already markup-bearing) to @p out as trimmed, non-empty lines, carrying the
/// active foreground color per line. Strips ASS override blocks {\...} and HTML-ish tags <...>
/// (extracting any color they set), converts "\N"/"\n" line breaks and "\h" hard space, and splits
/// on real newlines. Per line the color is the last color override seen before the line break.
auto AppendLinesFromMarkup(std::string_view text, std::vector<cSubtitleConverter::Line> &out) -> void {
    std::string cur;
    int32_t curColor = -1;
    const auto flush = [&out, &cur, &curColor]() -> void {
        const size_t b = cur.find_first_not_of(" \t\r\n");
        if (b != std::string::npos) {
            const size_t e = cur.find_last_not_of(" \t\r\n");
            cSubtitleConverter::Line line;
            line.text = cur.substr(b, e - b + 1);
            if (curColor >= 0) {
                line.rgb = static_cast<uint32_t>(curColor);
                line.hasColor = true;
            }
            out.push_back(std::move(line));
        }
        cur.clear();
    };

    for (size_t i = 0; i < text.size();) {
        const char c = text[i];
        if (c == '{') { // ASS override block
            const size_t close = text.find('}', i + 1);
            if (close == std::string_view::npos) {
                cur.push_back(c); // unmatched '{' -> keep as literal text, don't drop the rest
                ++i;
                continue;
            }
            if (const int32_t col = ParseAssColor(text.substr(i + 1, close - (i + 1))); col >= 0) {
                curColor = col;
            }
            i = close + 1;
        } else if (c == '<') { // HTML-ish tag (subrip <i>/<b>/<font color>)
            const size_t close = text.find('>', i + 1);
            if (close == std::string_view::npos) {
                cur.push_back(c); // unmatched '<' (e.g. "5 < 10") -> keep as literal text
                ++i;
                continue;
            }
            if (const int32_t col = ParseHtmlColor(text.substr(i + 1, close - (i + 1))); col >= 0) {
                curColor = col;
            }
            i = close + 1;
        } else if (c == '\\' && i + 1 < text.size()) {
            const char n = text[i + 1];
            if (n == 'N' || n == 'n') {
                flush();
            } else if (n == 'h') {
                cur.push_back(' ');
            } else {
                cur.push_back(n); // unknown escape: drop the backslash, keep the char
            }
            i += 2;
        } else if (c == '\n') {
            flush();
            ++i;
        } else if (c == '\r') {
            ++i;
        } else {
            cur.push_back(c);
            ++i;
        }
    }
    flush();
}

/// Append the text of an ffmpeg ASS event line. The decoded ass field is
/// "ReadOrder,Layer,Style,Name,MarginL,MarginR,MarginV,Effect,Text" -- 8 commas precede the text.
auto AppendAssLines(const char *ass, std::vector<cSubtitleConverter::Line> &out) -> void {
    const std::string_view s{ass};
    size_t pos = 0;
    int commas = 0;
    for (; pos < s.size() && commas < 8; ++pos) {
        if (s[pos] == ',') {
            ++commas;
        }
    }
    AppendLinesFromMarkup(commas == 8 ? s.substr(pos) : s, out);
}
// NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

} // namespace

// ============================================================================
// === cSubtitleConverter ===
// ============================================================================

// Start the pacing thread up front: it must already be running when the user enables a track, so the
// first cue is timed against the clock immediately. It idles cheaply (50 ms tick) until cues arrive.
cSubtitleConverter::cSubtitleConverter(cVaapiDevice *device) : cThread("vaapi subtitle"), device_(device) {
    hasExited_.store(false, std::memory_order_release);
    if (!Start()) {
        // Keep the latch consistent so Shutdown() doesn't wait on / warn about a thread that never ran.
        esyslog("vaapivideo/subtitle: failed to start subtitle thread");
        stopping_.store(true, std::memory_order_release);
        hasExited_.store(true, std::memory_order_release);
    }
}

cSubtitleConverter::~cSubtitleConverter() noexcept { Shutdown(); }

[[nodiscard]] auto cSubtitleConverter::Open(const AVCodecParameters *codecpar) -> bool {
    Close(); // drop any prior decoder, cues, and on-screen text
    if (codecpar == nullptr) {
        return false;
    }
    const AVCodec *decoder = avcodec_find_decoder(codecpar->codec_id);
    if (decoder == nullptr) {
        esyslog("vaapivideo/subtitle: no decoder for codec %s", avcodec_get_name(codecpar->codec_id));
        return false;
    }
    std::unique_ptr<AVCodecContext, FreeAVCodecContext> ctx{avcodec_alloc_context3(decoder)};
    if (!ctx) {
        return false;
    }
    if (const int ret = avcodec_parameters_to_context(ctx.get(), codecpar); ret < 0) {
        esyslog("vaapivideo/subtitle: parameters_to_context: %s", AvErr(ret).data());
        return false;
    }
    // Packets reach Convert() already rebased to the 90 kHz domain, so tell the decoder that base;
    // its start/end_display_time then track real cue timing when a stream relies on them.
    ctx->pkt_timebase = AVRational{.num = 1, .den = 90000};
    if (const int ret = avcodec_open2(ctx.get(), decoder, nullptr); ret < 0) {
        esyslog("vaapivideo/subtitle: avcodec_open2: %s", AvErr(ret).data());
        return false;
    }
    codecCtx_ = std::move(ctx);
    isyslog("vaapivideo/subtitle: track open (%s)", avcodec_get_name(codecpar->codec_id));
    return true;
}

// Close vs Reset differ only in the decoder: Close drops it (track turned off / switched -> the next
// Open() rebuilds for the new stream); Reset keeps it (seek within the same track -> only the cues are
// now stale). Both ask Action() to clear the screen via the atomic rather than touching the OSD here,
// because the OSD is owned solely by the Action thread.
auto cSubtitleConverter::Close() -> void {
    codecCtx_.reset();
    {
        const cMutexLock lock(&cueMutex_);
        cues_.clear();
    }
    hideRequested_.store(true, std::memory_order_release);
}

auto cSubtitleConverter::Reset() -> void {
    {
        const cMutexLock lock(&cueMutex_);
        cues_.clear();
    }
    hideRequested_.store(true, std::memory_order_release);
}

auto cSubtitleConverter::Convert(const AVPacket *packet) -> void {
    if (!codecCtx_ || packet == nullptr || packet->data == nullptr || packet->size <= 0) {
        return;
    }
    // Fall back to DTS: text subtitles have no frame reorder, so DTS == PTS and either anchors the cue.
    const int64_t packetPts90k = packet->pts != AV_NOPTS_VALUE ? packet->pts : packet->dts;
    if (packetPts90k == AV_NOPTS_VALUE) {
        return; // cannot place an untimestamped cue on the timeline -> drop it rather than guess
    }

    // avcodec_decode_subtitle2 takes a non-const AVPacket; clone (ref-counted, cheap, rare) so the
    // caller's packet is untouched.
    const std::unique_ptr<AVPacket, FreeAVPacket> clone{av_packet_clone(packet)};
    if (!clone) {
        return;
    }

    AVSubtitle sub{};
    int gotSub = 0;
    const int ret = avcodec_decode_subtitle2(codecCtx_.get(), &sub, &gotSub, clone.get());
    if (ret < 0) {
        esyslog("vaapivideo/subtitle: decode: %s", AvErr(ret).data());
        return;
    }
    if (gotSub == 0) {
        avsubtitle_free(&sub); // NOLINT(clang-analyzer-unix.Malloc) -- C-API frees rects/owned bufs
        return;
    }

    const int64_t start90k = packetPts90k + (static_cast<int64_t>(sub.start_display_time) * PTS_TICKS_PER_MS);
    int64_t end90k = 0;
    if (packet->duration > 0) {
        end90k = packetPts90k + packet->duration;
    } else if (sub.end_display_time != 0 && sub.end_display_time != UINT32_MAX) {
        end90k = packetPts90k + (static_cast<int64_t>(sub.end_display_time) * PTS_TICKS_PER_MS);
    } else {
        end90k = packetPts90k + DEFAULT_CUE_DURATION_90K;
    }
    // Guard malformed timing (end <= start): without a positive window the cue's [start,end) test in
    // Action() never matches the clock, so it would silently never appear.
    if (end90k <= start90k) {
        end90k = start90k + DEFAULT_CUE_DURATION_90K;
    }

    Cue cue;
    cue.start90k = start90k;
    cue.end90k = end90k;
    for (unsigned i = 0; i < sub.num_rects; ++i) {
        const AVSubtitleRect *rect = sub.rects[i];
        if (rect == nullptr) {
            continue;
        }
        if (rect->type == SUBTITLE_ASS && rect->ass != nullptr) {
            AppendAssLines(rect->ass, cue.lines);
        } else if (rect->type == SUBTITLE_TEXT && rect->text != nullptr) {
            AppendLinesFromMarkup(rect->text, cue.lines);
        }
    }
    avsubtitle_free(&sub); // NOLINT(clang-analyzer-unix.Malloc) -- C-API frees rects/owned bufs

    if (cue.lines.empty()) {
        return; // markup that stripped to nothing -> don't queue a blank cue that draws an empty band
    }
    // Demux delivers packets in ascending start order, so a plain push_back keeps cues_ sorted -- which
    // Action()'s front-pruning and first-match scan rely on.
    const cMutexLock lock(&cueMutex_);
    if (cues_.size() >= SUBTITLE_QUEUE_CAPACITY) {
        return; // a front-loaded / malformed stream could queue cues faster than the clock prunes them
    }
    cues_.push_back(std::move(cue));
}

auto cSubtitleConverter::Shutdown() -> void {
    if (hasShutdown_.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    stopping_.store(true, std::memory_order_release);
    Cancel(SUBTITLE_SHUTDOWN_TIMEOUT_S); // join Action(): from here this thread solely owns the OSD
    if (!hasExited_.load(std::memory_order_acquire)) {
        // Cancel() timed out and force-cancelled mid-Action(); the OSD it owns may be half-torn-down.
        esyslog("vaapivideo/subtitle: subtitle thread did not exit cleanly");
    }

    HideCue();
    codecCtx_.reset();
    const cMutexLock lock(&cueMutex_);
    cues_.clear();
}

// ============================================================================
// === THREAD ===
// ============================================================================

auto cSubtitleConverter::Action() -> void {
    while (!stopping_.load(std::memory_order_acquire)) {
        if (hideRequested_.exchange(false, std::memory_order_acq_rel)) {
            HideCue();
        }

        // Pace against the audio master clock -- the real presentation position. GetSTC() is the last
        // decoded video PTS and runs ahead of output (decoder+display depth), which would show cues
        // early; fall back to it only when there is no audio clock (e.g. video-only / pre-anchor).
        int64_t clock90k = AV_NOPTS_VALUE;
        if (device_ != nullptr) {
            clock90k = device_->GetAudioClock();
            if (clock90k == AV_NOPTS_VALUE) {
                clock90k = device_->GetSTC();
            }
        }
        if (clock90k >= 0) {
            Cue activeCue;
            bool haveActive = false;
            {
                const cMutexLock lock(&cueMutex_);
                // Drop cues that have already ended; the queue is in demux (ascending start) order.
                while (!cues_.empty() && cues_.front().end90k <= clock90k) {
                    cues_.pop_front();
                }
                for (const Cue &cue : cues_) {
                    if (cue.start90k > clock90k) {
                        break; // future cue: nothing later can match
                    }
                    if (clock90k < cue.end90k) {
                        activeCue = cue; // latest cue whose window contains the clock
                        haveActive = true;
                    }
                }
            }

            if (haveActive) {
                // Only latch shownStart90k_ on a successful draw: a transient OSD failure then retries
                // on the next tick instead of silently swallowing the whole cue.
                if (activeCue.start90k != shownStart90k_ && ShowCue(activeCue)) {
                    shownStart90k_ = activeCue.start90k;
                }
            } else if (shownStart90k_ != AV_NOPTS_VALUE) {
                HideCue();
            }
        }

        cCondWait::SleepMs(SUBTITLE_TICK_MS);
    }
    hasExited_.store(true, std::memory_order_release); // tells Shutdown() the OSD-owning thread really left
}

// ============================================================================
// === RENDERING (core OSD) ===
// ============================================================================

auto cSubtitleConverter::ShowCue(const Cue &cue) -> bool {
    int osdWidth = 0;
    int osdHeight = 0;
    double aspect = 1.0;
    if (device_ == nullptr) {
        return false;
    }
    device_->GetOsdSize(osdWidth, osdHeight, aspect);
    if (osdWidth <= 0 || osdHeight <= 0) {
        return false;
    }

    // Scale the font with the OSD height; SD modes still get a readable floor.
    const int fontHeight = std::max(28, osdHeight / 20);
    const std::unique_ptr<cFont> font{cFont::CreateFont(Setup.FontOsd, fontHeight)};
    if (!font) {
        esyslog("vaapivideo/subtitle: font creation failed");
        return false;
    }
    const int lineHeight = font->Height();
    if (lineHeight <= 0) {
        return false;
    }

    const int padX = std::max(6, fontHeight / 3);
    const int padY = std::max(2, fontHeight / 8);
    // Bound the band to the visible height: a hostile / very long cue (many wrapped rows) must not grow
    // drawLines or the OSD buffer without limit. Applied during the build below (keep the bottom-most rows).
    const int maxBandLines = std::max(1, (osdHeight - (2 * padY)) / lineHeight);

    // Honor VDR's subtitle transparency settings (0..10, same scale + mapping as cDvbSubtitleConverter:
    // 0 = opaque, 10 = fully transparent). Foreground defaults to white but follows the line's color.
    const auto alpha = [](int transparency) -> uint8_t {
        return static_cast<uint8_t>(ALPHA_OPAQUE * (10 - std::clamp(transparency, 0, 10)) / 10);
    };
    const uint8_t fgAlpha = alpha(Setup.SubtitleFgTransparency);

    // Wrap each cue line to the usable width with VDR's cTextWrapper (reuses the core text layout),
    // carrying the line's color onto each wrapped row -- long lines wrap instead of being clipped.
    struct DrawLine {
        std::string text;
        tColor fg;
    };
    std::vector<DrawLine> drawLines;
    const int maxTextWidth = std::max(1, osdWidth - (2 * padX));
    for (const Line &line : cue.lines) {
        if (line.text.empty()) {
            continue;
        }
        const uint32_t rgb = line.hasColor ? line.rgb : 0xFFFFFF;
        const tColor fg = ArgbToColor(fgAlpha, (rgb >> 16) & 0xFF, (rgb >> 8) & 0xFF, rgb & 0xFF);
        cTextWrapper wrapper(line.text.c_str(), font.get(), maxTextWidth);
        for (int k = 0; k < wrapper.Lines(); ++k) {
            // Cap peak size as we go (drop the oldest row) so a pathological cue can't balloon the vector.
            if (drawLines.size() == static_cast<size_t>(maxBandLines)) {
                drawLines.erase(drawLines.begin());
            }
            drawLines.push_back({.text = wrapper.GetLine(k), .fg = fg});
        }
    }
    if (drawLines.empty()) {
        HideCue();   // cue had only empty lines -> nothing to show
        return true; // not a failure: a legitimate no-op, latch it so Action() doesn't busy-retry
    }

    // Fixed band height sized for up to kReuseBandLines (grown only for taller cues): consecutive
    // 1/2/3-line cues then share one OSD geometry, so ShowCue reuses the live OSD (clear + redraw)
    // instead of a per-cue dumb-buffer + KMS-framebuffer alloc/free. Lines are bottom-anchored within
    // the band so a short cue still sits at the bottom of the screen.
    constexpr int kReuseBandLines = 3;
    const auto lineCount = static_cast<int>(drawLines.size());
    const int bandLines = std::min(std::max(lineCount, kReuseBandLines), maxBandLines);
    const int bandHeight = std::min(osdHeight, (lineHeight * bandLines) + (2 * padY));
    // Bottom-anchored, honoring the user's subtitle offset; clamped fully on-screen.
    const int top = std::clamp(osdHeight - std::max(8, osdHeight / 14) - bandHeight + Setup.SubtitleOffset, 0,
                               std::max(0, osdHeight - bandHeight));
    const int firstLineY = padY + ((bandLines - lineCount) * lineHeight); // bottom-align within the band

    // Reuse the live OSD when the new cue has the same shape (common for back-to-back dialogue):
    // just clear and redraw, avoiding a dumb-buffer + KMS-framebuffer alloc/free per cue. Geometry
    // is fixed at NewOsd/SetAreas, so a different size/position needs a fresh OSD. Gaps (no active
    // cue) still destroy the OSD via HideCue(), freeing the single OSD plane for menus / the replay bar.
    const bool reuse = osd_ != nullptr && osdAreaWidth_ == osdWidth && osdAreaHeight_ == bandHeight && osdTop_ == top;
    if (reuse) {
        osd_->DrawRectangle(0, 0, osdWidth - 1, bandHeight - 1, clrTransparent); // clear prior text
    } else {
        HideCue(); // drop the previous (differently-shaped) OSD, if any
        // Core OSD at the subtitle level (separate level from menus), drawn exactly like the DVB path.
        osd_ = cOsdProvider::NewOsd(0, top, OSD_LEVEL_SUBTITLES);
        if (osd_ == nullptr) {
            return false;
        }
        const tArea area = {.x1 = 0, .y1 = 0, .x2 = osdWidth - 1, .y2 = bandHeight - 1, .bpp = 32};
        if (osd_->SetAreas(&area, 1) != oeOk) {
            HideCue();
            return false;
        }
        osdTop_ = top;
        osdAreaWidth_ = osdWidth;
        osdAreaHeight_ = bandHeight;
    }

    // Backing box: black at 50% opacity, thinned further by SubtitleBgTransparency (0 -> 50%, 10 -> 0%).
    const auto boxAlpha =
        static_cast<uint8_t>((ALPHA_OPAQUE / 2) * (10 - std::clamp(Setup.SubtitleBgTransparency, 0, 10)) / 10);
    const tColor bg = ArgbToColor(boxAlpha, 0x00, 0x00, 0x00);
    for (int i = 0; i < lineCount; ++i) {
        const DrawLine &line = drawLines.at(static_cast<size_t>(i));
        // Centered box just wide enough for the text (clamped to the OSD); DrawText paints the
        // background box and centers the glyphs within it.
        const int boxWidth = std::min(osdWidth, font->Width(line.text.c_str()) + (2 * padX));
        const int x = (osdWidth - boxWidth) / 2;
        const int y = firstLineY + (i * lineHeight);
        osd_->DrawText(x, y, line.text.c_str(), line.fg, bg, font.get(), boxWidth, lineHeight, taCenter);
    }
    osd_->Flush();
    return true;
}

auto cSubtitleConverter::HideCue() -> void {
    delete osd_; // cVaapiOsd dtor hides the plane and frees its buffer safely
    osd_ = nullptr;
    osdTop_ = 0;
    osdAreaWidth_ = 0;
    osdAreaHeight_ = 0;
    shownStart90k_ = AV_NOPTS_VALUE;
}
