// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file mediaplayer.cpp
 * @brief Integrated media player: libavformat demux feeding cVaapiDevice directly.
 *
 * cVaapiMediaSource (libavformat) -> AVPacket -> cVaapiDevice::Submit{Video,Audio}Packet
 *                                                 -> shared decoder / display / audio threads.
 * The PES path (live TV, VDR recordings) is unchanged; mediaplayer just bypasses PES.
 *
 * Threading:
 *   - VDR main thread       cVaapiControl: OSD + key dispatch (non-blocking into the player).
 *   - One private cThread   cVaapiPlayer::Action(): demux pump and sole packet submitter.
 *   - Decoder/display/audio threads: unchanged from the PES path.
 *
 * A/V sync follows the audio master clock; pause routes through cDevice::Freeze/Play so
 * the clock halts with the demux loop.
 *
 * Invariants:
 *   1. sourceMutex MUST be released before calling Open/CloseCurrentEntry; cMutex is
 *      PTHREAD_MUTEX_ERRORCHECK (non-recursive). Action() defers playlist advancement.
 *   2. cVaapiControl owns the player (unique_ptr); cControl holds a borrowed alias.
 *      The dtor nulls the base before resetting (see ~cVaapiControl).
 *   3. cVaapiMediaSource emits a zero-based 90 kHz timeline so GetIndex/Seek math is
 *      stable across files with non-zero container start_time.
 *   4. Network I/O is interruptible via InterruptOnStop on cVaapiPlayer::stopping.
 */

#include "mediaplayer.h"
#include "audio.h"
#include "common.h"
#include "device.h"
#include "stream.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <dirent.h>

// FFmpeg
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
extern "C" {
#include <libavcodec/codec_id.h>
#include <libavcodec/codec_par.h>
#include <libavcodec/defs.h>
#include <libavcodec/packet.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/error.h>
#include <libavutil/mathematics.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
#include <libavutil/rational.h>
}
#pragma GCC diagnostic pop

// VDR
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <vdr/device.h>
#include <vdr/i18n.h>
#include <vdr/keys.h>
#include <vdr/menu.h>
#include <vdr/osdbase.h>
#include <vdr/player.h>
#include <vdr/remote.h>
#include <vdr/skins.h>
#include <vdr/status.h>
#include <vdr/thread.h>
#include <vdr/tools.h>
#pragma GCC diagnostic pop

namespace {

// ============================================================================
// === LOCAL CONSTANTS ===
// ============================================================================

constexpr int OSD_REFRESH_INTERVAL_MS =
    500;                                   ///< Replay-bar update cadence; ~2 Hz is enough to feel live without flicker.
constexpr int OSD_DEFAULT_TIMEOUT_S = 4;   ///< Auto-hide delay after a key event; matches VDR replay-control feel.
constexpr int DEMUX_IDLE_SLEEP_MS = 5;     ///< Back-off when ReadPacket reports EAGAIN or no work is available.
constexpr int DEMUX_PAUSE_WAKEUP_MS = 100; ///< Periodic re-check while paused; safety net against missed broadcasts.
constexpr int STATUS_LOG_INTERVAL_MS = 2000; ///< Cadence for the periodic mediaplayer status dsyslog.
constexpr int64_t TICKS_PER_MS = 90;         ///< VDR's 90 kHz PTS domain: 90 ticks == 1 ms.

/// Stringify cVaapiPlayer::State for the status log. Defined here so the header doesn't need a
/// helper, and so the switch is exhaustive (compiler warns if a new state is added).
[[nodiscard]] auto StateName(cVaapiPlayer::State s) noexcept -> const char * {
    switch (s) {
        case cVaapiPlayer::State::Opening:
            return "Opening";
        case cVaapiPlayer::State::Playing:
            return "Playing";
        case cVaapiPlayer::State::Paused:
            return "Paused";
        case cVaapiPlayer::State::Seeking:
            return "Seeking";
        case cVaapiPlayer::State::Eof:
            return "Eof";
        case cVaapiPlayer::State::Stopped:
            return "Stopped";
    }
    return "?";
}

/// Convert the container's start_time (AV_TIME_BASE units) to 90 kHz. Returns AV_NOPTS_VALUE
/// when the demuxer didn't populate start_time (typical for some streams + raw containers).
[[nodiscard]] auto FormatStart90k(const AVFormatContext *ctx) noexcept -> int64_t {
    if (ctx == nullptr || ctx->start_time == AV_NOPTS_VALUE) {
        return AV_NOPTS_VALUE;
    }
    constexpr AVRational kAvTimeBase{.num = 1, .den = AV_TIME_BASE};
    constexpr AVRational k90kHz{.num = 1, .den = 90000};
    return av_rescale_q(ctx->start_time, kAvTimeBase, k90kHz);
}

/// Stream-level start_time in 90 kHz units, or AV_NOPTS_VALUE if unset. Container-level
/// start_time is the *lowest* across all streams; we need each stream's own first PTS so the
/// caller can pick the latest (= sync point) and drop pre-sync leading packets.
[[nodiscard]] auto StreamStart90k(const AVStream *stream) noexcept -> int64_t {
    if (stream == nullptr || stream->start_time == AV_NOPTS_VALUE) {
        return AV_NOPTS_VALUE;
    }
    constexpr AVRational k90kHz{.num = 1, .den = 90000};
    return av_rescale_q(stream->start_time, stream->time_base, k90kHz);
}

/// Best-effort packet clock: prefers PTS, falls back to DTS. TS streams occasionally emit audio
/// packets carrying only DTS; without this fallback the downstream throttle and the post-seek
/// audio discard would skip such packets and the audio clock would never anchor.
[[nodiscard]] auto PacketClock90k(const AVPacket *pkt) noexcept -> int64_t {
    if (pkt == nullptr) {
        return AV_NOPTS_VALUE;
    }
    return pkt->pts != AV_NOPTS_VALUE ? pkt->pts : pkt->dts;
}

/// Deep-copy @p p->extradata into @p storage and wire the non-owning view in the StreamInfo
/// out-params. VideoStreamInfo / AudioStreamInfo hold non-owning extradata pointers; libavformat's
/// AVCodecParameters own the original buffer only while formatCtx lives. The copy keeps the
/// descriptors usable for the device's reopen logic and across audio-codec reopens mid-track.
auto CopyExtradata(const AVCodecParameters *p, std::vector<uint8_t> &storage, const uint8_t *&infoExtra,
                   int &infoSize) noexcept -> void {
    if (p->extradata != nullptr && p->extradata_size > 0) {
        storage.assign(p->extradata, p->extradata + p->extradata_size);
        infoExtra = storage.data();
        infoSize = static_cast<int>(storage.size());
    }
}

/// libavformat interrupt_callback. Polled by avformat_open_input / av_read_frame on slow
/// network I/O; returning non-zero makes those calls bail out with AVERROR_EXIT so both
/// player shutdown AND pending seek/next are bounded. @p opaque is the cVaapiMediaSource.
extern "C" auto InterruptOnStop(void *opaque) -> int {
    const auto *source = static_cast<const cVaapiMediaSource *>(opaque);
    return (source != nullptr && source->IoInterrupted()) ? 1 : 0;
}

// File-browser extension whitelist. Lowercase, dot-prefixed. libavformat can autodetect
// containers without an extension hint, but the browser filters the listing for the user;
// audio-only formats are intentionally absent because cVaapiMediaSource::Open requires a
// video stream and would fail at open. Extend with care -- adding here implies the entire
// decode path supports the format.
constexpr std::array<std::string_view, 7> kMediaExtensions{{".mp4", ".mkv", ".avi", ".mov", ".ts", ".m4v", ".webm"}};

constexpr std::array<std::string_view, 2> kPlaylistExtensions{{".m3u", ".m3u8"}};

// URI schemes we hand straight to libavformat rather than resolving as filesystem paths.
// HLS .m3u8 over http(s) deliberately goes here rather than through our local m3u parser.
// file:// is included so an M3U line like "file:///media/movie.mkv" is taken verbatim
// instead of being mangled into "<playlist-dir>/file:///media/movie.mkv".
constexpr std::array<std::string_view, 4> kUrlSchemes{{"file://", "http://", "https://", "ftp://"}};

[[nodiscard]] auto AsciiToLower(char c) noexcept -> char {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
}

/// Case-insensitive ASCII comparison. Fine for our literal extensions and URL schemes;
/// would NOT case-fold UTF-8 correctly (no callers feed it non-ASCII input).
[[nodiscard]] auto IEquals(std::string_view a, std::string_view b) noexcept -> bool {
    return std::ranges::equal(a, b, [](char l, char r) noexcept -> bool { return AsciiToLower(l) == AsciiToLower(r); });
}

[[nodiscard]] auto HasExtension(std::string_view path, std::string_view extension) noexcept -> bool {
    if (path.size() < extension.size()) {
        return false;
    }
    return IEquals(path.substr(path.size() - extension.size()), extension);
}

[[nodiscard]] auto HasUrlScheme(std::string_view path) noexcept -> bool {
    return std::ranges::any_of(kUrlSchemes, [path](std::string_view scheme) noexcept -> bool {
        return path.size() >= scheme.size() && IEquals(path.substr(0, scheme.size()), scheme);
    });
}

/// Typed accessor for the primary device. The mediaplayer feed surface (OpenForMediaPlayer,
/// SubmitVideoPacket, ClearForMediaPlayer, ...) lives on cVaapiDevice; if some other plugin
/// is primary, playback through this path is not possible -- callers must handle nullptr.
[[nodiscard]] auto FindPrimaryVaapiDevice() noexcept -> cVaapiDevice * {
    auto *primary = cDevice::PrimaryDevice();
    return dynamic_cast<cVaapiDevice *>(primary);
}

[[nodiscard]] auto Basename(std::string_view path) -> std::string {
    if (const auto pos = path.find_last_of('/'); pos != std::string_view::npos) {
        return std::string{path.substr(pos + 1)};
    }
    return std::string{path};
}

[[nodiscard]] auto Dirname(std::string_view path) -> std::string {
    if (const auto pos = path.find_last_of('/'); pos != std::string_view::npos) {
        return std::string{path.substr(0, pos)};
    }
    return ".";
}

/// Format a millisecond duration as @c h:mm:ss. Used by the replay bar (SetCurrent / SetTotal)
/// and the info dialog. Negative / zero inputs render as @c 0:00:00.
[[nodiscard]] auto FormatHms(int ms) -> cString {
    if (ms <= 0) {
        return "0:00:00";
    }
    const int sec = ms / 1000;
    return cString::sprintf("%d:%02d:%02d", sec / 3600, (sec / 60) % 60, sec % 60);
}

/// Format a byte count as a whole number of MiB (rounded to nearest), labelled "MB".
/// Used by the file browser to annotate each media file. A non-empty file never
/// rounds to "0 MB": sub-half-MiB sizes are floored up to "1 MB" so the column
/// always reflects that there is content.
[[nodiscard]] auto FormatSizeMb(std::uintmax_t bytes) -> cString {
    constexpr std::uintmax_t BYTES_PER_MIB = 1024U * 1024U;
    std::uintmax_t mib = (bytes + (BYTES_PER_MIB / 2)) / BYTES_PER_MIB;
    if (mib == 0 && bytes > 0) {
        mib = 1;
    }
    return cString::sprintf("%ju MB", mib);
}

[[nodiscard]] auto Trim(std::string_view s) -> std::string_view {
    constexpr std::string_view kWhitespace = " \t\r\n";
    const auto start = s.find_first_not_of(kWhitespace);
    if (start == std::string_view::npos) {
        return {};
    }
    const auto end = s.find_last_not_of(kWhitespace);
    return s.substr(start, end - start + 1);
}

/// Luma bit depth derived from AVCodecParameters. Layered fallbacks:
///   1. AVCodecParameters::format -- the AVPixelFormat libavformat assigns after
///      find_stream_info. Authoritative for every mainstream container.
///   2. AVCodecParameters::bits_per_raw_sample -- set by some demuxers when format isn't.
///   3. Profile heuristic -- last resort. Only honors profiles that are unambiguously
///      10-bit by spec: H.264 HIGH_10 and HEVC MAIN_10. REXT / HIGH_422 / HIGH_444 are
///      intentionally NOT special-cased: they straddle 8 / 10 / 12-bit and a bad guess
///      sends the wrong row of kVideoBackendTable to the decoder. Defaulting to k8 there
///      sacrifices a HW open attempt to FFmpeg's get_format SW fallback, which is safe.
///
/// >10-bit streams are reported as k10. The backend table has no row for 12-bit, so
/// SelectVideoBackendCap returns nullptr and the decoder opens SW.
[[nodiscard]] auto VideoFormatBitDepth(const AVCodecParameters *p) noexcept -> BitDepth {
    if (p == nullptr) {
        return BitDepth::k8;
    }
    if (p->format != AV_PIX_FMT_NONE) {
        if (const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(p->format));
            desc != nullptr) {
            return desc->comp[0].depth >= 10 ? BitDepth::k10 : BitDepth::k8;
        }
    }
    if (p->bits_per_raw_sample >= 10) {
        return BitDepth::k10;
    }
    if (p->codec_id == AV_CODEC_ID_H264 && p->profile == AV_PROFILE_H264_HIGH_10) {
        return BitDepth::k10;
    }
    if (p->codec_id == AV_CODEC_ID_HEVC && p->profile == AV_PROFILE_HEVC_MAIN_10) {
        return BitDepth::k10;
    }
    // VP9 Profile 2/3 are 10/12-bit by spec; profile alone is sufficient (unlike AV1 Profile 0
    // and VVC Main 10 which span both bit-depths and rely on the pixel-format descriptor above).
    if (p->codec_id == AV_CODEC_ID_VP9 && (p->profile == AV_PROFILE_VP9_2 || p->profile == AV_PROFILE_VP9_3)) {
        return BitDepth::k10;
    }
    return BitDepth::k8;
}

} // namespace

// ============================================================================
// === PLAYLIST PARSING ===
// ============================================================================

auto ParseM3U(std::string_view playlistPath) -> std::vector<PlaylistEntry> {
    std::vector<PlaylistEntry> result;

    // Canonicalize first: resolves symlinks and gives us a stable parent directory for
    // relative-URI resolution below. Failure here is fatal -- a non-existent playlist
    // can't yield entries.
    std::error_code ec;
    const auto canonical = std::filesystem::canonical(std::string{playlistPath}, ec);
    if (ec) {
        esyslog("vaapivideo/mediaplayer: playlist %.*s: %s", static_cast<int>(playlistPath.size()), playlistPath.data(),
                ec.message().c_str());
        return result;
    }

    const std::string parentDir = canonical.parent_path().string();

    FILE *fp = std::fopen(canonical.c_str(), "r");
    if (fp == nullptr) [[unlikely]] {
        esyslog("vaapivideo/mediaplayer: cannot open playlist %s: %s", canonical.c_str(), std::strerror(errno));
        return result;
    }

    // M3U grammar we accept: any line not starting with '#' is a URI; "#EXTINF:duration,title"
    // optionally precedes a URI and supplies its display title. All other '#'-lines are ignored.
    std::string pendingTitle;
    std::array<char, 4096> lineBuf{};
    while (std::fgets(lineBuf.data(), static_cast<int>(lineBuf.size()), fp) != nullptr) {
        const auto line = Trim(std::string_view{lineBuf.data()});
        if (line.empty()) {
            continue;
        }
        if (line.front() == '#') {
            constexpr std::string_view kExtInf = "#EXTINF:";
            if (line.size() > kExtInf.size() && line.starts_with(kExtInf)) {
                if (const auto commaPos = line.find(',', kExtInf.size()); commaPos != std::string_view::npos) {
                    pendingTitle = std::string{Trim(line.substr(commaPos + 1))};
                }
            }
            continue;
        }

        // Relative paths are resolved against the playlist's parent directory; absolute paths
        // and any URL scheme are taken verbatim.
        std::string uri{line};
        if (!HasUrlScheme(uri) && !uri.empty() && uri.front() != '/') {
            std::string resolved = parentDir;
            resolved += '/';
            resolved += uri;
            uri = std::move(resolved);
        }

        PlaylistEntry entry;
        entry.uri = std::move(uri);
        entry.title = pendingTitle.empty() ? Basename(entry.uri) : pendingTitle;
        pendingTitle.clear();
        result.push_back(std::move(entry));
    }
    // Distinguish clean EOF from a truncated read (disk error mid-playlist) -- the latter
    // sets ferror but leaves earlier entries looking like a successful partial parse.
    if (std::ferror(fp) != 0) {
        esyslog("vaapivideo/mediaplayer: read error in playlist %s: %s", canonical.c_str(), std::strerror(errno));
        result.clear();
    }
    if (std::fclose(fp) != 0) {
        esyslog("vaapivideo/mediaplayer: fclose(%s): %s", canonical.c_str(), std::strerror(errno));
        result.clear();
    }

    isyslog("vaapivideo/mediaplayer: playlist %s -- %zu entries", canonical.c_str(), result.size());
    return result;
}

auto IsMediaUri(std::string_view path) noexcept -> bool {
    if (HasUrlScheme(path)) {
        return true;
    }
    return std::ranges::any_of(kMediaExtensions,
                               [path](std::string_view ext) noexcept -> bool { return HasExtension(path, ext); });
}

auto IsPlaylistUri(std::string_view path) noexcept -> bool {
    // HLS manifests (.m3u8 served over http(s)) belong in libavformat, not our local m3u parser.
    if (HasUrlScheme(path)) {
        return false;
    }
    return std::ranges::any_of(kPlaylistExtensions,
                               [path](std::string_view ext) noexcept -> bool { return HasExtension(path, ext); });
}

auto StartPlayback(std::vector<PlaylistEntry> entries) -> bool {
    if (entries.empty()) [[unlikely]] {
        esyslog("vaapivideo/mediaplayer: StartPlayback called with empty playlist");
        return false;
    }
    // Reject up front when the device is absent OR not yet attached to hardware: launching the
    // control would close the file browser (osEnd) and only then fail in Activate(true), leaving
    // the user staring at the previous channel with no error. Checking IsReady() here lets the
    // caller keep the menu open and surface "Cannot start playback" immediately.
    auto *vaapiDev = FindPrimaryVaapiDevice();
    if (vaapiDev == nullptr || !vaapiDev->IsReady()) [[unlikely]] {
        esyslog("vaapivideo/mediaplayer: no ready primary vaapivideo device -- playback rejected");
        return false;
    }
    // cControl::Launch takes ownership and destroys via cControl::Shutdown / next Launch.
    cControl::Launch(new cVaapiControl(std::move(entries)));
    return true;
}

// One-shot "return to browser" target. Set on Stop/EOF, consumed by MainMenuAction(); both run on
// the VDR main thread (never concurrent), so no lock is needed.
[[nodiscard]] static auto PendingBrowserReturn() -> std::optional<std::string> & {
    static std::optional<std::string> pending;
    return pending;
}

auto RequestReturnToBrowser(std::string jumpToPath) -> void {
    PendingBrowserReturn() = std::move(jumpToPath);
    // CallPlugin queues a k_Plugin key -> MainMenuAction() next main-loop pass. Fails only if another
    // plugin call is pending; drop our request then so an unrelated menu open won't show the browser.
    if (!cRemote::CallPlugin(PLUGIN_NAME)) [[unlikely]] {
        esyslog("vaapivideo/mediaplayer: CallPlugin busy -- cannot reopen file browser");
        PendingBrowserReturn().reset();
    }
}

auto TakeReturnToBrowser() -> std::optional<std::string> {
    auto &pending = PendingBrowserReturn();
    if (!pending.has_value()) {
        return std::nullopt;
    }
    std::optional<std::string> result = std::move(pending);
    pending.reset();
    return result;
}

// ============================================================================
// === cVaapiMediaSource ===
// ============================================================================

cVaapiMediaSource::~cVaapiMediaSource() noexcept { Close(); }

[[nodiscard]] auto cVaapiMediaSource::IoInterrupted() const noexcept -> bool {
    const bool stop = stopFlag != nullptr && stopFlag->load(std::memory_order_acquire);
    const bool command = interruptFlag != nullptr && interruptFlag->load(std::memory_order_acquire);
    return stop || command;
}

auto cVaapiMediaSource::Close() noexcept -> void {
    formatCtx.reset();
    videoStreamIndex = -1;
    audioStreamIndex = -1;
    videoExtradataStorage.clear();
    audioExtradataStorage.clear();
    videoInfo = VideoStreamInfo{};
    audioInfo = AudioStreamInfo{};
    ptsOrigin90k = AV_NOPTS_VALUE;
    discardAudioBefore90k = AV_NOPTS_VALUE;
    eofReached = false;
}

[[nodiscard]] auto cVaapiMediaSource::Open(std::string_view uri) -> bool {
    Close();

    const std::string uriStr{uri};
    // Pre-allocate so we can install the interrupt callback before avformat_open_input runs,
    // which makes the connection phase itself interruptible on shutdown for network URLs.
    AVFormatContext *raw = avformat_alloc_context();
    if (raw == nullptr) [[unlikely]] {
        esyslog("vaapivideo/mediaplayer: avformat_alloc_context failed");
        return false;
    }
    raw->interrupt_callback.callback = InterruptOnStop;
    raw->interrupt_callback.opaque = this;
    if (const int ret = avformat_open_input(&raw, uriStr.c_str(), nullptr, nullptr); ret < 0) {
        // AVERROR_EXIT = our interrupt callback fired; not a user-visible error.
        if (ret != AVERROR_EXIT) {
            esyslog("vaapivideo/mediaplayer: avformat_open_input(%s): %s", uriStr.c_str(), AvErr(ret).data());
        }
        avformat_close_input(&raw);
        return false;
    }
    formatCtx.reset(raw);

    if (const int ret = avformat_find_stream_info(formatCtx.get(), nullptr); ret < 0) {
        if (ret != AVERROR_EXIT) {
            esyslog("vaapivideo/mediaplayer: avformat_find_stream_info(%s): %s", uriStr.c_str(), AvErr(ret).data());
        }
        Close();
        return false;
    }

    videoStreamIndex = av_find_best_stream(formatCtx.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    audioStreamIndex = av_find_best_stream(formatCtx.get(), AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);

    if (videoStreamIndex < 0) {
        esyslog("vaapivideo/mediaplayer: no video stream in %s", uriStr.c_str());
        Close();
        return false;
    }

    PopulateStreamInfo();
    isyslog("vaapivideo/mediaplayer: opened %s -- video=%s audio=%s duration=%dms", uriStr.c_str(),
            avcodec_get_name(videoInfo.codecId), audioStreamIndex >= 0 ? avcodec_get_name(audioInfo.codecId) : "none",
            DurationMs());
    return true;
}

auto cVaapiMediaSource::PopulateStreamInfo() -> void {
    videoInfo = VideoStreamInfo{};
    audioInfo = AudioStreamInfo{};
    videoExtradataStorage.clear();
    audioExtradataStorage.clear();

    // Pick ptsOrigin90k = MAX of tracked-stream start_times so both streams begin at
    // rebased PTS 0 together. Files where audio (or any other stream) leads video by
    // hundreds of ms in the container would otherwise present as: audio plays from t=0,
    // video frames at rebased PTS=+lead sit "too far in future" against the audio clock
    // and get continuously dropped -- the user sees a still image with progressing
    // audio. Dropping the small leading-audio prefix (handled by the < 0 guard in
    // ReadPacket) starts both streams together, no audio-only intro.
    // Falls back to formatCtx->start_time if neither stream advertises start_time, then
    // to the first packet seen by ReadPacket's lazy-set path if even that is unknown.
    const int64_t videoStart =
        videoStreamIndex >= 0 ? StreamStart90k(formatCtx->streams[videoStreamIndex]) : AV_NOPTS_VALUE;
    const int64_t audioStart =
        audioStreamIndex >= 0 ? StreamStart90k(formatCtx->streams[audioStreamIndex]) : AV_NOPTS_VALUE;
    ptsOrigin90k = FormatStart90k(formatCtx.get()); // fall-back
    if (videoStart != AV_NOPTS_VALUE) {
        ptsOrigin90k = videoStart;
    }
    if (audioStart != AV_NOPTS_VALUE) {
        ptsOrigin90k = (ptsOrigin90k == AV_NOPTS_VALUE) ? audioStart : std::max(ptsOrigin90k, audioStart);
    }

    videoFps = 0.0;
    if (videoStreamIndex >= 0) {
        const AVStream *stream = formatCtx->streams[videoStreamIndex];
        videoTimeBase = stream->time_base;
        // avg_frame_rate is the most reliable container-level fps. Fall back to r_frame_rate
        // (raw frame rate) only when avg is unset; some MKV files lack avg but have r.
        const AVRational fr = (stream->avg_frame_rate.num > 0 && stream->avg_frame_rate.den > 0)
                                  ? stream->avg_frame_rate
                                  : stream->r_frame_rate;
        if (fr.num > 0 && fr.den > 0) {
            videoFps = av_q2d(fr);
            videoInfo.fpsNum = fr.num;
            videoInfo.fpsDen = fr.den;
        }
        const AVCodecParameters *p = stream->codecpar;
        videoInfo.codecId = p->codec_id;
        videoInfo.codedWidth = p->width;
        videoInfo.codedHeight = p->height;
        videoInfo.profile = p->profile;
        videoInfo.level = p->level;
        videoInfo.primaries = p->color_primaries;
        videoInfo.transfer = p->color_trc;
        videoInfo.colorSpace = p->color_space;
        videoInfo.range = p->color_range;
        // Bit depth drives SelectVideoBackendCap (HW-vs-SW decode decision). Profile alone is
        // ambiguous: HEVC REXT can be 8-16 bit, AV1 Main can be 8 or 10, etc. Inspect the
        // pixel format descriptor for the authoritative answer.
        videoInfo.bitDepth = VideoFormatBitDepth(p);
        // The PES path waits for an in-band SPS before opening the codec; the mediaplayer
        // path always has the container's extradata, so the decoder can open immediately.
        videoInfo.hasSps = true;
        CopyExtradata(p, videoExtradataStorage, videoInfo.extradata, videoInfo.extradataSize);
    }

    if (audioStreamIndex >= 0) {
        const AVStream *stream = formatCtx->streams[audioStreamIndex];
        audioTimeBase = stream->time_base;
        const AVCodecParameters *p = stream->codecpar;
        audioInfo.codecId = p->codec_id;
        audioInfo.sampleRate = p->sample_rate;
        // Always open ALSA at 2 channels and let swresample downmix from the container's
        // native layout (5.1, 7.1, ...) -- matches the live-TV path in cVaapiDevice::PlayAudio
        // (hardcoded to 2ch there too). Without this, 5.1 AC3 files open ALSA at 6 channels
        // and the plug device's automatic downmix produces audibly distorted output on a
        // stereo sink. Multi-channel passthrough would need separate plumbing.
        audioInfo.channels = 2;
        CopyExtradata(p, audioExtradataStorage, audioInfo.extradata, audioInfo.extradataSize);
    }
}

[[nodiscard]] auto cVaapiMediaSource::DurationMs() const noexcept -> int {
    if (!formatCtx || formatCtx->duration <= 0) {
        return 0;
    }
    return static_cast<int>(formatCtx->duration / (AV_TIME_BASE / 1000));
}

[[nodiscard]] auto cVaapiMediaSource::ReadPacket(AVPacket *out, MediaPacketStream &stream) -> int {
    if (!formatCtx) {
        return AVERROR_EOF;
    }
    if (eofReached) {
        return AVERROR_EOF;
    }

    // One demuxer cursor, one packet per call, tagged with its owning stream. Splitting
    // reads across separate video/audio methods would need per-stream side FIFOs that drop
    // packets when one stream races ahead -- audible glitches in practice. Demux-order
    // delivery lets libavformat pace the pump and avoids artificial drops.
    while (true) {
        // Interruptibility for local files: av_read_frame() polls the InterruptOnStop
        // callback only when libavformat performs blocking I/O. Streams composed of many
        // skipped packets (damaged TS with empty / untracked stream_index / pre-target
        // audio prefix) can spin in this loop without ever entering a blocking read, so
        // poll stopFlag at the top of each iteration too.
        if (stopFlag != nullptr && stopFlag->load(std::memory_order_acquire)) {
            return AVERROR_EXIT;
        }
        std::unique_ptr<AVPacket, FreeAVPacket> pkt{av_packet_alloc()};
        if (!pkt) [[unlikely]] {
            return AVERROR(ENOMEM);
        }
        const int ret = av_read_frame(formatCtx.get(), pkt.get());
        if (ret == AVERROR_EOF) {
            eofReached = true;
            return AVERROR_EOF;
        }
        if (ret == AVERROR(EAGAIN)) {
            return AVERROR(EAGAIN);
        }
        if (ret == AVERROR_EXIT) {
            // Interrupt callback fired. Two reasons, distinguished by which flag is set:
            //   - shutdown (stopFlag): pass AVERROR_EXIT through so Action() bails without the
            //     playlist-advance path -- otherwise a user-pressed exit logs "playlist exhausted".
            //   - seek/next (interruptFlag only): the source stays valid. Consume the interrupt
            //     and return EAGAIN so Action() loops back and services the pending command at the
            //     top of its loop. Do NOT set eofReached -- the post-seek read must succeed.
            if (stopFlag != nullptr && stopFlag->load(std::memory_order_acquire)) {
                eofReached = true;
                return AVERROR_EXIT;
            }
            if (interruptFlag != nullptr) {
                interruptFlag->store(false, std::memory_order_release);
            }
            return AVERROR(EAGAIN);
        }
        if (ret < 0) {
            esyslog("vaapivideo/mediaplayer: av_read_frame: %s", AvErr(ret).data());
            eofReached = true;
            return AVERROR_EOF;
        }

        // Skip empty / padding packets. Corrupt TS streams (e.g. tvheadend recordings) sometimes
        // emit size=0 packets that FFmpeg interprets downstream as drain markers.
        if (pkt->data == nullptr || pkt->size <= 0) {
            continue;
        }

        AVRational tb{};
        if (pkt->stream_index == videoStreamIndex) {
            tb = videoTimeBase;
            stream = MediaPacketStream::Video;
        } else if (pkt->stream_index == audioStreamIndex) {
            tb = audioTimeBase;
            stream = MediaPacketStream::Audio;
        } else {
            continue; // untracked (subtitle/data)
        }

        // Rebase to a zero-based 90 kHz timeline. Files with non-zero container start_time
        // (TS recordings, some MP4s) would otherwise hand huge absolute PTS values to the
        // downstream audio clock and break GetIndex() / Seek() math.
        pkt->pts = Rebase90k(pkt->pts, tb);
        pkt->dts = Rebase90k(pkt->dts, tb);
        // Audio packets sometimes carry DTS only (TS containers). Audio has no B-frame
        // reorder so PTS == DTS; video keeps its real PTS to preserve reorder offset.
        if (stream == MediaPacketStream::Audio && pkt->pts == AV_NOPTS_VALUE) {
            pkt->pts = pkt->dts;
        }

        const int64_t clock90k = PacketClock90k(pkt.get());
        // Drop pre-sync prefix (rebased clock < 0): PopulateStreamInfo() picks
        // ptsOrigin90k = MAX(stream.start_time) so the trailing stream defines t=0 and the
        // leading stream's prefix is discarded here -- both streams start together.
        if (clock90k != AV_NOPTS_VALUE && clock90k < 0) {
            continue;
        }
        // Post-seek audio discard: keep video preroll (rebuilds H.264/HEVC reference chain)
        // but drop audio earlier than the seek target. Without this the master clock anchors
        // at the earlier keyframe libavformat landed on, and the drain sits in a re-arm
        // freerun loop until audio plays forward to the requested target.
        if (stream == MediaPacketStream::Audio && discardAudioBefore90k != AV_NOPTS_VALUE) {
            if (clock90k == AV_NOPTS_VALUE || clock90k < discardAudioBefore90k) {
                continue;
            }
            discardAudioBefore90k = AV_NOPTS_VALUE;
        }

        av_packet_move_ref(out, pkt.get());
        return 0;
    }
}

[[nodiscard]] auto cVaapiMediaSource::Rebase90k(int64_t ts, AVRational tb) noexcept -> int64_t {
    // tb.num/den <= 0 guards damaged container metadata: av_rescale_q divides by tb.den
    // and would produce garbage / UB on a zero denominator. Returning NOPTS makes the
    // downstream pre-sync and discard checks skip the packet safely.
    if (ts == AV_NOPTS_VALUE || tb.num <= 0 || tb.den <= 0) {
        return AV_NOPTS_VALUE;
    }
    constexpr AVRational k90kHz{.num = 1, .den = 90000};
    const int64_t ts90k = av_rescale_q(ts, tb, k90kHz);
    if (ptsOrigin90k == AV_NOPTS_VALUE) {
        ptsOrigin90k = ts90k;
    }
    return ts90k - ptsOrigin90k;
}

auto cVaapiMediaSource::Flush() -> void {
    if (formatCtx) {
        avformat_flush(formatCtx.get());
    }
    eofReached = false;
}

[[nodiscard]] auto cVaapiMediaSource::Seek(int64_t targetPts90k) -> bool {
    if (!formatCtx || videoStreamIndex < 0) {
        return false;
    }
    // Reset upfront so a failed seek doesn't leave a stale discard window armed.
    discardAudioBefore90k = AV_NOPTS_VALUE;
    // Convert from 90 kHz to the video stream's time base (av_seek_frame's units). Player Seek()
    // talks the zero-based timeline; we re-add the origin offset so we land at the matching
    // wall-clock keyframe inside the container's native timeline.
    const AVRational dstTb = formatCtx->streams[videoStreamIndex]->time_base;
    constexpr AVRational k90kHz{.num = 1, .den = 90000};
    const int64_t origin = (ptsOrigin90k == AV_NOPTS_VALUE) ? 0 : ptsOrigin90k;
    const int64_t seekTs = av_rescale_q(std::max<int64_t>(targetPts90k, 0) + origin, k90kHz, dstTb);
    const int ret = av_seek_frame(formatCtx.get(), videoStreamIndex, seekTs, AVSEEK_FLAG_BACKWARD);
    if (ret < 0) {
        esyslog("vaapivideo/mediaplayer: av_seek_frame: %s", AvErr(ret).data());
        return false;
    }
    Flush();
    discardAudioBefore90k = audioStreamIndex >= 0 ? std::max<int64_t>(targetPts90k, 0) : AV_NOPTS_VALUE;
    return true;
}

// ============================================================================
// === cVaapiPlayer ===
// ============================================================================

cVaapiPlayer::cVaapiPlayer(std::vector<PlaylistEntry> entries)
    : cPlayer(pmAudioVideo), cThread("vaapi mediaplayer demux"), playlist(std::move(entries)) {
    if (playlist.empty()) {
        esyslog("vaapivideo/mediaplayer: cVaapiPlayer constructed with empty playlist");
    }
}

cVaapiPlayer::~cVaapiPlayer() noexcept {
    // Order: raise stop, wake pause-waiter, join (bounded 3s via InterruptOnStop on slow
    // network URLs), drop the source under the mutex. ioInterrupt is raised too so a thread
    // parked in a network read bails immediately rather than after the I/O timeout.
    stopping.store(true, std::memory_order_release);
    ioInterrupt.store(true, std::memory_order_release);
    {
        const cMutexLock lock(&pauseMutex);
        pauseCondition.Broadcast();
    }
    Cancel(3);
    const cMutexLock lock(&sourceMutex);
    source.reset();
}

[[nodiscard]] auto cVaapiPlayer::CurrentPositionMs() const noexcept -> int {
    auto *vaapiDev = FindPrimaryVaapiDevice();
    if (vaapiDev == nullptr) {
        return 0;
    }
    if (const int64_t stc = vaapiDev->GetSTC(); stc >= 0) {
        return static_cast<int>(stc / TICKS_PER_MS);
    }
    // STC briefly NOPTS post-Clear (~50ms). Fall back to the last seek target so a rapid
    // follow-up Seek() computes its delta against the right base, not 0.
    const int target = pendingSeekTargetMs.load(std::memory_order_acquire);
    return (target >= 0) ? target : 0;
}

[[nodiscard]] auto cVaapiPlayer::Lookahead90k(const cVaapiDevice *vaapiDev) const noexcept -> int64_t {
    if (vaapiDev == nullptr) {
        return AV_NOPTS_VALUE;
    }
    // During pause the demux loop is intentionally stopped and the audio clock extrapolates from
    // its last anchor while ALSA is dropped, so lastAudio - audioClock turns into a fake negative
    // value (the status log surfaced this as e.g. lookahead=-600ms while Paused). Returning NOPTS
    // skips both the throttle and the misleading status line; the demuxer is paused-gated above
    // it anyway.
    if (paused.load(std::memory_order_acquire)) {
        return AV_NOPTS_VALUE;
    }
    const int64_t audioClock = vaapiDev->GetAudioClock();
    const int64_t lastAudio = latestAudioPts90k.load(std::memory_order_acquire);
    if (audioClock == AV_NOPTS_VALUE || lastAudio == AV_NOPTS_VALUE) {
        return AV_NOPTS_VALUE;
    }
    return lastAudio - audioClock;
}

[[nodiscard]] auto cVaapiPlayer::OpenCurrentEntry() -> bool {
    const cMutexLock lock(&sourceMutex);
    const size_t idx = currentIndex.load(std::memory_order_relaxed);
    if (idx >= playlist.size()) {
        return false;
    }
    // Commit `source` only after every step succeeds, so other threads never observe a
    // partially-initialized source between Open and OpenForMediaPlayer.
    auto nextSource = std::make_unique<cVaapiMediaSource>(&stopping, &ioInterrupt);
    if (!nextSource->Open(playlist.at(idx).uri)) {
        return false;
    }
    // Local is `vaapiDev` (not `device`) to avoid shadowing cPlayer::device under -Wshadow.
    auto *vaapiDev = FindPrimaryVaapiDevice();
    if (vaapiDev == nullptr) {
        esyslog("vaapivideo/mediaplayer: primary device is not vaapivideo -- cannot play");
        return false;
    }
    if (!vaapiDev->OpenForMediaPlayer(nextSource->VideoInfo(), nextSource->AudioInfo())) {
        vaapiDev->ClearForMediaPlayer();
        return false;
    }
    source = std::move(nextSource);
    // New entry = new PTS timeline; the throttle's high-water mark and the seek-target
    // fallback must not carry over from the previous entry.
    latestAudioPts90k.store(AV_NOPTS_VALUE, std::memory_order_release);
    pendingSeekTargetMs.store(-1, std::memory_order_release);
    return true;
}

auto cVaapiPlayer::CloseCurrentEntry() noexcept -> void {
    const cMutexLock lock(&sourceMutex);
    if (auto *vaapiDev = FindPrimaryVaapiDevice(); vaapiDev != nullptr) {
        vaapiDev->ClearForMediaPlayer();
    }
    source.reset();
}

auto cVaapiPlayer::Activate(bool On) -> void {
    // On=true: attached to device. Open first entry, start demux. Failure -> Stopped
    //          and cVaapiControl exits via IsFinished().
    // On=false: about to detach. Mirror the dtor's shutdown; idempotent.
    if (On) {
        if (!OpenCurrentEntry()) {
            esyslog("vaapivideo/mediaplayer: Activate(true) failed -- no entry could be opened");
            state.store(State::Stopped, std::memory_order_release);
            return;
        }
        state.store(State::Playing, std::memory_order_release);
        Start();
    } else {
        stopping.store(true, std::memory_order_release);
        ioInterrupt.store(true, std::memory_order_release); // break a parked network read so Cancel(3) joins fast
        {
            const cMutexLock lock(&pauseMutex);
            pauseCondition.Broadcast();
        }
        Cancel(3);
        CloseCurrentEntry();
        state.store(State::Stopped, std::memory_order_release);
    }
}

auto cVaapiPlayer::SetPaused(bool wantPaused) -> void {
    const bool wasPaused = paused.exchange(wantPaused, std::memory_order_acq_rel);
    if (wasPaused == wantPaused) {
        return;
    }
    // BOTH halves are required: DeviceFreeze() halts the audio master clock (else resume
    // re-anchors with a stutter); the demux flag halts packet flow (else queues fill while
    // frozen and OOM on long pauses).
    if (wantPaused) {
        DeviceFreeze();
        state.store(State::Paused, std::memory_order_release);
    } else {
        DevicePlay();
        state.store(State::Playing, std::memory_order_release);
    }
    const cMutexLock lock(&pauseMutex);
    pauseCondition.Broadcast();
}

auto cVaapiPlayer::Seek(int64_t deltaMs) -> void {
    if (deltaMs == 0) {
        return;
    }
    // fetch_add (not store) so rapid key repeats sum: 5x kRight in one demux cycle = +50s,
    // not +10s. Per-key repeat shaping is RcRepeatDelay/RcRepeatDelta in setup.conf.
    seekDeltaMs.fetch_add(deltaMs, std::memory_order_relaxed);
    seekPending.store(true, std::memory_order_release);
    // Break a blocking network read so the seek is serviced now, not after the I/O timeout.
    ioInterrupt.store(true, std::memory_order_release);
    // If paused, wake the demux thread so it can service the seek immediately.
    const cMutexLock lock(&pauseMutex);
    pauseCondition.Broadcast();
}

auto cVaapiPlayer::Next() -> void {
    nextRequested.store(true, std::memory_order_release);
    ioInterrupt.store(true, std::memory_order_release); // see Seek(): break a parked network read
    const cMutexLock lock(&pauseMutex);
    pauseCondition.Broadcast();
}

[[nodiscard]] auto cVaapiPlayer::Title() const -> std::string {
    const size_t idx = currentIndex.load(std::memory_order_relaxed);
    return (idx < playlist.size()) ? playlist.at(idx).title : std::string{};
}

[[nodiscard]] auto cVaapiPlayer::CurrentUri() const -> std::string {
    // Unlocked read like Title() (`playlist` is fixed after construction). Clamp the index: EOF
    // leaves it one past the end, so a Stop after a file finishes still resolves to that file.
    if (playlist.empty()) {
        return {};
    }
    size_t idx = currentIndex.load(std::memory_order_relaxed);
    if (idx >= playlist.size()) {
        idx = playlist.size() - 1;
    }
    return playlist.at(idx).uri;
}

[[nodiscard]] auto cVaapiPlayer::FramesPerSecond() -> double {
    // Skins use this for the ".ff" frame-count suffix; fall back to cPlayer's default 25.
    const cMutexLock lock(&sourceMutex);
    if (source) {
        const double fps = source->VideoFps();
        if (fps > 0.0) {
            return fps;
        }
    }
    return cPlayer::FramesPerSecond();
}

[[nodiscard]] auto cVaapiPlayer::InfoText() const -> std::string {
    const cMutexLock lock(&sourceMutex);
    if (!source) {
        return {};
    }
    const auto [w, h] = source->VideoCodedSize();
    const auto &v = source->VideoInfo();
    const auto &a = source->AudioInfo();
    const double fps = source->VideoFps();
    const size_t idx = currentIndex.load(std::memory_order_relaxed);

    std::string text;
    if (idx < playlist.size()) {
        text += std::format("Title:    {}\n", playlist.at(idx).title);
        text += std::format("URI:      {}\n\n", playlist.at(idx).uri);
    }
    text += std::format("Duration: {}\n", *FormatHms(source->DurationMs()));
    text += std::format("Video:    {} {}x{}", avcodec_get_name(v.codecId), w, h);
    if (fps > 0.0) {
        text += std::format(" @ {:.3f} fps", fps);
    }
    text += "\n";
    if (a.codecId != AV_CODEC_ID_NONE) {
        text += std::format("Audio:    {} {} Hz {} ch\n", avcodec_get_name(a.codecId), a.sampleRate, a.channels);
    } else {
        text += "Audio:    (none)\n";
    }
    if (playlist.size() > 1) {
        text += std::format("Playlist: {}/{}\n", idx + 1, playlist.size());
    }
    return text;
}

[[nodiscard]] auto cVaapiPlayer::GetReplayMode(bool &Play, bool &Forward, int &Speed) -> bool {
    Play = !paused.load(std::memory_order_acquire);
    Forward = true;
    Speed = -1; // trick-speed is out of scope for this player
    return true;
}

[[nodiscard]] auto cVaapiPlayer::GetIndex(int &Current, int &Total, bool /*SnapToIFrame*/) -> bool {
    Current = 0;
    Total = 0;
    const cMutexLock lock(&sourceMutex);
    if (!source) {
        return false;
    }
    Total = source->DurationMs();
    // CurrentPositionMs() falls back to pendingSeekTargetMs during the post-Clear NOPTS
    // window. Reading GetSTC() directly would snap the bar to 0 every time the OSD opens
    // during a seek burst.
    Current = CurrentPositionMs();
    return true;
}

auto cVaapiPlayer::PerformSeek(int64_t deltaMs) -> void {
    const cMutexLock lock(&sourceMutex);
    if (!source) {
        return;
    }
    auto *vaapiDev = FindPrimaryVaapiDevice();
    if (vaapiDev == nullptr) {
        return;
    }

    const int currentMs = CurrentPositionMs();
    const int totalMs = source->DurationMs();
    // 1 s tail margin: landing past the last keyframe would look like a hang.
    int64_t targetMs = std::max<int64_t>(0, static_cast<int64_t>(currentMs) + deltaMs);
    if (totalMs > 0 && targetMs > totalMs - 1000) {
        targetMs = std::max<int64_t>(0, totalMs - 1000);
    }
    const int64_t targetPts90k = targetMs * TICKS_PER_MS;

    state.store(State::Seeking, std::memory_order_release);

    // Seek the source FIRST; if it fails we leave the device state untouched so the user keeps
    // playing the old position instead of staring at a blanked frame after a wiped pipeline.
    if (!source->Seek(targetPts90k)) {
        esyslog("vaapivideo/mediaplayer: seek to %lldms failed", static_cast<long long>(targetMs));
        state.store(paused.load(std::memory_order_acquire) ? State::Paused : State::Playing, std::memory_order_release);
        return;
    }

    vaapiDev->FlushForSeek();
    // Reset the throttle high-water mark so a post-seek PTS smaller than pre-seek doesn't
    // stall the demuxer until audio "catches up" to a stale value.
    latestAudioPts90k.store(AV_NOPTS_VALUE, std::memory_order_release);
    // CurrentPositionMs() returns this while GetSTC() is briefly NOPTS post-flush; without
    // it a rapid follow-up Seek() would compute its delta against 0.
    pendingSeekTargetMs.store(static_cast<int>(targetMs), std::memory_order_release);
    dsyslog("vaapivideo/mediaplayer: seek %+lldms -- %dms -> %lldms (total=%dms)", static_cast<long long>(deltaMs),
            currentMs, static_cast<long long>(targetMs), totalMs);
    state.store(paused.load(std::memory_order_acquire) ? State::Paused : State::Playing, std::memory_order_release);
}

auto cVaapiPlayer::LogStatus() -> void {
    int totalMs = 0;
    {
        const cMutexLock lock(&sourceMutex);
        if (source) {
            totalMs = source->DurationMs();
        }
    }
    const int posMs = CurrentPositionMs();
    const int64_t lookahead = Lookahead90k(FindPrimaryVaapiDevice());
    const int lookaheadMs = (lookahead != AV_NOPTS_VALUE) ? static_cast<int>(lookahead / TICKS_PER_MS) : -1;
    const size_t idx = currentIndex.load(std::memory_order_relaxed);
    const size_t totalCount = playlist.size();
    dsyslog("vaapivideo/mediaplayer: status mode=%s pos=%dms total=%dms entry=%zu/%zu lookahead=%dms",
            StateName(state.load(std::memory_order_acquire)), posMs, totalMs, idx + 1, totalCount, lookaheadMs);
}

auto cVaapiPlayer::DrainTailAtEof() -> void {
    auto *vaapiDev = FindPrimaryVaapiDevice();
    if (vaapiDev == nullptr) {
        return;
    }
    // Bail the instant the user wants something else, so a held tail can't delay it. Pause matters
    // most: it stops the presenter, so depth freezes and the stall watchdog would otherwise advance
    // the playlist behind the user's back -- instead Action()'s pause branch holds and the drain
    // resumes when EOF is re-detected on Play.
    const auto aborted = [this]() noexcept -> bool {
        return stopping.load(std::memory_order_acquire) || paused.load(std::memory_order_acquire) ||
               seekPending.load(std::memory_order_acquire) || nextRequested.load(std::memory_order_acquire);
    };
    // Wait for depthFn to reach 0, bailing on user command, hard timeout, or stall. A real-time
    // presenter advances every frame, so no decrease within STALL_MS means a wedged pipeline.
    const auto drainUntilEmpty = [&](auto &&depthFn) noexcept -> void {
        const cTimeMs cap(MEDIAPLAYER_EOF_DRAIN_TIMEOUT_MS);
        cTimeMs sinceProgress;
        size_t lastDepth = SIZE_MAX;
        while (true) {
            const size_t depth = depthFn();
            if (depth == 0) {
                return;
            }
            // Reset on ANY decrease vs the previous reading, not a lifetime low: phase 2's codec drain
            // can RAISE depth (tail flushed into the reserve), and a lifetime-min tracker would then
            // read the legitimate drain that follows as a stall and cut the tail early.
            if (depth < lastDepth) {
                sinceProgress.Set();
            }
            lastDepth = depth;
            if (aborted() || cap.TimedOut() ||
                sinceProgress.Elapsed() > static_cast<uint64_t>(MEDIAPLAYER_EOF_DRAIN_STALL_MS)) {
                return;
            }
            cCondWait::SleepMs(MEDIAPLAYER_BACKPRESSURE_SLEEP_MS);
        }
    };

    // Phase 1: feed every queued packet into the codec first. A codec flush with packets still
    // queued re-arms mid-stream, leaving the rest undecodable until the next I-frame (lost tail).
    drainUntilEmpty([vaapiDev]() noexcept -> size_t { return vaapiDev->MediaPlayerDecodeQueueDepth(); });
    if (aborted()) {
        return;
    }
    // Phase 2: flush the reorder tail into the reserve, then drain it to the screen at real-time pace.
    vaapiDev->RequestMediaPlayerEosDrain();
    drainUntilEmpty([vaapiDev]() noexcept -> size_t { return vaapiDev->MediaPlayerBufferedDepth(); });
}

auto cVaapiPlayer::AdvancePlaylist() -> void {
    const size_t next = currentIndex.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (next >= playlist.size()) {
        // Release the source at final EOF so libavformat buffers / network sockets don't
        // sit pinned until the control destructs.
        CloseCurrentEntry();
        state.store(State::Eof, std::memory_order_release);
        isyslog("vaapivideo/mediaplayer: playlist exhausted");
        return;
    }
    CloseCurrentEntry();
    if (!OpenCurrentEntry()) {
        esyslog("vaapivideo/mediaplayer: failed to open next entry, stopping");
        state.store(State::Eof, std::memory_order_release);
    }
}

auto cVaapiPlayer::Action() -> void {
    // Each iteration: service pending command (seek/next) -> wait if paused -> honor device
    // backpressure -> pull and dispatch one packet. EOF advances the playlist; EAGAIN sleeps.
    const std::unique_ptr<AVPacket, FreeAVPacket> packet{av_packet_alloc()};
    if (!packet) [[unlikely]] {
        esyslog("vaapivideo/mediaplayer: AVPacket allocation failed -- aborting");
        state.store(State::Stopped, std::memory_order_release);
        return;
    }

    lastStatusLog.Set(0); // Emit the first status line immediately on the first iteration.
    // packetPending carries one read packet across iterations when the device's submit
    // queue is briefly full (audio cAudioProcessor::EnqueuePacket returns false on overflow).
    // Without this retry the lookahead throttle would advance latestAudioPts90k even though
    // the packet was silently dropped, breaking A/V sync subtly until the next forced flush.
    MediaPacketStream packetStream{MediaPacketStream::Video};
    bool packetPending = false;

    while (!stopping.load(std::memory_order_acquire)) {
        // -- periodic status diagnostic ------------------------------------------
        if (lastStatusLog.TimedOut()) {
            LogStatus();
            lastStatusLog.Set(STATUS_LOG_INTERVAL_MS);
        }

        // -- seek ----------------------------------------------------------------
        // Repeat-rate shaping (so a held key does not pile up many FlushForSeek -> snd_pcm_drop
        // cycles) is handled upstream in VDR's RcRepeatDelay / RcRepeatDelta. Any seeks that
        // arrive between iterations still coalesce via the fetch_add in Seek().
        if (seekPending.exchange(false, std::memory_order_acq_rel)) {
            // The command is now latched, so the interrupt that broke our blocking read has done
            // its job. Clear it before PerformSeek(): av_seek_frame() shares this interrupt_callback,
            // so a still-set ioInterrupt (from the triggering seek, or one that fired between the
            // read abort and here) would abort the seek itself and fail it. A genuinely newer seek
            // arriving during PerformSeek re-sets the flag and is serviced on the next iteration.
            ioInterrupt.store(false, std::memory_order_release);
            // Two-step (seekPending + seekDeltaMs) is not atomic. A key arriving between
            // this exchange(false) and the delta swap below can leave seekPending=true with
            // an already-drained delta on the next iteration; PerformSeek(0) would then
            // flush decoder + audio and seek to the current position for no movement.
            if (const int64_t deltaMs = seekDeltaMs.exchange(0, std::memory_order_relaxed); deltaMs != 0) {
                // Discard any held pre-seek packet: PerformSeek flushes the device's video +
                // audio queues, so submitting a pre-seek AU after the flush would push a
                // frame at the old PTS through the new GOP and contaminate the catch-up
                // window (visible as repeating stale-jitter / catch-up cycles every 2 s).
                if (packetPending) {
                    av_packet_unref(packet.get());
                    packetPending = false;
                }
                PerformSeek(deltaMs);
            }
        }

        // -- next ----------------------------------------------------------------
        if (nextRequested.exchange(false, std::memory_order_acq_rel)) {
            // Clear before AdvancePlaylist(): it opens the next entry's source (a network connect
            // for URLs), which also polls this interrupt_callback -- a still-set ioInterrupt would
            // abort the open. Same rationale as the seek path.
            ioInterrupt.store(false, std::memory_order_release);
            // Same rationale as the seek path: a held packet belongs to the previous entry.
            if (packetPending) {
                av_packet_unref(packet.get());
                packetPending = false;
            }
            AdvancePlaylist();
            if (state.load(std::memory_order_acquire) == State::Eof) {
                break;
            }
        }

        // -- pause ---------------------------------------------------------------
        if (paused.load(std::memory_order_acquire)) {
            const cMutexLock lock(&pauseMutex);
            if (paused.load(std::memory_order_acquire) && !stopping.load(std::memory_order_acquire) &&
                !seekPending.load(std::memory_order_acquire) && !nextRequested.load(std::memory_order_acquire)) {
                pauseCondition.TimedWait(pauseMutex, DEMUX_PAUSE_WAKEUP_MS);
            }
            continue;
        }

        // -- backpressure ---------------------------------------------------------
        auto *vaapiDev = FindPrimaryVaapiDevice();
        if (vaapiDev == nullptr) [[unlikely]] {
            cCondWait::SleepMs(DEMUX_IDLE_SLEEP_MS);
            continue;
        }
        // Backpressure gates NEW demux reads only. A packetPending has already advanced
        // libavformat's cursor; retry it even while queues are high. Otherwise a held audio
        // packet can be blocked by the very audioHighwater condition that submitting it
        // would help clear (the SubmitAudioPacket highwater check is the proper pacing
        // signal -- false return -> packetPending stays true -> retry next iter).
        if (!packetPending && vaapiDev->IsMediaPlayerBackpressured()) {
            cCondWait::SleepMs(MEDIAPLAYER_BACKPRESSURE_SLEEP_MS);
            continue;
        }

        // -- read and dispatch one packet in demux order --------------------------
        // AdvancePlaylist() re-locks sourceMutex via Close/OpenCurrentEntry; cMutex is
        // non-recursive (PTHREAD_MUTEX_ERRORCHECK), so we defer it past the lock scope.
        bool didWork = false;
        bool advanceAfterUnlock = false;
        if (!packetPending) {
            // -- real-time pacing ------------------------------------------------
            // libavformat reads local files much faster than wall-clock; without a lookahead
            // gate the decoder queue saturates and HW decode outruns audio-paced drain. Push
            // freely until audio anchors (Lookahead90k returns NOPTS then). Gate only NEW reads:
            // a held packet has already left the demuxer cursor and is not yet reflected in
            // latestAudioPts90k (that updates on successful submit only), so the lookahead doesn't
            // even account for it. Worse, if the held packet is audio, blocking its retry here
            // because video pushed the lookahead high would withhold the very packet that lets
            // the audio clock advance -- a self-inflicted underrun. Retries fall straight through
            // to the submit block below.
            if (const int64_t lookahead = Lookahead90k(vaapiDev);
                lookahead != AV_NOPTS_VALUE && lookahead > MEDIAPLAYER_MAX_LOOKAHEAD_90K) {
                cCondWait::SleepMs(MEDIAPLAYER_BACKPRESSURE_SLEEP_MS);
                continue;
            }

            const cMutexLock lock(&sourceMutex);
            if (!source) {
                state.store(State::Eof, std::memory_order_release);
                break;
            }
            const int ret = source->ReadPacket(packet.get(), packetStream);
            if (ret == 0) {
                packetPending = true;
            } else if (ret == AVERROR_EOF) {
                advanceAfterUnlock = true;
            } else if (ret == AVERROR_EXIT) {
                // Interrupted by shutdown; let the outer loop's stopping check exit cleanly.
                break;
            }
            // AVERROR(EAGAIN) falls through to the idle sleep below.
        }

        if (packetPending) {
            const bool submitted = (packetStream == MediaPacketStream::Video)
                                       ? vaapiDev->SubmitVideoPacket(packet.get())
                                       : vaapiDev->SubmitAudioPacket(packet.get());
            if (submitted) {
                // Lookahead reference tracks the AUDIO (master-clock) stream only. In an MPEG-TS
                // the video PTS leads the audio PTS at the same file position by a large mux
                // interleave offset (hundreds of ms .. ~1.5 s). If this reference were the max of
                // BOTH streams it would be dominated by the leading video PTS, so the lookahead
                // (= reference - audioClock) would read mux_offset + audio_buffer_depth and trip
                // MEDIAPLAYER_MAX_LOOKAHEAD_90K while the audio buffer is still tiny -- throttling
                // the demuxer, starving the audio queue, stalling the master clock, and wedging
                // the post-seek video-ahead drain in a re-arm-freerun loop that never converges.
                // Keying off audio measures the real audio buffer depth, immune to the video lead.
                // (Video-only streams leave this NOPTS -> Lookahead90k returns NOPTS, no throttle;
                // the jitterBuf-depth backpressure bounds them while the clock is unanchored.)
                // Monotonic-max: Action() is the sole writer, so a plain load/store suffices.
                // PacketClock90k falls back to DTS so TS audio packets carrying only DTS advance it.
                if (packetStream == MediaPacketStream::Audio) {
                    if (const int64_t packetPts = PacketClock90k(packet.get()); packetPts != AV_NOPTS_VALUE) {
                        const int64_t prev = latestAudioPts90k.load(std::memory_order_relaxed);
                        if (prev == AV_NOPTS_VALUE || packetPts > prev) {
                            latestAudioPts90k.store(packetPts, std::memory_order_release);
                        }
                    }
                }
                av_packet_unref(packet.get());
                packetPending = false;
                didWork = true;
            }
            // !submitted: keep packetPending=true so the next iteration retries after the
            // device drains; backpressure / lookahead checks above bound the spin frequency.
        }

        if (advanceAfterUnlock) {
            // Natural EOF: present the buffered tail before teardown so playback runs to the real end.
            DrainTailAtEof();
            if (stopping.load(std::memory_order_acquire)) {
                break;
            }
            // The drain bailed on a user command: let the loop top service it instead of advancing
            // (pause holds; a seek resets eofReached; the tail re-drains once EOF is hit again).
            if (paused.load(std::memory_order_acquire) || seekPending.load(std::memory_order_acquire) ||
                nextRequested.load(std::memory_order_acquire)) {
                continue;
            }
            AdvancePlaylist();
            if (state.load(std::memory_order_acquire) == State::Eof) {
                break;
            }
            continue;
        }

        if (!didWork) {
            cCondWait::SleepMs(DEMUX_IDLE_SLEEP_MS);
        }
    }
}

// ============================================================================
// === cVaapiControl ===
// ============================================================================

cVaapiControl::cVaapiControl(cVaapiPlayer *typedPlayer) : cControl(typedPlayer), player(typedPlayer) {
    barTimeout.Set(0);
    lastBarRefresh.Set(0);

    if (player) {
        const std::string title = player->Title();
        if (!title.empty()) {
            cStatus::MsgReplaying(this, title.c_str(), title.c_str(), true);
        }
    }
    isyslog("vaapivideo/mediaplayer: control launched");
}

cVaapiControl::~cVaapiControl() noexcept {
    HideReplayBar();
    cStatus::MsgReplaying(this, nullptr, nullptr, false);
    // Null the base alias BEFORE deleting the player (cf. cDvbPlayerControl::Stop in
    // vdr/dvbplayer.c). The unique_ptr owns/deletes; cControl does not.
    cControl::player = nullptr;
    player.reset();
    isyslog("vaapivideo/mediaplayer: control destroyed");
}

auto cVaapiControl::Hide() -> void { HideReplayBar(); }

[[nodiscard]] auto cVaapiControl::GetHeader() -> cString {
    return player ? cString{player->Title().c_str()} : cString{""};
}

[[nodiscard]] auto cVaapiControl::GetInfo() -> cOsdObject * {
    // VDR takes ownership and shows it through the OSD on the Info key.
    if (!player) {
        return nullptr;
    }
    const std::string body = player->InfoText();
    if (body.empty()) {
        return nullptr;
    }
    return new cMenuText(tr("File Info"), body.c_str());
}

auto cVaapiControl::ShowReplayBar() -> void {
    if (!barVisible) {
        displayReplay = Skins.Current()->DisplayReplay(false);
        barVisible = true;
    }
    barTimeout.Set(OSD_DEFAULT_TIMEOUT_S * 1000);
    RefreshReplayBar();
}

auto cVaapiControl::HideReplayBar() -> void {
    if (barVisible) {
        delete displayReplay;
        displayReplay = nullptr;
        barVisible = false;
    }
}

auto cVaapiControl::RefreshReplayBar() -> void {
    if (!barVisible || displayReplay == nullptr || !player) {
        return;
    }
    int current = 0;
    int total = 0;
    (void)player->GetIndex(current, total);

    displayReplay->SetTitle(player->Title().c_str());
    displayReplay->SetProgress(current, total);
    displayReplay->SetCurrent(FormatHms(current));
    displayReplay->SetTotal(FormatHms(total));
    displayReplay->SetMode(!player->IsPaused(), true, -1);
    displayReplay->Flush();
    lastBarRefresh.Set();
}

[[nodiscard]] auto cVaapiControl::HandleSeekKey(const char *label, int deltaMs) -> eOSState {
    dsyslog("vaapivideo/mediaplayer: key %s -- seek %+dms", label, deltaMs);
    player->Seek(deltaMs);
    ShowReplayBar();
    return osContinue;
}

[[nodiscard]] auto cVaapiControl::ProcessKey(eKeys Key) -> eOSState {
    // Key bindings (plugin spec):
    //   OK              toggle replay bar
    //   Play / Up       resume if paused
    //   Pause / Down    toggle pause
    //   Left  / Right   short seek (-/+ 10 s)
    //   Green / Yellow  long  seek (-/+ 60 s)
    //   Blue            cycle manual zoom (Off -> 1 -> .. -> N -> Off)
    //   Next            advance playlist
    //   Back / Stop     exit
    if (!player) {
        return osEnd;
    }
    if (player->IsFinished()) {
        // EOF / failed open: reopen the browser instead of dropping to live TV, like Stop below.
        RequestReturnToBrowser(player->CurrentUri());
        return osEnd;
    }

    // ProcessKey fires on every remote event, so it doubles as the bar's pacing tick.
    if (barVisible) {
        if (barTimeout.TimedOut()) {
            HideReplayBar();
        } else if (lastBarRefresh.Elapsed() >= OSD_REFRESH_INTERVAL_MS) {
            RefreshReplayBar();
        }
    }

    switch (Key & ~k_Repeat) {
        case kOk:
            dsyslog("vaapivideo/mediaplayer: key OK -- %s replay bar", barVisible ? "hide" : "show");
            if (barVisible) {
                HideReplayBar();
            } else {
                ShowReplayBar();
            }
            return osContinue;

        case kPlay:
        case kUp:
            dsyslog("vaapivideo/mediaplayer: key Play/Up -- %s",
                    player->IsPaused() ? "resume from pause" : "already playing (no-op)");
            if (player->IsPaused()) {
                player->SetPaused(false);
                RefreshReplayBar();
            }
            return osContinue;

        case kPause:
        case kDown:
            dsyslog("vaapivideo/mediaplayer: key Pause/Down -- toggle (was %s)",
                    player->IsPaused() ? "paused" : "playing");
            player->SetPaused(!player->IsPaused());
            ShowReplayBar();
            return osContinue;

        case kLeft:
            return HandleSeekKey("Left", -MEDIAPLAYER_SEEK_SHORT_MS);
        case kRight:
            return HandleSeekKey("Right", +MEDIAPLAYER_SEEK_SHORT_MS);
        case kGreen:
            return HandleSeekKey("Green", -MEDIAPLAYER_SEEK_LONG_MS);
        case kYellow:
            return HandleSeekKey("Yellow", +MEDIAPLAYER_SEEK_LONG_MS);

        case kNext:
            dsyslog("vaapivideo/mediaplayer: key Next -- advance playlist");
            player->Next();
            ShowReplayBar();
            return osContinue;

        case kBlue: {
            // Cycle the manual zoom (Off -> 1 -> .. -> N -> Off) and flash the new stop on the OSD.
            auto *vaapiDev = FindPrimaryVaapiDevice();
            if (vaapiDev == nullptr || !vaapiDev->IsReady()) {
                Skins.QueueMessage(mtWarning, tr("VAAPI device not ready"));
            } else {
                const int stop = vaapiDev->CycleZoom();
                dsyslog("vaapivideo/mediaplayer: key Blue -- zoom cycle to stop %d", stop);
                Skins.QueueMessage(mtInfo, vaapiDev->ZoomStatusLabel().c_str());
            }
            return osContinue;
        }

        case kBack:
        case kStop:
            // Reopen the browser at the played file instead of live TV; URLs / non-selectable
            // paths fall back to the media-dir in cVaapiFileBrowser.
            dsyslog("vaapivideo/mediaplayer: key Back/Stop -- return to file browser");
            RequestReturnToBrowser(player->CurrentUri());
            return osEnd;

        default:
            return osContinue;
    }
}

// ============================================================================
// === cVaapiFileBrowser ===
// ============================================================================

cVaapiFileBrowser::cVaapiFileBrowser(std::string startDir, const std::string &selectPath) : cOsdMenu("") {
    if (startDir.empty()) {
        startDir = "/";
    }
    // Jump back to a specific file (Stop from replay): open its parent dir, cursor on it. A URL or a
    // non-selectable path isn't browseable -- fall back to the start folder (svdrpsend / URL case).
    if (!selectPath.empty() && !HasUrlScheme(selectPath)) {
        std::error_code ec;
        const std::string parent = Dirname(selectPath);
        if (std::filesystem::is_regular_file(selectPath, ec) && !ec && std::filesystem::is_directory(parent, ec) &&
            !ec) {
            LoadDirectory(parent);
            if (SelectEntryByName(Basename(selectPath))) {
                return;
            }
        }
    }
    LoadDirectory(startDir);
}

auto cVaapiFileBrowser::SelectEntryByName(std::string_view name) -> bool {
    // entries[] index == menu index (built in Add() order). Re-Display() to repaint the highlight.
    for (size_t i = 0; i < entries.size(); ++i) {
        if (entries.at(i).name == name) {
            SetCurrent(Get(static_cast<int>(i)));
            Display();
            return true;
        }
    }
    return false;
}

auto cVaapiFileBrowser::LoadDirectory(const std::string &dir) -> void {
    // Layout order in the menu (matches user expectation from typical file managers):
    //   [..]            parent navigation (unless we're at "/")
    //   [<dir>]         subdirectories, alphabetical
    //   # <playlist>    .m3u/.m3u8 files, alphabetical
    //   <file>          media files, alphabetical
    // Falling back to a parent-only menu on opendir failure lets the user navigate up
    // out of an unreadable directory instead of being stuck.
    entries.clear();
    Clear();

    std::error_code ec;
    const auto canonical = std::filesystem::canonical(dir, ec);
    currentDir = ec ? dir : canonical.string();

    SetTitle(cString::sprintf("%s: %s", tr("Mediaplayer"), currentDir.c_str()));

    if (currentDir != "/") {
        entries.push_back({.kind = EntryKind::Parent, .name = ".."});
    }

    DIR *d = ::opendir(currentDir.c_str());
    if (d == nullptr) {
        esyslog("vaapivideo/mediaplayer: opendir(%s): %s", currentDir.c_str(), std::strerror(errno));
    } else {
        std::vector<BrowserEntry> dirs;
        std::vector<BrowserEntry> files;
        std::vector<BrowserEntry> playlists;

        while (true) {
            errno = 0;
            dirent *de = ::readdir(d);
            if (de == nullptr) {
                break;
            }
            std::string name = de->d_name;
            if (name.empty() || name.front() == '.') {
                continue;
            }
            std::string full = currentDir;
            if (full.back() != '/') {
                full += '/';
            }
            full += name;
            std::error_code statEc;
            // Named fileStatus (not `status`) because cOsdBase has an inherited `status` member.
            const auto fileStatus = std::filesystem::status(full, statEc);
            if (statEc) {
                continue;
            }
            if (fileStatus.type() == std::filesystem::file_type::directory) {
                dirs.push_back({.kind = EntryKind::Directory, .name = std::move(name)});
            } else if (fileStatus.type() == std::filesystem::file_type::regular) {
                // file_size() is a separate query that can fail independently of status() (e.g. a
                // race with deletion); fall back to 0 so the entry still lists, just without a size.
                std::error_code sizeEc;
                const std::uintmax_t bytes = std::filesystem::file_size(full, sizeEc);
                const std::uintmax_t size = sizeEc ? 0 : bytes;
                if (IsPlaylistUri(name)) {
                    playlists.push_back({.kind = EntryKind::Playlist, .name = std::move(name), .size = size});
                } else if (IsMediaUri(name)) {
                    files.push_back({.kind = EntryKind::File, .name = std::move(name), .size = size});
                }
            }
        }
        // readdir returns nullptr both for end-of-directory (errno unchanged) and on error
        // (errno != 0). The loop body resets errno before each readdir; check it here.
        if (errno != 0) {
            esyslog("vaapivideo/mediaplayer: readdir(%s): %s", currentDir.c_str(), std::strerror(errno));
        }
        if (::closedir(d) != 0) {
            esyslog("vaapivideo/mediaplayer: closedir(%s): %s", currentDir.c_str(), std::strerror(errno));
        }

        // std::sort (not std::ranges::sort) because the latter trips some IDE/IntelliSense
        // parsers on libstdc++'s sortable-concept resolution. clang-tidy modernize-use-ranges
        // would prefer the ranges form; suppressed here for that reason.
        const auto byName = [](const BrowserEntry &a, const BrowserEntry &b) -> bool { return a.name < b.name; };
        std::sort(dirs.begin(), dirs.end(), byName);           // NOLINT(modernize-use-ranges)
        std::sort(playlists.begin(), playlists.end(), byName); // NOLINT(modernize-use-ranges)
        std::sort(files.begin(), files.end(), byName);         // NOLINT(modernize-use-ranges)

        for (auto &e : dirs) {
            entries.push_back(std::move(e));
        }
        for (auto &e : playlists) {
            entries.push_back(std::move(e));
        }
        for (auto &e : files) {
            entries.push_back(std::move(e));
        }
    }

    for (const auto &entry : entries) {
        cString label;
        switch (entry.kind) {
            case EntryKind::Parent:
                label = cString::sprintf("[..]");
                break;
            case EntryKind::Directory:
                label = cString::sprintf("[%s]", entry.name.c_str());
                break;
            case EntryKind::Playlist:
                label = cString::sprintf("# %s", entry.name.c_str());
                break;
            case EntryKind::File:
                // Size appended inline (not a \t column): without SetCols the skin draws only the
                // first tab-column, so a tabbed size would be invisible. Inline always renders.
                label = cString::sprintf("%s  (%s)", entry.name.c_str(), *FormatSizeMb(entry.size));
                break;
        }
        Add(new cOsdItem(label, osUnknown));
    }
    Display();
}

[[nodiscard]] auto cVaapiFileBrowser::SelectedEntry() const -> const BrowserEntry * {
    const int idx = Current();
    if (idx < 0 || static_cast<size_t>(idx) >= entries.size()) {
        return nullptr;
    }
    return &entries.at(static_cast<size_t>(idx));
}

[[nodiscard]] auto cVaapiFileBrowser::BuildFullPath(const BrowserEntry &entry) const -> std::string {
    if (entry.kind == EntryKind::Parent) {
        if (currentDir == "/" || currentDir.empty()) {
            return "/";
        }
        return Dirname(currentDir);
    }
    return currentDir + (currentDir.back() == '/' ? "" : "/") + entry.name;
}

[[nodiscard]] auto cVaapiFileBrowser::ProcessKey(eKeys Key) -> eOSState {
    // kBack must be intercepted BEFORE cOsdMenu::ProcessKey(): the base menu returns osBack
    // for kBack (osdbase.c), which would close the whole browser instead of letting us walk
    // up to the parent directory. Only fall back to osBack (pop the menu) when already at root.
    if ((Key & ~k_Repeat) == kBack) {
        if (currentDir != "/" && !currentDir.empty()) {
            LoadDirectory(Dirname(currentDir));
            return osContinue;
        }
        return osBack;
    }

    // Stop leaves the browser outright (osEnd -> live TV) from any depth, vs. walking kBack to the
    // root. Intercepted before the base, which swallows kStop.
    if ((Key & ~k_Repeat) == kStop) {
        return osEnd;
    }

    // Let the base menu handle navigation keys (Up/Down/PageUp/PageDown) first. We only
    // see the key here when it returned osUnknown, i.e. nothing the menu knew how to do.
    const eOSState state = cOsdMenu::ProcessKey(Key);
    if (state != osUnknown) {
        return state;
    }

    switch (Key & ~k_Repeat) {
        case kOk: {
            const auto *entry = SelectedEntry();
            if (entry == nullptr) {
                return osContinue;
            }
            const std::string fullPath = BuildFullPath(*entry);
            switch (entry->kind) {
                case EntryKind::Parent:
                case EntryKind::Directory:
                    LoadDirectory(fullPath);
                    return osContinue;
                case EntryKind::Playlist: {
                    auto playlist = ParseM3U(fullPath);
                    if (playlist.empty()) {
                        Skins.Message(mtError, tr("Empty or unreadable playlist"));
                        return osContinue;
                    }
                    if (!StartPlayback(std::move(playlist))) {
                        Skins.Message(mtError, tr("Cannot start playback"));
                        return osContinue;
                    }
                    return osEnd;
                }
                case EntryKind::File:
                    if (!StartPlayback({PlaylistEntry{.uri = fullPath, .title = entry->name}})) {
                        Skins.Message(mtError, tr("Cannot start playback"));
                        return osContinue;
                    }
                    return osEnd;
            }
            return osContinue;
        }
        default:
            return osContinue;
    }
}
