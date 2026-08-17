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
 * the clock halts with the demux loop. Trick play (fast/slow, forward/backward) mirrors
 * cDvbPlayer's state machine and reuses the decoder's trick-speed machinery unchanged:
 * forward tricks keep the linear demux (the decoder's FF filter keeps keyframes only),
 * reverse steps keyframes backward via av_seek_frame -- there is no VDR index file.
 *
 * Invariants:
 *   1. sourceMutex MUST be released before calling Open/CloseCurrentEntry: they relock it
 *      internally and would otherwise hold it across blocking container I/O (invariant 5).
 *      Action() defers playlist advancement past its lock scope for the same reason.
 *      (cMutex tolerates same-thread relock -- this is a hold-time rule, not a deadlock rule.)
 *   2. cVaapiControl owns the player (unique_ptr); cControl holds a borrowed alias.
 *      The dtor nulls the base before resetting (see ~cVaapiControl).
 *   3. cVaapiMediaSource emits a zero-based 90 kHz timeline so GetIndex/Seek math is
 *      stable across files with non-zero container start_time.
 *   4. Network I/O is interruptible via InterruptOnStop on cVaapiPlayer::stopping.
 *   5. sourceMutex may be held across blocking demux I/O (ReadPacket/Seek), but never across the
 *      container open (OpenCurrentEntry opens unlocked, publishes locked). Main-thread hot paths
 *      (GetIndex, FramesPerSecond, MakeBookmark) are lock-free via cached atomics -- they must
 *      never take sourceMutex, or a stalled network read freezes the VDR main loop.
 */

#include "mediaplayer.h"
#include "audio.h"
#include "common.h"
#include "config.h"
#include "device.h"
#include "stream.h"
#include "subtitle.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <limits>
#include <memory>
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
#include <libavcodec/codec.h>
#include <libavcodec/codec_id.h>
#include <libavcodec/codec_par.h>
#include <libavcodec/defs.h>
#include <libavcodec/packet.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/dict.h>
#include <libavutil/dovi_meta.h>
#include <libavutil/error.h>
#include <libavutil/mastering_display_metadata.h>
#include <libavutil/mathematics.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
#include <libavutil/rational.h>
}
#pragma GCC diagnostic pop

// VDR
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <vdr/config.h>
#include <vdr/device.h>
#include <vdr/i18n.h>
#include <vdr/keys.h>
#include <vdr/menu.h>
#include <vdr/osdbase.h>
#include <vdr/player.h>
#include <vdr/plugin.h>
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

constexpr int DEMUX_IDLE_SLEEP_MS = 5;       ///< Back-off when ReadPacket reports EAGAIN or no work is available.
constexpr int DEMUX_PAUSE_WAKEUP_MS = 100;   ///< Periodic re-check while paused; safety net against a missed broadcast.
constexpr int OSD_DEFAULT_TIMEOUT_S = 4;     ///< Auto-hide delay after a key event; matches VDR replay-control feel.
constexpr int OSD_REFRESH_INTERVAL_MS = 500; ///< Replay-bar update cadence; ~2 Hz feels live without flicker.

/// Demux thread back-off when the device queues are full or input stalls.
constexpr int MEDIAPLAYER_BACKPRESSURE_SLEEP_MS = 5;

/// Real-time pacing brake: max AUDIO lookahead (90 kHz ticks) of the latest pushed audio PTS over the
/// audio master clock before the demux throttles. libavformat reads files far faster than wall-clock, so
/// without it the decoder queue + jitterBuf overrun their caps. Keyed off the audio tail only
/// (latestAudioPts90k); the resulting video depth (audio_tail + per-file mux offset) is bounded instead by
/// DECODER_RESERVE_HARD_CAP, where the excess lead waits COMPRESSED in the packetQueue. 1.5 s (not 1 s) so
/// the budget still reaches the ~1.3 s reserve cap when interlaced content deinterlaces to 50 fps.
/// MEDIAPLAYER_JITTERBUF_BACKPRESSURE_FRAMES (the pre-anchor video-depth gate) lives in device.cpp, its
/// only user, derived from DECODER_RESERVE_HARD_CAP so it stays coupled to the buffer it protects.
constexpr int64_t MEDIAPLAYER_MAX_LOOKAHEAD_90K = 135000;

/// Default seek deltas applied by the key bindings (milliseconds).
constexpr int MEDIAPLAYER_SEEK_SHORT_MS = 10000;
constexpr int MEDIAPLAYER_SEEK_LONG_MS = 60000;

/// Trick-play notch table, verbatim from vdr/dvbplayer.c (Speeds[]): positive entries are fast
/// divisors, negative slow multipliers, the 0 sentinels saturate silently. Kept verbatim so the
/// derived device repeat counts (fast 6/3/1, slow fwd 2/4/8, slow rev 24/48/96->63) are exactly
/// the values cVaapiDecoder::SetTrickSpeed()'s mapping is tuned for.
constexpr std::array<int, 9> MEDIAPLAYER_TRICK_SPEEDS{0, -2, -4, -8, 1, 2, 4, 12, 0};
static_assert(MEDIAPLAYER_TRICK_SPEEDS.at(MEDIAPLAYER_TRICK_NORMAL_IDX) == 1);
constexpr int MEDIAPLAYER_TRICK_STEPS_MAX = 3;   ///< Notches from normal to the extreme in either direction.
constexpr int MEDIAPLAYER_TRICK_SPEED_MULT = 12; ///< dvbplayer SPEED_MULT: repeat-count numerator (except slow-fwd).
constexpr int MEDIAPLAYER_TRICK_DEVICE_SPEED_MAX = 63; ///< dvbplayer MAX_VIDEO_SLOWMOTION clamp on the repeat count.

/// Reverse stepping: seek target offset below the last shown keyframe (1 ms -- av_seek_frame with
/// AVSEEK_FLAG_BACKWARD then lands on the preceding keyframe), and the extra back-step applied when
/// a container with coarse seek granularity lands on the already-shown keyframe again (0.5 s).
constexpr int64_t MEDIAPLAYER_REVERSE_EPSILON_90K = 90;
constexpr int64_t MEDIAPLAYER_REVERSE_RETRY_STEP_90K = 45000;

/// End-of-stream tail drain (cVaapiPlayer::DrainTailAtEof): at EOF the decode queue (~4 s) and decoded
/// reserve (~1.3 s) still hold unseen frames, so immediate teardown cuts playback seconds short (worst on
/// video-only clips, where no audio clock throttles the demuxer). Flush that tail at real-time pace first.
///   - TIMEOUT_MS: backstop so a wedged pipeline can't hang shutdown (covers the ~5 s queue+reserve tail).
///   - STALL_MS: bail when depth stops shrinking; must exceed one frame interval.
constexpr int MEDIAPLAYER_EOF_DRAIN_TIMEOUT_MS = 20000;
constexpr int MEDIAPLAYER_EOF_DRAIN_STALL_MS = 1500;

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
    // time_base guard as in Rebase90k: damaged container metadata with den == 0 would make
    // av_rescale_q divide by zero (SIGFPE).
    if (stream == nullptr || stream->start_time == AV_NOPTS_VALUE || stream->time_base.num <= 0 ||
        stream->time_base.den <= 0) {
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

/// Channel-layout label for a track description; empty for uncommon counts.
[[nodiscard]] auto AudioLayoutLabel(int channels) -> std::string_view {
    switch (channels) {
        case 1:
            return "Mono";
        case 2:
            return "Stereo";
        case 6:
            return "5.1";
        case 8:
            return "7.1";
        default:
            return {};
    }
}

/// cDisplayTracks label, e.g. "AC-3 5.1 (eng)". Kept short: tTrackId.description is char[32].
[[nodiscard]] auto AudioTrackDescription(const cVaapiMediaSource::AudioTrackDesc &track) -> std::string {
    std::string desc = avcodec_get_name(track.info.codecId);
    if (const std::string_view layout = AudioLayoutLabel(track.srcChannels); !layout.empty()) {
        desc += std::format(" {}", layout);
    }
    if (!track.language.empty()) {
        desc += std::format(" ({})", track.language);
    }
    return desc;
}

/// Short codec label for the UI; "SRT"/"DVB" read better than FFmpeg's "subrip"/"dvb_subtitle".
/// Other codecs use their name.
[[nodiscard]] auto SubtitleCodecName(AVCodecID codecId) noexcept -> const char * {
    if (codecId == AV_CODEC_ID_SUBRIP) {
        return "SRT";
    }
    if (codecId == AV_CODEC_ID_DVB_SUBTITLE) {
        return "DVB";
    }
    return avcodec_get_name(codecId);
}

/// cDisplaySubtitleTracks label, e.g. "SRT (ger)". Kept short: tTrackId.description is char[32].
[[nodiscard]] auto SubtitleTrackDescription(const cVaapiMediaSource::SubtitleTrackDesc &track) -> std::string {
    std::string desc = SubtitleCodecName(track.codecId);
    if (!track.language.empty()) {
        desc += std::format(" ({})", track.language);
    }
    return desc;
}

/// Subtitle codecs the converter can render: text codecs plus DVB bitmap. Other bitmap codecs
/// (hdmv_pgs, xsub) are excluded -- the render path is generic, so they are easy but untested follow-ups.
[[nodiscard]] auto IsSupportedSubtitle(AVCodecID codec) noexcept -> bool {
    const bool knownCodec = codec == AV_CODEC_ID_SUBRIP || codec == AV_CODEC_ID_TEXT || codec == AV_CODEC_ID_MOV_TEXT ||
                            codec == AV_CODEC_ID_ASS || codec == AV_CODEC_ID_SSA || codec == AV_CODEC_ID_DVB_SUBTITLE;
    // Don't advertise a track Open() would reject: this FFmpeg build may lack the decoder.
    return knownCodec && avcodec_find_decoder(codec) != nullptr;
}

/// Renderable-subtitle-stream predicate; one definition so the count and populate passes can't drift.
[[nodiscard]] auto IsSupportedSubtitleStream(const AVStream *stream) noexcept -> bool {
    if (stream == nullptr || stream->codecpar == nullptr) {
        return false;
    }
    const AVCodecParameters *p = stream->codecpar;
    return p->codec_type == AVMEDIA_TYPE_SUBTITLE && IsSupportedSubtitle(p->codec_id);
}

/// Match cDevice::EnsureAudioTrack's pick (Setup.AudioLanguages, else track 0) so opening
/// cDisplayTracks never silently re-switches the track. -1 when the source has no audio.
[[nodiscard]] auto ChoosePreferredAudioTrack(const cVaapiMediaSource &source) -> int {
    const auto &tracks = source.AudioTracks();
    if (tracks.empty()) {
        return -1;
    }
    int preferred = -1;
    int languagePreference = -1;
    for (size_t i = 0; i < tracks.size(); ++i) {
        int pos = 0;
        if (I18nIsPreferredLanguage(Setup.AudioLanguages, tracks.at(i).language.c_str(), languagePreference, &pos)) {
            preferred = static_cast<int>(i);
        }
    }
    return preferred >= 0 ? preferred : 0;
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
constexpr std::array<std::string_view, 7> MEDIA_EXTENSIONS{{".mp4", ".mkv", ".avi", ".mov", ".ts", ".m4v", ".webm"}};

constexpr std::array<std::string_view, 2> PLAYLIST_EXTENSIONS{{".m3u", ".m3u8"}};

/// Cap on the in-memory playlist read. Real .m3u files are tiny; the browser filters only by
/// extension, so a mislabeled huge file must not balloon VDR's memory.
constexpr size_t MAX_PLAYLIST_BYTES = 8U * 1024U * 1024U;

// URI schemes we hand straight to libavformat rather than resolving as filesystem paths.
// HLS .m3u8 over http(s) deliberately goes here rather than through our local m3u parser.
// file:// is included so an M3U line like "file:///media/movie.mkv" is taken verbatim
// instead of being mangled into "<playlist-dir>/file:///media/movie.mkv".
constexpr std::array<std::string_view, 4> URL_SCHEMES{{"file://", "http://", "https://", "ftp://"}};

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
    return std::ranges::any_of(URL_SCHEMES, [path](std::string_view scheme) noexcept -> bool {
        return path.size() >= scheme.size() && IEquals(path.substr(0, scheme.size()), scheme);
    });
}

// One canonical spelling for the bookmark identity (matches the browser's canonicalized currentDir).
// URLs and unresolvable paths pass through, so an SVDRP "PLAY ../a.mkv" still plays.
[[nodiscard]] auto NormalizeBookmarkUri(std::string uri) -> std::string {
    if (uri.empty() || HasUrlScheme(uri)) {
        return uri;
    }
    std::error_code ec;
    const auto canonical = std::filesystem::canonical(uri, ec);
    return ec ? uri : canonical.string();
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
    // Strip trailing slashes first ("/media/" -> "/media", "foo/" -> "foo") so the split below
    // matches POSIX dirname(); a lone "/" stays intact.
    while (path.size() > 1 && path.back() == '/') {
        path.remove_suffix(1);
    }
    if (const auto pos = path.find_last_of('/'); pos != std::string_view::npos) {
        // A slash only at position 0 ("/media") must yield "/" like POSIX dirname();
        // substr(0, 0) would yield "" and break walking up to the filesystem root.
        return pos == 0 ? std::string{"/"} : std::string{path.substr(0, pos)};
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

/// Format a byte count as a whole number of MiB (rounded to nearest), labeled "MB".
/// Used by the file browser to annotate each media file. A non-empty file never
/// rounds to "0 MB": sub-half-MiB sizes are floored up to "1 MB" so the column
/// always reflects that there is content.
[[nodiscard]] auto FormatSizeMb(std::uintmax_t bytes) -> cString {
    constexpr std::uintmax_t kBytesPerMiB = 1024U * 1024U;
    std::uintmax_t mib = (bytes + (kBytesPerMiB / 2)) / kBytesPerMiB;
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
///      sends the wrong row of VIDEO_BACKEND_TABLE to the decoder. Defaulting to k8 there
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

    // Read the whole file up front and split on '\n' below: a fixed fgets() line buffer
    // would silently split an over-long URI (e.g. a URL with a long query string) into two
    // bogus entries. Playlists are small, so buffering the file is cheap.
    std::string content;
    std::array<char, 4096> chunk{};
    bool readOk = true;
    // The common exit is the short-read break: fread coming up short means EOF or error.
    // The feof/ferror gate exists so fread is never called again once EOF or an error is
    // already flagged (a failed read leaves the file position indeterminate; and some
    // implementations flag EOF on the full final read of an exact-multiple-sized file).
    while (std::feof(fp) == 0 && std::ferror(fp) == 0) {
        const size_t n = std::fread(chunk.data(), 1, chunk.size(), fp);
        if (content.size() + n > MAX_PLAYLIST_BYTES) {
            esyslog("vaapivideo/mediaplayer: playlist %s larger than %zu bytes -- rejected", canonical.c_str(),
                    MAX_PLAYLIST_BYTES);
            readOk = false;
            break;
        }
        content.append(chunk.data(), n);
        if (n < chunk.size()) {
            break;
        }
    }
    // Distinguish clean EOF from a truncated read (disk error mid-playlist) -- parsing a
    // partial file would look like a successful short playlist.
    if (std::ferror(fp) != 0) {
        esyslog("vaapivideo/mediaplayer: read error in playlist %s: %s", canonical.c_str(), std::strerror(errno));
        readOk = false;
    }
    if (std::fclose(fp) != 0) {
        esyslog("vaapivideo/mediaplayer: fclose(%s): %s", canonical.c_str(), std::strerror(errno));
        readOk = false;
    }
    if (!readOk) {
        return result;
    }
    // Binary data misnamed .m3u: a NUL can't occur in a valid playlist, and letting one through
    // would embed NULs in URIs/titles that downstream C-string consumers silently truncate.
    if (content.find('\0') != std::string::npos) {
        esyslog("vaapivideo/mediaplayer: NUL byte in playlist %s -- rejected", canonical.c_str());
        return result;
    }

    // M3U grammar we accept: any line not starting with '#' is a URI; "#EXTINF:duration,title"
    // optionally precedes a URI and supplies its display title. All other '#'-lines are ignored.
    std::string pendingTitle;
    std::string_view rest{content};
    while (!rest.empty()) {
        const auto newline = rest.find('\n');
        const auto line = Trim(rest.substr(0, newline));
        rest = (newline == std::string_view::npos) ? std::string_view{} : rest.substr(newline + 1);
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

    isyslog("vaapivideo/mediaplayer: playlist %s -- %zu entries", canonical.c_str(), result.size());
    return result;
}

auto IsMediaUri(std::string_view path) noexcept -> bool {
    if (HasUrlScheme(path)) {
        return true;
    }
    return std::ranges::any_of(MEDIA_EXTENSIONS,
                               [path](std::string_view ext) noexcept -> bool { return HasExtension(path, ext); });
}

auto IsPlaylistUri(std::string_view path) noexcept -> bool {
    // HLS manifests (.m3u8 served over http(s)) belong in libavformat, not our local m3u parser.
    if (HasUrlScheme(path)) {
        return false;
    }
    return std::ranges::any_of(PLAYLIST_EXTENSIONS,
                               [path](std::string_view ext) noexcept -> bool { return HasExtension(path, ext); });
}

// ============================================================================
// === BOOKMARK PERSISTENCE ===
// ============================================================================
namespace {
// Guards vaapiConfig.bookmark + BookmarkDirty(): SVDRP PLAY tears a control down on the SVDRP thread,
// racing the main thread. Function-local static: lazy init stays off the throwing-static-init path.
[[nodiscard]] auto BookmarkMutex() -> cMutex & {
    static cMutex mutex;
    return mutex;
}

// Set while vaapiConfig.bookmark differs from setup.conf on disk.
[[nodiscard]] auto BookmarkDirty() -> bool & {
    static bool dirty = false;
    return dirty;
}

// Stage, then flush inline on the main thread (survives the emergency-exit path that skips VDR's own
// Setup.Save()) or defer to Housekeeping() for off-thread SVDRP teardown. Empty/unchanged URI: no-op.
auto PersistBookmark(const MediaBookmark &bm) -> void {
    if (bm.uri.empty()) [[unlikely]] {
        return;
    }
    bool onMainThread = false;
    {
        const cMutexLock lock(&BookmarkMutex());
        if (vaapiConfig.bookmark.uri == bm.uri && vaapiConfig.bookmark.positionMs == bm.positionMs &&
            !BookmarkDirty()) {
            return;
        }
        vaapiConfig.bookmark = bm;
        BookmarkDirty() = true;
        onMainThread = cThread::IsMainThread();
    }
    if (onMainThread) {
        FlushPendingBookmarkSave();
    }
}
} // namespace

auto LoadBookmark() -> MediaBookmark {
    const cMutexLock lock(&BookmarkMutex());
    return vaapiConfig.bookmark;
}

auto FlushPendingBookmarkSave() -> void {
    // Main thread only: SetupStore/Setup.Save mutate VDR's global setup store.
    MediaBookmark bm;
    {
        const cMutexLock lock(&BookmarkMutex());
        if (!BookmarkDirty()) {
            return;
        }
        bm = vaapiConfig.bookmark;
    }
    // SetupStore writes the keys; only Setup.Save() hits disk. On failure the dirty flag stays set to retry.
    auto *plugin = cPluginManager::GetPlugin(PLUGIN_NAME);
    if (plugin == nullptr) [[unlikely]] {
        esyslog("vaapivideo/mediaplayer: plugin instance missing -- cannot save bookmark");
        return;
    }
    plugin->SetupStore("BookmarkUri", bm.uri.c_str());
    plugin->SetupStore("BookmarkPositionMs", bm.positionMs);
    if (!Setup.Save()) [[unlikely]] {
        esyslog("vaapivideo/mediaplayer: cannot save bookmark to setup.conf");
        return;
    }
    {
        const cMutexLock lock(&BookmarkMutex());
        if (vaapiConfig.bookmark.uri == bm.uri && vaapiConfig.bookmark.positionMs == bm.positionMs) {
            BookmarkDirty() = false; // keep dirty if a newer bookmark was staged mid-save
        }
    }
    dsyslog("vaapivideo/mediaplayer: bookmark saved -- %s @ %dms", bm.uri.c_str(), bm.positionMs);
}

auto StartPlayback(PlaylistEntry origin) -> StartPlaybackResult {
    if (origin.uri.empty()) [[unlikely]] {
        esyslog("vaapivideo/mediaplayer: StartPlayback called with empty URI");
        return StartPlaybackResult::EmptyPlaylist;
    }
    origin.uri = NormalizeBookmarkUri(std::move(origin.uri)); // spelling-independent bookmark identity
    // Reject before Launch: launching closes the browser (osEnd) and only then fails in Activate(true),
    // stranding the user on the old channel with no error. IsReady() here keeps the menu open to report it.
    auto *vaapiDev = FindPrimaryVaapiDevice();
    if (vaapiDev == nullptr || !vaapiDev->IsReady()) [[unlikely]] {
        esyslog("vaapivideo/mediaplayer: no ready primary vaapivideo device -- playback rejected");
        return StartPlaybackResult::DeviceNotReady;
    }

    // Expand playlists here, not at the call sites, so the origin .m3u path survives into the bookmark.
    std::vector<PlaylistEntry> entries;
    if (IsPlaylistUri(origin.uri)) {
        entries = ParseM3U(origin.uri);
        if (entries.empty()) [[unlikely]] {
            return StartPlaybackResult::EmptyPlaylist;
        }
    } else {
        entries.push_back(origin);
    }

    // Resume exactly where we stopped when the origin matches the bookmark; playlists/streams save
    // position 0, so they stay at resumeMs 0 by construction.
    int resumeMs = 0;
    if (const MediaBookmark bm = LoadBookmark(); origin.uri == NormalizeBookmarkUri(bm.uri)) {
        resumeMs = bm.positionMs;
    }

    // cControl::Launch takes ownership and destroys via cControl::Shutdown / next Launch.
    cControl::Launch(new cVaapiControl(std::move(origin.uri), std::move(entries), resumeMs));
    return StartPlaybackResult::Started;
}

// One-shot "reopen the browser, not live TV" flag. Set on Stop/EOF, consumed by MainMenuAction() --
// both main-thread, no lock. The browser reads the bookmark itself to place its cursor.
namespace {
[[nodiscard]] auto PendingBrowserReopen() -> bool & {
    static bool pending = false;
    return pending;
}
} // namespace

auto RequestBrowserReopen() -> void {
    PendingBrowserReopen() = true;
    // CallPlugin queues a k_Plugin key for MainMenuAction(). Fails only if another plugin call is
    // pending; drop our flag then so an unrelated menu open won't show the browser.
    if (!cRemote::CallPlugin(PLUGIN_NAME)) [[unlikely]] {
        esyslog("vaapivideo/mediaplayer: CallPlugin busy -- cannot reopen file browser");
        PendingBrowserReopen() = false;
    }
}

auto TakeBrowserReopen() noexcept -> bool { return std::exchange(PendingBrowserReopen(), false); }

// ============================================================================
// === cVaapiMediaSource ===
// ============================================================================

cVaapiMediaSource::~cVaapiMediaSource() noexcept { Close(); }

[[nodiscard]] auto cVaapiMediaSource::IoInterrupted() const noexcept -> bool {
    const bool stop = stopFlag != nullptr && stopFlag->load(std::memory_order_acquire);
    const bool command = interruptFlag != nullptr && interruptFlag->load(std::memory_order_acquire);
    return stop || command;
}

// Reset to the pre-Open state; idempotent (dtor and Open() both call it). Every field Open()/
// PopulateStreamInfo() set must be cleared here, or it leaks into the next entry.
auto cVaapiMediaSource::Close() noexcept -> void {
    formatCtx.reset();
    videoStreamIndex = -1;
    audioStreamIndex = -1;
    videoExtradataStorage.clear();
    audioTracks.clear();
    currentAudioTrack = -1;
    subtitleTracks.clear();
    currentSubtitleTrack = -1;
    subtitleStreamIndex = -1;
    subtitleTimeBase = AVRational{.num = 1, .den = 90000};
    videoInfo = VideoStreamInfo{};
    audioInfo = AudioStreamInfo{};
    ptsOrigin90k = AV_NOPTS_VALUE;
    discardAudioBefore90k = AV_NOPTS_VALUE;
    discardVideoBefore90k = AV_NOPTS_VALUE;
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
    // Only called from Open() with a live context; the guard documents that precondition
    // (and keeps GCC's null-dereference analysis quiet about formatCtx->streams below).
    if (!formatCtx) [[unlikely]] {
        return;
    }
    videoInfo = VideoStreamInfo{};
    audioInfo = AudioStreamInfo{};
    videoExtradataStorage.clear();
    audioTracks.clear();
    currentAudioTrack = -1;
    subtitleTracks.clear();
    currentSubtitleTrack = -1;
    subtitleStreamIndex = -1;
    subtitleTimeBase = AVRational{.num = 1, .den = 90000};

    // Pick ptsOrigin90k = MAX of tracked-stream start_times so both streams begin at
    // rebased PTS 0 together. Files where audio (or any other stream) leads video by
    // hundreds of ms in the container would otherwise present as: audio plays from t=0,
    // video frames at rebased PTS=+lead sit "too far in future" against the audio clock
    // and get continuously dropped -- the user sees a still image with progressing
    // audio. Dropping the small leading-audio prefix (handled by the < 0 guard in
    // ReadPacket) starts both streams together, no audio-only intro.
    // Falls back to formatCtx->start_time if neither stream advertises start_time, then
    // to the first packet seen by ReadPacket's lazy-set path if even that is unknown.
    // One load + explicit null guard for the video stream: StreamStart90k tolerates nullptr,
    // and reusing the guarded pointer below keeps the deref provably safe (-Wnull-dereference).
    const AVStream *videoStream = videoStreamIndex >= 0 ? formatCtx->streams[videoStreamIndex] : nullptr;
    const int64_t videoStart = StreamStart90k(videoStream);
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
    if (videoStream != nullptr) {
        const AVStream *stream = videoStream;
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
        // field_order is the container's interlace verdict (the mediaplayer analog of MPEG-2
        // progressive_sequence); UNKNOWN leaves the hint off so the per-frame flag decides.
        videoInfo.hasStreamInterlaceInfo = p->field_order != AV_FIELD_UNKNOWN;
        videoInfo.streamInterlaced = p->field_order != AV_FIELD_UNKNOWN && p->field_order != AV_FIELD_PROGRESSIVE;
        // Bit depth drives SelectVideoBackendCap (HW-vs-SW decode decision). Profile alone is
        // ambiguous: HEVC REXT can be 8-16 bit, AV1 Main can be 8 or 10, etc. Inspect the
        // pixel format descriptor for the authoritative answer.
        videoInfo.bitDepth = VideoFormatBitDepth(p);
        // The PES path waits for an in-band SPS before opening the codec; the mediaplayer
        // path always has the container's extradata, so the decoder can open immediately.
        videoInfo.hasSps = true;
        CopyExtradata(p, videoExtradataStorage, videoInfo.extradata, videoInfo.extradataSize);
        // VP9/AV1 carry HDR static metadata in the container, not the bitstream -- copy it so the
        // HDR_OUTPUT_METADATA blob gets real mastering luminance instead of zeros.
        if (const AVPacketSideData *sd = av_packet_side_data_get(p->coded_side_data, p->nb_coded_side_data,
                                                                 AV_PKT_DATA_MASTERING_DISPLAY_METADATA);
            sd != nullptr && sd->size >= sizeof(AVMasteringDisplayMetadata)) {
            videoInfo.hasMasteringDisplay = true;
            std::memcpy(&videoInfo.masteringDisplay, sd->data, sizeof(videoInfo.masteringDisplay));
        }
        if (const AVPacketSideData *sd =
                av_packet_side_data_get(p->coded_side_data, p->nb_coded_side_data, AV_PKT_DATA_CONTENT_LIGHT_LEVEL);
            sd != nullptr && sd->size >= sizeof(AVContentLightMetadata)) {
            videoInfo.hasContentLight = true;
            std::memcpy(&videoInfo.contentLight, sd->data, sizeof(videoInfo.contentLight));
        }
        // Dolby Vision: VAAPI has no DV path, so only the HEVC base layer decodes (RPU + enhancement
        // layer dropped). A BL compatibility id of 1 (HDR10) / 4 (HLG) renders correctly; 0/2 (e.g.
        // profile 5 IPT-PQ, profile 4) have no standard fallback and show wrong colors. Log it so the
        // result is never a silent mystery.
        if (const AVPacketSideData *sd =
                av_packet_side_data_get(p->coded_side_data, p->nb_coded_side_data, AV_PKT_DATA_DOVI_CONF);
            sd != nullptr && sd->size >= sizeof(AVDOVIDecoderConfigurationRecord)) {
            AVDOVIDecoderConfigurationRecord dovi{};
            std::memcpy(&dovi, sd->data, sizeof(dovi));
            const unsigned compat = dovi.dv_bl_signal_compatibility_id;
            const bool hasHdrBase = compat == 1 || compat == 4;
            isyslog("vaapivideo/mediaplayer: Dolby Vision profile %u (BL compatibility id %u) -- DV not decodable; "
                    "base layer %s",
                    static_cast<unsigned>(dovi.dv_profile), compat,
                    hasHdrBase ? "renders as standard HDR10/HLG" : "may show wrong colors (no standard fallback)");
        }
    }

    // reserve() and never push_back after: each descriptor's info.extradata aliases its own
    // extradataStorage, so a reallocation would dangle the active mirror.
    unsigned audioCount = 0;
    for (unsigned i = 0; i < formatCtx->nb_streams; ++i) {
        if (formatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            ++audioCount;
        }
    }
    audioTracks.reserve(audioCount);
    for (unsigned i = 0; i < formatCtx->nb_streams; ++i) {
        const AVStream *stream = formatCtx->streams[i];
        if (stream->codecpar->codec_type != AVMEDIA_TYPE_AUDIO) {
            continue;
        }
        AudioTrackDesc desc;
        desc.avStreamIndex = static_cast<int>(i);
        desc.timeBase = stream->time_base;
        desc.srcChannels = stream->codecpar->ch_layout.nb_channels;
        if (const AVDictionaryEntry *lang = av_dict_get(stream->metadata, "language", nullptr, 0);
            lang != nullptr && lang->value != nullptr) {
            desc.language = lang->value;
        }
        PopulateAudioInfo(stream, desc.info, desc.extradataStorage);
        audioTracks.push_back(std::move(desc));
    }

    // Default to libavformat's best-stream pick; the player overrides it with the preferred-language
    // track before the codec opens.
    currentAudioTrack = -1;
    for (size_t i = 0; i < audioTracks.size(); ++i) {
        if (audioTracks.at(i).avStreamIndex == audioStreamIndex) {
            currentAudioTrack = static_cast<int>(i);
            break;
        }
    }
    if (currentAudioTrack < 0 && !audioTracks.empty()) {
        currentAudioTrack = 0;
    }
    ApplyCurrentAudioTrack();

    // Enumerate supported subtitle streams (reserve once: SubtitleTrackDesc holds a borrowed
    // codecpar, but the vector itself must not move while the chooser indexes into it). Unsupported
    // bitmap codecs (hdmv_pgs, xsub) are skipped here.
    unsigned subtitleCount = 0;
    for (unsigned i = 0; i < formatCtx->nb_streams; ++i) {
        if (IsSupportedSubtitleStream(formatCtx->streams[i])) {
            ++subtitleCount;
        }
    }
    subtitleTracks.reserve(subtitleCount);
    for (unsigned i = 0; i < formatCtx->nb_streams; ++i) {
        const AVStream *stream = formatCtx->streams[i];
        if (!IsSupportedSubtitleStream(stream)) {
            continue;
        }
        const AVCodecParameters *p = stream->codecpar;
        SubtitleTrackDesc desc;
        desc.avStreamIndex = static_cast<int>(i);
        desc.timeBase = stream->time_base;
        desc.codecId = p->codec_id;
        desc.codecpar = p;
        if (const AVDictionaryEntry *lang = av_dict_get(stream->metadata, "language", nullptr, 0);
            lang != nullptr && lang->value != nullptr) {
            desc.language = lang->value;
        }
        subtitleTracks.push_back(std::move(desc));
    }
    // Subtitles default to off: the user enables them with the Subtitles key (like dvbplayer).
    currentSubtitleTrack = -1;
    ApplyCurrentSubtitleTrack();
}

auto cVaapiMediaSource::PopulateAudioInfo(const AVStream *stream, AudioStreamInfo &info, std::vector<uint8_t> &storage)
    -> void {
    const AVCodecParameters *p = stream->codecpar;
    info.codecId = p->codec_id;
    // Containers can leave the rate at 0 when it only rides in-band (raw-ADTS AAC in TS is the
    // classic case -- avformat_find_stream_info can give up before a usable frame). Open at the
    // DVB-nominal 48 kHz exactly like the live path does (OpenCodec(detected, 48000, 2)): the
    // decoder re-derives the true rate from each frame and swresample converts at a fixed ratio
    // (AVSYNC.md invariant 2), so a nominal mismatch costs nothing. SetStreamParams rejects <= 0,
    // which would needlessly drop the track and play video-only.
    info.sampleRate = p->sample_rate > 0 ? p->sample_rate : 48000;
    // True channel count; the sink picks the PCM layout (ChooseOutputChannels). 0/unknown -> stereo
    // (SetStreamParams rejects <= 0).
    info.channels = p->ch_layout.nb_channels > 0 ? p->ch_layout.nb_channels : 2;
    CopyExtradata(p, storage, info.extradata, info.extradataSize);
}

auto cVaapiMediaSource::ApplyCurrentAudioTrack() -> void {
    if (currentAudioTrack < 0 || currentAudioTrack >= static_cast<int>(audioTracks.size())) {
        audioStreamIndex = -1;
        audioTimeBase = AVRational{.num = 1, .den = 90000};
        audioInfo = AudioStreamInfo{};
        return;
    }
    const AudioTrackDesc &desc = audioTracks.at(static_cast<size_t>(currentAudioTrack));
    audioStreamIndex = desc.avStreamIndex;
    audioTimeBase = desc.timeBase;
    audioInfo = desc.info; // .extradata still aliases desc.extradataStorage (stable in the table)
}

[[nodiscard]] auto cVaapiMediaSource::SelectAudioTrack(int trackIdx) -> bool {
    if (trackIdx < 0 || trackIdx >= static_cast<int>(audioTracks.size())) {
        return false;
    }
    if (trackIdx == currentAudioTrack) {
        return true;
    }
    currentAudioTrack = trackIdx;
    ApplyCurrentAudioTrack();
    // Clear any stale discard window so a pre-open repoint (no following seek) keeps the new track's
    // opening packets.
    discardAudioBefore90k = AV_NOPTS_VALUE;
    return true;
}

auto cVaapiMediaSource::ApplyCurrentSubtitleTrack() -> void {
    if (currentSubtitleTrack < 0 || currentSubtitleTrack >= static_cast<int>(subtitleTracks.size())) {
        subtitleStreamIndex = -1;
        subtitleTimeBase = AVRational{.num = 1, .den = 90000};
        return;
    }
    const SubtitleTrackDesc &desc = subtitleTracks.at(static_cast<size_t>(currentSubtitleTrack));
    subtitleStreamIndex = desc.avStreamIndex;
    subtitleTimeBase = desc.timeBase;
}

[[nodiscard]] auto cVaapiMediaSource::SelectSubtitleTrack(int trackIdx) -> bool {
    // trackIdx < 0 disables routing (subtitles off). A valid index >= track count is rejected.
    if (trackIdx >= static_cast<int>(subtitleTracks.size())) {
        return false;
    }
    currentSubtitleTrack = trackIdx < 0 ? -1 : trackIdx;
    ApplyCurrentSubtitleTrack();
    return true;
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
    // Reads go straight into @p out (clean on entry: the caller unrefs it before every new
    // read, and av_read_frame leaves it blank on failure) -- no per-packet AVPacket
    // allocation on this hot path.
    while (true) {
        // Interruptibility for local files: av_read_frame() polls the InterruptOnStop
        // callback only when libavformat performs blocking I/O. Streams composed of many
        // skipped packets (damaged TS with empty / untracked stream_index / pre-target
        // audio prefix) can spin in this loop without ever entering a blocking read, so
        // poll stopFlag at the top of each iteration too.
        if (stopFlag != nullptr && stopFlag->load(std::memory_order_acquire)) {
            return AVERROR_EXIT;
        }
        const int ret = av_read_frame(formatCtx.get(), out);
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
            // The abort latches AVERROR_EXIT + eof_reached in the AVIOContext: without this reset
            // every later read fails instantly on the sticky error without touching the file -- a
            // permanent silent stall for any consumer that does not follow up with a container
            // seek (trick feed, subtitle switch). Both fields are public AVIOContext API.
            if (formatCtx->pb != nullptr) {
                formatCtx->pb->error = 0;
                formatCtx->pb->eof_reached = 0;
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
        if (out->data == nullptr || out->size <= 0) {
            av_packet_unref(out);
            continue;
        }

        AVRational tb{};
        if (out->stream_index == videoStreamIndex) {
            tb = videoTimeBase;
            stream = MediaPacketStream::Video;
        } else if (out->stream_index == audioStreamIndex) {
            tb = audioTimeBase;
            stream = MediaPacketStream::Audio;
        } else if (currentSubtitleTrack >= 0 && out->stream_index == subtitleStreamIndex) {
            tb = subtitleTimeBase; // MUST set before Rebase90k -- else tb stays {0,0} and pts -> NOPTS
            // Subtitle cue. Rebase pts AND duration to 90 kHz (cues carry a display duration),
            // then hand straight to the consumer: subtitles bypass the pre-sync / audio-discard gates
            // below -- the converter keys display off cue start/end vs the clock, so a stale cue simply
            // never matches the clock. duration uses the same 90 kHz target as Rebase90k's pts path.
            // The tb validity check mirrors Rebase90k: a damaged tb would SIGFPE in av_rescale_q.
            out->pts = Rebase90k(out->pts, tb, false); // never let a subtitle seed the timeline origin
            out->dts = Rebase90k(out->dts, tb, false);
            if (out->duration > 0 && tb.num > 0 && tb.den > 0) {
                constexpr AVRational k90kHz{.num = 1, .den = 90000};
                out->duration = av_rescale_q(out->duration, tb, k90kHz);
            }
            stream = MediaPacketStream::Subtitle;
            return 0;
        } else {
            av_packet_unref(out); // untracked (unselected subtitle / data)
            continue;
        }

        // Rebase to a zero-based 90 kHz timeline. Files with non-zero container start_time
        // (TS recordings, some MP4s) would otherwise hand huge absolute PTS values to the
        // downstream audio clock and break GetIndex() / Seek() math.
        out->pts = Rebase90k(out->pts, tb);
        out->dts = Rebase90k(out->dts, tb);
        // Audio packets sometimes carry DTS only (TS containers). Audio has no B-frame
        // reorder so PTS == DTS; video keeps its real PTS to preserve reorder offset.
        if (stream == MediaPacketStream::Audio && out->pts == AV_NOPTS_VALUE) {
            out->pts = out->dts;
        }

        const int64_t clock90k = PacketClock90k(out);
        // Drop pre-sync prefix (rebased clock < 0): PopulateStreamInfo() picks
        // ptsOrigin90k = MAX(stream.start_time) so the trailing stream defines t=0 and the
        // leading stream's prefix is discarded here -- both streams start together.
        if (clock90k != AV_NOPTS_VALUE && clock90k < 0) {
            av_packet_unref(out);
            continue;
        }
        // Post-seek audio discard: keep video preroll (rebuilds H.264/HEVC reference chain)
        // but drop audio earlier than the seek target. Without this the master clock anchors
        // at the earlier keyframe libavformat landed on, and the drain sits in a re-arm
        // freerun loop until audio plays forward to the requested target.
        if (stream == MediaPacketStream::Audio && discardAudioBefore90k != AV_NOPTS_VALUE) {
            if (clock90k == AV_NOPTS_VALUE || clock90k < discardAudioBefore90k) {
                av_packet_unref(out);
                continue;
            }
            discardAudioBefore90k = AV_NOPTS_VALUE;
        }
        // Slow-motion entry: the video preroll must be decoded (reference chain) but never shown
        // (trick pacing has no clock gate to swallow it) -- AV_PKT_FLAG_DISCARD makes libavcodec
        // drop the decoded output. Keyed on PacketClock90k (PTS-or-DTS, like the audio discard
        // above) so DTS-only TS video doesn't disarm the window early; a packet with no timing at
        // all keeps it armed. A few reorder-late B-frames may still slip through the disarm; they
        // sit within a frame or two of the target and are invisible.
        if (stream == MediaPacketStream::Video && discardVideoBefore90k != AV_NOPTS_VALUE) {
            if (clock90k != AV_NOPTS_VALUE && clock90k < discardVideoBefore90k) {
                out->flags |= AV_PKT_FLAG_DISCARD;
            } else if (clock90k != AV_NOPTS_VALUE) {
                discardVideoBefore90k = AV_NOPTS_VALUE;
            }
        }

        return 0;
    }
}

[[nodiscard]] auto cVaapiMediaSource::Rebase90k(int64_t ts, AVRational tb, bool seedOrigin) noexcept -> int64_t {
    // tb.num/den <= 0 guards damaged container metadata: av_rescale_q divides by tb.den
    // and would produce garbage / UB on a zero denominator. Returning NOPTS makes the
    // downstream pre-sync and discard checks skip the packet safely.
    if (ts == AV_NOPTS_VALUE || tb.num <= 0 || tb.den <= 0) {
        return AV_NOPTS_VALUE;
    }
    constexpr AVRational k90kHz{.num = 1, .den = 90000};
    const int64_t ts90k = av_rescale_q(ts, tb, k90kHz);
    if (ptsOrigin90k == AV_NOPTS_VALUE) {
        // Only audio/video may seed the origin; a subtitle arriving first rebases to NOPTS (and is
        // dropped by the converter) rather than anchoring the A/V timeline on subtitle timing.
        if (!seedOrigin) {
            return AV_NOPTS_VALUE;
        }
        ptsOrigin90k = ts90k;
    }
    return ts90k - ptsOrigin90k;
}

auto cVaapiMediaSource::Flush() -> void {
    if (formatCtx) {
        avformat_flush(formatCtx.get());
    }
    eofReached = false; // a prior EOF must not latch: the post-seek/flush read has to resume
}

[[nodiscard]] auto cVaapiMediaSource::Seek(int64_t targetPts90k) -> bool {
    if (!formatCtx || videoStreamIndex < 0) {
        return false;
    }
    // Reset upfront so a failed seek doesn't leave stale discard windows armed.
    discardAudioBefore90k = AV_NOPTS_VALUE;
    discardVideoBefore90k = AV_NOPTS_VALUE;
    // Convert from 90 kHz to the video stream's time base (av_seek_frame's units). Player Seek()
    // talks the zero-based timeline; we re-add the origin offset so we land at the matching
    // wall-clock keyframe inside the container's native timeline.
    const AVRational dstTb = formatCtx->streams[videoStreamIndex]->time_base;
    // Same damaged-metadata guard as Rebase90k: av_rescale_q divides by dstTb.num here,
    // so a zero would raise SIGFPE on a corrupt file.
    if (dstTb.num <= 0 || dstTb.den <= 0) {
        esyslog("vaapivideo/mediaplayer: seek rejected -- invalid video time base %d/%d", dstTb.num, dstTb.den);
        return false;
    }
    constexpr AVRational k90kHz{.num = 1, .den = 90000};
    const int64_t origin = (ptsOrigin90k == AV_NOPTS_VALUE) ? 0 : ptsOrigin90k;
    const int64_t target = std::max<int64_t>(targetPts90k, 0);
    // Corrupt container start_time can saturate the origin to INT64_MAX (av_rescale_q clamps),
    // so adding blindly would be signed-overflow UB before av_rescale_q could saturate again.
    if (origin > 0 && target > std::numeric_limits<int64_t>::max() - origin) [[unlikely]] {
        esyslog("vaapivideo/mediaplayer: seek rejected -- timestamp overflow");
        return false;
    }
    const int64_t seekTs = av_rescale_q(target + origin, k90kHz, dstTb);
    const int ret = av_seek_frame(formatCtx.get(), videoStreamIndex, seekTs, AVSEEK_FLAG_BACKWARD);
    if (ret < 0) {
        esyslog("vaapivideo/mediaplayer: av_seek_frame: %s", AvErr(ret).data());
        return false;
    }
    Flush();
    discardAudioBefore90k = audioStreamIndex >= 0 ? target : AV_NOPTS_VALUE;
    return true;
}

// ============================================================================
// === cVaapiPlayer ===
// ============================================================================

cVaapiPlayer::cVaapiPlayer(std::string uri, std::vector<PlaylistEntry> entries, int startMs)
    : cPlayer(pmAudioVideo), cThread("vaapivideo/mediaplayer"), originUri(std::move(uri)), startPositionMs(startMs),
      playlist(std::move(entries)) {
    if (playlist.empty()) {
        esyslog("vaapivideo/mediaplayer: cVaapiPlayer constructed with empty playlist");
    }
}

cVaapiPlayer::~cVaapiPlayer() noexcept {
    // Must call CloseCurrentEntry here, not a bare source.reset(): ~cPlayer's Detach()->Activate(false)
    // dispatches to the empty base cPlayer::Activate during base destruction, so this is the only path
    // that runs our teardown -- without it ClrAvailableTracks/ClearForMediaPlayer never fire and
    // mediaPlayerAudioActive stays latched, wedging live-TV audio after the player exits.
    // ioInterrupt breaks a parked network read so Cancel(3) joins fast.
    stopping.store(true, std::memory_order_release);
    ioInterrupt.store(true, std::memory_order_release);
    {
        const cMutexLock lock(&pauseMutex);
        pauseCondition.Broadcast();
    }
    Cancel(3);
    // Stop the subtitle thread + free its overlay while the device/display are still attached
    // (AwaitSubtitleHidden needs the live display). Then run the per-entry teardown.
    if (subtitles) {
        subtitles->Shutdown();
    }
    CloseCurrentEntry();
    // Back to the default mode, here for the same reason CloseCurrentEntry is (see above): this
    // destructor is the only teardown path that actually runs, and leaving the output at a film
    // rate makes the whole VDR UI sluggish. Not inside CloseCurrentEntry() -- a playlist advance
    // calls that too, and the next entry picks its own mode.
    if (auto *vaapiDev = FindPrimaryVaapiDevice(); vaapiDev != nullptr) {
        vaapiDev->ResetDisplayModeToDefault();
    }
}

[[nodiscard]] auto cVaapiPlayer::CurrentPositionMs() const noexcept -> int {
    auto *vaapiDev = FindPrimaryVaapiDevice();
    if (vaapiDev == nullptr) {
        return 0;
    }
    if (const int64_t stc = vaapiDev->GetSTC(); stc >= 0) {
        return static_cast<int>(stc / PTS_TICKS_PER_MS);
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
    // Outside normal play the reference is meaningless: during pause the demux loop is stopped and
    // the audio clock extrapolates from its last anchor while ALSA is dropped (lastAudio - audioClock
    // turns into a fake negative value); during trick modes no audio is fed at all, so a stale
    // lastAudio would deadlock the throttle. Returning NOPTS skips both the throttle and the
    // misleading status line.
    if (playMode.load(std::memory_order_acquire) != PlayMode::Play) {
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
    const size_t idx = currentIndex.load(std::memory_order_relaxed);
    if (idx >= playlist.size()) {
        return false;
    }
    // Everything up to the publish below runs UNLOCKED: nextSource is invisible to other threads,
    // and avformat_open_input can block for seconds on a network URL -- holding sourceMutex across
    // it would stall every main-thread reader (e.g. InfoText) for the whole connect. Only this
    // thread (demux; main pre-Start) ever opens/replaces the source, so there is no writer race.
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
    // Pick the preferred-language track BEFORE OpenForMediaPlayer so the right codec opens directly
    // (no startup reopen).
    if (const int preferred = ChoosePreferredAudioTrack(*nextSource); preferred >= 0) {
        (void)nextSource->SelectAudioTrack(preferred);
    }
    // Proactive display-mode match: the container already carries the authoritative frame rate and
    // coded size, so unlike live TV / recordings this needs no decoded frame and no stability gate.
    // CloseCurrentEntry() has drained the pipeline, so any modeset lands in an idle window rather
    // than mid-picture. Computed here, requested below once the codecs are known to have opened.
    StreamModeRequest modeRequest{};
    if (const auto &videoInfo = nextSource->VideoInfo();
        videoInfo.codedWidth > 0 && videoInfo.codedHeight > 0 && videoInfo.fpsNum > 0 && videoInfo.fpsDen > 0) {
        // Field rate for interlaced content: the VPP deinterlacer emits 2 frames per coded frame,
        // which is what makes 1080i25 correctly ask for 50 Hz rather than 25.
        // Same halves-up rounding as the filter chain: the post-build republish must land on the
        // identical millihertz or the stability gate treats it as a second, distinct format. The
        // upper bound guards the cast -- a malformed container can report an absurd r_frame_rate.
        const int64_t rateNum = static_cast<int64_t>(videoInfo.fpsNum) * 1000 * (videoInfo.streamInterlaced ? 2 : 1);
        const int64_t rateMilliHz = (rateNum + (videoInfo.fpsDen / 2)) / videoInfo.fpsDen;
        if (rateMilliHz > 0 && rateMilliHz <= UINT32_MAX) {
            modeRequest = {.height = static_cast<uint32_t>(videoInfo.codedHeight),
                           .rateMilliHz = static_cast<uint32_t>(rateMilliHz),
                           .width = static_cast<uint32_t>(videoInfo.codedWidth)};
        }
    }
    if (!vaapiDev->OpenForMediaPlayer(nextSource->VideoInfo(), nextSource->AudioInfo())) {
        vaapiDev->ClearForMediaPlayer();
        return false;
    }
    // Only once the entry is known to be playable: the codec-open failure path above restores
    // nothing, so requesting first would strand the panel on an unplayable entry's rate. Still
    // ahead of every submitted frame, so the switch lands before playback, not a second into it.
    if (modeRequest.IsValid()) {
        vaapiDev->EvaluateDisplayMode(modeRequest, PlaybackSource::MediaPlayer, /*immediate=*/true);
    }
    // Publish + per-entry resets under the lock: readers see either the old source or the fully
    // initialized new one, never a half-open state.
    const cMutexLock lock(&sourceMutex);
    source = std::move(nextSource);
    cachedDurationMs.store(source->DurationMs(), std::memory_order_release);
    cachedVideoFps.store(source->VideoFps(), std::memory_order_release);
    // New entry = new PTS timeline; the throttle's high-water mark and the seek-target
    // fallback must not carry over from the previous entry.
    latestAudioPts90k.store(AV_NOPTS_VALUE, std::memory_order_release);
    pendingSeekTargetMs.store(-1, std::memory_order_release);
    // Drop any audio-switch request left pending from the previous entry before re-registering.
    audioSwitch.pending.store(false, std::memory_order_release);
    audioSwitch.targetIdx.store(-1, std::memory_order_release);
    RegisterAudioTracks();
    // Subtitle converter: one per player, created lazily on the first entry. Start each entry off
    // (closed) and drop any stale switch request.
    if (!subtitles) {
        subtitles = std::make_unique<cSubtitleConverter>(vaapiDev);
    } else {
        subtitles->Close();
    }
    subtitleSwitch.pending.store(false, std::memory_order_release);
    subtitleSwitch.targetIdx.store(-1, std::memory_order_release);
    RegisterSubtitleTracks();
    return true;
}

auto cVaapiPlayer::CloseCurrentEntry() noexcept -> void {
    const cMutexLock lock(&sourceMutex);
    if (subtitles) {
        subtitles->Close(); // drop decoder + cues + hide overlay before the source goes away
    }
    if (auto *vaapiDev = FindPrimaryVaapiDevice(); vaapiDev != nullptr) {
        vaapiDev->ClrAvailableTracks(); // drop this entry's audio + subtitle tracks (no stale list into live TV)
        vaapiDev->ClearForMediaPlayer();
    }
    audioSwitch.menuIndex.store(-1, std::memory_order_release);
    audioSwitch.trackCount.store(0, std::memory_order_release);
    subtitleSwitch.trackCount.store(0, std::memory_order_release);
    subtitleSwitch.menuIndex.store(-1, std::memory_order_release);
    cachedDurationMs.store(-1, std::memory_order_release); // -1 = no entry open (GetIndex reports false)
    cachedVideoFps.store(0.0, std::memory_order_release);
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
        ApplyStartPosition(); // resume the first entry before the demux thread submits any packet
        state.store(State::Running, std::memory_order_release);
        Start();
    } else {
        stopping.store(true, std::memory_order_release);
        ioInterrupt.store(true, std::memory_order_release); // break a parked network read so Cancel(3) joins fast
        {
            const cMutexLock lock(&pauseMutex);
            pauseCondition.Broadcast();
        }
        Cancel(3);
        if (subtitles) {
            subtitles->Shutdown(); // stop subtitle thread + free overlay while display is still attached
        }
        CloseCurrentEntry();
        // Mode restore lives in ~cVaapiPlayer, not here: VDR reaches Activate(false) only through
        // ~cPlayer, where the dynamic type has already decayed to the base and this override is
        // never dispatched. Kept as a belt-and-braces no-op for the (currently unreachable)
        // cDevice::AttachPlayer detach path -- ResetDisplayModeToDefault is idempotent.
        if (auto *vaapiDev = FindPrimaryVaapiDevice(); vaapiDev != nullptr) {
            vaapiDev->ResetDisplayModeToDefault();
        }
        state.store(State::Stopped, std::memory_order_release);
    }
}

auto cVaapiPlayer::Play() -> void {
    switch (playMode.load(std::memory_order_acquire)) {
        case PlayMode::Play:
            return;
        case PlayMode::Pause:
            // BOTH halves are required: DevicePlay() restarts the audio master clock; the mode
            // change un-parks the demux loop (else the queues never refill). `state` tracks
            // lifecycle, not transient phases, so this main-thread path never writes it.
            playMode.store(PlayMode::Play, std::memory_order_release);
            DevicePlay();
            WakeDemux();
            return;
        case PlayMode::Fast:
        case PlayMode::Slow:
            ExitTrick(false);
            return;
    }
}

auto cVaapiPlayer::Pause() -> void {
    switch (playMode.load(std::memory_order_acquire)) {
        case PlayMode::Pause:
            Play(); // toggle, like cDvbPlayer::Pause()
            return;
        case PlayMode::Play:
            // BOTH halves are required: DeviceFreeze() halts the audio master clock (else resume
            // re-anchors with a stutter); the mode change parks the demux loop (else queues fill
            // while frozen and OOM on long pauses).
            playMode.store(PlayMode::Pause, std::memory_order_release);
            DeviceFreeze();
            return;
        case PlayMode::Fast:
        case PlayMode::Slow:
            ExitTrick(true);
            return;
    }
}

auto cVaapiPlayer::Forward() -> void { CycleTrick(true); }

auto cVaapiPlayer::Backward() -> void { CycleTrick(false); }

auto cVaapiPlayer::LeaveTrickWithoutReanchor() -> void {
    if (IsTrickMode()) {
        EndTrick(PlayMode::Play);
    }
}

auto cVaapiPlayer::CycleTrick(bool towardForward) -> void {
    // cDvbPlayer::Forward()/Backward() folded over the direction: they are exact mirrors, and one
    // body keeps the transition graph in one place. sameDir = the active trick already runs in the
    // pressed direction; the opposite direction winds down one notch instead (through Speeds' '1'
    // entry into Play()/Pause()), so a scan is always left via normal play, never flipped abruptly.
    const bool multiSpeed = Setup.MultiSpeedMode != 0;
    const bool sameDir = trickForward.load(std::memory_order_acquire) == towardForward;
    switch (playMode.load(std::memory_order_acquire)) {
        case PlayMode::Fast:
            if (multiSpeed) {
                TrickSpeedStep(sameDir ? +1 : -1);
                return;
            }
            if (sameDir) {
                Play(); // single-speed: the second press (or the key release) ends the scan
                return;
            }
            [[fallthrough]]; // single-speed opposite fast: restart in the pressed direction
        case PlayMode::Play:
            EnterTrick(PlayMode::Fast, towardForward, multiSpeed ? +1 : +MEDIAPLAYER_TRICK_STEPS_MAX);
            return;
        case PlayMode::Slow:
            if (multiSpeed) {
                TrickSpeedStep(sameDir ? -1 : +1);
                return;
            }
            if (sameDir) {
                Pause(); // single-speed: the second press ends slow motion
                return;
            }
            [[fallthrough]]; // single-speed opposite slow: restart in the pressed direction
        case PlayMode::Pause:
            EnterTrick(PlayMode::Slow, towardForward, multiSpeed ? -1 : -MEDIAPLAYER_TRICK_STEPS_MAX);
            return;
    }
}

auto cVaapiPlayer::EnterTrick(PlayMode mode, bool forward, int firstStep) -> void {
    const int durationMs = cachedDurationMs.load(std::memory_order_acquire);
    if (durationMs < 0) {
        return; // no entry open
    }
    if (!forward && durationMs == 0) {
        // Reverse needs seekable, bounded input (cf. cDvbPlayer gating trick play on the index file).
        isyslog("vaapivideo/mediaplayer: reverse trick play refused -- source has no known duration");
        return;
    }
    // Anchor position FIRST: TrickSpeedStep's generation flush wipes the decoder's lastPts, after
    // which the demux-side transition could only fall back to the audio clock (stale in trick) or 0.
    trickAnchorMs.store(CurrentPositionMs(), std::memory_order_release);
    trickForward.store(forward, std::memory_order_release);
    trickSpeedIdx.store(MEDIAPLAYER_TRICK_NORMAL_IDX, std::memory_order_release);
    // Command BEFORE playMode: the demux gates its trick feed on "no pending command", so the mode
    // becoming visible must imply the command is visible too (release/acquire); otherwise a
    // mid-iteration mode snapshot runs the reverse feed on uninitialized step targets.
    ArmDemuxInterrupt(); // and the interrupt before the command -- see the invariant at the declaration
    TrickCommand cmd = TrickCommand::EnterReverse;
    if (forward) {
        cmd = mode == PlayMode::Slow ? TrickCommand::EnterSlowForward : TrickCommand::EnterForward;
    }
    trickCommand.store(cmd, std::memory_order_release);
    playMode.store(mode, std::memory_order_release);
    // Device trick state before waking the demux: SetTrickSpeed's generation flush then usually
    // precedes the aligned trick feed (first trick frames not wiped). A fast demux may consume the
    // command earlier; the flush then wipes a few just-fed frames, which the feed simply re-sends.
    // Coming from Pause the device is still frozen -- which is exactly what makes
    // cVaapiDevice::TrickSpeed() derive isFast=false for the slow modes (Freeze-before-slow).
    TrickSpeedStep(firstStep);
    WakeDemux();
}

auto cVaapiPlayer::TrickSpeedStep(int increment) -> void {
    const int idx = trickSpeedIdx.load(std::memory_order_acquire) + increment;
    if (idx < 0 || idx >= static_cast<int>(MEDIAPLAYER_TRICK_SPEEDS.size())) [[unlikely]] {
        return; // outside the table; unreachable via the key state machine
    }
    const int entry = MEDIAPLAYER_TRICK_SPEEDS.at(static_cast<size_t>(idx));
    if (entry == 0) {
        return; // sentinel: the speed saturates -- dvbplayer-style silent no-op, no device call
    }
    trickSpeedIdx.store(idx, std::memory_order_release);
    if (entry == 1) {
        // Wound back to normal: leave the trick -- resume play from fast, pause from slow.
        if (playMode.load(std::memory_order_acquire) == PlayMode::Fast) {
            Play();
        } else {
            Pause();
        }
        return;
    }
    const bool forward = trickForward.load(std::memory_order_acquire);
    const bool slow = playMode.load(std::memory_order_acquire) == PlayMode::Slow;
    // Repeat count exactly like cDvbPlayer::TrickSpeed(): Mult is 1 only for slow-forward (all
    // frames repeated 2/4/8 times); every stepping mode uses SPEED_MULT over the table entry.
    const int mult = (slow && forward) ? 1 : MEDIAPLAYER_TRICK_SPEED_MULT;
    const int speed = std::min(entry > 0 ? mult / entry : -entry * mult, MEDIAPLAYER_TRICK_DEVICE_SPEED_MAX);
    dsyslog("vaapivideo/mediaplayer: trick %s %s notch %d (device speed %d)", slow ? "slow" : "fast",
            forward ? "forward" : "backward", std::abs(idx - MEDIAPLAYER_TRICK_NORMAL_IDX), speed);
    DeviceTrickSpeed(speed, forward);
}

auto cVaapiPlayer::EndTrick(PlayMode nextMode) -> void {
    trickForward.store(true, std::memory_order_release);
    trickSpeedIdx.store(MEDIAPLAYER_TRICK_NORMAL_IDX, std::memory_order_release);
    playMode.store(nextMode, std::memory_order_release);
    // Safe from either thread: DevicePlay() is atomics + decoder/display notifications only.
    DevicePlay();
}

auto cVaapiPlayer::AbortTrick(const char *reason) -> void {
    // Slow modes were entered from pause, fast from play; return the user where they came from.
    const bool toPause = playMode.load(std::memory_order_acquire) == PlayMode::Slow;
    EndTrick(toPause ? PlayMode::Pause : PlayMode::Play);
    if (toPause) {
        DeviceFreeze();
    }
    esyslog("vaapivideo/mediaplayer: %s -- leaving trick mode", reason);
}

auto cVaapiPlayer::ExitTrick(bool toPause) -> void {
    // Anchor the resume position before DevicePlay() ends the trick (lastPts is the last shown
    // trick frame). Keep the entry anchor when nothing was presented yet (position reads 0).
    if (const int pos = CurrentPositionMs(); pos > 0) {
        trickAnchorMs.store(pos, std::memory_order_release);
    }
    EndTrick(toPause ? PlayMode::Pause : PlayMode::Play);
    // The pause exit re-freezes right after DevicePlay() -- the queues are empty during trick,
    // so nothing plays out in between.
    if (toPause) {
        DeviceFreeze();
    }
    ArmDemuxInterrupt(); // before the command, never after -- see the invariant at the declaration
    trickCommand.store(TrickCommand::Exit, std::memory_order_release);
    WakeDemux();
    dsyslog("vaapivideo/mediaplayer: trick exit -> %s", toPause ? "pause" : "play");
}

auto cVaapiPlayer::WakeDemux() -> void {
    // Wake the demux thread out of a pause park so it services the staged command immediately.
    // ArmDemuxInterrupt() (called BEFORE the command store) covers the blocked-in-read case.
    const cMutexLock lock(&pauseMutex);
    pauseCondition.Broadcast();
}

auto cVaapiPlayer::Seek(int64_t deltaMs) -> void {
    if (deltaMs == 0) {
        return;
    }
    // fetch_add (not store) so rapid key repeats sum: 5x kRight in one demux cycle = +50s,
    // not +10s. Per-key repeat shaping is RcRepeatDelay/RcRepeatDelta in setup.conf.
    ArmDemuxInterrupt();
    seekDeltaMs.fetch_add(deltaMs, std::memory_order_relaxed);
    seekPending.store(true, std::memory_order_release);
    WakeDemux();
}

auto cVaapiPlayer::Next() -> void {
    LeaveTrickWithoutReanchor(); // AdvancePlaylist reopens the pipeline; the next entry starts at normal speed
    ArmDemuxInterrupt();
    nextRequested.store(true, std::memory_order_release);
    WakeDemux();
}

[[nodiscard]] auto cVaapiPlayer::Title() const -> std::string {
    const size_t idx = currentIndex.load(std::memory_order_relaxed);
    return (idx < playlist.size()) ? playlist.at(idx).title : std::string{};
}

auto cVaapiPlayer::ApplyStartPosition() -> void {
    const int targetMs = std::exchange(startPositionMs, 0); // first entry only; playlist advance stays fresh
    if (targetMs <= 0) {
        return;
    }
    const cMutexLock lock(&sourceMutex);
    if (!source) {
        return;
    }
    // Live/unseekable or at/past the tail: start from 0. The 1 s margin mirrors SeekToMs.
    const int totalMs = source->DurationMs();
    if (totalMs <= 0 || targetMs >= totalMs - 1000) {
        return;
    }
    // Bare container seek: no demux thread yet and nothing submitted, so no FlushForSeek / state dance.
    // Seek() arms the audio-discard window so the clock anchors at the target, not the earlier keyframe.
    if (!source->Seek(static_cast<int64_t>(targetMs) * PTS_TICKS_PER_MS)) {
        esyslog("vaapivideo/mediaplayer: resume seek to %dms failed -- starting from 0", targetMs);
        return;
    }
    pendingSeekTargetMs.store(targetMs, std::memory_order_release); // read until GetSTC() anchors
    isyslog("vaapivideo/mediaplayer: resuming at %dms (bookmark)", targetMs);
}

[[nodiscard]] auto cVaapiPlayer::MakeBookmark() const -> MediaBookmark {
    MediaBookmark bm{.uri = originUri, .positionMs = 0};
    // A position is meaningful only for a single seekable local file mid-playback; playlists, streams,
    // EOF and failed opens keep 0 (bookmark the URI only).
    if (bm.uri.empty() || HasUrlScheme(originUri) || IsPlaylistUri(originUri) || IsFinished()) {
        return bm;
    }
    // Lock-free: the control dtor calls this on every teardown; sourceMutex could be held by a demux
    // thread parked in a stalled network read, delaying the stop for seconds.
    if (cachedDurationMs.load(std::memory_order_acquire) <= 0) {
        return bm; // no entry open, or live/unseekable
    }
    bm.positionMs = CurrentPositionMs(); // resume lands here exactly on the next start
    return bm;
}

[[nodiscard]] auto cVaapiPlayer::FramesPerSecond() -> double {
    // Skins use this for the ".ff" frame-count suffix; fall back to cPlayer's default 25.
    // Lock-free (cached at open) -- main-thread caller, same rationale as GetIndex.
    const double fps = cachedVideoFps.load(std::memory_order_acquire);
    return fps > 0.0 ? fps : cPlayer::FramesPerSecond();
}

[[nodiscard]] auto cVaapiPlayer::InfoText() const -> std::string {
    const cMutexLock lock(&sourceMutex);
    if (!source) {
        return {};
    }
    const auto [w, h] = source->VideoCodedSize();
    const auto &v = source->VideoInfo();
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
    const auto &tracks = source->AudioTracks();
    const int current = source->CurrentAudioTrack();
    if (tracks.empty()) {
        text += "Audio:    (none)\n";
    } else {
        for (size_t i = 0; i < tracks.size(); ++i) {
            const auto &track = tracks.at(i);
            const std::string_view layout = AudioLayoutLabel(track.srcChannels);
            const std::string channels = layout.empty() ? std::format("{} ch", track.srcChannels) : std::string{layout};
            const std::string lang = track.language.empty() ? std::string{} : std::format(" [{}]", track.language);
            // A leading "* " marks the active track; "  " keeps the others column-aligned.
            text += std::format("Audio {}: {}{} {} Hz {}{}\n", i + 1, (static_cast<int>(i) == current) ? "* " : "  ",
                                avcodec_get_name(track.info.codecId), track.info.sampleRate, channels, lang);
        }
    }
    const auto &subtitleTracks = source->SubtitleTracks();
    const int currentSub = source->CurrentSubtitleTrack();
    if (!subtitleTracks.empty()) {
        for (size_t i = 0; i < subtitleTracks.size(); ++i) {
            const auto &track = subtitleTracks.at(i);
            const char *codecName = SubtitleCodecName(track.codecId);
            const std::string lang = track.language.empty() ? std::string{} : std::format(" [{}]", track.language);
            // A leading "* " marks the active subtitle track; "off" reads as none selected.
            text += std::format("Subs {}:  {}{}{}\n", i + 1, (static_cast<int>(i) == currentSub) ? "* " : "  ",
                                codecName, lang);
        }
    }
    if (playlist.size() > 1) {
        text += std::format("Playlist: {}/{}\n", idx + 1, playlist.size());
    }
    return text;
}

[[nodiscard]] auto cVaapiPlayer::GetReplayMode(bool &Play, bool &Forward, int &Speed) -> bool {
    // cDvbPlayer::GetReplayMode() semantics: Speed -1 = normal play/pause, 0 = single-speed trick,
    // >0 = multi-speed notch. Slow motion reports Play=false, so skins render "1|>" style symbols.
    const PlayMode mode = playMode.load(std::memory_order_acquire);
    const bool trick = mode == PlayMode::Fast || mode == PlayMode::Slow;
    Play = mode == PlayMode::Play || mode == PlayMode::Fast;
    Forward = !trick || trickForward.load(std::memory_order_acquire);
    Speed = -1;
    if (trick) {
        Speed = Setup.MultiSpeedMode != 0
                    ? std::abs(trickSpeedIdx.load(std::memory_order_acquire) - MEDIAPLAYER_TRICK_NORMAL_IDX)
                    : 0;
    }
    return true;
}

[[nodiscard]] auto cVaapiPlayer::GetIndex(int &Current, int &Total, bool /*SnapToIFrame*/) -> bool {
    Current = 0;
    Total = 0;
    // Lock-free: this runs on the VDR main thread on every replay-bar refresh, and sourceMutex may
    // be held by the demux thread across a blocking network read -- taking it here would freeze the
    // whole main loop until the read times out.
    const int durationMs = cachedDurationMs.load(std::memory_order_acquire);
    if (durationMs < 0) {
        return false; // no entry open
    }
    Total = durationMs;
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
    (void)SeekToMs(static_cast<int64_t>(CurrentPositionMs()) + deltaMs);
}

[[nodiscard]] auto cVaapiPlayer::SeekToMs(int64_t targetMs) -> bool {
    // sourceMutex is held by the caller (PerformSeek / PerformAudioSwitch); not re-locked here
    // so the hold stays visible at the call site (relock would be tolerated but obscures it).
    if (!source) {
        return false;
    }
    auto *vaapiDev = FindPrimaryVaapiDevice();
    if (vaapiDev == nullptr) {
        return false;
    }

    const int totalMs = source->DurationMs();
    targetMs = std::max<int64_t>(0, targetMs);
    // 1 s tail margin: landing past the last keyframe would look like a hang.
    if (totalMs > 0 && targetMs > totalMs - 1000) {
        targetMs = std::max<int64_t>(0, totalMs - 1000);
    }
    const int64_t targetPts90k = targetMs * PTS_TICKS_PER_MS;

    // Seek the source FIRST; if it fails we leave the device state untouched so the user keeps
    // playing the old position instead of staring at a blanked frame after a wiped pipeline.
    if (!source->Seek(targetPts90k)) {
        esyslog("vaapivideo/mediaplayer: seek to %lldms failed", static_cast<long long>(targetMs));
        return false;
    }

    vaapiDev->FlushForSeek();
    // Drop pending cues + hide any on-screen subtitle so a stale cue doesn't linger across the seek;
    // new cues arrive as the post-seek packets are read.
    if (subtitles) {
        subtitles->Reset();
    }
    // Reset the throttle high-water mark so a post-seek PTS smaller than pre-seek doesn't
    // stall the demuxer until audio "catches up" to a stale value.
    latestAudioPts90k.store(AV_NOPTS_VALUE, std::memory_order_release);
    // CurrentPositionMs() returns this while GetSTC() is briefly NOPTS post-flush; without
    // it a rapid follow-up Seek() would compute its delta against 0.
    pendingSeekTargetMs.store(static_cast<int>(targetMs), std::memory_order_release);
    dsyslog("vaapivideo/mediaplayer: seek -> %lldms (total=%dms)", static_cast<long long>(targetMs), totalMs);
    return true;
}

auto cVaapiPlayer::PerformTrickTransition(TrickCommand cmd) -> void {
    const cMutexLock lock(&sourceMutex);
    if (!source) {
        return;
    }
    // The anchor was captured on the main thread at the keypress; by now the trick flush has wiped
    // the decoder's lastPts, so reading the position HERE would fall back to the (trick-stale)
    // audio clock or 0. -1 = no capture (never staged without one; belt-and-braces fallback).
    const int anchorMs = trickAnchorMs.load(std::memory_order_acquire);
    const int posMs = anchorMs >= 0 ? anchorMs : CurrentPositionMs();
    // Every transition re-anchors at the shown position via the jump-seek machinery. This is not
    // optional, even for slow-forward: the decoder purges its decoded reserve on every trick
    // generation boundary (clearEpoch bump in SetTrickSpeed) and Freeze() already dropped the
    // packet queue, so continuing from the demux cursor would jump ~the reserve depth (1.5 s+)
    // ahead. SeekToMs() also flushes the pre-trick audio queue (must never play into a trick)
    // and re-arms the position fallback for the replay bar.
    if (!SeekToMs(posMs)) {
        // Fail closed on entry: trick mode without the re-anchor would run the wrong timeline
        // (and reverse would retry a seek that can never work). A failed Exit re-anchor just
        // continues unmoved from the demux cursor -- the least-surprise fallback.
        if (cmd == TrickCommand::EnterForward || cmd == TrickCommand::EnterReverse) {
            AbortTrick("trick entry re-anchor seek failed");
        }
        return;
    }
    switch (cmd) {
        case TrickCommand::EnterSlowForward:
            // Slow motion must resume exactly at the shown frame, but the seek re-feeds from the
            // keyframe at/below it and trick pacing presents every decoded frame -- without this
            // the preroll GOP would replay in slow motion (seconds of content at 1/8 speed).
            source->DiscardVideoPrerollBefore(static_cast<int64_t>(posMs) * PTS_TICKS_PER_MS);
            break;
        case TrickCommand::EnterForward:
            // Fast-forward wants no discard: its first keyframe at/below the position IS the
            // intended start frame (non-keys are dropped by the decoder's FF filter anyway).
            break;
        case TrickCommand::EnterReverse:
            // The first step's backward container seek then lands on the keyframe at/before here.
            reverseTargetPts90k = static_cast<int64_t>(posMs) * PTS_TICKS_PER_MS;
            reverseShownPts90k = std::numeric_limits<int64_t>::max();
            break;
        case TrickCommand::Exit:
        case TrickCommand::None:
            break;
    }
}

[[nodiscard]] auto cVaapiPlayer::PerformReverseStep(cVaapiDevice *vaapiDev, AVPacket *packet) -> bool {
    const cMutexLock lock(&sourceMutex);
    if (!source) {
        return false;
    }
    if (reverseTargetPts90k < 0) {
        // Ran off the file start: auto-resume normal play from 0, like cDvbPlayer's rewind.
        EndTrick(PlayMode::Play);
        (void)SeekToMs(0);
        isyslog("vaapivideo/mediaplayer: reverse reached the start -- resuming play");
        return true;
    }
    if (!source->Seek(reverseTargetPts90k)) {
        // Container seeks don't fail transiently; retrying would loop forever at 5 ms cadence.
        AbortTrick("reverse step seek failed"); // av_seek_frame error logged by Seek()
        return true;
    }
    // Read forward to the first video keyframe the backward seek landed on.
    MediaPacketStream stream{MediaPacketStream::Video};
    while (true) {
        const int ret = source->ReadPacket(packet, stream);
        if (ret == AVERROR_EOF) {
            // Seek landed in a keyframe-less tail; step further back instead of spinning on EOF.
            reverseTargetPts90k -= MEDIAPLAYER_REVERSE_RETRY_STEP_90K;
            return true;
        }
        if (ret != 0) {
            return false; // EAGAIN (command interrupt) / EXIT: let the outer loop service it
        }
        if (stream == MediaPacketStream::Video && (packet->flags & AV_PKT_FLAG_KEY) != 0 &&
            packet->pts != AV_NOPTS_VALUE) {
            break;
        }
        av_packet_unref(packet); // audio / subtitles / GOP tail: reverse shows keyframes only
    }
    const int64_t keyPts = packet->pts;
    if (keyPts >= reverseShownPts90k) {
        // Coarse seek granularity landed on the keyframe already shown: force real progress.
        av_packet_unref(packet);
        reverseTargetPts90k -= MEDIAPLAYER_REVERSE_RETRY_STEP_90K;
        return true;
    }
    if (!vaapiDev->SubmitVideoPacket(packet)) {
        av_packet_unref(packet); // transient (queue raced full); the whole step is redone
        return false;
    }
    av_packet_unref(packet);
    reverseShownPts90k = keyPts;
    reverseTargetPts90k = keyPts - MEDIAPLAYER_REVERSE_EPSILON_90K;
    // Keep the replay bar honest between the (sparse) reverse presents.
    pendingSeekTargetMs.store(static_cast<int>(keyPts / PTS_TICKS_PER_MS), std::memory_order_release);
    return true;
}

auto cVaapiPlayer::SetAudioTrack(eTrackType Type, const tTrackId * /*TrackId*/) -> void {
    // Must NOT take sourceMutex (the demux thread may hold it): only atomics + pause wake, like Seek().
    // All tracks register as ttAudio, so idx = Type - ttAudioFirst and the range check rejects any
    // non-audio Type -- no explicit IS_AUDIO_TRACK test needed.
    const int idx = static_cast<int>(Type) - static_cast<int>(ttAudioFirst);
    if (idx < 0 || idx >= audioSwitch.trackCount.load(std::memory_order_acquire)) {
        return;
    }
    if (idx == audioSwitch.menuIndex.load(std::memory_order_acquire)) {
        return; // initial set / re-select of the active track: nothing to do
    }
    LeaveTrickWithoutReanchor(); // PerformAudioSwitch re-anchors itself; run it in normal play
    ArmDemuxInterrupt();
    audioSwitch.targetIdx.store(idx, std::memory_order_release);
    audioSwitch.pending.store(true, std::memory_order_release);
    WakeDemux();
}

auto cVaapiPlayer::SetSubtitleTrack(eTrackType Type, const tTrackId * /*TrackId*/) -> void {
    // Atomics + pause wake only (no sourceMutex), like SetAudioTrack/Seek. ttNone disables subtitles;
    // a valid subtitle Type maps to a descriptor index range-checked against subtitleSwitch.trackCount.
    int idx = -1;
    if (Type != ttNone) {
        // No explicit IS_SUBTITLE_TRACK test: a non-subtitle Type yields an idx outside
        // [0, subtitleSwitch.trackCount) and is rejected by the range check (cf. SetAudioTrack).
        idx = static_cast<int>(Type) - static_cast<int>(ttSubtitleFirst);
        if (idx < 0 || idx >= subtitleSwitch.trackCount.load(std::memory_order_acquire)) {
            return;
        }
    }
    // No-op the redundant re-selection VDR fires (EnsureSubtitleTrack / chooser re-apply): without this
    // each repeat re-arms ioInterrupt, aborting the demux's av_read_frame again and again, which starves
    // audio/video and stalls the clock until a seek. Mirrors SetAudioTrack()'s audioSwitch.menuIndex guard.
    // exchange so concurrent callers collapse to a single switch request for a given index.
    if (subtitleSwitch.menuIndex.exchange(idx, std::memory_order_acq_rel) == idx) {
        return;
    }
    ArmDemuxInterrupt();
    subtitleSwitch.targetIdx.store(idx, std::memory_order_release);
    subtitleSwitch.pending.store(true, std::memory_order_release);
    WakeDemux();
}

auto cVaapiPlayer::RegisterAudioTracks() -> void {
    // sourceMutex held (from OpenCurrentEntry). Publishes the source's audio streams so the Audio
    // button (cDisplayTracks) lists them, then selects the current track. Uses cPlayer's device
    // wrappers (they forward to the attached device) instead of reaching for the cVaapiDevice.
    if (!source) {
        return;
    }
    DeviceClrAvailableTracks(); // also resets the device's currentAudioTrack to ttNone
    const auto &tracks = source->AudioTracks();
    constexpr int kMaxAudioSlots = static_cast<int>(ttAudioLast) - static_cast<int>(ttAudioFirst) + 1;
    const int count = std::min(static_cast<int>(tracks.size()), kMaxAudioSlots);
    if (static_cast<int>(tracks.size()) > count) {
        isyslog("vaapivideo/mediaplayer: %zu audio tracks -- only the first %d are selectable", tracks.size(), count);
    }
    for (int i = 0; i < count; ++i) {
        const auto &track = tracks.at(static_cast<size_t>(i));
        const std::string desc = AudioTrackDescription(track);
        // Id = avStreamIndex + 1: nonzero (VDR's availability gate) and unique per stream.
        (void)DeviceSetAvailableTrack(ttAudio, i, static_cast<uint16_t>(track.avStreamIndex + 1),
                                      track.language.c_str(), desc.c_str());
    }
    // Clamp the source's current track into the selectable range (only matters for the >32 edge).
    int current = source->CurrentAudioTrack();
    if (current < 0 || current >= count) {
        current = count > 0 ? 0 : -1;
    }
    // Set audioSwitch.menuIndex BEFORE SetCurrentAudioTrack so the SetAudioTrack callback it fires no-ops
    // (no spurious startup re-anchor).
    audioSwitch.menuIndex.store(current, std::memory_order_release);
    audioSwitch.trackCount.store(count, std::memory_order_release);
    if (current >= 0) {
        // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange) -- clamped to ttAudio range above
        (void)DeviceSetCurrentAudioTrack(static_cast<eTrackType>(static_cast<int>(ttAudioFirst) + current));
    }
}

auto cVaapiPlayer::RegisterSubtitleTracks() -> void {
    // sourceMutex held (from OpenCurrentEntry). Publishes the source's supported subtitle streams so the
    // Subtitles button (cDisplaySubtitleTracks) lists them. Subtitles start off (ttNone) -- like
    // dvbplayer, the user enables them with the Subtitles key. Uses cPlayer's device wrappers.
    if (!source) {
        return;
    }
    const auto &tracks = source->SubtitleTracks();
    constexpr int kMaxSubtitleSlots = static_cast<int>(ttSubtitleLast) - static_cast<int>(ttSubtitleFirst) + 1;
    const int count = std::min(static_cast<int>(tracks.size()), kMaxSubtitleSlots);
    if (static_cast<int>(tracks.size()) > count) {
        isyslog("vaapivideo/mediaplayer: %zu subtitle tracks -- only the first %d are selectable", tracks.size(),
                count);
    }
    for (int i = 0; i < count; ++i) {
        const auto &track = tracks.at(static_cast<size_t>(i));
        const std::string desc = SubtitleTrackDescription(track);
        // Id = avStreamIndex + 1: nonzero (VDR's availability gate) and unique per stream.
        (void)DeviceSetAvailableTrack(ttSubtitle, i, static_cast<uint16_t>(track.avStreamIndex + 1),
                                      track.language.c_str(), desc.c_str());
    }
    subtitleSwitch.menuIndex.store(-1, std::memory_order_release);
    subtitleSwitch.trackCount.store(count, std::memory_order_release);
    // Force VDR's current subtitle track to ttNone so preferred-language / DisplaySubtitles handling
    // can't silently preselect one. Subtitles start off (like dvbplayer); the source already defaults
    // currentSubtitleTrack to -1 and the converter stays closed until the user selects a track.
    (void)DeviceSetCurrentSubtitleTrack(ttNone);
}

auto cVaapiPlayer::PerformAudioSwitch(int trackIdx) -> void {
    const cMutexLock lock(&sourceMutex);
    if (!source) {
        return;
    }
    auto *vaapiDev = FindPrimaryVaapiDevice();
    if (vaapiDev == nullptr) {
        return;
    }
    const int previous = source->CurrentAudioTrack();
    if (trackIdx == previous) {
        return; // already active (a duplicate request raced the no-op guard)
    }
    // Capture position BEFORE disturbing the clock; the re-anchor seeks back to it.
    const int64_t currentMs = CurrentPositionMs();

    // (a) repoint the demuxer to the new stream (no device / codec / seek side effects)
    if (!source->SelectAudioTrack(trackIdx)) {
        esyslog("vaapivideo/mediaplayer: audio track %d out of range -- keeping current", trackIdx);
        return;
    }
    // (b) reopen ONLY the audio codec for the new stream
    if (!vaapiDev->ReopenMediaPlayerAudio(source->AudioInfo())) {
        esyslog("vaapivideo/mediaplayer: audio reopen failed -- reverting to track %d", previous);
        if (source->SelectAudioTrack(previous)) {
            (void)vaapiDev->ReopenMediaPlayerAudio(source->AudioInfo()); // best-effort restore of the old codec
        }
        return;
    }
    // (c) re-anchor: seek to the captured position so the new stream flows from here and A/V resyncs.
    (void)SeekToMs(currentMs);
    audioSwitch.menuIndex.store(trackIdx, std::memory_order_release);
    isyslog("vaapivideo/mediaplayer: audio track -> %d", trackIdx);
}

auto cVaapiPlayer::PerformSubtitleSwitch(int trackIdx) -> void {
    const cMutexLock lock(&sourceMutex);
    if (!source) {
        return;
    }
    // The Subtitles chooser fires SetSubtitleTrack twice per pick (cycle + OK); skip the redundant
    // re-open (which would needlessly reopen the decoder and drop cues), like PerformAudioSwitch.
    const int previous = source->CurrentSubtitleTrack();
    if (trackIdx == previous) {
        return;
    }
    // Repoint demux routing (trackIdx < 0 = off). No A/V re-anchor: subtitles ride the same clock,
    // and cue display keys off the clock, so no seek/flush is needed.
    if (!source->SelectSubtitleTrack(trackIdx)) {
        esyslog("vaapivideo/mediaplayer: subtitle track %d out of range -- keeping current", trackIdx);
        subtitleSwitch.menuIndex.store(previous, std::memory_order_release); // keep the chooser state honest
        return;
    }
    if (!subtitles) {
        return;
    }
    if (trackIdx >= 0) {
        const auto &tracks = source->SubtitleTracks();
        const auto &track = tracks.at(static_cast<size_t>(trackIdx));
        const AVCodecParameters *codecpar = track.codecpar; // borrowed; converter deep-copies what it needs
        if (!subtitles->Open(codecpar)) {
            esyslog("vaapivideo/mediaplayer: subtitle decoder open failed -- turning subtitles off");
            (void)source->SelectSubtitleTrack(-1);
            subtitles->Close();
            subtitleSwitch.menuIndex.store(-1, std::memory_order_release); // let the user retry a different track
            (void)DeviceSetCurrentSubtitleTrack(ttNone);                   // push VDR's UI state back to off
            return;
        }
        isyslog("vaapivideo/mediaplayer: subtitle track -> %d", trackIdx);
    } else {
        subtitles->Close();
        isyslog("vaapivideo/mediaplayer: subtitles off");
    }
}

auto cVaapiPlayer::DrainTailAtEof() -> void {
    auto *vaapiDev = FindPrimaryVaapiDevice();
    if (vaapiDev == nullptr) {
        return;
    }
    // Bail the instant the user wants something else, so a held tail can't delay it. Pause matters
    // most: it stops the presenter, so depth freezes and the stall watchdog would otherwise advance
    // the playlist behind the user's back -- instead Action()'s pause branch holds and the drain
    // resumes when EOF is re-detected on Play. Any non-Play mode (pause, a freshly entered trick)
    // or a staged trick transition aborts for the same reason.
    const auto aborted = [this]() noexcept -> bool {
        return stopping.load(std::memory_order_acquire) || playMode.load(std::memory_order_acquire) != PlayMode::Play ||
               trickCommand.load(std::memory_order_acquire) != TrickCommand::None ||
               seekPending.load(std::memory_order_acquire) || nextRequested.load(std::memory_order_acquire) ||
               audioSwitch.pending.load(std::memory_order_acquire) ||
               subtitleSwitch.pending.load(std::memory_order_acquire);
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
            // Reset on ANY decrease vs the previous reading, not a lifetime low: the codec drain
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

    // Flush the reorder tail into the reserve (the decode thread defers that until its queue has
    // emptied), then drain everything to the screen at real-time pace.
    vaapiDev->RequestEosDrain();
    drainUntilEmpty([vaapiDev]() noexcept -> size_t { return vaapiDev->PendingPlayoutDepth(); });
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
    // Each iteration: service pending commands (seek/trick/next) -> wait if paused -> honor device
    // backpressure -> pull and dispatch one packet (or one reverse trick step). EOF advances the
    // playlist; EAGAIN sleeps.
    const std::unique_ptr<AVPacket, FreeAVPacket> packet{av_packet_alloc()};
    if (!packet) [[unlikely]] {
        esyslog("vaapivideo/mediaplayer: AVPacket allocation failed -- aborting");
        state.store(State::Stopped, std::memory_order_release);
        return;
    }

    // packetPending carries one read packet across iterations when the device's submit
    // queue is briefly full (audio cAudioProcessor::EnqueuePacket returns false on overflow).
    // Without this retry the lookahead throttle would advance latestAudioPts90k even though
    // the packet was silently dropped, breaking A/V sync subtly until the next forced flush.
    MediaPacketStream packetStream{MediaPacketStream::Video};
    bool packetPending = false;

    while (!stopping.load(std::memory_order_acquire)) {
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

        // -- trick transition ----------------------------------------------------
        // Staged by the trick state machine (EnterTrick/ExitTrick); one-shot like the seek branch.
        // Before the pause branch so an exit-to-pause still re-anchors while parked.
        if (const TrickCommand cmd = trickCommand.exchange(TrickCommand::None, std::memory_order_acq_rel);
            cmd != TrickCommand::None) {
            ioInterrupt.store(false, std::memory_order_release); // same rationale as the seek branch
            if (packetPending) { // a held packet belongs to the previous feed mode; the transition flushes
                av_packet_unref(packet.get());
                packetPending = false;
            }
            PerformTrickTransition(cmd);
        }

        // -- audio track switch --------------------------------------------------
        // Before the pause branch so a frozen player still switches (then plays the new track on resume).
        if (audioSwitch.pending.exchange(false, std::memory_order_acq_rel)) {
            ioInterrupt.store(false, std::memory_order_release); // av_seek_frame shares the callback
            if (packetPending) { // held packet belongs to the old stream; PerformAudioSwitch flushes
                av_packet_unref(packet.get());
                packetPending = false;
            }
            if (const int target = audioSwitch.targetIdx.exchange(-1, std::memory_order_relaxed); target >= 0) {
                PerformAudioSwitch(target);
            }
        }

        // -- subtitle track switch ----------------------------------------------
        // Before the pause branch (like the audio switch) so a frozen player still toggles subtitles.
        // The pending flag gates the target read: -1 is a valid target here (subtitles off).
        if (subtitleSwitch.pending.exchange(false, std::memory_order_acq_rel)) {
            // SetSubtitleTrack() set ioInterrupt to break a parked read; clear it like the audio/seek
            // paths or the next av_read_frame aborts. The held packet stays valid (no flush/seek here).
            ioInterrupt.store(false, std::memory_order_release);
            PerformSubtitleSwitch(subtitleSwitch.targetIdx.load(std::memory_order_acquire));
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
        if (playMode.load(std::memory_order_acquire) == PlayMode::Pause) {
            const cMutexLock lock(&pauseMutex);
            if (playMode.load(std::memory_order_acquire) == PlayMode::Pause &&
                !stopping.load(std::memory_order_acquire) && !seekPending.load(std::memory_order_acquire) &&
                trickCommand.load(std::memory_order_acquire) == TrickCommand::None &&
                !nextRequested.load(std::memory_order_acquire) &&
                !audioSwitch.pending.load(std::memory_order_acquire) &&
                !subtitleSwitch.pending.load(std::memory_order_acquire)) {
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
        // Feed-mode snapshot: dvbplayer-style trick modes replace the normal audio-clock-paced pump.
        const PlayMode mode = playMode.load(std::memory_order_acquire);
        const bool trickActive = mode == PlayMode::Fast || mode == PlayMode::Slow;

        // A staged-but-unconsumed Enter* command means this mode's feed state (reverse step
        // targets, aligned cursor) is not initialized yet: the command can land after this
        // iteration's consume point but before the mode snapshot. Let the loop top take it first.
        if (trickActive && trickCommand.load(std::memory_order_acquire) != TrickCommand::None) {
            continue;
        }

        // -- reverse trick feed: one keyframe step per iteration -------------------
        if (trickActive && !trickForward.load(std::memory_order_acquire)) {
            if (packetPending) { // left over from another feed mode; reverse re-reads after seeking
                av_packet_unref(packet.get());
                packetPending = false;
            }
            // Gate BEFORE reading: the trick queue is 1 deep and drops overflow, so a keyframe must
            // only be demuxed once the decoder is ready to take it.
            if (!vaapiDev->IsMediaPlayerTrickReady()) {
                cCondWait::SleepMs(MEDIAPLAYER_BACKPRESSURE_SLEEP_MS);
                continue;
            }
            if (!PerformReverseStep(vaapiDev, packet.get())) {
                cCondWait::SleepMs(DEMUX_IDLE_SLEEP_MS);
            }
            continue;
        }

        // Backpressure gates NEW demux reads only. A packetPending has already advanced
        // libavformat's cursor; retry it even while queues are high. Otherwise a held audio
        // packet can be blocked by the very audioHighwater condition that submitting it
        // would help clear (the SubmitAudioPacket highwater check is the proper pacing
        // signal -- false return -> packetPending stays true -> retry next iter).
        // Trick modes skip this gate: it measures the normal-replay queues; the trick feed
        // paces on IsMediaPlayerTrickReady() at the submit site instead.
        if (!packetPending && !trickActive && vaapiDev->IsMediaPlayerBackpressured()) {
            cCondWait::SleepMs(MEDIAPLAYER_BACKPRESSURE_SLEEP_MS);
            continue;
        }

        // -- read and dispatch one packet in demux order --------------------------
        // AdvancePlaylist() re-locks sourceMutex via Close/OpenCurrentEntry and can block on
        // container I/O, so defer it past the lock scope (invariant 1).
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

        if (packetPending && packetStream == MediaPacketStream::Subtitle) {
            // Subtitles never touch the device queues or the lookahead/backpressure throttle: hand the
            // cue to the converter and consume the packet unconditionally (tiny, sparse). During trick
            // the cue is dropped -- there is no meaningful display timing at trick pace.
            if (subtitles && !trickActive) {
                subtitles->Convert(packet.get());
            }
            av_packet_unref(packet.get());
            packetPending = false;
            didWork = true;
        } else if (packetPending && trickActive && packetStream == MediaPacketStream::Audio) {
            // No audio in trick modes: never submitted (nothing may play, and the lookahead
            // reference must not advance) -- the device-side trick swallow is only the backstop.
            av_packet_unref(packet.get());
            packetPending = false;
            didWork = true;
        } else if (packetPending && trickActive) {
            // Forward trick video. Slow motion decodes every frame; fast-forward wants keyframes
            // only -- non-key packets go through ungated so the decoder's FF filter (the single
            // authority, cf. cVaapiDecoder::EnqueuePacket) drops them at demux skim speed. Paced
            // packets gate on the trick queue (1 deep, drops overflow) via IsMediaPlayerTrickReady().
            const bool paced = mode == PlayMode::Slow || (packet->flags & AV_PKT_FLAG_KEY) != 0;
            if (paced && !vaapiDev->IsMediaPlayerTrickReady()) {
                cCondWait::SleepMs(MEDIAPLAYER_BACKPRESSURE_SLEEP_MS); // retry the held packet
            } else if (vaapiDev->SubmitVideoPacket(packet.get())) {
                av_packet_unref(packet.get());
                packetPending = false;
                didWork = true;
            }
            // !submitted: keep packetPending=true and retry, like the normal path below.
        } else if (packetPending) {
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
            if (trickActive) {
                // A forward trick ran into EOF: leave the trick first so the tail drain and the
                // next playlist entry run at normal speed (dvbplayer ends fast-forward the same way).
                EndTrick(PlayMode::Play);
                dsyslog("vaapivideo/mediaplayer: trick reached EOF -- resuming play");
            }
            // Natural EOF: present the buffered tail before teardown so playback runs to the real end.
            DrainTailAtEof();
            if (stopping.load(std::memory_order_acquire)) {
                break;
            }
            // The drain bailed on a user command: let the loop top service it instead of advancing
            // (pause holds; a seek resets eofReached; the tail re-drains once EOF is hit again).
            if (playMode.load(std::memory_order_acquire) != PlayMode::Play ||
                trickCommand.load(std::memory_order_acquire) != TrickCommand::None ||
                seekPending.load(std::memory_order_acquire) || nextRequested.load(std::memory_order_acquire) ||
                audioSwitch.pending.load(std::memory_order_acquire) ||
                subtitleSwitch.pending.load(std::memory_order_acquire)) {
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
    dsyslog("vaapivideo/mediaplayer: control launched");
}

cVaapiControl::~cVaapiControl() noexcept {
    // The single save hook, covering every teardown path (Stop, EOF, channel switch, shutdown, SVDRP
    // replace). Runs while the player (device attachment / STC) is still alive.
    if (player) {
        PersistBookmark(player->MakeBookmark());
    }
    HideReplayBar();
    cStatus::MsgReplaying(this, nullptr, nullptr, false);
    // Null the base alias BEFORE deleting the player (cf. cDvbPlayerControl::Stop in
    // vdr/dvbplayer.c). The unique_ptr owns/deletes; cControl does not.
    cControl::player = nullptr;
    player.reset();
    dsyslog("vaapivideo/mediaplayer: control destroyed");
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
    // The play/trick state drives the skin's mode symbols ("1>>", "<|1", ...), cf. cReplayControl.
    bool play = true;
    bool forward = true;
    int speed = -1;
    (void)player->GetReplayMode(play, forward, speed);
    displayReplay->SetMode(play, forward, speed);
    displayReplay->Flush();
    lastBarRefresh.Set();
}

[[nodiscard]] auto cVaapiControl::HandleSeekKey(const char *label, int deltaMs) -> eOSState {
    dsyslog("vaapivideo/mediaplayer: key %s -- seek %+dms", label, deltaMs);
    // Jumps leave trick mode first (cDvbPlayer::SkipSeconds semantics); the jump's own seek
    // re-anchors, so the Exit re-anchor would only double the flush.
    player->LeaveTrickWithoutReanchor();
    player->Seek(deltaMs);
    ShowReplayBar();
    return osContinue;
}

[[nodiscard]] auto cVaapiControl::HandleTrickKey(eKeys key, bool forward) -> eOSState {
    // Discrete presses only: our dispatch masks k_Repeat (VDR's menu.c matches raw values, where
    // repeats fall to default), so a held key must not step a notch per repeat event.
    if ((key & k_Repeat) != 0) {
        return osContinue;
    }
    // Release semantics follow menu.c: in single-speed mode the release ends a hold-to-scan
    // (Forward()/Backward() from an active same-direction trick resumes play); multi-speed
    // ignores releases -- each press steps one notch.
    if ((key & k_Release) != 0 && Setup.MultiSpeedMode != 0) {
        return osContinue;
    }
    dsyslog("vaapivideo/mediaplayer: key %s", forward ? "FastFwd" : "FastRew");
    if (forward) {
        player->Forward();
    } else {
        player->Backward();
    }
    ShowReplayBar();
    return osContinue;
}

[[nodiscard]] auto cVaapiControl::ProcessKey(eKeys Key) -> eOSState {
    // Key bindings (plugin spec):
    //   OK              toggle replay bar
    //   Play / Up       resume normal playback (from pause or any trick mode)
    //   Pause / Down    toggle pause; exits a trick mode into pause
    //   FastFwd/FastRew trick play (dvbplayer-style): fast fwd/rew from play, slow motion from
    //                   pause; repeated presses cycle the speed notches (Setup.MultiSpeedMode)
    //   Left  / Right   short seek (-/+ 10 s); exits a trick mode first
    //   Green / Yellow  long  seek (-/+ 60 s); exits a trick mode first
    //   Blue            cycle manual zoom (Off -> 1 -> .. -> N -> Off)
    //   Next            advance playlist
    //   Back / Stop     exit
    if (!player) {
        return osEnd;
    }
    if (player->IsFinished()) {
        // EOF / failed open: reopen the browser instead of dropping to live TV, like Stop below.
        RequestBrowserReopen();
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

    if (Key == kPlayPause) {
        // Combined-key normalization like cReplayControl::ProcessKey(): inside a trick mode the
        // "matching" half leaves the trick; in normal modes it toggles play/pause.
        bool play = false;
        bool forward = false;
        int speed = -1;
        (void)player->GetReplayMode(play, forward, speed);
        if (speed >= 0) {
            Key = play ? kPlay : kPause; // trick mode: the matching half leaves it
        } else {
            Key = play ? kPause : kPlay; // normal: toggle
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
            dsyslog("vaapivideo/mediaplayer: key Play/Up -- resume normal playback");
            player->Play();
            ShowReplayBar();
            return osContinue;

        case kPause:
        case kDown:
            dsyslog("vaapivideo/mediaplayer: key Pause/Down -- toggle (was %s)",
                    player->IsPaused() ? "paused" : "playing");
            player->Pause();
            ShowReplayBar();
            return osContinue;

        case kFastFwd:
        case kFastFwd | k_Release:
            return HandleTrickKey(Key, true);
        case kFastRew:
        case kFastRew | k_Release:
            return HandleTrickKey(Key, false);

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
            // Reopen the browser (cursor on the bookmark, set by the dtor) instead of live TV.
            dsyslog("vaapivideo/mediaplayer: key Back/Stop -- return to file browser");
            RequestBrowserReopen();
            return osEnd;

        default:
            return osContinue;
    }
}

// ============================================================================
// === cVaapiFileBrowser ===
// ============================================================================

cVaapiFileBrowser::cVaapiFileBrowser(std::string startDir) : cOsdMenu("") {
    if (startDir.empty()) {
        startDir = "/";
    }
    // Open on the bookmark (parent dir, cursor on the file); a URL / deleted file / gone m3u entry
    // falls back to the start folder. Read here so every browser open lands on the bookmark.
    if (const std::string mark = NormalizeBookmarkUri(LoadBookmark().uri); !mark.empty() && !HasUrlScheme(mark)) {
        std::error_code ec;
        const std::string parent = Dirname(mark);
        if (std::filesystem::is_regular_file(mark, ec) && !ec && std::filesystem::is_directory(parent, ec) && !ec) {
            LoadDirectory(parent);
            if (SelectEntryByName(Basename(mark))) {
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
                case EntryKind::Playlist:
                case EntryKind::File:
                    // StartPlayback expands a .m3u itself, so both kinds share one path.
                    switch (StartPlayback(PlaylistEntry{.uri = fullPath, .title = entry->name})) {
                        case StartPlaybackResult::Started:
                            return osEnd;
                        case StartPlaybackResult::EmptyPlaylist:
                            Skins.Message(mtError, tr("Empty or unreadable playlist"));
                            return osContinue;
                        case StartPlaybackResult::DeviceNotReady:
                            Skins.Message(mtError, tr("Cannot start playback"));
                            return osContinue;
                    }
                    return osContinue;
            }
            return osContinue;
        }
        default:
            return osContinue;
    }
}
