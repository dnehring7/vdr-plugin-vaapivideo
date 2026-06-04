// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file mediaplayer.h
 * @brief Integrated media player: libavformat demux + cVaapiDevice feed surface.
 *
 * Plays one of:
 *   - a local file:  /path/to/video.mp4
 *   - a remote URL:  http(s)/ftp via libavformat protocols
 *   - an m3u(8) playlist: lines are dispatched as the above when reached
 *
 * Reuses the existing VAAPI decoder, ALSA audio processor, and DRM display unchanged.
 * Demux runs on a private cThread; A/V sync uses the existing audio-master clock,
 * so pause/seek route through cDevice::Freeze() and cVaapiDevice::ClearForMediaPlayer().
 *
 * Entry points:
 *   - main menu     -> cVaapiQuickMenu -> cVaapiFileBrowser
 *   - replay end    -> MainMenuAction reopens cVaapiFileBrowser via pending return state
 *   - SVDRP PLAY    -> StartPlayback(...) launches cVaapiControl
 *
 * Threading: the demux thread is the only writer for the source FIFO; the player
 * thread reads from the source and pushes packets to the device. The control
 * runs on the VDR main thread and does not share state with the player except
 * through atomics and the player's public command methods.
 */

#ifndef VDR_VAAPIVIDEO_MEDIAPLAYER_H
#define VDR_VAAPIVIDEO_MEDIAPLAYER_H

#include "common.h"
#include "stream.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// VDR
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <vdr/osdbase.h>
#include <vdr/player.h>
#include <vdr/skins.h>
#include <vdr/thread.h>
#include <vdr/tools.h>
#pragma GCC diagnostic pop

class cVaapiDevice;

// ============================================================================
// === CONSTANTS ===
// ============================================================================

/// Demux thread back-off when the device queues are full or input stalls.
inline constexpr int MEDIAPLAYER_BACKPRESSURE_SLEEP_MS = 5;

/// Audio packet-queue high-watermark for demux pacing. Matches VDR replay's AUDIO_REPLAY_QUEUE_
/// HIGHWATER (cDvbPlayer in vdr/dvbplayer.c). The PTS-distance lookahead (MEDIAPLAYER_MAX_LOOKAHEAD_90K)
/// is a coarse brake: it lets the demuxer push up to 1 s of audio time ahead of the master clock, and
/// on TS containers where the video PTS leads the audio PTS at the same demux cursor by up to ~2 s
/// (mux interleave offset) that fills the decoder's 150-frame jitterBuf and triggers the overflow
/// loop you can see in the syslog after a seek burst. Capping the audio packet queue at this
/// low-water depth paces the demuxer to the audio thread's consumption rate (~1 x playback rate),
/// shrinking both audio packet depth and the co-buffered video lead. Audio resilience to demux
/// stalls = HIGHWATER * audio_packet_ms + ALSA buffer (~400 ms): for AC3 32 ms/packet that's
/// ~720 ms total, comfortably above any normal demux hiccup. The AUDIO_QUEUE_CAPACITY (300) hard
/// cap inside cAudioProcessor remains as an overflow backstop for SubmitAudioPacket return value.
inline constexpr size_t MEDIAPLAYER_AUDIO_QUEUE_HIGHWATER = 10;

/// Max AUDIO lookahead, in 90 kHz ticks, between the latest pushed audio PTS and the audio
/// master clock before the demux throttles. libavformat reads local files much faster than
/// wall-clock; without a real-time pacing limit the decoder packet queue and the downstream
/// jitter buffer overrun their caps. The throttle reference is the audio-stream PTS only
/// (see latestAudioPts90k), so this bound is an audio-tail budget. The actual VIDEO buffer
/// depth is (audio_tail + mux_offset) where mux_offset is the per-file inter-stream PTS
/// spread at the same demux cursor -- ~0 for tight MP4/MKV, but up to ~1 s for broadcast
/// TS (video PTS leads audio PTS). At 50 fps:
///   - 2 s audio tail + 1 s TS mux offset = 150 frames = exactly the decoder's 150-frame
///     hard cap -> jitterBuf hits the cap -> drop oldest -> drain stall / re-arm-freerun loop
///     after seek or pause on TS, the bug this constant exists to prevent.
///   - 1 s audio tail + 1 s TS mux offset = 100 frames -- under the cap with ~50 frames
///     headroom for transient excursions, and well below FUTURE_MAX (3 s).
/// 1 s gives the demuxer enough buffered audio to absorb wall-clock jitter while keeping the
/// resulting video depth safely below the hard cap on every container we have seen.
inline constexpr int64_t MEDIAPLAYER_MAX_LOOKAHEAD_90K = 90000;

/// Jitter-buffer high-water mark that triggers backpressure on the mediaplayer demux
/// path -- but ONLY while the audio master clock is NOPTS (the post-flush window where
/// ALSA refills before the first decoded sample anchors the clock).
/// The lookahead-vs-clock throttle is blind in that window, so without a buffer-depth
/// signal the demuxer pushes packets at file-read speed and the jitterBuf overflows past
/// its 150-frame hard cap. Once the clock anchors, IsMediaPlayerBackpressured() drops this
/// gate entirely: the demuxer is a single-cursor demux-order pump, so a video-depth gate
/// there would stall the shared cursor and starve the audio packet queue, underrunning ALSA
/// and stalling the very clock the video drain depends on (sustained-stutter-after-seek bug).
/// With a live clock the lookahead throttle is the sole gate and the 150-frame hard cap is the
/// overflow backstop. 100 frames = ~2 s @ 50 fps -- above the steady-state video depth with the
/// 1 s audio-tail lookahead (50 frames + per-file mux offset, up to ~100 on TS) but well below
/// the hard cap, so this only catches the pre-anchor blind window where the lookahead can't gate.
inline constexpr size_t MEDIAPLAYER_JITTERBUF_BACKPRESSURE_FRAMES = 100;

/// Default seek deltas applied by the key bindings (milliseconds).
inline constexpr int MEDIAPLAYER_SEEK_SHORT_MS = 10000;
inline constexpr int MEDIAPLAYER_SEEK_LONG_MS = 60000;

/// End-of-stream tail drain (cVaapiPlayer::DrainTailAtEof). At EOF the decode queue (~4 s @ 50 fps)
/// and present reserve (~3 s) still hold unseen frames; immediate teardown cuts playback seconds
/// short -- worst on video-only clips, where no audio clock throttles the demuxer so both buffers
/// fill to their caps. The drain flushes that tail at real-time pace first.
///   - TIMEOUT_MS: backstop so a wedged pipeline can't hang shutdown; covers the ~7 s worst case.
///   - STALL_MS: bail when depth stops shrinking (wedged pipeline). Must exceed one frame interval.
inline constexpr int MEDIAPLAYER_EOF_DRAIN_TIMEOUT_MS = 20000;
inline constexpr int MEDIAPLAYER_EOF_DRAIN_STALL_MS = 1500;

// ============================================================================
// === PLAYLIST HELPERS ===
// ============================================================================

/// One playlist row. @c uri is the absolute path (or URL); @c title is the
/// display label (defaults to the basename of @c uri).
struct PlaylistEntry {
    std::string uri;
    std::string title;
};

/// Parse an m3u / m3u8 file. Honors @c #EXTINF:duration,title rows; falls back
/// to the basename when no title is given. Relative paths are resolved against
/// the playlist's parent directory. Comment / empty lines are skipped.
/// Returns an empty vector on read error.
[[nodiscard]] auto ParseM3U(std::string_view playlistPath) -> std::vector<PlaylistEntry>;

/// True for the small extension whitelist we expose in the file browser, or for
/// http(s)/ftp prefixes. Case-insensitive.
[[nodiscard]] auto IsMediaUri(std::string_view path) noexcept -> bool;

/// True for @c .m3u / @c .m3u8 paths (case-insensitive).
[[nodiscard]] auto IsPlaylistUri(std::string_view path) noexcept -> bool;

/// Launch playback. @p entries.size()==1 plays a single file/URL; >1 is a playlist.
/// Wraps cControl::Launch(); does not block. Returns false iff there is no primary
/// vaapivideo device to attach to.
auto StartPlayback(std::vector<PlaylistEntry> entries) -> bool;

/// Ask the file browser to reopen (instead of live TV) on VDR's next main-menu hook, cursor on
/// @p jumpToPath -- or the media-dir root when that path is a URL/unreachable. Wraps
/// cRemote::CallPlugin(); call before returning osEnd. VDR main thread only.
auto RequestReturnToBrowser(std::string jumpToPath) -> void;

/// Consume a pending RequestReturnToBrowser(): the jump-to path if one was set (and clear it), else
/// std::nullopt. MainMenuAction() uses it to choose the browser over the quick menu.
[[nodiscard]] auto TakeReturnToBrowser() -> std::optional<std::string>;

// ============================================================================
// === MEDIA SOURCE ===
// ============================================================================

/// libavformat-based input. One source per uri; playlist advancement constructs a
/// new source on Open(). PTS values returned to callers are already rebased to
/// the VDR 90 kHz domain (relative to the source's first packet).
class cVaapiMediaSource final : public IMediaSource {
  public:
    /// @p stop (optional, non-owning) is polled by libavformat's interrupt callback so
    /// network-backed avformat_open_input() / av_read_frame() exit promptly when the player
    /// is shutting down. @p interrupt (optional, non-owning) does the same for a pending
    /// seek/next so those commands don't wait out a network read timeout. Pass
    /// &cVaapiPlayer::stopping and &cVaapiPlayer::ioInterrupt. Pointers are non-const because
    /// interrupt_callback.opaque is void*; the callback only reads via std::atomic::load().
    explicit cVaapiMediaSource(std::atomic<bool> *stop = nullptr, std::atomic<bool> *interrupt = nullptr) noexcept
        : stopFlag(stop), interruptFlag(interrupt) {}
    ~cVaapiMediaSource() noexcept override;
    cVaapiMediaSource(const cVaapiMediaSource &) = delete;
    cVaapiMediaSource(cVaapiMediaSource &&) noexcept = delete;
    auto operator=(const cVaapiMediaSource &) -> cVaapiMediaSource & = delete;
    auto operator=(cVaapiMediaSource &&) noexcept -> cVaapiMediaSource & = delete;

    // ========================================================================
    // === PUBLIC API ===
    // ========================================================================
    [[nodiscard]] auto Open(std::string_view uri) -> bool; ///< avformat_open_input + stream selection + extradata copy
    auto Close() noexcept -> void;                         ///< Idempotent; releases AVFormatContext

    // IMediaSource overrides
    [[nodiscard]] auto ReadPacket(AVPacket *out, MediaPacketStream &stream) -> int override;
    [[nodiscard]] auto VideoInfo() const noexcept -> const VideoStreamInfo & override { return videoInfo; }
    [[nodiscard]] auto AudioInfo() const noexcept -> const AudioStreamInfo & override { return audioInfo; }
    auto Flush() -> void override; ///< avformat_flush (post-seek)

    [[nodiscard]] auto Seek(int64_t targetPts90k) -> bool; ///< av_seek_frame to nearest keyframe at/below target
    [[nodiscard]] auto DurationMs() const noexcept -> int; ///< 0 when unknown (live streams)
    [[nodiscard]] auto IoInterrupted() const noexcept
        -> bool; ///< True if a blocking libavformat I/O should bail (shutdown via stopFlag, or a
                 ///< pending seek/next via interruptFlag). Polled by the interrupt_callback.
    [[nodiscard]] auto HasAudio() const noexcept -> bool { return audioStreamIndex >= 0; }
    [[nodiscard]] auto HasVideo() const noexcept -> bool { return videoStreamIndex >= 0; }
    [[nodiscard]] auto VideoFps() const noexcept -> double {
        return videoFps;
    } ///< Container-reported avg_frame_rate; 0.0 if unknown.
    [[nodiscard]] auto VideoCodedSize() const noexcept -> std::pair<int, int> {
        return {videoInfo.codedWidth, videoInfo.codedHeight};
    }

  private:
    auto PopulateStreamInfo() -> void;
    /// Rescale @p ts from @p tb to 90 kHz and subtract the source's PTS origin.
    /// Lazily seeds @c ptsOrigin90k on first call so files whose container/streams
    /// don't advertise start_time still emit a zero-based timeline.
    [[nodiscard]] auto Rebase90k(int64_t ts, AVRational tb) noexcept -> int64_t;

    std::unique_ptr<AVFormatContext, FreeAVFormatContext> formatCtx;
    std::atomic<bool> *stopFlag{nullptr};      ///< Non-owning; shutdown signal polled by the interrupt_callback.
    std::atomic<bool> *interruptFlag{nullptr}; ///< Non-owning; seek/next signal polled by the interrupt_callback.
    int videoStreamIndex{-1};
    int audioStreamIndex{-1};
    AVRational videoTimeBase{.num = 1, .den = 90000};
    AVRational audioTimeBase{.num = 1, .den = 90000};
    int64_t ptsOrigin90k{AV_NOPTS_VALUE};          ///< First-packet PTS in 90 kHz (or formatCtx->start_time
                                                   ///< when known); subtracted from every emitted PTS so the
                                                   ///< replay bar and Seek() math share a zero-based timeline.
    int64_t discardAudioBefore90k{AV_NOPTS_VALUE}; ///< Post-seek guard: armed to the seek target, drops
                                                   ///< audio packets earlier than that so the master
                                                   ///< clock anchors at the requested timeline (and not
                                                   ///< at the earlier video keyframe libavformat lands on).
    VideoStreamInfo videoInfo;
    AudioStreamInfo audioInfo;
    std::vector<uint8_t> videoExtradataStorage;
    std::vector<uint8_t> audioExtradataStorage;
    double videoFps{0.0}; ///< Snapshot of avg_frame_rate at Open(); 0.0 if not advertised.
    bool eofReached{false};
};

// ============================================================================
// === PLAYER ===
// ============================================================================

/// VDR cPlayer + private demux cThread. Owns the cVaapiMediaSource and walks the
/// playlist. Action() runs the packet pump: pull one video + one audio packet
/// per iteration, submit to the device, throttle on IsMediaPlayerBackpressured().
// NOLINTNEXTLINE(misc-multiple-inheritance) -- standard VDR pattern; cf. cDvbPlayer in VDR core.
class cVaapiPlayer final : public cPlayer, public cThread {
  public:
    enum class State : uint8_t { Opening, Playing, Paused, Seeking, Eof, Stopped };

    explicit cVaapiPlayer(std::vector<PlaylistEntry> entries);
    ~cVaapiPlayer() noexcept override;
    cVaapiPlayer(const cVaapiPlayer &) = delete;
    cVaapiPlayer(cVaapiPlayer &&) noexcept = delete;
    auto operator=(const cVaapiPlayer &) -> cVaapiPlayer & = delete;
    auto operator=(cVaapiPlayer &&) noexcept -> cVaapiPlayer & = delete;

    // ========================================================================
    // === PUBLIC API (called from cVaapiControl on the VDR main thread) ===
    // ========================================================================
    auto SetPaused(bool paused) -> void;
    [[nodiscard]] auto IsPaused() const noexcept -> bool { return paused.load(std::memory_order_acquire); }
    auto Seek(int64_t deltaMs) -> void; ///< Relative seek; deltaMs may be negative
    auto Next() -> void;                ///< Skip to next playlist entry, if any
    [[nodiscard]] auto Title() const -> std::string;
    /// URI of the current entry (clamped to the last on EOF, so Stop after a file ends still
    /// resolves to it). Empty only for an empty playlist. Used by cVaapiControl to reopen the browser.
    [[nodiscard]] auto CurrentUri() const -> std::string;
    /// True once playback can no longer continue: natural EOF, fatal open failure, or shutdown.
    /// cVaapiControl uses this to exit on its next key event.
    [[nodiscard]] auto IsFinished() const noexcept -> bool {
        const State current = state.load(std::memory_order_acquire);
        return current == State::Eof || current == State::Stopped;
    }

    // ========================================================================
    // === cPlayer overrides (public) ===
    // ========================================================================
    [[nodiscard]] auto GetReplayMode(bool &Play, bool &Forward, int &Speed) -> bool override;
    [[nodiscard]] auto GetIndex(int &Current, int &Total, bool SnapToIFrame = false) -> bool override;
    [[nodiscard]] auto FramesPerSecond() -> double override;
    [[nodiscard]] auto InfoText() const -> std::string; ///< Multi-line file metadata for cControl::GetInfo().

  protected:
    // ========================================================================
    // === cPlayer overrides (protected) ===
    // ========================================================================
    auto Activate(bool On) -> void override; ///< Visibility intentionally matches cPlayer base (protected).

    // ========================================================================
    // === cThread overrides ===
    // ========================================================================
    auto Action() -> void override;

  private:
    [[nodiscard]] auto OpenCurrentEntry() -> bool;
    auto CloseCurrentEntry() noexcept -> void;
    /// Current playback position in milliseconds from the device's audio-mastered STC.
    /// Returns 0 when no vaapivideo device is attached or the audio clock has not anchored.
    [[nodiscard]] auto CurrentPositionMs() const noexcept -> int;
    /// Demux-thread lookahead in 90 kHz ticks: how far ahead of the audio master clock the
    /// most recently submitted packet is. Returns AV_NOPTS_VALUE when either side is unanchored
    /// (startup, post-seek, no device) -- callers must skip the comparison in that case.
    [[nodiscard]] auto Lookahead90k(const cVaapiDevice *vaapiDev) const noexcept -> int64_t;
    auto PerformSeek(int64_t deltaMs) -> void;
    /// Block until the decode + present pipeline has flushed the buffered end-of-stream tail to the
    /// screen, so a natural EOF does not cut playback short. Returns early if the user issues
    /// stop / pause / seek / next during the wait, or if the pipeline stalls. Demux thread only;
    /// called before AdvancePlaylist() tears the entry down. See MEDIAPLAYER_EOF_DRAIN_* .
    auto DrainTailAtEof() -> void;
    auto AdvancePlaylist() -> void;
    auto LogStatus() -> void; ///< Periodic mediaplayer dsyslog (state + position + lookahead).

    std::vector<PlaylistEntry> playlist;
    std::atomic<size_t> currentIndex{0};

    std::unique_ptr<cVaapiMediaSource> source;
    std::atomic<State> state{State::Opening};
    std::atomic<bool> paused{false};
    std::atomic<bool> seekPending{false};
    std::atomic<int64_t> seekDeltaMs{0};
    std::atomic<bool> nextRequested{false};
    std::atomic<bool> stopping{false};
    std::atomic<bool> ioInterrupt{false}; ///< Set by Seek()/Next() (and shutdown) to break a blocking
                                          ///< av_read_frame()/av_seek_frame() on a slow network URL so the
                                          ///< command is serviced promptly instead of after the I/O timeout.
                                          ///< Polled via cVaapiMediaSource::IoInterrupted(); cleared in ReadPacket.
    std::atomic<int64_t> latestAudioPts90k{
        AV_NOPTS_VALUE};                      ///< Max AUDIO packet PTS submitted (90 kHz); drives the
                                              ///< lookahead-vs-audio-clock throttle in Action(). Audio
                                              ///< only: keying off the max of both streams lets a TS's
                                              ///< video-leads-audio mux PTS offset inflate the lookahead
                                              ///< and starve the audio queue. AV_NOPTS_VALUE for video-only.
    std::atomic<int> pendingSeekTargetMs{-1}; ///< Most recent seek target (ms). Used by CurrentPositionMs() as a
                                              ///< fallback while GetSTC() is still NOPTS in the ~50 ms window between
                                              ///< Clear() and the first decoded frame at the new position. Without
                                              ///< this, rapid follow-up Seek()s read position 0 and the playhead
                                              ///< snaps to the file start.

    mutable cMutex sourceMutex; ///< Guards source-pointer swaps across Action() and command methods
    cCondVar pauseCondition;    ///< Wakes Action() out of pause loop
    mutable cMutex pauseMutex;
    cTimeMs lastStatusLog; ///< Throttles LogStatus() to its periodic cadence.
};

// ============================================================================
// === CONTROL ===
// ============================================================================

/// Replay control: owns the player, draws the OSD replay bar, dispatches key events.
/// The base cControl stores only VDR's borrowed cPlayer pointer; it does not delete it
/// (cf. cControl::~cControl in vdr/player.c). Replay-control subclasses own/destroy the
/// player themselves -- same convention as cDvbPlayerControl::Stop() in vdr/dvbplayer.c.
/// Lifetime: created by cControl::Launch(), destroyed by VDR when ProcessKey() returns
/// osEnd or after Stop() (kBlue / kBack / kStop).
class cVaapiControl final : public cControl {
  public:
    explicit cVaapiControl(std::vector<PlaylistEntry> entries) : cVaapiControl(new cVaapiPlayer(std::move(entries))) {}
    ~cVaapiControl() noexcept override;
    cVaapiControl(const cVaapiControl &) = delete;
    cVaapiControl(cVaapiControl &&) noexcept = delete;
    auto operator=(const cVaapiControl &) -> cVaapiControl & = delete;
    auto operator=(cVaapiControl &&) noexcept -> cVaapiControl & = delete;

    auto Hide() -> void override;
    [[nodiscard]] auto ProcessKey(eKeys Key) -> eOSState override;
    [[nodiscard]] auto GetHeader() -> cString override;
    [[nodiscard]] auto GetInfo() -> cOsdObject * override; ///< File metadata dialog shown via kInfo.

  private:
    /// Delegating ctor: takes the typed pointer once and hands it to cControl as cPlayer*
    /// (upcast) and to @c player as cVaapiPlayer*. Avoids a static_cast downcast off
    /// cControl::player just to re-acquire the type we already had.
    explicit cVaapiControl(cVaapiPlayer *typedPlayer);
    auto ShowReplayBar() -> void;
    auto HideReplayBar() -> void;
    auto RefreshReplayBar() -> void;
    /// Handle a seek key: log, dispatch the relative seek, and pop up the replay bar so the
    /// user gets immediate visual feedback. @p label is the key name for the log line.
    [[nodiscard]] auto HandleSeekKey(const char *label, int deltaMs) -> eOSState;

    std::unique_ptr<cVaapiPlayer> player; ///< Sole owner. cControl::player is VDR's borrowed alias of the same pointer.
    cSkinDisplayReplay *displayReplay{nullptr};
    cTimeMs barTimeout;
    cTimeMs lastBarRefresh;
    bool barVisible{false};
};

// ============================================================================
// === FILE BROWSER ===
// ============================================================================

/// Directory browser. Lists subdirectories first, then media files and .m3u
/// playlists. kOk enters a directory or launches playback. kBack pops to parent.
class cVaapiFileBrowser final : public cOsdMenu {
  public:
    /// @p startDir is the root / fallback (the -m media-dir). When @p selectPath is a reachable local
    /// file, open its parent dir with the cursor on it; otherwise fall back to @p startDir.
    explicit cVaapiFileBrowser(std::string startDir, const std::string &selectPath = {});
    ~cVaapiFileBrowser() noexcept override = default;
    cVaapiFileBrowser(const cVaapiFileBrowser &) = delete;
    cVaapiFileBrowser(cVaapiFileBrowser &&) noexcept = delete;
    auto operator=(const cVaapiFileBrowser &) -> cVaapiFileBrowser & = delete;
    auto operator=(cVaapiFileBrowser &&) noexcept -> cVaapiFileBrowser & = delete;

    [[nodiscard]] auto ProcessKey(eKeys Key) -> eOSState override;

  private:
    enum class EntryKind : uint8_t { Parent, Directory, File, Playlist };
    struct BrowserEntry {
        EntryKind kind;
        std::string name;       ///< Display name (basename only)
        std::uintmax_t size{0}; ///< File size in bytes; 0 / unused for Parent and Directory
    };

    auto LoadDirectory(const std::string &dir) -> void;
    /// Move the cursor to the entry whose basename matches @p name and redraw; false if none match.
    [[nodiscard]] auto SelectEntryByName(std::string_view name) -> bool;
    [[nodiscard]] auto SelectedEntry() const -> const BrowserEntry *;
    [[nodiscard]] auto BuildFullPath(const BrowserEntry &entry) const -> std::string;

    std::string currentDir;
    std::vector<BrowserEntry> entries;
};

#endif // VDR_VAAPIVIDEO_MEDIAPLAYER_H
