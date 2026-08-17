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
 *   - replay stop   -> MainMenuAction reopens cVaapiFileBrowser (cursor on the persistent bookmark)
 *   - SVDRP PLAY    -> StartPlayback(...) launches cVaapiControl
 *
 * Resume: a single bookmark (origin URI + position) persists in setup.conf, staged at control
 * teardown and flushed inline on the main thread or via Housekeeping() for SVDRP-thread teardown.
 * Starting the bookmarked local file resumes at the saved position; the browser opens with
 * the cursor on it. Playlists and non-local URLs bookmark the origin URI only; bookmarks are replaced,
 * not auto-cleared.
 *
 * Threading: the demux thread is the only writer for the source FIFO; the player
 * thread reads from the source and pushes packets to the device. The control
 * runs on the VDR main thread and does not share state with the player except
 * through atomics and the player's public command methods.
 */

#ifndef VDR_VAAPIVIDEO_MEDIAPLAYER_H
#define VDR_VAAPIVIDEO_MEDIAPLAYER_H

#include "common.h"
#include "config.h"
#include "stream.h"

#include <atomic>
#include <cstdint>
#include <memory>
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
class cSubtitleConverter;

// The MEDIAPLAYER_* pacing/seek/EOF-drain constants live in mediaplayer.cpp, their only user
// (MEDIAPLAYER_JITTERBUF_BACKPRESSURE_FRAMES, the pre-anchor video-depth gate, in device.cpp likewise).

// ============================================================================
// === PLAYLIST HELPERS ===
// ============================================================================

/// One playlist row. @c uri is the absolute path (or URL); @c title is the
/// display label (defaults to the basename of @c uri).
struct PlaylistEntry {
    std::string uri;   ///< Absolute path or URL to open
    std::string title; ///< Display label; defaults to the basename of @c uri
};

/// Parse an m3u / m3u8 file. Honors `#EXTINF:duration,title` rows; falls back
/// to the basename when no title is given. Relative paths are resolved against
/// the playlist's parent directory. Comment / empty lines are skipped.
/// Returns an empty vector on read error.
[[nodiscard]] auto ParseM3U(std::string_view playlistPath) -> std::vector<PlaylistEntry>;

/// True for the small extension whitelist we expose in the file browser, or for
/// http(s)/ftp prefixes. Case-insensitive.
[[nodiscard]] auto IsMediaUri(std::string_view path) noexcept -> bool;

/// True for @c .m3u / @c .m3u8 paths (case-insensitive).
[[nodiscard]] auto IsPlaylistUri(std::string_view path) noexcept -> bool;

/// Outcome of StartPlayback(); lets callers surface a precise OSD / SVDRP error.
enum class StartPlaybackResult : uint8_t { Started, EmptyPlaylist, DeviceNotReady };

/// Launch playback of @p origin -- a media file, URL, or .m3u playlist (expanded here so the origin
/// URI survives into the bookmark). Resumes at the bookmarked position when @p origin matches the
/// bookmark. Wraps cControl::Launch(); does not block.
[[nodiscard]] auto StartPlayback(PlaylistEntry origin) -> StartPlaybackResult;

/// Ask the file browser to reopen (instead of live TV) on VDR's next main-menu hook; the browser
/// places its cursor on the bookmark itself. Wraps cRemote::CallPlugin(). VDR main thread only.
auto RequestBrowserReopen() -> void;

/// Consume a pending RequestBrowserReopen(): true (and clear it) if one was set, else false.
[[nodiscard]] auto TakeBrowserReopen() noexcept -> bool;

/// Flush a bookmark staged off-thread (SVDRP teardown) to setup.conf; no-op when nothing is pending.
/// Called from the plugin's Housekeeping(). VDR main thread only.
auto FlushPendingBookmarkSave() -> void;

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

    /// One demuxed audio stream. @c info.extradata aliases @c extradataStorage, so the table must
    /// never reallocate after Open (reserve()d once, never appended to).
    struct AudioTrackDesc {
        int avStreamIndex{-1};                       ///< Index into formatCtx->streams
        AVRational timeBase{.num = 1, .den = 90000}; ///< Stream time base (drives Rebase90k)
        AudioStreamInfo info;                        ///< Decoder descriptor; .extradata aliases extradataStorage
        std::vector<uint8_t> extradataStorage;       ///< Owns the extradata bytes for this track
        std::string language;                        ///< ISO-639 from the "language" metadata tag; "" if absent
        int srcChannels{0};                          ///< Container channel count (info.channels is forced to 2)
    };

    /// One demuxed supported subtitle stream (text or DVB bitmap). @c codecpar aliases the AVStream
    /// owned by formatCtx, so it is valid only while the source is open; the subtitle converter copies
    /// what it needs when opening its decoder.
    struct SubtitleTrackDesc {
        int avStreamIndex{-1};                       ///< Index into formatCtx->streams
        AVRational timeBase{.num = 1, .den = 90000}; ///< Stream time base (drives Rebase90k)
        AVCodecID codecId{AV_CODEC_ID_NONE};         ///< Subtitle codec (text or dvb_subtitle)
        const AVCodecParameters *codecpar{nullptr};  ///< Borrowed; used to open the converter's decoder
        std::string language;                        ///< ISO-639 from the "language" metadata tag; "" if absent
    };

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
    /// Arm a one-shot window flagging video packets below @p before90k with AV_PKT_FLAG_DISCARD:
    /// libavcodec decodes them (the reference chain needs the preroll) but drops the output, so a
    /// re-anchor's preroll neither burns the VPP chain nor replays in slow motion (trick pacing
    /// has no clock gate). Armed by every SeekToMs(); AV_NOPTS_VALUE disarms (FF entry starts at
    /// the keyframe at/below the target). Also disarmed by the first video packet at/after the
    /// target, or by the next Seek().
    auto DiscardVideoPrerollBefore(int64_t before90k) noexcept -> void { discardVideoBefore90k = before90k; }
    [[nodiscard]] auto DurationMs() const noexcept -> int; ///< 0 when unknown (live streams)
    [[nodiscard]] auto IoInterrupted() const noexcept
        -> bool; ///< True if a blocking libavformat I/O should bail (shutdown via stopFlag, or a
                 ///< pending seek/next via interruptFlag). Polled by the interrupt_callback.
    /// True once a stream is routed; audio-less video (and vice versa) is legal, so both are checked.
    [[nodiscard]] auto HasAudio() const noexcept -> bool { return audioStreamIndex >= 0; }
    [[nodiscard]] auto HasVideo() const noexcept -> bool { return videoStreamIndex >= 0; } ///< @see HasAudio()
    /// All audio streams found at Open(); the index doubles as the cDisplayTracks menu index.
    [[nodiscard]] auto AudioTracks() const noexcept -> const std::vector<AudioTrackDesc> & { return audioTracks; }
    /// Number of AudioTracks(); 0 for video-only input.
    [[nodiscard]] auto AudioTrackCount() const noexcept -> int { return static_cast<int>(audioTracks.size()); }
    /// Index into AudioTracks() of the stream being decoded; -1 before the first ApplyCurrentAudioTrack().
    [[nodiscard]] auto CurrentAudioTrack() const noexcept -> int { return currentAudioTrack; }
    /// Repoint the active audio stream (demux state only -- never the device/codec/seek). False on a
    /// bad index. Caller serializes via sourceMutex once the source is published.
    [[nodiscard]] auto SelectAudioTrack(int trackIdx) -> bool;
    /// Selectable subtitle streams (text + DVB bitmap only); the index doubles as the chooser menu index.
    [[nodiscard]] auto SubtitleTracks() const noexcept -> const std::vector<SubtitleTrackDesc> & {
        return subtitleTracks;
    }
    /// Number of SubtitleTracks(); 0 when the input carries none we can render.
    [[nodiscard]] auto SubtitleTrackCount() const noexcept -> int { return static_cast<int>(subtitleTracks.size()); }
    /// Index into SubtitleTracks() of the routed stream; -1 = subtitles off (the default).
    [[nodiscard]] auto CurrentSubtitleTrack() const noexcept -> int { return currentSubtitleTrack; }
    /// Repoint (or disable, idx < 0) the active subtitle stream -- demux routing only; the converter
    /// owns decode/render. False on an out-of-range index. Caller serializes via sourceMutex.
    [[nodiscard]] auto SelectSubtitleTrack(int trackIdx) -> bool;
    [[nodiscard]] auto VideoFps() const noexcept -> double {
        return videoFps;
    } ///< Container-reported avg_frame_rate; 0.0 if unknown.
    /// Coded (width, height) of the video stream in px; {0, 0} for audio-only input.
    [[nodiscard]] auto VideoCodedSize() const noexcept -> std::pair<int, int> {
        return {videoInfo.codedWidth, videoInfo.codedHeight};
    }

  private:
    /// One pass over formatCtx->streams after Open(): fills the track tables, picks the initial
    /// streams, snapshots what the cached getters serve.
    auto PopulateStreamInfo() -> void;
    /// Fill @p info (codec / rate / forced-stereo / extradata) for one audio AVStream into @p storage.
    static auto PopulateAudioInfo(const AVStream *stream, AudioStreamInfo &info, std::vector<uint8_t> &storage) -> void;
    /// Mirror audioTracks[currentAudioTrack] into audioStreamIndex / audioTimeBase / audioInfo.
    auto ApplyCurrentAudioTrack() -> void;
    /// Mirror subtitleTracks[currentSubtitleTrack] into subtitleStreamIndex / subtitleTimeBase
    /// (or clear them when no subtitle track is selected).
    auto ApplyCurrentSubtitleTrack() -> void;
    /// Rescale @p ts from @p tb to 90 kHz and subtract the source's PTS origin.
    /// Lazily seeds @c ptsOrigin90k on first call (when @p seedOrigin) so files whose
    /// container/streams don't advertise start_time still emit a zero-based timeline.
    /// Subtitle packets pass @p seedOrigin = false: they must never define the playback
    /// timeline, so before audio/video has seeded the origin they rebase to NOPTS and drop.
    [[nodiscard]] auto Rebase90k(int64_t ts, AVRational tb, bool seedOrigin = true) noexcept -> int64_t;

    std::unique_ptr<AVFormatContext, FreeAVFormatContext> formatCtx; ///< Open input; null until Open() succeeds
    std::atomic<bool> *stopFlag{nullptr};      ///< Non-owning; shutdown signal polled by the interrupt_callback.
    std::atomic<bool> *interruptFlag{nullptr}; ///< Non-owning; seek/next signal polled by the interrupt_callback.
    int videoStreamIndex{-1};                  ///< formatCtx->streams index of the video stream; -1 = none
    int audioStreamIndex{-1};                  ///< formatCtx->streams index of the decoding audio stream
    AVRational videoTimeBase{.num = 1, .den = 90000}; ///< Video stream time base (drives Rebase90k)
    AVRational audioTimeBase{.num = 1, .den = 90000}; ///< Audio stream time base (drives Rebase90k)
    int64_t ptsOrigin90k{AV_NOPTS_VALUE};             ///< First-packet PTS in 90 kHz (or formatCtx->start_time
                                                      ///< when known); subtracted from every emitted PTS so the
                                                      ///< replay bar and Seek() math share a zero-based timeline.
    int64_t discardAudioBefore90k{AV_NOPTS_VALUE};    ///< Post-seek guard: armed to the seek target, drops
                                                      ///< audio packets earlier than that so the master
                                                      ///< clock anchors at the requested timeline (and not
                                                      ///< at the earlier video keyframe libavformat lands on).
    int64_t discardVideoBefore90k{AV_NOPTS_VALUE};    ///< Re-anchor preroll guard: video below this gets
                                                      ///< AV_PKT_FLAG_DISCARD (decoded for refs, never shown).
                                                      ///< See DiscardVideoPrerollBefore().
    VideoStreamInfo videoInfo;                  ///< Video decoder descriptor; .extradata aliases videoExtradataStorage
    AudioStreamInfo audioInfo;                  ///< Mirror of audioTracks[currentAudioTrack].info (the decoding stream)
    std::vector<uint8_t> videoExtradataStorage; ///< Owns the bytes videoInfo.extradata points at
    std::vector<AudioTrackDesc> audioTracks;    ///< All audio streams; index == cDisplayTracks menu index
    int currentAudioTrack{-1};                  ///< Index into audioTracks of the decoding stream (-1 = none)
    std::vector<SubtitleTrackDesc> subtitleTracks; ///< All selectable subtitle streams; index == chooser menu index
    int currentSubtitleTrack{-1};                  ///< Index into subtitleTracks of the routed stream (-1 = off)
    int subtitleStreamIndex{-1};                   ///< Mirror of subtitleTracks[currentSubtitleTrack].avStreamIndex
    AVRational subtitleTimeBase{.num = 1, .den = 90000}; ///< Mirror of the routed subtitle stream's time base
    double videoFps{0.0};   ///< Snapshot of avg_frame_rate at Open(); 0.0 if not advertised.
    bool eofReached{false}; ///< Latched when av_read_frame() returns AVERROR_EOF; cleared by Seek()/Flush()
};

// ============================================================================
// === PLAYER ===
// ============================================================================

/// Index of the normal-speed ('1') entry in the trick-speed notch table (vdr/dvbplayer.c
/// Speeds[]; the table itself lives in mediaplayer.cpp). Here only for member initialization.
inline constexpr int MEDIAPLAYER_TRICK_NORMAL_IDX = 4;

/// VDR cPlayer + private demux cThread. Owns the cVaapiMediaSource and walks the
/// playlist. Action() runs the packet pump: pull one video + one audio packet
/// per iteration, submit to the device, throttle on IsMediaPlayerBackpressured().
// NOLINTNEXTLINE(misc-multiple-inheritance) -- standard VDR pattern; cf. cDvbPlayer in VDR core.
class cVaapiPlayer final : public cPlayer, public cThread {
  public:
    /// Only the phases anyone observes (IsFinished / Action's EOF checks). Transient pause/seek/
    /// trick phases are NOT states -- they live in the dedicated `playMode` / `seekPending` atomics;
    /// folding them in here made two threads write one variable for values nobody read.
    enum class State : uint8_t { Running, Eof, Stopped };

    /// dvbplayer-style play phase (cf. ePlayModes in vdr/dvbplayer.c). The VDR main thread (key
    /// handlers) is the primary writer; the demux thread writes only on auto-exit edges (reverse
    /// play reaching the file start, EOF during a trick mode). Fast = keyframe stepping paced by
    /// the decoder's trick hold; Slow = full decode paced below real time. Direction lives in the
    /// separate `trickForward` flag, the speed notch in `trickSpeedIdx`.
    enum class PlayMode : uint8_t { Play, Pause, Slow, Fast };

    /// @p uri is what the user selected (media file, .m3u path, or URL) -- the bookmark identity.
    /// @p startMs is the absolute resume position for the FIRST entry (0 = start).
    explicit cVaapiPlayer(std::string uri, std::vector<PlaylistEntry> entries, int startMs);
    ~cVaapiPlayer() noexcept override;
    cVaapiPlayer(const cVaapiPlayer &) = delete;
    cVaapiPlayer(cVaapiPlayer &&) noexcept = delete;
    auto operator=(const cVaapiPlayer &) -> cVaapiPlayer & = delete;
    auto operator=(cVaapiPlayer &&) noexcept -> cVaapiPlayer & = delete;

    // ========================================================================
    // === PUBLIC API (called from cVaapiControl on the VDR main thread) ===
    // ========================================================================
    /// Resume normal playback from pause or any trick mode (cf. cDvbPlayer::Play()); no-op while playing.
    auto Play() -> void;
    /// Toggle pause (cf. cDvbPlayer::Pause()): freezes from normal play (parks the demux thread and
    /// halts the audio master clock), exits a trick mode into pause, resumes when already paused.
    auto Pause() -> void;
    /// One notch toward faster-forward (cf. cDvbPlayer::Forward()): from play -> fast forward, deeper
    /// on repeat; from pause -> slow motion; winds an active backward mode down through normal.
    auto Forward() -> void;
    /// Mirror of Forward() toward backward playback (cf. cDvbPlayer::Backward()).
    auto Backward() -> void;
    /// Leave an active trick mode into normal play WITHOUT the Exit re-anchor; no-op otherwise.
    /// For callers about to reposition anyway (jump seek, playlist advance, audio-track switch) --
    /// their own seek/reopen re-anchors, so ExitTrick()'s staged seek would just double the flush.
    auto LeaveTrickWithoutReanchor() -> void;
    /// True while frozen (not during trick modes); safe from any thread.
    [[nodiscard]] auto IsPaused() const noexcept -> bool {
        return playMode.load(std::memory_order_acquire) == PlayMode::Pause;
    }
    /// True while a trick mode (fast or slow, either direction) is active; safe from any thread.
    [[nodiscard]] auto IsTrickMode() const noexcept -> bool {
        const PlayMode mode = playMode.load(std::memory_order_acquire);
        return mode == PlayMode::Fast || mode == PlayMode::Slow;
    }
    auto Seek(int64_t deltaMs) -> void; ///< Relative seek; deltaMs may be negative
    auto Next() -> void;                ///< Skip to next playlist entry, if any
    /// Display title of the entry playing now (playlist label, else the basename).
    [[nodiscard]] auto Title() const -> std::string;
    /// Snapshot the resume bookmark. A position is captured only for a single seekable local file
    /// still playing; playlists, streams, EOF and failed opens yield position 0 (URI only). Lock-free
    /// (cached atomics + STC) so the control dtor never blocks behind a stalled demux read.
    [[nodiscard]] auto MakeBookmark() const -> MediaBookmark;
    /// True once playback can no longer continue: natural EOF, fatal open failure, or shutdown.
    /// cVaapiControl uses this to exit on its next key event.
    [[nodiscard]] auto IsFinished() const noexcept -> bool {
        const State current = state.load(std::memory_order_acquire);
        return current == State::Eof || current == State::Stopped;
    }

    // ========================================================================
    // === cPlayer overrides (public) ===
    // ========================================================================
    /// cPlayer hook for the replay bar (cf. cDvbPlayer::GetReplayMode): Play/Forward/Speed from the
    /// trick state, so skins render the usual "1>>".."3>>" / "<<1" / "1|>" / "<|1" mode symbols.
    [[nodiscard]] auto GetReplayMode(bool &Play, bool &Forward, int &Speed) -> bool override;
    /// Position and length in MILLISECONDS, not cPlayer's frame unit -- cVaapiControl, the only
    /// caller, formats them directly. Served from cached atomics so it never blocks behind the
    /// demux thread. @p SnapToIFrame is ignored.
    [[nodiscard]] auto GetIndex(int &Current, int &Total, bool SnapToIFrame = false) -> bool override;
    /// Cached container fps, else cPlayer's default (25) -- skins divide by it for the frame counter.
    [[nodiscard]] auto FramesPerSecond() -> double override;
    [[nodiscard]] auto InfoText() const -> std::string; ///< Multi-line file metadata for cControl::GetInfo().
    /// cPlayer hook for the Audio-button track menu (VDR main thread). Maps Type to a descriptor index
    /// and hands it to the demux thread. @p TrackId unused (the player owns the mapping).
    auto SetAudioTrack(eTrackType Type, const tTrackId *TrackId) -> void override;
    /// cPlayer hook for the Subtitles-button chooser (VDR main thread). Maps Type to a subtitle
    /// descriptor index (ttNone = off) and hands it to the demux thread. @p TrackId unused.
    auto SetSubtitleTrack(eTrackType Type, const tTrackId *TrackId) -> void override;

  protected:
    // ========================================================================
    // === cPlayer overrides (protected) ===
    // ========================================================================
    auto Activate(bool On) -> void override; ///< Visibility intentionally matches cPlayer base (protected).

    // ========================================================================
    // === cThread overrides ===
    // ========================================================================
    /// Demux loop: read -> rebase PTS -> push to the device, with pause/seek/next serviced between packets.
    auto Action() -> void override;

  private:
    /// Construct the source for playlist[currentIndex], register its tracks, cache the metadata.
    /// False when it cannot be opened -- the caller then advances or finishes.
    [[nodiscard]] auto OpenCurrentEntry() -> bool;
    /// Tear the current entry down and clear the cached metadata. Idempotent.
    auto CloseCurrentEntry() noexcept -> void;
    /// Seek the just-opened first entry to the one-shot resume position before the demux thread starts.
    /// Consumes startPositionMs (playlist advancement unaffected); skipped for live / at-or-past-end.
    /// Takes sourceMutex.
    auto ApplyStartPosition() -> void;
    /// Current playback position in milliseconds from the device's audio-mastered STC.
    /// Returns 0 when no vaapivideo device is attached or the audio clock has not anchored.
    [[nodiscard]] auto CurrentPositionMs() const noexcept -> int;
    /// Demux-thread lookahead in 90 kHz ticks: how far ahead of the audio master clock the
    /// most recently submitted packet is. Returns AV_NOPTS_VALUE when either side is unanchored
    /// (startup, post-seek, no device) -- callers must skip the comparison in that case.
    [[nodiscard]] auto Lookahead90k(const cVaapiDevice *vaapiDev) const noexcept -> int64_t;
    /// Demux-thread half of Seek(): resolves the delta against the current position, then SeekToMs().
    auto PerformSeek(int64_t deltaMs) -> void;
    /// Absolute seek + re-anchor; assumes sourceMutex held. Shared by PerformSeek, the track
    /// switch's re-anchor (which Seek() would skip as a delta==0 no-op), and trick transitions.
    /// False when nothing was flushed (no source/device, or the container seek failed -- playback
    /// then continues at the old position). Trick transitions MUST check it and fail closed;
    /// jump/track-switch callers deliberately ignore it (continuing unmoved is the right fallback).
    [[nodiscard]] auto SeekToMs(int64_t targetMs) -> bool;

    // ========================================================================
    // === TRICK PLAY (state machine on the VDR main thread; feed in Action) ===
    // ========================================================================
    /// One-shot feed-transition command for the demux thread, staged by the state machine and
    /// consumed in Action() like seekPending. Enter* start a trick feed at the anchored position;
    /// Exit re-anchors normal playback after leaving a trick. Slow-forward is its own command
    /// (not derived from playMode in the consumer): the command is staged BEFORE playMode is
    /// published, so the consumer must not depend on the mode store having landed.
    enum class TrickCommand : uint8_t { None, EnterForward, EnterSlowForward, EnterReverse, Exit };

    /// cDvbPlayer's Forward()/Backward() transition graph, folded over the direction: one notch
    /// toward faster playback @p towardForward. An active opposite-direction mode winds down
    /// through normal play/pause (multi-speed) or restarts in the new direction (single-speed).
    auto CycleTrick(bool towardForward) -> void;
    /// Shared trick entry: publish the trick state, apply the first speed notch, stage the feed
    /// transition, wake the demux thread. Refused (no-op) without an open entry, or backward on a
    /// source without a known duration (reverse needs seekable, bounded input).
    auto EnterTrick(PlayMode mode, bool forward, int firstStep) -> void;
    /// cDvbPlayer::TrickSpeed(Increment) verbatim: walk trickSpeedIdx one notch, saturate silently
    /// on the table's 0 sentinels, resolve the '1' entry to Play()/Pause(), else map the table
    /// entry to the device frame-repeat count and call DeviceTrickSpeed().
    auto TrickSpeedStep(int increment) -> void;
    /// Shared trick-exit core: publish @p nextMode with the trick state reset and end the device
    /// trick via DevicePlay() (trickSpeed=0 + decoder trick exit + unpause). Callers add their own
    /// re-anchor: ExitTrick() stages Exit; the demux auto-exits (EOF, reverse at start) re-anchor
    /// inline.
    auto EndTrick(PlayMode nextMode) -> void;
    /// Main-thread trick exit: EndTrick(); exit-to-pause re-freezes right after (queues are empty
    /// during trick, so nothing plays out); stages the Exit re-anchor for the demux thread.
    auto ExitTrick(bool toPause) -> void;
    /// Demux-thread fail-closed exit: a trick transition/step cannot proceed (its seek failed), so
    /// running on in trick mode would play the wrong timeline or retry forever. Returns to where
    /// the trick was entered from: pause for slow modes, play for fast. No re-anchor -- seeks are
    /// exactly what just failed.
    auto AbortTrick(const char *reason) -> void;
    /// Arm ioInterrupt BEFORE staging a demux command; the consuming branch clears it after taking
    /// the command. Armed AFTER the command it can go stray: the demux may consume the command in
    /// between, and the then-unmatched interrupt aborts an innocent read (the abort also latches
    /// AVERROR_EXIT/eof in the AVIOContext -- see ReadPacket's reset).
    auto ArmDemuxInterrupt() noexcept -> void { ioInterrupt.store(true, std::memory_order_release); }
    /// Wake the demux thread out of a pause park so a staged command is serviced promptly.
    /// Blocking-read interruption is ArmDemuxInterrupt()'s job, BEFORE the command store.
    auto WakeDemux() -> void;
    /// Demux-thread half of a staged trick transition: SeekToMs() re-anchor at the shown position,
    /// slow-motion preroll discard, and reverse-stepping init. Takes sourceMutex.
    auto PerformTrickTransition(TrickCommand cmd) -> void;
    /// One reverse trick step: container-seek just below the last shown keyframe, read exactly that
    /// keyframe, submit it. Steps the target back further when a coarse seek lands on the same
    /// keyframe; auto-resumes normal play at the file start. Returns false when no progress was
    /// made (read failure / submit refused) so the caller sleeps. Takes sourceMutex.
    [[nodiscard]] auto PerformReverseStep(cVaapiDevice *vaapiDev, AVPacket *packet) -> bool;
    /// Publish the source's audio streams to the device for cDisplayTracks and select the
    /// Setup.AudioLanguages-preferred initial track. sourceMutex held (from OpenCurrentEntry).
    auto RegisterAudioTracks() -> void;
    /// Demux-thread track switch: repoint source, reopen audio codec, re-anchor A/V. Reverts to the
    /// old track if the new codec fails.
    auto PerformAudioSwitch(int trackIdx) -> void;
    /// Publish the source's supported subtitle streams to the device for the Subtitles-button chooser.
    /// Subtitles default off. sourceMutex held (from OpenCurrentEntry).
    auto RegisterSubtitleTracks() -> void;
    /// Demux-thread subtitle switch: repoint source routing and (re)open or close the converter's
    /// decoder. @p trackIdx < 0 turns subtitles off. No A/V re-anchor (subtitles ride the same clock).
    auto PerformSubtitleSwitch(int trackIdx) -> void;
    /// Block until the decode + present pipeline has flushed the buffered end-of-stream tail to the
    /// screen, so a natural EOF does not cut playback short. Returns early if the user issues
    /// stop / pause / seek / next during the wait, or if the pipeline stalls. Demux thread only;
    /// called before AdvancePlaylist() tears the entry down. See MEDIAPLAYER_EOF_DRAIN_* .
    auto DrainTailAtEof() -> void;
    /// Move to the next playlist entry, or finish when the list is exhausted.
    auto AdvancePlaylist() -> void;

    const std::string originUri; ///< What the user selected (file / .m3u path / URL); the bookmark identity
    int startPositionMs{0};      ///< First-entry resume position; consumed once in Activate(true) before Start()
    std::vector<PlaylistEntry> playlist; ///< One entry for a single file/URL; expanded rows for an .m3u
    std::atomic<size_t> currentIndex{0}; ///< Index into playlist of the entry playing now

    std::unique_ptr<cVaapiMediaSource> source;      ///< Current entry's demuxer; swapped under sourceMutex
    std::unique_ptr<cSubtitleConverter> subtitles;  ///< Subtitle decode + overlay; created lazily in OpenCurrentEntry
    std::atomic<State> state{State::Running};       ///< Lifecycle state; Eof/Stopped are what IsFinished() reports
    std::atomic<PlayMode> playMode{PlayMode::Play}; ///< Play phase (see PlayMode); the demux loop parks while Pause
    std::atomic<bool> trickForward{true};           ///< Trick direction; meaningful while playMode is Slow/Fast
    std::atomic<int> trickSpeedIdx{MEDIAPLAYER_TRICK_NORMAL_IDX}; ///< Speed notch: index into the trick-speed table
    std::atomic<TrickCommand> trickCommand{TrickCommand::None};   ///< Staged feed transition; consumed in Action()
    std::atomic<int> trickAnchorMs{-1}; ///< Shown position (ms) captured on the MAIN thread at the trick keypress,
                                        ///< BEFORE DeviceTrickSpeed()/DevicePlay() wipe the decoder's lastPts --
                                        ///< a demux-side position read races the flush and anchors at 0.
                                        ///< PerformTrickTransition re-anchors here.
    int64_t reverseTargetPts90k{-1};    ///< Demux-thread only: next reverse-step container seek target (90 kHz)
    int64_t reverseShownPts90k{-1}; ///< Demux-thread only: PTS of the last submitted reverse keyframe (progress guard)
    std::atomic<bool> seekPending{false}; ///< Set by Seek(), consumed by Action() before the next read
    std::atomic<int64_t> seekDeltaMs{0};  ///< Relative seek staged with seekPending (ms, may be negative)
    /// Per-track-type switch coordination. A Set*Track() call (VDR thread) stages a request here;
    /// the demux Action() services it before the pause branch so a frozen player still switches.
    struct TrackSwitchState {
        std::atomic<bool> pending{false}; ///< Set by Set*Track(); serviced (and cleared) in Action().
        std::atomic<int> targetIdx{-1};   ///< Descriptor index requested; -1 = none/off.
        std::atomic<int> menuIndex{-1};   ///< Index last selected; lets Set*Track() no-op redundant re-selections.
        std::atomic<int> trackCount{0};   ///< Registered track count; range check in Set*Track().
    };
    TrackSwitchState audioSwitch;           ///< Audio-track switch request (menuIndex = index currently decoding).
    TrackSwitchState subtitleSwitch;        ///< Subtitle-track switch request (menuIndex = last requested, -1 = off).
    std::atomic<bool> nextRequested{false}; ///< Set by Next(), consumed by Action() to advance the playlist
    std::atomic<bool> stopping{false};      ///< Shutdown signal; also breaks blocking libavformat I/O
    std::atomic<bool> ioInterrupt{false};   ///< Breaks a blocking av_read_frame()/av_seek_frame() on a slow
                                            ///< network URL so a staged command is serviced promptly instead of
                                            ///< after the I/O timeout. INVARIANT: armed via ArmDemuxInterrupt()
                                            ///< strictly BEFORE the command it belongs to is staged (see there).
                                            ///< Polled via cVaapiMediaSource::IoInterrupted(); cleared by the
                                            ///< command-consuming branches in Action() and in ReadPacket.
    /// Max AUDIO packet PTS submitted (90 kHz); drives the lookahead-vs-audio-clock throttle in Action(). Audio only:
    /// keying off the max of both streams lets a TS's video-leads-audio mux PTS offset inflate the lookahead and starve
    /// the audio queue. AV_NOPTS_VALUE for video-only.
    std::atomic<int64_t> latestAudioPts90k{AV_NOPTS_VALUE};
    std::atomic<int> pendingSeekTargetMs{-1}; ///< Most recent seek target (ms). Used by CurrentPositionMs() as a
                                              ///< fallback while GetSTC() is still NOPTS in the ~50 ms window between
                                              ///< Clear() and the first decoded frame at the new position. Without
                                              ///< this, rapid follow-up Seek()s read position 0 and the playhead
                                              ///< snaps to the file start.
    // Entry-metadata snapshots (set on open, cleared on close). Main-thread queries -- GetIndex,
    // FramesPerSecond, MakeBookmark -- read these instead of taking sourceMutex, which the demux
    // thread holds across blocking network I/O (cf. cDvbPlayer, which serves GetIndex from cached
    // indexes for the same reason).
    std::atomic<int> cachedDurationMs{-1};   ///< Current entry's duration; -1 = no entry open, 0 = live/unknown
    std::atomic<double> cachedVideoFps{0.0}; ///< Current entry's container fps; 0.0 = unknown

    mutable cMutex sourceMutex; ///< Guards source-pointer swaps across Action() and command methods
    cCondVar pauseCondition;    ///< Wakes Action() out of pause loop
    mutable cMutex pauseMutex;  ///< Pairs with pauseCondition; never held across a demux read
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
    /// Constructs the player it owns; @p startPositionMs is the one-shot resume offset. Go through
    /// cControl::Launch(), not this directly.
    cVaapiControl(std::string originUri, std::vector<PlaylistEntry> entries, int startPositionMs)
        : cVaapiControl(new cVaapiPlayer(std::move(originUri), std::move(entries), startPositionMs)) {}
    ~cVaapiControl() noexcept override;
    cVaapiControl(const cVaapiControl &) = delete;
    cVaapiControl(cVaapiControl &&) noexcept = delete;
    auto operator=(const cVaapiControl &) -> cVaapiControl & = delete;
    auto operator=(cVaapiControl &&) noexcept -> cVaapiControl & = delete;

    /// cControl hook: drop the replay bar so a menu can take the shared OSD plane.
    auto Hide() -> void override;
    /// cControl hook: play/pause, seek, next, info and stop; returns osEnd once playback is finished.
    [[nodiscard]] auto ProcessKey(eKeys Key) -> eOSState override;
    /// cControl hook: the title VDR shows while this control is active.
    [[nodiscard]] auto GetHeader() -> cString override;
    [[nodiscard]] auto GetInfo() -> cOsdObject * override; ///< File metadata dialog shown via kInfo.

  private:
    /// Delegating ctor: takes the typed pointer once and hands it to cControl as cPlayer*
    /// (upcast) and to @c player as cVaapiPlayer*. Avoids a static_cast downcast off
    /// cControl::player just to re-acquire the type we already had.
    explicit cVaapiControl(cVaapiPlayer *typedPlayer);
    auto ShowReplayBar() -> void;    ///< Create the skin replay display and arm its auto-hide timeout
    auto HideReplayBar() -> void;    ///< Destroy the replay display; idempotent
    auto RefreshReplayBar() -> void; ///< Repaint position/duration; rate-limited via lastBarRefresh
    /// Handle a seek key: log, dispatch the relative seek, and pop up the replay bar so the
    /// user gets immediate visual feedback. @p label is the key name for the log line.
    [[nodiscard]] auto HandleSeekKey(const char *label, int deltaMs) -> eOSState;
    /// Handle FastFwd/FastRew (press and release events): filter autorepeat, apply the
    /// single-speed hold-to-scan release semantics, then cycle the trick state.
    [[nodiscard]] auto HandleTrickKey(eKeys key, bool forward) -> eOSState;

    std::unique_ptr<cVaapiPlayer> player; ///< Sole owner. cControl::player is VDR's borrowed alias of the same pointer.
    cSkinDisplayReplay *displayReplay{nullptr}; ///< Skin replay display; owned while barVisible
    cTimeMs barTimeout;                         ///< Auto-hide deadline for the replay bar
    cTimeMs lastBarRefresh;                     ///< Rate limit for RefreshReplayBar()
    bool barVisible{false};                     ///< Whether the replay bar is currently on screen
};

// ============================================================================
// === FILE BROWSER ===
// ============================================================================

/// Directory browser. Lists subdirectories first, then media files and .m3u
/// playlists. kOk enters a directory or launches playback. kBack pops to parent.
class cVaapiFileBrowser final : public cOsdMenu {
  public:
    /// @p startDir is the root / fallback (the -m media-dir). Opens on the bookmark when it is a
    /// reachable local file (parent dir, cursor on it), else falls back to @p startDir.
    explicit cVaapiFileBrowser(std::string startDir);
    ~cVaapiFileBrowser() noexcept override = default;
    cVaapiFileBrowser(const cVaapiFileBrowser &) = delete;
    cVaapiFileBrowser(cVaapiFileBrowser &&) noexcept = delete;
    auto operator=(const cVaapiFileBrowser &) -> cVaapiFileBrowser & = delete;
    auto operator=(cVaapiFileBrowser &&) noexcept -> cVaapiFileBrowser & = delete;

    /// cOsdMenu hook: kOk enters a directory or starts playback, kBack pops to the parent.
    [[nodiscard]] auto ProcessKey(eKeys Key) -> eOSState override;

  private:
    /// What a row stands for; also the sort key (parent, then directories, then files/playlists).
    enum class EntryKind : uint8_t { Parent, Directory, File, Playlist };
    /// One row of the listing.
    struct BrowserEntry {
        EntryKind kind;         ///< Row type; decides the icon prefix and what kOk does
        std::string name;       ///< Display name (basename only)
        std::uintmax_t size{0}; ///< File size in bytes; 0 / unused for Parent and Directory
    };

    /// Read @p dir, rebuild @c entries (directories first, then media files) and repaint the menu.
    auto LoadDirectory(const std::string &dir) -> void;
    /// Move the cursor to the entry whose basename matches @p name and redraw; false if none match.
    [[nodiscard]] auto SelectEntryByName(std::string_view name) -> bool;
    /// Entry under the cursor, or nullptr for an empty listing.
    [[nodiscard]] auto SelectedEntry() const -> const BrowserEntry *;
    /// Absolute path of @p entry, i.e. currentDir joined with its name.
    [[nodiscard]] auto BuildFullPath(const BrowserEntry &entry) const -> std::string;

    std::string currentDir;            ///< Directory being listed; the base for BuildFullPath()
    std::vector<BrowserEntry> entries; ///< Rows in display order; index matches the cOsdMenu item index
};

#endif // VDR_VAAPIVIDEO_MEDIAPLAYER_H
