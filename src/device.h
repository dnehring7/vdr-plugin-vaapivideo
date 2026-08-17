// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file device.h
 * @brief VDR device integration, PES routing, and lifecycle
 */

#ifndef VDR_VAAPIVIDEO_DEVICE_H
#define VDR_VAAPIVIDEO_DEVICE_H

#include "caps.h"
#include "common.h"

// VDR
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <vdr/device.h>
#include <vdr/osd.h>
#include <vdr/tools.h>
#pragma GCC diagnostic pop

class cAudioProcessor;
class cVaapiDecoder;
class cVaapiDisplay;
struct AudioStreamInfo;
struct VideoStreamInfo;

// ============================================================================
// === STRUCTURES ===
// ============================================================================

/// Shared VAAPI hardware context passed to decoder and display subsystems.
/// Decode profile flags and VPP filter capabilities live on @ref caps (populated
/// once by ProbeGpuCaps() at device init).
struct VaapiContext {
    AVBufferRef *hwDeviceRef{}; ///< VAAPI hardware device (owned, freed via av_buffer_unref())
    int drmFd{-1};              ///< DRM file descriptor (borrowed -- owned by cVaapiDevice)
    GpuCaps caps{};             ///< GPU decode + VPP capability snapshot (see caps.h)
};

// ============================================================================
// === DISPLAY MODE MATCHING ===
// ============================================================================
// Runtime mode switching splits in two: the device owns the connector's mode inventory and the
// policy (this section), the display owns applying the winner (cVaapiDisplay::RequestDisplayMode).
// Everything here is off unless the operator enables it -- see the DISPLAY MODE SWITCHING block
// in config.h.

/// One connector mode that survived the usability filter, pre-digested for matching.
/// Built once per attach by BuildModeCandidates(); indices stay valid for cVaapiDevice::connectorModes.
struct DisplayModeCandidate {
    uint32_t height{};         ///< vdisplay (px)
    uint32_t refreshMilliHz{}; ///< Exact rate from ModeRefreshMilliHz(); never the truncated vrefresh
    bool preferred{};          ///< Mode carries DRM_MODE_TYPE_PREFERRED (the sink's native timing)
    uint16_t index{};          ///< Index into cVaapiDevice::connectorModes
    uint32_t width{};          ///< hdisplay (px)
};

/// What the stream currently playing wants from the output.
struct StreamModeRequest {
    uint32_t height{};      ///< Coded frame height (px)
    uint32_t rateMilliHz{}; ///< VPP *output* rate, i.e. the field rate when the chain deinterlaces --
                            ///< 1080i25 arrives here as 50000, which is why 25i naturally lands on 50 Hz
    uint32_t width{};       ///< Coded frame width (px)

    /// Field-wise equality; the stability gate uses it to recognise an unchanged request.
    [[nodiscard]] auto operator==(const StreamModeRequest &other) const noexcept -> bool = default;
    /// False while any field is still unknown, i.e. before the filter chain has published a format.
    [[nodiscard]] auto IsValid() const noexcept -> bool { return width > 0 && height > 0 && rateMilliHz > 0; }
};

/// Playback path a request came from; selects which of the three scope switches gates it.
enum class PlaybackSource : uint8_t {
    LiveTv = 0,      ///< Transfer Mode (cTransferControl)
    Replay = 1,      ///< Recording replay (cDvbPlayer)
    MediaPlayer = 2, ///< The plugin's own mediaplayer
};

/// Human label for a PlaybackSource; used in the decision log and the SVDRP MODE reply.
[[nodiscard]] constexpr auto PlaybackSourceName(PlaybackSource source) noexcept -> const char * {
    switch (source) {
        case PlaybackSource::LiveTv:
            return "live";
        case PlaybackSource::Replay:
            return "replay";
        case PlaybackSource::MediaPlayer:
            return "mediaplayer";
    }
    return "?"; // unreachable for a valid enum value; silences control-reaches-end warning
}

/// Snapshot of the mode-switching policy, taken from vaapiConfig at evaluation time so one
/// evaluation can never observe a half-applied setup-menu change.
struct DisplayModePolicy {
    uint32_t defaultHeight{};         ///< Fallback mode (the --resolution one) height
    uint32_t defaultRefreshMilliHz{}; ///< Fallback mode refresh
    uint32_t defaultWidth{};          ///< Fallback mode width
    bool matchRefresh{};              ///< Track the source frame rate
    bool matchResolution{};           ///< Track the source coded size
    uint32_t maxRefreshMilliHz{};     ///< Ceiling for the k*source search (k==1 may exceed it)
    uint32_t minHeight{};             ///< Floor for the resolution search
};

/// Outcome of SelectDisplayMode(): an index into the candidate list plus a human rationale.
struct DisplayModeMatch {
    int index{-1};      ///< Candidate index; < 0 means "no usable candidate, keep the current mode"
    std::string reason; ///< One-line explanation for the decision log / SVDRP MODE
};

/// Pick the connector mode that best serves @p request under @p policy.
///
/// Resolution first: the smallest candidate at or above the stream's coded size, never below
/// policy.minHeight (setting that to the panel's native height therefore pins the resolution).
/// Then refresh, restricted to that resolution: the HIGHEST exact integer multiple k*source that
/// stays at or below policy.maxRefreshMilliHz, so 25p lands on 50 Hz and 29.97 on 59.94 Hz while
/// 24p stays at 24 (unless the panel offers 48). k==1 is always allowed even above the cap.
/// Exposed (rather than file-local) so the SVDRP MODE command can explain the current decision.
[[nodiscard]] auto SelectDisplayMode(std::span<const DisplayModeCandidate> candidates, const StreamModeRequest &request,
                                     const DisplayModePolicy &policy) -> DisplayModeMatch;

/// Digest a connector's raw mode list into the matchable subset.
///
/// Drops interlaced modes -- the VPP always emits progressive frames and the scanout path has no
/// field interleaving -- and modes whose aspect ratio differs from @p defaultMode's by more than
/// 2%, so a stray 4:3 timing can never stretch the picture.
[[nodiscard]] auto BuildModeCandidates(std::span<const drmModeModeInfo> modes, const drmModeModeInfo &defaultMode)
    -> std::vector<DisplayModeCandidate>;

// ============================================================================
// === DRM DEVICES CLASS ===
// ============================================================================

/// Enumerates all DRM devices visible to the system via libdrm, to pick a primary node when no explicit
/// device path is configured. Iteration follows the standard range-for protocol via begin()/end().
class DrmDevices {
  public:
    DrmDevices() = default;
    ~DrmDevices() noexcept;
    DrmDevices(const DrmDevices &) = delete;
    DrmDevices(DrmDevices &&) noexcept = delete;
    auto operator=(const DrmDevices &) -> DrmDevices & = delete;
    auto operator=(DrmDevices &&) noexcept -> DrmDevices & = delete;

    // ========================================================================
    // === ITERATORS ===
    // ========================================================================
    [[nodiscard]] auto begin() -> std::vector<drmDevicePtr>::iterator; ///< Iterator to first enumerated DRM device
    [[nodiscard]] auto end() -> std::vector<drmDevicePtr>::iterator;   ///< Past-the-end iterator for DRM device list

    // ========================================================================
    // === QUERIES ===
    // ========================================================================
    [[nodiscard]] auto Enumerate() -> bool; ///< Populate device list via drmGetDevices2(); returns false if none found
    [[nodiscard]] auto HasDevices() const noexcept
        -> bool; ///< True if Enumerate() succeeded and found at least one device

  private:
    // ========================================================================
    // === STATE ===
    // ========================================================================
    std::vector<drmDevicePtr> deviceList; ///< Pointers to enumerated DRM device descriptors (freed in destructor)
};

// ============================================================================
// === VAAPI DEVICE CLASS ===
// ============================================================================

/// Primary VDR output device: demuxes PES, decodes via VAAPI, and renders over DRM/KMS.
/// Lifecycle: Initialize() latches device arguments and either attaches hardware immediately or stays
/// detached (--detached); once attached, VDR routes SetPlayMode/PlayVideo/PlayAudio here and Stop() tears
/// the subsystems down. All playback transitions (pause, trick speed, track switch) are handled inline.
class cVaapiDevice : public cDevice {
  public:
    cVaapiDevice();
    ~cVaapiDevice() noexcept override;
    cVaapiDevice(const cVaapiDevice &) = delete;
    cVaapiDevice(cVaapiDevice &&) noexcept = delete;
    auto operator=(const cVaapiDevice &) -> cVaapiDevice & = delete;
    auto operator=(cVaapiDevice &&) noexcept -> cVaapiDevice & = delete;

    // ========================================================================
    // === VDR DEVICE INTERFACE (public in cDevice) ===
    // ========================================================================
    [[nodiscard]] auto CanScaleVideo(const cRect &rect, int alignment = taCenter)
        -> cRect override;         ///< VPP resize path accepts normalized/clipped output rects.
    auto Clear() -> void override; ///< Flush decoder and audio queues without releasing hardware
    [[nodiscard]] auto DeviceName() const
        -> cString override; ///< Descriptive name (DRM path + connector) for SVDRP PRIM/LSTD replies
    [[nodiscard]] auto DeviceType() const -> cString override; ///< Returns "VAAPI"
#if APIVERSNUM >= 30014
    [[nodiscard]] auto Drain() -> bool override; ///< EOS: true once all buffered A/V has played out; never blocks.
#endif
    [[nodiscard]] auto Flush(int TimeoutMs = 0)
        -> bool override;           ///< Wait until packet queue drains; returns true when empty
    auto Freeze() -> void override; ///< Pause output: drain queue and stop audio
    auto GetOsdSize(int &Width, int &Height, double &PixelAspect)
        -> void override;                            ///< Return display framebuffer dimensions for OSD allocation
    [[nodiscard]] auto GetSTC() -> int64_t override; ///< Return presentation clock in VDR 90 kHz ticks
    auto GetVideoSize(int &Width, int &Height, double &VideoAspect)
        -> void override; ///< Return active video stream resolution
    [[nodiscard]] auto GrabImage(int &Size, bool Jpeg = true, int Quality = -1, int SizeX = -1, int SizeY = -1)
        -> uchar * override; ///< SVDRP GRAB: snapshot the displayed video + OSD as PNM (Jpeg=false) or JPEG.
                             ///< Returns malloc()'d buffer of @p Size bytes; caller (VDR core) free()s.
    [[nodiscard]] auto HasDecoder() const -> bool override; ///< True when a VAAPI codec context is open and ready
    [[nodiscard]] auto HasIBPTrickSpeed()
        -> bool override; ///< True while the replay carries video: all I/B/P frame types are submitted in trick
                          ///< mode and the decoder paces them. False for an audio-only replay -- see definition.
    [[nodiscard]] auto HardwareReady() const noexcept -> bool {
        return initState.load(std::memory_order_acquire) == 2;
    } ///< True once hardware is attached and decoder/display/audio are live -- the plugin's real
      ///< "usable" gate (and the acquire for those subsystem pointers); distinct from the Ready() override.
    [[nodiscard]] auto IsReady() const noexcept -> bool {
        return HardwareReady();
    } ///< Public "hardware attached" accessor (SVDRP ATTA/DETA, mediaplayer, services).
    auto Mute() -> void override; ///< Drop the already-queued audio tail (DropOutput); persistent mute is driven
                                  ///< by VDR through SetVolumeDevice(0), not this rarely-invoked override
    auto Play() -> void override; ///< Resume normal playback: clear trick speed and unpause
    [[nodiscard]] auto Poll(cPoller &Poller, int TimeoutMs = 0)
        -> bool override; ///< Return true when at least one queue has space for more data
    auto ScaleVideo(const cRect &rect = cRect::Null)
        -> void override; ///< Stage target rect; VPP rebuilds on dim change, KMS scanout stays 1:1.
    // === Manual zoom (transient; resets to Off on every content change) ===
    [[nodiscard]] auto SetZoom(int stop)
        -> int; ///< Apply cycle stop (0=Off, 1..N=preset); returns clamped stop, rebuilds VPP on change
    [[nodiscard]] auto CycleZoom() -> int; ///< Advance Off->1->..->N->Off; returns the new stop
    auto RefreshZoom() -> void;            ///< Rebuild VPP for edited crop values without changing the active stop
    auto RefreshVideoFilters()
        -> void; ///< Rebuild VPP for policy-only setup changes (post-processing policies); leaves zoom/rect
    auto ResetZoom() -> void; ///< Force back to Off; the new stream's graph rebuild picks it up
    [[nodiscard]] auto ZoomStatusLabel() const
        -> std::string; ///< Human-readable active-zoom label for OSD/SVDRP feedback (e.g. "Zoom 2: +12.5%")
    auto SetPrimary(bool On) -> void { MakePrimaryDevice(On); } ///< Public accessor for protected MakePrimaryDevice()
    auto StillPicture(const uchar *Data, int Length)
        -> void override;                                      ///< Decode and hold a single PES frame as a still image
    auto TrickSpeed(int Speed, bool Forward) -> void override; ///< Enter trick-speed mode at the given VDR speed index

    // ========================================================================
    // === PUBLIC API ===
    // ========================================================================
    [[nodiscard]] auto Attach() -> bool; ///< Open DRM/VAAPI hardware and start decoder/display/audio threads; used
                                         ///< for the first attach after a detached startup and to resume after Detach()
    auto Detach() -> bool;               ///< Stop all threads and release DRM/VAAPI hardware; use Attach() to resume.
                           ///< Returns true iff VDR's VT was yielded so fbcon owns the text console; false
                           ///< means the hardware IS released but fbcon did not reclaim the display (user
                           ///< needs to press `Alt+F<n>`, or add CAP_SYS_TTY_CONFIG to the systemd drop-in --
                           ///< see README)
    [[nodiscard]] auto Initialize(std::string_view drmDevicePath, std::string_view audioDevicePath,
                                  std::string_view connectorNameFilter = {}, bool deferred = false)
        -> bool; ///< Latch device arguments and, unless @p deferred is true, immediately open hardware and start
                 ///< threads. When deferred, the device stays in the detached state until Attach() is called (by
                 ///< MakePrimaryDevice() on the first primary promotion, or by the SVDRP ATTA command).
    auto MarkStartupComplete() noexcept -> void; ///< Called by the plugin once VDR startup has finished. Enables the
                                                 ///< MakePrimaryDevice() deferred-attach hook so a setup.conf-driven
                                                 ///< primary promotion during VDR bring-up does not defeat --detached.

    // ========================================================================
    // === DISPLAY MODE SWITCHING ===
    // ========================================================================
    // All entry points are cheap and non-blocking: they evaluate the policy and, at most, stage a
    // request the display thread picks up. Safe from the decode thread, the player thread and the
    // VDR main thread. Everything is inert until the operator enables a scope switch.

    /// Run @p request through the policy and, if it wins a different mode, stage it on the display.
    /// @p immediate skips the stability gate -- used for the mediaplayer, where the container has
    /// already told us the authoritative frame rate before a single frame is decoded.
    auto EvaluateDisplayMode(const StreamModeRequest &request, PlaybackSource source, bool immediate) -> void;
    /// Decoder hook: publish the stream format observed after a filter-graph build. Resolves the
    /// playback source itself and never bypasses the stability gate.
    auto NotifyStreamFormat(const StreamModeRequest &request) -> void;
    /// Decode-loop tick hook: mature a candidate armed by NotifyStreamFormat() once it has been
    /// stable long enough. Required because the reactive publish fires only once per filter-graph
    /// build -- a steady stream never sends the second notification the gate would otherwise need.
    auto PollPendingDisplayMode() -> void;
    /// Re-run the last published request through the (possibly just edited) policy. Called from the
    /// setup menu so a changed option takes effect without waiting for the next stream event.
    auto ReevaluateDisplayMode() -> void;
    /// Drop a stability-gate candidate before it goes stale. Called at every stream boundary
    /// (Clear, SetPlayMode, trick entry): the gate ages on wall-clock time, so a format armed just
    /// before a channel switch would otherwise mature afterwards and reprogram the CRTC for a
    /// stream that is gone. A genuinely stable format is republished by the rebuild that follows.
    auto InvalidateDisplayModeCandidate() -> void;
    /// Playback stopped: arm a deadline after which an output still sitting on a non-default mode
    /// is handed back. Deferred rather than immediate because VDR emits pmNone right before
    /// pmAudioVideo on a channel zap; anything that starts decoding disarms it, so it only fires
    /// when playback really ended into a source that decodes nothing (radio, scrambled, no tuner).
    auto ScheduleIdleModeRestore() -> void;
    /// Return to the mode selected at attach (the --resolution one) and forget the debounce state.
    /// Called when playback ends and when the operator disables mode switching.
    auto ResetDisplayModeToDefault() -> void;
    /// Which playback path is feeding the device right now; picks the governing scope switch.
    [[nodiscard]] auto CurrentPlaybackSource() const noexcept -> PlaybackSource;
    /// Refresh rate actually programmed on the CRTC, in millihertz. Falls back to the configured
    /// --resolution rate while the display is unavailable. Reported by SVDRP STAT.
    [[nodiscard]] auto ActiveRefreshMilliHz() const noexcept -> uint32_t;
    /// Multi-line human-readable mode inventory, active mode, and current decision (SVDRP MODE).
    [[nodiscard]] auto DisplayModeReport() const -> std::string;

    // ========================================================================
    // === MEDIAPLAYER FEED SURFACE ===
    // ========================================================================
    // Narrow, encapsulated entry points for the libavformat-based mediaplayer path
    // (see src/mediaplayer.{h,cpp}). The PES path remains the only writer through
    // PlayVideo/PlayAudio; these methods exist so the mediaplayer never touches the
    // private decoder / audioProcessor pointers directly. The EOS-drain pair
    // (RequestEosDrain/PendingPlayoutDepth) is also what cDevice::Drain() runs on.
    [[nodiscard]] auto OpenForMediaPlayer(const VideoStreamInfo &video, const AudioStreamInfo &audio)
        -> bool; ///< Opens video + audio codecs with full stream descriptors. Returns false iff either codec failed.
    [[nodiscard]] auto SubmitVideoPacket(const AVPacket *packet)
        -> bool; ///< Clones a pre-demuxed video AU onto the decoder queue. False when hardware is not attached
                 ///< (HardwareReady()) or the queue is full; caller must hold the packet and retry to avoid
                 ///< silently dropping AUs while the lookahead throttle still advances as if they were accepted.
    [[nodiscard]] auto SubmitAudioPacket(const AVPacket *packet)
        -> bool; ///< Clones a pre-demuxed audio AU onto the audio queue. False when hardware is not attached
                 ///< (HardwareReady()) or the queue is full.
    auto ClearForMediaPlayer()
        -> void; ///< Heavy flush: drops queues AND tears down the filter chain. Used at open/close of an entry.
    auto RequestEosDrain()
        -> void; ///< Flush the codec reorder buffer + temporal-filter hold into the present reserve. Safe with
                 ///< packets still queued: the decode thread defers the flush until its queue has emptied.
    [[nodiscard]] auto PendingPlayoutDepth() const noexcept
        -> size_t; ///< Un-presented work: decode queue + decoded reserve + pending codec drain + display
                   ///< backlog + unplayed audio tail. EOS drains wait for 0 -- only then has the final
                   ///< PTS actually been played out.
    auto FlushForSeek()
        -> void; ///< Light flush: drops queues but keeps filter chain and swresample alive. Used at seek.
    [[nodiscard]] auto ReopenMediaPlayerAudio(const AudioStreamInfo &audio)
        -> bool; ///< Reconfigure only the audio path to a new stream mid-playback, leaving video untouched.
                 ///< Demux thread only; caller re-anchors via FlushForSeek. False on codec/ALSA failure.
    [[nodiscard]] auto IsMediaPlayerBackpressured() const noexcept
        -> bool; ///< True iff either decoder or audio queue is at capacity. Mediaplayer demux thread polls this.
    [[nodiscard]] auto IsMediaPlayerTrickReady() const noexcept
        -> bool; ///< Trick-feed gate for the mediaplayer demux (the HasFeedSpace() predicate Poll() uses on the
                 ///< PES path). Paced trick submissions must gate here, not on IsMediaPlayerBackpressured():
                 ///< the trick queue is 1 deep and EnqueuePacket() DROPS overflow.
    [[nodiscard]] auto GetAudioClock() const noexcept
        -> int64_t; ///< Audio master clock in 90 kHz ticks, or AV_NOPTS_VALUE before audio anchors / after Clear().
                    ///< Used by the mediaplayer demux to pace itself against wall-clock playback.

  protected:
    // ========================================================================
    // === VDR DEVICE OVERRIDES (protected in cDevice) ===
    // ========================================================================
    [[nodiscard]] auto CanReplay() const -> bool override; ///< True when hardware is ready and decoder is open
    auto MakePrimaryDevice(bool On) -> void override; ///< Install or remove OSD provider when becoming/leaving primary
    [[nodiscard]] auto PlayAudio(const uchar *Data, int Length, uchar Id)
        -> int override; ///< Demux one audio PES packet and enqueue for decoding
    [[nodiscard]] auto PlayVideo(const uchar *Data, int Length)
        -> int override;                         ///< Demux one video PES packet and enqueue for decoding
    [[nodiscard]] auto Ready() -> bool override; ///< VDR startup-readiness, polled only by
                                                 ///< WaitForAllDevicesReady() (30 s cap). Ready when attached OR
                                                 ///< deliberately --detached, so a deferred device doesn't pin
                                                 ///< startup. Use HardwareReady()/IsReady() for hardware state.
    auto SetAudioTrackDevice(eTrackType Type)
        -> void override; ///< Reset audio codec state and flush on track switch (live TV path)
    auto SetDigitalAudioDevice(bool On)
        -> void override; ///< Audio-track-change hook fired by cDevice::SetCurrentAudioTrack() in BOTH live and
                          ///< replay; it is the only hook that fires during replay. On=true signals a dolby-track
                          ///< switch, but fires BEFORE currentAudioTrack is assigned (VDR vdr/device.c:1172-1180),
                          ///< so GetCurrentAudioTrack() would return the stale old track. HandleAudioTrackChange()
                          ///< has a dedicated dolby walk-around for this; see its implementation.
    [[nodiscard]] auto SetPlayMode(ePlayMode PlayMode)
        -> bool override;                              ///< Reset state machine and flush on mode transitions
    auto SetVolumeDevice(int Volume) -> void override; ///< Forward VDR volume [0..255] to ALSA renderer; 0 inserts
                                                       ///< digital silence on both PCM and passthrough,
                                                       ///< intermediate values scale PCM only

  private:
    // ========================================================================
    // === INTERNAL METHODS ===
    // ========================================================================
    [[nodiscard]] auto AttachHardware() -> bool; ///< Open DRM/VAAPI and start decoder/display/audio; shared by
                                                 ///< Initialize() and Attach(). Guarded by initState CAS.
    [[nodiscard]] auto BuildRadioText(uint32_t &presentEventId) const
        -> std::string; ///< Compose the radio splash text (channel name + present EPG title) and report the present
                        ///< event id (0 = none). Takes VDR's Channels/Schedules read locks; empty text => plain black.
    auto RefreshRadioSplash(bool force)
        -> void; ///< (Re)render the radio splash: build text, submit a black frame, and record the depicted EPG event.
                 ///< force=true always paints (channel entry); force=false paints only when the present event changed.
    auto ResetNoVideoMonitors() noexcept
        -> void; ///< Clear all radio-splash + encrypted-notice state; call on every lifecycle boundary.
    auto ResetReplayAudioEofBaseline() noexcept
        -> void; ///< Clear the replay EOF-repeat baseline; call on every replay-audio timeline reset.
    [[nodiscard]] auto HasFeedSpace(int currentSpeed) const
        -> bool; ///< Poll() gate: true when the decoder can accept another packet. Trick mode (currentSpeed != 0)
                 ///< also gates on the per-frame pacing timer; normal replay gates on the packet + audio highwater.
    auto CheckEncryptionTimeout()
        -> void; ///< Driven by the decode loop's per-iteration tick (so it ticks even when a scrambled channel
                 ///< delivers no PES): once the grace elapses with nothing decoding, show the encrypted notice
                 ///< (TV or radio). Cheap no-op until armed by pmAudioVideo/pmAudioOnly.
    auto ShowEncryptedScreen()
        -> void; ///< Paint "Channel N - <name> / encrypted" over black when the current channel is encrypted (Ca != 0)
                 ///< and its primary content (video for TV, audio for radio) is not decoding; no-op otherwise.
    [[nodiscard]] auto CurrentChannelIsEncrypted() const
        -> bool; ///< True iff the current channel carries a CA id; lets the radio path defer encrypted channels to the
                 ///< encrypted watchdog instead of painting a (silent) radio splash over them.
    auto CheckRadioSplash()
        -> void; ///< Decoder-tick half of radio detection: resolves radio-vs-encrypted and refreshes the splash.
                 ///< PlayAudio() only raises radioCheckPending -- it runs on the receiver thread under
                 ///< cDevice::mutexReceiver, where taking the Channels lock inverts against
                 ///< GetDevice()->Priority() (mutexReceiver under the Channels lock).
    auto HandleAudioTrackChange(const char *reason, bool enteringDolby)
        -> void; ///< Log + re-detect audio on track change. @p enteringDolby works around VDR firing the hook
                 ///< BEFORE assigning currentAudioTrack from SetDigitalAudioDevice(true).
    [[nodiscard]] auto OpenHardware() -> bool; ///< Open DRM fd, create VAAPI hw device context, find render node
    [[nodiscard]] auto PlayTrickAudio(const uchar *Data, int Length)
        -> int; ///< PlayAudio()'s trick branch for an audio-only replay: paces cDvbPlayer's feed and latches the
                ///< step PTS for GetSTC(); audio stays dropped. PlayAudio()'s return contract (0 = not due yet).
    [[nodiscard]] auto ProbeVppCapabilities(std::string_view renderNode)
        -> bool;                         ///< Query VAAPI decode profiles and VPP filter capabilities
    auto ReleaseHardware() -> void;      ///< Close VAAPI device reference and DRM file descriptor
    auto ResetAudioCodecState() -> void; ///< Drop the cached audio codec id and any in-flight 2-of-2 confirmation state
                                         ///< so the next PlayAudio() packet re-runs codec detection
    auto ApplyDisplayModePolicy(const StreamModeRequest &request, PlaybackSource source, uint64_t nowMs)
        -> void; ///< Run the matcher and stage the winner. Caller holds displayModeMutex and has already
                 ///< cleared the scope/policy/stability gates; this applies only the rate limit.
    auto RestoreDefaultModeLocked(uint64_t nowMs)
        -> void; ///< Hand the output back to defaultMode when mode switching is not in charge of the
                 ///< current source. Caller holds displayModeMutex.
    auto ArmModeCandidateLocked(const StreamModeRequest &request, uint64_t nowMs)
        -> void; ///< Start the stability window for @p request. Caller holds displayModeMutex.
    auto ClearModeCandidateLocked()
        -> void; ///< Disarm the stability window. Caller holds displayModeMutex. Both helpers exist so
                 ///< modeCandidateDueMs can never drift from modeCandidateSinceMs.
    [[nodiscard]] auto SelectDrmConnector()
        -> bool; ///< Scan connectors, pick a display mode, and store crtcId/connectorId
    [[nodiscard]] auto TryAcceptConnector(drmModeConnector *connector, bool allowModeFallback, drmModeRes *resources)
        -> bool; ///< SelectDrmConnector() helper: validate one connector, pick its mode (exact; else, when
                 ///< allowModeFallback, PREFERRED then first) and latch activeMode/crtcId/connectorId/connectorName.
    auto Stop() -> void; ///< Shut down decoder, display, and audio in dependency order
    [[nodiscard]] auto SubmitBlackFrame(std::string_view centerText = {})
        -> bool; ///< Submit a VAAPI NV12 black surface (optional centered text baked into the luma plane);
                 ///< false means it never reached the display queue (the caller may retry).
    auto SuspendHardware() -> void; ///< Release DRM/VAAPI/ALSA/OSD without touching cControl or VT;
                                    ///< shared by Detach() (SVDRP DETA) and SetPlayMode(pmExtern).

    // ========================================================================
    // === STATE ===
    // ========================================================================
    drmModeModeInfo activeMode{};                          ///< Selected DRM display mode
    drmModeModeInfo defaultMode{};                         ///< Mode chosen at attach; the mode-matcher's fallback and
                                                           ///< the target ResetDisplayModeToDefault() returns to
    std::vector<drmModeModeInfo> connectorModes;           ///< Full mode list of the selected connector, captured
                                                           ///< during the attach scan. Never re-enumerated: a fresh
                                                           ///< drmModeGetConnector() forces DDC and costs 100-500 ms
                                                           ///< per port (seconds on a CEC-standby sink).
    std::vector<DisplayModeCandidate> modeCandidates;      ///< connectorModes digested for matching
    std::atomic<AVCodecID> audioCodecId{AV_CODEC_ID_NONE}; ///< Active audio codec
    std::string audioDevice;                               ///< ALSA device name
    std::unique_ptr<cAudioProcessor> audioProcessor;       ///< Threaded ALSA renderer
    uint32_t connectorId{};                                ///< DRM connector ID
    std::string connectorName;                             ///< Selected connector: -c or auto-latched; empty = auto
    bool connectorUserSupplied{false};                     ///< connectorName came from -c (sticky); auto-latched
                                                           ///< names are cleared on Suspend to re-select on re-attach
    uint32_t crtcId{};                                     ///< DRM CRTC ID
    std::unique_ptr<cVaapiDecoder> decoder;                ///< Threaded VAAPI decoder
    std::unique_ptr<cVaapiDisplay> display;                ///< DRM page-flip display manager
    int drmFd{-1};                                         ///< DRM primary node fd
    std::string drmPath;                                   ///< DRM primary device path
    std::atomic<int> initState{0};                         ///< 0=detached, 1=pending, 2=ready
    std::atomic<bool> externActive{false};                 ///< True between SetPlayMode(pmExtern) and the next
                                                           ///< SetPlayMode call; gates the resume-Attach path.
    std::atomic<bool> startupComplete{false};              ///< Gates deferred-attach against --detached
    std::atomic<bool> liveMode{false};                     ///< True in Transfer Mode (live TV)
    std::atomic<bool> mediaPlayerAudioActive{false};       ///< Gates HandleAudioTrackChange off while the
                                                           ///< mediaplayer owns audio (track switches go via
                                                           ///< cVaapiPlayer::SetAudioTrack, not the live-TV reset)
    int osdHeight{};                                       ///< Cached display height (px)
    int osdWidth{};                                        ///< Cached display width (px)
    uint64_t osdModeGeneration{};                          ///< Display mode generation the osdWidth/osdHeight cache
                                                           ///< was taken at; a mismatch invalidates it so VDR's 1 Hz
                                                           ///< UpdateOsdSize() poll sees the new size after a modeset

    // --- Display mode switching (all guarded by displayModeMutex) ---
    mutable cMutex displayModeMutex;   ///< Serializes the debounce state below; EvaluateDisplayMode() is
                                       ///< reached from the decode thread, the player thread and the main thread
    StreamModeRequest modeCandidate{}; ///< Format currently accumulating toward the stability gate
    uint64_t modeCandidateSinceMs{};   ///< When modeCandidate was first observed (0 = none pending)
    /// Lock-free mirror of modeCandidateSinceMs + DISPLAY_MODE_STABLE_MS; 0 = nothing armed. PollPendingDisplayMode()
    /// runs on every decode-loop iteration, so the (overwhelmingly common) no-candidate case must not cost a mutex
    /// acquisition.
    std::atomic<uint64_t> modeCandidateDueMs{0};
    /// Wall clock at which an idle output on a non-default mode is handed back to defaultMode; 0 = disarmed. Covers
    /// playback ending into a source that decodes nothing (radio, scrambled, no free tuner), where no format is ever
    /// published to drive the restore.
    std::atomic<uint64_t> modeIdleRestoreDueMs{0};
    uint64_t lastModeChangeMs{};     ///< Wall clock of the last applied change; enforces the minimum interval
    StreamModeRequest lastRequest{}; ///< Most recently published format; replayed by ReevaluateDisplayMode()
    PlaybackSource lastRequestSource{PlaybackSource::LiveTv};     ///< Source that published lastRequest
    std::atomic<AVCodecID> audioCodecCandidate{AV_CODEC_ID_NONE}; ///< Pending 2-of-2 audio codec confirm
    std::atomic<int> audioCodecCandidateCount{0};                 ///< Confirmation count for audioCodecCandidate
    std::vector<uint8_t> audioDetectBuffer;   ///< AAC-LATM fallback window for DetectAudioCodec() (see
                                              ///< AUDIO_DETECT_WINDOW). Owned solely by the PlayAudio feed thread;
                                              ///< the reset paths never touch it (see audioDetectGen).
    uint32_t audioDetectGenSeen{};            ///< PlayAudio's last-seen audioDetectGen; a mismatch clears the window
    std::atomic<uint32_t> audioDetectGen{0};  ///< Bumped by ResetAudioCodecState() to invalidate the window across
                                              ///< threads without racing the vector
    std::atomic<unsigned> clearsSinceLog{0};  ///< Clear()s coalesced since the last burst log
    std::atomic<uint64_t> lastClearLogMs{0};  ///< Walltime of the last Clear() diagnostic log (rate-limit)
    std::atomic<uint64_t> lastClearMs{0};     ///< Last Clear() timestamp (diagnostic)
    eTrackType lastHandledAudioTrack{ttNone}; ///< (with lastHandledAudioPid) dedup track-change
    uint16_t lastHandledAudioPid{};           ///<   hooks during PMT churn
    std::atomic<bool> paused{false};          ///< True while frozen via Freeze()
#if APIVERSNUM >= 30014
    /// One RequestEosDrain() per Drain() cycle; Clear() and SetPlayMode() cancel and re-arm it (Drain() contract).
    std::atomic<bool> eosDrainRequested{false};
#endif
    /// Last confirmed audio codec; survives Clear() so a same-codec re-detect after a scrub seek logs nothing
    std::atomic<AVCodecID> previousAudioCodec{AV_CODEC_ID_NONE};
    std::atomic<AVCodecID> previousVideoCodec{AV_CODEC_ID_NONE}; ///< Previous channel's video codec (stale guard)
    bool inStillPicture{false};                                  ///< Re-entry guard for cDevice::StillPicture
    std::atomic<bool> radioBlackPending{false};                  ///< Awaiting radio-only channel detection
    cTimeMs radioBlackTimer;                                     ///< Radio-mode detection timeout
    /// PlayAudio saw no video after the grace; CheckRadioSplash resolves it
    std::atomic<bool> radioCheckPending{false};
    std::atomic<bool> radioSplashActive{false}; ///< A refreshable radio (no-video) splash is on screen
    /// EPG id last queued into the radio splash; top-of-range sentinels = empty/dirty
    std::atomic<uint32_t> radioSplashEventId{0};

    cTimeMs radioSplashPoll; ///< Next EPG re-check; touched only on the decoder tick thread (CheckRadioSplash)
    /// Encrypted-notice watchdog, armed on pmAudioVideo/pmAudioOnly: grace deadline on the cTimeMs::Now() clock; 0 =
    /// disarmed. Deliberately ONE word: the decoder tick thread outlives play modes, so it disarms/claims via CAS
    /// against the deadline it observed -- a concurrent re-arm stores a strictly-future value an expired observation
    /// never matches, making it impossible to cancel a fresh arm (see CheckEncryptionTimeout).
    std::atomic<uint64_t> encryptedDeadlineMs{0};
    /// PTS of the last replay PES fed to the decoder, 90 kHz. At EOF cDvbPlayer re-pushes the last PES; decoding the
    /// repeats keeps the DAC clock alive, so radio replay never hits VDR's StuckAtEof. Dropping an exact PTS repeat
    /// lets the clock stall instead. Reset via ResetReplayAudioEofBaseline() on every replay-audio timeline break.
    std::atomic<int64_t> lastReplayAudioPts{AV_NOPTS_VALUE};
    /// PTS of the last step PlayTrickAudio() let through, 90 kHz; AV_NOPTS_VALUE = none. Serves as both the audio-only
    /// replay's trick STC (read only while trickSpeed != 0) and the pacing hold's previous-step reference. Reset by
    /// Clear() and TrickSpeed().
    std::atomic<int64_t> trickAudioPts{AV_NOPTS_VALUE};
    std::atomic<int> trickSpeed{0};                               ///< VDR trick speed; 0 = normal
    VaapiContext vaapi{};                                         ///< Shared VAAPI context
    std::atomic<AVCodecID> videoCodecCandidate{AV_CODEC_ID_NONE}; ///< Pending 2-of-2 video codec confirm
    std::atomic<int> videoCodecCandidateCount{0};                 ///< Confirmation count for videoCodecCandidate
    std::atomic<AVCodecID> videoCodecId{AV_CODEC_ID_NONE};        ///< Active video codec
};

#endif // VDR_VAAPIVIDEO_DEVICE_H
