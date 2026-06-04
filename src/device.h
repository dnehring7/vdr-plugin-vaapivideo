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
// === DRM DEVICES CLASS ===
// ============================================================================

/**
 * @class DrmDevices
 * @brief Enumerates all DRM devices visible to the system via libdrm.
 *
 * Used to pick a primary node when no explicit device path is configured.
 * Iteration follows the standard range-for protocol via begin()/end().
 */
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

/**
 * @class cVaapiDevice
 * @brief Primary VDR output device: demuxes PES, decodes via VAAPI, and renders over DRM/KMS.
 *
 * Lifecycle: Initialize() latches device arguments and either attaches hardware immediately
 * or leaves the device detached (--detached). Once attached, VDR routes
 * SetPlayMode/PlayVideo/PlayAudio here; Stop() tears the subsystems down.
 * All playback state transitions (pause, trick speed, track switch) are handled inline.
 */
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
        -> bool override; ///< Always true: all I/B/P frame types are submitted in trick mode
    [[nodiscard]] auto HardwareReady() const noexcept -> bool {
        return initState.load(std::memory_order_acquire) == 2;
    } ///< True once hardware is attached and decoder/display/audio are live -- the plugin's real
      ///< "usable" gate (and the acquire for those subsystem pointers); distinct from the Ready() override.
    [[nodiscard]] auto IsReady() const noexcept -> bool {
        return HardwareReady();
    } ///< Public "hardware attached" accessor (SVDRP ATTA/DETA, mediaplayer, services).
    auto Mute() -> void override; ///< Drop pending audio frames (hardware mute handled by VDR)
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
    auto ResetZoom() -> void;              ///< Force back to Off; the new stream's graph rebuild picks it up
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
                           ///< needs to press Alt+F<n>, or add CAP_SYS_TTY_CONFIG to the systemd drop-in --
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
    // === MEDIAPLAYER FEED SURFACE ===
    // ========================================================================
    // Narrow, encapsulated entry points for the libavformat-based mediaplayer path
    // (see src/mediaplayer.{h,cpp}). The PES path remains the only writer through
    // PlayVideo/PlayAudio; these methods exist so the mediaplayer never touches the
    // private decoder / audioProcessor pointers directly.
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
    auto RequestMediaPlayerEosDrain()
        -> void; ///< Flush the codec reorder buffer + temporal-filter hold into the present reserve. Call only once
                 ///< the decode queue is drained (MediaPlayerDecodeQueueDepth()): a mid-queue flush re-arms the
                 ///< codec, leaving the remaining non-keyframe packets undecodable until the next I-frame.
    [[nodiscard]] auto MediaPlayerDecodeQueueDepth() const noexcept
        -> size_t; ///< Packets still waiting in the decode queue (0 when closed). Mediaplayer EOS-drain phase 1.
    [[nodiscard]] auto MediaPlayerBufferedDepth() const noexcept
        -> size_t; ///< Total un-presented work: decode queue + decoded reserve + audio pending work + pending codec
                   ///< drain. Mediaplayer EOS-drain phase 2 waits for this to reach 0 before teardown.
    auto FlushForSeek()
        -> void; ///< Light flush: drops queues but keeps filter chain and swresample alive. Used at seek.
    [[nodiscard]] auto IsMediaPlayerBackpressured() const noexcept
        -> bool; ///< True iff either decoder or audio queue is at capacity. Mediaplayer demux thread polls this.
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
    auto SetVolumeDevice(int Volume) -> void override; ///< Forward PCM volume [0..255] to ALSA renderer

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
    auto HandleAudioTrackChange(const char *reason, bool enteringDolby)
        -> void; ///< Log + re-detect audio on track change. @p enteringDolby works around VDR firing the hook
                 ///< BEFORE assigning currentAudioTrack from SetDigitalAudioDevice(true).
    [[nodiscard]] auto OpenHardware() -> bool; ///< Open DRM fd, create VAAPI hw device context, find render node
    [[nodiscard]] auto ProbeVppCapabilities(std::string_view renderNode)
        -> bool;                         ///< Query VAAPI decode profiles and VPP filter capabilities
    auto ReleaseHardware() -> void;      ///< Close VAAPI device reference and DRM file descriptor
    auto ResetAudioCodecState() -> void; ///< Drop the cached audio codec id and any in-flight 2-of-2 confirmation state
                                         ///< so the next PlayAudio() packet re-runs codec detection
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
    std::atomic<int> initState;                            ///< 0=detached, 1=pending, 2=ready
    std::atomic<bool> externActive{false};                 ///< True between SetPlayMode(pmExtern) and the next
                                                           ///< SetPlayMode call; gates the resume-Attach path.
    std::atomic<bool> startupComplete{false};              ///< Gates deferred-attach against --detached
    std::atomic<bool> liveMode;                            ///< True in Transfer Mode (live TV)
    int osdHeight{};                                       ///< Cached display height (px)
    int osdWidth{};                                        ///< Cached display width (px)
    std::atomic<AVCodecID> audioCodecCandidate{AV_CODEC_ID_NONE}; ///< Pending 2-of-2 audio codec confirm
    std::atomic<int> audioCodecCandidateCount;                    ///< Confirmation count for audioCodecCandidate
    std::atomic<unsigned> clearsSinceLog{0};                      ///< Clear()s coalesced since the last burst log
    std::atomic<uint64_t> lastClearLogMs{0};  ///< Walltime of the last Clear() diagnostic log (rate-limit)
    std::atomic<uint64_t> lastClearMs{0};     ///< Last Clear() timestamp (diagnostic)
    eTrackType lastHandledAudioTrack{ttNone}; ///< (with lastHandledAudioPid) dedup track-change
    uint16_t lastHandledAudioPid{};           ///<   hooks during PMT churn
    std::atomic<bool> paused;                 ///< True while frozen via Freeze()
    std::atomic<AVCodecID> previousAudioCodec{
        AV_CODEC_ID_NONE}; ///< Last confirmed audio codec; survives Clear() so a
                           ///< same-codec re-detect after a scrub seek logs nothing
    std::atomic<AVCodecID> previousVideoCodec{AV_CODEC_ID_NONE}; ///< Previous channel's video codec (stale guard)
    bool inStillPicture{false};                                  ///< Re-entry guard for cDevice::StillPicture
    std::atomic<bool> radioBlackPending{false};                  ///< Awaiting radio-only channel detection
    cTimeMs radioBlackTimer;                                     ///< Radio-mode detection timeout
    std::atomic<bool> radioSplashActive{false};                  ///< A refreshable radio (no-video) splash is on screen
    std::atomic<uint32_t> radioSplashEventId{
        0}; ///< EPG id last queued into the radio splash; top-of-range sentinels = empty/dirty

    cTimeMs radioSplashPoll;                   ///< Next EPG re-check; touched only on the PlayAudio thread
    std::atomic<bool> encryptedPending{false}; ///< Armed on pmAudioVideo; monitors for an undecodable (encrypted) video
                                               ///< stream. One-shot: cleared when video decodes, when the notice fires,
                                               ///< or on the next SetPlayMode.
    cTimeMs encryptedTimer;                    ///< Grace period before declaring a non-decoding channel encrypted
    std::atomic<int> trickSpeed;               ///< VDR trick speed; 0 = normal
    VaapiContext vaapi{};                      ///< Shared VAAPI context
    std::atomic<AVCodecID> videoCodecCandidate{AV_CODEC_ID_NONE}; ///< Pending 2-of-2 video codec confirm
    std::atomic<int> videoCodecCandidateCount;                    ///< Confirmation count for videoCodecCandidate
    std::atomic<AVCodecID> videoCodecId{AV_CODEC_ID_NONE};        ///< Active video codec
};

#endif // VDR_VAAPIVIDEO_DEVICE_H
