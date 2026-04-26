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

class cAudioProcessor;
class cVaapiDecoder;
class cVaapiDisplay;

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
    [[nodiscard]] auto IsReady() -> bool { return Ready(); } ///< Public accessor for protected Ready()
    auto Mute() -> void override; ///< Drop pending audio frames (hardware mute handled by VDR)
    auto Play() -> void override; ///< Resume normal playback: clear trick speed and unpause
    [[nodiscard]] auto Poll(cPoller &Poller, int TimeoutMs = 0)
        -> bool override; ///< Return true when at least one queue has space for more data
    auto SetPrimary(bool On) -> void { MakePrimaryDevice(On); } ///< Public accessor for protected MakePrimaryDevice()
    auto StillPicture(const uchar *Data, int Length)
        -> void override;                                      ///< Decode and hold a single PES frame as a still image
    auto TrickSpeed(int Speed, bool Forward) -> void override; ///< Enter trick-speed mode at the given VDR speed index

    // ========================================================================
    // === PUBLIC API ===
    // ========================================================================
    [[nodiscard]] auto Attach() -> bool; ///< Open DRM/VAAPI hardware and start decoder/display/audio threads; used
                                         ///< for the first attach after a detached startup and to resume after Detach()
    auto Detach() -> bool; ///< Stop all threads and release DRM/VAAPI hardware; use Attach() to resume. Returns
                           ///< true iff the VT bounce restored the text console; false means the hardware IS
                           ///< released but fbcon did not reclaim the display (user needs to press Alt+F1, or
                           ///< add the 'tty' group + CAP_SYS_TTY_CONFIG + the /dev/tty0 udev rule -- see README)
    [[nodiscard]] auto Initialize(std::string_view drmDevicePath, std::string_view audioDevicePath,
                                  std::string_view connectorNameFilter = {}, bool deferred = false)
        -> bool; ///< Latch device arguments and, unless @p deferred is true, immediately open hardware and start
                 ///< threads. When deferred, the device stays in the detached state until Attach() is called (by
                 ///< MakePrimaryDevice() on the first primary promotion, or by the SVDRP ATTA command).
    auto MarkStartupComplete() noexcept -> void; ///< Called by the plugin once VDR startup has finished. Enables the
                                                 ///< MakePrimaryDevice() deferred-attach hook so a setup.conf-driven
                                                 ///< primary promotion during VDR bring-up does not defeat --detached.

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
    [[nodiscard]] auto Ready() -> bool override; ///< True once DRM/VAAPI/ALSA hardware is attached and the
                                                 ///< decoder, display, and audio subsystems are initialized
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
    [[nodiscard]] auto AttachHardware()
        -> bool; ///< Open DRM/VAAPI hardware and start the decoder, display, and audio subsystems. Guarded by the
                 ///< initState CAS so double-attach is a no-op error. Requires drmPath to be populated (latched by
                 ///< Initialize()). Internal implementation shared by Initialize() (non-deferred) and Attach().
    auto HandleAudioTrackChange(const char *reason, bool enteringDolby)
        -> void; ///< Log + run full audio re-detection on track change. @p enteringDolby is set when called from
                 ///< SetDigitalAudioDevice(true), where VDR fires the hook BEFORE assigning currentAudioTrack and
                 ///< the read would otherwise return the OLD (stale) selection.
    [[nodiscard]] auto OpenHardware() -> bool; ///< Open DRM fd, create VAAPI hw device context, find render node
    [[nodiscard]] auto ProbeVppCapabilities(std::string_view renderNode)
        -> bool;                         ///< Query VAAPI decode profiles and VPP filter capabilities
    auto ReleaseHardware() -> void;      ///< Close VAAPI device reference and DRM file descriptor
    auto ResetAudioCodecState() -> void; ///< Drop the cached audio codec id and any in-flight 2-of-2 confirmation state
                                         ///< so the next PlayAudio() packet re-runs codec detection
    [[nodiscard]] auto SelectDrmConnector()
        -> bool;                     ///< Scan connectors, pick a display mode, and store crtcId/connectorId
    auto Stop() -> void;             ///< Shut down decoder, display, and audio in dependency order
    auto SubmitBlackFrame() -> void; ///< Allocate a VAAPI NV12 black surface and submit it through the display pipeline

    // ========================================================================
    // === STATE ===
    // ========================================================================
    drmModeModeInfo activeMode{};                          ///< DRM display mode selected by SelectDrmConnector()
    std::atomic<AVCodecID> audioCodecId{AV_CODEC_ID_NONE}; ///< Codec active for the current audio stream
    std::string audioDevice;                               ///< ALSA device name (e.g. "default", "hw:0,3")
    std::unique_ptr<cAudioProcessor> audioProcessor;       ///< Threaded ALSA renderer
    uint32_t connectorId{};                                ///< DRM connector ID chosen by SelectDrmConnector()
    std::string connectorName;                             ///< User-requested connector (e.g. "HDMI-A-1"); empty = auto
    uint32_t crtcId{};                                     ///< DRM CRTC ID associated with the selected connector
    std::unique_ptr<cVaapiDecoder> decoder;                ///< Threaded VAAPI packet decoder
    std::unique_ptr<cVaapiDisplay> display;                ///< DRM atomic page-flip display manager
    int drmFd{-1};                                         ///< Open file descriptor for the DRM primary node
    std::string drmPath;                      ///< Path to the DRM primary device node (e.g. "/dev/dri/card0")
    std::atomic<int> initState;               ///< Init state machine: 0=detached/uninit, 1=pending, 2=ready
    std::atomic<bool> startupComplete{false}; ///< True after plugin Start() returns; gates the MakePrimaryDevice()
                                              ///< deferred-attach hook so setup.conf-driven primary promotion at
                                              ///< VDR bring-up cannot defeat --detached
    std::atomic<bool> liveMode;               ///< True in Transfer Mode (live TV); false during replay
    int osdHeight{};                          ///< Cached display height for OSD allocation (pixels)
    int osdWidth{};                           ///< Cached display width for OSD allocation (pixels)
    AVCodecID audioCodecCandidate{AV_CODEC_ID_NONE}; ///< Candidate audio codec pending confirmation
    int audioCodecCandidateCount{};                  ///< Consecutive detections of candidate audio codec
    std::atomic<uint64_t> lastClearMs{0};     ///< Timestamp of last Clear() for diagnostic delta logging; 0 = never
    eTrackType lastHandledAudioTrack{ttNone}; ///< Dedup pair (with lastHandledAudioPid): skip track change if
    uint16_t lastHandledAudioPid{};           ///<   both (type, PID) match; suppresses resets during PMT churn
    std::atomic<bool> paused;                 ///< True while playback is frozen via Freeze()
    AVCodecID previousAudioCodec{AV_CODEC_ID_NONE};  ///< Codec from previous channel (stale-data guard)
    AVCodecID previousVideoCodec{AV_CODEC_ID_NONE};  ///< Codec from previous channel (stale-data guard)
    bool inStillPicture{false};                      ///< Re-entry guard: true while cDevice::StillPicture re-enters
    bool radioBlackPending{false};                   ///< True while waiting to detect radio-only channel
    cTimeMs radioBlackTimer;                         ///< Timeout for radio-mode detection (no video -> black frame)
    std::atomic<int> trickSpeed;                     ///< Active VDR trick speed index; 0 = normal playback
    VaapiContext vaapi{};                            ///< Shared VAAPI context (hwDeviceRef + drmFd borrow)
    AVCodecID videoCodecCandidate{AV_CODEC_ID_NONE}; ///< Candidate video codec pending confirmation
    int videoCodecCandidateCount{};                  ///< Consecutive detections of candidate video codec
    std::atomic<AVCodecID> videoCodecId{AV_CODEC_ID_NONE}; ///< Codec active for the current video stream
};

#endif // VDR_VAAPIVIDEO_DEVICE_H
