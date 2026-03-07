// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file device.h
 * @brief VDR device integration, PES routing, and lifecycle
 */

#ifndef VDR_VAAPIVIDEO_DEVICE_H
#define VDR_VAAPIVIDEO_DEVICE_H

#include "common.h"

class cAudioProcessor;
class cVaapiDecoder;
class cVaapiDisplay;

// ============================================================================
// === STRUCTURES ===
// ============================================================================

/// Shared VAAPI hardware context passed to decoder and display subsystems.
/// Decode profile flags and VPP filter capabilities are probed once at device
/// init by ProbeVppCapabilities() and cached here for the decoder's filter graph.
struct VaapiContext {
    AVBufferRef *hwDeviceRef{};  ///< VAAPI hardware device (owned, freed via av_buffer_unref())
    int drmFd{-1};               ///< DRM file descriptor (borrowed -- owned by cVaapiDevice)
    bool hasDenoise{};           ///< VAProcFilterNoiseReduction supported
    bool hasSharpness{};         ///< VAProcFilterSharpening supported
    bool hwH264{};               ///< GPU supports VAAPI hardware H.264 decode
    bool hwHevc{};               ///< GPU supports VAAPI hardware HEVC decode
    bool hwMpeg2{};              ///< GPU supports VAAPI hardware MPEG-2 decode
    std::string deinterlaceMode; ///< Best deinterlace mode name (empty = none)
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
    int deviceCount{0};                   ///< Number of DRM devices found by the last Enumerate() call
    std::vector<drmDevicePtr> deviceList; ///< Pointers to enumerated DRM device descriptors (freed in destructor)
};

// ============================================================================
// === VAAPI DEVICE CLASS ===
// ============================================================================

/**
 * @class cVaapiDevice
 * @brief Primary VDR output device: demuxes PES, decodes via VAAPI, and renders over DRM/KMS.
 *
 * Lifecycle: Initialize() opens hardware -> VDR calls SetPlayMode/PlayVideo/PlayAudio -> Stop() tears down.
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
    // === VDR DEVICE INTERFACE ===
    // ========================================================================
    [[nodiscard]] auto CanReplay() const -> bool override; ///< True when hardware is ready and decoder is open
    auto Clear() -> void override; ///< Flush decoder and audio queues without releasing hardware
    [[nodiscard]] auto DeviceType() const -> cString override; ///< Returns "VAAPI"
    [[nodiscard]] auto Flush(int TimeoutMs = 0)
        -> bool override;           ///< Wait until packet queue drains; returns true when empty
    auto Freeze() -> void override; ///< Pause output: drain queue and stop audio
    auto GetOsdSize(int &Width, int &Height, double &PixelAspect)
        -> void override;                            ///< Return display framebuffer dimensions for OSD allocation
    [[nodiscard]] auto GetSTC() -> int64_t override; ///< Return presentation clock in VDR 90 kHz ticks
    auto GetVideoSize(int &Width, int &Height, double &VideoAspect)
        -> void override;                                   ///< Return active video stream resolution
    [[nodiscard]] auto HasDecoder() const -> bool override; ///< True when a VAAPI codec context is open and ready
    [[nodiscard]] auto HasIBPTrickSpeed()
        -> bool override; ///< Always true: all I/B/P frame types are submitted in trick mode
    auto MakePrimaryDevice(bool On) -> void override; ///< Install or remove OSD provider when becoming/leaving primary
    auto Mute() -> void override;                     ///< Drop pending audio frames (hardware mute handled by VDR)
    auto Play() -> void override;                     ///< Resume normal playback: clear trick speed and unpause
    [[nodiscard]] auto PlayAudio(const uchar *Data, int Length, uchar Id)
        -> int override; ///< Demux one audio PES packet and enqueue for decoding
    [[nodiscard]] auto PlayVideo(const uchar *Data, int Length)
        -> int override; ///< Demux one video PES packet and enqueue for decoding
    [[nodiscard]] auto Poll(cPoller &Poller, int TimeoutMs = 0)
        -> bool override;                        ///< Return true when at least one queue has space for more data
    [[nodiscard]] auto Ready() -> bool override; ///< True once Initialize() has completed successfully
    auto SetAudioTrackDevice(eTrackType Type)
        -> void override; ///< Reset audio codec detection on user-initiated track switch
    [[nodiscard]] auto SetPlayMode(ePlayMode PlayMode)
        -> bool override;                              ///< Reset state machine and flush on mode transitions
    auto SetVolumeDevice(int Volume) -> void override; ///< Forward PCM volume [0..255] to ALSA renderer
    auto StillPicture(const uchar *Data, int Length)
        -> void override;                                      ///< Decode and hold a single PES frame as a still image
    auto TrickSpeed(int Speed, bool Forward) -> void override; ///< Enter trick-speed mode at the given VDR speed index

    // ========================================================================
    // === PUBLIC API ===
    // ========================================================================
    [[nodiscard]] auto Attach() -> bool; ///< Re-open hardware and restart all threads after a prior Detach()
    auto Detach() -> void;               ///< Stop all threads and release DRM/VAAPI hardware; use Attach() to resume
    [[nodiscard]] auto Initialize(std::string_view drmDevicePath, std::string_view audioDevicePath)
        -> bool; ///< Open DRM/VAAPI hardware and start decoder, display, and audio threads

  private:
    // ========================================================================
    // === INTERNAL METHODS ===
    // ========================================================================
    [[nodiscard]] auto OpenHardware() -> bool;         ///< Open DRM fd, create VAAPI device context, log codec support
    [[nodiscard]] auto ProbeVppCapabilities() -> bool; ///< Query VAAPI decode profiles and VPP filter capabilities
    auto ReleaseHardware() -> void;                    ///< Close VAAPI device reference and DRM file descriptor
    [[nodiscard]] auto SelectDrmConnector()
        -> bool;         ///< Scan connectors, pick a display mode, and store crtcId/connectorId
    auto Stop() -> void; ///< Shut down decoder, display, and audio in dependency order

    // ========================================================================
    // === STATE ===
    // ========================================================================
    drmModeModeInfo activeMode{};                          ///< DRM display mode selected by SelectDrmConnector()
    std::atomic<AVCodecID> audioCodecId{AV_CODEC_ID_NONE}; ///< Codec active for the current audio stream
    std::string audioDevice;                               ///< ALSA device name (e.g. "default", "hw:0,3")
    std::unique_ptr<cAudioProcessor> audioProcessor;       ///< Threaded ALSA renderer
    AVCodecID codecHysteresis{AV_CODEC_ID_NONE};           ///< Candidate codec pending hysteresis confirmation
    int codecHysteresisCount{};                            ///< Consecutive detections of the candidate codec
    uint32_t connectorId{};                                ///< DRM connector ID chosen by SelectDrmConnector()
    uint32_t crtcId{};                                     ///< DRM CRTC ID associated with the selected connector
    std::unique_ptr<cVaapiDecoder> decoder;                ///< Threaded VAAPI packet decoder
    std::unique_ptr<cVaapiDisplay> display;                ///< DRM atomic page-flip display manager
    int drmFd{-1};                                         ///< Open file descriptor for the DRM primary node
    std::string drmPath;                             ///< Path to the DRM primary device node (e.g. "/dev/dri/card0")
    std::atomic<int> initState;                      ///< Init state machine: 0=uninit, 1=pending, 2=ready
    bool liveMode{};                                 ///< True in Transfer Mode (live TV); false during replay
    int osdHeight{};                                 ///< Cached display height for OSD allocation (pixels)
    int osdWidth{};                                  ///< Cached display width for OSD allocation (pixels)
    std::atomic<bool> paused;                        ///< True while playback is frozen via Freeze()
    std::atomic<uchar> prevAudioStreamId{0xFF};      ///< Stream ID of last audio PES; 0xFF = none seen yet
    AVCodecID previousVideoCodec{AV_CODEC_ID_NONE};  ///< Codec from previous channel (stale-data guard)
    std::atomic<int> trickSpeed;                     ///< Active VDR trick speed index; 0 = normal playback
    VaapiContext vaapi{};                            ///< Shared VAAPI context (hwDeviceRef + drmFd borrow)
    AVCodecID videoCodecCandidate{AV_CODEC_ID_NONE}; ///< Candidate video codec pending confirmation
    int videoCodecCandidateCount{};                  ///< Consecutive detections of candidate video codec
    std::atomic<AVCodecID> videoCodecId{AV_CODEC_ID_NONE}; ///< Codec active for the current video stream
};

#endif // VDR_VAAPIVIDEO_DEVICE_H
