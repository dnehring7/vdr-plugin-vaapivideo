// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file display.h
 * @brief Zero-copy VAAPI->DRM display: PRIME import, atomic modesetting, and page-flip pacing.
 */

#ifndef VDR_VAAPIVIDEO_DISPLAY_H
#define VDR_VAAPIVIDEO_DISPLAY_H

#include "caps.h"
#include "common.h"
#include "config.h"

struct VaapiFrame;

// ============================================================================
// === ATOMIC REQUEST ===
// ============================================================================

/// RAII wrapper around a DRM atomic request. Accumulates (object, property, value) triples
/// for a single atomic commit. Move-only.
class AtomicRequest {
  public:
    AtomicRequest();
    ~AtomicRequest() noexcept;
    AtomicRequest(const AtomicRequest &) = delete;
    auto operator=(const AtomicRequest &) -> AtomicRequest & = delete;
    AtomicRequest(AtomicRequest &&other) noexcept;
    auto operator=(AtomicRequest &&other) noexcept -> AtomicRequest &;

    // ========================================================================
    // === PUBLIC API ===
    // ========================================================================
    auto AddProperty(uint32_t objId, uint32_t propId, uint64_t value)
        -> void; ///< Append (objId, propId, value); silently skips propId==0 (optional property absent on this driver)
    [[nodiscard]] auto Count() const noexcept -> int;
    [[nodiscard]] auto Handle() const noexcept -> drmModeAtomicReq *;

  private:
    // ========================================================================
    // === STATE ===
    // ========================================================================
    int propCount{};             ///< Properties accumulated so far
    drmModeAtomicReq *request{}; ///< Owned DRM atomic request handle
};

// ============================================================================
// === VAAPI DISPLAY ===
// ============================================================================

/// Zero-copy VAAPI->DRM display. Imports decoded surfaces as PRIME framebuffers and
/// pages them at the monitor's refresh rate. OSD overlay is composited atomically in
/// the same commit (single vblank for both planes).
///
/// Thread safety: SubmitFrame(), SetOsd(), BeginStreamSwitch(), EndStreamSwitch() are
/// safe from any thread. Initialize() and Shutdown() must be called from the same thread.
/// Lock order: importMutex -> bufferMutex -> osdMutex (see display.cpp file header).
class cVaapiDisplay : public cThread {
  public:
    // ========================================================================
    // === NESTED TYPES ===
    // ========================================================================

    // -------------------------------------------------------------------------
    // OsdOverlay
    // -------------------------------------------------------------------------

    /// Active OSD plane geometry. Updated from the OSD thread; committed atomically
    /// with the next video frame so both planes switch on the same vblank.
    struct OsdOverlay {
        // === DATA ===
        uint32_t fbId{};   ///< KMS FB object ID; 0 = plane hidden
        uint32_t height{}; ///< Overlay height in pixels
        uint32_t width{};  ///< Overlay width in pixels
        int32_t x{};       ///< Left edge relative to display origin
        int32_t y{};       ///< Top edge relative to display origin
    };

    // ========================================================================
    // === LIFECYCLE ===
    // ========================================================================

    cVaapiDisplay();
    ~cVaapiDisplay() noexcept override;
    cVaapiDisplay(const cVaapiDisplay &) = delete;
    auto operator=(const cVaapiDisplay &) -> cVaapiDisplay & = delete;
    cVaapiDisplay(cVaapiDisplay &&) noexcept = delete;
    auto operator=(cVaapiDisplay &&) noexcept -> cVaapiDisplay & = delete;

    // ========================================================================
    // === PUBLIC API ===
    // ========================================================================

    /// Block until fbId is no longer the committed OSD FB. Called by cVaapiOsd destructor
    /// before freeing the dumb buffer; returning early would allow KMS to scan freed memory.
    auto AwaitOsdHidden(uint32_t fbId) -> void;
    /// Pause frame delivery and hold importMutex while the codec is being torn down.
    /// Must be paired with EndStreamSwitch(). See display.cpp for the required call order.
    auto BeginStreamSwitch() -> void;
    /// Release importMutex and resume frame delivery after a channel switch.
    auto EndStreamSwitch() -> void;
    [[nodiscard]] auto GetAspectRatio() const noexcept -> double { return aspectRatio; }
    [[nodiscard]] auto GetDrmFd() const noexcept -> int { return drmFd; }
    [[nodiscard]] auto GetOutputHeight() const noexcept -> uint32_t { return outputHeight; }
    [[nodiscard]] auto GetOutputRefreshRate() const noexcept -> uint32_t { return refreshRate; }
    [[nodiscard]] auto GetOutputWidth() const noexcept -> uint32_t { return outputWidth; }
    /// Set up planes, cache DRM property IDs, program the initial display mode, start the thread.
    [[nodiscard]] auto Initialize(int fileDescriptor, AVBufferRef *hwDevice, uint32_t crtcIdentifier,
                                  uint32_t connectorIdentifier, const drmModeModeInfo &displayMode) -> bool;
    [[nodiscard]] auto IsInitialized() const noexcept -> bool;
    /// Hide the OSD plane only if fbId is the currently committed FB (avoids a spurious hide
    /// when another overlay has already replaced it).
    auto ClearOsdIfActive(uint32_t fbId) -> void;
    /// Stage new OSD geometry; applied on the next video commit (same vblank). Always marks
    /// dirty even for an unchanged fbId: VDR may repaint in-place, requiring FBC/PSR invalidation.
    auto SetOsd(const OsdOverlay &osd) -> void;
    /// Stop display thread, blank both planes, deactivate CRTC, release all resources. Idempotent.
    auto Shutdown() -> void;
    /// Hand a decoded frame to the display thread (single-slot queue).
    /// timeoutMs: -1 = block indefinitely (decoder's VSync backpressure), 0 = non-blocking, >0 = ms.
    [[nodiscard]] auto SubmitFrame(std::unique_ptr<VaapiFrame> frame, int timeoutMs = -1) -> bool;
    /// Wall-clock ms of the most recent page-flip event; used by the decoder for VSync pacing.
    [[nodiscard]] auto GetLastVSyncTimeMs() const noexcept -> uint64_t {
        return lastVSyncTimeMs.load(std::memory_order_relaxed);
    }
    /// Mutex serializing VA-driver calls between display (MapVaapiFrame) and decoder (VPP).
    /// iHD's VEBOX path is not thread-safe when shared with concurrent filter execution.
    [[nodiscard]] auto GetVaDriverMutex() noexcept -> cMutex & { return vaDriverMutex; }
    /// True iff all KMS commit-path prerequisites for HDR are present: plane supports P010,
    /// COLOR_ENCODING has BT.2020 and BT.709 enums, connector exposes HDR_OUTPUT_METADATA,
    /// Colorspace, and max bpc. HdrMode::On uses this to bypass the sink EDID gate.
    [[nodiscard]] auto CanDriveHdrPlane() const noexcept -> bool;
    /// Stage HDR signalling for the next atomic commit. Thread-safe (hdrStateMutex).
    auto SetHdrOutputState(const HdrStreamInfo &info) -> void;
    /// True iff both the KMS stack AND sink EDID advertise support for @p kind.
    /// HdrMode::Auto calls this; always returns false for Sdr (caller must check).
    [[nodiscard]] auto SupportsHdrPassthrough(StreamHdrKind kind) const noexcept -> bool;

  protected:
    // ========================================================================
    // === THREAD ===
    // ========================================================================

    auto Action() -> void override; ///< Display thread: drain DRM events -> map frame -> atomic commit -> wait for flip

  private:
    // ========================================================================
    // === INTERNAL TYPES ===
    // ========================================================================

    // -------------------------------------------------------------------------
    // DrmFramebuffer
    // -------------------------------------------------------------------------

    /// Owns a KMS framebuffer backed by a VAAPI PRIME surface. Move-only.
    /// Destructor release order: AVFrame -> FB -> GEM (reversing this is a kernel use-after-free).
    struct DrmFramebuffer {
        DrmFramebuffer() = default;
        ~DrmFramebuffer() noexcept;
        DrmFramebuffer(const DrmFramebuffer &) = delete;
        auto operator=(const DrmFramebuffer &) -> DrmFramebuffer & = delete;
        DrmFramebuffer(DrmFramebuffer &&other) noexcept;
        auto operator=(DrmFramebuffer &&other) noexcept -> DrmFramebuffer &;

        // === API ===
        [[nodiscard]] auto IsValid() const noexcept -> bool { return fbId != 0; }

        // === DATA ===
        int drmFd{-1};        ///< Borrowed DRM fd (lifetime owned by cVaapiDisplay)
        uint32_t fbId{};      ///< KMS FB object ID; 0 = invalid/unregistered
        AVFrame *frame{};     ///< Owned AVFrame keeping the VA surface ref alive for KMS scanout
        uint32_t gemHandle{}; ///< GEM BO handle imported from the PRIME fd
        uint32_t height{};    ///< Full surface height (may include codec padding beyond crop)
        uint64_t modifier{};  ///< DRM format modifier (tiling/compression layout)
        uint32_t width{};     ///< Full surface width (may include codec padding beyond crop)
    };

    // -------------------------------------------------------------------------
    // DrmPlaneProps
    // -------------------------------------------------------------------------

    /// Cached DRM atomic property IDs for a single display plane. Populated once by
    /// BindDrmPlane(); IDs are stable for the lifetime of the DRM device fd.
    struct DrmPlaneProps {
        // === DATA ===
        uint32_t colorEncoding{};              ///< COLOR_ENCODING prop ID (YUV colorimetry override)
        uint64_t colorEncodingBt2020{};        ///< "ITU-R BT.2020 YCbCr" enum value
        bool colorEncodingBt2020Valid{};       ///< True when BT.2020 enum was resolved
        uint64_t colorEncodingBt709{};         ///< "ITU-R BT.709 YCbCr" enum value
        bool colorEncodingValid{};             ///< True when BT.709 enum was resolved
        uint32_t colorRange{};                 ///< COLOR_RANGE prop ID
        uint64_t colorRangeLimited{};          ///< "YCbCr limited range" enum value
        bool colorRangeValid{};                ///< True when limited-range enum was resolved
        uint32_t crtcH{};                      ///< CRTC_H prop ID (dest rect height, pixels)
        uint32_t crtcId{};                     ///< CRTC_ID prop ID
        uint32_t crtcW{};                      ///< CRTC_W prop ID (dest rect width, pixels)
        uint32_t crtcX{};                      ///< CRTC_X prop ID (dest rect X offset, pixels)
        uint32_t crtcY{};                      ///< CRTC_Y prop ID (dest rect Y offset, pixels)
        uint32_t fbId{};                       ///< FB_ID prop ID
        uint32_t pixelBlendMode{};             ///< "pixel blend mode" prop ID; 1=Coverage (straight ARGB from VDR)
        uint32_t srcH{};                       ///< SRC_H prop ID (source crop height, 16.16 fixed-point)
        uint32_t srcW{};                       ///< SRC_W prop ID (source crop width, 16.16 fixed-point)
        uint32_t srcX{};                       ///< SRC_X prop ID (source crop X, 16.16 fixed-point)
        uint32_t srcY{};                       ///< SRC_Y prop ID (source crop Y, 16.16 fixed-point)
        bool supportsP010{};                   ///< IN_FORMATS blob lists DRM_FORMAT_P010 (required for HDR passthrough)
        uint32_t type{DRM_PLANE_TYPE_OVERLAY}; ///< DRM_PLANE_TYPE_PRIMARY / _OVERLAY / _CURSOR
        uint32_t zpos{};                       ///< zpos prop ID; never written at commit time (see AppendOsdPlane)
    };

    // -------------------------------------------------------------------------
    // ModesetProps
    // -------------------------------------------------------------------------

    /// Cached CRTC and connector property IDs needed for ALLOW_MODESET commits.
    /// Populated once by LoadDrmProperties().
    struct ModesetProps {
        // === DATA ===
        uint32_t connectorCrtcId{}; ///< Connector CRTC_ID prop ID
        uint32_t crtcActive{};      ///< CRTC ACTIVE prop ID
        uint32_t crtcModeId{};      ///< CRTC MODE_ID blob prop ID
        bool isValid{};             ///< True once all three IDs are resolved
    };

    /// Connector property IDs and enum values for HDR output signalling.
    /// Populated by ProbeHdrCapabilities(); a zero ID means the driver lacks that feature
    /// and the corresponding property is silently skipped at commit time (SDR still works).
    struct HdrConnectorProps {
        // === DATA ===
        uint32_t colorspace{};          ///< "Colorspace" enum prop ID
        uint64_t colorspaceBt2020Ycc{}; ///< "BT2020_YCC" enum value
        uint64_t colorspaceDefault{};   ///< "Default" enum value
        bool colorspaceValid{};         ///< True iff both enums resolved AND are distinct (same value -> unusable)
        uint32_t hdrOutputMetadata{};   ///< "HDR_OUTPUT_METADATA" blob prop ID
        uint32_t maxBpc{};    ///< "max bpc" range prop ID; 0 when range has < 2 values (clamp(10,0,0)=0 rejected)
        uint64_t maxBpcMin{}; ///< Minimum bpc from the range property
        uint64_t maxBpcMax{}; ///< Maximum bpc from the range property
    };

    // ========================================================================
    // === INTERNAL METHODS ===
    // ========================================================================

    /// Add OSD plane properties to @p req, clipping to screen bounds.
    /// No-op if osd.fbId==0 or osdPlaneId==0 (no OSD plane on this hardware).
    auto AppendOsdPlane(AtomicRequest &req, const OsdOverlay &osd) const -> void;
    /// Program a new display mode via an ALLOW_MODESET commit; resets HDR state to SDR.
    [[nodiscard]] auto ApplyDisplayMode(const drmModeModeInfo &mode) -> bool;
    /// Submit an atomic commit. Sets isFlipPending for page-flip commits (non-modeset).
    /// EBUSY is silently swallowed; all other errors are logged.
    [[nodiscard]] auto AtomicCommit(AtomicRequest &req, uint32_t flags) -> bool;
    /// Find the planeIndex-th plane supporting @p format on our CRTC; cache its property IDs.
    /// Prefers HDR-capable planes (P010 + COLOR_ENCODING BT.2020) for the NV12 video slot.
    [[nodiscard]] auto BindDrmPlane(int planeIndex, uint32_t format) -> bool;
    /// Poll for and dispatch one pending DRM event; returns true when an event was handled.
    [[nodiscard]] auto DrainDrmEvents(int timeoutMs) -> bool;
    /// Opt in to UNIVERSAL_PLANES + ATOMIC client caps, then cache CRTC and connector prop IDs.
    [[nodiscard]] auto LoadDrmProperties() -> bool;
    /// If staged HDR state differs from applied, append HDR_OUTPUT_METADATA + Colorspace + max bpc
    /// to @p req. Returns true when properties were appended. Sets @p failed on blob-alloc error
    /// (caller must abort the commit -- partial HDR metadata produces a green-cast image).
    [[nodiscard]] auto MaybeAppendHdrOutputState(AtomicRequest &req, bool &failed) -> bool;
    /// Populate hdrProps and displayCaps from the connector and EDID. Best-effort: failure
    /// disables HDR only; SDR playback is never affected.
    auto ProbeHdrCapabilities() -> void;
    /// Export a VAAPI surface as a KMS framebuffer via PRIME. Takes ownership of vaapiFrame;
    /// the AVFrame ref is transferred to the returned DrmFramebuffer to keep the surface alive.
    [[nodiscard]] auto MapVaapiFrame(std::unique_ptr<VaapiFrame> vaapiFrame) const -> DrmFramebuffer;
    /// libdrm page-flip callback. @p data is the cVaapiDisplay* from drmModeAtomicCommit.
    static auto OnPageFlipEvent(int fd, unsigned int seq, unsigned int sec, unsigned int usec, void *data) -> void;
    /// Submit a page-flip for @p fb, bundling any pending OSD change in the same atomic commit.
    [[nodiscard]] auto PresentBuffer(const DrmFramebuffer &fb) -> bool;
    /// Spin-drain DRM events until the in-flight flip completes or @p timeoutMs elapses.
    /// Must be called before taking importMutex in BeginStreamSwitch().
    auto WaitForPageFlip(int timeoutMs) -> void;

    // ========================================================================
    // === STATE ===
    // ========================================================================

    drmModeModeInfo activeMode{};                     ///< Currently programmed display mode
    double aspectRatio{DISPLAY_DEFAULT_ASPECT_RATIO}; ///< Output pixel aspect ratio (width/height)
    mutable cMutex bufferMutex;                       ///< Guards pendingFrame, pendingBuffer, displayedBuffer
    uint32_t connectorId{};                           ///< DRM connector object ID
    uint32_t crtcId{};                                ///< DRM CRTC object ID
    OsdOverlay currentOsd{};                          ///< OSD staged for the next commit (guarded by osdMutex)
    DrmFramebuffer displayedBuffer; ///< Front buffer currently being scanned out; kept alive until flip completes
    int drmFd{-1};                  ///< Borrowed DRM fd; lifetime owned by cVaapiDevice
    drmEventContext eventContext{}; ///< libdrm event dispatch table; only page_flip_handler is wired
    cCondVar frameSlotCond;         ///< Signalled when pendingFrame slot is consumed (under bufferMutex)
    std::atomic<bool> hasExited;    ///< Set by Action() just before return; Shutdown() polls this
    AVBufferRef *hwDeviceRef{};     ///< Owned VAAPI hw-device context ref (av_buffer_ref of hwDevice)
    mutable cMutex importMutex;     ///< Held across VAAPI->PRIME import + atomic commit; BeginStreamSwitch holds it
                                ///< while the codec is being torn down to prevent MapVaapiFrame racing the teardown.
    mutable cMutex vaDriverMutex; ///< Serializes VA-driver calls: MapVaapiFrame (display) vs VPP pull (decoder).
                                  ///< iHD VEBOX is not re-entrant when shared with filter execution.
    std::atomic<bool> isClearing; ///< Set during stream switch; gates new frame imports in Action() and SubmitFrame()
    std::atomic<bool> isFlipPending; ///< True between commit and page-flip event; Action() waits on this
    std::atomic<bool> ready;         ///< True after Initialize() succeeds; cleared first in Shutdown()
    std::atomic<bool> stopping;      ///< Tells Action() to exit; set after isClearing to avoid import/exit race
    std::atomic<uint64_t> lastVSyncTimeMs{0}; ///< Wall-clock ms of the most recent page-flip event
    uint32_t modeBlobId{};                    ///< KMS MODE_ID blob; must outlive CRTC enable, freed in Shutdown()
    ModesetProps modesetProps{};              ///< Cached CRTC + connector prop IDs for modeset commits
    mutable cMutex osdMutex;                  ///< Guards currentOsd and osdDirty
    bool osdDirty{};                          ///< True from SetOsd() until PresentBuffer() commits it
    uint32_t osdPlaneId{};                    ///< DRM plane object ID for OSD (0 = no overlay plane on this hardware)
    DrmPlaneProps osdProps{};                 ///< Cached atomic prop IDs for the OSD plane
    uint32_t outputHeight{DISPLAY_DEFAULT_HEIGHT}; ///< Active display height in pixels
    uint32_t outputWidth{DISPLAY_DEFAULT_WIDTH};   ///< Active display width in pixels
    DrmFramebuffer pendingBuffer; ///< Back buffer staged for the next flip; promoted to displayedBuffer on success
    std::unique_ptr<VaapiFrame>
        pendingFrame; ///< One-slot queue: decoded frame waiting for MapVaapiFrame (guarded by bufferMutex)
    uint32_t refreshRate{DISPLAY_DEFAULT_REFRESH_RATE}; ///< Active refresh rate in Hz; defaults to 50 if EDID reports 0
    uint32_t videoPlaneId{};                            ///< DRM plane object ID for the video primary plane
    DrmPlaneProps videoProps{};                         ///< Cached atomic prop IDs for the video plane

    // ========================================================================
    // === HDR STATE ===
    // ========================================================================
    uint32_t appliedHdrBlobId{};        ///< HDR_OUTPUT_METADATA blob ID currently held by the kernel (0 = none)
    HdrStreamInfo appliedHdrState{};    ///< HDR state last successfully committed; MaybeAppendHdrOutputState compares
                                        ///< staged vs applied to decide whether a new commit is needed
    DisplayCaps displayCaps{};          ///< KMS + EDID capability snapshot; set by ProbeHdrCapabilities()
    HdrConnectorProps hdrProps{};       ///< Connector HDR prop IDs (HDR_OUTPUT_METADATA, Colorspace, max bpc)
    cMutex hdrStateMutex;               ///< Guards stagedHdrState (written by decoder, read by display thread)
    uint32_t pendingDestroyHdrBlobId{}; ///< Previous blob ID to destroy on the next successful flip
    HdrStreamInfo stagedHdrState{}; ///< HDR state staged by SetHdrOutputState(); consumed by MaybeAppendHdrOutputState
};

#endif // VDR_VAAPIVIDEO_DISPLAY_H
