// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file display.h
 * @brief DRM atomic modesetting and page-flip display
 */

#ifndef VDR_VAAPIVIDEO_DISPLAY_H
#define VDR_VAAPIVIDEO_DISPLAY_H

#include "common.h"
#include "config.h"

struct VaapiFrame;

// ============================================================================
// === ATOMIC REQUEST ===
// ============================================================================

/**
 * @class AtomicRequest
 * @brief RAII wrapper around a DRM atomic commit request
 *
 * Accumulates object/property/value triples and submits them as a single
 * atomic DRM commit. Move-only; copying is deleted.
 */
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
        -> void; ///< Append an object-property-value triple; silently skips an invalid propId
    [[nodiscard]] auto Count() const noexcept -> int; ///< Return the number of properties accumulated so far
    [[nodiscard]] auto Handle() const noexcept
        -> drmModeAtomicReq *; ///< Return the underlying DRM atomic request handle

  private:
    // ========================================================================
    // === STATE ===
    // ========================================================================
    int propCount{};             ///< Number of properties accumulated in this request
    drmModeAtomicReq *request{}; ///< Underlying DRM atomic request handle
};

// ============================================================================
// === VAAPI DISPLAY ===
// ============================================================================

/**
 * @class cVaapiDisplay
 * @brief Zero-copy VAAPI -> DRM atomic modesetting display output
 *
 * Runs a dedicated display thread that imports VAAPI surfaces directly as DRM
 * PRIME framebuffers and schedules page-flips at the monitor's refresh rate.
 * OSD overlay is composited atomically in the same commit.
 *
 * Thread safety:
 *  - @c SubmitFrame(), @c SetOsd(), @c BeginStreamSwitch(), @c EndStreamSwitch() are safe to call from any thread.
 *  - @c Initialize() and @c Shutdown() must be called from the same thread.
 */
class cVaapiDisplay : public cThread {
  public:
    // ========================================================================
    // === NESTED TYPES ===
    // ========================================================================

    // -------------------------------------------------------------------------
    // OsdOverlay
    // -------------------------------------------------------------------------

    /**
     * @struct OsdOverlay
     * @brief Describes the currently active OSD plane geometry
     *
     * Updated from the OSD thread; applied atomically together with the next
     * video frame commit.
     */
    struct OsdOverlay {
        // === DATA ===
        uint32_t fbId{};   ///< KMS framebuffer object ID (0 = hidden)
        uint32_t height{}; ///< OSD overlay height in pixels
        uint32_t width{};  ///< OSD overlay width in pixels
        int32_t x{};       ///< OSD overlay left edge, relative to display origin
        int32_t y{};       ///< OSD overlay top edge, relative to display origin
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

    auto AwaitOsdHidden(uint32_t fbId)
        -> void;                      ///< Block until the given OSD framebuffer is no longer committed to the display
    auto BeginStreamSwitch() -> void; ///< Pause frame delivery and acquire the import lock for a channel/stream switch
    auto EndStreamSwitch() -> void; ///< Release the import lock and resume normal frame delivery after a channel switch
    [[nodiscard]] auto GetAspectRatio() const noexcept
        -> double { ///< Return the display pixel aspect ratio (width / height)
        return aspectRatio;
    }
    [[nodiscard]] auto GetDrmFd() const noexcept -> int { ///< Return the open DRM file descriptor
        return drmFd;
    }
    [[nodiscard]] auto GetOutputHeight() const noexcept -> uint32_t { ///< Return the active display height in pixels
        return outputHeight;
    }
    [[nodiscard]] auto GetOutputWidth() const noexcept -> uint32_t { ///< Return the active display width in pixels
        return outputWidth;
    }
    [[nodiscard]] auto Initialize(int fileDescriptor, AVBufferRef *hwDevice, uint32_t crtcIdentifier,
                                  uint32_t connectorIdentifier,
                                  const drmModeModeInfo &displayMode)
        -> bool; ///< Set up planes, cache DRM properties, program the initial display mode and start the display thread
    [[nodiscard]] auto IsInitialized() const noexcept
        -> bool; ///< Return true after a successful Initialize() and before Shutdown()
    auto ClearOsdIfActive(uint32_t fbId)
        -> void; ///< Clear the OSD overlay only if the given framebuffer ID is the one currently committed
    auto SetOsd(const OsdOverlay &osd) -> void; ///< Update the OSD overlay geometry; applied on the next frame commit
    auto Shutdown() -> void; ///< Stop the display thread, detach all planes and release all resources
    [[nodiscard]] auto SubmitFrame(std::unique_ptr<VaapiFrame> frame,
                                   int timeoutMs = -1)
        -> bool; ///< Hand a decoded VAAPI frame to the display thread (-1 =
                 ///< wait indefinitely, 0 = non-blocking, >0 = timeout ms)
    [[nodiscard]] auto GetLastVSyncTimeMs() const noexcept
        -> uint64_t { ///< Return the wall-clock timestamp (ms) of the most recent page-flip event
        return lastVSyncTimeMs.load(std::memory_order_relaxed);
    }

  protected:
    // ========================================================================
    // === THREAD ===
    // ========================================================================

    auto Action() -> void override; ///< Display thread: import frames, schedule page-flips, drain DRM events

  private:
    // ========================================================================
    // === INTERNAL TYPES ===
    // ========================================================================

    // -------------------------------------------------------------------------
    // DrmFramebuffer
    // -------------------------------------------------------------------------

    /**
     * @struct DrmFramebuffer
     * @brief Owns a single DRM framebuffer backed by a VAAPI PRIME surface
     *
     * Move-only; destructor removes the FB and closes the GEM handle.
     */
    struct DrmFramebuffer {
        DrmFramebuffer() = default;
        ~DrmFramebuffer() noexcept;
        DrmFramebuffer(const DrmFramebuffer &) = delete;
        auto operator=(const DrmFramebuffer &) -> DrmFramebuffer & = delete;
        DrmFramebuffer(DrmFramebuffer &&other) noexcept;
        auto operator=(DrmFramebuffer &&other) noexcept -> DrmFramebuffer &;

        // === API ===
        [[nodiscard]] auto IsValid() const noexcept -> bool {
            return fbId != 0;
        } ///< Return true when the framebuffer has been registered with the KMS driver

        // === DATA ===
        int drmFd{-1};        ///< DRM file descriptor used for cleanup (not owned)
        uint32_t fbId{};      ///< KMS framebuffer object ID (0 = invalid)
        AVFrame *frame{};     ///< AVFrame whose VAAPI surface backs this FB (owned)
        uint32_t gemHandle{}; ///< GEM buffer handle imported from the PRIME fd
        uint32_t height{};    ///< Frame height in pixels
        uint64_t modifier{};  ///< DRM format modifier (e.g. tiling layout)
        uint32_t width{};     ///< Frame width in pixels
    };

    // -------------------------------------------------------------------------
    // DrmPlaneProps
    // -------------------------------------------------------------------------

    /**
     * @struct DrmPlaneProps
     * @brief Cached DRM atomic property IDs for a single display plane
     *
     * Populated once during initialization; property IDs are stable for the
     * lifetime of the DRM device.
     */
    struct DrmPlaneProps {
        // === DATA ===
        uint32_t colorEncoding{};              ///< COLOR_ENCODING property ID for YUV planes
        uint64_t colorEncodingBt709{};         ///< Enum value for BT.709 encoding
        bool colorEncodingValid{};             ///< True when the BT.709 enum value was resolved
        uint32_t colorRange{};                 ///< COLOR_RANGE property ID for YUV planes
        uint64_t colorRangeLimited{};          ///< Enum value for limited range
        bool colorRangeValid{};                ///< True when the limited-range enum value was resolved
        uint32_t crtcH{};                      ///< CRTC_H property ID (destination rectangle height)
        uint32_t crtcId{};                     ///< CRTC_ID property ID (CRTC this plane is attached to)
        uint32_t crtcW{};                      ///< CRTC_W property ID (destination rectangle width)
        uint32_t crtcX{};                      ///< CRTC_X property ID (destination rectangle X offset)
        uint32_t crtcY{};                      ///< CRTC_Y property ID (destination rectangle Y offset)
        uint32_t fbId{};                       ///< FB_ID property ID (framebuffer to scan out)
        uint32_t pixelBlendMode{};             ///< pixel blend mode property ID (alpha blending method)
        uint32_t srcH{};                       ///< SRC_H property ID (source crop height, 16.16 fixed-point)
        uint32_t srcW{};                       ///< SRC_W property ID (source crop width, 16.16 fixed-point)
        uint32_t srcX{};                       ///< SRC_X property ID (source crop X, 16.16 fixed-point)
        uint32_t srcY{};                       ///< SRC_Y property ID (source crop Y, 16.16 fixed-point)
        uint32_t type{DRM_PLANE_TYPE_OVERLAY}; ///< Plane type value (DRM_PLANE_TYPE_PRIMARY / _OVERLAY / _CURSOR)
        uint32_t zpos{};                       ///< zpos property ID (Z-order / layer priority)
    };

    // -------------------------------------------------------------------------
    // ModesetProps
    // -------------------------------------------------------------------------

    /**
     * @struct ModesetProps
     * @brief Cached DRM atomic property IDs for the CRTC and connector
     *
     * Used when programming a new display mode; populated once on
     * initialization.
     */
    struct ModesetProps {
        // === DATA ===
        uint32_t connectorCrtcId{}; ///< Connector CRTC_ID property ID
        uint32_t crtcActive{};      ///< CRTC ACTIVE property ID
        uint32_t crtcModeId{};      ///< CRTC MODE_ID property ID (blob)
        bool isValid{};             ///< True once all three property IDs have been resolved
    };

    // ========================================================================
    // === INTERNAL METHODS ===
    // ========================================================================

    auto AppendOsdPlane(AtomicRequest &req, const OsdOverlay &osd) const
        -> void; ///< Add OSD plane properties to an existing atomic request, clipping to screen bounds
    [[nodiscard]] auto ApplyDisplayMode(const drmModeModeInfo &mode) -> bool; ///< Program a new display mode via an
                                                                              ///< atomic ALLOW_MODESET commit
    [[nodiscard]] auto AtomicCommit(AtomicRequest &req, uint32_t flags)
        -> bool; ///< Submit an atomic commit; sets isFlipPending for non-modeset commits
    [[nodiscard]] auto BindDrmPlane(int planeIndex, uint32_t format)
        -> bool; ///< Find the nth plane that supports @p format and cache its atomic property IDs
    [[nodiscard]] auto DrainDrmEvents(int timeoutMs) -> bool; ///< Poll for and dispatch pending DRM events; returns
                                                              ///< true if an event was handled
    [[nodiscard]] auto LoadDrmProperties() -> bool; ///< Cache the DRM atomic property IDs for the CRTC, connector
                                                    ///< and planes
    [[nodiscard]] auto MapVaapiFrame(std::unique_ptr<VaapiFrame> vaapiFrame) const
        -> DrmFramebuffer; ///< Import a VAAPI surface as a DRM PRIME framebuffer, taking ownership of the frame
    static auto OnPageFlipEvent(int fd, unsigned int seq, unsigned int sec, unsigned int usec, void *data)
        -> void; ///< DRM page-flip event callback; clears isFlipPending
    [[nodiscard]] auto PresentBuffer(const DrmFramebuffer &fb)
        -> bool;                                 ///< Submit an atomic page-flip for @p fb, atomically updating
                                                 ///< the OSD plane if changed
    auto WaitForPageFlip(int timeoutMs) -> void; ///< Drain DRM events until the in-flight page-flip completes
                                                 ///< or @p timeoutMs elapses

    // ========================================================================
    // === STATE ===
    // ========================================================================

    drmModeModeInfo activeMode{};                     ///< Currently programmed display mode
    double aspectRatio{DISPLAY_DEFAULT_ASPECT_RATIO}; ///< Display pixel aspect ratio (width / height)
    mutable cMutex bufferMutex;                       ///< Guards pendingFrame, pendingBuffer and displayedBuffer
    uint32_t connectorId{};                           ///< DRM connector object ID
    uint32_t crtcId{};                                ///< DRM CRTC object ID
    OsdOverlay currentOsd{};                          ///< OSD overlay parameters staged for the next frame commit
    DrmFramebuffer displayedBuffer;    ///< Framebuffer currently being scanned out by the CRTC (front buffer)
    int drmFd{-1};                     ///< Open DRM device file descriptor (not owned)
    drmEventContext eventContext{};    ///< Registered DRM event handler table
    cCondVar frameSlotCond;            ///< Signalled when pendingFrame is consumed and the slot becomes free
    std::atomic<bool> hasThreadExited; ///< Set by the display thread just before it returns (happens-before Shutdown())
    AVBufferRef *hwDeviceRef{};        ///< VAAPI hardware device context reference (owned)
    mutable cMutex importMutex;      ///< Guards the VAAPI -> DRM import + commit cycle; held across BeginStreamSwitch()
    std::atomic<bool> isClearing;    ///< Set during a stream switch to block new frame imports
    std::atomic<bool> isFlipPending; ///< Set between an atomic page-flip commit and the corresponding DRM flip event
    std::atomic<bool> isReady;       ///< Set after successful Initialize(); cleared at the start of Shutdown()
    std::atomic<bool> isStopping;    ///< Signals the display thread to exit its run loop
    std::atomic<uint64_t> lastVSyncTimeMs{0}; ///< Wall-clock timestamp (ms) of the most recent page-flip event
    uint32_t modeBlobId{};                    ///< KMS property blob ID for the current mode (managed lifetime)
    ModesetProps modesetProps{};              ///< Cached DRM atomic property IDs for the CRTC and connector
    mutable cMutex osdMutex;                  ///< Guards currentOsd and osdDirty across threads
    bool osdDirty{};          ///< True from each SetOsd() call until the change is committed by PresentBuffer()
    uint32_t osdPlaneId{};    ///< DRM plane object ID for the OSD overlay (0 if unavailable)
    DrmPlaneProps osdProps{}; ///< Cached atomic property IDs for the OSD plane
    uint32_t outputHeight{DISPLAY_DEFAULT_HEIGHT}; ///< Active display height in pixels
    uint32_t outputWidth{DISPLAY_DEFAULT_WIDTH};   ///< Active display width in pixels
    DrmFramebuffer pendingBuffer;                  ///< Next framebuffer staged for the upcoming page-flip (back buffer)
    std::unique_ptr<VaapiFrame> pendingFrame;      ///< Decoded VAAPI frame queued for import by the display thread
    uint32_t refreshRate{DISPLAY_DEFAULT_REFRESH_RATE}; ///< Active display refresh rate in Hz
    uint32_t videoPlaneId{};                            ///< DRM plane object ID for the video primary plane
    DrmPlaneProps videoProps{};                         ///< Cached atomic property IDs for the video plane
};

#endif // VDR_VAAPIVIDEO_DISPLAY_H
