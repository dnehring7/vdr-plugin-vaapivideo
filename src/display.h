// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file display.h
 * @brief Zero-copy VAAPI->DRM display: PRIME import, atomic mode-setting, and page-flip pacing.
 */

#ifndef VDR_VAAPIVIDEO_DISPLAY_H
#define VDR_VAAPIVIDEO_DISPLAY_H

#include "caps.h"
#include "common.h"
#include "config.h"
#include "filter.h" // FilterGraphToken (DrmFramebuffer keeps the producing VPP graph alive)

#include <deque>

// VDR
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <vdr/osd.h>
#include <vdr/thread.h>
#include <vdr/tools.h>
#pragma GCC diagnostic pop

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
    AtomicRequest(AtomicRequest &&other) noexcept; ///< Steals the request handle; @p other is left with none
    /// Destroys our own request first, then steals @p other's.
    auto operator=(AtomicRequest &&other) noexcept -> AtomicRequest &;

    // ========================================================================
    // === PUBLIC API ===
    // ========================================================================
    auto AddProperty(uint32_t objId, uint32_t propId, uint64_t value)
        -> void; ///< Append (objId, propId, value); silently skips propId==0 (optional property absent on this driver)
    [[nodiscard]] auto Count() const noexcept -> int;   ///< Number of staged triples; 0 means the commit is a no-op
    [[nodiscard]] auto Failed() const noexcept -> bool; ///< True if any requested property could not be staged
    /// Borrowed libdrm handle for drmModeAtomicCommit(); valid only while this object lives.
    [[nodiscard]] auto Handle() const noexcept -> drmModeAtomicReq *;

  private:
    // ========================================================================
    // === STATE ===
    // ========================================================================
    bool failed{};               ///< True once alloc or a property append failed (request incomplete)
    int propCount{};             ///< Properties accumulated so far
    drmModeAtomicReq *request{}; ///< Owned DRM atomic request handle
};

// ============================================================================
// === PLACEMENT HELPER ===
// ============================================================================

/// Result of fitting (srcWidth x srcHeight) into a target rect with DAR preservation.
/// All fields are zero when the input was invalid (use `width == 0` as the failure flag).
struct VideoPlacement {
    uint32_t destX{};  ///< Top-left X in viewport coordinates (even).
    uint32_t destY{};  ///< Top-left Y in viewport coordinates (even).
    uint32_t height{}; ///< Scaled height (even, >= 2).
    uint32_t width{};  ///< Scaled width (even, >= 2).
};

/// DAR-preserving fit of (srcWidth x srcHeight) into @p rect, centered, 2-px aligned for
/// NV12/P010 chroma. Returns an empty placement on invalid input or a rect smaller than 2x2.
[[nodiscard]] auto FitVideoToRect(uint32_t srcWidth, uint32_t srcHeight, const cRect &rect) noexcept -> VideoPlacement;

// ============================================================================
// === VAAPI DISPLAY ===
// ============================================================================

/// Zero-copy VAAPI->DRM display. Imports decoded surfaces as PRIME framebuffers and
/// pages them at the monitor's refresh rate. OSD overlay is composited atomically in
/// the same commit (single vblank for both planes).
///
/// Thread safety: SubmitFrame(), SetOsd(), BeginStreamSwitch(), EndStreamSwitch() are
/// safe from any thread. Initialize() and Shutdown() must be called from the same thread.
/// Lock order: importMutex -> vaDriverMutex; importMutex -> bufferMutex -> vaDriverMutex
/// (a framebuffer released under bufferMutex may free a retired VPP graph). bufferMutex and
/// vaDriverMutex may precede the leaf mutexes videoRectMutex/osdMutex/hdrStateMutex, which
/// are never nested with each other.
/// See display.cpp.
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
    /// Stream-start trace: first fresh frame committed to the CRTC after the switch, vs the switch epoch.
    auto ArmStartTrace(uint64_t epochMs) noexcept -> void;
    auto DisarmStartTrace() noexcept -> void; ///< Drop the pending stream-start milestone.
    /// HDR classification of the stream currently programmed for scanout. Used by GrabImage to
    /// drive HDR-aware tonemapping; reading the AVFrame's color_trc isn't reliable because
    /// av_hwframe_transfer_data strips the transfer-characteristic metadata on download.
    [[nodiscard]] auto GetActiveHdrKind() const noexcept -> StreamHdrKind;
    /// FbId of the OSD overlay currently programmed for scanout, or 0 when no OSD is on screen.
    /// Used by GrabImage to composite only the visible OSD (multiple OSDs may be allocated).
    [[nodiscard]] auto GetActiveOsdFbId() const noexcept -> uint32_t;
    /// Picture aspect the active mode is shown at (16/9 = 1.778), which on the anamorphic SD
    /// timings is NOT hdisplay/vdisplay. Use GetOutputPixelAspect() for the fit math.
    [[nodiscard]] auto GetAspectRatio() const noexcept -> double;
    /// Scanout pixel aspect: 1:1 on square-pixel timings, 64:45 on a 720x576 flagged 16:9. KMS
    /// scanout is 1:1, so the VPP-fitted framebuffer is the last chance to compensate for it.
    [[nodiscard]] auto GetOutputPixelAspect() const noexcept -> AspectRatio {
        const uint64_t packed = outputPixelAspect.load(std::memory_order_acquire);
        return {.den = static_cast<uint32_t>(packed & 0xFFFFFFFFULL), .num = static_cast<uint32_t>(packed >> 32)};
    }
    /// Borrowed DRM fd; the display owns it, so callers must not close it.
    [[nodiscard]] auto GetDrmFd() const noexcept -> int { return drmFd; }
    /// Snapshot the most recently displayed VAAPI surface as a host-side AVFrame (NV12 or P010).
    /// Returns nullptr if no frame has been displayed yet or the GPU download fails. Used by GrabImage.
    [[nodiscard]] auto GrabDisplayedFrame() -> std::unique_ptr<AVFrame, FreeAVFrame>;
    /// Active scanout height (px); use GetOutputGeometry() when both dimensions must agree.
    [[nodiscard]] auto GetOutputHeight() const noexcept -> uint32_t;
    /// Both dimensions from ONE load of the packed atomic. Calling GetOutputWidth() and
    /// GetOutputHeight() separately is two loads, which a mode change landing between them turns
    /// into a new width beside an old height -- use this wherever the pair must agree.
    auto GetOutputGeometry(uint32_t &width, uint32_t &height) const noexcept -> void;
    /// Active refresh rate rounded to whole Hz. Use GetOutputRefreshMilliHz() wherever 59.94 must
    /// stay distinguishable from 60 (mode matching, fps-filter decision).
    [[nodiscard]] auto GetOutputRefreshRate() const noexcept -> uint32_t;
    /// Active refresh rate in millihertz, computed from the mode timings (not the integer-truncated
    /// drmModeModeInfo::vrefresh, which reports both 59.94 and 60 as 60).
    [[nodiscard]] auto GetOutputRefreshMilliHz() const noexcept -> uint32_t {
        return outputRefreshMilliHz.load(std::memory_order_acquire);
    }
    /// Active scanout width (px); see GetOutputHeight().
    [[nodiscard]] auto GetOutputWidth() const noexcept -> uint32_t;
    /// Snapshot of the mode currently programmed on the CRTC. Used by the device to skip a
    /// request that would be a no-op. Display-thread state, so this takes modeRequestMutex.
    [[nodiscard]] auto GetActiveMode() const -> drmModeModeInfo;
    /// Bumped once per successful runtime mode change. Consumers cache against it to detect that
    /// the output geometry moved under them (see cVaapiDevice::GetOsdSize).
    [[nodiscard]] auto GetModeGeneration() const noexcept -> uint64_t {
        return modeGeneration.load(std::memory_order_acquire);
    }
    /// Stage @p mode for the display thread to program on its next loop iteration. Non-blocking
    /// and safe from any thread; a second request before the first is serviced replaces it.
    auto RequestDisplayMode(const drmModeModeInfo &mode) -> void;
    /// Consume the "output geometry changed" edge. The decoder calls this once per decode
    /// iteration and rebuilds the VPP graph when it fires (same contract as filterRebuildPending).
    [[nodiscard]] auto TakeGeometryChange() noexcept -> bool {
        return geometryChanged.exchange(false, std::memory_order_acq_rel);
    }
    /// Thread-safe snapshot of the active scanout rect.
    [[nodiscard]] auto GetVideoRect() const -> cRect;
    /// Thread-safe snapshot of the rect the next VPP build should target (differs from videoRect
    /// only between a ScaleVideo() call and the first fb of the new size).
    [[nodiscard]] auto GetTargetVideoRect() const -> cRect;
    /// Rect SetVideoRect() will store: full-screen for empty, otherwise clipped + 2-px aligned.
    [[nodiscard]] auto NormalizeVideoRect(const cRect &rect) const -> cRect;
    /// Set up planes, cache DRM property IDs, program the initial display mode, start the thread.
    [[nodiscard]] auto Initialize(int fileDescriptor, AVBufferRef *hwDevice, uint32_t crtcIdentifier,
                                  uint32_t connectorIdentifier, const drmModeModeInfo &displayMode) -> bool;
    /// True once Initialize() has planes, properties and the display thread up.
    [[nodiscard]] auto IsInitialized() const noexcept -> bool;
    /// Hide the OSD plane only if fbId is the currently committed FB (avoids a spurious hide
    /// when another overlay has already replaced it).
    auto ClearOsdIfActive(uint32_t fbId) -> void;
    /// Stage new OSD geometry; applied on the next video commit (same vblank). Always marks
    /// dirty even for an unchanged fbId: VDR may repaint in-place, requiring FBC/PSR invalidation.
    auto SetOsd(const OsdOverlay &osd) -> void;
    /// Stage the target video output rect (empty/null = full-screen). Returns true on dimension
    /// change, signaling the caller to trigger a VPP filter rebuild.
    [[nodiscard]] auto SetVideoRect(const cRect &rect) -> bool;
    /// Stop display thread, blank both planes, deactivate CRTC, release all resources. Idempotent.
    auto Shutdown() -> void;
    /// Tell the display thread that the decoder is intentionally pacing slow (trick play).
    /// While set, the underrun detector ignores re-presents -- they reflect trick pacing, not a stall.
    /// Cleared by the decoder when trick mode ends.
    auto SetTrickActive(bool enable) noexcept -> void { trickActive.store(enable, std::memory_order_relaxed); }
    /// Decoder is intentionally sleeping inside a sync correction (hard-ahead / soft-ahead).
    /// While set, the underrun detector ignores re-presents -- they reflect the sleep, not a stall.
    /// Decoder pairs each SleepMs with on/off so the suppression window matches the actual sleep.
    auto SetSyncSleeping(bool enable) noexcept -> void { syncSleeping.store(enable, std::memory_order_relaxed); }
    /// Device is paused (cVaapiDevice::Freeze()/Play()): the drain holds deliberately, so the
    /// underrun detector ignores re-presents -- a short pause would otherwise spam "queue empty"
    /// before the idle catch-all kicks in.
    auto SetDevicePaused(bool enable) noexcept -> void { devicePaused.store(enable, std::memory_order_relaxed); }
    /// Hand a decoded frame to the display thread (DISPLAY_PRERENDER_SLOTS-deep queue).
    /// timeoutMs: -1 = block until a slot opens (VSync backpressure), 0 = non-blocking, >0 = ms.
    [[nodiscard]] auto SubmitFrame(std::unique_ptr<VaapiFrame> frame, int timeoutMs = -1) -> bool;
    /// Lock-free pendingFrames depth poll. Decoder uses depth==0 to decide whether to pre-submit
    /// one frame ahead of strict-due (avoids a VSync re-present from audio-clock vs VSync drift).
    [[nodiscard]] auto PendingDepth() const noexcept -> size_t { return pendingDepth.load(std::memory_order_acquire); }
    /// True from the pop of a submitted frame until its flip has landed (import, commit, VSync wait). EOS
    /// drains count it so the last picture is not cut; the re-presents that follow are not counted.
    [[nodiscard]] auto HasFrameInFlight() const noexcept -> bool {
        return frameInFlight.load(std::memory_order_acquire);
    }
    /// Wall-clock ms of the most recent page-flip event; used by the decoder for VSync pacing.
    [[nodiscard]] auto GetLastVSyncTimeMs() const noexcept -> uint64_t {
        return lastVSyncTimeMs.load(std::memory_order_relaxed);
    }
    /// Mutex serializing VA-driver calls between display (MapVaapiFrame) and decoder (VPP):
    /// one VADisplay may not be driven from two threads at once.
    [[nodiscard]] auto GetVaDriverMutex() noexcept -> cMutex & { return vaDriverMutex; }
    /// True iff all KMS commit-path prerequisites for HDR are present: plane supports P010,
    /// COLOR_ENCODING has BT.2020 and BT.709 enums, connector exposes HDR_OUTPUT_METADATA,
    /// Colorspace, and max bpc. HdrMode::On uses this to bypass the sink EDID gate.
    [[nodiscard]] auto CanDriveHdrPlane() const noexcept -> bool;
    /// Stage HDR signaling for the next atomic commit. Thread-safe (hdrStateMutex).
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
    /// Destructor release order: AVFrame -> FB -> GEM (reversing this is a kernel use-after-free),
    /// then graphToken with the rest of the members (a VA-driver use-after-free if reversed).
    struct DrmFramebuffer {
        DrmFramebuffer() = default;
        ~DrmFramebuffer() noexcept;
        DrmFramebuffer(const DrmFramebuffer &) = delete;
        auto operator=(const DrmFramebuffer &) -> DrmFramebuffer & = delete;
        DrmFramebuffer(DrmFramebuffer &&other) noexcept; ///< Steals fb/GEM ownership; @p other is left invalid
        /// Releases our own fb in destructor order first, then steals @p other's.
        auto operator=(DrmFramebuffer &&other) noexcept -> DrmFramebuffer &;

        // === API ===
        /// True once the FB is registered with KMS, i.e. safe to reference from a commit.
        [[nodiscard]] auto IsValid() const noexcept -> bool { return fbId != 0; }

        // === DATA ===
        int drmFd{-1};        ///< Borrowed DRM fd (lifetime owned by cVaapiDisplay)
        uint32_t fbId{};      ///< KMS FB object ID; 0 = invalid/unregistered
        AVFrame *frame{};     ///< Owned AVFrame keeping the VA surface ref alive for KMS scanout
        uint32_t gemHandle{}; ///< GEM BO handle imported from the PRIME fd
        /// Moved in from the VaapiFrame with `frame`: keeps the producing VPP graph alive while we
        /// hold its surface (see FilterGraphToken). Released after `frame` (dtor body order).
        FilterGraphToken graphToken;
        uint32_t height{};   ///< Full surface height (may include codec padding beyond crop)
        uint64_t modifier{}; ///< DRM format modifier (tiling/compression layout)
        uint32_t width{};    ///< Full surface width (may include codec padding beyond crop)
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

    /// Connector property IDs and enum values for HDR output signaling.
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
    /// Returns true iff the commit will attach @p osd.fbId; false means the plane was hidden
    /// (no OSD plane on this hardware, fully clipped off-screen, or zero-size after clipping).
    [[nodiscard]] auto AppendOsdPlane(AtomicRequest &req, const OsdOverlay &osd) const -> bool;
    /// Program a new display mode via an ALLOW_MODESET commit; resets HDR state to SDR.
    /// @p blankPlanes additionally detaches both planes in the SAME commit -- mandatory for a
    /// runtime change, because a framebuffer larger than the incoming mode makes the kernel
    /// reject the whole atomic request with EINVAL.
    [[nodiscard]] auto ApplyDisplayMode(const drmModeModeInfo &mode, bool blankPlanes) -> bool;
    /// Service a pending RequestDisplayMode() from the display thread: drop the frame queue and
    /// both framebuffers, run the ALLOW_MODESET commit, then republish the output geometry and
    /// invalidate every cached plane property. Never called from any other thread.
    auto ChangeDisplayMode() -> void;
    /// Reset the last-committed VIDEO plane property caches to their sentinel so the next commit
    /// re-writes every stateful property. Shared by Initialize() and ChangeDisplayMode().
    /// Deliberately does NOT touch lastCommittedOsdFbId: that one describes what KMS is actually
    /// scanning out and is what AwaitOsdHidden() blocks on, so it may only be cleared under
    /// osdMutex once a commit that really detached the OSD plane has landed.
    auto ResetPlaneStateCaches() -> void;
    /// Publish @p mode's geometry to the lock-free getters and remap the scanout rects into the
    /// new mode. Shared by Initialize() and ChangeDisplayMode().
    auto PublishOutputGeometry(const drmModeModeInfo &mode) -> void;
    /// Commit the staged OSD overlay on its own, with no video framebuffer attached.
    /// PresentBuffer() normally carries the OSD, but it needs a framebuffer to ride on; after a
    /// mode change (or on an audio-only stream) none may ever arrive, which would leave the OSD
    /// staged forever. Returns true when a commit was submitted.
    [[nodiscard]] auto CommitOsdOnly() -> bool;
    /// Submit an atomic commit. flags==0 selects the async page-flip path; pass
    /// DRM_MODE_ATOMIC_ALLOW_MODESET for a synchronous transition. EBUSY is silently swallowed.
    /// @p osdHdrCommit marks a commit touching the OSD plane under HDR: a NONBLOCK EINVAL retries
    /// under ALLOW_MODESET (CDCLK bump) and latches; if the modeset also fails (or the commit was
    /// already ALLOW_MODESET, e.g. an HDR transition) it suppresses OSD enables while HDR is active.
    [[nodiscard]] auto AtomicCommit(AtomicRequest &req, uint32_t flags, bool osdHdrCommit = false) -> bool;
    /// Find the planeIndex-th plane supporting @p format on the active CRTC; cache its property IDs.
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
    /// Wait until the consumer drains the in-flight flip or @p timeoutMs elapses. Returns false
    /// only when normal stream-switch waiting timed out with the flip still pending; teardown
    /// returns true because the wait result no longer matters. Observe-only (never touches the
    /// DRM fd -- see the regression note in the definition).
    [[nodiscard]] auto WaitForPageFlip(int timeoutMs) -> bool;

    // ========================================================================
    // === STATE ===
    // ========================================================================

    drmModeModeInfo activeMode{};        ///< Currently programmed display mode (guarded by modeRequestMutex)
    cTimeMs atomicFailureLogCooldown{0}; ///< Rate-limits commit-failure logs; display-thread-only like AtomicCommit
    bool awaitingResizedFb{};           ///< Display-thread-only. Set by ChangeDisplayMode(): holds the video plane dark
                                        ///< until an fb sized for the new mode arrives, because KMS never scales and a
                                        ///< stale-sized one would be cropped, not shrunk. Cleared by the first fitting
                                        ///< fb or by the DISPLAY_MODE_RESIZE_WAIT_MS watchdog.
    uint64_t resizeWaitSince{};         ///< cTimeMs::Now() when awaitingResizedFb was set; arms that watchdog.
    mutable cMutex bufferMutex;         ///< Guards pendingFrames, pendingBuffer, displayedBuffer
    uint32_t connectorId{};             ///< DRM connector object ID
    uint32_t crtcId{};                  ///< DRM CRTC object ID
    OsdOverlay currentOsd{};            ///< OSD staged for the next commit (guarded by osdMutex)
    DrmFramebuffer displayedBuffer;     ///< Front buffer currently being scanned out; kept alive until flip completes
    int drmFd{-1};                      ///< Borrowed DRM fd; lifetime owned by cVaapiDevice
    drmEventContext eventContext{};     ///< libdrm event dispatch table; only page_flip_handler is wired
    cCondVar frameSlotCond;             ///< Signaled when a pendingFrames slot opens up (under bufferMutex)
    std::atomic<bool> hasExited{false}; ///< Set by Action() just before return; Shutdown() polls this
    AVBufferRef *hwDeviceRef{};         ///< Owned VAAPI hw-device context ref (av_buffer_ref of hwDevice)
    mutable cMutex importMutex;         ///< Held across VAAPI->PRIME import + atomic commit; BeginStreamSwitch holds it
                                ///< while the codec is being torn down to prevent MapVaapiFrame racing the teardown.
    /// Serializes VA-driver calls: MapVaapiFrame (display), VPP pull (decoder), retired-graph
    /// teardown (FreeFilterGraphLocked, any thread) -- one VADisplay may not be driven from two
    /// threads at once. Precedes only the leaf mutexes (see the class lock-order comment); the
    /// token deleter takes it alone.
    mutable cMutex vaDriverMutex;
    /// Set during stream switch; gates new frame imports in Action() and SubmitFrame()
    std::atomic<bool> isClearing{false};
    std::atomic<bool> isFlipPending{false};      ///< True between commit and page-flip event; Action() waits on this
    std::atomic<uint64_t> flipPendingSinceMs{0}; ///< cTimeMs::Now() when isFlipPending was set; 0 = not pending.
                                                 ///< Action() force-clears the flag if no event arrives within a
                                                 ///< few vblanks (kernel can swallow events on first plane attach).
    std::atomic<bool> ready{false};              ///< True after Initialize() succeeds; cleared first in Shutdown()
    std::atomic<bool> stopping{false}; ///< Tells Action() to exit; set after isClearing to avoid import/exit race
    /// Decoder is in trick play (slow-paced commits expected); suppresses underrun log
    std::atomic<bool> trickActive{false};
    std::atomic<bool> syncSleeping{false}; ///< Decoder is inside a sync-correction sleep (hard-ahead / soft-ahead);
                                           ///< suppresses underrun log so an intentional sleep doesn't fire it.
    std::atomic<bool> devicePaused{false}; ///< Mirrors cVaapiDevice::Freeze()/Play(). Suppresses the underrun log
                                           ///< during the pause window so re-presents of the last frame don't surface
                                           ///< as "queue empty" at vsync rate.
    std::atomic<uint64_t> lastFrameCommitMs{0};  ///< Wall-clock ms of the most recent fresh-frame commit;
                                                 ///< 0 = inactive (post-Clear). Gates the underrun tracker.
    StreamStartTrace startTrace;                 ///< Stream-start milestone: first fresh commit after a switch
    std::atomic<uint64_t> lastVSyncTimeMs{0};    ///< Wall-clock ms of the most recent page-flip event
    uint32_t modeBlobId{};                       ///< KMS MODE_ID blob; must outlive CRTC enable, freed in Shutdown()
    std::atomic<uint64_t> modeGeneration{0};     ///< Incremented after every successful runtime mode change
    drmModeModeInfo modeRequest{};               ///< Mode staged by RequestDisplayMode (guarded by modeRequestMutex)
    std::atomic<bool> modeRequestPending{false}; ///< True while modeRequest awaits the display thread
    mutable cMutex modeRequestMutex;             ///< Guards modeRequest, modeRequestPending and activeMode
                                                 ///< (written by the display thread, read by GetActiveMode())
    std::atomic<bool> modesetActive{false};      ///< Set while the display thread is inside ChangeDisplayMode();
                                                 ///< gates SubmitFrame(). A separate flag from isClearing, which
                                                 ///< BeginStreamSwitch() owns -- sharing one would let whichever
                                                 ///< finished first tear down a gate it never armed.
    ModesetProps modesetProps{};                 ///< Cached CRTC + connector prop IDs for modeset commits
    mutable cMutex osdMutex;                     ///< Guards currentOsd and osdDirty
    bool osdDirty{};                             ///< True from SetOsd() until PresentBuffer() commits it
    uint64_t osdGeneration{};                    ///< Bumps on every staged OSD update (guarded by osdMutex);
                                                 ///< PresentBuffer clears osdDirty only for the generation it
                                                 ///< committed, so a SetOsd racing the commit stays queued.
    uint32_t osdPlaneId{};    ///< DRM plane object ID for OSD (0 = no overlay plane on this hardware)
    DrmPlaneProps osdProps{}; ///< Cached atomic prop IDs for the OSD plane
    /// Active display size, packed as (width << 32) | height -- one atomic so GetOutputGeometry()
    /// can take both from a single load and never observe a new width beside an old height. The
    /// single-value getters load independently, so pairing THOSE is still racy.
    std::atomic<uint64_t> outputGeometry{(static_cast<uint64_t>(DISPLAY_DEFAULT_WIDTH) << 32) | DISPLAY_DEFAULT_HEIGHT};
    /// Scanout pixel aspect of the active mode, packed as (num << 32) | den. 1:1 on every
    /// square-pixel timing; GetAspectRatio() derives the picture aspect from it and the geometry.
    /// Published before the geometry -- see PublishOutputGeometry().
    std::atomic<uint64_t> outputPixelAspect{(1ULL << 32) | 1ULL};
    std::atomic<bool> geometryChanged{false}; ///< Set by ChangeDisplayMode; consumed once by the decoder via
                                              ///< TakeGeometryChange() to trigger a VPP rebuild for the new size.
    DrmFramebuffer pendingBuffer; ///< Back buffer staged for the next flip; promoted to displayedBuffer on success
    /// Up to DISPLAY_PRERENDER_SLOTS frames awaiting MapVaapiFrame (guarded by bufferMutex).
    std::deque<std::unique_ptr<VaapiFrame>> pendingFrames;
    std::atomic<bool> frameInFlight{false}; ///< A popped frame has not flipped yet (set before pendingDepth drops,
                                            ///< cleared once the flip gate passes); see HasFrameInFlight().
    std::atomic<size_t> pendingDepth{0};    ///< Lock-free mirror of pendingFrames.size(); updated under bufferMutex
                                            ///< on every push/pop/clear, polled by the decoder via PendingDepth().
    /// Active refresh rate in millihertz, derived from the mode timings; falls back to 50000 when the mode
    /// reports no clock.
    std::atomic<uint32_t> outputRefreshMilliHz{DISPLAY_DEFAULT_REFRESH_RATE * 1000};
    uint32_t videoPlaneId{};    ///< DRM plane object ID for the video primary plane
    DrmPlaneProps videoProps{}; ///< Cached atomic prop IDs for the video plane
    /// Requested rect (next VPP build target).
    cRect targetVideoRect{0, 0, static_cast<int>(DISPLAY_DEFAULT_WIDTH), static_cast<int>(DISPLAY_DEFAULT_HEIGHT)};
    /// Active scanout rect; matches the current fb 1:1. Advances to targetVideoRect once a matching fb arrives,
    /// so old frames keep painting during a rebuild.
    cRect videoRect{0, 0, static_cast<int>(DISPLAY_DEFAULT_WIDTH), static_cast<int>(DISPLAY_DEFAULT_HEIGHT)};
    mutable cMutex videoRectMutex; ///< Guards targetVideoRect and videoRect.

    // Last-committed plane property caches. Sentinel ~0 forces a write on the first commit after
    // Initialize; cache advances only on commit success so failed commits retry next frame.
    uint32_t lastCommittedOsdFbId{};               ///< OSD fbId in scanout after last successful commit; 0 = none.
    uint64_t lastOsdPixelBlendMode{~uint64_t{0}};  ///< OSD plane pixel blend mode last committed
    uint64_t lastVideoColorEncoding{~uint64_t{0}}; ///< Video plane COLOR_ENCODING last committed
    uint64_t lastVideoColorRange{~uint64_t{0}};    ///< Video plane COLOR_RANGE last committed
    uint64_t lastVideoSrcW{~uint64_t{0}};          ///< Video plane SRC_W last committed (16.16 fixed point)
    uint64_t lastVideoSrcH{~uint64_t{0}};          ///< Video plane SRC_H last committed (16.16 fixed point)
    uint64_t lastVideoCrtcX{~uint64_t{0}};         ///< Video plane CRTC_X last committed (px)
    uint64_t lastVideoCrtcY{~uint64_t{0}};         ///< Video plane CRTC_Y last committed (px)
    uint64_t lastVideoCrtcW{~uint64_t{0}};         ///< Video plane CRTC_W last committed (px)
    uint64_t lastVideoCrtcH{~uint64_t{0}};         ///< Video plane CRTC_H last committed (px)

    // ========================================================================
    // === HDR STATE ===
    // ========================================================================
    uint32_t appliedHdrBlobId{};        ///< HDR_OUTPUT_METADATA blob ID currently held by the kernel (0 = none)
    HdrStreamInfo appliedHdrState{};    ///< HDR state last successfully committed; MaybeAppendHdrOutputState compares
                                        ///< staged vs applied to decide whether a new commit is needed
    DisplayCaps displayCaps{};          ///< KMS + EDID capability snapshot; set by ProbeHdrCapabilities()
    HdrConnectorProps hdrProps{};       ///< Connector HDR prop IDs (HDR_OUTPUT_METADATA, Colorspace, max bpc)
    mutable cMutex hdrStateMutex;       ///< Guards stagedHdrState (written by decoder, read by display thread)
    uint32_t pendingDestroyHdrBlobId{}; ///< Previous blob ID to destroy on the next successful flip
    HdrStreamInfo stagedHdrState{}; ///< HDR state staged by SetHdrOutputState(); consumed by MaybeAppendHdrOutputState

    // OSD-over-HDR commit policy (display-thread-only; see AtomicCommit). On bandwidth-limited GPUs
    // the OSD plane beside the 4K 10-bpc video plane forces a clock-bump modeset; no static cap
    // predicts it, so it's detected at runtime. Reset on mode change and Initialize().
    bool osdHdrNeedsModeset{}; ///< OSD-over-HDR commits must use ALLOW_MODESET (CDCLK bump needed).
    bool osdHdrSuppressed{};   ///< Modeset fails too (bandwidth ceiling): block OSD enables under HDR.
};

#endif // VDR_VAAPIVIDEO_DISPLAY_H
