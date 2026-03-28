// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file display.cpp
 * @brief DRM atomic modesetting and page-flip display
 */

#include "display.h"
#include "common.h"
#include "decoder.h"

// POSIX
#include <pthread.h>
#include <sys/poll.h>

// C++ Standard Library
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>

// FFmpeg
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
extern "C" {
#include <libavutil/buffer.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_drm.h>
#include <libavutil/pixfmt.h>
}
#pragma GCC diagnostic pop

// DRM
#include <libdrm/drm.h>
#include <libdrm/drm_fourcc.h>
#include <libdrm/drm_mode.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

// VDR
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <vdr/thread.h>
#include <vdr/tools.h>
#pragma GCC diagnostic pop

// ============================================================================
// === CONSTANTS ===
// ============================================================================

constexpr int DISPLAY_PAGE_FLIP_TIMEOUT_MS = 40; ///< Maximum wait for a DRM page-flip event (ms)
constexpr int MAX_DRAIN_ITERATIONS = 10;         ///< Maximum DRM event drain passes during shutdown

// ============================================================================
// === HELPER FUNCTIONS ===
// ============================================================================

[[nodiscard]] static auto GetPlaneTypeName(uint32_t type) -> const char * {
    switch (type) {
        case DRM_PLANE_TYPE_OVERLAY:
            return "OVL";
        case DRM_PLANE_TYPE_PRIMARY:
            return "PRI";
        case DRM_PLANE_TYPE_CURSOR:
            return "CUR";
        default:
            return "???";
    }
}

// ============================================================================
// === ATOMIC REQUEST ===
// ============================================================================

AtomicRequest::AtomicRequest() : request(drmModeAtomicAlloc()) {}

AtomicRequest::~AtomicRequest() noexcept {
    if (request) {
        drmModeAtomicFree(request);
    }
}

AtomicRequest::AtomicRequest(AtomicRequest &&other) noexcept : propCount(other.propCount), request(other.request) {
    other.request = nullptr;
    other.propCount = 0;
}

auto AtomicRequest::operator=(AtomicRequest &&other) noexcept -> AtomicRequest & {
    if (this != &other) {
        if (request) {
            drmModeAtomicFree(request);
        }
        request = other.request;
        propCount = other.propCount;
        other.request = nullptr;
        other.propCount = 0;
    }
    return *this;
}

// ============================================================================
// === ATOMIC REQUEST -- PUBLIC API ===
// ============================================================================

auto AtomicRequest::AddProperty(uint32_t objId, uint32_t propId, uint64_t value) -> void {
    // propId == 0: optional property not found during discovery; skip silently.
    if (!request || propId == 0) {
        return;
    }
    if (drmModeAtomicAddProperty(request, objId, propId, value) >= 0) {
        propCount++;
    }
}

[[nodiscard]] auto AtomicRequest::Count() const noexcept -> int { return propCount; }

[[nodiscard]] auto AtomicRequest::Handle() const noexcept -> drmModeAtomicReq * { return request; }

// ============================================================================
// === DRM FRAMEBUFFER ===
// ============================================================================

cVaapiDisplay::DrmFramebuffer::DrmFramebuffer(DrmFramebuffer &&other) noexcept
    : drmFd(other.drmFd), fbId(other.fbId), frame(other.frame), gemHandle(other.gemHandle), height(other.height),
      modifier(other.modifier), width(other.width) {
    // Nullify the source so its destructor is a no-op; ownership is now ours.
    other.drmFd = -1;
    other.fbId = 0;
    other.gemHandle = 0;
    other.frame = nullptr;
}

cVaapiDisplay::DrmFramebuffer::~DrmFramebuffer() noexcept {
    // Release in reverse-acquisition order:
    // 1. AVFrame -- drops the libva surface reference (and closes the DMA-BUF fd).
    // 2. KMS FB -- the KMS driver stops scanning out and drops its DMA-BUF reference.
    // 3. GEM handle -- the imported buffer object is finally freed by the DRM driver.
    if (frame) {
        av_frame_free(&frame);
    }
    if (fbId != 0 && drmFd >= 0) {
        dsyslog("vaapivideo/display: destroy FB %u", fbId);
        drmModeRmFB(drmFd, fbId);
    }
    if (gemHandle != 0 && drmFd >= 0) {
        drm_gem_close closeArgs{.handle = gemHandle, .pad = 0};
        drmIoctl(drmFd, DRM_IOCTL_GEM_CLOSE, &closeArgs);
    }
}

auto cVaapiDisplay::DrmFramebuffer::operator=(DrmFramebuffer &&other) noexcept -> DrmFramebuffer & {
    if (this != &other) {
        if (frame) {
            av_frame_free(&frame);
        }
        if (fbId != 0 && drmFd >= 0) {
            drmModeRmFB(drmFd, fbId);
        }
        if (gemHandle != 0 && drmFd >= 0) {
            drm_gem_close closeArgs{.handle = gemHandle, .pad = 0};
            drmIoctl(drmFd, DRM_IOCTL_GEM_CLOSE, &closeArgs);
        }
        drmFd = other.drmFd;
        fbId = other.fbId;
        frame = other.frame;
        gemHandle = other.gemHandle;
        height = other.height;
        modifier = other.modifier;
        width = other.width;
        other.drmFd = -1;
        other.fbId = 0;
        other.gemHandle = 0;
        other.frame = nullptr;
    }
    return *this;
}

// ============================================================================
// === VAAPI DISPLAY ===
// ============================================================================

cVaapiDisplay::cVaapiDisplay()
    // Register only the page-flip handler; vblank and sequence events are not used, so those slots are left null to
    // avoid spurious callbacks.
    : eventContext{.version = DRM_EVENT_CONTEXT_VERSION,
                   .vblank_handler = nullptr,
                   .page_flip_handler = OnPageFlipEvent,
                   .page_flip_handler2 = nullptr,
                   .sequence_handler = nullptr} {
    dsyslog("vaapivideo/display: created");
}

cVaapiDisplay::~cVaapiDisplay() noexcept {
    dsyslog("vaapivideo/display: destroying (isReady=%d)", isReady.load(std::memory_order_relaxed));
    // Shutdown() must run before the mode blob is destroyed: the display thread may still be committing frames that
    // reference modesetProps.
    Shutdown();

    if (hwDeviceRef) {
        av_buffer_unref(&hwDeviceRef);
    }
    if (modeBlobId != 0 && drmFd >= 0) {
        drmModeDestroyPropertyBlob(drmFd, modeBlobId);
    }
}
// ============================================================================
// === PUBLIC API ===
// ============================================================================

auto cVaapiDisplay::AwaitOsdHidden(uint32_t fbId) -> void {
    if (fbId == 0 || !isReady.load(std::memory_order_relaxed)) {
        return;
    }

    // Wait until PresentBuffer() has committed the hide (osdDirty cleared) and the plane no longer references fbId.
    const cTimeMs deadline(100);
    while (!deadline.TimedOut()) {
        {
            const cMutexLock lock(&osdMutex);
            if (!osdDirty && currentOsd.fbId != fbId) {
                return;
            }
        }
        cCondWait::SleepMs(5);
    }
    esyslog("vaapivideo/display: AwaitOsdHidden timed out for fbId=%u", fbId);
}

auto cVaapiDisplay::BeginStreamSwitch() -> void {
    // Signal the display thread to stop accepting new frames. Must happen before anything else so the thread enters the
    // idle branch promptly.
    isClearing.store(true, std::memory_order_release);

    // Drop any queued frame so the decoder is not blocked waiting for the slot.
    {
        const cMutexLock lock(&bufferMutex);
        pendingFrame.reset();
        frameSlotCond.Broadcast();
    }

    // Wait for the current page-flip to land before locking importMutex. If we lock first, the display thread cannot
    // call DrainDrmEvents() to clear isFlipPending, causing a deadlock.
    WaitForPageFlip(DISPLAY_PAGE_FLIP_TIMEOUT_MS);

    // Lock importMutex so the display thread cannot start a new VAAPI->DRM import while the codec is being torn down.
    // The thread is either: - Spinning in the isClearing idle branch (safe), or - About to re-check isClearing under
    // importMutex (also safe). The existing framebuffers are intentionally kept alive so the last decoded picture stays
    // visible on screen during the channel switch.
    importMutex.Lock();
}

auto cVaapiDisplay::EndStreamSwitch() -> void {
    // Codec is fully destroyed; release the import lock so the display thread can resume. Clearing isClearing after the
    // unlock is safe because the thread will re-check it under importMutex before touching any surface.
    importMutex.Unlock();
    isClearing.store(false, std::memory_order_release);
}

[[nodiscard]] auto cVaapiDisplay::Initialize(int fileDescriptor, AVBufferRef *hwDevice, uint32_t crtcIdentifier,
                                             uint32_t connectorIdentifier, const drmModeModeInfo &displayMode) -> bool {
    dsyslog("vaapivideo/display: initializing %ux%u@%uHz", displayMode.hdisplay, displayMode.vdisplay,
            displayMode.vrefresh);

    if (fileDescriptor < 0 || !hwDevice) [[unlikely]] {
        esyslog("vaapivideo/display: invalid parameters");
        return false;
    }

    drmFd = fileDescriptor;
    crtcId = crtcIdentifier;
    connectorId = connectorIdentifier;
    activeMode = displayMode;
    outputWidth = displayMode.hdisplay;
    outputHeight = displayMode.vdisplay;
    // drmModeModeInfo.vrefresh can be 0 for uncommon modes; fall back to 50 Hz.
    refreshRate = displayMode.vrefresh > 0 ? displayMode.vrefresh : 50;
    aspectRatio = static_cast<double>(outputWidth) / static_cast<double>(outputHeight);

    hwDeviceRef = av_buffer_ref(hwDevice);
    if (!hwDeviceRef) {
        esyslog("vaapivideo/display: failed to ref hw device");
        return false;
    }

    // Initialization order is significant:
    // 1. Cache property IDs (needed by every subsequent atomic commit).
    // 2. Bind the NV12 video plane (mandatory).
    // 3. Bind the ARGB OSD plane (optional).
    // 4. Program the initial display mode (performs an ALLOW_MODESET commit).
    // 5. Mark as ready and start the display thread.
    if (!LoadDrmProperties()) {
        esyslog("vaapivideo/display: failed to cache DRM properties");
        return false;
    }

    if (!BindDrmPlane(0, DRM_FORMAT_NV12)) {
        esyslog("vaapivideo/display: no video plane found");
        return false;
    }

    // OSD plane is optional; failures are silently ignored.
    (void)BindDrmPlane(0, DRM_FORMAT_ARGB8888);

    if (!ApplyDisplayMode(displayMode)) {
        esyslog("vaapivideo/display: failed to set display mode");
        return false;
    }

    isReady.store(true, std::memory_order_release);
    Start();

    isyslog("vaapivideo/display: initialized %ux%u@%uHz", outputWidth, outputHeight, refreshRate);
    return true;
}

[[nodiscard]] auto cVaapiDisplay::IsInitialized() const noexcept -> bool {
    return isReady.load(std::memory_order_acquire);
}

auto cVaapiDisplay::ClearOsdIfActive(uint32_t fbId) -> void {
    if (fbId == 0) {
        return;
    }

    const cMutexLock lock(&osdMutex);
    if (currentOsd.fbId == fbId) {
        dsyslog("vaapivideo/display: OSD hide (conditional) - fbId=%u", fbId);
        currentOsd = {};
        osdDirty = true;
    }
}

auto cVaapiDisplay::SetOsd(const OsdOverlay &osd) -> void {
    const cMutexLock lock(&osdMutex);

    const bool wasHidden = (currentOsd.fbId == 0);
    const bool nowHidden = (osd.fbId == 0);
    const bool showing = wasHidden && !nowHidden;
    const bool hiding = !wasHidden && nowHidden;

    if (showing) {
        dsyslog("vaapivideo/display: OSD show - fbId=%u pos=(%d,%d) size=%ux%u", osd.fbId, osd.x, osd.y, osd.width,
                osd.height);
    } else if (hiding) {
        dsyslog("vaapivideo/display: OSD hide - fbId=%u", currentOsd.fbId);
    }

    // Always mark dirty even for identical geometry: the OSD may have written new pixels into the same dumb buffer,
    // and the kernel needs a commit to invalidate FBC/PSR tile caches.
    currentOsd = osd;
    osdDirty = true;
}

auto cVaapiDisplay::Shutdown() -> void {
    const bool wasInitialized = isReady.exchange(false, std::memory_order_acq_rel);
    dsyslog("vaapivideo/display: shutting down (wasInitialized=%d)", wasInitialized);

    if (!wasInitialized) {
        dsyslog("vaapivideo/display: already shut down, skipping");
        return;
    }

    // isClearing must be set before isStopping: it prevents new frame imports while isStopping tells the run loop to
    // exit.
    isClearing.store(true, std::memory_order_release);
    isStopping.store(true, std::memory_order_release);

    // Force-clear isFlipPending in case the CRTC has been disabled and the page-flip event will never arrive, which
    // would stall the thread.
    isFlipPending.store(false, std::memory_order_release);

    frameSlotCond.Broadcast();
    Cancel(1);

    const cTimeMs timeout(500);
    while (!hasThreadExited.load(std::memory_order_acquire) && !timeout.TimedOut()) {
        frameSlotCond.Broadcast();
        isFlipPending.store(false, std::memory_order_release);
        cCondWait::SleepMs(10);
    }

    if (!hasThreadExited.load(std::memory_order_acquire)) {
        esyslog("vaapivideo/display: thread did not exit in 500ms, waiting longer...");
        const cTimeMs timeout2(2000);
        while (!hasThreadExited.load(std::memory_order_acquire) && !timeout2.TimedOut()) {
            frameSlotCond.Broadcast();
            isFlipPending.store(false, std::memory_order_release);
            cCondWait::SleepMs(50);
        }
    }

    if (!hasThreadExited.load(std::memory_order_acquire)) {
        esyslog("vaapivideo/display: thread did not exit - may cause resource leak");
    }

    // Drain any page-flip events that arrived after the thread exited.
    for (int i = 0; i < MAX_DRAIN_ITERATIONS && DrainDrmEvents(0); ++i) {
    }

    // Detach planes and deactivate the CRTC so the kernel/console can reclaim the display.
    if (drmFd >= 0) {
        AtomicRequest req;
        if (videoPlaneId != 0) {
            req.AddProperty(videoPlaneId, videoProps.fbId, 0);
            req.AddProperty(videoPlaneId, videoProps.crtcId, 0);
        }
        if (osdPlaneId != 0) {
            req.AddProperty(osdPlaneId, osdProps.fbId, 0);
            req.AddProperty(osdPlaneId, osdProps.crtcId, 0);
        }
        // Deactivate the CRTC: clear mode, set ACTIVE=0, disconnect connector.
        if (modesetProps.isValid) {
            req.AddProperty(crtcId, modesetProps.crtcActive, 0);
            req.AddProperty(crtcId, modesetProps.crtcModeId, 0);
            req.AddProperty(connectorId, modesetProps.connectorCrtcId, 0);
        }
        (void)AtomicCommit(req, DRM_MODE_ATOMIC_ALLOW_MODESET);
    }

    {
        const cMutexLock lock(&bufferMutex);
        pendingFrame.reset();
        displayedBuffer = DrmFramebuffer{};
        pendingBuffer = DrmFramebuffer{};
    }
}

[[nodiscard]] auto cVaapiDisplay::SubmitFrame(std::unique_ptr<VaapiFrame> frame, int timeoutMs) -> bool {
    // Relaxed early-exit polls; bufferMutex below provides full synchronization.
    if (!frame || !isReady.load(std::memory_order_relaxed) || isClearing.load(std::memory_order_relaxed)) [[unlikely]] {
        return false;
    }

    const cMutexLock lock(&bufferMutex);

    // Single pending-frame slot; block until the display thread consumes the previous frame.
    if (pendingFrame) {
        if (timeoutMs == 0) {
            return false;
        }

        // Clamp "infinite" to 1 second chunks so we still check isClearing regularly instead of sleeping indefinitely
        // in TimedWait().
        const int waitMs = (timeoutMs < 0) ? 1000 : timeoutMs;
        const cTimeMs deadline(waitMs);
        while (pendingFrame && isReady.load(std::memory_order_relaxed) && !deadline.TimedOut()) {
            // Break out immediately if a stream switch is in progress to avoid deadlocking: BeginStreamSwitch() drains
            // pendingFrame under bufferMutex, so continuing to wait here would block it.
            if (isClearing.load(std::memory_order_relaxed)) {
                return false;
            }
            frameSlotCond.TimedWait(bufferMutex, 10);
        }

        if (pendingFrame || isClearing.load(std::memory_order_relaxed)) [[unlikely]] {
            return false;
        }
    }

    pendingFrame = std::move(frame);
    return true;
}

// ============================================================================
// === THREAD ===
// ============================================================================

auto cVaapiDisplay::Action() -> void {
    isyslog("vaapivideo/display: thread started (thread=%lu)", (unsigned long)pthread_self());

    // VDR's Running() is not thread-safe; use our own atomic flags (relaxed for polling, acquire under importMutex).
    while (!isStopping.load(std::memory_order_relaxed) && isReady.load(std::memory_order_relaxed)) {
        // Drain all pending DRM events before doing any work.  DrainDrmEvents(0) is non-blocking; the post-drain
        // flag check below catches shutdown immediately after.
        while (DrainDrmEvents(0)) {
        }

        if (isStopping.load(std::memory_order_relaxed) || !isReady.load(std::memory_order_relaxed)) {
            break;
        }

        // A flip is outstanding; poll briefly so we stay responsive to shutdown.
        if (isFlipPending.load(std::memory_order_relaxed)) {
            (void)DrainDrmEvents(5);
            continue;
        }

        // importMutex is held across the entire map+commit cycle to prevent BeginStreamSwitch() from tearing down the
        // codec while av_hwframe_map() is still accessing the VAAPI surface.
        bool frameCommitted = false;
        bool skipDueToClearing = false;
        bool hadFrame = false;

        {
            const cMutexLock importLock(&importMutex);

            // Re-check flags under the lock; BeginStreamSwitch() or Shutdown() may have arrived between the flip-wait
            // above and here.
            if (isClearing.load(std::memory_order_acquire) || isStopping.load(std::memory_order_acquire) ||
                !isReady.load(std::memory_order_acquire)) {
                skipDueToClearing = true;
            } else {
                std::unique_ptr<VaapiFrame> frameToShow;
                {
                    const cMutexLock lock(&bufferMutex);
                    if (pendingFrame) {
                        frameToShow = std::move(pendingFrame);
                        frameSlotCond.Broadcast();
                    }
                }

                // BeginStreamSwitch() may have fired between grabbing the frame and reaching this point; discard the
                // frame in that case so the caller is not blocked waiting for BeginStreamSwitch() to return.
                if (frameToShow && !isClearing.load(std::memory_order_acquire)) {
                    hadFrame = true;
                    DrmFramebuffer newFb = MapVaapiFrame(std::move(frameToShow));

                    // importMutex must still be held here: av_hwframe_map() keeps a reference into the VAAPI surface,
                    // which must outlive the DRM commit. Check isClearing once more because MapVaapiFrame() is the
                    // slowest step in this path.
                    if (newFb.IsValid() && !isClearing.load(std::memory_order_acquire)) {
                        const cMutexLock lock(&bufferMutex);
                        displayedBuffer = std::move(pendingBuffer);
                        pendingBuffer = std::move(newFb);

                        if (PresentBuffer(pendingBuffer)) {
                            frameCommitted = true;
                        }
                    }
                }
            }
        } // importMutex released; BeginStreamSwitch() may now proceed

        // All isClearing checks after this point are outside the lock, so sleep briefly to yield to BeginStreamSwitch()
        // rather than spinning.
        if (skipDueToClearing) {
            if (isStopping.load(std::memory_order_relaxed) || !isReady.load(std::memory_order_relaxed)) {
                break;
            }
            cCondWait::SleepMs(5);
            continue;
        }

        // No new frame arrived this iteration; re-submit the last buffer to keep the CRTC actively scanning. Without
        // this the CRTC could go dark (or display a stale hardware cursor) on drivers that require a flip per refresh
        // cycle. Guard with isClearing to avoid racing with BeginStreamSwitch().
        if (!frameCommitted && !hadFrame && !isClearing.load(std::memory_order_relaxed)) {
            const cMutexLock lock(&bufferMutex);
            if (pendingBuffer.IsValid()) {
                (void)PresentBuffer(pendingBuffer);
            }
        }
    }

    // Store before the final log so Shutdown()'s spin sees it immediately.
    hasThreadExited.store(true, std::memory_order_release);
}

// ============================================================================
// === INTERNAL METHODS ===
// ============================================================================

auto cVaapiDisplay::AppendOsdPlane(AtomicRequest &req, const OsdOverlay &osd) const -> void {
    if (osd.fbId == 0 || osdPlaneId == 0) {
        return;
    }

    if (osd.x < 0 || osd.y < 0) {
        esyslog("vaapivideo/display: OSD invalid negative position (%d,%d)", osd.x, osd.y);
        return;
    }

    if (static_cast<uint32_t>(osd.x) >= outputWidth || static_cast<uint32_t>(osd.y) >= outputHeight) {
        esyslog("vaapivideo/display: OSD off-screen pos=(%d,%d) screen=%ux%u", osd.x, osd.y, outputWidth, outputHeight);
        return;
    }

    // Clip the OSD rectangle to the screen boundary. This is mandatory: the KMS driver rejects any atomic commit where
    // the destination rectangle (CRTC_X + CRTC_W) exceeds the CRTC dimensions.
    const auto clippedW = std::min(osd.width, outputWidth - static_cast<uint32_t>(osd.x));
    const auto clippedH = std::min(osd.height, outputHeight - static_cast<uint32_t>(osd.y));

    if (clippedW == 0 || clippedH == 0) {
        return;
    }

    req.AddProperty(osdPlaneId, osdProps.crtcId, crtcId);
    req.AddProperty(osdPlaneId, osdProps.fbId, osd.fbId);
    req.AddProperty(osdPlaneId, osdProps.srcX, 0);
    req.AddProperty(osdPlaneId, osdProps.srcY, 0);
    // SRC_* use 16.16 fixed-point (value = pixels << 16).
    req.AddProperty(osdPlaneId, osdProps.srcW, static_cast<uint64_t>(clippedW) << 16);
    req.AddProperty(osdPlaneId, osdProps.srcH, static_cast<uint64_t>(clippedH) << 16);
    req.AddProperty(osdPlaneId, osdProps.crtcX, static_cast<uint64_t>(osd.x));
    req.AddProperty(osdPlaneId, osdProps.crtcY, static_cast<uint64_t>(osd.y));
    req.AddProperty(osdPlaneId, osdProps.crtcW, clippedW);
    req.AddProperty(osdPlaneId, osdProps.crtcH, clippedH);

    // VDR's tColor is straight (non-pre-multiplied) ARGB, so request "Coverage" blending (value 1) rather than
    // "Pre-multiplied".
    if (osdProps.pixelBlendMode != 0) {
        req.AddProperty(osdPlaneId, osdProps.pixelBlendMode, 1);
    }
    // zpos is intentionally omitted: it is immutable on Intel i915 overlay planes.
}

[[nodiscard]] auto cVaapiDisplay::AtomicCommit(AtomicRequest &req, uint32_t flags) -> bool {
    if (req.Count() == 0) {
        return true;
    }

    // Non-blocking page-flip with event notification. ASYNC not used: requires linear modifiers (VAAPI is tiled).
    uint32_t commitFlags = flags;
    if ((flags & DRM_MODE_ATOMIC_ALLOW_MODESET) == 0) {
        commitFlags |= DRM_MODE_PAGE_FLIP_EVENT | DRM_MODE_ATOMIC_NONBLOCK;
    }

    const int ret = drmModeAtomicCommit(drmFd, req.Handle(), commitFlags, this);
    if (ret == 0) {
        if ((flags & DRM_MODE_ATOMIC_ALLOW_MODESET) == 0) {
            isFlipPending.store(true, std::memory_order_release);
        }
        return true;
    }

    // EBUSY means the CRTC is still processing the previous flip; the caller will retry on the next iteration.
    if (errno != EBUSY) {
        esyslog("vaapivideo/display: atomic commit failed - %s (flags=0x%x)", strerror(errno), commitFlags);
    }
    return false;
}

[[nodiscard]] auto cVaapiDisplay::ApplyDisplayMode(const drmModeModeInfo &mode) -> bool {
    dsyslog("vaapivideo/display: setting display mode %ux%u@%uHz", mode.hdisplay, mode.vdisplay, mode.vrefresh);
    // Mode is passed as a property blob; destroy the old one to avoid leaking kernel resources.
    if (modeBlobId != 0) {
        drmModeDestroyPropertyBlob(drmFd, modeBlobId);
        modeBlobId = 0;
    }

    if (drmModeCreatePropertyBlob(drmFd, &mode, sizeof(mode), &modeBlobId) < 0) {
        esyslog("vaapivideo/display: failed to create mode blob");
        return false;
    }

    AtomicRequest req;
    req.AddProperty(crtcId, modesetProps.crtcActive, 1);
    req.AddProperty(crtcId, modesetProps.crtcModeId, modeBlobId);
    req.AddProperty(connectorId, modesetProps.connectorCrtcId, crtcId);

    if (!AtomicCommit(req, DRM_MODE_ATOMIC_ALLOW_MODESET)) {
        esyslog("vaapivideo/display: failed to set mode");
        return false;
    }

    activeMode = mode;
    return true;
}

[[nodiscard]] auto cVaapiDisplay::BindDrmPlane(int planeIndex, uint32_t format) -> bool {
    // Find the planeIndex'th plane that supports the given format. Uses the IN_FORMATS blob for modifier awareness.
    auto planeRes = std::unique_ptr<drmModePlaneRes, decltype(&drmModeFreePlaneResources)>(
        drmModeGetPlaneResources(drmFd), drmModeFreePlaneResources);
    auto res =
        std::unique_ptr<drmModeRes, decltype(&drmModeFreeResources)>(drmModeGetResources(drmFd), drmModeFreeResources);

    if (!planeRes || !res) {
        return false;
    }

    // possible_crtcs is a bitmask by index, not by CRTC object ID.
    int crtcIndex = -1;
    for (int i = 0; i < res->count_crtcs; ++i) {
        if (res->crtcs[i] == crtcId) {
            crtcIndex = i;
            break;
        }
    }
    if (crtcIndex < 0) {
        return false;
    }

    dsyslog("vaapivideo/display: searching for plane %d (format 0x%08x)", planeIndex, format);

    int found = 0;
    for (uint32_t i = 0; i < planeRes->count_planes; ++i) {
        auto plane = std::unique_ptr<drmModePlane, decltype(&drmModeFreePlane)>(
            drmModeGetPlane(drmFd, planeRes->planes[i]), drmModeFreePlane);
        if (!plane) {
            continue;
        }

        // Skip if already assigned.
        if (videoPlaneId != 0 && plane->plane_id == videoPlaneId) {
            continue;
        }

        // Check CRTC compatibility.
        if (!(plane->possible_crtcs & (1U << crtcIndex))) {
            continue;
        }

        auto planeProps = std::unique_ptr<drmModeObjectProperties, decltype(&drmModeFreeObjectProperties)>(
            drmModeObjectGetProperties(drmFd, plane->plane_id, DRM_MODE_OBJECT_PLANE), drmModeFreeObjectProperties);
        if (!planeProps) {
            continue;
        }

        // Single pass: collect format support, type, and all atomic property IDs.
        bool hasFormatSupport = false;
        uint32_t planeType = DRM_PLANE_TYPE_OVERLAY;
        DrmPlaneProps tempProps{};

        for (uint32_t j = 0; j < planeProps->count_props; ++j) {
            auto prop = std::unique_ptr<drmModePropertyRes, decltype(&drmModeFreeProperty)>(
                drmModeGetProperty(drmFd, planeProps->props[j]), drmModeFreeProperty);
            if (!prop) {
                continue;
            }

            const char *name = prop->name;

            // IN_FORMATS blob: (format, modifier) pairs supported by the plane.
            if (!hasFormatSupport && strcmp(name, "IN_FORMATS") == 0) {
                const auto blobId = static_cast<uint32_t>(planeProps->prop_values[j]);
                if (blobId != 0) {
                    auto blob = std::unique_ptr<drmModePropertyBlobRes, decltype(&drmModeFreePropertyBlob)>(
                        drmModeGetPropertyBlob(drmFd, blobId), drmModeFreePropertyBlob);
                    if (blob && blob->data) {
                        const auto *modBlob = static_cast<const drm_format_modifier_blob *>(blob->data);
                        const auto *base = static_cast<const uint8_t *>(blob->data);
                        const auto *formats =
                            reinterpret_cast<const uint32_t *>( // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
                                base + modBlob->formats_offset);

                        for (uint32_t k = 0; k < modBlob->count_formats; ++k) {
                            if (formats[k] == format) {
                                hasFormatSupport = true;
                                break;
                            }
                        }
                    }
                }
            } else if ((prop->flags & DRM_MODE_PROP_IMMUTABLE) && strcmp(name, "type") == 0) {
                planeType = static_cast<uint32_t>(planeProps->prop_values[j]);
                tempProps.type = planeType;
            } else if (strcmp(name, "CRTC_ID") == 0) {
                tempProps.crtcId = prop->prop_id;
            } else if (strcmp(name, "FB_ID") == 0) {
                tempProps.fbId = prop->prop_id;
            } else if (strcmp(name, "SRC_X") == 0) {
                tempProps.srcX = prop->prop_id;
            } else if (strcmp(name, "SRC_Y") == 0) {
                tempProps.srcY = prop->prop_id;
            } else if (strcmp(name, "SRC_W") == 0) {
                tempProps.srcW = prop->prop_id;
            } else if (strcmp(name, "SRC_H") == 0) {
                tempProps.srcH = prop->prop_id;
            } else if (strcmp(name, "CRTC_X") == 0) {
                tempProps.crtcX = prop->prop_id;
            } else if (strcmp(name, "CRTC_Y") == 0) {
                tempProps.crtcY = prop->prop_id;
            } else if (strcmp(name, "CRTC_W") == 0) {
                tempProps.crtcW = prop->prop_id;
            } else if (strcmp(name, "CRTC_H") == 0) {
                tempProps.crtcH = prop->prop_id;
            } else if (strcmp(name, "zpos") == 0) {
                tempProps.zpos = prop->prop_id;
            } else if (strcmp(name, "pixel blend mode") == 0) {
                tempProps.pixelBlendMode = prop->prop_id;
            } else if (strcmp(name, "COLOR_ENCODING") == 0) {
                tempProps.colorEncoding = prop->prop_id;
                for (int e = 0; e < prop->count_enums; ++e) {
                    if (strcmp(prop->enums[e].name, "ITU-R BT.709 YCbCr") == 0) {
                        tempProps.colorEncodingBt709 = prop->enums[e].value;
                        tempProps.colorEncodingValid = true;
                        break;
                    }
                }
                dsyslog("vaapivideo/display: plane %u COLOR_ENCODING prop=%u bt709_value=%lu found=%d", plane->plane_id,
                        tempProps.colorEncoding, (unsigned long)tempProps.colorEncodingBt709,
                        tempProps.colorEncodingValid);
            } else if (strcmp(name, "COLOR_RANGE") == 0) {
                tempProps.colorRange = prop->prop_id;
                for (int e = 0; e < prop->count_enums; ++e) {
                    if (strcmp(prop->enums[e].name, "YCbCr limited range") == 0) {
                        tempProps.colorRangeLimited = prop->enums[e].value;
                        tempProps.colorRangeValid = true;
                        break;
                    }
                }
                dsyslog("vaapivideo/display: plane %u COLOR_RANGE prop=%u limited_value=%lu found=%d", plane->plane_id,
                        tempProps.colorRange, (unsigned long)tempProps.colorRangeLimited, tempProps.colorRangeValid);
            }
        }

        if (!hasFormatSupport) {
            continue;
        }

        if (planeType == DRM_PLANE_TYPE_CURSOR) {
            dsyslog("vaapivideo/display: skipping cursor plane %u", plane->plane_id);
            continue;
        }

        if (tempProps.fbId == 0 || tempProps.crtcId == 0 || tempProps.srcX == 0 || tempProps.srcY == 0 ||
            tempProps.srcW == 0 || tempProps.srcH == 0 || tempProps.crtcX == 0 || tempProps.crtcY == 0 ||
            tempProps.crtcW == 0 || tempProps.crtcH == 0) {
            esyslog("vaapivideo/display: plane %u missing required atomic properties", plane->plane_id);
            continue;
        }

        dsyslog("vaapivideo/display: candidate plane %u type=%s (found=%d, need=%d)", plane->plane_id,
                GetPlaneTypeName(planeType), found, planeIndex);

        if (found == planeIndex) {
            const uint32_t planeId = plane->plane_id;
            const bool isVideo = (videoPlaneId == 0);
            DrmPlaneProps &props = isVideo ? videoProps : osdProps;
            props = tempProps;

            if (isVideo) {
                videoPlaneId = planeId;
                isyslog("vaapivideo/display: video plane %u type=%s", videoPlaneId, GetPlaneTypeName(props.type));
            } else {
                osdPlaneId = planeId;
                isyslog("vaapivideo/display: OSD plane %u type=%s zpos=%s", osdPlaneId, GetPlaneTypeName(props.type),
                        props.zpos ? "yes" : "no");
            }
            return true;
        }
        found++;
    }

    esyslog("vaapivideo/display: no suitable plane found for index %d format 0x%08x", planeIndex, format);
    return false;
}

[[nodiscard]] auto cVaapiDisplay::DrainDrmEvents(int timeoutMs) -> bool {
    // No mutex needed: called only from the display thread.
    pollfd pfd{.fd = drmFd, .events = POLLIN, .revents = 0};
    const int ret = poll(&pfd, 1, timeoutMs);

    if (ret > 0 && (pfd.revents & POLLIN)) {
        return drmHandleEvent(drmFd, &eventContext) == 0;
    }
    return false;
}

[[nodiscard]] auto cVaapiDisplay::LoadDrmProperties() -> bool {
    // Mandatory since kernel 4.8/4.2; the ioctl still requires them to be explicitly enabled.
    (void)drmSetClientCap(drmFd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1);
    (void)drmSetClientCap(drmFd, DRM_CLIENT_CAP_ATOMIC, 1);

    // Probing this capability (kernel 6.8+) serves as a minimum kernel version check.
    uint64_t asyncCap = 0;
    if (drmGetCap(drmFd, DRM_CAP_ATOMIC_ASYNC_PAGE_FLIP, &asyncCap) != 0 || asyncCap == 0) {
        esyslog("vaapivideo/display: kernel 6.8+ required (atomic async page-flip capability missing)");
        return false;
    }

    // Cache CRTC property IDs (ACTIVE, MODE_ID).
    auto crtcProps = std::unique_ptr<drmModeObjectProperties, decltype(&drmModeFreeObjectProperties)>(
        drmModeObjectGetProperties(drmFd, crtcId, DRM_MODE_OBJECT_CRTC), drmModeFreeObjectProperties);

    if (crtcProps) {
        for (uint32_t i = 0; i < crtcProps->count_props; ++i) {
            auto prop = std::unique_ptr<drmModePropertyRes, decltype(&drmModeFreeProperty)>(
                drmModeGetProperty(drmFd, crtcProps->props[i]), drmModeFreeProperty);
            if (!prop) {
                continue;
            }
            if (strcmp(prop->name, "ACTIVE") == 0) {
                modesetProps.crtcActive = prop->prop_id;
            } else if (strcmp(prop->name, "MODE_ID") == 0) {
                modesetProps.crtcModeId = prop->prop_id;
            }
        }
    }

    // Cache connector CRTC_ID property.
    auto connProps = std::unique_ptr<drmModeObjectProperties, decltype(&drmModeFreeObjectProperties)>(
        drmModeObjectGetProperties(drmFd, connectorId, DRM_MODE_OBJECT_CONNECTOR), drmModeFreeObjectProperties);

    if (connProps) {
        for (uint32_t i = 0; i < connProps->count_props; ++i) {
            auto prop = std::unique_ptr<drmModePropertyRes, decltype(&drmModeFreeProperty)>(
                drmModeGetProperty(drmFd, connProps->props[i]), drmModeFreeProperty);
            if (!prop) {
                continue;
            }
            if (strcmp(prop->name, "CRTC_ID") == 0) {
                modesetProps.connectorCrtcId = prop->prop_id;
            }
        }
    }

    modesetProps.isValid =
        (modesetProps.crtcActive != 0 && modesetProps.crtcModeId != 0 && modesetProps.connectorCrtcId != 0);
    return modesetProps.isValid;
}

[[nodiscard]] auto cVaapiDisplay::MapVaapiFrame(std::unique_ptr<VaapiFrame> vaapiFrame) const -> DrmFramebuffer {
    if (!vaapiFrame || !vaapiFrame->avFrame || vaapiFrame->avFrame->format != AV_PIX_FMT_VAAPI) [[unlikely]] {
        return {};
    }

    const AVFrame *srcFrame = vaapiFrame->avFrame;

    // Map the VAAPI surface to a DRM PRIME descriptor so we can import its file descriptors into the KMS driver.
    AVFrame *mappedFrame = av_frame_alloc();
    if (!mappedFrame) [[unlikely]] {
        return {};
    }

    mappedFrame->format = AV_PIX_FMT_DRM_PRIME;
    // AV_HWFRAME_MAP_READ -- we need read access to the surface data. AV_HWFRAME_MAP_DIRECT -- request a zero-copy map
    // (the DRM fd points directly into the VA surface, no intermediate copy).
    const int ret = av_hwframe_map(mappedFrame, srcFrame, AV_HWFRAME_MAP_READ | AV_HWFRAME_MAP_DIRECT);
    if (ret < 0) [[unlikely]] {
        // EIO is normal when av_hwframe_map() is called while the codec is being destroyed; the surface is already gone
        // from the VA driver.
        if (ret != AVERROR(EIO) && !isClearing.load(std::memory_order_relaxed)) {
            dsyslog("vaapivideo/display: av_hwframe_map failed: %s", AvErr(ret).data());
        }
        av_frame_free(&mappedFrame);
        return {};
    }

    const auto *desc =
        reinterpret_cast<const AVDRMFrameDescriptor *>( // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
            mappedFrame->data[0]);
    // Only single-object NV12 surfaces supported (one DMA-BUF fd, two planes at different offsets).
    if (!desc || desc->nb_objects == 0 || desc->nb_layers == 0 || desc->nb_objects != 1) [[unlikely]] {
        av_frame_free(&mappedFrame);
        return {};
    }

    if (desc->objects[0].fd < 0) [[unlikely]] {
        esyslog("vaapivideo/display: invalid PRIME FD %d", desc->objects[0].fd);
        av_frame_free(&mappedFrame);
        return {};
    }

    uint32_t gemHandle = 0;
    if (drmPrimeFDToHandle(drmFd, desc->objects[0].fd, &gemHandle) != 0) [[unlikely]] {
        esyslog("vaapivideo/display: drmPrimeFDToHandle failed: %s", strerror(errno));
        av_frame_free(&mappedFrame);
        return {};
    }

    // Build the per-plane arrays required by drmModeAddFB2WithModifiers().
    const uint32_t format = DRM_FORMAT_NV12;
    const auto width = static_cast<uint32_t>(srcFrame->width);
    const auto height = static_cast<uint32_t>(srcFrame->height);

    uint32_t handles[4] = {0};
    uint32_t pitches[4] = {0};
    uint32_t offsets[4] = {0};
    uint64_t modifiers[4] = {0};

    int planeIdx = 0;
    for (int i = 0; i < desc->nb_layers && planeIdx < 4; ++i) {
        const auto &layer = desc->layers[i];
        for (int j = 0; j < layer.nb_planes && planeIdx < 4; ++j) {
            const auto &plane = layer.planes[j];
            handles[planeIdx] = gemHandle;
            pitches[planeIdx] = static_cast<uint32_t>(plane.pitch);
            offsets[planeIdx] = static_cast<uint32_t>(plane.offset);
            modifiers[planeIdx] = desc->objects[plane.object_index].format_modifier;
            planeIdx++;
        }
    }

    // NV12 has exactly 2 planes: Y (luma) and UV (interleaved chroma).
    if (planeIdx != 2) [[unlikely]] {
        esyslog("vaapivideo/display: unexpected plane count %d (expected 2 for NV12)", planeIdx);
        drm_gem_close closeArgs{.handle = gemHandle, .pad = 0};
        drmIoctl(drmFd, DRM_IOCTL_GEM_CLOSE, &closeArgs);
        av_frame_free(&mappedFrame);
        return {};
    }

    uint32_t fbId = 0;
    if (drmModeAddFB2WithModifiers(drmFd, width, height, format, handles, pitches, offsets, modifiers, &fbId,
                                   DRM_MODE_FB_MODIFIERS) != 0) {
        esyslog("vaapivideo/display: drmModeAddFB2WithModifiers failed: %s", strerror(errno));
        drm_gem_close closeArgs{.handle = gemHandle, .pad = 0};
        drmIoctl(drmFd, DRM_IOCTL_GEM_CLOSE, &closeArgs);
        av_frame_free(&mappedFrame);
        return {};
    }

    DrmFramebuffer fb;
    fb.drmFd = drmFd;
    fb.fbId = fbId;
    fb.gemHandle = gemHandle;
    fb.width = width;
    fb.height = height;
    fb.modifier = modifiers[0];
    // Move AVFrame into the DrmFramebuffer to keep the VAAPI surface alive while KMS scans out the DMA-BUF.
    fb.frame = vaapiFrame->avFrame;
    vaapiFrame->avFrame = nullptr;
    vaapiFrame->ownsFrame = false;

    av_frame_free(&mappedFrame);

    return fb;
}

auto cVaapiDisplay::OnPageFlipEvent([[maybe_unused]] int fd, [[maybe_unused]] unsigned int seq,
                                    [[maybe_unused]] unsigned int sec, [[maybe_unused]] unsigned int usec, void *data)
    -> void {
    // drmHandleEvent() callback; 'data' is the userdata passed to drmModeAtomicCommit().
    auto *display = static_cast<cVaapiDisplay *>(data);
    if (display) {
        display->isFlipPending.store(false, std::memory_order_release);
    }
}

[[nodiscard]] auto cVaapiDisplay::PresentBuffer(const DrmFramebuffer &fb) -> bool {
    if (!fb.IsValid()) {
        return false;
    }

    // Filter graph already scaled to display-fitted dimensions (DAR-preserving letterbox/pillarbox).
    // Center the frame on screen; DRM presents at 1:1 pixel mapping.
    const uint32_t destX = (fb.width < outputWidth) ? (outputWidth - fb.width) / 2 : 0;
    const uint32_t destY = (fb.height < outputHeight) ? (outputHeight - fb.height) / 2 : 0;

    AtomicRequest req;
    req.AddProperty(videoPlaneId, videoProps.crtcId, crtcId);
    req.AddProperty(videoPlaneId, videoProps.fbId, fb.fbId);
    // Explicitly set BT.709 limited-range to match scale_vaapi output; driver default may mismatch.
    if (videoProps.colorEncodingValid) {
        req.AddProperty(videoPlaneId, videoProps.colorEncoding, videoProps.colorEncodingBt709);
    }
    if (videoProps.colorRangeValid) {
        req.AddProperty(videoPlaneId, videoProps.colorRange, videoProps.colorRangeLimited);
    }
    req.AddProperty(videoPlaneId, videoProps.srcX, 0);
    req.AddProperty(videoPlaneId, videoProps.srcY, 0);
    req.AddProperty(videoPlaneId, videoProps.srcW, static_cast<uint64_t>(fb.width) << 16);
    req.AddProperty(videoPlaneId, videoProps.srcH, static_cast<uint64_t>(fb.height) << 16);
    req.AddProperty(videoPlaneId, videoProps.crtcX, destX);
    req.AddProperty(videoPlaneId, videoProps.crtcY, destY);
    req.AddProperty(videoPlaneId, videoProps.crtcW, fb.width);
    req.AddProperty(videoPlaneId, videoProps.crtcH, fb.height);

    // Bundle any pending OSD change into the same atomic commit as the video flip so both planes switch on the exact
    // same vblank.
    bool osdCommitted = false;
    {
        const cMutexLock lock(&osdMutex);
        if (osdDirty) {
            if (currentOsd.fbId != 0) {
                AppendOsdPlane(req, currentOsd);
                osdCommitted = true;
            } else if (osdPlaneId != 0) {
                req.AddProperty(osdPlaneId, osdProps.fbId, 0);
                req.AddProperty(osdPlaneId, osdProps.crtcId, 0);
                osdCommitted = true;
            }
        }
    }

    const bool success = AtomicCommit(req, 0);

    // Clear dirty only on success so a failed commit retries next time.
    if (success && osdCommitted) {
        const cMutexLock lock(&osdMutex);
        osdDirty = false;
    }

    return success;
}

auto cVaapiDisplay::WaitForPageFlip(int timeoutMs) -> void {
    const cTimeMs deadline(timeoutMs);
    while (isFlipPending.load(std::memory_order_relaxed) && !deadline.TimedOut()) {
        // Abort early on shutdown or stream-switch so callers are not stalled.
        if (isStopping.load(std::memory_order_relaxed) || !isReady.load(std::memory_order_relaxed) ||
            isClearing.load(std::memory_order_relaxed)) {
            break;
        }
        (void)DrainDrmEvents(5);
    }
}
