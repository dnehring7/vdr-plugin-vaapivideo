// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file display.cpp
 * @brief DRM atomic-modeset display: VAAPI->PRIME import + page-flip pacing.
 *
 * Threading:
 *   - Producer (decoder thread): SubmitFrame() under bufferMutex; one-slot pendingFrame.
 *   - Consumer (this cThread Action()): map -> commit -> waits for page-flip event.
 *   - Stream-switch (Clear() on the main thread): BeginStreamSwitch() takes importMutex
 *     so the consumer cannot start a new av_hwframe_map() while the codec is being torn down.
 *   - OSD updates (any thread): SetOsd() under osdMutex; consumer bundles OSD changes into
 *     the next video commit to share a single vblank.
 *
 * Lock order: importMutex BEFORE bufferMutex BEFORE osdMutex. WaitForPageFlip() must run
 * BEFORE acquiring importMutex (BeginStreamSwitch) -- otherwise the consumer can't drain
 * DRM events to clear isFlipPending and we deadlock.
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

constexpr int DISPLAY_PAGE_FLIP_TIMEOUT_MS = 40; ///< ~2 vblank periods @ 50 Hz: covers one missed flip + retry
constexpr int MAX_DRAIN_ITERATIONS = 10;         ///< Bound on post-shutdown DRM-event drain loops (defensive)

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
    // Null the source so its destructor is a no-op.
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
    // propId==0 means the optional property wasn't found during BindDrmPlane discovery
    // (e.g. zpos, blend mode, COLOR_RANGE on older drivers). Silently skipping lets callers
    // unconditionally enqueue every property without per-driver branching at every call site.
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
    // Hand over ownership; null source so its destructor releases nothing.
    other.drmFd = -1;
    other.fbId = 0;
    other.gemHandle = 0;
    other.frame = nullptr;
}

cVaapiDisplay::DrmFramebuffer::~DrmFramebuffer() noexcept {
    // STRICT release order -- reversing any of these is a kernel use-after-free:
    //   1. AVFrame: drops the VA surface ref, which is what backs the DMA-BUF fd.
    //   2. KMS FB:  CRTC stops scanning, kernel drops its DMA-BUF ref.
    //   3. GEM:     the imported BO is finally freed by the DRM driver.
    // Doing (2)/(3) before (1) would free a surface the CRTC is still scanning out.
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
    // Only the v1 page-flip handler is wired; nulling the others is intentional so libdrm
    // doesn't dispatch through stale pointers if the kernel queues an event type we don't
    // expect (vblank, sequence, page_flip2).
    : eventContext{.version = DRM_EVENT_CONTEXT_VERSION,
                   .vblank_handler = nullptr,
                   .page_flip_handler = OnPageFlipEvent,
                   .page_flip_handler2 = nullptr,
                   .sequence_handler = nullptr} {
    dsyslog("vaapivideo/display: created");
}

cVaapiDisplay::~cVaapiDisplay() noexcept {
    dsyslog("vaapivideo/display: destroying (ready=%d)", ready.load(std::memory_order_relaxed));
    Shutdown();
}
// ============================================================================
// === PUBLIC API ===
// ============================================================================

auto cVaapiDisplay::AwaitOsdHidden(uint32_t fbId) -> void {
    // Called by cVaapiOsd::~cVaapiOsd before the underlying dumb buffer is freed: we must
    // not let the destructor return while the KMS plane still references fbId, or the
    // kernel scanout will read freed memory.
    if (fbId == 0 || !ready.load(std::memory_order_relaxed)) {
        return;
    }

    // Wait for PresentBuffer() to clear osdDirty AND advance currentOsd off this fbId.
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
    // Step 1: signal the consumer to stop accepting new frames. Must come first so the
    // consumer hits the isClearing idle branch promptly and doesn't start one more import.
    isClearing.store(true, std::memory_order_release);

    // Step 2: drop any queued frame. Otherwise SubmitFrame() callers blocked on the slot
    // would wait until their timeout, stalling the decoder during the switch.
    {
        const cMutexLock lock(&bufferMutex);
        pendingFrame.reset();
        frameSlotCond.Broadcast();
    }

    // Step 3: wait for the in-flight page-flip BEFORE taking importMutex. Reversing the
    // order deadlocks: importMutex held here would prevent the consumer from running
    // DrainDrmEvents() to clear isFlipPending.
    WaitForPageFlip(DISPLAY_PAGE_FLIP_TIMEOUT_MS);

    // Step 4: lock importMutex so the consumer cannot start a new av_hwframe_map() while
    // the codec is being destroyed. By this point the consumer is either spinning in the
    // isClearing idle branch or about to re-check isClearing under importMutex -- both safe.
    // We deliberately KEEP the displayed/pendingBuffer alive: the last decoded picture
    // stays on screen during the switch instead of flashing to black.
    importMutex.Lock();
}

auto cVaapiDisplay::EndStreamSwitch() -> void {
    // Codec is destroyed; release the import gate. Clearing isClearing AFTER the unlock is
    // safe because the consumer re-checks isClearing under importMutex before touching any
    // surface, so it can't observe a stale "false" mid-teardown.
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
    // Some EDIDs report vrefresh==0 for non-CEA modes; default to 50 Hz so downstream
    // A/V sync math never divides by zero. The 50 Hz figure is the DVB baseline and is
    // intentionally consistent with decoder.cpp's framerate fallback in InitFilterGraph()
    // -- changing one without the other will desync the controllers, so update both.
    refreshRate = displayMode.vrefresh > 0 ? displayMode.vrefresh : 50;
    aspectRatio = static_cast<double>(outputWidth) / static_cast<double>(outputHeight);

    hwDeviceRef = av_buffer_ref(hwDevice);
    if (!hwDeviceRef) {
        esyslog("vaapivideo/display: failed to ref hw device");
        return false;
    }

    // Init order matters:
    //   1. property IDs -- every subsequent atomic commit needs them
    //   2. video plane (NV12) -- mandatory
    //   3. OSD plane (ARGB8888) -- optional, ignored on hardware that lacks a second plane
    //   4. display mode -- one ALLOW_MODESET commit
    //   5. ready=true + Start() -- only now is the consumer thread allowed to run
    if (!LoadDrmProperties()) {
        esyslog("vaapivideo/display: failed to cache DRM properties");
        return false;
    }

    if (!BindDrmPlane(0, DRM_FORMAT_NV12)) {
        esyslog("vaapivideo/display: no video plane found");
        return false;
    }

    // OSD is best-effort: a missing ARGB plane only loses on-screen overlays, not playback.
    (void)BindDrmPlane(0, DRM_FORMAT_ARGB8888);

    if (!ApplyDisplayMode(displayMode)) {
        esyslog("vaapivideo/display: failed to set display mode");
        return false;
    }

    ready.store(true, std::memory_order_release);
    Start();

    isyslog("vaapivideo/display: initialized %ux%u@%uHz", outputWidth, outputHeight, refreshRate);
    return true;
}

[[nodiscard]] auto cVaapiDisplay::IsInitialized() const noexcept -> bool {
    return ready.load(std::memory_order_acquire);
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

    // Mark dirty unconditionally, even for an identical (fbId, geometry) pair: VDR may
    // have repainted into the same dumb buffer in place, and on Intel/AMD the kernel
    // only invalidates FBC (Framebuffer Compression) and PSR tile caches when a commit
    // touches the plane. Without the dirty flag the screen would show stale OSD pixels.
    currentOsd = osd;
    osdDirty = true;
}

auto cVaapiDisplay::Shutdown() -> void {
    const bool wasInitialized = ready.exchange(false, std::memory_order_acq_rel);
    dsyslog("vaapivideo/display: shutting down (wasInitialized=%d)", wasInitialized);

    if (!wasInitialized) {
        dsyslog("vaapivideo/display: already shut down, skipping");
        return;
    }

    // Order: isClearing BEFORE stopping. isClearing alone gates new imports; stopping
    // alone tells the loop to exit. If we set stopping first, the consumer could observe
    // (stopping=false, isClearing=false) on its next iteration and start one more import
    // before noticing the exit signal.
    isClearing.store(true, std::memory_order_release);
    stopping.store(true, std::memory_order_release);

    // Force-clear the flip-pending flag: if the CRTC was disabled externally or the
    // display disconnected, the page-flip event will never arrive and the consumer
    // would otherwise spin forever in DrainDrmEvents(5).
    isFlipPending.store(false, std::memory_order_release);

    frameSlotCond.Broadcast();
    Cancel(1);

    const cTimeMs timeout(500);
    while (!hasExited.load(std::memory_order_acquire) && !timeout.TimedOut()) {
        frameSlotCond.Broadcast();
        isFlipPending.store(false, std::memory_order_release);
        cCondWait::SleepMs(10);
    }

    if (!hasExited.load(std::memory_order_acquire)) {
        esyslog("vaapivideo/display: thread did not exit in 500ms, waiting longer...");
        const cTimeMs timeout2(2000);
        while (!hasExited.load(std::memory_order_acquire) && !timeout2.TimedOut()) {
            frameSlotCond.Broadcast();
            isFlipPending.store(false, std::memory_order_release);
            cCondWait::SleepMs(50);
        }
    }

    if (!hasExited.load(std::memory_order_acquire)) {
        esyslog("vaapivideo/display: thread did not exit - may cause resource leak");
    }

    // Drain page-flip events that landed after the consumer exited. Without this, the
    // kernel keeps the events pending on the fd and the next process to open the device
    // would inherit them.
    for (int i = 0; i < MAX_DRAIN_ITERATIONS && DrainDrmEvents(0); ++i) {
    }

    // Detach planes + deactivate CRTC so fbcon (or the next DRM client) can take over.
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
        // CRTC teardown: ACTIVE=0, MODE_ID=0, and unbind the connector. All three in one
        // atomic commit so the kernel never observes a half-disabled CRTC state.
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

    // Free VAAPI hw context + KMS mode blob. The mode blob is only safe to destroy AFTER
    // the CRTC has been disabled above (kernel still references it otherwise). Both fields
    // are nulled so a second Shutdown() is a true no-op even past the wasInitialized guard.
    if (hwDeviceRef) {
        av_buffer_unref(&hwDeviceRef);
    }
    if (modeBlobId != 0 && drmFd >= 0) {
        drmModeDestroyPropertyBlob(drmFd, modeBlobId);
        modeBlobId = 0;
    }
}

[[nodiscard]] auto cVaapiDisplay::SubmitFrame(std::unique_ptr<VaapiFrame> frame, int timeoutMs) -> bool {
    // Cheap relaxed pre-checks; bufferMutex below provides the real synchronization.
    if (!frame || !ready.load(std::memory_order_relaxed) || isClearing.load(std::memory_order_relaxed)) [[unlikely]] {
        return false;
    }

    const cMutexLock lock(&bufferMutex);

    // pendingFrame is a single slot. The decoder calls this with timeoutMs > 0 to block on
    // VSync backpressure -- that's how the decoder paces itself to vrefresh.
    if (pendingFrame) {
        if (timeoutMs == 0) {
            return false;
        }

        // Negative ("infinite") timeout is clamped to 1 s slices so a stream switch never
        // sleeps past the isClearing transition below.
        const int waitMs = (timeoutMs < 0) ? 1000 : timeoutMs;
        const cTimeMs deadline(waitMs);
        while (pendingFrame && ready.load(std::memory_order_relaxed) && !deadline.TimedOut()) {
            // Bail on stream switch -- BeginStreamSwitch() also takes bufferMutex to drain
            // pendingFrame, so continuing to wait would deadlock the switch on us.
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

    while (!stopping.load(std::memory_order_relaxed) && ready.load(std::memory_order_relaxed)) {
        // Drain any queued DRM events first so isFlipPending reflects reality before we
        // make decisions on it. Non-blocking; just consumes whatever poll() already has.
        while (DrainDrmEvents(0)) {
        }

        // VSync gate: wait for the previous flip's event before queuing the next one.
        // 5 ms is short enough that the page_flip_handler latency is the bottleneck, not us.
        if (isFlipPending.load(std::memory_order_relaxed)) {
            (void)DrainDrmEvents(5);
            continue;
        }

        // Stream switch in progress: yield so BeginStreamSwitch() can take importMutex
        // (which we'd otherwise hold across the entire map/commit block below).
        if (isClearing.load(std::memory_order_relaxed)) {
            cCondWait::SleepMs(5);
            continue;
        }

        // importMutex is held across map AND commit. Releasing between them would let
        // BeginStreamSwitch() race in, free the codec context, and leave us committing
        // a freed VAAPI surface.
        bool frameCommitted = false;
        {
            const cMutexLock importLock(&importMutex);

            // Re-check under the lock: BeginStreamSwitch() may have flipped isClearing
            // between the unlocked check above and now.
            if (!isClearing.load(std::memory_order_acquire) && !stopping.load(std::memory_order_acquire) &&
                ready.load(std::memory_order_acquire)) {
                std::unique_ptr<VaapiFrame> frameToShow;
                {
                    const cMutexLock lock(&bufferMutex);
                    if (pendingFrame) {
                        frameToShow = std::move(pendingFrame);
                        frameSlotCond.Broadcast();
                    }
                }

                if (frameToShow && !isClearing.load(std::memory_order_acquire)) {
                    DrmFramebuffer newFb = MapVaapiFrame(std::move(frameToShow));

                    // MapVaapiFrame is the slow step (PRIME export + GEM import + AddFB2);
                    // re-check isClearing one more time before committing.
                    if (newFb.IsValid() && !isClearing.load(std::memory_order_acquire)) {
                        const cMutexLock lock(&bufferMutex);
                        if (PresentBuffer(newFb)) {
                            // Buffer chain (displayed <- pending <- new) advances ONLY on a
                            // successful commit. A failed commit must not destroy the currently-
                            // scanned FB or the kernel reads freed memory on next scanout.
                            displayedBuffer = std::move(pendingBuffer);
                            pendingBuffer = std::move(newFb);
                            frameCommitted = true;
                        }
                    }
                }
            }
        }

        // No new frame arrived this iteration: re-present the previous buffer. Two reasons:
        //   1. Keeps the page-flip cadence going so OSD geometry changes (which piggyback
        //      on the video commit in PresentBuffer) don't stall waiting for the next
        //      decoded frame -- otherwise OSD updates would freeze on a paused stream.
        //   2. Maintains continuous vsync timing for the decoder's backpressure on us.
        if (!frameCommitted && !isClearing.load(std::memory_order_relaxed)) {
            const cMutexLock lock(&bufferMutex);
            if (pendingBuffer.IsValid()) {
                (void)PresentBuffer(pendingBuffer);
            } else {
                // Pre-first-frame: nothing to present, don't spin.
                cCondWait::SleepMs(5);
            }
        }
    }

    hasExited.store(true, std::memory_order_release);
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

    // KMS atomic-check rejects commits where CRTC_X+CRTC_W exceeds the CRTC width (and
    // similarly for Y). Clipping here is cheaper than letting the whole commit fail.
    const auto clippedW = std::min(osd.width, outputWidth - static_cast<uint32_t>(osd.x));
    const auto clippedH = std::min(osd.height, outputHeight - static_cast<uint32_t>(osd.y));

    if (clippedW == 0 || clippedH == 0) {
        return;
    }

    req.AddProperty(osdPlaneId, osdProps.crtcId, crtcId);
    req.AddProperty(osdPlaneId, osdProps.fbId, osd.fbId);
    req.AddProperty(osdPlaneId, osdProps.srcX, 0);
    req.AddProperty(osdPlaneId, osdProps.srcY, 0);
    // SRC_* properties use 16.16 fixed-point (value = pixels << 16).
    req.AddProperty(osdPlaneId, osdProps.srcW, static_cast<uint64_t>(clippedW) << 16);
    req.AddProperty(osdPlaneId, osdProps.srcH, static_cast<uint64_t>(clippedH) << 16);
    req.AddProperty(osdPlaneId, osdProps.crtcX, static_cast<uint64_t>(osd.x));
    req.AddProperty(osdPlaneId, osdProps.crtcY, static_cast<uint64_t>(osd.y));
    req.AddProperty(osdPlaneId, osdProps.crtcW, clippedW);
    req.AddProperty(osdPlaneId, osdProps.crtcH, clippedH);

    // VDR's tColor stores straight (non-premultiplied) ARGB. Pick "Coverage" blending
    // (enum value 1) rather than "Pre-multiplied" (0) to avoid double-multiplied alpha,
    // which produces visibly darker OSD edges and ringing on translucent backgrounds.
    if (osdProps.pixelBlendMode != 0) {
        req.AddProperty(osdPlaneId, osdProps.pixelBlendMode, 1);
    }
    // zpos is intentionally NOT written. Observed behavior:
    //   * On at least one tested Intel i915 stack, writing zpos here caused the entire
    //     atomic commit to fail (the property is immutable on its overlay planes).
    //   * On the AMD/amdgpu stacks tested, the plane stacking the driver picks already
    //     puts our OSD plane above the video plane, so an override isn't needed.
    // This is empirical, not derived from a stable spec, so re-test on new hardware
    // before re-introducing a zpos write.
}

[[nodiscard]] auto cVaapiDisplay::AtomicCommit(AtomicRequest &req, uint32_t flags) -> bool {
    if (req.Count() == 0) {
        return true; // empty commit -- nothing to do, treat as success
    }

    // Page-flip path uses NONBLOCK + PAGE_FLIP_EVENT so the kernel queues the work and
    // delivers a completion event we drain in Action(). Modeset path (ALLOW_MODESET) is
    // synchronous and event-less because mode programming must be sequenced against the
    // initial CRTC enable.
    //
    // We do NOT use DRM_MODE_ATOMIC_ASYNC: that flag requires linear (untiled) buffer
    // modifiers, but VAAPI scanout surfaces are tiled (Y/X-tiled or compressed CCS) and
    // the kernel rejects async commits on them with -EINVAL.
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

    // EBUSY = the previous flip's event has not been consumed yet. The Action() loop
    // retries on the next iteration after DrainDrmEvents() catches up, so it's not worth
    // logging on the hot path.
    if (errno != EBUSY) {
        esyslog("vaapivideo/display: atomic commit failed - %s (flags=0x%x)", strerror(errno), commitFlags);
    }
    return false;
}

[[nodiscard]] auto cVaapiDisplay::ApplyDisplayMode(const drmModeModeInfo &mode) -> bool {
    dsyslog("vaapivideo/display: setting display mode %ux%u@%uHz", mode.hdisplay, mode.vdisplay, mode.vrefresh);
    // KMS takes the mode as a property blob (kernel-side). Destroy the previous blob to
    // release the kernel allocation -- libdrm has no GC for these.
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
    // Find the planeIndex'th plane that supports the given fourcc on our CRTC, cache its
    // atomic property IDs, and assign it as either the video or OSD plane (whichever is
    // still unbound). Format support is read from the IN_FORMATS blob (the modern modifier-
    // aware list); the legacy plane->formats array doesn't include tiling info, which we
    // need because VAAPI surfaces are always tiled.
    auto planeRes = std::unique_ptr<drmModePlaneRes, decltype(&drmModeFreePlaneResources)>(
        drmModeGetPlaneResources(drmFd), drmModeFreePlaneResources);
    auto res =
        std::unique_ptr<drmModeRes, decltype(&drmModeFreeResources)>(drmModeGetResources(drmFd), drmModeFreeResources);

    if (!planeRes || !res) {
        return false;
    }

    // possible_crtcs is a bitmask indexed by *position* in res->crtcs, NOT by CRTC object
    // id. Resolve our crtcId to its array index before testing the mask below.
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

        // Skip the video plane on the second pass (we're then looking for the OSD plane).
        if (videoPlaneId != 0 && plane->plane_id == videoPlaneId) {
            continue;
        }

        if (!(plane->possible_crtcs & (1U << crtcIndex))) {
            continue;
        }

        auto planeProps = std::unique_ptr<drmModeObjectProperties, decltype(&drmModeFreeObjectProperties)>(
            drmModeObjectGetProperties(drmFd, plane->plane_id, DRM_MODE_OBJECT_PLANE), drmModeFreeObjectProperties);
        if (!planeProps) {
            continue;
        }

        // One sweep over the plane's properties: format support (via IN_FORMATS blob),
        // plane type, and every atomic prop id we'll later need to write to. Doing it in
        // one pass avoids re-querying drm_mode_obj_get_props per property.
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

            // IN_FORMATS blob lists (format, modifier) pairs the plane accepts. We only
            // need to confirm the format is present -- the actual modifier check happens
            // implicitly when KMS validates the FB at commit time.
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
    // Single-consumer in steady state (the Action thread). Shutdown() also calls this
    // from the main thread to drain residual events post-Cancel(), but only AFTER the
    // Action thread has set hasExited, so there is no overlap.
    pollfd pfd{.fd = drmFd, .events = POLLIN, .revents = 0};
    const int ret = poll(&pfd, 1, timeoutMs);

    if (ret > 0 && (pfd.revents & POLLIN)) {
        return drmHandleEvent(drmFd, &eventContext) == 0;
    }
    return false;
}

[[nodiscard]] auto cVaapiDisplay::LoadDrmProperties() -> bool {
    // Both client caps must be opted into per-fd: UNIVERSAL_PLANES exposes overlay planes
    // (without it the kernel only reports primary/cursor), ATOMIC switches us to the atomic
    // modeset uAPI we use for everything.
    (void)drmSetClientCap(drmFd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1);
    (void)drmSetClientCap(drmFd, DRM_CLIENT_CAP_ATOMIC, 1);

    // Proxy capability gate. We do NOT actually use async flips (see AtomicCommit -- they
    // require linear modifiers, our surfaces are tiled), but DRM_CAP_ATOMIC_ASYNC_PAGE_FLIP
    // is reliably present on the kernels we've tested against and serves as a single
    // "kernel new enough for our atomic-modeset assumptions" check.
    // CAUTION: do not infer a precise kernel version from the presence of this cap.
    //   Replace with an explicit drmGetVersion() check if a future kernel ships atomic
    //   modeset without this cap, OR if any of our other atomic assumptions need their
    //   own version gate.
    uint64_t asyncCap = 0;
    if (drmGetCap(drmFd, DRM_CAP_ATOMIC_ASYNC_PAGE_FLIP, &asyncCap) != 0 || asyncCap == 0) {
        esyslog("vaapivideo/display: required DRM atomic capability missing "
                "(DRM_CAP_ATOMIC_ASYNC_PAGE_FLIP) -- kernel too old or driver lacks atomic support");
        return false;
    }

    // CRTC ACTIVE / MODE_ID -- needed by every modeset commit.
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

    // Connector CRTC_ID -- needed to (un)bind the connector during enable/disable.
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

    // Export the VAAPI surface as DRM PRIME so we can wrap it in a KMS framebuffer below.
    AVFrame *mappedFrame = av_frame_alloc();
    if (!mappedFrame) [[unlikely]] {
        return {};
    }

    mappedFrame->format = AV_PIX_FMT_DRM_PRIME;
    // MAP_READ: KMS scanout reads pixels.
    // MAP_DIRECT: zero-copy -- the PRIME fd refers to the same memory as the VA surface,
    //   no intermediate copy is allocated. Without this we'd duplicate every frame on
    //   the GPU heap.
    // vaDriverMutex: serializes VA-driver entry against the decoder thread's filter-graph
    //   execution. The iHD driver's VEBOX path is not thread-safe when shared with VPP
    //   filter execution on the same VADisplay (observed sporadic VA_STATUS_ERROR_OPERATION
    //   _FAILED on iHD). The decoder takes the same lock around its filter push/pull.
    int ret = 0;
    {
        const cMutexLock vaLock(&vaDriverMutex);
        ret = av_hwframe_map(mappedFrame, srcFrame, AV_HWFRAME_MAP_READ | AV_HWFRAME_MAP_DIRECT);
    }
    if (ret < 0) [[unlikely]] {
        // EIO during teardown: the VA surface is already gone (expected race).
        // isClearing is the secondary guard for the same race. Suppress both to avoid
        // spam during channel switches.
        if (ret != AVERROR(EIO) && !isClearing.load(std::memory_order_relaxed)) {
            dsyslog("vaapivideo/display: av_hwframe_map failed: %s", AvErr(ret).data());
        }
        av_frame_free(&mappedFrame);
        return {};
    }

    const auto *desc =
        reinterpret_cast<const AVDRMFrameDescriptor *>( // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
            mappedFrame->data[0]);
    // The rest of this function assumes a single DMA-BUF object holding both NV12 layers
    // (Y + UV at different offsets). On the iHD/Mesa stacks tested this is always the
    // shape we get; on hypothetical drivers that split planes across multiple objects
    // we would need a per-object GEM import + multi-fd AddFB2 path. Reject early so the
    // failure mode is "no scanout, log line" rather than "scanout reads from one object's
    // GEM handle plus another object's offset".
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

    // drmModeAddFB2WithModifiers takes parallel arrays per FB plane (handle/pitch/offset/
    // modifier). Walk the AVDRMFrameDescriptor's layers/planes to populate them.
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

    // NV12 = Y plane + interleaved UV plane, exactly two. Anything else means the
    // descriptor doesn't actually describe NV12 and AddFB2 would reject it anyway.
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
    // Move the AVFrame ownership into the DrmFramebuffer so the VA surface ref outlives
    // the scanout. Dropping it earlier would let the VA driver recycle the surface while
    // the kernel is still reading from the DMA-BUF -> green/garbled frames on the next flip.
    fb.frame = vaapiFrame->avFrame;
    vaapiFrame->avFrame = nullptr;
    vaapiFrame->ownsFrame = false;

    av_frame_free(&mappedFrame);

    return fb;
}

auto cVaapiDisplay::OnPageFlipEvent([[maybe_unused]] int fd, [[maybe_unused]] unsigned int seq,
                                    [[maybe_unused]] unsigned int sec, [[maybe_unused]] unsigned int usec, void *data)
    -> void {
    // libdrm dispatches this from drmHandleEvent() on the consumer thread (we never call
    // drmHandleEvent from anywhere else). The `data` cookie is the `this` pointer we passed
    // to drmModeAtomicCommit. Release-store on isFlipPending makes the consumer's next
    // acquire-load see "flip done" and submit the next frame.
    auto *display = static_cast<cVaapiDisplay *>(data);
    if (display) {
        display->lastVSyncTimeMs.store(cTimeMs::Now(), std::memory_order_release);
        display->isFlipPending.store(false, std::memory_order_release);
    }
}

[[nodiscard]] auto cVaapiDisplay::PresentBuffer(const DrmFramebuffer &fb) -> bool {
    if (!fb.IsValid()) {
        return false;
    }

    // Letterbox/pillarbox by centering. The decoder's scale_vaapi already produced
    // display-fitted dimensions preserving DAR; KMS scans out 1:1 (no plane scaler on
    // the planes we use), so all we have to do here is offset.
    const uint32_t destX = (fb.width < outputWidth) ? (outputWidth - fb.width) / 2 : 0;
    const uint32_t destY = (fb.height < outputHeight) ? (outputHeight - fb.height) / 2 : 0;

    AtomicRequest req;
    req.AddProperty(videoPlaneId, videoProps.crtcId, crtcId);
    req.AddProperty(videoPlaneId, videoProps.fbId, fb.fbId);
    // Force BT.709 limited-range explicitly. Defaults vary by driver: i915 is BT.601 full
    // range, AMD is BT.709 full range. Either mismatches our scale_vaapi output (BT.709 TV)
    // and produces washed-out / green-tinted colors. Properties may be absent on older
    // drivers; AddProperty(propId==0) silently no-ops in that case.
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

    // Bundle pending OSD geometry into the same atomic commit as the video flip. Both
    // planes switch on the exact same vblank: separate commits would let one plane lag
    // by a vblank period and produce visible OSD tearing on top of moving video.
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

    // Only clear osdDirty when the commit actually landed -- a failed commit (e.g. EBUSY)
    // must keep the OSD update queued for the next attempt.
    if (success && osdCommitted) {
        const cMutexLock lock(&osdMutex);
        osdDirty = false;
    }

    return success;
}

auto cVaapiDisplay::WaitForPageFlip(int timeoutMs) -> void {
    // Called from BeginStreamSwitch() on the main thread BEFORE taking importMutex (the
    // ordering rule documented at the top of this file). DrainDrmEvents() here works
    // because the consumer thread runs poll() too -- only one of us actually sees a
    // given event, and isFlipPending's release/acquire serializes the result.
    const cTimeMs deadline(timeoutMs);
    while (isFlipPending.load(std::memory_order_relaxed) && !deadline.TimedOut()) {
        // Bail on shutdown / stream-switch so callers don't sit through the full timeout
        // when the answer is "we're tearing down anyway".
        if (stopping.load(std::memory_order_relaxed) || !ready.load(std::memory_order_relaxed) ||
            isClearing.load(std::memory_order_relaxed)) {
            break;
        }
        (void)DrainDrmEvents(5);
    }
}
