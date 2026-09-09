// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file display.cpp
 * @brief DRM atomic-modeset display: VAAPI->PRIME import and page-flip pacing.
 *
 * Threading model:
 *   Producer (decoder):    SubmitFrame() under bufferMutex; pushes onto pendingFrames (DISPLAY_PRERENDER_SLOTS deep).
 *   Consumer (Action()):   map -> commit -> drain page-flip event.
 *   Stream-switch (main):  BeginStreamSwitch() holds importMutex while codec tears down.
 *   OSD (any thread):      SetOsd() under osdMutex; bundled into next video commit.
 *
 * Lock order: importMutex -> vaDriverMutex (frame import); importMutex -> bufferMutex ->
 * vaDriverMutex (releasing a DrmFramebuffer / clearing pendingFrames can drop the last
 * FilterGraphToken, whose deleter locks vaDriverMutex and nothing else). vaDriverMutex may
 * precede only the leaf mutexes {videoRectMutex, osdMutex, hdrStateMutex} -- decoder rebuilds
 * query geometry/HDR state under it -- and the leaves have no outgoing edges, so the lock
 * graph stays acyclic.
 * PresentBuffer() may run under bufferMutex; its leaf locks
 * {videoRectMutex, osdMutex, hdrStateMutex} are never nested with each other.
 * DRM fd rule: the consumer thread is the ONLY drmHandleEvent dispatcher while Action()
 * runs; WaitForPageFlip() observes isFlipPending without touching the fd. A second reader
 * was bisect-verified to permanently halve the post-switch present cadence on i915/UHD.
 */

#include "display.h"
#include "caps.h"
#include "common.h"
#include "config.h"
#include "decoder.h"

// POSIX
#include <sys/poll.h>

// C++ Standard Library
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <utility>
#include <vector>

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
#include <libavutil/mastering_display_metadata.h>
#include <libavutil/pixfmt.h>
#include <libavutil/rational.h>
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
#include <vdr/osd.h>
#include <vdr/thread.h>
#include <vdr/tools.h>
#pragma GCC diagnostic pop

// ============================================================================
// === CONSTANTS ===
// ============================================================================

namespace {

// --- Prerender queue ---
constexpr size_t DISPLAY_PRERENDER_SLOTS =
    8; ///< Decoder->display handoff queue depth (= 160 ms tolerance @ 50 fps). Sized to absorb a
       ///< single UHD VPP/memory-bandwidth spike (observed ~80 ms in replay) AND the per-frame
       ///< variance of CPU-side SW decoders (libdav1d 1080p50 spikes 30-40 ms on complex frames)
       ///< without draining the cache and forcing a re-present. SubmitFrame blocks when all slots
       ///< are full so audio clock stays in lipsync (the whole pipeline is delayed in lockstep, not
       ///< just video). FHD HW paths never fill past 1-2 slots; the extra depth is a no-op there.
       ///< COUPLED to DISPLAY_UNDERRUN_THRESHOLD_VSYNCS (= SLOTS + 2) below; revisit that margin if
       ///< you change this (the relationship is not linear -- see the note at that definition).

// --- Page flip ---
constexpr int DISPLAY_PAGE_FLIP_TIMEOUT_MS = 40; ///< ~2 vblanks @ 50 Hz: tolerates one missed flip before giving up
constexpr uint64_t DISPLAY_PAGE_FLIP_STUCK_MS =
    200; ///< Stuck-flip watchdog: force-clear isFlipPending if the kernel swallows the page-flip event.
constexpr int DISPLAY_MAX_DRAIN_ITERATIONS =
    10; ///< Safety bound on post-shutdown DRM event drain (guards against infinite loops)
constexpr uint32_t PAGE_FLIP_COMMIT_FLAGS = DRM_MODE_PAGE_FLIP_EVENT | DRM_MODE_ATOMIC_NONBLOCK;
constexpr int DISPLAY_ATOMIC_FAILURE_LOG_INTERVAL_MS = 1000; ///< Commit-failure log rate limit (retry path ~200 Hz)

// --- Runtime mode change ---
constexpr uint64_t DISPLAY_MODE_RESIZE_WAIT_MS =
    2000; ///< Watchdog on awaitingResizedFb. After a mode change the video plane stays dark until a
          ///< framebuffer matching the new scanout rect arrives (a stale-sized one would be cropped,
          ///< not scaled). Should take a frame or two; past this the gate opens unconditionally so a
          ///< filter chain that never delivers the expected size cannot strand the screen on black.

// --- Underrun / warmup-grace tracking (display consumer thread) ---
constexpr uint64_t DISPLAY_UNDERRUN_IDLE_MAX_MS =
    10000; ///< Gap beyond which a re-present streak is treated as paused / stopped, not an underrun: a
           ///< stream-paused / radio-mode / suspended-stream scenario can leave lastFrameCommitMs hours in
           ///< the past, and the recovery log on resume would otherwise scream a multi-hour "queue refilled"
           ///< event. inTrick / inSyncSleep catch the explicit pause paths; this is the catch-all.
constexpr int DISPLAY_UNDERRUN_LOG_INTERVAL_MS = 2000; ///< Min interval between underrun-onset dsyslog lines.
constexpr auto DISPLAY_UNDERRUN_THRESHOLD_VSYNCS = static_cast<unsigned>(DISPLAY_PRERENDER_SLOTS + 2);
///< Empty-VSync streak that trips an underrun log (thresholdMs = DISPLAY_UNDERRUN_THRESHOLD_VSYNCS * vsyncMs).
///< SLOTS + 2 is tightly coupled to DISPLAY_PRERENDER_SLOTS:
///<   SLOTS -> one missed VSync inside a freshly-drained queue (queue absorbs it, no log)
///<   + 1   -> grace VSync for the decoder to catch up (single hiccup, no log)
///<   + 1   -> one more sample so the trigger fires on sustained gaps, not transients
///< Revisit the +2 margin if you change DISPLAY_PRERENDER_SLOTS -- the relationship doesn't scale linearly:
///< at slots=1 you'd want +3 (more noise), at slots=8 +2 is plenty.
constexpr int DISPLAY_WARMUP_ACTIVE_WINDOW_MS =
    500; ///< Min idle gap on a fresh commit that arms the warmup grace (does not gate the underrun log).
constexpr int DISPLAY_WARMUP_GRACE_MS =
    3000; ///< Post-idle grace suppressing underrun logs while the pipeline re-anchors after a resume.

// --- Stream-start trace milestone (cVaapiDisplay::startTrace; see StreamStartTrace in common.h) ---
constexpr uint32_t TRACE_FIRST_COMMIT = 1U << 0; ///< First fresh frame committed to the CRTC after the switch

// ============================================================================
// === HELPER FUNCTIONS ===
// ============================================================================

[[nodiscard]] auto GetPlaneTypeName(uint32_t type) -> const char * {
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

// Read a DRM device cap, returning 0 if unsupported. Used only by the one-time init diagnostic.
[[nodiscard]] auto GetDrmCap(int drmFd, uint64_t cap) noexcept -> uint64_t {
    uint64_t value = 0;
    return drmGetCap(drmFd, cap, &value) == 0 ? value : uint64_t{0};
}

} // namespace

// ============================================================================
// === ATOMIC REQUEST ===
// ============================================================================

AtomicRequest::AtomicRequest() : request(drmModeAtomicAlloc()) {}

AtomicRequest::~AtomicRequest() noexcept {
    if (request) {
        drmModeAtomicFree(request);
    }
}

AtomicRequest::AtomicRequest(AtomicRequest &&other) noexcept
    : failed(other.failed), propCount(other.propCount), request(other.request) {
    other.request = nullptr; // prevents double-free in moved-from destructor
    other.failed = false;
    other.propCount = 0;
}

auto AtomicRequest::operator=(AtomicRequest &&other) noexcept -> AtomicRequest & {
    if (this != &other) {
        if (request) {
            drmModeAtomicFree(request);
        }
        failed = other.failed;
        request = other.request;
        propCount = other.propCount;
        other.failed = false;
        other.request = nullptr;
        other.propCount = 0;
    }
    return *this;
}

// ============================================================================
// === ATOMIC REQUEST -- PUBLIC API ===
// ============================================================================

auto AtomicRequest::AddProperty(uint32_t objId, uint32_t propId, uint64_t value) -> void {
    // propId==0 means the driver doesn't expose this optional property (e.g. zpos, blend mode,
    // COLOR_RANGE on older kernels). Skipping silently avoids per-driver branches at every caller.
    if (propId == 0) {
        return;
    }
    // A requested property that can't be staged poisons the request: a partial commit would
    // apply a torn plane state.
    if (!request) [[unlikely]] {
        failed = true;
        return;
    }
    if (drmModeAtomicAddProperty(request, objId, propId, value) >= 0) {
        propCount++;
    } else {
        failed = true;
    }
}

[[nodiscard]] auto AtomicRequest::Count() const noexcept -> int { return propCount; }

[[nodiscard]] auto AtomicRequest::Failed() const noexcept -> bool { return failed; }

[[nodiscard]] auto AtomicRequest::Handle() const noexcept -> drmModeAtomicReq * { return request; }

// ============================================================================
// === DRM FRAMEBUFFER ===
// ============================================================================

cVaapiDisplay::DrmFramebuffer::DrmFramebuffer(DrmFramebuffer &&other) noexcept
    : drmFd(other.drmFd), fbId(other.fbId), frame(other.frame), gemHandle(other.gemHandle),
      graphToken(std::move(other.graphToken)), height(other.height), modifier(other.modifier), width(other.width) {
    other.drmFd = -1; // prevents double-release in moved-from destructor
    other.fbId = 0;
    other.gemHandle = 0;
    other.frame = nullptr;
}

cVaapiDisplay::DrmFramebuffer::~DrmFramebuffer() noexcept {
    // Release order matters: (1) AVFrame drops the VA surface ref that backs the DMA-BUF;
    // (2) drmModeRmFB tells the CRTC to stop scanning and releases the kernel DMA-BUF ref;
    // (3) DRM_IOCTL_GEM_CLOSE frees the imported BO. Reversing (1)/(2) causes the kernel to
    // read freed GPU memory on the next scanout. (4) graphToken with the members after this
    // body: the VPP graph must outlive the surface it rendered into (see FilterGraphToken).
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
        graphToken = std::move(other.graphToken); // after our releases above: old token may free its graph
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
    // Without a description cThread logs no start/end line, skips prctl(PR_SET_NAME), and its
    // "thread won't end" error names nothing.
    : cThread("vaapivideo/display"),
      // Only page_flip_handler (v1) is wired; other slots are null so libdrm doesn't dispatch
      // to stale pointers on unexpected event types (vblank, sequence, page_flip2).
      eventContext{.version = DRM_EVENT_CONTEXT_VERSION,
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
    // Called from cVaapiOsd::~cVaapiOsd before freeing the dumb buffer; must not return while
    // the kernel still scans out fbId. lastCommittedOsdFbId tracks the fbId of the most recent
    // successful OSD commit, so fbIds that never reached the kernel return immediately.
    if (fbId == 0 || !ready.load(std::memory_order_relaxed)) {
        return;
    }
    constexpr int kTimeoutMs = 500;
    const cTimeMs deadline(kTimeoutMs);
    while (!deadline.TimedOut()) {
        {
            const cMutexLock lock(&osdMutex);
            if (lastCommittedOsdFbId != fbId) {
                return;
            }
        }
        cCondWait::SleepMs(5);
    }
    esyslog("vaapivideo/display: AwaitOsdHidden timed out for fbId=%u after %d ms", fbId, kTimeoutMs);
}

auto cVaapiDisplay::BeginStreamSwitch() -> void {
    // Order matters: gate consumer, drop queue (unblock submitters), try to let the consumer
    // drain the in-flight flip, then hold importMutex for codec teardown.
    // displayedBuffer/pendingBuffer stay alive so the last frame remains on screen.
    isClearing.store(true, std::memory_order_release);
    {
        const cMutexLock lock(&bufferMutex);
        pendingFrames.clear();
        pendingDepth.store(0, std::memory_order_release);
        frameSlotCond.Broadcast();
    }
    // Best-effort (see WaitForPageFlip): if this times out, old buffers remain alive and
    // the consumer still owns DRM event dispatch, but the drain was not confirmed.
    if (!WaitForPageFlip(DISPLAY_PAGE_FLIP_TIMEOUT_MS)) [[unlikely]] {
        esyslog("vaapivideo/display: timed out waiting for page flip before stream switch");
    }
    importMutex.Lock();
    // Reset under importMutex so an in-flight fresh commit cannot republish a pre-Clear timestamp.
    lastFrameCommitMs.store(0, std::memory_order_release);
}

auto cVaapiDisplay::ArmStartTrace(uint64_t epochMs) noexcept -> void { startTrace.Arm(epochMs, TRACE_FIRST_COMMIT); }

auto cVaapiDisplay::DisarmStartTrace() noexcept -> void { startTrace.Disarm(); }

auto cVaapiDisplay::EndStreamSwitch() -> void {
    // Unlock first, then clear isClearing -- consumer re-checks isClearing under importMutex
    // so it can't observe a stale "false" mid-teardown.
    importMutex.Unlock();
    isClearing.store(false, std::memory_order_release);
}

[[nodiscard]] auto cVaapiDisplay::Initialize(int fileDescriptor, AVBufferRef *hwDevice, uint32_t crtcIdentifier,
                                             uint32_t connectorIdentifier, const drmModeModeInfo &displayMode) -> bool {
    // Rate from the timings, not drmModeModeInfo::vrefresh: that field is rounded to whole Hz, so it
    // would print this one line as "@50Hz" while every other mode line in the log says "@50.000Hz".
    dsyslog("vaapivideo/display: initializing %ux%u@%.3fHz", displayMode.hdisplay, displayMode.vdisplay,
            static_cast<double>(ModeRefreshMilliHz(displayMode)) / 1000.0);

    if (fileDescriptor < 0 || !hwDevice) [[unlikely]] {
        esyslog("vaapivideo/display: invalid parameters");
        return false;
    }

    if (ready.load(std::memory_order_acquire)) [[unlikely]] {
        esyslog("vaapivideo/display: Initialize called while already initialized");
        return false;
    }

    // Wipe probe state from a failed prior attempt: BindDrmPlane assigns the video slot only
    // while videoPlaneId==0, so a stale ID would route NV12 to the OSD slot on retry.
    displayCaps = {};
    hdrProps = {};
    modesetProps = {};
    osdPlaneId = 0;
    osdProps = {};
    videoPlaneId = 0;
    videoProps = {};
    {
        const cMutexLock lock(&osdMutex);
        currentOsd = {};
        osdDirty = false;
        osdGeneration = 0;
        // Nothing is on screen on a fresh CRTC. Reset here rather than in ResetPlaneStateCaches(),
        // which must not touch this field (see its comment); a stale id would make AwaitOsdHidden()
        // block for its full timeout on the first OSD destroyed after a re-attach.
        lastCommittedOsdFbId = 0;
    }

    drmFd = fileDescriptor;
    crtcId = crtcIdentifier;
    connectorId = connectorIdentifier;
    {
        const cMutexLock lock(&modeRequestMutex);
        activeMode = displayMode;
        modeRequest = {};
    }
    modeRequestPending.store(false, std::memory_order_release);
    modesetActive.store(false, std::memory_order_release);
    modeGeneration.store(0, std::memory_order_release);
    geometryChanged.store(false, std::memory_order_release);
    awaitingResizedFb = false;
    resizeWaitSince = 0;
    PublishOutputGeometry(displayMode);
    // Reset plane-state caches so the first commit re-writes every stateful property regardless
    // of any leftover values the kernel may carry from a prior session.
    ResetPlaneStateCaches();
    // Re-arm the OSD-over-HDR commit-path probe (see AtomicCommit).
    osdHdrNeedsModeset = false;
    osdHdrSuppressed = false;
    // Re-arm lifecycle flags: Shutdown() leaves stopping/isClearing latched and hasExited set,
    // so a re-Initialize()'d consumer thread would otherwise exit immediately.
    hasExited.store(false, std::memory_order_release);
    isClearing.store(false, std::memory_order_release);
    isFlipPending.store(false, std::memory_order_release);
    flipPendingSinceMs.store(0, std::memory_order_release);
    stopping.store(false, std::memory_order_release);

    // Local guard until init succeeds: failure paths never set ready, so Shutdown() -- the
    // usual owner of the unref -- would early-return and leak a member ref.
    std::unique_ptr<AVBufferRef, FreeAVBufferRef> hwDeviceGuard{av_buffer_ref(hwDevice)};
    if (!hwDeviceGuard) {
        esyslog("vaapivideo/display: failed to ref hw device");
        return false;
    }

    // Init order is load-bearing: each step depends on the previous one.
    //   1. LoadDrmProperties: every atomic commit needs CRTC/connector prop IDs.
    //   2. BindDrmPlane(NV12): mandatory; also populates IN_FORMATS for step 4.
    //   3. BindDrmPlane(ARGB): optional OSD plane; playback continues without it.
    //   4. ProbeHdrCapabilities: needs videoPlaneId (set in step 2) for P010 check.
    //   5. ApplyDisplayMode: ALLOW_MODESET commit; also initializes SDR connector state.
    //   6. ready=true + Start(): consumer thread must not run before step 5 completes.
    if (!LoadDrmProperties()) {
        esyslog("vaapivideo/display: failed to cache DRM properties");
        return false;
    }

    if (!BindDrmPlane(0, DRM_FORMAT_NV12)) {
        esyslog("vaapivideo/display: no video plane found");
        return false;
    }

    (void)BindDrmPlane(0, DRM_FORMAT_ARGB8888); // best-effort: no OSD plane -> no overlays, playback unaffected

    // Must run after BindDrmPlane(NV12): needs videoPlaneId to check IN_FORMATS for P010.
    ProbeHdrCapabilities();

    // blankPlanes=false: nothing is attached yet at init time, and adding plane properties for
    // planes the kernel has never seen only widens the surface for a rejected first commit.
    // No log here: ApplyDisplayMode already reported the rejected mode with its geometry, and the
    // attach path reports the overall failure. A third line would just say the same thing twice.
    if (!ApplyDisplayMode(displayMode, /*blankPlanes=*/false)) {
        return false;
    }

    hwDeviceRef = hwDeviceGuard.release();
    ready.store(true, std::memory_order_release);
    Start();

    isyslog("vaapivideo/display: initialized %ux%u@%.3fHz", GetOutputWidth(), GetOutputHeight(),
            static_cast<double>(GetOutputRefreshMilliHz()) / 1000.0);
    return true;
}

[[nodiscard]] auto cVaapiDisplay::IsInitialized() const noexcept -> bool {
    return ready.load(std::memory_order_acquire);
}

auto cVaapiDisplay::GetOutputGeometry(uint32_t &width, uint32_t &height) const noexcept -> void {
    const uint64_t packed = outputGeometry.load(std::memory_order_acquire);
    width = static_cast<uint32_t>(packed >> 32);
    height = static_cast<uint32_t>(packed & 0xFFFFFFFFULL);
}

[[nodiscard]] auto cVaapiDisplay::GetOutputWidth() const noexcept -> uint32_t {
    return static_cast<uint32_t>(outputGeometry.load(std::memory_order_acquire) >> 32);
}

[[nodiscard]] auto cVaapiDisplay::GetOutputHeight() const noexcept -> uint32_t {
    return static_cast<uint32_t>(outputGeometry.load(std::memory_order_acquire) & 0xFFFFFFFFULL);
}

[[nodiscard]] auto cVaapiDisplay::GetAspectRatio() const noexcept -> double {
    // Raster shape times pixel aspect, NOT width/height: an anamorphic CEA timing (720x576 flagged
    // 16:9) is shown 42% wider than its raster. Derived rather than stored so there is one fewer
    // atomic to keep coherent with the geometry across a mode change.
    uint32_t width = 0;
    uint32_t height = 0;
    GetOutputGeometry(width, height);
    const AspectRatio par = GetOutputPixelAspect();
    if (height == 0 || par.den == 0) [[unlikely]] {
        return DISPLAY_DEFAULT_ASPECT_RATIO;
    }
    return (static_cast<double>(width) * par.num) / (static_cast<double>(height) * par.den);
}

[[nodiscard]] auto cVaapiDisplay::GetOutputRefreshRate() const noexcept -> uint32_t {
    return (outputRefreshMilliHz.load(std::memory_order_acquire) + 500U) / 1000U;
}

[[nodiscard]] auto cVaapiDisplay::GetActiveMode() const -> drmModeModeInfo {
    const cMutexLock lock(&modeRequestMutex);
    return activeMode;
}

auto cVaapiDisplay::RequestDisplayMode(const drmModeModeInfo &mode) -> void {
    if (!ready.load(std::memory_order_acquire)) [[unlikely]] {
        return;
    }
    {
        const cMutexLock lock(&modeRequestMutex);
        modeRequest = mode;
    }
    // Published after the mode itself so the display thread never reads a half-written request.
    modeRequestPending.store(true, std::memory_order_release);
}

[[nodiscard]] auto cVaapiDisplay::GetActiveHdrKind() const noexcept -> StreamHdrKind {
    const cMutexLock lock(&hdrStateMutex);
    return appliedHdrState.kind;
}

[[nodiscard]] auto cVaapiDisplay::GetActiveOsdFbId() const noexcept -> uint32_t {
    // Return what KMS is actually scanning out, not the staged fbId -- GrabImage uses this to
    // composite only the OSD currently on screen.
    const cMutexLock lock(&osdMutex);
    return lastCommittedOsdFbId;
}

[[nodiscard]] auto cVaapiDisplay::GrabDisplayedFrame() -> std::unique_ptr<AVFrame, FreeAVFrame> {
    if (!ready.load(std::memory_order_acquire)) [[unlikely]] {
        return nullptr;
    }

    // Clone under bufferMutex so the displayed AVFrame's VA-surface ref stays alive past unlock;
    // hold the mutex only across the cheap ref bump, never across the GPU download below.
    // The token rides along: the download syncs the surface through the producing VPP context, so
    // a rebuild retiring that graph mid-grab must not free it (see FilterGraphToken). Declared
    // before `source` so it is released after it.
    FilterGraphToken sourceGraphToken;
    std::unique_ptr<AVFrame, FreeAVFrame> source;
    {
        const cMutexLock lock(&bufferMutex);
        if (!displayedBuffer.frame) [[unlikely]] {
            return nullptr;
        }
        source.reset(av_frame_clone(displayedBuffer.frame));
        sourceGraphToken = displayedBuffer.graphToken;
    }
    if (!source) [[unlikely]] {
        return nullptr;
    }

    std::unique_ptr<AVFrame, FreeAVFrame> dest{av_frame_alloc()};
    if (!dest) [[unlikely]] {
        return nullptr;
    }

    // Serialize against VPP: a download racing filter execution on one VADisplay hangs the driver.
    const cMutexLock vaLock(&vaDriverMutex);
    if (const int ret = av_hwframe_transfer_data(dest.get(), source.get(), 0); ret < 0) [[unlikely]] {
        esyslog("vaapivideo/display: grab transfer failed: %s", AvErr(ret).data());
        return nullptr;
    }
    return dest;
}

auto cVaapiDisplay::ClearOsdIfActive(uint32_t fbId) -> void {
    if (fbId == 0) {
        return;
    }

    const cMutexLock lock(&osdMutex);
    if (currentOsd.fbId == fbId) {
        dsyslog("vaapivideo/display: OSD hide (conditional) -- fbId=%u", fbId);
        currentOsd = {};
        osdDirty = true;
        ++osdGeneration;
    }
}

auto cVaapiDisplay::SetOsd(const OsdOverlay &osd) -> void {
    const cMutexLock lock(&osdMutex);

    const bool wasHidden = (currentOsd.fbId == 0);
    const bool nowHidden = (osd.fbId == 0);
    const bool showing = wasHidden && !nowHidden;
    const bool hiding = !wasHidden && nowHidden;

    if (showing) {
        dsyslog("vaapivideo/display: OSD show -- fbId=%u pos=(%d,%d) size=%ux%u", osd.fbId, osd.x, osd.y, osd.width,
                osd.height);
    } else if (hiding) {
        dsyslog("vaapivideo/display: OSD hide -- fbId=%u", currentOsd.fbId);
    }

    // Always mark dirty, even for an unchanged (fbId, geometry) pair. VDR may repaint
    // into the same dumb buffer in-place, and display-compression caches (FBC/PSR) are only
    // invalidated when the plane is touched by an atomic commit. Without this, stale compressed
    // pixels remain on screen.
    currentOsd = osd;
    osdDirty = true;
    ++osdGeneration;
}

auto cVaapiDisplay::Shutdown() -> void {
    const bool wasInitialized = ready.exchange(false, std::memory_order_acq_rel);
    if (!wasInitialized) {
        return;
    }
    dsyslog("vaapivideo/display: shutting down");

    // isClearing before stopping: if reversed, the consumer could observe (stopping=false,
    // isClearing=false) and start one more import between the two stores.
    isClearing.store(true, std::memory_order_release);
    stopping.store(true, std::memory_order_release);

    // Force-clear isFlipPending: if the CRTC was disabled externally (display disconnect,
    // TTY switch), the page-flip event never arrives and the consumer would spin forever.
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
        esyslog("vaapivideo/display: thread did not exit -- may cause resource leak");
    }

    // Drain residual page-flip events: without this the kernel keeps them pending on the fd
    // and the next process to open the DRM device inherits stale events. Only once the
    // consumer has exited -- a wedged consumer may still be inside drmHandleEvent, and a
    // second concurrent reader on the DRM fd is forbidden (see the file-header DRM fd rule).
    if (hasExited.load(std::memory_order_acquire)) {
        for (int i = 0; i < DISPLAY_MAX_DRAIN_ITERATIONS && DrainDrmEvents(0); ++i) {
        }
    } else {
        esyslog("vaapivideo/display: skipping residual DRM event drain -- display thread still running");
    }

    // Blank planes, reset HDR state, and deactivate the CRTC so fbcon or the next DRM client
    // can take over cleanly. All in one atomic to avoid a half-disabled CRTC state.
    if (drmFd >= 0) {
        AtomicRequest req;
        // Include HDR reset in the same ALLOW_MODESET commit that disables the CRTC;
        // a separate commit would leave BT.2020/10bpc active for the next DRM client.
        if (hdrProps.hdrOutputMetadata != 0) {
            req.AddProperty(connectorId, hdrProps.hdrOutputMetadata, 0);
        }
        if (hdrProps.colorspaceValid) {
            req.AddProperty(connectorId, hdrProps.colorspace, hdrProps.colorspaceDefault);
        }
        if (hdrProps.maxBpc != 0) {
            const uint64_t sdrBpc = std::clamp<uint64_t>(8U, hdrProps.maxBpcMin, hdrProps.maxBpcMax);
            req.AddProperty(connectorId, hdrProps.maxBpc, sdrBpc);
        }
        if (videoPlaneId != 0) {
            req.AddProperty(videoPlaneId, videoProps.fbId, 0);
            req.AddProperty(videoPlaneId, videoProps.crtcId, 0);
        }
        if (osdPlaneId != 0) {
            req.AddProperty(osdPlaneId, osdProps.fbId, 0);
            req.AddProperty(osdPlaneId, osdProps.crtcId, 0);
        }
        if (modesetProps.isValid) {
            req.AddProperty(crtcId, modesetProps.crtcActive, 0);
            req.AddProperty(crtcId, modesetProps.crtcModeId, 0);
            req.AddProperty(connectorId, modesetProps.connectorCrtcId, 0);
        }
        (void)AtomicCommit(req, DRM_MODE_ATOMIC_ALLOW_MODESET);
    }

    {
        const cMutexLock lock(&bufferMutex);
        pendingFrames.clear();
        pendingDepth.store(0, std::memory_order_release);
        displayedBuffer = DrmFramebuffer{};
        pendingBuffer = DrmFramebuffer{};
    }

    // The mode blob must be destroyed AFTER the CRTC is disabled: the kernel holds an
    // internal reference to it while ACTIVE=1.
    if (hwDeviceRef) {
        av_buffer_unref(&hwDeviceRef);
    }
    if (modeBlobId != 0 && drmFd >= 0) {
        drmModeDestroyPropertyBlob(drmFd, modeBlobId);
        modeBlobId = 0;
    }
    // HDR blobs: same constraint -- CRTC must already be disabled before freeing.
    if (appliedHdrBlobId != 0 && drmFd >= 0) {
        if (drmModeDestroyPropertyBlob(drmFd, appliedHdrBlobId) != 0) [[unlikely]] {
            esyslog("vaapivideo/display: failed to destroy applied HDR blob: %s", std::strerror(errno));
        }
        appliedHdrBlobId = 0;
    }
    if (pendingDestroyHdrBlobId != 0 && drmFd >= 0) {
        if (drmModeDestroyPropertyBlob(drmFd, pendingDestroyHdrBlobId) != 0) [[unlikely]] {
            esyslog("vaapivideo/display: failed to destroy pending HDR blob: %s", std::strerror(errno));
        }
        pendingDestroyHdrBlobId = 0;
    }
    // Reset tracked HDR state so a re-Initialize() starts from a known SDR baseline.
    // Held under hdrStateMutex so a concurrent GetActiveHdrKind() / SetHdrOutputState()
    // observes consistent state -- SVDRP can in theory still reach this device while
    // the destructor walks Shutdown().
    {
        const cMutexLock lock(&hdrStateMutex);
        appliedHdrState = {};
        stagedHdrState = {};
    }
}

[[nodiscard]] auto cVaapiDisplay::SubmitFrame(std::unique_ptr<VaapiFrame> frame, int timeoutMs) -> bool {
    // Relaxed pre-checks (cheap); bufferMutex below provides the actual memory ordering.
    // Two independent gates: isClearing (owned by BeginStreamSwitch) and modesetActive (owned by
    // the display thread's ChangeDisplayMode). They can overlap, so each owner clears only its own.
    if (!frame || !ready.load(std::memory_order_relaxed) || isClearing.load(std::memory_order_relaxed) ||
        modesetActive.load(std::memory_order_relaxed)) [[unlikely]] {
        return false;
    }

    const cMutexLock lock(&bufferMutex);

    // VSync-paced backpressure: block when the prerender queue is full.
    if (pendingFrames.size() >= DISPLAY_PRERENDER_SLOTS) {
        if (timeoutMs == 0) {
            return false;
        }

        // timeoutMs < 0 blocks until a slot opens; per-slice isClearing/ready checks keep an
        // "infinite" wait from outliving a stream switch or shutdown.
        const cTimeMs deadline(timeoutMs > 0 ? timeoutMs : 0);
        while (pendingFrames.size() >= DISPLAY_PRERENDER_SLOTS && ready.load(std::memory_order_relaxed)) {
            if (isClearing.load(std::memory_order_relaxed) || modesetActive.load(std::memory_order_relaxed)) {
                return false;
            }
            if (timeoutMs > 0 && deadline.TimedOut()) {
                break;
            }
            frameSlotCond.TimedWait(bufferMutex, 10);
        }

        if (pendingFrames.size() >= DISPLAY_PRERENDER_SLOTS) [[unlikely]] {
            return false;
        }
    }

    // Re-check under bufferMutex: every teardown path sets its flag before taking bufferMutex
    // to drop the queue, so either the flag is visible here or this push is swept by the
    // clear -- a stale frame can never survive into the new stream (or the new mode).
    if (!ready.load(std::memory_order_relaxed) || isClearing.load(std::memory_order_relaxed) ||
        modesetActive.load(std::memory_order_relaxed)) [[unlikely]] {
        return false;
    }

    pendingFrames.push_back(std::move(frame));
    pendingDepth.store(pendingFrames.size(), std::memory_order_release);
    return true;
}

// ============================================================================
// === THREAD ===
// ============================================================================

auto cVaapiDisplay::Action() -> void {
    // Queue-underrun tracker: wall-clock duration of consecutive empty VSyncs during active
    // playback; warmup grace suppresses spurious counts after Clear/cold-start while the
    // filter graph + audio anchor. Durations are wall-clock deltas, never vsyncCount * nominal
    // vsyncMs -- counts lie when the loop is preempted or flip events arrive late.
    // Re-derived every iteration rather than hoisted: a runtime mode change moves the refresh
    // rate under this thread, and a stale threshold would mis-scale every underrun verdict.
    uint64_t gapStartMs = 0;      ///< Wall-clock baseline for the current gap; 0 = "anchor on next re-present".
                                  ///< Reset on commit / isClearing / inTrick / inSyncSleep / inPause so deliberate
                                  ///< holds do not surface their duration as a fake underrun.
    uint64_t peakGapMs = 0;       ///< Wall-clock peak duration of the current gap; reset on commit.
    unsigned emptyVSyncTotal = 0; ///< Cumulative empty-VSync count since the display thread started.
    // cTimeMs(0) starts timed-out: the first underrun log fires undelayed, and no warmup
    // grace exists until the first fresh commit arms it (pendingBuffer.IsValid() gates the
    // counter before that, so "no grace at construction" is safe).
    cTimeMs underrunLogCooldown(0);
    cTimeMs warmupGraceUntil(0);

    while (!stopping.load(std::memory_order_relaxed) && ready.load(std::memory_order_relaxed)) {
        const auto refreshHz = GetOutputRefreshRate();
        const auto vsyncMs = refreshHz > 0 ? 1000U / refreshHz : 20U;
        const uint64_t thresholdMs = DISPLAY_UNDERRUN_THRESHOLD_VSYNCS * vsyncMs;

        // Non-blocking drain so isFlipPending is up-to-date before the gate check below.
        while (DrainDrmEvents(0)) {
        }

        // VSync gate: don't queue a new flip until the previous one's event has arrived.
        // 5 ms poll avoids busy-spinning; page_flip_handler latency is the real bottleneck.
        // Recovery: on first plane attach over HDMI, the kernel occasionally swallows the page
        // flip event (link renegotiation, sink probe failure, etc.). After DISPLAY_PAGE_FLIP_STUCK_MS the
        // event isn't coming -- force-clear so the next commit can proceed.
        if (isFlipPending.load(std::memory_order_relaxed)) {
            (void)DrainDrmEvents(5);
            const uint64_t since = flipPendingSinceMs.load(std::memory_order_acquire);
            if (since != 0 && cTimeMs::Now() - since > DISPLAY_PAGE_FLIP_STUCK_MS &&
                isFlipPending.load(std::memory_order_relaxed)) {
                esyslog("vaapivideo/display: page-flip event missing after %llums, forcing recovery",
                        static_cast<unsigned long long>(cTimeMs::Now() - since));
                flipPendingSinceMs.store(0, std::memory_order_release);
                isFlipPending.store(false, std::memory_order_release);
            }
            continue;
        }

        // Runtime mode change, serviced here and nowhere else: this thread owns every DRM commit
        // and is the sole drmHandleEvent dispatcher, so doing the ALLOW_MODESET inline needs no
        // extra lock and cannot stall the VDR main loop. Placed after the flip gate above so the
        // previous flip has always completed before the CRTC is reprogrammed.
        if (modeRequestPending.load(std::memory_order_acquire)) [[unlikely]] {
            ChangeDisplayMode();
            gapStartMs = 0;
            peakGapMs = 0;
            warmupGraceUntil.Set(DISPLAY_WARMUP_GRACE_MS);
            continue;
        }

        // Yield importMutex to BeginStreamSwitch() and reset the underrun counters so
        // post-switch pre-roll re-presents don't trigger a spurious log. gapStartMs reset to 0
        // so the next non-suppressed iteration re-anchors at nowMs rather than measuring back
        // through the isClearing window.
        if (isClearing.load(std::memory_order_relaxed)) {
            gapStartMs = 0;
            peakGapMs = 0;
            cCondWait::SleepMs(5);
            continue;
        }

        // Trick hold, sync-correction sleep, and pause all re-present deliberately -- not
        // underruns. Reset both counters so the window's duration doesn't surface as a fake
        // underrun the moment it ends (gapStartMs=0 re-anchors on the next re-present).
        const bool inTrick = trickActive.load(std::memory_order_relaxed);
        const bool inSyncSleep = syncSleeping.load(std::memory_order_relaxed);
        const bool inPause = devicePaused.load(std::memory_order_relaxed);
        if (inTrick || inSyncSleep || inPause) {
            gapStartMs = 0;
            peakGapMs = 0;
        }

        // importMutex spans both map AND commit: releasing between them lets BeginStreamSwitch()
        // free the codec while a pointer to its surface is still held.
        bool frameCommitted = false;
        {
            const cMutexLock importLock(&importMutex);

            // Re-check under the lock: isClearing may have been set between the unlocked
            // check above and acquiring importMutex.
            if (!isClearing.load(std::memory_order_acquire) && !stopping.load(std::memory_order_acquire) &&
                ready.load(std::memory_order_acquire)) {
                std::unique_ptr<VaapiFrame> frameToShow;
                {
                    const cMutexLock lock(&bufferMutex);
                    if (!pendingFrames.empty()) {
                        frameToShow = std::move(pendingFrames.front());
                        pendingFrames.pop_front();
                        pendingDepth.store(pendingFrames.size(), std::memory_order_release);
                        frameSlotCond.Broadcast();
                    }
                }

                if (frameToShow && !isClearing.load(std::memory_order_acquire)) {
                    DrmFramebuffer newFb = MapVaapiFrame(std::move(frameToShow));

                    // MapVaapiFrame is the slow path (PRIME export + GEM import + AddFB2);
                    // re-check isClearing one more time before committing the result.
                    if (newFb.IsValid() && !isClearing.load(std::memory_order_acquire)) {
                        const cMutexLock lock(&bufferMutex);
                        if (PresentBuffer(newFb)) {
                            // Buffer chain advances only on a successful commit. On failure,
                            // displayedBuffer must NOT be released while the CRTC is still
                            // scanning it (kernel use-after-free on next scanout).
                            displayedBuffer = std::move(pendingBuffer);
                            pendingBuffer = std::move(newFb);
                            // Arm warmup grace whenever decoder resumes from idle: post-Clear
                            // (lastCommit==0) OR after a long gap (commit > DISPLAY_WARMUP_ACTIVE_WINDOW_MS old).
                            // The latter covers transitions where BeginStreamSwitch wasn't invoked
                            // (track switch, post-trick re-anchor). Done under importMutex so a
                            // BeginStreamSwitch racing this commit cannot republish a stale value.
                            const uint64_t prevCommitMs = lastFrameCommitMs.load(std::memory_order_relaxed);
                            const uint64_t nowMs = cTimeMs::Now();
                            if (prevCommitMs == 0 || nowMs - prevCommitMs > DISPLAY_WARMUP_ACTIVE_WINDOW_MS) {
                                warmupGraceUntil.Set(DISPLAY_WARMUP_GRACE_MS);
                            }
                            lastFrameCommitMs.store(nowMs, std::memory_order_release);
                            // The new stream's first picture is on its way to the panel (flip lands next VSync).
                            if (const int64_t traceMs = startTrace.Fire(TRACE_FIRST_COMMIT); traceMs >= 0)
                                [[unlikely]] {
                                tsyslog("vaapivideo/display: trace +%lldms first frame committed to CRTC (%ux%u)",
                                        static_cast<long long>(traceMs), pendingBuffer.width, pendingBuffer.height);
                            }
                            // Recovery log: onset fires at THRESHOLD regardless of how long the
                            // gap actually lasts; peak captures the real wall-clock length.
                            if (peakGapMs >= thresholdMs) {
                                dsyslog("vaapivideo/display: queue refilled after %llums; total=%u",
                                        static_cast<unsigned long long>(peakGapMs), emptyVSyncTotal);
                            }
                            gapStartMs = 0;
                            peakGapMs = 0;
                            frameCommitted = true;
                        }
                    }
                }
            }
        }

        // No new frame: re-present the previous buffer to keep flip cadence + OSD updates alive.
        if (!frameCommitted && !isClearing.load(std::memory_order_relaxed)) {
            bool didPresent = false;
            bool haveVideoBuffer = false;
            {
                const cMutexLock lock(&bufferMutex);
                haveVideoBuffer = pendingBuffer.IsValid();
                if (haveVideoBuffer) {
                    // Failure falls through to the SleepMs(5) below -- an immediate retry
                    // would busy-spin while the kernel keeps rejecting the plane state.
                    didPresent = PresentBuffer(pendingBuffer);
                }
            }
            // No framebuffer at all: the OSD has nothing to ride on. That is the state right
            // after a mode change (ChangeDisplayMode drops both buffers) and for any audio-only
            // stream, and without this the staged overlay -- typically the VDR menu the user is
            // looking at -- would never reach the screen.
            if (!haveVideoBuffer) {
                (void)CommitOsdOnly();
            }
            // bufferMutex is released BEFORE the tracking and SleepMs below: pthread mutexes
            // are not FIFO, and holding it across the pre-first-frame sleep starved
            // SubmitFrame() badly enough to push first-picture latency from ms to seconds.
            if (didPresent) {
                // Count only when the decoder was active AND past warmup grace. Gaps anchor on
                // gapStartMs (first re-present), NOT lastFrameCommitMs, so a deliberate sleep /
                // trick hold / stream switch doesn't surface its own duration as a fake
                // underrun. Past DISPLAY_UNDERRUN_IDLE_MAX_MS stop accumulating (paused stream).
                const uint64_t nowMs = cTimeMs::Now();
                const uint64_t lastCommitMs = lastFrameCommitMs.load(std::memory_order_acquire);
                if (!inTrick && !inSyncSleep && !inPause && lastCommitMs != 0 && warmupGraceUntil.TimedOut()) {
                    if (gapStartMs == 0) {
                        gapStartMs = nowMs; // anchor: first re-present of the current gap
                    }
                    const uint64_t currentGapMs = nowMs - gapStartMs;
                    if (currentGapMs < DISPLAY_UNDERRUN_IDLE_MAX_MS) {
                        ++emptyVSyncTotal;
                        peakGapMs = std::max(peakGapMs, currentGapMs);
                        if (currentGapMs >= thresholdMs && underrunLogCooldown.TimedOut()) {
                            dsyslog("vaapivideo/display: queue empty %llums; total=%u",
                                    static_cast<unsigned long long>(currentGapMs), emptyVSyncTotal);
                            underrunLogCooldown.Set(DISPLAY_UNDERRUN_LOG_INTERVAL_MS);
                        }
                    } else {
                        // Paused/stopped, not an underrun: clear peak so the recovery log stays
                        // silent on resume; gapStartMs stays put to avoid re-anchoring a fresh
                        // accounting window every iteration.
                        peakGapMs = 0;
                    }
                } else {
                    gapStartMs = 0;
                    peakGapMs = 0;
                }
            } else {
                cCondWait::SleepMs(5);
            }
        }
    }

    hasExited.store(true, std::memory_order_release);
}

// ============================================================================
// === INTERNAL METHODS ===
// ============================================================================

auto cVaapiDisplay::AppendOsdPlane(AtomicRequest &req, const OsdOverlay &osd) const -> bool {
    if (osd.fbId == 0 || osdPlaneId == 0) {
        return false;
    }

    // Skins may place the OSD partly off-screen on the left/top: crop the source by the
    // negative offset and place the visible part at zero. Right/bottom overflow is clipped
    // by capping clippedW/clippedH below. int64_t avoids the -INT32_MIN UB corner case.
    const auto sourceOffsetX = static_cast<uint32_t>(std::clamp<int64_t>(-static_cast<int64_t>(osd.x), 0, osd.width));
    const auto sourceOffsetY = static_cast<uint32_t>(std::clamp<int64_t>(-static_cast<int64_t>(osd.y), 0, osd.height));
    const auto destX = static_cast<uint32_t>(std::max(0, osd.x));
    const auto destY = static_cast<uint32_t>(std::max(0, osd.y));

    // KMS rejects CRTC_X+CRTC_W > CRTC width (similarly for Y); clip here so the entire atomic
    // commit doesn't fail over a slightly oversized OSD. If nothing remains visible after clipping,
    // emit a hide commit -- a silent no-op would leave the previously-shown OSD scanned out
    // forever (PresentBuffer clears osdDirty on success).
    // Snapshot the geometry once: a runtime mode change can move it between reads, and after one
    // the live OSD is still sized for the previous mode -- this clip is what keeps it committable.
    const auto screenWidth = GetOutputWidth();
    const auto screenHeight = GetOutputHeight();
    const bool offScreen =
        sourceOffsetX >= osd.width || sourceOffsetY >= osd.height || destX >= screenWidth || destY >= screenHeight;
    const auto clippedW = offScreen ? 0U : std::min(osd.width - sourceOffsetX, screenWidth - destX);
    const auto clippedH = offScreen ? 0U : std::min(osd.height - sourceOffsetY, screenHeight - destY);
    if (clippedW == 0 || clippedH == 0) {
        req.AddProperty(osdPlaneId, osdProps.fbId, 0);
        req.AddProperty(osdPlaneId, osdProps.crtcId, 0);
        return false;
    }

    req.AddProperty(osdPlaneId, osdProps.crtcId, crtcId);
    req.AddProperty(osdPlaneId, osdProps.fbId, osd.fbId);
    req.AddProperty(osdPlaneId, osdProps.srcX, static_cast<uint64_t>(sourceOffsetX) << 16);
    req.AddProperty(osdPlaneId, osdProps.srcY, static_cast<uint64_t>(sourceOffsetY) << 16);
    // SRC_* properties use 16.16 fixed-point (value = pixels << 16).
    req.AddProperty(osdPlaneId, osdProps.srcW, static_cast<uint64_t>(clippedW) << 16);
    req.AddProperty(osdPlaneId, osdProps.srcH, static_cast<uint64_t>(clippedH) << 16);
    req.AddProperty(osdPlaneId, osdProps.crtcX, destX);
    req.AddProperty(osdPlaneId, osdProps.crtcY, destY);
    req.AddProperty(osdPlaneId, osdProps.crtcW, clippedW);
    req.AddProperty(osdPlaneId, osdProps.crtcH, clippedH);

    // VDR stores straight (non-premultiplied) ARGB via tColor. "Coverage" blending (enum 1)
    // applies alpha correctly; "Pre-multiplied" (enum 0) would double-apply it and produce
    // darker edges with ringing on translucent backgrounds. Write only on change -- the value
    // is sticky in the kernel; rewriting every animator frame is wasted validation.
    if (osdProps.pixelBlendMode != 0 && lastOsdPixelBlendMode != 1) {
        req.AddProperty(osdPlaneId, osdProps.pixelBlendMode, 1);
    }
    // zpos is NOT written: on tested i915 it causes the commit to fail (the property is
    // immutable on overlay planes). On amdgpu the driver's default z-order already places
    // the OSD above the video plane. Empirical -- re-verify on new hardware before adding.
    return true;
}

[[nodiscard]] auto cVaapiDisplay::AtomicCommit(AtomicRequest &req, uint32_t flags, bool osdHdrCommit) -> bool {
    if (!req.Handle() || req.Failed()) [[unlikely]] {
        // Incomplete request (alloc failure): reporting success would let PresentBuffer RmFB
        // the framebuffer KMS still scans out. No errno -- the failure was at build time.
        if (atomicFailureLogCooldown.TimedOut()) {
            esyslog("vaapivideo/display: failed to build atomic request (out of memory?)");
            atomicFailureLogCooldown.Set(DISPLAY_ATOMIC_FAILURE_LOG_INTERVAL_MS);
        }
        return false;
    }
    if (req.Count() == 0) {
        return true; // empty commit -- nothing to do, treat as success
    }

    // flags==0: async page-flip (PAGE_FLIP_EVENT | NONBLOCK), event clears isFlipPending. Covers
    //   steady-state video frames, plane-position updates (ScaleVideo), color encoding/range
    //   updates, and OSD show/hide -- fastset-eligible because scale_vaapi emits the final
    //   framebuffer size, so SRC == CRTC and KMS does not need to drive a plane scaler.
    // flags==DRM_MODE_ATOMIC_ALLOW_MODESET (sync, no event): used for ApplyDisplayMode, CRTC
    //   disable on shutdown, HDR connector-state changes that may link-retrain, and (via the
    //   osdHdrCommit fallback below) OSD-over-HDR frames on GPUs that need a CDCLK bump for them.
    //   Display thread blocks until applied.
    // ATOMIC_ASYNC is unused -- it requires linear buffers, our VAAPI surfaces are tiled.
    const uint32_t commitFlags = (flags == 0) ? PAGE_FLIP_COMMIT_FLAGS : flags;
    if (drmModeAtomicCommit(drmFd, req.Handle(), commitFlags, this) == 0) {
        if ((commitFlags & DRM_MODE_PAGE_FLIP_EVENT) != 0) {
            flipPendingSinceMs.store(cTimeMs::Now(), std::memory_order_release);
            isFlipPending.store(true, std::memory_order_release);
        }
        return true;
    }
    const int origErrno = errno;
    // EBUSY: previous flip event not yet consumed; Action() retries after DrainDrmEvents().
    if (origErrno == EBUSY) {
        return false;
    }

    // OSD-over-HDR EINVAL recovery on bandwidth-limited GPUs: the OSD plane beside the 4K 10-bpc
    // video plane forces a pixel-clock bump that only a modeset can apply, so the NONBLOCK flip is
    // rejected every frame. Retry the same req (drmModeAtomicCommit leaves it intact on failure)
    // under ALLOW_MODESET -- accepted even though parts without cdclk-squash may briefly retrain the
    // link at OSD show/hide, since otherwise the menu is unusable over HDR. Latch so later frames
    // skip the doomed NONBLOCK attempt.
    if (origErrno == EINVAL && osdHdrCommit) [[unlikely]] {
        if ((commitFlags & DRM_MODE_ATOMIC_ALLOW_MODESET) == 0) {
            if (!osdHdrNeedsModeset) {
                isyslog("vaapivideo/display: OSD over HDR needs a modeset on this GPU -- using "
                        "synchronous commits while the OSD is shown (brief A/V interruption possible)");
                osdHdrNeedsModeset = true;
            }
            if (drmModeAtomicCommit(drmFd, req.Handle(), DRM_MODE_ATOMIC_ALLOW_MODESET, this) == 0) {
                return true; // sync path carries no PAGE_FLIP_EVENT -> isFlipPending stays clear
            }
        }
        // A modeset was rejected too: the config exceeds the hardware bandwidth ceiling, which no
        // flag can fix. Drop the OSD plane (PresentBuffer honors the latch) so the thread stops
        // spinning on a doomed sync commit that would also stall video -- menu hidden, video plays.
        if (!osdHdrSuppressed) {
            esyslog("vaapivideo/display: OSD over HDR exceeds the display bandwidth on this GPU -- "
                    "hiding the OSD while HDR is active so video keeps playing");
            osdHdrSuppressed = true;
        }
        return false;
    }

    // Pure-video / SDR / HDR-transition flips are fastset-eligible (scale_vaapi emits the final fb
    // size), so an EINVAL here is a genuinely invalid plane state -- surface it loud, no retry.
    // Rate-limited: the 5 ms re-present retry path would otherwise emit this at up to 200 Hz.
    if (atomicFailureLogCooldown.TimedOut()) {
        if (origErrno == EACCES) {
            // EACCES on a DRM_MASTER ioctl means one thing: this fd is no longer the current master, i.e.
            // another client took the display. No mode is at fault, whatever the caller reports.
            esyslog("vaapivideo/display: atomic commit refused -- not DRM master, another client holds the display "
                    "(flags=0x%x)",
                    commitFlags);
        } else {
            esyslog("vaapivideo/display: atomic commit failed -- %s (flags=0x%x)", std::strerror(origErrno),
                    commitFlags);
        }
        atomicFailureLogCooldown.Set(DISPLAY_ATOMIC_FAILURE_LOG_INTERVAL_MS);
    }
    return false;
}

[[nodiscard]] auto cVaapiDisplay::ApplyDisplayMode(const drmModeModeInfo &mode, bool blankPlanes) -> bool {
    // Deliberately silent on success: both callers log the mode they ended up with ("initialized",
    // "mode changed to"), so announcing the attempt as well printed every switch twice. The failure
    // paths below carry the geometry instead, which is where it is actually needed.
    const double rateHz = static_cast<double>(ModeRefreshMilliHz(mode)) / 1000.0;

    // KMS stores the mode as a property blob; libdrm has no GC, so the old one is freed by hand.
    // Keep it alive across the commit: the kernel holds an internal reference while ACTIVE=1, and
    // it is also the state to fall back on if the new mode is rejected. Freed only once the new
    // blob is live (or immediately, if the new blob was rejected).
    const uint32_t previousBlobId = modeBlobId;
    uint32_t newBlobId = 0;
    if (drmModeCreatePropertyBlob(drmFd, &mode, sizeof(mode), &newBlobId) < 0) {
        esyslog("vaapivideo/display: failed to create mode blob for %ux%u@%.3fHz: %s", mode.hdisplay, mode.vdisplay,
                rateHz, std::strerror(errno));
        return false;
    }

    AtomicRequest req;
    // Detach both planes in the SAME commit as the mode change. On a runtime switch the attached
    // framebuffers are still sized for the outgoing mode, and a plane larger than the incoming
    // CRTC makes the kernel reject the whole atomic request with EINVAL.
    if (blankPlanes) {
        if (videoPlaneId != 0) {
            req.AddProperty(videoPlaneId, videoProps.fbId, 0);
            req.AddProperty(videoPlaneId, videoProps.crtcId, 0);
        }
        if (osdPlaneId != 0) {
            req.AddProperty(osdPlaneId, osdProps.fbId, 0);
            req.AddProperty(osdPlaneId, osdProps.crtcId, 0);
        }
    }
    req.AddProperty(crtcId, modesetProps.crtcActive, 1);
    req.AddProperty(crtcId, modesetProps.crtcModeId, newBlobId);
    req.AddProperty(connectorId, modesetProps.connectorCrtcId, crtcId);
    // Include the SDR baseline for HDR connector properties in this same ALLOW_MODESET commit
    // to clear any state left by a previous DRM client (e.g. BT.2020 / 10 bpc from HDR
    // playback). A second ALLOW_MODESET later would cause a second HDMI link retrain;
    // if the AVR is locked onto an IEC61937 bitstream at that point it drops out of passthrough
    // and treats the subsequent payload as raw PCM noise.
    if (hdrProps.hdrOutputMetadata != 0) {
        req.AddProperty(connectorId, hdrProps.hdrOutputMetadata, 0);
    }
    if (hdrProps.colorspaceValid) {
        req.AddProperty(connectorId, hdrProps.colorspace, hdrProps.colorspaceDefault);
    }
    if (hdrProps.maxBpc != 0) {
        const uint64_t sdrBpc = std::clamp<uint64_t>(8U, hdrProps.maxBpcMin, hdrProps.maxBpcMax);
        req.AddProperty(connectorId, hdrProps.maxBpc, sdrBpc);
    }

    if (!AtomicCommit(req, DRM_MODE_ATOMIC_ALLOW_MODESET)) {
        esyslog("vaapivideo/display: driver rejected mode %ux%u@%.3fHz", mode.hdisplay, mode.vdisplay, rateHz);
        // Drop only the rejected blob; the previous one is still referenced by the live CRTC.
        if (drmModeDestroyPropertyBlob(drmFd, newBlobId) != 0) [[unlikely]] {
            esyslog("vaapivideo/display: failed to destroy rejected mode blob: %s", std::strerror(errno));
        }
        return false;
    }

    modeBlobId = newBlobId;
    if (previousBlobId != 0 && drmModeDestroyPropertyBlob(drmFd, previousBlobId) != 0) [[unlikely]] {
        esyslog("vaapivideo/display: failed to destroy previous mode blob: %s", std::strerror(errno));
    }

    {
        const cMutexLock lock(&modeRequestMutex);
        activeMode = mode;
    }
    // Mark applied state as Sdr so MaybeAppendHdrOutputState() skips the first frame's
    // HDR write (staged==applied), keeping subsequent page flips in the non-ALLOW_MODESET
    // fast path and preventing spurious AVR retrains during IEC61937 lock-in.
    {
        const cMutexLock lock(&hdrStateMutex);
        appliedHdrState = HdrStreamInfo{};
    }
    // The commit above wrote HDR_OUTPUT_METADATA=0, so the kernel has dropped its reference to
    // whatever infoframe blob was live -- destroy it rather than just forgetting the id. At
    // Initialize() these are always 0; on the runtime path they are not, and simply zeroing them
    // leaked one kernel blob per mode change made during HDR playback.
    for (uint32_t *blobId : {&appliedHdrBlobId, &pendingDestroyHdrBlobId}) {
        if (*blobId != 0) {
            if (drmModeDestroyPropertyBlob(drmFd, *blobId) != 0) [[unlikely]] {
                esyslog("vaapivideo/display: failed to destroy HDR blob on mode change: %s", std::strerror(errno));
            }
            *blobId = 0;
        }
    }
    // Fresh CDCLK headroom: re-probe so a smaller HDR mode isn't needlessly forced onto sync commits.
    osdHdrNeedsModeset = false;
    osdHdrSuppressed = false;
    return true;
}

auto cVaapiDisplay::ResetPlaneStateCaches() -> void {
    // Video-plane caches only. lastCommittedOsdFbId describes what the kernel is really scanning
    // out (AwaitOsdHidden blocks on it before a dumb buffer is freed) and is guarded by osdMutex,
    // so it is cleared by the caller -- under that lock, and only once a commit that actually
    // detached the OSD plane has landed.
    constexpr uint64_t kCacheSentinel = ~uint64_t{0};
    lastOsdPixelBlendMode = kCacheSentinel;
    lastVideoColorEncoding = kCacheSentinel;
    lastVideoColorRange = kCacheSentinel;
    lastVideoSrcW = lastVideoSrcH = kCacheSentinel;
    lastVideoCrtcX = lastVideoCrtcY = lastVideoCrtcW = lastVideoCrtcH = kCacheSentinel;
}

auto cVaapiDisplay::PublishOutputGeometry(const drmModeModeInfo &mode) -> void {
    const uint64_t previousGeometry = outputGeometry.load(std::memory_order_acquire);
    const auto previousWidth = static_cast<uint32_t>(previousGeometry >> 32);
    const auto previousHeight = static_cast<uint32_t>(previousGeometry & 0xFFFFFFFFULL);

    // Pixel aspect before the geometry, and modeGeneration (bumped by the caller) after both. Store
    // order alone cannot make the pair coherent for a reader -- it may load one atomic before this
    // runs and the other after -- so GetOsdSize() brackets its reads with the generation and
    // retries; publishing in this order is what makes that bracket sufficient.
    const AspectRatio pixelAspect = ModePixelAspectRatio(mode);
    outputPixelAspect.store((static_cast<uint64_t>(pixelAspect.num) << 32) | static_cast<uint64_t>(pixelAspect.den),
                            std::memory_order_release);
    outputGeometry.store((static_cast<uint64_t>(mode.hdisplay) << 32) | static_cast<uint64_t>(mode.vdisplay),
                         std::memory_order_release);
    // A degenerate mode (no clock / zero totals) yields 0. 50 Hz is the DVB baseline and must
    // match decoder.cpp's framerate fallback in InitFilterGraph() -- the two values are coupled;
    // changing one without the other desyncs the A/V controllers.
    const uint32_t rateMilliHz = ModeRefreshMilliHz(mode);
    outputRefreshMilliHz.store(rateMilliHz > 0 ? rateMilliHz : 50000U, std::memory_order_release);

    const cRect fullRect(0, 0, static_cast<int>(mode.hdisplay), static_cast<int>(mode.vdisplay));
    const cMutexLock lock(&videoRectMutex);
    // A skin (e.g. skindesigner) may be holding a scaled video window via ScaleVideo(); the device
    // keeps no copy of it, so resetting to full-screen here would silently destroy its layout with
    // nothing left to restore it. Map the request proportionally into the new mode instead, and
    // only fall back to full-screen when it genuinely was full-screen (or the old size is unknown).
    const auto remap = [&](const cRect &rect) -> cRect {
        if (previousWidth == 0 || previousHeight == 0) {
            return fullRect;
        }
        if (rect.X() == 0 && rect.Y() == 0 && rect.Width() == static_cast<int>(previousWidth) &&
            rect.Height() == static_cast<int>(previousHeight)) {
            return fullRect;
        }
        const auto scale = [](int value, uint32_t from, uint32_t to) -> int {
            return static_cast<int>((static_cast<int64_t>(value) * to) / from);
        };
        return NormalizeVideoRect(
            {scale(rect.X(), previousWidth, mode.hdisplay), scale(rect.Y(), previousHeight, mode.vdisplay),
             scale(rect.Width(), previousWidth, mode.hdisplay), scale(rect.Height(), previousHeight, mode.vdisplay)});
    };
    videoRect = remap(videoRect);
    targetVideoRect = remap(targetVideoRect);
}

[[nodiscard]] auto cVaapiDisplay::CommitOsdOnly() -> bool {
    // The OSD normally rides the next video commit, but PresentBuffer() needs a framebuffer to
    // carry it. After a mode change there is none (ChangeDisplayMode drops both), and an
    // audio-only stream never produces one either, so without this path a staged OSD -- the VDR
    // menu the user is looking at -- would never reach the screen.
    if (osdPlaneId == 0) {
        return false;
    }

    AtomicRequest req;
    bool osdCommitted = false;
    uint32_t osdFbId = 0;
    uint64_t osdCommitGeneration = 0;
    const bool hdrActive = GetActiveHdrKind() != StreamHdrKind::Sdr;
    {
        const cMutexLock lock(&osdMutex);
        if (!osdDirty) {
            return false;
        }
        // Same suppression rule as PresentBuffer: hides always land, enables are blocked while a
        // bandwidth-limited GPU has latched osdHdrSuppressed under HDR.
        if (currentOsd.fbId != 0 && osdHdrSuppressed && hdrActive) {
            return false;
        }
        osdCommitGeneration = osdGeneration;
        if (currentOsd.fbId != 0) {
            osdFbId = AppendOsdPlane(req, currentOsd) ? currentOsd.fbId : 0;
        } else {
            req.AddProperty(osdPlaneId, osdProps.fbId, 0);
            req.AddProperty(osdPlaneId, osdProps.crtcId, 0);
        }
        osdCommitted = true;
    }
    // Pull the CRTC into the atomic state. The kernel derives the page-flip event we ask for by
    // walking the request's CRTCs, and an already-detached plane -- exactly what ApplyDisplayMode
    // leaves behind -- contributes neither an old nor a new one, so the commit would succeed with
    // no event and latch isFlipPending until the 200 ms watchdog fires. ACTIVE=1 is already its
    // value, so this stays on the fast non-blocking path.
    req.AddProperty(crtcId, modesetProps.crtcActive, 1);
    if (!osdCommitted || req.Count() == 0) {
        return false;
    }

    if (!AtomicCommit(req, 0, /*osdHdrCommit=*/hdrActive)) {
        return false;
    }
    const cMutexLock lock(&osdMutex);
    lastCommittedOsdFbId = osdFbId;
    if (osdGeneration == osdCommitGeneration) {
        osdDirty = false;
    }
    if (osdFbId != 0) {
        lastOsdPixelBlendMode = 1; // AppendOsdPlane wrote it iff it differed.
    }
    return true;
}

auto cVaapiDisplay::ChangeDisplayMode() -> void {
    drmModeModeInfo wanted{};
    bool alreadyActive = false;
    {
        // Take the request and clear the flag under the SAME lock RequestDisplayMode() publishes
        // under: clearing it afterwards would silently swallow a request that landed in between.
        const cMutexLock lock(&modeRequestMutex);
        wanted = modeRequest;
        modeRequestPending.store(false, std::memory_order_release);
        alreadyActive = wanted.hdisplay == activeMode.hdisplay && wanted.vdisplay == activeMode.vdisplay &&
                        ModeRefreshMilliHz(wanted) == ModeRefreshMilliHz(activeMode);
    }
    // Requests are idempotent so callers can fire one unconditionally (ResetDisplayModeToDefault
    // cannot tell whether an earlier change is still in flight). Nothing to do, and skipping the
    // commit spares the sink a pointless link retrain.
    if (alreadyActive || wanted.hdisplay == 0 || wanted.vdisplay == 0) {
        return;
    }

    // Gate the producer: SubmitFrame() rejects while this is set, so nothing sized for the outgoing
    // mode can queue up behind our back. Its own flag, not isClearing -- that one belongs to
    // BeginStreamSwitch(), and a modeset finishing inside a stream-switch window would clear a gate
    // it never armed, letting pre-Clear frames through.
    modesetActive.store(true, std::memory_order_release);
    {
        const cMutexLock lock(&bufferMutex);
        pendingFrames.clear();
        pendingDepth.store(0, std::memory_order_release);
        frameSlotCond.Broadcast();
        // Dropping both framebuffers is what makes the transition safe: Action()'s re-present
        // branch runs outside importMutex, so an invalid pendingBuffer is the one thing that
        // reliably stops it from committing a stale-sized fb mid-modeset.
        displayedBuffer = DrmFramebuffer{};
        pendingBuffer = DrmFramebuffer{};
    }

    const bool applied = ApplyDisplayMode(wanted, /*blankPlanes=*/true);
    // Cached plane properties describe state the blank commit just cleared -- invalidate them
    // even on failure, since a rejected request may still have landed partially.
    ResetPlaneStateCaches();

    if (applied) {
        PublishOutputGeometry(wanted);
        // The commit detached the OSD plane; record that under osdMutex or AwaitOsdHidden() hangs
        // a cVaapiOsd destructor for its full 500 ms timeout on a stale fbId. currentOsd is kept
        // and re-armed in the same critical section, so a skin that never repaints gets its overlay
        // back clipped to the new mode rather than losing it.
        {
            const cMutexLock lock(&osdMutex);
            lastCommittedOsdFbId = 0;
            osdDirty = true;
            ++osdGeneration;
        }
        awaitingResizedFb = true;
        resizeWaitSince = cTimeMs::Now();
        // Publish last: the decoder rebuilds its VPP graph off this edge and must observe the new
        // geometry when it does.
        geometryChanged.store(true, std::memory_order_release);
        modeGeneration.fetch_add(1, std::memory_order_acq_rel);
        isyslog("vaapivideo/display: mode changed to %ux%u@%.3fHz", GetOutputWidth(), GetOutputHeight(),
                static_cast<double>(GetOutputRefreshMilliHz()) / 1000.0);
    } else {
        // An atomic commit is all-or-nothing: the old mode is still programmed AND the old OSD fb
        // still attached, so lastCommittedOsdFbId must keep describing it. The next commit
        // re-attaches the video plane from the sentinel caches. No retry -- a rejected mode will be
        // rejected again. ApplyDisplayMode already named it; report only the consequence.
        esyslog("vaapivideo/display: keeping current mode %ux%u@%.3fHz", GetOutputWidth(), GetOutputHeight(),
                static_cast<double>(GetOutputRefreshMilliHz()) / 1000.0);
    }

    modesetActive.store(false, std::memory_order_release);
}

[[nodiscard]] auto cVaapiDisplay::BindDrmPlane(int planeIndex, uint32_t format) -> bool {
    // Find the planeIndex-th plane supporting @p format on the active CRTC, cache its atomic prop IDs,
    // and assign it as the video or OSD plane (whichever is still unbound). Format support is
    // checked via the IN_FORMATS blob -- the legacy plane->formats array has no modifier
    // information and VAAPI surfaces are always tiled.
    auto planeRes = std::unique_ptr<drmModePlaneRes, decltype(&drmModeFreePlaneResources)>(
        drmModeGetPlaneResources(drmFd), drmModeFreePlaneResources);
    auto res =
        std::unique_ptr<drmModeRes, decltype(&drmModeFreeResources)>(drmModeGetResources(drmFd), drmModeFreeResources);

    if (!planeRes || !res) {
        return false;
    }

    // possible_crtcs is a position bitmask into res->crtcs[], not a CRTC object-ID bitmask.
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

    // DRM fourccs are four printable ASCII bytes, little-endian, so spell the format out: the raw
    // 0x3231564e this used to print is the same value but nobody reads it as "NV12".
    dsyslog("vaapivideo/display: searching for plane index %d (format %c%c%c%c)", planeIndex,
            static_cast<char>(format & 0xFFU), static_cast<char>((format >> 8U) & 0xFFU),
            static_cast<char>((format >> 16U) & 0xFFU), static_cast<char>((format >> 24U) & 0xFFU));

    // For the NV12 video plane, prefer an HDR-capable plane (P010 + both COLOR_ENCODING enums).
    // Some GPUs put P010 on a later plane and list a SDR-only plane first; taking the first
    // match would silently disable HDR. Fall back to the first SDR-only plane if no HDR
    // capable plane exists.
    const bool preferHdrCapable = (videoPlaneId == 0 && planeIndex == 0 && format == DRM_FORMAT_NV12);
    int found = 0;
    uint32_t fallbackPlaneId = 0;
    uint32_t fallbackPlaneType = DRM_PLANE_TYPE_OVERLAY;
    DrmPlaneProps fallbackProps{};
    for (uint32_t i = 0; i < planeRes->count_planes; ++i) {
        auto plane = std::unique_ptr<drmModePlane, decltype(&drmModeFreePlane)>(
            drmModeGetPlane(drmFd, planeRes->planes[i]), drmModeFreePlane);
        if (!plane) {
            continue;
        }

        // On the OSD pass (videoPlaneId already set), skip the already-claimed video plane.
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

        // Single sweep over all plane properties: format support, plane type, and every
        // atomic prop ID needed later. One pass avoids re-querying per-property.
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

            // IN_FORMATS blob lists (format, modifier) pairs the plane accepts. Check the
            // requested format and also record P010 support in one pass -- ProbeHdrCapabilities
            // would otherwise have to re-parse the same blob. The modifier is validated by
            // KMS at commit time; matching the fourcc alone is sufficient here.
            if (!hasFormatSupport && std::strcmp(name, "IN_FORMATS") == 0) {
                const auto blobId = static_cast<uint32_t>(planeProps->prop_values[j]);
                if (blobId != 0) {
                    auto blob = std::unique_ptr<drmModePropertyBlobRes, decltype(&drmModeFreePropertyBlob)>(
                        drmModeGetPropertyBlob(drmFd, blobId), drmModeFreePropertyBlob);
                    if (blob && blob->data && blob->length >= sizeof(drm_format_modifier_blob)) {
                        const auto *modBlob = static_cast<const drm_format_modifier_blob *>(blob->data);
                        const auto *base = static_cast<const uint8_t *>(blob->data);
                        // drm_format_modifier_blob: formats_offset is a byte offset to the
                        // uint32_t format[] array (DRM ABI, not a pointer). Bound it to
                        // [header end, blob length] so a buggy driver blob can't cause an OOB
                        // read or alias header fields as fourccs; memcpy because a malformed
                        // offset may be unaligned.
                        const auto formatsBegin = static_cast<uint64_t>(modBlob->formats_offset);
                        const uint64_t formatsEnd =
                            formatsBegin + (static_cast<uint64_t>(modBlob->count_formats) * sizeof(uint32_t));
                        if (formatsBegin >= sizeof(*modBlob) && formatsEnd <= blob->length) {
                            for (uint32_t k = 0; k < modBlob->count_formats; ++k) {
                                uint32_t planeFormat = 0;
                                const size_t formatOffset =
                                    static_cast<size_t>(formatsBegin) + (static_cast<size_t>(k) * sizeof(planeFormat));
                                std::memcpy(&planeFormat, base + formatOffset, sizeof(planeFormat));
                                if (planeFormat == DRM_FORMAT_P010) {
                                    tempProps.supportsP010 = true;
                                }
                                if (planeFormat == format) {
                                    hasFormatSupport = true;
                                }
                            }
                        }
                    }
                }
            } else if ((prop->flags & DRM_MODE_PROP_IMMUTABLE) && std::strcmp(name, "type") == 0) {
                planeType = static_cast<uint32_t>(planeProps->prop_values[j]);
                tempProps.type = planeType;
            } else if (std::strcmp(name, "CRTC_ID") == 0) {
                tempProps.crtcId = prop->prop_id;
            } else if (std::strcmp(name, "FB_ID") == 0) {
                tempProps.fbId = prop->prop_id;
            } else if (std::strcmp(name, "SRC_X") == 0) {
                tempProps.srcX = prop->prop_id;
            } else if (std::strcmp(name, "SRC_Y") == 0) {
                tempProps.srcY = prop->prop_id;
            } else if (std::strcmp(name, "SRC_W") == 0) {
                tempProps.srcW = prop->prop_id;
            } else if (std::strcmp(name, "SRC_H") == 0) {
                tempProps.srcH = prop->prop_id;
            } else if (std::strcmp(name, "CRTC_X") == 0) {
                tempProps.crtcX = prop->prop_id;
            } else if (std::strcmp(name, "CRTC_Y") == 0) {
                tempProps.crtcY = prop->prop_id;
            } else if (std::strcmp(name, "CRTC_W") == 0) {
                tempProps.crtcW = prop->prop_id;
            } else if (std::strcmp(name, "CRTC_H") == 0) {
                tempProps.crtcH = prop->prop_id;
            } else if (std::strcmp(name, "zpos") == 0) {
                tempProps.zpos = prop->prop_id;
            } else if (std::strcmp(name, "pixel blend mode") == 0) {
                tempProps.pixelBlendMode = prop->prop_id;
            } else if (std::strcmp(name, "COLOR_ENCODING") == 0) {
                tempProps.colorEncoding = prop->prop_id;
                for (int e = 0; e < prop->count_enums; ++e) {
                    const char *enumName = prop->enums[e].name;
                    if (std::strcmp(enumName, "ITU-R BT.709 YCbCr") == 0) {
                        tempProps.colorEncodingBt709 = prop->enums[e].value;
                        tempProps.colorEncodingValid = true;
                    } else if (std::strcmp(enumName, "ITU-R BT.2020 YCbCr") == 0) {
                        tempProps.colorEncodingBt2020 = prop->enums[e].value;
                        tempProps.colorEncodingBt2020Valid = true;
                    }
                }
                dsyslog("vaapivideo/display: plane %u COLOR_ENCODING prop=%u bt709=%lu(%s) bt2020=%lu(%s)",
                        plane->plane_id, tempProps.colorEncoding, (unsigned long)tempProps.colorEncodingBt709,
                        tempProps.colorEncodingValid ? "yes" : "no", (unsigned long)tempProps.colorEncodingBt2020,
                        tempProps.colorEncodingBt2020Valid ? "yes" : "no");
            } else if (std::strcmp(name, "COLOR_RANGE") == 0) {
                tempProps.colorRange = prop->prop_id;
                for (int e = 0; e < prop->count_enums; ++e) {
                    if (std::strcmp(prop->enums[e].name, "YCbCr limited range") == 0) {
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

        dsyslog("vaapivideo/display: candidate plane %u type=%s (match #%d, want #%d)", plane->plane_id,
                GetPlaneTypeName(planeType), found, planeIndex);

        if (preferHdrCapable) {
            const bool hdrCapable =
                tempProps.supportsP010 && tempProps.colorEncodingValid && tempProps.colorEncodingBt2020Valid;
            if (!hdrCapable) {
                if (fallbackPlaneId == 0) {
                    fallbackPlaneId = plane->plane_id;
                    fallbackPlaneType = planeType;
                    fallbackProps = tempProps;
                }
                continue;
            }
        } else if (found != planeIndex) {
            found++;
            continue;
        }

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
        // The *Found flags say the enum value was located, not what it is; the per-plane probe
        // lines above print the values, and shorter names made the two lines look contradictory.
        dsyslog("vaapivideo/display: plane %u props: fbId=%u crtcId=%u srcX/Y/W/H=%u/%u/%u/%u "
                "crtcX/Y/W/H=%u/%u/%u/%u zpos=%u blend=%u colorEncoding=%u(bt709Found=%d,bt2020Found=%d) "
                "colorRange=%u(limitedFound=%d) supportsP010=%d",
                planeId, props.fbId, props.crtcId, props.srcX, props.srcY, props.srcW, props.srcH, props.crtcX,
                props.crtcY, props.crtcW, props.crtcH, props.zpos, props.pixelBlendMode, props.colorEncoding,
                props.colorEncodingValid ? 1 : 0, props.colorEncodingBt2020Valid ? 1 : 0, props.colorRange,
                props.colorRangeValid ? 1 : 0, props.supportsP010 ? 1 : 0);
        return true;
    }

    if (fallbackPlaneId != 0) {
        videoPlaneId = fallbackPlaneId;
        videoProps = fallbackProps;
        isyslog("vaapivideo/display: video plane %u type=%s (fallback: no HDR-capable plane)", videoPlaneId,
                GetPlaneTypeName(fallbackPlaneType));
        return true;
    }

    esyslog("vaapivideo/display: no suitable plane found for index %d format 0x%08x", planeIndex, format);
    return false;
}

[[nodiscard]] auto cVaapiDisplay::DrainDrmEvents(int timeoutMs) -> bool {
    // The consumer thread is the ONLY caller while Action() runs; Shutdown() drains from
    // the main thread only after hasExited. WaitForPageFlip() deliberately does NOT call
    // this (see the regression note there) -- never add a second concurrent reader.
    if (drmFd < 0) [[unlikely]] {
        return false;
    }
    pollfd pfd{.fd = drmFd, .events = POLLIN, .revents = 0};
    const int ret = poll(&pfd, 1, timeoutMs);

    // Reject error revents: dispatching after POLLERR/POLLHUP (device gone, TTY switch)
    // would hand drmHandleEvent a dead fd.
    if (ret <= 0 || (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0 || (pfd.revents & POLLIN) == 0) {
        return false;
    }
    return drmHandleEvent(drmFd, &eventContext) == 0;
}

[[nodiscard]] auto cVaapiDisplay::LoadDrmProperties() -> bool {
    // UNIVERSAL_PLANES: exposes overlay and cursor planes (default: only primary/cursor).
    // ATOMIC: switches the fd to the atomic mode-setting uAPI used everywhere below.
    // Both are per-fd opt-ins; no-ops on already-set caps.
    (void)drmSetClientCap(drmFd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1);
    (void)drmSetClientCap(drmFd, DRM_CLIENT_CAP_ATOMIC, 1);

    // One-time DRM identification + relevant device caps. Logged at init so the syslog
    // of any field deploy carries the kernel driver fingerprint relevant to atomic-commit
    // behavior (modifiers, async page-flip, vblank events).
    if (drmVersionPtr v = drmGetVersion(drmFd); v != nullptr) {
        dsyslog("vaapivideo/display: DRM driver=%.*s %d.%d.%d caps: addfb2_modifiers=%lu async_flip=%lu "
                "vblank_event=%lu prime=%lu dumb=%lu",
                v->name_len, v->name ? v->name : "?", v->version_major, v->version_minor, v->version_patchlevel,
                static_cast<unsigned long>(GetDrmCap(drmFd, DRM_CAP_ADDFB2_MODIFIERS)),
                static_cast<unsigned long>(GetDrmCap(drmFd, DRM_CAP_ASYNC_PAGE_FLIP)),
                static_cast<unsigned long>(GetDrmCap(drmFd, DRM_CAP_CRTC_IN_VBLANK_EVENT)),
                static_cast<unsigned long>(GetDrmCap(drmFd, DRM_CAP_PRIME)),
                static_cast<unsigned long>(GetDrmCap(drmFd, DRM_CAP_DUMB_BUFFER)));
        drmFreeVersion(v);
    }

    // CRTC ACTIVE / MODE_ID: needed by every modeset commit.
    auto crtcProps = std::unique_ptr<drmModeObjectProperties, decltype(&drmModeFreeObjectProperties)>(
        drmModeObjectGetProperties(drmFd, crtcId, DRM_MODE_OBJECT_CRTC), drmModeFreeObjectProperties);

    if (crtcProps) {
        for (uint32_t i = 0; i < crtcProps->count_props; ++i) {
            auto prop = std::unique_ptr<drmModePropertyRes, decltype(&drmModeFreeProperty)>(
                drmModeGetProperty(drmFd, crtcProps->props[i]), drmModeFreeProperty);
            if (!prop) {
                continue;
            }
            if (std::strcmp(prop->name, "ACTIVE") == 0) {
                modesetProps.crtcActive = prop->prop_id;
            } else if (std::strcmp(prop->name, "MODE_ID") == 0) {
                modesetProps.crtcModeId = prop->prop_id;
            }
        }
    }

    // Connector CRTC_ID: needed to bind/unbind the connector during enable/disable.
    auto connProps = std::unique_ptr<drmModeObjectProperties, decltype(&drmModeFreeObjectProperties)>(
        drmModeObjectGetProperties(drmFd, connectorId, DRM_MODE_OBJECT_CONNECTOR), drmModeFreeObjectProperties);

    if (connProps) {
        for (uint32_t i = 0; i < connProps->count_props; ++i) {
            auto prop = std::unique_ptr<drmModePropertyRes, decltype(&drmModeFreeProperty)>(
                drmModeGetProperty(drmFd, connProps->props[i]), drmModeFreeProperty);
            if (!prop) {
                continue;
            }
            if (std::strcmp(prop->name, "CRTC_ID") == 0) {
                modesetProps.connectorCrtcId = prop->prop_id;
            }
        }
    }

    modesetProps.isValid =
        (modesetProps.crtcActive != 0 && modesetProps.crtcModeId != 0 && modesetProps.connectorCrtcId != 0);
    return modesetProps.isValid;
}

auto cVaapiDisplay::ProbeHdrCapabilities() -> void {
    // Populates hdrProps (prop IDs used in every commit) and displayCaps (capability bits
    // consumed by CanDriveHdrPlane / SupportsHdrPassthrough) from the same connector/EDID walk.
    // The DRM I/O (property IDs, plane formats, raw EDID blob) lives here because hdrProps must
    // stay with cVaapiDisplay; the pure EDID byte parse is delegated to ParseEdidHdrCaps (caps.cpp).
    // Clear first: a re-probe after hotplug must not inherit bits from the previous sink.
    hdrProps = HdrConnectorProps{};
    displayCaps = DisplayCaps{};

    auto connProps = std::unique_ptr<drmModeObjectProperties, decltype(&drmModeFreeObjectProperties)>(
        drmModeObjectGetProperties(drmFd, connectorId, DRM_MODE_OBJECT_CONNECTOR), drmModeFreeObjectProperties);

    std::vector<uint8_t> edidBlob;
    bool haveBt2020Ycc = false;
    bool haveDefault = false;
    if (connProps) {
        for (uint32_t i = 0; i < connProps->count_props; ++i) {
            auto prop = std::unique_ptr<drmModePropertyRes, decltype(&drmModeFreeProperty)>(
                drmModeGetProperty(drmFd, connProps->props[i]), drmModeFreeProperty);
            if (!prop) {
                continue;
            }
            const char *name = prop->name;
            if (std::strcmp(name, "HDR_OUTPUT_METADATA") == 0) {
                hdrProps.hdrOutputMetadata = prop->prop_id;
            } else if (std::strcmp(name, "Colorspace") == 0) {
                hdrProps.colorspace = prop->prop_id;
                for (int e = 0; e < prop->count_enums; ++e) {
                    if (std::strcmp(prop->enums[e].name, "BT2020_YCC") == 0) {
                        hdrProps.colorspaceBt2020Ycc = prop->enums[e].value;
                        haveBt2020Ycc = true;
                    } else if (std::strcmp(prop->enums[e].name, "Default") == 0) {
                        hdrProps.colorspaceDefault = prop->enums[e].value;
                        haveDefault = true;
                    }
                }
                // Guard: if BT2020_YCC and Default map to the same value, SDR and HDR
                // commits would be indistinguishable and the colorspace would never actually change.
                hdrProps.colorspaceValid =
                    haveBt2020Ycc && haveDefault && hdrProps.colorspaceBt2020Ycc != hdrProps.colorspaceDefault;
            } else if (std::strcmp(name, "max bpc") == 0 && prop->count_values >= 2) {
                // Range property [min, max]: require both bounds so the commit path can't
                // produce clamp(10, 0, 0) == 0 bpc, which the kernel rejects.
                hdrProps.maxBpc = prop->prop_id;
                hdrProps.maxBpcMin = prop->values[0];
                hdrProps.maxBpcMax = prop->values[1];
            } else if (std::strcmp(name, "EDID") == 0) {
                const auto blobId = static_cast<uint32_t>(connProps->prop_values[i]);
                if (blobId != 0) {
                    auto blob = std::unique_ptr<drmModePropertyBlobRes, decltype(&drmModeFreePropertyBlob)>(
                        drmModeGetPropertyBlob(drmFd, blobId), drmModeFreePropertyBlob);
                    if (blob && blob->data && blob->length > 0) {
                        const auto *src = static_cast<const uint8_t *>(blob->data);
                        edidBlob.assign(src, src + blob->length);
                    }
                }
            }
        }
    }

    if (!edidBlob.empty()) {
        ParseEdidHdrCaps(std::span<const uint8_t>{edidBlob.data(), edidBlob.size()}, displayCaps);
    }
    // P010 and COLOR_ENCODING flags were already sniffed by BindDrmPlane(NV12); copy here.
    displayCaps.planeSupportsP010 = (videoPlaneId != 0) && videoProps.supportsP010;
    displayCaps.planeColorEncodingValid = videoProps.colorEncodingValid;
    displayCaps.planeColorEncodingBt2020 = videoProps.colorEncodingBt2020Valid;
    displayCaps.hasHdrOutputMetadata = (hdrProps.hdrOutputMetadata != 0);
    displayCaps.hasColorspaceEnum = hdrProps.colorspaceValid;
    displayCaps.colorspaceBt2020Ycc = hdrProps.colorspaceValid; // distinct from Default (validated above)
    displayCaps.hasMaxBpc = (hdrProps.maxBpc != 0);
    displayCaps.maxBpcSupported = static_cast<uint8_t>(std::clamp<uint64_t>(hdrProps.maxBpcMax, 8U, 16U));

    isyslog("vaapivideo/display: HDR caps -- connector: metadata=%s colorspace=%s max_bpc=%s[%lu..%lu]; "
            "sink: pq=%s hlg=%s bt2020ycc=%s; plane: p010=%s bt2020enc=%s",
            hdrProps.hdrOutputMetadata ? "yes" : "no", hdrProps.colorspaceValid ? "yes" : "no",
            hdrProps.maxBpc ? "yes" : "no", static_cast<unsigned long>(hdrProps.maxBpcMin),
            static_cast<unsigned long>(hdrProps.maxBpcMax), displayCaps.sinkHdr10Pq ? "yes" : "no",
            displayCaps.sinkHlg ? "yes" : "no", displayCaps.sinkBt2020Ycc ? "yes" : "no",
            displayCaps.planeSupportsP010 ? "yes" : "no", displayCaps.planeColorEncodingBt2020 ? "yes" : "no");
}

[[nodiscard]] auto cVaapiDisplay::CanDriveHdrPlane() const noexcept -> bool { return displayCaps.CanDriveHdrPlane(); }

auto cVaapiDisplay::SetHdrOutputState(const HdrStreamInfo &info) -> void {
    // hdrStateMutex ensures the display thread reads a torn-write-free snapshot of the
    // multi-word AVMasteringDisplayMetadata / AVContentLightMetadata fields.
    const cMutexLock lock(&hdrStateMutex);
    stagedHdrState = info;
}

namespace {

/// Round a non-negative double to uint16_t with clamping.
/// std::lround is correct for all signs; (d + 0.5) cast fails for small negative d (NaN-safe too).
[[nodiscard]] auto RoundU16(double d) noexcept -> uint16_t {
    if (!(d > 0.0)) { // covers NaN too
        return 0;
    }
    if (d >= 65535.0) {
        return 0xFFFF;
    }
    return static_cast<uint16_t>(std::lround(d));
}

/// Convert an AVRational in [0, 1] range to the unsigned 16-bit EDID/HDMI primary
/// coordinate (units of 0.00002, so 1.0 == 0xC350). Clamped to [0, 0xFFFF].
[[nodiscard]] auto EncodePrimary(AVRational r) noexcept -> uint16_t {
    if (r.den == 0) {
        return 0;
    }
    return RoundU16(av_q2d(r) * 50000.0);
}

[[nodiscard]] auto AvRationalEqual(AVRational x, AVRational y) noexcept -> bool {
    return x.num == y.num && x.den == y.den;
}

/// Compare two 2-element AVRational arrays (chromaticity XY, an FFmpeg ABI C-array type).
/// A named helper keeps the paired element compares readable at the call sites.
[[nodiscard]] auto XyRationalEqual(const AVRational (&a)[2], const AVRational (&b)[2]) noexcept -> bool {
    return AvRationalEqual(a[0], b[0]) && AvRationalEqual(a[1], b[1]);
}

[[nodiscard]] auto MasteringEqual(const AVMasteringDisplayMetadata &a, const AVMasteringDisplayMetadata &b) noexcept
    -> bool {
    // Field-by-field: memcmp would compare implementation-defined padding bytes that
    // AVMasteringDisplayMetadata's trivial assignment operator does NOT copy.
    if (a.has_primaries != b.has_primaries || a.has_luminance != b.has_luminance) {
        return false;
    }
    if (a.has_primaries) {
        for (int i = 0; i < 3; ++i) {
            if (!XyRationalEqual(a.display_primaries[i], b.display_primaries[i])) {
                return false;
            }
        }
        if (!XyRationalEqual(a.white_point, b.white_point)) {
            return false;
        }
    }
    if (a.has_luminance &&
        (!AvRationalEqual(a.max_luminance, b.max_luminance) || !AvRationalEqual(a.min_luminance, b.min_luminance))) {
        return false;
    }
    return true;
}

[[nodiscard]] auto HdrOutputStateEqual(const HdrStreamInfo &a, const HdrStreamInfo &b) noexcept -> bool {
    if (a.kind != b.kind || a.hasMasteringDisplay != b.hasMasteringDisplay || a.hasContentLight != b.hasContentLight) {
        return false;
    }
    if (a.hasMasteringDisplay && !MasteringEqual(a.mastering, b.mastering)) {
        return false;
    }
    if (a.hasContentLight &&
        (a.contentLight.MaxCLL != b.contentLight.MaxCLL || a.contentLight.MaxFALL != b.contentLight.MaxFALL)) {
        return false;
    }
    return true;
}

// HDMI EOTF codes per CTA-861.3 / HDMI 2.0a.
constexpr uint8_t HDMI_EOTF_SMPTE_ST_2084 = 2; // HDR10 PQ
constexpr uint8_t HDMI_EOTF_ARIB_STD_B67 = 3;  // HLG

/// Populate a Static Metadata Type 1 infoframe from a stream's HDR side-data. Missing
/// mastering / content-light side-data leaves the corresponding fields zero, which per
/// HDMI 2.1 section 7.6.1 the sink interprets as "unknown" (accepted for HDR10 and HLG alike).
[[nodiscard]] auto BuildHdrMetadataInfoframe(const HdrStreamInfo &info) noexcept -> hdr_output_metadata {
    hdr_output_metadata meta{};
    meta.metadata_type = 0; // HDMI_STATIC_METADATA_TYPE1
    // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access) -- DRM ABI requires union access
    auto &m = meta.hdmi_metadata_type1;
    m.metadata_type = 0;
    m.eotf = (info.kind == StreamHdrKind::Hlg) ? HDMI_EOTF_ARIB_STD_B67 : HDMI_EOTF_SMPTE_ST_2084;
    if (info.hasMasteringDisplay) {
        if (info.mastering.has_primaries) {
            // AVMasteringDisplayMetadata and hdr_metadata_infoframe both document (r, g, b)
            // primary order; FFmpeg's HEVC decoder already translates the SEI's native
            // (g, b, r) layout, so no further reordering is needed here.
            for (int i = 0; i < 3; ++i) {
                m.display_primaries[i].x = EncodePrimary(info.mastering.display_primaries[i][0]);
                m.display_primaries[i].y = EncodePrimary(info.mastering.display_primaries[i][1]);
            }
            m.white_point.x = EncodePrimary(info.mastering.white_point[0]);
            m.white_point.y = EncodePrimary(info.mastering.white_point[1]);
        }
        if (info.mastering.has_luminance) {
            // drm_mode.h: max_display_mastering_luminance in cd/m^2, min in 0.0001 cd/m^2.
            m.max_display_mastering_luminance = RoundU16(av_q2d(info.mastering.max_luminance));
            m.min_display_mastering_luminance = RoundU16(av_q2d(info.mastering.min_luminance) * 10000.0);
        }
    }
    if (info.hasContentLight) {
        // AVContentLightMetadata fields are unsigned int; the HDMI infoframe slots are u16.
        constexpr unsigned int kLightMax = 0xFFFFU;
        m.max_cll = static_cast<uint16_t>(std::min(info.contentLight.MaxCLL, kLightMax));
        m.max_fall = static_cast<uint16_t>(std::min(info.contentLight.MaxFALL, kLightMax));
    }
    // NOLINTEND(cppcoreguidelines-pro-type-union-access)
    return meta;
}

} // namespace

[[nodiscard]] auto cVaapiDisplay::MaybeAppendHdrOutputState(AtomicRequest &req, bool &failed) -> bool {
    failed = false;
    // hdrStateMutex guards the multi-word struct snapshots; keep the blob-create ioctl below
    // outside the lock so it can't stall the decoder's SetHdrOutputState.
    HdrStreamInfo staged;
    HdrStreamInfo applied;
    {
        const cMutexLock lock(&hdrStateMutex);
        staged = stagedHdrState;
        applied = appliedHdrState;
    }
    if (HdrOutputStateEqual(staged, applied)) {
        return false; // ApplyDisplayMode() pre-programmed the SDR baseline, so the first
                      // real frame with staged == Sdr legitimately skips the write here.
    }

    const bool wantActive = (staged.kind != StreamHdrKind::Sdr);
    uint32_t newBlobId = 0;

    if (wantActive && hdrProps.hdrOutputMetadata != 0) {
        const hdr_output_metadata meta = BuildHdrMetadataInfoframe(staged);
        if (drmModeCreatePropertyBlob(drmFd, &meta, sizeof(meta), &newBlobId) != 0) [[unlikely]] {
            // Shipping BT.2020 + 10 bpc without an EOTF blob renders as crushed green on most
            // sinks, so abort the whole commit and retry on the next frame.
            esyslog("vaapivideo/display: drmModeCreatePropertyBlob(HDR_OUTPUT_METADATA) failed: %s",
                    std::strerror(errno));
            failed = true;
            return false;
        }
    }

    bool appended = false;
    if (hdrProps.hdrOutputMetadata != 0) {
        req.AddProperty(connectorId, hdrProps.hdrOutputMetadata, newBlobId);
        appended = true;
    }
    if (hdrProps.colorspaceValid) {
        req.AddProperty(connectorId, hdrProps.colorspace,
                        wantActive ? hdrProps.colorspaceBt2020Ycc : hdrProps.colorspaceDefault);
        appended = true;
    }
    if (hdrProps.maxBpc != 0) {
        // 10 bpc for HDR, 8 bpc for SDR -- clamped to the property's advertised range because
        // some HDR-only displays expose a minimum > 8 bpc and would reject an unconditional 8.
        const uint64_t requestedBpc = wantActive ? 10U : 8U;
        const uint64_t bpc = std::clamp(requestedBpc, hdrProps.maxBpcMin, hdrProps.maxBpcMax);
        req.AddProperty(connectorId, hdrProps.maxBpc, bpc);
        appended = true;
    }

    // Optimistically promote the new blob; PresentBuffer() rolls back on commit failure.
    // Blob IDs are display-thread-only (no lock); appliedHdrState is only ever touched
    // under hdrStateMutex.
    pendingDestroyHdrBlobId = appliedHdrBlobId;
    appliedHdrBlobId = newBlobId;
    {
        const cMutexLock lock(&hdrStateMutex);
        appliedHdrState = staged;
    }
    if (appended) {
        isyslog("vaapivideo/display: HDR state -- committing kind=%s blob=%u", StreamHdrKindName(staged.kind),
                newBlobId);
    }
    return appended;
}

[[nodiscard]] auto cVaapiDisplay::SupportsHdrPassthrough(StreamHdrKind kind) const noexcept -> bool {
    // Delegates to DisplayCaps::SupportsHdrKind which combines CanDriveHdrPlane()
    // with the sink-side EDID gate. Sdr always returns true inside the delegate
    // when the plane is drivable; Auto-mode callers that want Sdr treated as "no
    // passthrough required" must check kind == Sdr before calling.
    if (kind == StreamHdrKind::Sdr) {
        return false;
    }
    return displayCaps.SupportsHdrKind(kind);
}

[[nodiscard]] auto cVaapiDisplay::MapVaapiFrame(std::unique_ptr<VaapiFrame> vaapiFrame) const -> DrmFramebuffer {
    if (!vaapiFrame || !vaapiFrame->avFrame || vaapiFrame->avFrame->format != AV_PIX_FMT_VAAPI) [[unlikely]] {
        return {};
    }

    const AVFrame *srcFrame = vaapiFrame->avFrame;

    // Export the VAAPI surface as DRM PRIME so it can be wrapped in a KMS framebuffer below.
    AVFrame *mappedFrame = av_frame_alloc();
    if (!mappedFrame) [[unlikely]] {
        return {};
    }

    mappedFrame->format = AV_PIX_FMT_DRM_PRIME;
    // MAP_READ: KMS scanout reads pixels.
    // MAP_DIRECT: zero-copy -- the PRIME fd refers to the same memory as the VA surface,
    //   no intermediate copy is allocated. Without this every frame would be duplicated on
    //   the GPU heap.
    // vaDriverMutex: serializes VA-driver entry against the decoder thread's filter-graph
    //   execution -- driving one VADisplay from two threads at once fails sporadically with
    //   VA_STATUS_ERROR_OPERATION_FAILED. The decoder takes the same lock around its push/pull.
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

    // FFmpeg ABI: an AV_PIX_FMT_DRM_PRIME frame's data[0] is the AVDRMFrameDescriptor pointer.
    const auto *desc =
        reinterpret_cast<const AVDRMFrameDescriptor *>( // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
            mappedFrame->data[0]);
    // The rest of this function assumes a single DMA-BUF object holding both NV12 layers
    // (Y + UV at different offsets) -- the shape every stack tested returns. A driver that split
    // planes across multiple objects would need a per-object GEM import + multi-fd AddFB2 path. Reject early so the
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
        esyslog("vaapivideo/display: drmPrimeFDToHandle failed: %s", std::strerror(errno));
        av_frame_free(&mappedFrame);
        return {};
    }

    // drmModeAddFB2WithModifiers takes parallel arrays per FB plane (handle/pitch/offset/
    // modifier). Walk the AVDRMFrameDescriptor's layers/planes to populate them.
    //
    // Pick the DRM fourcc from hw_frames_ctx->sw_format -- the VPP surface was explicitly
    // allocated with this layout, so it's the authoritative source. The PRIME descriptor's
    // layer[0].format is NOT reliable: a driver can report a fourcc that KMS rejects in
    // combination with the exported modifier (seen on iHD 25.x), producing spurious AddFB2
    // EINVAL on plain SDR NV12 scanout.
    uint32_t format = DRM_FORMAT_NV12;
    if (srcFrame->hw_frames_ctx) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- FFmpeg ABI
        const auto *framesCtx = reinterpret_cast<const AVHWFramesContext *>(srcFrame->hw_frames_ctx->data);
        if (framesCtx->sw_format == AV_PIX_FMT_P010) {
            format = DRM_FORMAT_P010;
        }
    }
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

    // NV12 and P010 both: one Y plane + one interleaved UV plane. Anything else means the
    // descriptor doesn't actually describe a 4:2:0 two-plane layout and AddFB2 would reject
    // it anyway.
    if (planeIdx != 2) [[unlikely]] {
        esyslog("vaapivideo/display: unexpected plane count %d (expected 2 for NV12/P010)", planeIdx);
        drm_gem_close closeArgs{.handle = gemHandle, .pad = 0};
        drmIoctl(drmFd, DRM_IOCTL_GEM_CLOSE, &closeArgs);
        av_frame_free(&mappedFrame);
        return {};
    }

    uint32_t fbId = 0;
    if (drmModeAddFB2WithModifiers(drmFd, width, height, format, handles, pitches, offsets, modifiers, &fbId,
                                   DRM_MODE_FB_MODIFIERS) != 0) {
        esyslog("vaapivideo/display: drmModeAddFB2WithModifiers failed: %s", std::strerror(errno));
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
    // The token travels with it: this fb is now what keeps its VPP graph alive.
    fb.frame = vaapiFrame->avFrame;
    fb.graphToken = std::move(vaapiFrame->graphToken);
    vaapiFrame->avFrame = nullptr;
    vaapiFrame->ownsFrame = false;

    av_frame_free(&mappedFrame);

    return fb;
}

auto cVaapiDisplay::OnPageFlipEvent([[maybe_unused]] int fd, [[maybe_unused]] unsigned int seq,
                                    [[maybe_unused]] unsigned int sec, [[maybe_unused]] unsigned int usec, void *data)
    -> void {
    // Dispatched from drmHandleEvent() on the consumer thread (sole reader while Action()
    // runs -- file-header DRM fd rule). `data` is the `this` cookie from drmModeAtomicCommit.
    // Release-store on isFlipPending publishes "flip done" to acquire-loaders.
    auto *display = static_cast<cVaapiDisplay *>(data);
    if (display) {
        display->lastVSyncTimeMs.store(cTimeMs::Now(), std::memory_order_release);
        display->flipPendingSinceMs.store(0, std::memory_order_release);
        display->isFlipPending.store(false, std::memory_order_release);
    }
}

[[nodiscard]] auto FitVideoToRect(uint32_t srcWidth, uint32_t srcHeight, const cRect &rect) noexcept -> VideoPlacement {
    if (srcWidth == 0 || srcHeight == 0 || rect.Width() < 2 || rect.Height() < 2) [[unlikely]] {
        return {};
    }
    const auto rectW = static_cast<uint32_t>(rect.Width());
    const auto rectH = static_cast<uint32_t>(rect.Height());
    uint32_t w = 0;
    uint32_t h = 0;
    if (static_cast<uint64_t>(srcWidth) * rectH >= static_cast<uint64_t>(srcHeight) * rectW) {
        w = rectW;
        h = static_cast<uint32_t>(static_cast<uint64_t>(rectW) * srcHeight / srcWidth);
    } else {
        h = rectH;
        w = static_cast<uint32_t>(static_cast<uint64_t>(rectH) * srcWidth / srcHeight);
    }
    w = std::max(2U, w & ~1U);
    h = std::max(2U, h & ~1U);
    const int32_t cx = rect.X() + ((static_cast<int32_t>(rectW) - static_cast<int32_t>(w)) / 2);
    const int32_t cy = rect.Y() + ((static_cast<int32_t>(rectH) - static_cast<int32_t>(h)) / 2);
    return {.destX = static_cast<uint32_t>(std::max(0, cx) & ~1),
            .destY = static_cast<uint32_t>(std::max(0, cy) & ~1),
            .height = h,
            .width = w};
}

[[nodiscard]] auto cVaapiDisplay::GetVideoRect() const -> cRect {
    const cMutexLock lock(&videoRectMutex);
    return videoRect;
}

[[nodiscard]] auto cVaapiDisplay::GetTargetVideoRect() const -> cRect {
    const cMutexLock lock(&videoRectMutex);
    return targetVideoRect;
}

[[nodiscard]] auto cVaapiDisplay::NormalizeVideoRect(const cRect &rect) const -> cRect {
    const int outW = static_cast<int>(GetOutputWidth());
    const int outH = static_cast<int>(GetOutputHeight());
    // Guard invalid modes; std::clamp() with lo > hi is UB.
    if (outW < 2 || outH < 2) [[unlikely]] {
        return cRect::Null;
    }
    if (rect.IsEmpty()) {
        return {0, 0, outW, outH};
    }
    // 2-px alignment: NV12/P010 chroma is 4:2:0 (same constraint for 8- and 10-bit).
    const int x = std::clamp(rect.X(), 0, outW - 2) & ~1;
    const int y = std::clamp(rect.Y(), 0, outH - 2) & ~1;
    const int w = std::clamp(rect.Width() & ~1, 2, (outW - x) & ~1);
    const int h = std::clamp(rect.Height() & ~1, 2, (outH - y) & ~1);
    return {x, y, w, h};
}

[[nodiscard]] auto cVaapiDisplay::SetVideoRect(const cRect &rect) -> bool {
    const cRect normalized = NormalizeVideoRect(rect);
    if (normalized.IsEmpty()) [[unlikely]] {
        return false;
    }
    // Stage the target only; PresentBuffer promotes the active scanout rect (videoRect) once a fb
    // of the new size arrives. Old-sized jitterBuf frames keep painting at the old rect during the
    // VPP rebuild, so no underrun and no crop.
    bool dimsChanged = false;
    {
        const cMutexLock lock(&videoRectMutex);
        dimsChanged = normalized.Width() != targetVideoRect.Width() || normalized.Height() != targetVideoRect.Height();
        targetVideoRect = normalized;
    }
    return dimsChanged;
}

[[nodiscard]] auto cVaapiDisplay::PresentBuffer(const DrmFramebuffer &fb) -> bool {
    if (!fb.IsValid()) {
        return false;
    }

    // The filter chain pre-fits the fb to targetVideoRect; the active videoRect advances only once
    // a fb matching the new target arrives, so old-sized jitterBuf frames keep painting at the old
    // rect during the rebuild (no underrun, no crop). KMS scanout is always 1:1. Promotion is
    // deferred: if this commit fails, videoRect must not have moved ahead of what KMS holds.
    cRect vr;
    cRect promoteRect;
    bool promoteVideoRect = false;
    {
        const cMutexLock lock(&videoRectMutex);
        const auto fitTarget = FitVideoToRect(fb.width, fb.height, targetVideoRect);
        if (fitTarget.width == fb.width && fitTarget.height == fb.height) {
            vr = targetVideoRect;
            promoteRect = targetVideoRect;
            promoteVideoRect = (vr.Width() != videoRect.Width() || vr.Height() != videoRect.Height() ||
                                vr.X() != videoRect.X() || vr.Y() != videoRect.Y());
        } else {
            vr = videoRect;
        }
    }
    const auto placement = FitVideoToRect(fb.width, fb.height, vr);
    if (placement.width == 0) [[unlikely]] {
        return false;
    }

    // Post-modeset gate. SRC_W/H are written equal to CRTC_W/H below, an invariant that only holds
    // because the VPP pre-fits every fb to the scanout rect -- KMS itself does not scale. A frame
    // that was already in flight when the mode changed does not fit, and committing it would crop
    // the picture rather than shrink it, so hold the plane dark until the rebuilt chain catches up.
    if (awaitingResizedFb) [[unlikely]] {
        if (placement.width == fb.width && placement.height == fb.height) {
            awaitingResizedFb = false;
        } else if (cTimeMs::Now() - resizeWaitSince > DISPLAY_MODE_RESIZE_WAIT_MS) {
            esyslog("vaapivideo/display: no fb matching %ux%u after %llums -- resuming with mismatched geometry",
                    vr.Width(), vr.Height(), static_cast<unsigned long long>(cTimeMs::Now() - resizeWaitSince));
            awaitingResizedFb = false;
        } else {
            return false; // Action() falls through to its 5 ms sleep and retries with the next fb.
        }
    }

    const uint32_t destX = placement.destX;
    const uint32_t destY = placement.destY;
    const uint32_t planeW = placement.width;
    const uint32_t planeH = placement.height;

    AtomicRequest req;
    req.AddProperty(videoPlaneId, videoProps.crtcId, crtcId);
    req.AddProperty(videoPlaneId, videoProps.fbId, fb.fbId);

    // HDR connector signaling: must precede the plane color-space write so the kernel sees one
    // coherent HDR picture per commit. Snapshot for rollback on commit failure.
    HdrStreamInfo previousHdrState;
    {
        const cMutexLock lock(&hdrStateMutex);
        previousHdrState = appliedHdrState;
    }
    const uint32_t previousHdrBlobId = appliedHdrBlobId;
    bool hdrStateFailed = false;
    const bool hdrStateChanged = MaybeAppendHdrOutputState(req, hdrStateFailed);
    if (hdrStateFailed) [[unlikely]] {
        return false;
    }

    // Plane state: write each stateful property only on change. Some drivers treat redundant
    // rewrites as transitions and reject them on the steady-state page-flip path. Cache advances
    // only on commit success so a failed commit retries cleanly next frame.
    // Read AFTER MaybeAppendHdrOutputState: the plane color space must match the connector
    // state staged in this same commit.
    const bool hdrActive = GetActiveHdrKind() != StreamHdrKind::Sdr;
    const uint64_t stagedColorEncoding = (hdrActive && videoProps.colorEncodingBt2020Valid)
                                             ? videoProps.colorEncodingBt2020
                                             : videoProps.colorEncodingBt709;
    const uint64_t stagedSrcW = static_cast<uint64_t>(planeW) << 16;
    const uint64_t stagedSrcH = static_cast<uint64_t>(planeH) << 16;
    if (videoProps.colorEncodingValid && stagedColorEncoding != lastVideoColorEncoding) {
        req.AddProperty(videoPlaneId, videoProps.colorEncoding, stagedColorEncoding);
    }
    if (videoProps.colorRangeValid && videoProps.colorRangeLimited != lastVideoColorRange) {
        req.AddProperty(videoPlaneId, videoProps.colorRange, videoProps.colorRangeLimited);
    }
    // SRC_X/Y are always 0 (no source crop); bundled with the SRC_W/H write so they're emitted
    // once on the first commit and again only when the source rect actually changes.
    if (stagedSrcW != lastVideoSrcW || stagedSrcH != lastVideoSrcH) {
        req.AddProperty(videoPlaneId, videoProps.srcX, 0);
        req.AddProperty(videoPlaneId, videoProps.srcY, 0);
        req.AddProperty(videoPlaneId, videoProps.srcW, stagedSrcW);
        req.AddProperty(videoPlaneId, videoProps.srcH, stagedSrcH);
    }
    if (destX != lastVideoCrtcX) {
        req.AddProperty(videoPlaneId, videoProps.crtcX, destX);
    }
    if (destY != lastVideoCrtcY) {
        req.AddProperty(videoPlaneId, videoProps.crtcY, destY);
    }
    if (planeW != lastVideoCrtcW) {
        req.AddProperty(videoPlaneId, videoProps.crtcW, planeW);
    }
    if (planeH != lastVideoCrtcH) {
        req.AddProperty(videoPlaneId, videoProps.crtcH, planeH);
    }

    // Bundle the OSD plane into the same atomic commit -- separate commits let one plane lag a
    // vblank and tear over moving video. osdFbId reflects what the kernel will actually scan out
    // after this commit lands: if AppendOsdPlane hides (clipped off-screen / no OSD plane), it
    // reports false and we record 0 instead of currentOsd.fbId.
    // osdHdrSuppressed blocks only OSD *enables* (doomed commits); hides always land -- they only
    // reduce plane load, and a plane enabled before HDR must stay closable. Queued enables
    // (osdDirty stays set) land once HDR ends.
    const bool osdHidden = osdHdrSuppressed && hdrActive;
    bool osdCommitted = false;
    uint32_t osdFbId = 0;
    uint64_t osdCommitGeneration = 0;
    uint32_t prevOsdFbId = 0;
    {
        const cMutexLock lock(&osdMutex);
        prevOsdFbId = lastCommittedOsdFbId;
        osdFbId = prevOsdFbId;
        if (osdDirty && (currentOsd.fbId == 0 || !osdHidden)) {
            osdCommitted = true;
            osdCommitGeneration = osdGeneration;
            if (currentOsd.fbId != 0) {
                osdFbId = AppendOsdPlane(req, currentOsd) ? currentOsd.fbId : 0;
            } else if (osdPlaneId != 0) {
                req.AddProperty(osdPlaneId, osdProps.fbId, 0);
                req.AddProperty(osdPlaneId, osdProps.crtcId, 0);
                osdFbId = 0;
            }
        }
    }

    // Two commit paths: steady-state nonblocking page flip (flags=0), and sync ALLOW_MODESET for
    // HDR connector-state changes that may link-retrain. Plane/OSD updates ride the page-flip
    // path because scale_vaapi already emits the final framebuffer size, so KMS sees SRC == CRTC
    // and no plane scaler is involved -- the kernel internally picks fastset where applicable.
    //
    // Exception: an OSD plane transition over HDR may need a CDCLK-bump modeset (see AtomicCommit).
    // Once latched, only transitions pay the sync cost -- steady repaints keep their data rate and
    // stay fastset on the async path. HDR-transition commits (already ALLOW_MODESET) carry
    // osdHdrCommit for suppression eligibility, so a bandwidth-ceiling GPU can't spin forever.
    const bool osdHdrCommit = hdrActive && osdCommitted;
    const bool osdTransition = osdFbId != prevOsdFbId;
    const uint32_t commitFlags =
        (hdrStateChanged || (osdHdrCommit && osdTransition && osdHdrNeedsModeset)) ? DRM_MODE_ATOMIC_ALLOW_MODESET : 0U;
    const bool success = AtomicCommit(req, commitFlags, osdHdrCommit);
    if (success) {
        if (promoteVideoRect) {
            const cMutexLock lock(&videoRectMutex);
            videoRect = promoteRect;
        }
        lastVideoColorEncoding = stagedColorEncoding;
        lastVideoColorRange = videoProps.colorRangeLimited;
        lastVideoSrcW = stagedSrcW;
        lastVideoSrcH = stagedSrcH;
        lastVideoCrtcX = destX;
        lastVideoCrtcY = destY;
        lastVideoCrtcW = planeW;
        lastVideoCrtcH = planeH;
        if (osdCommitted) {
            // osdDirty clears only on a landed commit -- a failed one (e.g. EBUSY) must keep
            // the OSD update queued -- and only for the generation this commit actually staged:
            // a SetOsd racing the commit (e.g. in-place repaint of the same fbId) must stay dirty.
            const cMutexLock lock(&osdMutex);
            lastCommittedOsdFbId = osdFbId;
            if (osdGeneration == osdCommitGeneration) {
                osdDirty = false;
            }
            if (osdFbId != 0) {
                lastOsdPixelBlendMode = 1; // AppendOsdPlane wrote it iff it differed.
            }
        }
    }

    // HDR blob lifecycle. On success the kernel has taken over the reference; drop the
    // previous userspace one. On failure, discard the new blob that never reached the
    // kernel and restore the previously-applied state so the next frame can retry cleanly.
    if (hdrStateChanged) {
        if (success) {
            if (pendingDestroyHdrBlobId != 0) {
                if (drmModeDestroyPropertyBlob(drmFd, pendingDestroyHdrBlobId) != 0) [[unlikely]] {
                    esyslog("vaapivideo/display: failed to free previous HDR blob: %s", std::strerror(errno));
                }
                pendingDestroyHdrBlobId = 0;
            }
        } else {
            if (appliedHdrBlobId != 0 && drmModeDestroyPropertyBlob(drmFd, appliedHdrBlobId) != 0) [[unlikely]] {
                esyslog("vaapivideo/display: failed to free rejected HDR blob: %s", std::strerror(errno));
            }
            appliedHdrBlobId = previousHdrBlobId;
            pendingDestroyHdrBlobId = 0;
            // Rollback under hdrStateMutex so GetActiveHdrKind() observes the restored state
            // atomically with the same lock it acquires for reads.
            {
                const cMutexLock lock(&hdrStateMutex);
                appliedHdrState = previousHdrState;
            }
            esyslog("vaapivideo/display: atomic commit failed during HDR transition -- will retry next frame");
        }
    }

    return success;
}

[[nodiscard]] auto cVaapiDisplay::WaitForPageFlip(int timeoutMs) -> bool {
    // Observe-only wait for the consumer to drain the in-flight flip before BeginStreamSwitch()
    // takes importMutex (its flip-pending branch runs ahead of the isClearing check, so it
    // drains even mid-switch). NEVER drain the fd from here: a second drmHandleEvent reader
    // was bisect-verified to permanently halve the post-switch present cadence on i915/UHD.
    // Best-effort: a commit racing importMutex.Lock() can leave a flip in flight; old buffers
    // stay alive while the consumer continues to own DRM event dispatch.
    const cTimeMs deadline(timeoutMs);
    while (isFlipPending.load(std::memory_order_acquire) && !deadline.TimedOut()) {
        if (stopping.load(std::memory_order_relaxed) || !ready.load(std::memory_order_relaxed)) {
            return true; // teardown no longer needs a stream-switch drain result
        }
        cCondWait::SleepMs(1);
    }
    return !isFlipPending.load(std::memory_order_acquire);
}
