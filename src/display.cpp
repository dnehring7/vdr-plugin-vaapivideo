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
 * Lock order: importMutex -> bufferMutex -> osdMutex.
 *             {bufferMutex, vaDriverMutex} -> videoRectMutex (innermost; never held while acquiring any other lock).
 * WaitForPageFlip() MUST complete before BeginStreamSwitch() takes importMutex;
 * reversing that order deadlocks because the consumer cannot drain DRM events while
 * importMutex is held by the main thread.
 */

#include "display.h"
#include "caps.h"
#include "common.h"
#include "config.h"
#include "decoder.h"

// POSIX
#include <pthread.h>
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
#include <vdr/thread.h>
#include <vdr/tools.h>
#pragma GCC diagnostic pop

// ============================================================================
// === CONSTANTS ===
// ============================================================================

constexpr int DISPLAY_PAGE_FLIP_TIMEOUT_MS = 40; ///< ~2 vblanks @ 50 Hz: tolerates one missed flip before giving up
constexpr int DISPLAY_MAX_DRAIN_ITERATIONS =
    10; ///< Safety bound on post-shutdown DRM event drain (guards against infinite loops)

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

namespace {

// CTA-861-G sec.7.5.13 HDR Static Metadata Data Block (extended tag 0x06):
//   byte 1 = supported EOTFs bitmap: bit 0=SDR, bit 1=HDR gamma, bit 2=PQ, bit 3=HLG.
// CTA-861-G sec.7.5.6 Colorimetry Data Block (extended tag 0x05):
//   byte 1 = colorimetry flags: bit 6=BT.2020 YCC.
constexpr uint8_t EDID_EOTF_PQ = 1U << 2;
constexpr uint8_t EDID_EOTF_HLG = 1U << 3;
constexpr uint8_t EDID_COLORIMETRY_BT2020_YCC = 1U << 6;
constexpr size_t EDID_BLOCK_SIZE = 128;             ///< Every EDID block is exactly 128 bytes
constexpr size_t EDID_CHECKSUM_OFFSET = 127;        ///< Byte 127 is the block checksum (not a data block)
constexpr size_t EDID_EXTENSION_COUNT_OFFSET = 126; ///< Byte 126 of base block: number of extension blocks

/// Parse one 128-byte CTA-861 extension block (tag byte 0x02) for HDR EOTF and BT.2020 YCC.
/// Per CTA-861-G sec.7.3, a DTD offset of 0 means "no DTDs"; data blocks span [4, checksum).
/// Malformed blocks are silently ignored. Results are OR'd into @p caps so multiple extension
/// blocks accumulate rather than overwrite.
auto ParseCtaExtension(std::span<const uint8_t> ext, DisplayCaps &caps) -> void {
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- spans are size-checked
    if (ext.size() < 4 || ext[0] != 0x02) {
        return;
    }
    const size_t dtdOffset = ext[2];
    const size_t dataBlockEnd = (dtdOffset == 0) ? EDID_CHECKSUM_OFFSET : dtdOffset;
    if (dataBlockEnd < 4 || dataBlockEnd > ext.size()) {
        return;
    }
    size_t offset = 4;
    while (offset < dataBlockEnd) {
        const uint8_t header = ext[offset];
        const uint8_t blockTag = header >> 5;
        const uint8_t payloadLen = header & 0x1F;
        if (offset + 1 + payloadLen > dataBlockEnd) {
            break;
        }
        const std::span<const uint8_t> payload = ext.subspan(offset + 1, payloadLen);
        // CTA-861-G sec.7.5: blockTag==7 = "Extended Tag" block; payload[0] is the extended tag.
        if (blockTag == 7 && !payload.empty()) {
            const uint8_t extTag = payload[0];
            if (extTag == 0x06 && payload.size() >= 3) {
                // HDR Static Metadata Data Block (sec.7.5.13): payload[1] = EOTF bitmap.
                caps.sinkHdr10Pq = caps.sinkHdr10Pq || (payload[1] & EDID_EOTF_PQ) != 0;
                caps.sinkHlg = caps.sinkHlg || (payload[1] & EDID_EOTF_HLG) != 0;
            } else if (extTag == 0x05 && payload.size() >= 2) {
                // Colorimetry Data Block (sec.7.5.6): payload[1] = colorimetry bitmap.
                caps.sinkBt2020Ycc = caps.sinkBt2020Ycc || (payload[1] & EDID_COLORIMETRY_BT2020_YCC) != 0;
            }
        }
        offset += 1 + payloadLen;
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
}

/// Walk every CTA-861 extension block in a raw EDID blob and OR the sink HDR bits into @p caps.
auto ParseEdidHdrCaps(std::span<const uint8_t> edid, DisplayCaps &caps) -> void {
    if (edid.size() < EDID_BLOCK_SIZE) {
        return;
    }
    // Bounded by edid.size() >= EDID_BLOCK_SIZE (128) check above; the offset is < 128.
    const size_t extCount =
        edid[EDID_EXTENSION_COUNT_OFFSET]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
    for (size_t i = 0; i < extCount; ++i) {
        const size_t extOffset = EDID_BLOCK_SIZE * (i + 1);
        if (extOffset + EDID_BLOCK_SIZE > edid.size()) {
            break;
        }
        ParseCtaExtension(edid.subspan(extOffset, EDID_BLOCK_SIZE), caps);
    }
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

AtomicRequest::AtomicRequest(AtomicRequest &&other) noexcept : propCount(other.propCount), request(other.request) {
    other.request = nullptr; // prevents double-free in moved-from destructor
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
    // propId==0 means the driver doesn't expose this optional property (e.g. zpos, blend mode,
    // COLOR_RANGE on older kernels). Skipping silently avoids per-driver branches at every caller.
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
    other.drmFd = -1; // prevents double-release in moved-from destructor
    other.fbId = 0;
    other.gemHandle = 0;
    other.frame = nullptr;
}

cVaapiDisplay::DrmFramebuffer::~DrmFramebuffer() noexcept {
    // Release order matters: (1) AVFrame drops the VA surface ref that backs the DMA-BUF;
    // (2) drmModeRmFB tells the CRTC to stop scanning and releases the kernel DMA-BUF ref;
    // (3) DRM_IOCTL_GEM_CLOSE frees the imported BO. Reversing (1)/(2) causes the kernel to
    // read freed GPU memory on the next scanout.
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
    // Only page_flip_handler (v1) is wired; other slots are null so libdrm doesn't dispatch
    // to stale pointers on unexpected event types (vblank, sequence, page_flip2).
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
    // Called by cVaapiOsd::~cVaapiOsd before the underlying dumb buffer is freed: the destructor
    // must not return while the KMS plane still references fbId, or the kernel scanout will read
    // freed memory.
    if (fbId == 0 || !ready.load(std::memory_order_relaxed)) {
        return;
    }

    // Poll until PresentBuffer() clears osdDirty AND advances currentOsd past this fbId.
    // 100 ms is generous: a page flip at 25 Hz takes 40 ms; the OSD is committed on the
    // next video frame, which must arrive before the VDR OSD destructor returns.
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
    // Order matters: gate consumer, drop queue (unblock submitters), drain flip BEFORE
    // taking importMutex (reversed = deadlock), then hold importMutex for codec teardown.
    // displayedBuffer/pendingBuffer stay alive so the last frame remains on screen.
    isClearing.store(true, std::memory_order_release);
    {
        const cMutexLock lock(&bufferMutex);
        pendingFrames.clear();
        pendingDepth.store(0, std::memory_order_release);
        frameSlotCond.Broadcast();
    }
    WaitForPageFlip(DISPLAY_PAGE_FLIP_TIMEOUT_MS);
    importMutex.Lock();
    // Reset under importMutex so an in-flight fresh commit cannot republish a pre-Clear timestamp.
    lastFrameCommitMs.store(0, std::memory_order_release);
}

auto cVaapiDisplay::EndStreamSwitch() -> void {
    // Unlock first, then clear isClearing -- consumer re-checks isClearing under importMutex
    // so it can't observe a stale "false" mid-teardown.
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
    videoRect = cRect(0, 0, static_cast<int>(outputWidth), static_cast<int>(outputHeight));
    // vrefresh==0 occurs for non-CEA modes on some EDIDs. 50 Hz is the DVB baseline and
    // must match decoder.cpp's framerate fallback in InitFilterGraph() -- the two values
    // are coupled; changing one without the other desyncs the A/V controllers.
    refreshRate = displayMode.vrefresh > 0 ? displayMode.vrefresh : 50;
    aspectRatio = static_cast<double>(outputWidth) / static_cast<double>(outputHeight);

    hwDeviceRef = av_buffer_ref(hwDevice);
    if (!hwDeviceRef) {
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

[[nodiscard]] auto cVaapiDisplay::GetActiveHdrKind() const noexcept -> StreamHdrKind {
    const cMutexLock lock(&hdrStateMutex);
    return appliedHdrState.kind;
}

[[nodiscard]] auto cVaapiDisplay::GetActiveOsdFbId() const noexcept -> uint32_t {
    const cMutexLock lock(&osdMutex);
    return currentOsd.fbId;
}

[[nodiscard]] auto cVaapiDisplay::GrabDisplayedFrame() -> std::unique_ptr<AVFrame, FreeAVFrame> {
    if (!ready.load(std::memory_order_acquire)) [[unlikely]] {
        return nullptr;
    }

    // Clone under bufferMutex so the displayed AVFrame's VA-surface ref stays alive past unlock;
    // hold the mutex only across the cheap ref bump, never across the GPU download below.
    std::unique_ptr<AVFrame, FreeAVFrame> source;
    {
        const cMutexLock lock(&bufferMutex);
        if (!displayedBuffer.frame) [[unlikely]] {
            return nullptr;
        }
        source.reset(av_frame_clone(displayedBuffer.frame));
    }
    if (!source) [[unlikely]] {
        return nullptr;
    }

    std::unique_ptr<AVFrame, FreeAVFrame> dest{av_frame_alloc()};
    if (!dest) [[unlikely]] {
        return nullptr;
    }

    // Serialize with VPP: iHD's VEBOX deadlocks if download races with concurrent filter execution.
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

    // Always mark dirty, even for an unchanged (fbId, geometry) pair. VDR may repaint
    // into the same dumb buffer in-place; on Intel/AMD, FBC/PSR tile caches are only
    // invalidated when the plane is touched by an atomic commit. Without this, stale
    // compressed pixels remain on screen.
    currentOsd = osd;
    osdDirty = true;
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
        esyslog("vaapivideo/display: thread did not exit - may cause resource leak");
    }

    // Drain residual page-flip events: without this the kernel keeps them pending on the fd
    // and the next process to open the DRM device inherits stale events.
    for (int i = 0; i < DISPLAY_MAX_DRAIN_ITERATIONS && DrainDrmEvents(0); ++i) {
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
    if (!frame || !ready.load(std::memory_order_relaxed) || isClearing.load(std::memory_order_relaxed)) [[unlikely]] {
        return false;
    }

    const cMutexLock lock(&bufferMutex);

    // VSync-paced backpressure: block when the prerender queue is full.
    if (pendingFrames.size() >= DISPLAY_PRERENDER_SLOTS) {
        if (timeoutMs == 0) {
            return false;
        }

        // Slice infinite waits into 1 s windows so a stream switch isn't blocked indefinitely.
        const int waitMs = (timeoutMs < 0) ? 1000 : timeoutMs;
        const cTimeMs deadline(waitMs);
        while (pendingFrames.size() >= DISPLAY_PRERENDER_SLOTS && ready.load(std::memory_order_relaxed) &&
               !deadline.TimedOut()) {
            if (isClearing.load(std::memory_order_relaxed)) {
                return false;
            }
            frameSlotCond.TimedWait(bufferMutex, 10);
        }

        if (pendingFrames.size() >= DISPLAY_PRERENDER_SLOTS || isClearing.load(std::memory_order_relaxed))
            [[unlikely]] {
            return false;
        }
    }

    pendingFrames.push_back(std::move(frame));
    pendingDepth.store(pendingFrames.size(), std::memory_order_release);
    return true;
}

// ============================================================================
// === THREAD ===
// ============================================================================

auto cVaapiDisplay::Action() -> void {
    dsyslog("vaapivideo/display: thread started (thread=%lu)", (unsigned long)pthread_self());

    // Queue-underrun tracker: measures the wall-clock duration of consecutive empty VSyncs
    // during active playback. Warmup grace suppresses spurious counts after Clear/cold-start
    // while the filter graph + audio anchor.
    //
    // Duration is taken directly from (nowMs - lastCommitMs) -- a wall-clock delta -- rather
    // than a VSync count multiplied by a nominal vsyncMs. The latter lies when the consumer
    // loop is preempted or page_flip events arrive late: a 4 s real-time stall could be
    // mis-reported as 25 vsyncs * 20 ms (=500 ms) or, with a stale-reset bug, as 238 * 20 ms.
    // Wall-clock makes the printed value true by construction.
    //
    // UNDERRUN_THRESHOLD_VSYNCS = SLOTS + 2 is tightly coupled to DISPLAY_PRERENDER_SLOTS:
    //   SLOTS    -> one missed VSync inside a freshly-drained queue (queue absorbs it, no log)
    //   + 1      -> grace VSync for the decoder to catch up (single hiccup, no log)
    //   + 1      -> one more sample so the trigger fires on sustained gaps, not transients
    // Revisit the +2 margin if you change DISPLAY_PRERENDER_SLOTS -- the relationship doesn't
    // scale linearly: at slots=1 you'd want +3 (more noise), at slots=8 +2 is plenty.
    constexpr auto UNDERRUN_THRESHOLD_VSYNCS = static_cast<unsigned>(DISPLAY_PRERENDER_SLOTS + 2);
    constexpr int ACTIVE_WINDOW_MS = 500;
    constexpr int UNDERRUN_LOG_COOLDOWN_MS = 2000;
    constexpr int WARMUP_GRACE_MS = 3000;
    // Beyond this the gap is no longer treated as an underrun: a stream-paused / radio-mode /
    // suspended-stream scenario can leave lastCommitMs hours in the past, and the recovery log
    // on resume would otherwise scream a multi-hour "queue refilled" event. inTrick / inSyncSleep
    // catch the explicit pause paths; this is the catch-all for everything else.
    constexpr uint64_t IDLE_THRESHOLD_MS = 10000;
    // refreshRate is set once in EndStreamSwitch() before Action() starts and stays constant for
    // the consumer-thread lifetime, so vsyncMs / thresholdMs are loop invariants -- hoisted.
    const auto vsyncMs = refreshRate > 0 ? 1000U / refreshRate : 20U;
    const uint64_t thresholdMs = UNDERRUN_THRESHOLD_VSYNCS * vsyncMs;
    uint64_t gapStartMs = 0;      ///< Wall-clock baseline for the current gap; 0 = "anchor on next re-present".
                                  ///< Reset on commit / isClearing / inTrick / inSyncSleep so a deliberate sleep
                                  ///< or trick-pace exit does not surface its duration as a fake underrun.
    uint64_t peakGapMs = 0;       ///< Wall-clock peak duration of the current gap; reset on commit.
    unsigned emptyVSyncTotal = 0; ///< Cumulative empty-VSync count since the display thread started.
    // cTimeMs(0) constructs a timer that's already timed out (begin == end == Now()), so
    // TimedOut() returns true immediately. underrunLogCooldown: first log fires without delay.
    // warmupGraceUntil: no grace active until the first fresh commit arms it via the
    // prevCommitMs==0 branch below; before that the IsValid() gate on pendingBuffer prevents
    // the underrun counter from running, so "no grace at construction" is the safe state.
    cTimeMs underrunLogCooldown(0);
    cTimeMs warmupGraceUntil(0);

    while (!stopping.load(std::memory_order_relaxed) && ready.load(std::memory_order_relaxed)) {
        // Non-blocking drain so isFlipPending is up-to-date before the gate check below.
        while (DrainDrmEvents(0)) {
        }

        // VSync gate: don't queue a new flip until the previous one's event has arrived.
        // 5 ms poll avoids busy-spinning; page_flip_handler latency is the real bottleneck.
        if (isFlipPending.load(std::memory_order_relaxed)) {
            (void)DrainDrmEvents(5);
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

        // Trick play: decoder commits at the trick hold (60-2000 ms), so most VSyncs re-present.
        // That's not an underrun; clear the counters so trick-exit doesn't carry a stale peak
        // OR a stale gapStartMs (which would otherwise log the trick hold itself as an underrun).
        // Sync sleep (hard-ahead / soft-ahead): decoder is deliberately paused inside a correction
        // sleep, so re-presents during that window are intentional, not underruns. Same reset --
        // gapStartMs=0 means the post-sleep re-present re-anchors instead of reporting the
        // sleep duration as a gap.
        const bool inTrick = trickActive.load(std::memory_order_relaxed);
        const bool inSyncSleep = syncSleeping.load(std::memory_order_relaxed);
        if (inTrick || inSyncSleep) {
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
                            // (lastCommit==0) OR after a long gap (commit > ACTIVE_WINDOW_MS old).
                            // The latter covers transitions where BeginStreamSwitch wasn't invoked
                            // (track switch, post-trick re-anchor). Done under importMutex so a
                            // BeginStreamSwitch racing this commit cannot republish a stale value.
                            const uint64_t prevCommitMs = lastFrameCommitMs.load(std::memory_order_relaxed);
                            const uint64_t nowMs = cTimeMs::Now();
                            if (prevCommitMs == 0 || nowMs - prevCommitMs > ACTIVE_WINDOW_MS) {
                                warmupGraceUntil.Set(WARMUP_GRACE_MS);
                            }
                            lastFrameCommitMs.store(nowMs, std::memory_order_release);
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
            const cMutexLock lock(&bufferMutex);
            if (pendingBuffer.IsValid()) {
                (void)PresentBuffer(pendingBuffer);
                // Only count an underrun when decoder was previously active AND past warmup
                // grace. Gap duration is measured from gapStartMs (NOT lastFrameCommitMs):
                // anchoring on the first re-present in a streak means a deliberate hard-ahead
                // sleep, trick hold or stream-switch does not surface its own duration as a fake
                // underrun the moment it ends. Beyond IDLE_THRESHOLD_MS we stop accumulating so
                // a paused stream doesn't grow peakGapMs without bound.
                const uint64_t nowMs = cTimeMs::Now();
                const uint64_t lastCommitMs = lastFrameCommitMs.load(std::memory_order_acquire);
                if (!inTrick && !inSyncSleep && lastCommitMs != 0 && warmupGraceUntil.TimedOut()) {
                    if (gapStartMs == 0) {
                        gapStartMs = nowMs; // anchor: first re-present of the current gap
                    }
                    const uint64_t currentGapMs = nowMs - gapStartMs;
                    if (currentGapMs < IDLE_THRESHOLD_MS) {
                        ++emptyVSyncTotal;
                        peakGapMs = std::max(peakGapMs, currentGapMs);
                        if (currentGapMs >= thresholdMs && underrunLogCooldown.TimedOut()) {
                            dsyslog("vaapivideo/display: queue empty %llums; total=%u",
                                    static_cast<unsigned long long>(currentGapMs), emptyVSyncTotal);
                            underrunLogCooldown.Set(UNDERRUN_LOG_COOLDOWN_MS);
                        }
                    } else {
                        // Gap exceeded IDLE_THRESHOLD_MS: treat as paused / stopped, not as an
                        // underrun. Clear peak so the recovery log stays silent on resume;
                        // gapStartMs stays put so we don't re-anchor and start a fresh accounting
                        // window every iteration during a long pause.
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

    // KMS rejects a commit where CRTC_X+CRTC_W > CRTC width (similarly for Y).
    // Clip here so the entire atomic commit doesn't fail over a slightly oversized OSD.
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

    // VDR stores straight (non-premultiplied) ARGB via tColor. "Coverage" blending (enum 1)
    // applies alpha correctly; "Pre-multiplied" (enum 0) would double-apply it and produce
    // darker edges with ringing on translucent backgrounds.
    if (osdProps.pixelBlendMode != 0) {
        req.AddProperty(osdPlaneId, osdProps.pixelBlendMode, 1);
    }
    // zpos is NOT written: on tested i915 it causes the commit to fail (the property is
    // immutable on overlay planes). On amdgpu the driver's default z-order already places
    // the OSD above the video plane. Empirical -- re-verify on new hardware before adding.
}

[[nodiscard]] auto cVaapiDisplay::AtomicCommit(AtomicRequest &req, uint32_t flags) -> bool {
    if (req.Count() == 0) {
        return true; // empty commit -- nothing to do, treat as success
    }

    // Page-flip: NONBLOCK + PAGE_FLIP_EVENT -- kernel queues the work and delivers a
    // completion event that Action() drains via DrainDrmEvents().
    // Modeset: synchronous and event-less -- mode programming must complete before the
    // next page-flip is submitted.
    // DRM_MODE_ATOMIC_ASYNC is intentionally not used: it requires linear (untiled) buffers,
    // but VAAPI surfaces are always tiled (Y/X-tile or CCS compressed); the kernel rejects
    // async commits on tiled BOs with EINVAL.
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

    // EBUSY: previous flip event not yet consumed; Action() retries after DrainDrmEvents().
    // Not worth logging -- it's expected on the hot path when the consumer runs slightly ahead.
    if (errno != EBUSY) {
        esyslog("vaapivideo/display: atomic commit failed - %s (flags=0x%x)", std::strerror(errno), commitFlags);
    }
    return false;
}

[[nodiscard]] auto cVaapiDisplay::ApplyDisplayMode(const drmModeModeInfo &mode) -> bool {
    dsyslog("vaapivideo/display: setting display mode %ux%u@%uHz", mode.hdisplay, mode.vdisplay, mode.vrefresh);
    // KMS stores the mode as a property blob. Destroy the old one explicitly -- libdrm has no GC.
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
        esyslog("vaapivideo/display: failed to set mode");
        return false;
    }

    activeMode = mode;
    // Mark applied state as Sdr so MaybeAppendHdrOutputState() skips the first frame's
    // HDR write (staged==applied), keeping subsequent page flips in the non-ALLOW_MODESET
    // fast path and preventing spurious AVR retrains during IEC61937 lock-in.
    appliedHdrState = HdrStreamInfo{};
    appliedHdrBlobId = 0;
    return true;
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

    dsyslog("vaapivideo/display: searching for plane %d (format 0x%08x)", planeIndex, format);

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
                    if (blob && blob->data) {
                        const auto *modBlob = static_cast<const drm_format_modifier_blob *>(blob->data);
                        const auto *base = static_cast<const uint8_t *>(blob->data);
                        // drm_format_modifier_blob: formats_offset is a byte offset into the same
                        // buffer where the uint32_t format[] array begins (DRM ABI, not a pointer).
                        const auto *formats =
                            reinterpret_cast<const uint32_t *>( // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
                                base + modBlob->formats_offset);

                        for (uint32_t k = 0; k < modBlob->count_formats; ++k) {
                            if (formats[k] == format) {
                                hasFormatSupport = true;
                            } else if (formats[k] == DRM_FORMAT_P010) {
                                tempProps.supportsP010 = true;
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

        dsyslog("vaapivideo/display: candidate plane %u type=%s (found=%d, need=%d)", plane->plane_id,
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
    // Single-consumer in steady state. Shutdown() also calls this from the main thread,
    // but only after hasExited is set, so there is never concurrent access.
    pollfd pfd{.fd = drmFd, .events = POLLIN, .revents = 0};
    const int ret = poll(&pfd, 1, timeoutMs);

    if (ret > 0 && (pfd.revents & POLLIN)) {
        return drmHandleEvent(drmFd, &eventContext) == 0;
    }
    return false;
}

[[nodiscard]] auto cVaapiDisplay::LoadDrmProperties() -> bool {
    // UNIVERSAL_PLANES: exposes overlay and cursor planes (default: only primary/cursor).
    // ATOMIC: switches the fd to the atomic modesetting uAPI used everywhere below.
    // Both are per-fd opt-ins; no-ops on already-set caps.
    (void)drmSetClientCap(drmFd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1);
    (void)drmSetClientCap(drmFd, DRM_CLIENT_CAP_ATOMIC, 1);

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
        ParseEdidHdrCaps(std::span<const uint8_t>{edidBlob}, displayCaps);
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

/// Compare two 2-element AVRational arrays (chromaticity XY). Wraps the constant-index
/// accesses to a C array (FFmpeg ABI type) in a single scope, so call sites stay NOLINT-free.
[[nodiscard]] auto XyRationalEqual(const AVRational (&a)[2], const AVRational (&b)[2]) noexcept -> bool {
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
    return AvRationalEqual(a[0], b[0]) && AvRationalEqual(a[1], b[1]);
    // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
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
                // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index) -- FFmpeg ABI fixed [3][2] array
                m.display_primaries[i].x = EncodePrimary(info.mastering.display_primaries[i][0]);
                m.display_primaries[i].y = EncodePrimary(info.mastering.display_primaries[i][1]);
                // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
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
        constexpr unsigned int LIGHT_MAX = 0xFFFFU;
        m.max_cll = static_cast<uint16_t>(std::min(info.contentLight.MaxCLL, LIGHT_MAX));
        m.max_fall = static_cast<uint16_t>(std::min(info.contentLight.MaxFALL, LIGHT_MAX));
    }
    // NOLINTEND(cppcoreguidelines-pro-type-union-access)
    return meta;
}

} // namespace

[[nodiscard]] auto cVaapiDisplay::MaybeAppendHdrOutputState(AtomicRequest &req, bool &failed) -> bool {
    failed = false;
    // Held across staged-read and appliedHdrState write so GetActiveHdrKind() never sees a torn snapshot.
    const cMutexLock lock(&hdrStateMutex);
    const HdrStreamInfo staged = stagedHdrState;
    if (HdrOutputStateEqual(staged, appliedHdrState)) {
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
    // Blob IDs (uint32_t) are display-thread-only, no SVDRP reader, so the lock above is
    // strictly for appliedHdrState.
    pendingDestroyHdrBlobId = appliedHdrBlobId;
    appliedHdrBlobId = newBlobId;
    appliedHdrState = staged;
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

    // FFmpeg ABI: an AV_PIX_FMT_DRM_PRIME frame's data[0] is the AVDRMFrameDescriptor pointer.
    const auto *desc =
        reinterpret_cast<const AVDRMFrameDescriptor *>( // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
            mappedFrame->data[0]);
    // The rest of this function assumes a single DMA-BUF object holding both NV12 layers
    // (Y + UV at different offsets). On the iHD/Mesa stacks tested this is always the
    // shape returned; on hypothetical drivers that split planes across multiple objects
    // a per-object GEM import + multi-fd AddFB2 path would be required. Reject early so the
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
    // layer[0].format is NOT reliable: iHD 25.x has been observed to report a fourcc that
    // KMS rejects in combination with the exported modifier, producing spurious AddFB2
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
    fb.frame = vaapiFrame->avFrame;
    vaapiFrame->avFrame = nullptr;
    vaapiFrame->ownsFrame = false;

    av_frame_free(&mappedFrame);

    return fb;
}

auto cVaapiDisplay::OnPageFlipEvent([[maybe_unused]] int fd, [[maybe_unused]] unsigned int seq,
                                    [[maybe_unused]] unsigned int sec, [[maybe_unused]] unsigned int usec, void *data)
    -> void {
    // libdrm dispatches this from drmHandleEvent() on the consumer thread (drmHandleEvent is
    // never called from anywhere else). The `data` cookie is the `this` pointer passed to
    // drmModeAtomicCommit. Release-store on isFlipPending makes the consumer's next
    // acquire-load see "flip done" and submit the next frame.
    auto *display = static_cast<cVaapiDisplay *>(data);
    if (display) {
        display->lastVSyncTimeMs.store(cTimeMs::Now(), std::memory_order_release);
        display->isFlipPending.store(false, std::memory_order_release);
    }
}

[[nodiscard]] auto cVaapiDisplay::GetVideoRectHeight() const -> uint32_t {
    const cMutexLock lock(&videoRectMutex);
    return static_cast<uint32_t>(videoRect.Height());
}

[[nodiscard]] auto cVaapiDisplay::GetVideoRectWidth() const -> uint32_t {
    const cMutexLock lock(&videoRectMutex);
    return static_cast<uint32_t>(videoRect.Width());
}

[[nodiscard]] auto cVaapiDisplay::SetVideoRect(const cRect &rect) -> bool {
    const cMutexLock lock(&videoRectMutex);
    cRect newRect;
    if (rect.IsEmpty()) {
        newRect = cRect(0, 0, static_cast<int>(outputWidth), static_cast<int>(outputHeight));
    } else {
        // NV12 (4:2:0) requires plane coordinates to be 2-pixel aligned on Intel DRM.
        newRect = cRect(rect.X() & ~1, rect.Y() & ~1, std::max(2, rect.Width() & ~1), std::max(2, rect.Height() & ~1));
    }
    const bool dimsChanged = (newRect.Width() != videoRect.Width() || newRect.Height() != videoRect.Height());
    dsyslog("vaapivideo/display: SetVideoRect x=%d y=%d w=%d h=%d%s", newRect.X(), newRect.Y(), newRect.Width(),
            newRect.Height(), dimsChanged ? " (rebuild)" : "");
    videoRect = newRect;
    return dimsChanged;
}

[[nodiscard]] auto cVaapiDisplay::PresentBuffer(const DrmFramebuffer &fb) -> bool {
    if (!fb.IsValid()) {
        return false;
    }

    // Snapshot the current video rect. scale_vaapi produces a framebuffer exactly sized to this
    // rect, so CRTC_W/H == SRC_W/H -- no DRM plane scaler. CRTC_X/Y positions the plane.
    cRect vr;
    {
        const cMutexLock lock(&videoRectMutex);
        vr = videoRect;
    }
    // Center scale_vaapi output within the rect (handles brief transition before filter rebuild).
    const int32_t destX =
        vr.X() + (static_cast<int>(fb.width) < vr.Width() ? (vr.Width() - static_cast<int>(fb.width)) / 2 : 0);
    const int32_t destY =
        vr.Y() + (static_cast<int>(fb.height) < vr.Height() ? (vr.Height() - static_cast<int>(fb.height)) / 2 : 0);

    AtomicRequest req;
    req.AddProperty(videoPlaneId, videoProps.crtcId, crtcId);
    req.AddProperty(videoPlaneId, videoProps.fbId, fb.fbId);

    // HDR connector signaling is written only on state transitions, and must precede the
    // per-frame COLOR_ENCODING below so the kernel sees a coherent HDR picture in one atomic.
    // Snapshot the previous applied state for rollback: MaybeAppendHdrOutputState updates
    // appliedHdrState/appliedHdrBlobId optimistically even though the kernel only observes
    // them once this commit lands.
    const HdrStreamInfo previousHdrState = appliedHdrState;
    const uint32_t previousHdrBlobId = appliedHdrBlobId;
    bool hdrStateFailed = false;
    const bool hdrStateChanged = MaybeAppendHdrOutputState(req, hdrStateFailed);
    if (hdrStateFailed) [[unlikely]] {
        return false;
    }

    // COLOR_ENCODING follows the applied HDR state. Forcing the enum explicitly matters
    // because driver defaults (i915 BT.601 full / AMD BT.709 full) mismatch scale_vaapi's
    // limited-range output. An HDR-active state with no BT.2020 enum is refused upstream
    // by CanDriveHdrPlane(); intentionally no BT.709 fallback here (would produce a green cast).
    if (appliedHdrState.kind != StreamHdrKind::Sdr) {
        if (videoProps.colorEncodingBt2020Valid) {
            req.AddProperty(videoPlaneId, videoProps.colorEncoding, videoProps.colorEncodingBt2020);
        }
    } else if (videoProps.colorEncodingValid) {
        req.AddProperty(videoPlaneId, videoProps.colorEncoding, videoProps.colorEncodingBt709);
    }
    if (videoProps.colorRangeValid) {
        req.AddProperty(videoPlaneId, videoProps.colorRange, videoProps.colorRangeLimited);
    }
    // Clamp source + dest together to display bounds (1:1 crop, no DRM scaler).
    // Also enforce 2-pixel alignment: NV12 (4:2:0) requires even CRTC_X/Y on Intel DRM.
    const auto safeDestX = static_cast<uint32_t>(std::max(0, destX) & ~1);
    const auto safeDestY = static_cast<uint32_t>(std::max(0, destY) & ~1);
    const uint32_t planeW = std::min(fb.width, safeDestX < outputWidth ? outputWidth - safeDestX : 0U) & ~1U;
    const uint32_t planeH = std::min(fb.height, safeDestY < outputHeight ? outputHeight - safeDestY : 0U) & ~1U;
    if (planeW == 0 || planeH == 0) [[unlikely]] {
        return false;
    }
    req.AddProperty(videoPlaneId, videoProps.srcX, 0);
    req.AddProperty(videoPlaneId, videoProps.srcY, 0);
    req.AddProperty(videoPlaneId, videoProps.srcW, static_cast<uint64_t>(planeW) << 16);
    req.AddProperty(videoPlaneId, videoProps.srcH, static_cast<uint64_t>(planeH) << 16);
    req.AddProperty(videoPlaneId, videoProps.crtcX, static_cast<uint64_t>(safeDestX));
    req.AddProperty(videoPlaneId, videoProps.crtcY, static_cast<uint64_t>(safeDestY));
    req.AddProperty(videoPlaneId, videoProps.crtcW, planeW);
    req.AddProperty(videoPlaneId, videoProps.crtcH, planeH);

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

    // ALLOW_MODESET is required on AMDGPU (and i915 on some kernels) whenever
    // HDR_OUTPUT_METADATA / Colorspace / max bpc changes -- those can force a link retrain that a
    // "pure page-flip" commit is not permitted to trigger. The brief blackout is acceptable
    // because HDR transitions only happen at channel-switch time, never per-frame.
    const uint32_t commitFlags = hdrStateChanged ? DRM_MODE_ATOMIC_ALLOW_MODESET : 0U;
    const bool success = AtomicCommit(req, commitFlags);

    // Only clear osdDirty when the commit actually landed -- a failed commit (e.g. EBUSY)
    // must keep the OSD update queued for the next attempt.
    if (success && osdCommitted) {
        const cMutexLock lock(&osdMutex);
        osdDirty = false;
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

auto cVaapiDisplay::WaitForPageFlip(int timeoutMs) -> void {
    // Called from BeginStreamSwitch() on the main thread BEFORE taking importMutex (the
    // ordering rule documented at the top of this file). DrainDrmEvents() here works
    // because the consumer thread runs poll() too -- only one caller actually sees a
    // given event, and isFlipPending's release/acquire serializes the result.
    const cTimeMs deadline(timeoutMs);
    while (isFlipPending.load(std::memory_order_relaxed) && !deadline.TimedOut()) {
        // Bail on shutdown / stream-switch so callers don't sit through the full timeout
        // when the answer is "tearing down anyway".
        if (stopping.load(std::memory_order_relaxed) || !ready.load(std::memory_order_relaxed) ||
            isClearing.load(std::memory_order_relaxed)) {
            break;
        }
        (void)DrainDrmEvents(5);
    }
}
