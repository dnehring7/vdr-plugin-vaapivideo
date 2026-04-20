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

namespace {

// CTA-861-G decode constants. EOTF bits are in HDR SMDB byte 2; BT.2020 flags in
// Colorimetry Data Block byte 2. Bit 0 (SDR) / bit 1 (HDR gamma) are implicit.
constexpr uint8_t EDID_EOTF_PQ = 1U << 2;
constexpr uint8_t EDID_EOTF_HLG = 1U << 3;
constexpr uint8_t EDID_COLORIMETRY_BT2020_YCC = 1U << 6;
constexpr size_t EDID_BLOCK_SIZE = 128;
constexpr size_t EDID_CHECKSUM_OFFSET = 127;
constexpr size_t EDID_EXTENSION_COUNT_OFFSET = 126;

/// Parse one 128-byte CTA-861 extension block for HDR EOTF + BT.2020 YCC advertisement.
/// Per CTA-861-G section 7.3, a DTD offset of 0 means "no DTDs"; data blocks then fill [4, 127)
/// (byte 127 is the checksum). Malformed blocks are tolerated silently.
auto ParseCtaExtension(std::span<const uint8_t> ext, cVaapiDisplay::SinkHdrCaps &caps) -> void {
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
        // Tag 7 = "Use Extended Tag Code"; payload[0] carries the extended tag.
        if (blockTag == 7 && !payload.empty()) {
            const uint8_t extTag = payload[0];
            if (extTag == 0x06 && payload.size() >= 3) {
                // HDR Static Metadata Data Block. payload[1] = supported EOTFs bitmap.
                // OR across extension blocks -- a second block must not erase support
                // discovered in an earlier one.
                caps.hdr10 = caps.hdr10 || (payload[1] & EDID_EOTF_PQ) != 0;
                caps.hlg = caps.hlg || (payload[1] & EDID_EOTF_HLG) != 0;
            } else if (extTag == 0x05 && payload.size() >= 2) {
                // Colorimetry Data Block. payload[1] carries the BT.2020 flags. Same
                // accumulation rule as the HDR SMDB above.
                caps.bt2020Ycc = caps.bt2020Ycc || (payload[1] & EDID_COLORIMETRY_BT2020_YCC) != 0;
            }
        }
        offset += 1 + payloadLen;
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
}

/// Walk every CTA-861 extension block in a raw EDID blob and OR the results into `caps`.
auto ParseEdidHdrCaps(std::span<const uint8_t> edid) -> cVaapiDisplay::SinkHdrCaps {
    cVaapiDisplay::SinkHdrCaps caps{};
    if (edid.size() < EDID_BLOCK_SIZE) {
        return caps;
    }
    const size_t extCount =
        edid[EDID_EXTENSION_COUNT_OFFSET]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
    for (size_t i = 0; i < extCount; ++i) {
        const size_t extOffset = EDID_BLOCK_SIZE * (i + 1);
        if (extOffset + EDID_BLOCK_SIZE > edid.size()) {
            break;
        }
        ParseCtaExtension(edid.subspan(extOffset, EDID_BLOCK_SIZE), caps);
    }
    return caps;
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

    // HDR capability probe runs AFTER the video plane is bound -- planeSupportsP010 needs
    // videoPlaneId to query the plane's IN_FORMATS blob. The probe is best-effort: logs
    // the discovered sink + connector capabilities so operators can see why HDR is (or
    // isn't) available, but never blocks SDR output.
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
        // Reset connector HDR properties to SDR defaults in the same atomic that disables
        // the CRTC, otherwise the next DRM client inherits our BT.2020 / 10 bpc settings.
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
    // HDR property blobs: same rationale as modeBlobId -- the CRTC has already been
    // disabled, so the kernel is no longer referencing either of these. Not freeing them
    // here would leak kernel memory for the plugin-unload duration.
    if (appliedHdrBlobId != 0 && drmFd >= 0) {
        if (drmModeDestroyPropertyBlob(drmFd, appliedHdrBlobId) != 0) [[unlikely]] {
            esyslog("vaapivideo/display: failed to destroy applied HDR blob: %s", strerror(errno));
        }
        appliedHdrBlobId = 0;
    }
    if (pendingDestroyHdrBlobId != 0 && drmFd >= 0) {
        if (drmModeDestroyPropertyBlob(drmFd, pendingDestroyHdrBlobId) != 0) [[unlikely]] {
            esyslog("vaapivideo/display: failed to destroy pending HDR blob: %s", strerror(errno));
        }
        pendingDestroyHdrBlobId = 0;
    }
    // Reset tracked HDR state so a re-Initialize() starts from a known SDR baseline.
    appliedHdrState = {};
    stagedHdrState = {};
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
    // Fold the SDR baseline for HDR connector properties into the modeset atomic so no
    // second ALLOW_MODESET commit is needed later to clear state left by a previous DRM
    // client. Doing it here keeps the single HDMI link retrain that ApplyDisplayMode
    // already forces -- a second retrain after the audio sink has locked onto an IEC61937
    // bitstream causes the sink to fall out of passthrough and treat subsequent payload
    // as raw PCM (i.e. noise).
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
    // SDR baseline is now programmed; subsequent page flips suppress the HDR property
    // write while staged == applied (both Sdr), so the PresentBuffer() path runs without
    // ALLOW_MODESET and the AVR never sees a stray retrain while holding an IEC61937 lock.
    appliedHdrState = HdrStreamInfo{};
    appliedHdrBlobId = 0;
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

    // Video-plane discovery (first NV12 call) prefers a plane that also supports the HDR
    // triad (P010 + BT.2020 + BT.709 COLOR_ENCODING). Some GPUs expose the first NV12
    // plane as SDR-only and put P010 on a later one; falling for the first-match would
    // disable HDR unnecessarily. We keep the first SDR-only candidate as a fallback for
    // the case where no HDR-capable plane exists.
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

            // IN_FORMATS blob lists (format, modifier) pairs the plane accepts. We confirm
            // the requested format and, in the same sweep, record whether the plane can
            // also scan out DRM_FORMAT_P010 -- the HDR passthrough path needs that and
            // otherwise we'd re-iterate the whole blob from ProbeHdrCapabilities just to
            // answer the same question. The (format, modifier) pair is validated at commit
            // time by KMS; scanning for the fourcc alone is sufficient here.
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
                            } else if (formats[k] == DRM_FORMAT_P010) {
                                tempProps.supportsP010 = true;
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
                    const char *enumName = prop->enums[e].name;
                    if (strcmp(enumName, "ITU-R BT.709 YCbCr") == 0) {
                        tempProps.colorEncodingBt709 = prop->enums[e].value;
                        tempProps.colorEncodingValid = true;
                    } else if (strcmp(enumName, "ITU-R BT.2020 YCbCr") == 0) {
                        tempProps.colorEncodingBt2020 = prop->enums[e].value;
                        tempProps.colorEncodingBt2020Valid = true;
                    }
                }
                dsyslog("vaapivideo/display: plane %u COLOR_ENCODING prop=%u bt709=%lu(%s) bt2020=%lu(%s)",
                        plane->plane_id, tempProps.colorEncoding, (unsigned long)tempProps.colorEncodingBt709,
                        tempProps.colorEncodingValid ? "yes" : "no", (unsigned long)tempProps.colorEncodingBt2020,
                        tempProps.colorEncodingBt2020Valid ? "yes" : "no");
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

auto cVaapiDisplay::ProbeHdrCapabilities() -> void {
    // Optional connector properties: a missing id just disables HDR, never breaks SDR.
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
            if (strcmp(name, "HDR_OUTPUT_METADATA") == 0) {
                hdrProps.hdrOutputMetadata = prop->prop_id;
            } else if (strcmp(name, "Colorspace") == 0) {
                hdrProps.colorspace = prop->prop_id;
                for (int e = 0; e < prop->count_enums; ++e) {
                    if (strcmp(prop->enums[e].name, "BT2020_YCC") == 0) {
                        hdrProps.colorspaceBt2020Ycc = prop->enums[e].value;
                        haveBt2020Ycc = true;
                    } else if (strcmp(prop->enums[e].name, "Default") == 0) {
                        hdrProps.colorspaceDefault = prop->enums[e].value;
                        haveDefault = true;
                    }
                }
                // Refuse drivers that map BT2020_YCC and Default to the same numeric value --
                // SDR and HDR commits would be indistinguishable.
                hdrProps.colorspaceValid =
                    haveBt2020Ycc && haveDefault && hdrProps.colorspaceBt2020Ycc != hdrProps.colorspaceDefault;
            } else if (strcmp(name, "max bpc") == 0 && prop->count_values >= 2) {
                // Range property: [min, max]. Skipping entries without both bounds keeps the
                // commit path from requesting clamp(10, 0, 0) == 0 bpc, which the kernel rejects.
                hdrProps.maxBpc = prop->prop_id;
                hdrProps.maxBpcMin = prop->values[0];
                hdrProps.maxBpcMax = prop->values[1];
            } else if (strcmp(name, "EDID") == 0) {
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
        sinkHdrCaps = ParseEdidHdrCaps(std::span<const uint8_t>{edidBlob});
    }
    // BindDrmPlane(NV12) already sniffed the video plane's IN_FORMATS blob for P010.
    planeSupportsP010 = (videoPlaneId != 0) && videoProps.supportsP010;

    isyslog("vaapivideo/display: HDR caps -- connector: metadata=%s colorspace=%s max_bpc=%s[%lu..%lu]; "
            "sink: pq=%s hlg=%s bt2020ycc=%s; plane: p010=%s bt2020enc=%s",
            hdrProps.hdrOutputMetadata ? "yes" : "no", hdrProps.colorspaceValid ? "yes" : "no",
            hdrProps.maxBpc ? "yes" : "no", static_cast<unsigned long>(hdrProps.maxBpcMin),
            static_cast<unsigned long>(hdrProps.maxBpcMax), sinkHdrCaps.hdr10 ? "yes" : "no",
            sinkHdrCaps.hlg ? "yes" : "no", sinkHdrCaps.bt2020Ycc ? "yes" : "no", planeSupportsP010 ? "yes" : "no",
            videoProps.colorEncodingBt2020Valid ? "yes" : "no");
}

[[nodiscard]] auto cVaapiDisplay::CanDriveHdrPlane() const noexcept -> bool {
    // Every property PresentBuffer() writes on an HDR transition must be present; HdrMode::On
    // skips only the sink EDID gate (some TVs lie), never these commit-path prerequisites.
    return planeSupportsP010 && videoProps.colorEncodingValid && videoProps.colorEncodingBt2020Valid &&
           hdrProps.hdrOutputMetadata != 0 && hdrProps.colorspaceValid && hdrProps.maxBpc != 0 &&
           hdrProps.maxBpcMax >= 10;
}

auto cVaapiDisplay::SetHdrOutputState(const HdrStreamInfo &info) -> void {
    // Mutexed so the display thread reads a torn-write-free snapshot of the AVMasteringDisplay
    // / AVContentLight multi-word fields on the next PresentBuffer().
    const cMutexLock lock(&hdrStateMutex);
    stagedHdrState = info;
}

namespace {

/// Round a non-negative double to uint16_t with clamping. std::lround rounds half-away-from-
/// zero for both signs, whereas naive `cast(d + 0.5)` produces wrong results when d is very
/// small negative -- we never pass negatives here, but prefer the correct primitive anyway.
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
                // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
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
    HdrStreamInfo staged;
    {
        const cMutexLock lock(&hdrStateMutex);
        staged = stagedHdrState;
    }
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
            esyslog("vaapivideo/display: drmModeCreatePropertyBlob(HDR_OUTPUT_METADATA) failed: %s", strerror(errno));
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
    if (kind == StreamHdrKind::Sdr || !CanDriveHdrPlane()) {
        return false;
    }
    // Auto mode adds the sink-side EDID gate on top of CanDriveHdrPlane().
    switch (kind) {
        case StreamHdrKind::Hdr10:
            return sinkHdrCaps.hdr10 && sinkHdrCaps.bt2020Ycc;
        case StreamHdrKind::Hlg:
            return sinkHdrCaps.hlg && sinkHdrCaps.bt2020Ycc;
        case StreamHdrKind::Sdr:
            return false;
    }
    return false;
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

    // HDR connector signalling is written only on state transitions, and must precede the
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

    // ALLOW_MODESET is required on AMDGPU (and i915 on some kernels) whenever we change
    // HDR_OUTPUT_METADATA / Colorspace / max bpc -- those can force a link retrain that a
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
                    esyslog("vaapivideo/display: failed to free previous HDR blob: %s", strerror(errno));
                }
                pendingDestroyHdrBlobId = 0;
            }
        } else {
            if (appliedHdrBlobId != 0 && drmModeDestroyPropertyBlob(drmFd, appliedHdrBlobId) != 0) [[unlikely]] {
                esyslog("vaapivideo/display: failed to free rejected HDR blob: %s", strerror(errno));
            }
            appliedHdrBlobId = previousHdrBlobId;
            pendingDestroyHdrBlobId = 0;
            appliedHdrState = previousHdrState;
            esyslog("vaapivideo/display: atomic commit failed during HDR transition -- will retry next frame");
        }
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
