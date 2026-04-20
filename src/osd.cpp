// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file osd.cpp
 * @brief OSD overlay backed by DRM dumb buffers, scanned out on the OSD plane.
 *
 * Each cVaapiOsd owns one CPU-accessible (dumb) GEM buffer + KMS framebuffer; VDR's
 * cPixmap renderer paints ARGB8888 directly into the mmap'd region in Flush(), and the
 * display thread picks up the framebuffer-id (and geometry) on the next atomic commit
 * via cVaapiDisplay::SetOsd().
 *
 * Threading: cPixmap rendering runs under VDR's LOCK_PIXMAPS macro and feeds Flush() on
 * the main VDR thread; provider->UpdateOsd() is called *after* releasing LOCK_PIXMAPS so
 * we don't hold the global pixmap lock across the display's atomic-commit path. The
 * display reads no pixel data from us -- it scans the dumb buffer directly via KMS, so
 * the OSD lifetime must outlive any pageflip that referenced its fbId (see ~cVaapiOsd).
 */

#include "osd.h"
#include "display.h"

// C++ Standard Library
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>

// POSIX
#include <sys/mman.h>
#include <sys/types.h>

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
// === GLOBAL STATE ===
// ============================================================================

// VDR's cOsdProvider::Install() is a singleton with no detach hook, which doesn't survive
// the SVDRP DETA/ATTA cycle in device.cpp. We keep our own pointer so device.cpp can swap
// the live display under us across stream switches without re-creating the provider.
cOsdProvider *osdProvider = nullptr;

namespace {

// No-op cOsd returned by CreateOsd() while detached. cOsd's three-arg constructor is
// protected and only befriended to cOsdProvider, so we can't `new cOsd(...)` directly
// from cVaapiOsdProvider; this thin derived adapter exposes it. Inheriting cOsd's
// default behavior (no painting, level=999 not active) gives skins something safe to
// hold until SVDRP ATTA installs a real cVaapiOsd.
class cVaapiDummyOsd : public cOsd {
  public:
    cVaapiDummyOsd(int left, int top, uint level) : cOsd(left, top, level) {}
};

} // namespace

// ============================================================================
// === OSD PROVIDER ===
// ============================================================================

cVaapiOsdProvider::cVaapiOsdProvider(cVaapiDisplay *const display) : display_(display) {
    // display_ may be null when --detached defers hardware init: skin plugins still need to
    // see the provider during VDR Start() so they can probe TrueColor support, and a later
    // AttachDisplay() (from MakePrimaryDevice or Attach) wires up the live display. The
    // pixel-path methods (CreateOsd, UpdateOsd, HideOsd) already guard on display_.
    if (display_) {
        dsyslog("vaapivideo/osd: provider created %ux%u (TrueColor only)", display_->GetOutputWidth(),
                display_->GetOutputHeight());
    } else {
        dsyslog("vaapivideo/osd: provider created detached (no display yet, TrueColor only)");
    }
}

cVaapiOsdProvider::~cVaapiOsdProvider() noexcept {
    dsyslog("vaapivideo/osd: provider destructor start display_=%p", static_cast<void *>(display_));

    // Conditional clear: a fast restart can install a new provider before VDR destroys
    // the old one, in which case ::osdProvider already points at the replacement and we
    // must NOT null it.
    if (::osdProvider == this) {
        ::osdProvider = nullptr;
    }

    dsyslog("vaapivideo/osd: provider destroyed");
}

// ============================================================================
// === PUBLIC API ===
// ============================================================================

auto cVaapiOsdProvider::AttachDisplay(cVaapiDisplay *display) noexcept -> void {
    dsyslog("vaapivideo/osd: attaching display %p (was %p)", static_cast<void *>(display),
            static_cast<void *>(display_));
    display_ = display;
}

auto cVaapiOsdProvider::DetachDisplay() noexcept -> void {
    dsyslog("vaapivideo/osd: detaching display (was %p)", static_cast<void *>(display_));
    display_ = nullptr;
}

auto cVaapiOsdProvider::ReleaseAllOsdResources() -> void {
    // Called from cVaapiDevice::Detach(): VDR may keep cVaapiOsd objects alive past our
    // hardware release, but their dumb buffers hold kernel refs that would block
    // drmDropMaster/close. Force-release the buffers in place; the cVaapiOsd objects
    // become render-no-ops (pixels_==nullptr) until VDR finally destroys them.
    const cMutexLock lock(&osdListMutex_);
    for (auto *osd : activeOsds_) {
        osd->DestroyDumbBuffer();
    }
    dsyslog("vaapivideo/osd: released DRM resources for %zu active OSDs", activeOsds_.size());
}

auto cVaapiOsdProvider::TrackOsd(cVaapiOsd *osd) -> void {
    const cMutexLock lock(&osdListMutex_);
    activeOsds_.push_back(osd);
}

auto cVaapiOsdProvider::UntrackOsd(cVaapiOsd *osd) -> void {
    const cMutexLock lock(&osdListMutex_);
    std::erase(activeOsds_, osd);
}

[[nodiscard]] auto cVaapiOsdProvider::GetDisplay() const noexcept -> cVaapiDisplay * { return display_; }

auto cVaapiOsdProvider::HideOsd(const uint32_t fbId) -> void {
    if (display_) [[likely]] {
        // Conditional clear: VDR can have several cVaapiOsds alive concurrently (e.g. menu
        // over playback), only one of which is currently presented. Tearing down a
        // non-active OSD must NOT yank the visible OSD off the plane.
        display_->ClearOsdIfActive(fbId);
    }
}

auto cVaapiOsdProvider::UpdateOsd(cVaapiOsd &osd) const -> void {
    if (!display_) [[unlikely]] {
        return;
    }

    display_->SetOsd({.fbId = osd.GetFramebufferId(),
                      .height = static_cast<uint32_t>(osd.Height()),
                      .width = static_cast<uint32_t>(osd.Width()),
                      .x = osd.Left(),
                      .y = osd.Top()});
}

// ============================================================================
// === VDR INTERFACE ===
// ============================================================================

[[nodiscard]] auto cVaapiOsdProvider::CreateOsd(const int left, const int top, const uint level) -> cOsd * {
    // Returning nullptr from CreateOsd while a provider is installed is unsafe: VDR's
    // cOsdProvider::NewOsd (vdr/osd.c:2295-2304) only synthesizes a dummy when *no*
    // provider is registered; otherwise it hands the nullptr straight back to the caller
    // (and even dereferences cOsd::Osds[0] before returning). On any failure -- including
    // the --detached case where display_ is null because hardware init was deferred --
    // return the same kind of base-class dummy VDR uses as its own fallback so callers
    // (skins, OSD probes) get a no-op object instead of a null.
    if (!display_ || !display_->IsInitialized()) [[unlikely]] {
        dsyslog("vaapivideo/osd: CreateOsd while detached -- returning dummy cOsd");
        return new cVaapiDummyOsd(left, top, level);
    }

    const int drmFd = display_->GetDrmFd();
    if (drmFd < 0) [[unlikely]] {
        esyslog("vaapivideo/osd: CreateOsd - invalid DRM fd, returning dummy cOsd");
        return new cVaapiDummyOsd(left, top, level);
    }

    // Size the FB to "from (left, top) to the bottom-right screen corner" instead of
    // pre-asking the skin how big it wants to be: VDR's cOsd API has no width/height
    // negotiation at construction. Pixmaps that exceed this rectangle are clipped per
    // scanline in Flush() so we never write past the mmap'd region.
    const auto screenWidth = static_cast<int>(display_->GetOutputWidth());
    const auto screenHeight = static_cast<int>(display_->GetOutputHeight());
    const int osdWidth = screenWidth - left;
    const int osdHeight = screenHeight - top;

    if (osdWidth <= 0 || osdHeight <= 0) [[unlikely]] {
        esyslog("vaapivideo/osd: CreateOsd - invalid dimensions %dx%d, returning dummy cOsd", osdWidth, osdHeight);
        return new cVaapiDummyOsd(left, top, level);
    }

    auto *osd = new cVaapiOsd(left, top, level, drmFd, osdWidth, osdHeight, this);
    if (!osd->Allocate()) [[unlikely]] {
        esyslog("vaapivideo/osd: CreateOsd - allocation error, returning dummy cOsd");
        delete osd;
        return new cVaapiDummyOsd(left, top, level);
    }

    dsyslog("vaapivideo/osd: CreateOsd() fbId=%u %dx%d at (%d,%d) level=%u", osd->GetFramebufferId(), osdWidth,
            osdHeight, left, top, level);
    return osd;
}

// ============================================================================
// === OSD ===
// ============================================================================

cVaapiOsd::cVaapiOsd(const int posX, const int posY, const uint lvl, const int fd, const int fbWidth,
                     const int fbHeight, cVaapiOsdProvider *provider)
    // cOsd has no width/height accessors past construction, so we keep our own. Width and
    // height represent the dumb-buffer extent (= screen minus OSD origin), NOT the visible
    // pixmap region.
    : cOsd(posX, posY, lvl), drmFd_(fd), height_(static_cast<uint32_t>(fbHeight)), provider_(provider),
      width_(static_cast<uint32_t>(fbWidth)) {
    // drmFd_ is BORROWED from the display. The display outlives every cVaapiOsd by virtue
    // of cVaapiDevice::Detach() running provider->ReleaseAllOsdResources() before
    // ReleaseHardware(), so we don't need to ref-count the fd here.
    provider_->TrackOsd(this);
}

cVaapiOsd::~cVaapiOsd() noexcept {
    if (provider_) {
        provider_->UntrackOsd(this);
    }

    if (provider_ && framebufferId_ != 0) [[likely]] {
        provider_->HideOsd(framebufferId_);

        // STRICT ordering: hide -> await -> destroy. The display thread must observe the
        // hide and present at least one frame without our fbId before we free the GEM
        // object, otherwise the kernel reads freed memory on its next scanout. AwaitOsdHidden
        // is the synchronization point; without it the next two lines race the display thread.
        if (cVaapiDisplay *display = provider_->GetDisplay(); display && display->IsInitialized()) {
            display->AwaitOsdHidden(framebufferId_);
        }
    }

    // Snapshot before DestroyDumbBuffer() zeroes framebufferId_, otherwise we log "fbId=0".
    const uint32_t logFbId = framebufferId_;
    DestroyDumbBuffer();
    dsyslog("vaapivideo/osd: destroyed fbId=%u", logFbId);
}

// ============================================================================
// === PUBLIC API ===
// ============================================================================

[[nodiscard]] auto cVaapiOsd::Allocate() -> bool {
    if (!CreateDumbBuffer(width_, height_)) [[unlikely]] {
        esyslog("vaapivideo/osd: dumb buffer allocation failed");
        return false;
    }

    dsyslog("vaapivideo/osd: Allocate() %ux%u fbId=%u stride=%u", width_, height_, framebufferId_, stride_);
    return true;
}

[[nodiscard]] auto cVaapiOsd::GetFramebufferId() const noexcept -> uint32_t { return framebufferId_; }

[[nodiscard]] auto cVaapiOsd::Height() const noexcept -> int { return static_cast<int>(height_); }

[[nodiscard]] auto cVaapiOsd::Width() const noexcept -> int { return static_cast<int>(width_); }

// ============================================================================
// === VDR INTERFACE ===
// ============================================================================

[[nodiscard]] auto cVaapiOsd::CanHandleAreas(const tArea *areas, const int numAreas) -> eOsdError {
    // We expose ARGB8888 surfaces and VDR's TrueColor pipeline owns any palette/depth
    // conversion -- our framebuffer never sees indexed pixels through this entry point.
    // The base class implementation already validates the structural constraints (area
    // count, overlap, dimensions) we'd otherwise have to repeat, so just delegate.
    return cOsd::CanHandleAreas(areas, numAreas);
}

auto cVaapiOsd::Flush() -> void {
    if (!pixels_) [[unlikely]] {
        return;
    }

    // RenderPixmaps() pops one dirty pixmap per call and returns null when done. The
    // LOCK_PIXMAPS macro takes VDR's per-OSD pixmap mutex; it MUST be held for the whole
    // pop-loop, not just per call -- otherwise a concurrent SetPixmap()/AddPixmap() can
    // mutate the list mid-iteration and we'd skip or double-render regions.
    bool rendered = false;
    {
        LOCK_PIXMAPS;
        while (auto *pm = dynamic_cast<cPixmapMemory *>(RenderPixmaps())) {
            const cRect vp = pm->ViewPort();
            const uint8_t *src = pm->Data();
            const size_t srcStride = static_cast<size_t>(vp.Width()) * 4;

            // Clip the pixmap's destination rect to our FB. Skins routinely place pixmaps
            // partially off-screen (e.g. animated slides) and we'd write past mappedSize_
            // without this clamp.
            const int dstX = std::max(vp.X(), 0);
            const int dstY = std::max(vp.Y(), 0);
            const int dstRight = std::min(vp.X() + vp.Width(), static_cast<int>(width_));
            const int dstBottom = std::min(vp.Y() + vp.Height(), static_cast<int>(height_));

            if (dstX < dstRight && dstY < dstBottom) [[likely]] {
                const size_t copyBytes = static_cast<size_t>(dstRight - dstX) * 4;
                // stride_ may exceed width_*4 -- the DRM driver returned the pitch via
                // drm_mode_create_dumb.pitch and we use that value verbatim. Always step
                // dst rows by stride_, never by width_*4.
                uint8_t *dst = pixels_ + (static_cast<size_t>(dstY) * stride_) + (static_cast<size_t>(dstX) * 4);
                const uint8_t *srcRow =
                    src + (static_cast<size_t>(dstY - vp.Y()) * srcStride) + (static_cast<size_t>(dstX - vp.X()) * 4);
                for (int y = dstY; y < dstBottom; ++y) {
                    std::memcpy(dst, srcRow, copyBytes);
                    dst += stride_;
                    srcRow += srcStride;
                }
                rendered = true;
            }
            DestroyPixmap(pm);
        }
    }

    // UpdateOsd() is called OUTSIDE LOCK_PIXMAPS: it ends up in display->SetOsd() which
    // takes osdMutex, and the display thread holds osdMutex during PresentBuffer's atomic
    // commit. Holding LOCK_PIXMAPS across that path would gate every VDR pixmap operation
    // on the display's vsync cadence. Only fbId + geometry crosses; the kernel scans the
    // dumb buffer directly so we don't need to copy pixels here.
    if (rendered) {
        provider_->UpdateOsd(*this);
        return;
    }

    if (IsTrueColor()) {
        return;
    }

    // Indexed-color fallback (cBitmap path). VDR's TrueColor skins go through RenderPixmaps
    // above; only legacy 4/8bpp skins still use the bitmap API. We palette-convert per
    // pixel into ARGB8888 -- slow but only hit by old skins.
    bool anyDirty = false;
    for (int i = 0; cBitmap *bitmap = GetBitmap(i); ++i) {
        int x1 = 0;
        int y1 = 0;
        int x2 = 0;
        int y2 = 0;
        if (!bitmap->Dirty(x1, y1, x2, y2)) {
            continue;
        }

        // Clamp the dirty rect to the FB. Coordinate mapping: bitmap-local (x,y) lands at
        // FB (bmpX0+x, bmpY0+y), so the FB-bounds clamp on x is "x >= -bmpX0" / "x < width-bmpX0".
        const int bmpX0 = bitmap->X0();
        const int bmpY0 = bitmap->Y0();
        const int cx1 = std::max(x1, -bmpX0);
        const int cy1 = std::max(y1, -bmpY0);
        const int cx2 = std::min(x2, static_cast<int>(width_) - 1 - bmpX0);
        const int cy2 = std::min(y2, static_cast<int>(height_) - 1 - bmpY0);

        if (cx1 <= cx2 && cy1 <= cy2) {
            for (int y = cy1; y <= cy2; ++y) {
                uint8_t *dstRow =
                    pixels_ + (static_cast<size_t>(bmpY0 + y) * stride_) + (static_cast<size_t>(bmpX0 + cx1) * 4);
                for (int x = cx1; x <= cx2; ++x) {
                    const tColor color = bitmap->GetColor(x, y);
                    std::memcpy(dstRow, &color, sizeof(uint32_t));
                    dstRow += 4;
                }
            }
            anyDirty = true;
        }
        bitmap->Clean();
    }

    if (anyDirty) {
        provider_->UpdateOsd(*this);
    }
}

// ============================================================================
// === INTERNAL METHODS ===
// ============================================================================

[[nodiscard]] auto cVaapiOsd::CreateDumbBuffer(const uint32_t fbWidth, const uint32_t fbHeight) -> bool {
    if (drmFd_ < 0) [[unlikely]] {
        esyslog("vaapivideo/osd: CreateDumbBuffer - invalid DRM fd");
        return false;
    }

    // Dumb buffers are the lowest-common-denominator GEM allocation: every DRM driver
    // implements them, they're CPU-mappable (which we need for direct ARGB writes), and
    // they need no GPU API. The trade-off is no tiling and no compression -- fine here
    // because the OSD plane is small relative to the video plane.
    drm_mode_create_dumb createReq{};
    createReq.width = fbWidth;
    createReq.height = fbHeight;
    createReq.bpp = 32;

    if (drmIoctl(drmFd_, DRM_IOCTL_MODE_CREATE_DUMB, &createReq) < 0) [[unlikely]] {
        esyslog("vaapivideo/osd: DRM_IOCTL_MODE_CREATE_DUMB failed: %s", strerror(errno));
        return false;
    }

    gemHandle_ = createReq.handle;
    // Use the driver-returned pitch verbatim; do NOT recompute as fbWidth*4. Drivers may
    // pad rows for hardware alignment, and Flush() steps by this stride value.
    stride_ = createReq.pitch;
    mappedSize_ = createReq.size;

    // Register the GEM buffer as a KMS FB. ARGB8888 has one plane, so handles[0] only.
    const uint32_t handles[4] = {gemHandle_, 0, 0, 0};
    const uint32_t pitches[4] = {stride_, 0, 0, 0};
    const uint32_t offsets[4] = {0, 0, 0, 0};

    if (drmModeAddFB2(drmFd_, fbWidth, fbHeight, DRM_FORMAT_ARGB8888, handles, pitches, offsets, &framebufferId_, 0) <
        0) [[unlikely]] {
        esyslog("vaapivideo/osd: drmModeAddFB2 failed: %s", strerror(errno));
        DestroyDumbBuffer();
        return false;
    }

    // mmap the buffer so Flush() can write ARGB pixels directly into the scanout memory.
    // PROT_READ is requested in addition to PROT_WRITE so std::memcpy reads-around-writes
    // don't trap (some hardened libc builds and ASan modes verify the source mapping is
    // readable even when only writes are issued from this address). The kernel-side FBC/PSR
    // read paths run via the GPU mapping, NOT via our user mmap, and don't depend on this.
    drm_mode_map_dumb mapReq{};
    mapReq.handle = gemHandle_;

    if (drmIoctl(drmFd_, DRM_IOCTL_MODE_MAP_DUMB, &mapReq) < 0) [[unlikely]] {
        esyslog("vaapivideo/osd: DRM_IOCTL_MODE_MAP_DUMB failed: %s", strerror(errno));
        DestroyDumbBuffer();
        return false;
    }

    pixels_ = static_cast<uint8_t *>(
        mmap(nullptr, mappedSize_, PROT_READ | PROT_WRITE, MAP_SHARED, drmFd_, static_cast<off_t>(mapReq.offset)));

    if (pixels_ == MAP_FAILED) [[unlikely]] {
        esyslog("vaapivideo/osd: mmap failed: %s", strerror(errno));
        pixels_ = nullptr;
        DestroyDumbBuffer();
        return false;
    }

    // Zero-fill = fully transparent ARGB. Without this the first commit would show
    // whatever stale data the kernel handed us in the dumb-buffer pages.
    std::memset(pixels_, 0, mappedSize_);
    return true;
}

auto cVaapiOsd::DestroyDumbBuffer() -> void {
    // STRICT release order (reverse of allocation): munmap -> drmModeRmFB -> destroy GEM.
    // drmModeRmFB MUST come before destroy_dumb -- the kernel WARN_ON()s and refuses if a
    // GEM handle still has an FB attached. Idempotent: every step zeroes its handle so a
    // double-call (e.g. ReleaseAllOsdResources then ~cVaapiOsd) is a no-op.
    if (pixels_) [[likely]] {
        munmap(pixels_, mappedSize_);
        pixels_ = nullptr;
    }

    if (drmFd_ >= 0) [[likely]] {
        if (framebufferId_ != 0) {
            drmModeRmFB(drmFd_, framebufferId_);
            framebufferId_ = 0;
        }
        if (gemHandle_ != 0) {
            drm_mode_destroy_dumb destroyReq{};
            destroyReq.handle = gemHandle_;
            (void)drmIoctl(drmFd_, DRM_IOCTL_MODE_DESTROY_DUMB, &destroyReq);
            gemHandle_ = 0;
        }
    }
}
