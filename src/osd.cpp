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

// See declaration in osd.h for rationale.
cOsdProvider *osdProvider = nullptr;

namespace {

// cOsd's 3-arg constructor is protected and only befriended to cOsdProvider, so we can't
// use `new cOsd(...)` directly. This thin subclass exposes it so CreateOsd() can return a
// safe no-op object when the display is not yet attached (inheriting cOsd's default behavior:
// no painting, level=999, not active).
class cVaapiDummyOsd : public cOsd {
  public:
    cVaapiDummyOsd(int leftArg, int topArg, uint levelArg) : cOsd(leftArg, topArg, levelArg) {}
};

} // namespace

// ============================================================================
// === OSD PROVIDER ===
// ============================================================================

cVaapiOsdProvider::cVaapiOsdProvider(cVaapiDisplay *const display) : display_(display) {
    // display_ is null under --detached: skins probe ProvidesTrueColor() during VDR Start()
    // before hardware is ready. AttachDisplay() wires the real display later. All pixel-path
    // methods guard on display_ before use.
    if (display_) {
        dsyslog("vaapivideo/osd: provider created %ux%u (TrueColor only)", display_->GetOutputWidth(),
                display_->GetOutputHeight());
    } else {
        dsyslog("vaapivideo/osd: provider created detached (no display yet, TrueColor only)");
    }
}

cVaapiOsdProvider::~cVaapiOsdProvider() noexcept {
    dsyslog("vaapivideo/osd: provider destructor start display_=%p", static_cast<void *>(display_));

    // A fast restart installs the new provider before VDR destroys the old one; only null
    // the global if it still points at us.
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
    // Called from cVaapiDevice::Detach() before drmDropMaster. Dumb buffer GEM handles hold
    // kernel refs that block the fd close; force-free them in place. The cVaapiOsd objects
    // stay alive (VDR owns them) but become no-ops (pixels_==nullptr) until VDR destroys them.
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
        // VDR can have multiple cVaapiOsds alive (e.g. menu over playback); only the
        // topmost is scanned out. Guard so destroying a background OSD doesn't blank the plane.
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
    // VDR's cOsdProvider::NewOsd (vdr/osd.c ~2295) only synthesizes a dummy when *no*
    // provider is registered; with a provider installed it passes nullptr straight to the
    // caller. On any failure return cVaapiDummyOsd -- the same kind of no-op VDR uses
    // as its own internal fallback.
    if (!display_ || !display_->IsInitialized()) [[unlikely]] {
        dsyslog("vaapivideo/osd: CreateOsd while detached -- returning dummy cOsd");
        return new cVaapiDummyOsd(left, top, level);
    }

    const int drmFd = display_->GetDrmFd();
    if (drmFd < 0) [[unlikely]] {
        esyslog("vaapivideo/osd: CreateOsd - invalid DRM fd, returning dummy cOsd");
        return new cVaapiDummyOsd(left, top, level);
    }

    // VDR's cOsd API has no width/height negotiation at construction, so size the FB from
    // (left, top) to the screen corner. Flush() clips any pixmap that overflows this rect.
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
    // cOsd has no width/height accessors after construction; width_/height_ are the dumb-buffer
    // extent (screen minus OSD origin), not the visible pixmap region.
    : cOsd(posX, posY, lvl), drmFd_(fd), height_(static_cast<uint32_t>(fbHeight)), provider_(provider),
      width_(static_cast<uint32_t>(fbWidth)) {
    provider_->TrackOsd(this);
}

cVaapiOsd::~cVaapiOsd() noexcept {
    if (provider_) {
        provider_->UntrackOsd(this);
    }

    if (provider_ && framebufferId_ != 0) [[likely]] {
        provider_->HideOsd(framebufferId_);

        // STRICT ordering: hide -> await -> destroy. KMS continues scanning the GEM
        // buffer until the next atomic commit completes; freeing it before AwaitOsdHidden
        // returns races the display thread and causes the kernel to read freed memory.
        if (cVaapiDisplay *display = provider_->GetDisplay(); display && display->IsInitialized()) {
            display->AwaitOsdHidden(framebufferId_);
        }
    }

    const uint32_t logFbId = framebufferId_; // capture before DestroyDumbBuffer() zeroes it
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
    // We accept any area config: TrueColor pipeline handles palette/depth conversion upstream.
    // Delegate to cOsd for structural validation (count, overlap, dimensions).
    return cOsd::CanHandleAreas(areas, numAreas);
}

auto cVaapiOsd::Flush() -> void {
    if (!pixels_) [[unlikely]] {
        return;
    }

    // LOCK_PIXMAPS must span the entire pop-loop: a concurrent SetPixmap()/AddPixmap() can
    // mutate the list between calls, causing skipped or double-rendered regions.
    bool rendered = false;
    {
        LOCK_PIXMAPS;
        while (auto *pm = dynamic_cast<cPixmapMemory *>(RenderPixmaps())) {
            const cRect vp = pm->ViewPort();
            const uint8_t *src = pm->Data();
            const size_t srcStride = static_cast<size_t>(vp.Width()) * 4;

            // Skins routinely place pixmaps partially off-screen (animated slides, etc.);
            // clamp to FB bounds or we'd write past mappedSize_.
            const int dstX = std::max(vp.X(), 0);
            const int dstY = std::max(vp.Y(), 0);
            const int dstRight = std::min(vp.X() + vp.Width(), static_cast<int>(width_));
            const int dstBottom = std::min(vp.Y() + vp.Height(), static_cast<int>(height_));

            if (dstX < dstRight && dstY < dstBottom) [[likely]] {
                const size_t copyBytes = static_cast<size_t>(dstRight - dstX) * 4;
                // stride_ is the driver-returned pitch and may exceed width_*4 (row padding).
                // Always step dst by stride_, never by width_*4.
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

    // UpdateOsd() is outside LOCK_PIXMAPS: it takes display's osdMutex, which the display
    // thread holds during atomic commit. Holding LOCK_PIXMAPS through that would stall all
    // VDR pixmap operations on vsync. Only fbId + geometry is passed; KMS scans the dumb
    // buffer directly so no pixel copy is needed here.
    if (rendered) {
        provider_->UpdateOsd(*this);
        return;
    }

    if (IsTrueColor()) {
        return;
    }

    // Indexed-color fallback: only legacy 4/8bpp skins reach here; TrueColor skins exit
    // above via IsTrueColor(). Per-pixel palette lookup is slow but unavoidable for old skins.
    bool anyDirty = false;
    for (int i = 0; cBitmap *bitmap = GetBitmap(i); ++i) {
        int x1 = 0;
        int y1 = 0;
        int x2 = 0;
        int y2 = 0;
        if (!bitmap->Dirty(x1, y1, x2, y2)) {
            continue;
        }

        // Bitmap-local (x,y) maps to FB offset (bmpX0+x, bmpY0+y); clamp dirty rect so we
        // don't write outside the mmap'd region.
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

    // Dumb buffers: universal DRM fallback -- CPU-mappable, no GPU API required.
    // Tradeoff: no tiling/compression, acceptable because the OSD plane is small.
    drm_mode_create_dumb createReq{};
    createReq.width = fbWidth;
    createReq.height = fbHeight;
    createReq.bpp = 32;

    if (drmIoctl(drmFd_, DRM_IOCTL_MODE_CREATE_DUMB, &createReq) < 0) [[unlikely]] {
        esyslog("vaapivideo/osd: DRM_IOCTL_MODE_CREATE_DUMB failed: %s", strerror(errno));
        return false;
    }

    gemHandle_ = createReq.handle;
    // Use driver-returned pitch verbatim; recomputing as fbWidth*4 misses hardware row padding.
    stride_ = createReq.pitch;
    mappedSize_ = createReq.size;

    // Register as a KMS FB. ARGB8888 is single-plane so only handles[0] is populated.
    const uint32_t handles[4] = {gemHandle_, 0, 0, 0};
    const uint32_t pitches[4] = {stride_, 0, 0, 0};
    const uint32_t offsets[4] = {0, 0, 0, 0};

    if (drmModeAddFB2(drmFd_, fbWidth, fbHeight, DRM_FORMAT_ARGB8888, handles, pitches, offsets, &framebufferId_, 0) <
        0) [[unlikely]] {
        esyslog("vaapivideo/osd: drmModeAddFB2 failed: %s", strerror(errno));
        DestroyDumbBuffer();
        return false;
    }

    // PROT_READ | PROT_WRITE: some hardened libc / ASan builds verify that the source side
    // of memcpy is readable even when only writes occur at this address. Kernel FBC/PSR reads
    // go through the GPU mapping, not our user mmap.
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

    // Zero = fully transparent ARGB; dumb-buffer pages are not zeroed by the kernel.
    std::memset(pixels_, 0, mappedSize_);
    return true;
}

auto cVaapiOsd::DestroyDumbBuffer() -> void {
    // Reverse-allocation order: munmap -> drmModeRmFB -> destroy GEM.
    // drmModeRmFB MUST precede destroy_dumb -- kernel WARN_ON()s if the GEM handle still
    // has an FB attached. Idempotent: each step zeroes its handle for safe double-call
    // (e.g. ReleaseAllOsdResources() followed by ~cVaapiOsd).
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
