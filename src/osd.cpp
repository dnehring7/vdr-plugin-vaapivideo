// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file osd.cpp
 * @brief OSD overlay using DRM dumb buffers
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

// VDR's cOsdProvider::Install() is singleton-only and unsuitable for multi-plugin coexistence;
// we track the active provider manually so device.cpp can manage attach/detach across stream switches.
cOsdProvider *osdProvider = nullptr;

// ============================================================================
// === OSD PROVIDER ===
// ============================================================================

cVaapiOsdProvider::cVaapiOsdProvider(cVaapiDisplay *const display) : display_(display) {
    dsyslog("vaapivideo/osd: provider created %ux%u (TrueColor only)", display_->GetOutputWidth(),
            display_->GetOutputHeight());
}

cVaapiOsdProvider::~cVaapiOsdProvider() noexcept {
    dsyslog("vaapivideo/osd: provider destructor start display_=%p", static_cast<void *>(display_));

    // Guard against rapid teardown/restart: another provider instance may have already replaced us.
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
        // Only clear if this is the active framebuffer; destroying a non-active OSD must not hide the active one.
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
    if (!display_ || !display_->IsInitialized()) [[unlikely]] {
        esyslog("vaapivideo/osd: CreateOsd failed - display not ready");
        return nullptr;
    }

    const int drmFd = display_->GetDrmFd();
    if (drmFd < 0) [[unlikely]] {
        esyslog("vaapivideo/osd: CreateOsd failed - invalid DRM fd");
        return nullptr;
    }

    // Framebuffer spans from OSD origin to screen edge; oversized skin content is clipped scanline-by-scanline
    // in Flush() to avoid out-of-bounds writes into the mmap'd dumb buffer.
    const auto screenWidth = static_cast<int>(display_->GetOutputWidth());
    const auto screenHeight = static_cast<int>(display_->GetOutputHeight());
    const int osdWidth = screenWidth - left;
    const int osdHeight = screenHeight - top;

    if (osdWidth <= 0 || osdHeight <= 0) [[unlikely]] {
        esyslog("vaapivideo/osd: CreateOsd failed - invalid dimensions %dx%d", osdWidth, osdHeight);
        return nullptr;
    }

    auto *osd = new cVaapiOsd(left, top, level, drmFd, osdWidth, osdHeight, this);
    if (!osd->Allocate()) [[unlikely]] {
        esyslog("vaapivideo/osd: CreateOsd failed - initialization error");
        delete osd;
        return nullptr;
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
    // cOsd does not expose width/height as settable fields after construction; we store our own.
    : cOsd(posX, posY, lvl), drmFd_(fd), height_(static_cast<uint32_t>(fbHeight)), provider_(provider),
      width_(static_cast<uint32_t>(fbWidth)) {
    // drmFd_ is borrowed from the display; it must remain valid for the OSD lifetime.
    provider_->TrackOsd(this);
}

cVaapiOsd::~cVaapiOsd() noexcept {
    if (provider_) {
        provider_->UntrackOsd(this);
    }

    if (provider_ && framebufferId_ != 0) [[likely]] {
        provider_->HideOsd(framebufferId_);

        // Wait for the display thread to finish scanning out this framebuffer; freeing the GEM object
        // while KMS still references it causes a kernel use-after-free.
        if (cVaapiDisplay *display = provider_->GetDisplay(); display && display->IsInitialized()) {
            display->AwaitOsdHidden(framebufferId_);
        }
    }

    // Save before DestroyDumbBuffer() zeroes framebufferId_.
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
    // TrueColor provider: VDR handles color-depth conversion. Delegate structural checks to base class.
    return cOsd::CanHandleAreas(areas, numAreas);
}

auto cVaapiOsd::Flush() -> void {
    if (!pixels_) [[unlikely]] {
        return;
    }

    // RenderPixmaps() returns one dirty ARGB8888 region per call; loop until nullptr.
    // LOCK_PIXMAPS must be held across the entire sequence -- VDR's cPixmap is not thread-safe.
    bool rendered = false;
    {
        LOCK_PIXMAPS;
        while (auto *pm = dynamic_cast<cPixmapMemory *>(RenderPixmaps())) {
            const cRect vp = pm->ViewPort();
            const uint8_t *src = pm->Data();
            const size_t srcStride = static_cast<size_t>(vp.Width()) * 4;

            // Clip to framebuffer boundaries (skins may produce oversized pixmaps).
            const int dstX = std::max(vp.X(), 0);
            const int dstY = std::max(vp.Y(), 0);
            const int dstRight = std::min(vp.X() + vp.Width(), static_cast<int>(width_));
            const int dstBottom = std::min(vp.Y() + vp.Height(), static_cast<int>(height_));

            if (dstX < dstRight && dstY < dstBottom) [[likely]] {
                const size_t copyBytes = static_cast<size_t>(dstRight - dstX) * 4;
                // stride_ may exceed width_*4 due to DRM pitch alignment requirements.
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

    // Notify display after releasing LOCK_PIXMAPS to avoid holding VDR's global lock during an atomic commit.
    // UpdateOsd() passes only geometry; the display reads pixel data directly from the mmap'd dumb buffer.
    if (rendered) {
        provider_->UpdateOsd(*this);
        return;
    }

    if (IsTrueColor()) {
        return;
    }

    // Indexed-color fallback: convert dirty bitmap regions from VDR palette to ARGB8888 pixel-by-pixel.
    // Only reached by legacy 4/8bpp skins; TrueColor skins use the RenderPixmaps() path above.
    bool anyDirty = false;
    for (int i = 0; cBitmap *bitmap = GetBitmap(i); ++i) {
        int x1 = 0;
        int y1 = 0;
        int x2 = 0;
        int y2 = 0;
        if (!bitmap->Dirty(x1, y1, x2, y2)) {
            continue;
        }

        // Clamp dirty region to framebuffer; bitmap pixel (x,y) maps to framebuffer (X0()+x, Y0()+y).
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

    // Allocate a dumb (CPU-accessible) GEM buffer -- the only buffer type guaranteed by all DRM drivers.
    drm_mode_create_dumb createReq{};
    createReq.width = fbWidth;
    createReq.height = fbHeight;
    createReq.bpp = 32;

    if (drmIoctl(drmFd_, DRM_IOCTL_MODE_CREATE_DUMB, &createReq) < 0) [[unlikely]] {
        esyslog("vaapivideo/osd: DRM_IOCTL_MODE_CREATE_DUMB failed: %s", strerror(errno));
        return false;
    }

    gemHandle_ = createReq.handle;
    stride_ = createReq.pitch; // may exceed fbWidth*4 due to GPU-specific pitch alignment
    mappedSize_ = createReq.size;

    // Register as a KMS framebuffer. ARGB8888 is single-plane (index 0 only).
    const uint32_t handles[4] = {gemHandle_, 0, 0, 0};
    const uint32_t pitches[4] = {stride_, 0, 0, 0};
    const uint32_t offsets[4] = {0, 0, 0, 0};

    if (drmModeAddFB2(drmFd_, fbWidth, fbHeight, DRM_FORMAT_ARGB8888, handles, pitches, offsets, &framebufferId_, 0) <
        0) [[unlikely]] {
        esyslog("vaapivideo/osd: drmModeAddFB2 failed: %s", strerror(errno));
        DestroyDumbBuffer();
        return false;
    }

    // mmap for direct CPU pixel writes; PROT_READ is needed for potential read-back by FBC/PSR.
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

    // Start with a fully transparent framebuffer so no garbage is visible before the first Flush().
    std::memset(pixels_, 0, mappedSize_);
    return true;
}

auto cVaapiOsd::DestroyDumbBuffer() -> void {
    // Reverse allocation order: unmap -> remove FB -> destroy GEM handle.
    // FB must be removed before closing the GEM handle: the kernel asserts no active FB references remain.
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
