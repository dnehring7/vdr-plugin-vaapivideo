// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file osd.cpp
 * @brief OSD overlay using DRM dumb buffers
 */

#include "osd.h"

#include <sys/mman.h>
#include <sys/types.h>

// C++ Standard Library
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>

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
#include <vdr/tools.h>
#pragma GCC diagnostic pop

// ============================================================================
// === GLOBAL STATE ===
// ============================================================================

// VDR queries this global to find the active provider; we maintain it manually because VDR's own provider registry uses
// a different, plugin-incompatible path.
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

    // Only clear the global if it still points to us; another provider may have already replaced it during a rapid
    // teardown/restart cycle.
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

[[nodiscard]] auto cVaapiOsdProvider::GetDisplay() const noexcept -> cVaapiDisplay * { return display_; }

auto cVaapiOsdProvider::HideOsd() -> void {
    if (display_) [[likely]] {
        // An empty OsdInfo struct means "no OSD plane" -- the display thread will remove the overlay on its next atomic
        // commit.
        display_->SetOsd({});
    }
}

auto cVaapiOsdProvider::UpdateOsd(cVaapiOsd &osd) const -> void {
    if (!display_) [[unlikely]] {
        return;
    }

    // Pass the current framebuffer ID and geometry to the display thread. Left()/Top() come from cOsd base class and
    // are the screen-space position of the OSD origin, as originally passed to CreateOsd().
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

    // The framebuffer covers the region from the OSD origin to the screen edge -- exactly the area that
    // AppendOsdPlane() will scan out. Skin content that extends beyond this (oversized pixmaps, off-screen bitmaps) is
    // clipped in Flush() before writing to the framebuffer.
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
    // cOsd base stores position and level; we duplicate width/height ourselves because cOsd does not expose them as
    // settable fields after construction.
    : cOsd(posX, posY, lvl), drmFd_(fd), height_(static_cast<uint32_t>(fbHeight)), provider_(provider),
      width_(static_cast<uint32_t>(fbWidth)) {
    // drmFd_ is borrowed from the display; it must remain valid for the OSD lifetime.
}

cVaapiOsd::~cVaapiOsd() noexcept {
    if (provider_) [[likely]] {
        provider_->HideOsd();

        // Block until the display thread stops scanning out our framebuffer. DestroyDumbBuffer() below frees the GEM
        // object; doing that while the display is still reading it would cause visible corruption or a kernel page
        // fault, so this wait is mandatory.
        if (cVaapiDisplay *display = provider_->GetDisplay();
            display && framebufferId_ != 0 && display->IsInitialized()) {
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
    // Delegate structural checks (overlapping areas, out-of-bounds) to the base class.
    if (const eOsdError baseResult = cOsd::CanHandleAreas(areas, numAreas); baseResult != oeOk) {
        return baseResult;
    }

    // ProvidesTrueColor() is true, so VDR's pixmap layer handles all color-depth conversion before calling Flush(); any
    // bit depth is fine.
    return oeOk;
}

auto cVaapiOsd::Flush() -> void {
    if (!pixels_) [[unlikely]] {
        return;
    }

    // TrueColor path: VDR composites pixmaps into ARGB8888 dirty rectangles. RenderPixmaps() may return multiple
    // non-overlapping dirty regions from different pixmaps (e.g. background, text, borders). The caller must loop until
    // it returns nullptr, otherwise parts of the OSD are lost. The entire loop must be protected by LOCK_PIXMAPS (see
    // vdr/osd.h).
    {
        LOCK_PIXMAPS;
        bool rendered = false;
        while (auto *pm = dynamic_cast<cPixmapMemory *>(RenderPixmaps())) {
            const cRect vp = pm->ViewPort();
            const uint8_t *src = pm->Data();
            const size_t srcStride = static_cast<size_t>(vp.Width()) * 4;

            // Clip the viewport to the framebuffer boundaries. Skins may produce pixmaps that extend beyond the visible
            // area (e.g. skinflatplus with oversized OSD settings on a 1080p display).
            const int dstX = std::max(vp.X(), 0);
            const int dstY = std::max(vp.Y(), 0);
            const int dstRight = std::min(vp.X() + vp.Width(), static_cast<int>(width_));
            const int dstBottom = std::min(vp.Y() + vp.Height(), static_cast<int>(height_));

            if (dstX < dstRight && dstY < dstBottom) [[likely]] {
                const auto srcOffX = static_cast<size_t>(dstX - vp.X()) * 4;
                const size_t copyBytes = static_cast<size_t>(dstRight - dstX) * 4;
                for (int y = dstY; y < dstBottom; ++y) {
                    // stride_ (not width_*4) accounts for any DRM alignment padding added by the kernel during dumb
                    // buffer creation.
                    const size_t dstOff = (static_cast<size_t>(y) * stride_) + (static_cast<size_t>(dstX) * 4);
                    const size_t srcOff = (static_cast<size_t>(y - vp.Y()) * srcStride) + srcOffX;
                    std::memcpy(pixels_ + dstOff, src + srcOff, copyBytes);
                }
                rendered = true;
            }
            DestroyPixmap(pm);
        }
        if (rendered) {
            provider_->UpdateOsd(*this);
            return;
        }
    }

    // In TrueColor mode RenderPixmaps() composites everything (including bitmap areas) into pixmaps, so reaching here
    // simply means no dirty content exists this frame -- the indexed-color path is not needed.
    if (IsTrueColor()) {
        return;
    }

    // Indexed-color path: the skin uses palette-based bitmaps (4/8bpp). Iterate only the dirty rectangle of each bitmap
    // and convert each pixel through VDR's palette lookup to ARGB8888.
    bool anyDirty = false;
    for (int i = 0; cBitmap *bitmap = GetBitmap(i); ++i) {
        int x1 = 0;
        int y1 = 0;
        int x2 = 0;
        int y2 = 0;
        if (!bitmap->Dirty(x1, y1, x2, y2)) {
            continue;
        }

        // Clamp dirty region to framebuffer boundaries (bitmap-local coords). Bitmap origin (X0(), Y0()) is its
        // position within the OSD coordinate space; pixel (x, y) maps to framebuffer position (X0()+x, Y0()+y).
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

    // Step 1: Allocate a dumb (CPU-accessible) GEM buffer in kernel memory. The kernel returns a GEM handle, the actual
    // row stride, and the total size.
    drm_mode_create_dumb createReq{};
    createReq.width = fbWidth;
    createReq.height = fbHeight;
    createReq.bpp = 32;

    if (drmIoctl(drmFd_, DRM_IOCTL_MODE_CREATE_DUMB, &createReq) < 0) [[unlikely]] {
        esyslog("vaapivideo/osd: DRM_IOCTL_MODE_CREATE_DUMB failed: %s", strerror(errno));
        return false;
    }

    gemHandle_ = createReq.handle;
    stride_ = createReq.pitch; // may be larger than fbWidth*4 due to alignment
    mappedSize_ = createReq.size;

    // Step 2: Register the GEM buffer as a DRM framebuffer so the modesetting API can scan it out via a DRM plane.
    // Four-plane arrays required by the API; we use only the first plane for the single ARGB8888 buffer.
    const uint32_t handles[4] = {gemHandle_, 0, 0, 0};
    const uint32_t pitches[4] = {stride_, 0, 0, 0};
    const uint32_t offsets[4] = {0, 0, 0, 0};

    if (drmModeAddFB2(drmFd_, fbWidth, fbHeight, DRM_FORMAT_ARGB8888, handles, pitches, offsets, &framebufferId_, 0) <
        0) [[unlikely]] {
        esyslog("vaapivideo/osd: drmModeAddFB2 failed: %s", strerror(errno));
        DestroyDumbBuffer();
        return false;
    }

    // Step 3: Obtain a mmap offset from the kernel, then map the buffer into the process address space so we can write
    // pixels from the CPU side.
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
    // Teardown must follow the reverse allocation order: unmap CPU memory -> remove DRM framebuffer -> destroy GEM
    // object. Reversing the order would leave dangling kernel references.
    if (pixels_ && pixels_ != MAP_FAILED) [[likely]] {
        munmap(pixels_, mappedSize_);
        pixels_ = nullptr;
    }

    if (framebufferId_ != 0 && drmFd_ >= 0) [[likely]] {
        drmModeRmFB(drmFd_, framebufferId_);
        framebufferId_ = 0;
    }

    if (gemHandle_ != 0 && drmFd_ >= 0) [[likely]] {
        drm_mode_destroy_dumb destroyReq{};
        destroyReq.handle = gemHandle_;
        // Return value intentionally ignored; nothing useful to do on failure here.
        (void)drmIoctl(drmFd_, DRM_IOCTL_MODE_DESTROY_DUMB, &destroyReq);
        gemHandle_ = 0;
    }
}
