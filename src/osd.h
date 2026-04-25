// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file osd.h
 * @brief OSD overlay using DRM dumb buffers on the KMS OSD plane.
 *
 * Flush() has two paths:
 * - TrueColor skins (32bpp): RenderPixmaps() composites ARGB pixmaps that are
 *   memcpy'd scanline-by-scanline into the dumb buffer.
 * - Legacy indexed-color skins (4/8bpp): dirty bitmaps are palette-converted
 *   per pixel via GetColor() -- slow but only reached by old skins.
 *
 * See osd.cpp file header for threading and lifetime constraints.
 */

#ifndef VDR_VAAPIVIDEO_OSD_H
#define VDR_VAAPIVIDEO_OSD_H

#include "common.h"
#include "display.h"

// C++ Standard Library
#include <vector>

class cVaapiOsd;

// ============================================================================
// === OSD PROVIDER ===
// ============================================================================

/// VDR OSD provider -- allocates cVaapiOsd instances backed by DRM dumb buffers.
class cVaapiOsdProvider : public cOsdProvider {
  public:
    // ========================================================================
    // === SPECIAL MEMBERS ===
    // ========================================================================
    explicit cVaapiOsdProvider(cVaapiDisplay *display);
    ~cVaapiOsdProvider() noexcept override;
    cVaapiOsdProvider(const cVaapiOsdProvider &) = delete;
    cVaapiOsdProvider(cVaapiOsdProvider &&) noexcept = delete;
    auto operator=(const cVaapiOsdProvider &) -> cVaapiOsdProvider & = delete;
    auto operator=(cVaapiOsdProvider &&) noexcept -> cVaapiOsdProvider & = delete;

    // ========================================================================
    // === PUBLIC API ===
    // ========================================================================
    auto AttachDisplay(cVaapiDisplay *display) noexcept -> void; ///< Swap in a new display after SVDRP ATTA
    auto DetachDisplay() noexcept -> void;                       ///< Null the display ref; Flush() becomes a no-op
    auto ReleaseAllOsdResources() -> void; ///< Force-free DRM buffers of all live OSDs before drmDropMaster

  protected:
    // ========================================================================
    // === VDR INTERFACE ===
    // ========================================================================
    [[nodiscard]] auto CreateOsd(int left, int top, uint level) -> cOsd * override;
    [[nodiscard]] auto ProvidesTrueColor() -> bool override { return true; } ///< Enables RenderPixmaps() fast path

  private:
    friend class cVaapiOsd; ///< ~cVaapiOsd calls HideOsd/AwaitOsdHidden; Flush() calls UpdateOsd

    // ========================================================================
    // === INTERNAL METHODS ===
    // ========================================================================
    [[nodiscard]] auto GetDisplay() const noexcept -> cVaapiDisplay *;
    auto HideOsd(uint32_t fbId) -> void;          ///< Clears OSD plane only if fbId is the currently scanned-out FB
    auto UpdateOsd(cVaapiOsd &osd) const -> void; ///< Push FB id + geometry to display for the next atomic commit
    auto TrackOsd(cVaapiOsd *osd) -> void;        ///< Register OSD so ReleaseAllOsdResources() can reach it
    auto UntrackOsd(cVaapiOsd *osd) -> void;

    // ========================================================================
    // === STATE ===
    // ========================================================================
    std::vector<cVaapiOsd *> activeOsds_; ///< Live OSDs holding DRM resources; guarded by osdListMutex_
    cVaapiDisplay *display_;              ///< Borrowed; nulled on Detach -- never outlives the device
    cMutex osdListMutex_;                 ///< Guards activeOsds_ (SVDRP thread vs. VDR main thread)
};

// ============================================================================
// === OSD ===
// ============================================================================

/// VDR OSD backed by one DRM dumb GEM buffer + KMS framebuffer, mmap'd for direct ARGB writes.
class cVaapiOsd : public cOsd {
  public:
    // ========================================================================
    // === SPECIAL MEMBERS ===
    // ========================================================================
    cVaapiOsd(int posX, int posY, uint lvl, int fd, int fbWidth, int fbHeight, cVaapiOsdProvider *provider);
    ~cVaapiOsd() noexcept override;
    cVaapiOsd(const cVaapiOsd &) = delete;
    cVaapiOsd(cVaapiOsd &&) noexcept = delete;
    auto operator=(const cVaapiOsd &) -> cVaapiOsd & = delete;
    auto operator=(cVaapiOsd &&) noexcept -> cVaapiOsd & = delete;

    // ========================================================================
    // === VDR INTERFACE ===
    // ========================================================================
    [[nodiscard]] auto CanHandleAreas(const tArea *areas, int numAreas) -> eOsdError override;
    auto Flush() -> void override; ///< Write dirty pixmaps/bitmaps into the dumb buffer, then signal display

  private:
    friend class cVaapiOsdProvider; ///< CreateOsd() calls Allocate(); UpdateOsd() calls GetFramebufferId/Width/Height

    // ========================================================================
    // === INTERNAL METHODS ===
    // ========================================================================
    [[nodiscard]] auto Allocate() -> bool; ///< Must be called once after construction (CreateOsd does this)
    [[nodiscard]] auto CreateDumbBuffer(uint32_t fbWidth, uint32_t fbHeight) -> bool;
    auto DestroyDumbBuffer() -> void; ///< Idempotent: munmap -> drmModeRmFB -> destroy GEM (reverse alloc order)
    [[nodiscard]] auto GetFramebufferId() const noexcept -> uint32_t;
    [[nodiscard]] auto Height() const noexcept -> int;
    [[nodiscard]] auto Width() const noexcept -> int;

    // ========================================================================
    // === STATE ===
    // ========================================================================
    int drmFd_;                   ///< Borrowed from display; outlives all OSDs (see device Detach order)
    uint32_t framebufferId_{};    ///< KMS FB registered with drmModeAddFB2; 0 = not allocated
    uint32_t gemHandle_{};        ///< GEM dumb-buffer handle; 0 = not allocated
    uint32_t height_{};           ///< FB height = screen height - Top() (not the visible pixmap height)
    size_t mappedSize_{};         ///< Byte length of the mmap region (driver-aligned, >= stride*height)
    uint8_t *pixels_{};           ///< mmap'd ARGB8888 scanout memory; nullptr = released or not yet allocated
    cVaapiOsdProvider *provider_; ///< Borrowed; used for UpdateOsd/HideOsd -- valid while device is attached
    uint32_t stride_{};           ///< Row pitch in bytes from DRM_IOCTL_MODE_CREATE_DUMB; may exceed width*4
    uint32_t width_{};            ///< FB width = screen width - Left()
};

// ============================================================================
// === GLOBAL STATE ===
// ============================================================================

/// Singleton provider pointer -- kept here so device.cpp can swap the display under it across
/// SVDRP DETA/ATTA without re-registering the provider with VDR (which has no detach hook).
extern cOsdProvider *osdProvider;

#endif // VDR_VAAPIVIDEO_OSD_H
