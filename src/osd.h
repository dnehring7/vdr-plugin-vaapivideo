// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file osd.h
 * @brief OSD overlay using DRM dumb buffers
 *
 * Implements VDR's cOsd and cOsdProvider interfaces for DRM plane rendering.
 *
 * Dual-path design:
 * - TrueColor skins (32bpp): RenderPixmaps() returns a composited ARGB pixmap
 *   that is memcpy'd scanline-by-scanline into the DRM framebuffer.
 * - Indexed-color skins (4/8bpp): Flush() iterates dirty bitmaps and converts
 *   each pixel via VDR's GetColor() palette lookup into ARGB8888.
 *
 * ProvidesTrueColor() returns true so modern skins use the fast pixmap path,
 * while legacy indexed-color skins fall back to the per-pixel palette path.
 */

#ifndef VDR_VAAPIVIDEO_OSD_H
#define VDR_VAAPIVIDEO_OSD_H

#include "common.h"
#include "display.h"

class cVaapiOsd;

// ============================================================================
// === OSD PROVIDER ===
// ============================================================================

/**
 * @class cVaapiOsdProvider
 * @brief VDR OSD provider that creates cVaapiOsd instances backed by DRM dumb
 * buffers
 */
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
    auto AttachDisplay(cVaapiDisplay *display) noexcept -> void; ///< Reconnect to a (new) display instance
    auto DetachDisplay() noexcept -> void;                       ///< Sever the display reference for safe shutdown

  protected:
    // ========================================================================
    // === VDR INTERFACE ===
    // ========================================================================
    [[nodiscard]] auto CreateOsd(int left, int top, uint level)
        -> cOsd * override; ///< Allocate and return a new OSD instance
    [[nodiscard]] auto ProvidesTrueColor() -> bool override {
        return true;
    } ///< Advertise 32bpp TrueColor support to VDR

  private:
    friend class cVaapiOsd; ///< Needs HideOsd()/GetDisplay()/UpdateOsd() access from ~cVaapiOsd() and Flush()

    // ========================================================================
    // === INTERNAL METHODS ===
    // ========================================================================
    [[nodiscard]] auto GetDisplay() const noexcept -> cVaapiDisplay *; ///< Return the borrowed display pointer
    auto HideOsd(uint32_t fbId) -> void; ///< Remove the OSD overlay only if @p fbId is the currently displayed one
    auto UpdateOsd(cVaapiOsd &osd) const -> void; ///< Push OSD geometry and framebuffer to display

    // ========================================================================
    // === STATE ===
    // ========================================================================
    cVaapiDisplay *display_; ///< Borrowed display reference; nulled by DetachDisplay()
};

// ============================================================================
// === OSD ===
// ============================================================================

/**
 * @class cVaapiOsd
 * @brief VDR OSD backed by a DRM dumb buffer mapped into process memory
 */
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
    [[nodiscard]] auto CanHandleAreas(const tArea *areas, int numAreas)
        -> eOsdError override;     ///< Accept any area configuration; TrueColor path handles all formats
    auto Flush() -> void override; ///< Composite VDR pixmaps/bitmaps into the DRM framebuffer and present

  private:
    friend class cVaapiOsdProvider; ///< Needs Allocate()/GetFramebufferId()/Height()/Width() from CreateOsd() and
                                    ///< UpdateOsd()

    // ========================================================================
    // === INTERNAL METHODS ===
    // ========================================================================
    [[nodiscard]] auto Allocate() -> bool; ///< Create and mmap the DRM dumb buffer; must be called once
                                           ///< after construction
    [[nodiscard]] auto CreateDumbBuffer(uint32_t fbWidth, uint32_t fbHeight)
        -> bool;                      ///< Allocate DRM dumb buffer, register FB, and mmap pixel memory
    auto DestroyDumbBuffer() -> void; ///< Unmap pixel memory, remove FB registration, and free GEM handle
    [[nodiscard]] auto GetFramebufferId() const noexcept -> uint32_t; ///< Return the DRM framebuffer ID registered with
                                                                      ///< drmModeAddFB2()
    [[nodiscard]] auto Height() const noexcept -> int;                ///< Return the framebuffer height in pixels
    [[nodiscard]] auto Width() const noexcept -> int;                 ///< Return the framebuffer width in pixels

    // ========================================================================
    // === STATE ===
    // ========================================================================
    int drmFd_;                   ///< Borrowed DRM file descriptor
    uint32_t framebufferId_{};    ///< DRM framebuffer ID (from drmModeAddFB2)
    uint32_t gemHandle_{};        ///< GEM dumb-buffer handle (from DRM_IOCTL_MODE_CREATE_DUMB)
    uint32_t height_{};           ///< Framebuffer height in pixels
    size_t mappedSize_{};         ///< Byte length of the mmap'd region
    uint8_t *pixels_{};           ///< Start of the mmap'd ARGB8888 pixel memory
    cVaapiOsdProvider *provider_; ///< Borrowed provider reference for UpdateOsd/GetDisplay
    uint32_t stride_{};           ///< Row stride in bytes (pitch returned by DRM_IOCTL_MODE_CREATE_DUMB)
    uint32_t width_{};            ///< Framebuffer width in pixels
};

// ============================================================================
// === GLOBAL STATE ===
// ============================================================================

extern cOsdProvider *osdProvider; ///< Active OSD provider instance (required by VDR plugin API)

#endif // VDR_VAAPIVIDEO_OSD_H
