// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file subtitle.h
 * @brief Subtitles: the mediaplayer's converter and the DVD subpicture decoder.
 *
 * cSubtitleConverter (mediaplayer): FFmpeg turns SubRip/ASS/mov_text into plain text (VDR core has no
 * text-subtitle decoder) and dvb_subtitle into palette bitmaps; a pacing thread then drives each cue onto
 * the OSD against the playback clock, mirroring core's cDvbSubtitleConverter (whose PES/STC pipeline this
 * self-demuxing libav player cannot reuse).
 *
 * Why a separate thread: a cue must show/hide on the clock, not when its packet happens to arrive --
 * so Convert() only decodes+queues, and Action() owns all display timing.
 *
 * Why the core OSD (NewOsd(OSD_LEVEL_SUBTITLES) + DrawText/DrawScaledBitmap) and not a dedicated KMS
 * plane: it reuses the proven DVB rendering path and shares the single OSD plane the display scans out.
 *
 * Why the locking is shaped the way it is: Open/Close/Convert/Reset/Shutdown all run on the demux
 * thread; Action() is the SOLE owner of the cOsd, so OSD access needs no lock; the only state crossing
 * threads is the cue queue (cueMutex_) and the hideRequested_ atomic.
 *
 * cVaapiSpuDecoder (dvd plugin): VDR's own DVD subpicture decoder, re-timed against the frame on screen.
 */

#ifndef VDR_VAAPIVIDEO_SUBTITLE_H
#define VDR_VAAPIVIDEO_SUBTITLE_H

#include "common.h"

// C++ Standard Library
#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// VDR
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <vdr/dvbspu.h>
#include <vdr/thread.h>
#include <vdr/tools.h>
#pragma GCC diagnostic pop

class cVaapiDevice;
class cOsd;
class cBitmap;

// ============================================================================
// === SUBTITLE CONVERTER CLASS ===
// ============================================================================

/// Threaded subtitle converter: decodes text (SubRip/ASS/mov_text) and DVB bitmap cues and paces them
/// onto the OSD. One instance per player. Thread safety: the public API (Open/Close/Convert/Reset/Shutdown)
/// is demux-thread only and Action() is the sole owner of the cOsd -- see the file header for rationale.
class cSubtitleConverter final : public cThread {
  public:
    /// A rendered line plus its color. Public only so the file-local markup parser in subtitle.cpp can
    /// populate it. Color is per-line because broadcaster SRT/ASS can switch `<font color>` mid-cue.
    struct Line {
        std::string text;       ///< line text, markup already stripped
        uint32_t rgb{0xFFFFFF}; ///< 0xRRGGBB foreground, honored only when hasColor (else Setup's white)
        bool hasColor{false};   ///< whether the cue specified a color at all
    };

    // ========================================================================
    // === SPECIAL MEMBERS ===
    // ========================================================================
    // Non-copyable/movable: owns a thread + an OSD, neither of which can be duplicated or relocated.
    /// Starts the pacing thread immediately (see the note above the definition); @p device is borrowed.
    explicit cSubtitleConverter(cVaapiDevice *device);
    ~cSubtitleConverter() noexcept override;
    cSubtitleConverter(const cSubtitleConverter &) = delete;
    cSubtitleConverter(cSubtitleConverter &&) noexcept = delete;
    auto operator=(const cSubtitleConverter &) -> cSubtitleConverter & = delete;
    auto operator=(cSubtitleConverter &&) noexcept -> cSubtitleConverter & = delete;

    // ========================================================================
    // === PUBLIC API (demux thread) ===
    // ========================================================================
    /// Point the converter at a subtitle stream. Re-open drops the old decoder/cues/text because a new
    /// stream invalidates them; returns false (caller leaves subtitles off) when the codec has no decoder.
    [[nodiscard]] auto Open(const AVCodecParameters *codecpar) -> bool;
    /// Turn subtitles off entirely. Unlike Reset() it also frees the decoder -- the stream is going away,
    /// so a later Open() must rebuild from scratch. Idempotent.
    auto Close() -> void;
    /// Decode one packet into cues and queue them. Never touches the screen (that is Action()'s job) and
    /// runs inline on the demux thread because subtitle decode is cheap and sparse. No-op until Open().
    auto Convert(const AVPacket *packet) -> void;
    /// Discard queued cues + hide the overlay after a seek, but keep the decoder: same stream, only the
    /// old cue timings no longer match the new position.
    auto Reset() -> void;
    /// Tear down: join Action() before freeing the OSD it solely owns, then drop the decoder/cues.
    /// Idempotent so the dtor and an explicit caller can both invoke it.
    auto Shutdown() -> void;

  protected:
    auto Action() -> void override; ///< Pacing loop; clock-driven so cues land in sync, not at arrival time.

  private:
    // ========================================================================
    // === INTERNAL TYPES ===
    // ========================================================================
    /// One DVB region: a palette bitmap at a fixed spot on the canvas. shared_ptr so Action()'s
    /// per-tick Cue copy doesn't deep-copy the pixel buffer.
    struct BitmapRegion {
        int x{};                               ///< top-left X on the subtitle canvas (pre-scale)
        int y{};                               ///< top-left Y on the subtitle canvas (pre-scale)
        std::shared_ptr<const cBitmap> bitmap; ///< VDR indexed bitmap (palette + pixels); read-only after decode
    };

    /// A cue is either text (lines) or bitmap (regions) -- never both: a track is purely text or
    /// purely DVB bitmap, so Action() picks the renderer by which vector is non-empty.
    struct Cue {
        uint64_t serial{};                 ///< monotonic queue id; dedups redraws even when two pages share a PTS
        int64_t start90k{};                ///< 90 kHz window start; Action() shows while start <= clock < end
        int64_t end90k{};                  ///< 90 kHz window end
        std::vector<Line> lines;           ///< text path: the cue's text, one entry per source line
        std::vector<BitmapRegion> regions; ///< bitmap path: DVB regions, empty for text cues
        int canvasW{};                     ///< subtitle display width (bitmap cues), for OSD scaling
        int canvasH{};                     ///< subtitle display height (bitmap cues), for OSD scaling
    };

    // ========================================================================
    // === INTERNAL METHODS (Action thread) ===
    // ========================================================================
    /// Draw text @p cue bottom-centered. Reuses the live OSD when the next cue is the same shape, to
    /// avoid a per-cue buffer alloc/free. Returns false on draw failure so Action() retries instead
    /// of dropping it.
    [[nodiscard]] auto ShowCue(const Cue &cue) -> bool;
    /// Draw DVB bitmap @p cue: scale the subtitle canvas to the OSD and blit each region via
    /// cOsd::DrawScaledBitmap (the core DVB path). Reuses the live OSD when the next page has the same
    /// geometry, else allocates fresh. Returns false on draw failure, like ShowCue.
    [[nodiscard]] auto ShowBitmapCue(const Cue &cue) -> bool;
    auto HideCue() -> void; ///< Drop the OSD, releasing the shared plane for menus / the replay bar.

    // ========================================================================
    // === STATE ===
    // ========================================================================
    cVaapiDevice *device_; ///< borrowed (player outlives us): clock + OSD-size source
    std::unique_ptr<AVCodecContext, FreeAVCodecContext> codecCtx_; ///< decoder; demux-thread only, hence unlocked
    cOsd *osd_{};                                                  ///< live OSD; Action-thread only, hence unlocked
    /// Geometry of the live OSD: a same-shape successor cue reuses it (clear + redraw) instead of
    /// realloc'ing the dumb buffer + KMS framebuffer. Action-thread only, valid only while osd_ != nullptr.
    /// osdLeft_ is 0 for text bands (full-width, centered text); bitmap cues place the OSD at the scaled
    /// region bounding box, so reuse must match the left edge too.
    int osdLeft_{};             ///< Left edge of the live OSD (px)
    int osdTop_{};              ///< Top edge of the live OSD (px)
    int osdAreaWidth_{};        ///< Width of the live OSD (px)
    int osdAreaHeight_{};       ///< Height of the live OSD (px)
    uint64_t shownCueSerial_{}; ///< Cue::serial on screen now (0 = nothing); dedups redraws. Action-thread only.
    uint64_t nextCueSerial_{1}; ///< next serial to hand out; demux-thread only. 0 is reserved for "nothing shown".

    std::deque<Cue> cues_;    ///< pending cues, kept in demux (ascending start) order for Action()'s scan
    mutable cMutex cueMutex_; ///< the only lock: guards cues_ across the demux <-> Action handoff
    std::atomic<bool> hideRequested_{false}; ///< Close()/Reset() ask Action() to clear the screen on its own thread
    std::atomic<bool> stopping_{false};      ///< Shutdown() tells Action() to exit
    std::atomic<bool> hasExited_{true};      ///< Set by Action() before it returns; Shutdown() checks it after Cancel()
    std::atomic<bool> hasShutdown_{false};   ///< guards double teardown (dtor + an explicit Shutdown())
};

// ============================================================================
// === DVD SUBPICTURE DECODER ===
// ============================================================================

inline constexpr size_t SUBTITLE_SPU_CLUT_ENTRIES = 16; ///< DVD subpicture colour lookup table size

/// VDR's DVD subpicture decoder (dvd plugin subtitles and menu buttons), timed against the frame on screen.
/// The player times subpictures by the packets it reads, and the decode buffer runs those over a second ahead
/// of the display -- every cue would show and hide that much early.
class cVaapiSpuDecoder final : public cDvbSpuDecoder {
  public:
    /// @p device is borrowed: it owns this decoder.
    explicit cVaapiSpuDecoder(cVaapiDevice &device) : device_(device) {}
    ~cVaapiSpuDecoder() noexcept override = default;
    cVaapiSpuDecoder(const cVaapiSpuDecoder &) = delete;
    cVaapiSpuDecoder(cVaapiSpuDecoder &&) noexcept = delete;
    auto operator=(const cVaapiSpuDecoder &) -> cVaapiSpuDecoder & = delete;
    auto operator=(cVaapiSpuDecoder &&) noexcept -> cVaapiSpuDecoder & = delete;

    // No locking: the dvd player makes every call under its thread lock.

    /// @p pts is the player's read position, 0 its still-frame tick (no position). The base decoder runs
    /// commands up to ScreenTime() instead.
    auto setTime(uint32_t pts) -> int override;
    /// Queues the subpicture until the screen reaches @p pts. The base decoder holds a single one, so taking it
    /// at read time would drop a cue that is still waiting for the screen. Takes ownership of @p buf (new[]).
    auto processSPU(uint32_t pts, uint8_t *buf, bool AllowedShow) -> void override;
    /// Kept for the subpictures read from now on: a queued one must draw with the CLUT of its own read time.
    auto setPalette(uint32_t *pal) -> void override;
    auto Empty() -> void override; ///< Also drops the queued subpictures (seek, new title).
    /// Not while another OSD is open (VDR's audio or subtitle menu): VDR would log an error and hand out a dummy
    /// OSD, and drawing into that counts as visible, so the player's redraw after the menu would be skipped.
    auto Draw() -> void override;

  private:
    using Clut = std::array<uint32_t, SUBTITLE_SPU_CLUT_ENTRIES>; ///< A DVD colour lookup table (YCrCb entries)

    /// A subpicture read ahead of the screen, held until its display time.
    struct PendingSpu {
        uint32_t pts{0};                 ///< Display time of the subpicture
        std::unique_ptr<uint8_t[]> data; ///< Complete SPU as the player assembled it
        bool allowedShow{true};          ///< Passed through to the base decoder
        std::optional<Clut> clut;        ///< CLUT at read time; none before the player set one
    };

    [[nodiscard]] auto ScreenTime() -> std::optional<uint32_t>; ///< How far commands may run; none = not yet
    auto Advance() -> int;                                      ///< Installs due subpictures, runs their commands
    auto InstallDueSpus(uint32_t time) -> void;                 ///< Hands queued subpictures due by @p time to the base

    cVaapiDevice &device_;               ///< Source of the presentation clock
    int64_t lastClock_{-1};              ///< Presentation clock at the previous call; -1 = none
    cTimeMs clockMovedAt_{0};            ///< When lastClock_ last changed
    std::deque<PendingSpu> pendingSpus_; ///< Read but not yet due, in stream order
    std::optional<Clut> clut_;           ///< CLUT at the player's read position
    std::optional<uint32_t> readPts_;    ///< Player's read position (an SPU's PTS may be a genuine 0)
    bool inBase_{false};                 ///< Inside a base-class call: its setTime()/Empty() are not the player's
};

#endif // VDR_VAAPIVIDEO_SUBTITLE_H
