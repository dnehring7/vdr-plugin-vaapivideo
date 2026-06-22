// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file subtitle.h
 * @brief Text-subtitle converter for the mediaplayer. Exists because VDR core has no text-subtitle
 *        decoder: FFmpeg turns SubRip/ASS/mov_text into plain text, then a pacing thread drives it onto
 *        the OSD against the playback clock, mirroring core's cDvbSubtitleConverter.
 *
 * Why a separate thread: a cue must show/hide on the clock, not when its packet happens to arrive --
 * so Convert() only decodes+queues, and Action() owns all display timing.
 *
 * Why the core OSD (NewOsd(OSD_LEVEL_SUBTITLES) + DrawText) and not a dedicated KMS plane: it reuses
 * the proven DVB rendering path and shares the single OSD plane the display already scans out.
 *
 * Why the locking is shaped the way it is: Open/Close/Convert/Reset/Shutdown all run on the demux
 * thread; Action() is the SOLE owner of the cOsd, so OSD access needs no lock; the only state crossing
 * threads is the cue queue (cueMutex_) and the hideRequested_ atomic.
 */

#ifndef VDR_VAAPIVIDEO_SUBTITLE_H
#define VDR_VAAPIVIDEO_SUBTITLE_H

#include "common.h"

// C++ Standard Library
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

// VDR
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <vdr/thread.h>
#pragma GCC diagnostic pop

class cVaapiDevice;
class cOsd;

// ============================================================================
// === SUBTITLE CONVERTER CLASS ===
// ============================================================================

/// Threaded text-subtitle converter: decodes SubRip/ASS/mov_text and paces cues onto the OSD. One
/// instance per player. Thread safety: the public API (Open/Close/Convert/Reset/Shutdown) is demux-thread
/// only and Action() is the sole owner of the cOsd -- see the @file block above for the threading rationale.
class cSubtitleConverter final : public cThread {
  public:
    /// A rendered line plus its color. Public only so the file-local markup parser in subtitle.cpp can
    /// populate it. Color is per-line because broadcaster SRT/ASS can switch <font color> mid-cue.
    struct Line {
        std::string text;       ///< line text, markup already stripped
        uint32_t rgb{0xFFFFFF}; ///< 0xRRGGBB foreground, honored only when hasColor (else Setup's white)
        bool hasColor{false};   ///< whether the cue specified a color at all
    };

    // ========================================================================
    // === SPECIAL MEMBERS ===
    // ========================================================================
    // Non-copyable/movable: owns a thread + an OSD, neither of which can be duplicated or relocated.
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
    struct Cue {
        int64_t start90k{};      ///< 90 kHz window start; Action() shows the cue while start <= clock < end
        int64_t end90k{};        ///< 90 kHz window end
        std::vector<Line> lines; ///< the cue's text, one entry per source line
    };

    // ========================================================================
    // === INTERNAL METHODS (Action thread) ===
    // ========================================================================
    /// Draw @p cue bottom-centered. Reuses the live OSD when the next cue is the same shape, to avoid a
    /// per-cue buffer alloc/free. Returns false on draw failure so Action() retries instead of dropping it.
    [[nodiscard]] auto ShowCue(const Cue &cue) -> bool;
    auto HideCue() -> void; ///< Drop the OSD, releasing the shared plane for menus / the replay bar.

    // ========================================================================
    // === STATE ===
    // ========================================================================
    cVaapiDevice *device_; ///< borrowed (player outlives us): clock + OSD-size source
    std::unique_ptr<AVCodecContext, FreeAVCodecContext> codecCtx_; ///< decoder; demux-thread only, hence unlocked
    cOsd *osd_{};                                                  ///< live OSD; Action-thread only, hence unlocked
    // Geometry of the live OSD: a same-shape successor cue reuses it (clear + redraw) instead of
    // realloc'ing the dumb buffer + KMS framebuffer. Action-thread only, valid only while osd_ != nullptr.
    int osdTop_{};
    int osdAreaWidth_{};
    int osdAreaHeight_{};
    int64_t shownStart90k_{AV_NOPTS_VALUE}; ///< start90k on screen now; dedups redraws of the same cue

    std::deque<Cue> cues_;    ///< pending cues, kept in demux (ascending start) order for Action()'s scan
    mutable cMutex cueMutex_; ///< the only lock: guards cues_ across the demux <-> Action handoff
    std::atomic<bool> hideRequested_{false}; ///< Close()/Reset() ask Action() to clear the screen on its own thread
    std::atomic<bool> stopping_{false};      ///< Shutdown() tells Action() to exit
    std::atomic<bool> hasExited_{true};      ///< Set by Action() before it returns; Shutdown() checks it after Cancel()
    std::atomic<bool> hasShutdown_{false};   ///< guards double teardown (dtor + an explicit Shutdown())
};

#endif // VDR_VAAPIVIDEO_SUBTITLE_H
