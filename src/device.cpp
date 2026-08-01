// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file device.cpp
 * @brief cDevice subclass: PES routing, codec detection, audio-track plumbing, lifecycle.
 *
 * Threading model:
 *   - Main VDR thread: Clear/SetPlayMode/TrickSpeed/Freeze, audio-track hooks, OSD queries.
 *   - Receiver / dvbplayer thread: PlayVideo / PlayAudio / Poll / Flush.
 *   - Decoder/audio/display threads: owned by their subsystems.
 *
 * Subsystem lifetime is bound to a tri-state initState (0=detached, 1=in-progress, 2=ready)
 * managed via CAS in AttachHardware() / Detach(). The decoder Action thread holds raw
 * pointers to display + audioProcessor, so Stop() MUST tear them down in the order
 * decoder -> display -> audioProcessor (see Stop() comment).
 *
 * The Detach() path also has a non-obvious upstream-shutdown ordering hazard with
 * cTransferControl that the destructor doesn't share -- documented inline.
 */

#include "device.h"
#include "audio.h"
#include "caps.h"
#include "common.h"
#include "config.h"
#include "decoder.h"
#include "display.h"
#include "mediaplayer.h"
#include "osd.h"
#include "pes.h"
#include "stream.h"

// C++ Standard Library
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <iterator>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// POSIX
#include <fcntl.h>
#include <linux/vt.h>
#include <pthread.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

// DRM/KMS
#include <libdrm/drm.h>
#include <libdrm/drm_mode.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

// FFmpeg
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/codec.h>
#include <libavcodec/codec_id.h>
#include <libavcodec/packet.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/avutil.h>
#include <libavutil/buffer.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/mem.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
}
#pragma GCC diagnostic pop

// VAAPI (only base types; profile/VPP enumeration lives in caps.cpp)
#include <va/va.h>

// VDR
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <vdr/channels.h>
#include <vdr/config.h>
#include <vdr/device.h>
#include <vdr/epg.h>
#include <vdr/font.h>
#include <vdr/i18n.h>
#include <vdr/osd.h>
#include <vdr/player.h>
#include <vdr/thread.h>
#include <vdr/tools.h>
#pragma GCC diagnostic pop

// ============================================================================
// === HELPER FUNCTIONS ===
// ============================================================================

namespace {

// AUDIO_QUEUE_HIGHWATER paces dvbplayer/PES replay (PlayAudio/Poll); AUDIO_QUEUE_HIGHWATER_MEDIAPLAYER
// paces the single-cursor mediaplayer demux. MEDIAPLAYER_MAX_LOOKAHEAD_90K is the coarser PTS-distance
// brake (so a TS mux interleave offset can't over-fill the decoder jitterBuf after a seek).

// Jitter-buffer high-water that backpressures the mediaplayer demux ONLY while the audio master clock is
// NOPTS (the post-flush window before the first decoded sample anchors the clock). For AUDIO streams
// AUDIO_QUEUE_HIGHWATER already covers this; it matters for VIDEO-ONLY streams (no audio clock ever), where
// it is the sole video-depth gate stopping the file-read-speed demuxer from flooding. Once the clock
// anchors, IsMediaPlayerBackpressured() drops it (a video gate on the single demux cursor would starve the
// audio queue -- sustained-stutter-after-seek bug). Derived from the reserve cap (3/4) and kept here, the
// only user, so it stays coupled to the buffer it protects and below the cap by construction. The
// static_assert must trip before the decode-ahead reserve hits its hard cap (else the drop-oldest runaway
// guard fires first and the demuxer never throttles).
constexpr size_t MEDIAPLAYER_JITTERBUF_BACKPRESSURE_FRAMES = (DECODER_RESERVE_HARD_CAP * 3) / 4;
static_assert(MEDIAPLAYER_JITTERBUF_BACKPRESSURE_FRAMES < DECODER_RESERVE_HARD_CAP,
              "mediaplayer backpressure must engage below the reserve cap");

/// How often PlayAudio re-checks the present EPG event while a radio splash is on screen. Program
/// boundaries land on minute scales, so a couple of seconds is responsive without taxing the
/// Schedules read-lock on the audio thread.
constexpr int RADIO_SPLASH_POLL_MS = 2000;

/// radioSplashEventId cache sentinels. DVB event ids are 16-bit, so top-of-range uint32 values can
/// never collide with a real id. EMPTY_TEXT = "a blank frame is queued" (distinct from EPG event 0,
/// so a blank->caption transition still repaints). DIRTY = "force a repaint next poll" -- published
/// before a forced submit so that, if the submit fails, the poll retries instead of matching the key.
constexpr uint32_t RADIO_SPLASH_EMPTY_TEXT_ID = std::numeric_limits<uint32_t>::max();
constexpr uint32_t RADIO_SPLASH_DIRTY_ID = RADIO_SPLASH_EMPTY_TEXT_ID - 1;

/// Grace period after a channel switch before a video stream that never decodes is declared
/// encrypted/undecodable and the on-screen notice is shown. Also the radio black-frame delay:
/// SetPlayMode arms both watchdogs from this constant so the two graces cannot drift apart.
constexpr int ENCRYPTED_NOTICE_DELAY_MS = 3000;

/// Fallback ES window for DetectAudioCodec(), used only when a single payload is inconclusive.
/// Sized for AAC-LATM alone: its AudioMuxElements span PES boundaries, so the LOAS sync chains
/// only across payloads and needs ~2 frames (~1 KB) visible at once. ADTS/MP2/AC-3 resolve on one
/// payload and must NOT reach here -- the window's front-erase lands mid-frame and a ~1.5 KB AC-3
/// frame can't fit a corroborating pair in 2 KB.
constexpr size_t AUDIO_DETECT_WINDOW = 2048;

// --- Runtime display-mode switching (stability / rate-limit / matcher tolerances) ---
constexpr uint64_t DISPLAY_MODE_STABLE_MS =
    1500; ///< How long a reactively observed format must hold before it may drive a modeset.
          ///< Live-TV formats churn faster than this while the tuner settles, so only the one the
          ///< viewer actually landed on gets through.
constexpr uint64_t DISPLAY_MODE_MIN_INTERVAL_MS =
    3000; ///< Minimum gap between two applied changes. Each modeset costs an HDMI link retrain
          ///< (~0.5-1 s of black), so back-to-back switches must be impossible even if the
          ///< stability gate is somehow satisfied twice in a row.
constexpr uint64_t DISPLAY_MODE_IDLE_RESTORE_MS =
    5000; ///< No stream format published for this long after a stop: hand the output back to the
          ///< default. Playback can end into a source that never decodes anything (radio,
          ///< scrambled, no free tuner), where nothing else would move the mode off the one the
          ///< previous stream installed.
constexpr uint32_t DISPLAY_MODE_EXACT_TOLERANCE_PPM =
    200; ///< Tier-A rate tolerance (0.02%): rounding noise only. Keeps 59.94 and 60 distinct.
constexpr uint32_t DISPLAY_MODE_LOOSE_TOLERANCE_PPM =
    5000; ///< Tier-B rate tolerance (0.5%): only consulted when tier A finds nothing. Lets 59.94
          ///< content land on a 60 Hz-only panel and 23.976 on a 24 Hz-only panel.
constexpr uint32_t DISPLAY_MODE_MAX_RATE_MULTIPLE = 8; ///< Highest k considered in the k*source refresh search.

// === VT helpers =============================================================
// Startup + ATTA: foreground VDR's VT (stdin) so the kernel delivers keypresses
// to VDR's KBD. DETA: yield to tty1 so the user lands on getty. Needs the
// systemd drop-in (TTYPath=/dev/ttyN + AmbientCapabilities=CAP_SYS_TTY_CONFIG);
// failures log once at INFO and fall back to manual Ctrl+Alt+F<n>. See README.
//
// VT_ACTIVATE/VT_WAITACTIVE work on any VT fd, so STDIN_FILENO is used directly
// (set up by TTYPath=) and /dev/tty0 is left alone -- no udev rule needed.

constexpr int VT_SWITCH_TIMEOUT_MS = 1500; ///< Cap on VT_WAITACTIVE polling. VT_PROCESS-mode owners
                                           ///< that refuse to release would otherwise block startup forever
                                           ///< and look like a 60 s "plugin hang" until the watchdog fires.

std::atomic<bool> capWarned{false}, noVtHinted{false};

[[nodiscard]] auto OwnVt() -> int {
    // TTY major=4, minor=N for /dev/ttyN (N=1..63); minor 0 is the current-VT alias.
    struct stat st{};
    if (fstat(STDIN_FILENO, &st) != 0 || !S_ISCHR(st.st_mode) || major(st.st_rdev) != 4) {
        return 0;
    }
    const auto vt = static_cast<int>(minor(st.st_rdev));
    return (vt >= 1 && vt <= 63) ? vt : 0;
}

[[nodiscard]] auto SwitchToVt(int vt) -> bool {
    if (ioctl(STDIN_FILENO, VT_ACTIVATE, vt) != 0) {
        const int err = errno;
        if (bool expected = false; capWarned.compare_exchange_strong(expected, true)) {
            isyslog("vaapivideo/device: VT_ACTIVATE(%d) denied (%s) -- VT auto-management disabled, "
                    "use Ctrl+Alt+F<n>; see README 'Console and keyboard integration'",
                    vt, std::strerror(err));
        } else {
            dsyslog("vaapivideo/device: VT_ACTIVATE(%d) denied (%s)", vt, std::strerror(err));
        }
        return false;
    }
    // Poll VT_GETSTATE instead of blocking on VT_WAITACTIVE: the latter can hang forever in
    // VT_PROCESS mode if the owning process never releases. Bounded wait keeps startup non-fatal.
    const cTimeMs timeout(VT_SWITCH_TIMEOUT_MS);
    while (!timeout.TimedOut()) {
        vt_stat state{};
        if (ioctl(STDIN_FILENO, VT_GETSTATE, &state) == 0 && static_cast<int>(state.v_active) == vt) {
            return true;
        }
        cCondWait::SleepMs(10);
    }

    isyslog("vaapivideo/device: VT%d activation timed out after %d ms -- continuing without waiting", vt,
            VT_SWITCH_TIMEOUT_MS);
    return false;
}

[[nodiscard]] auto ActivateOwnVt() -> bool {
    const int vt = OwnVt();
    if (vt == 0) {
        if (bool expected = false; noVtHinted.compare_exchange_strong(expected, true)) {
            isyslog("vaapivideo/device: stdin is not a VT -- KBD remote and VT auto-management "
                    "disabled; see README 'Console and keyboard integration'");
        }
        return true;
    }
    if (!SwitchToVt(vt)) {
        return false;
    }
    isyslog("vaapivideo/device: console VT%d activated for keyboard input", vt);
    return true;
}

[[nodiscard]] auto LeaveOwnVt() -> bool {
    const int ownVt = OwnVt();
    if (ownVt == 0) {
        return true;
    }
    int targetVt = (ownVt == 1) ? 2 : 1;
    if (const char *env = std::getenv("VDR_CONSOLE_TTY"); env != nullptr) {
        char *endp = nullptr;
        const long n = std::strtol(env, &endp, 10);
        if (endp != env && *endp == '\0' && n >= 1 && n <= 63 && n != ownVt) {
            targetVt = static_cast<int>(n);
        }
    }
    if (!SwitchToVt(targetVt)) {
        return false;
    }
    isyslog("vaapivideo/device: yielded VT%d -> VT%d for text console (DETA)", ownVt, targetVt);
    return true;
}

} // namespace

// ============================================================================
// === DRM DEVICES CLASS ===
// ============================================================================

DrmDevices::~DrmDevices() noexcept {
    if (!deviceList.empty()) {
        drmFreeDevices(deviceList.data(), static_cast<int>(deviceList.size()));
    }
}

[[nodiscard]] auto DrmDevices::begin() -> std::vector<drmDevicePtr>::iterator { return deviceList.begin(); }

[[nodiscard]] auto DrmDevices::end() -> std::vector<drmDevicePtr>::iterator { return deviceList.end(); }

[[nodiscard]] auto DrmDevices::Enumerate() -> bool {
    // drmFreeDevices walks libdrm's per-entry refs; must be called before reuse.
    if (!deviceList.empty()) {
        drmFreeDevices(deviceList.data(), static_cast<int>(deviceList.size()));
        deviceList.clear();
    }

    deviceList.resize(64); // Real systems have 1-4 nodes; 64 is a safe upper bound.
    const int result = drmGetDevices2(0, deviceList.data(), static_cast<int>(deviceList.size()));

    if (result > 0) {
        deviceList.resize(static_cast<size_t>(result));
        dsyslog("vaapivideo/device: found %d DRM device(s)", result);
    } else {
        if (result < 0) {
            esyslog("vaapivideo/device: drmGetDevices2 failed (%s)", std::strerror(-result));
        } else {
            dsyslog("vaapivideo/device: no DRM devices found");
        }
        deviceList.clear();
    }
    return HasDevices();
}

[[nodiscard]] auto DrmDevices::HasDevices() const noexcept -> bool { return !deviceList.empty(); }

// ============================================================================
// === VAAPI DEVICE CLASS ===
// ============================================================================

namespace {

constexpr uint8_t SPLASH_LUMA_BLACK = 16;  ///< NV12 TV-range black (matches SubmitBlackFrame fill).
constexpr uint8_t SPLASH_LUMA_WHITE = 235; ///< NV12 TV-range white for fully-covered glyph pixels.

/// Bake optional centered caption text into NV12 luma (chroma stays neutral, so glyphs render gray);
/// '\n' splits into block-centered lines. CPU-only. Safe off the main thread (radio/encrypted paths):
/// cFont::CreateFont() yields a self-contained font with its own FT_Library/FT_Face, sharing no global
/// FreeType state with VDR's OSD fonts -- so do not be tempted to cache/share one instance.
auto DrawCenteredLuma(AVFrame *nv12, std::string_view text) -> void {
    if (text.empty() || nv12 == nullptr || nv12->data[0] == nullptr) [[unlikely]] {
        return;
    }
    const int w = nv12->width;
    const int h = nv12->height;

    // Scale with output height; SD modes still need a readable floor.
    const int fontHeight = std::max(24, h / 24);
    const std::unique_ptr<cFont> font{cFont::CreateFont(Setup.FontOsd, fontHeight)};
    if (!font) [[unlikely]] {
        esyslog("vaapivideo/device: splash font creation failed");
        return;
    }
    const int lineHeight = font->Height();
    if (lineHeight <= 0) [[unlikely]] {
        return;
    }

    // DrawText is single-line; count lines up front to vertically center the whole block.
    const auto lineCount = static_cast<int>(std::ranges::count(text, '\n')) + 1;
    int lineY = (h - (lineHeight * lineCount)) / 2;

    for (size_t start = 0;;) {
        const size_t nl = text.find('\n', start);
        const std::string line{text.substr(start, (nl == std::string_view::npos ? text.size() : nl) - start)};
        if (const int fullW = font->Width(line.c_str()); fullW > 0) {
            // Clip an over-long line (long channel/EPG title) to the frame; DrawText's Width cap ends
            // on a whole glyph rather than mid-character.
            const int drawW = std::min(fullW, w);
            cBitmap bm(drawW, lineHeight, 8); // 8bpp == the full 256-level antialiasing ramp
            // Keep untouched pixels transparent; else DrawText's first Index() call claims palette
            // slot 0 for a glyph color and the index-0 background would inherit it.
            bm.DrawRectangle(0, 0, drawW - 1, lineHeight - 1, clrTransparent);
            font->DrawText(&bm, 0, 0, line.c_str(), clrWhite, clrTransparent, drawW);

            // startX >= 0 and startX + drawW <= w, so every dstX below is in-frame without clamping.
            const int startX = (w - drawW) / 2;
            for (int gy = 0; gy < lineHeight; ++gy) {
                const int dstY = lineY + gy;
                if (dstY < 0 || dstY >= h) { // a tall multi-line block can overflow the frame
                    continue;
                }
                uint8_t *row = nv12->data[0] + (static_cast<ptrdiff_t>(dstY) * nv12->linesize[0]);
                for (int gx = 0; gx < drawW; ++gx) {
                    // Alpha is the glyph's antialiased coverage -> use it as the luma weight; UV stays neutral.
                    const auto alpha = static_cast<uint8_t>((bm.GetColor(gx, gy) >> 24) & 0xFFU);
                    if (alpha == 0) {
                        continue;
                    }
                    row[startX + gx] = static_cast<uint8_t>(SPLASH_LUMA_BLACK +
                                                            ((alpha * (SPLASH_LUMA_WHITE - SPLASH_LUMA_BLACK)) / 255));
                }
            }
        }
        lineY += lineHeight;
        if (nl == std::string_view::npos) {
            break;
        }
        start = nl + 1;
    }
}

/// First splash line shared by the radio and encrypted screens: "Channel N - Name" (or "Channel N"
/// when the name is missing), so both notices read consistently. Caller must hold LOCK_CHANNELS_READ.
[[nodiscard]] auto FormatChannelLine(const cChannel &channel) -> std::string {
    const char *name = channel.Name();
    return (name != nullptr && *name != '\0') ? std::format("{} {} - {}", tr("Channel"), channel.Number(), name)
                                              : std::format("{} {}", tr("Channel"), channel.Number());
}

} // namespace

[[nodiscard]] auto cVaapiDevice::BuildRadioText(uint32_t &presentEventId) const -> std::string {
    // Radio splash content: "Channel N - Name" on line 1, the present EPG event "start-end title" on
    // line 2. Runs on the caller's thread (PlayAudio / SetPlayMode); takes VDR's list locks in the
    // canonical order
    // (Channels before Schedules) and copies the strings out before the locks drop. Reports the
    // present event id (0 = none) so the caller can detect a program change. Returns empty -- a plain
    // black frame -- during replay or when no channel/EPG is available.
    presentEventId = 0;
    std::string text;
    // Channel name / EPG are meaningful only for live broadcast; during replay CurrentChannel() is a
    // stale live channel, so fall back to a plain black frame. On a live-radio entry where liveMode
    // is not yet latched, the 2 s PlayAudio poll fills the caption in once the first PES arrives.
    if (!liveMode.load(std::memory_order_relaxed)) {
        return text;
    }
    LOCK_CHANNELS_READ;
    const cChannel *channel = Channels->GetByNumber(cDevice::CurrentChannel());
    if (channel == nullptr) [[unlikely]] {
        return text;
    }
    text = FormatChannelLine(*channel); // line 1, consistent with the encrypted notice

    LOCK_SCHEDULES_READ;
    if (const cSchedule *schedule = Schedules->GetSchedule(channel); schedule != nullptr) {
        if (const cEvent *present = schedule->GetPresentEvent(); present != nullptr) {
            presentEventId = present->EventID();
            if (const char *title = present->Title(); title != nullptr && *title != '\0') {
                // Line 2: "11:00-12:00 Title" -- the present event's schedule plus title. GetTimeString()
                // honors the user's 12/24 h setting; the cString temporaries live through the format call.
                const cString start = present->GetTimeString();
                const cString end = present->GetEndTimeString();
                text += std::format("\n{}-{} {}", *start, *end, title);
            }
        }
    }
    return text;
}

auto cVaapiDevice::RefreshRadioSplash(bool force) -> void {
    // force=true clears stale scanout on radio entry (main thread); the poll (audio thread) repaints
    // only when the cached key changes. Cross-thread state rides the atomics; SubmitBlackFrame is
    // serialized internally via vaDriverMutex.
    uint32_t presentEventId = 0;
    const std::string text = BuildRadioText(presentEventId);
    const uint32_t splashEventId = text.empty() ? RADIO_SPLASH_EMPTY_TEXT_ID : presentEventId;
    if (force) {
        // Publish DIRTY (never equal to any real splashEventId) so that if this forced submit fails,
        // the next poll still repaints instead of matching the key and returning early.
        radioSplashActive.store(true, std::memory_order_relaxed);
        radioSplashEventId.store(RADIO_SPLASH_DIRTY_ID, std::memory_order_relaxed);
    } else if (splashEventId == radioSplashEventId.load(std::memory_order_relaxed)) {
        return; // nothing changed since the last paint
    }
    // Publish the depicted key only once the frame actually queued, so a failed submit stays DIRTY.
    if (SubmitBlackFrame(text)) {
        radioSplashEventId.store(splashEventId, std::memory_order_relaxed);
    }
}

auto cVaapiDevice::CheckEncryptionTimeout() -> void {
    // Driven by the decode loop's per-iteration tick (decoder.SetLoopTickCallback), which keeps
    // firing on the ~100 ms idle waits even when a fully scrambled channel delivers no PES at all --
    // the case PlayAudio/PlayVideo never see (plenty for the grace period). The leading load
    // short-circuits the disarmed case. Every disarm below is a CAS against the deadline observed
    // here: this decoder thread outlives play modes, and a plain store could cancel a fresh re-arm
    // by SetPlayMode. A re-arm's deadline lies strictly in the future, so it never matches an
    // observation this tick found expired -- the CAS claims exactly the arm it decided on.
    uint64_t deadline = encryptedDeadlineMs.load(std::memory_order_relaxed);
    if (deadline == 0) {
        return;
    }
    // Any decoding elementary stream proves the CAM descrambles this service (a DVB service
    // scrambles under one ECM) -- the same rule ShowEncryptedScreen applies. Disarm now instead of
    // re-deciding at the deadline.
    if (videoCodecId.load(std::memory_order_relaxed) != AV_CODEC_ID_NONE ||
        audioCodecId.load(std::memory_order_relaxed) != AV_CODEC_ID_NONE) {
        encryptedDeadlineMs.compare_exchange_strong(deadline, 0, std::memory_order_relaxed);
        return;
    }
    if (cTimeMs::Now() < deadline) {
        return; // still inside the grace period
    }
    // Only live broadcast can be encrypted (a recording just fails to decode, and CurrentChannel()
    // would be a stale live channel during replay).
    if (!Transferring()) {
        encryptedDeadlineMs.compare_exchange_strong(deadline, 0, std::memory_order_relaxed);
        return;
    }
    // Claim the one-shot, then paint. No further monitoring needed: once the stream descrambles,
    // PlayVideo's codec detection decodes it and those frames overwrite the notice on screen.
    if (encryptedDeadlineMs.compare_exchange_strong(deadline, 0, std::memory_order_relaxed)) {
        ShowEncryptedScreen();
    }
}

auto cVaapiDevice::ShowEncryptedScreen() -> void {
    // "encrypted/undecodable" means the CAM cannot descramble this channel at all. ANY decoding
    // elementary stream -- video OR audio -- proves it IS descrambling (a DVB service scrambles under
    // one ECM, so clear audio implies the video will follow), so the notice must not fire: a TV channel
    // whose audio is already up while its video is a beat behind (slow keyframe, or the CAM just primed
    // after a replay/mediaplayer handover) is decryptable, not encrypted. Only fire when NOTHING
    // decodes. FTA channels (Ca 0) are excluded below. Codec state is the reliable discriminator, read
    // here under the channel lock.
    std::string text;
    int number = 0;
    {
        LOCK_CHANNELS_READ;
        const cChannel *channel = Channels->GetByNumber(cDevice::CurrentChannel());
        if (channel == nullptr) {
            dsyslog("vaapivideo/device: encrypted watchdog -- no current channel, skipping");
            return;
        }
        number = channel->Number();
        const int vpid = channel->Vpid();
        const int caId = channel->Ca();
        if (caId == 0) {
            dsyslog("vaapivideo/device: encrypted watchdog -- channel %d is FTA (ca=0), skipping", number);
            return;
        }
        const bool decoding = videoCodecId.load(std::memory_order_relaxed) != AV_CODEC_ID_NONE ||
                              audioCodecId.load(std::memory_order_relaxed) != AV_CODEC_ID_NONE;
        if (decoding) {
            dsyslog("vaapivideo/device: encrypted watchdog -- channel %d decrypts (vpid=%d), skipping", number, vpid);
            return;
        }
        text = std::format("{}\n{}", FormatChannelLine(*channel), tr("encrypted"));
    }
    isyslog("vaapivideo/device: channel %d encrypted/undecodable -- showing notice", number);
    // Take ownership of the screen: stop the radio poll repainting its splash over the notice.
    radioSplashActive.store(false, std::memory_order_relaxed);
    static_cast<void>(SubmitBlackFrame(text)); // one-shot notice; a transient submit failure just skips it
}

[[nodiscard]] auto cVaapiDevice::CurrentChannelIsEncrypted() const -> bool {
    LOCK_CHANNELS_READ;
    const cChannel *channel = Channels->GetByNumber(cDevice::CurrentChannel());
    return channel != nullptr && channel->Ca() != 0;
}

auto cVaapiDevice::ResetNoVideoMonitors() noexcept -> void {
    // Single funnel for tearing down both no-video screens (radio splash + encrypted notice) on every
    // lifecycle boundary, so no path forgets a field. DIRTY makes the next radio entry repaint.
    radioBlackPending.store(false, std::memory_order_relaxed);
    radioSplashActive.store(false, std::memory_order_relaxed);
    radioSplashEventId.store(RADIO_SPLASH_DIRTY_ID, std::memory_order_relaxed);
    // Plain store, not CAS: a lifecycle boundary means disarm WHATEVER is armed, by design.
    encryptedDeadlineMs.store(0, std::memory_order_relaxed);
}

auto cVaapiDevice::SubmitBlackFrame(std::string_view centerText) -> bool {
    // One black frame (optional caption) straight to the display, bypassing the decoder. Uses a
    // one-shot hw_frames_ctx because these paths run before/without the decoder's frame pool.
    // False if the frame never queued.
    if (!display || !vaapi.hwDeviceRef) [[unlikely]] {
        return false;
    }

    const auto w = static_cast<int>(display->GetOutputWidth());
    const auto h = static_cast<int>(display->GetOutputHeight());
    if (w <= 0 || h <= 0) [[unlikely]] { // no mode set yet: nothing valid to allocate or draw into
        return false;
    }

    std::unique_ptr<AVBufferRef, FreeAVBufferRef> framesRef{av_hwframe_ctx_alloc(vaapi.hwDeviceRef)};
    if (!framesRef) [[unlikely]] {
        esyslog("vaapivideo/device: black frame hw_frames_ctx alloc failed");
        return false;
    }
    // FFmpeg ABI: AVBufferRef::data points to a typed payload (here AVHWFramesContext) per the
    // hwframe context contract -- the cast is the documented access pattern.
    auto *ctx = reinterpret_cast<AVHWFramesContext *>( // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        framesRef->data);
    ctx->format = AV_PIX_FMT_VAAPI;
    ctx->sw_format = AV_PIX_FMT_NV12;
    ctx->width = w;
    ctx->height = h;
    ctx->initial_pool_size = 1;

    // SW staging frame uploaded to a single VAAPI surface via av_hwframe_transfer_data.
    std::unique_ptr<AVFrame, FreeAVFrame> hwFrame{av_frame_alloc()};
    std::unique_ptr<AVFrame, FreeAVFrame> swFrame{av_frame_alloc()};
    if (!hwFrame || !swFrame) [[unlikely]] {
        esyslog("vaapivideo/device: black frame surface alloc failed");
        return false;
    }

    // vaDriverMutex: hw_frames_ctx_init + get_buffer create/reserve a VAAPI surface, which
    // races the display thread's av_hwframe_map and the decoder's VPP on the iHD driver (not
    // thread-safe on a shared VADisplay). Serialize like MapVaapiFrame / the decoder paths do.
    {
        const cMutexLock vaLock(&display->GetVaDriverMutex());
        if (av_hwframe_ctx_init(framesRef.get()) < 0) [[unlikely]] {
            esyslog("vaapivideo/device: black frame hw_frames_ctx init failed");
            return false;
        }
        if (av_hwframe_get_buffer(framesRef.get(), hwFrame.get(), 0) < 0) [[unlikely]] {
            esyslog("vaapivideo/device: black frame surface alloc failed");
            return false;
        }
    }

    swFrame->format = AV_PIX_FMT_NV12;
    swFrame->width = w;
    swFrame->height = h;
    if (av_frame_get_buffer(swFrame.get(), 0) < 0) [[unlikely]] {
        esyslog("vaapivideo/device: black frame sw buffer alloc failed");
        return false;
    }

    // BT.709 TV-range "black": Y=16, UV=128. Y=0 would clip below black on TV-range
    // sinks and the rest of the pipeline (scale_vaapi out_range=tv) is TV-range too.
    const auto rows = static_cast<size_t>(h);
    std::memset(swFrame->data[0], 16, static_cast<size_t>(swFrame->linesize[0]) * rows);
    std::memset(swFrame->data[1], 128, static_cast<size_t>(swFrame->linesize[1]) * (rows / 2));

    // Optional centered splash text, baked into the luma plane (CPU-only, no VA driver calls).
    DrawCenteredLuma(swFrame.get(), centerText);

    // vaDriverMutex again: the upload is a VA driver call with the same race surface as above.
    // Released before SubmitFrame so the display thread can take the mutex to drain the queue.
    {
        const cMutexLock vaLock(&display->GetVaDriverMutex());
        if (av_hwframe_transfer_data(hwFrame.get(), swFrame.get(), 0) < 0) [[unlikely]] {
            esyslog("vaapivideo/device: black frame upload failed");
            return false;
        }
    }

    // Submit through the normal display path: modeset / DRM plane state mirrors a real frame.
    auto frame = std::make_unique<VaapiFrame>();
    frame->avFrame = hwFrame.release();
    // FFmpeg VAAPI ABI: data[3] holds the VASurfaceID directly, cast through uintptr_t.
    frame->vaSurfaceId = static_cast<VASurfaceID>(
        reinterpret_cast<uintptr_t>(frame->avFrame->data[3])); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    // Stage SDR with the commit (only now that a frame is ready): the decoder's normal SDR staging is
    // bypassed here, so without this an HDR-left connector would read this NV12 black as BT.2020
    // PQ/HLG (crushed blacks). The display thread applies the staged state atomically with this frame.
    display->SetHdrOutputState(HdrStreamInfo{});
    const bool queued = display->SubmitFrame(std::move(frame), 100);
    // Debug aid: trace every black-frame submission (startup splash, radio, encrypted, channel-switch
    // clear) and whether it reached the display queue. Flatten the caption to one line -- newlines/runs
    // of whitespace become a single space -- so multi-line captions stay on one log line.
    std::string logCaption;
    logCaption.reserve(centerText.size());
    for (const char c : centerText) {
        if (c == '\n' || c == '\r' || c == '\t' || c == ' ') {
            if (!logCaption.empty() && logCaption.back() != ' ') {
                logCaption += ' ';
            }
        } else {
            logCaption += c;
        }
    }
    while (!logCaption.empty() && logCaption.back() == ' ') {
        logCaption.pop_back();
    }
    dsyslog("vaapivideo/device: black frame %dx%d caption=\"%s\" -> queued=%s", w, h,
            logCaption.empty() ? "(none)" : logCaption.c_str(), queued ? "yes" : "no");
    return queued;
}

cVaapiDevice::cVaapiDevice() {
    isyslog("vaapivideo/device: created");
    SetDescription("VAAPI Video Device");
    SetVideoFormat(true);
}

cVaapiDevice::~cVaapiDevice() noexcept {
    // Pass CheckDecoder=false: the decoder may already be torn down (e.g. after Detach()),
    // and the default HasDecoder() gate would misreport this device as non-primary.
    dsyslog("vaapivideo/device: destroying (isPrimary=%d)", IsPrimaryDevice(/*CheckDecoder=*/false));

    // Demote before destroying subsystems: prevents VDR from routing PlayVideo/OSD calls
    // into a half-destroyed device.
    if (IsPrimaryDevice(/*CheckDecoder=*/false)) {
        cDevice::MakePrimaryDevice(false);
    }

    DetachAllReceivers();

    // Sever OSD provider's display pointer before display is destroyed. The OSD provider
    // outlives this device (VDR owns it via cOsdProvider::Shutdown).
    if (auto *vaapiProvider = dynamic_cast<cVaapiOsdProvider *>(::osdProvider)) {
        vaapiProvider->DetachDisplay();
        dsyslog("vaapivideo/device: OSD provider detached from display");
    }

    if (audioProcessor || decoder || display) {
        Stop();
    }

    ReleaseHardware();

    dsyslog("vaapivideo/device: destroyed");
}

// ============================================================================
// === VDR DEVICE INTERFACE ===
// ============================================================================

[[nodiscard]] auto cVaapiDevice::CanReplay() const -> bool {
    return (initState.load(std::memory_order_acquire) == 2) && decoder && decoder->IsReady();
}

[[nodiscard]] auto cVaapiDevice::CanScaleVideo(const cRect &rect, int /*alignment*/) -> cRect {
    return HardwareReady() && display ? display->NormalizeVideoRect(rect) : cRect::Null;
}

namespace {
constexpr uint64_t DEVICE_CLEAR_LOG_COOLDOWN_MS =
    1000; ///< Min spacing between Clear() diagnostic logs; collapses a scrub/seek Clear() burst to ~1 line/s.
} // namespace

auto cVaapiDevice::Clear() -> void {
    // Diagnostic: log thread + inter-call delta to diagnose unexpected Clear() bursts.
    // Expected callers: cDvbPlayer::Empty() (pause/play/trick/goto) and
    // cTransfer::Receive() (live-TV retry exhaustion). Rapid-fire calls outside those
    // paths should be traceable via the logged thread name.
    //
    // Rate-limited: a progress-bar scrub / seek auto-repeat fires one Clear() per jump (~80 in a
    // burst). Log isolated Clear()s in full immediately (the gate is open after a quiet gap); during a
    // burst, emit one line per DEVICE_CLEAR_LOG_COOLDOWN_MS carrying the coalesced count so the burst
    // stays diagnosable (thread + magnitude) without 80 near-identical lines.
    // Stream boundary: drop any format still ageing in the display-mode stability gate before it
    // matures against a stream that has already gone away.
    InvalidateDisplayModeCandidate();

    const auto nowMs = cTimeMs::Now();
    const uint64_t prevMs = lastClearMs.exchange(nowMs, std::memory_order_relaxed);
    clearsSinceLog.fetch_add(1, std::memory_order_relaxed);
    if (prevMs == 0 || nowMs - lastClearLogMs.load(std::memory_order_relaxed) >= DEVICE_CLEAR_LOG_COOLDOWN_MS) {
        std::array<char, 16> threadName{};
        (void)pthread_getname_np(pthread_self(), threadName.data(), threadName.size());
        const unsigned coalesced = clearsSinceLog.exchange(0, std::memory_order_relaxed);
        if (prevMs == 0) {
            dsyslog("vaapivideo/device: Clear() thread='%s' (first call)", threadName.data());
        } else if (coalesced > 1) {
            dsyslog("vaapivideo/device: Clear() thread='%s' delta=%llums (+%u more coalesced)", threadName.data(),
                    static_cast<unsigned long long>(nowMs - prevMs), coalesced - 1);
        } else {
            dsyslog("vaapivideo/device: Clear() thread='%s' delta=%llums", threadName.data(),
                    static_cast<unsigned long long>(nowMs - prevMs));
        }
        lastClearLogMs.store(nowMs, std::memory_order_relaxed);
    }

    cDevice::Clear();

    // trickSpeed intentionally NOT reset: Clear() is a buffer flush, not a mode change.
    // VDR calls Clear() at the start of trick play; resetting here would cancel the mode.

    // trickAudioPts IS position state, so it must go: keeping it would pin the radio STC at wherever the
    // trick stopped. Safe here because cDvbPlayer::Empty() reads GetSTC() for its resume index BEFORE
    // calling DeviceClear().
    trickAudioPts.store(AV_NOPTS_VALUE, std::memory_order_relaxed);
    // A seek jumps the PTS timeline; drop the baseline so the first frame at the new position isn't
    // mistaken for the old position's EOF repeat.
    ResetReplayAudioEofBaseline();

    // Force audio codec re-detection. This is the only place that catches the replay
    // track-switch path: "audi N" -> cDvbPlayer::SetAudioTrack -> Goto -> Empty ->
    // DeviceClear -> here. VDR routes around SetAudioTrackDevice() when a cPlayer is
    // attached, so without this reset audioCodecId stays pinned and the old decoder is
    // fed the new track's bitstream -- audio dies silently until the next channel switch.
    // Safe: Clear() runs on the main thread (or under cDvbPlayer's mutex); PlayAudio()
    // cannot race this reset.
    ResetAudioCodecState();

    // Stream-switch window: prevents the display thread from re-presenting a surface
    // that the decoder is about to invalidate during the flush.
    if (display) [[likely]] {
        display->BeginStreamSwitch();
    }

    if (decoder) [[likely]] {
        decoder->Clear();
    }

    if (display) [[likely]] {
        display->EndStreamSwitch();
    }

    if (audioProcessor) [[likely]] {
        audioProcessor->Clear();
    }
}

[[nodiscard]] auto cVaapiDevice::DeviceName() const -> cString {
    // SVDRP PRIM (no-arg) and LSTD render this. drmPath is latched by Initialize();
    // connectorName is either user-supplied (-c) or populated by SelectDrmConnector()
    // on first attach. Before either runs (e.g. between ctor and ProcessArgs, or
    // --detached before the first attach), fall back to whatever is available.
    if (drmPath.empty()) {
        return "vaapivideo";
    }
    if (connectorName.empty()) {
        return cString::sprintf("vaapivideo %s", drmPath.c_str());
    }
    return cString::sprintf("vaapivideo %s %s", drmPath.c_str(), connectorName.c_str());
}

[[nodiscard]] auto cVaapiDevice::DeviceType() const -> cString { return "VAAPI"; }

[[nodiscard]] auto cVaapiDevice::Flush(int TimeoutMs) -> bool {
    // Waits only on the decoder packet queue, not the audio queue. Audio drains under its
    // own ALSA backpressure and has not been observed to linger past channel switches.
    // If that changes, add audioProcessor->GetQueueSize()==0 here.
    dsyslog("vaapivideo/device: Flush(%d)", TimeoutMs);

    if (!decoder) {
        return true;
    }

    const cTimeMs timeout(TimeoutMs);
    while (!decoder->IsQueueEmpty() && !timeout.TimedOut()) {
        cCondWait::SleepMs(10);
    }

    return decoder->IsQueueEmpty();
}

auto cVaapiDevice::Freeze() -> void {
    cDevice::Freeze();
    paused.store(true, std::memory_order_relaxed);

    // Drop queued packets so un-pause shows the user's intended frame, not stale lookahead.
    // Does NOT reset sync EMA: that would cause a reseed transient on resume.
    if (decoder) [[likely]] {
        decoder->DrainQueue();
        // Hold the drain so the jitterBuf head's PTS doesn't drift during the pause: ALSA is
        // dropped below and WritePcmToAlsa stops, GetClock() goes stale within ~1 s, and the
        // decoder's no-clock-freerun would otherwise submit frames at vsync rate -- leaving the
        // head hundreds of ms ahead of the re-anchored audio clock on resume and triggering the
        // post-resume drain-stall loop.
        decoder->SetDevicePaused(true);
    }
    // Suppress the display's underrun ("queue empty Nms") log: with the decoder holding the drain,
    // re-presents of the last frame are intentional, not a stall. Without this gate the per-pause
    // syslog gains a stream of vsync-rate underrun lines until the display's 10 s DISPLAY_UNDERRUN_IDLE_MAX_MS
    // catch-all kicks in (i.e. several entries per pause are typical).
    if (display) [[likely]] {
        display->SetDevicePaused(true);
    }

    // Drop ALSA buffers: ~100-200 ms of audio is already queued in the sink and would
    // continue playing past the freeze without an explicit drain. DropOutput preserves
    // playbackPts; pauseClock=true additionally pins GetClock() so it does not extrapolate
    // through ALSA silence (a short pause < AUDIO_CLOCK_STALE_MS would otherwise produce a
    // fake-advanced clock on resume -> SkipStaleJitterFrames silently drops the preserved
    // jitterBuf head = playback-position skip on un-pause). The pin lifts on the next ALSA
    // write inside WritePcmToAlsa().
    if (audioProcessor) [[likely]] {
        audioProcessor->DropOutput(/*pauseClock=*/true);
    }
}

auto cVaapiDevice::GetOsdSize(int &Width, int &Height, double &PixelAspect) -> void {
    // Called frequently by skins during repaint. Three tiers:
    //   1. cached  -- normal hot path after display init.
    //   2. live    -- first post-init call populates the cache.
    //   3. config  -- pre-init fallback (NOT cached; real display may appear later).
    // Gate the display-touching tiers on HardwareReady() (the acquire paired with AttachHardware()'s
    // SVDRP-thread publish): a skin repaint here must not race a half-built display. Until ready, use
    // the config tier.
    //
    // Keying the cache on the mode generation is the whole OSD-resize hookup for runtime mode
    // switching: VDR polls cOsdProvider::UpdateOsdSize() once per second, which lands here, sees the
    // new size, recomputes Setup.OSDWidth/Height plus the font sizes and bumps osdState -- which
    // makes open menus redisplay, destroying the old cVaapiOsd for one sized to the new mode.
    //
    // PixelAspect is VDR's PIXEL aspect hint ("a circle drawn 100x100 shows up as a circle"), not
    // the shape of the screen -- cDevice's own implementation returns 1.0. It is 1.0 for every
    // square-pixel mode and only interesting on the anamorphic SD timings. Nothing in VDR core
    // reads it back; it lands in Setup.OSDAspect for skins to use.
    if (HardwareReady() && display) [[likely]] {
        // Size and pixel aspect live in separate atomics, so bracket both with the mode generation
        // and retry if it moved -- otherwise a mode change landing mid-read pairs a new size with
        // the previous aspect and the mismatch is CACHED. Cannot spin: a second iteration needs a
        // whole modeset to land inside it.
        uint64_t modeGeneration = display->GetModeGeneration();
        uint32_t outputWidth = 0;
        uint32_t outputHeight = 0;
        double pixelAspect = 1.0;
        for (bool coherent = false; !coherent;) {
            display->GetOutputGeometry(outputWidth, outputHeight);
            const AspectRatio par = display->GetOutputPixelAspect();
            pixelAspect = par.den > 0 ? static_cast<double>(par.num) / static_cast<double>(par.den) : 1.0;
            const uint64_t recheck = display->GetModeGeneration();
            coherent = (recheck == modeGeneration);
            modeGeneration = recheck;
        }

        if (osdWidth > 0 && osdHeight > 0 && osdModeGeneration == modeGeneration) [[likely]] {
            Width = osdWidth;
            Height = osdHeight;
            PixelAspect = pixelAspect;
            return;
        }
        if (display->IsInitialized()) {
            osdWidth = static_cast<int>(outputWidth);
            osdHeight = static_cast<int>(outputHeight);
            osdModeGeneration = modeGeneration;
            Width = osdWidth;
            Height = osdHeight;
            PixelAspect = pixelAspect;
            dsyslog("vaapivideo/device: OSD size %dx%d pixel-aspect=%.3f screen-aspect=%.3f", Width, Height,
                    PixelAspect, display->GetAspectRatio());
            return;
        }
    }

    Width = static_cast<int>(vaapiConfig.display.GetWidth());
    Height = static_cast<int>(vaapiConfig.display.GetHeight());
    PixelAspect = 1.0; // pre-init fallback: --resolution names a square-pixel raster
}

[[nodiscard]] auto cVaapiDevice::GetSTC() -> int64_t {
    // Video PTS outranks the audio clock (the truer STC) because cDvbPlayer only wants editing-mark /
    // position math, where one-frame accuracy is enough and video is already valid during the audio prime
    // window. Do NOT use for A/V sync: lags real audio output by decoder+display pipeline depth.
    if (decoder) [[likely]] {
        if (const int64_t pts = decoder->GetLastPts(); pts != AV_NOPTS_VALUE) {
            return pts;
        }
    }

    // Audio-only (radio) replay: nothing ever stamps lastPts, and VDR requires a valid STC in normal AND
    // trick modes (vdr/device.h) -- cDvbPlayer maps it back to a frame index for the position display and
    // for the resume point after a trick. While a trick runs, output is dropped and the DAC clock decays
    // to stale, so the last stepped PTS is the position; once trickSpeed is 0 the advancing audio clock
    // must win again (slow-forward exits without a Clear(), so the latch can outlive its trick).
    // Acquire pairs with TrickSpeed()'s release-store, which resets the latch first -- a reader that
    // sees a new trick's speed cannot read the previous trick's position.
    if (trickSpeed.load(std::memory_order_acquire) != 0) {
        if (const int64_t trickPts = trickAudioPts.load(std::memory_order_relaxed); trickPts != AV_NOPTS_VALUE) {
            return trickPts;
        }
    }
    if (audioProcessor) {
        if (const int64_t clock = audioProcessor->GetClock(); clock != AV_NOPTS_VALUE) {
            return clock;
        }
    }
    return -1;
}

auto cVaapiDevice::GetVideoSize(int &Width, int &Height, double &VideoAspect) -> void {
    // Skins watch the 0->non-zero transition to trigger a repaint; report 0x0 rather than
    // stale dimensions when no codec is open or no frame has been decoded yet.
    if (!HasDecoder()) [[unlikely]] {
        Width = Height = 0;
        VideoAspect = 1.0;
        return;
    }

    Width = decoder->GetStreamWidth();
    Height = decoder->GetStreamHeight();
    VideoAspect = decoder->GetStreamAspect();

    if (Width == 0 || Height == 0) [[unlikely]] {
        VideoAspect = 1.0; // codec open but pre-first-frame; aspect would be NaN
    }
}

// ----------------------------------------------------------------------------
// GrabImage helpers (anonymous namespace -- used only here)
// ----------------------------------------------------------------------------

namespace {

/// Upper bounds on GRAB's requested output dimensions (DCI-8K width, 8K UHD height). VDR's SVDRP
/// CmdGRAB forwards SizeX/SizeY with only a numeric check, so an absurd request must not drive a
/// multi-gigabyte scale allocation (remote OOM via an exposed SVDRP port). Oversize requests are
/// rejected outright -- silently clamping would return an image the caller did not ask for. The
/// per-axis caps also bound the pixel count (<= 8192*4320) with no separate area check needed.
constexpr int GRAB_MAX_WIDTH = 8192;
constexpr int GRAB_MAX_HEIGHT = 4320;

/// One-shot filter graph: feed @p in, pull a single frame whose pixel format is enforced
/// by appending `,format=<outFmt>` to @p chainPrefix. Caller-supplied chain stages
/// (e.g. "scale=W:H") run before that terminal format conversion. @p srcColorspace and
/// @p srcColorRange become buffersrc options; zscale reads them from the link, not the
/// frame, so for HDR sources they MUST be passed here even if also stamped on the AVFrame.
[[nodiscard]] auto RunGrabFilter(AVFrame *in, std::string_view chainPrefix, AVPixelFormat outFmt,
                                 AVColorSpace srcColorspace = AVCOL_SPC_UNSPECIFIED,
                                 AVColorRange srcColorRange = AVCOL_RANGE_UNSPECIFIED)
    -> std::unique_ptr<AVFrame, FreeAVFrame> {
    if (!in || in->width <= 0 || in->height <= 0) [[unlikely]] {
        return nullptr;
    }

    const std::unique_ptr<AVFilterGraph, FreeAVFilterGraph> graph{avfilter_graph_alloc()};
    if (!graph) [[unlikely]] {
        return nullptr;
    }

    // Numeric pix_fmt / colorspace / range: HW format aliases differ across FFmpeg versions;
    // integer values are stable.
    std::string srcArgs =
        std::format("video_size={}x{}:pix_fmt={}:time_base=1/1:pixel_aspect=1/1", in->width, in->height, in->format);
    if (srcColorspace != AVCOL_SPC_UNSPECIFIED) {
        srcArgs += std::format(":colorspace={}", static_cast<int>(srcColorspace));
    }
    if (srcColorRange != AVCOL_RANGE_UNSPECIFIED) {
        srcArgs += std::format(":range={}", static_cast<int>(srcColorRange));
    }

    AVFilterContext *src = nullptr;
    if (const int ret = avfilter_graph_create_filter(&src, avfilter_get_by_name("buffer"), "in", srcArgs.c_str(),
                                                     nullptr, graph.get());
        ret < 0) [[unlikely]] {
        esyslog("vaapivideo/device: grab buffer source: %s", AvErr(ret).data());
        return nullptr;
    }

    AVFilterContext *sink = nullptr;
    if (const int ret = avfilter_graph_create_filter(&sink, avfilter_get_by_name("buffersink"), "out", nullptr, nullptr,
                                                     graph.get());
        ret < 0) [[unlikely]] {
        esyslog("vaapivideo/device: grab buffer sink: %s", AvErr(ret).data());
        return nullptr;
    }

    AVFilterInOut *outputs = avfilter_inout_alloc();
    AVFilterInOut *inputs = avfilter_inout_alloc();
    if (!outputs || !inputs) [[unlikely]] {
        avfilter_inout_free(&outputs);
        avfilter_inout_free(&inputs);
        return nullptr;
    }
    outputs->name = av_strdup("in");
    outputs->filter_ctx = src;
    inputs->name = av_strdup("out");
    inputs->filter_ctx = sink;
    if (!outputs->name || !inputs->name) [[unlikely]] {
        avfilter_inout_free(&outputs);
        avfilter_inout_free(&inputs);
        return nullptr;
    }

    // The trailing `format=` is what guarantees the output pixfmt; sink itself accepts anything.
    const std::string chain = chainPrefix.empty()
                                  ? std::format("format={}", av_get_pix_fmt_name(outFmt))
                                  : std::format("{},format={}", chainPrefix, av_get_pix_fmt_name(outFmt));

    const int parseRet = avfilter_graph_parse_ptr(graph.get(), chain.c_str(), &inputs, &outputs, nullptr);
    avfilter_inout_free(&inputs);
    avfilter_inout_free(&outputs);
    if (parseRet < 0) [[unlikely]] {
        esyslog("vaapivideo/device: grab parse '%s': %s", chain.c_str(), AvErr(parseRet).data());
        return nullptr;
    }

    if (const int ret = avfilter_graph_config(graph.get(), nullptr); ret < 0) [[unlikely]] {
        esyslog("vaapivideo/device: grab config '%s': %s", chain.c_str(), AvErr(ret).data());
        return nullptr;
    }

    if (const int ret = av_buffersrc_add_frame_flags(src, in, AV_BUFFERSRC_FLAG_KEEP_REF); ret < 0) [[unlikely]] {
        esyslog("vaapivideo/device: grab send: %s", AvErr(ret).data());
        return nullptr;
    }
    // Flush so buffersink emits the frame even though the chain is finite. EOS-flush errors
    // are not actionable here (sink may still have a buffered frame), but FFmpeg marks the
    // function nodiscard so capture the result explicitly.
    [[maybe_unused]] const int eosRet = av_buffersrc_add_frame_flags(src, nullptr, 0);

    std::unique_ptr<AVFrame, FreeAVFrame> out{av_frame_alloc()};
    if (!out) [[unlikely]] {
        return nullptr;
    }
    if (const int ret = av_buffersink_get_frame(sink, out.get()); ret < 0) [[unlikely]] {
        esyslog("vaapivideo/device: grab recv: %s", AvErr(ret).data());
        return nullptr;
    }
    return out;
}

// Reproduce on-screen geometry inside the grab canvas. The VPP filter has already DAR-fitted the
// fb to videoRect (no DRM scaler), so frame dimensions equal placement dimensions and the only
// remaining step is a pad to full output size when videoRect is a sub-rect (skin thumbnail mode).
[[nodiscard]] auto BuildGrabLayoutChain(const AVFrame *frame, const cVaapiDisplay &display) -> std::string {
    if (!frame || frame->width <= 0 || frame->height <= 0) [[unlikely]] {
        return {};
    }
    const auto placement = FitVideoToRect(static_cast<uint32_t>(frame->width), static_cast<uint32_t>(frame->height),
                                          display.GetVideoRect());
    const uint32_t outW = display.GetOutputWidth();
    const uint32_t outH = display.GetOutputHeight();
    if (placement.width == 0 ||
        (placement.destX == 0 && placement.destY == 0 && placement.width == outW && placement.height == outH)) {
        return {};
    }
    return std::format("pad={}:{}:{}:{}:color=black", outW, outH, placement.destX, placement.destY);
}

/// Serialize an RGB24 AVFrame as a P6 PNM byte stream into a malloc()'d buffer.
/// PNM = trivial text header + raw RGB; no codec required.
[[nodiscard]] auto MakePnm(const AVFrame *rgb24, int &outSize) -> uchar * {
    const std::string header = std::format("P6\n{} {}\n255\n", rgb24->width, rgb24->height);
    const size_t rowBytes = static_cast<size_t>(rgb24->width) * 3;
    const size_t pixelBytes = rowBytes * static_cast<size_t>(rgb24->height);
    const size_t total = header.size() + pixelBytes;

    // VDR's SVDRP code free()s the returned pointer, so malloc -- not new[].
    auto *buf = static_cast<uchar *>(std::malloc(total)); // NOLINT(cppcoreguidelines-no-malloc)
    if (!buf) [[unlikely]] {
        return nullptr;
    }

    // NOLINTNEXTLINE(bugprone-not-null-terminated-result) -- header is a sized blob, not a C string
    std::memcpy(buf, header.data(), header.size());
    uchar *dst = buf + header.size();
    const uint8_t *src = rgb24->data[0];
    for (int y = 0; y < rgb24->height; ++y) {
        std::memcpy(dst, src, rowBytes); // skip linesize padding
        dst += rowBytes;
        src += rgb24->linesize[0];
    }

    outSize = static_cast<int>(total);
    return buf;
}

/// Encode a single YUV420P AVFrame as a JFIF JPEG via libavcodec MJPEG into a malloc()'d buffer.
/// @p quality is 1..100 (higher = better); -1 maps to 95.
[[nodiscard]] auto EncodeMjpeg(AVFrame *yuv420p, int quality, int &outSize) -> uchar * {
    const AVCodec *codec = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
    if (!codec) [[unlikely]] {
        esyslog("vaapivideo/device: MJPEG encoder unavailable in this FFmpeg build");
        return nullptr;
    }

    std::unique_ptr<AVCodecContext, FreeAVCodecContext> ctx{avcodec_alloc_context3(codec)};
    if (!ctx) [[unlikely]] {
        return nullptr;
    }

    // Modern FFmpeg deprecates YUVJ formats; emit yuv420p with color_range=JPEG so the MJPEG
    // encoder writes a JFIF full-range stream without warnings.
    ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    ctx->width = yuv420p->width;
    ctx->height = yuv420p->height;
    ctx->time_base = {.num = 1, .den = 25}; // arbitrary; MJPEG is stateless
    ctx->color_range = AVCOL_RANGE_JPEG;

    // FFmpeg's quality scale: smaller global_quality = better. Map 1..100 -> [FF_QP2LAMBDA*100 .. FF_QP2LAMBDA*1].
    const int q = (quality > 0) ? std::clamp(quality, 1, 100) : 95;
    ctx->global_quality = FF_QP2LAMBDA * (101 - q);
    ctx->flags |= AV_CODEC_FLAG_QSCALE;

    if (const int ret = avcodec_open2(ctx.get(), codec, nullptr); ret < 0) [[unlikely]] {
        esyslog("vaapivideo/device: MJPEG open: %s", AvErr(ret).data());
        return nullptr;
    }

    if (const int ret = avcodec_send_frame(ctx.get(), yuv420p); ret < 0) [[unlikely]] {
        esyslog("vaapivideo/device: MJPEG send_frame: %s", AvErr(ret).data());
        return nullptr;
    }

    std::unique_ptr<AVPacket, FreeAVPacket> pkt{av_packet_alloc()};
    if (!pkt) [[unlikely]] {
        return nullptr;
    }
    if (const int ret = avcodec_receive_packet(ctx.get(), pkt.get()); ret < 0) [[unlikely]] {
        esyslog("vaapivideo/device: MJPEG receive_packet: %s", AvErr(ret).data());
        return nullptr;
    }

    auto *buf =
        static_cast<uchar *>(std::malloc(static_cast<size_t>(pkt->size))); // NOLINT(cppcoreguidelines-no-malloc)
    if (!buf) [[unlikely]] {
        return nullptr;
    }
    std::memcpy(buf, pkt->data, static_cast<size_t>(pkt->size));
    outSize = pkt->size;
    return buf;
}

} // namespace

[[nodiscard]] auto cVaapiDevice::GrabImage(int &Size, bool Jpeg, int Quality, int SizeX, int SizeY) -> uchar * {
    if (!HardwareReady() || !display) [[unlikely]] {
        esyslog("vaapivideo/device: GrabImage -- device not ready");
        return nullptr;
    }

    // Reject oversize requests up front, before the (expensive) surface grab and HDR tonemap --
    // see GRAB_MAX_WIDTH/GRAB_MAX_HEIGHT for why VDR core does not bound these itself.
    if (SizeX > GRAB_MAX_WIDTH || SizeY > GRAB_MAX_HEIGHT) [[unlikely]] {
        esyslog("vaapivideo/device: GrabImage -- requested size %dx%d exceeds limit %dx%d", SizeX, SizeY,
                GRAB_MAX_WIDTH, GRAB_MAX_HEIGHT);
        return nullptr;
    }

    // 1. Snapshot the displayed VAAPI surface to host memory (NV12 or P010).
    auto srcFrame = display->GrabDisplayedFrame();
    if (!srcFrame) [[unlikely]] {
        esyslog("vaapivideo/device: GrabImage -- no displayed frame yet");
        return nullptr;
    }

    // 2. Drive the HDR branch from the display's cached state, NOT srcFrame->color_trc:
    // av_hwframe_transfer_data strips all color metadata on download. We re-stamp colorimetry
    // for zscale/tonemap (per-frame) AND pass colorspace+range as buffersrc options below
    // (zscale reads those from the filter LINK, not the frame).
    const StreamHdrKind hdrKind = display->GetActiveHdrKind();
    const bool isHdr = (hdrKind != StreamHdrKind::Sdr);
    if (isHdr) {
        srcFrame->color_primaries = AVCOL_PRI_BT2020;
        srcFrame->color_trc = (hdrKind == StreamHdrKind::Hlg) ? AVCOL_TRC_ARIB_STD_B67 : AVCOL_TRC_SMPTE2084;
        srcFrame->colorspace = AVCOL_SPC_BT2020_NCL;
        srcFrame->color_range = AVCOL_RANGE_MPEG;
    }
    // HDR-to-SDR tonemap (canonical FFmpeg wiki shape; requires libzimg for zscale):
    //   PQ/HLG -> linear (npl=200, ~ BT.2408 graphics white) -> gbrpf32le (linearization needs
    //   RGB; YUV matrix is undefined for linear samples) -> BT.2020->BT.709 gamut -> hable
    //   tonemap -> BT.709 gamma/matrix, PC range -> yuv420p (RunGrabFilter appends rgb24).
    // Tuning:
    //   peak=1.0 must be EXPLICIT. The filter's default peak=0 means "auto-detect from MaxCLL /
    //     mastering metadata", forwarded from the source frame -- for typical 1000-nit
    //     mastered HDR10 that derives peak~5 and produces visibly gray whites. peak=1.0 forces
    //     anything >= npl (200 nits) to clip to pure white; aggressive, but the right call for
    //     an SDR screen grab where preserving HDR specular detail isn't worth dim mid-tones.
    //   r=pc (not r=tv) avoids the swscale TV->PC expansion clamping max RGB near Y=235.
    //   desat=1 is the mid-ground; the filter default of 2 over-desaturates toward white.
    std::string videoChain;
    if (isHdr) {
        videoChain = "zscale=t=linear:npl=200,format=gbrpf32le,zscale=p=bt709,"
                     "tonemap=hable:desat=1:peak=1.0,"
                     "zscale=t=bt709:m=bt709:r=pc,format=yuv420p";
    }
    if (const std::string layoutChain = BuildGrabLayoutChain(srcFrame.get(), *display); !layoutChain.empty()) {
        if (!videoChain.empty()) {
            videoChain += ',';
        }
        videoChain += layoutChain;
    }
    auto rgb24 = RunGrabFilter(srcFrame.get(), videoChain, AV_PIX_FMT_RGB24,
                               isHdr ? AVCOL_SPC_BT2020_NCL : AVCOL_SPC_UNSPECIFIED,
                               isHdr ? AVCOL_RANGE_MPEG : AVCOL_RANGE_UNSPECIFIED);
    if (!rgb24) [[unlikely]] {
        return nullptr;
    }

    // 3. Composite the visible OSD on top at native display geometry, before any user-requested
    // resize -- OSD geometry is screen-space and would misalign if the video was scaled first.
    if (auto *provider = dynamic_cast<cVaapiOsdProvider *>(::osdProvider)) {
        provider->CompositeOntoRgb24(rgb24->data[0], rgb24->width, rgb24->height, rgb24->linesize[0],
                                     display->GetActiveOsdFbId());
    }

    // 4. Encode. SizeX/SizeY <= 0 = native display resolution (upper bounds enforced at entry).
    // "Native" means what the viewer sees, not the raster: on an anamorphic SD mode the scanout is
    // 720x576 but the panel shows it as 16:9, so an unscaled grab would arrive squashed in the
    // browser (the live plugin renders these). Undo the stretch by widening to square pixels. An
    // explicit SizeX/SizeY still wins -- that is the caller stating exactly what it wants.
    const AspectRatio grabPar = display->GetOutputPixelAspect();
    const int squareWidth =
        (grabPar.den > 0 && grabPar.num != grabPar.den)
            ? static_cast<int>((static_cast<int64_t>(rgb24->width) * grabPar.num) / grabPar.den) & ~1
            : rgb24->width;
    const int outW = (SizeX > 0) ? SizeX : std::max(squareWidth, 2);
    const int outH = (SizeY > 0) ? SizeY : rgb24->height;
    const bool needScale = (outW != rgb24->width) || (outH != rgb24->height);

    // The rgb24 frame from filter graph #1 is tagged csp=gbr range=pc; the second graph's
    // buffersrc must declare the same or libavfilter logs "Changing video frame properties on
    // the fly" and falls back through swscale.
    constexpr AVColorSpace kRgbSrcColorspace = AVCOL_SPC_RGB;
    constexpr AVColorRange kRgbSrcColorRange = AVCOL_RANGE_JPEG;

    if (Jpeg) {
        // yuv420p + AVCOL_RANGE_JPEG on the encoder context yields JFIF full-range without the
        // deprecated YUVJ pixfmts. scale fused into this pass to skip an intermediate frame.
        const std::string chain = needScale ? std::format("scale={}:{}", outW, outH) : std::string{};
        auto yuv = RunGrabFilter(rgb24.get(), chain, AV_PIX_FMT_YUV420P, kRgbSrcColorspace, kRgbSrcColorRange);
        if (!yuv) [[unlikely]] {
            return nullptr;
        }
        return EncodeMjpeg(yuv.get(), Quality, Size);
    }

    if (needScale) {
        auto scaled = RunGrabFilter(rgb24.get(), std::format("scale={}:{}", outW, outH), AV_PIX_FMT_RGB24,
                                    kRgbSrcColorspace, kRgbSrcColorRange);
        if (!scaled) [[unlikely]] {
            return nullptr;
        }
        rgb24 = std::move(scaled);
    }
    return MakePnm(rgb24.get(), Size);
}

[[nodiscard]] auto cVaapiDevice::HasDecoder() const -> bool {
    // HardwareReady() first: it is the acquire that makes the cross-thread `decoder` read race-free
    // (VDR/GetVideoSize poll this from the main thread vs the SVDRP-thread attach publish).
    return HardwareReady() && decoder && decoder->IsReady();
}

[[nodiscard]] auto cVaapiDevice::HasIBPTrickSpeed() -> bool {
    // True promises cDvbPlayer we pace its contiguous feed ourselves -- a promise only the decoder's
    // per-frame pacing can keep. An audio-only replay has no frame to pace on, so false instead puts
    // cDvbPlayer on its seeking path (~0.4 s of content per delivered frame), which PlayTrickAudio() can
    // pace and which is the better radio FF anyway.
    // The device cannot know a replay is audio-only up front (VDR replays radio recordings under
    // pmAudioVideo too), so this has to be inferred from the stream. IsPlayingVideo() is VDR's own latch,
    // set on the FIRST video TS packet -- earlier than videoCodecId, which waits for detection + SPS +
    // codec open -- and it is the same predicate cDvbPlayer's VideoOnly delivery gate keys off, so player
    // and device always agree. A trick started in the brief window before a video replay's first packet
    // begins on the seeking path and converges as soon as the first delivered packet flips the latch.
    return IsPlayingVideo();
}

auto cVaapiDevice::MakePrimaryDevice(bool On) -> void {
    dsyslog("vaapivideo/device: MakePrimaryDevice(%s) called", On ? "true" : "false");

    // Deferred-init for --detached: open hardware on first post-startup primary promotion.
    // The startupComplete gate keeps a setup.conf-driven primary restore during VDR bring-up
    // from defeating --detached. On failure, fall through (do NOT return early) -- VDR has
    // already assigned primaryDevice (vdr/device.c:201-202), so bailing leaves the slot
    // non-functional. The device lands in "detached primary" state, recoverable via SVDRP ATTA.
    if (On && initState.load(std::memory_order_acquire) == 0 && !drmPath.empty() &&
        startupComplete.load(std::memory_order_acquire)) {
        isyslog("vaapivideo/device: primary activation after detached -- attaching hardware");
        if (!AttachHardware()) [[unlikely]] {
            esyslog("vaapivideo/device: deferred hardware init failed -- staying detached as "
                    "primary; use SVDRP ATTA to retry");
        }
    }

    cDevice::MakePrimaryDevice(On);

    if (On) {
        // CheckDecoder=false: under --detached the decoder isn't open yet; the default
        // HasDecoder() gate would misreport this device as non-primary.
        if (IsPrimaryDevice(/*CheckDecoder=*/false)) {
            isyslog("vaapivideo/device: activated as primary device");
            // VDR owns the OSD provider's lifetime (frees it via cOsdProvider::Shutdown() at
            // process exit), so create once + reattach thereafter. Must register even when
            // display==nullptr so skins probing cOsdProvider::SupportsTrueColor() during
            // Start() find a provider; pixel-path methods tolerate a null display until
            // AttachDisplay() wires one in.
            if (auto *provider = dynamic_cast<cVaapiOsdProvider *>(::osdProvider)) {
                if (display) {
                    provider->AttachDisplay(display.get());
                    dsyslog("vaapivideo/device: OSD provider reattached to display");
                }
            } else {
                ::osdProvider = new cVaapiOsdProvider(display.get());
                isyslog("vaapivideo/device: OSD provider registered%s", display ? "" : " (detached)");
            }
        } else {
            esyslog("vaapivideo/device: failed to activate as primary device");
        }
    } else {
        isyslog("vaapivideo/device: deactivated as primary device");
        // OSD provider not torn down here: VDR's cOsdProvider::Shutdown() handles it at exit.
    }
}

auto cVaapiDevice::Mute() -> void {
    cDevice::Mute();

    // DropOutput silences the queued tail without nulling the playback clock; the persistent mute
    // case is handled by SetVolumeDevice(0) (VDR's mute key drives that path, not this override).
    // Full Clear() here would null GetClock() and cause a video freerun + display underrun for
    // every OSD-mediated mute. HardwareReady()-gated: skins can mute on menu open/close (main
    // thread) during an in-flight ATTA.
    if (HardwareReady() && audioProcessor) [[likely]] {
        audioProcessor->DropOutput();
    }
}

auto cVaapiDevice::Play() -> void {
    cDevice::Play();

    // VDR fires Play() both for genuine resume AND as the middle leg of REW->PLAY->REW. Request
    // trick exit and let the decoder confirm: a TrickSpeed() within one frame period cancels
    // the request; otherwise the decoder leaves trick mode on the next frame.
    if (trickSpeed.exchange(0, std::memory_order_release) != 0) {
        if (decoder) [[likely]] {
            decoder->RequestTrickExit();
        }
    }

    paused.store(false, std::memory_order_relaxed);
    // Release the drain-loop hold. The next WritePcmToAlsa anchors the clock and lifts the
    // pause pin (Freeze() set clockPaused=true via DropOutput); until then GetClock() returns
    // the pinned playbackPts so the drain due-gate stays stable across the resume window.
    // Seek/startup NOPTS handling remains covered by DECODER_NO_CLOCK_HOLD_MS.
    if (decoder) [[likely]] {
        decoder->SetDevicePaused(false);
    }
    if (display) [[likely]] {
        display->SetDevicePaused(false);
    }
}

[[nodiscard]] auto cVaapiDevice::PlayTrickAudio(const uchar *Data, int Length) -> int {
    // Without video the decoder paces nothing, leaving this the only brake on cDvbPlayer's feed and the
    // only source of a position. Audio stays dropped (TrickSpeed() -> DropOutput).
    const auto pes = ParsePes({Data, static_cast<size_t>(Length)});
    if (!pes.isAudio || pes.payloadSize == 0) [[unlikely]] {
        return Length; // nothing to pace on; swallow it so the player moves to the next frame
    }

    // On the seeking path one PES == one ~0.4 s content step, so gating these calls IS the trick speed.
    // A mode's first step has no reference and passes straight through.
    const int64_t prevPts = trickAudioPts.load(std::memory_order_relaxed);
    if (decoder && !decoder->TakeTrickStep(pes.pts, prevPts)) {
        return 0; // hold still running: VDR parks on Poll(), which gates on the same deadline
    }
    if (pes.pts != AV_NOPTS_VALUE) {
        trickAudioPts.store(pes.pts, std::memory_order_relaxed);
    }
    return Length;
}

[[nodiscard]] auto cVaapiDevice::PlayAudio(const uchar *Data, int Length, uchar /*Id*/) -> int {
    if (!Data || Length <= 0) [[unlikely]] {
        return Length;
    }

    // Trick output is dropped either way; a video replay is already paced and positioned by PlayVideo(),
    // an audio-only one needs PlayTrickAudio() to do both. Ahead of `paused` for the same reason
    // PlayVideo() allows paused+trick: VDR freezes before entering a slow trick mode.
    if (trickSpeed.load(std::memory_order_relaxed) != 0) [[unlikely]] {
        return IsPlayingVideo() ? Length : PlayTrickAudio(Data, Length);
    }

    if (paused.load(std::memory_order_relaxed)) [[unlikely]] {
        return Length;
    }

    if (!audioProcessor || !audioProcessor->IsInitialized()) [[unlikely]] {
        return Length;
    }

    const auto pes = ParsePes({Data, static_cast<size_t>(Length)});
    if (!pes.isAudio || pes.payloadSize == 0) [[unlikely]] {
        return Length;
    }

    // audioCodecId == NONE triggers detection; reset by SetPlayMode(pmNone),
    // HandleAudioTrackChange(), or Clear() (replay audi-N path).
    AVCodecID currentCodec = audioCodecId.load(std::memory_order_relaxed);
    bool isLive = liveMode.load(std::memory_order_relaxed);

    // Sink escalation: repeated decode-failure cascades with no decoded frame mean the confirmed
    // codec is a misdetection (or the PID's payload changed). Reset the whole audio codec domain
    // like a track change -- stale clock and wrong-codec packets must not survive into re-detection.
    // Clearing previousAudioCodec forces the fresh confirmation to log even on the same id.
    if (currentCodec != AV_CODEC_ID_NONE && audioProcessor->TakeCodecRedetectRequest()) [[unlikely]] {
        esyslog("vaapivideo/device: audio codec %s never produced a frame -- re-detecting",
                avcodec_get_name(currentCodec));
        ResetAudioCodecState();
        ResetReplayAudioEofBaseline();
        previousAudioCodec.store(AV_CODEC_ID_NONE, std::memory_order_relaxed);
        audioProcessor->Clear();
        if (decoder) [[likely]] {
            decoder->NotifyAudioChange();
        }
        currentCodec = AV_CODEC_ID_NONE;
    }

    if (currentCodec == AV_CODEC_ID_NONE) {
        // pmAudioOnly skips PlayVideo() entirely, so latch liveMode on first audio PES.
        if (!isLive) {
            isLive = Transferring();
            liveMode.store(isLive, std::memory_order_relaxed);
            if (decoder) {
                decoder->SetLiveMode(isLive);
            }
        }

        // Payload-first: AC-3/MP2/ADTS carry whole frames per PES payload, so one is decisive and
        // the window would only hurt them (see AUDIO_DETECT_WINDOW). Only AAC-LATM, whose frames
        // span PES boundaries, needs the cross-payload window -- reached solely on a NONE here.
        AVCodecID detectedCodec = ::DetectAudioCodec({pes.payload, pes.payloadSize});
        if (detectedCodec == AV_CODEC_ID_NONE) {
            // A reset on another thread bumps audioDetectGen; drop stale bytes before mixing PIDs.
            if (const uint32_t gen = audioDetectGen.load(std::memory_order_relaxed); gen != audioDetectGenSeen) {
                audioDetectGenSeen = gen;
                audioDetectBuffer.clear();
            }
            audioDetectBuffer.insert(audioDetectBuffer.end(), pes.payload, pes.payload + pes.payloadSize);
            if (audioDetectBuffer.size() > AUDIO_DETECT_WINDOW) {
                audioDetectBuffer.erase(
                    audioDetectBuffer.begin(),
                    audioDetectBuffer.begin() +
                        static_cast<std::ptrdiff_t>(audioDetectBuffer.size() - AUDIO_DETECT_WINDOW));
            }
            detectedCodec = ::DetectAudioCodec({audioDetectBuffer.data(), audioDetectBuffer.size()});
        }

        if (detectedCodec == AV_CODEC_ID_NONE) [[unlikely]] {
            // Reset the candidate so 2-of-2 means two CONSECUTIVE decisive payloads -- else two
            // uncorrelated false positives minutes apart could accumulate into a bogus confirmation.
            audioCodecCandidate.store(AV_CODEC_ID_NONE, std::memory_order_relaxed);
            audioCodecCandidateCount.store(0, std::memory_order_relaxed);
            return Length;
        }

        // Always require 2-of-2 confirmation. SVDRP "audi N" resets audioCodecId on the main
        // thread while PlayTs may have already buffered a full PES from the OLD PID in
        // tsToPesAudio. That stale PES arrives here first and would detect the wrong codec.
        // Depth=2 suffices: tsToPesAudio.Reset() fires on every PUSI, so at most one stale
        // PES can sneak through. Cost: ~24-32 ms extra latency on codec open.
        if (detectedCodec == audioCodecCandidate.load(std::memory_order_relaxed)) {
            audioCodecCandidateCount.fetch_add(1, std::memory_order_relaxed);
        } else {
            audioCodecCandidate.store(detectedCodec, std::memory_order_relaxed);
            audioCodecCandidateCount.store(1, std::memory_order_relaxed);
        }
        // A scrub seek resets audioCodecId on every Clear() (the audi-N track-switch path above), so the
        // same codec re-confirms constantly. Gate the awaiting/confirmed logs on an ACTUAL change from the
        // last confirmed codec: previousAudioCodec survives Clear(), and SetPlayMode(pmNone) clears it so a
        // fresh session still logs once. The 2-of-2 detection itself always runs -- only the logging is gated.
        const bool codecChanged = detectedCodec != previousAudioCodec.load(std::memory_order_relaxed);
        const int candidateCount = audioCodecCandidateCount.load(std::memory_order_relaxed);
        if (candidateCount < 2) {
            if (codecChanged) {
                dsyslog("vaapivideo/device: audio codec %s -- awaiting confirmation (%d/2)",
                        avcodec_get_name(detectedCodec), candidateCount);
            }
            return Length;
        }

        audioCodecCandidate.store(AV_CODEC_ID_NONE, std::memory_order_relaxed);
        audioCodecCandidateCount.store(0, std::memory_order_relaxed);
        audioDetectBuffer.clear(); // detection done; free the window until the next codec-less phase
        audioDetectBuffer.shrink_to_fit();

        if (!audioProcessor->OpenCodec(detectedCodec, 48000, 2)) [[unlikely]] {
            esyslog("vaapivideo/device: failed to open audio codec %s", avcodec_get_name(detectedCodec));
            return Length;
        }

        audioCodecId.store(detectedCodec, std::memory_order_relaxed);
        previousAudioCodec.store(detectedCodec, std::memory_order_relaxed);
        if (codecChanged) {
            isyslog("vaapivideo/device: audio codec %s confirmed (%s, %s)", avcodec_get_name(detectedCodec),
                    isLive ? "live" : "replay", audioProcessor->IsPassthrough() ? "passthrough" : "PCM");
        }

        // Mirrors HandleAudioTrackChange: re-arms freerun so a cold-VPP stall can't anchor
        // sync to a clock that ran ~5 s ahead while the GPU loaded firmware on first use.
        if (decoder) [[likely]] {
            decoder->NotifyAudioChange();
        }
    }

    // Radio detection: 3 s grace set by SetPlayMode(pmAudioVideo) with no video arriving;
    // paint black to clear the previous channel's residual scanout picture.
    if (radioBlackPending.load(std::memory_order_relaxed) && radioBlackTimer.TimedOut()) [[unlikely]] {
        radioBlackPending.store(false, std::memory_order_relaxed);
        if (videoCodecId.load(std::memory_order_relaxed) == AV_CODEC_ID_NONE) {
            // An encrypted channel with no video is owned by the encrypted watchdog (which shows the
            // "encrypted" notice); painting a silent radio splash here would race and flicker with it.
            if (CurrentChannelIsEncrypted()) {
                dsyslog("vaapivideo/device: no video on encrypted channel -- deferring to encrypted watchdog");
            } else {
                isyslog("vaapivideo/device: no video stream detected -- radio mode, showing black frame");
                RefreshRadioSplash(/*force=*/true);
                radioSplashPoll.Set(RADIO_SPLASH_POLL_MS);
            }
        }
    }

    // Radio splash refresh: while a no-video splash is up, re-render when the present EPG event
    // changes so the on-screen "now playing" stays current. Throttled and confined to this audio
    // thread (radioSplashPoll is not shared). The videoCodecId gate drops out of radio mode the
    // moment a channel starts sending video -- the decoder then owns the scanout.
    if (radioSplashActive.load(std::memory_order_relaxed) &&
        videoCodecId.load(std::memory_order_relaxed) == AV_CODEC_ID_NONE && radioSplashPoll.TimedOut()) [[unlikely]] {
        radioSplashPoll.Set(RADIO_SPLASH_POLL_MS);
        RefreshRadioSplash(/*force=*/false);
    }

    // End-of-replay flush: at EOF cDvbPlayer continuously re-pushes the last PES to drain the device
    // (vdr/dvbplayer.c). Decoding the repeats keeps the DAC clock advancing, so GetSTC() never stalls and
    // radio replay never hits VDR's StuckAtEof (no video PTS pins the STC). Drop the repeat -- the real tail
    // is already queued, ALSA drains, the clock ages stale, replay auto-stops. Audio PTS is strictly rising
    // in normal replay (trick play uses PlayTrickAudio), so an exact repeat is the EOF re-push, never a real
    // frame. Reset on every timeline break via ResetReplayAudioEofBaseline().
    if (!isLive && pes.pts != AV_NOPTS_VALUE && pes.pts == lastReplayAudioPts.load(std::memory_order_relaxed))
        [[unlikely]] {
        return Length;
    }

    // Replay backpressure: cap queue to avoid tail-drops that create PTS gaps. Live: always accept.
    if (!isLive && audioProcessor->GetQueueSize() >= AUDIO_QUEUE_HIGHWATER) [[unlikely]] {
        return 0;
    }

    audioProcessor->Decode(pes.payload, pes.payloadSize, pes.pts);
    if (!isLive && pes.pts != AV_NOPTS_VALUE) {
        lastReplayAudioPts.store(pes.pts, std::memory_order_relaxed);
    }
    return Length;
}

[[nodiscard]] auto cVaapiDevice::PlayVideo(const uchar *Data, int Length) -> int {
    if (!Data || Length <= 0) [[unlikely]] {
        return Length;
    }

    // Allow paused+trick (VDR calls Freeze before slow TrickSpeed), block paused-only.
    if (paused.load(std::memory_order_relaxed) && trickSpeed.load(std::memory_order_relaxed) == 0) [[unlikely]] {
        return Length;
    }

    if (!decoder || !decoder->IsReady()) [[unlikely]] {
        return Length;
    }

    radioBlackPending.store(false, std::memory_order_relaxed);
    // Any video PES (even scrambled) ends radio-splash ownership now, before the codec opens -- else
    // the audio-thread poll could repaint the radio splash over a late video stream or the just-shown
    // encrypted notice during the codec-detection window.
    radioSplashActive.store(false, std::memory_order_relaxed);

    const auto pes = ParsePes({Data, static_cast<size_t>(Length)});
    if (!pes.isVideo || pes.payloadSize == 0) [[unlikely]] {
        return Length;
    }

    const AVCodecID currentCodec = videoCodecId.load(std::memory_order_relaxed);
    bool isLive = liveMode.load(std::memory_order_relaxed);

    if (currentCodec == AV_CODEC_ID_NONE) [[unlikely]] {
        // Latch live/replay on first video PES: cTransferControl state is not yet stable
        // at SetPlayMode() time (the cTransfer object may not exist yet).
        isLive = Transferring();
        liveMode.store(isLive, std::memory_order_relaxed);
        decoder->SetLiveMode(isLive);

        const AVCodecID detectedCodec = ::DetectVideoCodec({pes.payload, pes.payloadSize});
        if (detectedCodec == AV_CODEC_ID_NONE) [[unlikely]] {
            return Length;
        }

        // Stale-PES guard only when codec matches previous channel: prevents ring-buffer
        // residue from reopening the old decoder. Narrower than audio's unconditional guard
        // because video has no audi-N race (no out-of-band track switch at this layer).
        const AVCodecID prevVideo = previousVideoCodec.load(std::memory_order_relaxed);
        if (detectedCodec == prevVideo && prevVideo != AV_CODEC_ID_NONE) [[unlikely]] {
            if (detectedCodec == videoCodecCandidate.load(std::memory_order_relaxed)) {
                videoCodecCandidateCount.fetch_add(1, std::memory_order_relaxed);
            } else {
                videoCodecCandidate.store(detectedCodec, std::memory_order_relaxed);
                videoCodecCandidateCount.store(1, std::memory_order_relaxed);
            }
            const int candidateCount = videoCodecCandidateCount.load(std::memory_order_relaxed);
            if (candidateCount < 2) {
                dsyslog("vaapivideo/device: video codec %s same as previous -- awaiting confirmation (%d/2)",
                        avcodec_get_name(detectedCodec), candidateCount);
                return Length;
            }
            dsyslog("vaapivideo/device: video codec %s confirmed after %d detections", avcodec_get_name(detectedCodec),
                    candidateCount);
        }

        // Wait for in-band config before opening: H.264/HEVC need the SPS bit-depth/profile for
        // backend selection (else the 8-bit row misclassifies 10-bit streams); MPEG-2 needs the
        // sequence_extension progressive_sequence for the deinterlace verdict (FrameNeedsDeinterlace).
        // DVB carries both each GOP, so the wait is bounded.
        VideoStreamInfo streamInfo;
        streamInfo.codecId = detectedCodec;
        if (detectedCodec == AV_CODEC_ID_MPEG2VIDEO || detectedCodec == AV_CODEC_ID_H264 ||
            detectedCodec == AV_CODEC_ID_HEVC) {
            streamInfo = ::ProbeVideoSps(detectedCodec, {pes.payload, pes.payloadSize});
            if (!streamInfo.hasSps || (detectedCodec == AV_CODEC_ID_MPEG2VIDEO && !streamInfo.hasStreamInterlaceInfo))
                [[unlikely]] {
                return Length;
            }
        }

        // Clear the same-codec confirmation only now that config is in hand. A `return Length` above
        // leaves the candidate confirmed, so the next PES resumes at the probe instead of restarting
        // the 2-of-2 count -- which could otherwise keep skipping the sparse GOP-header PES.
        videoCodecCandidate.store(AV_CODEC_ID_NONE, std::memory_order_relaxed);
        videoCodecCandidateCount.store(0, std::memory_order_relaxed);

        if (!decoder->OpenCodecWithInfo(streamInfo)) [[unlikely]] {
            esyslog("vaapivideo/device: failed to open video codec %s", avcodec_get_name(detectedCodec));
            return Length;
        }

        videoCodecId.store(detectedCodec, std::memory_order_relaxed);
        ResetNoVideoMonitors(); // video now decodes: neither no-video screen may repaint over it
        isyslog("vaapivideo/device: video codec %s (%s)", avcodec_get_name(detectedCodec), isLive ? "live" : "replay");
    }

    // Replay backpressure: return 0 so VDR retries via Poll(). Live: never block.
    // Trick mode caps queue to DECODER_TRICK_QUEUE_DEPTH for immediate keyframe visibility.
    if (!isLive) [[unlikely]] {
        const int currentSpeed = trickSpeed.load(std::memory_order_relaxed);

        if (currentSpeed != 0) {
            if (!decoder->IsReadyForNextTrickFrame() || decoder->GetQueueSize() >= DECODER_TRICK_QUEUE_DEPTH) {
                return 0;
            }
        } else if (decoder->IsQueueFull()) {
            return 0;
        }
    }

    decoder->EnqueueData(pes.payload, pes.payloadSize, pes.pts);
    return Length;
}

[[nodiscard]] auto cVaapiDevice::HasFeedSpace(int currentSpeed) const -> bool {
    // Trick: also gate on the per-frame pacing timer to avoid burst-feeding past display rate.
    if (currentSpeed != 0) {
        // Audio-only trick is paced solely by that timer; its decoder queue stays empty, so the depth
        // check would only add packetMutex traffic to the Poll() spin.
        if (!IsPlayingVideo()) {
            return decoder->IsReadyForNextTrickFrame();
        }
        return decoder->IsReadyForNextTrickFrame() && decoder->GetQueueSize() < DECODER_TRICK_QUEUE_DEPTH;
    }
    return !decoder->IsQueueFull() && (!audioProcessor || audioProcessor->GetQueueSize() < AUDIO_QUEUE_HIGHWATER);
}

[[nodiscard]] auto cVaapiDevice::Poll(cPoller & /*Poller*/, int TimeoutMs) -> bool {
    if (!decoder) [[unlikely]] {
        return true;
    }

    // cTransfer pushes via PlayVideo/PlayAudio directly and never calls Poll().
    if (liveMode.load(std::memory_order_relaxed)) {
        return true;
    }

    const int currentSpeed = trickSpeed.load(std::memory_order_relaxed);

    if (HasFeedSpace(currentSpeed)) {
        return true;
    }

    // Spin in-place: returning false causes VDR to retry through its outer Poll() loop,
    // which adds a full MainLoopInterval (~100 ms) per cycle and visibly slows startup/seek.
    if (TimeoutMs > 0) {
        const cTimeMs timeout(TimeoutMs);
        while (!HasFeedSpace(currentSpeed) && !timeout.TimedOut()) {
            cCondWait::SleepMs(5);
        }
        return HasFeedSpace(currentSpeed);
    }

    return false;
}

[[nodiscard]] auto cVaapiDevice::Ready() -> bool {
    // VDR override, polled ONLY by cDevice::WaitForAllDevicesReady() (30 s cap) at startup, which
    // blocks bring-up until every device is ready. A --detached device never attaches during startup,
    // so gating on initState==2 pinned it for the full timeout. A deliberately-detached device is done
    // with startup -> report ready; only an in-progress attach (initState==1) is genuinely not-ready.
    // Hardware-dependent paths use HardwareReady()/IsReady(), never this override.
    return initState.load(std::memory_order_acquire) != 1;
}

auto cVaapiDevice::ScaleVideo(const cRect &rect) -> void {
    // HardwareReady()-gated: skindesigner can drive ScaleVideo() (main thread) during an in-flight ATTA.
    if (!HardwareReady() || !display) [[unlikely]] {
        return;
    }
    // On dim change, the decoder rebuilds the VPP chain to emit fbs pre-sized for the new rect.
    const bool needsFilterRebuild = display->SetVideoRect(rect);
    if (needsFilterRebuild) {
        dsyslog("vaapivideo/device: ScaleVideo rect=(%d,%d %dx%d) (rebuild)", rect.X(), rect.Y(), rect.Width(),
                rect.Height());
    }
    if (needsFilterRebuild && decoder) {
        decoder->RequestFilterRebuild();
    }
}

[[nodiscard]] auto cVaapiDevice::SetZoom(int stop) -> int {
    // Crop is applied in the VPP chain; a stop change just needs a filter-graph rebuild, reusing
    // the ScaleVideo() mechanism. zoomActive is read by the decoder when it rebuilds. Selecting a
    // disabled (level 0) preset is treated as Off so the label/crop never claim a no-op zoom.
    int clamped = std::clamp(stop, 0, CONFIG_ZOOM_PRESET_COUNT);
    if (clamped > 0 && vaapiConfig.zoomLevel[clamped - 1].load(std::memory_order_relaxed) <= 0) {
        clamped = 0;
    }
    const int previous = vaapiConfig.zoomActive.exchange(clamped, std::memory_order_relaxed);
    if (previous != clamped) {
        dsyslog("vaapivideo/device: zoom stop %d -> %d (rebuild)", previous, clamped);
        // zoomActive (set above) persists regardless; HardwareReady()-gate the decoder read so a zoom
        // key during an in-flight ATTA can't touch a half-published decoder.
        if (HardwareReady() && decoder) {
            decoder->RequestFilterRebuild();
        }
    }
    return clamped;
}

namespace {
// Next cycle stop after `current`, skipping disabled (level 0) presets. Off (stop 0) is always a
// valid stop, so the loop always terminates; returns `current` only when every level is disabled.
auto NextZoomStop(int current) -> int {
    for (int step = 1; step <= CONFIG_ZOOM_PRESET_COUNT + 1; ++step) {
        const int cand = (current + step) % (CONFIG_ZOOM_PRESET_COUNT + 1);
        if (cand == 0 || vaapiConfig.zoomLevel[cand - 1].load(std::memory_order_relaxed) > 0) {
            return cand;
        }
    }
    return current; // unreachable: Off is always a stop
}
} // namespace

[[nodiscard]] auto cVaapiDevice::CycleZoom() -> int {
    // Advance to the next stop, skipping disabled (level 0) presets so a single configured level
    // still cycles Off <-> that level. CAS so two callers (replay ProcessKey on the main thread,
    // ZOOM next on the SVDRP thread) can't both read the same stop and collapse into one step.
    int current = vaapiConfig.zoomActive.load(std::memory_order_relaxed);
    int next = NextZoomStop(current);
    while (!vaapiConfig.zoomActive.compare_exchange_weak(current, next, std::memory_order_relaxed)) {
        next = NextZoomStop(current); // compare_exchange_weak refreshed `current`
    }
    if (next != current) {
        dsyslog("vaapivideo/device: zoom stop %d -> %d (rebuild)", current, next);
        if (HardwareReady() && decoder) { // acquire-gate the decoder read against an in-flight ATTA
            decoder->RequestFilterRebuild();
        }
    }
    return next;
}

auto cVaapiDevice::RefreshZoom() -> void {
    // Setup-menu edited the crop values of the live preset: request a rebuild WITHOUT touching
    // zoomActive, so a concurrent CycleZoom()/SetZoom() selection can't be reverted by a stale snapshot.
    const int stop = vaapiConfig.zoomActive.load(std::memory_order_relaxed);
    if (stop < 1 || stop > CONFIG_ZOOM_PRESET_COUNT ||
        vaapiConfig.zoomLevel[stop - 1].load(std::memory_order_relaxed) <= 0) {
        return;
    }
    dsyslog("vaapivideo/device: zoom stop %d refresh (rebuild)", stop);
    if (HardwareReady() && decoder) { // acquire-gate the decoder read against an in-flight ATTA
        decoder->RequestFilterRebuild();
    }
}

auto cVaapiDevice::RefreshVideoFilters() -> void {
    // Reuse the zoom rebuild path so a post-processing policy change applies to live playback now, not at the next
    // channel/codec change; the decoder re-reads vaapiConfig when it reassembles BuildParams.
    if (HardwareReady() && decoder) { // acquire-gate the decoder read against an in-flight ATTA
        dsyslog("vaapivideo/device: filter policy refresh (rebuild)");
        decoder->RequestFilterRebuild();
    }
}

auto cVaapiDevice::ResetZoom() -> void {
    // Content change: drop to Off. The incoming stream rebuilds its own graph (reading zoomActive),
    // so no explicit RequestFilterRebuild here -- and decoder may be mid-teardown on some callers.
    const int previous = vaapiConfig.zoomActive.exchange(0, std::memory_order_relaxed);
    if (previous != 0) {
        dsyslog("vaapivideo/device: zoom reset to off (was stop %d)", previous);
    }
}

[[nodiscard]] auto cVaapiDevice::ZoomStatusLabel() const -> std::string {
    const int stop = vaapiConfig.zoomActive.load(std::memory_order_relaxed);
    if (stop < 1 || stop > CONFIG_ZOOM_PRESET_COUNT) {
        return "Zoom: off";
    }
    const int level = vaapiConfig.zoomLevel[stop - 1].load(std::memory_order_relaxed);
    if (level <= 0) {
        return "Zoom: off";
    }
    return std::format("Zoom {}: +{:.1f}%", stop, static_cast<double>(level) / 10.0);
}

auto cVaapiDevice::SetAudioTrackDevice(eTrackType /*Type*/) -> void {
    // Fires AFTER currentAudioTrack is assigned, so the read in the handler is reliable.
    // In practice only reached when no cPlayer is attached (live=cTransfer,
    // replay=cDvbPlayer both count as cPlayers in modern VDR).
    HandleAudioTrackChange("SetAudioTrackDevice", /*enteringDolby=*/false);
}

auto cVaapiDevice::SetDigitalAudioDevice(bool On) -> void {
    // Fired for both live and replay track changes. On=true means dolby entry, and VDR
    // fires it BEFORE assigning currentAudioTrack (vdr/device.c:1172-1180), so the handler
    // cannot rely on GetCurrentAudioTrack() and uses the dolby slot walk instead.
    HandleAudioTrackChange("SetDigitalAudioDevice", /*enteringDolby=*/On);
}

[[nodiscard]] auto cVaapiDevice::SetPlayMode(ePlayMode PlayMode) -> bool {
    static constexpr const char *kModeNames[] = {
        "pmNone", "pmAudioVideo", "pmAudioOnly", "pmAudioOnlyBlack", "pmVideoOnly", "pmExtern",
    };
    const auto idx = static_cast<unsigned>(PlayMode);
    dsyslog("vaapivideo/device: SetPlayMode(%s) called", idx < std::size(kModeNames) ? kModeNames[idx] : "unknown");

    // Stream boundary, same reason as in Clear(): a candidate armed for the outgoing stream must
    // not mature against the incoming one.
    InvalidateDisplayModeCandidate();

    // External-player handover (vdr-mpv): suspend hardware so it can grab DRM/VAAPI/ALSA.
    // SuspendHardware (not Detach) -- cMpvControl is live, cControl::Shutdown would double-free.
    // Arm externActive only on actual suspend; --detached startup uses MakePrimaryDevice's
    // deferred-init path instead.
    if (PlayMode == pmExtern_THIS_SHOULD_BE_AVOIDED) {
        if (initState.load(std::memory_order_acquire) != 0) {
            isyslog("vaapivideo/device: pmExtern -- releasing hardware for external player");
            SuspendHardware();
            externActive.store(true, std::memory_order_release);
        }
        return true;
    }

    // Resume before the pmNone branch dereferences `decoder` (SuspendHardware nulled it).
    if (externActive.exchange(false, std::memory_order_acq_rel) && initState.load(std::memory_order_acquire) == 0 &&
        !drmPath.empty()) {
        isyslog("vaapivideo/device: SetPlayMode(%s) -- resuming hardware after pmExtern",
                idx < std::size(kModeNames) ? kModeNames[idx] : "unknown");
        if (!Attach()) [[unlikely]] {
            esyslog("vaapivideo/device: failed to resume hardware after pmExtern");
            return false;
        }
    }

    // Clear stale Freeze() flag first -- would otherwise block PlayVideo() in the new mode.
    // Propagate the un-pause to decoder + display: VDR drives "exit playback while paused" as
    // Blue/Stop -> SetPlayMode(pmNone) WITHOUT a prior Play(), so without this the drain hold
    // set by Freeze() (decoder->devicePaused, display->devicePaused) stays asserted across the
    // SetPlayMode transition. The next live-TV / replay session then feeds packets into the
    // decoder, jitterBuf grows to DECODER_RESERVE_HARD_CAP, and every subsequent frame overflows
    // (visible as "jitterBuf overflow -- dropped N" spam at ~50 fps with no picture on screen).
    // Audio's clockPaused is cleared by the Clear() path below via ResetPlaybackClock().
    paused.store(false, std::memory_order_relaxed);
    if (decoder) [[likely]] {
        decoder->SetDevicePaused(false);
    }
    if (display) [[likely]] {
        display->SetDevicePaused(false);
    }

    switch (PlayMode) {
        case pmNone:
            // Capture previous video codec BEFORE clearing: PlayVideo's 2-of-2 guard uses it
            // to reject stale TS-buffer bytes after the switch. Audio path runs the guard
            // unconditionally so it doesn't need a previous-codec capture.
            ResetNoVideoMonitors();
            previousVideoCodec.store(videoCodecId.exchange(AV_CODEC_ID_NONE, std::memory_order_relaxed),
                                     std::memory_order_relaxed);
            // Drop the audio log-dedup baseline so the next session logs its codec once (scrub Clear()s
            // keep it, suppressing the per-seek re-confirm spam; a real session boundary re-arms the log).
            previousAudioCodec.store(AV_CODEC_ID_NONE, std::memory_order_relaxed);
            liveMode.store(false, std::memory_order_relaxed);
            trickSpeed.store(0, std::memory_order_release);
            trickAudioPts.store(AV_NOPTS_VALUE, std::memory_order_relaxed);
            ResetReplayAudioEofBaseline(); // fresh session: no stale flush-repeat baseline
            if (decoder) [[likely]] {
                decoder->SetTrickSpeed(0);
                decoder->SetLiveMode(false);
                decoder->RequestCodecReopen();
            }
            videoCodecCandidate.store(AV_CODEC_ID_NONE, std::memory_order_relaxed);
            videoCodecCandidateCount.store(0, std::memory_order_relaxed);
            // Reset dedup so the new channel's first audio-track hook re-detects.
            lastHandledAudioTrack = ttNone;
            lastHandledAudioPid = 0;
            // Reset transient zoom BEFORE Clear() so any post-clear rebuild observes Off, never the
            // old content's stop. Manual zoom is a per-content choice -- drop it on every change.
            ResetZoom();
            Clear();
            // Playback ended (or a zap started). Arm the deferred mode restore -- if something
            // decodes within the window this is disarmed again, so an ordinary zap never bounces
            // the mode; if nothing does, the output comes back off the previous stream's mode.
            ScheduleIdleModeRestore();
            // User opt-in: paint black during channel-switch gap instead of holding the
            // previous channel's last frame until the new one decodes its first frame.
            if (vaapiConfig.clearOnChannelSwitch.load(std::memory_order_relaxed)) {
                static_cast<void>(SubmitBlackFrame());
            }
            break;
        case pmAudioOnly:
        case pmAudioOnlyBlack:
            // Radio: paint channel name + present EPG title over black so DRM scanout doesn't hold
            // the previous channel's picture; PlayAudio keeps it current as the program changes.
            ResetNoVideoMonitors(); // RefreshRadioSplash below re-arms radioSplashActive for this channel
            ResetZoom();            // Audio-only is still a content change; keep transient-zoom semantics.
            Clear();
            // Encrypted radio: arm the watchdog too, so a scrambled audio-only channel that never
            // decodes gets the "encrypted" notice instead of a silent radio splash.
            encryptedDeadlineMs.store(cTimeMs::Now() + ENCRYPTED_NOTICE_DELAY_MS, std::memory_order_relaxed);
            // Poll deadline is left to the PlayAudio thread (which owns radioSplashPoll); its first
            // refresh after entry recomputes the same event and no-ops, then arms the 2 s cadence.
            RefreshRadioSplash(/*force=*/true);
            break;
        case pmAudioVideo:
        case pmVideoOnly:
            // Codec state already clean (VDR always emits pmNone before pmAudioVideo).
            // liveMode is latched on the first PES in PlayVideo/PlayAudio.
            ResetNoVideoMonitors(); // a video stream is starting; re-armed below for the no-video grace
            ResetZoom(); // Before Clear(): belt-and-braces so a new stream starts at Off even if pmNone was skipped.
            Clear();
            // Shared grace: if no video arrives, PlayAudio() paints black (radio channel).
            radioBlackTimer.Set(ENCRYPTED_NOTICE_DELAY_MS);
            radioBlackPending.store(true, std::memory_order_relaxed);
            // Same grace for the encrypted-channel notice: armed unconditionally, it self-cancels
            // when a codec opens and only paints if the channel turns out encrypted with a video PID.
            encryptedDeadlineMs.store(cTimeMs::Now() + ENCRYPTED_NOTICE_DELAY_MS, std::memory_order_relaxed);
            dsyslog("vaapivideo/device: pmAudioVideo -- armed no-video watchdogs (grace %d ms)",
                    ENCRYPTED_NOTICE_DELAY_MS);
            break;
        default:
            ResetNoVideoMonitors(); // pmExtern / unknown: not our scanout
            break;
    }
    return true;
}

auto cVaapiDevice::SetVolumeDevice(int Volume) -> void {
    // HardwareReady()-gated: VDR-core volume path runs on the main thread during an in-flight ATTA.
    if (HardwareReady() && audioProcessor) [[likely]] {
        audioProcessor->SetVolume(Volume);
    }
}

auto cVaapiDevice::StillPicture(const uchar *Data, int Length) -> void {
    if (!Data || Length <= 0) [[unlikely]] {
        return;
    }

    // Re-entry guard: TS path re-enters this method as PES; only the outermost call manages state.
    const bool isOuterCall = !inStillPicture;

    bool wasPaused = false;
    if (isOuterCall) {
        inStillPicture = true;
        // Mirror the decoder reset below in the device's own trick state: VDR's Goto(..., Still) calls
        // Play() only when paused (vdr/dvbplayer.c cDvbPlayer::Goto), so a mark-jump out of FF/REW would
        // otherwise leave PlayVideo/Poll applying trick pacing to the still frames while the decoder
        // already runs at normal speed.
        trickSpeed.store(0, std::memory_order_release);
        if (decoder) [[likely]] {
            decoder->Clear();
            decoder->SetTrickSpeed(0);
            decoder->SetStillPictureMode(true);
        }
        wasPaused = paused.exchange(false, std::memory_order_relaxed);
    }

    if (Data[0] == 0x47) {
        cDevice::StillPicture(Data, Length);
    } else {
        // Raw PES buffer: walk each start-code-prefixed packet (PlayVideo takes one per call).
        int offset = 0;
        while (offset < Length) {
            const int remaining = Length - offset;
            if (remaining < 9 || Data[offset] != 0x00 || Data[offset + 1] != 0x00 || Data[offset + 2] != 0x01)
                [[unlikely]] {
                break;
            }
            const auto pesField = static_cast<int>((Data[offset + 4] << 8) | Data[offset + 5]);
            const int pesLen = (pesField > 0) ? std::min(pesField + 6, remaining) : remaining;
            if (pesLen < 9) [[unlikely]] {
                break;
            }
            (void)PlayVideo(Data + offset, pesLen);
            offset += pesLen;
        }
    }

    if (isOuterCall) {
        if (decoder) [[likely]] {
            decoder->FlushParser();
            decoder->RequestCodecDrain();
        }
        if (wasPaused) {
            paused.store(true, std::memory_order_relaxed);
        }
        inStillPicture = false;
    }
}

auto cVaapiDevice::TrickSpeed(int Speed, bool Forward) -> void {
    dsyslog("vaapivideo/device: TrickSpeed(%d, %s)", Speed, Forward ? "forward" : "backward");

    // VDR convention: Freeze() is called BEFORE slow trick but NOT before fast FF/REW.
    // paused==true here therefore implies slow mode.
    const bool isFast = !paused.load(std::memory_order_relaxed);

    // A reference surviving from an earlier trick would fake the first step's content distance (slow-
    // forward exits without a Clear(); speed/direction changes never see one). NOPTS = free first step.
    // BEFORE the trickSpeed release-store: GetSTC()'s acquire pairs with it, so a reader seeing the new
    // speed can no longer read the previous trick's position.
    trickAudioPts.store(AV_NOPTS_VALUE, std::memory_order_relaxed);
    trickSpeed.store(Speed, std::memory_order_release);

    // Both display-mode entry points bail while trickSpeed != 0, so a candidate armed just before
    // this would sit untouched for the whole (arbitrarily long) trick session and then be applied
    // on the first tick after it ends -- possibly against different content entirely.
    InvalidateDisplayModeCandidate();

    if (decoder) [[likely]] {
        decoder->SetTrickSpeed(Speed, Forward, isFast);
    }

    // Drop audio explicitly: VDR usually calls Mute() too but timing varies, and the
    // queued sink content (~100-200 ms) would be audible after the trick command. DropOutput
    // preserves the clock -- on trick-exit, WritePcmToAlsa()'s >5s-jump guard handles any
    // timeline shift automatically.
    if (audioProcessor) [[likely]] {
        audioProcessor->DropOutput();
    }
}

// ============================================================================
// === PUBLIC API ===
// ============================================================================

[[nodiscard]] auto cVaapiDevice::Attach() -> bool {
    // Opens hardware using arguments latched by Initialize(). Used after detached startup
    // and after a Detach() -> SVDRP ATTA cycle.
    if (drmPath.empty()) [[unlikely]] {
        esyslog("vaapivideo/device: cannot attach -- Initialize() has not run yet");
        return false;
    }
    isyslog("vaapivideo/device: attaching to hardware (DRM=%s audio=%s connector=%s)", drmPath.c_str(),
            audioDevice.c_str(), connectorName.empty() ? "auto" : connectorName.c_str());

    if (!AttachHardware()) [[unlikely]] {
        return false;
    }

    // Mirrors the MakePrimaryDevice() OSD branch: the detached-primary -> SVDRP ATTA path
    // runs MakePrimaryDevice() before display exists, so Attach() must wire it up here.
    if (IsPrimaryDevice()) {
        if (auto *provider = dynamic_cast<cVaapiOsdProvider *>(::osdProvider)) {
            provider->AttachDisplay(display.get());
            dsyslog("vaapivideo/device: OSD provider reattached to display");
        } else {
            ::osdProvider = new cVaapiOsdProvider(display.get());
            isyslog("vaapivideo/device: OSD provider registered");
        }
    }

    return true;
}

auto cVaapiDevice::Detach() -> bool {
    // Idempotent: skip teardown if never attached, avoiding a spurious VT yield.
    if (initState.load(std::memory_order_acquire) == 0) [[unlikely]] {
        dsyslog("vaapivideo/device: detach requested but hardware is not attached (no-op)");
        return true;
    }

    isyslog("vaapivideo/device: detaching from hardware");

    // cControl::Shutdown before SuspendHardware: ~cPlayer's unwinding fires
    // SetPlayMode(pmNone)->Clear() and still needs decoder/display/audio alive.
    cControl::Shutdown();
    SuspendHardware();
    const bool vtYielded = LeaveOwnVt();
    isyslog("vaapivideo/device: detached");
    return vtYielded;
}

auto cVaapiDevice::SuspendHardware() -> void {
    // Hardware-only release; callers add cControl::Shutdown / LeaveOwnVt as appropriate.

    // Teardown barrier: leave "ready" up front so HardwareReady() turns false for the whole teardown
    // and the main-thread OSD/skin readers stop dereferencing display/decoder/audioProcessor before
    // Stop() frees them (symmetric with AttachHardware()'s release publish). Use 1, not 0: a racing
    // AttachHardware() CAS expects 0, so 1 keeps it rejected. The terminal store(0) below finalizes.
    initState.store(1, std::memory_order_release);

    DetachAllReceivers();

    if (auto *provider = dynamic_cast<cVaapiOsdProvider *>(::osdProvider)) {
        provider->DetachDisplay();
        dsyslog("vaapivideo/device: OSD provider detached from display");
    }

    Stop();

    // Force-release OSD mmap'd dumb buffers: VDR may keep cVaapiOsd objects alive past this
    // point, and their kernel refs would block drmDropMaster/close.
    if (auto *provider = dynamic_cast<cVaapiOsdProvider *>(::osdProvider)) {
        provider->ReleaseAllOsdResources();
    }

    // Drop master before close so another DRM client can take it immediately.
    if (drmFd >= 0) {
        drmDropMaster(drmFd);
    }

    ReleaseHardware();

    // Reset all playback state so a subsequent Attach() re-detects codecs from scratch.
    videoCodecId.store(AV_CODEC_ID_NONE, std::memory_order_relaxed);
    previousVideoCodec.store(AV_CODEC_ID_NONE, std::memory_order_relaxed);
    videoCodecCandidate.store(AV_CODEC_ID_NONE, std::memory_order_relaxed);
    videoCodecCandidateCount.store(0, std::memory_order_relaxed);
    // The no-video monitors are not cleared via SetPlayMode on the SVDRP DETA path, so reset them
    // here too -- otherwise a stale splash flag could fire against the next attached stream.
    ResetNoVideoMonitors();
    ResetAudioCodecState();
    ResetReplayAudioEofBaseline();
    liveMode.store(false, std::memory_order_relaxed);
    trickSpeed.store(0, std::memory_order_relaxed);
    trickAudioPts.store(AV_NOPTS_VALUE, std::memory_order_relaxed);
    paused.store(false, std::memory_order_relaxed);
    lastHandledAudioTrack = ttNone;
    lastHandledAudioPid = 0;
    osdWidth = 0;
    osdHeight = 0;
    osdModeGeneration = 0;

    // Drop the mode inventory: the next attach re-enumerates against the topology as it is then
    // (the sink may have moved), and the debounce state must not carry a pre-detach timestamp.
    {
        const cMutexLock lock(&displayModeMutex);
        connectorModes.clear();
        modeCandidates.clear();
        defaultMode = {};
        ClearModeCandidateLocked();
        modeIdleRestoreDueMs.store(0, std::memory_order_release);
        lastModeChangeMs = 0;
        lastRequest = {};
    }

    // Drop an auto-detected connector so the next attach re-selects against the current topology
    // (sticky auto-latching would hard-fail a re-attach if the sink moved). A -c connector stays pinned.
    if (!connectorUserSupplied) {
        connectorName.clear();
    }

    // Reset the Clear()-diagnostic baseline so the next attach's first Clear() logs as "(first call)".
    lastClearMs.store(0, std::memory_order_relaxed);
    lastClearLogMs.store(0, std::memory_order_relaxed);
    clearsSinceLog.store(0, std::memory_order_relaxed);

    initState.store(0, std::memory_order_release);
}

[[nodiscard]] auto cVaapiDevice::Initialize(std::string_view drmDevicePath, std::string_view audioDevicePath,
                                            std::string_view connectorNameFilter, bool deferred) -> bool {
    // One-shot latch: called exactly once per device lifetime by the plugin.
    // Subsequent attach cycles reuse these latched args via AttachHardware() / Attach().
    if (!drmPath.empty()) [[unlikely]] {
        esyslog("vaapivideo/device: Initialize() called twice (already latched DRM '%s')", drmPath.c_str());
        return false;
    }

    drmPath = drmDevicePath;
    audioDevice = audioDevicePath;
    connectorName = connectorNameFilter;
    connectorUserSupplied = !connectorNameFilter.empty(); // -c pins the connector; auto-latch is transient per attach

    if (deferred) {
        // Stay in state 0: hardware opens on first primary-device promotion after startup,
        // or on explicit SVDRP ATTA.
        isyslog("vaapivideo/device: starting detached -- DRM '%s', audio '%s', connector '%s' "
                "(hardware init deferred until attach)",
                drmPath.c_str(), audioDevice.c_str(), connectorName.empty() ? "auto" : connectorName.c_str());
        return true;
    }

    dsyslog("vaapivideo/device: initializing -- DRM '%s', audio '%s', connector '%s'", drmPath.c_str(),
            audioDevice.c_str(), connectorName.empty() ? "auto" : connectorName.c_str());
    return AttachHardware();
}

auto cVaapiDevice::MarkStartupComplete() noexcept -> void { startupComplete.store(true, std::memory_order_release); }

// ============================================================================
// === DISPLAY MODE SWITCHING ===
// ============================================================================

[[nodiscard]] auto cVaapiDevice::CurrentPlaybackSource() const noexcept -> PlaybackSource {
    // mediaPlayerAudioActive spans an open mediaplayer entry; liveMode is latched from
    // Transferring() on the first video PES. Anything else is cDvbPlayer replay.
    if (mediaPlayerAudioActive.load(std::memory_order_acquire)) {
        return PlaybackSource::MediaPlayer;
    }
    return liveMode.load(std::memory_order_relaxed) ? PlaybackSource::LiveTv : PlaybackSource::Replay;
}

[[nodiscard]] auto cVaapiDevice::ActiveRefreshMilliHz() const noexcept -> uint32_t {
    if (HardwareReady() && display && display->IsInitialized()) [[likely]] {
        return display->GetOutputRefreshMilliHz();
    }
    return vaapiConfig.display.GetRefreshRate() * 1000U;
}

namespace {

/// True when the operator has enabled mode switching for @p source.
[[nodiscard]] auto ModeSwitchEnabledFor(PlaybackSource source) noexcept -> bool {
    switch (source) {
        case PlaybackSource::LiveTv:
            return vaapiConfig.modeSwitchLiveTv.load(std::memory_order_relaxed);
        case PlaybackSource::Replay:
            return vaapiConfig.modeSwitchReplay.load(std::memory_order_relaxed);
        case PlaybackSource::MediaPlayer:
            return vaapiConfig.modeSwitchMediaplayer.load(std::memory_order_relaxed);
    }
    return false;
}

/// Snapshot the live policy. Taken once per evaluation so a setup-menu edit landing mid-decision
/// cannot mix old and new settings.
[[nodiscard]] auto SnapshotModePolicy(const drmModeModeInfo &defaultMode) noexcept -> DisplayModePolicy {
    return {.defaultHeight = defaultMode.vdisplay,
            .defaultRefreshMilliHz = ModeRefreshMilliHz(defaultMode),
            .defaultWidth = defaultMode.hdisplay,
            .matchRefresh = vaapiConfig.matchRefreshRate.load(std::memory_order_relaxed),
            .matchResolution = vaapiConfig.matchResolution.load(std::memory_order_relaxed),
            .maxRefreshMilliHz = MaxRefreshModeMilliHz(vaapiConfig.maxRefreshRate.load(std::memory_order_relaxed)),
            .minHeight = MinResolutionModeHeight(vaapiConfig.minResolution.load(std::memory_order_relaxed))};
}

/// Two modes are "the same output" iff size and exact rate agree; drmModeModeInfo also carries
/// timing detail that is irrelevant to whether a modeset is worth its HDMI relink.
[[nodiscard]] auto SameOutputMode(const drmModeModeInfo &a, const drmModeModeInfo &b) noexcept -> bool {
    return a.hdisplay == b.hdisplay && a.vdisplay == b.vdisplay && ModeRefreshMilliHz(a) == ModeRefreshMilliHz(b);
}

} // namespace

auto cVaapiDevice::ArmModeCandidateLocked(const StreamModeRequest &request, uint64_t nowMs) -> void {
    modeCandidate = request;
    modeCandidateSinceMs = nowMs;
    // Publish a non-zero due time last so PollPendingDisplayMode()'s lock-free probe never sees an
    // armed window before the payload behind it. (+1 guards the theoretical nowMs==0 tick.)
    modeCandidateDueMs.store(nowMs + DISPLAY_MODE_STABLE_MS + 1, std::memory_order_release);
}

auto cVaapiDevice::ClearModeCandidateLocked() -> void {
    modeCandidateDueMs.store(0, std::memory_order_release);
    modeCandidate = {};
    modeCandidateSinceMs = 0;
}

auto cVaapiDevice::ApplyDisplayModePolicy(const StreamModeRequest &request, PlaybackSource source, uint64_t nowMs)
    -> void {
    // Caller holds displayModeMutex. Runs the matcher and stages the winner; the stability gate
    // and the scope/policy gates are the caller's business.

    // Rate limit: each change costs an HDMI link retrain, so back-to-back switches must be
    // impossible even when a playlist advances through short entries. Deliberately returns with
    // the candidate STILL ARMED -- PollPendingDisplayMode() re-enters at the cooldown boundary the
    // due time is pushed to, so a throttled switch is deferred rather than dropped. Leaving the
    // due time in the past would instead have the poll take this mutex on every decode tick.
    if (lastModeChangeMs != 0 && nowMs - lastModeChangeMs < DISPLAY_MODE_MIN_INTERVAL_MS) {
        modeCandidateDueMs.store(lastModeChangeMs + DISPLAY_MODE_MIN_INTERVAL_MS, std::memory_order_release);
        return;
    }

    const auto match = SelectDisplayMode(modeCandidates, request, SnapshotModePolicy(defaultMode));
    // Disarm before the verdict is acted on: a matcher that found nothing for this inventory will
    // find nothing every time, so leaving the candidate armed would re-run it (and re-allocate its
    // reason string) on every single decode-loop tick for the rest of the session.
    ClearModeCandidateLocked();
    if (match.index < 0) {
        return;
    }
    const auto &winner = connectorModes.at(modeCandidates.at(static_cast<size_t>(match.index)).index);
    if (SameOutputMode(winner, display->GetActiveMode())) {
        // The common case once settled -- and what closes the rebuild -> publish -> evaluate loop,
        // since the rebuilt filter graph republishes the very format that produced this mode.
        return;
    }

    // Carries the stream too: without it the log cannot say whether the chosen resolution tracked
    // the source or was clamped by the configured minimum.
    isyslog("vaapivideo/device: display mode switch (%s): stream %ux%u@%.3fHz -> %s", PlaybackSourceName(source),
            request.width, request.height, static_cast<double>(request.rateMilliHz) / 1000.0, match.reason.c_str());
    display->RequestDisplayMode(winner);
    lastModeChangeMs = nowMs;
}

auto cVaapiDevice::RestoreDefaultModeLocked(uint64_t nowMs) -> void {
    // Caller holds displayModeMutex. Used whenever mode switching is not in charge of the current
    // source: without it a mode installed by a *different* source stays on the CRTC forever (stop
    // a 24p recording with live-TV switching off and live TV would run at 24 Hz for the session).
    ClearModeCandidateLocked();
    modeIdleRestoreDueMs.store(0, std::memory_order_release);
    if (defaultMode.hdisplay == 0) {
        return;
    }
    if (SameOutputMode(defaultMode, display->GetActiveMode())) {
        return; // the overwhelmingly common case: nothing ever moved the mode
    }
    // No rate limit here, unlike ApplyDisplayModePolicy: this path is edge-driven with nothing left
    // armed to retry it, so throttling it would silently drop the restore and strand the output on
    // another source's mode. Returning to the default is idempotent and cannot oscillate.
    isyslog("vaapivideo/device: restoring default display mode %ux%u@%.3fHz", defaultMode.hdisplay,
            defaultMode.vdisplay, static_cast<double>(ModeRefreshMilliHz(defaultMode)) / 1000.0);
    display->RequestDisplayMode(defaultMode);
    lastModeChangeMs = nowMs;
}

auto cVaapiDevice::EvaluateDisplayMode(const StreamModeRequest &request, PlaybackSource source, bool immediate)
    -> void {
    if (!HardwareReady() || !display) {
        return;
    }
    // Disarmed ahead of every other gate: a publish means something is decoding, which is exactly
    // what the idle watchdog waits to rule out -- and a stream that cannot be matched, or one being
    // scrubbed, is still decoding. SetPlayMode(pmNone) arms it, and a zap emits pmNone and then
    // decodes, so this is what keeps an ordinary zap from bouncing the mode.
    modeIdleRestoreDueMs.store(0, std::memory_order_release);
    // Trick play paces frames off-cadence and is transient: a modeset would cost a second of black
    // mid-scrub for a rate that reverts the moment it ends.
    if (trickSpeed.load(std::memory_order_relaxed) != 0) {
        return;
    }

    const cMutexLock lock(&displayModeMutex);
    // A connector whose timings are all interlaced or all off-aspect leaves the candidates empty;
    // the matcher can never satisfy anything, so there is nothing to remember or arm.
    if (modeCandidates.empty()) {
        return;
    }
    // Remembered even when the switches are off, so flipping one on in the setup menu can act on
    // the stream already playing (see ReevaluateDisplayMode).
    lastRequest = request;
    lastRequestSource = source;

    const uint64_t nowMs = cTimeMs::Now();
    const auto policy = SnapshotModePolicy(defaultMode);
    // No usable format (an unmatchable rate is published as 0 rather than guessed), source switch
    // off, or neither policy enabled: all three mean this stream cannot drive the mode, so hand the
    // output back to the default rather than leaving it on whatever a PREVIOUS stream installed.
    // Done on the first filter build so the transition lands in the same window as any other
    // switch. A no-op in the default configuration, where the mode never moved.
    if (!request.IsValid() || !ModeSwitchEnabledFor(source) || (!policy.matchRefresh && !policy.matchResolution)) {
        RestoreDefaultModeLocked(nowMs);
        return;
    }

    if (immediate) {
        ClearModeCandidateLocked();
        ApplyDisplayModePolicy(request, source, nowMs);
        return;
    }

    // Stability gate: live-TV formats churn while the tuner settles, so only a format that held
    // long enough to be the one the viewer landed on may drive a modeset. Armed here, matured by
    // PollPendingDisplayMode() -- the reactive publish fires once per filter-graph build, so a
    // second identical notification would never arrive on its own.
    if (modeCandidateSinceMs == 0 || !(modeCandidate == request)) {
        ArmModeCandidateLocked(request, nowMs);
        return;
    }
    if (nowMs - modeCandidateSinceMs < DISPLAY_MODE_STABLE_MS) {
        return;
    }
    ApplyDisplayModePolicy(request, source, nowMs);
}

auto cVaapiDevice::PollPendingDisplayMode() -> void {
    // Driven by the decode loop's per-iteration tick (which also runs on the ~100 ms idle ticks),
    // so an armed candidate matures on wall-clock time instead of waiting for a second publish that
    // a steady stream never produces.
    //
    // Lock-free fast path first: with nothing armed -- the steady state -- this must cost two
    // loads, not a mutex acquisition.
    const uint64_t candidateDueMs = modeCandidateDueMs.load(std::memory_order_acquire);
    const uint64_t idleDueMs = modeIdleRestoreDueMs.load(std::memory_order_acquire);
    if (candidateDueMs == 0 && idleDueMs == 0) {
        return;
    }
    const uint64_t probeMs = cTimeMs::Now();
    const bool candidateDue = candidateDueMs != 0 && probeMs >= candidateDueMs;
    const bool idleDue = idleDueMs != 0 && probeMs >= idleDueMs;
    if (!candidateDue && !idleDue) {
        return;
    }
    if (!HardwareReady() || !display) {
        return;
    }
    if (trickSpeed.load(std::memory_order_relaxed) != 0) {
        return;
    }

    const cMutexLock lock(&displayModeMutex);
    if (modeCandidates.empty()) {
        return;
    }
    const uint64_t nowMs = cTimeMs::Now();

    // Re-check under the lock: another thread may have disarmed or re-armed since the probe.
    if (modeCandidateSinceMs != 0 && modeCandidate.IsValid() &&
        nowMs - modeCandidateSinceMs >= DISPLAY_MODE_STABLE_MS) {
        // Re-check the gates: the operator may have changed a setting while the candidate aged.
        const auto policy = SnapshotModePolicy(defaultMode);
        if (!ModeSwitchEnabledFor(lastRequestSource) || (!policy.matchRefresh && !policy.matchResolution)) {
            RestoreDefaultModeLocked(nowMs);
            return;
        }
        ApplyDisplayModePolicy(modeCandidate, lastRequestSource, nowMs);
        return;
    }

    // Idle watchdog: nothing has published a format for a while, i.e. playback stopped without
    // anything decoding afterwards -- hand the mode back so the VDR menu does not sit at a film
    // rate for the session. The DEADLINE is re-read under the lock, not just its non-zero-ness:
    // judging a re-arm that landed between the probe and the lock (stop, then an immediate zap)
    // against the pre-lock snapshot restored the default ~5 s early.
    const uint64_t idleDueNowMs = modeIdleRestoreDueMs.load(std::memory_order_acquire);
    if (idleDueNowMs != 0 && nowMs >= idleDueNowMs && modeCandidateSinceMs == 0) {
        // The stream that installed the mode is gone, so its format must not be resurrected by a
        // later setup-menu confirm (ReevaluateDisplayMode replays lastRequest).
        lastRequest = {};
        RestoreDefaultModeLocked(nowMs);
    }
}

auto cVaapiDevice::NotifyStreamFormat(const StreamModeRequest &request) -> void {
    EvaluateDisplayMode(request, CurrentPlaybackSource(), /*immediate=*/false);
}

auto cVaapiDevice::InvalidateDisplayModeCandidate() -> void {
    // Cheap lock-free guard: the candidate is armed for at most ~1.5 s at a time, so on the vast
    // majority of Clear()/SetPlayMode() calls there is nothing to drop.
    if (modeCandidateDueMs.load(std::memory_order_acquire) == 0) {
        return;
    }
    const cMutexLock lock(&displayModeMutex);
    ClearModeCandidateLocked();
}

auto cVaapiDevice::ScheduleIdleModeRestore() -> void {
    // Deferred rather than restoring now: VDR emits pmNone immediately before pmAudioVideo on an
    // ordinary zap, so restoring here would bounce the mode twice per zap. Anything that starts
    // decoding disarms it (EvaluateDisplayMode, ahead of every other gate), leaving this to fire
    // only when playback really ended into something that decodes nothing.
    if (!HardwareReady() || !display) {
        return;
    }
    // Armed unconditionally, not only when off the default: GetActiveMode() names the last mode the
    // display thread finished PROGRAMMING, so a stop landing inside an in-flight switch would see
    // the outgoing mode and leave the incoming one unwatched. A spurious arm costs one no-op poll --
    // RestoreDefaultModeLocked() early-outs when the mode really is the default.
    modeIdleRestoreDueMs.store(cTimeMs::Now() + DISPLAY_MODE_IDLE_RESTORE_MS, std::memory_order_release);
}

auto cVaapiDevice::ReevaluateDisplayMode() -> void {
    StreamModeRequest request;
    PlaybackSource source{};
    {
        const cMutexLock lock(&displayModeMutex);
        request = lastRequest;
        source = lastRequestSource;
        ClearModeCandidateLocked();
        lastModeChangeMs = 0; // an explicit operator action must not be swallowed by the rate limit
    }
    if (request.IsValid()) {
        EvaluateDisplayMode(request, source, /*immediate=*/true);
    } else {
        ResetDisplayModeToDefault();
    }
}

auto cVaapiDevice::ResetDisplayModeToDefault() -> void {
    if (!HardwareReady() || !display) {
        return;
    }
    const cMutexLock lock(&displayModeMutex);
    if (defaultMode.hdisplay == 0) {
        return;
    }
    ClearModeCandidateLocked();
    modeIdleRestoreDueMs.store(0, std::memory_order_release); // the request below is the restore
    lastRequest = {};
    lastModeChangeMs = 0; // an explicit stop must not be swallowed by the rate limit
    // Deliberately NOT gated on GetActiveMode(): that names the last mode the display thread
    // finished programming, so a stop landing inside an in-flight switch would see the old mode and
    // strand the output on the new one. RequestDisplayMode is idempotent, so requesting
    // unconditionally is free and overrides a still-queued request.
    // Worded apart from RestoreDefaultModeLocked()'s line: that one is the automatic policy edge,
    // this one an explicit external reset (setup change, player teardown, SVDRP).
    dsyslog("vaapivideo/device: display mode reset -- requesting default %ux%u@%.3fHz", defaultMode.hdisplay,
            defaultMode.vdisplay, static_cast<double>(ModeRefreshMilliHz(defaultMode)) / 1000.0);
    display->RequestDisplayMode(defaultMode);
}

namespace {

/// " [anamorphic 64:45]" for a candidate whose raster is not square-pixel, empty otherwise. Tagged
/// because it is what lets a 1.25:1 720x576 raster pass an aspect gate anchored on a 16:9 panel --
/// otherwise the inventory looks wrong. Shared by the attach dump and the SVDRP MODE report.
[[nodiscard]] auto AnamorphicNote(std::span<const drmModeModeInfo> modes, const DisplayModeCandidate &candidate)
    -> std::string {
    if (candidate.index >= modes.size()) [[unlikely]] {
        return {};
    }
    const AspectRatio par = ModePixelAspectRatio(*std::next(modes.begin(), candidate.index));
    if (par.num == par.den) {
        return {};
    }
    return std::format(" [anamorphic {}:{}]", par.num, par.den);
}

} // namespace

[[nodiscard]] auto cVaapiDevice::DisplayModeReport() const -> std::string {
    if (!HardwareReady() || !display) {
        return "display not attached\n";
    }
    const auto programmedMode = display->GetActiveMode();
    // Whole report under displayModeMutex: runs on the SVDRP thread, where an attach or a
    // DETA-driven suspend would otherwise reallocate the inventory mid-iteration.
    const cMutexLock lock(&displayModeMutex);

    std::string out = std::format(
        "Active:  {}x{}@{:.3f}Hz\nDefault: {}x{}@{:.3f}Hz\n", programmedMode.hdisplay, programmedMode.vdisplay,
        static_cast<double>(ModeRefreshMilliHz(programmedMode)) / 1000.0, defaultMode.hdisplay, defaultMode.vdisplay,
        static_cast<double>(ModeRefreshMilliHz(defaultMode)) / 1000.0);

    if (lastRequest.IsValid()) {
        const auto match = SelectDisplayMode(modeCandidates, lastRequest, SnapshotModePolicy(defaultMode));
        out += std::format("Stream:  {}x{}@{:.3f}Hz ({}, switching {})\nDecision: {}\n", lastRequest.width,
                           lastRequest.height, static_cast<double>(lastRequest.rateMilliHz) / 1000.0,
                           PlaybackSourceName(lastRequestSource),
                           ModeSwitchEnabledFor(lastRequestSource) ? "enabled" : "disabled", match.reason);
    } else if (lastRequest.width > 0) {
        // Spelled out rather than reported as "none", which would read as nothing playing at all:
        // a stream IS playing, the matcher is just inert because guessing its rate could drive the
        // CRTC somewhere bogus.
        out += std::format("Stream:  {}x{}, no declared frame rate -- not matchable\n", lastRequest.width,
                           lastRequest.height);
    } else {
        out += "Stream:  none\n";
    }

    out += std::format("Modes ({} usable of {}):\n", modeCandidates.size(), connectorModes.size());
    for (const auto &candidate : modeCandidates) {
        out += std::format("  {}x{}@{:.3f}Hz{}{}\n", candidate.width, candidate.height,
                           static_cast<double>(candidate.refreshMilliHz) / 1000.0,
                           candidate.preferred ? " (preferred)" : "",
                           AnamorphicNote({connectorModes.data(), connectorModes.size()}, candidate));
    }
    return out;
}

// ============================================================================
// === MEDIAPLAYER FEED SURFACE ===
// ============================================================================

[[nodiscard]] auto cVaapiDevice::OpenForMediaPlayer(const VideoStreamInfo &video, const AudioStreamInfo &audio)
    -> bool {
    if (!HardwareReady()) [[unlikely]] {
        esyslog("vaapivideo/device: OpenForMediaPlayer rejected -- hardware not attached");
        return false;
    }
    if (!decoder || !audioProcessor) [[unlikely]] {
        esyslog("vaapivideo/device: OpenForMediaPlayer rejected -- subsystems missing");
        return false;
    }

    // Mediaplayer paths are always replay -- no live-TV jitter buffer, no Transferring() race. Drop
    // any leftover no-video monitors from a prior live channel so they cannot fire over playback.
    ResetNoVideoMonitors();
    liveMode.store(false, std::memory_order_relaxed);
    decoder->SetLiveMode(false);
    ResetZoom(); // Each opened file/URL starts unzoomed; the codec-open below rebuilds the graph.

    // Skip the 2-of-2 codec-detection dance: the demuxer reported authoritative codec IDs.
    videoCodecCandidate.store(AV_CODEC_ID_NONE, std::memory_order_relaxed);
    videoCodecCandidateCount.store(0, std::memory_order_relaxed);
    audioCodecCandidate.store(AV_CODEC_ID_NONE, std::memory_order_relaxed);
    audioCodecCandidateCount.store(0, std::memory_order_relaxed);

    if (!decoder->OpenCodecWithInfo(video)) [[unlikely]] {
        esyslog("vaapivideo/device: OpenForMediaPlayer video codec %s open failed", avcodec_get_name(video.codecId));
        return false;
    }
    videoCodecId.store(video.codecId, std::memory_order_relaxed);

    // Audio failure is non-fatal: a codec FFmpeg cannot open must not block an otherwise-fine HDR
    // video. Degrade to video-only. (A missing container sample rate -- raw-ADTS AAC in TS -- no
    // longer lands here: PopulateAudioInfo defaults it to the nominal 48 kHz the live path uses.)
    bool audioOpened = false;
    if (audio.codecId != AV_CODEC_ID_NONE) {
        if (audioProcessor->OpenCodecWithInfo(audio)) {
            audioCodecId.store(audio.codecId, std::memory_order_relaxed);
            audioOpened = true;
        } else {
            esyslog("vaapivideo/device: OpenForMediaPlayer audio codec %s open failed -- playing video-only",
                    avcodec_get_name(audio.codecId));
        }
    }
    if (!audioOpened) {
        // No audio: drop codec/clock state so the decoder freeruns and SubmitAudioPacket swallows
        // any AUs the demuxer still delivers (no demux stall).
        audioCodecId.store(AV_CODEC_ID_NONE, std::memory_order_relaxed);
        audioProcessor->Clear();
    }
    decoder->NotifyAudioChange();

    // Menu-driven track switches now route through cVaapiPlayer::SetAudioTrack, so the live-TV
    // HandleAudioTrackChange reset must stand down for this session.
    mediaPlayerAudioActive.store(true, std::memory_order_release);

    isyslog("vaapivideo/device: mediaplayer open -- video=%s audio=%s", avcodec_get_name(video.codecId),
            audioOpened ? avcodec_get_name(audio.codecId) : "none");
    return true;
}

[[nodiscard]] auto cVaapiDevice::ReopenMediaPlayerAudio(const AudioStreamInfo &audio) -> bool {
    if (!HardwareReady()) [[unlikely]] {
        esyslog("vaapivideo/device: ReopenMediaPlayerAudio rejected -- hardware not attached");
        return false;
    }
    if (!audioProcessor || !decoder) [[unlikely]] {
        esyslog("vaapivideo/device: ReopenMediaPlayerAudio rejected -- subsystems missing");
        return false;
    }

    if (audio.codecId == AV_CODEC_ID_NONE) {
        // Switching to "no audio": drop the codec/clock state so the decoder freeruns.
        audioCodecId.store(AV_CODEC_ID_NONE, std::memory_order_relaxed);
        audioProcessor->Clear();
        decoder->NotifyAudioChange();
        return true;
    }

    // SetStreamParams drains + reconfigures ALSA (PCM<->IEC61937, rate/channel) + reopens the decoder
    // under the audio mutex; the present thread reads audio lock-free, so this cannot stall it. The
    // caller re-anchors the clock via FlushForSeek afterwards.
    if (!audioProcessor->OpenCodecWithInfo(audio)) [[unlikely]] {
        esyslog("vaapivideo/device: ReopenMediaPlayerAudio %s open failed", avcodec_get_name(audio.codecId));
        return false;
    }
    audioCodecId.store(audio.codecId, std::memory_order_relaxed);
    decoder->NotifyAudioChange(); // arm freerun: the new audio clock is NOPTS until it primes
    isyslog("vaapivideo/device: mediaplayer audio reopen -- %s", avcodec_get_name(audio.codecId));
    return true;
}

[[nodiscard]] auto cVaapiDevice::SubmitVideoPacket(const AVPacket *packet) -> bool {
    if (!HardwareReady() || !decoder || decoder->IsQueueFull()) [[unlikely]] {
        return false;
    }
    decoder->EnqueuePacket(packet);
    return true;
}

[[nodiscard]] auto cVaapiDevice::SubmitAudioPacket(const AVPacket *packet) -> bool {
    if (!HardwareReady() || !audioProcessor) [[unlikely]] {
        return false;
    }
    // Video-only degrade (no audio codec): swallow the AU so the shared demux cursor doesn't stall
    // retrying a packet that can never be queued.
    if (audioCodecId.load(std::memory_order_relaxed) == AV_CODEC_ID_NONE) {
        return true;
    }
    if (!audioProcessor->IsInitialized()) [[unlikely]] {
        return false;
    }
    // Queue HIGHWATER is a pacing signal, not a hard failure. Returning false keeps the
    // mediaplayer's packetPending=true so this already-demuxed audio packet is retried after
    // the queue drains a slot; the shared demux cursor halts without dropping audio or
    // falsely advancing latestAudioPts90k via the post-submit update in cVaapiPlayer::Action.
    //
    // Use the DEEPER mediaplayer high-water, not the shallow PES one. This gate is the TRUE bound
    // on the single-cursor demux read-ahead: when it trips, the cursor stalls on the held audio
    // packet and stops reading VIDEO too (demux-order pump). At the PES value (~320 ms) the video
    // decoded reserve is therefore capped at ~320 ms -- fine for progressive (25 fps, ~16 frames is
    // plenty of slack) but starves interlaced content (deinterlaced to 50 fps: ~320 ms is ~16 frames,
    // and a post-seek catch-up drains it to zero, then the decoder runs input-starved and can never
    // rebuild the cushion -> sustained "decoder unable to keep up"). The PES path is immune: it feeds
    // video via a separate PlayVideo() call, so video fills the reserve to the cap independent of this
    // audio gate. Matching the audio read-ahead to AUDIO_QUEUE_HIGHWATER_MEDIAPLAYER (~1 s) lets the
    // shared cursor pull video ~1 s ahead too, so the reserve fills toward DECODER_RESERVE_HARD_CAP
    // exactly like the live/dvbplayer paths. (This is mediaplayer-only: PlayAudio() keeps the shallow
    // real-time gate.)
    if (audioProcessor->GetQueueSize() >= AUDIO_QUEUE_HIGHWATER_MEDIAPLAYER) {
        return false;
    }
    return audioProcessor->EnqueuePacket(packet);
}

auto cVaapiDevice::ClearForMediaPlayer() -> void {
    // Used at entry open/close (where codec params may change). Drops in-flight packets,
    // tears down the filter chain, and lets the next frame rebuild it with new params.
    // Codec contexts stay open and play-mode is unchanged.
    // Leaving media-player mode re-arms the live-TV HandleAudioTrackChange path.
    mediaPlayerAudioActive.store(false, std::memory_order_release);
    if (display) [[likely]] {
        display->BeginStreamSwitch();
    }
    if (decoder) [[likely]] {
        decoder->Clear();
    }
    if (display) [[likely]] {
        display->EndStreamSwitch();
    }
    if (audioProcessor) [[likely]] {
        audioProcessor->Clear();
    }
}

auto cVaapiDevice::RequestMediaPlayerEosDrain() -> void {
    if (decoder) [[likely]] {
        decoder->RequestCodecDrain();
    }
}

[[nodiscard]] auto cVaapiDevice::MediaPlayerDecodeQueueDepth() const noexcept -> size_t {
    return decoder ? decoder->GetQueueSize() : 0;
}

[[nodiscard]] auto cVaapiDevice::MediaPlayerBufferedDepth() const noexcept -> size_t {
    if (!decoder) {
        return 0;
    }
    // decode queue + decoded reserve + audio pending work. The pending-drain flag bridges the gap
    // between RequestMediaPlayerEosDrain() and the reorder tail landing in the reserve, so the
    // drain wait can't read a premature zero.
    size_t depth = decoder->GetQueueSize() + decoder->GetDecodedReserveSize();
    if (decoder->IsCodecDrainPending()) {
        ++depth;
    }
    if (audioProcessor) {
        depth += audioProcessor->GetPendingWorkSize();
    }
    return depth;
}

auto cVaapiDevice::FlushForSeek() -> void {
    // Light flush for the mediaplayer's seek path: same flush semantics as
    // ClearForMediaPlayer() (queues drained, codec buffers reset, ALSA drained, audio
    // clock re-anchors on next frame) BUT the filter graph and swresample state are
    // retained. Stream parameters do not change across a seek inside one file, so the
    // ~100 ms rebuild plus its "VAAPI filter initialized" / "initialized swresample"
    // log spam is pure overhead.
    if (display) [[likely]] {
        display->BeginStreamSwitch();
    }
    if (decoder) [[likely]] {
        decoder->FlushForSeek();
    }
    if (display) [[likely]] {
        display->EndStreamSwitch();
    }
    if (audioProcessor) [[likely]] {
        audioProcessor->Clear();
    }
}

[[nodiscard]] auto cVaapiDevice::IsMediaPlayerBackpressured() const noexcept -> bool {
    const bool videoFull = decoder && decoder->IsQueueFull();
    const size_t audioDepth = audioProcessor ? audioProcessor->GetQueueSize() : 0;
    // Gates BOTH the audioHighwater pacing and the audioCanReanchor escape below. Without it,
    // a video-only stream (no audio codec opened) reports audioDepth=0 forever -> audioCanReanchor
    // would always be true and permanently disable the jitterFull pre-anchor backpressure, letting
    // the decoder freerun-overrun the jitterBuf hard cap during startup.
    const bool audioOpen = audioCodecId.load(std::memory_order_relaxed) != AV_CODEC_ID_NONE;
    // Audio queue HIGHWATER (not the AUDIO_QUEUE_CAPACITY hard cap). The PTS lookahead alone is a
    // loose brake -- on TS where video PTS leads audio at the same demux cursor, the demuxer can
    // pump enough audio to keep the latest_audio_PTS within 1 s of the clock while video stacks up
    // and saturates the jitterBuf. Capping the audio packet queue at a small low-water depth (~10,
    // mirroring VDR's replay HIGHWATER) paces the demuxer to audio consumption rate and shrinks
    // both audio packet depth and the co-buffered video lead, keeping jitterBuf well below the cap.
    // Mediaplayer-only deeper highwater (single-cursor demux: audio depth bounds the video reserve).
    // See AUDIO_QUEUE_HIGHWATER_MEDIAPLAYER -- the shallow shared gate starves heavy interlaced 4K.
    const bool audioHighwater = audioOpen && audioDepth >= AUDIO_QUEUE_HIGHWATER_MEDIAPLAYER;
    // jitterBuf depth: guards ONLY the pre-anchor window (post-seek / startup) where the audio
    // clock is still NOPTS, so the demuxer's lookahead throttle can't gate delivery and HW decode
    // would overrun the jitterBuf hard cap (dropping ~30 frames per rapid-seek burst).
    //
    // Once the audio clock is live, this gate MUST be disabled: the demuxer is a single-cursor,
    // demux-order pump, so any throttle stops BOTH streams. Asserting backpressure on video
    // jitterBuf depth then starves the audio packet queue -> WritePcmToAlsa stops -> ALSA
    // underruns -> the master clock stalls -> the video drain gate (dueIn vs clock) stalls ->
    // jitterBuf stays full -> backpressure stays asserted. That positive-feedback starvation is
    // the "stutters after a long seek and never recovers" bug. With a live clock the lookahead
    // throttle (audio-referenced, self-releasing at the audio 1x consumption rate) is the correct
    // and sufficient gate; DECODER_RESERVE_HARD_CAP remains the ultimate overflow backstop.
    //
    // Audio-reanchor escape: pause -> Play() drains the audio queue but preserves jitterBuf
    // (decoder hold). On resume jitterFull would block the demuxer just when audio packets MUST
    // flow to re-anchor the clock -- deadlock, since the gate stays asserted until the clock
    // anchors and the clock can't anchor without audio. Yielding jitterFull while audio has room
    // below AUDIO_QUEUE_HIGHWATER_MEDIAPLAYER lets audio packets through; ALSA writes; clock anchors;
    // the gate becomes moot. The audioHighwater gate above caps the co-pumped burst at ~32 AC-3 packets.
    // Video-only streams (audioOpen=false) cannot anchor an audio clock, so the escape stays
    // disarmed and jitterFull bounds the decoder's pre-anchor depth.
    const bool clockAnchored = audioProcessor && audioProcessor->GetClock() != AV_NOPTS_VALUE;
    const bool audioCanReanchor = audioOpen && audioDepth < AUDIO_QUEUE_HIGHWATER_MEDIAPLAYER;
    const bool jitterFull = !clockAnchored && !audioCanReanchor && decoder &&
                            decoder->GetDecodedReserveSize() >= MEDIAPLAYER_JITTERBUF_BACKPRESSURE_FRAMES;
    return videoFull || audioHighwater || jitterFull;
}

[[nodiscard]] auto cVaapiDevice::GetAudioClock() const noexcept -> int64_t {
    if (!audioProcessor) {
        return AV_NOPTS_VALUE;
    }
    return audioProcessor->GetClock();
}

// ============================================================================
// === INTERNAL METHODS ===
// ============================================================================

[[nodiscard]] auto cVaapiDevice::AttachHardware() -> bool {
    // State: 0 (detached) -> 1 (in-progress) -> 2 (ready); the CAS rejects double-attach. NOT
    // main-thread-only: Initialize() runs this on the main thread, SVDRP ATTA on the SVDRP handler
    // thread (a cThread), a primary promotion via MakePrimaryDevice(). The subsystem pointers and
    // osdWidth/osdHeight written below are published by the terminal initState.store(2, release);
    // non-player main-thread readers (GetOsdSize/HasDecoder/ScaleVideo/Mute/...) must take the
    // matching acquire via HardwareReady() first. The player path (SetPlayMode/Clear/PlayVideo/...)
    // is reached only after ATTA returns through cControl, which supplies its own happens-before.
    // Every failure rollback below also stores 0 with release, so all initState transitions publish.
    int expected = 0;
    if (!initState.compare_exchange_strong(expected, 1)) [[unlikely]] {
        esyslog("vaapivideo/device: AttachHardware rejected -- already attached (state=%d)", expected);
        return false;
    }

    if (!OpenHardware()) [[unlikely]] {
        esyslog("vaapivideo/device: hardware initialization failed");
        initState.store(0, std::memory_order_release);
        return false;
    }

    // Construction order: audioProcessor -> display -> decoder. The decoder holds raw
    // pointers to both and queries them on every frame; they must outlive its thread.
    // Stop() reverses this order.
    audioProcessor = std::make_unique<cAudioProcessor>();
    if (!audioProcessor->Initialize(audioDevice)) [[unlikely]] {
        esyslog("vaapivideo/device: audio initialization failed");
        audioProcessor.reset();
        ReleaseHardware();
        initState.store(0, std::memory_order_release);
        return false;
    }

    display = std::make_unique<cVaapiDisplay>();
    if (!display->Initialize(drmFd, vaapi.hwDeviceRef, crtcId, connectorId, activeMode)) [[unlikely]] {
        esyslog("vaapivideo/device: display initialization failed");
        display.reset();
        audioProcessor->Shutdown();
        audioProcessor.reset();
        ReleaseHardware();
        initState.store(0, std::memory_order_release);
        return false;
    }

    // Decoder starts codec-less; PlayVideo() opens the codec on the first PES.
    decoder = std::make_unique<cVaapiDecoder>(display.get(), &vaapi);
    // Drive the encrypted-channel watchdog off the decode loop's idle tick: a fully scrambled channel
    // delivers no PES to PlayAudio/PlayVideo, but the decode thread still wakes ~every 100 ms with an
    // empty queue. Set before Initialize() starts that thread. CheckEncryptionTimeout no-ops unless armed.
    decoder->SetLoopTickCallback([this]() -> void {
        CheckEncryptionTimeout();
        // Same tick drives the display-mode stability gate: the reactive publish below fires once
        // per filter-graph build, so a candidate needs a wall-clock nudge to ever mature.
        PollPendingDisplayMode();
    });
    // Reactive display-mode matching for live TV and recordings: neither path knows the frame rate
    // before FFmpeg has parsed the VUI, so the chain reports it after each build. Also set before
    // Initialize(); a no-op unless the operator enabled a scope switch.
    decoder->SetStreamFormatCallback([this](uint32_t width, uint32_t height, uint32_t rateMilliHz) -> void {
        NotifyStreamFormat({.height = height, .rateMilliHz = rateMilliHz, .width = width});
    });
    if (!decoder->Initialize()) [[unlikely]] {
        esyslog("vaapivideo/device: decoder initialization failed");
        decoder.reset();
        display->Shutdown();
        display.reset();
        audioProcessor->Shutdown();
        audioProcessor.reset();
        ReleaseHardware();
        initState.store(0, std::memory_order_release);
        return false;
    }

    decoder->SetAudioProcessor(audioProcessor.get()); // A/V sync clock source

    // Pre-populate OSD cache: MakePrimaryDevice() (called next) may query GetOsdSize()
    // before any skin repaint triggers the lazy-init path.
    osdWidth = static_cast<int>(display->GetOutputWidth());
    osdHeight = static_cast<int>(display->GetOutputHeight());
    dsyslog("vaapivideo/device: OSD size %dx%d (pre-cached)", osdWidth, osdHeight);

    initState.store(2, std::memory_order_release);
    ResetZoom(); // Fresh hardware (plugin start or SVDRP ATTA) always begins at Off.
    isyslog("vaapivideo/device: attached -- DRM=%s audio=%s", drmPath.c_str(), audioDevice.c_str());

    // Startup splash: cover the console immediately with a black frame and a centered title.
    // Safe here -- display is ready and the decoder is codec-less/idle, and SubmitBlackFrame
    // serializes its VAAPI calls under vaDriverMutex.
    dsyslog("vaapivideo/device: showing startup splash");
    const bool splashShown = SubmitBlackFrame(tr("VDR with vaapivideo is getting ready..."));
    dsyslog("vaapivideo/device: startup splash %s", splashShown ? "submitted" : "FAILED");

    // Foreground VDR's VT so KBD receives keys after startup / SVDRP ATTA. Non-fatal.
    (void)ActivateOwnVt();

    return true;
}

auto cVaapiDevice::HandleAudioTrackChange(const char *reason, bool enteringDolby) -> void {
    // Single funnel for audio-track notifications: resolve track, dedup against last
    // (type, PID), and on a real change reset audioCodecId + drain audio so PlayAudio()
    // re-detects on the next PES. All entry points run on VDR's main thread under
    // mutexCurrentAudioTrack -- no additional locking needed:
    //   SetAudioTrackDevice   - reliable post-assignment read (no cPlayer attached).
    //   SetDigitalAudioDevice - On=true fires BEFORE assignment; uses dolby slot walk.
    //   Clear() via Empty/Goto - replay-path safety net.

    // In media-player mode cVaapiPlayer::SetAudioTrack owns audio; this PES-oriented reset would only
    // fight it. Live TV never sets the flag.
    if (mediaPlayerAudioActive.load(std::memory_order_acquire)) {
        dsyslog("vaapivideo/device: %s ignored -- media-player audio is player-driven", reason);
        return;
    }

    eTrackType type = ttNone;
    const tTrackId *track = nullptr;

    if (enteringDolby) {
        // GetCurrentAudioTrack() still returns the OLD track (pre-assignment race).
        // Walk dolby slots and pick the unique populated one. If multiple are present,
        // fall back to ttNone -- the reset still fires and PlayAudio() re-detects.
        // Enum casts are valid per VDR's IS_DOLBY_TRACK range (vdr/device.h).
        for (int offset = 0; offset <= ttDolbyLast - ttDolbyFirst; ++offset) {
            // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange) -- VDR API range
            const auto candidateType = static_cast<eTrackType>(static_cast<int>(ttDolbyFirst) + offset);
            const tTrackId *candidate = GetTrack(candidateType);
            if (candidate == nullptr || candidate->id == 0) {
                continue;
            }
            if (track == nullptr) {
                type = candidateType;
                track = candidate;
            } else {
                // Ambiguous: multiple dolby tracks. Fall back to ttNone.
                type = ttNone;
                track = nullptr;
                break;
            }
        }
    } else {
        type = GetCurrentAudioTrack();
        track = GetTrack(type);
    }

    const auto pid = static_cast<uint16_t>(track ? track->id : 0);

    if (type != ttNone && type == lastHandledAudioTrack && pid == lastHandledAudioPid) {
        return; // PMT churn / duplicate hook: VDR fires this repeatedly per channel switch.
    }
    lastHandledAudioTrack = type;
    lastHandledAudioPid = pid;

    const char *kind = "unknown";
    if (IS_AUDIO_TRACK(type)) {
        kind = "audio";
    } else if (IS_DOLBY_TRACK(type) || enteringDolby) {
        kind = "dolby"; // enteringDolby covers the ambiguous-dolby fallback
    }

    if (track != nullptr) {
        isyslog("vaapivideo/device: %s -> %s track %d (lang=%s, desc=%s, PID=%u)", reason, kind, static_cast<int>(type),
                (track->language[0] != '\0') ? track->language : "?",
                (track->description[0] != '\0') ? track->description : "?", pid);
    } else {
        // Ambiguous dolby: next-packet log identifies the actual codec.
        isyslog("vaapivideo/device: %s -> %s track switch (codec re-detect on next packet)", reason, kind);
    }

    ResetAudioCodecState();
    ResetReplayAudioEofBaseline();
    if (audioProcessor) [[likely]] {
        audioProcessor->Clear();
    }
    if (decoder) [[likely]] {
        decoder->NotifyAudioChange();
    }
}

[[nodiscard]] auto cVaapiDevice::OpenHardware() -> bool {
    if (drmPath.empty()) [[unlikely]] {
        esyslog("vaapivideo/device: no DRM device specified");
        return false;
    }

    // R+W required: resource query ioctls need R, KMS atomic modesetting needs W.
    // Missing group membership ('video' or 'render') is the common failure mode.
    if (access(drmPath.c_str(), R_OK | W_OK) != 0) [[unlikely]] {
        esyslog("vaapivideo/device: DRM device '%s' not accessible -- %s", drmPath.c_str(), std::strerror(errno));
        esyslog("vaapivideo/device: ensure user is in 'video' or 'render' group");
        return false;
    }

    drmFd = open(drmPath.c_str(), O_RDWR | O_CLOEXEC);
    if (drmFd < 0) [[unlikely]] {
        esyslog("vaapivideo/device: failed to open '%s' -- %s", drmPath.c_str(), std::strerror(errno));
        return false;
    }
    dsyslog("vaapivideo/device: opened DRM fd=%d", drmFd);

    if (!SelectDrmConnector()) [[unlikely]] {
        esyslog("vaapivideo/device: no connected display found on %s", drmPath.c_str());
        close(drmFd);
        drmFd = -1;
        return false;
    }

    // VAAPI requires the render node (/dev/dri/renderD*); drmGetDevice2 resolves the
    // matching render path for the card node already open.
    ::drmDevicePtr rawDevInfo = nullptr;
    if (drmGetDevice2(drmFd, 0, &rawDevInfo) != 0 || !rawDevInfo) [[unlikely]] {
        esyslog("vaapivideo/device: drmGetDevice2 failed");
        close(drmFd);
        drmFd = -1;
        return false;
    }

    std::unique_ptr<::drmDevice, FreeDrmDevice> devInfo{rawDevInfo};

    if (!(devInfo->available_nodes & (1 << DRM_NODE_RENDER)) || !devInfo->nodes[DRM_NODE_RENDER]) [[unlikely]] {
        esyslog("vaapivideo/device: no render node on %s", drmPath.c_str());
        close(drmFd);
        drmFd = -1;
        return false;
    }

    const std::string renderNode = devInfo->nodes[DRM_NODE_RENDER];
    dsyslog("vaapivideo/device: render node %s (primary %s)", renderNode.c_str(), drmPath.c_str());

    AVBufferRef *hwDevice = nullptr;
    const int ret = av_hwdevice_ctx_create(&hwDevice, AV_HWDEVICE_TYPE_VAAPI, renderNode.c_str(), nullptr, 0);
    if (ret < 0) [[unlikely]] {
        esyslog("vaapivideo/device: av_hwdevice_ctx_create failed -- %s", AvErr(ret).data());
        esyslog("vaapivideo/device: test with: vainfo --display drm --device %s", renderNode.c_str());
        close(drmFd);
        drmFd = -1;
        return false;
    }

    vaapi.hwDeviceRef = hwDevice;
    vaapi.drmFd = drmFd;

    if (!ProbeVppCapabilities(renderNode)) [[unlikely]] {
        esyslog("vaapivideo/device: VPP unavailable -- GPU is not suitable for this plugin");
        av_buffer_unref(&vaapi.hwDeviceRef);
        close(drmFd);
        drmFd = -1;
        return false;
    }

    return true;
}

[[nodiscard]] auto cVaapiDevice::ProbeVppCapabilities(std::string_view renderNode) -> bool {
    // ProbeGpuCaps() hard-fails (std::nullopt) if the render node / VADisplay / VPP
    // entrypoint is unavailable -- abort attach. A probe that succeeds with all decode
    // flags false means VPP is usable but no HW decode profiles were advertised;
    // OpenCodec() will fall back to SW decode per codec.
    auto probed = ProbeGpuCaps(renderNode);
    if (!probed) [[unlikely]] {
        return false;
    }
    vaapi.caps = std::move(*probed);
    return true;
}

auto cVaapiDevice::ReleaseHardware() -> void {
    if (vaapi.hwDeviceRef) {
        av_buffer_unref(&vaapi.hwDeviceRef);
        vaapi.drmFd = -1; // borrowed copy of drmFd; real close below
    }

    if (drmFd >= 0) {
        close(drmFd);
        drmFd = -1;
    }
}

auto cVaapiDevice::ResetAudioCodecState() -> void {
    // Clears candidate fields too: a stale partial 2-of-2 count would otherwise let one
    // detection confirm against the previous cycle's codec and reopen the wrong decoder.
    audioCodecId.store(AV_CODEC_ID_NONE, std::memory_order_relaxed);
    audioCodecCandidate.store(AV_CODEC_ID_NONE, std::memory_order_relaxed);
    audioCodecCandidateCount.store(0, std::memory_order_relaxed);
    // Invalidate the PlayAudio detection window across threads without touching the vector:
    // it drops its accumulated bytes when it next sees this counter move (old-PID ES must not
    // corroborate against new-PID ES).
    audioDetectGen.fetch_add(1, std::memory_order_relaxed);
}

auto cVaapiDevice::ResetReplayAudioEofBaseline() noexcept -> void {
    lastReplayAudioPts.store(AV_NOPTS_VALUE, std::memory_order_relaxed);
}

// ============================================================================
// === DISPLAY MODE MATCHING ===
// ============================================================================

namespace {

/// Relative difference between two rates, in parts per million. Used instead of a percentage so
/// the exact tier (200 ppm) can stay tight enough to keep 59.94 and 60 apart.
[[nodiscard]] auto RateErrorPpm(uint32_t actualMilliHz, uint32_t targetMilliHz) noexcept -> uint32_t {
    if (targetMilliHz == 0) [[unlikely]] {
        return UINT32_MAX;
    }
    const uint64_t delta =
        actualMilliHz > targetMilliHz ? actualMilliHz - targetMilliHz : targetMilliHz - actualMilliHz;
    return static_cast<uint32_t>(std::min<uint64_t>(delta * 1000000ULL / targetMilliHz, UINT32_MAX));
}

/// Resolution stage: the (width, height) the output should run at.
///
/// Smallest candidate size that still covers the stream's coded frame, so the picture maps 1:1 and
/// the sink does its own upscaling; never below policy.minHeight, which doubles as the "pin the
/// resolution" control when set to the panel's native height. A stream larger than anything the
/// panel offers falls back to the largest size (VPP downscales, as it does today).
[[nodiscard]] auto SelectTargetResolution(std::span<const DisplayModeCandidate> candidates,
                                          const StreamModeRequest &request, const DisplayModePolicy &policy)
    -> std::pair<uint32_t, uint32_t> {
    const std::pair<uint32_t, uint32_t> fallback{policy.defaultWidth, policy.defaultHeight};
    if (!policy.matchResolution) {
        return fallback;
    }

    bool haveFit = false;
    std::pair<uint32_t, uint32_t> best{};
    std::pair<uint32_t, uint32_t> largest{};
    uint64_t bestArea = 0;
    uint64_t largestArea = 0;
    for (const auto &candidate : candidates) {
        if (candidate.height < policy.minHeight) {
            continue;
        }
        const uint64_t area = static_cast<uint64_t>(candidate.width) * candidate.height;
        if (area > largestArea) {
            largestArea = area;
            largest = {candidate.width, candidate.height};
        }
        if (candidate.width < request.width || candidate.height < request.height) {
            continue;
        }
        if (!haveFit || area < bestArea) {
            haveFit = true;
            bestArea = area;
            best = {candidate.width, candidate.height};
        }
    }

    if (haveFit) {
        return best;
    }
    if (largestArea > 0) {
        return largest; // stream exceeds the panel: run native, let the VPP downscale
    }
    return fallback; // minHeight excluded everything
}

/// Refresh stage, restricted to candidates already at the target resolution.
///
/// Two tiers over k = 1..DISPLAY_MODE_MAX_RATE_MULTIPLE. Tier A absorbs rounding only, so 59.94
/// and 60 stay distinct and a genuine 59.94 mode wins when the panel has both. Tier B is the
/// 0.5% net that lets 59.94 content settle on a 60-Hz-only panel. Within a tier the HIGHEST k at
/// or below the cap wins -- that is what turns 25p into 50 Hz and 29.97 into 59.94 -- while k==1
/// is always in play so a source faster than the cap is never forced down.
[[nodiscard]] auto SelectTargetRefresh(std::span<const DisplayModeCandidate> candidates, uint32_t targetWidth,
                                       uint32_t targetHeight, const StreamModeRequest &request,
                                       const DisplayModePolicy &policy) -> DisplayModeMatch {
    const auto atTargetSize = [&](const DisplayModeCandidate &candidate) noexcept -> bool {
        return candidate.width == targetWidth && candidate.height == targetHeight;
    };
    // Winners are tracked as pointers into the span and only converted to an index at the very end;
    // that keeps every access a dereference of something already known to be in range.
    const auto indexOf = [&](const DisplayModeCandidate *candidate) noexcept -> int {
        return static_cast<int>(candidate - candidates.data());
    };
    // One shape for every decision string: "<mode picked> (<why>)". The caller prefixes the stream it
    // was picked for, so the reason must never restate the source -- SVDRP MODE prints the stream on
    // its own line right above the decision, and the syslog line does the same inline.
    const auto describe = [](std::string_view why, const DisplayModeCandidate &candidate) -> std::string {
        return std::format("{}x{}@{:.3f}Hz ({})", candidate.width, candidate.height,
                           static_cast<double>(candidate.refreshMilliHz) / 1000.0, why);
    };

    // Fallback used both when matching is off and when no multiple fits: prefer the mode the
    // operator configured, otherwise the fastest one within the cap, otherwise PREFERRED/first.
    const auto pickFallback = [&](std::string_view why) -> DisplayModeMatch {
        const DisplayModeCandidate *exactDefault = nullptr;
        const DisplayModeCandidate *fastestCapped = nullptr;
        const DisplayModeCandidate *preferred = nullptr;
        const DisplayModeCandidate *anyMode = nullptr;
        for (const auto &candidate : candidates) {
            if (!atTargetSize(candidate)) {
                continue;
            }
            if (anyMode == nullptr) {
                anyMode = &candidate;
            }
            if (preferred == nullptr && candidate.preferred) {
                preferred = &candidate;
            }
            if (exactDefault == nullptr && RateErrorPpm(candidate.refreshMilliHz, policy.defaultRefreshMilliHz) <=
                                               DISPLAY_MODE_EXACT_TOLERANCE_PPM) {
                exactDefault = &candidate;
            }
            if (candidate.refreshMilliHz <= policy.maxRefreshMilliHz &&
                (fastestCapped == nullptr || candidate.refreshMilliHz > fastestCapped->refreshMilliHz)) {
                fastestCapped = &candidate;
            }
        }
        for (const auto *candidate : {exactDefault, fastestCapped, preferred, anyMode}) {
            if (candidate != nullptr) {
                return {.index = indexOf(candidate), .reason = describe(why, *candidate)};
            }
        }
        return {.index = -1, .reason = std::format("no mode at {}x{} ({})", targetWidth, targetHeight, why)};
    };

    if (!policy.matchRefresh) {
        return pickFallback("refresh matching off");
    }
    if (request.rateMilliHz == 0) [[unlikely]] {
        return pickFallback("source rate unknown");
    }

    for (const uint32_t tolerancePpm : {DISPLAY_MODE_EXACT_TOLERANCE_PPM, DISPLAY_MODE_LOOSE_TOLERANCE_PPM}) {
        const DisplayModeCandidate *best = nullptr;
        uint32_t bestK = 0;
        uint32_t bestErrorPpm = UINT32_MAX;
        for (uint32_t k = 1; k <= DISPLAY_MODE_MAX_RATE_MULTIPLE; ++k) {
            const uint64_t target = static_cast<uint64_t>(request.rateMilliHz) * k;
            if (target > UINT32_MAX) {
                break;
            }
            // k==1 reproduces the source rate exactly, so it stays eligible even above the cap --
            // capping it would leave a 120 fps source with no honest option at all.
            if (k > 1 && target > policy.maxRefreshMilliHz) {
                break;
            }
            for (const auto &candidate : candidates) {
                if (!atTargetSize(candidate)) {
                    continue;
                }
                const uint32_t errorPpm = RateErrorPpm(candidate.refreshMilliHz, static_cast<uint32_t>(target));
                if (errorPpm > tolerancePpm) {
                    continue;
                }
                // Higher k wins first (ascending loop order lets a later k overwrite an earlier
                // one), then the closest rate, then the faster mode on a dead heat.
                const bool better =
                    best == nullptr || k > bestK || (k == bestK && errorPpm < bestErrorPpm) ||
                    (k == bestK && errorPpm == bestErrorPpm && candidate.refreshMilliHz > best->refreshMilliHz);
                if (better) {
                    best = &candidate;
                    bestK = k;
                    bestErrorPpm = errorPpm;
                }
            }
        }
        if (best != nullptr) {
            return {.index = indexOf(best),
                    .reason = describe(std::format("{} x{}, {} ppm",
                                                   tolerancePpm == DISPLAY_MODE_EXACT_TOLERANCE_PPM ? "exact" : "near",
                                                   bestK, bestErrorPpm),
                                       *best)};
        }
    }

    return pickFallback(std::format("no multiple of {:.3f}Hz available", //
                                    static_cast<double>(request.rateMilliHz) / 1000.0));
}

/// Timing identity, stricter than SameOutputMode(): a CEA and a DMT 1080p60 look alike but are
/// different rasters. All programmed fields -- equal totals can still split into a different
/// porch/sync -- but not type/name/vrefresh, which are metadata or derived.
[[nodiscard]] auto SameRaster(const drmModeModeInfo &a, const drmModeModeInfo &b) noexcept -> bool {
    return a.clock == b.clock && a.hdisplay == b.hdisplay && a.hsync_start == b.hsync_start &&
           a.hsync_end == b.hsync_end && a.htotal == b.htotal && a.hskew == b.hskew && a.vdisplay == b.vdisplay &&
           a.vsync_start == b.vsync_start && a.vsync_end == b.vsync_end && a.vtotal == b.vtotal && a.vscan == b.vscan &&
           a.flags == b.flags;
}

} // namespace

[[nodiscard]] auto BuildModeCandidates(std::span<const drmModeModeInfo> modes, const drmModeModeInfo &defaultMode)
    -> std::vector<DisplayModeCandidate> {
    std::vector<DisplayModeCandidate> candidates;
    candidates.reserve(modes.size());

    // Aspect gate: keep only timings shaped like the mode the operator/EDID settled on, so a legacy
    // 4:3 entry can never be picked and stretch the picture. Compared on the PICTURE aspect, not
    // hdisplay/vdisplay -- a raw pixel-ratio test rejects every anamorphic SD mode on a 16:9 panel,
    // which silently reduced the 576p "Minimum resolution" setting to a duplicate of 720p.
    const double referenceAspect = (defaultMode.hdisplay > 0 && defaultMode.vdisplay > 0)
                                       ? ModePictureAspect(defaultMode)
                                       : DISPLAY_DEFAULT_ASPECT_RATIO;
    constexpr double kAspectTolerance = 0.02;

    uint16_t index = 0;
    for (const auto &mode : modes) {
        const uint16_t modeIndex = index++;
        if (mode.hdisplay == 0 || mode.vdisplay == 0) {
            continue;
        }
        // Interlaced output is unusable here: the VPP always emits progressive frames and the
        // scanout path has no field interleaving, so an interlaced mode would show half the lines.
        if ((mode.flags & DRM_MODE_FLAG_INTERLACE) != 0) {
            continue;
        }
        const uint32_t rateMilliHz = ModeRefreshMilliHz(mode);
        if (rateMilliHz == 0) {
            continue;
        }
        if (std::abs(ModePictureAspect(mode) - referenceAspect) > referenceAspect * kAspectTolerance) {
            continue;
        }
        candidates.push_back({.height = mode.vdisplay,
                              .refreshMilliHz = rateMilliHz,
                              .preferred = (mode.type & DRM_MODE_TYPE_PREFERRED) != 0,
                              .index = modeIndex,
                              .width = mode.hdisplay});
    }

    // Second pass: one candidate per distinct output. A connector routinely lists 1920x1080@60
    // twice (CEA and DMT rasters for the same picture), and keeping both lets the matcher install
    // the default mode's twin -- which SameOutputMode() calls "already there", so
    // RestoreDefaultModeLocked() never puts the sink back on its own timing. Keep the default
    // raster, else a PREFERRED one, else the first listed.
    const auto rankOf = [modes, &defaultMode](const DisplayModeCandidate &candidate) noexcept -> int {
        if (SameRaster(*std::next(modes.begin(), candidate.index), defaultMode)) {
            return 2;
        }
        return candidate.preferred ? 1 : 0;
    };
    std::vector<DisplayModeCandidate> unique;
    unique.reserve(candidates.size());
    for (const auto &candidate : candidates) {
        const auto twin = std::ranges::find_if(unique, [&candidate](const DisplayModeCandidate &kept) noexcept -> bool {
            return kept.width == candidate.width && kept.height == candidate.height &&
                   kept.refreshMilliHz == candidate.refreshMilliHz;
        });
        if (twin == unique.end()) {
            unique.push_back(candidate);
        } else if (rankOf(candidate) > rankOf(*twin)) {
            *twin = candidate;
        }
    }
    candidates = std::move(unique);

    // Third pass: drop the CEA pixel-repetition rasters (1440x576, 2880x576 -- 720x576 with every
    // pixel sent two or four times). Same picture, 2x/4x the pixel clock, so picking one would make
    // the VPP upscale horizontally for no added detail.
    //
    // Redundant means: another candidate shares its height, refresh rate and EXACT picture aspect
    // at an integer fraction of its width. The exact-aspect key is what keeps genuinely distinct
    // near-16:9 timings apart -- 1360x768 and 1366x768 match on height and rate but differ in
    // picture aspect and are not multiples, so both survive.
    const auto aspectOf = [modes](const DisplayModeCandidate &candidate) noexcept -> AspectRatio {
        return ModePictureAspectRatio(*std::next(modes.begin(), candidate.index));
    };
    // The default mode is exempt: it is the resolution the matcher falls back to whenever
    // resolution matching is off or finds no fit, so dropping it (--resolution 1440x576 names a
    // repeated raster) would leave the refresh stage without a candidate at the target size.
    const uint32_t defaultRateMilliHz = ModeRefreshMilliHz(defaultMode);
    const auto isDefaultMode = [&](const DisplayModeCandidate &candidate) noexcept -> bool {
        return candidate.width == defaultMode.hdisplay && candidate.height == defaultMode.vdisplay &&
               candidate.refreshMilliHz == defaultRateMilliHz;
    };
    std::vector<DisplayModeCandidate> distinct;
    distinct.reserve(candidates.size());
    // Writes a separate vector: an erase-in-place pass would let the predicate see moved-from ones.
    for (const auto &candidate : candidates) {
        if (isDefaultMode(candidate)) {
            distinct.push_back(candidate);
            continue;
        }
        const AspectRatio aspect = aspectOf(candidate);
        const bool repeats = std::ranges::any_of(candidates, [&](const DisplayModeCandidate &base) -> bool {
            if (base.height != candidate.height || base.refreshMilliHz != candidate.refreshMilliHz ||
                base.width >= candidate.width || (candidate.width % base.width) != 0) {
                return false;
            }
            const AspectRatio baseAspect = aspectOf(base);
            return baseAspect.num == aspect.num && baseAspect.den == aspect.den;
        });
        if (!repeats) {
            distinct.push_back(candidate);
        }
    }
    return distinct;
}

[[nodiscard]] auto SelectDisplayMode(std::span<const DisplayModeCandidate> candidates, const StreamModeRequest &request,
                                     const DisplayModePolicy &policy) -> DisplayModeMatch {
    if (candidates.empty()) [[unlikely]] {
        return {.index = -1, .reason = "no usable connector modes"};
    }
    const auto [targetWidth, targetHeight] = SelectTargetResolution(candidates, request, policy);
    return SelectTargetRefresh(candidates, targetWidth, targetHeight, request, policy);
}

// SelectDrmConnector() helper: validate one connector and, when it carries the wanted mode, latch the
// chosen mode + CRTC + connector into the device. Pulled out of the two connector loops so the same
// logic serves the fast cached pass and the slow full-probe pass. Mode priority: exact (w,h,rate) match
// (operator wins); else, only when allowModeFallback, the driver PREFERRED mode then the first listed.
// allowModeFallback=false (cached pass) accepts only an exact match, so a partial cached mode list can't
// lock in preferred/first before the full EDID probe finds the operator-selected mode.
[[nodiscard]] auto cVaapiDevice::TryAcceptConnector(drmModeConnector *connector, bool allowModeFallback,
                                                    drmModeRes *resources) -> bool {
    const auto targetWidth = vaapiConfig.display.GetWidth();
    const auto targetHeight = vaapiConfig.display.GetHeight();
    const auto targetRate = vaapiConfig.display.GetRefreshRate();

    if (!connector || connector->connection != DRM_MODE_CONNECTED || connector->count_modes == 0) {
        return false;
    }

    // Build the kernel-style name (e.g. "HDMI-A-1", "DP-2"). Always computed -- used
    // both for the user-supplied filter (when set) and to populate connectorName below
    // for SVDRP DeviceName() / sticky re-selection across DETA/ATTA cycles.
    const char *typeName = drmModeGetConnectorTypeName(connector->connector_type);
    const auto name = std::format("{}-{}", typeName ? typeName : "Unknown", connector->connector_type_id);
    if (!connectorName.empty() && name != connectorName) {
        dsyslog("vaapivideo/device: skipping connector %s (want %s)", name.c_str(), connectorName.c_str());
        return false;
    }

    // Match --resolution within its integer-Hz bucket, nearest true rate first. A connector
    // routinely exposes 59.94 AND 60.000 at the same size and drm_mode_vrefresh() rounds BOTH to
    // 60, so comparing that field latched whichever came first in EDID order -- "@60" could
    // silently settle on 59.94. The loose tolerance IS the bucket (0.5%, wide enough for 59.94 and
    // 23.976 to reach their integer neighbours); ties go to a PREFERRED timing.
    const drmModeModeInfo *bestMode = nullptr;
    uint32_t bestErrorPpm = 0;
    for (int modeIdx = 0; modeIdx < connector->count_modes; ++modeIdx) {
        const auto &mode = connector->modes[modeIdx];
        if (mode.hdisplay != targetWidth || mode.vdisplay != targetHeight) {
            continue;
        }
        const uint32_t errorPpm = RateErrorPpm(ModeRefreshMilliHz(mode), targetRate * 1000U);
        if (errorPpm > DISPLAY_MODE_LOOSE_TOLERANCE_PPM) {
            continue;
        }
        const bool preferred = (mode.type & DRM_MODE_TYPE_PREFERRED) != 0;
        const bool bestPreferred = bestMode != nullptr && (bestMode->type & DRM_MODE_TYPE_PREFERRED) != 0;
        if (bestMode == nullptr || errorPpm < bestErrorPpm ||
            (errorPpm == bestErrorPpm && preferred && !bestPreferred)) {
            bestMode = &mode;
            bestErrorPpm = errorPpm;
        }
    }
    bool modeFound = bestMode != nullptr;
    if (modeFound) {
        activeMode = *bestMode;
    }
    if (!modeFound) {
        if (!allowModeFallback) {
            return false;
        }
        for (int modeIdx = 0; modeIdx < connector->count_modes; ++modeIdx) {
            if (connector->modes[modeIdx].type & DRM_MODE_TYPE_PREFERRED) {
                activeMode = connector->modes[modeIdx];
                modeFound = true;
                break;
            }
        }
        if (!modeFound) {
            activeMode = connector->modes[0];
        }
    }

    uint32_t localCrtc = 0;
    std::unique_ptr<drmModeObjectProperties, FreeDrmObjectProperties> props{
        drmModeObjectGetProperties(drmFd, connector->connector_id, DRM_MODE_OBJECT_CONNECTOR)};
    if (props) {
        for (uint32_t propIdx = 0; propIdx < props->count_props; ++propIdx) {
            std::unique_ptr<drmModePropertyRes, FreeDrmProperty> prop{drmModeGetProperty(drmFd, props->props[propIdx])};
            // Use CRTC_ID atomic property, not legacy encoder_id walk: the legacy path
            // can return a stale CRTC on atomic drivers.
            if (prop && std::strcmp(prop->name, "CRTC_ID") == 0) {
                localCrtc = static_cast<uint32_t>(props->prop_values[propIdx]);
                break;
            }
        }
    }
    if (localCrtc == 0 && resources->count_crtcs > 0) {
        localCrtc = resources->crtcs[0];
    }
    if (localCrtc == 0) {
        return false;
    }

    crtcId = localCrtc;
    connectorId = connector->connector_id;
    // Latch the chosen connector once. Sticky on subsequent re-attaches (operator
    // expectation: a DETA/ATTA cycle keeps the same output). Also surfaces in
    // SVDRP PRIM/LSTD via DeviceName().
    if (connectorName.empty()) {
        connectorName = name;
    }

    // Retain the mode list for runtime switching. This is the only place it is ever read: the
    // connector object is freed by the caller, and re-querying it later means a fresh DDC probe
    // (100-500 ms per port, seconds on a CEC-standby sink). defaultMode is the fallback the
    // matcher returns to and the mode ResetDisplayModeToDefault() restores.
    // Under displayModeMutex: SVDRP MODE (DisplayModeReport) iterates these from its own thread,
    // and an attach racing that iteration would reallocate the vectors under it.
    {
        const cMutexLock lock(&displayModeMutex);
        defaultMode = activeMode;
        connectorModes.assign(connector->modes, connector->modes + connector->count_modes);
        modeCandidates = BuildModeCandidates(connectorModes, defaultMode);
    }

    isyslog("vaapivideo/device: display %ux%u@%.3fHz (%s, connector %u, CRTC %u)", activeMode.hdisplay,
            activeMode.vdisplay, static_cast<double>(ModeRefreshMilliHz(activeMode)) / 1000.0, name.c_str(),
            connectorId, crtcId);
    // Inventory dump: the primary field diagnostic for mode matching, and the only record of what
    // the aspect/interlace filter threw away.
    dsyslog("vaapivideo/device: %zu connector modes, %zu usable for mode matching", connectorModes.size(),
            modeCandidates.size());
    for (const auto &candidate : modeCandidates) {
        dsyslog("vaapivideo/device:   mode %ux%u@%.3fHz%s%s", candidate.width, candidate.height,
                static_cast<double>(candidate.refreshMilliHz) / 1000.0, candidate.preferred ? " (preferred)" : "",
                AnamorphicNote({connectorModes.data(), connectorModes.size()}, candidate).c_str());
    }
    return true;
}

[[nodiscard]] auto cVaapiDevice::SelectDrmConnector() -> bool {

    if (drmSetClientCap(drmFd, DRM_CLIENT_CAP_ATOMIC, 1) != 0) [[unlikely]] {
        esyslog("vaapivideo/device: failed to enable DRM atomic capability");
        return false;
    }
    // Must precede the drmModeGetConnector() calls below: without it the kernel masks
    // DRM_MODE_FLAG_PIC_AR_* out of every mode, leaving the anamorphic CEA SD timings
    // indistinguishable from square-pixel 1.25:1 rasters -- the aspect gate would drop them and the
    // VPP would mis-fit them. ATOMIC already implies this cap in every kernel to date; asking
    // explicitly keeps the dependency visible. Best-effort: a refusal just falls back to pixel
    // ratios.
    (void)drmSetClientCap(drmFd, DRM_CLIENT_CAP_ASPECT_RATIO, 1);

    std::unique_ptr<drmModeRes, FreeDrmResources> resources{drmModeGetResources(drmFd)};
    if (!resources) [[unlikely]] {
        esyslog("vaapivideo/device: failed to get DRM resources");
        return false;
    }

    // Existence-check only: display re-queries planes during modeset. Fail fast here if
    // the kernel doesn't expose plane resources (atomic requires them).
    if (!std::unique_ptr<drmModePlaneRes, FreeDrmPlaneResources>{drmModeGetPlaneResources(drmFd)}) [[unlikely]] {
        esyslog("vaapivideo/device: failed to get DRM plane resources (atomic required)");
        return false;
    }

    // Pass 1: cached state (no DDC). drmModeGetConnector() forces a fresh probe -- 100-500 ms
    // per disconnected port, seconds on a CEC-standby sink. On boxes with 6-8 connectors that
    // stacks up to the reported "60 s startup". Boot-probe / hot-plug cache suffices normally.
    for (int i = 0; i < resources->count_connectors; ++i) {
        const std::unique_ptr<drmModeConnector, FreeDrmConnector> connector{
            drmModeGetConnectorCurrent(drmFd, resources->connectors[i])};
        if (TryAcceptConnector(connector.get(), false, resources.get())) {
            return true;
        }
    }

    // Pass 2: cold cache (DRM driver just loaded) or sink woke from standby after boot probe.
    // Full DDC probe -- only when pass 1 missed. When a
    // connector name is configured, pre-filter via the cached connector so we don't pay the
    // probe cost on every connector just to find out the name doesn't match.
    dsyslog("vaapivideo/device: no candidate in cached state -- forcing EDID probe");
    for (int i = 0; i < resources->count_connectors; ++i) {
        if (!connectorName.empty()) {
            const std::unique_ptr<drmModeConnector, FreeDrmConnector> cached{
                drmModeGetConnectorCurrent(drmFd, resources->connectors[i])};
            if (cached) {
                const char *typeName = drmModeGetConnectorTypeName(cached->connector_type);
                const auto name = std::format("{}-{}", typeName ? typeName : "Unknown", cached->connector_type_id);
                if (name != connectorName) {
                    continue;
                }
            }
        }
        const std::unique_ptr<drmModeConnector, FreeDrmConnector> connector{
            drmModeGetConnector(drmFd, resources->connectors[i])};
        if (TryAcceptConnector(connector.get(), true, resources.get())) {
            return true;
        }
    }

    if (!connectorName.empty()) {
        esyslog("vaapivideo/device: connector '%s' not found or not connected", connectorName.c_str());
    } else {
        esyslog("vaapivideo/device: no connected display found");
    }
    return false;
}

auto cVaapiDevice::Stop() -> void {
    // Reverse of AttachHardware(): decoder first -- its Action thread dereferences raw
    // display + audioProcessor pointers on every frame; destroying either while the thread
    // runs is use-after-free. No back-references flow the other way. Each Shutdown() is
    // idempotent.
    if (decoder) [[likely]] {
        decoder->Shutdown();
        decoder.reset();
    }
    if (display) [[likely]] {
        display->Shutdown();
        display.reset();
    }
    if (audioProcessor) [[likely]] {
        audioProcessor->Shutdown();
        audioProcessor.reset();
    }
}
