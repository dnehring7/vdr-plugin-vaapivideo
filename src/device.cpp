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
 * Subsystem lifetime is bound to a tri-state initState (0=none, 1=in-progress, 2=ready)
 * managed via CAS in Initialize() / Stop() / Detach(). The decoder Action thread holds raw
 * pointers to display + audioProcessor, so Stop() MUST tear them down in the order
 * decoder -> display -> audioProcessor (see Stop() comment).
 *
 * The Detach() path also has a non-obvious upstream-shutdown ordering hazard with
 * cTransferControl that the destructor doesn't share -- documented inline.
 */

#include "device.h"
#include "audio.h"
#include "common.h"
#include "config.h"
#include "decoder.h"
#include "osd.h"
#include "pes.h"

// C++ Standard Library
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

// POSIX
#include <fcntl.h>
#include <linux/vt.h>
#include <sys/ioctl.h>
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
#include <libavcodec/codec_id.h>
#include <libavutil/avutil.h>
#include <libavutil/buffer.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixfmt.h>
}
#pragma GCC diagnostic pop

// VAAPI
#include <va/va.h>
#include <va/va_drm.h>
#include <va/va_vpp.h>

// VDR
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <vdr/device.h>
#include <vdr/player.h>
#include <vdr/thread.h>
#include <vdr/tools.h>
#pragma GCC diagnostic pop

// ============================================================================
// === HELPER FUNCTIONS ===
// ============================================================================

/// VT bounce after dropping DRM master. fbcon does not auto-reclaim the display when another
/// DRM client (typically systemd-logind) still holds the device open, so we VT_ACTIVATE to a
/// different VT and back -- the kernel reprograms the CRTC on each switch and the second
/// switch hands the framebuffer back to the console.
static auto RestoreConsole() -> void {
    const int ttyFd = open("/dev/tty0", O_RDWR | O_CLOEXEC);
    if (ttyFd < 0) {
        dsyslog("vaapivideo/device: console restore skipped (no /dev/tty0)");
        return;
    }

    struct vt_stat vtState{};
    if (ioctl(ttyFd, VT_GETSTATE, &vtState) != 0) {
        close(ttyFd);
        return;
    }

    const auto currentVt = static_cast<int>(vtState.v_active);

    // VT_ACTIVATE to the current VT is a no-op in the kernel; pick any other VT to force a
    // real switch (and therefore a CRTC reprogram).
    const int tempVt = (currentVt == 1) ? 2 : 1;

    if (ioctl(ttyFd, VT_ACTIVATE, tempVt) == 0) {
        ioctl(ttyFd, VT_WAITACTIVE, tempVt);
    }
    if (ioctl(ttyFd, VT_ACTIVATE, currentVt) == 0) {
        ioctl(ttyFd, VT_WAITACTIVE, currentVt);
    }

    close(ttyFd);
    dsyslog("vaapivideo/device: console VT%d restored", currentVt);
}

/// True iff the driver advertises VLD bitstream decode + YUV420 surface output for this
/// profile. Both are required for our pipeline; HW-accel pickers in OpenCodec() use the
/// per-profile flags this populates.
[[nodiscard]] static auto HasVldDecode(VADisplay display, VAProfile profile) -> bool {
    const int maxEp = vaMaxNumEntrypoints(display);
    if (maxEp <= 0) [[unlikely]] {
        return false;
    }

    std::vector<VAEntrypoint> entrypoints(static_cast<size_t>(maxEp));
    int epCount = 0;
    if (vaQueryConfigEntrypoints(display, profile, entrypoints.data(), &epCount) != VA_STATUS_SUCCESS) {
        return false;
    }

    const auto end = entrypoints.begin() + epCount;
    if (std::find(entrypoints.begin(), end, VAEntrypointVLD) == end) {
        return false;
    }

    VAConfigAttrib attrib{};
    attrib.type = VAConfigAttribRTFormat;
    if (vaGetConfigAttributes(display, profile, VAEntrypointVLD, &attrib, 1) != VA_STATUS_SUCCESS) {
        return false;
    }
    return (attrib.value & VA_RT_FORMAT_YUV420) != 0;
}

// ============================================================================
// === DRM DEVICES CLASS ===
// ============================================================================

DrmDevices::~DrmDevices() noexcept {
    if (!deviceList.empty()) {
        drmFreeDevices(deviceList.data(), static_cast<int>(deviceList.size()));
    }
}

// ============================================================================
// === ITERATORS ===
// ============================================================================

[[nodiscard]] auto DrmDevices::begin() -> std::vector<drmDevicePtr>::iterator { return deviceList.begin(); }

[[nodiscard]] auto DrmDevices::end() -> std::vector<drmDevicePtr>::iterator { return deviceList.end(); }

// ============================================================================
// === QUERIES ===
// ============================================================================

[[nodiscard]] auto DrmDevices::Enumerate() -> bool {
    dsyslog("vaapivideo/device: enumerating DRM devices");

    // Re-enumeration: free the previous batch (drmFreeDevices walks libdrm's per-entry refs).
    if (!deviceList.empty()) {
        drmFreeDevices(deviceList.data(), static_cast<int>(deviceList.size()));
        deviceList.clear();
    }

    deviceList.resize(64); // Generous overprovision: real systems have 1-4 DRM nodes.
    const int result = drmGetDevices2(0, deviceList.data(), static_cast<int>(deviceList.size()));

    if (result > 0) {
        deviceList.resize(static_cast<size_t>(result));
        dsyslog("vaapivideo/device: found %d DRM device(s)", result);
    } else {
        if (result < 0) {
            esyslog("vaapivideo/device: drmGetDevices2 failed (%s)", strerror(-result));
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

auto cVaapiDevice::SubmitBlackFrame() -> void {
    // Used to clear the previous channel's residual picture on radio channels and on
    // hardware (re)init -- the DRM scanout buffer would otherwise hold whatever was last
    // displayed. Allocates its own one-shot hw_frames_ctx so it does not depend on the
    // decoder pipeline being up.
    const auto w = static_cast<int>(display->GetOutputWidth());
    const auto h = static_cast<int>(display->GetOutputHeight());

    std::unique_ptr<AVBufferRef, FreeAVBufferRef> framesRef{av_hwframe_ctx_alloc(vaapi.hwDeviceRef)};
    if (!framesRef) [[unlikely]] {
        esyslog("vaapivideo/device: black frame hw_frames_ctx alloc failed");
        return;
    }
    auto *ctx =
        reinterpret_cast<AVHWFramesContext *>(framesRef->data); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    ctx->format = AV_PIX_FMT_VAAPI;
    ctx->sw_format = AV_PIX_FMT_NV12;
    ctx->width = w;
    ctx->height = h;
    ctx->initial_pool_size = 1;
    if (av_hwframe_ctx_init(framesRef.get()) < 0) [[unlikely]] {
        esyslog("vaapivideo/device: black frame hw_frames_ctx init failed");
        return;
    }

    // SW staging frame -> uploaded to a single VAAPI surface via av_hwframe_transfer_data.
    std::unique_ptr<AVFrame, FreeAVFrame> hwFrame{av_frame_alloc()};
    std::unique_ptr<AVFrame, FreeAVFrame> swFrame{av_frame_alloc()};
    if (!hwFrame || !swFrame || av_hwframe_get_buffer(framesRef.get(), hwFrame.get(), 0) < 0) [[unlikely]] {
        esyslog("vaapivideo/device: black frame surface alloc failed");
        return;
    }
    swFrame->format = AV_PIX_FMT_NV12;
    swFrame->width = w;
    swFrame->height = h;
    if (av_frame_get_buffer(swFrame.get(), 0) < 0) [[unlikely]] {
        esyslog("vaapivideo/device: black frame sw buffer alloc failed");
        return;
    }

    // BT.709 TV-range "black": Y=16, UV=128. Y=0 would clip below black on TV-range
    // sinks and the rest of the pipeline (scale_vaapi out_range=tv) is TV-range too.
    const auto rows = static_cast<size_t>(h);
    std::memset(swFrame->data[0], 16, static_cast<size_t>(swFrame->linesize[0]) * rows);
    std::memset(swFrame->data[1], 128, static_cast<size_t>(swFrame->linesize[1]) * (rows / 2));
    if (av_hwframe_transfer_data(hwFrame.get(), swFrame.get(), 0) < 0) [[unlikely]] {
        esyslog("vaapivideo/device: black frame upload failed");
        return;
    }

    // Wrap and submit through the normal display path so the modeset / DRM plane state
    // matches a regular frame (no separate "blank" code path to maintain).
    auto frame = std::make_unique<VaapiFrame>();
    frame->avFrame = hwFrame.release();
    // FFmpeg VAAPI ABI: data[3] holds the VASurfaceID directly, cast through uintptr_t.
    frame->vaSurfaceId = static_cast<VASurfaceID>(
        reinterpret_cast<uintptr_t>(frame->avFrame->data[3])); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    static_cast<void>(display->SubmitFrame(std::move(frame), 100));
}

cVaapiDevice::cVaapiDevice() {
    isyslog("vaapivideo/device: created");
    SetDescription("VAAPI Video Device");
    SetVideoFormat(true);
}

cVaapiDevice::~cVaapiDevice() noexcept {
    dsyslog("vaapivideo/device: destroying (isPrimary=%d)", IsPrimaryDevice());

    // Demote ourselves first so VDR cannot route any further calls (PlayVideo / OSD lookups
    // / etc.) into a half-destroyed device.
    if (IsPrimaryDevice()) {
        cDevice::MakePrimaryDevice(false);
    }

    DetachAllReceivers();

    // Sever the OSD provider's display pointer before display is destroyed -- the OSD
    // provider outlives this device (VDR owns its lifecycle via cOsdProvider::Shutdown).
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

auto cVaapiDevice::Clear() -> void {
    cDevice::Clear();

    // trickSpeed is intentionally NOT reset: Clear() is a buffer flush, not a mode change.
    // VDR also calls Clear() at the start of trick play and resetting the speed here would
    // bounce us out of the mode the user just selected.

    // Force audio codec re-detection. This is the *only* place that catches the replay
    // audio-track-switch path: SVDRP "audi N" -> cDevice::SetCurrentAudioTrack -> player path
    // -> cDvbPlayer::SetAudioTrack -> Goto(Current) -> Empty() -> DeviceClear() -> here.
    // VDR routes around our SetAudioTrackDevice() override when a player is attached, so
    // without this reset audioCodecId stays pinned to the old codec and the still-open old
    // decoder is fed the new track's bitstream -- audio dies silently until the next channel
    // switch. Re-detection is cheap: SetStreamParams() early-returns on matching codec/rate.
    // Concurrency: Clear() runs on the main thread (or cDvbPlayer's mutex via Goto/Empty),
    // so PlayAudio() cannot race this reset.
    ResetAudioCodecState();

    // Bracket the decoder flush in a stream-switch window so the display thread cannot
    // re-present a surface that the decoder is about to invalidate.
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

[[nodiscard]] auto cVaapiDevice::DeviceType() const -> cString { return "VAAPI"; }

[[nodiscard]] auto cVaapiDevice::Flush(int TimeoutMs) -> bool {
    // VDR contract: return true once queued data has reached the output. We currently
    // only wait on the decoder packet queue, NOT on the audio queue. Audio is paced by
    // its own thread + ALSA backpressure and a stuck audio queue would never block VDR's
    // channel-switch flow in observed behavior. If a future regression shows lingering
    // audio across channel switches when Flush returns true, also waiting on
    // audioProcessor->GetQueueSize()==0 here is the place to add it.
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

    // Drop the queued packets so the next frame after un-pause is the user's intended one,
    // not stale lookahead. Deliberately does NOT touch sync state -- pause/resume should
    // preserve the EMA so we don't reseed from a transient.
    if (decoder) [[likely]] {
        decoder->DrainQueue();
    }

    // Drop ALSA buffers so audio cuts out in sync with the visual freeze (otherwise the
    // sink keeps playing whatever is already queued, ~100-200 ms past the freeze).
    if (audioProcessor) [[likely]] {
        audioProcessor->Clear();
    }
}

auto cVaapiDevice::GetOsdSize(int &Width, int &Height, double &PixelAspect) -> void {
    // Three tiers because VDR may call this before display is up (e.g. during plugin
    // setup-menu rendering) and because skins re-call it constantly during repaint.
    //   1. cached  -- normal hot path, no display call.
    //   2. live    -- first call after display init populates the cache.
    //   3. config  -- pre-init fallback. NOT cached: the real display may resolve later.
    if (osdWidth > 0 && osdHeight > 0 && display) [[likely]] {
        Width = osdWidth;
        Height = osdHeight;
        PixelAspect = display->GetAspectRatio();
        return;
    }

    if (display && display->IsInitialized()) {
        osdWidth = static_cast<int>(display->GetOutputWidth());
        osdHeight = static_cast<int>(display->GetOutputHeight());
        Width = osdWidth;
        Height = osdHeight;
        PixelAspect = display->GetAspectRatio();
        dsyslog("vaapivideo/device: OSD size cached: %dx%d aspect=%.3f", Width, Height, PixelAspect);
        return;
    }

    Width = static_cast<int>(vaapiConfig.display.GetWidth());
    Height = static_cast<int>(vaapiConfig.display.GetHeight());
    PixelAspect = static_cast<double>(Width) / Height;
}

[[nodiscard]] auto cVaapiDevice::GetSTC() -> int64_t {
    // VDR consumes this in 90 kHz units (PTS scale). The "true" STC would be the audio
    // clock from cAudioProcessor::GetClock(), but cDvbPlayer only uses the value for
    // editing-mark / position math where last-decoded-video PTS is accurate to within
    // one frame period and avoids returning AV_NOPTS_VALUE during the audio prime window.
    // Do NOT reuse this for A/V sync -- it can lag the real audio output by the decoder/
    // display pipeline depth.
    if (!decoder) [[unlikely]] {
        return -1;
    }
    const int64_t pts = decoder->GetLastPts();
    return pts != AV_NOPTS_VALUE ? pts : -1;
}

auto cVaapiDevice::GetVideoSize(int &Width, int &Height, double &VideoAspect) -> void {
    // Skins watch the 0->real transition to trigger a repaint, so report 0x0 (not last
    // known dimensions) whenever no codec is open or no frame has been decoded yet.
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

[[nodiscard]] auto cVaapiDevice::HasDecoder() const -> bool { return decoder && decoder->IsReady(); }

[[nodiscard]] auto cVaapiDevice::HasIBPTrickSpeed() -> bool { return true; }

auto cVaapiDevice::MakePrimaryDevice(bool On) -> void {
    dsyslog("vaapivideo/device: MakePrimaryDevice(%s) called", On ? "true" : "false");

    cDevice::MakePrimaryDevice(On);

    if (On) {
        if (IsPrimaryDevice()) {
            isyslog("vaapivideo/device: activated as primary device");
            // OSD provider lifecycle is owned by VDR (cOsdProvider::Shutdown()), so we
            // create at most one and reuse on subsequent activations.
            if (!::osdProvider && display) {
                ::osdProvider = new cVaapiOsdProvider(display.get());
                isyslog("vaapivideo/device: OSD provider registered");
            } else if (::osdProvider) {
                dsyslog("vaapivideo/device: OSD provider already exists, reusing");
            }
        } else {
            esyslog("vaapivideo/device: failed to activate as primary device");
        }
    } else {
        isyslog("vaapivideo/device: deactivated as primary device");
        // OSD provider is intentionally NOT torn down here -- VDR will run
        // cOsdProvider::Shutdown() globally after our Stop() during process exit.
    }
}

auto cVaapiDevice::Mute() -> void {
    cDevice::Mute();

    if (audioProcessor) [[likely]] {
        audioProcessor->Clear();
    }
}

auto cVaapiDevice::Play() -> void {
    cDevice::Play();

    // VDR calls Play() both for a genuine "resume normal speed" AND as a transition step
    // inside compound trick sequences like REW->PLAY->REW. We can't distinguish the two
    // up front, so we *request* trick exit and let the decoder confirm: if a TrickSpeed()
    // call follows within one frame period, the request is canceled, otherwise the decoder
    // exits trick mode for real on the next frame.
    if (trickSpeed.exchange(0, std::memory_order_release) != 0) {
        if (decoder) [[likely]] {
            decoder->RequestTrickExit();
        }
    }

    paused.store(false, std::memory_order_relaxed);
}

[[nodiscard]] auto cVaapiDevice::PlayAudio(const uchar *Data, int Length, uchar /*Id*/) -> int {
    if (!Data || Length <= 0 || paused.load(std::memory_order_relaxed) ||
        trickSpeed.load(std::memory_order_relaxed) != 0) [[unlikely]] {
        return Length;
    }

    if (!audioProcessor || !audioProcessor->IsInitialized()) [[unlikely]] {
        return Length;
    }

    const auto pes = ParsePes({Data, static_cast<size_t>(Length)});
    if (!pes.isAudio || pes.payloadSize == 0) [[unlikely]] {
        return Length;
    }

    // Codec detection runs while audioCodecId == NONE, reset by SetPlayMode(pmNone),
    // HandleAudioTrackChange(), or Clear() (replay audi-N path).
    const AVCodecID currentCodec = audioCodecId.load(std::memory_order_relaxed);
    bool isLive = liveMode.load(std::memory_order_relaxed);

    if (currentCodec == AV_CODEC_ID_NONE) {
        // pmAudioOnly skips video path, so latch liveMode here on first audio PES.
        if (!isLive) {
            isLive = Transferring();
            liveMode.store(isLive, std::memory_order_relaxed);
            if (decoder) {
                decoder->SetLiveMode(isLive);
            }
        }

        const AVCodecID detectedCodec = ::DetectAudioCodec({pes.payload, pes.payloadSize});
        if (detectedCodec == AV_CODEC_ID_NONE) [[unlikely]] {
            return Length;
        }

        // 2-of-2 confirmation ALWAYS. Race: SVDRP `audi N` updates currentAudioTrack on
        // the main thread but PlayTs may have already buffered a full PES from the OLD PID
        // inside tsToPesAudio. That stale PES arrives here after audioCodecId was reset,
        // detecting the wrong codec. Depth=2 suffices: tsToPesAudio.Reset() fires on every
        // PUSI, so at most one stale PES sneaks through. Cost: ~24-32 ms extra latency.
        if (detectedCodec == audioCodecCandidate) {
            ++audioCodecCandidateCount;
        } else {
            audioCodecCandidate = detectedCodec;
            audioCodecCandidateCount = 1;
        }
        if (audioCodecCandidateCount < 2) {
            dsyslog("vaapivideo/device: audio codec %s -- awaiting confirmation (%d/2)",
                    avcodec_get_name(detectedCodec), audioCodecCandidateCount);
            return Length;
        }

        audioCodecCandidate = AV_CODEC_ID_NONE;
        audioCodecCandidateCount = 0;

        if (!audioProcessor->OpenCodec(detectedCodec, 48000, 2)) [[unlikely]] {
            esyslog("vaapivideo/device: failed to open audio codec %s", avcodec_get_name(detectedCodec));
            return Length;
        }

        audioCodecId.store(detectedCodec, std::memory_order_relaxed);
        isyslog("vaapivideo/device: audio codec %s confirmed (%s, %s)", avcodec_get_name(detectedCodec),
                isLive ? "live" : "replay", audioProcessor->IsPassthrough() ? "passthrough" : "PCM");
    }

    // Radio detection: 3 s after pmAudioVideo with no video codec, paint black to clear
    // the previous channel's residual picture.
    if (radioBlackPending && radioBlackTimer.TimedOut()) [[unlikely]] {
        radioBlackPending = false;
        if (videoCodecId.load(std::memory_order_relaxed) == AV_CODEC_ID_NONE && display && vaapi.hwDeviceRef) {
            isyslog("vaapivideo/device: no video stream detected -- radio mode, showing black frame");
            SubmitBlackFrame();
        }
    }

    // Replay: return 0 for Poll() backpressure. Live: always accept, rely on the audio
    // queue's drop-oldest policy on overflow.
    if (!isLive && audioProcessor->IsQueueFull()) [[unlikely]] {
        return 0;
    }

    audioProcessor->Decode(pes.payload, pes.payloadSize, pes.pts);
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

    radioBlackPending = false;

    const auto pes = ParsePes({Data, static_cast<size_t>(Length)});
    if (!pes.isVideo || pes.payloadSize == 0) [[unlikely]] {
        return Length;
    }

    const AVCodecID currentCodec = videoCodecId.load(std::memory_order_relaxed);
    bool isLive = liveMode.load(std::memory_order_relaxed);

    if (currentCodec == AV_CODEC_ID_NONE) [[unlikely]] {
        // Latch live/replay here: cTransferControl isn't visible at SetPlayMode() time.
        isLive = Transferring();
        liveMode.store(isLive, std::memory_order_relaxed);
        decoder->SetLiveMode(isLive);

        const AVCodecID detectedCodec = ::DetectVideoCodec({pes.payload, pes.payloadSize});
        if (detectedCodec == AV_CODEC_ID_NONE) [[unlikely]] {
            return Length;
        }

        // Stale-PES guard: require 2-of-2 confirmation when codec matches previous channel.
        // Prevents leftover ring-buffer bytes from reopening the OLD decoder after a switch.
        // Narrower than PlayAudio's unconditional guard -- video has no audi-N race path.
        if (detectedCodec == previousVideoCodec && previousVideoCodec != AV_CODEC_ID_NONE) [[unlikely]] {
            if (detectedCodec == videoCodecCandidate) {
                ++videoCodecCandidateCount;
            } else {
                videoCodecCandidate = detectedCodec;
                videoCodecCandidateCount = 1;
            }
            if (videoCodecCandidateCount < 2) {
                dsyslog("vaapivideo/device: video codec %s same as previous -- awaiting confirmation (%d/2)",
                        avcodec_get_name(detectedCodec), videoCodecCandidateCount);
                return Length;
            }
            dsyslog("vaapivideo/device: video codec %s confirmed after %d detections", avcodec_get_name(detectedCodec),
                    videoCodecCandidateCount);
        }

        videoCodecCandidate = AV_CODEC_ID_NONE;
        videoCodecCandidateCount = 0;

        if (!decoder->OpenCodec(detectedCodec)) [[unlikely]] {
            esyslog("vaapivideo/device: failed to open video codec %s", avcodec_get_name(detectedCodec));
            return Length;
        }

        videoCodecId.store(detectedCodec, std::memory_order_relaxed);
        isyslog("vaapivideo/device: video codec %s (%s)", avcodec_get_name(detectedCodec), isLive ? "live" : "replay");
    }

    // Replay backpressure (return 0 -> VDR retries via Poll). Live: never block.
    // Trick mode caps queue depth at 1 for immediate keyframe visibility.
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

[[nodiscard]] auto cVaapiDevice::Poll(cPoller & /*Poller*/, int TimeoutMs) -> bool {
    if (!decoder) [[unlikely]] {
        return true;
    }

    // Live TV: cTransfer pushes via PlayVideo/PlayAudio and never asks Poll(). Returning
    // true is the safe default for any unexpected live caller.
    if (liveMode.load(std::memory_order_relaxed)) {
        return true;
    }

    const int currentSpeed = trickSpeed.load(std::memory_order_relaxed);

    auto hasSpace = [&]() -> bool {
        // Trick mode also gates on the per-frame pacing timer so we don't burst-feed the
        // decoder past what the display will actually show.
        if (currentSpeed != 0) {
            return decoder->IsReadyForNextTrickFrame() && decoder->GetQueueSize() < DECODER_TRICK_QUEUE_DEPTH;
        }
        return !decoder->IsQueueFull() && (!audioProcessor || !audioProcessor->IsQueueFull());
    };

    if (hasSpace()) {
        return true;
    }

    // Spin in-place rather than returning false. Returning false makes VDR retry through
    // its outer Poll() loop, which adds a full MainLoopInterval (~100 ms) per cycle and
    // visibly slows down replay startup / seek-confirm.
    if (TimeoutMs > 0) {
        const cTimeMs timeout(TimeoutMs);
        while (!hasSpace() && !timeout.TimedOut()) {
            cCondWait::SleepMs(5);
        }
        return hasSpace();
    }

    return false;
}

[[nodiscard]] auto cVaapiDevice::Ready() -> bool { return initState.load(std::memory_order_acquire) == 2; }

auto cVaapiDevice::SetAudioTrackDevice(eTrackType /*Type*/) -> void {
    // Legacy/safety hook: VDR only routes through here when no cPlayer is attached, which
    // in modern VDR is effectively never (live=cTransfer, replay=cDvbPlayer, both cPlayers).
    // Fires AFTER currentAudioTrack is assigned -> read in the handler is reliable, so we
    // take the dedup-against-current-track path.
    HandleAudioTrackChange("SetAudioTrackDevice", /*enteringDolby=*/false);
}

auto cVaapiDevice::SetDigitalAudioDevice(bool On) -> void {
    // The actually-fired audio-track-change hook for both live and replay. cDevice fires
    // it with On=true *BEFORE* assigning currentAudioTrack (vdr/device.c:1172-1180), so on
    // dolby-track switches GetCurrentAudioTrack() would still return the OLD track and the
    // dedup would silently swallow the change. Pass On through so the handler takes the
    // "walk dolby slots" stale-read workaround.
    HandleAudioTrackChange("SetDigitalAudioDevice", /*enteringDolby=*/On);
}

[[nodiscard]] auto cVaapiDevice::SetPlayMode(ePlayMode PlayMode) -> bool {
    static constexpr const char *kModeNames[] = {
        "pmNone", "pmAudioVideo", "pmAudioOnly", "pmAudioOnlyBlack", "pmVideoOnly", "pmExtern",
    };
    const auto idx = static_cast<unsigned>(PlayMode);
    dsyslog("vaapivideo/device: SetPlayMode(%s) called", idx < std::size(kModeNames) ? kModeNames[idx] : "unknown");

    // Clear stale Freeze() flag first -- would otherwise block PlayVideo() in the new mode.
    paused.store(false, std::memory_order_relaxed);

    switch (PlayMode) {
        case pmNone:
            // Full teardown. Capture previous codec ids BEFORE clearing: the 2-of-2
            // guards in PlayVideo/PlayAudio use them to catch stale TS-buffer bytes.
            // Audio reset is deferred to Clear() -> ResetAudioCodecState().
            radioBlackPending = false;
            previousVideoCodec = videoCodecId.exchange(AV_CODEC_ID_NONE, std::memory_order_relaxed);
            previousAudioCodec = audioCodecId.load(std::memory_order_relaxed);
            liveMode.store(false, std::memory_order_relaxed);
            trickSpeed.store(0, std::memory_order_release);
            if (decoder) [[likely]] {
                decoder->SetTrickSpeed(0);
                decoder->SetLiveMode(false);
                decoder->RequestCodecReopen();
            }
            videoCodecCandidate = AV_CODEC_ID_NONE;
            videoCodecCandidateCount = 0;
            // Reset dedup so the new channel's first audio-track hook re-detects.
            lastHandledAudioTrack = ttNone;
            lastHandledAudioPid = 0;
            Clear();
            break;
        case pmAudioOnly:
        case pmAudioOnlyBlack:
            // Radio: paint black so DRM scanout doesn't hold the previous channel's picture.
            radioBlackPending = false;
            Clear();
            if (display && vaapi.hwDeviceRef) [[likely]] {
                SubmitBlackFrame();
            }
            break;
        case pmAudioVideo:
        case pmVideoOnly:
            // Codec state already clean (VDR emits pmNone first). liveMode latched on first PES.
            Clear();
            // 3 s grace for video; if only audio arrives, PlayAudio() paints black.
            radioBlackTimer.Set(3000);
            radioBlackPending = true;
            break;
        default:
            break;
    }
    return true;
}

auto cVaapiDevice::SetVolumeDevice(int Volume) -> void {
    if (audioProcessor) [[likely]] {
        audioProcessor->SetVolume(Volume);
    }
}

auto cVaapiDevice::StillPicture(const uchar *Data, int Length) -> void {
    if (!Data || Length <= 0) [[unlikely]] {
        return;
    }

    // Re-entry guard: TS path re-enters us as PES; only outermost call manages state.
    const bool isOuterCall = !inStillPicture;

    bool wasPaused = false;
    if (isOuterCall) {
        inStillPicture = true;
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
        // PES buffer: iterate packets individually (PlayVideo handles one per call).
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

    // VDR's API doesn't pass fast/slow directly. The convention it uses is: Freeze() (which
    // sets paused=true) is called BEFORE slow trick, but NOT before fast FF/REW. So paused
    // at this point implies slow mode.
    const bool isFast = !paused.load(std::memory_order_relaxed);

    trickSpeed.store(Speed, std::memory_order_release);

    if (decoder) [[likely]] {
        decoder->SetTrickSpeed(Speed, Forward, isFast);
    }

    // Drop audio explicitly. VDR usually calls Mute() too but not always at the same edge,
    // and a delay leaves the user hearing 100-200 ms of stale audio after the trick command.
    if (audioProcessor) [[likely]] {
        audioProcessor->Clear();
    }
}

// ============================================================================
// === PUBLIC API ===
// ============================================================================

[[nodiscard]] auto cVaapiDevice::Attach() -> bool {
    // SVDRP ATTA: re-open the same DRM/VAAPI/audio combo we last had. drmPath/audioDevice
    // were latched in the prior Initialize() so SVDRP doesn't need to re-pass them.
    if (drmPath.empty()) [[unlikely]] {
        esyslog("vaapivideo/device: cannot attach - no DRM device path recorded");
        return false;
    }
    isyslog("vaapivideo/device: re-attaching to hardware (DRM=%s audio=%s)", drmPath.c_str(), audioDevice.c_str());

    if (!Initialize(drmPath, audioDevice)) [[unlikely]] {
        return false;
    }

    // Reconnect the OSD provider to the newly created display instance.
    if (auto *provider = dynamic_cast<cVaapiOsdProvider *>(::osdProvider)) {
        provider->AttachDisplay(display.get());
        dsyslog("vaapivideo/device: OSD provider re-attached to display");
    }

    return true;
}

auto cVaapiDevice::Detach() -> void {
    isyslog("vaapivideo/device: detaching from hardware");

    // Quiesce upstream sources BEFORE Stop() to avoid use-after-free: cTransfer's tuner
    // thread calls PlayVideo/PlayAudio which dereference `decoder`. cControl::Shutdown()
    // must precede Stop() because the unwinding cPlayer::Detach() fires SetPlayMode(pmNone)
    // -> Clear(), which needs decoder/display/audioProcessor still alive.
    // NOTE: ~cVaapiDevice does NOT need this -- VDR's shutdown sequence handles the order.
    cControl::Shutdown();
    DetachAllReceivers();

    // Disconnect OSD provider from display before display destruction.
    if (auto *provider = dynamic_cast<cVaapiOsdProvider *>(::osdProvider)) {
        provider->DetachDisplay();
        dsyslog("vaapivideo/device: OSD provider detached from display");
    }

    Stop();

    // Force-release OSD mmap'd dumb buffers: VDR may keep cVaapiOsd objects alive past
    // Detach(), and their kernel refs would block drmDropMaster/close.
    if (auto *provider = dynamic_cast<cVaapiOsdProvider *>(::osdProvider)) {
        provider->ReleaseAllOsdResources();
    }

    // drmDropMaster before close lets another client take master immediately.
    if (drmFd >= 0) {
        drmDropMaster(drmFd);
    }

    ReleaseHardware();
    RestoreConsole();

    // Reset all state so a subsequent Attach() re-detects codecs from scratch.
    videoCodecId.store(AV_CODEC_ID_NONE, std::memory_order_relaxed);
    previousVideoCodec = AV_CODEC_ID_NONE;
    previousAudioCodec = AV_CODEC_ID_NONE;
    videoCodecCandidate = AV_CODEC_ID_NONE;
    videoCodecCandidateCount = 0;
    ResetAudioCodecState();
    liveMode.store(false, std::memory_order_relaxed);
    trickSpeed.store(0, std::memory_order_relaxed);
    paused.store(false, std::memory_order_relaxed);
    lastHandledAudioTrack = ttNone;
    lastHandledAudioPid = 0;
    osdWidth = 0;
    osdHeight = 0;
    initState.store(0, std::memory_order_release);
    isyslog("vaapivideo/device: detached");
}

[[nodiscard]] auto cVaapiDevice::Initialize(std::string_view drmDevicePath, std::string_view audioDevicePath) -> bool {
    // 0 -> 1 (in-progress) -> 2 (ready). The CAS protects against double-Initialize from
    // a racing SVDRP ATTA + plugin Start() during process bring-up.
    int expected = 0;
    if (!initState.compare_exchange_strong(expected, 1)) [[unlikely]] {
        esyslog("vaapivideo/device: already initialized (state=%d)", expected);
        return false;
    }

    drmPath = drmDevicePath;
    audioDevice = audioDevicePath;

    dsyslog("vaapivideo/device: initializing - DRM '%s', audio '%s'", drmPath.c_str(), audioDevice.c_str());

    if (!OpenHardware()) [[unlikely]] {
        esyslog("vaapivideo/device: hardware initialization failed");
        initState.store(0, std::memory_order_relaxed);
        return false;
    }

    // Construction order: audioProcessor, then display, then decoder. The decoder is given
    // raw pointers to the other two and queries them on every frame, so they must already
    // exist (and be valid for its thread's lifetime). Stop() reverses this order.
    audioProcessor = std::make_unique<cAudioProcessor>();
    if (!audioProcessor->Initialize(audioDevice)) [[unlikely]] {
        esyslog("vaapivideo/device: audio initialization failed");
        audioProcessor.reset();
        ReleaseHardware();
        initState.store(0, std::memory_order_relaxed);
        return false;
    }

    display = std::make_unique<cVaapiDisplay>();
    if (!display->Initialize(drmFd, vaapi.hwDeviceRef, crtcId, connectorId, activeMode)) [[unlikely]] {
        esyslog("vaapivideo/device: display initialization failed");
        display.reset();
        audioProcessor->Shutdown();
        audioProcessor.reset();
        ReleaseHardware();
        initState.store(0, std::memory_order_relaxed);
        return false;
    }

    // Decoder starts codec-less; PlayVideo()'s detection path opens it on the first PES.
    decoder = std::make_unique<cVaapiDecoder>(display.get(), &vaapi);
    if (!decoder->Initialize()) [[unlikely]] {
        esyslog("vaapivideo/device: decoder initialization failed");
        decoder.reset();
        display->Shutdown();
        display.reset();
        audioProcessor->Shutdown();
        audioProcessor.reset();
        ReleaseHardware();
        initState.store(0, std::memory_order_relaxed);
        return false;
    }

    // Hand the decoder its audio-clock source for A/V sync.
    decoder->SetAudioProcessor(audioProcessor.get());

    // Populate the OSD-size cache eagerly: the upcoming MakePrimaryDevice() may call
    // GetOsdSize() before display has been queried otherwise.
    osdWidth = static_cast<int>(display->GetOutputWidth());
    osdHeight = static_cast<int>(display->GetOutputHeight());
    dsyslog("vaapivideo/device: pre-cached OSD size %dx%d", osdWidth, osdHeight);

    initState.store(2, std::memory_order_release);
    isyslog("vaapivideo/device: initialized - DRM=%s audio=%s", drmPath.c_str(), audioDevice.c_str());

    // Paint black so we don't expose whatever the last DRM client (or boot splash) left
    // in the scanout buffer until the first real frame arrives.
    SubmitBlackFrame();

    return true;
}

auto cVaapiDevice::Stop() -> void {
    // Reverse of Initialize(): decoder first because its Action thread holds raw pointers
    // to display + audioProcessor and dereferences both per frame. Destroying either while
    // the thread is alive is a use-after-free. No back-references the other way, so the
    // remaining order is for cleanliness. Each Shutdown() is idempotent.
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

// ============================================================================
// === INTERNAL METHODS ===
// ============================================================================

auto cVaapiDevice::HandleAudioTrackChange(const char *reason, bool enteringDolby) -> void {
    // Single funnel for audio-track notifications: resolve track, dedup against last
    // (type, PID), and on a real change reset audioCodecId + drain audio so PlayAudio()
    // re-detects on the next PES.
    //
    // Entry points (all main-thread under mutexCurrentAudioTrack, no extra locking):
    //   SetAudioTrackDevice   - no cPlayer attached; reliable post-assignment read.
    //   SetDigitalAudioDevice - On=true fires BEFORE assignment (dolby stale-read).
    //   Clear() via Empty/Goto - replay-path safety net.

    eTrackType type = ttNone;
    const tTrackId *track = nullptr;

    if (enteringDolby) {
        // Stale-read workaround: GetCurrentAudioTrack() still returns the OLD track.
        // Walk dolby slots and pick the unique populated one; for multi-dolby channels
        // fall back to "unknown" -- the reset still fires and PlayAudio re-detects.
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
                // Ambiguous -- multiple dolby tracks present. Fall back to "unknown".
                type = ttNone;
                track = nullptr;
                break;
            }
        }
    } else {
        // Reliable post-assignment read path.
        type = GetCurrentAudioTrack();
        track = GetTrack(type);
    }

    const auto pid = static_cast<uint16_t>(track ? track->id : 0);

    if (type != ttNone && type == lastHandledAudioTrack && pid == lastHandledAudioPid) {
        // PMT churn / same track re-selection. Log for SVDRP correlation but skip reset.
        isyslog("vaapivideo/device: %s -> %s track %d (PID=%u) -- no change", reason,
                IS_AUDIO_TRACK(type) ? "audio" : (IS_DOLBY_TRACK(type) ? "dolby" : "unknown"), static_cast<int>(type),
                pid);
        return;
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
        // Ambiguous-dolby fallback; the next-packet log will identify the actual codec.
        isyslog("vaapivideo/device: %s -> %s track switch (codec re-detect on next packet)", reason, kind);
    }

    // Trigger PlayAudio()'s detection cycle and clear stale audio state.
    ResetAudioCodecState();
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

    // R for the resource query ioctls, W for KMS modeset/atomic ioctls. Group membership
    // ('video' or 'render') is the usual gotcha here, hence the explicit hint below.
    if (access(drmPath.c_str(), R_OK | W_OK) != 0) [[unlikely]] {
        esyslog("vaapivideo/device: DRM device '%s' not accessible -- %s", drmPath.c_str(), strerror(errno));
        esyslog("vaapivideo/device: ensure user is in 'video' or 'render' group");
        return false;
    }

    drmFd = open(drmPath.c_str(), O_RDWR | O_CLOEXEC);
    if (drmFd < 0) [[unlikely]] {
        esyslog("vaapivideo/device: failed to open '%s' - %s", drmPath.c_str(), strerror(errno));
        return false;
    }
    dsyslog("vaapivideo/device: opened DRM fd=%d", drmFd);

    if (!SelectDrmConnector()) [[unlikely]] {
        esyslog("vaapivideo/device: no connected display found on %s", drmPath.c_str());
        close(drmFd);
        drmFd = -1;
        return false;
    }

    // VAAPI must use the render node (/dev/dri/renderD*), not the primary card node we
    // opened above. drmGetDevice2 gives us the matching render path for the same GPU.
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
        esyslog("vaapivideo/device: av_hwdevice_ctx_create failed - %s", AvErr(ret).data());
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
    // Populates the vaapi.hw* / has* / deinterlaceMode flags consumed by the decoder and
    // filter graph. Fatal-fails if VAEntrypointVideoProc is missing -- without VPP we can
    // neither deinterlace nor color-convert and the GPU is unusable for this plugin.
    vaapi.hasDenoise = false;
    vaapi.hasSharpness = false;
    vaapi.hwH264 = false;
    vaapi.hwHevc = false;
    vaapi.hwMpeg2 = false;
    vaapi.deinterlaceMode.clear();

    // Throwaway VADisplay for probing. Reusing FFmpeg's VADisplay tickles a bug in some
    // iHD driver versions (observed on iHD 24.x) where vaCreateContext starts failing
    // intermittently after the first probe context is destroyed.
    const std::string renderPath{renderNode};
    const int renderFd = open(renderPath.c_str(), O_RDWR | O_CLOEXEC);
    if (renderFd < 0) [[unlikely]] {
        esyslog("vaapivideo/device: VPP probe failed -- cannot open render node: %s", strerror(errno));
        return false;
    }

    VADisplay vaDisplay = vaGetDisplayDRM(renderFd);
    if (!vaDisplay) [[unlikely]] {
        esyslog("vaapivideo/device: VPP probe failed -- vaGetDisplayDRM error");
        close(renderFd);
        return false;
    }

    int vaMajor = 0;
    int vaMinor = 0;
    if (vaInitialize(vaDisplay, &vaMajor, &vaMinor) != VA_STATUS_SUCCESS) [[unlikely]] {
        esyslog("vaapivideo/device: VPP probe failed -- vaInitialize error");
        close(renderFd);
        return false;
    }

    const char *vendorStr = vaQueryVendorString(vaDisplay);
    isyslog("vaapivideo/device: VA-API driver -- %s", vendorStr ? vendorStr : "(unknown)");

    // Walk the driver's profile list and tag the broadcast codecs we care about as HW
    // capable iff they support both VLD (= bitstream decode) and YUV420 surfaces.
    const int maxProfiles = vaMaxNumProfiles(vaDisplay);
    if (maxProfiles <= 0) [[unlikely]] {
        esyslog("vaapivideo/device: vaMaxNumProfiles failed");
        vaTerminate(vaDisplay);
        close(renderFd);
        return false;
    }
    std::vector<VAProfile> profiles(static_cast<size_t>(maxProfiles));
    int numProfiles = 0;
    if (vaQueryConfigProfiles(vaDisplay, profiles.data(), &numProfiles) == VA_STATUS_SUCCESS) {
        for (size_t i = 0; i < static_cast<size_t>(numProfiles); ++i) {
            const VAProfile profile =
                profiles[i]; // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            switch (profile) {
                case VAProfileMPEG2Simple:
                case VAProfileMPEG2Main:
                    if (!vaapi.hwMpeg2 && HasVldDecode(vaDisplay, profile)) {
                        vaapi.hwMpeg2 = true;
                    }
                    break;
                case VAProfileH264ConstrainedBaseline:
                case VAProfileH264Main:
                case VAProfileH264High:
                    if (!vaapi.hwH264 && HasVldDecode(vaDisplay, profile)) {
                        vaapi.hwH264 = true;
                    }
                    break;
                case VAProfileHEVCMain:
                case VAProfileHEVCMain10:
                    if (!vaapi.hwHevc && HasVldDecode(vaDisplay, profile)) {
                        vaapi.hwHevc = true;
                    }
                    break;
                default:
                    break;
            }
            if (vaapi.hwMpeg2 && vaapi.hwH264 && vaapi.hwHevc) {
                break;
            }
        }
    }

    isyslog("vaapivideo/device: VAAPI decode -- mpeg2=%s h264=%s hevc=%s", vaapi.hwMpeg2 ? "hw" : "sw",
            vaapi.hwH264 ? "hw" : "sw", vaapi.hwHevc ? "hw" : "sw");

    // vaQueryVideoProcFilters needs a live VPP context (config + surface + context).
    // Build a minimal one just for the probe and tear it all down at the bottom.
    VAConfigID configId = VA_INVALID_ID;
    if (vaCreateConfig(vaDisplay, VAProfileNone, VAEntrypointVideoProc, nullptr, 0, &configId) != VA_STATUS_SUCCESS)
        [[unlikely]] {
        esyslog("vaapivideo/device: VAEntrypointVideoProc unavailable -- cannot initialize");
        vaTerminate(vaDisplay);
        close(renderFd);
        return false;
    }

    // vaCreateContext requires at least one render target surface. Use 64x64 instead of
    // a 1x1 placeholder: the iHD driver SIGSEGVs on zero-size dimensions and some VPP
    // entrypoints reject sub-block sizes outright.
    VASurfaceID surface = VA_INVALID_SURFACE;
    if (vaCreateSurfaces(vaDisplay, VA_RT_FORMAT_YUV420, 64, 64, &surface, 1, nullptr, 0) != VA_STATUS_SUCCESS)
        [[unlikely]] {
        vaDestroyConfig(vaDisplay, configId);
        esyslog("vaapivideo/device: VPP probe failed -- vaCreateSurfaces error");
        vaTerminate(vaDisplay);
        close(renderFd);
        return false;
    }

    VAContextID contextId = VA_INVALID_ID;
    if (vaCreateContext(vaDisplay, configId, 64, 64, 0, &surface, 1, &contextId) != VA_STATUS_SUCCESS) [[unlikely]] {
        vaDestroySurfaces(vaDisplay, &surface, 1);
        vaDestroyConfig(vaDisplay, configId);
        esyslog("vaapivideo/device: VPP probe failed -- vaCreateContext error");
        vaTerminate(vaDisplay);
        close(renderFd);
        return false;
    }

    VAProcFilterType filters[VAProcFilterCount];
    auto numFilters = static_cast<unsigned int>(VAProcFilterCount);
    if (vaQueryVideoProcFilters(vaDisplay, contextId, filters, &numFilters) != VA_STATUS_SUCCESS) {
        numFilters = 0;
    }

    for (unsigned int i = 0; i < numFilters; ++i) {
        if (filters[i] == VAProcFilterNoiseReduction) {
            vaapi.hasDenoise = true;
        } else if (filters[i] == VAProcFilterSharpening) {
            vaapi.hasSharpness = true;
        }
    }

    // Pick the best deinterlace mode the driver advertises. Quality order:
    // motion_compensated > motion_adaptive > weave > bob. The decoder uses this string in
    // its deinterlace_vaapi filter on the HW-decode path; if empty, the HW path runs with
    // no deinterlacer (the SW-decode path uses bwdif unconditionally).
    VAProcFilterCapDeinterlacing deintCaps[VAProcDeinterlacingCount];
    auto numDeintCaps = static_cast<unsigned int>(VAProcDeinterlacingCount);
    if (vaQueryVideoProcFilterCaps(vaDisplay, contextId, VAProcFilterDeinterlacing, deintCaps, &numDeintCaps) ==
        VA_STATUS_SUCCESS) {
        static constexpr struct {
            VAProcDeinterlacingType type;
            const char *name;
        } kDeintModes[] = {
            {.type = VAProcDeinterlacingMotionCompensated, .name = "motion_compensated"},
            {.type = VAProcDeinterlacingMotionAdaptive, .name = "motion_adaptive"},
            {.type = VAProcDeinterlacingWeave, .name = "weave"},
            {.type = VAProcDeinterlacingBob, .name = "bob"},
        };

        const auto *begin = static_cast<const VAProcFilterCapDeinterlacing *>(deintCaps);
        const auto *end = begin + numDeintCaps;
        for (const auto &[type, name] : kDeintModes) {
            if (std::any_of(begin, end, [type](const auto &c) -> bool { return c.type == type; })) {
                vaapi.deinterlaceMode = name;
                break;
            }
        }
    }

    // libva offers no RAII wrappers; tear down in reverse creation order by hand.
    vaDestroyContext(vaDisplay, contextId);
    vaDestroySurfaces(vaDisplay, &surface, 1);
    vaDestroyConfig(vaDisplay, configId);
    vaTerminate(vaDisplay);
    close(renderFd);

    isyslog("vaapivideo/device: VPP capabilities -- denoise=%s sharpen=%s deinterlace=%s",
            vaapi.hasDenoise ? "yes" : "no", vaapi.hasSharpness ? "yes" : "no",
            vaapi.deinterlaceMode.empty() ? "none" : vaapi.deinterlaceMode.c_str());
    return true;
}

auto cVaapiDevice::ReleaseHardware() -> void {
    if (vaapi.hwDeviceRef) {
        av_buffer_unref(&vaapi.hwDeviceRef);
        vaapi.drmFd = -1; // vaapi.drmFd is a borrowed copy of drmFd; the real close is below.
    }

    if (drmFd >= 0) {
        close(drmFd);
        drmFd = -1;
    }
}

auto cVaapiDevice::ResetAudioCodecState() -> void {
    // Forget the audio stream identity. PlayAudio() re-runs detection on its next PES.
    // Clearing the candidate fields too is REQUIRED -- a stale partial 2-of-2 count would
    // otherwise let the next single detection result confirm against the previous cycle's
    // codec and re-open the wrong decoder.
    audioCodecId.store(AV_CODEC_ID_NONE, std::memory_order_relaxed);
    audioCodecCandidate = AV_CODEC_ID_NONE;
    audioCodecCandidateCount = 0;
}

[[nodiscard]] auto cVaapiDevice::SelectDrmConnector() -> bool {

    if (drmSetClientCap(drmFd, DRM_CLIENT_CAP_ATOMIC, 1) != 0) [[unlikely]] {
        esyslog("vaapivideo/device: failed to enable DRM atomic capability");
        return false;
    }

    std::unique_ptr<drmModeRes, FreeDrmResources> resources{drmModeGetResources(drmFd)};
    if (!resources) [[unlikely]] {
        esyslog("vaapivideo/device: failed to get DRM resources");
        return false;
    }

    // Existence-check the plane resources -- atomic modesetting needs them, but we don't
    // use the values here (the display module re-queries planes during modeset).
    if (!std::unique_ptr<drmModePlaneRes, FreeDrmPlaneResources>{drmModeGetPlaneResources(drmFd)}) [[unlikely]] {
        esyslog("vaapivideo/device: failed to get DRM plane resources (atomic required)");
        return false;
    }

    const auto targetWidth = vaapiConfig.display.GetWidth();
    const auto targetHeight = vaapiConfig.display.GetHeight();
    const auto targetRate = vaapiConfig.display.GetRefreshRate();

    // Mode selection priority:
    //   1. exact (width, height, refresh) match against the plugin config -- operator wins
    //   2. driver's PREFERRED mode (typically the panel's native mode)
    //   3. first listed mode (the connector won't expose any if nothing is connected)

    for (int i = 0; i < resources->count_connectors; ++i) {
        std::unique_ptr<drmModeConnector, FreeDrmConnector> connector{
            drmModeGetConnector(drmFd, resources->connectors[i])};
        if (!connector || connector->connection != DRM_MODE_CONNECTED || connector->count_modes == 0) [[likely]] {
            continue;
        }

        bool modeFound = false;
        for (int modeIdx = 0; modeIdx < connector->count_modes; ++modeIdx) {
            const auto &mode = connector->modes[modeIdx];
            if (mode.hdisplay == targetWidth && mode.vdisplay == targetHeight && mode.vrefresh == targetRate) {
                activeMode = mode;
                modeFound = true;
                break;
            }
        }

        if (!modeFound) {
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

        std::unique_ptr<drmModeObjectProperties, FreeDrmObjectProperties> props{
            drmModeObjectGetProperties(drmFd, connector->connector_id, DRM_MODE_OBJECT_CONNECTOR)};
        if (props) {
            for (uint32_t propIdx = 0; propIdx < props->count_props; ++propIdx) {
                std::unique_ptr<drmModePropertyRes, FreeDrmProperty> prop{
                    drmModeGetProperty(drmFd, props->props[propIdx])};
                // Read CRTC_ID directly from the connector's atomic property: on atomic
                // drivers the legacy encoder_id walk can return a stale CRTC, while the
                // CRTC_ID property always reflects the current binding.
                if (prop && std::strcmp(prop->name, "CRTC_ID") == 0) {
                    crtcId = static_cast<uint32_t>(props->prop_values[propIdx]);
                    break;
                }
            }
        }

        if (crtcId == 0 && resources->count_crtcs > 0) {
            crtcId = resources->crtcs[0];
        }

        if (crtcId != 0) {
            connectorId = connector->connector_id;
            isyslog("vaapivideo/device: display %ux%u@%uHz (connector %u, CRTC %u)", activeMode.hdisplay,
                    activeMode.vdisplay, activeMode.vrefresh, connectorId, crtcId);
            return true;
        }
    }

    esyslog("vaapivideo/device: no connected display found");
    return false;
}
