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
#include <cstring>
#include <format>
#include <iterator>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

// POSIX
#include <fcntl.h>
#include <linux/vt.h>
#include <pthread.h>
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

// VAAPI (only base types; profile/VPP enumeration lives in caps.cpp)
#include <va/va.h>

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

/// ~320 ms at AC-3 32 ms framing. Shallow cap prevents tail-drops that would create PTS
/// gaps and derail A/V sync long after the drop event.
static constexpr size_t AUDIO_REPLAY_QUEUE_HIGHWATER = 10;

/// VT bounce to hand the framebuffer back to fbcon after dropping DRM master.
/// fbcon does not auto-reclaim when another DRM client still holds the device open;
/// VT_ACTIVATE to a different VT and back forces the kernel to reprogram the CRTC.
/// Returns true iff the bounce completed; on failure the caller surfaces the error in
/// the SVDRP reply (user can press Alt+F1 manually).
///
/// VT_ACTIVATE takes a VT number, not an fd; only /dev/tty0 needs to be readable.
///
/// Three prerequisites must ALL hold (see README "SVDRP commands"):
///   1. /dev/tty0 mode 0660  (distros ship 0600; fix via udev KERNEL=="tty0",MODE="0660")
///   2. The vdr user in the 'tty' supplementary group.
///   3. CAP_SYS_TTY_CONFIG in the vdr process's ambient set. Critical subtlety: VDR's
///      '-u vdr' does setuid(0->vdr) inside the process and the kernel clears ambient
///      caps on any 0->non-0 UID change. AmbientCapabilities= alone is NOT enough when
///      vdr starts as root. The systemd unit must do the UID switch itself:
///          [Service]
///          User=vdr
///          Group=video
///          SupplementaryGroups=tty
///          AmbientCapabilities=CAP_SYS_TTY_CONFIG
///      Then vdr starts already as the vdr user, '-u vdr' is a no-op, and the cap survives.
[[nodiscard]] static auto RestoreConsole() -> bool {
    const int ttyFd = open("/dev/tty0", O_RDWR | O_CLOEXEC);
    if (ttyFd < 0) {
        esyslog("vaapivideo/device: console restore FAILED -- cannot open /dev/tty0: %s", strerror(errno));
        esyslog("vaapivideo/device: add 'vdr' to the 'tty' group and widen /dev/tty0 to 0660 via udev "
                "(KERNEL==\"tty0\", GROUP=\"tty\", MODE=\"0660\")");
        return false;
    }

    struct vt_stat vtState{};
    if (ioctl(ttyFd, VT_GETSTATE, &vtState) != 0) {
        esyslog("vaapivideo/device: console restore FAILED -- VT_GETSTATE: %s", strerror(errno));
        close(ttyFd);
        return false;
    }

    const auto currentVt = static_cast<int>(vtState.v_active);

    // VT_ACTIVATE to the current VT is a no-op in the kernel; pick any other VT to force a
    // real switch (and therefore a CRTC reprogram).
    const int tempVt = (currentVt == 1) ? 2 : 1;

    // VT_ACTIVATE/VT_WAITACTIVE are gated by CAP_SYS_TTY_CONFIG for processes without a
    // matching controlling tty. EPERM here means /dev/tty0 perms are fine but the daemon
    // lacks the capability -- log once, abort the bounce cleanly.
    if (ioctl(ttyFd, VT_ACTIVATE, tempVt) != 0) {
        const int savedErrno = errno;
        esyslog("vaapivideo/device: console restore FAILED -- VT_ACTIVATE(%d): %s", tempVt, strerror(savedErrno));
        if (savedErrno == EPERM) {
            esyslog("vaapivideo/device: vdr lacks CAP_SYS_TTY_CONFIG -- systemd must switch user itself "
                    "(drop-in: User=vdr, Group=video, SupplementaryGroups=tty, "
                    "AmbientCapabilities=CAP_SYS_TTY_CONFIG) so the ambient cap survives; setting "
                    "AmbientCapabilities alone is cleared by vdr's internal -u vdr setuid");
        }
        close(ttyFd);
        return false;
    }
    ioctl(ttyFd, VT_WAITACTIVE, tempVt);

    if (ioctl(ttyFd, VT_ACTIVATE, currentVt) == 0) {
        ioctl(ttyFd, VT_WAITACTIVE, currentVt);
    }

    close(ttyFd);
    isyslog("vaapivideo/device: console VT%d restored", currentVt);
    return true;
}

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
    dsyslog("vaapivideo/device: enumerating DRM devices");

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
    // Clears residual scanout on radio channels, hardware reinit, and opt-in channel switch.
    // Uses a one-shot hw_frames_ctx so it works before the decoder pipeline is up.
    if (!display || !vaapi.hwDeviceRef) [[unlikely]] {
        return;
    }

    // Stage SDR HDR state before submitting: if the connector was left in HDR mode the sink
    // would interpret SDR NV12 as BT.2020 PQ/HLG (crushed blacks, wrong colors). The atomic
    // commit clears HDR_OUTPUT_METADATA / Colorspace / BT.2020 COLOR_ENCODING together with
    // the black framebuffer. Normal SDR staging via the decoder is bypassed on all these
    // call sites, so we must do it explicitly here.
    display->SetHdrOutputState(HdrStreamInfo{});

    const auto w = static_cast<int>(display->GetOutputWidth());
    const auto h = static_cast<int>(display->GetOutputHeight());

    std::unique_ptr<AVBufferRef, FreeAVBufferRef> framesRef{av_hwframe_ctx_alloc(vaapi.hwDeviceRef)};
    if (!framesRef) [[unlikely]] {
        esyslog("vaapivideo/device: black frame hw_frames_ctx alloc failed");
        return;
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
    if (av_hwframe_ctx_init(framesRef.get()) < 0) [[unlikely]] {
        esyslog("vaapivideo/device: black frame hw_frames_ctx init failed");
        return;
    }

    // SW staging frame uploaded to a single VAAPI surface via av_hwframe_transfer_data.
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

    // Submit through the normal display path: modeset / DRM plane state mirrors a real frame.
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
    // Pass CheckDecoder=false: the decoder may already be torn down (e.g. after Detach()),
    // and the default HasDecoder() gate would misreport us as non-primary.
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

auto cVaapiDevice::Clear() -> void {
    // Diagnostic: log thread + inter-call delta to diagnose unexpected Clear() bursts.
    // Expected callers: cDvbPlayer::Empty() (pause/play/trick/goto) and
    // cTransfer::Receive() (live-TV retry exhaustion). Rapid-fire calls outside those
    // paths should be traceable via the logged thread name.
    const auto nowMs = cTimeMs::Now();
    const uint64_t prevMs = lastClearMs.exchange(nowMs, std::memory_order_relaxed);
    std::array<char, 16> threadName{};
    (void)pthread_getname_np(pthread_self(), threadName.data(), threadName.size());
    if (prevMs == 0) {
        isyslog("vaapivideo/device: Clear() thread='%s' (first call)", threadName.data());
    } else {
        isyslog("vaapivideo/device: Clear() thread='%s' delta=%llums", threadName.data(),
                static_cast<unsigned long long>(nowMs - prevMs));
    }

    cDevice::Clear();

    // trickSpeed intentionally NOT reset: Clear() is a buffer flush, not a mode change.
    // VDR calls Clear() at the start of trick play; resetting here would cancel the mode.

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
    }

    // Drop ALSA buffers: ~100-200 ms of audio is already queued in the sink and would
    // continue playing past the freeze without an explicit drain.
    if (audioProcessor) [[likely]] {
        audioProcessor->Clear();
    }
}

auto cVaapiDevice::GetOsdSize(int &Width, int &Height, double &PixelAspect) -> void {
    // Called frequently by skins during repaint. Three tiers:
    //   1. cached  -- normal hot path after display init.
    //   2. live    -- first post-init call populates the cache.
    //   3. config  -- pre-init fallback (NOT cached; real display may appear later).
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
    // Returns 90 kHz PTS of the last decoded video frame. The true STC would be the audio
    // clock, but cDvbPlayer only uses this for editing-mark / position math where one-frame
    // accuracy is sufficient, and it avoids AV_NOPTS_VALUE during the audio prime window.
    // Do NOT use for A/V sync: lags real audio output by decoder+display pipeline depth.
    if (!decoder) [[unlikely]] {
        return -1;
    }
    const int64_t pts = decoder->GetLastPts();
    return pts != AV_NOPTS_VALUE ? pts : -1;
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

[[nodiscard]] auto cVaapiDevice::HasDecoder() const -> bool { return decoder && decoder->IsReady(); }

[[nodiscard]] auto cVaapiDevice::HasIBPTrickSpeed() -> bool { return true; }

auto cVaapiDevice::MakePrimaryDevice(bool On) -> void {
    dsyslog("vaapivideo/device: MakePrimaryDevice(%s) called", On ? "true" : "false");

    // Deferred-init trigger: after VDR startup finishes, promote-to-primary opens hardware
    // if still detached. startupComplete gate prevents setup.conf-driven primary restore
    // during VDR bring-up from defeating --detached (normal path skips this, state is 2).
    //
    // Failure is NOT a veto: VDR assigns primaryDevice before calling this hook
    // (vdr/device.c:201-202). Returning early would strand us as a non-functional primary.
    // Log and fall through so the device lands in the recoverable "detached primary" state;
    // SVDRP ATTA can retry.
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
        // CheckDecoder=false: during --detached the decoder is not yet open; the default
        // HasDecoder() gate would misreport us as non-primary.
        if (IsPrimaryDevice(/*CheckDecoder=*/false)) {
            isyslog("vaapivideo/device: activated as primary device");
            // OSD provider lifecycle is owned by VDR (cOsdProvider::Shutdown()): create once,
            // reattach thereafter. Must register even when detached so skins see TrueColor
            // support during Start() -- cOsdProvider::SupportsTrueColor() logs an error if
            // no provider is registered. Pixel-path methods are display-null-safe; a later
            // AttachDisplay() wires up the live display pointer.
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
        // OSD provider intentionally NOT torn down: VDR calls cOsdProvider::Shutdown()
        // globally during process exit, after all devices are stopped.
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

    // VDR calls Play() for genuine resume AND as a transition inside REW->PLAY->REW.
    // Request trick exit and let the decoder confirm: if TrickSpeed() follows within one
    // frame period the request is canceled; otherwise the decoder exits trick mode on the
    // next frame.
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

    // audioCodecId == NONE triggers detection; reset by SetPlayMode(pmNone),
    // HandleAudioTrackChange(), or Clear() (replay audi-N path).
    const AVCodecID currentCodec = audioCodecId.load(std::memory_order_relaxed);
    bool isLive = liveMode.load(std::memory_order_relaxed);

    if (currentCodec == AV_CODEC_ID_NONE) {
        // pmAudioOnly skips PlayVideo() entirely, so latch liveMode on first audio PES.
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

        // Always require 2-of-2 confirmation. SVDRP "audi N" resets audioCodecId on the main
        // thread while PlayTs may have already buffered a full PES from the OLD PID in
        // tsToPesAudio. That stale PES arrives here first and would detect the wrong codec.
        // Depth=2 suffices: tsToPesAudio.Reset() fires on every PUSI, so at most one stale
        // PES can sneak through. Cost: ~24-32 ms extra latency on codec open.
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

    // Radio detection: 3 s grace set by SetPlayMode(pmAudioVideo) with no video arriving;
    // paint black to clear the previous channel's residual scanout picture.
    if (radioBlackPending && radioBlackTimer.TimedOut()) [[unlikely]] {
        radioBlackPending = false;
        if (videoCodecId.load(std::memory_order_relaxed) == AV_CODEC_ID_NONE) {
            isyslog("vaapivideo/device: no video stream detected -- radio mode, showing black frame");
            SubmitBlackFrame();
        }
    }

    // Replay backpressure: cap queue to avoid tail-drops that create PTS gaps. Live: always accept.
    if (!isLive && audioProcessor->GetQueueSize() >= AUDIO_REPLAY_QUEUE_HIGHWATER) [[unlikely]] {
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

        // Wait for an SPS before opening H.264/HEVC: backend selection (8-bit vs High10/
        // Main10) keys on bit-depth/profile from the SPS. Opening without it would hit the
        // 8-bit table row and misclassify 10-bit streams. DVB carries SPS at least once per
        // GOP, so the wait is bounded.
        VideoStreamInfo streamInfo;
        streamInfo.codecId = detectedCodec;
        if (detectedCodec == AV_CODEC_ID_H264 || detectedCodec == AV_CODEC_ID_HEVC) {
            streamInfo = ::ProbeVideoSps(detectedCodec, {pes.payload, pes.payloadSize});
            if (!streamInfo.hasSps) [[unlikely]] {
                return Length;
            }
        }

        if (!decoder->OpenCodecWithInfo(streamInfo)) [[unlikely]] {
            esyslog("vaapivideo/device: failed to open video codec %s", avcodec_get_name(detectedCodec));
            return Length;
        }

        videoCodecId.store(detectedCodec, std::memory_order_relaxed);
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

[[nodiscard]] auto cVaapiDevice::Poll(cPoller & /*Poller*/, int TimeoutMs) -> bool {
    if (!decoder) [[unlikely]] {
        return true;
    }

    // cTransfer pushes via PlayVideo/PlayAudio directly and never calls Poll().
    if (liveMode.load(std::memory_order_relaxed)) {
        return true;
    }

    const int currentSpeed = trickSpeed.load(std::memory_order_relaxed);

    auto hasSpace = [&]() -> bool {
        // Trick: also gate on per-frame pacing timer to avoid burst-feeding past display rate.
        if (currentSpeed != 0) {
            return decoder->IsReadyForNextTrickFrame() && decoder->GetQueueSize() < DECODER_TRICK_QUEUE_DEPTH;
        }
        return !decoder->IsQueueFull() &&
               (!audioProcessor || audioProcessor->GetQueueSize() < AUDIO_REPLAY_QUEUE_HIGHWATER);
    };

    if (hasSpace()) {
        return true;
    }

    // Spin in-place: returning false causes VDR to retry through its outer Poll() loop,
    // which adds a full MainLoopInterval (~100 ms) per cycle and visibly slows startup/seek.
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

    // Clear stale Freeze() flag first -- would otherwise block PlayVideo() in the new mode.
    paused.store(false, std::memory_order_relaxed);

    switch (PlayMode) {
        case pmNone:
            // Capture previous codec ids BEFORE clearing: PlayVideo/PlayAudio 2-of-2 guards
            // use them to reject stale TS-buffer bytes after the switch.
            // Audio codec reset is deferred to Clear() -> ResetAudioCodecState().
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
            // User opt-in: paint black during channel-switch gap instead of holding the
            // previous channel's last frame until the new one decodes its first frame.
            if (vaapiConfig.clearOnChannelSwitch.load(std::memory_order_relaxed)) {
                SubmitBlackFrame();
            }
            break;
        case pmAudioOnly:
        case pmAudioOnlyBlack:
            // Radio: paint black so DRM scanout doesn't hold the previous channel's picture.
            radioBlackPending = false;
            Clear();
            SubmitBlackFrame();
            break;
        case pmAudioVideo:
        case pmVideoOnly:
            // Codec state already clean (VDR always emits pmNone before pmAudioVideo).
            // liveMode is latched on the first PES in PlayVideo/PlayAudio.
            Clear();
            // 3 s grace: if no video arrives, PlayAudio() paints black (radio channel).
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

    trickSpeed.store(Speed, std::memory_order_release);

    if (decoder) [[likely]] {
        decoder->SetTrickSpeed(Speed, Forward, isFast);
    }

    // Drop audio explicitly: VDR usually calls Mute() too but timing varies, and the
    // queued sink content (~100-200 ms) would be audible after the trick command.
    if (audioProcessor) [[likely]] {
        audioProcessor->Clear();
    }
}

// ============================================================================
// === PUBLIC API ===
// ============================================================================

[[nodiscard]] auto cVaapiDevice::Attach() -> bool {
    // Opens hardware using arguments latched by Initialize(). Used after detached startup
    // and after a Detach() -> SVDRP ATTA cycle.
    if (drmPath.empty()) [[unlikely]] {
        esyslog("vaapivideo/device: cannot attach - Initialize() has not run yet");
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
    // Idempotent: skip teardown if never attached, avoiding a spurious VT bounce.
    if (initState.load(std::memory_order_acquire) == 0) [[unlikely]] {
        dsyslog("vaapivideo/device: detach requested but hardware is not attached (no-op)");
        return true;
    }

    isyslog("vaapivideo/device: detaching from hardware");

    // Stop upstream sources before Stop(): cTransfer's tuner thread calls PlayVideo/PlayAudio
    // which dereference `decoder`. cControl::Shutdown() must precede Stop() because the
    // unwinding cPlayer::Detach() fires SetPlayMode(pmNone)->Clear(), which still needs the
    // subsystems alive. Note: ~cVaapiDevice does NOT need this; VDR's shutdown sequence
    // guarantees the correct order there.
    cControl::Shutdown();
    DetachAllReceivers();

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

    // Drop master before close so another DRM client can take it immediately.
    if (drmFd >= 0) {
        drmDropMaster(drmFd);
    }

    ReleaseHardware();
    const bool consoleRestored = RestoreConsole();

    // Reset all playback state so a subsequent Attach() re-detects codecs from scratch.
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
    return consoleRestored;
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

    if (deferred) {
        // Stay in state 0: hardware opens on first primary-device promotion after startup,
        // or on explicit SVDRP ATTA.
        isyslog("vaapivideo/device: starting detached - DRM '%s', audio '%s', connector '%s' "
                "(hardware init deferred until attach)",
                drmPath.c_str(), audioDevice.c_str(), connectorName.empty() ? "auto" : connectorName.c_str());
        return true;
    }

    dsyslog("vaapivideo/device: initializing - DRM '%s', audio '%s', connector '%s'", drmPath.c_str(),
            audioDevice.c_str(), connectorName.empty() ? "auto" : connectorName.c_str());
    return AttachHardware();
}

auto cVaapiDevice::MarkStartupComplete() noexcept -> void { startupComplete.store(true, std::memory_order_release); }

// ============================================================================
// === INTERNAL METHODS ===
// ============================================================================

[[nodiscard]] auto cVaapiDevice::AttachHardware() -> bool {
    // State: 0 (detached) -> 1 (in-progress) -> 2 (ready). All callers run on VDR's main
    // thread; the CAS rejects double-attach defensively rather than guarding a data race.
    int expected = 0;
    if (!initState.compare_exchange_strong(expected, 1)) [[unlikely]] {
        esyslog("vaapivideo/device: AttachHardware rejected -- already attached (state=%d)", expected);
        return false;
    }

    if (!OpenHardware()) [[unlikely]] {
        esyslog("vaapivideo/device: hardware initialization failed");
        initState.store(0, std::memory_order_relaxed);
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

    // Decoder starts codec-less; PlayVideo() opens the codec on the first PES.
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

    decoder->SetAudioProcessor(audioProcessor.get()); // A/V sync clock source

    // Pre-populate OSD cache: MakePrimaryDevice() (called next) may query GetOsdSize()
    // before any skin repaint triggers the lazy-init path.
    osdWidth = static_cast<int>(display->GetOutputWidth());
    osdHeight = static_cast<int>(display->GetOutputHeight());
    dsyslog("vaapivideo/device: pre-cached OSD size %dx%d", osdWidth, osdHeight);

    initState.store(2, std::memory_order_release);
    isyslog("vaapivideo/device: attached - DRM=%s audio=%s", drmPath.c_str(), audioDevice.c_str());

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
        // Same (type, PID): PMT churn or duplicate hook. Log for correlation, skip reset.
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
        // Ambiguous dolby: next-packet log identifies the actual codec.
        isyslog("vaapivideo/device: %s -> %s track switch (codec re-detect on next packet)", reason, kind);
    }

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

    // R+W required: resource query ioctls need R, KMS atomic modesetting needs W.
    // Missing group membership ('video' or 'render') is the common failure mode.
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

    // VAAPI requires the render node (/dev/dri/renderD*); drmGetDevice2 resolves the
    // matching render path for the card node we already have open.
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

    // Existence-check only: display re-queries planes during modeset. Fail fast here if
    // the kernel doesn't expose plane resources (atomic requires them).
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

        // Build the kernel-style name (e.g. "HDMI-A-1", "DP-2") and filter if requested.
        if (!connectorName.empty()) {
            const char *typeName = drmModeGetConnectorTypeName(connector->connector_type);
            const auto name = std::format("{}-{}", typeName ? typeName : "Unknown", connector->connector_type_id);
            if (name != connectorName) {
                dsyslog("vaapivideo/device: skipping connector %s (want %s)", name.c_str(), connectorName.c_str());
                continue;
            }
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
                // Use CRTC_ID atomic property, not legacy encoder_id walk: the legacy path
                // can return a stale CRTC on atomic drivers.
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
