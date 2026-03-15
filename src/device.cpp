// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file device.cpp
 * @brief VDR device integration, PES routing, and lifecycle
 */

#include "device.h"
#include "audio.h"
#include "common.h"
#include "config.h"
#include "decoder.h"
#include "osd.h"
#include "pes.h"

// POSIX
#include <fcntl.h>
#include <linux/vt.h>
#include <sys/ioctl.h>
#include <unistd.h>

// C++ Standard Library
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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
#include <libavutil/hwcontext_vaapi.h>
#include <libavutil/pixfmt.h>
}
#pragma GCC diagnostic pop

// VAAPI
#include <va/va.h>
#include <va/va_vpp.h>

// DRM
#include <libdrm/drm.h>
#include <libdrm/drm_mode.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

// VDR
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <vdr/device.h>
#include <vdr/thread.h>
#include <vdr/tools.h>
#pragma GCC diagnostic pop

// ============================================================================
// === HELPER FUNCTIONS ===
// ============================================================================

/// Restore the Linux console after releasing DRM hardware.
/// fbcon does not automatically reclaim the display when a DRM master drops
/// while other processes hold the device open (e.g. systemd-logind).
/// A VT round-trip forces the kernel to modeset fbcon's framebuffer.
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

    // VT_ACTIVATE to the already-active VT is a kernel no-op, so switch to a temporary VT first to force fbcon to
    // reprogram the CRTC.
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

/// Check whether a profile supports VLD decode with YUV420 output.
/// Validates both the VLD entrypoint and the VA_RT_FORMAT_YUV420 config attribute.
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
    deviceList.resize(64); // Hard upper bound; real systems have 1-4 DRM devices at most
    deviceCount = drmGetDevices2(0, deviceList.data(), static_cast<int>(deviceList.size()));

    if (deviceCount > 0) {
        deviceList.resize(static_cast<size_t>(deviceCount));
        dsyslog("vaapivideo/device: found %d DRM device(s)", deviceCount);
    } else {
        dsyslog("vaapivideo/device: no DRM devices found");
        deviceList.clear();
    }
    return HasDevices();
}

[[nodiscard]] auto DrmDevices::HasDevices() const noexcept -> bool { return deviceCount > 0; }

// ============================================================================
// === VAAPI DEVICE CLASS ===
// ============================================================================

auto cVaapiDevice::SubmitBlackFrame() -> void {
    const auto w = static_cast<int>(display->GetOutputWidth());
    const auto h = static_cast<int>(display->GetOutputHeight());

    // One-shot VAAPI hw_frames_ctx for a single NV12 surface
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

    // Allocate VAAPI surface and software staging frame
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

    // Fill with limited-range NV12 black (Y=16, UV=128) and upload to VAAPI surface
    const auto rows = static_cast<size_t>(h);
    std::memset(swFrame->data[0], 16, static_cast<size_t>(swFrame->linesize[0]) * rows);
    std::memset(swFrame->data[1], 128, static_cast<size_t>(swFrame->linesize[1]) * (rows / 2));
    if (av_hwframe_transfer_data(hwFrame.get(), swFrame.get(), 0) < 0) [[unlikely]] {
        esyslog("vaapivideo/device: black frame upload failed");
        return;
    }

    // Wrap in VaapiFrame and submit through the normal display pipeline
    auto frame = std::make_unique<VaapiFrame>();
    frame->avFrame = hwFrame.release();
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

    // Remove as primary device to prevent VDR calling into us.
    if (IsPrimaryDevice()) {
        cDevice::MakePrimaryDevice(false);
    }

    DetachAllReceivers();

    // Detach display from OSD provider before destroying display.
    if (::osdProvider) {
        if (auto *vaapiProvider = dynamic_cast<cVaapiOsdProvider *>(::osdProvider)) {
            vaapiProvider->DetachDisplay();
            dsyslog("vaapivideo/device: OSD provider detached from display");
        }
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
    return (initState.load(std::memory_order_relaxed) == 2) && decoder && decoder->IsReady();
}

auto cVaapiDevice::Clear() -> void {
    cDevice::Clear(); // resets VDR-level frame counters and internal demux bookkeeping

    // trickSpeed not reset -- Clear() is a buffer flush, not a mode change. Managed by TrickSpeed(), Play(),
    // SetPlayMode(pmNone).

    // Hold the display import lock across the decoder flush to prevent a race where an in-flight DRM frame export
    // overtakes the cleared queue and is re-presented stale.
    if (display) [[likely]] {
        display->BeginStreamSwitch();
    }

    if (decoder) [[likely]] {
        decoder->Clear(); // preserves the open codec context
    }

    if (display) [[likely]] {
        display->EndStreamSwitch();
    }

    if (audioProcessor) [[likely]] {
        audioProcessor->Clear();
    }

    // Audio tracking preserved -- reset in SetPlayMode(pmNone) and SetAudioTrackDevice().
}

[[nodiscard]] auto cVaapiDevice::DeviceType() const -> cString { return "VAAPI"; }

[[nodiscard]] auto cVaapiDevice::Flush(int TimeoutMs) -> bool {
    dsyslog("vaapivideo/device: Flush(%d)", TimeoutMs);

    if (!decoder) {
        return true;
    }

    // Wait for decoder queue to drain.
    const cTimeMs timeout(TimeoutMs);
    while (!decoder->IsQueueEmpty() && !timeout.TimedOut()) {
        cCondWait::SleepMs(10);
    }

    return decoder->IsQueueEmpty();
}

auto cVaapiDevice::Freeze() -> void {
    cDevice::Freeze();
    paused.store(true, std::memory_order_relaxed);

    // Drain queue for instant visual pause (at most one in-flight frame). Do not call ResetSync() -- preserves sync
    // across pause/resume.
    if (decoder) [[likely]] {
        decoder->DrainQueue();
    }

    // Drop ALSA buffers immediately so audio stops in sync with the visual freeze.
    if (audioProcessor) [[likely]] {
        audioProcessor->Clear();
    }
}

auto cVaapiDevice::GetOsdSize(int &Width, int &Height, double &PixelAspect) -> void {
    // OSD size must match display output (VDR allocates buffers based on this).

    if (!display) [[unlikely]] {
        // Detached or not yet initialized -- return config defaults without logging. VDR calls this every second; using
        // esyslog here would flood the log.
        Width = static_cast<int>(vaapiConfig.display.GetWidth());
        Height = static_cast<int>(vaapiConfig.display.GetHeight());
        PixelAspect = static_cast<double>(Width) / Height;
        return;
    }

    // Fast path: cached values.
    if (osdWidth > 0 && osdHeight > 0) [[likely]] {
        Width = osdWidth;
        Height = osdHeight;
        PixelAspect = display->GetAspectRatio();
        return;
    }

    // Slow path: populate cache.
    if (display->IsInitialized()) [[likely]] {
        osdWidth = static_cast<int>(display->GetOutputWidth());
        osdHeight = static_cast<int>(display->GetOutputHeight());
        Width = osdWidth;
        Height = osdHeight;
        PixelAspect = display->GetAspectRatio();
        dsyslog("vaapivideo/device: OSD size cached: %dx%d aspect=%.3f", Width, Height, PixelAspect);
    } else [[unlikely]] {
        // Fallback to config defaults.
        osdWidth = static_cast<int>(vaapiConfig.display.GetWidth());
        osdHeight = static_cast<int>(vaapiConfig.display.GetHeight());
        Width = osdWidth;
        Height = osdHeight;
        PixelAspect = static_cast<double>(vaapiConfig.display.GetWidth()) / vaapiConfig.display.GetHeight();
        dsyslog("vaapivideo/device: OSD size from config: %dx%d", Width, Height);
    }
}

[[nodiscard]] auto cVaapiDevice::GetSTC() -> int64_t {
    // STC (System Time Clock) is VDR's 90 kHz presentation clock. VDR reads it to detect playback stalls, drive
    // adaptive buffering, and align subtitles. We return the PTS of the last decoded frame as the best approximation.
    if (!decoder) [[unlikely]] {
        return -1;
    }

    const int64_t pts = decoder->GetLastPts();
    if (pts == AV_NOPTS_VALUE) [[unlikely]] {
        return -1;
    }

    return pts;
}

auto cVaapiDevice::GetVideoSize(int &Width, int &Height, double &VideoAspect) -> void {
    // Return 0,0 when no decoder is active -- skins detect the 0 -> real-size transition to trigger repaints.

    if (!HasDecoder()) [[unlikely]] {
        Width = Height = 0;
        VideoAspect = 1.0;
        return;
    }

    Width = decoder->GetStreamWidth();
    Height = decoder->GetStreamHeight();
    VideoAspect = decoder->GetStreamAspect();

    // Codec open but no frames decoded yet -- dimensions still unknown.
    if (Width == 0 || Height == 0) [[unlikely]] {
        VideoAspect = 1.0;
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
            // VDR owns the OSD provider lifecycle; we only create it once.
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
        // VDR calls cOsdProvider::Shutdown() to tear down all providers after our Stop().
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

    if (trickSpeed.load(std::memory_order_relaxed) != 0) {
        trickSpeed.store(0, std::memory_order_release);
        if (decoder) [[likely]] {
            decoder->SetTrickSpeed(0);
        }
    }

    paused.store(false, std::memory_order_relaxed);
}

[[nodiscard]] auto cVaapiDevice::PlayAudio(const uchar *Data, int Length, uchar Id) -> int {
    if (!Data || Length <= 0 || paused.load(std::memory_order_relaxed) ||
        trickSpeed.load(std::memory_order_relaxed) != 0) [[unlikely]] {
        return Length;
    }

    if (!audioProcessor || !audioProcessor->IsInitialized()) [[unlikely]] {
        return Length;
    }

    // Stream-type detection is deferred to here because cTransferControl -- which determines Transferring() -- may not
    // exist yet when SetPlayMode() is called.
    const uchar lastId = prevAudioStreamId.load(std::memory_order_relaxed);
    if (lastId != Id) [[unlikely]] {
        if (lastId != 0xFF) {
            dsyslog("vaapivideo/device: audio stream ID change 0x%02X -> 0x%02X", lastId, Id);
        }
        // Different stream ID means a new elementary stream (e.g. after a channel change that races with Transfer Mode
        // startup). Force full codec re-detection.
        audioCodecId.store(AV_CODEC_ID_NONE, std::memory_order_relaxed);
        codecHysteresis = AV_CODEC_ID_NONE;
        codecHysteresisCount = 0;
    }
    prevAudioStreamId.store(Id, std::memory_order_relaxed);

    const auto pes = ParsePes({Data, static_cast<size_t>(Length)});
    if (!pes.isAudio || pes.payloadSize == 0) [[unlikely]] {
        return Length;
    }

    const AVCodecID currentCodec = audioCodecId.load(std::memory_order_relaxed);

    if (currentCodec == AV_CODEC_ID_NONE) {
        // Detect live mode (covers pmAudioOnly where no video arrives).
        if (!liveMode) {
            liveMode = Transferring();
        }

        const AVCodecID detectedCodec = ::DetectAudioCodec({pes.payload, pes.payloadSize});
        if (detectedCodec == AV_CODEC_ID_NONE) [[unlikely]] {
            return Length; // Not enough data yet
        }

        // Open codec with default parameters (48kHz stereo)
        if (!audioProcessor->OpenCodec(detectedCodec, 48000, 2)) [[unlikely]] {
            esyslog("vaapivideo/device: failed to open audio codec %s", avcodec_get_name(detectedCodec));
            return Length;
        }

        audioCodecId.store(detectedCodec, std::memory_order_relaxed);
        // Reset A/V sync -- audio clock is invalid after codec change.
        if (decoder) {
            decoder->NotifyAudioChange();
        }
        isyslog("vaapivideo/device: audio codec %s (%s)", avcodec_get_name(detectedCodec),
                liveMode ? "live" : "replay");
    } else {
        // Mid-stream codec change with hysteresis (3 consecutive detections)
        const AVCodecID detectedCodec = ::DetectAudioCodec({pes.payload, pes.payloadSize});
        if (detectedCodec != AV_CODEC_ID_NONE && detectedCodec != currentCodec) {
            if (detectedCodec == codecHysteresis) {
                // Require 3 consecutive detections before committing to a codec switch. A single malformed PES packet
                // could produce a spurious detection; the hysteresis prevents unnecessary OpenCodec() teardown cycles.
                if (++codecHysteresisCount >= 3) {
                    isyslog("vaapivideo/device: audio codec change %s -> %s", avcodec_get_name(currentCodec),
                            avcodec_get_name(detectedCodec));

                    if (audioProcessor->OpenCodec(detectedCodec, 48000, 2)) {
                        audioCodecId.store(detectedCodec, std::memory_order_relaxed);
                        if (decoder) {
                            decoder->NotifyAudioChange();
                        }
                    } else {
                        esyslog("vaapivideo/device: failed to switch to audio codec %s",
                                avcodec_get_name(detectedCodec));
                        return Length;
                    }
                    codecHysteresis = AV_CODEC_ID_NONE;
                    codecHysteresisCount = 0;
                }
            } else {
                codecHysteresis = detectedCodec;
                codecHysteresisCount = 1;
            }
        } else if (detectedCodec == currentCodec && codecHysteresisCount > 0) {
            // Confirmed back to original codec -- cancel the pending switch.
            codecHysteresis = AV_CODEC_ID_NONE;
            codecHysteresisCount = 0;
        }
    }

    // Replay flow control (same as PlayVideo()): return 0 so VDR calls Poll() and retries. In live TV mode the queue is
    // always accepted to keep pace with cTransfer.
    if (!liveMode && audioProcessor->IsQueueFull()) [[unlikely]] {
        return 0;
    }

    // Audio plays independently -- parser handles frame extraction and PTS tracking.
    audioProcessor->Decode(pes.payload, pes.payloadSize, pes.pts);
    return Length;
}

[[nodiscard]] auto cVaapiDevice::PlayVideo(const uchar *Data, int Length) -> int {
    if (!Data || Length <= 0) [[unlikely]] {
        return Length;
    }

    // Block while paused (except trick mode)
    if (paused.load(std::memory_order_relaxed) && trickSpeed.load(std::memory_order_relaxed) == 0) [[unlikely]] {
        return Length;
    }

    if (!decoder || !decoder->IsReady()) [[unlikely]] {
        return Length;
    }

    const auto pes = ParsePes({Data, static_cast<size_t>(Length)});
    if (!pes.isVideo || pes.payloadSize == 0) [[unlikely]] {
        return Length;
    }

    const AVCodecID currentCodec = videoCodecId.load(std::memory_order_relaxed);
    AVCodecID detectedCodec = AV_CODEC_ID_NONE;

    if (currentCodec == AV_CODEC_ID_NONE) [[unlikely]] {
        // Detect live vs replay (cTransferControl may not exist at SetPlayMode() time).
        liveMode = Transferring();

        detectedCodec = ::DetectVideoCodec({pes.payload, pes.payloadSize});
        if (detectedCodec == AV_CODEC_ID_NONE) [[unlikely]] {
            return Length; // No strong marker found -- wait for next packet
        }

        // Guard against stale PES data from the previous channel lingering in VDR's transfer buffer after a channel
        // switch. When the detected codec differs from the previously active codec, the data must belong to the new
        // channel and is accepted immediately. When it matches the previous codec, stale buffer data is possible:
        // require two consecutive detections to confirm. For same-codec channel switches the extra confirmation adds
        // one GOP period (~1-2 s) but the codec type is correct regardless, so no harm is done.
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
        isyslog("vaapivideo/device: video codec %s (%s)", avcodec_get_name(detectedCodec),
                liveMode ? "live" : "replay");
    }
    // Mid-stream codec changes (e.g. splice from MPEG-2 to H.264 broadcast) start a new SetPlayMode() cycle. Codec ID
    // is reset in pmNone so re-detection fires.

    // Replay flow control: return 0 -> VDR calls Poll() and retries. Live TV: never block (cTransfer has only 100 ms
    // retry budget). Trick mode: single-packet queue (DECODER_TRICK_QUEUE_DEPTH=1).
    if (!liveMode) [[unlikely]] {
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

    // Live TV: cTransfer calls PlayVideo/PlayAudio directly and never calls Poll.
    if (liveMode) {
        return true;
    }

    const int currentSpeed = trickSpeed.load(std::memory_order_relaxed);

    auto hasSpace = [&]() -> bool {
        // Trick mode: gate on both the time budget and the single-slot queue.
        if (currentSpeed != 0) {
            return decoder->IsReadyForNextTrickFrame() && decoder->GetQueueSize() < DECODER_TRICK_QUEUE_DEPTH;
        }
        return !decoder->IsQueueFull() && (!audioProcessor || !audioProcessor->IsQueueFull());
    };

    if (hasSpace()) {
        return true;
    }

    // Spin-wait up to TimeoutMs to avoid a round-trip through VDR's Poll retry loop.
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

auto cVaapiDevice::SetAudioTrackDevice(eTrackType Type) -> void {
    // VDR calls this when user switches audio track (green button / Audio menu). Set sentinel to trigger codec reset
    // when first packet of new stream arrives in PlayAudio().

    const tTrackId *track = GetTrack(Type);
    dsyslog("vaapivideo/device: SetAudioTrackDevice(%s %d%s%s)",
            IS_AUDIO_TRACK(Type)   ? "audio"
            : IS_DOLBY_TRACK(Type) ? "dolby"
                                   : "unknown",
            static_cast<int>(Type), track ? ", lang=" : "", track ? track->language : "");

    // Resetting to 0xFF forces PlayAudio() to treat the very next packet as a stream-ID change and drop any accumulated
    // hysteresis state for the previous track.
    prevAudioStreamId.store(0xFF, std::memory_order_relaxed);

    // Flush now so the tail of the old track is never rendered during the switch.
    if (audioProcessor) [[likely]] {
        audioProcessor->Clear();
    }
}

[[nodiscard]] auto cVaapiDevice::SetPlayMode(ePlayMode PlayMode) -> bool {
    static constexpr const char *kModeNames[] = {
        "pmNone", "pmAudioVideo", "pmAudioOnly", "pmAudioOnlyBlack", "pmVideoOnly", "pmExtern",
    };
    const auto idx = static_cast<unsigned>(PlayMode);
    dsyslog("vaapivideo/device: SetPlayMode(%s) called", idx < std::size(kModeNames) ? kModeNames[idx] : "unknown");

    // Reset paused -- prevents stale Freeze() state from blocking next session.
    paused.store(false, std::memory_order_relaxed);

    switch (PlayMode) {
        case pmNone:
            // Hard reset of all codec state. Doing this here rather than in Clear() allows skip/seek within a
            // recording to reuse the open codec without a teardown cycle.
            previousVideoCodec = videoCodecId.load(std::memory_order_relaxed);
            videoCodecId.store(AV_CODEC_ID_NONE, std::memory_order_relaxed);
            audioCodecId.store(AV_CODEC_ID_NONE, std::memory_order_relaxed);
            liveMode = false;
            trickSpeed.store(0, std::memory_order_release);
            if (decoder) [[likely]] {
                decoder->SetTrickSpeed(0);
                decoder->RequestCodecReopen();
            }
            prevAudioStreamId.store(0xFF, std::memory_order_relaxed);
            codecHysteresis = AV_CODEC_ID_NONE;
            codecHysteresisCount = 0;
            videoCodecCandidate = AV_CODEC_ID_NONE;
            videoCodecCandidateCount = 0;
            Clear();
            // Submit a black VAAPI frame so the previous channel's last picture does not stay on screen.
            // Uses the normal SubmitFrame() path so the display thread and OSD keep running.
            if (display && vaapi.hwDeviceRef) {
                SubmitBlackFrame();
            }
            break;
        case pmAudioVideo:
        case pmAudioOnly:
        case pmAudioOnlyBlack:
        case pmVideoOnly:
            // Codec state was already reset by the preceding pmNone call. liveMode is determined from the first
            // incoming PES packet via Transferring().
            Clear();
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

    dsyslog("vaapivideo/device: StillPicture(%d bytes)", Length);

    // Input: TS (0x47) or PES (0x00). Base class converts TS -> PES.
    if (Data[0] == 0x47) {
        dsyslog("vaapivideo/device: StillPicture received TS packet, waiting for PES");
        cDevice::StillPicture(Data, Length);
        return;
    }

    // exchange() atomically clears paused so PlayVideo() will accept the packet. Storing the original value lets us
    // restore frozen state after submission.
    const bool wasPaused = paused.exchange(false, std::memory_order_relaxed);
    PlayVideo(Data, Length);
    if (wasPaused) {
        paused.store(true, std::memory_order_relaxed);
    }
}

auto cVaapiDevice::TrickSpeed(int Speed, bool Forward) -> void {
    dsyslog("vaapivideo/device: TrickSpeed(%d, %s)", Speed, Forward ? "forward" : "backward");

    // VDR always calls Freeze() before slow trick modes (setting paused=true), but fast forward/rewind arrives directly
    // without a preceding Freeze(). Checking paused here lets SetTrickSpeed() distinguish the two cases.
    const bool isFast = !paused.load(std::memory_order_relaxed);

    trickSpeed.store(Speed, std::memory_order_release);

    if (decoder) [[likely]] {
        decoder->SetTrickSpeed(Speed, Forward, isFast);
    }

    // Mute audio during trick modes (VDR may also call Mute() separately).
    if (audioProcessor) [[likely]] {
        audioProcessor->Clear();
    }
}

// ============================================================================
// === PUBLIC API ===
// ============================================================================

[[nodiscard]] auto cVaapiDevice::Attach() -> bool {
    // Re-open DRM/VAAPI hardware and restart all subsystem threads. Uses the drmPath and audioDevice stored from the
    // last Initialize() call.
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
    // Full hardware detach: stop all threads then release the DRM fd and VAAPI context so that other applications (e.g.
    // a compositor) can take over the display. initState is reset to 0 so that a subsequent Attach() call can re-run
    // Initialize().
    isyslog("vaapivideo/device: detaching from hardware");

    // Sever the OSD provider's display reference before destroying the display.
    if (auto *provider = dynamic_cast<cVaapiOsdProvider *>(::osdProvider)) {
        provider->DetachDisplay();
        dsyslog("vaapivideo/device: OSD provider detached from display");
    }

    Stop();

    // Drop DRM master before closing the fd so the kernel can hand master status to another DRM client (compositor,
    // console, etc.) immediately.
    if (drmFd >= 0) {
        drmDropMaster(drmFd);
    }

    ReleaseHardware();

    // Force the kernel console to redraw after releasing the DRM device.
    RestoreConsole();

    osdWidth = 0;
    osdHeight = 0;
    initState.store(0, std::memory_order_release);
    isyslog("vaapivideo/device: detached");
}

[[nodiscard]] auto cVaapiDevice::Initialize(std::string_view drmDevicePath, std::string_view audioDevicePath) -> bool {
    // initState transitions: 0 (uninit) -> 1 (init in progress) -> 2 (ready). compare_exchange atomically claims slot
    // 1; a racing second call sees a non-zero state and bails out without touching any shared resources.
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
        initState.store(0);
        return false;
    }

    // Audio must start before display so the decoder already has a clock reference for A/V sync.
    audioProcessor = std::make_unique<cAudioProcessor>();
    if (!audioProcessor->Initialize(audioDevice)) [[unlikely]] {
        esyslog("vaapivideo/device: audio initialization failed");
        audioProcessor.reset();
        ReleaseHardware();
        initState.store(0);
        return false;
    }

    display = std::make_unique<cVaapiDisplay>();
    if (!display->Initialize(drmFd, vaapi.hwDeviceRef, crtcId, connectorId, activeMode)) [[unlikely]] {
        esyslog("vaapivideo/device: display initialization failed");
        display.reset();
        audioProcessor->Stop();
        audioProcessor.reset();
        ReleaseHardware();
        initState.store(0);
        return false;
    }

    // Decoder starts without an open codec; OpenCodec() is called on the first video PES packet.
    decoder = std::make_unique<cVaapiDecoder>(display.get(), &vaapi);
    if (!decoder->Initialize()) [[unlikely]] {
        esyslog("vaapivideo/device: decoder initialization failed");
        decoder.reset();
        display->Shutdown();
        display.reset();
        audioProcessor->Stop();
        audioProcessor.reset();
        ReleaseHardware();
        initState.store(0);
        return false;
    }

    // Give decoder access to audio for A/V sync.
    decoder->SetAudioProcessor(audioProcessor.get());

    // Pre-cache OSD size before MakePrimaryDevice() calls GetOsdSize().
    osdWidth = static_cast<int>(display->GetOutputWidth());
    osdHeight = static_cast<int>(display->GetOutputHeight());
    dsyslog("vaapivideo/device: pre-cached OSD size %dx%d", osdWidth, osdHeight);

    initState.store(2, std::memory_order_release);
    isyslog("vaapivideo/device: initialized - DRM=%s audio=%s", drmPath.c_str(), audioDevice.c_str());

    // Show black immediately so the console is covered even if VDR starts on a radio channel.
    SubmitBlackFrame();

    return true;
}

auto cVaapiDevice::Stop() -> void {
    // Shutdown order is strictly: display -> decoder -> audio. display holds exported DRM frames that reference
    // decoder's hw_frames_ctx, so display must stop before the decoder can be safely torn down. The decoder holds a
    // pointer to audioProcessor for A/V sync callbacks, so audioProcessor must outlive the decoder.
    if (display) [[likely]] {
        display->Shutdown();
        display.reset();
    }

    if (decoder) [[likely]] {
        decoder->Shutdown();
        decoder.reset();
    }

    if (audioProcessor) [[likely]] {
        audioProcessor->Stop();
        audioProcessor.reset();
    }
}

// ============================================================================
// === INTERNAL METHODS ===
// ============================================================================

[[nodiscard]] auto cVaapiDevice::OpenHardware() -> bool {
    if (drmPath.empty()) [[unlikely]] {
        esyslog("vaapivideo/device: no DRM device specified");
        return false;
    }

    // Both read and write access are required: read for resource queries (drmModeGetResources), write for KMS
    // mode-setting ioctls (DRM_IOCTL_MODE_*).
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

    // VAAPI must be opened on the render node (/dev/dri/renderD*), not the primary node. drmGetDevice2() takes the open
    // primary fd and fills in a drmDevice descriptor that also carries the render node path, so we never need to parse
    // sysfs directly.
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

    if (!ProbeVppCapabilities()) [[unlikely]] {
        esyslog("vaapivideo/device: VPP unavailable -- GPU is not suitable for this plugin");
        av_buffer_unref(&vaapi.hwDeviceRef);
        close(drmFd);
        drmFd = -1;
        return false;
    }

    return true;
}

[[nodiscard]] auto cVaapiDevice::ProbeVppCapabilities() -> bool {
    // Query VAAPI Video Processing Pipeline capabilities once at device init. The results are cached in vaapi and used
    // by InitFilterGraph() on every stream change to build a filter chain that only contains supported filters. Returns
    // false when VAEntrypointVideoProc is unavailable (fatal).
    vaapi.hasDenoise = false;
    vaapi.hasSharpness = false;
    vaapi.hwH264 = false;
    vaapi.hwHevc = false;
    vaapi.hwMpeg2 = false;
    vaapi.deinterlaceMode.clear();

    if (!vaapi.hwDeviceRef) [[unlikely]] {
        esyslog("vaapivideo/device: VPP probe skipped -- no VAAPI device");
        return false;
    }

    // Extract VADisplay from FFmpeg's hardware device context.
    const auto *hwDeviceCtx =
        reinterpret_cast<const AVHWDeviceContext *>( // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
            vaapi.hwDeviceRef->data);
    const auto *vaapiDevCtx = static_cast<const AVVAAPIDeviceContext *>(hwDeviceCtx->hwctx);
    const VADisplay vaDisplay = vaapiDevCtx->display; // NOLINT(misc-misplaced-const)

    // Log the VA-API driver identity so the user can verify which backend is active (e.g. "Mesa Gallium" for AMD
    // radeonsi).
    const char *vendorStr = vaQueryVendorString(vaDisplay);
    isyslog("vaapivideo/device: VA-API driver -- %s", vendorStr ? vendorStr : "(unknown)");

    // --- Probe hardware decode profiles ---
    // Check which broadcast codecs have VLD (decode) entrypoints with YUV420 output on this GPU. OpenCodec() uses
    // these flags to decide between HW and SW decode paths. A profile listed without VLD is encode-only.
    const int maxProfiles = vaMaxNumProfiles(vaDisplay);
    if (maxProfiles <= 0) [[unlikely]] {
        esyslog("vaapivideo/device: vaMaxNumProfiles failed");
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

    // Create a minimal VPP config + context needed for the query API.
    VAConfigID configId = VA_INVALID_ID;
    if (vaCreateConfig(vaDisplay, VAProfileNone, VAEntrypointVideoProc, nullptr, 0, &configId) != VA_STATUS_SUCCESS)
        [[unlikely]] {
        esyslog("vaapivideo/device: VAEntrypointVideoProc unavailable -- cannot initialize");
        return false;
    }

    // A small dummy surface satisfies the vaCreateContext signature requirement.
    VASurfaceID surface = VA_INVALID_SURFACE;
    if (vaCreateSurfaces(vaDisplay, VA_RT_FORMAT_YUV420, 64, 64, &surface, 1, nullptr, 0) != VA_STATUS_SUCCESS)
        [[unlikely]] {
        vaDestroyConfig(vaDisplay, configId);
        esyslog("vaapivideo/device: VPP probe failed -- vaCreateSurfaces error");
        return false;
    }

    VAContextID contextId = VA_INVALID_ID;
    if (vaCreateContext(vaDisplay, configId, 64, 64, 0, &surface, 1, &contextId) != VA_STATUS_SUCCESS) [[unlikely]] {
        vaDestroySurfaces(vaDisplay, &surface, 1);
        vaDestroyConfig(vaDisplay, configId);
        esyslog("vaapivideo/device: VPP probe failed -- vaCreateContext error");
        return false;
    }

    // --- Query supported filter types ---
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

    // --- Query deinterlacing capabilities ---
    // Walk supported algorithms from best to worst; keep the first (best) match.
    VAProcFilterCapDeinterlacing deintCaps[VAProcDeinterlacingCount];
    auto numDeintCaps = static_cast<unsigned int>(VAProcDeinterlacingCount);
    if (vaQueryVideoProcFilterCaps(vaDisplay, contextId, VAProcFilterDeinterlacing, deintCaps, &numDeintCaps) ==
        VA_STATUS_SUCCESS) {
        // Preference order: motion_compensated > motion_adaptive > weave > bob. Map each type to a priority; higher is
        // better.
        static constexpr struct {
            VAProcDeinterlacingType type;
            int priority;
            const char *name;
        } kDeintModes[] = {
            {.type = VAProcDeinterlacingMotionCompensated, .priority = 4, .name = "motion_compensated"},
            {.type = VAProcDeinterlacingMotionAdaptive, .priority = 3, .name = "motion_adaptive"},
            {.type = VAProcDeinterlacingWeave, .priority = 2, .name = "weave"},
            {.type = VAProcDeinterlacingBob, .priority = 1, .name = "bob"},
        };

        int bestPriority = 0;
        for (unsigned int i = 0; i < numDeintCaps; ++i) {
            for (const auto &[type, priority, name] : kDeintModes) {
                if (deintCaps[i].type == type && priority > bestPriority) {
                    bestPriority = priority;
                    vaapi.deinterlaceMode = name;
                }
            }
        }
    }

    // --- Cleanup temporary VAAPI resources ---
    vaDestroyContext(vaDisplay, contextId);
    vaDestroySurfaces(vaDisplay, &surface, 1);
    vaDestroyConfig(vaDisplay, configId);

    isyslog("vaapivideo/device: VPP capabilities -- denoise=%s sharpen=%s deinterlace=%s",
            vaapi.hasDenoise ? "yes" : "no", vaapi.hasSharpness ? "yes" : "no",
            vaapi.deinterlaceMode.empty() ? "none" : vaapi.deinterlaceMode.c_str());
    return true;
}

auto cVaapiDevice::ReleaseHardware() -> void {
    if (vaapi.hwDeviceRef) {
        av_buffer_unref(&vaapi.hwDeviceRef);
        vaapi.hwDeviceRef = nullptr;
        vaapi.drmFd = -1; // Clear borrowed reference only -- actual fd is closed below.
    }

    if (drmFd >= 0) {
        close(drmFd);
        drmFd = -1;
    }
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

    const std::unique_ptr<drmModePlaneRes, FreeDrmPlaneResources> planeRes{drmModeGetPlaneResources(drmFd)};
    if (!planeRes) [[unlikely]] {
        esyslog("vaapivideo/device: failed to get DRM plane resources (atomic required)");
        return false;
    }

    const auto targetWidth = vaapiConfig.display.GetWidth();
    const auto targetHeight = vaapiConfig.display.GetHeight();
    const auto targetRate = vaapiConfig.display.GetRefreshRate();

    // Mode selection priority: (1) exact config match, (2) driver-preferred flag, (3) first listed. The first connector
    // that yields a CRTC ID wins; remaining connectors are ignored.

    for (int i = 0; i < resources->count_connectors; ++i) {
        std::unique_ptr<drmModeConnector, FreeDrmConnector> connector{
            drmModeGetConnector(drmFd, resources->connectors[i])};
        if (!connector || connector->connection != DRM_MODE_CONNECTED || connector->count_modes == 0) [[unlikely]] {
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
                // Using the CRTC_ID connector property is more reliable than walking the encoder chain: it reflects the
                // currently active CRTC and is always present on atomic-capable drivers.
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
