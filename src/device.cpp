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
#include <memory>
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
#include <vdr/device.h>
#include <vdr/osd.h>
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

static std::atomic<bool> capWarned{false}, noVtHinted{false};

[[nodiscard]] static auto OwnVt() -> int {
    // TTY major=4, minor=N for /dev/ttyN (N=1..63); minor 0 is the current-VT alias.
    struct stat st{};
    if (fstat(STDIN_FILENO, &st) != 0 || !S_ISCHR(st.st_mode) || major(st.st_rdev) != 4) {
        return 0;
    }
    const auto vt = static_cast<int>(minor(st.st_rdev));
    return (vt >= 1 && vt <= 63) ? vt : 0;
}

[[nodiscard]] static auto SwitchToVt(int vt) -> bool {
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

[[nodiscard]] static auto ActivateOwnVt() -> bool {
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

[[nodiscard]] static auto LeaveOwnVt() -> bool {
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
    // call sites, so it must be done explicitly here.
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
    return Ready() && display ? display->NormalizeVideoRect(rect) : cRect::Null;
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
        dsyslog("vaapivideo/device: Clear() thread='%s' (first call)", threadName.data());
    } else {
        dsyslog("vaapivideo/device: Clear() thread='%s' delta=%llums", threadName.data(),
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
    // syslog gains a stream of vsync-rate underrun lines until the display's 10 s IDLE_THRESHOLD
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

// ----------------------------------------------------------------------------
// GrabImage helpers (anonymous namespace -- used only here)
// ----------------------------------------------------------------------------

namespace {

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
    if (!Ready() || !display) [[unlikely]] {
        esyslog("vaapivideo/device: GrabImage - device not ready");
        return nullptr;
    }

    // 1. Snapshot the displayed VAAPI surface to host memory (NV12 or P010).
    auto srcFrame = display->GrabDisplayedFrame();
    if (!srcFrame) [[unlikely]] {
        esyslog("vaapivideo/device: GrabImage - no displayed frame yet");
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

    // 4. Encode. SizeX/SizeY <= 0 = native display resolution.
    const int outW = (SizeX > 0) ? SizeX : rgb24->width;
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

[[nodiscard]] auto cVaapiDevice::HasDecoder() const -> bool { return decoder && decoder->IsReady(); }

[[nodiscard]] auto cVaapiDevice::HasIBPTrickSpeed() -> bool { return true; }

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

    // DropOutput silences the sink without nulling the playback clock; full Clear() here would
    // null GetClock() and cause a video freerun + display underrun for every OSD-mediated mute
    // (some skins toggle mute on menu open/close).
    if (audioProcessor) [[likely]] {
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
        if (detectedCodec == audioCodecCandidate.load(std::memory_order_relaxed)) {
            audioCodecCandidateCount.fetch_add(1, std::memory_order_relaxed);
        } else {
            audioCodecCandidate.store(detectedCodec, std::memory_order_relaxed);
            audioCodecCandidateCount.store(1, std::memory_order_relaxed);
        }
        const int candidateCount = audioCodecCandidateCount.load(std::memory_order_relaxed);
        if (candidateCount < 2) {
            dsyslog("vaapivideo/device: audio codec %s -- awaiting confirmation (%d/2)",
                    avcodec_get_name(detectedCodec), candidateCount);
            return Length;
        }

        audioCodecCandidate.store(AV_CODEC_ID_NONE, std::memory_order_relaxed);
        audioCodecCandidateCount.store(0, std::memory_order_relaxed);

        if (!audioProcessor->OpenCodec(detectedCodec, 48000, 2)) [[unlikely]] {
            esyslog("vaapivideo/device: failed to open audio codec %s", avcodec_get_name(detectedCodec));
            return Length;
        }

        audioCodecId.store(detectedCodec, std::memory_order_relaxed);
        isyslog("vaapivideo/device: audio codec %s confirmed (%s, %s)", avcodec_get_name(detectedCodec),
                isLive ? "live" : "replay", audioProcessor->IsPassthrough() ? "passthrough" : "PCM");

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

    radioBlackPending.store(false, std::memory_order_relaxed);

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

        videoCodecCandidate.store(AV_CODEC_ID_NONE, std::memory_order_relaxed);
        videoCodecCandidateCount.store(0, std::memory_order_relaxed);

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

auto cVaapiDevice::ScaleVideo(const cRect &rect) -> void {
    if (!display) [[unlikely]] {
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
        if (decoder) {
            decoder->RequestFilterRebuild();
        }
    }
    return clamped;
}

// Next cycle stop after `current`, skipping disabled (level 0) presets. Off (stop 0) is always a
// valid stop, so the loop always terminates; returns `current` only when every level is disabled.
static auto NextZoomStop(int current) -> int {
    for (int step = 1; step <= CONFIG_ZOOM_PRESET_COUNT + 1; ++step) {
        const int cand = (current + step) % (CONFIG_ZOOM_PRESET_COUNT + 1);
        if (cand == 0 || vaapiConfig.zoomLevel[cand - 1].load(std::memory_order_relaxed) > 0) {
            return cand;
        }
    }
    return current; // unreachable: Off is always a stop
}

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
        if (decoder) {
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
    if (decoder) {
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
    // decoder, jitterBuf grows to its 150-frame hard cap, and every subsequent frame overflows
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
            radioBlackPending.store(false, std::memory_order_relaxed);
            previousVideoCodec.store(videoCodecId.exchange(AV_CODEC_ID_NONE, std::memory_order_relaxed),
                                     std::memory_order_relaxed);
            liveMode.store(false, std::memory_order_relaxed);
            trickSpeed.store(0, std::memory_order_release);
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
            // User opt-in: paint black during channel-switch gap instead of holding the
            // previous channel's last frame until the new one decodes its first frame.
            if (vaapiConfig.clearOnChannelSwitch.load(std::memory_order_relaxed)) {
                SubmitBlackFrame();
            }
            break;
        case pmAudioOnly:
        case pmAudioOnlyBlack:
            // Radio: paint black so DRM scanout doesn't hold the previous channel's picture.
            radioBlackPending.store(false, std::memory_order_relaxed);
            ResetZoom(); // Audio-only is still a content change; keep transient-zoom semantics.
            Clear();
            SubmitBlackFrame();
            break;
        case pmAudioVideo:
        case pmVideoOnly:
            // Codec state already clean (VDR always emits pmNone before pmAudioVideo).
            // liveMode is latched on the first PES in PlayVideo/PlayAudio.
            ResetZoom(); // Before Clear(): belt-and-braces so a new stream starts at Off even if pmNone was skipped.
            Clear();
            // 3 s grace: if no video arrives, PlayAudio() paints black (radio channel).
            radioBlackTimer.Set(3000);
            radioBlackPending.store(true, std::memory_order_relaxed);
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

    // Re-entry guard: TS path re-enters this method as PES; only the outermost call manages state.
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
    ResetAudioCodecState();
    liveMode.store(false, std::memory_order_relaxed);
    trickSpeed.store(0, std::memory_order_relaxed);
    paused.store(false, std::memory_order_relaxed);
    lastHandledAudioTrack = ttNone;
    lastHandledAudioPid = 0;
    osdWidth = 0;
    osdHeight = 0;
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
// === MEDIAPLAYER FEED SURFACE ===
// ============================================================================

[[nodiscard]] auto cVaapiDevice::OpenForMediaPlayer(const VideoStreamInfo &video, const AudioStreamInfo &audio)
    -> bool {
    if (!Ready()) [[unlikely]] {
        esyslog("vaapivideo/device: OpenForMediaPlayer rejected -- hardware not attached");
        return false;
    }
    if (!decoder || !audioProcessor) [[unlikely]] {
        esyslog("vaapivideo/device: OpenForMediaPlayer rejected -- subsystems missing");
        return false;
    }

    // Mediaplayer paths are always replay -- no live-TV jitter buffer, no Transferring() race.
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

    if (audio.codecId != AV_CODEC_ID_NONE) {
        if (!audioProcessor->OpenCodecWithInfo(audio)) [[unlikely]] {
            esyslog("vaapivideo/device: OpenForMediaPlayer audio codec %s open failed",
                    avcodec_get_name(audio.codecId));
            return false;
        }
        audioCodecId.store(audio.codecId, std::memory_order_relaxed);
        decoder->NotifyAudioChange();
    } else {
        // Video-only entry: drop any audio codec / clock state left over from a previous entry
        // so the decoder enters freerun and does not try to lip-sync to the prior file's audio.
        audioCodecId.store(AV_CODEC_ID_NONE, std::memory_order_relaxed);
        audioProcessor->Clear();
        decoder->NotifyAudioChange();
    }

    isyslog("vaapivideo/device: mediaplayer open -- video=%s audio=%s", avcodec_get_name(video.codecId),
            audio.codecId == AV_CODEC_ID_NONE ? "none" : avcodec_get_name(audio.codecId));
    return true;
}

[[nodiscard]] auto cVaapiDevice::SubmitVideoPacket(const AVPacket *packet) -> bool {
    if (!Ready() || !decoder || decoder->IsQueueFull()) [[unlikely]] {
        return false;
    }
    decoder->EnqueuePacket(packet);
    return true;
}

[[nodiscard]] auto cVaapiDevice::SubmitAudioPacket(const AVPacket *packet) -> bool {
    if (!Ready() || !audioProcessor || !audioProcessor->IsInitialized()) [[unlikely]] {
        return false;
    }
    // Queue HIGHWATER is a pacing signal, not a hard failure. Returning false keeps the
    // mediaplayer's packetPending=true so this already-demuxed audio packet is retried after
    // the queue drains a slot; the shared demux cursor halts without dropping audio or
    // falsely advancing latestAudioPts90k via the post-submit update in cVaapiPlayer::Action.
    if (audioProcessor->GetQueueSize() >= MEDIAPLAYER_AUDIO_QUEUE_HIGHWATER) {
        return false;
    }
    return audioProcessor->EnqueuePacket(packet);
}

auto cVaapiDevice::ClearForMediaPlayer() -> void {
    // Used at entry open/close (where codec params may change). Drops in-flight packets,
    // tears down the filter chain, and lets the next frame rebuild it with new params.
    // Codec contexts stay open and play-mode is unchanged.
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
    const bool audioHighwater = audioOpen && audioDepth >= MEDIAPLAYER_AUDIO_QUEUE_HIGHWATER;
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
    // and sufficient gate; the 150-frame hard cap remains the ultimate overflow backstop.
    //
    // Audio-reanchor escape: pause -> Play() drains the audio queue but preserves jitterBuf
    // (decoder hold). On resume jitterFull would block the demuxer just when audio packets MUST
    // flow to re-anchor the clock -- deadlock, since the gate stays asserted until the clock
    // anchors and the clock can't anchor without audio. Yielding jitterFull while audio has room
    // below HIGHWATER lets audio packets through; ALSA writes; clock anchors; the gate becomes
    // moot. The audioHighwater gate above caps the co-pumped burst at ~10 audio packets (~250 ms).
    // Video-only streams (audioOpen=false) cannot anchor an audio clock, so the escape stays
    // disarmed and jitterFull bounds the decoder's pre-anchor depth.
    const bool clockAnchored = audioProcessor && audioProcessor->GetClock() != AV_NOPTS_VALUE;
    const bool audioCanReanchor = audioOpen && audioDepth < MEDIAPLAYER_AUDIO_QUEUE_HIGHWATER;
    const bool jitterFull = !clockAnchored && !audioCanReanchor && decoder &&
                            decoder->GetJitterBufSize() >= MEDIAPLAYER_JITTERBUF_BACKPRESSURE_FRAMES;
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
    ResetZoom(); // Fresh hardware (plugin start or SVDRP ATTA) always begins at Off.
    isyslog("vaapivideo/device: attached - DRM=%s audio=%s", drmPath.c_str(), audioDevice.c_str());

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
        esyslog("vaapivideo/device: failed to open '%s' - %s", drmPath.c_str(), std::strerror(errno));
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
    audioCodecCandidate.store(AV_CODEC_ID_NONE, std::memory_order_relaxed);
    audioCodecCandidateCount.store(0, std::memory_order_relaxed);
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

    // Acceptance check for one connector. Returns true if the connector is accepted and
    // crtcId/connectorId/activeMode have been latched. Pulled out of the loop so the same
    // logic serves the fast cached pass and the slow full-probe fallback below.
    // allowModeFallback=false (cached pass): accept only an exact (w,h,rate) match, so a
    // partial cached mode list doesn't lock in preferred/first before the full EDID probe
    // gets a chance to find the operator-selected mode.
    const auto tryAccept = [&](drmModeConnector *connector, bool allowModeFallback) -> bool {
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
                std::unique_ptr<drmModePropertyRes, FreeDrmProperty> prop{
                    drmModeGetProperty(drmFd, props->props[propIdx])};
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
        isyslog("vaapivideo/device: display %ux%u@%uHz (%s, connector %u, CRTC %u)", activeMode.hdisplay,
                activeMode.vdisplay, activeMode.vrefresh, name.c_str(), connectorId, crtcId);
        return true;
    };

    // Pass 1: cached state (no DDC). drmModeGetConnector() forces a fresh probe -- 100-500 ms
    // per disconnected port, seconds on a CEC-standby sink. On boxes with 6-8 connectors that
    // stacks up to the reported "60 s startup". Boot-probe / hot-plug cache suffices normally.
    for (int i = 0; i < resources->count_connectors; ++i) {
        const std::unique_ptr<drmModeConnector, FreeDrmConnector> connector{
            drmModeGetConnectorCurrent(drmFd, resources->connectors[i])};
        if (tryAccept(connector.get(), false)) {
            return true;
        }
    }

    // Pass 2: cold cache (DRM driver just loaded) or sink woke from standby after boot probe.
    // Full DDC probe -- same cost as the pre-fix single pass, only when pass 1 missed. When a
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
        if (tryAccept(connector.get(), true)) {
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
