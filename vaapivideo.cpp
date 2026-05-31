// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file vaapivideo.cpp
 * @brief VDR plugin entry point: argument parsing, device lifecycle, and setup-menu registration.
 *
 * Plugin lifecycle (VDR call order):
 *   1. Constructor    -- no hardware touched
 *   2. ProcessArgs()  -- args stored; hardware untouched
 *   3. Initialize()   -- cVaapiDevice created; hardware opened unless --detached
 *   4. Start()        -- attached: confirm primary; detached: release attach gate
 *   5. Housekeeping() -- periodic VDR main-loop tick (empty)
 *   6. Stop()         -- plugin cleanup; VDR still owns cVaapiDevice
 *   7. Destructor     -- trivial; VDR destroys cVaapiDevice via cDevice::Shutdown()
 */

#include "src/common.h"
#include "src/config.h"
#include "src/device.h"
#include "src/mediaplayer.h"
#include "src/osd.h"

// POSIX
#include <getopt.h> // NOLINT(misc-include-cleaner) -- clang-tidy cannot map getopt_long/option/optind/optarg back to this header
#include <strings.h>
#include <unistd.h>

// DRM
#include <xf86drm.h>

// C++ Standard Library
#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cstring>
#include <format>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

// FFmpeg
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
extern "C" {
#include <libavutil/log.h>
}
#pragma GCC diagnostic pop

// VDR
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <vdr/channels.h>
#include <vdr/device.h>
#include <vdr/i18n.h>
#include <vdr/keys.h>
#include <vdr/menuitems.h>
#include <vdr/osdbase.h>
#include <vdr/plugin.h>
#include <vdr/skins.h>
#include <vdr/tools.h>
#pragma GCC diagnostic pop

namespace {

// ============================================================================
// === cMenuSetupVaapi ===
// ============================================================================

/// VDR setup page. All edits are staged in local int copies and written back to
/// vaapiConfig atomics only when the user confirms via Store().
class cMenuSetupVaapi : public cMenuSetupPage {
  public:
    cMenuSetupVaapi()
        : editClearOnChannelSwitch(vaapiConfig.clearOnChannelSwitch.load(std::memory_order_relaxed) ? 1 : 0),
          editHdrMode(
              std::clamp(static_cast<int>(vaapiConfig.hdrMode.load(std::memory_order_relaxed)), 0, kHdrModeCount - 1)),
          editPassthroughLatency(vaapiConfig.passthroughLatency.load(std::memory_order_relaxed)),
          // Clamp so cMenuEditStraItem never indexes past kPassthroughModeLabels.
          editPassthroughMode(std::clamp(static_cast<int>(vaapiConfig.passthroughMode.load(std::memory_order_relaxed)),
                                         0, kPassthroughModeCount - 1)),
          editPcmLatency(vaapiConfig.pcmLatency.load(std::memory_order_relaxed)) {
        SetSection(tr("VAAPI Video"));
        // Separate A/V offsets for PCM and IEC61937 passthrough paths; bounds from config.h
        // keep the menu and setup.conf parser in sync.
        Add(new cMenuEditIntItem(tr("PCM Audio Latency (ms)"), &editPcmLatency, CONFIG_AUDIO_LATENCY_MIN_MS,
                                 CONFIG_AUDIO_LATENCY_MAX_MS));
        Add(new cMenuEditIntItem(tr("Passthrough Audio Latency (ms)"), &editPassthroughLatency,
                                 CONFIG_AUDIO_LATENCY_MIN_MS, CONFIG_AUDIO_LATENCY_MAX_MS));
        Add(new cMenuEditStraItem(tr("Audio Passthrough"), &editPassthroughMode, kPassthroughModeCount,
                                  kPassthroughModeLabels.data()));
        Add(new cMenuEditStraItem(tr("HDR Passthrough"), &editHdrMode, kHdrModeCount, kHdrModeLabels.data()));
        // Use "off"/"on" labels to match the string-select items above; default is "no"/"yes".
        Add(new cMenuEditBoolItem(tr("Clear display on channel switch"), &editClearOnChannelSwitch, tr("off"),
                                  tr("on")));
        // Manual zoom levels: each is a zoom-in factor in tenths-of-% (344 == +34.4% enlargement),
        // applied as an equal crop on all sides; 0 disables the level (skipped while cycling). The
        // active stop is transient and not edited here. cMenuEditItem strdup()s its label, so the
        // temporary cString is safe.
        for (int i = 0; i < CONFIG_ZOOM_PRESET_COUNT; ++i) {
            editZoomLevel[i] = vaapiConfig.zoomLevel[i].load(std::memory_order_relaxed);
            Add(new cMenuEditIntItem(*cString::sprintf(tr("Zoom level %d (0.1%% larger, 0=off)"), i + 1),
                                     &editZoomLevel[i], CONFIG_ZOOM_LEVEL_MIN, CONFIG_ZOOM_LEVEL_MAX));
        }
    }

  protected:
    auto Store() -> void override {
        vaapiConfig.pcmLatency.store(editPcmLatency, std::memory_order_relaxed);
        vaapiConfig.passthroughLatency.store(editPassthroughLatency, std::memory_order_relaxed);
        vaapiConfig.passthroughMode.store(static_cast<PassthroughMode>(editPassthroughMode), std::memory_order_relaxed);
        vaapiConfig.hdrMode.store(static_cast<HdrMode>(editHdrMode), std::memory_order_relaxed);
        vaapiConfig.clearOnChannelSwitch.store(editClearOnChannelSwitch != 0, std::memory_order_relaxed);
        SetupStore("PcmLatency", editPcmLatency);
        SetupStore("PassthroughLatency", editPassthroughLatency);
        SetupStore("PassthroughMode", editPassthroughMode);
        SetupStore("HdrMode", editHdrMode);
        SetupStore("ClearOnChannelSwitch", editClearOnChannelSwitch);
        // Decide whether the *live* level actually changed BEFORE overwriting the atomics -- only
        // then is a filter rebuild warranted (a bare setup OK must not glitch the picture).
        const int activeZoom = vaapiConfig.zoomActive.load(std::memory_order_relaxed);
        bool activePresetChanged = false;
        if (activeZoom >= 1 && activeZoom <= CONFIG_ZOOM_PRESET_COUNT) {
            activePresetChanged =
                editZoomLevel[activeZoom - 1] != vaapiConfig.zoomLevel[activeZoom - 1].load(std::memory_order_relaxed);
        }
        // Persist only the level definitions; zoomActive is intentionally not stored (transient).
        for (int i = 0; i < CONFIG_ZOOM_PRESET_COUNT; ++i) {
            vaapiConfig.zoomLevel[i].store(editZoomLevel[i], std::memory_order_relaxed);
            SetupStore(*cString::sprintf("Zoom%d", i + 1), editZoomLevel[i]);
        }
        // Re-apply only when the live preset's level changed, so the picture updates now instead of
        // waiting for the next zoom/content event. If the live preset was just disabled (set to 0),
        // turn zoom off rather than refresh a no-op crop. (This page is a separate class from the
        // plugin, so reach the device via the primary-device registry.)
        if (activePresetChanged) {
            if (auto *device = dynamic_cast<cVaapiDevice *>(cDevice::PrimaryDevice()); device != nullptr) {
                if (editZoomLevel[activeZoom - 1] <= 0) {
                    (void)device->SetZoom(0);
                } else {
                    device->RefreshZoom();
                }
            }
        }
    }

  private:
    // Labels derived from PassthroughModeName() -- keeps enum, setup.conf, and menu in sync.
    // Order must match enum numeric values; cMenuEditStraItem stores the index, which is
    // round-tripped through setup.conf as PassthroughMode(int).
    static constexpr std::array kPassthroughModeLabels{
        PassthroughModeName(PassthroughMode::Auto),
        PassthroughModeName(PassthroughMode::On),
        PassthroughModeName(PassthroughMode::Off),
    };
    static constexpr int kPassthroughModeCount = static_cast<int>(kPassthroughModeLabels.size());

    // Same pattern as kPassthroughModeLabels; rooted in HdrModeName().
    static constexpr std::array kHdrModeLabels{
        HdrModeName(HdrMode::Auto),
        HdrModeName(HdrMode::On),
        HdrModeName(HdrMode::Off),
    };
    static constexpr int kHdrModeCount = static_cast<int>(kHdrModeLabels.size());

    int editClearOnChannelSwitch; ///< Scratch copy of clearOnChannelSwitch (0/1 for cMenuEditBoolItem).
    int editHdrMode;              ///< Scratch copy of hdrMode as int (index into kHdrModeLabels).
    int editPassthroughLatency;   ///< Scratch copy of passthroughLatency; not committed until Store().
    int editPassthroughMode;      ///< Scratch copy of passthroughMode as int (index into kPassthroughModeLabels).
    int editPcmLatency;           ///< Scratch copy of pcmLatency; not committed until Store().
    int editZoomLevel[CONFIG_ZOOM_PRESET_COUNT]{}; ///< Scratch copies of per-preset zoom level (tenths-of-%).
};

// ============================================================================
// === cVaapiQuickMenu ===
// ============================================================================

/// The plugin's main menu: a two-line menu always showing both entries. OK on line 1 cycles the zoom
/// one stop and closes the OSD immediately (flash the new stop on the skin) -- a one-shot toggle, no
/// lingering menu; OK on line 2 opens the mediaplayer. This is how the single @vaapivideo main-menu
/// hook reaches both actions.
class cVaapiQuickMenu : public cOsdMenu {
  public:
    cVaapiQuickMenu(cVaapiDevice *device, std::string mediaDir)
        : cOsdMenu(tr("VAAPI Video")), device_(device), mediaDir_(std::move(mediaDir)) {
        AddItems();
    }

    [[nodiscard]] auto ProcessKey(eKeys key) -> eOSState override {
        const eOSState state = cOsdMenu::ProcessKey(key); // handles Up/Down/Back
        if (state == osUnknown && (key & ~k_Repeat) == kOk) {
            if (Current() == 0) { // Zoom line: cycle one stop and leave the menu
                if (device_ == nullptr || !device_->IsReady()) {
                    Skins.QueueMessage(mtWarning, tr("VAAPI device not ready"));
                } else {
                    (void)device_->CycleZoom();
                    const std::string label = device_->ZoomStatusLabel();
                    Skins.QueueMessage(mtInfo, label.c_str());
                }
                return osEnd;
            }
            return AddSubMenu(new cVaapiFileBrowser(mediaDir_)); // Mediaplayer line
        }
        return state;
    }

  private:
    auto AddItems() -> void {
        const int current = Current();
        Clear();
        Add(new cOsdItem(device_ != nullptr ? device_->ZoomStatusLabel().c_str() : "Zoom: off"));
        Add(new cOsdItem(tr("Mediaplayer")));
        SetCurrent(Get(current < 0 ? 0 : current));
    }

    cVaapiDevice *device_; ///< Borrowed; for CycleZoom() / ZoomStatusLabel().
    std::string mediaDir_; ///< Browser root for the mediaplayer item.
};

// ============================================================================
// === cVaapiVideoPlugin ===
// ============================================================================

/// Top-level VDR plugin: arg parsing, lifecycle, setup menu, SVDRP, and service API.
class cVaapiVideoPlugin : public cPlugin {
  public:
    cVaapiVideoPlugin();
    ~cVaapiVideoPlugin() noexcept override;
    cVaapiVideoPlugin(const cVaapiVideoPlugin &) = delete;
    cVaapiVideoPlugin(cVaapiVideoPlugin &&) = delete;
    auto operator=(const cVaapiVideoPlugin &) -> cVaapiVideoPlugin & = delete;
    auto operator=(cVaapiVideoPlugin &&) -> cVaapiVideoPlugin & = delete;

    // VDR plugin API -- called in the order documented at the top of this file
    [[nodiscard]] auto CommandLineHelp() -> const char * override;
    [[nodiscard]] auto Description() -> const char * override { return PLUGIN_DESCRIPTION; }
    auto Housekeeping() -> void override;
    [[nodiscard]] auto Initialize() -> bool override;
    [[nodiscard]] auto MainMenuEntry() -> const char * override;
    [[nodiscard]] auto MainMenuAction() -> cOsdObject * override;
    [[nodiscard]] auto ProcessArgs(int argc, char *argv[]) -> bool override;
    [[nodiscard]] auto Service(const char *serviceId, void *data = nullptr) -> bool override;
    [[nodiscard]] auto SetupMenu() -> cMenuSetupPage * override;
    [[nodiscard]] auto SetupParse(const char *Name, const char *Value) -> bool override;
    [[nodiscard]] auto Start() -> bool override;
    auto Stop() -> void override;
    [[nodiscard]] auto SVDRPCommand(const char *command, const char *option, int &replyCode) -> cString override;
    [[nodiscard]] auto SVDRPHelpPages() -> const char ** override;
    [[nodiscard]] auto Version() -> const char * override { return PLUGIN_VERSION; }

  private:
    /// Resolves the DRM device path: explicit arg > /dev/dri/card0 > libdrm enumeration.
    /// Returns an empty cString and logs errors on failure.
    [[nodiscard]] auto ResolveDrmDevice() const -> cString;

    cString audioDevice;       ///< ALSA device (-a / --audio); defaults to "default".
    cString connectorName;     ///< DRM connector (-c / --connector); empty = first connected.
    cString drmPath;           ///< DRM device path (-d / --drm); empty = auto-detect.
    cString mediaDir{"/"};     ///< Mediaplayer initial directory (-m / --media-dir); defaults to "/".
    bool startDetached{false}; ///< -D / --detached: defer all hardware init until SVDRP ATTA or
                               ///<   primary-device switch.

    /// Non-owning pointer. cDevice self-registers with VDR on construction; VDR destroys it
    /// via cDevice::Shutdown() after all Stop() calls. Never delete this pointer.
    cVaapiDevice *vaapiDevice{nullptr};
};

// ============================================================================
// === cVaapiVideoPlugin -- special members ===
// ============================================================================

cVaapiVideoPlugin::cVaapiVideoPlugin() : audioDevice("default") { dsyslog("vaapivideo: plugin created"); }

cVaapiVideoPlugin::~cVaapiVideoPlugin() noexcept = default;

// ============================================================================
// === cVaapiVideoPlugin -- private helpers ===
// ============================================================================

auto cVaapiVideoPlugin::ResolveDrmDevice() const -> cString {
    if (!isempty(*drmPath)) {
        isyslog("vaapivideo: using specified DRM device: %s", *drmPath);
        return drmPath;
    }

    // /dev/dri/card0 covers the vast majority of single-GPU setups; fall back to libdrm
    // enumeration only when that node is absent or inaccessible.
    if (access("/dev/dri/card0", R_OK | W_OK) == 0) {
        isyslog("vaapivideo: auto-detected /dev/dri/card0 (primary GPU)");
        return "/dev/dri/card0";
    }

    dsyslog("vaapivideo: /dev/dri/card0 not accessible, enumerating all DRM "
            "devices...");
    DrmDevices drmEnum;
    if (!drmEnum.Enumerate()) {
        esyslog("vaapivideo: could not enumerate DRM devices");
        esyslog("vaapivideo: specify device explicitly: vdr -P 'vaapivideo -d "
                "/dev/dri/cardX'");
        return {};
    }

    for (const auto &dev : drmEnum) {
        if (dev && (dev->available_nodes & (1 << DRM_NODE_PRIMARY))) {
            const char *primaryNode = dev->nodes[DRM_NODE_PRIMARY];
            dsyslog("vaapivideo: found DRM primary node: %s", primaryNode);
            return primaryNode;
        }
    }

    esyslog("vaapivideo: no DRM device with a primary node found");
    esyslog("vaapivideo: run 'ls -l /dev/dri/' to inspect available devices");
    return {};
}

// ============================================================================
// === cVaapiVideoPlugin -- VDR plugin API ===
// ============================================================================

auto cVaapiVideoPlugin::CommandLineHelp() -> const char * {
    // VDR stores the returned pointer for the process lifetime; must be static.
    static const std::string kHelp =
        std::format("  -a DEV, --audio=DEV         Use ALSA audio device DEV "
                    "(default: 'default')\n"
                    "  -c NAME, --connector=NAME   Use DRM connector NAME "
                    "(default: first connected)\n"
                    "  -D, --detached              Start without opening the "
                    "DRM/VAAPI/ALSA hardware;\n"
                    "                              attach later via the primary-device "
                    "switch or SVDRP ATTA\n"
                    "  -d DEV, --drm=DEV           Use DRM device DEV "
                    "(default: auto-detect)\n"
                    "  -m DIR, --media-dir=DIR     Mediaplayer initial directory "
                    "(default: '/')\n"
                    "  -r RES, --resolution=RES    Output resolution "
                    "WIDTHxHEIGHT@RATE (default: {}x{}@{})\n",
                    DISPLAY_DEFAULT_WIDTH, DISPLAY_DEFAULT_HEIGHT, DISPLAY_DEFAULT_REFRESH_RATE);
    return kHelp.c_str();
}

auto cVaapiVideoPlugin::Housekeeping() -> void {}

auto cVaapiVideoPlugin::MainMenuEntry() -> const char * { return tr("VAAPI Video"); }

auto cVaapiVideoPlugin::MainMenuAction() -> cOsdObject * {
    // Always the quick menu: line 1 cycles zoom, line 2 opens the mediaplayer. One @vaapivideo hook
    // exposes both. The dir falls back to "/" if empty.
    std::string dir = isempty(*mediaDir) ? std::string{"/"} : std::string{*mediaDir};
    return new cVaapiQuickMenu(vaapiDevice, std::move(dir));
}

auto cVaapiVideoPlugin::Initialize() -> bool {
    // PPS/SPS parse errors during initial stream acquisition flood the log; only fatal FFmpeg errors are relevant.
    av_log_set_level(AV_LOG_FATAL);

    // Must create cVaapiDevice in Initialize(), not Start(): VDR calls all Initialize() first, then sets the primary
    // device, then calls all Start(). Skin plugins query cOsdProvider::SupportsTrueColor() in their Start(), which
    // requires the OSD provider registered via MakePrimaryDevice() -- deferring to Start() misses that window.

    const cString resolvedDrm = ResolveDrmDevice();
    if (isempty(*resolvedDrm)) {
        return false;
    }

    isyslog("vaapivideo: using audio device: %s", *audioDevice);

    dsyslog("vaapivideo: creating cVaapiDevice (DRM=%s, audio=%s, detached=%d)", *resolvedDrm, *audioDevice,
            startDetached ? 1 : 0);
    vaapiDevice = new cVaapiDevice(); // ownership immediately transferred to VDR's device registry

    if (!vaapiDevice->Initialize(*resolvedDrm, *audioDevice,
                                 isempty(*connectorName) ? std::string_view{} : std::string_view{*connectorName},
                                 startDetached)) {
        esyslog("vaapivideo: ========================================");
        esyslog("vaapivideo: device initialization FAILED");
        esyslog("vaapivideo: plugin will not be available");
        esyslog("vaapivideo: VDR will continue with DVB devices only");
        esyslog("vaapivideo: see error messages above for details");
        esyslog("vaapivideo: ========================================");
        vaapiDevice = nullptr; // VDR still destroys the instance; just relinquish the pointer
        return false;
    }

    isyslog("vaapivideo: initialized, version %s", PLUGIN_VERSION);
    return true;
}

auto cVaapiVideoPlugin::ProcessArgs(int argc, char *argv[]) -> bool {
    // NOLINTBEGIN(misc-include-cleaner) -- getopt symbols (option, getopt_long, optind, optarg,
    // required_argument, no_argument) come from <getopt.h>, but clang-tidy's IWYU doesn't track them.
    static constexpr std::array<option, 7> kLongOptions = {
        {{.name = "audio", .has_arg = required_argument, .flag = nullptr, .val = 'a'},
         {.name = "connector", .has_arg = required_argument, .flag = nullptr, .val = 'c'},
         {.name = "detached", .has_arg = no_argument, .flag = nullptr, .val = 'D'},
         {.name = "drm", .has_arg = required_argument, .flag = nullptr, .val = 'd'},
         {.name = "media-dir", .has_arg = required_argument, .flag = nullptr, .val = 'm'},
         {.name = "resolution", .has_arg = required_argument, .flag = nullptr, .val = 'r'},
         {.name = nullptr, .has_arg = 0, .flag = nullptr, .val = 0}}};

    optind = 1; // getopt state is global; reset so re-invocation by VDR parses cleanly.
    int opt{};
    while ((opt = getopt_long(argc, argv, "d:a:c:r:m:D", kLongOptions.data(), nullptr)) != -1) {
        switch (opt) {
            case 'd':
                if (optarg == nullptr || *optarg == '\0') {
                    esyslog("vaapivideo: empty DRM device argument");
                    return false;
                }
                drmPath = optarg;
                dsyslog("vaapivideo: DRM device set to '%s'", *drmPath);
                break;
            case 'a':
                if (optarg == nullptr || *optarg == '\0') {
                    esyslog("vaapivideo: empty audio device argument");
                    return false;
                }
                audioDevice = optarg;
                dsyslog("vaapivideo: audio device set to '%s'", *audioDevice);
                break;
            case 'c':
                if (optarg == nullptr || *optarg == '\0') {
                    esyslog("vaapivideo: empty connector argument");
                    return false;
                }
                connectorName = optarg;
                dsyslog("vaapivideo: connector set to '%s'", *connectorName);
                break;
            case 'r':
                if (optarg == nullptr || *optarg == '\0') {
                    esyslog("vaapivideo: empty resolution argument");
                    return false;
                }
                if (!vaapiConfig.display.ParseResolution(optarg)) {
                    esyslog("vaapivideo: failed to parse resolution '%s'", optarg);
                    return false;
                }
                dsyslog("vaapivideo: resolution set to %ux%u@%u", vaapiConfig.display.GetWidth(),
                        vaapiConfig.display.GetHeight(), vaapiConfig.display.GetRefreshRate());
                break;
            case 'm':
                if (optarg == nullptr || *optarg == '\0') {
                    esyslog("vaapivideo: empty media-dir argument");
                    return false;
                }
                mediaDir = optarg;
                dsyslog("vaapivideo: mediaplayer initial directory '%s'", *mediaDir);
                break;
            case 'D':
                startDetached = true;
                dsyslog("vaapivideo: detached startup requested");
                break;
            default:
                esyslog("vaapivideo: unrecognized command-line option (see stderr)");
                return false;
        }
    }
    return true;
    // NOLINTEND(misc-include-cleaner)
}

// Inter-plugin service API. data==nullptr is a capability probe; non-null fills the typed result.
//   "VaapiVideo-Available-v1.0"  -- bool*    -- hardware decoder is ready
//   "VaapiVideo-IsReady-v1.0"    -- bool*    -- device fully initialized (not detached)
//   "VaapiVideo-DeviceType-v1.0" -- cString* -- human-readable GPU/driver string
auto cVaapiVideoPlugin::Service(const char *serviceId, void *data) -> bool {
    if (serviceId == nullptr) {
        return false;
    }

    if (strcmp(serviceId, "VaapiVideo-Available-v1.0") == 0) {
        if (data != nullptr) {
            *static_cast<bool *>(data) = vaapiDevice && vaapiDevice->HasDecoder();
        }
        return true;
    }

    if (strcmp(serviceId, "VaapiVideo-IsReady-v1.0") == 0) {
        if (data != nullptr) {
            *static_cast<bool *>(data) = vaapiDevice && vaapiDevice->IsReady();
        }
        return true;
    }

    if (strcmp(serviceId, "VaapiVideo-DeviceType-v1.0") == 0) {
        if (data != nullptr) {
            *static_cast<cString *>(data) = vaapiDevice ? vaapiDevice->DeviceType() : "N/A";
        }
        return true;
    }

    return false; // unknown ID -- VDR will try other plugins
}

auto cVaapiVideoPlugin::SetupMenu() -> cMenuSetupPage * { return new cMenuSetupVaapi(); }

auto cVaapiVideoPlugin::SetupParse(const char *Name, const char *Value) -> bool {
    return vaapiConfig.SetupParse(Name, Value);
}

auto cVaapiVideoPlugin::Start() -> bool {
    isyslog("vaapivideo: starting VAAPI Video Plugin v%s", PLUGIN_VERSION);

    if (!vaapiDevice) [[unlikely]] {
        esyslog("vaapivideo: device not created -- cannot start");
        return false;
    }

    // Detached startup: hardware is not open yet. MarkStartupComplete() arms the
    // MakePrimaryDevice() gate so post-startup primary switches will trigger Attach() --
    // but the setup.conf-driven promotion that already ran will not.
    if (!vaapiDevice->IsReady()) {
        isyslog("vaapivideo: started detached -- hardware init deferred until attach");
        vaapiDevice->MarkStartupComplete();
        return true;
    }

    // VDR has processed setup.conf and called MakePrimaryDevice() for the stored primary.
    // If the device was not promoted (first-run or misconfigured setup), force it now.
    if (!vaapiDevice->IsPrimaryDevice()) {
        const int primaryIndex = vaapiDevice->DeviceNumber() + 1;
        if (cDevice::SetPrimaryDevice(primaryIndex)) {
            isyslog("vaapivideo: set as primary device %d", primaryIndex);
        } else {
            // SetPrimaryDevice() uses the global index table; SetPrimary() is the direct
            // fallback when that lookup fails (e.g. device registered late).
            esyslog("vaapivideo: SetPrimaryDevice(%d) failed, falling back to SetPrimary", primaryIndex);
            vaapiDevice->SetPrimary(true);
        }
    }

    if (!vaapiDevice->IsPrimaryDevice()) {
        esyslog("vaapivideo: could not establish as primary device -- aborting");
        vaapiDevice = nullptr;
        return false;
    }
    isyslog("vaapivideo: primary device confirmed -- startup complete");
    vaapiDevice->MarkStartupComplete();
    return true;
}

auto cVaapiVideoPlugin::Stop() -> void {
    // Do not delete vaapiDevice: cDevice::Shutdown() (called from main() after all Stop()s) owns it.
    if (vaapiDevice) {
        // cVaapiOsdProvider holds a raw pointer to cVaapiDisplay; detach before VDR destroys
        // cVaapiDevice to prevent a dangling-pointer dereference in the display destructor.
        if (auto *provider = dynamic_cast<cVaapiOsdProvider *>(::osdProvider)) {
            provider->DetachDisplay();
            dsyslog("vaapivideo: OSD provider detached from display");
        }
        vaapiDevice = nullptr;
    }

    isyslog("vaapivideo: plugin stopped");
}

auto cVaapiVideoPlugin::SVDRPCommand(const char *command, const char *option, int &replyCode) -> cString {
    // SVDRP reply codes: 900=success, 550=action not taken, 500=unknown command.

    if (strcasecmp(command, "DETA") == 0) {
        if (!vaapiDevice) [[unlikely]] {
            replyCode = 550;
            return "No VAAPI device";
        }
        if (!vaapiDevice->IsReady()) [[unlikely]] {
            replyCode = 550;
            return "VAAPI device is already detached";
        }
        const bool vtYielded = vaapiDevice->Detach();
        replyCode = 900;
        if (vtYielded) {
            return "VAAPI device detached from hardware";
        }
        // VT yield failed: CAP_SYS_TTY_CONFIG missing or stdin not a VT. Log has details.
        return "VAAPI device detached -- press Alt+F<n> to restore the console "
               "(see README: console and keyboard integration)";
    }

    if (strcasecmp(command, "ATTA") == 0) {
        if (!vaapiDevice) [[unlikely]] {
            replyCode = 550;
            return "No VAAPI device";
        }
        if (vaapiDevice->IsReady()) [[unlikely]] {
            replyCode = 550;
            return "VAAPI device is already attached";
        }
        if (!vaapiDevice->Attach()) {
            replyCode = 550;
            return "VAAPI device attach failed - check logs for details";
        }
        // Promote to primary on ATTA: a non-primary vaapivideo holds the hardware but does
        // not render. SetPrimaryDevice() is synchronous, so the IsPrimaryDevice() gate below
        // observes the new state.
        if (!vaapiDevice->IsPrimaryDevice()) {
            isyslog("vaapivideo: ATTA promoting device %d to primary", vaapiDevice->DeviceNumber() + 1);
            cDevice::SetPrimaryDevice(vaapiDevice->DeviceNumber() + 1);
        }
        if (vaapiDevice->IsPrimaryDevice()) {
            LOCK_CHANNELS_READ;
            if (const cChannel *channel = Channels->GetByNumber(cDevice::CurrentChannel())) {
                cDevice::PrimaryDevice()->SwitchChannel(channel, true);
            }
        }
        replyCode = 900;
        return "VAAPI device attached to hardware";
    }

    if (strcasecmp(command, "STAT") == 0) {
        if (!vaapiDevice) [[unlikely]] {
            replyCode = 550;
            return "No VAAPI device";
        }
        if (!vaapiDevice->IsReady()) [[unlikely]] {
            replyCode = 550;
            return "VAAPI device detached (hardware not attached) -- use ATTA to attach";
        }

        int width = 0;
        int height = 0;
        double aspect = 1.0;
        vaapiDevice->GetOsdSize(width, height, aspect);

        replyCode = 900;
        return cString::sprintf("VAAPI Device Status:\n"
                                "Type: %s\n"
                                "Decoder: %s\n"
                                "Display Resolution: %dx%d\n"
                                "Refresh Rate: %u Hz",
                                *vaapiDevice->DeviceType(), vaapiDevice->HasDecoder() ? "Ready" : "Not Ready", width,
                                height, vaapiConfig.display.GetRefreshRate());
    }

    if (strcasecmp(command, "CONFIG") == 0) {
        replyCode = 900;
        return cString::sprintf("Configuration:\n%s", vaapiConfig.GetSummary().c_str());
    }

    if (strcasecmp(command, "PLAY") == 0) {
        if (option == nullptr || *option == '\0') {
            replyCode = 550;
            return "PLAY needs a path or URL";
        }
        if (!vaapiDevice || !vaapiDevice->IsReady()) {
            replyCode = 550;
            return "VAAPI device not ready -- cannot start playback";
        }

        std::vector<PlaylistEntry> entries;
        const std::string_view uri{option};
        if (IsPlaylistUri(uri)) {
            entries = ParseM3U(uri);
            if (entries.empty()) {
                replyCode = 550;
                return cString::sprintf("Empty or unreadable playlist: %s", option);
            }
        } else {
            entries.push_back(PlaylistEntry{.uri = std::string{uri}, .title = std::string{uri}});
        }

        if (!StartPlayback(std::move(entries))) {
            replyCode = 550;
            return "Could not start mediaplayer (no primary vaapivideo device?)";
        }
        replyCode = 900;
        return cString::sprintf("Playing %s", option);
    }

    if (strcasecmp(command, "ZOOM") == 0) {
        if (!vaapiDevice || !vaapiDevice->IsReady()) {
            replyCode = 550;
            return "VAAPI device not ready";
        }
        // No arg or "next" cycles; a number selects a stop directly (0 = Off, 1..N = preset).
        if (option == nullptr || *option == '\0' || strcasecmp(option, "next") == 0) {
            (void)vaapiDevice->CycleZoom();
        } else {
            int parsed{};
            const auto *end = option + std::strlen(option);
            const auto [ptr, ec] = std::from_chars(option, end, parsed);
            if (ec != std::errc{} || ptr != end || parsed < 0 || parsed > CONFIG_ZOOM_PRESET_COUNT) {
                replyCode = 550;
                return cString::sprintf("ZOOM needs 'next' or 0-%d", CONFIG_ZOOM_PRESET_COUNT);
            }
            if (parsed > 0 && vaapiConfig.zoomLevel[parsed - 1].load(std::memory_order_relaxed) <= 0) {
                replyCode = 550;
                return cString::sprintf("Zoom stop %d is disabled (level 0)", parsed);
            }
            (void)vaapiDevice->SetZoom(parsed);
        }
        // Flash the new stop on the OSD too: live-TV zoom is driven via this command (a key bound to
        // `svdrpsend ... ZOOM next`), so the SVDRP reply alone would leave the viewer no feedback.
        // QueueMessage is the thread-safe, deferred variant -- safe to call from the SVDRP thread.
        const std::string label = vaapiDevice->ZoomStatusLabel();
        Skins.QueueMessage(mtInfo, label.c_str());
        replyCode = 900;
        return cString::sprintf("%s", label.c_str());
    }

    replyCode = 500;
    return {"Unknown SVDRP command"};
}

auto cVaapiVideoPlugin::SVDRPHelpPages() -> const char ** {
    // Built once; *zoomHelp stays valid for the array's lifetime because the cString is static.
    static const cString zoomHelp = cString::sprintf(
        "ZOOM [next|0-%d]\n    Cycle manual zoom or pick a stop (0=off, 1-%d=preset). Resets on content change.",
        CONFIG_ZOOM_PRESET_COUNT, CONFIG_ZOOM_PRESET_COUNT);
    static const char *const kHelpPages[] = {
        "DETA\n    Detach from the DRM/VAAPI hardware, allowing other applications to use the display.",
        "ATTA\n    Re-attach to the DRM/VAAPI hardware and restart all subsystem threads.",
        "STAT\n    Show detailed device status and statistics.",
        "CONFIG\n    Display current configuration settings.",
        "PLAY <uri>\n    Play a local file, URL, or .m3u/.m3u8 playlist via the integrated mediaplayer.",
        *zoomHelp,
        nullptr};
    // VDR's SVDRPHelpPages() signature is const char ** but the literal array is const char *const *;
    // both pointee levels are read-only at the call site, so stripping the inner const is safe.
    return const_cast<const char **>(kHelpPages); // NOLINT(cppcoreguidelines-pro-type-const-cast)
}

// ============================================================================
// === VDR plugin factory ===
// ============================================================================

} // namespace

VDRPLUGINCREATOR(cVaapiVideoPlugin);
