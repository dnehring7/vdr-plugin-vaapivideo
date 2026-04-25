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
#include <cstring>
#include <format>
#include <string>

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
#include <vdr/menuitems.h>
#include <vdr/plugin.h>
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
    }

  private:
    // Labels derived from PassthroughModeName() -- keeps enum, setup.conf, and menu in sync.
    // Order must match enum numeric values; cMenuEditStraItem stores the index and we
    // round-trip it through setup.conf as PassthroughMode(int).
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
        std::format("  -d DEV, --drm=DEV           Use DRM device DEV "
                    "(default: auto-detect)\n"
                    "  -a DEV, --audio=DEV         Use ALSA audio device DEV "
                    "(default: 'default')\n"
                    "  -c NAME, --connector=NAME   Use DRM connector NAME "
                    "(default: first connected)\n"
                    "  -r RES, --resolution=RES    Output resolution "
                    "WIDTHxHEIGHT@RATE (default: {}x{}@{})\n"
                    "  -D, --detached              Start without opening the "
                    "DRM/VAAPI/ALSA hardware;\n"
                    "                              attach later via the primary-device "
                    "switch or SVDRP ATTA\n",
                    DISPLAY_DEFAULT_WIDTH, DISPLAY_DEFAULT_HEIGHT, DISPLAY_DEFAULT_REFRESH_RATE);
    return kHelp.c_str();
}

auto cVaapiVideoPlugin::Housekeeping() -> void {}

auto cVaapiVideoPlugin::Initialize() -> bool {
    // PPS/SPS parse errors during initial stream acquisition flood the log; only fatal FFmpeg errors are relevant.
    av_log_set_level(AV_LOG_FATAL);

    // Must create cVaapiDevice in Initialize(), not Start(): VDR calls all Initialize() first, then sets the primary
    // device, then calls all Start(). Skin plugins query cOsdProvider::SupportsTrueColor() in their Start(), which
    // requires our OSD provider registered via MakePrimaryDevice() -- deferring to Start() misses that window.

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
        vaapiDevice = nullptr; // VDR still destroys the instance; just relinquish our pointer
        return false;
    }

    isyslog("vaapivideo: initialized, version %s", PLUGIN_VERSION);
    return true;
}

auto cVaapiVideoPlugin::ProcessArgs(int argc, char *argv[]) -> bool {
    // NOLINTBEGIN(misc-include-cleaner) -- getopt symbols (option, getopt_long, optind, optarg,
    // required_argument, no_argument) come from <getopt.h>, but clang-tidy's IWYU doesn't track them.
    static constexpr std::array<option, 6> kLongOptions = {
        {{.name = "drm", .has_arg = required_argument, .flag = nullptr, .val = 'd'},
         {.name = "audio", .has_arg = required_argument, .flag = nullptr, .val = 'a'},
         {.name = "connector", .has_arg = required_argument, .flag = nullptr, .val = 'c'},
         {.name = "resolution", .has_arg = required_argument, .flag = nullptr, .val = 'r'},
         {.name = "detached", .has_arg = no_argument, .flag = nullptr, .val = 'D'},
         {.name = nullptr, .has_arg = 0, .flag = nullptr, .val = 0}}};

    optind = 1; // getopt state is global; reset so re-invocation by VDR parses cleanly.
    int opt{};
    while ((opt = getopt_long(argc, argv, "d:a:c:r:D", kLongOptions.data(), nullptr)) != -1) {
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
    // If our device was not promoted (first-run or misconfigured setup), force it now.
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

auto cVaapiVideoPlugin::SVDRPCommand(const char *command, [[maybe_unused]] const char *option, int &replyCode)
    -> cString {
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
        const bool consoleRestored = vaapiDevice->Detach();
        replyCode = 900;
        if (consoleRestored) {
            return "VAAPI device detached from hardware";
        }
        // fbcon restore failed: missing /dev/tty0 perms or CAP_SYS_TTY_CONFIG. Log has details.
        return "VAAPI device detached -- press Alt+F1 to restore the console "
               "(see README: tty group + udev + CAP_SYS_TTY_CONFIG)";
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
        // Only retune if we are the primary device. With --detached, ATTA is valid for
        // non-primary devices too; retuning the current primary would interrupt other playback.
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

    replyCode = 500;
    return {"Unknown SVDRP command"};
}

auto cVaapiVideoPlugin::SVDRPHelpPages() -> const char ** {
    static const char *const kHelpPages[] = {
        "DETA\n    Detach from the DRM/VAAPI hardware, allowing other applications to use the display.",
        "ATTA\n    Re-attach to the DRM/VAAPI hardware and restart all subsystem threads.",
        "STAT\n    Show detailed device status and statistics.", "CONFIG\n    Display current configuration settings.",
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
