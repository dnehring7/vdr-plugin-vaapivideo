// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Dirk Nehring <dnehring@gmx.net>
/**
 * @file vaapivideo.cpp
 * @brief VDR plugin entry point and lifecycle management
 *
 * VDR plugin lifecycle call order (relevant to this file):
 *   1. Constructor      -- object created, no hardware touched yet
 *   2. ProcessArgs()    -- command-line args parsed, stored for later use
 *   3. Initialize()     -- hardware opened, cVaapiDevice created and initialized
 *   4. Start()          -- primary device confirmed, startup finalized
 *   5. Housekeeping()   -- called periodically by VDR's main loop (empty here)
 *   6. Stop()           -- hardware released; VDR still owns cVaapiDevice
 *   7. Destructor       -- trivial; VDR destroys cVaapiDevice separately via
 * cDevice::Shutdown()
 */

#include "src/common.h"
#include "src/config.h"
#include "src/device.h"
#include "src/osd.h"

// POSIX
#include <getopt.h> // NOLINT(misc-include-cleaner)
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

/// VDR setup page for the VAAPI plugin. Edits are applied to a local copy and
/// only written back to the global config when the user confirms with Store().
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
        // Two independent A/V offsets: one used while audio is decoded to PCM, the other while audio
        // is forwarded as IEC61937 passthrough. Bounds come from config.h so the menu and the
        // setup.conf parser share a single source of truth.
        Add(new cMenuEditIntItem(tr("PCM Audio Latency (ms)"), &editPcmLatency, CONFIG_AUDIO_LATENCY_MIN_MS,
                                 CONFIG_AUDIO_LATENCY_MAX_MS));
        Add(new cMenuEditIntItem(tr("Passthrough Audio Latency (ms)"), &editPassthroughLatency,
                                 CONFIG_AUDIO_LATENCY_MIN_MS, CONFIG_AUDIO_LATENCY_MAX_MS));
        Add(new cMenuEditStraItem(tr("Audio Passthrough"), &editPassthroughMode, kPassthroughModeCount,
                                  kPassthroughModeLabels.data()));
        Add(new cMenuEditStraItem(tr("HDR Passthrough"), &editHdrMode, kHdrModeCount, kHdrModeLabels.data()));
        // Override the default "no"/"yes" so both boolean-style items in this page read "off"/"on".
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
    // Labels for cMenuEditStraItem, derived from PassthroughModeName() so the enum, the
    // setup.conf log output, and the setup-menu display can never drift apart. Order must
    // match the enum's numeric values: cMenuEditStraItem stores the selected index and we
    // round-trip it through setup.conf as PassthroughMode(int).
    static constexpr std::array kPassthroughModeLabels{
        PassthroughModeName(PassthroughMode::Auto),
        PassthroughModeName(PassthroughMode::On),
        PassthroughModeName(PassthroughMode::Off),
    };
    static constexpr int kPassthroughModeCount = static_cast<int>(kPassthroughModeLabels.size());

    // HdrMode menu labels -- same pattern, rooted in HdrModeName() for single-source-of-truth.
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

/// Top-level VDR plugin class. Responsible for command-line argument parsing,
/// plugin lifecycle (Initialize / Start / Stop), the setup menu, the SVDRP
/// interface, and the inter-plugin service API.
class cVaapiVideoPlugin : public cPlugin {
  public:
    cVaapiVideoPlugin();
    ~cVaapiVideoPlugin() noexcept override;
    cVaapiVideoPlugin(const cVaapiVideoPlugin &) = delete;
    cVaapiVideoPlugin(cVaapiVideoPlugin &&) = delete;
    auto operator=(const cVaapiVideoPlugin &) -> cVaapiVideoPlugin & = delete;
    auto operator=(cVaapiVideoPlugin &&) -> cVaapiVideoPlugin & = delete;

    // VDR plugin API -- called by VDR in the order described in the file-level comment
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
    /// Returns the DRM device path to use: the user-supplied value, or the best
    /// candidate found by probing /dev/dri/card0 first and then falling back to
    /// libdrm enumeration.
    [[nodiscard]] auto ResolveDrmDevice() const -> cString;

    cString audioDevice;   ///< ALSA device passed via -a / --audio; defaults to
                           ///< "default".
    cString connectorName; ///< DRM connector name passed via -c / --connector;
                           ///< empty means first connected output.
    cString drmPath;       ///< DRM device path passed via -d / --drm; empty means
                           ///< auto-detect.

    /// Non-owning pointer to the active output device.
    /// All cDevice instances self-register with VDR in their constructor and
    /// are destroyed by cDevice::Shutdown() (called from main()) -- never delete
    /// this pointer manually.
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

    // No device specified -- try /dev/dri/card0 first (covers the vast majority of single-GPU setups),
    // then fall back to full libdrm enumeration when that node is absent or inaccessible.
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
    // Static storage: VDR keeps a pointer to the returned string for the process lifetime.
    static const std::string kHelp =
        std::format("  -d DEV, --drm=DEV           Use DRM device DEV "
                    "(default: auto-detect)\n"
                    "  -a DEV, --audio=DEV         Use ALSA audio device DEV "
                    "(default: 'default')\n"
                    "  -c NAME, --connector=NAME   Use DRM connector NAME "
                    "(default: first connected)\n"
                    "  -r RES, --resolution=RES    Output resolution "
                    "WIDTHxHEIGHT@RATE (default: {}x{}@{})\n",
                    DISPLAY_DEFAULT_WIDTH, DISPLAY_DEFAULT_HEIGHT, DISPLAY_DEFAULT_REFRESH_RATE);
    return kHelp.c_str();
}

auto cVaapiVideoPlugin::Housekeeping() -> void {}

auto cVaapiVideoPlugin::Initialize() -> bool {
    // FFmpeg's PPS/SPS parse errors during initial stream acquisition flood the log; only fatal errors matter here.
    av_log_set_level(AV_LOG_FATAL);

    // Device must be created in Initialize(), not Start(): VDR's startup is all Initialize() -> set primary device ->
    // all Start(). Skin plugins call cOsdProvider::SupportsTrueColor() in their Start(), which requires our OSD
    // provider (installed via MakePrimaryDevice()). Deferring to Start() would miss the primary device window.

    const cString resolvedDrm = ResolveDrmDevice();
    if (isempty(*resolvedDrm)) {
        return false;
    }

    isyslog("vaapivideo: using audio device: %s", *audioDevice);

    // cVaapiDevice self-registers with VDR's device list in its cDevice constructor; ownership transfers to VDR
    // immediately and cDevice::Shutdown() (called from main()) will delete it. Never delete vaapiDevice manually.
    dsyslog("vaapivideo: creating cVaapiDevice (DRM=%s, audio=%s)", *resolvedDrm, *audioDevice);
    vaapiDevice = new cVaapiDevice();

    if (!vaapiDevice->Initialize(*resolvedDrm, *audioDevice,
                                 isempty(*connectorName) ? std::string_view{} : std::string_view{*connectorName})) {
        esyslog("vaapivideo: ========================================");
        esyslog("vaapivideo: device initialization FAILED");
        esyslog("vaapivideo: plugin will not be available");
        esyslog("vaapivideo: VDR will continue with DVB devices only");
        esyslog("vaapivideo: see error messages above for details");
        esyslog("vaapivideo: ========================================");
        vaapiDevice = nullptr; // clear our pointer; VDR will still destroy the instance
        return false;
    }

    isyslog("vaapivideo: initialized, version %s", PLUGIN_VERSION);
    return true;
}

auto cVaapiVideoPlugin::ProcessArgs(int argc, char *argv[]) -> bool {
    // NOLINTNEXTLINE(misc-include-cleaner)
    static constexpr std::array<option, 5> kLongOptions = {
        {{.name = "drm", .has_arg = required_argument, .flag = nullptr, .val = 'd'}, // NOLINT(misc-include-cleaner)
         {.name = "audio", .has_arg = required_argument, .flag = nullptr, .val = 'a'},
         {.name = "connector", .has_arg = required_argument, .flag = nullptr, .val = 'c'},
         {.name = "resolution", .has_arg = required_argument, .flag = nullptr, .val = 'r'},
         {.name = nullptr, .has_arg = 0, .flag = nullptr, .val = 0}}};

    optind = 1; // Reset global getopt state; VDR may call ProcessArgs() more than once. // NOLINT(misc-include-cleaner)
    int opt{};
    // NOLINTNEXTLINE(misc-include-cleaner)
    while ((opt = getopt_long(argc, argv, "d:a:c:r:", kLongOptions.data(), nullptr)) != -1) {
        switch (opt) {
            case 'd':
                if (optarg == nullptr || *optarg == '\0') { // NOLINT(misc-include-cleaner)
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
            default:
                esyslog("vaapivideo: unrecognized command-line option (see stderr)");
                return false;
        }
    }
    return true;
}

// Inter-plugin service API.
//
// Service IDs and data contracts:
//   "VaapiVideo-Available-v1.0"  -- bool*    -- true if hardware decoder is ready
//   "VaapiVideo-IsReady-v1.0"    -- bool*    -- true if device is fully initialized
//   "VaapiVideo-DeviceType-v1.0" -- cString* -- human-readable device type string
//
// data==nullptr is a pure capability probe ("does this service exist?").
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

    return false; // unknown service ID -- let VDR try other plugins
}

auto cVaapiVideoPlugin::SetupMenu() -> cMenuSetupPage * { return new cMenuSetupVaapi(); }

auto cVaapiVideoPlugin::SetupParse(const char *Name, const char *Value) -> bool {
    return vaapiConfig.SetupParse(Name, Value);
}

auto cVaapiVideoPlugin::Start() -> bool {
    isyslog("vaapivideo: starting VAAPI Video Plugin v%s", PLUGIN_VERSION);

    if (!vaapiDevice || !vaapiDevice->IsReady()) [[unlikely]] {
        esyslog("vaapivideo: device not initialized -- cannot start");
        return false;
    }

    // By this point VDR has processed setup.conf and called MakePrimaryDevice() on the stored primary device.
    // The fallback below handles first-run or misconfigured setups where VDR did not select our device.
    if (!vaapiDevice->IsPrimaryDevice()) {
        const int primaryIndex = vaapiDevice->DeviceNumber() + 1;
        if (cDevice::SetPrimaryDevice(primaryIndex)) {
            isyslog("vaapivideo: set as primary device %d", primaryIndex);
        } else {
            // SetPrimaryDevice() can fail if called too early in VDR's startup; SetPrimary() is the
            // direct instance-level fallback that bypasses the global device-index lookup.
            esyslog("vaapivideo: SetPrimaryDevice(%d) failed, falling back to "
                    "SetPrimary",
                    primaryIndex);
            vaapiDevice->SetPrimary(true);
        }
    }

    if (!vaapiDevice->IsPrimaryDevice()) {
        esyslog("vaapivideo: could not establish as primary device -- aborting");
        vaapiDevice = nullptr;
        return false;
    }
    isyslog("vaapivideo: primary device confirmed -- startup complete");
    return true;
}

auto cVaapiVideoPlugin::Stop() -> void {
    // Do not delete vaapiDevice: VDR's device registry owns it; cDevice::Shutdown() (called from main() after all
    // Stop() invocations) destroys it. Deleting here would cause a double-free.

    if (vaapiDevice) {
        // The OSD provider holds a raw pointer to cVaapiDisplay. Detach it before VDR tears down cVaapiDevice
        // to prevent a dangling-pointer dereference when the display destructor runs.
        if (auto *provider = dynamic_cast<cVaapiOsdProvider *>(::osdProvider)) {
            provider->DetachDisplay();
            dsyslog("vaapivideo: OSD provider detached from display");
        }
        vaapiDevice = nullptr; // Relinquish non-owning pointer; VDR destroys the object via cDevice::Shutdown().
    }

    isyslog("vaapivideo: plugin stopped");
}

auto cVaapiVideoPlugin::SVDRPCommand(const char *command, [[maybe_unused]] const char *option, int &replyCode)
    -> cString {
    // VDR SVDRP reply codes: 900=success, 550=action not taken, 500=unknown command.

    if (strcasecmp(command, "DETA") == 0) {
        if (!vaapiDevice) [[unlikely]] {
            replyCode = 550;
            return "No VAAPI device";
        }
        vaapiDevice->Detach();
        replyCode = 900;
        return "VAAPI device detached from hardware";
    }

    if (strcasecmp(command, "ATTA") == 0) {
        if (!vaapiDevice) [[unlikely]] {
            replyCode = 550;
            return "No VAAPI device";
        }
        if (!vaapiDevice->Attach()) {
            replyCode = 550;
            return "VAAPI device attach failed - check logs for details";
        }
        // Force VDR to restart the current channel's transfer so data flows through the freshly
        // re-initialized decoder/display pipeline.
        {
            LOCK_CHANNELS_READ;
            if (const cChannel *channel = Channels->GetByNumber(cDevice::CurrentChannel())) {
                cDevice::PrimaryDevice()->SwitchChannel(channel, true);
            }
        }
        replyCode = 900;
        return "VAAPI device attached to hardware";
    }

    if (strcasecmp(command, "STAT") == 0) {
        if (!vaapiDevice || !vaapiDevice->IsReady()) [[unlikely]] {
            replyCode = 550;
            return "VAAPI device inactive";
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
    return const_cast<const char **>(kHelpPages); // NOLINT(cppcoreguidelines-pro-type-const-cast)
}

// ============================================================================
// === VDR plugin factory ===
// ============================================================================

} // namespace

VDRPLUGINCREATOR(cVaapiVideoPlugin);
