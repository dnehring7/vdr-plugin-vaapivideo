# VDR VAAPI Video Plugin

Hardware-accelerated video output for [VDR](https://www.tvdr.de/) using VAAPI
decode, DRM atomic modesetting, and ALSA audio output.

Unlike older VDR output plugins that rely on X11 or OpenGL, this plugin drives
the display directly through the kernel DRM/KMS subsystem. No display server
is required -- it runs on a bare console, in a systemd service, or headless.

The entire video path from decoder to screen is zero-copy: VAAPI surfaces are
exported as DRM PRIME buffers and handed to the kernel without ever touching
system memory. Audio passthrough formats (AC-3, DTS, TrueHD and others) are
detected automatically from the HDMI sink's EDID at runtime -- no manual
configuration is needed.

GPU capabilities (decode profiles, VPP filters) are probed once at startup via
the native VAAPI API. The Video Processing Pipeline (VPP) must be available --
the plugin will not start without it. Codecs without hardware decode support
(e.g. MPEG-2 on AMD) fall back to FFmpeg software decoding automatically --
the filter graph handles the upload to VAAPI surfaces transparently.


## Features

| Component | Capabilities                                                                                        |
|-----------|-----------------------------------------------------------------------------------------------------|
| Decode    | H.264, HEVC, MPEG-2 -- hardware (VAAPI) with automatic software fallback per codec                  |
| Filters   | Deinterlace, denoise, scale (DAR-preserving), sharpen -- SW path (bwdif, hqdn3d) or HW (VAAPI VPP)  |
| Audio     | PCM decode (AAC, MP2); IEC61937 passthrough (AC-3, E-AC-3, DTS, DTS-HD, TrueHD, AC-4, MPEG-H)       |
| Display   | DRM atomic modesetting, double-buffered page-flip, BT.709 SDR; up to 3840×2160                      |
| OSD       | True-color hardware overlay on a dedicated DRM plane, alpha-blended                                 |


## Architecture

```
VDR ──PES──▶ cVaapiDevice ──▶ PES Parser ──▶ cVaapiDecoder
                                                 │
                                    ┌────────────┴────────────┐
                                    ▼                         ▼
                              VAAPI HW Decode          FFmpeg SW Decode
                                    │                         │
                                    ▼                         ▼
                              VAAPI VPP Filters     SW Filters (bwdif, hqdn3d)
                              (deinterlace, denoise)     + hwupload
                                    │                         │
                                    └────────────┬────────────┘
                                                 ▼
                                   scale_vaapi (BT.709 NV12)
                                   + sharpness_vaapi
                                                 │
                                                 ▼
                                     DRM PRIME (zero-copy)
                                                 │
                                    ┌────────────┴────────────┐
                                    ▼                         ▼
                              Video Plane (NV12)       OSD Plane (ARGB8888)
                                    │                         │
                                    └────────────┬────────────┘
                                                 ▼
                                    DRM Atomic Page-Flip ──▶ Display
```

| Source File    | Responsibility                                                |
|----------------|---------------------------------------------------------------|
| audio.cpp      | ALSA output, IEC61937 passthrough                             |
| common.h       | AvErr() helper, RAII deleters, version guards, shared headers |
| config.cpp     | Plugin configuration, display parameters, setup storage       |
| decoder.cpp    | VAAPI decode, filter graphs, A/V sync                         |
| device.cpp     | VDR integration, PES routing, lifecycle                       |
| display.cpp    | DRM atomic modesetting, page flips                            |
| osd.cpp        | Hardware OSD overlay                                          |
| pes.cpp        | PES parsing, codec detection                                  |
| vaapivideo.cpp | Plugin entry point, VDR registration                          |


## Requirements

| Dependency   | Minimum   | Notes                                                          |
|--------------|-----------|----------------------------------------------------------------|
| Linux Kernel | 6.8+      | Atomic async page-flip, universal planes, COLOR_ENCODING/RANGE |
| VDR          | 2.6.0+    | APIVERSNUM >= 20600                                            |
| FFmpeg       | 7.0+      | libavcodec >= 61.3.100, built with VAAPI support               |

**Supported VAAPI drivers:**

- **Intel** (Broadwell and later): `intel-media-driver` (iHD)
- **AMD**: `mesa-va-drivers` (radeonsi)

NVIDIA GPUs are **not** supported -- the third-party `nvidia-vaapi-driver` does
not implement the Video Processing Pipeline (VPP) required by this plugin.


## Setup

### 1. Install build dependencies

**Fedora / RHEL / openSUSE:**

    dnf install gcc-c++ make git pkgconf \
        vdr-devel \
        libdrm-devel \
        alsa-lib-devel \
        ffmpeg-devel \
        libva-devel

**Debian / Ubuntu:**

    apt install g++ make git pkgconf \
        vdr-dev \
        libdrm-dev \
        libasound2-dev \
        libavcodec-dev \
        libavformat-dev \
        libavfilter-dev \
        libavutil-dev \
        libswresample-dev \
        libva-dev

**Gentoo:**

    echo "media-fonts/corefonts MSttfEULA" >> /etc/portage/package.license
    emerge -av \
        sys-devel/gcc \
        sys-devel/make \
        dev-vcs/git \
        dev-util/pkgconf \
        media-video/vdr \
        x11-libs/libdrm \
        media-libs/alsa-lib \
        media-video/ffmpeg \
        media-libs/libva

### 2. Build and install

    git clone https://github.com/dnehring7/vdr-plugin-vaapivideo.git
    cd vdr-plugin-vaapivideo
    make
    sudo make install

An RPM spec file is included for Fedora/RHEL/openSUSE:

    rpmbuild -ta vdr-vaapivideo-*.tar.gz

### 3. Permissions

The user that runs VDR needs access to DRM, video render, and ALSA devices:

    sudo usermod -aG video,render,audio vdr

A logout or service restart is required for the group change to take effect.

### 4. Install the VAAPI driver

This plugin requires a VAAPI driver with **Video Processing Pipeline (VPP)**
support. VPP provides hardware scaling, deinterlacing, denoising, and colorspace
conversion -- the plugin will not start without it.

Install the driver for your GPU:

**Fedora / RHEL / openSUSE:**

    dnf install intel-media-driver          # Intel (Broadwell+)
    dnf install mesa-va-drivers-freeworld   # AMD (radeonsi)

**Debian / Ubuntu:**

    apt install intel-media-va-driver                   # Intel (Broadwell+)
    apt install mesa-va-drivers firmware-amd-graphics   # AMD (radeonsi)

**Gentoo:**

    emerge -av media-libs/intel-media-driver                        # Intel (Broadwell+)
    USE="vaapi" VIDEO_CARDS="radeonsi" emerge -av media-libs/mesa   # AMD (radeonsi)

### 5. Verify VAAPI

Run `vainfo` to confirm the driver is working:

    vainfo --display drm --device /dev/dri/renderD128

The output must list at least one decode profile (H.264 or HEVC) and the VPP
entry point:

    VAProfileNone                   : VAEntrypointVideoProc

If this line is missing, VPP is not available and the plugin will not start.
Make sure the correct driver for your GPU is installed (see step 4).

#### Detailed capability check with vaapivideo-probe

The repository includes a standalone diagnostic tool `vaapivideo-probe.cpp`
that probes the exact capabilities the plugin needs: VLD decode profiles,
P010/NV12 surface support, VPP filters (denoise, sharpen, deinterlace),
and HDR tone mapping. It is **not** part of the plugin build -- compile and
run it manually when troubleshooting:

    g++ -std=c++20 $(pkg-config --cflags --libs libdrm libva libva-drm) \
        -o vaapivideo-probe vaapivideo-probe.cpp

    ./vaapivideo-probe                     # uses /dev/dri/card0
    ./vaapivideo-probe /dev/dri/card1      # explicit device

The output shows device info, then each capability with a color-coded
yes/no verdict:

    VAAPI Capability Prober (DVB-S2 / YUV420-NV12)
    ==============================================
    DRM device:  /dev/dri/card1
    Render node: /dev/dri/renderD128
    VA-API:      1.23
    Driver:      Intel iHD driver for Intel(R) Gen Graphics - 25.4.6 ()

    --- Hardware Decode (VLD + YUV420) ---
      MPEG-2 Main                        yes
      H.264 Main                         yes
      H.264 High                         yes
      HEVC Main 10                       yes

    --- Video Processing Pipeline (VPP) ---
      General (VideoProc)                yes
      Scaling                            yes
      P010 (10-bit) surfaces             yes
      NV12 (8-bit) surfaces              yes
      10-bit -> 8-bit conversion         yes
      Noise Reduction (Denoise)          yes
      Sharpening                         yes
      Color Balance                      yes
      Skin Tone Enhancement              yes
      Total Color Correction             yes
      HVS Noise Reduction                no
      HDR Tone Mapping (HDR10)           HDR->HDR, HDR->SDR

    --- DVB-S2 Color Conversion Paths ---
      BT.601 -> BT.709  (MPEG-2 SD)      yes
      BT.709 passthrough (H.264 HD)      yes
      BT.2020 -> BT.709  (HEVC UHD)      yes
      P010 -> NV12      (10->8 bit)      yes
      HLG -> SDR   (no TM required)      yes
      PQ/HDR10 -> SDR    (tone map)      yes

    --- Deinterlacing Algorithms ---
      Motion Compensated                 yes
      Motion Adaptive                    yes
      Weave                              no
      Bob                                yes

Any line showing **no** indicates a missing driver capability. Compare the
output against the plugin log (`vdr -l 3`) to identify mismatches.

### 6. Configure the ALSA audio device

The plugin defaults to the ALSA `default` device (stereo PCM only).
For IEC61937 bitstream passthrough (AC-3, DTS, TrueHD, ...) a direct hardware
device node is required. Passthrough formats are detected automatically from
the HDMI sink's EDID -- no manual selection is needed.

Find the card and device numbers for your HDMI output:

    aplay -l | grep -E "HDMI|DisplayPort"

Pass the result to VDR as `-a hw:CARD,DEVICE`, for example:

    vdr -P 'vaapivideo -a hw:0,3'

If A/V sync is off, adjust the audio latency in the VDR setup menu:

    Setup → Plugins → vaapivideo → Audio Latency (ms)


## Configuration

### Command-line options

    vdr -P 'vaapivideo [-d DEV] [-a DEV] [-r WxH@R]'

| Option   | Default      | Description                                            |
|----------|--------------|--------------------------------------------------------|
| `-d DEV` | auto-detect  | DRM device path (`/dev/dri/cardN`)                     |
| `-a DEV` | `default`    | ALSA audio device (use `hw:CARD,DEV` for passthrough)  |
| `-r WxH@R` | `1920x1080@50` | Output resolution and refresh rate (max 3840×2160) |

The DRM device is auto-detected: `/dev/dri/card0` is tried first; if that is
not accessible, all DRM devices are enumerated via libdrm and the first primary
node is used. Specify `-d` explicitly when multiple GPUs are present.

### VDR setup menu

    Setup → Plugins → vaapivideo

| Setting            | Range   | Description                                       |
|--------------------|---------|---------------------------------------------------|
| Audio Latency (ms) | 0 – 200 | A/V offset to compensate for external audio delay |

### SVDRP commands

| Command                  | Description                                            |
|--------------------------|--------------------------------------------------------|
| `PLUG vaapivideo STAT`   | Device status, active resolution, refresh rate         |
| `PLUG vaapivideo CONFIG` | Current configuration summary                          |
| `PLUG vaapivideo DETA`   | Detach from DRM/VAAPI hardware (release for other use) |
| `PLUG vaapivideo ATTA`   | Re-attach to DRM/VAAPI hardware and restart pipeline   |

### Inter-plugin service API

Other plugins can query device state via VDR's service interface:

| Service ID                     | Data type  | Description                                   |
|--------------------------------|------------|-----------------------------------------------|
| `VaapiVideo-Available-v1.0`    | `bool*`    | `true` if a hardware decoder is ready         |
| `VaapiVideo-IsReady-v1.0`     | `bool*`    | `true` if the device is fully initialized      |
| `VaapiVideo-DeviceType-v1.0`  | `cString*` | Human-readable device type string              |

Passing `data = nullptr` is valid and acts as a capability probe (returns `true`
for known IDs without writing data).


## Troubleshooting

| Problem                   | Solution                                                                   |
|---------------------------|----------------------------------------------------------------------------|
| No video                  | Check group membership (video, render) and run `vainfo`                    |
| No audio                  | Run: `speaker-test -D hw:0,3 -c 2 -r 48000 -t sine -l 1`                   |
| Passthrough not working   | Use `hw:CARD,DEV`; verify `/proc/asound/card0/eld#0.N` is non-empty        |
| A/V drift                 | Increase Audio Latency in setup menu (positive = delay audio)              |
| DRM device not found      | Run: `ls -l /dev/dri/` and specify `-d /dev/dri/cardN` explicitly          |
| Black screen after resume | Send SVDRP: `PLUG vaapivideo DETA` then `PLUG vaapivideo ATTA`             |


## Development

### Build targets

| Target         | Description                                          |
|----------------|------------------------------------------------------|
| `make`         | Release build (`-O3`, LTO, strip)                    |
| `make install` | Install plugin to VDR plugin directory               |
| `make clean`   | Remove build artifacts                               |
| `make dist`    | Create source tarball                                |
| `make indent`  | Format all sources with clang-format                 |
| `make lint`    | Static analysis with clang-tidy (requires bear)      |
| `make docs`    | Generate Doxygen HTML documentation                  |

`make lint` generates a `compile_commands.json` via
[bear](https://github.com/rizsotto/Bear) before running clang-tidy. Install
`bear` alongside `clang-tidy` before using it.

### Debug builds

Uncomment the matching pair of sanitizer lines in the Makefile:

**AddressSanitizer + UndefinedBehaviorSanitizer:**

    CXXFLAGS = -g -Og -fno-omit-frame-pointer -fno-lto -fsanitize=address,undefined
    LDFLAGS += -fsanitize=address,undefined

    export ASAN_OPTIONS=detect_leaks=1:abort_on_error=0:symbolize=1
    export UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1

**ThreadSanitizer** (mutually exclusive with ASan):

    CXXFLAGS = -g -Og -fno-omit-frame-pointer -fno-lto -fsanitize=thread
    LDFLAGS += -fsanitize=thread

    export TSAN_OPTIONS=halt_on_error=0:second_deadlock_stack=1:history_size=7

### Verbose logging

Enable debug-level syslog output:

    vdr -l 3 -P vaapivideo


## Roadmap

- Expose filter controls (denoise strength, sharpen level) in VDR setup menu
- Dynamic resolution switching on SD / HD / UHD channel change
- HDR10 / HLG passthrough
- GPU memory usage and ALSA underrun counters in STAT output
- Detached-state startup with deferred hardware initialization


## Credits

- **Author:** Dirk Nehring <<dnehring@gmx.net>>
- **Inspired by:** [vdr-plugin-softhdcuvid](https://github.com/jojo61/vdr-plugin-softhdcuvid)


## License

[AGPL-3.0-or-later](LICENSE) -- Copyright © 2026 Dirk Nehring.
Modified distributions must publish their source under the same terms.
