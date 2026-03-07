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

| Source File | Responsibility                                                 |
|-------------|----------------------------------------------------------------|
| audio.cpp   | ALSA output, IEC61937 passthrough                              |
| common.h    | AvErr() helper, RAII deleters, version guards, shared headers  |
| config.cpp  | Plugin configuration, display parameters, setup storage        |
| decoder.cpp | VAAPI decode, filter graphs, A/V sync                          |
| device.cpp  | VDR integration, PES routing, lifecycle                        |
| display.cpp | DRM atomic modesetting, page flips                             |
| osd.cpp     | Hardware OSD overlay                                           |
| pes.cpp     | PES parsing, codec detection                                   |


## Requirements

| Dependency   | Minimum   | Notes                                                          |
|--------------|-----------|----------------------------------------------------------------|
| Linux Kernel | 6.8+      | Atomic async page-flip, universal planes, COLOR_ENCODING/RANGE |
| VDR          | 2.7.9+    | APIVERSNUM >= 30012                                            |
| FFmpeg       | 7.0+      | libavcodec >= 61.3.100, built with VAAPI support               |
| libdrm       | 2.4.131+  |                                                                |

**Supported VAAPI drivers:**

- **Intel** (Broadwell and later): `intel-media-driver` (iHD)
- **AMD**: `mesa-va-drivers` (radeonsi)

NVIDIA GPUs are **not** supported -- the third-party `nvidia-vaapi-driver` does
not implement the Video Processing Pipeline (VPP) required by this plugin.


## Setup

### 1. Install build dependencies

**Fedora / RHEL / openSUSE:**

    sudo dnf install gcc-c++ make git \
        vdr-devel \
        libdrm-devel \
        alsa-lib-devel \
        ffmpeg-devel \
        libva-devel

**Debian / Ubuntu:**

    sudo apt install g++ make git \
        vdr-dev \
        libdrm-dev \
        libasound2-dev \
        libavcodec-dev \
        libavformat-dev \
        libavfilter-dev \
        libavutil-dev \
        libswresample-dev \
        libva-dev

### 2. Build and install

    git clone https://github.com/dnehring7/vdr-vaapivideo.git
    cd vdr-vaapivideo
    make
    sudo make install

An RPM spec file is included for Fedora/RHEL/openSUSE:

    rpmbuild -ta vdr-vaapivideo-*.tar.gz

### 3. Permissions

The user that runs VDR needs access to DRM, video render, and ALSA devices:

    sudo usermod -aG video,render,audio vdr

A logout or service restart is required for the group change to take effect.

### 4. Verify VAAPI

Confirm that the VAAPI driver is functional before starting VDR:

    vainfo --display drm --device /dev/dri/renderD128

The output must list at least one decode profile for H.264 or HEVC.
If `vainfo` fails, install the correct driver package first.

### 5. Configure the ALSA audio device

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

| Option   | Default      | Description                                         |
|----------|--------------|-----------------------------------------------------|
| `-d DEV` | auto-detect  | DRM device path (`/dev/dri/cardN`)                  |
| `-a DEV` | `default`    | ALSA audio device (use `hw:CARD,DEV` for passthrough) |
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

### Finding devices

    ls -la /dev/dri/card*               # DRM primary nodes
    aplay -l | grep -E "HDMI|Display"   # ALSA HDMI/DP outputs


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


## Credits

- **Author:** Dirk Nehring <<dnehring@gmx.net>>
- **Inspired by:** [vdr-plugin-softhdcuvid](https://github.com/jojo61/vdr-plugin-softhdcuvid)


## License

[AGPL-3.0-or-later](LICENSE) -- Copyright © 2026 Dirk Nehring.
Modified distributions must publish their source under the same terms.
