# VDR VAAPI Video Plugin

Hardware-accelerated video output for [VDR](https://www.tvdr.de/) using VAAPI
decode, DRM atomic modesetting, and ALSA audio -- no X11, Wayland, or OpenGL
required. Runs on a bare console, in a systemd service, or headless.

The video path is zero-copy: VAAPI surfaces are exported as DRM PRIME buffers
and scanned out without touching system memory. Audio passthrough formats are
detected automatically from the HDMI sink's EDID. Codecs lacking hardware
decode support (e.g. MPEG-2 on AMD) fall back to FFmpeg software decoding
transparently. The VAAPI Video Processing Pipeline (VPP) **must** be
available -- the plugin will not start without it.


## Features

| Component | Capabilities                                                                                       |
|-----------|----------------------------------------------------------------------------------------------------|
| Decode    | H.264, HEVC, MPEG-2 -- hardware (VAAPI) with automatic software fallback per codec                 |
| Filters   | Deinterlace, denoise, scale (DAR-preserving), sharpen -- SW path (bwdif, hqdn3d) or HW (VAAPI VPP) |
| Audio     | PCM decode (AAC, MP2); IEC61937 passthrough (AC-3, E-AC-3, DTS, DTS-HD, TrueHD, AC-4, MPEG-H)      |
| Display   | DRM atomic modesetting, double-buffered page-flip, BT.709 SDR; up to 3840x2160                     |
| OSD       | True-color hardware overlay on a dedicated DRM plane, alpha-blended                                |


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

### Source layout

| File             | Responsibility                                              |
|------------------|-------------------------------------------------------------|
| `vaapivideo.cpp` | Plugin entry point, VDR lifecycle, setup menu, SVDRP        |
| `src/device.cpp` | VDR device integration, PES routing, hardware init/teardown |
| `src/decoder.cpp`| VAAPI decode, FFmpeg filter graph, A/V sync                 |
| `src/display.cpp`| DRM atomic modesetting, PRIME import, page-flip thread      |
| `src/audio.cpp`  | ALSA output, IEC61937 passthrough, HDMI ELD/EDID probe      |
| `src/osd.cpp`    | DRM dumb-buffer OSD overlay (ARGB8888 plane)                |
| `src/pes.cpp`    | PES header parsing, video/audio codec detection             |
| `src/config.cpp` | Resolution parsing, setup.conf storage                      |
| `src/common.h`   | RAII deleters, `AvErr()` helper, version/API guards         |


## Requirements

| Dependency   | Minimum | Notes                                                          |
|--------------|---------|----------------------------------------------------------------|
| Linux kernel | 6.8+    | Atomic async page-flip, universal planes, COLOR_ENCODING/RANGE |
| VDR          | 2.6.0+  | `APIVERSNUM >= 20600`                                          |
| FFmpeg       | 7.0+    | `libavcodec >= 61.3.100`, built with `--enable-vaapi`          |
| C++ compiler |         | C++20 support required (GCC 12+, Clang 16+)                    |

### Supported VAAPI drivers

| GPU     | Driver package                | Backend             |
|---------|-------------------------------|---------------------|
| Intel   | `intel-media-driver` (iHD)    | Broadwell and later |
| AMD     | `mesa-va-drivers` (radeonsi)  | GCN 3 and later     |

NVIDIA GPUs are **not supported** -- the third-party `nvidia-vaapi-driver` does
not implement the Video Processing Pipeline (VPP) required by this plugin.


## Setup

### 1. Install build dependencies

<details>
<summary>Fedora / RHEL / openSUSE</summary>

    dnf install gcc-c++ make git pkgconf \
        vdr-devel \
        libdrm-devel \
        alsa-lib-devel \
        ffmpeg-devel \
        libva-devel

</details>

<details>
<summary>Debian / Ubuntu</summary>

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

</details>

<details>
<summary>Gentoo</summary>

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

</details>

### 2. Build and install

    git clone https://github.com/dnehring7/vdr-plugin-vaapivideo.git
    cd vdr-plugin-vaapivideo
    make
    sudo make install

An RPM spec file (`vdr-vaapivideo.spec`) is included for Fedora/RHEL/openSUSE:

    rpmbuild -ta vdr-vaapivideo-*.tar.gz

### 3. Permissions

The VDR user needs access to DRM, video render, and ALSA devices:

    sudo usermod -aG video,render,audio vdr

A logout or service restart is required for group changes to take effect.

### 4. Install the VAAPI driver

The plugin requires a VAAPI driver with **Video Processing Pipeline (VPP)**
support (hardware scaling, deinterlacing, denoising, colorspace conversion).
The plugin will refuse to start if VPP is not available.

<details>
<summary>Fedora / RHEL / openSUSE</summary>

    dnf install intel-media-driver          # Intel (Broadwell+)
    dnf install mesa-va-drivers-freeworld   # AMD (radeonsi)

</details>

<details>
<summary>Debian / Ubuntu</summary>

    apt install intel-media-va-driver                   # Intel (Broadwell+)
    apt install mesa-va-drivers firmware-amd-graphics   # AMD (radeonsi)

</details>

<details>
<summary>Gentoo</summary>

    emerge -av media-libs/intel-media-driver                        # Intel (Broadwell+)
    USE="vaapi" VIDEO_CARDS="radeonsi" emerge -av media-libs/mesa   # AMD (radeonsi)

</details>

### 5. Verify VAAPI

Run `vainfo` to confirm the driver is loaded and VPP is available:

    vainfo --display drm --device /dev/dri/renderD128

Look for the VPP entry point in the output:

    VAProfileNone                   : VAEntrypointVideoProc

If this line is missing, the plugin will not start. Verify the correct driver
is installed (step 4) and that the user has access to the render node (step 3).

#### vaapivideo-probe

Standalone diagnostic tool (not part of the plugin build) that probes decode
profiles, VPP filters, surface formats, and HDR tone mapping:

    g++ -std=c++20 $(pkg-config --cflags --libs libdrm libva libva-drm) \
        -o vaapivideo-probe vaapivideo-probe.cpp

    ./vaapivideo-probe                     # uses /dev/dri/card0
    ./vaapivideo-probe /dev/dri/card1      # explicit device

Any line showing **no** indicates a missing driver capability. Compare against
the plugin log (`vdr -l 3`) to identify mismatches.

### 6. Configure the ALSA audio device

Defaults to ALSA `default` (stereo PCM). For IEC61937 passthrough use a direct
hardware device -- passthrough formats are detected from the sink's EDID:

    aplay -l | grep -E "HDMI|DisplayPort"
    vdr -P 'vaapivideo -a hw:0,3'


## Configuration

### Command-line options

    vdr -P 'vaapivideo [-d DEV] [-a DEV] [-r WxH@R]'

| Option      | Default        | Description                                           |
|-------------|----------------|-------------------------------------------------------|
| `-d DEV`    | auto-detect    | DRM device path (`/dev/dri/cardN`)                    |
| `-a DEV`    | `default`      | ALSA audio device (use `hw:CARD,DEV` for passthrough) |
| `-r WxH@R`  | `1920x1080@50` | Output resolution and refresh rate (max 3840x2160)    |

Use `-d` explicitly when multiple GPUs are present.

### VDR setup menu

    Setup -> Plugins -> vaapivideo

| Setting            | Range    | Description                                      |
|--------------------|----------|--------------------------------------------------|
| Audio Latency (ms) | 0 - 200 | A/V offset to compensate for external audio delay |

### SVDRP commands

| Command                  | Description                                            |
|--------------------------|--------------------------------------------------------|
| `PLUG vaapivideo STAT`   | Device status, active resolution, refresh rate         |
| `PLUG vaapivideo CONFIG` | Current configuration summary                          |
| `PLUG vaapivideo DETA`   | Detach from DRM/VAAPI hardware (release for other use) |
| `PLUG vaapivideo ATTA`   | Re-attach to DRM/VAAPI hardware and restart pipeline   |

DETA/ATTA hands the display to another application and reclaims it without
restarting VDR.

### Inter-plugin service API

Query device state via VDR's `cPlugin::Service()` interface:

| Service ID                   | Data type  | Description                               |
|------------------------------|------------|-------------------------------------------|
| `VaapiVideo-Available-v1.0`  | `bool*`    | `true` if a hardware decoder is ready     |
| `VaapiVideo-IsReady-v1.0`    | `bool*`    | `true` if the device is fully initialized |
| `VaapiVideo-DeviceType-v1.0` | `cString*` | Human-readable device type string         |

`data = nullptr` acts as a capability probe (`true` for known IDs).


## Troubleshooting

| Symptom                   | Diagnosis and fix                                                       |
|---------------------------|-------------------------------------------------------------------------|
| No video output           | Check group membership (`video`, `render`) and run `vainfo`             |
| No audio                  | Test ALSA directly: `speaker-test -D hw:0,3 -c 2 -r 48000 -t sine -l 1` |
| Passthrough not working   | Use `hw:CARD,DEV`; verify `/proc/asound/card0/eld#0.N` is non-empty     |
| A/V drift                 | Increase Audio Latency in setup menu (positive = delay audio)           |
| DRM device not found      | Run `ls -l /dev/dri/` and specify `-d /dev/dri/cardN` explicitly        |
| Black screen after resume | Send SVDRP: `PLUG vaapivideo DETA` then `PLUG vaapivideo ATTA`          |
| Plugin refuses to start   | VPP missing -- run `vainfo` and check for `VAEntrypointVideoProc`       |


## Development

### Build targets

| Target         | Description                                     |
|----------------|-------------------------------------------------|
| `make`         | Release build (`-O3`, LTO, strip)               |
| `make install` | Install plugin to VDR plugin directory          |
| `make clean`   | Remove build artifacts                          |
| `make dist`    | Create source tarball                           |
| `make indent`  | Format sources with clang-format                |
| `make lint`    | Static analysis with clang-tidy (requires bear) |
| `make docs`    | Generate Doxygen HTML documentation             |

`make lint` generates `compile_commands.json` via
[bear](https://github.com/rizsotto/Bear) before running clang-tidy.

### Debug builds

Uncomment the matching sanitizer lines in the Makefile (ASan+UBSan **or**
TSan -- mutually exclusive). See the Makefile comments for runtime env vars.

### Verbose logging

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

[AGPL-3.0-or-later](LICENSE) -- Copyright (C) 2026 Dirk Nehring.
Modified distributions must publish their source under the same terms.
