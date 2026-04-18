# VDR VAAPI Video Plugin

Hardware-accelerated video output for [VDR](https://www.tvdr.de/) using VAAPI
decode, DRM atomic modesetting, and ALSA audio. No X11, Wayland, or OpenGL is
required — the plugin runs on the bare console, in a systemd service, or fully
headless.

The video path is zero-copy: VAAPI surfaces are exported as DRM PRIME buffers
and scanned out without ever touching system memory. Audio passthrough formats
are detected automatically from the HDMI sink's EDID. Codecs that lack hardware
decode support on the host GPU (e.g. MPEG-2 on AMD) fall back to FFmpeg
software decoding transparently. The VAAPI Video Processing Pipeline (VPP)
**must** be available — the plugin will refuse to start without it.


## Features

| Component | Capabilities                                                                                       |
|-----------|----------------------------------------------------------------------------------------------------|
| Decode    | H.264, HEVC, MPEG-2 — hardware (VAAPI) with automatic per-codec software fallback                  |
| Filters   | Deinterlace, denoise, DAR-preserving scale, sharpen — SW path (bwdif, hqdn3d) or HW (VAAPI VPP)    |
| Audio     | PCM decode (AAC, MP2); IEC61937 passthrough (AC-3, E-AC-3, DTS, DTS-HD, TrueHD, AC-4, MPEG-H 3D)   |
| Display   | DRM atomic modesetting, double-buffered page-flip, BT.709 SDR, up to 3840×2160                     |
| OSD       | True-color hardware overlay on a dedicated DRM plane, alpha-blended over the video plane           |
| A/V sync  | Audio-mastered, EMA-smoothed, proportional with hard-transient bypass — see [AVSYNC.md](AVSYNC.md) |


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

| File              | Responsibility                                              |
|-------------------|-------------------------------------------------------------|
| `vaapivideo.cpp`  | Plugin entry point, VDR lifecycle, setup menu, SVDRP        |
| `src/device.cpp`  | VDR device integration, PES routing, hardware init/teardown |
| `src/decoder.cpp` | VAAPI decode, FFmpeg filter graph, A/V sync controller      |
| `src/display.cpp` | DRM atomic modesetting, PRIME import, page-flip thread      |
| `src/audio.cpp`   | ALSA output, IEC61937 passthrough, HDMI ELD/EDID probe      |
| `src/osd.cpp`     | DRM dumb-buffer OSD overlay (ARGB8888 plane)                |
| `src/pes.cpp`     | PES header parsing, video/audio codec detection             |
| `src/config.cpp`  | Resolution parsing, `setup.conf` storage                    |
| `src/common.h`    | RAII deleters, `AvErr()` helper, version/API guards         |

The A/V sync controller is documented separately in [AVSYNC.md](AVSYNC.md).
The coding conventions enforced across all sources are listed in
`.github/copilot-instructions.md`.


## Requirements

| Dependency   | Minimum | Notes                                                                      |
|--------------|---------|----------------------------------------------------------------------------|
| Linux kernel | 6.8+    | Atomic async page-flip cap, universal planes, COLOR_ENCODING / COLOR_RANGE |
| VDR          | 2.6.0+  | `APIVERSNUM >= 20600`                                                      |
| FFmpeg       | 7.0+    | `libavcodec >= 61.3.100`, built with `--enable-vaapi`                      |
| C++ compiler | C++20   | GCC 12+ or Clang 16+                                                       |

### Supported VAAPI drivers

| GPU     | Driver package                | Hardware                       |
|---------|-------------------------------|--------------------------------|
| Intel   | `intel-media-driver` (iHD)    | Broadwell and later            |
| AMD     | `mesa-va-drivers` (radeonsi)  | GCN 3 and later                |

NVIDIA GPUs are **not supported**: the third-party `nvidia-vaapi-driver` does
not implement the Video Processing Pipeline (VPP) that this plugin requires.


## Setup

### Pre-built packages

Pre-built packages for Fedora 44 and Debian 13 (Trixie) are published with
each [GitHub release](https://github.com/dnehring7/vdr-plugin-vaapivideo/releases)
and served as GPG-signed repositories via GitHub Pages. The signing key
is published at https://github.com/dnehring7.gpg (fingerprint
`617BD4B433CE88729B6D46816C7069D18A46683A`).

<details>
<summary>Fedora 44 (x86_64)</summary>

```sh
sudo dnf config-manager addrepo \
  --from-repofile=https://dnehring7.github.io/vdr-plugin-vaapivideo/fedora/44/vdr-vaapivideo.repo
sudo dnf install vdr-vaapivideo
```

</details>

<details>
<summary>Debian 13 / Trixie (amd64)</summary>

```sh
curl -fsSL https://github.com/dnehring7.gpg \
  | sudo gpg --dearmor -o /usr/share/keyrings/dnehring7.gpg
sudo curl -fsSL https://dnehring7.github.io/vdr-plugin-vaapivideo/debian/vdr-vaapivideo.sources \
  -o /etc/apt/sources.list.d/vdr-vaapivideo.sources
sudo apt update
sudo apt install vdr-plugin-vaapivideo
```

</details>

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

An RPM spec file (`vdr-vaapivideo.spec`) is included for Fedora/RHEL/openSUSE
packaging:

    rpmbuild -ta vdr-vaapivideo-*.tar.gz

### 3. Permissions

The VDR user needs access to DRM render, video, and ALSA devices:

    sudo usermod -aG video,render,audio vdr

A logout or service restart is required for group changes to take effect.

### 4. Install the VAAPI driver

The plugin requires a VAAPI driver with **Video Processing Pipeline (VPP)**
support — the source of all hardware scaling, deinterlacing, denoising, and
colorspace conversion. Initialization fails if VPP is not available.

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

A standalone diagnostic tool (not part of the plugin build) that probes decode
profiles, VPP filters, surface formats, and HDR tone mapping:

    g++ -std=c++20 $(pkg-config --cflags --libs libdrm libva libva-drm) \
        -o vaapivideo-probe vaapivideo-probe.cpp

    ./vaapivideo-probe                     # uses /dev/dri/card0
    ./vaapivideo-probe /dev/dri/card1      # explicit device

Any line showing **no** indicates a missing driver capability. Compare against
the plugin log (`vdr -l 3`) to identify mismatches.

### 6. Configure the ALSA audio device

The default ALSA device is `default` (stereo PCM). For IEC61937 passthrough use
a direct hardware device — supported passthrough formats are detected from the
sink's EDID at startup:

    aplay -l | grep -E "HDMI|DisplayPort"
    vdr -P 'vaapivideo -a hw:0,3'


## Configuration

### Command-line options

    vdr -P 'vaapivideo [-d DEV] [-a DEV] [-c NAME] [-r WxH@R]'

| Option     | Default           | Description                                           |
|------------|-------------------|-------------------------------------------------------|
| `-d DEV`   | auto-detect       | DRM device path (`/dev/dri/cardN`)                    |
| `-a DEV`   | `default`         | ALSA audio device (use `hw:CARD,DEV` for passthrough) |
| `-c NAME`  | first connected   | DRM connector name (e.g. `HDMI-A-1`, `DP-2`)          |
| `-r WxH@R` | `1920x1080@50`    | Output resolution and refresh rate (max 3840×2160)    |

Use `-d` explicitly when multiple GPUs are present. Use `-c` to select a
specific output when multiple displays are connected — connector names match
the kernel's naming scheme visible under `/sys/class/drm/`.

### VDR setup menu

    Setup → Plugins → vaapivideo

| Setting                          | Range             | Description                                                       |
|----------------------------------|-------------------|-------------------------------------------------------------------|
| `PCM Audio Latency (ms)`         | −200 … 200        | A/V offset applied when audio is decoded to PCM by the plugin     |
| `Passthrough Audio Latency (ms)` | −200 … 200        | A/V offset applied when audio is forwarded as IEC61937 to an AVR  |
| `Audio Passthrough`              | auto / on / off   | IEC61937 passthrough policy (see below)                           |
| `Clear display on channel switch`| off / on          | Paint a black frame on channel switch instead of leaving the previous channel's last frame on screen |

The two latency knobs are split because a downstream receiver doing its own
bitstream decode contributes a different delay than the PCM path. Both default
to **0 ms** — adjust only if a residual offset is visible after the controller
has settled. See [AVSYNC.md](AVSYNC.md) for the full sign convention and tuning
guidance.

`Clear display on channel switch` defaults to **off**: on `SetPlayMode(pmNone)`
the DRM scanout keeps the last decoded frame until the new channel produces
its first picture. Enable it to blank the screen between channels — the plugin
submits a BT.709 TV-range black VAAPI surface through the normal display path
right after the teardown. Radio (`pmAudioOnly`) always paints black and is not
affected by this setting.

`Audio Passthrough` defaults to **auto**: the plugin reads the HDMI sink's ELD
at startup and forwards a compressed codec as IEC61937 only when the sink
advertises support for it — everything else is decoded to stereo PCM. Use
**on** to unconditionally force passthrough for every wrappable codec (AC-3,
E-AC-3, TrueHD, DTS, AC-4, MPEG-H 3D Audio) and **ignore the ELD entirely**.
This is the knob for topologies where the probed capabilities are wrong — the
typical case being an AVR behind a TV, where the TV's EDID masks the AVR's
real decoder support. When **on** overrides a negative ELD, the plugin logs
a `PassthroughMode=on overriding ELD for X (sink advertises no support); …`
line so the override is visible in the journal. Codecs without IEC61937
framing (AAC, MP2, …) are always decoded to PCM. Use **off** to disable
passthrough entirely and always decode to PCM — convenient for sinks that
only accept stereo PCM or for troubleshooting.

Note that ALSA cannot signal a silent decode failure at the sink. The
`default` device in particular is a plug wrapper that accepts IEC61937 bursts
as plain S16LE PCM; if the real downstream device cannot decode the burst
you will simply hear nothing. When using **on**, make sure the downstream
device really does decode the codec — otherwise switch back to **auto** or
**off**.

Changes to this setting only take effect when the audio device is reopened —
i.e. on the next channel switch or codec change. Switch channels once after
leaving the setup menu to activate the new mode.

### SVDRP commands

| Command                  | Description                                              |
|--------------------------|----------------------------------------------------------|
| `PLUG vaapivideo STAT`   | Device status, active resolution, refresh rate           |
| `PLUG vaapivideo CONFIG` | Current configuration summary                            |
| `PLUG vaapivideo DETA`   | Detach from DRM/VAAPI hardware (release for other apps)  |
| `PLUG vaapivideo ATTA`   | Re-attach to DRM/VAAPI hardware and resume the pipeline  |

DETA hands the display to another application (an external player, a
diagnostic tool, etc.) and ATTA reclaims it without restarting VDR. ATTA also
forces a channel re-tune so data flows through the freshly initialized
decoder/display pipeline.

### Inter-plugin service API

Other plugins can query device state via VDR's `cPlugin::Service()` interface:

| Service ID                   | Data type   | Description                                  |
|------------------------------|-------------|----------------------------------------------|
| `VaapiVideo-Available-v1.0`  | `bool *`    | `true` if a hardware decoder is ready        |
| `VaapiVideo-IsReady-v1.0`    | `bool *`    | `true` if the device is fully initialized    |
| `VaapiVideo-DeviceType-v1.0` | `cString *` | Human-readable device type string            |

Passing `data == nullptr` acts as a capability probe — `Service()` returns
`true` for any known ID without writing to the buffer.


## Troubleshooting

| Symptom                   | Diagnosis and fix                                                          |
|---------------------------|----------------------------------------------------------------------------|
| Plugin refuses to start   | VPP missing — run `vainfo` and check for `VAEntrypointVideoProc`           |
| No video output           | Verify group membership (`video`, `render`); run `vainfo`                  |
| No audio                  | Test ALSA directly: `speaker-test -D hw:0,3 -c 2 -r 48000 -t sine -l 1`    |
| Passthrough not working   | Use `hw:CARD,DEV`; verify `/proc/asound/card0/eld#0.N` is non-empty        |
| Persistent A/V drift      | Tune `PCM Audio Latency` or `Passthrough Audio Latency`                    |
| DRM device not found      | Run `ls -l /dev/dri/` and specify `-d /dev/dri/cardN` explicitly           |
| Black screen after resume | Send SVDRP: `PLUG vaapivideo DETA` then `PLUG vaapivideo ATTA`             |

Increase the VDR log verbosity with `-l 3` to capture decoder, display, and
sync diagnostics; the periodic `sync d=… avg=…` line is described in
[AVSYNC.md](AVSYNC.md#log-format).


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

`make lint` invokes [bear](https://github.com/rizsotto/Bear) to produce
`compile_commands.json` and then runs clang-tidy across all sources.

### Debug builds

Uncomment the matching sanitizer block in the Makefile (ASan + UBSan **or**
TSan — they are mutually exclusive). The Makefile comments document the
runtime environment variables.

### Coding conventions

The project enforces a strict modern-C++ style: trailing return types on every
function, `[[nodiscard]]` on value-returning functions, RAII for every C-API
resource, `std::format`/`std::span` over their C equivalents, and VDR
threading primitives (`cThread`, `cMutex`, `cCondVar`) over `std::thread` /
`std::mutex`. The full rules and rationale are in
[CLAUDE.md](CLAUDE.md) / `.github/copilot-instructions.md`.

### Verbose logging

    vdr -l 3 -P vaapivideo


## Roadmap

- Expose filter controls (denoise strength, sharpen level) in the VDR setup menu
- Dynamic resolution switching on SD / HD / UHD channel changes
- HDR10 / HLG passthrough
- GPU memory and ALSA underrun counters in `STAT` output
- Detached-state startup with deferred hardware initialization


## Credits

- **Author:** Dirk Nehring &lt;<dnehring@gmx.net>&gt;
- **Inspired by:** [vdr-plugin-softhdcuvid](https://github.com/jojo61/vdr-plugin-softhdcuvid)


## License

[AGPL-3.0-or-later](LICENSE) — Copyright © 2026 Dirk Nehring.
Modified distributions must publish their source under the same terms.
