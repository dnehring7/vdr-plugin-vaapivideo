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

| Component   | Capabilities                                                                                       |
|-------------|----------------------------------------------------------------------------------------------------|
| Decode      | MPEG-2, H.264 (incl. High 10), HEVC (incl. Main 10), AV1 Main / Main 10 — hardware (VAAPI) with per-profile software fallback |
| Filters     | Deinterlace, denoise, DAR-preserving scale, sharpen — SW path (bwdif, hqdn3d) or HW (VAAPI VPP)    |
| Audio       | PCM decode (AAC, MP2); IEC61937 passthrough (AC-3, E-AC-3, DTS, TrueHD, AC-4, MPEG-H 3D)           |
| Display     | DRM atomic modesetting, double-buffered page-flip, BT.709 SDR + BT.2020 HDR10/HLG passthrough      |
| OSD         | True-color hardware overlay on a dedicated DRM plane, alpha-blended over the video plane           |
| Mediaplayer | Local files (MP4, MKV, TS, WebM, …), http(s)/ftp URLs, m3u/m3u8 playlists, runtime audio-track switching — see [Mediaplayer](#mediaplayer) |
| A/V sync    | Audio-mastered, EMA-smoothed, proportional with hard-transient bypass — see [AVSYNC.md](AVSYNC.md) |


## Architecture

```
VDR live/replay ──PES──▶ cVaapiDevice ──▶ PES Parser ─┐
                                                       │
Mediaplayer ──libavformat──▶ AVPacket ─────────────────┼──▶ cVaapiDecoder
                                                       │
                                          ┌────────────┴────────────┐
                                          ▼                         ▼
                                    VAAPI HW Decode          FFmpeg SW Decode
                                          │                         │
                                          ▼                         ▼
                                    VAAPI VPP Filters     SW Filters (bwdif, hqdn3d)
                                 (deinterlace, denoise)        + hwupload
                                          │                         │
                                          └────────────┬────────────┘
                                                       ▼
                                                  scale_vaapi
                                              + sharpness_vaapi
                                     (SDR: BT.709 NV12; HDR: BT.2020 P010)
                                                       │
                                                       ▼
                                           DRM PRIME (zero-copy)
                                                       │
                                          ┌────────────┴────────────┐
                                          ▼                         ▼
                                     Video Plane             OSD Plane (ARGB8888)
                                (NV12 SDR / P010 HDR)
                                          │                         │
                                          └────────────┬────────────┘
                                                       ▼
                                          DRM Atomic Page-Flip ──▶ Display
```

Two input paths share the decoder/filter/display pipeline unchanged: VDR's live and
replay traffic enters via PES through `cVaapiDevice::PlayVideo` / `PlayAudio`; the
integrated mediaplayer demuxes files and URLs with libavformat and pushes
pre-framed access units straight into the decoder via a narrow feed surface
(`SubmitVideoPacket` / `SubmitAudioPacket`). Codec selection, HDR routing, A/V
sync — all path-agnostic.

Inside `cVaapiDecoder`, decode and presentation run on **separate threads**: the
decode thread filters frames into a decode-ahead reserve, and a presentation
thread drains that reserve at the audio-synced cadence, so a slow 4K VPP step
spends the reserve instead of stalling the screen. See
[AVSYNC.md → Decode / present decouple](AVSYNC.md#decode--present-decouple).

### Source layout

| File                  | Responsibility                                                                        |
|-----------------------|---------------------------------------------------------------------------------------|
| `vaapivideo.cpp`      | Plugin entry point, VDR lifecycle, setup menu, SVDRP, main-menu hook                  |
| `src/device.cpp`      | VDR device integration, PES routing, hardware init/teardown, mediaplayer feed surface |
| `src/decoder.cpp`     | Decoupled VAAPI decode + presentation threads, A/V sync controller                     |
| `src/filter.cpp`      | FFmpeg filter-graph build (deinterlace / denoise / scale / sharpen; HW and SW chains) |
| `src/display.cpp`     | DRM atomic modesetting, PRIME import, page-flip thread                                |
| `src/audio.cpp`       | ALSA output, IEC61937 passthrough, HDMI ELD/EDID probe                                |
| `src/osd.cpp`         | DRM dumb-buffer OSD overlay (ARGB8888 plane)                                          |
| `src/mediaplayer.cpp` | libavformat demux, file browser, cControl with OSD replay bar                         |
| `src/stream.cpp`      | Shared codec/profile data model, H.264/HEVC SPS probe                                 |
| `src/pes.cpp`         | PES header parsing                                                                    |
| `src/caps.cpp`        | One-shot GPU/display/sink capability probes (GpuCaps, DisplayCaps, AudioSinkCaps)     |
| `src/config.cpp`      | Resolution parsing, `setup.conf` storage                                              |
| `src/common.h`        | RAII deleters, `AvErr()` helper, version/API guards                                   |

The A/V sync controller is documented separately in [AVSYNC.md](AVSYNC.md).
The coding conventions enforced across all sources are listed in
`.github/copilot-instructions.md`.


## Requirements

| Dependency   | Minimum | Notes                                                                                   |
|--------------|---------|-----------------------------------------------------------------------------------------|
| Linux kernel | 5.15+   | DRM atomic modeset, universal planes, COLOR_ENCODING / COLOR_RANGE, HDR_OUTPUT_METADATA |
| VDR          | 2.6.6+  | `APIVERSNUM >= 20606`                                                                   |
| FFmpeg       | 7.0+    | `libavcodec >= 61.3.100`, built with `--enable-vaapi`                                   |
| libva        | 1.22+   | `VAProfileVVCMain10` is unconditionally referenced                                      |
| C++ compiler | C++20   | GCC 12+ or Clang 16+                                                                    |

### Supported VAAPI drivers

| GPU     | Driver package                | Hardware                       |
|---------|-------------------------------|--------------------------------|
| Intel   | `intel-media-driver` (iHD)    | Broadwell and later            |
| AMD     | `mesa-va-drivers` (radeonsi)  | GCN 3 and later                |

NVIDIA GPUs are **not supported**: the third-party `nvidia-vaapi-driver` does
not implement the Video Processing Pipeline (VPP) that this plugin requires.


## Setup

### Pre-built packages

Signed Fedora 44, Debian 13, and Ubuntu 26.04 LTS package repositories are
published on every
[GitHub release](https://github.com/dnehring7/vdr-plugin-vaapivideo/releases)
and served via GitHub Pages. All configs reference the signing key at
<https://github.com/dnehring7.gpg> — DNF fetches it on first install; the
APT `.sources` files ship the key inline (modern DEB822 `Signed-By:`).

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
sudo curl -fsSL https://dnehring7.github.io/vdr-plugin-vaapivideo/debian/vdr-vaapivideo.sources \
  -o /etc/apt/sources.list.d/vdr-vaapivideo.sources
sudo apt update
sudo apt install vdr-plugin-vaapivideo
```

</details>

<details>
<summary>Ubuntu 26.04 LTS / Resolute Raccoon (amd64)</summary>

```sh
sudo curl -fsSL https://dnehring7.github.io/vdr-plugin-vaapivideo/ubuntu/vdr-vaapivideo.sources \
  -o /etc/apt/sources.list.d/vdr-vaapivideo.sources
sudo apt update
sudo apt install vdr-plugin-vaapivideo
```

The Ubuntu build links against FFmpeg 8 (libavcodec62) and Ubuntu's
`vdr-dev` 2.6.9, and is a separate ABI from the Debian Trixie build —
install one or the other, not both.

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

A standalone diagnostic tool that probes decode profiles, VPP filters, surface
formats, and HDR tone mapping. Built on demand — it is not part of `make`:

    make probe
    ./vaapivideo-probe                     # uses /dev/dri/card0
    ./vaapivideo-probe /dev/dri/card1      # explicit device

For each connected output it also parses the **sink EDID** and reports what the
display advertises — HDR10 (PQ), HLG, BT.2020 Y'CbCr, HDR10+, desired max
luminance, and a Dolby Vision VSVDB — under *Sink HDR (EDID CTA-861)*. This is
the display half of the HDR decision: the plugin's `auto` HDR10 gate needs both
*HDR10 / PQ* and *BT.2020 Y'CbCr* to read **yes** here, and the probe flags that
correlation directly. The Dolby Vision line is informational only — it shows
whether the panel supports DV; the plugin cannot decode it (see
[Dolby Vision and the VAAPI limit](#dolby-vision-and-the-vaapi-limit)).

Any line showing **no** indicates a missing driver or sink capability. Compare
against the plugin log (`vdr -l 3`) to identify mismatches.

### 6. Configure the ALSA audio device

The default ALSA device is `default` (stereo PCM). For IEC61937 passthrough use
a direct hardware device — supported passthrough formats are detected from the
sink's EDID at startup:

    aplay -l | grep -E "HDMI|DisplayPort"
    vdr -P 'vaapivideo -a hw:0,3'


## Configuration

### Command-line options

    vdr -P 'vaapivideo [-a DEV] [-c NAME] [-D] [-d DEV] [-m DIR] [-r WxH@R]'

| Option                           | Default         | Description                                           |
|----------------------------------|-----------------|-------------------------------------------------------|
| `-a DEV`, `--audio=DEV`          | `default`       | ALSA audio device (use `hw:CARD,DEV` for passthrough) |
| `-c NAME`, `--connector=NAME`    | first connected | DRM connector name (e.g. `HDMI-A-1`, `DP-2`)          |
| `-D`, `--detached`               | off             | Start without opening the DRM/VAAPI/ALSA hardware     |
| `-d DEV`, `--drm=DEV`            | auto-detect     | DRM device path (`/dev/dri/cardN`)                    |
| `-m DIR`, `--media-dir=DIR`      | `/`             | Mediaplayer file-browser root directory               |
| `-r WxH@R`, `--resolution=WxH@R` | `1920x1080@50`  | Output resolution and refresh rate (max 3840×2160)    |

Use `-d` explicitly when multiple GPUs are present. Use `-c` to select a
specific output when multiple displays are connected — connector names match
the kernel's naming scheme visible under `/sys/class/drm/`.

`--detached` brings VDR up without grabbing the GPU, the DRM master, or the
ALSA device. The plugin stays loaded but idle; hardware initialization runs on
the first primary-device promotion (e.g. `Setup → OSD → Primary DVB interface`)
or when `PLUG vaapivideo ATTA` is issued via SVDRP. Useful for hosts that want
to yield the display to another application at boot, or for systemd units
that start VDR before a user session claims the console.

### VDR setup menu

    Setup → Plugins → vaapivideo

| Setting                          | Range            | Description                                                                                          |
|----------------------------------|------------------|------------------------------------------------------------------------------------------------------|
| `PCM Audio Latency (ms)`         | −200 … 200       | A/V offset applied when audio is decoded to PCM by the plugin                                        |
| `Passthrough Audio Latency (ms)` | −200 … 200       | A/V offset applied when audio is forwarded as IEC61937 to an AVR                                     |
| `Audio Passthrough`              | auto / on / off  | IEC61937 passthrough policy (see below)                                                              |
| `HDR Passthrough`                | auto / on / off  | HDR10 / HLG BT.2020 + P010 output policy (see [HDR passthrough](#hdr-passthrough))                   |
| `Clear display on channel switch`| off / on         | Paint a black frame on channel switch instead of leaving the previous channel's last frame on screen |
| `Zoom level N (0.1% larger, 0=off)` | 0 … 499       | Level N (1–5): zoom-in factor in tenths-of-% (`344` = +34.4%, picture enlarged 1.34×); 0 disables the level (skipped while cycling) |
| `Deinterlace`                    | auto / hardware: motion adaptive / weave / bob / software: bwdif / w3fdif | Deinterlacer policy (see [Post-processing](#post-processing)) |
| `Denoise`                        | auto (hardware) / off / software: light / strong | `auto` = HW `denoise_vaapi`; `software:` = `hqdn3d` presets (force the SW block) |
| `Sharpen`                        | auto (hardware) / off / software: mild / medium | `auto` = HW `sharpness_vaapi`; `software:` = `unsharp` presets (force the SW block) |
| `Scaling`                        | auto (hardware, HQ) / hardware: fast / software: HQ / software: fast | `scale_vaapi:mode=hq` / `scale_vaapi` / `swscale` lanczos / `swscale` bilinear |

The two latency knobs are split because a downstream receiver doing its own
bitstream decode contributes a different delay than the PCM path. Both default
to **0 ms** — adjust only if a residual offset is visible after the controller
has settled. See [AVSYNC.md](AVSYNC.md) for the full sign convention and tuning
guidance.

`Clear display on channel switch` defaults to **off**: the screen keeps the last
decoded frame until the new channel produces its first picture. Enable it to
blank the screen between channels instead. Radio channels always blank,
regardless of this setting.

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

### Manual zoom

Five **zoom levels** let you magnify the picture to fill the screen — useful for
cropping away the black bars that broadcasters bake into the frame (2.39:1 scope,
2.00:1, and similar). Each level is a **zoom-in factor**: the picture is enlarged
uniformly (aspect preserved) and the overflow is cropped equally off all sides.
The value is in tenths-of-a-percent of enlargement, so `344` = **+34.4%** (the
picture is 1.34× its size). The crop is rounded to the nearest 2-pixel-aligned
rectangle (NV12/P010 chroma alignment), so the realised factor matches the
configured one to within a pixel; a residual ≤1% gap to a full-screen fit is
absorbed by a single uniform stretch (imperceptible — genuine letterbox is far
larger and stays untouched). Out of the box, level 1 is **+34.4%** (fills 2.39:1
CinemaScope) and level 2 **+12.5%** (fills 2.00:1) on a 16:9 screen; levels 3–5
are off. The maximum is **+49.9%** (1.5×).

Cycling steps **Off → 1 → 2 → 3 → 4 → 5 → Off**, but **levels set to 0 are skipped**, so
if you only want one zoom level, set the other four to `0` and the key toggles
Off ↔ that level. The active stop is **transient and personal**: it is never written
to `setup.conf` and resets to **Off** automatically on every content change (plugin
start, SVDRP `ATTA`, channel switch / replay start, and each mediaplayer file). Only
the five level *definitions* persist.

Cycling the zoom:

- **Mediaplayer replay** — the **Blue** key cycles zoom and flashes the new level
  on the OSD.
- **Live TV** — VDR routes no live-TV keypresses to output plugins, so the plugin's
  single main-menu hook (`@vaapivideo`) does it. It always opens a two-line menu —
  **Zoom** (OK cycles one stop and closes the menu, flashing the new level) and
  **Mediaplayer** (OK opens the browser) — so one hook reaches both. Bind a key to it
  in `keymacros.conf`; VDR can append the follow-up keypresses, giving you one key per
  action:

      Blue      @vaapivideo Ok          # open menu, cycle zoom, menu closes itself
      Yellow    @vaapivideo Down Ok     # open menu, go to Mediaplayer, open browser

  Or just `Blue @vaapivideo` to open the menu and navigate by hand. The
  `PLUG vaapivideo ZOOM [next|0-5]` SVDRP command remains available for scripting.

### Post-processing

Four independent policies control deinterlace, denoise, sharpen, and scaling. They
all default to **auto**, which keeps the zero-copy VAAPI VPP path (the original
behaviour). Decoding always stays on the GPU; these options only shape the
post-processing that follows it.

The chain runs in one of two **domains**, chosen automatically:

- **GPU domain** (all `auto` / `hardware:` choices) — `deinterlace_vaapi` → `denoise_vaapi`
  → `scale_vaapi` → `sharpness_vaapi`, entirely on the GPU.
- **SW block** (any `software:` choice) — the decoded surface is pulled to
  system memory once (`hwdownload`), the whole post-process runs in software
  (`bwdif`/`w3fdif` → `hqdn3d` → `swscale` lanczos → `unsharp`), then a single
  `hwupload` puts it back on a VAAPI surface for display. This is the high-quality
  path for interlaced DVB broadcast on GPUs whose VAAPI deinterlacer is weak (notably
  AMD/Mesa, where the hardware deinterlacer leaves visible combing). HW decode is
  retained, so only the post-processing costs CPU. The SW block is automatically
  bypassed (forced back to the GPU domain) for HDR, UHD, and trick-play/still frames.

Each option's label says where it runs: `auto` and `hardware:` stay on the GPU VPP
path; `software:` routes the whole post-process through the SW block.

**Deinterlace**

- **auto** — best deinterlace mode the driver advertises (motion-compensated when
  present). This reproduces the original behaviour exactly.
- **hardware: motion adaptive / weave / bob** — request a specific VAAPI mode (there is
  no separate "motion compensated" entry — `auto` already picks it when available). The
  request is clamped to a mode the driver actually advertises (some iHD GPUs expose just
  one); the advertised set is logged at startup (`vaapivideo/caps: VPP … deinterlace=…`).
- **software: bwdif / w3fdif** — software deinterlace (field-rate). `bwdif` is the
  recommended quality choice for broadcast; `w3fdif` (simple 3-tap) is a **lighter**
  alternative that costs less CPU. (`yadif` is used internally only as a fallback if
  `bwdif` is unavailable.) The software filters are slice-threaded across cores so a
  1080i50 deinterlace doesn't peg a single core.

**Denoise** — `auto (hardware)` is HW `denoise_vaapi` (codec-tuned, heavier for MPEG-2);
`off` skips it; `software: light` / `strong` are `hqdn3d` presets that route the chain
through the SW block. `auto` is *HW-only*: it adds no software denoise, so inside the SW
block it does nothing — pick a `software:` preset if you want denoise there.

**Sharpen** — `auto (hardware)` is HW `sharpness_vaapi` (codec-tuned); `off` skips it;
`software: mild` / `medium` are `unsharp` presets that route the chain through the SW
block. Like denoise, `auto` adds no software sharpen inside the SW block.

**Scaling** — `auto (hardware, HQ)` uses `scale_vaapi:mode=hq` (bicubic); `hardware:
fast` drops `mode=hq` for a cheaper GPU scale; `software: HQ` uses `swscale` lanczos
and `software: fast` uses `swscale` bilinear (both route through the SW block). In the
SW block the software scale is emitted only when it does real work (a resize, a manual
zoom, or a BT.601→BT.709 / range conversion); a 1080i broadcast already at 1920×1080
BT.709 skips it. `scale_vaapi` always normalises pixel format and colorimetry on the GPU
path. The `software:` scalers are niche — `auto` (hardware `scale_vaapi`) is both cheaper
and better; reach for them only when a GPU's `scale_vaapi` itself is suspect.

> **CPU cost:** the SW block runs the deinterlacer at *field rate* (e.g. 1080i50 →
> 1080p50), so every following software node (`hqdn3d`, `swscale`, `unsharp`) also
> processes 50 frames/s. On a low-power CPU, `software: strong` denoise plus
> `software: bwdif` on 1080i50 can saturate cores and drop frames. If playback can't
> keep up, set denoise to `off`, leave sharpen on `auto`, or — on a GPU with a good
> hardware deinterlacer (Intel iHD) — use `Deinterlace = auto` and stay on the GPU path.

These apply to **live playback immediately**: on leaving the setup menu the plugin
rebuilds the filter graph (the same path a zoom edit uses), so there is no need to
switch channels. Switching `auto`↔`software:` is a filter-graph rebuild only — decode
stays on the GPU, so there is no channel re-tune or black flash. The options only
ever affect the video filter chain — decoding, audio, HDR, and zoom are untouched.

### SVDRP commands

| Command                        | Description                                                |
|--------------------------------|------------------------------------------------------------|
| `PLUG vaapivideo STAT`         | Device status, active resolution, refresh rate             |
| `PLUG vaapivideo CONF`         | Current configuration summary                              |
| `PLUG vaapivideo DETA`         | Detach from DRM/VAAPI hardware (release for other apps)    |
| `PLUG vaapivideo ATTA`         | Re-attach to DRM/VAAPI hardware; if primary, resume output |
| `PLUG vaapivideo PLAY <uri>`   | Start mediaplayer on a file, URL, or `.m3u/.m3u8` playlist |
| `PLUG vaapivideo ZOOM [next\|0-5]` | Cycle manual zoom (`next`) or select a stop (0 = off, 1–5 = preset) |

DETA hands the display to another application (an external player, a
diagnostic tool, etc.) and ATTA reclaims it without restarting VDR. When the
VAAPI device is the current primary device, ATTA also forces a channel
re-tune so data flows through the freshly initialized decoder/display
pipeline.

PLAY launches the integrated mediaplayer — see [Mediaplayer](#mediaplayer)
for accepted URI forms, replay key bindings, and playlist semantics.

### Console and keyboard integration

The plugin uses the Linux console for two things:

1. **KBD remote** — VDR reads keypresses from `stdin`; needs `stdin` bound to a VT.
2. **VT auto-management** — startup and `ATTA` pull VDR's VT to the foreground;
   `DETA` yields to `tty1` (override with `VDR_CONSOLE_TTY=N`) so the user lands on getty. Needs
   `CAP_SYS_TTY_CONFIG`.

A single systemd drop-in covers both. `tty7` is conventional and keeps
`getty@tty1.service` running on `tty1` for a login shell:

        sudo install -d -m 0755 /etc/systemd/system/vdr.service.d
        sudo tee /etc/systemd/system/vdr.service.d/50-vaapivideo-console.conf > /dev/null <<'EOF'
        [Service]
        User=vdr
        Group=video
        AmbientCapabilities=CAP_SYS_TTY_CONFIG
        StandardInput=tty
        TTYPath=/dev/tty7
        TTYReset=yes
        TTYVHangup=yes
        EOF
        sudo systemctl daemon-reload
        sudo systemctl restart vdr.service

`User=vdr` makes systemd switch user *before* applying the ambient capability —
the kernel clears ambient caps on any `setuid()` from root, so a `runvdr -u vdr`
wrapper would strip `CAP_SYS_TTY_CONFIG` before the plugin can use it.

Verify:

        journalctl -u vdr -b | grep -E 'kbd|console VT'
        # KBD remote control thread started
        # console VT7 activated for keyboard input

Switch to VDR with `Ctrl+Alt+F7`; back to a login shell with `Ctrl+Alt+F1`.

#### Behavior during `DETA` / `ATTA`

`DETA` releases DRM and switches the foreground to `tty1` so the user lands on
the getty login (and `fbcon` takes over the screen). Set `VDR_CONSOLE_TTY=N` in
the drop-in `[Service]` section to override the target VT. KBD keeps reading from
`stdin`; the kernel only delivers keypresses to the foreground VT, so KBD
pauses while you are on `tty1` and resumes on the next `ATTA` (which pulls
VDR's VT, e.g. `tty7`, back to the foreground) or a manual `Ctrl+Alt+F7`.

Diagnostics — the plugin logs once at INFO when the configuration is incomplete:

- `stdin is not a VT` — drop-in missing; **KBD does not start** and VT
  switches are manual.
- `VT_ACTIVATE denied` — `CAP_SYS_TTY_CONFIG` missing; KBD works, only VT
  switches are manual.

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

| Area    | Symptom                              | Diagnosis and fix                                          |
|---------|--------------------------------------|------------------------------------------------------------|
| Startup | Plugin refuses to start              | VPP missing — `vainfo` must list `VAEntrypointVideoProc`   |
| Startup | DRM device not found                 | `ls -l /dev/dri/`; pass `-d /dev/dri/cardN` explicitly     |
| Startup | No video output                      | Check group membership (`video`, `render`); run `vainfo`   |
| Startup | Black screen after resume            | SVDRP `PLUG vaapivideo DETA` then `ATTA`                   |
| Picture | Combing on interlaced (AMD/Mesa)     | Weak HW deinterlacer — `Deinterlace = software: bwdif` (or `w3fdif`) |
| Picture | Blocky / smeared (Intel Nxxx)        | VPP denoiser broken on these iGPUs — `Denoise = off`       |
| Audio   | No audio                             | `speaker-test -D hw:0,3 -c 2 -r 48000 -t sine -l 1`        |
| Audio   | Passthrough not working              | Use `hw:CARD,DEV`; `/proc/asound/card0/eld#0.N` must be non-empty |
| Audio   | Persistent A/V drift                 | Tune `PCM` / `Passthrough Audio Latency` (see [AVSYNC.md](AVSYNC.md)) |
| Perf    | AMD iGPU stutters / drops            | GPU pinned `low` DPM — `power_dpm_force_performance_level=auto` |
| Perf    | Drops only with `software:` filters  | CPU can't sustain field-rate SW — use `w3fdif` or HW `Deinterlace = auto` |
| Perf    | High CPU on encrypted HD             | Software CSA descrambling (CAM/softcam), not the plugin — a CI+ CAM offloads it |

Picture fixes map to the [Post-processing](#post-processing) options; confirm drops in the `sync … drop=N` log line.

Increase the VDR log verbosity with `-l 3` to capture decoder, display, and
sync diagnostics; the periodic `sync d=… avg=…` line is described in
[AVSYNC.md](AVSYNC.md#diagnostic-log).


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
`.github/copilot-instructions.md`.

### Verbose logging

    vdr -l 3 -P vaapivideo


## HDR passthrough

HDR10 (BT.2020 + SMPTE ST 2084 / PQ) and HLG (BT.2020 + ARIB STD-B67) streams
are detected on the first decoded frame from the AVFrame color metadata and
the frame's bit depth (10-bit minimum). Detection is codec-agnostic — HEVC
Main 10 is the common case, but any FFmpeg decoder that produces BT.2020 + PQ/HLG
10-bit frames will engage passthrough.

When passthrough is active, the entire output chain switches in lockstep:

- `scale_vaapi` emits `P010` 10-bit samples with BT.2020 primaries and the
  stream-native transfer function preserved (no tone-mapping, no BT.709 clamp).
- The DRM video plane scans out `DRM_FORMAT_P010` with `COLOR_ENCODING` set to
  BT.2020 YCbCr.
- The connector's `HDR_OUTPUT_METADATA` blob carries the stream's mastering
  display and content-light side data (HDMI Static Metadata Type 1), with
  `Colorspace=BT2020_YCC` and `max_bpc=10`.

HDR→SDR, SDR→HDR, and HDR10↔HLG transitions are handled atomically in a single
KMS commit per channel switch. SDR streams bypass this path entirely and run
the BT.709 NV12 TV-range pipeline unchanged; any previously programmed HDR
connector state is explicitly reset to SDR defaults before the next frame
lands, so a stream change never leaves stale BT.2020 signaling on the wire.

Three gates must all pass for HDR to engage in `auto` mode:
1. **Stream** — ≥10-bit samples, BT.2020 primaries, PQ or HLG transfer.
2. **GPU** — VAAPI VPP can allocate `P010` (YUV420_10) surfaces. Hardware HEVC
   Main 10 decode is preferred; if it's missing the FFmpeg software HEVC
   decoder fills in, its output is uploaded to a P010 VAAPI surface, and HDR
   still engages.
3. **Display** — the connector exposes `HDR_OUTPUT_METADATA`, the bound video
   plane advertises `DRM_FORMAT_P010` and both BT.709 + BT.2020 `COLOR_ENCODING`
   enums, and the sink's EDID CTA-861 HDR Static Metadata block advertises the
   requested EOTF.

User configuration (VDR setup menu → "VAAPI Video" → **HDR Passthrough**):

- `auto` — autodetect via the three gates above (default).
- `on` — force HDR output whenever the stream is HDR; skips only the sink-EDID
  gate. The plane-support and connector-property gates still apply, so
  configurations that would produce a black screen are refused.
- `off` — never pass through; always use the existing SDR output path.

Tone-mapping (HDR→SDR or SDR→HDR) is deliberately **not** implemented. HDR
content forced through the SDR pipeline (HdrMode::Off, or `auto` with any gate
failing) will show clipped highlights and compressed primaries because no
PQ/HLG inverse EOTF is applied. This is why `auto` is the default.

### HDR signalling sources

HDR10 / HLG is detected from BT.2020 primaries + a PQ/HLG transfer + ≥10-bit
samples. That metadata reaches the decoded frame from one of two places:

- **In-bitstream** (HEVC/AV1 VUI + SEI) — the common case for HEVC HDR10.
- **Container tags** (Matroska/MP4/WebM `Colour` element + mastering/CLL side
  data) — required for **VP9**, whose bitstream signals only a colour-space
  matrix and no transfer function. The mediaplayer seeds the decoder's colour
  fields and the `HDR_OUTPUT_METADATA` mastering luminance from these container
  tags, so a properly tagged VP9 HDR file engages HDR10 with correct mastering
  metadata. An untagged HDR file (no in-bitstream and no container colour info)
  cannot be distinguished from SDR and plays as SDR.

### Dolby Vision and the VAAPI limit

**Dolby Vision is not decoded.** This is a hard VAAPI limitation, not an
omission: no VAAPI driver exposes a DV entry point, and DV's per-scene dynamic
metadata lives in a proprietary **RPU** (plus, for dual-layer profiles, a second
HEVC enhancement layer) that the open VPP/KMS path cannot process or tunnel to
the sink. FFmpeg only ever decodes the **HEVC base layer**. What you get
therefore depends on that base layer's *cross-compatibility*, which the plugin
logs on open (`Dolby Vision profile N (BL compatibility id M)`):

| DV profile | BL compatibility | Result |
|------------|------------------|--------|
| 8.1        | HDR10 (id 1)     | Plays as **HDR10** via the base layer's standard HDR10 metadata (no dynamic DV) |
| 8.4        | HLG (id 4)       | Plays as **HLG** |
| 7          | HDR10 (id 1)     | Base layer plays as **HDR10**; the BD enhancement layer is ignored |
| 5          | none (id 0)      | **SDR fallback** — the base is IPT-PQ with no standard colour signalling |
| 4          | none (id 0)      | **SDR fallback** — dual-layer, no HDR10-compatible base |

So DV files that carry an HDR10/HLG-compatible base play correctly as HDR10/HLG;
profile 4/5 (compatibility id 0) carry their colour only in the RPU and fall back
to SDR. Full DV would require a DV-capable decode + a DV output path neither of
which exists for VAAPI/DRM, so it is **out of scope** rather than planned.

## Mediaplayer

An integrated player for local files, http(s) / ftp URLs, and m3u / m3u8
playlists. Demuxing is done by libavformat; the demuxed access units are
pushed straight into the existing decoder, filter and display pipeline,
so HDR passthrough, deinterlacing, VPP scaling and IEC61937 audio
passthrough work identically to the live-TV path.

### Entry points

- **Main menu** → *Mediaplayer*: opens the file browser rooted at the
  directory passed via `-m DIR / --media-dir=DIR` (default `/`).
  Subdirectories enter on `OK`; m3u files launch as playlists; other
  media files play directly.
- **SVDRP**: `PLUG vaapivideo PLAY <uri>` — `<uri>` is a local path, an
  http(s) / ftp URL, or a local `.m3u/.m3u8` playlist.
- **Remote key**: bind a button on your remote to launch the file
  browser via VDR's `keymacros.conf`. `@vaapivideo` opens the two-line
  quick menu (Zoom / Mediaplayer); append `Down Ok` to jump straight to
  the browser:

      User1   @vaapivideo Down Ok

  Then assign your remote's button to `User1` in `remote.conf` (or via
  *Setup → Remote control → Learning*). Pressing it opens the file
  browser at `--media-dir`. Launching a specific URI from a key needs
  SVDRP, e.g. a wrapper that calls `svdrpsend PLUG vaapivideo PLAY …`.

### Replay controls

| Key                        | Action                       |
|----------------------------|------------------------------|
| `OK`                       | Toggle replay-bar OSD        |
| `Play` / `Up`              | Resume if paused             |
| `Pause` / `Down`           | Toggle pause                 |
| `Left` / `Right`           | Seek −/+ 10 s                |
| `Green` / `Yellow`         | Seek −/+ 60 s                |
| `Blue`                     | Cycle manual zoom (Off → 1–5)|
| `Audio`                    | Audio-track menu (multi-track files) |
| `Next`                     | Skip to next playlist entry  |
| `Back` / `Stop`            | Return to the file browser   |

Rapid key repeats sum: pressing `Right` three times before the demuxer
services the first one lands at +30 s, not +10 s.

### Audio tracks

Files carrying more than one audio track — multiple languages, or a stereo plus
an AC-3 mix — can be switched during playback with the **Audio** key. It opens
the same on-screen track menu VDR uses in live TV and recording replay, listing
each track by codec, channel layout and language (e.g. `AC-3 5.1 (eng)`). Pick
one and the player re-opens the audio decoder and re-anchors A/V at the current
position (a brief, seek-like resync); switching also works while paused and the
new track plays on resume.

On open, the initial track follows your VDR audio-language preference
(*Setup → DVB → Audio languages*), exactly as live and replay do; with no
preference match it defaults to the file's first audio track. AC-3 / DTS and the
other compressed formats still passthrough as IEC61937 when selected — the
wrapping is decided by the codec, not by the menu.

The **Info** key's file page lists every audio track with codec, sample rate,
channel layout and language, marking the active one with a leading `*`.
Subtitle-track selection is not yet supported.

If an audio codec cannot be opened — e.g. a container that omits the sample rate
until a frame is decoded (raw-ADTS AAC in some TS files) — playback degrades to
**video-only** rather than refusing the file; the log notes `audio codec … open
failed -- playing video-only`. The video (including HDR) plays normally.

### File-browser scope

The browser filters its listing to `.mp4 .mkv .avi .mov .ts .m4v .webm`
plus `.m3u/.m3u8`. The filter is for usability only; `PLUG vaapivideo
PLAY` via SVDRP accepts any URI libavformat can open. Audio-only
formats are not supported — the source requires a video stream.

After playback ends, or when leaving playback with `Back` / `Stop`, the browser
reopens at the played local file. URLs and non-selectable paths fall back to `--media-dir`.
Inside the browser, `Back` moves to the parent directory and `Stop` exits to live TV.

### Playlist format

Plain or extended m3u / m3u8. Lines starting with
`#EXTINF:<duration>,<title>` supply the display title for the URI on
the following line; other `#`-prefixed lines are ignored. Relative
paths are resolved against the playlist's parent directory; absolute
paths and URLs are taken verbatim. HLS manifests over http(s) are
deliberately *not* parsed locally — they are forwarded to libavformat
instead.

### Seeking

Seeking lands on the keyframe at or before the requested position, so the resume
point may be a second or two earlier than the exact offset.

### Frame-rate handling

Source frame rates that differ from the display refresh rate are matched to the
display so playback always runs at real-time speed (otherwise a 60 fps source on
a 50 Hz panel would play too slow, a 24 fps source too fast). Frames are
duplicated or dropped as needed — there is no motion interpolation.


## Roadmap

- Dynamic resolution switching on SD / HD / UHD channel changes.
- AV1 live decode path: OBU sequence-header probe + Main / Main 10 backend
  routing, closing the AV1 live branch (the mediaplayer path already opens
  AV1 with codec params from the container).
- Variable refresh rate (HDMI VRR / FreeSync) for judder-free 24p and 25p
  film playback: tie the DRM page-flip cadence to the decoded stream's
  frame rate instead of the panel's fixed refresh.
- Mediaplayer: subtitle rendering, trick-speed (fast/slow forward and
  reverse) and persistent resume position.


## Credits

- **Author:** Dirk Nehring &lt;<dnehring@gmx.net>&gt;
- **Inspired by:** [vdr-plugin-softhdcuvid](https://github.com/jojo61/vdr-plugin-softhdcuvid)


## License

[AGPL-3.0-or-later](LICENSE) — Copyright © 2026 Dirk Nehring.
Modified distributions must publish their source under the same terms.
