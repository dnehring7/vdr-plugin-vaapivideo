# VDR VAAPI Video Plugin

Hardware-accelerated video output for [VDR](https://www.tvdr.de/) using VAAPI
decode, DRM atomic mode-setting, and ALSA audio. No X11, Wayland, or OpenGL is
required — the plugin runs on the bare console, in a systemd service, or fully
headless.

The video path is zero-copy: VAAPI surfaces are exported as DRM PRIME buffers
and scanned out without ever touching system memory. Audio passthrough formats
are detected automatically from the HDMI sink's EDID. Codecs that lack hardware
decode support on the host GPU fall back to FFmpeg software decoding
transparently. The VAAPI Video Processing Pipeline (VPP) **must** be available —
the plugin refuses to start without it.


## Features

| Component   | Capabilities                                                                                       |
|-------------|----------------------------------------------------------------------------------------------------|
| Decode      | MPEG-2, H.264 (incl. High 10), HEVC (incl. Main 10), AV1 Main / Main 10 — hardware (VAAPI) with per-profile software fallback |
| Filters     | Deinterlace, denoise, DAR-preserving scale, sharpen — hardware (VAAPI VPP) or software (bwdif, hqdn3d) |
| Audio       | PCM decode/downmix with sink-driven multichannel output; IEC61937 passthrough (AC-3, E-AC-3, DTS, TrueHD, AC-4, MPEG-H 3D) |
| Display     | DRM atomic mode-setting, double-buffered page-flip, BT.709 SDR + BT.2020 HDR10/HLG passthrough, optional runtime resolution / refresh-rate matching |
| OSD         | True-color hardware overlay on a dedicated DRM plane, alpha-blended over the video plane           |
| Mediaplayer | Local files (MP4, MKV, TS, WebM, …), http(s)/ftp URLs, m3u/m3u8 playlists, trick play (fast/slow, forward/backward), audio-track switching, text subtitles — see [Mediaplayer](#mediaplayer) |
| A/V sync    | Audio-mastered, EMA-smoothed, proportional with hard-transient bypass — see [AVSYNC.md](AVSYNC.md) |


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


## Installation

### Pre-built packages

Signed Fedora 44, Debian 13, and Ubuntu 26.04 LTS package repositories are
published on every
[GitHub release](https://github.com/dnehring7/vdr-plugin-vaapivideo/releases)
and served via GitHub Pages. All configs reference the signing key at
<https://github.com/dnehring7.gpg>.

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

The Ubuntu build links against FFmpeg 8 and Ubuntu's `vdr-dev` 2.6.9, and is a
separate ABI from the Debian Trixie build — install one or the other, not both.

</details>

After installing a package, continue with [Permissions](#3-permissions) and
[Install the VAAPI driver](#4-install-the-vaapi-driver).

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
packaging: `rpmbuild -ta vdr-vaapivideo-*.tar.gz`

### 3. Permissions

The VDR user needs access to DRM render, video, and ALSA devices:

    sudo usermod -aG video,render,audio vdr

A logout or service restart is required for group changes to take effect.

### 4. Install the VAAPI driver

The plugin requires a VAAPI driver with **Video Processing Pipeline (VPP)**
support — the source of all hardware scaling, deinterlacing, denoising, and
colorspace conversion.

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

If this line is missing, the plugin will not start — verify the driver
(step 4) and the render-node permissions (step 3).

For deeper diagnostics, a standalone probe tool reports decode profiles, VPP
filters, surface formats, HDR tone mapping (including the HLG → HDR10 H2H
path), variable refresh rate support (kernel `vrr_capable` / `VRR_ENABLED`
plus the sink's advertised VRR ranges: HDMI 2.1 game-VRR, AMD FreeSync, and
VESA range limits), and the sink's EDID HDR capabilities.
It is built on demand:

    make probe
    ./vaapivideo-probe [/dev/dri/cardN]     # default: /dev/dri/card0

Any line showing **no** indicates a missing driver or sink capability; compare
against the plugin log (`vdr -l 3`).

### 6. Configure the ALSA audio device

Prefer a device bound **directly to the HDMI/DisplayPort output** — either
`hw:CARD,DEV` (bit-exact) or `plughw:CARD,DEV` (same device plus rate/format
conversion). Both expose the real sink and its channel map, so IEC61937
passthrough and native multichannel PCM both work. Avoid the bare `default`:
it routes through dmix/PulseAudio, downmixes multichannel to stereo, and
blocks the channel-map query used to put surround channels on the right
speakers. Find the device with:

    aplay -l | grep -E "HDMI|DisplayPort"
    vdr -P 'vaapivideo -a plughw:0,3'


## Configuration

### Command-line options

    vdr -P 'vaapivideo [-a DEV] [-c NAME] [-D] [-d DEV] [-m DIR] [-r WxH@R] [-t]'

| Option                           | Default         | Description                                           |
|----------------------------------|-----------------|-------------------------------------------------------|
| `-a DEV`, `--audio=DEV`          | `default`       | ALSA audio device — prefer `hw:`/`plughw:CARD,DEV`    |
| `-c NAME`, `--connector=NAME`    | first connected | DRM connector name (e.g. `HDMI-A-1`, `DP-2`)          |
| `-D`, `--detached`               | off             | Start without opening the DRM/VAAPI/ALSA hardware     |
| `-d DEV`, `--drm=DEV`            | auto-detect     | DRM device path (`/dev/dri/cardN`)                    |
| `-m DIR`, `--media-dir=DIR`      | `/`             | Mediaplayer file-browser root directory               |
| `-r WxH@R`, `--resolution=WxH@R` | `1920x1080@50`  | Default output resolution and refresh rate (whole Hz, max 3840×2160) |
| `-t`, `--trace`                  | off             | Emit the A/V-sync and stream-start diagnostics (needs `vdr -l 3`) |

Use `-d` explicitly when multiple GPUs are present, and `-c` to select a
specific output when multiple displays are connected (names as under
`/sys/class/drm/`).

`--resolution` names the **default** mode: it is programmed at startup, is the
fallback for [display mode switching](#display-mode-switching), and is restored
when playback ends. Its rate is whole Hz: where a panel offers both 59.94 and
60.000, `@60` takes 60.000 and falls back to 59.94 only when 60.000 is absent.

`--detached` brings VDR up without grabbing the GPU, DRM master, or ALSA
device. Hardware initialization runs on the first primary-device promotion or
on SVDRP `PLUG vaapivideo ATTA` — useful for hosts that yield the display to
another application at boot.

### Setup menu

    Setup → Plugins → vaapivideo

| Setting                          | Range            | Description                                                                                          |
|----------------------------------|------------------|------------------------------------------------------------------------------------------------------|
| **Audio** | | |
| `Audio Passthrough`              | auto / on / off  | IEC61937 passthrough policy (see [Audio settings](#audio-settings))                                  |
| `PCM Channels`                   | auto / stereo / multichannel | Decoded-PCM channel layout (see [Audio settings](#audio-settings))                        |
| `PCM Audio Latency (ms)`         | −200 … 200       | A/V offset applied when audio is decoded to PCM                                                      |
| `Passthrough Audio Latency (ms)` | −200 … 200       | A/V offset applied when audio is forwarded as IEC61937                                               |
| **Video** | | |
| `Deinterlace`                    | auto / hardware: motion adaptive / weave / bob / software: bwdif / w3fdif | Deinterlacer policy (see [Post-processing](#post-processing)) |
| `Denoise`                        | auto (hardware) / off / software: light / strong | Denoise policy                                                                       |
| `Scaling`                        | auto (hardware, HQ) / hardware: fast / software: HQ / software: fast | Scaler selection                                                 |
| `Sharpen`                        | auto (hardware) / off / software: mild / medium | Sharpen policy                                                                        |
| `HDR Passthrough`                | auto / on / off  | HDR10 / HLG output policy (see [HDR](#hdr))                                                          |
| **Display Mode** — all off by default | | |
| `Match refresh rate`             | off / on         | Track the source frame rate with the display refresh rate                                            |
| `Match resolution`               | off / on         | Track the source coded size with the display resolution                                              |
| `Minimum resolution`             | 576p / 720p / 1080p / 2160p | Floor for the resolution search; set to the panel's native height to pin the resolution |
| `Maximum refresh rate`           | 50 / 60 / 100 / 120 Hz / unlimited | Ceiling for the refresh-multiple search                                             |
| `Switch for live TV`             | off / on         | Allow mode switching while watching live TV                                                          |
| `Switch for recordings`          | off / on         | Allow mode switching while replaying recordings                                                      |
| `Switch for mediaplayer`         | off / on         | Allow mode switching in the integrated mediaplayer                                                   |
| **Zoom** | | |
| `Zoom level N`                   | 0 … 499          | Zoom-in factor of level N (1–5) in tenths-of-% (`344` = +34.4%); 0 disables the level                |
| **General** | | |
| `Clear display on channel switch`| off / on         | Paint a black frame on channel switch instead of keeping the previous channel's last frame           |

### Audio settings

**`Audio Passthrough`** — `auto` (default) reads the HDMI sink's ELD at startup
and forwards a compressed codec as IEC61937 only when the sink advertises
support for it; everything else is decoded to PCM. If the ELD is unreadable at
that point (AVR asleep, TV off), the probe repeats on the next codec change, so
passthrough and multichannel light up as soon as the sink answers — no VDR
restart needed. `on` forces passthrough for
every wrappable codec (AC-3, E-AC-3, TrueHD, DTS, AC-4, MPEG-H 3D) and ignores
the ELD — for topologies where the probed capabilities are wrong, typically an
AVR behind a TV whose EDID masks the AVR's real decoders. Make sure the
downstream device really decodes the codec: ALSA cannot detect a silent decode
failure at the sink, you will simply hear nothing. `off` always decodes to PCM.
Changes take effect when the audio device is reopened — switch channels once
after leaving the setup menu.

Passthrough also flips the IEC 60958-3 "non-audio" bit (AES0 bit 1) that tells
the sink a bitstream, not PCM, is coming, and clears it again on every PCM open
(HDMI codecs keep the bit across `snd_pcm_close()`). The control is resolved per
output at startup — HDA cards expose one `IEC958 Playback Default` per digital
converter on `iface=MIXER`, indexed in PCM-device order, so `hw:0,3` (the first
HDMI pin) uses index 0. The log names the element it picked:
`IEC958 Playback Default on hw:0 -- iface=MIXER device=0 index=0`, and every
*change* as `IEC958 AES0 0x04 -> 0x06 (non-audio)`. The bit is re-asserted on
every device open (the kernel's cached value can drift from the link across an
AVR power-cycle or a hotplug), but a write that changes nothing is not logged:
no line on a passthrough-to-passthrough channel switch means the sink was
already armed, not that the bit was skipped. If the log instead reports
`not resolved`, the card exposes no such control (an ALSA `default`/dmix device
does not) and the bit is left alone — passthrough still works on sinks that key
off the IEC61937 preamble alone.

**`PCM Channels`** — applies whenever audio is decoded to PCM (no passthrough,
or a codec without IEC61937 framing such as AAC or MP2):

- **auto** (default): native multichannel up to the sink's advertised PCM
  channel count; falls back to stereo when no ELD is readable.
- **stereo**: always downmix to 2.0.
- **multichannel**: force native multichannel even without a readable ELD —
  for sinks whose capabilities are masked but known to handle multichannel PCM.

The output layout follows the decoded stream (a 5.1 broadcast plays as 5.1,
stereo stays stereo — surround is never fabricated) and adapts mid-stream when
a broadcast switches layouts. Correct surround channel ordering requires a
direct `hw:`/`plughw:` device (see
[step 6](#6-configure-the-alsa-audio-device)).

**Latency** — the two knobs are split because a receiver doing its own
bitstream decode adds a different delay than the PCM path. Both default to
0 ms; adjust only if a residual offset remains after the sync controller has
settled. See [AVSYNC.md](AVSYNC.md#steady-state-offset) for the sign
convention.

### Post-processing

Deinterlace, denoise, scaling, and sharpen are four independent policies. They
all default to **auto**, the zero-copy VAAPI VPP path. Each option's label says
where it runs: `auto` and `hardware:` choices stay on the GPU; any `software:`
choice pulls the decoded frame to system memory once, runs the whole
post-process in software, and uploads the result back for display. Decoding
always stays on the GPU, and the software block is automatically bypassed for
HDR, UHD, and trick play.

- **Deinterlace** — `auto` picks the best mode the driver advertises
  (motion-compensated when present). `hardware:` requests a specific VAAPI
  mode, clamped to what the driver offers. `software: bwdif` is the quality
  choice for interlaced broadcast on GPUs with a weak hardware deinterlacer
  (notably AMD/Mesa, which leaves visible combing); `software: w3fdif` is a
  lighter alternative for slower CPUs.
- **Denoise** — `auto (hardware)` is the codec-tuned VAAPI denoiser; `off`
  skips it; `software: light / strong` are `hqdn3d` presets.
- **Sharpen** — `auto (hardware)` is the codec-tuned VAAPI sharpener; `off`
  skips it; `software: mild / medium` are `unsharp` presets.
- **Scaling** — `auto (hardware, HQ)` is high-quality GPU scaling and the right
  choice almost always; the alternatives are niche fallbacks for GPUs whose
  scaler is suspect.

The software block runs at field rate (1080i50 becomes 50 fps through every
software filter), so a low-power CPU can saturate and drop frames. If playback
can't keep up: use `w3fdif`, set denoise to `off`, or return to the hardware
path.

Changes apply to live playback immediately on leaving the setup menu — the
filter graph is rebuilt in place, no channel switch needed.

### Display mode switching

By default the plugin programs the `--resolution` mode once at startup and
never changes it. **Everything in this group is off by default**: two policies
decide *what* is tracked, three source switches decide *when* a change is
allowed, and nothing happens until at least one of each is enabled.

- **Match refresh rate** — picks the highest exact integer multiple of the
  source frame rate at or below `Maximum refresh rate`: 25p → 50 Hz, 24p → 24
  or 48 Hz, 29.97p → 59.94 Hz. Exact rates are recomputed from the mode
  timings, so 59.94 and 60.000 are distinguished; when no exact multiple
  exists, a 0.5% tolerance retry lets 59.94 content settle on a 60 Hz-only
  panel. Interlaced sources match on their field rate (1080i25 lands on
  50 Hz). Landing on the right rate removes frame-rate resampling entirely —
  the judder of 24p on a 50 Hz mode disappears.
- **Match resolution** — picks the smallest mode that still covers the
  stream's coded size, never below `Minimum resolution`. Mapping 1:1 lets the
  TV do the upscale (usually better) and cuts memory bandwidth on UHD panels.
  Setting `Minimum resolution` to the panel's native height pins the
  resolution while leaving refresh matching free. Off-aspect modes and CEA
  pixel-repetition rasters (1440×576, 2880×576) are filtered out. Anamorphic
  SD modes (720×576 flagged 16∶9) are fully compensated: the picture is fitted
  for the non-square pixels, the OSD stays aligned, and `GRAB` screenshots are
  widened back to square pixels. The mode inventory tags such modes
  `[anamorphic 64:45]`.

Switches happen **proactively** when the mediaplayer opens a file, and
**reactively** for live TV and recordings once the stream's timing is stable
(1.5 s stability, at most one switch per 3 s — rapid zapping produces at most
one switch, on the channel you stayed on). Trick play never triggers a switch.
The default mode is restored whenever mode switching is no longer in charge of
what is playing.

A mode change re-trains the HDMI link, so the panel goes black for roughly half
a second — the same blank a source switch on the TV produces. The OSD re-lays
itself out within about a second.

> **Caution:** a **resolution** change resizes the OSD mid-session, a path some
> skins have never had to handle. Enabling only `Match refresh rate` avoids the
> OSD resize entirely. Also note an AVR locked onto an IEC61937 bitstream may
> briefly drop out of passthrough when the link re-trains.

Use `svdrpsend PLUG vaapivideo MODE` to see the connector's usable modes with
their exact rates, the active and default mode, and the matcher's decision for
the stream currently playing.

### Manual zoom

Five zoom levels magnify the picture to fill the screen — useful for cropping
away black bars baked into the broadcast (2.39:1 scope, 2.00:1, and similar).
Each level is a zoom-in factor in tenths-of-a-percent; aspect is preserved and
the overflow is cropped equally off all sides. Out of the box, level 1 is
+34.4% (fills 2.39:1 on a 16:9 screen) and level 2 is +12.5% (fills 2.00:1);
levels 3–5 are off. The maximum is +49.9%.

Cycling steps Off → 1 → 2 → 3 → 4 → 5 → Off, skipping levels set to 0. The
active stop is transient: it resets to Off on every content change and is
never written to `setup.conf` — only the five level definitions persist.

- **Mediaplayer replay** — the **Blue** key cycles zoom.
- **Live TV** — VDR routes no live-TV keypresses to output plugins, so the
  plugin's main-menu hook (`@vaapivideo`) opens a two-line menu (**Zoom** /
  **Mediaplayer**). Bind it in `keymacros.conf`; VDR can append follow-up
  keypresses for one-key actions:

      Blue      @vaapivideo Ok          # cycle zoom, menu closes itself
      Yellow    @vaapivideo Down Ok     # open the mediaplayer browser

- **Scripting** — `svdrpsend PLUG vaapivideo ZOOM [next|0-5]`.


## Mediaplayer

An integrated player for local files, http(s)/ftp URLs, and m3u/m3u8
playlists. Demuxing is done by libavformat; the demuxed packets feed the same
decoder, filter, and display pipeline as live TV, so HDR passthrough,
deinterlacing, and IEC61937 audio passthrough work identically.

### Starting playback

- **Main menu → Mediaplayer** — file browser rooted at `--media-dir`
  (default `/`). Directories enter on `OK`; m3u files launch as playlists;
  media files play directly. The browser lists
  `.mp4 .mkv .avi .mov .ts .m4v .webm` plus `.m3u/.m3u8`.
- **SVDRP** — `PLUG vaapivideo PLAY <uri>` accepts any URI libavformat can
  open (a video stream is required — audio-only formats are not supported).
- **Remote key** — bind `@vaapivideo Down Ok` to a key in `keymacros.conf`
  (see [Manual zoom](#manual-zoom)).

### Replay controls

| Key                        | Action                                             |
|----------------------------|----------------------------------------------------|
| `OK`                       | Toggle replay-bar OSD                              |
| `Play` / `Up`              | Resume normal playback (from pause or trick play)  |
| `Pause` / `Down`           | Toggle pause; exits trick play into pause          |
| `FastFwd` / `FastRew`      | Trick play (see below)                             |
| `Left` / `Right`           | Seek −/+ 10 s (exits trick play first)             |
| `Green` / `Yellow`         | Seek −/+ 60 s (exits trick play first)             |
| `Blue`                     | Cycle manual zoom                                  |
| `Audio`                    | Audio-track menu                                   |
| `Subtitles`                | Subtitle-track menu                                |
| `Next`                     | Skip to next playlist entry                        |
| `Back` / `Stop`            | Return to the file browser                         |

Rapid seek presses sum (`Right` three times = +30 s). Seeking lands on the
keyframe at or before the requested position, so the resume point may be a
second or two early.

### Trick play

`FastFwd` / `FastRew` follow VDR's dvbplayer semantics. From normal play they
enter fast forward/rewind, stepping through keyframes at ×2/×4/×8 (repeated
presses cycle the speed, shown in the replay bar as `1>>` … `3>>` / `<<1` …
`<<3`); from pause they enter slow motion (`1|>` / `<|1`, audio muted), which
resumes at the shown position.
With `Setup → Replay → Multi speed mode` on, pressing the opposite key winds
an active mode back down through normal play/pause. With it off there is a
single speed per mode: holding the key scans until release, and pressing the
opposite key restarts the scan in the new direction. Fast rewind steps keyframes
backward through the container (there is no VDR index file), so the effective
rewind smoothness depends on the file's keyframe interval; audio and
subtitles are off during all trick modes, and reaching the file start while
rewinding resumes normal playback.

### Tracks and subtitles

The **Audio** key opens VDR's standard track menu for files with multiple
audio tracks, listed by codec, layout, and language (e.g. `AC-3 5.1 (eng)`).
The initial track follows the VDR audio-language preference; compressed
formats still pass through as IEC61937 when selected. If an audio codec cannot
be opened, playback degrades to video-only instead of refusing the file.

Embedded **text** subtitles (SubRip, ASS/SSA, mov_text) are selected with the
**Subtitles** key via VDR's standard track chooser and rendered on the OSD,
following VDR's subtitle transparency and offset settings. Bitmap formats
(DVB subtitles, PGS) are not rendered on the mediaplayer path.

### Playlists and resume

Playlists are plain or extended m3u/m3u8: `#EXTINF` titles are honored,
relative paths resolve against the playlist's directory, and http(s) HLS
manifests are forwarded to libavformat rather than parsed locally.

The player keeps a single resume bookmark in `setup.conf`, updated whenever
playback stops. Restarting the bookmarked local file resumes at the saved
position; playing to the end resets it. After playback the browser reopens
with the cursor on the last-played file.

### Frame-rate handling

Sources whose frame rate differs from the display refresh are duplicated or
dropped to real-time speed (no motion interpolation); rates within 0.2% count
as matched and are left alone. Enable
[display mode switching](#display-mode-switching) to move the display to the
source's cadence instead of resampling.


## HDR

HDR10 and HLG streams are detected on the first decoded frame from the color
metadata (BT.2020 primaries, PQ or HLG transfer, ≥10-bit) — codec-agnostic,
with HEVC Main 10 the common case. When passthrough engages, the whole chain
switches in lockstep: the filter graph emits 10-bit P010, the video plane
scans out BT.2020, and the connector carries the stream's HDR metadata to the
sink. SDR streams run the BT.709 pipeline unchanged, and every transition is
atomic — a stream change never leaves stale HDR signaling on the wire.

**`HDR Passthrough`** in the setup menu:

- **auto** (default) — engage only when stream, GPU, and display all support
  it, including the sink's EDID advertising the stream's EOTF.
- **on** — skip only the sink-EDID check (for sinks with wrong EDID data);
  combinations that would produce a black screen are still refused.
- **off** — always use the SDR path.

Tone-mapping is deliberately **not** implemented: HDR content forced through
the SDR path shows clipped highlights and washed-out color, which is why
`auto` is the default. VP9 HDR needs container color tags (Matroska/WebM),
which the mediaplayer forwards to the decoder; an untagged HDR file cannot be
distinguished from SDR and plays as SDR.

### Dolby Vision

Dolby Vision is **not decoded** — no VAAPI driver exposes a DV entry point,
and FFmpeg decodes only the HEVC base layer. What you get depends on the base
layer's cross-compatibility, which the plugin logs when the file opens:

| DV profile | Base layer compatibility | Result                              |
|------------|--------------------------|-------------------------------------|
| 8.1        | HDR10                    | Plays as HDR10 (no dynamic metadata)|
| 8.4        | HLG                      | Plays as HLG                        |
| 7          | HDR10                    | Base layer plays as HDR10           |
| 4 / 5      | none                     | SDR fallback                        |

### Dynamic HDR

DVB broadcasts carry dynamic HDR as HDR10+, SL-HDR2, or Dolby Vision. None of
them can be passed through: the kernel's `HDR_OUTPUT_METADATA` property carries
only the static HDR10 metadata infoframe, so the plugin sends the PQ or HLG base
layer and the sink applies its own tone mapping.

`make probe` reports what the sink advertises for each system, read from the
EDID CTA-861 HDR Dynamic Metadata block (HDR10+, SL-HDR1/2/3, ST 2094-10) and
the Dolby vendor block. The Dolby line shows the vendor block's layout version
(0–2), which is an EDID format revision — Dolby Vision 2 has no published EDID
signaling and cannot be detected.


## SVDRP commands

| Command                        | Description                                                |
|--------------------------------|------------------------------------------------------------|
| `PLUG vaapivideo STAT`         | Device status, active resolution, refresh rate             |
| `PLUG vaapivideo CONF`         | Current configuration summary                              |
| `PLUG vaapivideo MODE`         | Display-mode inventory, active/default mode, current match decision |
| `PLUG vaapivideo DETA`         | Detach from DRM/VAAPI hardware (release for other apps)    |
| `PLUG vaapivideo ATTA`         | Re-attach to DRM/VAAPI hardware; if primary, resume output |
| `PLUG vaapivideo PLAY <uri>`   | Start mediaplayer on a file, URL, or `.m3u/.m3u8` playlist |
| `PLUG vaapivideo ZOOM [next\|0-5]` | Cycle manual zoom (`next`) or select a stop (0 = off)  |
| `PLUG vaapivideo TRACE [on\|off]` | Turn A/V-sync and stream-start tracing on or off (no argument queries) |

`DETA` hands the display to another application and `ATTA` reclaims it without
restarting VDR; when the plugin is the primary device, `ATTA` also re-tunes the
channel so data flows through the fresh pipeline.


## Console and keyboard integration

The plugin uses the Linux console for two things: the **KBD remote** (VDR
reads keypresses from `stdin`, which must be bound to a VT) and **VT
auto-management** (startup and `ATTA` pull VDR's VT to the foreground; `DETA`
yields to `tty1`, override with `VDR_CONSOLE_TTY=N`, so the user lands on a
login shell — this needs `CAP_SYS_TTY_CONFIG`).

A single systemd drop-in covers both; `tty7` keeps `tty1` free for a getty:

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

`User=vdr` must be set here: the kernel clears ambient capabilities on any
`setuid()` from root, so a `runvdr -u vdr` wrapper would strip
`CAP_SYS_TTY_CONFIG` before the plugin can use it.

Verify with `journalctl -u vdr -b | grep -E 'kbd|console VT'` — expect
`KBD remote control thread started` and `console VT7 activated`. Switch to VDR
with `Ctrl+Alt+F7`, back to a login shell with `Ctrl+Alt+F1`. The plugin logs
`stdin is not a VT` (drop-in missing, KBD disabled) or `VT_ACTIVATE denied`
(capability missing, VT switches manual) when the configuration is incomplete.


## Inter-plugin service API

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
| Audio   | Multichannel plays as stereo / wrong speakers | Use a direct `hw:`/`plughw:CARD,DEV`, not `default` |
| Audio   | Persistent A/V drift                 | Tune `PCM` / `Passthrough Audio Latency` (see [AVSYNC.md](AVSYNC.md)) |
| Audio   | Short dropout shortly after a channel switch | ALSA ring ran dry — the log reports `ALSA … recovered -- N xrun(s)`; check `state:`/`avail_max` in `/proc/asound/card0/pcm3p/sub0/status` and see [AVSYNC.md](AVSYNC.md#ring-cushion) |
| Perf    | AMD iGPU stutters / drops            | GPU pinned `low` DPM — `power_dpm_force_performance_level=auto` |
| Perf    | Drops only with `software:` filters  | CPU can't sustain field-rate SW — use `w3fdif` or HW `Deinterlace = auto` |
| Perf    | High CPU on encrypted HD             | Software CSA descrambling (CAM/softcam), not the plugin — a CI+ CAM offloads it |

Increase the VDR log verbosity with `-l 3` to capture decoder, display, and
audio diagnostics.

The A/V-sync and channel-switch diagnostics need one switch more, because they
narrate the presentation thread frame by frame: run VDR at `-l 3` *and* turn
tracing on, either with the plugin's `-t` / `--trace` option or at runtime with
`svdrpsend PLUG vaapivideo TRACE on`. That yields

- the periodic `sync d=… avg=…` line plus a line per drop, skip and catch-up —
  see [AVSYNC.md](AVSYNC.md#diagnostic-log), and
- for slow channel switches (picture or sound arriving late), the `trace +Nms …`
  timeline: first audio/video PES, first keyframe, codec open, first decoded /
  presented / committed frame, DAC start, A/V lock, all relative to the same
  switch epoch — see [AVSYNC.md](AVSYNC.md#stream-start-trace).

Faults report themselves without tracing: `catch-up cycling sustained`,
`jitterBuf overflow`, and every warning and error stay at the ordinary levels.


## Development

### Architecture

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

Two input paths share the decoder/filter/display pipeline unchanged: VDR live
and replay traffic enters as PES through `cVaapiDevice::PlayVideo` /
`PlayAudio`; the integrated mediaplayer demuxes with libavformat and pushes
pre-framed access units into the same decoder. Codec selection, HDR routing,
and A/V sync are path-agnostic.

Inside `cVaapiDecoder`, decode and presentation run on separate threads: the
decode thread filters frames into a decode-ahead reserve, and a presentation
thread drains it at the audio-synced cadence, so a slow 4K VPP step spends the
reserve instead of stalling the screen. The complete A/V sync design —
threading, buffering, correction regimes, diagnostics — is documented in
[AVSYNC.md](AVSYNC.md).

### Source layout

| File                  | Responsibility                                                                        |
|-----------------------|---------------------------------------------------------------------------------------|
| `vaapivideo.cpp`      | Plugin entry point, VDR lifecycle, setup menu, SVDRP, main-menu hook                  |
| `src/device.cpp`      | VDR device integration, PES routing, hardware init/teardown, mediaplayer feed surface |
| `src/decoder.cpp`     | Decoupled VAAPI decode + presentation threads, A/V sync controller                    |
| `src/filter.cpp`      | FFmpeg filter-graph build (deinterlace / denoise / scale / sharpen; HW and SW chains) |
| `src/display.cpp`     | DRM atomic mode-setting, PRIME import, page-flip thread                                |
| `src/audio.cpp`       | ALSA output (multichannel PCM / downmix, chmap), IEC61937 passthrough, HDMI ELD read  |
| `src/osd.cpp`         | DRM dumb-buffer OSD overlay (ARGB8888 plane)                                          |
| `src/mediaplayer.cpp` | libavformat demux, file browser, cControl with OSD replay bar                         |
| `src/subtitle.cpp`    | Mediaplayer text-subtitle decode (SubRip/ASS/mov_text) + OSD rendering                |
| `src/stream.cpp`      | Shared codec/profile data model, H.264/HEVC SPS probe                                 |
| `src/pes.cpp`         | PES header parsing                                                                    |
| `src/caps.cpp`        | GPU / display / audio-sink capability probing (VAAPI, EDID, ELD)                      |
| `src/config.cpp`      | Resolution parsing, `setup.conf` storage                                              |
| `src/common.h`        | RAII deleters, `AvErr()` helper, version/API guards                                   |

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
| `make probe`   | Build the `vaapivideo-probe` diagnostic tool    |

For debug builds, uncomment the matching sanitizer block in the Makefile
(ASan + UBSan **or** TSan — mutually exclusive); the Makefile comments document
the runtime environment variables. Run with verbose logging via
`vdr -l 3 -P vaapivideo`.

The project enforces a strict modern-C++ style — trailing return types,
`[[nodiscard]]`, RAII for every C-API resource, VDR threading primitives. The
full rules are in `.github/copilot-instructions.md`.


## Roadmap

- **VRR presentation** — on displays with variable refresh rate (FreeSync /
  VESA Adaptive-Sync), latch each frame at its PTS deadline instead of
  switching display modes: the existing audio-master pacing goes straight to
  the glass at the source's true cadence (e.g. 50 Hz PAL on a panel with no
  fixed 50 Hz mode). `Match refresh rate` becomes a three-way choice —
  off / mode switch / VRR preferred — falling back to mode switching when the
  display has no usable VRR range.
- **HLG → HDR10 (PQ) mapping** — for HDR panels that accept only the PQ EOTF
  (common on laptop eDP), convert HLG streams to PQ and signal ST 2084 to the
  sink. Uses the driver's VAAPI HDR tone-mapping filter
  (`VAProcFilterHighDynamicRangeToneMapping`, HDR-to-HDR mode) when the GPU
  exposes it; on drivers without it (Mesa `radeonsi`) falls back to a static
  per-channel transfer-function swap in the CRTC gamma LUT at zero per-frame
  cost. Includes accepting BT.2020-RGB-only sinks in the HDR gate, which
  today fall back to SDR even for native HDR10.


## Credits

- **Author:** Dirk Nehring &lt;<dnehring@gmx.net>&gt;
- **Inspired by:** [vdr-plugin-softhdcuvid](https://github.com/jojo61/vdr-plugin-softhdcuvid)


## License

[AGPL-3.0-or-later](LICENSE) — Copyright © 2026 Dirk Nehring.
Modified distributions must publish their source under the same terms.
