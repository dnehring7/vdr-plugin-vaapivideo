# A/V Synchronization

## Problem

A DVB stream encodes audio and video against a single 90 kHz program clock
(PCR). On playback the audio DAC runs on its own oscillator — typically
5–50 ppm off the broadcaster, several hundred ppm on poor SAT>IP gear.
Without active correction, lip-sync drifts by milliseconds per minute.

## Architecture

```
DVB stream / replay file (PCR)
  ├── audio PTS → decoder → ALSA ring → DAC ── GetClock() ── master
  └── video PTS → decoder → filter → jitterBuf → SyncAndSubmitFrame
                                                       │
                                                       ▼
                                       pendingFrames (3 slots) → display thread → KMS commit
```

Three invariants:

1. **Audio is master.** `cAudioProcessor::GetClock()` returns the DAC
   playback PTS, computed as `playbackPts + (now − lastClockUpdateMs)`.
   `playbackPts` is refreshed once per ALSA period (~25 ms); the
   age-extrapolation collapses period quantization to scheduling-jitter
   precision (~1 ms). After `AUDIO_CLOCK_STALE_MS = 1 s` without a write,
   or before any write has fired, `GetClock()` returns `AV_NOPTS_VALUE`
   and the controller falls into freerun.

2. **Audio is never resampled.** No `swr_set_compensation`, no software
   PLL. Only video adapts. This keeps the system stateless across
   channel switches and avoids the feedback-loop instabilities of
   software audio resampling.

3. **Video output cadence is uniform.** The filter graph appends
   `fps=<displayFps>` whenever the source rate is below the display
   refresh, so 25p, 25i-deinterlaced and native 50p all reach
   `SyncAndSubmitFrame` at the same cadence. One controller regime fits
   every source.

## Filter pipeline

```
SW decode: [bwdif] → [hqdn3d] → format=nv12 → hwupload → scale_vaapi → [sharpness_vaapi] [→ fps]
HW decode: [deinterlace_vaapi=rate=field] → [denoise_vaapi] → scale_vaapi → [sharpness_vaapi] [→ fps]
```

`fps` is metadata-only in FFmpeg (`AVFILTER_FLAG_METADATA_ONLY`), so it
duplicates `AVFrame` references without touching pixels. Placement at the
chain tail keeps scale/denoise/sharpen at one execution per *input* frame.

## Per-frame sync (`SyncAndSubmitFrame`)

```
rawDelta = videoPTS − GetClock() − pipelineLatency
```

`rawDelta > 0` ⇒ video ahead, `< 0` ⇒ behind. `pipelineLatency` is the
configured operator knob plus a fixed two-frame tail (one decoder→scanout
period, one HDMI/panel period). The knob is split per output mode:

| Mode                 | `setup.conf` key     | Range       | Default |
| -------------------- | -------------------- | ----------- | ------- |
| PCM (decoded)        | `PcmLatency`         | −200…200 ms | 0       |
| IEC61937 passthrough | `PassthroughLatency` | −200…200 ms | 0       |

Active variant is selected per stream via `cAudioProcessor::IsPassthrough()`.

### EMA smoother

`rawDelta` carries up to ~150 ms field-alternation aliasing on
deinterlaced 50p output; using it directly for soft corrections would
churn. The smoother runs in two phases:

1. **Warmup.** First `WARMUP_SAMPLES = 50` samples (~1 s @ 50 fps) feed a
   simple mean. Soft corrections are gated on `smoothedDeltaValid`, so
   no correction fires off a partial mean.
2. **Steady-state EMA.** Integer EMA with a residual accumulator that
   carries the `diff mod N` remainder across samples — guarantees exact
   convergence to the rawDelta mean even when `|diff| < N`. Time
   constant ~1 s (`EMA_SAMPLES = 50`).

`ResetSmoothedDelta()` clears warmup, EMA, residual, hard-streaks and
catch-up state in one call. Called on channel switch, hard-behind fire,
catch-up exit, and `WaitForAudioCatchUp`.

## Correction regimes

Symmetric: every regime has a behind and an ahead path. Hard transients
bypass the cooldown.

### Soft corridor — `|smoothed| > CORRIDOR (50 ms)`, cooldown elapsed

Let `correctMs = min(|smoothed|/90, MAX_CORRECTION_MS = 200)`.

| Direction | Action |
| --------- | ------ |
| ahead     | `SleepMs(correctMs + frameDur)`, submit, then `smoothed −= (elapsed − frameDur) × 90` |
| behind    | Drop `max(1, round(correctMs / frameDur))` frames (one now, rest via `pendingDrops`); per drop `smoothed += frameDur × 90` |

The `+ frameDur` padding on the sleep is load-bearing: a bare
`SleepMs(correctMs)` only lengthens the iteration by `correctMs − frameDur`
(the missing `frameDur` is absorbed by the next iteration's natural
packet wait), so without padding the smoother sees half the requested
shift and re-fires forever.

Both paths feed back the **measured** effect, not the requested amount.
This preserves the smoother across the correction and is what makes
post-correction `d ≈ 0` reliably reproducible.

`MAX_CORRECTION_MS = HARD_THRESHOLD = 200` so a single soft event can
fully close the corridor; no sub-corridor residual is left for the
controller to re-fire on.

### Cooldown — `COOLDOWN_MS = 5 s`

Armed by every soft fire, hard-ahead transient, and `WaitForAudioCatchUp`.
**Not** armed by catch-up exit or hard-behind: both reset the EMA, and
warmup (~1 s) alone gates soft until the EMA reseeds. The smoother has
therefore absorbed the previous correction plus several cycles of fresh
samples before the next soft event can fire.

### Hard transients (raw delta, no cooldown gate)

| Condition                                | Action |
| ---------------------------------------- | ------ |
| `rawDelta < −HARD_THRESHOLD` (−200 ms)   | Drop N frames (N = round(\|rawDelta\|/frameDur)), reset EMA |
| `rawDelta > +HARD_THRESHOLD`, replay     | `WaitForAudioCatchUp()` blocks (≤ 5 s) until audio catches up, then submit; reset EMA, arm cooldown |
| `rawDelta > +HARD_THRESHOLD`, live       | One sleep ≤ `DECODER_SYNC_HARD_AHEAD_MAX_MS = 500 ms`, submit; EMA `−= measured`, arm cooldown |

Both directions are **streak-debounced** (`hardAheadStreak`,
`hardBehindStreak`): a single sample over threshold submits unpaced and
waits for the next sample to confirm. Real PCR discontinuities shift
`pts` for every subsequent frame, so the streak reaches 2 within one
frame period and the correction still fires within ~20 ms. Isolated
outliers (`snd_pcm_delay` quantization, scheduler hiccups, the
`GetClock()` load-pair race) clear on the next sample and never trigger
a 500 ms freeze.

The live hard-ahead path exists because the soft corridor caps at
~40 ms/s correction rate and a marginal transponder can drift faster.
A single ≤ 500 ms glitch back into the corridor is preferable to an
indefinite slow chase, and the cap prevents the upstream packet queue
from overflowing during the sleep.

### Catch-up — silent bulk drop

Three entry conditions, same exit (`rawDelta > −CORRIDOR`):

| Entry      | Condition                                                       | Triggers |
| ---------- | --------------------------------------------------------------- | -------- |
| spike      | `rawDelta < −2 × HARD_THRESHOLD` (−400 ms)                      | Catastrophic backlog (cold start, post-seek, multi-second decoder stall) |
| warmup     | `!smoothedDeltaValid && rawDelta < −2 × CORRIDOR` (−100 ms)     | Stale pre-roll backlog about to poison the EMA seed |
| sustained  | `smoothedDeltaValid && smoothedDelta < −2 × CORRIDOR` (−100 ms) | Replay queue lag soft-behind can't clear within its cooldown |

While `catchingUp`, every incoming frame is dropped silently — no
per-event log, no EMA churn, no cooldown arm — until `rawDelta` rises
above `−CORRIDOR`. Two log lines bracket the pass:

```
vaapivideo/decoder: catch-up entered (spike|warmup|sustained) raw=-2738ms
vaapivideo/decoder: catch-up complete dropped=143 wall=314ms exit-raw=-38ms
```

Entry threshold (−100 ms) and exit threshold (−50 ms) give at least
`CORRIDOR` of hysteresis, well above single-sample jitter. The exit at
`−CORRIDOR` places the residual inside the soft corridor so no follow-up
soft event is required.

`SkipStaleJitterFrames()` lifts its "keep ≥ 1" guard while catching up,
since the kept frame would be dropped next iteration anyway. On exit
the EMA is reset, `pendingDrops` is cleared, and the exiting frame is
submitted normally.

The **warmup entry** is the post-Clear safety net. After `Clear()` the
EMA is reset and the first 50 samples seed the next mean. If the input
queue still holds pre-Clear-stale frames their `rawDelta` would be
deeply negative, biasing the seed and tripping soft-behind on the very
next frame. The warmup catch-up drains those frames silently before
they reach the EMA accumulator.

## Jitter buffer (unified drain)

The video drain is unified across live and replay: both push decoded
frames onto `jitterBuf` (`std::deque`) and pop when due.

```
push: pendingFrames → jitterBuf
SkipStaleJitterFrames(): bulk-drop heads more than HARD_THRESHOLD behind clock
loop:
  dueIn = headPts − GetClock() − latency
  if dueIn > halfFrame: break          // head not yet due, hold buffer
  else:                 SyncAndSubmitFrame(head)
```

`liveMode` selects only the **hard-ahead policy** inside
`SyncAndSubmitFrame` (replay → `WaitForAudioCatchUp`, live → bounded
sleep). The drain itself is identical for both — `SkipStaleJitterFrames`
plus the pre-warmup catch-up handle every trick-exit and pre-roll
backlog.

### Drain bypasses

The due check is bypassed for:

- **Freerun** (`freerunFrames > 0` after `Clear()`, trick exit, audio
  codec change) — gives an instant first picture.
- **`pendingDrops`** from a soft-behind burst — one drop per drain
  iteration until exhausted.

`SkipStaleJitterFrames()` runs at the top of every drain pass and
bulk-drops heads more than `HARD_THRESHOLD` (200 ms) behind the clock.
Cheaper than routing each through catch-up.

### Steady-state `buf` depth

`buf` (jitterBuf depth at log emission) reflects the difference between
input arrival rate and the gate's release rate.

- **Live TV.** `AUDIO_ALSA_BUFFER_MS = 400 ms` sizes the ALSA hardware
  ring. `GetClock()` therefore lags wall time by roughly that amount,
  so head frames sit in `jitterBuf` until the lagged clock catches
  them. Steady state: `buf ≈ (AUDIO_ALSA_BUFFER_MS + broadcastLead) / frameDur`.
  Higher-bitrate streams ship more lead and run a deeper `buf`. 4K VBR
  can swing `buf` by ~1 s within seconds as bitrate peaks stall packet
  arrival; this is the cushion absorbing jitter, not a problem.

- **Replay, cold start.** VDR's dvbplayer bursts disk reads to refill
  an empty PES ring. Decoder receives input faster than real-time and
  builds a backlog of ~40–60 frames, then dvbplayer throttles to
  playback rate and `buf` stabilizes there.

- **Replay, post-Clear (skip / track switch).** PES ring is drained
  but dvbplayer feeds at real-time from frame zero. Decoder produces
  at audio-clock pace; `buf` stays near 0.

### Audio packet queue (`aq`)

`aq` is the FIFO between `Decode()` and the audio thread. The thread
decodes packets as fast as they arrive and writes PCM to ALSA, so `aq`
drains almost instantaneously and `aq = 0` is healthy. A persistently
non-zero `aq` indicates the audio decoder is falling behind real-time
(CPU contention, ALSA write stall). The real audio cushion is the
**ALSA ring**, not this queue.

## Display prerender

`SyncAndSubmitFrame` hands the chosen frame to
`cVaapiDisplay::SubmitFrame`, which pushes onto `pendingFrames` (a
`std::deque`, depth `DISPLAY_PRERENDER_SLOTS = 3`). The display thread
pops one per VSync, maps via VAAPI→PRIME, and commits via DRM atomic.
`SubmitFrame` **blocks** when all slots are full — this is the VSync
backpressure that paces the decoder to the display refresh rate.

The 3-slot depth lets the decoder stay ahead so an occasional VPP spike
(heavy keyframe through 4K HQ scaling, GPU contention) drains a slot
instead of missing the next VSync.

### Starve detection

The display thread tracks `lastFrameCommitMs` (atomic, updated on every
fresh commit). On a VSync where no fresh frame is available it
re-presents the previous buffer to keep flip cadence + OSD updates
alive. A starve is logged when:

- the last fresh commit was within `ACTIVE_WINDOW_MS = 500 ms`
  (decoder is "active"), AND
- the empty-VSync streak reaches `DISPLAY_PRERENDER_SLOTS + 2 = 5`
  (the prerender depth could not absorb it, so the user *will* see a
  missed VSync), AND
- the warmup grace has expired (see below).

Logged at most once per `STARVE_LOG_COOLDOWN_MS = 2 s`. Outside the
active window (post-Clear, paused, pre-roll, post-trick) re-presents
are expected and neither count nor log.

### Warmup grace

`WARMUP_GRACE_MS = 3 s` suppresses the starve log for the first 3 s
after the decoder resumes from idle. Armed in two cases:

- **Post-Clear.** `BeginStreamSwitch()` zeroes `lastFrameCommitMs`;
  the next fresh commit detects the zero and arms the grace.
- **Idle resume.** Any fresh commit where the previous commit is
  older than `ACTIVE_WINDOW_MS` arms the grace. Covers transitions
  where `BeginStreamSwitch` wasn't invoked (track switch, post-trick
  re-anchor).

Without the grace, every Clear logs a spurious starve while the filter
graph rebuilds and the audio clock anchors.

## Sync bypass

The sync gate is bypassed (frame submitted unpaced) in:

- Trick mode (`SubmitTrickFrame()` paces via its own timer; audio is muted).
- Freerun window after `Clear()`, trick exit, or `NotifyAudioChange()`.
- Radio mode / NOPTS frame (no audio processor or no PTS to align on).
- Audio not yet running (`GetClock()` is NOPTS until the first
  `WritePcmToAlsa()` fires).

## Lifecycle

| Event                      | EMA                | Cooldown   | Jitter buffer                  |
| -------------------------- | ------------------ | ---------- | ------------------------------ |
| Plugin start               | invalid            | —          | empty                          |
| Channel switch (`Clear()`) | reset              | unchanged  | flushed; freerun armed         |
| Catch-up enter             | (drops silent)     | unchanged  | drained silently to alignment  |
| Catch-up exit              | reset              | unchanged  | one frame submitted normally   |
| Soft drop                  | `+= frameDur` / drop | armed    | one frame removed per drop     |
| Soft sleep                 | `−= measured`      | armed      | unchanged                      |
| Hard-behind                | reset              | unchanged  | N frames dropped               |
| Hard-ahead (replay)        | reset              | armed      | unchanged                      |
| Hard-ahead (live)          | `−= measured`      | armed      | unchanged                      |
| Audio codec / track change | unchanged          | unchanged  | preserved; freerun armed       |

Audio codec / track change preserves the buffer across the switch — the
catch-up path will silently realign against the new clock once it
arrives, so dropping ~1 s of still-valid video buys nothing.

## Diagnostic log

```
sync d=+12.5ms avg=+11.7ms lat=40ms buf=50 aq=0 miss=0 drop=0 skip=0
```

| Field  | Meaning |
| ------ | ------- |
| `d`    | Interval mean of `rawDelta` since the last log; comparable to `avg` |
| `avg`  | EMA-smoothed delta; drives every soft-correction decision |
| `lat`  | Active `SyncLatency90k` (2-frame tail + active operator knob) |
| `buf`  | `jitterBuf` depth in frames at log emission |
| `aq`   | Audio packet queue depth |
| `miss` | Drain gaps > 2 × output frame period since last log (VSync backpressure, our sleeps) |
| `drop` | Frames dropped (video behind) since last log — soft + hard combined |
| `skip` | Render delays (video ahead) since last log — soft sleep + hard-ahead combined |

`d ≈ avg` in steady state means the EMA has converged on current
reality. The line is suppressed during warmup and reissued immediately
on warmup completion. Periodic interval `LOG_INTERVAL_MS = 2 s`.

**Healthy steady state:** `avg` inside `±CORRIDOR`, `d ≈ avg`,
`miss = drop = skip = 0`. `buf` depth varies by mode (live:
ALSA-cushion-driven; replay cold start: ~40–60; replay post-Clear: ~0).

Each soft / hard event also emits a per-event `dsyslog` line naming the
cause (`soft-ahead`, `soft-behind`, `hard-ahead live`, `hard-ahead
replay`, `hard-behind`, `stale-jitter bulk`, `catch-up
entered (spike|warmup|sustained)`, `catch-up complete`) for "why did
this fire?" without waiting for the next periodic line.

### Steady-state offset

The EMA does not settle at zero. The baseline depends on whether the
decoder is gate-bound or throughput-bound:

- **Gate-bound** (live TV, replay cold start, anything with `buf > 0`):
  `d ≈ +halfFrame` (~+7 ms @ 50 fps). The gate releases head when its
  pts is `+halfFrame` ahead of `clock + latency`, so submitted frames
  carry that small positive offset.
- **Throughput-bound** (replay post-Clear, `buf ≈ 0`): `d ≈ −2 × frameDur`
  (~−40 ms @ 50 fps). Each new frame is submitted as soon as decoded;
  no buffered lead, and the audio clock has already advanced past the
  latency target by the time `SyncAndSubmit` runs.

Both baselines sit inside `±CORRIDOR` and are **not** drift — the
controller leaves them alone, since correcting a non-drifting bias
would only introduce visible jank. To re-center either regime, tune
`PcmLatency` or `PassthroughLatency` (negative values pull video
earlier vs audio).

## Constants

File-scope constants live in [src/decoder.cpp](src/decoder.cpp),
[src/audio.cpp](src/audio.cpp), and [src/config.h](src/config.h); each
carries a `///<` comment with purpose and unit. Constants marked
*(local)* live inside the function that uses them.

Naming conventions:

- `_MS` suffix means the value is in milliseconds.
- `_90K` suffix means the value is in 90 kHz PTS ticks (matches code
  variables like `rawDelta`, `smoothedDelta90k`, `latency90k`).
- No suffix means the value is dimensionless (sample counts, frame
  counts, depths).

| Constant                            | Value | Purpose |
| ----------------------------------- | ----- | ------- |
| `PTS_TICKS_PER_MS`                  | 90    | DVB 90 kHz PTS clock factor: ticks = ms × this |
| `AUDIO_ALSA_BUFFER_MS`              | 400   | ALSA ring size (ms); lagged audio clock pulls live `buf` to ~MS/frameDur |
| `AUDIO_CLOCK_STALE_MS`              | 1000  | `GetClock()` extrapolation timeout before returning NOPTS |
| `AUDIO_QUEUE_CAPACITY`              | 300   | Audio packet queue depth (~10 s AC-3); sized for slow-start decoders |
| `DECODER_QUEUE_CAPACITY`            | 200   | Video packet queue depth (~4 s @ 50 fps) |
| `DECODER_SYNC_CORRIDOR_90K`         | 4500  | Soft corridor half-width (= 50 ms × PTS_TICKS_PER_MS); below lipsync percept threshold |
| `DECODER_SYNC_HARD_THRESHOLD_90K`   | 18000 | Hard-transient threshold (= 200 ms × PTS_TICKS_PER_MS); 2× = catch-up spike entry |
| `DECODER_SYNC_MAX_CORRECTION_MS`    | 200   | Soft-event cap (matches HARD_THRESHOLD) so one event fully closes corridor |
| `DECODER_SYNC_HARD_AHEAD_MAX_MS`    | 500   | Live hard-ahead sleep cap (ms) |
| `DECODER_SYNC_COOLDOWN_MS`          | 5000  | Min interval between soft corrections (= 5 EMA time constants) |
| `DECODER_SYNC_EMA_SAMPLES`          | 50    | EMA divisor (~1 s @ 50 fps); residual accumulator → exact convergence |
| `DECODER_SYNC_WARMUP_SAMPLES`       | 50    | Samples averaged before EMA seed (~1 s @ 50 fps) |
| `DECODER_SYNC_FREERUN_FRAMES`       | 1     | Unpaced frames after sync-disrupting events |
| `DECODER_SYNC_LOG_INTERVAL_MS`      | 2000  | Periodic sync diagnostic interval (ms) |
| `DISPLAY_PRERENDER_SLOTS`           | 3     | Decoder→display handoff queue depth; absorbs single-VSync VPP spikes |
| `ACTIVE_WINDOW_MS` *(local)*        | 500   | Window after last fresh commit during which a VSync starve counts |
| `STARVE_LOG_COOLDOWN_MS` *(local)*  | 2000  | Min interval between starve dsyslog lines |
| `WARMUP_GRACE_MS` *(local)*         | 3000  | Post-idle grace suppressing starve logs while pipeline anchors |
| `STARVE_STREAK_THRESHOLD` *(local)* | 5     | Empty-VSync streak length that trips a starve log (= PRERENDER_SLOTS + 2) |
