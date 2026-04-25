# A/V Synchronization

## Problem

A DVB stream encodes audio and video against a single 90 kHz program clock
(PCR). On playback the audio DAC runs on its own oscillator — typically
5–50 ppm off the broadcaster, several hundred ppm on poor SAT>IP gear.
Without active correction, lip-sync drifts by milliseconds per minute.

## Architecture

```
DVB stream (PCR)
  ├── audio PTS → decoder → ALSA ring → DAC ── GetClock() ── master
  └── video PTS → decoder → filter → jitterBuf → SyncAndSubmitFrame → display
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

`fps` is metadata-only in FFmpeg (`AVFILTER_FLAG_METADATA_ONLY`) so it
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
   simple mean. The soft path is gated on `smoothedDeltaValid`, so no
   correction fires off a partial mean.
2. **Steady-state EMA.** Integer EMA with a residual accumulator that
   carries the `diff mod N` remainder across samples — guarantees exact
   convergence to the rawDelta mean even when `|diff| < N`. Time
   constant ~1 s (`EMA_SAMPLES = 50`).

`ResetSmoothedDelta()` clears warmup, EMA, residual, hard-streaks and
catch-up state in one call. Triggered on channel switch, hard-behind
fire, catch-up exit and `WaitForAudioCatchUp`.

## Correction regimes

Symmetric: every regime has a behind and an ahead path. Hard transients
bypass the cooldown.

### Soft corridor — `|smoothed| > CORRIDOR (30 ms)`, cooldown elapsed

Let `correctMs = min(|smoothed|/90, MAX_CORRECTION_MS = 200)`.

| Direction | Action                                                                                           |
| --------- | ------------------------------------------------------------------------------------------------ |
| ahead     | `SleepMs(correctMs + frameDur)`, submit, then `smoothed −= (elapsed − frameDur) × 90`            |
| behind    | Drop `max(1, round(correctMs / frameDur))` frames (one now, rest via `pendingDrops`); per drop `smoothed += frameDur × 90` |

The `+ frameDur` padding on the sleep is load-bearing: a bare
`SleepMs(correctMs)` only lengthens the iteration by `correctMs − frameDur`
(the missing `frameDur` is absorbed by the next iteration's natural
packet wait), so without padding the smoother sees half the requested
shift and re-fires forever.

Both paths feed back the **measured** effect, not the requested amount —
this preserves the smoother across the correction and is what makes
post-correction `d ≈ 0` reliably reproducible.

`MAX_CORRECTION_MS = HARD_THRESHOLD = 200` so a single soft event can
fully close the corridor; no sub-corridor residual is left for the
controller to refuse-fire on. `CORRIDOR = 30 ms` sits above the
EMA-smoothed envelope of `snd_pcm_delay`'s IEC61937 quantization noise
(raw AC-3 bursts of 1536 samples ≈ 32 ms, heavily attenuated by the
50-sample EMA in steady state) and below the human perceptual threshold
(~80 ms). A narrower corridor would re-fire on residual quantization
noise; a wider one would let genuine drift accumulate.

### Cooldown

`COOLDOWN_MS = 5 s` = 5 EMA time constants. Armed by every soft fire,
hard transient and `WaitForAudioCatchUp`. Catch-up exit does *not* arm
it — the tighter exit threshold leaves the residual inside the corridor,
and warmup alone gates soft until the EMA reseeds. The smoother
has therefore absorbed the previous correction plus several cycles of
fresh samples before the next event can fire.

### Hard transients (raw delta, no cooldown gate)

| Condition                                | Action                                                                                |
| ---------------------------------------- | ------------------------------------------------------------------------------------- |
| `rawDelta < −HARD_THRESHOLD` (−200 ms)   | Drop frame, reset EMA, arm cooldown                                                   |
| `rawDelta > +HARD_THRESHOLD`, replay     | `WaitForAudioCatchUp()` blocks (≤ 5 s) until audio catches up, then submit            |
| `rawDelta > +HARD_THRESHOLD`, live       | One sleep ≤ `HARD_AHEAD_MAX_MS = 500 ms`, bypassing `MAX_CORRECTION_MS` and cooldown  |

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

### Catch-up — `rawDelta < −2 × HARD_THRESHOLD` (−400 ms)

Catastrophic backlog (cold start with slow codec prime, post-seek,
multi-second decoder stall, or a hardware-limited decoder whose audio
side publishes a large forward clock jump). Every incoming frame is
dropped silently — no per-event log, no EMA churn, no cooldown arm —
until `rawDelta > −CORRIDOR`. Two log lines bracket the pass:

```
vaapivideo/decoder: catch-up entered raw=-2738ms
vaapivideo/decoder: catch-up complete dropped=143 wall=314ms exit-raw=-38ms
```

Hysteresis is entry −400 ms vs exit −30 ms (370 ms gap), well above any
single-sample clock jitter. The exit at `-CORRIDOR` (rather than
`-HARD_THRESHOLD`) places the residual inside the soft corridor so no
follow-up soft event is required; a tighter exit avoids a visible
multi-second desync when `COOLDOWN + WARMUP` would otherwise block the
post-catch-up soft-behind from firing.

`SkipStaleJitterFrames()` lifts its "keep ≥ 1" guard while catching up
since the kept frame would be dropped next iteration anyway. On exit,
EMA is reset, `pendingDrops` cleared, and the exiting frame is submitted
normally. Cooldown is **not** re-armed here: the residual is inside the
corridor so soft cannot fire anyway, and warmup (~1 s) alone keeps the
controller quiescent while the EMA reseeds.

## Jitter buffer (live TV)

Audio and video share a single end-to-end cushion controlled by
`AUDIO_ALSA_BUFFER_MS` (default 400 ms, in `src/audio.cpp`).

- **Audio side** = ALSA hardware ring sized to `bufferSize =
  AUDIO_ALSA_BUFFER_MS`. Steady-state the ring stays near full so
  `GetClock()` lags wall time by roughly that amount.
- **Video side** = `jitterBuf` (`std::deque`). Drain is **due-based**:

  ```
  dueIn = headPts − GetClock() − latency
  if dueIn > halfFrame: break          // head not yet due, hold buffer
  else:                 submit head    // via SyncAndSubmitFrame
  ```

  Head frames sit in the buffer for as long as the lagged clock plus
  the broadcaster's own `video_pts − audio_pts` offset puts them in the
  future, so the deque accumulates `(MS + broadcastLead) / frameDur`
  frames at steady state.

Both sides delay by the same amount, so lip-sync is preserved. The
trade-off is channel-switch-to-audio latency: the first audio plays at
~`MS / 3` (the ALSA `startThreshold`).

### Per-stream variance

`buf` reflects the broadcaster's own offset on top of the ALSA floor.
Sampled values: SWR ~0 ms, RTLup ~400 ms, Das Erste ~500 ms, ZDF HD /
UHD1 ~700–750 ms, SES UHD Demo ~1700 ms. 4K VBR streams can swing `buf`
by a full second within seconds as bitrate peaks stall packet arrival.
None of this can be compressed without breaking lip-sync; the jitter
knob is a **floor, not a ceiling**.

### Drain bypasses

The due check is bypassed for:
- **Freerun** (`freerunFrames > 0` after `Clear()`, trick exit, audio
  codec change) — gives an instant first picture.
- **`pendingDrops`** from a soft-behind burst — one drop per drain
  iteration until exhausted.

`SkipStaleJitterFrames()` runs at the top of every drain pass and
bulk-drops heads more than `HARD_THRESHOLD` (200 ms) behind the clock.
Cheaper than routing each through catch-up.

### Cold-boot startup

On a cold VDR start, VAAPI codec init + filter graph + parser warm-up
emit the first video frame significantly later than audio reaches the
DAC. A large catch-up correction (hundreds of frames in tens of ms)
followed by a single soft-behind is normal and expected.

The cushion rebuilds over the next 1–2 s: the audio packet queue
accumulated during the OSD load drains into the ALSA ring faster than
real-time, `snd_pcm_delay` climbs from `startThreshold` toward full
`bufferSize`, and the lagged clock pulls `buf` up to its steady value.
The first sync-log line typically shows a small `buf`; one or two
intervals later it has converged.

If `buf` stays at 0 across many sync-log lines, the configured
`AUDIO_ALSA_BUFFER_MS` is too small — raise it.

### Audio packet queue (`aq`)

`aq` is the FIFO between `Decode()` and the audio thread. The thread
decodes packets as fast as they arrive and writes PCM to ALSA, so `aq`
drains almost instantaneously and `aq = 0` is healthy. A persistently
non-zero `aq` indicates the audio decoder is falling behind real-time
(CPU contention, ALSA write stall). The real audio cushion is the
**ALSA ring**, not this queue.

Replay mode skips the jitter buffer entirely — the source is local and
there is no arrival jitter to absorb.

## Sync bypass

The sync gate is bypassed (frame submitted unpaced) in:

- Trick mode (`SubmitTrickFrame()` paces via its own timer; audio is muted).
- Freerun window after `Clear()`, trick exit or `NotifyAudioChange()`.
- Radio mode / NOPTS frame (no audio processor or no PTS to align on).
- Audio not yet running (`GetClock()` is NOPTS until the first
  `WritePcmToAlsa()` fires).

## Lifecycle

| Event                      | EMA            | Cooldown   | Jitter buffer                    |
| -------------------------- | -------------- | ---------- | -------------------------------- |
| Plugin start               | invalid        | —          | empty                            |
| Channel switch (`Clear()`) | reset          | —          | flushed; freerun armed           |
| Catch-up enter             | (drops silent) | —          | drained silently to alignment    |
| Catch-up exit              | reset          | —          | one frame submitted normally     |
| Soft drop                  | `+= frameDur × 90` per drop | armed | one frame removed per drop |
| Soft sleep                 | `−= measured`  | armed      | unchanged                        |
| Hard-behind                | reset          | armed      | one frame dropped                |
| Hard-ahead (replay)        | (post-wait resync) | armed  | unchanged                        |
| Hard-ahead (live)          | `−= measured`  | armed      | unchanged                        |
| Audio codec / track change | unchanged      | unchanged  | preserved; freerun armed         |

Audio codec / track change preserves the buffer across the switch — the
catch-up path will silently realign against the new clock once it
arrives, so dropping ~1 s of still-valid video buys nothing.

## Log format

```
sync d=+12.5ms avg=+11.7ms lat=40ms buf=50 aq=0 miss=0 drop=0 skip=0
```

| Field  | Meaning                                                                        |
| ------ | ------------------------------------------------------------------------------ |
| `d`    | Interval mean of `rawDelta` since the last log; comparable to `avg`            |
| `avg`  | EMA-smoothed delta; drives every soft-correction decision                      |
| `lat`  | Active `SyncLatency90k` (2-frame tail + active operator knob)                  |
| `buf`  | `jitterBuf` depth in frames at log emission                                    |
| `aq`   | Audio packet queue depth                                                       |
| `miss` | Drain gaps > 2 × output frame period since last log (vsync backpressure, our sleeps) |
| `drop` | Frames dropped (video behind) since last log — soft + hard combined            |
| `skip` | Render delays (video ahead) since last log — soft sleep + hard-ahead combined  |

`d ≈ avg` in steady state means the EMA has converged on current
reality. The line is suppressed during warmup and reissued immediately
on warmup completion. Periodic interval `LOG_INTERVAL_MS = 2 s`.

**Healthy steady state:** `avg` inside `±CORRIDOR`, `d ≈ avg`, `buf`
stable, `miss = drop = skip = 0`.

Each soft / hard event also emits a per-event `dsyslog` line naming the
cause (`soft-ahead`, `soft-behind`, `hard-ahead live`, `hard-ahead
replay`, `hard-behind`, `stale-jitter bulk`, `catch-up entered`,
`catch-up complete`) for "why did this fire?" without waiting for the
next periodic line.

### Steady-state offset

The EMA does not necessarily settle at zero — a constant few-ms offset
typically reflects residual HDMI/TV scanout bias beyond the 2-frame
tail. Inside `±CORRIDOR` the controller leaves it alone (correcting a
non-drifting bias would only introduce visible jank). To re-center,
tune `PcmLatency` or `PassthroughLatency`.

## Constants

Sync constants live at file scope in [src/decoder.cpp](src/decoder.cpp);
the cushion floor sits in [src/audio.cpp](src/audio.cpp). Each carries a
`///<` comment with purpose and unit.

| Constant                         | Value | Purpose                                                                 |
| -------------------------------- | ----- | ----------------------------------------------------------------------- |
| `AUDIO_ALSA_BUFFER_MS`           | 400   | ALSA ring size (ms); lagged audio clock pulls video buf to ~MS/frameDur |
| `DECODER_SYNC_HARD_THRESHOLD`    | 200   | Raw-delta threshold for hard transients (ms); 2× = catch-up entry       |
| `DECODER_SYNC_CORRIDOR`          | 30    | Soft corridor half-width (ms); above EMA-smoothed IEC61937 quantization |
| `DECODER_SYNC_MAX_CORRECTION_MS` | 200   | Soft-event cap (= HARD_THRESHOLD) so one event fully closes corridor    |
| `HARD_AHEAD_MAX_MS` *(local)*    | 500   | Live hard-ahead sleep cap (ms)                                          |
| `DECODER_SYNC_COOLDOWN_MS`       | 5000  | Min interval between soft corrections (= 5 EMA time constants)          |
| `DECODER_SYNC_EMA_SAMPLES`       | 50    | EMA divisor (~1 s @ 50 fps); residual accumulator → exact convergence   |
| `DECODER_SYNC_WARMUP_SAMPLES`    | 50    | Samples averaged before EMA seed (~1 s @ 50 fps)                        |
| `DECODER_SYNC_FREERUN_FRAMES`    | 1     | Unpaced frames after sync-disrupting events                             |
| `DECODER_SYNC_LOG_INTERVAL_MS`   | 2000  | Periodic sync diagnostic interval (ms)                                  |
| `DECODER_QUEUE_CAPACITY`         | 200   | Video packet queue depth (~4 s @ 50 fps)                                |
| `AUDIO_CLOCK_STALE_MS`           | 1000  | `GetClock()` extrapolation timeout before returning NOPTS               |
| `AUDIO_QUEUE_CAPACITY`           | 300   | Audio packet queue depth (~10 s AC-3); sized for slow-start decoders    |
