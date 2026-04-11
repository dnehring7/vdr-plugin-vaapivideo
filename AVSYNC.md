# A/V Synchronization

## Problem

A DVB stream encodes audio and video PTS values against a single 90 kHz program
clock (PCR). On playback the audio DAC is driven by a sink oscillator that is
independent of the source — 5–50 ppm off on healthy hardware, several hundred
ppm on pathological SAT>IP deployments. Without active correction, lip sync
drifts by milliseconds per minute.

## Design

```
DVB stream (PCR)
    │
    ├─► Audio PTS ─► decoder ─► ALSA ─► DAC
    │                                   │
    │                             GetClock()  ◄── master clock
    │                                   │
    │                                   ▼
    └─► Video PTS ─► decoder ─► filter ─► jitter buffer ─► SyncAndSubmitFrame
```

**Three invariants.**

1. **Audio is master.** `cAudioProcessor::GetClock()` returns the PTS of the
   sample currently at the DAC: `playbackPts + (age × 90)`, where
   `age = cTimeMs::Now() − lastClockUpdateMs`. ALSA writes refresh
   `playbackPts` once per ~25 ms period; the age-compensated read gives a
   live DAC-position estimate instead of a piecewise-constant snapshot, so
   `rawDelta` has scheduling-jitter precision (~1 ms) rather than period-size
   quantisation (~25 ms). Beyond `AUDIO_CLOCK_STALE_MS = 1 s`, or before PCM
   has been written, `GetClock()` returns `AV_NOPTS_VALUE` and the controller
   falls through to the unpaced freerun branch.

2. **Audio is never resampled.** No `swr_set_compensation`, no software PLL.
   Audio plays at the DAC's native rate; only video adapts. This keeps the
   system stateless across channel switches and eliminates the feedback-loop
   instabilities that plague software audio resampling.

3. **Video is uniformly 50 fps (or display refresh).** The filter graph
   upconverts sub-display-rate progressive sources with `fps=<displayFps>`
   so 25p, 25i-deinterlaced, and native 50p all reach `SyncAndSubmitFrame()`
   at the same cadence. One controller regime covers every source type.

## Filter pipeline

```
SW decode: [bwdif] → [hqdn3d] → format=nv12 → hwupload → scale_vaapi → [sharpness_vaapi] [→ fps]
HW decode: [deinterlace_vaapi=rate=field] → [denoise_vaapi] → scale_vaapi → [sharpness_vaapi] [→ fps]
```

The trailing `fps=<displayFps>` node is appended whenever the natural output
rate (source fps, doubled for `rate=field` deinterlacing) is below the
display refresh rate. `fps` has `AVFILTER_FLAG_METADATA_ONLY` in FFmpeg so it
duplicates `AVFrame` references without touching pixel data and works equally
well with VAAPI surfaces and SW frames. Placement at the chain tail is
deliberate: scale/denoise/sharpen each run once per *input* frame, only the
metadata is duplicated to hit the target cadence.

## Measurement

For every video frame, `SyncAndSubmitFrame()` computes:

```
rawDelta = videoPTS − GetClock() − pipelineLatency
```

`rawDelta > 0` = video ahead of audio; `rawDelta < 0` = behind.

### Pipeline latency

`SyncLatency90k(ap)` returns the operator knob plus a fixed tail of **two
output frame periods** (`2 × outputFrameDurationMs`):

1. **Decoder → scanout.** A frame submitted in iteration N appears at vsync
   N+1 — one full frame period.
2. **HDMI link + panel input lag.** Another empirical frame period, not
   exposed by any DRM API.

Two operator knobs, selected per-stream via `cAudioProcessor::IsPassthrough()`:

| Mode                 | Setup-menu key                   | `setup.conf` key     | Range       | Default |
| -------------------- | -------------------------------- | -------------------- | ----------- | ------- |
| PCM (decoded by us)  | `PCM Audio Latency (ms)`         | `PcmLatency`         | −200…200 ms | 0       |
| IEC61937 passthrough | `Passthrough Audio Latency (ms)` | `PassthroughLatency` | −200…200 ms | 0       |

Defaults are zero because the 2-frame tail already covers typical HDMI/TV
bias. Positive → audio delayed (TV in cinema/movie mode); negative → audio
advanced (TV in gaming/bypass mode, slow AV receiver).

### Two-phase smoother

Raw delta carries ~150 ms field-alternation aliasing on deinterlaced 50p. It
is used only for the hard-transient threshold and as a log diagnostic; every
soft-correction decision goes through the smoother.

1. **Warmup.** After any reset (channel switch, hard transient, prime-sync),
   the first `WARMUP_SAMPLES = 50` samples feed a simple mean instead of the
   EMA. At 50 fps the window is ~1 s — within the cooldown, so no soft
   correction fires off a partial mean. `√50 ≈ 7×` seed-bias reduction.

2. **Steady-state residual EMA.**

   ```
   residual += rawDelta − smoothed
   step      = residual / EMA_SAMPLES      (integer)
   smoothed += step
   residual -= step × EMA_SAMPLES
   ```

   With `EMA_SAMPLES = 50` the time constant is ~1 s. The residual accumulator
   carries the `diff mod N` remainder that integer division would otherwise
   discard, so the smoother converges **exactly** to the rawDelta mean instead
   of stalling whenever `|diff| < N` ticks (as a plain integer EMA does).

The soft path is gated on `smoothedDeltaValid`, so nothing fires during
warmup.

## Correction

Correction is **symmetric**: both directions use the same proportional,
rate-limited mechanism. Hard transients bypass the rate limit.

### Soft corridor (`|smoothed| > CORRIDOR = 20 ms`, cooldown elapsed)

`CORRIDOR` is sized at `frameDur` (20 ms at 50 fps) because any sleep shorter
than `frameDur` is absorbed into the natural packet wait at the top of the
next `Action()` iteration — a narrower corridor would re-fire forever without
effect.

Let `correctMs = min(|smoothed|/90, MAX_CORRECTION_MS = 100)`.

| Smoothed                | Action                                                                                                                                  |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `> +CORRIDOR` (ahead)   | `SleepMs(correctMs + frameDur)`, submit; post-submit adjust `smoothed −= (elapsed − frameDur) × 90` (measured after `SubmitFrame`)      |
| `< −CORRIDOR` (behind)  | Drop `max(1, round(correctMs / frameDur))` frames: first now, remainder via `pendingDrops`; adjust `smoothed += frameDur × 90` per drop |

Both paths adjust `smoothed` by the **measured** effect of the correction,
not by the requested amount. This preserves the smoother's history — the next
sample no longer reseeds from a single noisy raw value, which was the dominant
source of post-correction oscillation in earlier designs.

The `+ frameDur` padding on the sleep is critical. A bare `SleepMs(correctMs)`
only lengthens the iteration by `(correctMs − frameDur)`: the missing
`frameDur` is absorbed by the next iteration's natural packet wait (the packet
was queued while we were sleeping). Padding forces the iteration one full
frame period past normal, so the measured `extraMs ≈ correctMs` and the
controller converges in a single fire. Drop rounding is nearest-integer,
bounding the residual to `±frameDur/2`.

### Cooldown

One event per `COOLDOWN_MS = 5 s` = 5 EMA time constants, so the smoother has
absorbed the previous correction plus several cycles of fresh samples before
re-firing. Also armed by hard transients, prime-sync, and
`WaitForAudioCatchUp`. With `MAX_CORRECTION_MS = 100` per event the soft
controller absorbs up to ~20 ms/s sustained drift — orders of magnitude above
any realistic stream. A healthy stream requires zero corrections in steady
state.

### Hard transients (raw delta, not gated by cooldown)

| Condition                                   | Action                                                                                                     |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `rawDelta < −HARD_THRESHOLD` (−200 ms)      | Drop frame, reset EMA, clear `pendingDrops`, arm cooldown                                                  |
| `rawDelta > +HARD_THRESHOLD`, replay        | `WaitForAudioCatchUp()` blocks until audio catches up (capped at 5 s), then submit; arm cooldown           |
| `rawDelta > +HARD_THRESHOLD`, live          | One sleep up to `HARD_AHEAD_MAX_MS = 500 ms`, bypassing `MAX_CORRECTION_MS` and cooldown; measured adjust  |

The live-mode hard-ahead path exists because the soft corridor caps at
20 ms/s correction rate. A marginal transponder can drift faster than that,
and the soft controller then chases the bias indefinitely. A single ~500 ms
glitch to drag the system back into the soft corridor is a better UX than an
indefinite stall, and the 500 ms cap prevents the upstream packet queue from
overflowing during the sleep.

The 200 ms threshold sits well above the EMA noise floor and below the
perceptual annoyance threshold for a single corrective event.

## Jitter buffer (live TV)

A count-based buffer at ~1 s of output rate (`DECODER_JITTER_BUFFER_MS = 1000`
→ 50 frames at 50 fps) absorbs DVB-over-IP arrival jitter and weak-signal
bursts. Three states:

1. **Filling.** Frames accumulate until `jitterTarget` is reached.

2. **Prime-sync.** `RunJitterPrimeSync()` performs a one-shot coarse alignment
   of the buffer head against the audio clock before the first drain:
   - **Behind** (`initDelta < 0`): drop stale heads until the head pts
     reaches the clock.
   - **Ahead** (`initDelta > 0`): busy-wait in 10 ms slices until audio
     catches up, then trim any heads that overshot during the wait. At least
     one frame is always retained so the display is not starved.

   If the audio clock is still `AV_NOPTS_VALUE` when prime-sync runs (typical
   right after a track switch), the one-shot is **deferred** rather than
   fired against a clockless buffer. The buffer continues to drain via
   `SyncAndSubmitFrame()`'s "no clock" branch, and the alignment runs the
   moment a real clock arrives. This avoids the failure mode where the
   one-shot fires against NOPTS and never re-runs, leaving the EMA to seed
   from a several-hundred-ms transient skew.

   Prime-sync resets the EMA and arms the cooldown so the steady-state
   controller does not react to the (possibly large) raw deltas right after
   the coarse jump.

3. **Draining.** The drain loop consumes exactly `pushedCount` fresh frames
   per `Action()` iteration — the same count the decoder just pushed (1 for
   native 50p, 2 for 25i `rate=field` or 25p upconverted) — plus any
   `pendingDrops` carried over from a soft-correction burst. This keeps the
   buffer level constant from iteration to iteration regardless of source
   type. `SubmitFrame()`'s single-slot VSync backpressure paces the loop.

`SkipStaleJitterFrames()` purges heads more than `HARD_THRESHOLD` behind the
audio clock at CPU speed, in one shot, so the soft path doesn't have to drop
them one-by-one and burn a per-drop grace window. At least one frame is
always retained.

Replay mode skips the jitter buffer entirely — the source is local and there
is no arrival jitter to absorb.

## Sync bypass

The sync gate is bypassed in these cases (frame submitted unpaced):

- **Trick mode** — `SubmitTrickFrame()` paces via its own timer; audio is muted.
- **Freerun window** — `freerunFrames > 0` after `Clear()`, trick exit, or
  `NotifyAudioChange()`; the audio clock has not yet re-anchored.
- **No audio processor / NOPTS frame** — radio mode, MPEG-2 B-frames without PTS.
- **Audio not yet running** — `GetClock()` is NOPTS until `WritePcmToAlsa()`
  has fired at least once.

## Lifecycle

| Event                          | EMA                          | Cooldown  | Jitter buffer                                 |
| ------------------------------ | ---------------------------- | --------- | --------------------------------------------- |
| Plugin start                   | invalid                      | —         | empty                                         |
| Channel switch (`Clear()`)     | reset                        | —         | flushed; freerun armed                        |
| Jitter prime-sync              | reset                        | armed     | aligned by `RunJitterPrimeSync()`             |
| Soft drop                      | `+= frameDur × 90` per drop  | armed     | one frame removed per drop                    |
| Soft sleep                     | `-= measured extra`          | armed     | unchanged                                     |
| Hard transient (behind)        | reset                        | armed     | one frame dropped                             |
| Hard-ahead (replay)            | (post-wait resync)           | armed     | unchanged                                     |
| Hard-ahead (live)              | `-= measured extra`          | armed     | unchanged                                     |
| Audio codec / track change     | unchanged                    | unchanged | preserved; freerun armed; prime-sync re-armed |

"Audio codec / track change" is the non-obvious row: dropping the buffer
would strand ~1 s of valid video and force a full re-prime, so we re-arm the
prime-sync one-shot against the existing buffer and let it run the moment a
new audio clock arrives.

## Log format

```
sync d=+12.5ms avg=+11.7ms lat=40ms buf=50 aq=0 miss=0 drop=0 skip=0
```

| Field  | Meaning                                                                                                                     |
| ------ | --------------------------------------------------------------------------------------------------------------------------- |
| `d`    | **Interval mean** of raw delta since the last log (tenths of ms). Directly comparable to `avg` — both are means             |
| `avg`  | EMA-smoothed delta (tenths of ms); drives every soft-correction decision                                                    |
| `lat`  | Active `SyncLatency90k` value in ms (2-frame tail + operator knob); flags whether the PCM or passthrough knob is in effect  |
| `buf`  | Jitter buffer depth (frames) at log emission                                                                                |
| `aq`   | Audio queue depth (packets)                                                                                                 |
| `miss` | Per-frame drain gaps > 2 × output frame period since last log (vsync backpressure, decode stall, our own sleeps)            |
| `drop` | Frames dropped (video behind) since last log — soft + hard combined                                                         |
| `skip` | Render delays (video ahead) since last log — soft sleep + live hard-ahead + replay hard-hold combined                       |

`d` is the interval mean, not a point sample, so `d ≈ avg` in steady state
means the EMA has converged on the current reality. The log line is
suppressed during warmup and reissued immediately after warmup completes.
Periodic interval is `LOG_INTERVAL_MS = 30 s`.

**Healthy steady state:** `avg` inside `±CORRIDOR`, `d ≈ avg`, `buf` stable,
`miss = drop = skip = 0`.

### Steady-state offset

The EMA does not necessarily settle at zero. A constant few-millisecond
offset typically reflects residual HDMI scanout / TV input lag beyond the
two output-frame periods that `SyncLatency90k()` already covers. As long as
the offset is inside `±CORRIDOR` the controller leaves it alone: correcting
a non-drifting bias would only introduce visible jank. Operators who want
`avg` to centre on zero can absorb the bias by tuning the appropriate
latency knob (`PcmLatency` for PCM output, `PassthroughLatency` for IEC61937).

## Constants

All sync constants live in [src/decoder.cpp](src/decoder.cpp) at file scope as
`constexpr` values. Grep for `DECODER_SYNC_` and `DECODER_JITTER_`; each
carries a `///<` comment documenting purpose and unit.

| Constant                          | Value  | Purpose                                                               |
| --------------------------------- | ------ | --------------------------------------------------------------------- |
| `DECODER_SYNC_HARD_THRESHOLD`     | 200 ms | Raw-delta threshold for emergency drop / hold                         |
| `DECODER_SYNC_CORRIDOR`           | 20 ms  | Soft corridor = frameDur at 50 fps                                    |
| `DECODER_SYNC_MAX_CORRECTION_MS`  | 100    | Soft-event cap (sleep ms or drop-burst ms)                            |
| `HARD_AHEAD_MAX_MS` *(local)*     | 500    | Live-mode hard-ahead sleep cap                                        |
| `DECODER_SYNC_COOLDOWN_MS`        | 5000   | Min interval between soft corrections (= 5 EMA time constants)        |
| `DECODER_SYNC_EMA_SAMPLES`        | 50     | EMA divisor (~1 s @ 50 fps; residual accumulator → exact convergence) |
| `DECODER_SYNC_WARMUP_SAMPLES`     | 50     | Samples averaged before EMA seed (~1 s @ 50 fps)                      |
| `DECODER_SYNC_FREERUN_FRAMES`     | 1      | Unpaced frames after sync-disrupting events (Clear, track switch, …)  |
| `DECODER_SYNC_LOG_INTERVAL_MS`    | 30000  | Periodic sync diagnostic interval                                     |
| `DECODER_JITTER_BUFFER_MS`        | 1000   | Jitter buffer target depth (absorbs weak-signal arrival bursts)       |
| `DECODER_QUEUE_CAPACITY`          | 200    | Video packet queue depth (~4 s @ 50 fps / ~8 s @ 25 fps)              |
| `AUDIO_CLOCK_STALE_MS` *(audio)*  | 1000   | Max age for `GetClock()` extrapolation before returning NOPTS         |
| `AUDIO_QUEUE_CAPACITY` *(audio)*  | 100    | Audio packet queue depth (~3 s at AC-3 ~32 ms framing)                |
