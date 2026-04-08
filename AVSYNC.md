# A/V Synchronization

## Problem

A DVB stream carries audio and video PTS values derived from a single 90 kHz
program clock (PCR). On the playback device the audio DAC clock is driven by
the HDMI sink's pixel clock -- an independent oscillator that differs from the
DVB source by 5-50 ppm (normal hardware) up to several hundred ppm
(pathological SAT>IP setups). Without correction, lip-sync degrades by
milliseconds per minute.

## Design

```
DVB stream (PCR)
    |
    +--> Audio PTS --> decoder --> ALSA --> DAC  (HDMI sink clock)
    |                                |
    |                            GetClock()    <-- master clock
    |                                v
    +--> Video PTS --> decoder --> filter --> jitter buffer --> display
                                                                   ^
                                                       SyncAndSubmitFrame
```

**Audio is the master clock.** `cAudioProcessor::GetClock()` returns the PTS
of the sample currently at the DAC output: `endPts - snd_pcm_delay`, updated
once per ALSA write period (~24-32 ms). The value is piecewise-constant
between writes.

Audio is **not** rate-corrected -- no `swr_set_compensation()`, no software
PLL. Audio plays at the DAC's native rate; only video adapts. This keeps the
system stateless across channel switches and eliminates feedback-loop
instabilities.

## Measurement

For every video frame, `SyncAndSubmitFrame()` computes:

```
rawDelta = videoPTS - GetClock() - pipelineLatency
```

Sign convention: `rawDelta > 0` means video is ahead of audio,
`rawDelta < 0` means video is behind.

`pipelineLatency` (`SyncLatency90k()`) is the configured external audio
latency (AV receiver) plus one output frame period (DRM atomic-commit ->
next VSync scanout).

The raw delta carries large aliasing noise -- on deinterlaced 50p output it
oscillates by ~150 ms between alternating fields, because `GetClock()` updates
at the ALSA write period (~30 ms) while video drains at the output frame rate.
The smoother is therefore strictly necessary for any correction decision; the
raw delta only appears in logs as a diagnostic.

### Two-phase smoother

1. **Warmup.** After every reset (channel switch, hard transient, prime-sync)
   the first `WARMUP_SAMPLES = 50` samples are accumulated into a simple mean
   instead of feeding the EMA directly. Seeding the EMA from a single noisy
   sample would bias it for an entire time constant; the 50-sample mean cuts
   the seed error by `sqrt(50) ~= 7x`. At 50 fps the warmup window is ~1 s
   (~2 s at 25 fps), well within the cooldown.

2. **Steady state.** Once seeded, the smoother is a single-pole IIR
   low-pass:

   ```
   smoothed += (rawDelta - smoothed) / EMA_SAMPLES
   ```

   With `EMA_SAMPLES = 250` the time constant is ~5 s at 50 fps, sufficient
   to suppress the per-frame oscillation to a few ms while still tracking
   real drift.

The soft-correction path is gated on `smoothedDeltaValid`, so corrections are
suppressed during warmup.

## Correction

Correction is **symmetric**: both directions of error are handled by the same
proportional, rate-limited mechanism. Hard transients bypass the rate limit
because they represent emergencies that cannot wait.

### Soft corrections (rate-limited, proportional)

When `|smoothed| > CORRIDOR` (50 ms) and the cooldown timer has expired, one
correction event fires. The amount is "enough to bring the smoothed value back
to ~0", capped at `MAX_CORRECTION_MS = 100 ms` per event:

| Smoothed value | Action |
|---|---|
| `smoothed > +CORRIDOR` (video ahead) | `cCondWait::SleepMs(min(smoothed/90, MAX_CORRECTION_MS))`, then submit |
| `smoothed < -CORRIDOR` (video behind) | Drop `ceil(|smoothed|/frameDur)` frames, capped at `MAX_CORRECTION_MS`. The first drop happens immediately; remaining drops are queued in `pendingDrops` and executed one per call so the jitter buffer is not flushed in a burst |

Both paths adjust `smoothedDelta90k` by the **exact known correction amount**
instead of resetting the EMA:

```
sleep N ms     ->  smoothedDelta -= N * 90   (audio advances, video sits still)
drop one frame ->  smoothedDelta += frameDur * 90  (next display PTS jumps forward)
```

This preserves the smoother's history. The next sample no longer reseeds from
a single noisy raw value, which was the dominant source of post-correction
oscillation in earlier designs.

### Cooldown

A single `cTimeMs syncCooldown` rate-limits soft corrections to one event per
`COOLDOWN_MS = 5 s` (chosen to equal one EMA time constant, so the smoother
has had a full window of samples since the last correction). The cooldown is
also armed by hard transients, prime-sync completion, and `WaitForAudioCatchUp`
so that the steady-state controller does not react on top of those events.

With `MAX_CORRECTION_MS = 100 ms` per cooldown the controller can absorb up to
~20 ms/s sustained drift while holding the corridor -- orders of magnitude
above any realistic stream. The actual correction rate is governed by the
corridor and physical drift, not the cooldown ceiling: a healthy stream
typically requires zero corrections in steady state.

### Hard transients (raw delta, large excursion)

Hard transients are **not** gated by the cooldown -- a burst of stale frames
must be flushed immediately or audio runs visibly ahead.

| Condition | Action |
|---|---|
| `rawDelta < -HARD_THRESHOLD` (-200 ms) | Drop frame, reset EMA, clear `pendingDrops`, arm cooldown |
| `rawDelta > +HARD_THRESHOLD` (+200 ms), replay only | `WaitForAudioCatchUp()` blocks until audio catches up (capped at 5 s), then submit; reset EMA, arm cooldown |

The replay-side hold is disabled in live mode: a live source cannot be paused
server-side, so blocking would just grow the upstream buffer until it
overflows. Live transients are handled by the negative branch only.

The 200 ms threshold sits well above the EMA noise floor and below the
perceptual annoyance threshold for a single corrective event.

## Jitter buffer (live TV)

A count-based jitter buffer (~500 ms at the output frame rate) absorbs
DVB-over-IP packet arrival jitter. Three states:

1. **Filling** -- frames accumulate until the target depth is reached.
2. **Prime-sync** -- `RunJitterPrimeSync()` performs a one-shot coarse
   alignment of the buffer head against the audio clock before the first
   drain:
   - **Behind** (`initDelta < 0`): drop stale frames until the head reaches
     the clock. If the buffer empties, the caller re-primes on the next
     refill.
   - **Ahead** (`initDelta > 0`): busy-wait in 10 ms slices until audio
     catches up, then drop any frames that overshot during the wait. At
     least one frame is always retained so the display is not starved.
   Prime-sync resets the EMA and arms the cooldown so the steady-state
   controller does not react to the (possibly large) raw deltas seen
   immediately after the coarse jump.
3. **Draining** -- one frame per `Action()` iteration, paced by
   `SubmitFrame()`'s single-slot VSync backpressure (not a wall-clock timer).

The buffer level is **unregulated**: audio and video share the source rate, so
the level naturally tracks the audio queue depth.
`SkipStaleJitterFrames()` purges frames more than `HARD_THRESHOLD` behind
audio at CPU speed, in one shot, so the soft path does not have to drop them
one-by-one. At least one frame is always retained.

## Sync bypass

Sync is bypassed in these cases (the frame is submitted unpaced):

- **Trick mode** -- `SubmitTrickFrame()` paces via a per-frame timer; the
  audio clock is irrelevant because audio is muted.
- **Freerun window** -- `freerunFrames > 0` after `Clear()` or after a
  trick-mode exit; the audio clock has not yet re-anchored to the new PTS
  stream.
- **No audio processor / no PTS** -- radio mode, MPEG-2 B-frames without PTS,
  etc.
- **Audio not yet running** -- `GetClock()` returns `AV_NOPTS_VALUE` until
  the audio thread has pushed PCM to the sink.

## Lifecycle

| Event | EMA | Cooldown | Jitter buffer |
|---|---|---|---|
| Plugin start | invalid | -- | empty |
| Channel switch (`Clear()`) | reset | -- | flushed, freerun armed |
| Jitter prime-sync | reset | armed | aligned by `RunJitterPrimeSync()` |
| Soft drop or sleep | EMA adjusted by exact correction | armed | one frame removed (drop only) |
| Hard transient | reset | armed | unchanged |
| `WaitForAudioCatchUp` (replay) | reset | armed | unchanged |
| Audio codec change | -- | -- | freerun re-armed |

## Log format

```
sync d=+12ms avg=+5.4ms buf=15 aq=0 miss=0 drop=0 skip=0
```

| Field | Meaning |
|---|---|
| `d`    | Raw instantaneous delta (noisy -- diagnosis only) |
| `avg`  | EMA-smoothed delta (drives all soft correction decisions) |
| `buf`  | Jitter buffer depth (frames) |
| `aq`   | Audio queue depth (packets) |
| `miss` | Drain stalls > 2x output frame period since last log (any cause: vsync backpressure, decode stall, our own sleeps) |
| `drop` | Frames dropped (video behind) since last log -- soft + hard combined |
| `skip` | Render delays (video ahead) since last log -- soft sleep + replay-only hard hold combined |

The log line is suppressed during EMA warmup (the partial mean would be
misleading) and reissued immediately after warmup completes. The periodic
interval is `LOG_INTERVAL_MS = 30 s`.

Healthy steady state: `avg` inside `+/-CORRIDOR`, `buf` stable,
`aq ~= 0`, `miss = drop = skip = 0`.

Relationship between counters:
- `skip` is a subset of `miss` on output frame periods <= 50 ms (e.g. 50 fps).
- `miss - skip` ~= stalls from external causes (display backpressure, decode
  hiccups, scheduler latency).
- `drop > 0` with `avg` strongly negative -> upstream is delivering video
  late or the decoder is falling behind.

## Steady-state offset

The EMA does not necessarily settle at zero. A constant offset of a few tens
of milliseconds typically reflects the unaccounted-for HDMI scanout / TV
input lag beyond the single output-frame period that `SyncLatency90k()`
includes. As long as the offset is inside `+/-CORRIDOR` the controller leaves
it alone -- correcting a bias that is not actually drifting would only
introduce visible jank. Users who want the avg to centre on zero can absorb
the bias by raising the configured `audioLatency` parameter for their
hardware.

## Constants

All sync constants are in [src/decoder.cpp](src/decoder.cpp) (file-scope
`constexpr`). Grep for `DECODER_SYNC_` and `DECODER_JITTER_` to find them.
Each constant has a one-line `///<` comment documenting purpose and unit.

Current values:

| Constant | Value | Purpose |
|---|---|---|
| `DECODER_SYNC_HARD_THRESHOLD`     | 200 ms | Raw-delta threshold for emergency drop / hold |
| `DECODER_SYNC_CORRIDOR`           | 50 ms  | Soft target corridor (`|EMA|` outside this triggers correction) |
| `DECODER_SYNC_COOLDOWN_MS`        | 5000   | Min interval between soft corrections (= 1 EMA time constant) |
| `DECODER_SYNC_MAX_CORRECTION_MS`  | 100    | Cap on a single correction event |
| `DECODER_SYNC_EMA_SAMPLES`        | 250    | EMA divisor (~5 s time constant @ 50 fps) |
| `DECODER_SYNC_WARMUP_SAMPLES`     | 50     | Samples averaged before EMA seed (~1 s @ 50 fps) |
| `DECODER_SYNC_FREERUN_FRAMES`     | 1      | Unpaced frames after `Clear()` to show first picture fast |
| `DECODER_SYNC_LOG_INTERVAL_MS`    | 30000  | Periodic sync diagnostic interval |
| `DECODER_JITTER_BUFFER_MS`        | 500    | Jitter buffer target depth |
