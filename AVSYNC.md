# A/V Synchronization

## Problem

A DVB stream carries audio and video PTS values derived from a single 90 kHz
program clock (PCR).  On the playback device, the audio DAC clock is driven by
the HDMI sink's pixel clock — an oscillator entirely independent of the DVB
source.  The two clocks differ by 5–50 ppm in normal hardware and up to a few
hundred ppm on pathological SAT>IP setups.  Without intervention, audio and
video drift apart by milliseconds per minute, eventually breaking lip sync.

This document describes how `vdr-plugin-vaapivideo` keeps them aligned.

## Architecture

```
DVB stream (PCR)
    │
    ├─► Audio PTS → audio decoder → ALSA → DAC (HDMI sink clock)
    │                                  │
    │                              GetClock() = endPts − snd_pcm_delay
    │                                  │  (master clock for sync)
    │                                  ▼
    └─► Video PTS → video decoder → filter ─► jitter buffer ─► display
                                                                  ▲
                                                       SyncAndSubmitFrame
                                                       (adapts to audio)
```

**Audio is the master clock.**  `cAudioProcessor::GetClock()` returns the PTS
of the sample currently at the DAC output, computed as `endPts − snd_pcm_delay`
on every ALSA write (one ALSA period, ~24–32 ms).  The value is piecewise-
constant between writes — there is no interpolation.

Audio is **not** rate-corrected.  No `swr_set_compensation()`, no software PLL.
Audio plays at the DAC's native rate; video adapts.  This makes the system
stateless across channel switches and removes an entire class of feedback-loop
bugs that plagued earlier PI-controller designs.

## Measurement

For every drained video frame, `SyncAndSubmitFrame()` computes:

```
delta = videoPTS − audioClock − pipelineLatency
```

- `delta = 0`  perfect sync
- `delta < 0`  video behind audio
- `delta > 0`  video ahead of audio

`pipelineLatency` accounts for the configured external audio latency plus one
frame period for the DRM scanout pipeline.

The raw delta carries ±50 ms aliasing noise because `GetClock()` updates only
once per ALSA period, while video drains at 50 fps.  An EMA low-pass filter
smooths it:

```cpp
smoothedDelta += (delta − smoothedDelta) / 250
```

α = 1/250 corresponds to a ~5 s time constant at 50 fps — long enough to
suppress the period-aliasing noise to a ±5–10 ms residual, short enough that
real drift converges within seconds.  All sync decisions use the smoothed
value; the raw delta is logged only as a diagnostic.

## Correction

The corrective action is asymmetric, by physics:

| Condition | Action |
|---|---|
| `\|smoothedDelta\| ≤ TARGET_BAND` (±15 ms) | Submit |
| `smoothedDelta > +TARGET_BAND` | Sleep `(smoothedDelta − 0)` ms (capped at `MAX_WAIT_MS`), then submit |
| `smoothedDelta < −TARGET_BAND` | (no action; see below) |
| `delta < −HARD_THRESHOLD` (−200 ms) | Drop the frame, reseed EMA, arm grace |
| `delta > +HARD_THRESHOLD` (replay only) | Block until audio catches up, reseed EMA, arm grace |

### Why scale the wait by the excess

A threshold-edge controller (sleep one frame whenever `smoothed > BAND`) parks
the system at the threshold: it crosses the band, sleeps one frame, decays back
into the band, crosses again, and repeats.  Sleeping the **excess** drives the
smoothed value to ~0 in a single shot.  After the sleep we apply
`smoothedDelta -= waitMs * 90` — a precise adjustment, because the audio clock
has advanced by exactly `waitMs` while the held video PTS did not.

A 1 s grace period after each correction prevents the slow EMA from triggering
a second wait while the previous adjustment is still propagating.

### Why no behind-side correction

The audio clock is `endPts − snd_pcm_delay`, where `endPts` advances at the
source PTS rate (DVB PCR).  In the long run the audio clock therefore tracks
the source rate exactly: the audio path **cannot** run permanently faster than
the stream.  Any sustained `delta < 0` is either a transient (handled by the
hard drop) or EMA noise.  Dropping frames on noise injects visible artifacts to
"fix" something that isn't real, so the soft path is wait-only.

The hard drop covers the real cases: channel switches, network gaps, decoder
hiccups.  After it fires, the EMA is reseeded and a grace period gives the
fresh measurement room to stabilize.

### Wait cap

`MAX_WAIT_MS = 80` limits a single sleep to ~5× the normal frame period.  On a
big transient (e.g. a channel switch leaves video 200 ms ahead of audio), the
correction is split across multiple grace cycles instead of one ~200 ms freeze.

## Jitter buffer

Live TV uses a 25-frame (~500 ms) jitter buffer to absorb DVB-over-IP packet
arrival jitter.  The buffer has three lifecycle states:

1. **Filling** — pending frames accumulate until the buffer reaches its target
   depth.
2. **Prime-sync** — one-shot pre-drain alignment by `RunJitterPrimeSync()`:
   - If video is behind audio, drop stale frames until the front PTS reaches
     the audio clock.  If the deficit exceeds one buffer's worth, the buffer
     empties and re-prime restarts the cycle until the deficit fits.
   - If video is ahead, sleep until the audio clock reaches the front PTS,
     then drop any frames the ~24 ms ALSA clock step overshot past.
3. **Draining** — one frame per `Action()` iteration.  Pacing comes from
   `SubmitFrame`'s single-slot backpressure (which blocks on the next VSync),
   not from a wall-clock timer.

The buffer level is **not regulated**.  Audio packets and decoded video frames
are produced at the same source rate, so the jitter buffer level naturally
tracks the audio queue: low aq ↔ low buf, high aq ↔ high buf.  Any sustained
growth indicates an audio-side issue, not a video one, and is best diagnosed
in the audio path.

`SkipStaleJitterFrames()` runs each iteration to drop frames more than
`HARD_THRESHOLD` behind the audio clock.  Per-iteration draining can only drop
one such frame per loop, which is too slow to recover from a multi-frame stale
burst (e.g. after a network gap), so the catch-up runs at CPU speed.

## Trick mode and freerun

Sync logic is bypassed entirely in three states:

- **Trick mode** (FF/REW/slow):  pacing is enforced by `SubmitTrickFrame()`'s
  per-frame timer, not the audio clock.
- **Freerun** (`freerunFrames > 0`, set by `Clear()`):  the next frame after a
  channel switch or seek is submitted immediately so the user sees something
  on the screen, before prime-sync kicks in.
- **No clock yet** (audio pipeline running but `GetClock()` returns
  `AV_NOPTS_VALUE`):  submit immediately and log a diagnostic.

## Lifecycle summary

| Event | Behavior |
|---|---|
| Plugin start | EMA invalid, no state to preserve |
| Channel switch (`Clear`) | Jitter buffer flushed, `freerunFrames` armed, EMA invalidated |
| Jitter prime | `RunJitterPrimeSync()` aligns front to audio, EMA invalidated, grace armed |
| Soft wait fires | EMA adjusted by `−waitMs × 90`, grace armed |
| Hard transient | Frame dropped, EMA invalidated, grace armed |
| Audio codec change (`NotifyAudioChange`) | `freerunFrames` re-armed for first post-change frame |

## Log format

```
sync d=+12ms avg=+5.4ms buf=15 aq=0 miss=0
```

| Field | Meaning |
|---|---|
| `d`    | Raw instantaneous delta (noisy by design — for diagnosis only) |
| `avg`  | EMA-smoothed delta (drives all corrections) |
| `buf`  | Jitter buffer depth (frames) |
| `aq`   | Audio decode queue depth (packets) |
| `miss` | Drain stalls since the last log: gap > 2× frame period.  Counts both display stalls and our own deliberate soft waits. |

A healthy steady state shows `avg` inside ±10 ms, `buf` stable near the
post-prime equilibrium, `aq` near zero, and `miss = 0` (or 1 immediately
after a soft wait).

## Tunables

All in `src/decoder.cpp`:

| Constant | Default | Purpose |
|---|---|---|
| `DECODER_SYNC_HARD_THRESHOLD` | 200 ms | Hard-drop / hard-wait safety threshold |
| `DECODER_SYNC_TARGET_BAND` | 15 ms | EMA dead band on the ahead side |
| `DECODER_SYNC_MAX_WAIT_MS` | 80 ms | Cap on a single soft wait |
| `DECODER_SYNC_GRACE_MS` | 1000 ms | Suppress corrections while the EMA reseeds |
| `DECODER_SYNC_EMA_SAMPLES` | 250 | EMA divisor → ~5 s @ 50 fps |
| `DECODER_JITTER_BUFFER_MS` | 500 ms | Target jitter buffer depth |

## Source map

| File | Component |
|---|---|
| [src/decoder.cpp](src/decoder.cpp) `SyncAndSubmitFrame()` | Per-frame sync decision |
| [src/decoder.cpp](src/decoder.cpp) `UpdateSmoothedDelta()` | EMA filter |
| [src/decoder.cpp](src/decoder.cpp) `RunJitterPrimeSync()` | One-shot pre-drain alignment |
| [src/decoder.cpp](src/decoder.cpp) `SkipStaleJitterFrames()` | Hard-late catch-up at CPU speed |
| [src/decoder.cpp](src/decoder.cpp) `WaitForAudioCatchUp()` | Replay-mode hard ahead block |
| [src/audio.cpp](src/audio.cpp) `WritePcmToAlsa()` | Updates `playbackPts = endPts − snd_pcm_delay` |
| [src/audio.cpp](src/audio.cpp) `GetClock()` | Returns the most recent `playbackPts` snapshot |
