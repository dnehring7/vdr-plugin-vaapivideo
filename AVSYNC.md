# A/V Sync Drift Correction

## The Problem

The DVB broadcast stream carries audio and video with timestamps (PTS) derived
from a 90 kHz master clock (PCR).  On the playback device, the audio DAC clock
is PLL-locked to the HDMI sink's pixel clock -- a completely independent
oscillator.  The rate difference between these two clocks (typically 5-50 ppm)
causes audio and video to slowly drift apart.  Without correction, lip sync
degrades by ~1-8 ms per second, requiring periodic frame drops.

## Architecture

```
DVB Stream (PCR = master clock)
    |
    +-- Video PTS --> decoder --> filter --> jitter buffer --> display
    |
    +-- Audio PTS --> decoder --> swresample --> ALSA --> DAC (HDMI sink clock)
                                     ^
                                     |
                              PI drift compensation
                              (micro-adjusts playback rate)
```

Audio is the master clock.  `GetClock()` returns the PTS of the sample
currently at the DAC output (`endPts - snd_pcm_delay`).  Video adapts to match.

## Measurement

On every video frame submission (`SyncAndSubmitFrame`), the decoder computes:

```
delta = videoPTS - audioClock - pipelineLatency
```

- `delta = 0`  -- perfect sync
- `delta < 0`  -- video behind audio (audio clock ahead)
- `delta > 0`  -- video ahead of audio

The raw delta has ~+/-20 ms jitter from `snd_pcm_delay` measurement noise.
An EMA low-pass filter (alpha ~0.01, ~2 s time constant at 50 fps) smooths it:

```
smoothedDelta += (delta - smoothedDelta) / 100
```

## PI Controller

The smoothed delta feeds a PI (Proportional-Integral) controller that computes
a drift compensation value:

```
I-term:  driftIntegral += smoothedDelta          (raw accumulation, /1000 on output)
comp  =  -(smoothedDelta + driftIntegral/1000) / 90
```

**P-term** (`-smoothedDelta/90`):  Reacts to the current offset.  If
avg=-30 ms, it pushes comp=+33 to close the gap.  Fast response (~10 s), but
drops to zero when avg=0.

**I-term** (`-driftIntegral/1000/90`):  Slowly accumulates the total error to
learn the steady-state hardware drift rate.  When the P-term has driven avg
to 0, the I-term alone sustains the exact compensation value needed to keep it
there.  Time constant ~20 s.  This prevents the oscillation that a P-only
controller would exhibit.

The comp value is clamped to +/-200 (~2 % rate adjustment).  Typical
steady-state values are 3-5 for normal hardware drift, 40-80 for HDMI setups
with larger PLL skew.

## Correction

The comp value is published to the audio processor via
`SetDriftCompensation()`.  In the audio decode loop, before each
`swr_convert()` call:

```cpp
swr_set_compensation(swrCtx, comp * sampleRate / 10000, sampleRate);
```

This tells FFmpeg's resampler: "over the next second of audio, produce
`comp * 48000 / 10000` extra (or fewer) samples."  For comp=5 (50 ppm drift),
that is 24 extra samples per second -- completely inaudible.

**Why this affects the clock:**  `WritePcmToAlsa` advances `endPts` by the
original frame count (DVB timeline), but the ALSA buffer receives the
compensated sample count.  This intentional mismatch means:

- `endPts` advances at DVB PCR rate (correct).
- `snd_pcm_delay` reflects the actual (compensated) buffer depth.
- `GetClock() = endPts - delay` naturally shifts as the buffer depth diverges
  from what endPts expects.

When the resampler adds samples, the buffer grows slightly deeper per write.
The delay increases faster than endPts, so the clock slows down.  This is
physically identical to what a hardware PCR-locked DAC would do.

## Lifecycle

**Startup:**  The I-term initializes at -40 ms (typical hardware drift).
First channel starts with comp~44 immediately -- near-instant sync.

**Channel switch:**  `smoothedDelta` resets (EMA re-seeds from first
post-correction frame).  The I-term is preserved -- the hardware drift has not
changed.  Convergence takes ~5-10 s instead of ~30 s.

**Hard drops** (delta < -200 ms):  Safety net for massive transients (initial
tune, PTS discontinuities).  Drops frames in bulk until delta recovers.  EMA
re-seeds from the post-correction delta.  I-term preserved.

**Passthrough audio** (AC-3/DTS bitstream):  `swr_set_compensation` cannot
adjust compressed audio.  The PI controller still computes comp, but it has no
effect.  Drift accumulates until delta hits -200 ms, triggering a hard drop.
This is the correct behavior -- there is no better option for passthrough
without modifying the compressed bitstream.

## Convergence Example

```
Time    d       avg      comp   i        State
0s     -40ms   -40.0ms   84   -40.0ms   Initial: P=44 + I=40 = 84
10s    -20ms   -25.0ms   72   -42.5ms   P shrinking, I accumulating
20s     -5ms    -8.0ms   53   -44.5ms   Converging
30s     +0ms    +0.0ms   43   -43.0ms   P=0, I sustains comp=43
60s     -2ms    +0.0ms   43   -43.0ms   Locked indefinitely
120s    +0ms    +0.0ms   43   -43.0ms   No drift, no frame drops
```

## Log Format

```
sync d=-8ms avg=-10.2ms comp=10 i=-43.1ms buf=34
```

| Field | Meaning |
|-------|---------|
| `d`   | Instantaneous delta (raw, jittery) |
| `avg` | EMA-smoothed delta (drives corrections) |
| `comp`| Drift compensation (samples per 10000 output) |
| `i`   | I-term value (learned steady-state drift) |
| `buf` | Jitter buffer depth (video fields queued) |

## Key Source Files

| File | Component |
|------|-----------|
| `src/decoder.cpp` `SyncAndSubmitFrame()` | PI controller, EMA, frame drop logic |
| `src/audio.cpp` `DecodeToPcm()` | `swr_set_compensation()` application |
| `src/audio.cpp` `WritePcmToAlsa()` | PTS tracking with original frame count |
| `src/audio.cpp` `GetClock()` | `endPts - snd_pcm_delay` snapshot |
