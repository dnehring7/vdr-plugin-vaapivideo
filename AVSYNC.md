# A/V Synchronization

## Problem

A DVB stream encodes audio and video against a single 90 kHz program clock
(PCR). On playback the audio DAC runs on its own oscillator — typically
5–50 ppm off the broadcaster, several hundred ppm on poor SAT>IP gear.
Without active correction, lip-sync drifts by milliseconds per minute.

## Architecture

```
DVB live / replay (PCR)   ─┐
                           ├─▶ 90 kHz PTS ─┬─ audio → ALSA ring → DAC ── GetClock() ── master
Mediaplayer (libavformat) ─┘               └─ video
                                                │
        DECODE thread (Action):         decode → VPP filter → handoffQueue   (epoch-stamped)
                                                │
        PRESENT thread (PresentAction): handoffQueue → jitterBuf → due-gate → SyncAndSubmitFrame
                                                │              (decode-ahead reserve = jitterBuf + handoffQueue)
                                                ▼
        pendingFrames (DISPLAY_PRERENDER_SLOTS = 8) → display thread → KMS commit
```

The controller is **input-path-agnostic**: VDR's PES path and the
libavformat-based mediaplayer (see [README.md → Mediaplayer](README.md#mediaplayer))
both deliver packets in VDR's 90 kHz PTS domain. PES arrives in that
domain natively; the mediaplayer rebases each packet from the container's
time base via `av_rescale_q` and subtracts `ptsOrigin90k`, set in
`PopulateStreamInfo` to `max(stream.start_time)` across the tracked
streams so the trailing stream defines t=0 and any pre-sync leading
packets (rebased PTS < 0) are dropped by `ReadPacket` — both streams
begin at rebased PTS 0 together. See `cVaapiMediaSource::PopulateStreamInfo`
and `::ReadPacket` in [src/mediaplayer.cpp](src/mediaplayer.cpp). Downstream
the EMA, soft corridor, hard transients, catch-up, drain and jitter buffer
behave identically for either source.

Three invariants:

1. **Audio is master.** `cAudioProcessor::GetClock()` returns the DAC
   playback PTS, computed as `playbackPts + (now − lastClockUpdateMs)`.
   `playbackPts` is refreshed once per ALSA period (~25 ms); the
   age-extrapolation collapses period quantization to scheduling-jitter
   precision (~1 ms). After `AUDIO_CLOCK_STALE_MS = 1 s` without a write,
   or before any write has fired, `GetClock()` returns `AV_NOPTS_VALUE`
   and the controller falls into freerun. Audio holds two write-path
   commands: **`Clear()`** flushes ALSA and resets the playback clock
   (seek / channel change); **`DropOutput()`** flushes ALSA + decode
   queue but preserves the playback clock — used from Mute / Freeze /
   SetTrickSpeed, where the stream is paused but not abandoned. Freeze
   passes `pauseClock=true`, which additionally *pins* `GetClock()` so it
   cannot extrapolate through ALSA silence and fake-advance the clock
   across the pause (a fake-advanced clock on resume would drop the
   preserved `jitterBuf` head via `SkipStaleJitterFrames`); the pin lifts
   on the next `WritePcmToAlsa()`. A full `Clear()` in those paths would
   null `GetClock()`, force the decoder into freerun, and tick a display
   underrun on resume. The audio
   thread additionally auto-resets the clock on any decoded-PTS jump
   >5 s (channel switch, seek, wrap) so external paths that bypass
   `Clear()` still re-anchor.

2. **Audio is never resampled.** No `swr_set_compensation`, no software
   PLL. Only video adapts. This keeps the system stateless across
   channel switches and avoids the feedback-loop instabilities of
   software audio resampling.

3. **Video is producer-paced to the display rate.** The filter graph
   appends `fps=<displayFps>` whenever the post-deinterlace output rate
   differs from the display rate (rational test
   `outputRateNum ≠ displayHz × outputRateDen`) — for *any* ratio, exact
   or not. The node paces the *decoder* at source rate by buffering its
   output up/down to the display rate, so source-rate consumption tracks
   real time; without it the decoder is paced only by `SubmitFrame`'s
   VSync backpressure (= display rate) and drifts (60→50 consumes source
   at 83%, 24→50 at 208%). On audio-clocked paths the due-gate would
   eventually correct that via catch-up drops / re-presents, but
   **video-only** playback (HDR demo files, etc.) depends entirely on
   `fps`; adding it on audio-clocked paths too is harmless and removes the
   routine source>display catch-up-drop log churn. The filter is
   nearest-neighbor (duplicate for source<display, decimate for
   source>display) — no motion interpolation exists in the VAAPI VPP
   chain. Sources already at the display rate (native 50p, 25i→50 via
   `rate=field`) skip the node. Either way every frame reaches
   `SyncAndSubmitFrame` at one cadence, so a single controller regime fits all.

## Decode / present decouple

Decode and presentation run on **two threads** so a slow VPP step never stalls
the screen. A 4K interlaced → 2160p upscale can spike to ~80 ms — far longer
than the 20 ms frame period — and SW decoders (libdav1d) add their own per-frame
variance. If the thread filtering the next frame also has to submit the current
one on time, that spike surfaces as a dropped frame.

- **Decode thread** (`cVaapiDecoder::Action`): pulls packets, runs VAAPI/SW
  decode + the VPP filter graph, and pushes each finished frame onto
  `handoffQueue`. Never touches the audio clock or the sync controller.
- **Present thread** (`PresentAction`, a nested `cPresenter : cThread` declared
  last so it is destroyed first): splices `handoffQueue` into its private
  `jitterBuf`, runs the due-gated drain, and calls `SyncAndSubmitFrame` at the
  audio-synced cadence — while the decode thread is already filtering the next
  frame.

The two are joined by a bounded **blocking** handoff: `handoffMutex` (a strict
leaf lock), with `handoffCondition` waking the present thread and `handoffNotFull`
waking the decode thread. When `handoffQueue` reaches `DECODER_RESERVE_HARD_CAP`
the decode thread *waits* rather than dropping — the upstream packet queue (and
through it VDR's own flow control) stays authoritative, so backpressure is never
resolved by discarding already-decoded frames.

### Decode-ahead reserve

`jitterBuf` (present side) + `handoffQueue` (handoff) together form the
**decode-ahead reserve**. In steady replay the decoder runs ~2 s ahead, so a
multi-second VPP stall drains the reserve instead of the screen. This is the
deep, low-frequency cushion; the 8-slot display prerender (below) is the
shallow, per-frame one — two buffers at different timescales. The total depth is
published as `publishedDecodedReserveSize` (read via `GetDecodedReserveSize()`)
so the mediaplayer backpressures its demux on the *whole* reserve, not one stage.

### Generation epochs

Because the present thread holds frames the decode thread produced earlier, a
`Clear()` / seek / trick transition must invalidate in-flight frames without a
lock handshake. A single atomic `clearEpoch` is the generation counter:

- `Clear()`, `FlushForSeek()`, `SetTrickSpeed(0)`, and the deferred trick-exit
  (`ResolvePendingTrickExit`) bump `clearEpoch`.
- The decode thread stamps each frame's `producedEpoch` from `clearEpoch` at
  production.
- The present thread snapshots `presentEpoch = clearEpoch` once per iteration and
  discards any frame with `producedEpoch < presentEpoch` (a superseded
  generation) — at the splice and again in a front-purge — so stale frames
  self-discard regardless of the race timing between decode, present, and the
  control thread. Other cross-thread control changes are applied at the top of
  `PresentAction` via atomics (e.g. `jitterFlushRequest`), never by reaching into
  present-thread state from another thread.

Lock order is unchanged — `codecMutex → parserMutex → packetMutex`, with
`handoffMutex` a strict leaf that never nests another lock — so the second thread
adds no new ordering edges. `jitterBuf` and the sync controller stay
present-thread-private (no lock).

## Filter pipeline

```
SW decode: [bwdif|yadif] → [hqdn3d] → format=nv12 → [crop] → hwupload → [denoise_vaapi] → scale_vaapi → [sharpness_vaapi] [→ fps]
HW decode: [deinterlace_vaapi=rate=field] → [denoise_vaapi] → [crop] → scale_vaapi → [sharpness_vaapi] [→ fps]
```

Bracketed nodes are conditional: `[bwdif|yadif]` on interlaced input
(`yadif`/`bob` in trick / still mode), `[hqdn3d]`/`[denoise_vaapi]`/`[sharpness_vaapi]`
on codec + GPU-VPP availability, `[crop]` only while a manual-zoom preset
is active, and `[fps]` per invariant 3. `scale_vaapi` is always present
(it normalizes pixel format + colorimetry even when not resizing).

`fps` is metadata-only in FFmpeg (`AVFILTER_FLAG_METADATA_ONLY`), so it
duplicates `AVFrame` references without touching pixels. Placement at the
chain tail keeps scale/denoise/sharpen at one execution per *input* frame.

## Per-frame sync (`SyncAndSubmitFrame`)

```
rawDelta = videoPTS − GetClock() − pipelineLatency
```

`rawDelta > 0` ⇒ video ahead, `< 0` ⇒ behind. `pipelineLatency` is the
configured operator knob plus a fixed one-frame tail (the dominant
scanout delay = commit + page flip for an empty prerender cache). The
knob is split per output mode:

| Mode                 | `setup.conf` key     | Range       | Default |
| -------------------- | -------------------- | ----------- | ------- |
| PCM (decoded)        | `PcmLatency`         | −200…200 ms | 0       |
| IEC61937 passthrough | `PassthroughLatency` | −200…200 ms | 0       |

Active variant is selected per stream via `cAudioProcessor::IsPassthrough()`.

### EMA smoother

`rawDelta` carries up to ~150 ms field-alternation aliasing on
deinterlaced 50p output; using it directly for soft corrections would
churn. The smoother is an **Exponential Moving Average** — a running
estimate that weights each new sample by `α` and the previous estimate
by `1 − α`:

```
ema = α × new_value + (1 − α) × previous_ema
```

Small `α` (here `1 / EMA_SAMPLES = 1/50`) ignores single-frame spikes
but tracks sustained drift; the time constant is `1 / α` samples (~1 s
@ 50 fps). The smoother runs in two phases:

1. **Warmup.** First `WARMUP_SAMPLES = 50` samples (~1 s @ 50 fps) feed a
   simple mean to seed the EMA. Soft corrections are gated on
   `smoothedDeltaValid`, so no correction fires off a partial mean.
2. **Steady-state EMA.** Integer form of the formula above with a
   residual accumulator that carries the `diff mod N` remainder across
   samples — guarantees exact convergence to the rawDelta mean even
   when `|diff| < N` (the naïve integer step would round to 0).

`ResetSmoothedDelta()` clears warmup, EMA, residual, hard-debounce counters
and catch-up state in one call. Called on channel switch, hard-behind fire,
catch-up exit, and `WaitForAudioCatchUp`.

**Fast-start seed.** A `FlushForSeek()` (same stream, same pipeline — see
[Lifecycle](#lifecycle)) carries the pre-seek converged delta across the
flush as `seekHintDelta90k` and seeds the EMA from it on the first
post-seek frame, skipping the 50-sample warmup. The GPU-vs-audio offset is
a property of the pipeline (decode + VPP + KMS latency vs ALSA hw_ptr), not
the playback position, so the pre-seek steady state is the right seed and
the right catch-up exit target. The hint is captured as the *pre-correction*
`stableDelta90k` (a sleep's predictive EMA bump makes the live value
transient during recovery), clamped into the soft corridor, and ages out
after `DECODER_SYNC_HINT_MAX_AGE_MS` so a stale snapshot can't dominate.
Plain `Clear()` (content boundary) drops the hint — different content can
have different decode latency.

## Correction regimes

Symmetric: every regime has a behind and an ahead path. Hard transients
bypass the cooldown.

### Soft corridor — `|smoothed| > CORRIDOR (50 ms)`, cooldown elapsed

Trigger uses **smoothed** (so a single bad frame can't fire the
correction); correction size uses **rawDelta** (to close the actual
gap, not the lagging average). Let
`correctMs = min(|rawDelta|/90, MAX_CORRECTION_MS = 200)`.

| Direction | Action |
| --------- | ------ |
| ahead     | `SleepMs(correctMs + frameDur)`, submit, then `smoothed −= (elapsed − frameDur) × 90` |
| behind    | Drop `N = max(1, round(correctMs / frameDur))` frames in one burst (one now, `N−1` via `pendingDrops`); reset the EMA |

The `+ frameDur` padding on the ahead sleep is load-bearing: a bare
`SleepMs(correctMs)` only lengthens the iteration by `correctMs − frameDur`
(the missing `frameDur` is absorbed by the next iteration's natural
packet wait), so without padding the smoother sees half the requested
shift and re-fires forever.

The behind path is the small cousin of hard-behind: same `rawDelta`-based
drop count, same `ResetSmoothedDelta()`, same one-per-iteration drain of
`pendingDrops`. The reset removes any open-loop EMA bump; honest
re-measurement during the next warmup (~1 s) seeds the smoother from the
post-correction reality. Earlier versions paced drops across the
cooldown to hide each skip, but in measurement the pacing window
overlapped persistent drift and the controller never converged — a
single short burst lands a real correction and is what makes
post-correction `d ≈ 0` reliably reproducible.

`MAX_CORRECTION_MS = HARD_THRESHOLD = 200` so a single soft event can
fully close the corridor; no sub-corridor residual is left for the
controller to re-fire on.

### Cooldown — `COOLDOWN_MS = 5 s`

Armed by every soft fire, hard-ahead transient, and `WaitForAudioCatchUp`.
Catch-up exit and hard-behind don't arm the cooldown; the EMA reset's
warmup (~1 s) alone gates the next soft event. Soft-behind itself
also resets the EMA *and* arms the cooldown, so the smoother has
absorbed the previous correction plus several cycles of fresh samples
before another soft event can fire.

### Hard transients (raw delta, no cooldown gate)

| Condition                                | Action |
| ---------------------------------------- | ------ |
| `rawDelta < −HARD_THRESHOLD` (−200 ms)   | Drop N frames (N = round(\|rawDelta\|/frameDur)), reset EMA |
| `rawDelta > +HARD_THRESHOLD`, replay     | `WaitForAudioCatchUp()` blocks (≤ 5 s) until audio catches up, then submit; reset EMA, arm cooldown |
| `rawDelta > +HARD_THRESHOLD`, live       | One sleep ≤ `DECODER_SYNC_HARD_AHEAD_MAX_MS = 500 ms`, submit; EMA `−= measured`, arm cooldown |

Both directions are **2-sample debounced** (`hardAheadDebounce`,
`hardBehindDebounce`): a single sample over threshold submits unpaced and
waits for the next sample to confirm. Real PCR discontinuities shift
`pts` for every subsequent frame, so the counter reaches 2 within one
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
vaapivideo/decoder: catch-up complete dropped=143 wall=314ms exit-raw=-38ms follow-up=4 (target=+10ms)
```

When catch-up *cycles* (e.g. a VVC SW decode that can't sustain real
time), those per-pass lines are throttled to one per
`DECODER_CATCHUP_LOG_THROTTLE_MS = 2 s`; the suppressed cycles are folded
into a periodic `catch-up cycling sustained: N cycles …` line and a final
`catch-up cycling settled: …` when the cycling stops.

Entry threshold (−100 ms) and exit threshold (−50 ms) give at least
`CORRIDOR` of hysteresis, well above single-sample jitter. The exit at
`−CORRIDOR` is also the **highest threshold guaranteed reachable**:
catch-up only progresses `rawDelta` when frames are already cached in
`jitterBuf` (drops are fast pops, audio barely advances). Once the
cache drains, each further drop has to wait one VPP cycle for the next
frame, so on marginal-VPP hardware (UHD upscale, ~50 fps == audio rate)
PTS and clock advance equally and `rawDelta` stops climbing. Targeting
`+halfFrame` would then hang catch-up forever, silently consuming every
newly decoded frame. The small permanent negative offset that may
remain after exit is well below the 80 ms lipsync percept threshold.

`SkipStaleJitterFrames()` lifts its "keep ≥ 1" guard while catching up,
since the kept frame would be dropped next iteration anyway. On exit the
EMA is reset and the exiting frame is submitted normally; a small
**follow-up drop burst** (≤ 8 frames, via `pendingDrops`) is armed to push
the head a touch past the clock (or to the seek hint, when one exists) —
but only when `jitterBuf` actually holds enough cached frames to satisfy
it cheaply. On a drained, marginal-VPP pipeline each follow-up drop would
instead wait a full VPP cycle and never gain ground, so it is skipped there.

The **warmup entry** is the post-Clear safety net. After `Clear()` the
EMA is reset and the first 50 samples seed the next mean. If the input
queue still holds pre-Clear-stale frames their `rawDelta` would be
deeply negative, biasing the seed and tripping soft-behind on the very
next frame. The warmup catch-up drains those frames silently before
they reach the EMA accumulator.

## Jitter buffer (unified drain)

The video drain is unified across live and replay and runs on the **present
thread** (see [Decode / present decouple](#decode--present-decouple)): each
iteration splices the decode thread's `handoffQueue` into the private `jitterBuf`
(`std::deque`) and pops when due.

```
splice: handoffQueue → jitterBuf      (drop frames with producedEpoch < presentEpoch)
front-purge: drop jitterBuf heads with producedEpoch < presentEpoch
runaway guard: if jitterBuf > RESERVE_HARD_CAP, drop oldest down to cap
SkipStaleJitterFrames(): bulk-drop heads more than HARD_THRESHOLD behind clock
loop:
  if devicePaused && !trick:            break                  // Freeze: hold, clock pinned
  if trick || freerun || pendingDrops || !ap:  SyncAndSubmitFrame(head)  // clock due-gate bypassed
  clock = GetClock()
  if clock == NOPTS:                                            // audio not yet anchored
      hold ≤ NO_CLOCK_HOLD_MS (escape if jitterBuf near cap), else no-clock freerun submit
  dueIn = headPts − clock − latency
  wake  = PresentWakeThreshold90k()     // frameDur if prerender empty (pre-fill), else halfFrame
  if dueIn > wake:
      if dueIn > FUTURE_MAX:            drop head; continue     // PTS discontinuity
      else:                            break                    // hold (still frame) until due
  else:                                SyncAndSubmitFrame(head)
```

The drain has three guards against startup / re-anchor / pause stalls:

- **`FUTURE_MAX` (`DECODER_DRAIN_FUTURE_MAX_MS = 3 s`).** Drops heads
  sitting more than 3 s ahead of the audio clock as PTS discontinuities
  (post-ATTA anchor mismatch, broadcast PCR break, post-seek backlog).
  Waiting for the gate to clear naturally would look like a multi-second
  startup freeze. Smaller future offsets remain paced normally.

- **Still-frame hold.** A head sitting more than `CORRIDOR` (50 ms) but
  under 3 s ahead of the clock — a post-seek / trick-exit re-anchor where
  the freshly anchored clock must advance to meet the head — is simply
  held: the drain breaks without submitting, so the single freerun frame
  already shown after the `Clear()` stays on screen as a still picture
  until the clock reaches its PTS. Each frame is then released exactly when
  due, so the transition is a brief freeze, not a crawl. A hold outside the
  corridor zeroes `lastDrainMs` so the resume drain isn't counted as a
  starvation miss.

- **No-clock hold (`DECODER_NO_CLOCK_HOLD_MS = 1.5 s`).** While
  `GetClock()` is NOPTS (audio priming after `Clear()` / seek) the drain
  *holds* a non-empty `jitterBuf` rather than freerunning pre-anchor video
  at VSync rate — which would land the head far ahead of the clock the
  moment audio anchors. A
  near-cap escape submits anyway if `jitterBuf` approaches
  `RESERVE_HARD_CAP`, so a fast HW decoder can't overflow waiting for an
  audio anchor that never comes (video-only stream). Covers the
  mux-interleave seek offset — a TS seek can land on a keyframe up to ~1 s
  ahead of the target audio.

- **`RESERVE_HARD_CAP` (`DECODER_RESERVE_HARD_CAP = 150`, ~3 s @ 50 fps).**
  Drop-oldest runaway guard for the case the gates above miss
  (`SkipStaleJitterFrames` only drops heads *behind* the clock, so PTS
  marching ahead with a valid clock could grow `jitterBuf` without
  bound). Drop-oldest preserves the closest-to-due tail. The same cap
  bounds each stage of the decode-ahead reserve: the decode thread also
  drop-oldest-trims `handoffQueue` if the present thread stalls past it
  (normally it backpressures there long before — see the decouple section).
  A trial reduction to 64 to reclaim GPU memory **regressed** replay: the
  shallow reserve keeps the decoder continuously producing instead of
  idling, multiplying `vaDriverMutex` contention with the display's
  per-frame PRIME map. The deep reserve is also the lower-contention
  operating point — keep it.

`Freeze()` (pause) holds the drain directly: while `devicePaused` and not
in trick play the loop breaks without submitting, so the head's PTS can't
drift against the pinned-but-static audio clock (see [Architecture](#architecture)
invariant 1). Resume (`Play()`) lifts the hold and the pin together.

The **pre-fill bypass** submits the head up to one prefillMargin early
when the display prerender queue is empty, keeping `PendingDepth()` at
1–2 instead of 0–1. `prefillMargin = frameDur / 2` (half a frame above
strict-due), so the total prefill window is `halfFrame + prefillMargin
= frameDur` — the decoder never runs more than one frame ahead of
strict-due. Absorbs audio-clock vs VSync phase drift that would
otherwise tick the underrun counter on a healthy stream. Originally
`prefillMargin = frameDur` but that pushed steady-state `d` so far
above `halfFrame` that a short audio glitch under heavy VPP load
drained the queue to 0 before recovery.

`liveMode` selects only the **hard-ahead policy** inside
`SyncAndSubmitFrame` (replay → `WaitForAudioCatchUp`, live → bounded
sleep). The drain itself is identical for both — `SkipStaleJitterFrames`
plus the pre-warmup catch-up handle every trick-exit and pre-roll
backlog.

### Drain bypasses

The due check is bypassed for:

- **Freerun** (`freerunFrames > 0` after `Clear()`, trick exit, audio
  codec change) — gives an instant first picture; the still-frame hold
  then freezes that one unpaced frame until the clock syncs.
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

`SyncAndSubmitFrame` (on the present thread) hands the chosen frame to
`cVaapiDisplay::SubmitFrame`, which pushes onto `pendingFrames` (a
`std::deque`, depth `DISPLAY_PRERENDER_SLOTS = 8`). The display thread
pops one per VSync, maps via VAAPI→PRIME, and commits via DRM atomic.
`SubmitFrame` **blocks** when all slots are full — this is the VSync
backpressure that paces the present thread (and through the handoff, the
decoder) to the display refresh rate.

This prerender is the **shallow, per-frame** cushion — distinct from the
deeper decode-ahead reserve upstream (see [Decode / present
decouple](#decode--present-decouple)). The 8-slot depth (= 160 ms tolerance
@ 50 fps) is sized to absorb a single UHD VPP / memory-bandwidth spike
(observed ~80 ms in replay on 1280×720 → 3840×2160 upscale) plus the
per-frame variance of CPU-side SW decoders (libdav1d 1080p50 spikes
30–40 ms on complex frames) without draining the cache and forcing a
re-present. FHD HW paths never fill past 1–2 slots, so the extra depth is
a no-op there; the lipsync pipeline is delayed in lockstep with audio,
not just video, so the extra slots do not shift `rawDelta`.

### Queue underrun detection

The display thread tracks `lastFrameCommitMs` (atomic, updated on every
fresh commit). On a VSync where no fresh frame is available it
re-presents the previous buffer to keep flip cadence + OSD updates
alive. Each re-present streak is measured in **wall-clock** instead of
VSync count, so the printed duration is true even when the consumer
loop is preempted or page-flip events arrive late.

State carried across iterations:

- `gapStartMs` — wall-clock baseline anchored on the first re-present
  of the current streak; `0` means "anchor on next re-present".
- `peakGapMs` — wall-clock peak of the current streak; reset on every
  fresh commit.

Per re-present iteration (only when `lastFrameCommitMs != 0` and outside
trick / sync-sleep / warmup grace):

1. Anchor `gapStartMs = nowMs` if it's `0`.
2. `currentGapMs = nowMs − gapStartMs`.
3. If `currentGapMs < IDLE_THRESHOLD_MS (10 s)`:
   - `peakGapMs = max(peakGapMs, currentGapMs)`.
   - If `currentGapMs ≥ thresholdMs` and the log cooldown has elapsed,
     emit `queue empty Nms; total=M`.
4. Else (gap ≥ 10 s): treat as paused / stopped, clear `peakGapMs`,
   leave `gapStartMs` put so a long pause doesn't re-anchor every
   iteration.

`thresholdMs = (DISPLAY_PRERENDER_SLOTS + 2) × vsyncMs` — at 50 Hz that's
`10 × 20 ms = 200 ms`. The `+2` margin gives one VSync of natural absorption
by the prerender depth plus one more so a single hiccup doesn't trip.

On the next fresh commit:
- If `peakGapMs ≥ thresholdMs` the recovery line `queue refilled after
  Nms; total=M` reports the actual peak.
- `gapStartMs` and `peakGapMs` reset; the streak ends.

Onset is rate-limited to once per `UNDERRUN_LOG_COOLDOWN_MS = 2 s`.
`isClearing`, `inTrick`, `inSyncSleep`, and `inPause` (device frozen) each
force `gapStartMs = 0` so a deliberate hard-ahead sleep (up to 500 ms),
trick hold (60–2000 ms), or pause does not surface its own duration as a
fake underrun.

### Warmup grace

`WARMUP_GRACE_MS = 3 s` suppresses the underrun gate for the first 3 s
after the decoder resumes from idle. Armed on a fresh commit when:

- `lastFrameCommitMs == 0` (post-`Clear()` reset), OR
- `nowMs − lastFrameCommitMs > ACTIVE_WINDOW_MS (500 ms)` (idle resume
  where `BeginStreamSwitch` wasn't invoked — track switch, post-trick
  re-anchor).

Without the grace, every Clear would log a spurious underrun while the
filter graph rebuilds and the audio clock anchors. `ACTIVE_WINDOW_MS`
exists only to arm this grace — it does **not** gate the underrun log
itself (that gate is `IDLE_THRESHOLD_MS`).

## Sync bypass

The sync gate is bypassed (frame submitted unpaced) in:

- Trick mode (`SubmitTrickFrame()` paces via its own timer; audio is muted).
- Freerun window after `Clear()`, trick exit, or `NotifyAudioChange()`.
- Radio mode / NOPTS frame (no audio processor or no PTS to align on).
- Audio not yet running (`GetClock()` is NOPTS until the first
  `WritePcmToAlsa()` fires).

## Lifecycle

| Event                          | EMA                | Cooldown   | Jitter buffer                  |
| ------------------------------ | ------------------ | ---------- | ------------------------------ |
| Plugin start                   | invalid            | —          | empty                          |
| Channel switch (`Clear()`)     | reset              | unchanged  | flushed; freerun armed         |
| Catch-up enter                 | (drops silent)     | unchanged  | drained silently to alignment  |
| Catch-up exit                  | reset              | unchanged  | one frame submitted normally   |
| Soft drop                      | reset              | armed      | N frames dropped (one now, N−1 burst via `pendingDrops`, one per drain iteration) |
| Soft sleep                     | `−= measured`      | armed      | unchanged                      |
| Hard-behind                    | reset              | unchanged  | N frames dropped               |
| Hard-ahead (replay)            | reset              | armed      | unchanged                      |
| Hard-ahead (live)              | `−= measured`      | armed      | unchanged                      |
| Trick entry (FF/REW/slow)      | reset              | unchanged  | reserve purged (generation bump); paced by `SubmitTrickFrame`, no freerun |
| Trick exit → normal (`Play`)   | reset              | unchanged  | reserve purged (generation bump); freerun armed |
| Pause / resume (`Freeze`/`Play`) | unchanged        | unchanged  | held (drain stops); audio clock pinned, no drops |
| Audio codec / track change     | unchanged          | unchanged  | preserved; freerun armed       |
| Mediaplayer seek               | reset              | unchanged  | flushed; freerun armed; filter graph **preserved** |
| Mediaplayer playlist advance   | reset (on reopen)  | unchanged  | flushed; freerun armed; filter graph rebuilt        |

Audio codec / track change preserves the buffer across the switch — the
catch-up path will silently realign against the new clock once it
arrives, so dropping ~1 s of still-valid video buys nothing.

Mediaplayer seek calls `cVaapiDevice::FlushForSeek()`, which fans out to
`decoder->FlushForSeek()` + `audioProcessor->Clear()`. Same drain semantics
as a channel switch (packet queues drained, codec buffers reset, ALSA
drained, audio clock re-anchors on next frame) **except the filter chain
stays alive** — `FlushForSeek` is `Clear` minus `filterChain.Reset()`,
since seek does not change stream parameters and the VAAPI VPP rebuild
would cost ~100 ms per seek for no benefit. Entry open/close uses the
heavier `ClearForMediaPlayer()` which does rebuild the filter (codec
params may change at the next entry). Playlist advance closes the current
`cVaapiMediaSource` and opens the next, which may reopen the codec when
codecId / extradata differ — `OpenCodecWithInfo()` then performs a full
teardown the same way a same-codec bit-depth change does on the PES path.

## Diagnostic log

```
sync d=+15.2ms avg=+15.1ms lat=20ms buf=40 aq=0 miss=0 drop=0 skip=0
```

| Field  | Meaning |
| ------ | ------- |
| `d`    | Interval mean of `rawDelta` since the last log; comparable to `avg` |
| `avg`  | EMA-smoothed delta; drives every soft-correction decision |
| `lat`  | Active `SyncLatency90k` (1-frame tail + active operator knob) |
| `buf`  | `jitterBuf` depth in frames at log emission |
| `aq`   | Audio packet queue depth |
| `miss` | Drain gaps > 2 × output frame period since last log (VSync backpressure, deliberate sync sleeps) |
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
entered (spike|warmup|sustained)`, `catch-up complete`, `head too far in
future … dropping`) for "why did this fire?" without waiting for the next
periodic line.

### Steady-state offset

The EMA does not settle at zero. The baseline depends on whether the
decoder is gate-bound or throughput-bound:

- **Gate-bound** (live TV, replay cold start, anything with `buf > 0`):
  `d` settles inside the prefill envelope `[halfFrame, halfFrame +
  prefillMargin]` — at 50 fps that's `[+10 ms, +20 ms]`, with observed
  values centering around `+15…+18 ms`. The exact position depends on
  how often the display drains the prerender queue to zero (which
  enables the prefill bypass) vs. holding at depth 1+ (which gates at
  strict `halfFrame`). On marginal-VPP hardware where the queue
  oscillates between 0 and 1, `d` lands near the middle of the
  envelope; on FHD where the queue stays at 1–2, `d` sits closer to
  `halfFrame`.
- **Throughput-bound** (replay post-Clear, `buf ≈ 0`): `d ≈ −frameDur`
  (~−20 ms @ 50 fps). Each new frame is submitted as soon as decoded;
  no buffered lead, and the audio clock has already advanced past the
  latency target by the time `SyncAndSubmit` runs. The 1-frame
  pipeline-latency tail keeps this baseline well inside `CORRIDOR`,
  giving ~30 ms of head room before a typical 10–15 ms/s
  pipeline/crystal drift can trip soft-behind.

Both baselines sit inside `±CORRIDOR` and are **not** drift — the
controller leaves them alone, since correcting a non-drifting bias
would only introduce visible jank. To re-center either regime, tune
`PcmLatency` or `PassthroughLatency` (negative values pull video
earlier vs audio).

## Constants

File-scope constants live in [src/decoder.cpp](src/decoder.cpp),
[src/decoder.h](src/decoder.h), [src/audio.cpp](src/audio.cpp), and
[src/config.h](src/config.h); each carries a `///<` comment with purpose and
unit. Constants marked *(local)* live inside the function that uses them.

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
| `DECODER_RESERVE_HARD_CAP`          | 150   | Per-stage cap on the decode-ahead reserve (handoffQueue + jitterBuf, ~3 s @ 50 fps each); decode-side backpressure / present-side drop-oldest runaway guard |
| `DECODER_TRICK_QUEUE_DEPTH`         | 1     | Handoff reserve depth while trick play is active (Poll() throttles the producer; overflow drops incoming) |
| `DECODER_TRICK_HOLD_MS`             | 20    | Base per-frame hold for slow trick (~one field period @ 50 Hz); fast trick scales it by the speed multiplier |
| `DECODER_SYNC_CORRIDOR_90K`         | 4500  | Soft corridor half-width (= 50 ms × PTS_TICKS_PER_MS); below lipsync percept threshold |
| `DECODER_SYNC_HARD_THRESHOLD_90K`   | 18000 | Hard-transient threshold (= 200 ms × PTS_TICKS_PER_MS); 2× = catch-up spike entry |
| `DECODER_SYNC_MAX_CORRECTION_MS`    | 200   | Soft-event cap (matches HARD_THRESHOLD) so one event fully closes corridor |
| `DECODER_SYNC_HARD_AHEAD_MAX_MS`    | 500   | Live hard-ahead sleep cap (ms) |
| `DECODER_DRAIN_FUTURE_MAX_MS`       | 3000  | Future-head discontinuity guard: drop heads >3 s ahead; smaller offsets hold (still frame) until due |
| `DECODER_NO_CLOCK_HOLD_MS`          | 1500  | Walltime the present drain holds a non-empty jitterBuf while `GetClock()` is NOPTS before falling back to no-clock freerun (covers the mux-interleave seek offset) |
| `DECODER_SYNC_COOLDOWN_MS`          | 5000  | Min interval between soft corrections (= 5 EMA time constants) |
| `DECODER_SYNC_EMA_SAMPLES`          | 50    | EMA divisor (~1 s @ 50 fps); residual accumulator → exact convergence |
| `DECODER_SYNC_WARMUP_SAMPLES`       | 50    | Samples averaged before EMA seed (~1 s @ 50 fps) |
| `DECODER_SYNC_FREERUN_FRAMES`       | 1     | Unpaced frames after sync-disrupting events |
| `DECODER_SYNC_LOG_INTERVAL_MS`      | 2000  | Periodic sync diagnostic interval (ms) |
| `DISPLAY_PRERENDER_SLOTS`             | 8     | Present→display prerender queue depth (= 160 ms tolerance @ 50 fps); absorbs a UHD VPP / memory-bandwidth spike plus SW-decoder per-frame variance |
| `ACTIVE_WINDOW_MS` *(local)*          | 500   | Min idle gap on a fresh commit that arms the warmup grace (does not gate the underrun log itself) |
| `IDLE_THRESHOLD_MS` *(local)*         | 10000 | Wall-clock streak length beyond which a re-present gap is treated as paused / stopped (peak cleared) |
| `UNDERRUN_LOG_COOLDOWN_MS` *(local)*  | 2000  | Min interval between underrun-onset dsyslog lines |
| `WARMUP_GRACE_MS` *(local)*           | 3000  | Post-idle grace suppressing underrun logs while pipeline anchors |
| `UNDERRUN_THRESHOLD_VSYNCS` *(local)* | 10    | `(PRERENDER_SLOTS + 2)`; `thresholdMs = THRESHOLD_VSYNCS × vsyncMs` is the wall-clock gap that trips a log |
| `PAGE_FLIP_STUCK_MS` *(local)*        | 200   | Stuck-flip watchdog: force-clear `isFlipPending` if the kernel swallows the page-flip event |
