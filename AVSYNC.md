# A/V Synchronization

## Problem

A DVB stream encodes audio and video against one 90 kHz program clock (PCR).
On playback the audio DAC runs on its own oscillator — typically 5–50 ppm off
the broadcaster, several hundred ppm on poor SAT>IP gear. Without active
correction, lip-sync drifts by milliseconds per minute. The plugin corrects
**video to the audio clock**: audio plays untouched, video is paced, dropped,
or delayed to track it.

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

The controller is **input-path-agnostic**. Both input paths deliver packets in
VDR's 90 kHz PTS domain, so everything downstream — EMA, corridor, hard
transients, catch-up, drain, jitter buffer — behaves identically for either
source:

- **PES** (live TV / dvbplayer replay) is already in that domain.
- **Mediaplayer** (libavformat) rebases each packet in `Rebase90k`: rescale the
  container timestamp to 90 kHz, then subtract `ptsOrigin90k`. `ptsOrigin90k` is
  `max(start_time)` across the tracked streams (`PopulateStreamInfo`), so the
  *trailing* stream defines t=0 and any leading pre-sync packets (rebased PTS < 0)
  are dropped in `ReadPacket` — both streams begin at rebased PTS 0 together.
  Subtitle packets never seed `ptsOrigin90k`, so a stray early cue can't shift the
  A/V timeline. A seek additionally arms `discardAudioBefore90k` so audio anchors at the requested
  position, not at the earlier keyframe libavformat lands on. See
  `cVaapiMediaSource` in [src/mediaplayer.cpp](src/mediaplayer.cpp).

Three invariants:

1. **Audio is master.** `cAudioProcessor::GetClock()` returns the PTS at the DAC
   output: `playbackPts + (now − lastClockUpdateMs) × 90`. `playbackPts` is
   republished on every ALSA write as `endPts − snd_pcm_delay()` (i.e. per
   decoded packet, well faster than the ~25 ms ALSA period); the wall-clock
   age-extrapolation fills the gaps between writes, so reads stay smooth to ~1 ms.
   A seqlock makes the read lock-free. `GetClock()` returns `AV_NOPTS_VALUE`
   before the first write or once a write is older than
   `AUDIO_CLOCK_STALE_MS = 1 s`, and the controller falls into freerun.

   Two write-path commands manage the clock:
   - **`Clear()`** (seek / channel change): `snd_pcm_drop`+`prepare`, drain the
     packet queue, and `ResetPlaybackClock()` — `GetClock()` goes NOPTS and the
     decoder freeruns until audio re-anchors.
   - **`DropOutput()`** (Mute / Freeze / SetTrickSpeed): `snd_pcm_drop`+`prepare`
     and drain the packet queue, but **keep** `playbackPts` — the clock stays
     valid, the decoder stays paced, the display queue does not underrun.
     `pauseClock=true` (Freeze) additionally **pins** `GetClock()` to the static
     `playbackPts` so it can't extrapolate through ALSA silence and fake-advance
     across the pause (a fake-advanced clock on resume would drop the preserved
     `jitterBuf` head via `SkipStaleJitterFrames`). The pin clears on the next
     write, `Clear()`, or `ResetPlaybackClock()`.

   The audio thread also auto-resets the clock on any decoded-PTS jump > 5 s
   (channel switch, seek, wrap), so paths that bypass `Clear()` still re-anchor.
   Volume 0 writes digital silence (zeroed PCM / zeroed IEC61937 burst) rather
   than stopping ALSA, so the master clock keeps advancing while muted.

   A mid-stream PCM channel-layout change re-anchors the clock the same way. When
   the decoded frame's layout implies a different output channel count than the open
   device (a broadcast switching stereo↔5.1, or the operator flipping **PCM
   Channels**), `DecodeToPcm()` flags it and the audio thread calls
   `ReconfigurePcmOutput()`, which reopens ALSA at the new count and
   `ResetPlaybackClock()`s. `GetClock()` goes NOPTS briefly (the triggering frame is
   dropped, and the next packet is held until ALSA reopens and swr rebuilds) and
   re-anchors on the next write; the decoder's no-clock hold (1.5 s) covers the gap.
   The reopen is one-shot per layout change — passthrough is never affected (it is a
   fixed 2-channel IEC61937 carrier).

2. **Audio is never *adaptively* resampled.** No `swr_set_compensation`, no software PLL.
   The PCM path may do fixed format/channel/rate conversion to the negotiated ALSA format
   (e.g. 44.1 kHz radio to a 48 kHz device), but the rate ratio is constant — only video
   adapts over time. This keeps the system stateless across channel switches and avoids the
   feedback-loop instabilities of a software audio PLL.

3. **Video is producer-paced to the display rate.** When post-deinterlace output
   rate differs from the display rate (rational test
   `outputRateNum ≠ displayHz × outputRateDen`, for any ratio), the filter graph
   appends `fps=<displayHz>`. The node buffers up/down so the *decoder* consumes
   source at real time; without it the decoder is paced only by `SubmitFrame`'s
   VSync backpressure (= display rate) and drifts (60→50 consumes source at 83 %,
   24→50 at 208 %). Audio-clocked paths would eventually correct that drift via
   catch-up drops / re-presents, but **video-only** playback (HDR demo files) has
   no clock and depends entirely on `fps`; adding it everywhere is harmless and
   removes routine source>display catch-up-drop churn. The filter is
   nearest-neighbor (duplicate or decimate — no motion interpolation exists in
   VAAPI VPP). Sources already at the display rate (native 50p, 25i→50 via
   `rate=field`) skip it, as do trick play and still-picture mode (which pace
   frames themselves). Either way every frame reaches `SyncAndSubmitFrame` at one
   cadence, so a single controller regime fits all.

## Decode / present decouple

Decode and presentation run on **two threads** so a slow VPP step never stalls
the screen. A 4K interlaced → 2160p upscale can spike to ~80 ms — far past the
20 ms frame period — and SW decoders (libdav1d) add their own per-frame variance.
If the thread filtering the next frame also had to submit the current one on
time, that spike would surface as a dropped frame.

- **Decode thread** (`cVaapiDecoder::Action`): pulls packets, runs VAAPI/SW
  decode + the VPP filter graph, and pushes each finished frame onto
  `handoffQueue`. Never touches the audio clock or the sync controller.
- **Present thread** (`PresentAction`, a nested `cPresenter : cThread` declared
  last so it is destroyed first): splices `handoffQueue` into its private
  `jitterBuf`, runs the due-gated drain, and calls `SyncAndSubmitFrame` at the
  audio-synced cadence — while the decode thread is already filtering ahead.

They are joined by a bounded **blocking** handoff: `handoffMutex` (a strict leaf
lock), with `handoffCondition` waking the present thread and `handoffNotFull`
waking the decode thread. When the decoded reserve reaches `DECODER_RESERVE_HARD_CAP`
— the published total `jitterBuf + handoffQueue`, or `handoffQueue` alone — the
decode thread *waits* rather than dropping, so the upstream packet queue (and
through it VDR's flow control) stays authoritative — backpressure is never resolved
by discarding already-decoded frames. (A drop-oldest exists only as an
`[[unlikely]]` memory-safety net for a wedged presenter.)

### Decode-ahead reserve

`jitterBuf` (present side) + `handoffQueue` (handoff) together form the
**decode-ahead reserve**, bounded to `DECODER_RESERVE_HARD_CAP` (~1.3 s @ 50 fps).
In steady replay it sits near that cap, so a VPP stall up to ~1.3 s drains the
reserve instead of the screen. This is the deep, low-frequency cushion; the 8-slot
display prerender (below) is the shallow, per-frame one — two buffers at different
timescales. The total depth is published as `publishedDecodedReserveSize` (read via
`GetDecodedReserveSize()`) so both the decode thread's own backpressure and the
mediaplayer's demux throttle gate on the *whole* reserve, not one stage.

### Generation epochs

Because the present thread holds frames the decode thread produced earlier, a
`Clear()` / seek / trick transition must invalidate in-flight frames without a
lock handshake. A single atomic `clearEpoch` is the generation counter:

- `Clear()`, `FlushForSeek()`, `SetTrickSpeed(0)`, and the deferred trick-exit
  (`ResolvePendingTrickExit`) bump `clearEpoch`.
- The decode thread stamps each frame's `producedEpoch` from `clearEpoch` while
  holding `codecMutex` for the producing decode, so the stamp is correct
  per-frame regardless of when the handoff happens.
- The present thread snapshots `presentEpoch = clearEpoch` once per iteration and
  drops any frame with `producedEpoch < presentEpoch` — at the splice and again
  in a front-purge — so superseded frames self-discard regardless of the race
  timing between decode, present, and the control thread. Other cross-thread
  control changes are applied at the top of `PresentAction` via atomics
  (e.g. `jitterFlushRequest`), never by reaching into present-thread state.

Lock order is `codecMutex → parserMutex → packetMutex`, with `handoffMutex` a
strict leaf that nests no other lock — so the second thread adds no ordering
edges. `jitterBuf` and the sync controller stay present-thread-private (no lock).

## Filter pipeline

Two filter domains, selected by `useSwPost` (true when a `sw-*`
deinterlace/denoise/scale/sharpen preset is active and the stream is not HDR
passthrough / UHD / simple-deint):

```
GPU VPP domain (default / "auto" presets):
  SW decode: [bwdif|yadif] → [hqdn3d] → format=nv12|p010le → [crop] → hwupload → [denoise_vaapi] → scale_vaapi → [sharpness_vaapi] [→ fps]
  HW decode: [deinterlace_vaapi=rate=field] → [denoise_vaapi] → [crop] → scale_vaapi → [sharpness_vaapi] [→ fps]

Hybrid SW/HW domain (a sw-* preset is active) — HW decode adds one hwdownload; all paths end with one hwupload:
  [hwdownload (HW decode only)] → [bwdif|w3fdif] → [hqdn3d] → [crop → swscale] → [unsharp] → hwupload → [denoise_vaapi] → [crop → scale_vaapi] → [sharpness_vaapi] [→ fps]
```

Bracketed nodes are conditional. In the GPU VPP domain the chain forks again on
decode path (`isSoftwareDecode`): a SW-decoded frame is uploaded mid-chain
(`format=nv12|p010le`, `p010le` under HDR, then `hwupload`), while a HW-decoded
frame deinterlaces and scales natively. `[bwdif|yadif]` runs on interlaced input
(`yadif`/`bob` in trick / still mode); `[hqdn3d]` is an MPEG-2-only SW-denoise
fallback used when the GPU lacks `denoise_vaapi`;
`[denoise_vaapi]`/`[sharpness_vaapi]` depend on codec + GPU-VPP availability;
`[crop]` is active only while a manual-zoom preset is; and `[fps]` per invariant
3. In the GPU VPP domain `scale_vaapi` is always present — it normalizes pixel
format + colorimetry even when not resizing (the hybrid domain may run `swscale`
instead when a `sw-*` scale/sharpen pulls scaling into the SW segment).

`fps` only duplicates or drops `AVFrame` references; it never reprocesses pixels.
Placing it at the chain tail keeps scale/denoise/sharpen at one execution per
*input* frame.

## Per-frame sync (`SyncAndSubmitFrame`)

```
rawDelta = videoPTS − GetClock() − pipelineLatency
```

`rawDelta > 0` ⇒ video ahead, `< 0` ⇒ behind. `pipelineLatency` is a configured
operator knob plus a fixed one-frame tail (the dominant scanout delay = commit +
page flip for an empty prerender cache). The knob is split per output mode and
selected per stream via `cAudioProcessor::IsPassthrough()`:

| Mode                 | `setup.conf` key     | Range       | Default |
| -------------------- | -------------------- | ----------- | ------- |
| PCM (decoded)        | `PcmLatency`         | −200…200 ms | 0       |
| IEC61937 passthrough | `PassthroughLatency` | −200…200 ms | 0       |

### EMA smoother

`rawDelta` carries up to ~150 ms field-alternation aliasing on deinterlaced 50p
output; using it directly for soft corrections would churn. The smoother is an
**Exponential Moving Average** — a running estimate weighting each new sample by
`α` and the prior estimate by `1 − α`:

```
ema = α × new_value + (1 − α) × previous_ema
```

Small `α` (`1 / EMA_SAMPLES = 1/50`) ignores single-frame spikes but tracks
sustained drift; the time constant is `1 / α` samples (~1 s @ 50 fps). Two phases:

1. **Warmup.** The first `WARMUP_SAMPLES = 50` samples (~1 s) feed a simple mean
   that seeds the EMA. Soft corrections gate on `smoothedDeltaValid`, so none
   fires off a partial mean.
2. **Steady-state EMA.** Integer form of the formula above with a residual
   accumulator carrying the `diff mod N` remainder across samples — guaranteeing
   exact convergence to the rawDelta mean even when `|diff| < N` (a naïve integer
   step would round to 0).

`ResetSmoothedDelta()` clears warmup, EMA, residual, hard-debounce counters, and
catch-up state in one call. Called on channel switch, hard-behind, catch-up exit,
and `WaitForAudioCatchUp`.

**Fast-start seed.** A `FlushForSeek()` (same stream, same pipeline) carries the
pre-seek converged delta across the flush as `seekHintDelta90k` and seeds the EMA
from it on the first post-seek frame, skipping the 50-sample warmup. The
GPU-vs-audio offset is a property of the pipeline (decode + VPP + KMS latency vs
ALSA hw_ptr), not the playback position, so the pre-seek steady state is the right
seed and the right catch-up exit target. The hint is captured as the
*pre-correction* `stableDelta90k` (a sleep's predictive EMA bump makes the live
value transient during recovery), clamped into the soft corridor, and ages out
after `DECODER_SYNC_HINT_MAX_AGE_MS`. Plain `Clear()` (content boundary) drops the
hint — different content can have different decode latency.

## Correction regimes

Symmetric: every regime has a behind and an ahead path. The trigger uses
**smoothed** delta (so a single bad frame can't fire); the correction size uses
**rawDelta** (to close the actual gap, not the lagging average). Hard transients
bypass the cooldown.

### Soft corridor — `|smoothed| > CORRIDOR (50 ms)`, cooldown elapsed

`correctMs = min(|rawDelta| / 90, DECODER_SYNC_CORRECTION_MAX_MS = 200)`.

| Direction | Action |
| --------- | ------ |
| ahead     | `SleepMs(correctMs + frameDur)`, submit, then `smoothed −= (elapsed − frameDur) × 90` |
| behind    | Drop `N = max(1, round(correctMs / frameDur))` frames in one burst (one now, `N−1` via `pendingDrops`); reset the EMA |

The `+ frameDur` padding on the ahead sleep is load-bearing: a bare
`SleepMs(correctMs)` only lengthens the iteration by `correctMs − frameDur` (the
missing `frameDur` is absorbed by the next iteration's natural packet wait), so
without padding the smoother sees half the requested shift and re-fires forever.

The behind path resets the EMA so the next warmup (~1 s) re-measures from the
post-correction reality, removing any open-loop bump. One short burst lands a real
correction; that is what makes post-correction `d ≈ 0` reproducible.

`DECODER_SYNC_CORRECTION_MAX_MS = HARD_THRESHOLD = 200` (it is *derived* from
`HARD_THRESHOLD`), so a single soft event can fully close the corridor with no
sub-corridor residual left to re-fire on.

### Cooldown — `COOLDOWN_MS = 5 s`

Armed by every soft fire, hard-ahead transient, and `WaitForAudioCatchUp`.
Catch-up exit and hard-behind don't arm it — the EMA reset's warmup (~1 s) alone
gates the next soft event. Soft-behind resets the EMA *and* arms the cooldown, so
the smoother absorbs the previous correction plus several cycles of fresh samples
before another soft event can fire.

### Hard transients (raw delta, no cooldown gate)

| Condition                              | Action |
| -------------------------------------- | ------ |
| `rawDelta < −HARD_THRESHOLD` (−200 ms) | Drop `N = round(\|rawDelta\|/frameDur)` frames, reset EMA |
| `rawDelta > +HARD_THRESHOLD`, replay   | `WaitForAudioCatchUp()` blocks (≤ 5 s) until audio reaches the head, then submit; reset EMA, arm cooldown |
| `rawDelta > +HARD_THRESHOLD`, live     | One sleep ≤ `HARD_AHEAD_MAX_MS = 500 ms`, submit; `EMA −= measured`, arm cooldown |

Both directions are **2-sample debounced** (`hardAheadDebounce`,
`hardBehindDebounce`): a single over-threshold sample submits unpaced and waits
for the next to confirm. A real PCR discontinuity shifts `pts` for every
subsequent frame, so the counter reaches 2 within one frame period and the
correction still fires within ~20 ms. Isolated outliers (`snd_pcm_delay`
quantization, scheduler hiccups, the `GetClock()` load-pair race) clear on the
next sample and never trigger a 500 ms freeze.

The live hard-ahead path exists because the soft corridor caps at ~40 ms/s and a
marginal transponder can drift faster. A single ≤ 500 ms glitch back into the
corridor beats an indefinite slow chase, and the cap keeps the upstream packet
queue from overflowing during the sleep. `liveMode` selects only this policy
(replay blocks, live sleeps); the drain is otherwise identical for both.

### Catch-up — silent bulk drop

Three entry conditions, one exit (`rawDelta > −CORRIDOR`):

| Entry      | Condition                                                      | Triggers |
| ---------- | ------------------------------------------------------------- | -------- |
| spike      | `rawDelta < −2 × HARD_THRESHOLD` (−400 ms)                    | Catastrophic backlog (cold start, post-seek, multi-second decoder stall) |
| warmup     | `!smoothedDeltaValid && rawDelta < −2 × CORRIDOR` (−100 ms)   | Stale pre-roll backlog about to poison the EMA seed |
| sustained  | `smoothedDeltaValid && smoothedDelta < −2 × CORRIDOR` (−100 ms) | Replay queue lag soft-behind can't clear within its cooldown |

While `catchingUp`, every incoming frame is dropped silently — no per-event log,
no EMA churn, no cooldown arm — until `rawDelta` rises above `−CORRIDOR`. Two log
lines bracket the pass:

```
vaapivideo/decoder: catch-up entered (spike) raw=-2738ms
vaapivideo/decoder: catch-up entered (sustained) avg=-118ms raw=-2738ms
vaapivideo/decoder: catch-up complete dropped=143 wall=314ms exit-raw=-38ms follow-up=4 (target=+10ms)
```

The `sustained` entry additionally carries the `avg=<smoothedDelta>ms` that
triggered it; `spike` / `warmup` print `raw=` only.

When catch-up *cycles* (e.g. a VVC SW decode that can't sustain real time), those
lines are throttled to one per `DECODER_SYNC_CATCHUP_LOG_INTERVAL_MS = 2 s`; suppressed
cycles fold into a periodic `catch-up cycling sustained: …` line every
`DECODER_SYNC_CATCHUP_SUMMARY_INTERVAL_MS = 10 s`, with a final `catch-up cycling
settled: …` when it stops.

The entry (−100 ms) and exit (−50 ms) thresholds give `CORRIDOR` of hysteresis.
The exit at `−CORRIDOR` is also the **highest threshold guaranteed reachable**:
catch-up only advances `rawDelta` while frames are cached in `jitterBuf` (drops
are fast pops, audio barely moves). Once the cache drains, each further drop waits
one VPP cycle for the next frame, so on marginal-VPP hardware (UHD upscale at
~50 fps == audio rate) PTS and clock advance equally and `rawDelta` stops
climbing — targeting `+halfFrame` would hang catch-up forever. The small residual
negative offset that may remain is well below the 80 ms lipsync percept threshold.

On exit the EMA is reset, the exiting frame is submitted normally, and a small
**follow-up drop burst** (≤ 8 frames via `pendingDrops`) nudges the head a touch
past the clock (or to the seek hint) — but only when `jitterBuf` holds enough
cached frames to satisfy it cheaply. On a drained marginal-VPP pipeline each
follow-up drop would wait a full VPP cycle and never gain ground, so it is skipped
there. `SkipStaleJitterFrames()` also lifts its "keep ≥ 1" guard while catching
up, since the kept frame would be dropped next iteration anyway.

The **warmup entry** is the post-`Clear()` safety net: if the input queue still
holds pre-Clear-stale frames, their deeply negative `rawDelta` would bias the EMA
seed and trip soft-behind on the next frame. The warmup catch-up drains them
silently before they reach the accumulator.

## Jitter buffer (unified drain)

The video drain is unified across live and replay and runs on the **present
thread**: each iteration splices `handoffQueue` into the private `jitterBuf`
(`std::deque`) and pops when due.

```
splice: handoffQueue → jitterBuf      (drop frames with producedEpoch < presentEpoch)
front-purge: drop jitterBuf heads with producedEpoch < presentEpoch
runaway guard: if jitterBuf > RESERVE_HARD_CAP, drop oldest down to cap
SkipStaleJitterFrames(): bulk-drop heads more than HARD_THRESHOLD behind clock
loop:
  if devicePaused && !trick:                    break                     // Freeze: hold, clock pinned
  if trick || freerun || pendingDrops || !ap:   SyncAndSubmitFrame(head)  // due-gate bypassed
  clock = GetClock()
  if clock == NOPTS:                                                       // audio not yet anchored
      hold ≤ NO_CLOCK_HOLD_MS (escape if jitterBuf near cap), else no-clock freerun submit
  dueIn = headPts − clock − latency
  wake  = PresentWakeThreshold90k()             // frameDur if prerender empty (pre-fill), else halfFrame
  if dueIn > wake:
      if dueIn > FUTURE_MAX:   drop head; continue                        // PTS discontinuity
      else:                    break                                       // hold (still frame) until due
  else:                        SyncAndSubmitFrame(head)
```

Guards against startup / re-anchor / pause stalls:

- **`FUTURE_MAX` (`DECODER_DRAIN_FUTURE_MAX_MS = 3 s`).** Drops heads sitting
  more than 3 s ahead of the audio clock as PTS discontinuities (post-ATTA anchor
  mismatch, broadcast PCR break, post-seek backlog). Smaller offsets stay paced.

- **Still-frame hold.** A head more than the wake threshold but under 3 s ahead —
  a post-seek / trick-exit re-anchor where the freshly anchored clock must advance
  to meet it — is held: the drain breaks without submitting, so the single freerun
  frame already shown after the `Clear()` stays on screen until the clock reaches
  its PTS. Each frame is then released exactly when due, so the transition is a
  brief freeze, not a crawl. A hold outside the corridor zeroes `lastDrainMs` so
  the resume drain isn't counted as a starvation miss.

- **No-clock hold (`DECODER_NO_CLOCK_HOLD_MS = 1.5 s`).** While `GetClock()` is
  NOPTS (audio priming after `Clear()` / seek) the drain *holds* a non-empty
  `jitterBuf` rather than freerunning pre-anchor video at VSync rate — which would
  land the head far ahead the moment audio anchors. A near-cap escape submits
  anyway if `jitterBuf` approaches `RESERVE_HARD_CAP`, so a fast HW decoder can't
  overflow waiting for an anchor that never comes (video-only stream). This covers
  the mux-interleave seek offset — a TS seek can land on a keyframe up to ~1 s
  ahead of the target audio.

- **`RESERVE_HARD_CAP` (`DECODER_RESERVE_HARD_CAP = 64`, ~1.3 s @ 50 fps).**
  Drop-oldest runaway guard for the case the gates above miss
  (`SkipStaleJitterFrames` only drops heads *behind* the clock, so PTS marching
  ahead with a valid clock could grow `jitterBuf` unbounded). Drop-oldest keeps
  the closest-to-due tail. The same cap bounds each stage of the reserve: the
  decode thread drop-oldest-trims `handoffQueue` if the present thread stalls past
  it (normally it backpressures long before — see the decouple section). It also
  bounds GPU surface retention: each held frame pins a 4K NV12 surface (~12 MB),
  so 64 ≈ 0.8 GB GTT — still dwarfing the < 40 ms VPP variance and the 8-slot
  prerender. If replay soft/hard-behind drops appear, `vaDriverMutex` contention
  from continuous decode is the suspect; raising the cap trades GTT for headroom.

`Freeze()` (pause) holds the drain directly: while `devicePaused` and not in
trick play the loop breaks without submitting, so the head's PTS can't drift
against the pinned-but-static audio clock (Architecture invariant 1). Resume
(`Play()`) lifts the hold and the pin together.

The **pre-fill bypass** releases the head up to `frameDur / 2` early when the
display prerender queue is empty (`PresentWakeThreshold90k()` returns `frameDur`
instead of `halfFrame`), keeping `PendingDepth()` at 1–2 instead of 0–1. The
total prefill window is therefore one `frameDur` — the decoder never runs more
than one frame ahead of strict-due. This absorbs audio-clock vs VSync phase drift
that would otherwise tick the underrun counter on a healthy stream.

### Drain bypasses

The due check is bypassed for:

- **Freerun** (`freerunFrames > 0` after `Clear()`, trick exit, audio codec
  change) — gives an instant first picture; the still-frame hold then freezes that
  one unpaced frame until the clock syncs.
- **`pendingDrops`** from a soft- or hard-behind burst — one drop per drain
  iteration until exhausted.

`SkipStaleJitterFrames()` runs at the top of every drain pass and bulk-drops
heads more than `HARD_THRESHOLD` (200 ms) behind the clock — cheaper than routing
each through catch-up.

### Steady-state `buf` depth

`buf` (jitterBuf depth at log emission) reflects input arrival rate minus the
gate's release rate.

- **Live TV.** `AUDIO_ALSA_BUFFER_MS = 400 ms` sizes the ALSA ring, so
  `GetClock()` lags wall time by roughly that; head frames sit in `jitterBuf`
  until the lagged clock catches them.
  `buf ≈ (AUDIO_ALSA_BUFFER_MS + broadcastLead) / frameDur`. Higher-bitrate
  streams ship more lead and run deeper; 4K VBR can swing `buf` by ~1 s within
  seconds as bitrate peaks stall packet arrival — that is the cushion working.
- **Replay, cold start.** dvbplayer bursts disk reads to refill an empty PES
  ring, so the decoder builds a ~40–60 frame backlog, then dvbplayer throttles to
  playback rate and `buf` stabilizes there.
- **Replay, post-`Clear()` (skip / track switch).** The PES ring is drained and
  dvbplayer feeds at real time from frame zero; the decoder produces at
  audio-clock pace and `buf` stays near 0.

### Audio packet queue (`aq`)

`aq` is the FIFO between `Decode()` and the audio thread. The thread decodes and
writes PCM as fast as packets arrive, so `aq` drains almost instantly and `aq = 0`
is healthy. A persistently non-zero `aq` means the audio decoder is falling behind
real time (CPU contention, ALSA write stall). The real audio cushion is the ALSA
ring, not this queue.

## Display prerender

`SyncAndSubmitFrame` hands the chosen frame to `cVaapiDisplay::SubmitFrame`, which
pushes onto `pendingFrames` (a `std::deque`, depth `DISPLAY_PRERENDER_SLOTS = 8`).
The display thread pops one per VSync, maps via VAAPI→PRIME, and commits via DRM
atomic. `SubmitFrame` **blocks** when all slots are full — this VSync backpressure
paces the present thread (and through the handoff, the decoder) to the display
refresh rate.

This is the **shallow, per-frame** cushion, distinct from the deeper decode-ahead
reserve upstream. The 8-slot depth (= 160 ms @ 50 fps) absorbs a single UHD VPP /
memory-bandwidth spike (~80 ms observed on 1280×720 → 3840×2160 upscale) plus
SW-decoder per-frame variance (libdav1d 1080p50 spikes 30–40 ms on complex frames)
without draining the cache. FHD HW paths never fill past 1–2 slots. The whole
pipeline is delayed in lockstep with audio, not just video, so the extra slots do
not shift `rawDelta`.

### Queue underrun detection

The display thread tracks `lastFrameCommitMs` (atomic, updated on every fresh
commit). On a VSync with no fresh frame it re-presents the previous buffer to keep
flip cadence + OSD updates alive. Each re-present streak is measured in
**wall-clock**, not VSync count, so the printed duration is true even when the
consumer loop is preempted or page-flip events arrive late.

State carried across iterations:

- `gapStartMs` — wall-clock baseline anchored on the first re-present of the
  current streak; `0` means "anchor on next re-present".
- `peakGapMs` — wall-clock peak of the current streak; reset on every fresh
  commit.

Per re-present (only when `lastFrameCommitMs != 0` and outside trick / sync-sleep /
warmup grace):

1. Anchor `gapStartMs = nowMs` if it's `0`.
2. `currentGapMs = nowMs − gapStartMs`.
3. If `currentGapMs < DISPLAY_UNDERRUN_IDLE_MAX_MS (10 s)`: update `peakGapMs`, and if
   `currentGapMs ≥ thresholdMs` and the log cooldown elapsed, emit
   `queue empty Nms; total=M`.
4. Else (gap ≥ 10 s): treat as paused / stopped, clear `peakGapMs`, leave
   `gapStartMs` so a long pause doesn't re-anchor every iteration.

`thresholdMs = (DISPLAY_PRERENDER_SLOTS + 2) × vsyncMs` — at 50 Hz that's
`10 × 20 ms = 200 ms`. The `+2` gives one VSync of natural prerender absorption
plus one so a single hiccup doesn't trip. On the next fresh commit, if
`peakGapMs ≥ thresholdMs` the recovery line `queue refilled after Nms; total=M`
reports the peak; then `gapStartMs`/`peakGapMs` reset.

Onset is rate-limited to once per `DISPLAY_UNDERRUN_LOG_INTERVAL_MS = 2 s`. `isClearing`,
`inTrick`, `inSyncSleep`, and `inPause` (device frozen) each force `gapStartMs = 0`
so a deliberate hard-ahead sleep (≤ 500 ms), trick hold, or pause does not surface
its own duration as a fake underrun. A `DISPLAY_PAGE_FLIP_STUCK_MS = 200 ms` watchdog
force-clears `isFlipPending` if the kernel swallows the page-flip event.

### Warmup grace

`DISPLAY_WARMUP_GRACE_MS = 3 s` suppresses the underrun gate for the first 3 s after the
decoder resumes from idle. Armed on a fresh commit when:

- `lastFrameCommitMs == 0` (post-`Clear()` reset), OR
- `nowMs − lastFrameCommitMs > DISPLAY_WARMUP_ACTIVE_WINDOW_MS (500 ms)` (idle resume where
  `BeginStreamSwitch` wasn't invoked — track switch, post-trick re-anchor).

Without it, every `Clear()` would log a spurious underrun while the filter graph
rebuilds and the audio clock anchors. `DISPLAY_WARMUP_ACTIVE_WINDOW_MS` exists only to arm this
grace — it does **not** gate the underrun log (that gate is `DISPLAY_UNDERRUN_IDLE_MAX_MS`).

## Sync bypass

The sync gate is bypassed (frame submitted unpaced) in:

- Trick mode (`SubmitTrickFrame()` paces via its own timer; audio is muted).
- Freerun window after `Clear()`, trick exit, or `NotifyAudioChange()`.
- Radio mode / NOPTS frame (no audio processor or no PTS to align on).
- Audio not yet running (`GetClock()` is NOPTS until the first `WritePcmToAlsa()`).

## Lifecycle

| Event                            | EMA               | Cooldown  | Jitter buffer |
| -------------------------------- | ----------------- | --------- | ------------- |
| Plugin start                     | invalid           | —         | empty |
| Channel switch (`Clear()`)       | reset             | unchanged | flushed; freerun armed |
| Catch-up enter                   | (drops silent)    | unchanged | drained silently to alignment |
| Catch-up exit                    | reset             | unchanged | one frame submitted normally |
| Soft drop                        | reset             | armed     | N frames dropped (one now, N−1 via `pendingDrops`, one per drain iteration) |
| Soft sleep                       | `−= measured`     | armed     | unchanged |
| Hard-behind                      | reset             | unchanged | N frames dropped |
| Hard-ahead (replay)              | reset             | armed     | unchanged |
| Hard-ahead (live)                | `−= measured`     | armed     | unchanged |
| Trick entry (FF/REW/slow)        | reset             | unchanged | reserve purged (epoch bump); paced by `SubmitTrickFrame`, no freerun |
| Trick exit → normal (`Play`)     | reset             | unchanged | reserve purged (epoch bump); freerun armed |
| Pause / resume (`Freeze`/`Play`) | unchanged         | unchanged | held (drain stops); clock pinned, no drops |
| Audio codec / track change       | unchanged         | unchanged | preserved; freerun armed |
| PCM channel-layout change        | unchanged         | unchanged | preserved; ALSA reopens, clock re-anchors (brief NOPTS) |
| Mediaplayer seek                 | reset             | unchanged | flushed; freerun armed; filter graph **preserved** |
| Mediaplayer playlist advance     | reset (on reopen) | unchanged | flushed; freerun armed; filter graph rebuilt |

Audio codec / track change preserves the buffer — catch-up silently realigns
against the new clock once it arrives, so dropping ~1 s of still-valid video buys
nothing.

Mediaplayer seek calls `cVaapiDevice::FlushForSeek()`, which fans out to
`decoder->FlushForSeek()` + `audioProcessor->Clear()`: same drain semantics as a
channel switch **except the filter chain stays alive**, since seek doesn't change
stream parameters and the VAAPI VPP rebuild would cost ~100 ms per seek for no
benefit (`FlushForSeek` rebuilds the filter only when the active chain contains an
`fps` node, which can't survive a seek-sized PTS jump). Entry open/close uses the
heavier `ClearForMediaPlayer()`, which rebuilds the filter (codec params may differ
at the next entry). Playlist advance closes the current `cVaapiMediaSource` and
opens the next; `OpenCodecWithInfo()` performs a full teardown when codecId /
extradata differ.

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
| `miss` | Drain gaps > 2 × output frame period since last log (upstream starvation; deliberate sync sleeps and trick-play holds are explicitly excluded via `sleptInLastSubmit`) |
| `drop` | Frames dropped (video behind) since last log — soft-behind, hard-behind, catch-up, stale-jitter, and pending-drop bursts combined |
| `skip` | Render delays (video ahead) since last log — soft sleep + hard-ahead combined |

`d ≈ avg` in steady state means the EMA has converged on current reality. The line
is suppressed during warmup and reissued immediately on warmup completion.
Periodic interval `LOG_INTERVAL_MS = 2 s`.

**Healthy steady state:** `avg` inside `±CORRIDOR`, `d ≈ avg`,
`miss = drop = skip = 0`. `buf` depth varies by mode (live: ALSA-cushion-driven;
replay cold start: ~40–60; replay post-`Clear()`: ~0).

Each soft / hard event also emits a per-event `dsyslog` line naming the cause
(`soft-ahead`, `soft-behind`, `hard-ahead live`, `hard-ahead replay`,
`hard-behind`, `stale-jitter bulk`, `catch-up entered (spike|warmup|sustained)`,
`catch-up complete`, `head too far in future … dropping`) for "why did this fire?"
without waiting for the next periodic line.

### Steady-state offset

The EMA does **not** settle at zero, and the non-zero baseline is **not** drift —
the controller leaves it alone, since correcting a non-drifting bias only
introduces visible jank. The baseline depends on whether the decoder is gate-bound
or throughput-bound:

- **Gate-bound** (live TV, replay cold start, anything with `buf > 0`): `d`
  settles inside the prefill envelope `[halfFrame, halfFrame + frameDur/2]` — at
  50 fps `[+10 ms, +20 ms]`, observed around `+15…+18 ms`. The exact position
  depends on how often the display drains the prerender queue to zero (enabling
  the pre-fill bypass) vs. holding at depth 1+ (gating at strict `halfFrame`).
- **Throughput-bound** (replay post-`Clear()`, `buf ≈ 0`): `d ≈ −frameDur`
  (~−20 ms @ 50 fps). Each frame is submitted as soon as decoded — no buffered
  lead — and the audio clock has advanced past the latency target by the time
  `SyncAndSubmitFrame` runs. The 1-frame pipeline-latency tail keeps this well
  inside `CORRIDOR`, leaving ~30 ms of headroom before a typical 10–15 ms/s
  pipeline/crystal drift could trip soft-behind.

To re-center either regime, tune `PcmLatency` / `PassthroughLatency`. Since
`rawDelta = videoPTS − GetClock() − pipelineLatency`, a **positive** value
subtracts more from `rawDelta`, releasing each frame at an earlier audio-clock
value — i.e. positive latency pulls video earlier vs audio (it delays audio
relative to video, per `config.h`).

## Constants

Every constant below is file-scope — in [src/config.h](src/config.h),
[src/audio.h](src/audio.h), [src/audio.cpp](src/audio.cpp),
[src/decoder.h](src/decoder.h), [src/decoder.cpp](src/decoder.cpp), or
[src/display.cpp](src/display.cpp) — and each carries a `///<` comment with
purpose and unit. Within each file they are grouped by sub-function under
`// --- label ---` rulers; the groups below mirror that layout.

Naming conventions:

- A module prefix (`PTS_` / `AUDIO_` / `DECODER_` / `DISPLAY_` / `CONFIG_` /
  `VDR_`) names the owning subsystem.
- `_MS` — milliseconds; `_90K` — 90 kHz PTS ticks (matches code variables like
  `rawDelta`, `smoothedDelta90k`, `latency90k`); `_VSYNCS` — display refresh
  periods; no suffix — dimensionless (sample / frame / slot counts, depths).
  (`PTS_TICKS_PER_MS` is the lone exception: there `_MS` means *per* millisecond.)
- `_MAX_MS` / `_MIN_MS` — a cap or clamp bound, with `MAX`/`MIN` trailing before
  the unit (`DECODER_SYNC_CORRECTION_MAX_MS`, `DECODER_DRAIN_FUTURE_MAX_MS`,
  `CONFIG_AUDIO_LATENCY_MAX_MS`).

**Clock & audio** (config.h, audio.h, audio.cpp)

| Constant                      | Value | Purpose |
| ----------------------------- | ----- | ------- |
| `PTS_TICKS_PER_MS`            | 90    | DVB 90 kHz PTS clock factor: ticks = ms × this (here `_MS` means *per* ms) |
| `AUDIO_ALSA_BUFFER_MS`        | 400   | ALSA ring size (ms); the lagged audio clock pulls live `buf` to ~MS/frameDur |
| `AUDIO_CLOCK_STALE_MS`        | 1000  | `GetClock()` extrapolation timeout before returning NOPTS |
| `AUDIO_QUEUE_HIGHWATER`            | 10    | Audio-feed backpressure gate for dvbplayer/PES replay (~320 ms AC-3) |
| `AUDIO_QUEUE_HIGHWATER_MEDIAPLAYER`| 32    | Audio-feed gate for the single-cursor mediaplayer demux (~1 s AC-3); deeper than `HIGHWATER` because one cursor feeds both audio and video |
| `AUDIO_QUEUE_CAPACITY`             | 100   | Audio packet-queue overflow backstop (~3.2 s); the HIGHWATER gates are the real limit |
| `CONFIG_AUDIO_LATENCY_MIN_MS` | −200  | Lower clamp on the `PcmLatency` / `PassthroughLatency` operator knobs |
| `CONFIG_AUDIO_LATENCY_MAX_MS` | 200   | Upper clamp on the `PcmLatency` / `PassthroughLatency` operator knobs |

**Mediaplayer feed pacing** (mediaplayer.h, device.cpp)

| Constant                                    | Value  | Purpose |
| ------------------------------------------- | ------ | ------- |
| `MEDIAPLAYER_MAX_LOOKAHEAD_90K`             | 135000 | Real-time demux brake: max audio lookahead (1.5 s @ 90 kHz) of the latest pushed audio PTS over the audio clock before the demux throttles; keeps libavformat's fast file reads from overrunning the reserve |
| `MEDIAPLAYER_JITTERBUF_BACKPRESSURE_FRAMES` | 48     | Pre-anchor video-depth gate (¾ of `DECODER_RESERVE_HARD_CAP`); the sole demux brake for VIDEO-ONLY streams while no audio clock exists yet |

**Video queues & decode-ahead reserve** (decoder.h, decoder.cpp)

| Constant                    | Value | Purpose |
| --------------------------- | ----- | ------- |
| `DECODER_QUEUE_CAPACITY`    | 200   | Video packet queue depth (~4 s @ 50 fps) |
| `DECODER_SUBMIT_TIMEOUT_MS` | 100   | Present-side VSync backpressure budget inside `display->SubmitFrame()` |
| `DECODER_RESERVE_HARD_CAP`  | 64    | Cap on the decode-ahead reserve (handoffQueue + jitterBuf, ~1.3 s @ 50 fps **total**); decode-side backpressure / present-side drop-oldest guard; also bounds 4K-surface GTT |

**Sync controller — corridor / EMA / cooldown** (decoder.cpp)

| Constant                       | Value | Purpose |
| ------------------------------ | ----- | ------- |
| `DECODER_SYNC_COOLDOWN_MS`     | 5000  | Min interval between soft corrections (= 5 EMA time constants) |
| `DECODER_SYNC_HINT_MAX_AGE_MS` | 5000  | Max age (= `…COOLDOWN_MS`) of a pre-correction `stableDelta` snapshot before the seek hint falls back to the current `smoothedDelta` |
| `DECODER_SYNC_CORRIDOR_90K`    | 4500  | Soft corridor half-width (= 50 ms × `PTS_TICKS_PER_MS`); below lipsync percept threshold |
| `DECODER_SYNC_EMA_SAMPLES`     | 50    | EMA divisor (~1 s @ 50 fps); residual accumulator → exact convergence |
| `DECODER_SYNC_WARMUP_SAMPLES`  | 50    | Samples averaged before the EMA seed (~1 s @ 50 fps) |
| `DECODER_SYNC_LOG_INTERVAL_MS` | 2000  | Periodic sync diagnostic interval (ms) |
| `DECODER_SYNC_FREERUN_FRAMES`  | 1     | Unpaced frames after sync-disrupting events |

**Sync controller — hard transients** (decoder.cpp)

| Constant                          | Value | Purpose |
| --------------------------------- | ----- | ------- |
| `DECODER_SYNC_HARD_THRESHOLD_90K` | 18000 | Hard-transient threshold (= 200 ms × `PTS_TICKS_PER_MS`); 2× = catch-up spike entry |
| `DECODER_SYNC_CORRECTION_MAX_MS`  | 200   | Soft-event cap, derived = `HARD_THRESHOLD ÷ PTS_TICKS_PER_MS`, so one event fully closes the corridor |
| `DECODER_SYNC_HARD_AHEAD_MAX_MS`  | 500   | Live hard-ahead sleep cap (ms) |

**Sync controller — catch-up logging** (decoder.cpp)

| Constant                                   | Value | Purpose |
| ------------------------------------------ | ----- | ------- |
| `DECODER_SYNC_CATCHUP_LOG_INTERVAL_MS`     | 2000  | Min interval between catch-up entry/exit log pairs; suppresses flood during sustained slow-decode cycling |
| `DECODER_SYNC_CATCHUP_SUMMARY_INTERVAL_MS` | 10000 | Cadence of the aggregated "cycling sustained" summary while catch-up keeps cycling under the log interval |

**Present-thread drain** (decoder.cpp)

| Constant                      | Value | Purpose |
| ----------------------------- | ----- | ------- |
| `DECODER_DRAIN_FUTURE_MAX_MS` | 3000  | Future-head discontinuity guard: drop heads > 3 s ahead; smaller offsets hold (still frame) until due |
| `DECODER_NO_CLOCK_HOLD_MS`    | 1500  | Walltime the drain holds a non-empty jitterBuf while `GetClock()` is NOPTS before no-clock freerun (covers the mux-interleave seek offset) |

**Trick-play pacing** (decoder.h, decoder.cpp)

| Constant                            | Value | Purpose |
| ----------------------------------- | ----- | ------- |
| `DECODER_TRICK_QUEUE_DEPTH`         | 1     | Handoff reserve depth during trick play (Poll() throttles the producer; overflow drops incoming) |
| `DECODER_TRICK_HOLD_MS`             | 20    | Base per-frame hold (~one field period @ 50 Hz). Slow **forward** scales it by the slowdown factor (capped at `DECODER_TRICK_SLOW_HOLD_MAX_MS`); fast trick keeps the base hold and scales the frame-skip multiplier; slow **reverse** uses the BASE/STEP/MAX holds below |
| `DECODER_TRICK_SLOW_HOLD_MAX_MS`    | 200   | Cap on the slow-forward per-frame hold (≥ 5 fps) |
| `VDR_SLOW_REVERSE_SPEED_MULT`       | 12    | VDR `dvbplayer.c` SPEED_MULT; divided back out in `SetTrickSpeed` to recover the slow-reverse slowdown level |
| `DECODER_SLOW_REVERSE_HOLD_BASE_MS` | 260   | Slow-reverse per-frame hold floor (reverse steps a fixed ~0.4 s of content per frame) |
| `DECODER_SLOW_REVERSE_HOLD_STEP_MS` | 70    | Added to the slow-reverse hold per slowdown unit |
| `DECODER_SLOW_REVERSE_HOLD_MAX_MS`  | 700   | Cap on the slow-reverse hold |

**Display prerender & underrun tracking** (config.h, display.cpp)

| Constant                             | Value | Purpose |
| ------------------------------------ | ----- | ------- |
| `DISPLAY_PRERENDER_SLOTS`            | 8     | Present→display prerender depth (= 160 ms @ 50 fps); absorbs a UHD VPP / bandwidth spike + SW-decoder variance |
| `DISPLAY_WARMUP_ACTIVE_WINDOW_MS`    | 500   | Min idle gap on a fresh commit that arms the warmup grace (does not gate the underrun log) |
| `DISPLAY_WARMUP_GRACE_MS`            | 3000  | Post-idle grace suppressing underrun logs while the pipeline anchors |
| `DISPLAY_UNDERRUN_IDLE_MAX_MS`       | 10000 | Wall-clock streak beyond which a re-present gap is treated as paused / stopped (peak cleared) |
| `DISPLAY_UNDERRUN_LOG_INTERVAL_MS`   | 2000  | Min interval between underrun-onset dsyslog lines |
| `DISPLAY_UNDERRUN_THRESHOLD_VSYNCS`  | 10    | `(DISPLAY_PRERENDER_SLOTS + 2)`; `thresholdMs = DISPLAY_UNDERRUN_THRESHOLD_VSYNCS × vsyncMs` is the gap that trips a log |
| `DISPLAY_PAGE_FLIP_STUCK_MS`         | 200   | Stuck-flip watchdog: force-clear `isFlipPending` if the kernel swallows the page-flip event |
