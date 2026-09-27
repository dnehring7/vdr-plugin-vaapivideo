# A/V Synchronization

Developer documentation for the plugin's A/V sync design: the clock model,
threading, buffering, correction regimes, and their diagnostics. For
user-facing setup and configuration see [README.md](README.md); the only
operator knobs are the two latency values described under
[Steady-state offset](#steady-state-offset).

Contents:

1. [Problem](#problem) — why correction is needed at all
2. [Architecture](#architecture) — data flow and the three invariants
3. [Decode / present decouple](#decode--present-decouple) — the two decoder threads
4. [Filter pipeline](#filter-pipeline) — the FFmpeg graph both domains build
5. [Per-frame sync](#per-frame-sync-syncandsubmitframe) — delta measurement and EMA smoothing
6. [Correction regimes](#correction-regimes) — soft corridor, hard transients, catch-up
7. [Sync bypass](#sync-bypass) — when frames are submitted unpaced
8. [Jitter buffer](#jitter-buffer-unified-drain) — the present-thread drain
9. [Display prerender](#display-prerender) — the per-VSync cushion and underrun detection
10. [Lifecycle](#lifecycle) — what each event resets
11. [Stream start](#stream-start) — channel-switch latency budget, first keyframe, first audio
12. [End of stream](#end-of-stream) — playing the tail out: VDR's `Drain()`, the mediaplayer's drain, what each buffer contributes
13. [Tracing](#tracing) — the `-t` / `TRACE` gate in front of the diagnostics below
14. [Diagnostic log](#diagnostic-log) — reading the `sync` line, tuning the baseline
15. [Stream-start trace](#stream-start-trace) — the `trace +Nms` timeline of a switch
16. [Constants](#constants) — the values that shape observable behaviour, and the naming rules

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
        pendingFrames (DISPLAY_PRERENDER_CAPACITY = 8) → display thread → KMS commit
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
  position, not at the earlier keyframe libavformat lands on; every re-anchor likewise arms
  `discardVideoBefore90k`, which flags the video preroll with `AV_PKT_FLAG_DISCARD` — decoded for
  the reference chain, never output, so it neither burns a GOP of VPP work on its way to being
  catch-up-dropped nor replays in slow motion (trick pacing has no clock gate). Fast-forward
  entry disarms the window: its start frame is deliberately the keyframe at/below the target.
  See `cVaapiMediaSource` in [src/mediaplayer.cpp](src/mediaplayer.cpp).

Three invariants:

1. **Audio is master.** `cAudioProcessor::GetClock()` returns the PTS at the DAC
   output: `playbackPts + (now − lastClockUpdateMs) × 90`. `playbackPts` is
   republished on every ALSA write as `endPts − snd_pcm_delay()` (i.e. per
   decoded packet, much faster than the ~25 ms ALSA period); the wall-clock
   age-extrapolation fills the gaps between writes, so reads stay smooth to ~1 ms.
   A seqlock makes the read lock-free. `GetClock()` returns `AV_NOPTS_VALUE`
   while the DAC is not running (below its start threshold `snd_pcm_delay()`
   only counts queued frames, so a clock would advance through audio nobody
   hears) or once a write is older than `AUDIO_CLOCK_STALE_MS = 1 s`; the
   controller then holds or freeruns.

   Three write-path commands manage the clock:
   - **`Clear()`** (seek / channel change): `snd_pcm_drop`+`prepare`, drain the
     packet queue, and `ResetPlaybackClock()` — `GetClock()` goes NOPTS and the
     decoder freeruns until audio re-anchors.
   - **`DropOutput()`** (Mute / SetTrickSpeed): `snd_pcm_drop`+`prepare`
     and drain the packet queue, but **keep** `playbackPts` — the clock stays
     valid until the next write, so the decoder is not thrown into freerun.
     That write finds the DAC stopped and publishes NOPTS: the picture holds
     until sound is audible again, then re-anchors on the real DAC position.
   - **`PauseOutput()`** / **`ResumeOutput()`** (Freeze / Play): lose nothing.
     The players resume from their read position, so the ring (~800 ms) and the
     packet queue are part of the stream — dropping them anchored the first
     write after `Play()` ~1 s ahead and the presenter cut that much video.
     `snd_pcm_pause` keeps the ring, the worker stops popping, and `GetClock()`
     is **pinned** at the DAC position; `Play()` restarts both where they
     stopped. A packet already popped finishes first (in a paused ring it would
     break the pin). Without hardware pause only the ring is dropped; a stopped
     ring reads NOPTS after `Play()` until audio restarts, except over the EOS
     tail. VDR enters slow motion from a pause without `Play()`: the trick's
     `DropOutput()` ends the pause (the pin stays until the next write), while
     `Mute()` leaves a paused output alone.

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

   Audio-only replay EOF is a deliberate exception to "keep the clock advancing":
   VDR core ends a replay only when `GetSTC()` reaches the last frame or stalls
   for 3 s (`StuckAtEof`), and with no video the STC *is* this audio clock. At EOF
   `cDvbPlayer` continuously re-pushes the last PES to flush the device, so
   decoding those repeats would keep the DAC clock advancing forever and radio
   replay would never auto-stop. `PlayAudio()` drops the re-delivered PES by exact
   repeated PTS (best-effort; baseline reset on every timeline boundary) so
   `GetClock()` ages stale and VDR ends the replay itself.

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
   VSync backpressure (= display rate) and drifts (60→50 consumes source at 83%,
   24→50 at 208%). Audio-clocked paths would eventually correct that drift via
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

They are joined by a bounded **blocking** handoff: `handoffMutex` (a near-leaf
lock — see below), with `handoffCondition` waking the present thread and
`handoffNotFull` waking the decode thread. When the decoded reserve reaches `DECODER_RESERVE_CAPACITY`
— the published total `jitterBuf + handoffQueue`, or `handoffQueue` alone — the
decode thread *waits* rather than dropping, so the upstream packet queue (and
through it VDR's flow control) stays authoritative — backpressure is never resolved
by discarding already-decoded frames. (A drop-oldest exists only as an
`[[unlikely]]` memory-safety net for a wedged presenter.)

### Decode-ahead reserve

`jitterBuf` (present side) + `handoffQueue` (handoff) together form the
**decode-ahead reserve**, bounded to `DECODER_RESERVE_CAPACITY` (~1.3 s @ 50 fps).
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

- `Clear()`, `FlushForSeek()`, `SetTrickSpeed()` on a trick generation boundary
  other than slow-forward entry, and the deferred trick-exit
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

Lock order is `codecMutex → parserMutex → packetMutex`. `handoffMutex` is a
near-leaf: destroying a frame under it may drop the last `FilterGraphToken`,
whose deleter locks the display's `vaDriverMutex` — its only outgoing edge, and
safe because `vaDriverMutex` never leads back to a decoder-side lock (see the
lock-order comments in decoder.h / display.cpp). `jitterBuf` and the sync
controller stay present-thread-private (no lock).

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

Frames leave the chain with the filters' own timestamps, rescaled from the
sink's time base (halved by a field-rate deinterlacer, `1/rate` after `fps`)
back to 90 kHz. A temporal deinterlacer emits frame N only once N+1 has
arrived, so the input's PTS would label every 1080i frame one frame late and
put video a frame behind audio. Slow forward keeps them too: it paces on the
distance between them. Only fast and reverse trick play stamp synthetically
(`sourcePts + i·frameDur`): they pace on source-PTS strides, and their
ghost-field drop recognizes the first output by that stamp.

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

### Graph lifetime and rebuild debounce

A rebuilt graph is retired, not freed in place: every frame it produced carries
a `FilterGraphToken` (a shared keep-alive on the graph), because iHD surfaces
hold a raw back-pointer to the producing VPP context that `vaSyncSurface()`
dereferences — freeing the graph while up to ~1.3 s of its frames (reserve +
display prerender) are still queued is a driver use-after-free on the display
thread's next PRIME export. The retired graph is freed, under the VA driver
mutex, when the last frame referencing it leaves the pipeline.

Rebuild *requests* (`ScaleVideo()` resize, zoom preset change) are debounced on
the decode thread: the rebuild runs once the burst has been quiet for 150 ms,
deferred at most 500 ms in total. A skin firing several
resize calls per menu transition costs one rebuild instead of one per call;
old-sized frames keep painting at the old scanout rect until the first
new-sized fb arrives (`PresentBuffer` promotes `videoRect` then), so the
deferral is invisible.

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
   exact convergence to the rawDelta mean even when `|diff| < N` (a naive integer
   step would round to 0).

`ResetSmoothedDelta()` clears warmup, EMA, residual, hard-debounce counters, and
catch-up state in one call. Called on channel switch, hard-behind, catch-up exit,
and `WaitForAudioCatchUp`.

**Fast-start seed.** A `FlushForSeek()` (same stream, same pipeline) carries the
pre-seek converged delta across the flush as `seekHintDelta90k` and seeds the EMA
from it on the first post-seek frame, skipping the 50-sample warmup. The
GPU-vs-audio offset is a property of the pipeline (decode + VPP + KMS latency vs.
ALSA hw_ptr), not the playback position, so the pre-seek steady state is the right
seed and the right catch-up exit target. The hint is captured as the
*pre-correction* `stableDelta90k` (a sleep's predictive EMA bump makes the live
value transient during recovery), clamped into the soft corridor, and ages out
after one cooldown (5 s). Plain `Clear()` (content boundary) drops the
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
lines are throttled to one per 2 s; suppressed cycles fold into a periodic
`catch-up cycling sustained: …` line every 10 s, with a final `catch-up cycling
settled: …` when it stops.

The entry (−100 ms) and exit (−50 ms) thresholds give `CORRIDOR` of hysteresis.
The exit at `−CORRIDOR` is also the **highest threshold guaranteed reachable**:
catch-up only advances `rawDelta` while frames are cached in `jitterBuf` (drops
are fast pops, audio barely moves). Once the cache drains, each further drop waits
one VPP cycle for the next frame, so on marginal-VPP hardware (UHD upscale at
~50 fps == audio rate) PTS and clock advance equally and `rawDelta` stops
climbing — targeting `+halfFrame` would hang catch-up forever. The small residual
negative offset that may remain is well below the 80 ms lip-sync perception threshold.

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

## Sync bypass

The sync gate is bypassed (frame submitted unpaced) in:

- Trick mode (`SubmitTrickFrame()` paces via its own timer; audio is muted).
- Freerun window after `Clear()`, trick exit, or `NotifyAudioChange()`.
- Radio mode / NOPTS frame (no audio processor or no PTS to align on).
- Audio not yet running (`GetClock()` is NOPTS until the DAC starts, i.e. until
  the ring holds `AUDIO_ALSA_START_MS`).

## Jitter buffer (unified drain)

The video drain is unified across live and replay and runs on the **present
thread**: each iteration splices `handoffQueue` into the private `jitterBuf`
(`std::deque`) and pops when due.

```
splice: handoffQueue → jitterBuf      (drop frames with producedEpoch < presentEpoch)
front-purge: drop jitterBuf heads with producedEpoch < presentEpoch
runaway guard: if jitterBuf > RESERVE_CAPACITY, drop oldest down to cap
SkipStaleJitterFrames(): bulk-drop heads more than HARD_THRESHOLD behind clock (not paused, not in trick)
loop:
  if devicePaused:                              break                     // Freeze: hold, clock pinned
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
  anyway if `jitterBuf` approaches `RESERVE_CAPACITY`, so a fast HW decoder can't
  overflow waiting for an anchor that never comes (video-only stream). This covers
  the mux-interleave seek offset — a TS seek can land on a keyframe up to ~1 s
  ahead of the target audio.

- **`RESERVE_CAPACITY` (`DECODER_RESERVE_CAPACITY = 64`, ~1.3 s @ 50 fps).**
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

`Freeze()` (pause) holds the drain directly: while `devicePaused` the loop
breaks without submitting, so the head's PTS can't drift against the
pinned-but-static audio clock (Architecture invariant 1). Resume (`Play()`)
lifts the hold and the pin together. `devicePaused` is the presentation hold,
not VDR's paused flag: VDR starts slow motion from a pause without `Play()`
(`TrickSpeed()` lifts the hold) and pauses it again with a bare `Freeze()`,
which must stop the picture — trick play included.

The **pre-fill bypass** releases the head up to `frameDur / 2` early when the
display prerender queue is empty (`PresentWakeThreshold90k()` returns `frameDur`
instead of `halfFrame`), keeping `PendingDepth()` at 1–2 instead of 0–1. The
total prefill window is therefore one `frameDur` — the decoder never runs more
than one frame ahead of strict-due. This absorbs audio-clock vs. VSync phase drift
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

- **Live TV.** The ALSA ring holds `AUDIO_ALSA_START_MS` (300 ms) once playback
  starts — see [Ring cushion](#ring-cushion) for why that number, not
  `AUDIO_ALSA_BUFFER_MS`, is the steady level — so `GetClock()` lags wall time by
  roughly that; head frames sit in `jitterBuf` until the lagged clock catches
  them. `buf ≈ (ringFill + broadcastLead) / frameDur`. Higher-bitrate
  streams ship more lead and run deeper; 4K VBR can swing `buf` by ~1 s within
  seconds as bitrate peaks stall packet arrival — that is the cushion working.
- **Replay, cold start.** dvbplayer bursts disk reads to refill an empty PES
  ring, so the decoder builds a ~40–60 frame backlog, then dvbplayer throttles to
  playback rate and `buf` stabilizes there.
- **Replay, post-`Clear()` (skip / track switch).** The PES ring is drained and
  dvbplayer feeds at real time from frame zero; the decoder produces at
  audio-clock pace and `buf` stays near 0.

### Audio packet queue (`aq`)

`aq` is the FIFO between `Decode()` and the audio thread. Its healthy depth
depends on the input path:

- **Live / PES replay** — packets arrive at real time and the thread decodes and
  writes them immediately, so `aq` drains almost instantly and sits at **0**. A
  persistently non-zero `aq` here means the audio decoder is falling behind real
  time (CPU contention, ALSA write stall).
- **Mediaplayer** — the single-cursor demux reads the file far faster than real
  time until the audio feed gate stops it, and the audio thread blocks in the
  ALSA write once the ring is full, so `aq` sits **pegged at
  `AUDIO_QUEUE_HIGHWATER_MEDIAPLAYER` (32) by design** — that is the pre-read
  lead working, not a decoder falling behind.

Either way the real audio cushion is the ALSA ring, not this queue.

### Ring cushion

The ALSA ring is sized by `AUDIO_ALSA_BUFFER_MS` (800 ms), but the level it
actually runs at is set by `AUDIO_ALSA_START_MS` (300 ms), the fill the DAC
starts at. A live feed arrives at exactly 1x, so once playback starts, writes and
playout advance at the same rate: **whatever is in the ring at the start is the
average level for the rest of the stream** — it never grows back. The level then
sawtooths with the arrival granularity, because a DVB audio PES is not one frame
but up to eight: 4608-byte MP2 payloads carry 192 ms, 7680-byte AC-3 payloads
160 ms, and each arrives one period after the last.

So the cushion has to clear one PES period plus arrival jitter. With the earlier
`bufferSize / 3` threshold (133 ms) it did not: measured on satip, the ring peaked
at 144 ms and grazed **8–19 ms** on every PES cycle, and a single late payload
underran it — `snd_pcm_writei` returned `-EPIPE`, `snd_pcm_recover()` re-prepared,
and the DAC then stayed silent until the threshold refilled. Audibly: sound
starts, drops out briefly ~0.5–1 s into the channel switch, and resumes. Nothing
logged it, because the recovery was silent; it was only visible as `state: XRUN`
in `/proc/asound/card*/pcm*p/sub*/status`. That recovery now logs (rate-limited),
and at 300 ms the same measurement gives min 206–226 ms, avg 313–320 ms, no xruns.

The cost is paid once per stream start: the DAC starts ~150 ms later, and A/V lock
follows ~190 ms later, since the video waits for an audio clock that is now
anchored further back. Volume changes also take up to one ring fill to become
audible (mute does not — `DropOutput()` drops the ring).

## Display prerender

`SyncAndSubmitFrame` hands the chosen frame to `cVaapiDisplay::SubmitFrame`, which
pushes onto `pendingFrames` (a `std::deque`, depth `DISPLAY_PRERENDER_CAPACITY = 8`).
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
3. If `currentGapMs` < 10 s: update `peakGapMs`, and if
   `currentGapMs ≥ thresholdMs` and the log cooldown elapsed, emit
   `queue empty Nms; total=M`.
4. Else (gap ≥ 10 s): treat as paused / stopped, clear `peakGapMs`, leave
   `gapStartMs` so a long pause doesn't re-anchor every iteration.

`thresholdMs = (DISPLAY_PRERENDER_CAPACITY + 2) × vsyncMs` — at 50 Hz that's
`10 × 20 ms = 200 ms`. The `+2` gives one VSync of natural prerender absorption
plus one so a single hiccup doesn't trip. On the next fresh commit, if
`peakGapMs ≥ thresholdMs` the recovery line `queue refilled after Nms; total=M`
reports the peak; then `gapStartMs`/`peakGapMs` reset.

Onset is rate-limited to once per 2 s. `isClearing`,
`inTrick`, `inSyncSleep`, and `inPause` (device frozen) each force `gapStartMs = 0`
so a deliberate hard-ahead sleep (≤ 500 ms), trick hold, or pause does not surface
its own duration as a fake underrun. A 200 ms watchdog
force-clears `isFlipPending` if the kernel swallows the page-flip event.

### Warmup grace

A 3 s grace suppresses the underrun gate after the
decoder resumes from idle. Armed on a fresh commit when:

- `lastFrameCommitMs == 0` (post-`Clear()` reset), OR
- `nowMs − lastFrameCommitMs` > 500 ms (idle resume where
  `BeginStreamSwitch` wasn't invoked — track switch, post-trick re-anchor).

Without it, every `Clear()` would log a spurious underrun while the filter graph
rebuilds and the audio clock anchors. The 500 ms idle window only arms this
grace — it does **not** gate the underrun log (the 10 s idle limit above does).

## Lifecycle

| Event                            | EMA               | Cooldown  | Jitter buffer |
| -------------------------------- | ----------------- | --------- | ------------- |
| Plugin start                     | invalid           | —         | empty |
| Channel switch (`Clear()`)       | reset             | unchanged | flushed; freerun armed; stream-start trace armed by `SetPlayMode()` when tracing is on |
| Catch-up enter                   | (drops silent)    | unchanged | drained silently to alignment |
| Catch-up exit                    | reset             | unchanged | one frame submitted normally |
| Soft drop                        | reset             | armed     | N frames dropped (one now, N−1 via `pendingDrops`, one per drain iteration) |
| Soft sleep                       | `−= measured`     | armed     | unchanged |
| Hard-behind                      | reset             | unchanged | N frames dropped |
| Hard-ahead (replay)              | reset             | armed     | unchanged |
| Hard-ahead (live)                | `−= measured`     | armed     | unchanged |
| Trick entry (FF/REW/slow REW)    | reset             | unchanged | reserve purged (epoch bump); paced by `SubmitTrickFrame`, no freerun |
| Slow-forward entry (from pause)  | unchanged         | unchanged | reserve and filter graph kept: VDR reads on where the pause stopped and never resends it; paced by PTS distance × slowdown, so the reserve's field-rate frames and later ones run at one speed |
| Trick exit → normal (`Play`)     | reset             | unchanged | reserve purged (epoch bump); freerun armed |
| Pause / resume (`Freeze`/`Play`) | unchanged         | unchanged | held (drain stops, slow motion included: VDR pauses it with a bare `Freeze()`); packet queues and ALSA ring kept, clock pinned; resume continues gapless; a trick from the pause drops the audio and ends the hold |
| Audio codec / track change       | unchanged         | unchanged | preserved; freerun armed |
| PCM channel-layout change        | unchanged         | unchanged | preserved; ALSA reopens, clock re-anchors (brief NOPTS) |
| Mediaplayer seek                 | reset             | unchanged | flushed; freerun armed; filter graph **preserved** |
| Mediaplayer trick entry/exit     | reset             | unchanged | flushed + re-anchored at the shown position (same `FlushForSeek` path as a seek) |
| Mediaplayer playlist advance     | reset (on reopen) | unchanged | flushed; freerun armed; filter graph rebuilt |

Audio codec / track change preserves the buffer — catch-up silently realigns
against the new clock once it arrives, so dropping ~1 s of still-valid video buys
nothing. The same path is also entered mid-stream when the audio sink forces a
re-detect: after repeated decode-failure cascades with no decoded frame
(`cAudioProcessor::TakeCodecRedetectRequest`, a misdetected/changed codec), the
device resets the audio codec, `Clear()`s the sink, and calls `NotifyAudioChange()`
— so the clock re-anchors and video freeruns exactly as on a track switch.

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

Mediaplayer **trick transitions** ride the same seek path: every entry and exit
re-anchors at the shown position via `SeekToMs()`. This is mandatory — the
decoder purges its decoded reserve on trick exits and fast/reverse entries
(`clearEpoch` bump in `SetTrickSpeed`), so continuing from the demux cursor would
jump the reserve depth ahead of what the viewer saw. The re-anchor flags the re-fed
preroll with `AV_PKT_FLAG_DISCARD` like every seek (see Architecture above);
fast-forward entry disarms that window, since its start frame is the keyframe
at/below the anchor.
Fast/slow reverse feeds isolated keyframes by stepping `av_seek_frame` backward
(there is no VDR index file), paced through `HasFeedSpace()` exactly like the
PES trick path. No audio or subtitle packets are fed during any trick mode —
the lookahead throttle returns NOPTS outside normal play for the same reason.
Two exceptions to the uniform re-anchor: exits immediately followed by another
repositioning command (jump keys, playlist advance, audio-track switch) skip it
— that command's own seek/reopen re-anchors, and the Exit seek would only
double the flush (`LeaveTrickWithoutReanchor`) — and a **failed** transition
seek fails closed: the player leaves trick mode (`AbortTrick`) rather than
running a trick on the wrong timeline or retrying a reverse seek forever.

## Stream start

What a viewer waits for after a zap is, in order: the tuner / stream join
(outside the plugin — ~1.7 s SAT>IP on the test rig, a multicast join or HTTP
connect on IPTV), the next keyframe (a random position inside the GOP), the
pipeline's own startup cost, and finally the moment picture and sound run
together. The plugin's share is kept to a few tens of milliseconds by three
rules in `cVaapiDevice::PlayVideo` / `PlayAudio`:

- **The first keyframe PES opens the codec, unconditionally.** `DetectVideoCodec`
  only fires on a parameter-set-bearing keyframe PES, so whatever it accepts is
  the first decodable picture of the new stream. An earlier "same codec as the
  previous channel → wait for a second detection" guard could only confirm on
  the *next* keyframe, i.e. it threw away the first one and cost a full GOP on
  every same-codec switch (0.6–0.9 s on DVB, several seconds on long-GOP IPTV)
  — the dominant reason video trailed audio after a zap. The old-channel PES
  residue it guarded against cannot occur with VDR 2.x transfer mode (the old
  player is detached and `PlayTs(NULL)` resets the TS→PES assemblers before the
  tuner is re-pointed; the new player attaches only after the new receiver
  exists; the dvb/satip/iptv devices clear their TS rings in `OpenDvr()`).

- **The first keyframe's access unit is released as soon as its PES is
  complete** (live only). `av_parser_parse2` holds an AU back until the next
  AU's start code arrives, which for the very first picture means waiting for
  the next PES — one frame period on DVB, a whole delivery burst on IPTV. In a
  TS the PES boundary is the AU boundary, and VDR chunks an oversized picture
  into PES packets of exactly 0xFFF0 + 6 bytes (remux.c `MAXPESLENGTH` plus
  the header), so any shorter video
  PES ends its picture: `PlayVideo` then calls
  `cVaapiDecoder::ReleasePendingAccessUnit()`, which drains the parser and
  recreates it (a flushed `AVCodecParser` keeps a stale `frame_start_found` and
  would otherwise cut the next AU at its first NAL). One-shot per codec open;
  replay is excluded because the pre-TS PES recording format splits a picture
  across many small PES packets.

- **Audio confirms from two frames, not necessarily two PES.** Codec certainty
  needs two corroborated audio frames (one frame misdetects), and an
  in-session `AUDI N` track switch
  can deliver one complete PES of the *old* PID first, which is why the 2-of-2
  rule spans payloads. On a **fresh stream** (`audioFreshStart`, armed by
  `SetPlayMode()`) nothing stale can precede the first PES, so a `chained`
  detection — two frame headers linked by an exact frame-length step inside one
  payload, i.e. two real audio frames — confirms immediately and the codec opens
  on the first PES (~25–100 ms sooner; the log says `chained 1-PES`). Weaker
  evidence (exact fill, Dolby head-span, DTS/TrueHD lone sync) keeps 2-of-2, and
  its 1-of-2 candidate PES is then held, not dropped, and fed once confirmed, so
  the wait costs no audio and the ALSA start threshold
  (`AUDIO_ALSA_START_MS`) fills one PES sooner. In-session track changes
  disable both fast paths: full 2-of-2 across payloads, candidate dropped. The
  test is "this stream has already delivered an audio PES" (`audioPesSeen`), not
  "a codec was confirmed" — a track change landing inside the detection window
  is in-session too.

- **A Dolby track switch outruns VDR's own PID switch.** VDR fires
  `SetDigitalAudioDevice(true)` *before* it assigns `currentAudioTrack`
  (device.c), so `PlayTs()` keeps routing the **old** track's PID for as long as
  the plugin's handler runs — and `cAudioProcessor::Clear()` in it can block tens
  of milliseconds on the ALSA mutex. Those leftovers are complete, decisive PES:
  fed to the detector they win the 2-of-2 vote for the codec being left behind,
  after which the new track's bitstream starves the wrong decoder until the
  cascade escalation re-detects (measured **9.5 s of silence** on an MP2 → Dolby
  switch before the fix). A DVB Dolby track always rides in `private_stream_1`,
  so `PlayAudio()` drops audio PES with any other stream id until the switch
  lands, bounded by 500 ms so an unusual mux
  costs a hiccup instead of the audio. The gate arms only for an **in-session**
  switch (`audioPesSeen`) — a stream that has not delivered audio yet has no
  leftovers to keep out — and is evaluated only on payloads `ParsePes()`
  accepted, so the PES-shaped garbage an encrypted channel produces before its
  CAM has keys can neither time it out nor be logged against it. Every other track hook fires *after* the
  assignment, where at most the one PES already complete in `tsToPesAudio` can be
  stale — that one the 2-of-2 rule covers. Measured after the fix: hook →
  `confirmed (2-of-2)` in 230–520 ms, both directions, no cascade.

What remains after the keyframe is inherent: decode + VPP build + one VSync
(a few tens of ms keyframe-PES → CRTC on progressive channels; on 1080i a
temporal deinterlacer first waits for its reference frames — the VAAPI one never
shows the very first picture, `bwdif` shows it one input later), then the
**still-frame hold**.
The first frame is shown unpaced (freerun) the moment it exists, but the audio
that belongs to its PTS has not even arrived yet — in a DVB mux video is sent
~0.5–1 s ahead of its PTS and audio only ~0.1–0.2 s, so the first picture
freezes for roughly `videoLead − audioLead + AUDIO_ALSA_START_MS` (0.6–1.0 s
observed, the last term being the ring cushion the stream needs anyway — see
[Ring cushion](#ring-cushion)) until the audio clock reaches it, and motion starts
A/V-locked from there. No player can start synced motion earlier than the arrival of that audio;
the alternatives (muting instead of freezing, or crawling the video) are worse.

### Before the stream reaches the plugin

VDR attaches a newly launched player only from its main loop, and
`cTransfer::Receive()` discards the new channel's TS until then. That pass can
come late: after a key zap the loop first draws the channel banner (a skin's
signal bars block in `FE_GET_PROPERTY`); after an SVDRP or plugin switch it
sleeps in `cRemote::Get(1000)`.

`cVaapiSwitchAttacher` (device.cpp) therefore attaches the transfer player from
`cStatus::ChannelSwitch()`, which VDR fires inside `SetChannel()` right after
`cControl::Launch()`; the main loop's attach becomes a no-op. This also closes a
race: on a primary device without a tuner `HasProgramme()` is true only once a
player is attached, so the main loop could re-switch a channel caught in that
gap. It gains time only where the first packet arrives before the banner is
drawn (SVDRP, fast sources); a slow tuner or CAM already hid the gap.

## End of stream

A replay ends when the pipeline has played out, not when the last packet was
accepted: decode queue (~4 s), decode-ahead reserve (~1.3 s), prerender slots,
the frame awaiting its flip and the ALSA ring (~800 ms) hold seconds of
material. Both players drain through the same pair — `RequestEosDrain()` starts
it, `PendingPlayoutDepth()` counts what is left — and stop at depth 0:

- **VDR replay** (API ≥ 30014): `cDvbPlayer` polls `cDevice::Drain()` every
  few ms once its file is exhausted. `Drain()` first delivers the PES VDR holds
  back (a video PES is complete only at the next payload start, which never
  comes at EOF), then calls `DrainDevice()`. That requests the drain once — a
  repeat would re-arm the codec drain and pin the depth above 0. `Clear()` and
  `SetPlayMode()` cancel it; a PES accepted afterwards re-arms it, because a
  growing recording can hit eof and resume without a `Clear()`.
- **Mediaplayer**: `DrainTailAtEof()` waits on the same depth at real-time pace
  and gives up on a user command, a stall, or a hard cap.

The drain releases the AU the video parser withholds until the next start
code, sends the codec its NULL packet once the queue is empty and flushes the
chain (a temporal deinterlacer holds the last frame). Audio flushes parser,
codec and resampler the same way and `snd_pcm_start()`s a ring still below its
start threshold, since no further write will push it over. The codec drain
counts as pending until its frames reach the reserve, a popped display frame
until its flip lands, so no poll reads 0 with work in flight.

Audio usually ends first (mux interleave), so its clock keeps extrapolating
past the last sample instead of going stale: the video tail stays paced. A
pause in the tail defers the audio drain until the pause ends — started in the
pause, the audio tail would unpin the clock and the video tail would read as
late. `Play()` lifts the pin itself, as no write follows that could; a trick
entered from the pause ends it through `DropOutput()`, or slow motion to the
end would never report the drain done.

Trace lines (see [Tracing](#tracing)): `EOS drain requested`, `codec drain`,
`audio: EOS drain`, and the cancel / re-arm lines. Always logged, once per end:
`EOS drained -- last frame on screen Nms after the request`, or in the
mediaplayer `EOS tail drained after Nms` / why it was abandoned.

## Tracing

The A/V-sync narration and the stream-start milestones are opt-in. They go
through `tsyslog()` (`src/common.h`), one gate on top of VDR's `dsyslog`, so
**both** switches have to be on:

- the plugin's `-t` / `--trace` startup option, or `svdrpsend PLUG vaapivideo
  TRACE on` at runtime (`TRACE off` turns it back off, `TRACE` with no argument
  reports the current state), **and**
- VDR's `-l 3` log level.

Gated: the periodic `sync d=…` line, every per-event correction line named
below, the whole [stream-start trace](#stream-start-trace), and the
[end-of-stream](#end-of-stream) narration. Not gated: `catch-up cycling
sustained` / `settled`, the `jitterBuf` / handoff overflow lines, the EOS
outcome (`EOS drained` / `EOS tail …`), and every warning and error — a real
fault still reports itself at the ordinary log levels, with no extra switch.

The gate exists because the correction lines are per-frame: through a sustained
mismatch the drop/skip paths would write ~50 lines a second from the
presentation thread, and that syslog I/O is itself a pacing hazard on the thread
that has to hit VSync. With tracing off the arguments are never evaluated.

## Diagnostic log

```
sync d=+15.2ms avg=+15.1ms av=+812ms lat=20ms buf=40 aq=0 miss=0 drop=0 skip=0
```

| Field  | Meaning |
| ------ | ------- |
| `d`    | Interval mean of `rawDelta` since the last log; comparable to `avg` |
| `avg`  | EMA-smoothed delta; drives every soft-correction decision |
| `av`   | The multiplex's A/V interleave, sampled at audio ingress — see [Stream-start trace](#stream-start-trace). A feed property, so it holds still while the other fields move; it changes only when the stream does |
| `lat`  | Active `SyncLatency90k` (1-frame tail + active operator knob) |
| `buf`  | `jitterBuf` depth in frames at log emission |
| `aq`   | Audio packet queue depth |
| `miss` | Drain gaps > 2 × output frame period since last log (upstream starvation; deliberate sync sleeps and trick-play holds are excluded via `sleptInLastSubmit`, and the first 3 s after a flush are excluded as transition cost — filter rebuild, mode-switch HDMI retrain, audio re-anchor) |
| `drop` | Frames dropped (video behind) since last log — soft-behind, hard-behind, catch-up, stale-jitter, and pending-drop bursts combined |
| `skip` | Render delays (video ahead) since last log — soft sleep + hard-ahead combined |

`d ≈ avg` in steady state means the EMA has converged on current reality. The line
is suppressed during warmup and reissued immediately on warmup completion.
Emission is event-driven: a 2 s timer only *evaluates*; a line is emitted when
a counter ticked (`miss`/`drop`/`skip`), when `avg` drifted ≥ 1 ms from the last
emitted line, on a forced request (warmup completion, `Clear()`, trick
transitions), or on the 30 s heartbeat (matching the systemd watchdog
cadence) — a stable
stream logs two lines a minute instead of thirty. Skipped evaluations keep
accumulating, so `d` still means "mean since
the last *emitted* line". The `sync freerun (no clock)` line follows the same
heartbeat (freerun is a state, not an event — a video-only source would
otherwise repeat it every 2 s).

**Healthy steady state:** `avg` inside `±CORRIDOR`, `d ≈ avg`,
`miss = drop = skip = 0`. `buf` depth varies by mode (live: ALSA-cushion-driven;
replay cold start: ~40–60; replay post-`Clear()`: ~0; mediaplayer: near the
reserve cap). `aq` is 0 on the live/PES path and pegged at the mediaplayer
highwater (32) during file replay — see
[Audio packet queue](#audio-packet-queue-aq).

Both this line and the per-event lines below require [tracing](#tracing).

Each soft / hard event also emits a per-event `tsyslog` line naming the cause
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
value — i.e. positive latency pulls video earlier vs. audio (it delays audio
relative to video, per `config.h`).

## Stream-start trace

With [tracing](#tracing) on, every stream start
(`SetPlayMode(pmAudioVideo / pmVideoOnly / pmAudioOnly*)`: channel switch,
replay start, mediaplayer open) arms a one-shot trace in the
device, decoder, audio processor, and display (`StreamStartTrace` in
`src/common.h`, lock-free; `pmNone` disarms it). Each component reports its
first milestones once, as `+N ms` after the common switch epoch, so one journal
excerpt shows where the latency went:

```
device:  trace +1682ms first video PES (pts=…, 1746 bytes)
device:  trace +1868ms first audio PES (pts=…, 3840 bytes)
device:  trace +2012ms audio codec mp2 opened (PCM, chained 1-PES confirm), first decodable pts=…
audio:   trace +2012ms first ALSA write -- queued from pts=… (1152 frames, delay=1193)
audio:   trace +2190ms DAC running -- clock anchored, audible from pts=… (ring=302ms)
device:  trace +2352ms first keyframe PES (h264, pts=…, video PES #40)
device:  trace +2353ms video codec h264 opened (profile=100, 8-bit), first fed pts=…
decoder: trace +2357ms first decoded frame pts=… (1280x720 type=I key, after 1 packet(s), filter to build)
decoder: trace +2399ms first frame presented (freerun) pts=… clock=… -- video +748ms vs audio: still-frame hold …
display: trace +2411ms first frame committed to CRTC (1920x1080)
decoder: trace +3170ms A/V locked pts=… av=+812ms lat=20ms raw=+17ms buf=31 vbuf=620ms abuf=302ms
```

Reading it: `first video PES` is when the tuner / stream delivers; the gap to
`first keyframe PES` is the GOP position (`video PES #N` = how many pictures
were skipped); keyframe → `committed to CRTC` is the plugin's own startup cost;
`DAC running` is when sound becomes audible and the audio clock anchors (the
ALSA start threshold — nothing is paced before it); the
`video +Nms vs audio` figure on the first presented frame predicts the
still-frame hold, and `A/V locked` is when motion starts in sync (see
[Stream start](#stream-start)). `after N packet(s)` on the first decoded frame
exposes the decoder's reorder delay; `filter to build` means the VPP graph was
built for this frame (the `filter initialized` line follows).

`A/V locked` marks the first frame the audio-clock gate released — everything
before it was freerun:

| Field | Meaning |
| ----- | ------- |
| `av`  | The **multiplex's** A/V interleave: when an audio AU arrives, how far ahead the video feed already is (+ = video ahead). Sampled at audio ingress against the video feed's position at that same instant, so no queue depth, jitter buffer, ALSA tail, or latency knob enters it. Several hundred ms is normal on DVB — it is the encoder's video buffer delay, not a fault, and nothing the plugin can shrink |
| `lat` | Compensation in force: the operator knob (`PcmLatency` / `PassthroughLatency`, whichever path is active) + the one-frame pipeline tail |
| `raw` | Residual the sync gate acts on — `videoPTS − audioClock − lat`, the same `raw` the correction lines and the periodic [sync line](#diagnostic-log) report |

`av` describes the *stream*, `lat` and `raw` the *pipeline*. The buffers are the
answer to `av`, never part of it: `av` is what the jitter buffer has to absorb,
which is why `vbuf` grows to roughly that size while the pipeline waits for audio
to catch up. `raw` near zero with a large `av` is the healthy case — the
interleave was absorbed. A large `lat` holding a small `raw` means the operator
knob is carrying the stream.

The sample is taken in `cVaapiDecoder::NoteAudioPts()`, called by the device from
both audio feed paths (`PlayAudio()` for PES, `SubmitAudioPacket()` for the
mediaplayer) against the video feed position recorded in `EnqueueData()` /
`EnqueuePacket()`. Every one of those points is upstream of the codecs, so the
figure is two ingress positions and nothing else. `Clear()` resets it, so a
pre-switch interleave can never be reported for the new stream.

`A/V locked` also reports the cushion on both sides at that moment: `vbuf` is the jitter
buffer (`buf` frames × the output frame duration) and `abuf` the unplayed ALSA
tail (end-of-queued PTS minus the DAC clock, so it covers PCM and passthrough
alike). Both are the margin the pipeline has before the next hiccup shows on
screen — a lock that arrives with `vbuf` near one frame or `abuf` near zero is a
lock that is about to underrun. `abuf` reads 0 while the clock has not anchored
yet or after a mute / trick-mode drop emptied the ring.

## Constants

The values below define what the pipeline does observably: latency, buffer
depth, and when the sync controller acts. Everything else (log cadences, poll
slices, retry budgets, watchdogs) is an implementation detail documented at
its definition. Shared constants are `inline constexpr` in a header,
single-user ones `constexpr` in the consuming `.cpp`'s anonymous namespace;
each carries a `///<` comment with purpose and unit.

Naming:

- **Prefix** — the module that defines it: `AUDIO_`, `DECODER_`, `DEVICE_`,
  `DISPLAY_`, `MEDIAPLAYER_`, `SUBTITLE_`, `FILTER_`, `STREAM_`, `CONFIG_`.
  Values fixed by a specification keep its namespace (`EDID_`, `CEA_`, `ELD_`,
  `HDMI_`, `PES_`); values mirrored from VDR carry `VDR_` and VDR's own name
  (`VDR_SPEED_MULT` is dvbplayer.c's `SPEED_MULT`).
- **Unit suffix** — `_MS`, `_S`, `_90K` (90 kHz PTS ticks, like the `…90k`
  variables), `_HZ`, `_PPM`, `_BYTES`, `_FRAMES`, `_PACKETS`, `_SAMPLES`,
  `_VSYNCS`. Sizes of a queue or buffer end in `_CAPACITY`, a feed gate in
  `_HIGHWATER`, a retry / iteration budget in `_LIMIT`. `PTS_TICKS_PER_MS`
  is the lone exception: there `_MS` means *per* millisecond.
- **Qualifiers** — `_MIN` / `_MAX` / `_DEFAULT` come right before the unit
  (`DECODER_SYNC_CORRECTION_MAX_MS`, `MEDIAPLAYER_LOOKAHEAD_MAX_90K`).

**Clock & audio** (config.h, audio.cpp)

| Constant                      | Value | Purpose |
| ----------------------------- | ----- | ------- |
| `PTS_TICKS_PER_MS`            | 90    | DVB 90 kHz PTS clock: ticks = ms × this |
| `AUDIO_ALSA_BUFFER_MS`        | 800   | ALSA ring size — an upper bound, not the running level |
| `AUDIO_ALSA_START_MS`         | 300   | Ring fill the DAC starts at, hence the cushion the stream keeps for good (the feed is 1x); must clear one audio PES period (up to 192 ms) plus jitter — see [Ring cushion](#ring-cushion) |
| `AUDIO_CLOCK_STALE_MS`        | 1000  | Age after which `GetClock()` stops extrapolating and returns NOPTS |
| `CONFIG_AUDIO_LATENCY_MIN_MS` | −200  | Lower bound of the `PcmLatency` / `PassthroughLatency` setup knobs |
| `CONFIG_AUDIO_LATENCY_MAX_MS` | 200   | Upper bound of the same knobs |

**Buffers** (audio.h, decoder.h, decoder.cpp, display.cpp, mediaplayer.cpp)

| Constant                            | Value  | Purpose |
| ----------------------------------- | ------ | ------- |
| `AUDIO_QUEUE_HIGHWATER`             | 10     | Audio packets queued before the dvbplayer / PES feed is held back (~320 ms AC-3) |
| `AUDIO_QUEUE_HIGHWATER_MEDIAPLAYER` | 32     | The same gate for the mediaplayer (~1 s AC-3): one demux cursor feeds audio and video, so the audio side needs more slack |
| `DECODER_QUEUE_CAPACITY`            | 200    | Compressed video packets queued ahead of the decoder (~4 s @ 50 fps) |
| `DECODER_RESERVE_CAPACITY`          | 64     | Decoded frames held ahead of presentation, handoff queue and jitter buffer together (~1.3 s @ 50 fps); also bounds 4K surface memory |
| `DISPLAY_PRERENDER_CAPACITY`        | 8      | Frames queued for scanout (160 ms @ 50 fps); absorbs a UHD VPP or bandwidth spike |
| `MEDIAPLAYER_LOOKAHEAD_MAX_90K`     | 135000 | Audio lead (1.5 s) the mediaplayer demuxes ahead of the audio clock before it throttles to real time |

**Sync controller** (decoder.cpp) — see [Correction regimes](#correction-regimes)

| Constant                          | Value | Purpose |
| --------------------------------- | ----- | ------- |
| `DECODER_SYNC_CORRIDOR_90K`       | 4500  | Soft corridor half-width (50 ms), below the lip-sync perception threshold |
| `DECODER_SYNC_HARD_THRESHOLD_90K` | 18000 | Hard-transient threshold (200 ms); twice this enters catch-up |
| `DECODER_SYNC_CORRECTION_MAX_MS`  | 200   | Cap on one soft correction, derived from the hard threshold so one event can close the corridor |
| `DECODER_SYNC_COOLDOWN_MS`        | 5000  | Minimum interval between soft corrections (5 EMA time constants) |
| `DECODER_SYNC_EMA_SAMPLES`        | 50    | EMA divisor (~1 s @ 50 fps) |
| `DECODER_SYNC_WARMUP_SAMPLES`     | 50    | Samples averaged to seed the EMA (~1 s @ 50 fps) |
| `DECODER_SYNC_HARD_AHEAD_MAX_MS`  | 500   | Longest sleep live TV takes when video is far ahead |
| `DECODER_SYNC_FREERUN_FRAMES`     | 1     | Frames shown unpaced after a sync-disrupting event |
| `DECODER_DRAIN_FUTURE_MAX_MS`     | 3000  | A head frame further ahead of the clock is a PTS discontinuity and dropped; nearer ones hold until due |
| `DECODER_NO_CLOCK_HOLD_MS`        | 1500  | How long video waits for the audio clock before it starts without one |

**Trick play.** Both slow directions run at **1/speed of content time**: each
step is held for the PTS distance it covers × the slowdown (2/4/8), whatever
one step carries — a 20 ms field, a 40 ms frame, an ~85 ms audio step, or a
GOP of keyframe stride in reverse. A step without two PTS counts one output
frame forward, or VDR's nominal 0.4 s stride in reverse. Slow reverse
therefore shows one picture per ~0.8 / 1.6 / 3.2 s at /2 / /4 / /8; a speed
change, `Play()` or `Clear()` cuts a running hold short.
