# Speculative inference

The brain's `think()` call is the largest per-step latency component on
warm requests (~1.5-2.0 s for Holo3 Q8). It was previously fully serial
with the post-action settle:

```
dispatch action → wait for settle → screenshot → think() → next action
```

Speculative inference overlaps `think()` with the settle window so the
brain's reasoning starts *before* the next frame is captured. When the
post-settle frame matches the frame the speculation started with, the
speculative result is consumed directly — saving a synchronous brain
round-trip.

Tracking issue: [#118](https://github.com/mercurialsolo/mantis/issues/118).

## How it works

`SpeculativeBrain` wraps the inner `Brain`. Each `think()` call:

1. If a pending speculation from the previous call exists AND the new
   `frames[-1]` matches the frame the speculation started with (per
   `phash_64` Hamming distance ≤ tolerance), consume the speculation —
   skip the synchronous round-trip entirely.
2. Otherwise, fall through to a synchronous `inner.think()`.
3. Either way, kick off a *new* speculation against the new frames for
   the *next* call to consume.

The validator defaults to `frames_close_enough(..., max_hamming_distance=0)`
— only pixel-equivalent frames pass. **Speculative results never drive
an action when the screen visibly changed.** Tighter than necessary on
SPAs with subtle animations; tunable per-deployment.

## Quality guard

The safety contract:

- **Strict validator**: with `max_hamming_distance=0`, a single bit of
  perceptual difference invalidates the speculation. Falls through to
  the synchronous path on every miss.
- **Synchronous fallback on exception**: any exception in the speculative
  `think()` aborts the speculation; the runner calls `inner.think()`
  fresh.
- **Cancel on invalidate**: the worker is freed as soon as the runner
  decides the speculation is stale.
- **Counter parity**: `hits + misses + synchronous_starts` always equals
  the number of `think()` calls in the episode. Hit rate is observable.

## Lifecycle

```
request 1:
  step 1: no pending → synchronous_start (synchronous think); kick off speculation
  step 2: pending validates → HIT (free think); kick off speculation
  step 3: pending invalidates (page changed) → MISS, synchronous fallback; kick off
  ...
  done() → runner returns
```

The runtime calls `brain.reset()` at the start of each `/v1/cua` run so
per-run counters surface cleanly. Cross-run state never leaks.

## API response signal

`/v1/cua` responses include a `speculation_summary` block:

```json
{
  "speculation_summary": {
    "hits": 4,
    "misses": 1,
    "synchronous_starts": 1,
    "hit_rate": 0.6667,
    "enabled": true
  }
}
```

Every run doubles as an ablation data point.

## Ablation toggle

Per [#261](https://github.com/mercurialsolo/mantis/issues/261) discipline:

```bash
MANTIS_SPECULATIVE_INFERENCE=disabled
```

When disabled, `runtime.brain` stays bare (no wrapper); the runner runs
in serial `think()` mode. `speculation_summary.enabled` reflects the
toggle state in the API response.

## Expected wall-time impact

Per-step decomposition on a warm container (post-#311 session reuse and
#294 adaptive settle):

| Phase | Pre-#118 | Post-#118 (cache hit) |
|---|---|---|
| Dispatch action | <0.1 s | unchanged |
| Settle (adaptive) | 0.3-0.7 s | overlapped with think() |
| `think()` inference | 1.5-2.0 s | **0** (already done) |
| Frame capture + framing | <0.5 s | unchanged |
| **Per-step total** | ~2.5 s | **~0.8-1.2 s** |

Projected ~50% per-step reduction on hit-heavy workloads. Click + scroll
flows have the highest hit rates (visual stability); navigate / form-
submit flows trip the validator more often and fall back to synchronous.

## What this does NOT do

- Doesn't speed up the FIRST think() call in a run (no pending speculation).
- Doesn't help when every action visibly changes the page (validator
  always misses; behaviour identical to synchronous).
- Doesn't widen the brain's parallelism — only one speculation is
  in-flight at a time, matching the serial step pattern.

## See also

- [Adaptive settle](adaptive-settle.md) — the other half of the warm-path
  speedup. Speculative inference benefits MORE when settle is short
  (the overlap window shrinks but so does the wasted time).
- [Chrome session reuse](chrome-session-reuse.md) — eliminates the
  cold-launch cost. Combined with #118, the warm-request wall is
  dominated by inference + dispatch.
