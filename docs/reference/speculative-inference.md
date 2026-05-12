# Speculative inference

`SpeculativeBrain` wraps the inner `Brain` to overlap `think()` with the
post-action settle window. The infrastructure is shipped; **the wrapper
is opt-in via `MANTIS_SPECULATIVE_INFERENCE=enabled`** because the
real-world E2E ablation on the production Holo3 + llama.cpp deployment
showed a wall-time **regression**, not a win.

Tracking issue: [#118](https://github.com/mercurialsolo/mantis/issues/118).

## What changed

| Component | Before | After |
|---|---|---|
| `BasetenCUARuntime.load()` | bare brain | optional `SpeculativeBrain` wrapper |
| `MANTIS_SPECULATIVE_INFERENCE` | — | env var; default `disabled` |
| `/v1/cua` payload `"speculation"` | — | per-request override |
| `/v1/cua` response | — | `speculation_summary` block |

## How the wrapper works

Each `think()` call:

1. If a pending speculation from the previous call exists AND the new
   `frames[-1]` matches the frame the speculation started with (per
   `phash_64` Hamming distance ≤ tolerance), consume the speculative
   result — skip the synchronous round-trip.
2. Otherwise, fall through to a synchronous `inner.think()`.
3. Either way, kick off a *new* speculation against the new frames for
   the *next* call to consume.

The validator defaults to `frames_close_enough(..., max_hamming_distance=0)`
— only pixel-equivalent frames pass.

## Quality guarantee (why this is safe even though it's slower today)

The strict validator makes false acceptances impossible:

- **`max_hamming_distance=0`**: a single bit of perceptual difference
  invalidates the speculation. Falls through to the synchronous path.
- **Synchronous fallback on exception**: any speculative `think()`
  exception aborts; runner calls `inner.think()` fresh.
- **Cancel on invalidate**: the worker is freed as soon as the runner
  decides the speculation is stale.

It is **mathematically impossible** for a speculative result to drive
an action when the page visibly changed.

## E2E ablation (Modal, Holo3 Q8 on llama.cpp)

Identical lu.ma extract instruction (18 steps), single-deploy A/B via
the per-request `"speculation"` override:

| Run | Speculation | Steps | Wall | Hit rate |
|---|---|---|---|---|
| A | OFF | 18 (max_steps) | **93 s** | n/a |
| B | ON | 18 (max_steps) | **145 s** | **55.6%** (10 hits / 18 think) |

**Speculation is 52% slower despite a 55.6% hit rate.** No quality
regression (no done_rejections, no predicate anomalies, validator behaved
correctly), but the perf claim from the original issue doesn't hold on
this backend.

### Root cause

`SpeculativeBrain` runs `think()` on a worker thread; Holo3 routes both
the speculative AND the synchronous `think()` to the same llama.cpp
inference server (single GPU). The two HTTP requests **serialize** on
the GPU — the speculative call holds GPU time during the action
dispatch, then the sync fallback (on misses) waits for the GPU to free.

The wrapper helps when:

- The inner brain serves requests from **separate GPUs / processes**
  (multi-replica deployment, multi-tenant inference fleet).
- The inner brain is **CPU-bound but heavily I/O-bound** (e.g. a remote
  Anthropic API call where Python threads can overlap network I/O).
- The hit rate × inference cost > GPU-contention penalty.

It hurts when the brain backend has a single GPU shared across both
concurrent requests, like Holo3 Q8 on llama.cpp today.

## API response signal

`/v1/cua` responses include a `speculation_summary` block on every run:

```json
{
  "speculation_summary": {
    "hits": 10,
    "misses": 7,
    "synchronous_starts": 1,
    "hit_rate": 0.5556,
    "enabled": true
  }
}
```

Every run doubles as an ablation data point.

## Toggles

| Lever | Effect |
|---|---|
| `MANTIS_SPECULATIVE_INFERENCE=enabled` | container-wide opt-in; wraps `runtime.brain` |
| `MANTIS_SPECULATIVE_INFERENCE=disabled` (default) | bare brain, legacy serial path |
| `payload["speculation"]=false` | per-request opt-out even when env-var is on |

Per-request override lets a single deploy serve both arms of an A/B
without redeploy — useful for measuring whether the wrapper's wall-time
profile has improved on a backend change.

## When to enable

Only on backends where the brain inference server has enough parallelism
to serve two concurrent `think()` requests without serializing:

- Anthropic Claude API (cloud, virtually unlimited parallelism)
- vLLM with TP > 1 across multiple GPUs and a router that load-balances
- Multi-replica llama.cpp behind a load balancer

For Holo3 Q8 on a single llama.cpp container (current Modal production),
**keep it disabled**.

## See also

- [Adaptive settle](adaptive-settle.md) — the warm-path speedup that
  actually works today.
- [Chrome session reuse](chrome-session-reuse.md) — eliminates the
  cold-launch cost.
- [#309](https://github.com/mercurialsolo/mantis/issues/309) Holo3 Q5_K_M
  quantization — separate per-step inference win that doesn't depend on
  parallelism.
