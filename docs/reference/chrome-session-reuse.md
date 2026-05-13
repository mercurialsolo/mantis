# Chrome session reuse

Each `/v1/cua` request previously paid the ~10 s Xvfb + Chrome launch
tax even on a warm Modal container. The runtime now keeps the live
browser process around in a container-scoped cache so successive
requests reuse it.

Tracking issue: [#311](https://github.com/mercurialsolo/mantis/issues/311).

## What changed

| Component | Before | After |
|---|---|---|
| `XdotoolGymEnv.close()` | Always force-killed Xvfb + Chrome | No-op when `reuse_session=True` |
| `XdotoolGymEnv.shutdown()` | — | New method that force-kills regardless of `reuse_session` |
| `runtime._chrome_env_cache` | — | Container-scoped `dict[(profile_dir, proxy_key), (env, proxy_proc)]` |
| Per-request env creation | Always fresh launch | Cache hit reuses live browser; cache miss launches once |

## Cache key

`(profile_dir, proxy_key)` where:

- **profile_dir** is the tenant-scoped Chrome profile path
  (`data_root / "chrome-profile"`).
- **proxy_key** is `""` when `proxy_disabled=true`, otherwise
  `"<proxy_city>__<proxy_state>"`. Two requests with different proxy
  configs don't share a browser.

Cross-tenant isolation is preserved because the profile_dir is
tenant-scoped (`tenants/<tenant_id>/chrome-profile/<safe_profile_id>`,
keyed by `profile_id` since #341 — fallback to legacy `state_key` when
`profile_id` is not set) and tenant requests will mismatch on the first
key component.

## Lifecycle

```
request 1 arrives:
  cache MISS
    → launch Xvfb + Chrome (~10 s)
    → cache[(profile, proxy)] = (env, proxy_proc)
    → run task, env.close() is a no-op
request 2 arrives (same key):
  cache HIT
    → reuse env; reset(start_url=...) navigates in-tab (<500 ms)
    → run task, env.close() is a no-op
container recycles:
  → _shutdown_chrome_env_cache() force-closes every cached env
```

The proxy process is also kept alive across requests when the env is
cached — terminated only on the non-reuse path or container shutdown.

## Ablation toggle

Per [#261](https://github.com/mercurialsolo/mantis/issues/261) discipline:

```bash
MANTIS_CHROME_REUSE=disabled
```

When disabled the runtime falls back to the per-request launch path
(legacy behaviour). Per-request opt-out is available via
`payload["reuse_session"] = false` on `/v1/cua`.

## API response signal

`/v1/cua` responses now include a `reused_session` boolean. Each run
doubles as an ablation data point:

```json
{
  "success": true,
  "reused_session": true,
  "elapsed_seconds": 6.2
}
```

A request that warms the cache emits `false` (it launched the browser);
subsequent requests against the same key emit `true`.

## Expected wall-time impact

From the lu.ma click flow profiled in [#310](https://github.com/mercurialsolo/mantis/issues/310):

| Phase | Pre-#311 (every request) | Post-#311 cache hit |
|---|---|---|
| Xvfb + Chrome launch + first navigation | ~10 s | 0 (reused) |
| Step 1 inference | 1.5-2.0 s | unchanged |
| Settle (adaptive — #294) | 0.3-0.7 s | unchanged |
| **Total for a 2-step task** | ~17 s | **~6-7 s** |

The first request on a warm container still pays the full launch (it
seeds the cache). The win is on every subsequent request against the
same profile + proxy.

## What this does NOT do

- Doesn't share state across containers. Modal scales replicas
  independently — each container has its own cache.
- Doesn't share state across tenants. Profile dir is tenant-scoped.
- Doesn't pre-warm the browser at container boot. That's a separate
  follow-up ([#310](https://github.com/mercurialsolo/mantis/issues/310)).
- Doesn't dedupe concurrent requests. The cache is locked but the env
  inside isn't thread-safe; Mantis's `/v1/cua` worker model is serial
  per container.

## See also

- [Adaptive settle](adaptive-settle.md) — the other half of the warm-
  request speedup.
- [#310](https://github.com/mercurialsolo/mantis/issues/310) Modal
  `keep_warm` — eliminates the cold-start cost on the FIRST request too.
