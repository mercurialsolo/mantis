# Rate limits

Two dimensions, both per-tenant, both in-process per replica.

| Dimension | Source | Default | On exceed |
|---|---|---|---|
| **Concurrent runs** | `tenant.max_concurrent_runs` | 5 | `429` with `Retry-After: 5` |
| **Rate** (token bucket) | `tenant.rate_limit_per_minute` | 30 | `429` with `Retry-After: <s-until-token>` |

## How the rate dimension works

A standard token bucket: each tenant has a bucket with `rate_limit_per_minute` capacity, refilled at `rate_limit_per_minute / 60` tokens/sec. Every `POST /v1/predict` (run mode only — polling actions don't consume tokens) takes one token. When the bucket is empty, the request gets `429` with `Retry-After` set to the time until the next token will be available.

Set `rate_limit_per_minute: 0` in the tenant config to disable the rate limit entirely (useful for trusted internal tenants).

## How the concurrency dimension works

A simple counter per tenant. Each accepted run increments it; each finished/failed/cancelled run decrements it. When the counter hits `max_concurrent_runs`, new requests get `429`. The Prometheus gauge `mantis_concurrent_runs{tenant_id=...}` exposes the live count.

## Tuning

| Symptom | Fix |
|---|---|
| Tenant reports frequent 429s on bursts | Raise `rate_limit_per_minute` |
| Tenant reports 429s on long-running parallel jobs | Raise `max_concurrent_runs` |
| GPU keeps OOMing because too many concurrent runs | Lower the global cap by setting a lower `max_concurrent_runs` for the busy tenants |
| Want strict cluster-wide caps across N replicas | Today: run `replicas: 1` and accept the SPOF. Future: Redis-backed limiter (Tier 2.5) |

## Per-replica vs cluster-wide

The current limiter is in-process per replica. With N replicas, the effective per-tenant cap is roughly `N × configured_cap` because each replica tracks its own counter.

For most workloads this is fine — Mantis runs are detached and stick to one replica, so traffic spreads naturally. For strict cluster-wide enforcement (e.g., regulatory caps), either:

1. Run a single replica (already the default for autoscale-aware deployments).
2. Wait for the planned Tier 2.5 Redis-backed limiter.
3. Roll your own — `src/mantis_agent/rate_limit.py` is small and the `TenantRateLimiter` interface is intentionally swap-friendly.

## Observability

```
mantis_predict_requests_total{tenant_id="acme", mode="run", outcome="rate_limited"}  ← rate-limit hits
mantis_rate_limit_rejections_total{tenant_id="acme", kind="rate"}                    ← bucket empty
mantis_rate_limit_rejections_total{tenant_id="acme", kind="concurrent"}              ← concurrency cap
mantis_concurrent_runs{tenant_id="acme"}                                             ← live in-flight
```

Alert on `rate(mantis_rate_limit_rejections_total[5m]) > 0.1 / s` per tenant for chronic limit hitting.

## See also

- [Tenant keys](tenant-keys.md) — where to set `rate_limit_per_minute` / `max_concurrent_runs`
- [Metrics](metrics.md) — full label set + scrape setup
