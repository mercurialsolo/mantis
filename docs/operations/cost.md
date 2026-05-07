# Cost model (#122)

Per-resource pricing for the CUA runtime lives in
`src/mantis_agent/cost_config.py` as a `CostConfig` dataclass. Defaults
match the historical hardcoded rates so existing deployments see no
change without an env-var override.

## Overridable knobs

All env vars are interpreted as floats. Bad values fail fast (operators
see the error instead of silent fallback).

| Env var | Default | Purpose |
|---|---|---|
| `MANTIS_COST_GPU_HOURLY_USD` | `3.25` | GPU compute, $/hour. Multiply by `gpu_seconds / 3600` for the run's GPU bill. |
| `MANTIS_COST_CLAUDE_CALL_USD` | `0.003` | Per-Claude-API-call rate. Applied to `claude_extract` + `claude_grounding` counters. |
| `MANTIS_COST_PROXY_PER_GB_USD` | `5.00` | Egress proxy bandwidth, $/GB. Multiply by `proxy_mb / 1024`. |
| `MANTIS_COST_GPU_SECONDS_PER_STEP` | `3.0` | Per-step GPU budget used when the runner doesn't measure exact seconds. |
| `MANTIS_COST_PROXY_MB_PER_NAV` | `5.0` | Estimated proxy MB consumed by one page load. |
| `MANTIS_COST_PROXY_MB_PER_SCROLL` | `0.5` | Estimated proxy MB per scroll. |

Set these once at deploy time (Modal secret, k8s ConfigMap,
`docker run -e ...`, Baseten Truss `environment_variables`). Per-tenant
overrides happen at the `MicroPlanRunner` constructor (`cost_config=`).

## Wiring

```
CostConfig.from_env()  →  CostMeter(cost_config=...)  →  MicroPlanRunner
                                       │
                                       └─ runs cost_meter.totals(),
                                          emits inflight + final
                                          gauges/histograms on Prometheus
```

The `CostMeter` rolls up per-resource counters from the runner's `costs`
dict (`gpu_steps`, `gpu_seconds`, `claude_extract`, `claude_grounding`,
`proxy_mb`) and produces the four-tuple `(gpu, claude, proxy, total)`
in USD.

## Prometheus

Two Prometheus surfaces ship with the cost model — see
[Metrics](metrics.md) for the full table.

| Metric | Type | Labels |
|---|---|---|
| `mantis_run_cost_usd` | histogram | `tenant_id`, `model`, `status` |
| `mantis_run_cost_usd_inflight` | gauge | `tenant_id`, `component` (`gpu` / `claude` / `proxy` / `total`) |

The histogram captures terminal cost per detached run; the inflight
gauge updates on every progress log so live runs show up on dashboards
without waiting for terminal observation.

## Tuning the rates

When upstream prices change (Modal A100 hourly, Anthropic per-token,
proxy provider) operators can:

1. Update the env var on the deployment (no code change).
2. Confirm the new rate is live by hitting `/metrics` and reading
   `mantis_run_cost_usd_inflight` — the next progress emission shows
   the override.
3. Tenant-specific rates are out of scope for this surface — pass
   a custom `CostConfig` instance to `MicroPlanRunner(cost_config=...)`
   inside the per-tenant request handler.

## Useful queries

```promql
# Per-tenant cost burn rate ($/hour averaged over the last hour)
sum by (tenant_id) (
  increase(mantis_run_cost_usd_sum[1h])
)

# Cost split by component, last 24h
sum by (component) (
  rate(mantis_run_cost_usd_inflight[24h])
)

# p95 cost per detached run (catches bug-runs blowing past the cap)
histogram_quantile(0.95,
  sum by (tenant_id, le) (
    rate(mantis_run_cost_usd_bucket[1h])
  )
)
```

## Tests

`tests/test_cost_config.py` covers env-override parsing + the three
cost computation helpers. `tests/test_cost_meter.py` covers the
roll-up + Prometheus emission surface.
