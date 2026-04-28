# Operations

For operators running a multi-tenant Mantis fleet. If you're a single-tenant deployment (one shared `MANTIS_API_TOKEN`), most of this section is optional — the features below are toggled on by mounting a tenant keys file and setting the right env vars.

| Topic | Tier | What it does |
|---|---|---|
| [Tenant keys](tenant-keys.md) | Tier 1 | Per-tenant tokens, scopes, caps, Anthropic key routing, allowed domains |
| [Rate limits](rate-limits.md) | Tier 2 | Per-tenant token bucket + concurrent-run gauge |
| [Idempotency](idempotency.md) | Tier 2 | `Idempotency-Key` header → cached `run_id` (24 h) |
| [Webhooks](webhooks.md) | Tier 2 | HMAC-signed POST to caller URL on terminal status |
| [URL allowlist](allowlist.md) | Tier 2 | Reject plans whose `navigate` URLs are off the tenant's domain list |
| [Metrics](metrics.md) | Tier 2 | Prometheus counters / gauges / histograms labeled by `tenant_id` |

## Toggle map

```
single tenant ──┬── set MANTIS_API_TOKEN env var
                └── (everything works; one shared token)

multi-tenant ──┬── mount JSON keys file at $MANTIS_TENANT_KEYS_PATH
               ├── per-tenant config: scopes, caps, anthropic_secret_name, allowed_domains, webhook_url
               └── all of Tier 1 + Tier 2 features apply
```

Without `MANTIS_TENANT_KEYS_PATH` set, the server runs in single-tenant mode using `MANTIS_API_TOKEN`. The Tier-2 features (rate limits, idempotency, webhooks, allowlist, metrics) all still work in single-tenant — they just apply to a single `default` tenant.

## What every operator needs to know

1. **Hot reload.** The keys file has a 5 s read cache. Update the file → 5 s later the new keys work / old keys are rejected. No pod restart.
2. **Per-replica state.** Rate limits + concurrency are in-process per replica. With N replicas, the effective per-tenant cap is roughly N × the configured cap. For strict cluster-wide limits, run a single replica or swap to a Redis-backed limiter (planned Tier 2.5).
3. **Per-tenant data isolation.** `state_key` is server-prefixed with `tenant_id`. Browser profiles and run state live under `$MANTIS_DATA_DIR/tenants/<tenant_id>/`. Tenants cannot read each other's checkpoints, profiles, or recordings.
4. **Hard caps come from env vars.** `MANTIS_MAX_STEPS_PER_PLAN`, `MANTIS_MAX_LOOP_ITERATIONS`, `MANTIS_MAX_RUNTIME_MINUTES`, `MANTIS_MAX_COST_USD` cap **above** every tenant cap. Tighten globally by lowering these.

## See also

- [Reference / Environment variables](../reference/env-vars.md) — full list of env vars the server reads
- [Hosting](../hosting/index.md) — platform-specific deploy paths
