# Idempotency keys

Caller passes `Idempotency-Key: <unique-id>` on `POST /v1/predict`; the server caches `(tenant_id, sha256(key)) → run_id` with a 24 h TTL. Subsequent retries with the same key return the cached `run_id` instead of starting a new run.

Useful when a caller's network is flaky and they can't tell whether a POST landed.

## Storage

File-backed: `$MANTIS_DATA_DIR/idempotency/<tenant_id>/<sha256-hex>.json`

```jsonc
{
  "run_id": "20260428_021432_076255ef",
  "response": { ... cached /v1/predict response ... },
  "stored_at": 1730068472.331
}
```

Per-process in-memory cache fronts the file lookups for speed; the sidecar means a replica restart preserves the entries.

## TTL

Default 24 h, fixed in code (`DEFAULT_TTL_SECONDS`). Expired entries are pruned **lazily** — they get deleted on next read attempt. There's no background sweeper, so a long-idle directory can accumulate stale files. Run a cron if that bothers you:

```bash
find $MANTIS_DATA_DIR/idempotency -name "*.json" -mtime +1 -delete
```

## Per-tenant isolation

The cache key includes `tenant_id`, so tenant A and tenant B can use the same idempotency key string without collision. They genuinely cannot read each other's cache entries — directories are tenant-prefixed.

## Multi-replica caveat

The on-disk sidecar only helps within one replica's view of the volume. With multiple replicas, the in-memory layer of replica A doesn't know about replica B's stores until the file gets re-read. For strict cluster-wide idempotency, swap to Redis (Tier 2.5 plan) — the interface in `src/mantis_agent/idempotency.py` is designed for it.

## Observability

```
mantis_predict_requests_total{tenant_id="acme", mode="run", outcome="idempotent_hit"}
```

A high `idempotent_hit` rate means callers are retrying often — investigate their network or your latency before raising rate limits.

## See also

- [Client / Runs and polling](../client/runs-and-polling.md#idempotency) — caller-side usage
- [Tenant keys](tenant-keys.md)
