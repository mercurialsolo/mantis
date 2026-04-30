# Runs and polling

After submitting a detached run you get a `run_id`. Everything else is `POST /v1/predict` with an `action` field.

## Actions

```jsonc
{ "action": "status", "run_id": "20260428_…" }
{ "action": "result", "run_id": "..." }
{ "action": "logs",   "run_id": "...", "tail": 200 }
{ "action": "cancel", "run_id": "..." }
```

| Action | Returns | Notes |
|---|---|---|
| `status` | Current state + `summary` block when terminal | Cheapest call; use for polling |
| `result` | Full result JSON (leads, per-step trace, costs) | Only meaningful after terminal |
| `logs` | Last `tail` events written by the runner | Useful for debugging stuck runs; max `tail` is 10000 |
| `cancel` | Marks the run cancelled; runner halts at the next checkpoint boundary | Cooperative — not instant |

## Status lifecycle

```
queued ──► running ──► succeeded
                  ├──► failed
                  └──► cancelled
```

| Status | Meaning |
|---|---|
| `queued` | Server accepted the request; not yet on a worker |
| `running` | Runner is working on it |
| `succeeded` | Run completed; `summary` populated |
| `failed` | Run threw an unhandled error; `error` populated |
| `cancelled` | A `cancel` action was processed |

## Polling pattern

```bash
RUN_ID="..."
while true; do
  STATE=$(curl -fsS -X POST "$ENDPOINT/v1/predict" \
    -H "Authorization: Api-Key $BASETEN_API_KEY" \
    -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"action\":\"status\",\"run_id\":\"$RUN_ID\"}" | jq -r .status)
  echo "$(date '+%H:%M:%S')  $STATE"
  case "$STATE" in succeeded|failed|cancelled) break ;; esac
  sleep 30
done
```

Recommended polling cadence:

| Phase | Interval |
|---|---|
| First minute | every 5–10 s (catch quick failures fast) |
| Next 5 min | every 30 s |
| Beyond 5 min | every 60 s |

Each `status` call hits the rate limiter; per-tenant `rate_limit_per_minute` defaults to 30, so a 1/min poll is fine.

## Status response

```jsonc
{
  "status": "succeeded",
  "run_id": "20260428_021432_076255ef",
  "started_at":  "2026-04-28T02:14:32Z",
  "finished_at": "2026-04-28T02:24:01Z",
  "summary": {
    "total_time_s": 569,
    "steps_executed": 17,
    "viable": 3,
    "leads_with_phone": 1,
    "cost_total": 0.42,
    "cost_breakdown": {
      "gpu":    0.12,
      "claude": 0.12,
      "proxy":  0.18
    },
    "dynamic_verification_summary": { ... }
  }
}
```

The `summary.dynamic_verification_summary` block contains the per-page health checks the runner emits for extraction workflows (`page_1_required_filters_present: pass`, etc.).

## Result response

```jsonc
{
  "result": {
    "leads": [
      "VIABLE | Year: <YYYY> | Make: <Make> | Model: <Model> | Price: <Price> | Phone: <Phone or 'none'>",
      "VIABLE | Year: <YYYY> | Make: <Make> | Model: <Model> | ...",
      "VIABLE | Year: <YYYY> | Make: <Make> | Model: <Model> | ..."
    ],
    "steps": [
      { "intent": "Navigate to ...", "type": "navigate", "success": true, "duration": 1.2, ... },
      ...
    ],
    "video": {
      "path":          "/.../recording.mp4",
      "polished_path": "/.../recording_polished.mp4",
      "actions": { "clicks": 17, "keys": 3, "types": 2, "scrolls": 8, "drags": 0 },
      "duration_seconds": 567.3
    },
    "summary": { ... same as status response ... }
  },
  ...
}
```

The lead format depends on the plan — for BoatTrader-style extractions it's the `VIABLE | …` strings above; for `task_suite` runs each task gets its own `result` block.

## Idempotency

Pass `Idempotency-Key: <unique-id>` on the original `POST /v1/predict` and retries with the same key return the cached `run_id` instead of starting a new run.

```bash
curl -X POST "$ENDPOINT/v1/predict" \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
  -H "Idempotency-Key: order-7afc3b91" \
  -H "Content-Type: application/json" \
  -d '{ ... }'
```

Cache TTL is 24 hours (configurable on the server). Useful when your client has a flaky network and you can't easily tell whether a POST landed.

See [Idempotency](../operations/idempotency.md) for the operator side.

## Webhooks (instead of polling)

If your tenant has `webhook_url` configured (or you pass `callback_url` per request), you don't need to poll — the server will POST a signed JSON to your URL when the run reaches a terminal state.

Body:

```jsonc
{
  "run_id": "20260428_021432_076255ef",
  "tenant_id": "tenant_a",
  "status": "succeeded",
  "summary": { ... same as status response ... },
  "delivered_at": "2026-04-28T02:24:01Z"
}
```

Verify the HMAC-SHA256 signature in `X-Mantis-Signature: sha256=<hex>`:

```python
import hmac, hashlib
def verify(body: bytes, header: str, secret: str) -> bool:
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header)
```

3 retries with exponential backoff (1 s, 5 s, 30 s) if your endpoint returns non-2xx. After that the server gives up — keep polling as a fallback. See [Webhooks](../operations/webhooks.md) for the operator side.

## Cancelling

```bash
curl -fsS -X POST "$ENDPOINT/v1/predict" \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"action\":\"cancel\",\"run_id\":\"$RUN_ID\"}"
```

Cancels are cooperative — the runner finishes the current step's checkpoint write before stopping. Expect 5–60 s before the status flips to `cancelled`.

## Resuming

If a run failed mid-way (network hiccup, replica restart, OOM), re-submit with the **same `state_key`** and `resume_state: true`:

```jsonc
{
  "detached": true,
  "micro": "plans/boattrader/...json",
  "state_key": "boattrader-prod",     // ← same key as the failed run
  "resume_state": true,
  "max_cost": 2
}
```

The runner picks up from the last checkpoint (extracted leads kept, browser profile + cookies still on the volume). You get a new `run_id` but the same logical workflow continues.

## See also

- [Recordings](recordings.md) — fetching the screencast
- [Errors](errors.md) — full error reference
- [Reference / HTTP API](../api.md) — endpoint detail
