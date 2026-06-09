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

The lead format depends on the plan — for marketplace-listings extractions it's the `VIABLE | …` strings above; for `task_suite` runs each task gets its own `result` block.

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

If a run failed mid-way (network hiccup, replica restart, OOM), re-submit with the **same `workflow_id`** and `resume_state: true`. Keep the same `profile_id` too so the resumed run reuses the logged-in Chrome session from the failed attempt (#341):

```jsonc
{
  "detached": true,
  "micro": "plans/example/...json",
  "profile_id":  "marketplace-prod",       // ← same profile (cookies / sessions)
  "workflow_id": "marketplace-listings-v1", // ← same workflow (checkpoint to resume from)
  "resume_state": true,
  "max_cost": 2
}
```

The runner picks up from the last checkpoint (extracted leads kept, browser profile + cookies still on the volume). You get a new `run_id` but the same logical workflow continues.

## Paused runs

> OTP / 2FA / human-in-the-loop confirmation — [#344](https://github.com/mercurialsolo/mantis/issues/344).

A run can pause mid-flight — for OTP / 2FA / explicit human confirmation — without failing. The default `request_user_input` host tool is wired into every `/v1/predict` run: if the brain emits `Action(TOOL_CALL, name="request_user_input", params={"prompt": "..."})`, the runner snapshots state and the run's status flips to `paused`.

```python
from mantis_agent.client import MantisClient

client = MantisClient.from_env()
handle = client.predict({"micro": "plans/login-flow.json", "profile_id": "alice"})

def on_status(s):
    if s.status == "paused":
        code = input(s.prompt + " ")    # surface to the human
        client.resume(handle.run_id, user_input=code)

# wait_for_completion polls past `paused` (it's NON-terminal); the
# on_status callback above handles the prompt + calls resume() and
# the loop continues to terminal succeeded / failed / cancelled.
final = client.wait_for_completion(handle.run_id, on_status=on_status)
```

Key points:

* `paused` is **non-terminal**. `wait_for_completion` keeps polling — it only stops on `succeeded` / `failed` / `cancelled` (the `TERMINAL_STATUSES` set is unchanged).
* `RunStatus.prompt`, `RunStatus.reason`, and `RunStatus.pause_state` are populated when status is `paused`. The pause_state is opaque — the server keeps the canonical copy on disk, so you don't need to send it back yourself.
* `client.resume(run_id, user_input=...)` is synchronous: it returns `{status: running}` once the worker has been kicked back into the runner. The continuation runs in the background; keep polling status until terminal.
* The runner can pause more than once on the same run (OTP → confirmation → another OTP). Each pause flips status back to `paused` with a fresh `prompt`; resume the same way.
* Plan-signature mismatch on resume returns 400 — usually a sign that someone edited the on-disk state, since the plan can't change between submit and resume.

See [API / Pause / resume](../api.md#pause-resume) for the full request/response shapes.

## Lifecycle endpoints (cheap-poll + queue) {#lifecycle}

The action-based `POST /v1/predict {action: status}` returns the full
detail payload — fine for one-off checks, expensive when polled in a
loop. The lifecycle endpoints below give a cheap *phase + backoff hint*
poll surface plus a per-tenant queue snapshot. Use these for active
polling, then call the action-based status route once a terminal phase
arrives if you need full detail.

### `GET /v1/runs/{run_id}` — phase + backoff hint

```bash
curl -fsS "$ENDPOINT/v1/runs/$RUN_ID" \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN"
```

Returns:

```json
{
  "run_id": "<run_id>",
  "phase": "running",
  "last_event_at": 1722500400.0,
  "polling_backoff_ms_hint": 1500,
  "started_at": 1722500390.0,
  "finished_at": null,
  "halt_class": null
}
```

Phases (`queued` / `running` / `recovering` / `complete` / `halted` /
`cancelled`) are stable wire constants. `complete` covers both fully
successful runs and `completed_with_failures`. `polling_backoff_ms_hint`
is adaptive: terminal phases give a long hint (state won't change),
fresh transitions give a short hint, idle runs grow progressively
longer.

Client pattern:

```python
import requests, time

def watch(run_id: str) -> dict:
    while True:
        r = requests.get(
            f"{ENDPOINT}/v1/runs/{run_id}",
            headers={"X-Mantis-Token": TOKEN},
        )
        r.raise_for_status()
        body = r.json()
        if body["phase"] in {"complete", "halted", "cancelled"}:
            return body
        time.sleep(body["polling_backoff_ms_hint"] / 1000)
```

### `GET /v1/queue` — tenant queue snapshot

```bash
curl -fsS "$ENDPOINT/v1/queue" \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN"
```

Returns:

```json
{
  "tenant_id": "<tenant>",
  "queued": 2,
  "running": 1,
  "recovering": 0,
  "eta_ms": null
}
```

Scoped to the calling tenant. Terminal runs are excluded — operators
wanting historical totals can scan `status.json` files via
`POST /v1/predict {action: status}`. `eta_ms` is best-effort and may
be `null` when no recent dispatch samples exist.

### `POST /v1/runs/{run_id}/cancel`

Already documented under [Cancelling](#cancelling) above. The route
accepts an optional JSON body `{"reason": "<operator-tag>"}` for log
attribution; the response carries the post-cancel status.

> **Note for self-hosters running multiple API replicas:** the
> lifecycle data structures shipped in
> `mantis_agent.run_lifecycle.RunLifecycleStore` are single-process
> (in-memory) by design. The deployed routes here derive phase from
> the file-backed `status.json` written by the executor, so they work
> across replicas without modification. If you build a richer
> integration on the `RunLifecycleStore` API directly (for example,
> adding `should_cancel()` polling inside an executor), back the store
> with Redis or Modal Dict before scaling past one replica.

## See also

- [Recordings](recordings.md) — fetching the screencast
- [Errors](errors.md) — full error reference
- [Reference / HTTP API](../api.md) — endpoint detail
