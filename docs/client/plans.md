# Sending plans

Once you have an authenticated session, every plan submission lands on `POST /v1/predict` with one of the four plan-shape fields. Pick the one that fits your data:

```jsonc
{
  "detached": true,                                  // (default) async with run_id
  "task_suite":  { ... },                            // OR
  "task_file":   "tasks/crm/crm_tasks.json",     // OR
  "micro":       "plans/boattrader/...json",         // OR
  "plan_text":   "Plain English description",        //

  "state_key":         "stable-workflow-id",         // namespaced server-side
  "resume_state":      false,                        // continue from last checkpoint
  "max_cost":          2.0,                          // clamped to tenant cap
  "max_time_minutes":  20,
  "proxy_city":        "Miami",                      // optional IPRoyal geo override
  "proxy_state":       "FL",
  "record_video":      false,                        // see Recordings page
  "video_format":      "mp4",
  "video_fps":         5,
  "callback_url":      "https://my-webhook"          // overrides tenant default
}
```

Field reference is on the [Reference / HTTP API](../api.md#post-v1predict) page.

## Decision tree

```
Have a stable workflow you'll run many times?
  ├─ yes → micro-plan via `micro: "plans/<domain>/<workflow>.json"`
  └─ no  → one-shot or rapidly-changing?
            ├─ yes → `plan_text` — server decomposes via Claude
            └─ no  → multi-task suite style?
                     ├─ yes → `task_suite` (multi-task dict)
                     └─ no  → `task_suite` with a flat micro-plan list
```

## Examples

=== "micro (high-volume reliable)"

    ```bash
    curl -X POST "$ENDPOINT/v1/predict" \
      -H "Authorization: Api-Key $BASETEN_API_KEY" \
      -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
      -H "Content-Type: application/json" \
      -d '{
        "detached": true,
        "micro": "plans/boattrader/extract_url_filtered_3listings.json",
        "state_key": "boattrader-prod",
        "max_cost": 2,
        "max_time_minutes": 20
      }'
    ```

=== "plan_text (one-shot)"

    ```bash
    curl -X POST "$ENDPOINT/v1/predict" \
      -H "Authorization: Api-Key $BASETEN_API_KEY" \
      -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
      -H "Content-Type: application/json" \
      -d '{
        "detached": true,
        "plan_text": "Go to BoatTrader, filter to Miami private sellers above $35,000, extract listing details for the first 3 listings.",
        "state_key": "ad-hoc-1",
        "max_cost": 2
      }'
    ```

=== "task_suite (multi-task)"

    ```bash
    curl -X POST "$ENDPOINT/v1/predict" \
      -H "Authorization: Api-Key $BASETEN_API_KEY" \
      -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
      -H "Content-Type: application/json" \
      -d "$(cat <<JSON
    {
      "detached": true,
      "task_suite": $(cat tasks/crm/crm_tasks.json),
      "state_key": "crm-$(date +%s)",
      "max_cost": 5,
      "max_time_minutes": 30
    }
    JSON
    )"
    ```

=== "task_file (baked-in path)"

    ```bash
    curl -X POST "$ENDPOINT/v1/predict" \
      -H "Authorization: Api-Key $BASETEN_API_KEY" \
      -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
      -H "Content-Type: application/json" \
      -d '{
        "detached": true,
        "task_file": "tasks/crm/crm_tasks.json",
        "state_key": "crm-prod"
      }'
    ```

## Caps and clamping

| Knob | Server hard cap (env-overridable) | Tenant cap (per-tenant config) | Request value |
|---|---|---|---|
| `max_cost` | `MANTIS_MAX_COST_USD` (default $25) | `max_cost_per_run` | `max_cost` |
| `max_time_minutes` | `MANTIS_MAX_RUNTIME_MINUTES` (60) | `max_time_minutes_per_run` | `max_time_minutes` |
| Plan size | `MANTIS_MAX_STEPS_PER_PLAN` (200) | n/a — rejects 400 | n/a |
| `loop_count` | `MANTIS_MAX_LOOP_ITERATIONS` (50) | n/a — silently clamps | n/a |

The effective value is `min(server_cap, tenant_cap, request_value)`. If you ask for $50, your tenant has $5 cap, and the server has $25 cap → you get $5.

## URL allowlist

If your tenant has `allowed_domains` configured, the server scans your plan's `navigate` steps + `task_suite.base_url` + each `task.start_url` and rejects 403 if any host is off-list. Wildcards like `*.boattrader.com` work; exact matches like `crm.example.com` work.

If you need to add a domain, ask your operator to update your tenant config — see [URL allowlist](../operations/allowlist.md).

## What the server returns

For `detached: true` (the default) — a queued handle:

```jsonc
{
  "status": "queued",
  "model": "holo3",
  "mode": "detached",
  "run_id": "20260428_021432_076255ef",
  "created_at": "2026-04-28T02:14:32.331Z",
  "payload": { ...echoed input... },
  "status_path":  "/workspace/.../status.json",
  "result_path":  "/workspace/.../result.json",
  "csv_path":     "/workspace/.../leads.csv",
  "events_path":  "/workspace/.../events.log"
}
```

Use `run_id` for [Runs and polling](runs-and-polling.md).

For `detached: false` — the call blocks until the run completes (5-30+ min). Useful only for short plans; the response is the same shape you'd otherwise fetch with `{action: "result"}`.

## See also

- [Plan formats](../getting-started/plan-formats.md) — full schemas
- [Runs and polling](runs-and-polling.md) — what to do with a `run_id`
- [Errors](errors.md) — when something goes wrong
