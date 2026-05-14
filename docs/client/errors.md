# Errors

What every status code means and how to handle it.

## Quick reference

| Status | Meaning | Retry? |
|---|---|---|
| **200** | Success | n/a |
| **400** | Bad request — malformed JSON, oversized plan, missing `intent`/`type` on a step | No — fix the payload |
| **401** missing | No `X-Mantis-Token` | No — add the header |
| **401** invalid | Token doesn't match a tenant | No — verify the token wasn't truncated |
| **403** scope | Token valid but tenant lacks the required scope (`run` / `status` / `result` / `logs`) | No — operator needs to add scope |
| **403** allowlist | Plan references a host not in tenant's `allowed_domains` | No — fix the plan or ask operator to add the domain |
| **404** run | `action=status\|result\|logs` referenced a `run_id` your tenant doesn't own | No — wrong `run_id` or wrong tenant |
| **404** video | No recording for the run (recording disabled or ffmpeg failed) | No |
| **409** profile-busy | (#342, Modal HTTP endpoint only) Another run is currently holding the requested `profile_id`'s Chrome user-data-dir lock. The `detail` includes the held `run_id` so you can poll it. | Yes — wait for the held run to finish, or submit with a different `profile_id` |
| **429** rate | Tenant exceeded `rate_limit_per_minute` | Yes — honor `Retry-After` header |
| **429** concurrent | Tenant at `max_concurrent_runs` | Yes — honor `Retry-After` |
| **500** | Unhandled exception | Sometimes — check `{"action":"logs", ...}` for the traceback before retrying |
| **502** upstream | Holo3 (`/v1/chat/completions`) or Anthropic API unreachable | Yes — exponential backoff |
| **503** auth-not-configured | Server has neither `MANTIS_API_TOKEN` nor `MANTIS_TENANT_KEYS_PATH` set | No — operator misconfiguration |
| **503** metrics | `prometheus_client` not installed in the container; `/metrics` only | n/a |

## Error body shape

```jsonc
{
  "detail": "human-readable error string"
}
```

The `detail` string is meant to be safe to surface to your users (no internals leaked). For deeper debugging, use `{"action": "logs", "run_id": "...", "tail": 500}` to get the runner's event log.

## Run-level failures

A run can return `200 OK` and still have `status: failed`:

```jsonc
{
  "status": "failed",
  "run_id": "...",
  "error": "page_blocked",       // or "max_cost_exceeded", "timeout", etc.
  "summary": { ... whatever was completed ... }
}
```

| `error` value | Cause | Fix |
|---|---|---|
| `page_blocked` | The site detected the agent / Cloudflare didn't pass | Different proxy, geo, or reduce request rate |
| `max_cost_exceeded` | Budget cap hit before the plan completed | Raise `max_cost` (within tenant cap) or shorten the plan |
| `timeout` | `max_time_minutes` hit | Same |
| `gate_failed` | A `gate: true` step's verify clause was false | The site didn't reach the expected state — different filters / start URL |
| `extract_failed` | Claude couldn't parse the screenshot into the expected schema | Schema mismatch with what's actually visible — adjust the plan |
| `navigation_blocked` | Cloudflare 403 or sustained 5xx from the target | Wait + retry; or try with a fresh `profile_id` (different cookies / IP rotation) |

These are partial successes — the `summary` block reflects whatever was completed (e.g., 2 of 3 listings extracted before the cap hit).

## Retry guidance

| Code | Retry strategy |
|---|---|
| 4xx (except 429) | Don't retry — fix the request |
| 429 | Honor `Retry-After`; capped exponential backoff after that |
| 500 | Inspect `{"action":"logs"}` first; retry with a fresh `workflow_id` (and a fresh `profile_id` if the failure looks tied to a corrupted Chrome session) if it looks transient |
| 502 | Exponential backoff (5 s → 30 s → 2 min); the upstream Holo3 / Anthropic might be down |
| 503 (auth) | Don't retry — operator action needed |

If you're hitting 429 frequently, ask your operator to raise the per-tenant `rate_limit_per_minute` or `max_concurrent_runs` — see [Rate limits](../operations/rate-limits.md).

## Idempotency for safe retries

For workflows where double-execution would be expensive (e.g., extracting from a paginated site twice), use `Idempotency-Key`:

```bash
KEY="my-job-$(uuidgen)"
curl -X POST "$ENDPOINT/v1/predict" \
  -H "Idempotency-Key: $KEY" \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -d '{...}'
```

Every retry of the same `Idempotency-Key` returns the cached `run_id` (24 h TTL). See [Idempotency](../operations/idempotency.md).

## Diagnosing a failed step

When a step fails, `action=result` returns each failed step with structured diagnostics — read these **before** falling back to `action=logs`. The full schema is documented in [`mantis plan run` → `result.json`](../getting-started/cli.md#resultjson-schema); the fields you'll care about per failed step are:

| Field | What it tells you |
|---|---|
| `failure_class` | Stable enum (see table below). Branch on this in dashboards / retry policies. |
| `final_url` | Browser URL at the moment of failure. |
| `page_title` | Page title at the moment of failure — CF interstitials surface here even when `data` is empty. |
| `last_action` | The final `Action` dispatched before failure (`{type, params, reasoning}`). |
| `screenshot_b64` | Base64-encoded PNG of the post-failure viewport. |
| `data` | Short prose from the handler (`gate:FAIL:Error 403`, `fill_error: not found`, …). |

### `failure_class` → likely cause → first action

| `failure_class` | Likely cause | First action |
|---|---|---|
| `cf_challenge` | Cloudflare / anti-bot interstitial didn't clear | Retry with a fresh `profile_id` (rotates the IP / cookies). For repeated failures, ask operator to bump `MANTIS_CF_PREWARM_MAX_SECONDS`. |
| `http_4xx` | Target returned 401 / 404 / 410 | Check `final_url` — usually a stale URL in the plan or a tenant-scoped page. Not retryable. |
| `http_5xx` | Target backend returned 5xx | Exponential backoff + retry. Usually transient. |
| `nav_timeout` | Page load exceeded the navigate budget | Bump `wait_after_load_seconds` on the step, or `MANTIS_NAV_WAIT_SECONDS`. Repeated → check egress / proxy. |
| `selector_miss` | Click / fill / submit couldn't locate the target | Inspect `screenshot_b64` — the page is often in a different state than the plan expects. Adjust the plan, or add a `wait` step before. |
| `no_state_change` | Action handler reported success but the runner-state snapshot saw no URL / page / scroll change. Self-healing demotion ([epic #377](https://github.com/mercurialsolo/mantis/issues/377) Phase A). Fires on `click` / `submit` / `navigate_back`. | Usually transient — the runner already triggered a retry and (after 2× repeats on the same step) routed through `Holo3StepHandler`. If it persists into terminal failure, the click is hitting an element that doesn't actually navigate; inspect `screenshot_b64` + `last_action`. |
| `extractor_error` | Claude extractor failed / returned empty | Schema mismatch with what's visible. Tighten the recipe or relax the schema. |
| `budget_exceeded` | `max_cost` / `max_time` / per-URL context budget tripped | Raise the cap or shorten the plan. |
| `unknown` | No rule matched | Pull `action=logs` (below) — the runner traceback usually identifies it. |

### Decoding the failure screenshot

```bash
# Pull the result, extract the failed step's screenshot to a PNG.
curl -fsS -X POST "$ENDPOINT/v1/predict" \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
  -d "{\"action\":\"result\",\"run_id\":\"$RUN_ID\"}" \
  | jq -r '.steps[] | select(.success == false) | .screenshot_b64' \
  | head -1 | base64 -d > failed_step.png
```

`screenshot_b64` is omitted on success and on steps where capture failed.

### When `failure_class` isn't enough

For `unknown` (or to read the traceback / per-step Holo3 logs), fall through to:

```bash
curl -fsS -X POST "$ENDPOINT/v1/predict" \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
  -d "{\"action\":\"logs\",\"run_id\":\"$RUN_ID\",\"tail\":500}" \
  | jq -r '.events[]'
```

`action=logs` returns the runner thread's Python logger tail (state transitions, step handler messages, exception tracebacks). This is the HTTP-API analog of operator-only `modal app logs` — same content, different access path.

If the `events.log` doesn't surface the cause either, ask your operator to pull `modal app logs mantis-plan-runner` for the container-side stdout / stderr. Open a ticket with the `run_id`.

## Debugging a stuck run

If a `running` status doesn't advance for ≥ 5 min:

```bash
# Pull the last 500 events the runner emitted
curl -fsS -X POST "$ENDPOINT/v1/predict" \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
  -d "{\"action\":\"logs\",\"run_id\":\"$RUN_ID\",\"tail\":500}" \
  | jq -r '.events[]'
```

Common stuck patterns and what they look like in the logs:

| Symptom | Likely cause |
|---|---|
| Many `[click] (0,0) grounding=NO` lines | Holo3 hallucinating coordinates; the agent might be on a different page than expected |
| `[runner] plan=False executor=False idx=0 in_range=False` repeating | The runner exhausted the per-step budget; the next step's `max_steps` ran out |
| `[content-control] parse failed: ...` | Claude returned prose instead of JSON. Non-fatal; the runner moves on |
| `Runner interrupted due to worker preemption` | (Modal only) Spot GPU was preempted. The function auto-restarts |
| `Cloudflare challenge timeout` | The page's anti-bot didn't auto-resolve. Try a different proxy geo |

If the run is genuinely stuck (no log progress for 10 + minutes within the time budget), `cancel` it and start fresh.

## See also

- [Authentication](auth.md) — auth-related error details
- [Operations / Rate limits](../operations/rate-limits.md) — operator-side cap tuning
