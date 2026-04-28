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
| `navigation_blocked` | Cloudflare 403 or sustained 5xx from the target | Wait + retry; or try with a fresh `state_key` |

These are partial successes — the `summary` block reflects whatever was completed (e.g., 2 of 3 listings extracted before the cap hit).

## Retry guidance

| Code | Retry strategy |
|---|---|
| 4xx (except 429) | Don't retry — fix the request |
| 429 | Honor `Retry-After`; capped exponential backoff after that |
| 500 | Inspect `{"action":"logs"}` first; retry with a fresh `state_key` if it looks transient |
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
