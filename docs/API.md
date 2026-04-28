# Mantis CUA HTTP API

Reference for callers who want to use the Mantis CUA service directly —
without going through vision_claude or another wrapper. For the
vision_claude integration walkthrough see
[`integration-vision_claude.md`](integration-vision_claude.md).

## Endpoints

| Path | Auth | Purpose |
|---|---|---|
| `POST /v1/predict` | `X-Mantis-Token` (run scope) | Run a plan, poll status, fetch result. The high-level orchestrator. |
| `POST /predict` | `X-Mantis-Token` (run scope) | Backwards-compat alias for `/v1/predict`. Identical behavior. |
| `POST /v1/chat/completions` | `X-Mantis-Token` (run scope) | OpenAI-compat reverse proxy to in-pod Holo3 (raw inference). |
| `GET /v1/models` | open | OpenAI-compat model list. Returns `holo3`. |
| `GET /v1/health`, `GET /health` | open | Liveness/readiness probe. |
| `GET /metrics` | open | Prometheus scrape endpoint. Returns 503 if `prometheus_client` not installed. |

When deployed behind Baseten, all requests must also carry
`Authorization: Api-Key <BASETEN_API_KEY>` (gateway auth, separate from
container auth).

## Authentication

The service uses **two layers of auth** when deployed on Baseten:

| Header | Layer | Purpose |
|---|---|---|
| `Authorization: Api-Key <BASETEN_API_KEY>` | Baseten gateway | Authenticates the platform request. Required for any call. |
| `X-Mantis-Token: <tenant_token>` | Container | Authenticates the tenant. Required for `/v1/predict` and `/v1/chat/completions`. |

`X-Mantis-Token` is split into a custom header (rather than another `Authorization: Bearer`) because the Baseten gateway's `Authorization: Api-Key` header is forwarded to the container; using the same header for both auth layers would clash.

If `MANTIS_TENANT_KEYS_PATH` is configured on the deployment, each tenant has its own token. Otherwise a single `MANTIS_API_TOKEN` works for all callers (single-tenant mode).

## Rate / scale caps

Per-request server-side caps that the caller cannot exceed:

| Env var | Default | Effect |
|---|---|---|
| `MANTIS_MAX_STEPS_PER_PLAN` | 200 | Plans larger than this are rejected with `400`. |
| `MANTIS_MAX_LOOP_ITERATIONS` | 50 | `loop_count` in any `loop` step is silently clamped to this. |
| `MANTIS_MAX_RUNTIME_MINUTES` | 60 | `max_time_minutes` in the request body is clamped. |
| `MANTIS_MAX_COST_USD` | 25.0 | `max_cost` in the request body is clamped. |

Plus per-tenant caps when multi-tenant is enabled (`max_concurrent_runs`, `max_cost_per_run`, `max_time_minutes_per_run`).

---

## `POST /v1/predict`

Run a plan, poll an existing run, fetch the result, or fetch live logs. The mode is determined by the `action` field (or its absence).

### Run a new plan

The request body must contain **exactly one** of these plan-shape fields, in priority order:

| Field | Type | Description |
|---|---|---|
| `task_suite` | object | Inline task-suite dict. Use this for arbitrary plans where you don't want to bake them into the container image. |
| `task_file_contents` | string | JSON-as-string. Same shape as `task_suite` but pre-serialized. |
| `task_file` | string | Path inside the container image (e.g. `tasks/crm/staffai_tasks.json`). |
| `micro` | string | Path to a micro-plan JSON or plain-text plan inside the image (e.g. `plans/boattrader/extract_url_filtered.json`). |
| `plan_text` | string | Inline plain-English plan. Decomposed via Claude on the server side. |

Plus the run options:

| Field | Default | Description |
|---|---|---|
| `detached` | `true` | Return a `run_id` immediately and continue work in the background. Set `false` to block until done (only useful for short plans — 5–10s). |
| `state_key` | `""` | Caller-chosen identifier; the server prefixes it with `tenant_id` so callers can't collide. Reuse the same key across runs to share checkpoint state and Chrome profile (cookies, sessions). |
| `resume_state` | `false` | Reconstruct browser state from the latest checkpoint at `state_key` before starting. |
| `max_cost` | `25.0` | Cap in USD; clamped against the tenant cap. |
| `max_time_minutes` | `60` | Wall-clock cap; clamped against the tenant cap. |
| `proxy_city`, `proxy_state` | unset | Optional IPRoyal geo overrides. Subject to allowlist. |

#### Detached response

```jsonc
{
  "status": "queued",
  "created_at": "2026-04-28T01:57:08.316Z",
  "model": "holo3",
  "mode": "detached",
  "run_id": "20260428_021432_076255ef",
  "payload": { ... echoed input ... },
  "updated_at": "2026-04-28T01:57:08.317Z",
  "status_path":  "/workspace/mantis-data/runs/<run_id>/status.json",
  "result_path":  "/workspace/mantis-data/runs/<run_id>/result.json",
  "csv_path":     "/workspace/mantis-data/runs/<run_id>/leads.csv",
  "events_path":  "/workspace/mantis-data/runs/<run_id>/events.log"
}
```

The `*_path` fields are server-internal; you fetch them through the polling actions (next section).

### Poll / fetch / cancel an existing run

Set `action` and `run_id` in the body:

```jsonc
{ "action": "status", "run_id": "20260428_021432_076255ef" }
{ "action": "result", "run_id": "..." }
{ "action": "logs",   "run_id": "...", "tail": 200 }
{ "action": "cancel", "run_id": "..." }
```

`status` returns the current state plus a `summary` block when the run is in a terminal state:

```jsonc
{
  "status": "succeeded",          // or running | failed | cancelled
  "run_id": "...",
  "started_at": "...",
  "finished_at": "...",
  "summary": {
    "total_time_s": 569,
    "steps_executed": 17,
    "viable": 3,
    "leads_with_phone": 1,
    "result_path": "...",
    "csv_path": "...",
    "dynamic_verification_summary": { ... },
    "cost_total": 0.42,
    "cost_breakdown": {
      "gpu":    0.12,
      "claude": 0.12,
      "proxy":  0.18
    }
  }
}
```

`result` returns the full lead list and per-step trace. `logs` returns the
last `tail` events written by the runner (default 200, max 10000).

### Errors

| Status | Meaning |
|---|---|
| `400` | Bad request. Common causes: no plan-shape provided, malformed JSON, plan exceeds `MANTIS_MAX_STEPS_PER_PLAN`, micro-step missing `intent`/`type`. |
| `401` | Missing or invalid `X-Mantis-Token`. |
| `403` | Token valid but tenant lacks `run` scope (read-only key). |
| `404` | `action=status\|result\|logs` referenced an unknown `run_id`. |
| `429` | (Tier 2) Tenant exceeded concurrent-run cap. |
| `500` | Unhandled exception — check `events_path` for traceback. |
| `502` | Upstream Holo3 (`/v1/chat/completions`) or Anthropic API unreachable. |
| `503` | Server auth not configured (`MANTIS_API_TOKEN` unset and no keys file). |

---

## `POST /v1/chat/completions`

OpenAI-compatible reverse proxy to the in-pod Holo3 server. **For raw inference only** — no plan orchestration, no Claude grounding, no checkpointing. Designed for clients that want to run their own perception-action loop and use Holo3 as the brain.

```bash
curl -X POST "https://model-qvvgkneq.api.baseten.co/production/v1/chat/completions" \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "holo3",
    "messages": [
      {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        {"type": "text", "text": "Click the boat listing title."}
      ]}
    ],
    "max_tokens": 256
  }'
```

Auth headers and Mantis-side cookies are stripped before the request is forwarded to llama.cpp; the upstream never sees your tenant credentials.

For the orchestrated/reliable path that handles the full plan, use `/v1/predict` instead.

---

## `GET /v1/models`

OpenAI-compatible model listing.

```jsonc
{
  "object": "list",
  "data": [
    { "id": "holo3", "object": "model", "owned_by": "mantis" }
  ]
}
```

---

## End-to-end example: 3-listing BoatTrader extraction

```bash
TOKEN=$(read -srp "MANTIS_API_TOKEN: " v && echo "$v")
BTKEY="$BASETEN_API_KEY"
ENDPOINT="https://model-qvvgkneq.api.baseten.co/production"

# 1. Launch detached run
RESP=$(curl -fsS -X POST "$ENDPOINT/v1/predict" \
  -H "Authorization: Api-Key $BTKEY" \
  -H "X-Mantis-Token: $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "detached": true,
    "micro": "plans/boattrader/extract_url_filtered_3listings.json",
    "state_key": "smoke-test",
    "resume_state": false,
    "max_cost": 2,
    "max_time_minutes": 20
  }')
RUN_ID=$(echo "$RESP" | jq -r .run_id)
echo "run_id: $RUN_ID"

# 2. Poll status until terminal
while true; do
  STATUS=$(curl -fsS -X POST "$ENDPOINT/v1/predict" \
    -H "Authorization: Api-Key $BTKEY" \
    -H "X-Mantis-Token: $TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"action\":\"status\",\"run_id\":\"$RUN_ID\"}" | jq -r .status)
  echo "$(date '+%H:%M:%S') $STATUS"
  case "$STATUS" in succeeded|failed|cancelled) break ;; esac
  sleep 30
done

# 3. Fetch leads
curl -fsS -X POST "$ENDPOINT/v1/predict" \
  -H "Authorization: Api-Key $BTKEY" \
  -H "X-Mantis-Token: $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"action\":\"result\",\"run_id\":\"$RUN_ID\"}" \
  | jq .result.leads
```

Real result from the verification run on this branch:

```
1997 Caroff CHATAM 52       — $254,000 — phone +596696520959
1987 Beneteau Idylle 15.50  — $130,000
2006 Luhrs 41 Convertible   — $235,000 (Private Seller)
```

## Plan shapes — when to use which

| Use case | Recommended shape |
|---|---|
| Recurring high-volume workflow with predictable steps | Hand-author a **micro-plan** JSON, ship it in the image at `plans/<domain>/<workflow>.json`, reference via `micro` |
| Arbitrary plain-English request | `plan_text` — server decomposes it via Claude (cached after first run) |
| Ad-hoc plan you don't want baked into the image | `task_suite` (inline JSON dict) |
| StaffAI-style multi-task suite with `task_id` + `verify` clauses | `task_suite` or `task_file` |

## Plan formats

### `micro` — micro-plan JSON

A flat list of step objects executed by `MicroPlanRunner`:

```jsonc
[
  {"intent": "Navigate to https://...", "type": "navigate",
   "section": "setup", "required": true},
  {"intent": "Verify filters applied",  "type": "extract_data",
   "claude_only": true, "section": "setup", "gate": true,
   "verify": "Page shows boat listings ..."},
  {"intent": "Click listing title",     "type": "click",
   "grounding": true, "section": "extraction"},
  {"intent": "Read URL",                "type": "extract_url",
   "claude_only": true, "section": "extraction"},
  {"intent": "Scroll to description",   "type": "scroll",
   "budget": 10, "section": "extraction"},
  {"intent": "Extract data",            "type": "extract_data",
   "claude_only": true, "section": "extraction"},
  {"intent": "Go back",                 "type": "navigate_back",
   "section": "extraction"},
  {"intent": "Loop",                    "type": "loop",
   "loop_target": 2, "loop_count": 3, "section": "extraction"}
]
```

Step types: `navigate`, `filter`, `click`, `scroll`, `extract_url`, `extract_data`, `navigate_back`, `paginate`, `loop`.

Key fields:

| Field | Effect |
|---|---|
| `section` | One of `setup`, `extraction`, `pagination`. Used by retry/halt logic. |
| `required` | If true, retry on fail then halt the whole run. |
| `gate` | Claude verifies a condition; halt on fail. |
| `verify` | Free-text condition Claude checks. |
| `claude_only` | Skip Holo3; Claude does the perception. Use for extract / gate steps. |
| `grounding` | Refine click coordinates with `ClaudeGrounding`. |
| `budget` | Max actions Holo3 can take in this step (default 8). |
| `loop_target` | Step index to jump back to. |
| `loop_count` | Max loop iterations (clamped to `MANTIS_MAX_LOOP_ITERATIONS`). |

### `task_suite` — multi-task JSON

For Claude-CUA-style autonomous-per-task workflows (the existing
`tasks/crm/staffai_tasks.json` is this shape):

```jsonc
{
  "session_name": "staffai_crm",
  "base_url": "https://staffai-test-crm.exe.xyz",
  "auth": { "user_id": "...", "password": "..." },
  "tasks": [
    {
      "task_id": "login",
      "intent": "Go to https://... and log in with user X and password Y",
      "save_session": true,
      "start_url": "https://...",
      "verify": { "type": "url_not_contains", "value": "login" }
    },
    {
      "task_id": "update_lead_industry",
      "intent": "Go to the Leads Page. Update industry of qualified lead to 'Space Exploration'.",
      "require_session": true,
      "start_url": "https://...",
      "verify": { "type": "page_contains_text", "value": "Space Exploration" }
    }
  ]
}
```

Each task runs with its own `max_steps` budget; Claude decides what to do per task. The runner verifies the `verify` clause after each.

### `plan_text` — plain-English

```jsonc
{
  "plan_text": "Go to BoatTrader, filter to Miami private sellers above $35,000, extract listing details for the first 3 listings, save year/make/model/price/phone."
}
```

`PlanDecomposer` (Claude-backed, cached by signature) converts this into a micro-plan and proceeds. Decomposition costs ~$0.10 the first time per unique plan text; subsequent runs hit the cache.

---

## Pricing (verified end-to-end)

Real numbers from the BoatTrader 3-listing run on Baseten:

| Item | Cost |
|---|---|
| GPU (Holo3 on H100, ~10 min) | ~$0.12 |
| Claude (gates + extract + grounding) | ~$0.12 |
| Proxy (IPRoyal residential) | ~$0.18 |
| **Total per 3-listing run** | **~$0.42** |
| **Per-listing** | **~$0.14** |

For comparison, equivalent Claude-only CUA flow ~$0.50–$1.50 per listing.

---

## Security model

| Concern | Guarantee |
|---|---|
| Tenant token confidentiality | Stored in Baseten secrets; constant-time compare on validation; never echoed in logs |
| Per-tenant Anthropic key | Resolved from the tenant's `anthropic_secret_name` — keys are not shared across tenants |
| Per-tenant browser profile | Mounted at `/workspace/mantis-data/tenants/<tenant_id>/chrome-profile/<state_key>/` — cookies cannot bleed across tenants |
| Per-tenant run state | Same volume layout — `state_key` is server-prefixed so callers cannot read another tenant's checkpoint |
| Plan injection (e.g., `loop_count: 999_999`) | Server-side hard caps clamp the values; oversized plans are rejected with `400` |
| Upstream credential leak | `/v1/chat/completions` strips `X-Mantis-Token`, `Authorization`, `Cookie` before forwarding to in-pod llama.cpp |

## Limits / caveats

- **Detached runs survive replica restart** (state on the data volume) but only on the same Baseten model. Cross-region failover not supported.
- **Pause/resume for OTP** is not yet wired through `/v1/predict`. Today it works in the vision_claude integration path because the loop runs in the vision_claude pod.
- **`/v1/chat/completions`** is unstreamed in v1. Streaming SSE is a Tier 2 follow-up.
- **Single Anthropic-key per tenant** at request time (re-resolved on every call).

## Tier 2 features (rate limits, idempotency, webhooks, allowlist, metrics)

### Rate limits

Two dimensions, both enforced per-tenant:

| Dimension | Source | Behavior on exceed |
|---|---|---|
| **Concurrent runs** | `tenant.max_concurrent_runs` (default 5) | `429 Too Many Requests` with `Retry-After: 5` |
| **Rate** (token bucket) | `tenant.rate_limit_per_minute` (default 30) | `429` with `Retry-After: <seconds-until-token>` |

State is in-process per replica. Behind a load balancer with N replicas, the effective per-tenant cap is roughly `N × configured_cap`. For strict cluster-wide limits, deploy a single replica or swap to a Redis-backed limiter (planned Tier 2.5).

### Idempotency keys

Send `Idempotency-Key: <unique-string>` on `POST /v1/predict`. The server caches `(tenant_id, key) → run_id` with a 24-hour TTL. Subsequent retries with the same key return the original `run_id` without starting a new run.

```bash
curl -X POST "$ENDPOINT/v1/predict" \
  -H "Authorization: Api-Key $BTKEY" \
  -H "X-Mantis-Token: $TOK" \
  -H "Idempotency-Key: order-7afc3b91" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

The cache is sidecar-backed (`$MANTIS_DATA_DIR/idempotency/<tenant_id>/<key_hash>.json`) so a replica restart preserves it.

### Webhook callbacks

Two ways to receive run-completion notifications:

1. **Per-tenant default** — set `webhook_url` and `webhook_secret_name` in the tenant keys file.
2. **Per-request override** — pass `callback_url` in the `/v1/predict` body.

When the run reaches a terminal state (`succeeded`, `failed`, `cancelled`), the server POSTs:

```jsonc
{
  "run_id": "20260428_021432_076255ef",
  "tenant_id": "vision_claude_prod",
  "status": "succeeded",
  "summary": { ... same shape as /v1/predict status response ... },
  "delivered_at": "2026-04-28T02:24:01.648Z"
}
```

With an HMAC-SHA256 signature in `X-Mantis-Signature: sha256=<hex>` (signed with the tenant's webhook secret). 3 retries with exponential backoff (1s, 5s, 30s) if the receiver returns non-2xx or fails to connect.

Verify the signature on receipt:

```python
import hmac, hashlib
def verify(body: bytes, header_sig: str, secret: str) -> bool:
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header_sig)
```

### URL allowlist enforcement

If a tenant has `allowed_domains` set in the keys file, every plan submitted via `/v1/predict` is scanned for `navigate`-type URLs and `task_suite.base_url` / `task.start_url`. Off-list hosts return `403 Forbidden` before any run starts:

```jsonc
{
  "detail": "plan references host(s) not in tenant allowlist: evil.com"
}
```

Wildcards: `*.boattrader.com` matches any subdomain but not `boattrader.com.evil.com`. Empty `allowed_domains` (the default) skips this check.

### Prometheus metrics

`GET /metrics` returns Prometheus text format. Metric names + labels:

| Metric | Type | Labels | Notes |
|---|---|---|---|
| `mantis_predict_requests_total` | counter | `tenant_id`, `mode`, `outcome` | mode = `run\|status\|result\|logs\|cancel`; outcome = `ok\|bad_request\|rate_limited\|denied_allowlist\|idempotent_hit\|error` |
| `mantis_chat_completions_total` | counter | `tenant_id`, `outcome` | outcome = `ok\|status_4xx\|status_5xx\|upstream_error` |
| `mantis_run_duration_seconds` | histogram | `tenant_id`, `model`, `status` | Buckets: 10s … 3600s |
| `mantis_run_cost_usd` | histogram | `tenant_id`, `model`, `status` | Buckets: $0.01 … $25 |
| `mantis_concurrent_runs` | gauge | `tenant_id` | Currently in-flight runs |
| `mantis_rate_limit_rejections_total` | counter | `tenant_id`, `kind` | kind = `rate\|concurrent` |

If `prometheus_client` isn't installed in the container (e.g., orchestrator-only install), `/metrics` returns `503` and all metric calls become no-ops — the rest of the API is unaffected.

## Tier roadmap

This API is at **Tier 2** — production-quality multi-tenant. Upcoming:

- **Tier 3:** billing records, admin API, multi-region.

See [`PROPOSAL-mantis-cua-replaces-vision_claude.md`](PROPOSAL-mantis-cua-replaces-vision_claude.md) for the bigger architectural picture and migration plan.
