# Mantis CUA HTTP API

Reference for callers who want to use the Mantis CUA service directly —
without going through a host wrapper. For library-shaped integrations
where you drive `MicroPlanRunner` in your own process, see
[Embedding MicroPlanRunner](integrations/embedding-microplanrunner.md)
and the [any-agent integration playbook](integrations/any-agent.md).

## Endpoints

| Path | Auth | Purpose |
|---|---|---|
| `POST /v1/predict` | `X-Mantis-Token` (run scope) | Run a plan, poll status, fetch result. The high-level orchestrator. |
| `POST /predict` | `X-Mantis-Token` (run scope) | Backwards-compat alias for `/v1/predict`. Identical behavior. |
| `POST /v1/cua` | `X-Mantis-Token` (run scope) | Pure CUA pass-through — Mantis as a thin Holo3 driver. No decomposition, no Claude. See [Pure CUA mode](client/pure-cua.md). |
| `POST /v1/chat/completions` | `X-Mantis-Token` (run scope) | OpenAI-compat reverse proxy to in-pod Holo3 (raw inference). |
| `GET /v1/models` | open | OpenAI-compat model list. Returns `holo3`. |
| `GET /v1/health`, `GET /health` | open | Liveness/readiness probe. |
| `GET /v1/version` | open | Runtime version snapshot — `version`, `model`, `ready`, `git_sha`, `build_time`. Useful for pinning client behavior to a specific build. |
| `GET /metrics` | open | Prometheus scrape endpoint. Returns 503 if `prometheus_client` not installed. |
| `GET /v1/runs/{run_id}` | `X-Mantis-Token` | **Cheap-poll lifecycle ([#806](https://github.com/mercurialsolo/mantis/issues/806))** — `phase` + adaptive `polling_backoff_ms_hint`. Use instead of `action=status` for active polling loops. |
| `GET /v1/runs/{run_id}/status` | `X-Mantis-Token` | Full-detail status (alias for `action=status`). |
| `GET /v1/runs/{run_id}/result` | `X-Mantis-Token` | Result payload once terminal (alias for `action=result`). |
| `POST /v1/runs/{run_id}/cancel` | `X-Mantis-Token` | Cancel a run (alias for `action=cancel`). |
| `GET /v1/runs/{run_id}/events` | `X-Mantis-Token` | **SSE event stream ([#808](https://github.com/mercurialsolo/mantis/issues/808))** with `?sse=true`. JSON parity with `action=reasoning_trace` otherwise. |
| `GET /v1/queue` | `X-Mantis-Token` | Per-tenant queue snapshot — counts of `queued` / `running` / `recovering` runs. |
| `POST /v1/recipes` | `X-Mantis-Token` | **Runtime recipe registration ([#809](https://github.com/mercurialsolo/mantis/issues/809))** — `{name, schema: ExtractionSchema}`. Tenant-scoped. |
| `GET /v1/recipes` | `X-Mantis-Token` | List runtime recipes registered under the caller's tenant. |
| `GET /v1/recipes/{name}` | `X-Mantis-Token` | Fetch a runtime recipe by name. |
| `DELETE /v1/recipes/{name}` | `X-Mantis-Token` | Delete a runtime recipe (idempotent). |
| `GET /v1/runs/{run_id}/video` | `X-Mantis-Token` | Download the screencast captured during a run. Returns 404 if `record_video` was not requested. |
| `GET /v1/runs/{run_id}/artifacts/{name}` | `X-Mantis-Token` | Download a run artifact ([#508](https://github.com/mercurialsolo/mantis/issues/508)). Allowlisted names: `leads.csv`, `extracted_rows.csv`, `extracted_rows.json`, `result.json`. Returns 404 when the artifact wasn't produced (no leads, no structured rows). |
| `GET /docs`, `GET /redoc` | open | Interactive Swagger UI / Redoc viewer over `/openapi.json`. Disable on production tenant fleets with `MANTIS_ENABLE_DOCS_UI=0`. |
| `GET /openapi.json` | open | Machine-readable OpenAPI spec. Always served, even when the interactive UIs are disabled — this is what client SDKs and IDE plugins consume. |

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
| `task_file` | string | Path inside the container image (e.g. `tasks/crm/crm_tasks.json`). |
| `micro` | string | Path to a micro-plan JSON or plain-text plan inside the image (e.g. `plans/example/extract_listings.json`). |
| `plan_text` | string | Inline plain-English plan. Decomposed via Claude on the server side. |

Plus the run options:

| Field | Default | Description |
|---|---|---|
| `detached` | `true` | Return a `run_id` immediately and continue work in the background. Set `false` to block until done (only useful for short plans — 5–10s). |
| `profile_id` | `"default"` | (#341) Chrome user-data-dir identity. Server prefixes with `tenant_id`. Sticky across plan revisions — same id ⇒ same cookies / logged-in sessions. |
| `workflow_id` | `plan_signature[:12]` | (#341) Checkpoint identity. Server prefixes with `tenant_id`. Rotate when the plan definition changes; pair with `resume_state` to pick up where the last run with this id stopped. |
| `state_key` | `""` | Legacy single-field identity. When set alone, the server routes it to both `profile_id` and `workflow_id` (back-compat). Prefer the split fields above in new code; see [#341](https://github.com/mercurialsolo/mantis/issues/341). |
| `resume_state` | `false` | Reconstruct browser state from the latest checkpoint at `workflow_id` before starting. |
| `max_cost` | `25.0` | Cap in USD; clamped against the tenant cap. |
| `max_time_minutes` | `60` | Wall-clock cap; clamped against the tenant cap. |
| `proxy_city`, `proxy_state` | unset | Optional IPRoyal geo overrides. Subject to allowlist. |
| `record_video` | `false` | If true, captures the Xvfb display while the run executes and saves a screencast under the per-tenant run dir. Fetch via `GET /v1/runs/{run_id}/video`. |
| `video_format` | `"mp4"` | One of `mp4`, `webm`, `gif`. |
| `video_fps` | `5` | Capture rate; clamped to `[1, 30]`. Higher fps = larger file + more CPU. |
| `live_viewer` | `false` | ([#416](https://github.com/mercurialsolo/mantis/issues/416)) Stand up an MJPEG tunnel onto the Xvfb display and surface its URL as `viewer_url` on `action=status`. Open the URL in a browser to watch the run live. Currently only the `holo3` executor wires this through. |

The following go **inside `task_suite`** (not top-level), alongside the plan:

| Suite field | Description |
|---|---|
| `_lora_adapter` | ([#911](https://github.com/mercurialsolo/mantis/issues/911)) Serve **base + this LoRA adapter** (the promotion-gate *challenger*). A ref `"<volume>:/checkpoints/<algo>"` (the trainer volume `mantis-trainer-vol` is mounted read-only on the executors) or a mounted path. For llama.cpp bases (`holo3`) prefer a pre-converted `.gguf` adapter; vLLM bases serve the PEFT dir directly. Omit to serve the base (the *champion*). **Modal only** — on Baseten the adapter is a deployment-level env (`MANTIS_LORA_ADAPTER`), see [Baseten hosting](hosting/baseten.md). |
| `_lora_name` | vLLM only — served-model-name for the adapter (default `challenger`). |
| `_lora_scale` | llama.cpp only — adapter scale (default `1.0`; emits `--lora-scaled`). |

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
  // Present only when the run was started with ``live_viewer: true``
  // and the executor has stood up the MJPEG tunnel. Hot-link in any
  // browser while the run is still running.
  "viewer_url": "https://ta-...-7860-....w.modal.host?token=...",
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
    },
    "wall_time_breakdown": {
      "perceive":       12.4,
      "think":          88.1,
      "act":             6.7,
      "settle":         32.0,
      "claude_ground":  18.9,
      "claude_extract": 71.2,
      "claude_verify":   4.3,
      "load":           12.1,
      "overhead":        1.3
    }
  }
}
```

`result` returns the full lead list and per-step trace. `logs` returns the
last `tail` events written by the runner (default 200, max 10000).

### Structured extraction artifacts ([#508](https://github.com/mercurialsolo/mantis/issues/508))

In addition to the legacy `leads` string list, every `result` carries an
`artifacts` array describing structured extracted data and downloadable
files. The legacy fields (`leads`, `csv_path`, `result_path`) are kept for
back-compat — `artifacts` is the new contract for callers that want
schema-keyed rows or want to fetch files over HTTP rather than read
server-local paths.

```jsonc
{
  "artifacts": [
    {
      "name": "extracted_rows",
      "kind": "structured_data",
      "mime_type": "application/json",
      "schema": { "fields": ["title", "url", "department"] },
      "row_count": 12,
      "data": [
        { "title": "ML Engineer", "url": "https://...", "department": "Eng" }
      ]
    },
    {
      "name": "leads.csv",
      "kind": "file",
      "mime_type": "text/csv",
      "row_count": 12,
      "download_url": "/v1/runs/<run_id>/artifacts/leads.csv"
    },
    {
      "name": "extracted_rows.csv",
      "kind": "file",
      "mime_type": "text/csv",
      "schema": { "fields": ["title", "url", "department"] },
      "row_count": 12,
      "download_url": "/v1/runs/<run_id>/artifacts/extracted_rows.csv"
    },
    {
      "name": "extracted_rows.json",
      "kind": "file",
      "mime_type": "application/json",
      "schema": { "fields": ["title", "url", "department"] },
      "row_count": 12,
      "download_url": "/v1/runs/<run_id>/artifacts/extracted_rows.json"
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `name` | string | Stable identifier (`extracted_rows`, `leads.csv`, `extracted_rows.csv`, `extracted_rows.json`). |
| `kind` | string | `structured_data` (inline rows) or `file` (downloadable via `download_url`). |
| `mime_type` | string | Content type — for `structured_data`, always `application/json`; for `file`, the on-the-wire MIME of the served file. |
| `schema.fields` | string[] | (where applicable) Column order matching the ExtractionSchema field names. `extracted_rows.csv` uses these names as the CSV header; `leads.csv` keeps the legacy fixed columns. |
| `row_count` | int | Number of rows / lines the artifact contains. |
| `data` | object[] | (`structured_data` only) The rows themselves, keyed by schema field name. |
| `download_url` | string | (`file` only) Path to fetch the file from the artifact endpoint. Auth follows the standard `X-Mantis-Token` rules. |

The `extracted_rows` structured artifact is the canonical form — it
includes every schema field on every row even when a value is missing
(empty string). `leads.csv` is preserved as a legacy file artifact with
the historic fixed columns (`status`, `year`, `make`, ...) for callers
that depend on that shape; new code should prefer `extracted_rows.csv`
which uses the schema's actual columns.

The `artifacts` array is empty when a run produces no leads and no
schema-driven extraction rows, so consumers can always iterate it
without a `KeyError` guard.

### Custom extraction schemas ([#508](https://github.com/mercurialsolo/mantis/issues/508))

`/v1/predict` accepts an optional top-level `extraction_schema` field
describing the columns the run should extract. When set, this schema
takes precedence over any plan-derived `ObjectiveSpec` schema and is
the dict that drives `ClaudeExtractor`'s tool-use JSON schema.

```jsonc
{
  "task_suite": { ... },
  "extraction_schema": {
    "entity_name": "job",
    "fields": [
      { "name": "title",      "type": "str", "required": true,  "example": "ML Engineer" },
      { "name": "url",        "type": "str", "required": true,  "example": "https://..." },
      { "name": "department", "type": "str", "required": false },
      { "name": "location",   "type": "str", "required": false }
    ],
    "required_fields": ["title", "url"]
  }
}
```

Fields not listed here are not extracted. The schema's `fields` order
becomes the column order in `extracted_rows.csv`; missing values show
up as empty strings rather than absent keys.

### Wall-time breakdown

`summary.wall_time_breakdown` (epic [#362](https://github.com/mercurialsolo/mantis/issues/362)) reports where wall-clock seconds went, alongside the existing `cost_breakdown` for dollars. Both come from the runner — `cost_breakdown` is owned by `CostMeter`, `wall_time_breakdown` by `TimeMeter`.

Each terminal `summary` carries the same nine buckets. `step_details[i].time_breakdown` (same shape, scoped to one step) lets you pinpoint *which* step dominated a bucket — e.g. one extract step consuming 65 s of `claude_extract`.

| Bucket | What lands here |
|---|---|
| `perceive` | `env.screenshot()` capture, viewport sync. |
| `think` | Brain inference — Holo3 / Gemma4 / Claude action emission. |
| `act` | `env.step(action)` — xdotool keystroke / mouse / scroll, CDP click. |
| `settle` | Post-action wait (fixed or adaptive); page-render quiescence. |
| `claude_ground` | `ClaudeGrounding.refine_*` coordinate refinement on grounded steps. |
| `claude_extract` | `ClaudeExtractor.find_*` extraction calls. |
| `claude_verify` | Gate verification, `DynamicPlanVerifier` checks. |
| `load` | `env.reset(url)` page-load + Cloudflare wait + proxy CONNECT. |
| `overhead` | Residual — runner orchestration, retry loops, dispatch, anything not above. |

Sum of the buckets tracks `total_time_s` within ±5 %; the residual lives in `overhead`. The mantis Python client exposes a typed accessor:

```python
status = client.status(run_id)
bd = status.wall_time_breakdown()  # {} on pre-terminal runs
if bd:
    largest = max(bd, key=bd.get)
    print(f"largest bucket: {largest} ({bd[largest]:.1f}s)")
```

Pre-Phase-B summaries omit the `wall_time_breakdown` key entirely; the accessor returns `{}` so existing client code keeps working.

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
curl -X POST "https://model-qvvgkneq.api.baseten.co/production/sync/v1/chat/completions" \
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
    { "id": "holo3", "object": "model", "owned_by": "mantis" },
    { "id": "fara",  "object": "model", "owned_by": "mantis" }
  ]
}
```

See [CUA models](reference/cua-models.md) for the full list of `cua_model` values the dispatcher accepts and the action-space differences between brains.

---

## End-to-end example: 3-listing extraction

```bash
TOKEN=$(read -srp "MANTIS_API_TOKEN: " v && echo "$v")
BTKEY="$BASETEN_API_KEY"
# Baseten gateway forwards /sync/<any path> to the container. /predict is
# the legacy default route (equivalent to /sync/predict).
ENDPOINT="https://your-model.api.baseten.co/production/sync"

# 1. Launch detached run — supply your own plan_text or a micro-plan.
RESP=$(curl -fsS -X POST "$ENDPOINT/v1/predict" \
  -H "Authorization: Api-Key $BTKEY" \
  -H "X-Mantis-Token: $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "detached": true,
    "plan_text": "Extract the first 3 listings from <your URL>: year, make, model, price, phone, url.",
    "profile_id":  "smoke",
    "workflow_id": "smoke-test-v1",
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

Result shape (one row per successfully extracted listing):

```
<year> <make> <model>  — <price> — phone <phone or 'none'>
<year> <make> <model>  — <price>
<year> <make> <model>  — <price>
```

## Plan shapes — when to use which

| Use case | Recommended shape |
|---|---|
| Recurring high-volume workflow with predictable steps | Hand-author a **micro-plan** JSON, ship it in the image at `plans/<domain>/<workflow>.json`, reference via `micro` |
| Arbitrary plain-English request | `plan_text` — server decomposes it via Claude (cached after first run) |
| Ad-hoc plan you don't want baked into the image | `task_suite` (inline JSON dict) |
| Multi-task suite with `task_id` + `verify` clauses | `task_suite` or `task_file` |

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

#### Plan-level `runtime` defaults

Plans can wrap the step list in `{steps, runtime}` so they declare their own proxy / cost / time defaults without every caller remembering the right submission flags:

```jsonc
{
  "runtime": {
    "proxy_disabled": false,
    "proxy_provider": "privateproxy",
    "proxy_city": "miami",
    "max_cost": 3.0,
    "max_time_minutes": 10
  },
  "steps": [ /* … */ ]
}
```

Submission overrides win — an explicit `proxy_disabled: true` in the HTTP body beats `proxy_disabled: false` in the plan, but omitting the body field falls back to the plan default. Schema and field reference: [Plan formats → Declaring runtime defaults](getting-started/plan-formats.md#declaring-runtime-defaults-inside-the-plan).

### `task_suite` — multi-task JSON

For Claude-CUA-style autonomous-per-task workflows (the existing
`tasks/crm/crm_tasks.json` is this shape):

```jsonc
{
  "session_name": "crm_demo",
  "base_url": "https://crm.example.com",
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
  "plan_text": "Go to a marketplace listings site, filter to private sellers above $35,000 in Florida, extract listing details for the first 3 listings, save year/make/model/price/phone."
}
```

`PlanDecomposer` (Claude-backed, cached by signature) converts this into a micro-plan and proceeds. Decomposition costs ~$0.10 the first time per unique plan text; subsequent runs hit the cache.

---

## Pricing (verified end-to-end)

Real numbers from a 3-listing marketplace-extraction run on Baseten:

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
| Per-tenant browser profile | Mounted at `/workspace/mantis-data/tenants/<tenant_id>/chrome-profile/<profile_id>/` — cookies cannot bleed across tenants |
| Per-tenant run state | Same volume layout — `profile_id` / `workflow_id` / legacy `state_key` are all server-prefixed so callers cannot read another tenant's checkpoint |
| Plan injection (e.g., `loop_count: 999_999`) | Server-side hard caps clamp the values; oversized plans are rejected with `400` |
| Upstream credential leak | `/v1/chat/completions` strips `X-Mantis-Token`, `Authorization`, `Cookie` before forwarding to in-pod llama.cpp |

## Limits / caveats

- **Detached runs survive replica restart** (state on the data volume) but only on the same Baseten model. Cross-region failover not supported.
- **`/v1/chat/completions`** is unstreamed in v1. Streaming SSE is a Tier 2 follow-up.
- **Single Anthropic-key per tenant** at request time (re-resolved on every call).

## Pause / resume

> OTP / 2FA / human-in-the-loop confirmation — [#344](https://github.com/mercurialsolo/mantis/issues/344).

When a plan hits an auth wall the agent can't get past on its own — an OTP code, a 2FA push, an explicit "yes, refund this" confirmation — a registered host tool raises [`PauseRequested`](reference/glossary.md). The runner snapshots its state, writes `pause_state.json` to the run dir, and flips the run's status to `paused`. The caller polls status, surfaces the prompt to a human (or fetches the code from a side channel), then resumes with the answer.

A default `request_user_input` host tool is registered on every detached `/v1/predict` run. Brains that emit `Action(TOOL_CALL, name="request_user_input", params={"prompt": "..."})` will pause the run on the first call and receive the caller's `user_input` on the second (after resume).

### The `plan_text` hand-over pattern (ask the user mid-run)

You don't drive steps yourself — submit a single `plan_text` (detached) and the decomposer breaks it into a `MicroPlan` and runs it step by step. To get a human hand-over, **write the hand-over into the plan in plain English** — phrase it as "ask the user … and wait for the answer, then use that answer to …". The decomposer emits a `request_user_input` step followed by a step that references the answer through the `{{user_input}}` token.

Example plan:

> "Go to news.ycombinator.com, read the top 3 story titles, ask the user which story title to open and wait for the answer, then open that story and report its title."

Decomposes to a 5-step plan:

```
[0] navigate           → news.ycombinator.com
[1] extract_data       → top 3 story titles
[2] request_user_input → pauses; prompt shown to you
[3] click {{user_input}}  → the answer is substituted in verbatim
[4] extract_data       → the opened story title
```

Step `[2]` is the hand-over: the run flips to `status=paused` and surfaces a `prompt`. You resume with `action=resume` + `user_input`, and the value is **substituted verbatim into every `{{user_input}}` token** on the remaining steps (intent + string params) before they execute. End to end:

```bash
# 1. submit (detached) — note the answer is used as a CLICK TARGET, so the
#    prompt should ask for something usable as one (a title/label, not "1/2/3").
RUN_ID=$(curl -s "$ENDPOINT/v1/predict" -H "X-Mantis-Token: $TOKEN" \
  -d '{"plan_text":"Go to news.ycombinator.com, read the top 3 story titles, ask the user which story title to open and wait for the answer, then open that story and report its title.","detached":true}' \
  | jq -r .run_id)

# 2. poll until paused, read the prompt
curl -s "$ENDPOINT/v1/predict" -H "X-Mantis-Token: $TOKEN" \
  -d "{\"action\":\"status\",\"run_id\":\"$RUN_ID\"}" | jq '{status, prompt, reason}'
# → { "status": "paused", "prompt": "Which story title should I open?", "reason": "user_input" }

# 3. resume with the human's answer → substituted into {{user_input}} on step [3]
curl -s "$ENDPOINT/v1/predict" -H "X-Mantis-Token: $TOKEN" \
  -d "{\"action\":\"resume\",\"run_id\":\"$RUN_ID\",\"user_input\":\"Show HN: my side project\"}"
# → { "status": "running", ... }

# 4. keep polling until terminal (succeeded / failed)
```

**Use the answer verbatim.** Whatever string you send as `user_input` is what replaces `{{user_input}}` — so phrase the plan's question so the answer is directly usable by the next step (a concrete title or on-screen label for a `click`, a code for an OTP field, etc.). An ordinal like "2" will be clicked literally as the text "2".

> **Troubleshooting — run never pauses, logs say `request_user_input step N: no host tool registered; downgrade to skip`.** This means the deployment is running **code older than [#883](https://github.com/mercurialsolo/mantis/issues/883)** — the `request_user_input` step is downgraded to a skip, `{{user_input}}` is never filled, the run finishes `completed_with_failures`, and `action=resume` is rejected with `requires a paused run`. The log line is byte-identical between the old and fixed handlers, so it does **not** tell you which code is live. Fix: redeploy the app from current `main` (`modal app stop <app>` then `modal deploy deploy/modal/modal_mantis_server.py` for the `mantis-server` endpoint). Pause/resume + `{{user_input}}` substitution require #883 (wiring), #885 (resume staging), and #887 (initial-path staging) — all on `main` as of 2026-06-13.

### Status poll on a paused run

```jsonc
POST /v1/predict
{"action": "status", "run_id": "20260513_180527_abc"}

→ {
  "status":       "paused",
  "run_id":       "20260513_180527_abc",
  "prompt":       "Enter the 6-digit code from your authenticator",
  "reason":       "user_input",
  "pause_state":  { /* opaque PauseState blob — hand back on resume */ }
}
```

`pause_state` is opaque — the server is the only thing that interprets it. Treat it as a token: store it if you want, but you don't need to send it back yourself. The server already has the canonical copy on disk under the run_id; it round-trips automatically on resume.

#### What's captured at pause time

| Captured | Restored on resume | Notes |
|---|---|---|
| Step index + plan signature | ✓ — runner picks up at the next un-run step | Round-trips via `pause_state.step_index` + `plan_signature`. |
| Step results so far | ✓ — replayed into the runner state | Lets `_handle_success` / dedup logic see prior outputs. |
| Pending tool call (`pending_tool` + `pending_arguments` + `prompt`) | ✓ — the resumed runner re-invokes the tool with `user_input` set | The mechanism that lets a paused tool finish its `call_tool` round-trip. |
| **URL + scroll + viewport** (`browser_state`, [epic #358](https://github.com/mercurialsolo/mantis/issues/358) Phase A) | ✓ — agent re-lands on the exact pixel | CDP-captured (`location.href`, `window.scrollX/Y`, `window.innerWidth/Height`) just before pause raises. Empty when the env doesn't expose CDP (legacy adapters). |
| Cookies / localStorage / IndexedDB | ✓ — but via the `profile_id` Chrome user-data-dir, not `pause_state` | Persists across runs on its own; `profile_id` is the identity that scopes the dir. |
| **Unsubmitted form input** (`browser_state.form_state`, [Phase B (#360)](https://github.com/mercurialsolo/mantis/issues/360)) | ✓ — half-filled inputs / selects / checkboxes / radios / contenteditable repopulate on resume | Keyed by stable selector (`data-*` > `id` > short CSS path). Passwords masked: the selector is kept so the caller knows *which* field to re-prompt, but the value is dropped before serialization (opt in via `MANTIS_PAUSE_CAPTURE_PASSWORDS=1` for test/debug only). Missing selectors on the resumed page are silently skipped. |
| In-memory JS state (React/Redux store, in-flight network) | ✗ — fresh page load | Container-level snapshots would be the right answer; out of scope. |

### Resume

```jsonc
POST /v1/predict
{"action": "resume", "run_id": "20260513_180527_abc", "user_input": "123456"}

→ {"status": "running", "run_id": "20260513_180527_abc", "resumed_at": "2026-05-13T18:09:12Z"}
```

The server rehydrates the stored `PauseState`, calls `runner.resume(state, user_input=...)` against the same `profile_id` / `workflow_id` the original run used, and continues from the paused step. Subsequent `action=status` polls return `running` until the run reaches a terminal status (`succeeded` / `failed` / `cancelled`) — or pauses again, in which case the cycle repeats with a fresh prompt.

### Error cases

| Status | Cause |
|---|---|
| `400 action='resume' requires user_input` | Missing the `user_input` field |
| `400 action='resume' requires a paused run` | Run isn't currently in `paused` status (succeeded, running, cancelled, ...) |
| `400 plan signature mismatch on resume` | Disk-stored `pause_state.plan_signature` doesn't match the current plan derived from the stored payload — usually means someone edited the on-disk state |
| `404 unknown run_id` | No `status.json` for that `run_id` on this tenant |

### Deployment coverage

| Deployment | Pause / resume |
|---|---|
| **Baseten** (`/v1/predict` via `BasetenCUARuntime`) | ✅ All brains — Holo3, Claude, EvoCUA, OpenCUA, Gemma4-CUA. Default `request_user_input` tool registered on every detached run ([#344](https://github.com/mercurialsolo/mantis/issues/344)). |
| **Modal `mantis-server`** (`<workspace>--mantis-server-api.modal.run`, `deploy/modal/modal_mantis_server.py`) | ✅ Serves the **same** `baseten_server` FastAPI app as Baseten, so the `plan_text` / `micro` / `task_suite` paths all register the default `request_user_input` tool and pause/resume identically. This is the endpoint to use for the `plan_text` hand-over pattern above. **Must be deployed from `main` ≥ #883** — older deploys skip the step (see the troubleshooting note above). |
| **Modal `mantis-cua-server`** (`<workspace>--mantis-cua-server-api.modal.run`, `deploy/modal/modal_cua_server.py`) | ✅ Holo3 micro path — constructs `MicroPlanRunner` directly. Claude / EvoCUA / OpenCUA / Gemma4-CUA on this app go through `task_loop.run_executor_lifecycle` and don't currently surface paused state ([#347](https://github.com/mercurialsolo/mantis/issues/347)). |
| **Modal `local_entrypoint`** (CLI: `modal run ...`) | ❌ Not wired. Use an HTTP endpoint or embed the library. |
| **Library-embedded** (`MicroPlanRunner` / `GymRunner` direct) | ✅ Always — pause/resume is a property of the runner. The HTTP surfaces above are wrappers on top. |

### Library-embedded integrations

If you embed `MicroPlanRunner` / `GymRunner` directly (no HTTP), the same primitives are available in-process via `runner.run_with_status(plan)` returning `RunnerResult(paused=True, pause_state=...)`, plus `runner.resume(state, user_input=..., plan=plan)`. See [Embedding MicroPlanRunner](integrations/embedding-microplanrunner.md) for the canonical walkthrough — the HTTP surface is just a wrapper on top of the same library API.

## Screencast / video recording

Send a plan with `record_video: true` and the runtime produces a feature-walkthrough video — title card → captioned run footage → outro card with the result summary. Fetch with `GET /v1/runs/{run_id}/video`. The raw screencast is preserved alongside; pass `?raw=1` to fetch it instead.

The walkthrough has three segments plus animated click ripples on top of the run footage:

```
┌─────────────────┐  ┌─────────────────────────┐  ┌─────────────────┐
│  Title card     │→ │  Run footage (captions  │→ │  Outro card     │
│  (3s)           │  │   + click ripples)      │  │  (5s)           │
│                 │  │  per-step intent shown  │  │                 │
│  Mantis CUA     │  │  with [OK] / [FAIL]     │  │  Run complete   │
│  ───            │  │  in the bottom strip    │  │  ───            │
│  <plan name>    │  │  while the action plays │  │  3 viable leads │
│  tenant: …      │  │  + expanding sky-blue   │  │  1 with phone   │
│  run: …         │  │  ripple at every click  │  │  17 steps · 9m  │
│                 │  │                         │  │  cost: $0.42    │
└─────────────────┘  └─────────────────────────┘  └─────────────────┘
```

Title and outro are rendered with PIL. Captions are SRT cues burned in by ffmpeg's `subtitles=` filter (libass). Click ripples are PNG-sequence overlay frames composited via ffmpeg's `overlay` filter. Polish is best-effort — if anything fails (PIL, ffmpeg, libass not built in the image), the raw recording is still saved and the endpoint serves it.

### Action overlays — universal computer use

Every kind of agent action gets a visual cue, regardless of what application is in focus (browser, file manager, terminal, dialogs, anything visible on the Xvfb display). The agent emits actions with pixel coordinates / key chords / text, and the overlay renderer composites the matching visual onto the recording.

| Agent action | Overlay |
|---|---|
| `CLICK` (single) | Sky-blue expanding ripple at (x, y), 0.6 s, fades out |
| `DOUBLE_CLICK` | Same as click + a second offset ring 0.1 s later |
| `KEY_PRESS` (e.g. `Ctrl+S`, `Tab`, `Enter`) | Slate badge in the bottom-right with the chord text, 1.5 s, slide-in then fade |
| `TYPE` (typed text) | "⌨ Typing: \"…\"" caption near the top, 1.8 s, fades after text appears on screen |
| `SCROLL` (`up` / `down` / `left` / `right`) | Sky-blue arrow at the matching screen edge, slides in the scroll direction, 0.8 s |
| `DRAG` | Animated trail line from start to end with a moving head dot, 0.9 s |
| `WAIT`, `NAVIGATE`, `DONE` | No overlay (no useful visual locus) |

All overlays are deliberately minimal — visible without being disruptive. Sky-blue accent color across the set so they read as a single visual language.

You'll see counts in the result metadata under `video.actions`:

```jsonc
{
  "video": {
    "path": ".../recording.mp4",
    "polished_path": ".../recording_polished.mp4",
    "actions": {
      "clicks": 17,
      "keys":   3,
      "types":  2,
      "scrolls": 8,
      "drags":  0
    },
    "clicks": 17,    // backwards-compat field
    ...
  }
}
```

```bash
# 1. Submit a recorded run
RESP=$(curl -fsS -X POST "$ENDPOINT/v1/predict" \
  -H "Authorization: Api-Key $BTKEY" \
  -H "X-Mantis-Token: $TOK" \
  -H "Content-Type: application/json" \
  -d '{
    "detached": true,
    "micro": "plans/example/extract_listings.json",
    "profile_id":  "demo-recording",
    "workflow_id": "demo-recording-v1",
    "max_cost": 2,
    "max_time_minutes": 20,
    "record_video": true,
    "video_format": "mp4",
    "video_fps": 8
  }')
RUN_ID=$(echo "$RESP" | jq -r .run_id)

# 2. Poll status until succeeded ... (same as the regular flow)

# 3. Download the screencast
curl -fsS -o demo.mp4 \
  -H "X-Mantis-Token: $TOK" \
  "$ENDPOINT/v1/runs/$RUN_ID/video"
```

Result-side metadata (in the `summary` block):

```jsonc
{
  "video": {
    "path": "/workspace/mantis-data/tenants/<tenant>/runs/<run_id>/recording.mp4",
    "polished_path": "/workspace/mantis-data/tenants/<tenant>/runs/<run_id>/recording_polished.mp4",
    "format": "mp4",
    "duration_seconds": 567.3,
    "bytes": 31457280,
    "error": null
  }
}
```

`polished_path` is set only when the post-process compose step succeeded; on failure it's omitted and the endpoint falls back to the raw recording.

### Endpoint behavior

| Request | Returns |
|---|---|
| `GET /v1/runs/{run_id}/video` | Polished mp4 (preferred) → raw mp4 (fallback) → 404 |
| `GET /v1/runs/{run_id}/video?raw=1` | Raw mp4 only → 404 |

### Format tradeoffs

| Format | Container | Encode cost | Output size (typical 10-min run) | Best for |
|---|---|---|---|---|
| `mp4` | H.264 (libx264, `ultrafast` preset, CRF 28) | low | ~30–80 MB | sharing, downloads |
| `webm` | VP9 (libvpx-vp9, cpu-used 5, CRF 32) | medium | ~25–60 MB | embedding in web pages |
| `gif` | palettegen + paletteuse | high | ~50–200 MB | docs, Slack, animated thumbnails (lossy) |

For long recordings or tight bandwidth, prefer `mp4` at 5 fps. The `gif` path uses a palette-aware filtergraph but file size grows fast — use only for short demos (< 60 s).

### Operational caveats

- The container image must have `ffmpeg` installed. Both `docker/server.Dockerfile` and `deploy/baseten/holo3/config.yaml` ship it; if you're rolling your own image, add `ffmpeg` to the apt deps. Without ffmpeg, `record_video: true` is a soft-fail — the run completes normally, and the response carries `video.error: "ffmpeg-not-installed"`.
- `video.error` envelopes you may see (soft-fail in every case — the run itself never fails because recording couldn't start):
  - `ffmpeg-not-installed` — ffmpeg isn't on PATH inside the container.
  - `x-display-not-ready:<display>` — the Xvfb display ffmpeg targets wasn't up when the recorder fired. The runtime now calls `env.ensure_display_ready()` before spawning the recorder, so this error only appears when a custom image hasn't installed `xvfb` / `xdpyinfo` or when a third-party caller spawns `ScreenRecorder` directly outside the standard runtime. Fix: install `xvfb` + `xdpyinfo` (apt: `xvfb x11-utils`) in your image.
  - `ffmpeg-startup-failed:<stderr>` — ffmpeg exited within ~300 ms of spawn. The trailing stderr blob is the actionable part; common cause was historically `Cannot open display :99, error 1` (fixed by the display-ready probe above).
  - `empty-output` — ffmpeg started and stopped cleanly but wrote a zero-byte file. Usually means the run completed before any frames captured.
  - `spawn-failed:<oserror>` — the Popen itself raised (e.g., process budget exhausted).
- Recordings live at `$MANTIS_DATA_DIR/tenants/<tenant_id>/runs/<run_id>/recording.<fmt>` so tenants cannot read each other's files. The download endpoint uses the authenticated tenant's dir; even if you guess another tenant's `run_id`, the file lookup is scoped.
- `video_fps` is clamped to `[1, 30]`. Higher fps doesn't help much (UI rarely changes faster than 5–10 fps) and bloats the file.
- Each second of recording is ~50 KB at 5 fps mp4. Multiply by your target run duration + tenant count to size the EFS / Filestore.

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
  "tenant_id": "tenant_a",
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

Wildcards: `*.example.com` matches any subdomain but not `example.com.evil.com`. Empty `allowed_domains` (the default) skips this check.

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

See [Architecture](architecture.md) for the bigger architectural picture.
