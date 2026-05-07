# Environment variables

Reference for every server-side env knob. Set on the container (Baseten / k8s Deployment / `docker run -e ...`) — never in client-side code.

## Auth

| Var | Default | Effect |
|---|---|---|
| `MANTIS_API_TOKEN` | unset | Single-tenant mode: any caller with this token gets `DEFAULT_TENANT` permissions. Ignored if `MANTIS_TENANT_KEYS_PATH` is set. |
| `MANTIS_TENANT_KEYS_PATH` | unset | Path to JSON keys file. When set, the server runs in multi-tenant mode. |

## Caps

| Var | Default | Effect |
|---|---|---|
| `MANTIS_MAX_STEPS_PER_PLAN` | 200 | Reject plans larger than this with 400 |
| `MANTIS_MAX_LOOP_ITERATIONS` | 50 | Silently clamp `loop_count` in micro-plans |
| `MANTIS_MAX_RUNTIME_MINUTES` | 60 | Hard wall-time cap on every run |
| `MANTIS_MAX_COST_USD` | 25.0 | Hard cost cap on every run |

These are global hard caps; tenant config can be tighter, never looser.

## Paths

| Var | Default | Effect |
|---|---|---|
| `MANTIS_DATA_DIR` | `/workspace/mantis-data` | Top-level data volume. Per-tenant subtree at `tenants/<tenant_id>/`. |
| `MANTIS_REPO_ROOT` | `/workspace/cua-agent` | Where `task_file` / `micro` paths are resolved from. |
| `MANTIS_DEBUG_DIR` | `<MANTIS_DATA_DIR>/screenshots/claude_debug` | Where Claude extraction prompt + screenshot debug bundles land. |
| `MANTIS_IDEMPOTENCY_DIR` | `<MANTIS_DATA_DIR>/idempotency` | Sidecar files for idempotency cache. |
| `MANTIS_CHROME_PROFILE_DIR` | set per-request by handler | Chrome profile dir used by the Xvfb env. The handler overrides this per tenant + state_key. |

## Inference

| Var | Default | Effect |
|---|---|---|
| `MANTIS_LLAMA_PORT` | 18080 | Internal port the in-pod llama.cpp server binds to. The `/v1/chat/completions` proxy forwards here. |
| `MANTIS_BRAIN` | `holo3` | Brain backend selector. One of `holo3`, `claude`, `opencua`, `llamacpp`, `gemma4`, `agent-s`. Wins over the legacy `MANTIS_MODEL`. |
| `MANTIS_MODEL` | (set by Truss) | Legacy alias of `MANTIS_BRAIN` for one minor release. `gemma4-cua` aliases to `gemma4`. |
| `MANTIS_HOLO3_MODEL_DIR` | `/models/holo3` | Where Holo3 GGUF weights are mounted. |
| `ANTHROPIC_API_KEY` | unset | Default Anthropic key. Per-tenant `anthropic_secret_name` overrides per request. |
| `MANTIS_PROMPTS_DIR` | unset | Override directory for prompt files. When set, the loader reads `<dir>/<name>.txt` before falling back to the in-tree constant — lets a tenant tune wording without forking the wheel. Names: `system_v1`, `gemma4_system`, `holo3_system`, `claude_system`, `opencua_system`, `llamacpp_system`. |

## Proxy (IPRoyal)

| Var | Default | Effect |
|---|---|---|
| `PROXY_URL` | unset | `host:port` for the upstream IPRoyal proxy |
| `PROXY_USER` | unset | session id |
| `PROXY_PASS` | unset | password |
| `MANTIS_PROXY_CITY` | unset | Default proxy geo override (caller can override per request) |
| `MANTIS_PROXY_STATE` | unset | Same |

## Webhooks

| Var | Default | Effect |
|---|---|---|
| `MANTIS_WEBHOOK_SECRET_DEFAULT` | unset | Fallback HMAC signing secret when a tenant's `webhook_secret_name` doesn't resolve |

## Cost model (#122)

| Var | Default | Effect |
|---|---|---|
| `MANTIS_COST_GPU_HOURLY_USD` | `3.25` | GPU compute, $/hour. Used by `CostConfig.gpu_cost`. |
| `MANTIS_COST_CLAUDE_CALL_USD` | `0.003` | Per-Claude-API-call rate. Multiplied by `claude_extract` + `claude_grounding` counters. |
| `MANTIS_COST_PROXY_PER_GB_USD` | `5.00` | Egress proxy bandwidth $/GB. |
| `MANTIS_COST_GPU_SECONDS_PER_STEP` | `3.0` | Per-step GPU seconds when the runner doesn't measure exact wall time. |
| `MANTIS_COST_PROXY_MB_PER_NAV` | `5.0` | Estimated proxy MB per page load. |
| `MANTIS_COST_PROXY_MB_PER_SCROLL` | `0.5` | Estimated proxy MB per scroll. |

See [operations/cost.md](../operations/cost.md) for the full rate-tuning workflow.

## Trace export (#155)

| Var | Default | Effect |
|---|---|---|
| `MANTIS_TRACE_EXPORT_DIR` | unset | Enable per-run trace export. When set, every completed / halted / cancelled / paused run writes `<dir>/<tenant_id>/<run_id>.json` with the full step list, costs, status, and predicted/observed outcomes. Empty `tenant_id` falls back to `__shared__/`. Off by default — feature flag for the continual-fine-tuning pipeline. |
| `MANTIS_TRACE_INCLUDE_SCREENSHOTS` | unset | When truthy (`1`/`true`/`yes`/`on`) and trace export is enabled, also persists per-step PNG screenshots to `<dir>/<tenant_id>/<run_id>_screens/<step:04d>.png`. Default off because screenshot bytes ~100× the on-disk trace size. |

## Logging

| Var | Default | Effect |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Standard Python logging level |
| `MANTIS_LOG_FORMAT` | `json` | `json` (default) emits one-line JSON per record with `tenant_id` enrichment; `plain` reverts to ad-hoc format |

## Context (set per request, not per deployment)

The handler sets these on every `/v1/predict` so downstream code (the runtime, the JSON log formatter) can read them via `os.environ`. **Don't rely on them being set at deployment time.**

- `MANTIS_TENANT_ID` — current request's tenant id
- `MANTIS_CHROME_PROFILE_DIR` — per-tenant per-state_key Chrome profile dir for this run

## See also

- [Operations / Tenant keys](../operations/tenant-keys.md) — multi-tenant config
- [Hosting](../hosting/index.md) — platform-specific deploy paths
