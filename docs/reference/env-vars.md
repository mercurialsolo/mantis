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
| `MANTIS_MODEL` | (set by Truss) | Model kind: `holo3`, `gemma4`, `evocua-32b`, etc. |
| `MANTIS_HOLO3_MODEL_DIR` | `/models/holo3` | Where Holo3 GGUF weights are mounted. |
| `ANTHROPIC_API_KEY` | unset | Default Anthropic key. Per-tenant `anthropic_secret_name` overrides per request. |

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
