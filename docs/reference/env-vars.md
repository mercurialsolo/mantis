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
| `MANTIS_BRAIN` | `holo3` | Brain backend selector. One of `holo3`, `claude`, `opencua`, `llamacpp`, `gemma4`, `agent-s`, `mock`. Wins over the legacy `MANTIS_MODEL`. `mock` is a deterministic always-DONE stub for plan authoring without GPU / API cost (#274). |
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

## Runner / verification

| Var | Default | Effect |
|---|---|---|
| `MANTIS_PREDICATE_VERIFY` | `enabled` | Per-step world-model verification (#291). When the brain emits a structured prediction (`{"expected": [...]}` or `Predicted: ...`), the runner parses, evaluates, and writes per-predicate booleans into the trajectory plus a `world_model_error` reward component. Set to `disabled` to ablate — `predicted_outcome` is still recorded for distillation, but no evaluation runs. See [Predicate grammar](predicates.md). |
| `MANTIS_DONE_GATE` | `enabled` | Deterministic done-acceptance gate (#303). Runs cheap predicates (empty summary, plan steps incomplete, pending form values, etc.) before the model-based `verify_done`. Set to `disabled` to ablate — the runner falls through to the existing model verifier and `done_rejections_by_reason` stays empty. See [Done-acceptance gate](done-gate.md). |
| `MANTIS_FORM_CONTROLLER` | `enabled` | First-class runtime form controller (#301) owning pending-values / used-regions / submit-latch state. Set to `disabled` to ablate — the runner falls back to the legacy scattered `force_fill_*` locals; `runner.form_controller` is `None`. See [Form controller](form-controller.md). |
| `MANTIS_ADAPTIVE_SETTLE` | `enabled` | Replaces post-action `time.sleep(settle_time)` (#294) with a frame-stability gate (xdotool path) or `wait_for_load_state("networkidle")` gate (Playwright path), capped at the legacy budget. Set to `disabled` to ablate — both gates short-circuit back to a fixed sleep without a redeploy. See [Adaptive settle](adaptive-settle.md). |
| `MANTIS_CHROME_REUSE` | `enabled` | Container-scoped Xvfb + Chrome session reuse (#311). Successive `/v1/cua` requests with the same `(profile_dir, proxy_key)` reuse the live browser instead of paying the ~10 s launch tax. Set to `disabled` to ablate. Per-request opt-out: `payload["reuse_session"]=false`. See [Chrome session reuse](chrome-session-reuse.md). |
| `MANTIS_SPECULATIVE_INFERENCE` | `disabled` | Wraps the inner brain in `SpeculativeBrain` (#118) so `think()` overlaps with the post-action settle. Default OFF because the E2E ablation on Holo3 Q8 + single-llama.cpp showed a wall-time regression (GPU contention between speculative + sync requests, 55.6% hit rate → +52% wall). Quality is preserved by the strict validator. Enable on multi-GPU backends where the two `think()` requests don't serialize. See [Speculative inference](speculative-inference.md). |
| `MANTIS_PERCEPTUAL_VERIFY` | `enabled` | Perceptual-diff verifier (#293) for high-risk actions (submit, confirm, buy, send, delete, login, save). Compares pre/post frame hashes — both global and a 200×200 region around the click — and emits `action_effect_observed: bool` per step. WARNING line injected into next step's feedback on no-effect. Observational only — never blocks or substitutes the action. Set to `disabled` to ablate. See [Perceptual diff verifier](perceptual-diff.md). |
| `MANTIS_LOOP_RECOVERY` | `enabled` | Action-class-transition policy (#302) that forces TAB / TYPE / RETURN when the brain loops on a no-effect click. Runs after the existing substitution chain (force-fill, force-submit, claude-director, top-click-guard) — the last gate before dispatch. Per-reason count surfaces on `RunResult.loop_recoveries_by_reason`. Set to `disabled` to ablate. See [Loop recovery policy](loop-recovery.md). |

## API documentation surface

| Var | Default | Effect |
|---|---|---|
| `MANTIS_ENABLE_DOCS_UI` | `1` | Serve `/docs` (Swagger) and `/redoc` (Redoc) over the FastAPI app. Set to `0` / `false` / `no` / `off` on production tenant fleets that don't want the interactive UIs exposed publicly. `/openapi.json` is served regardless. |
| `MANTIS_GIT_SHA` | unset | Surfaced verbatim in `GET /v1/version` so clients can pin to a specific build. Typically populated by the deploy pipeline. |
| `MANTIS_BUILD_TIME` | unset | Surfaced verbatim in `GET /v1/version`. Populated by the deploy pipeline. |

## Context (set per request, not per deployment)

The handler sets these on every `/v1/predict` so downstream code (the runtime, the JSON log formatter) can read them via `os.environ`. **Don't rely on them being set at deployment time.**

- `MANTIS_TENANT_ID` — current request's tenant id
- `MANTIS_CHROME_PROFILE_DIR` — per-tenant per-state_key Chrome profile dir for this run

## See also

- [Operations / Tenant keys](../operations/tenant-keys.md) — multi-tenant config
- [Hosting](../hosting/index.md) — platform-specific deploy paths
