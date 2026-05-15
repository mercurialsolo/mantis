# Modal

Modal is the easiest path for ad-hoc / batch runs that benefit from scale-to-zero. `deploy/modal/modal_cua_server.py` exposes two surfaces on the same app:

1. **HTTP API** ([#342](https://github.com/mercurialsolo/mantis/issues/342)) — `@modal.asgi_app()` at `https://<workspace>--mantis-cua-server-api.modal.run`, mirroring the Baseten `/v1/predict` shape. Tenant-keyed via `X-Mantis-Token`; the API container dispatches via `.spawn()` to the GPU executors and rehydrates `modal.FunctionCall.from_id(...)` on status polls.
2. **`local_entrypoint`** — one-off CLI runs (`modal run deploy/modal/...`) for ad-hoc debugging.

## Prerequisites

```bash
pip install modal
modal token new   # authenticates this machine to your Modal workspace
```

You'll also need an `.env` file at the repo root with the same five secrets Baseten uses. Modal's `Secret.from_dotenv()` picks them up at deploy time.

## Deploy

```bash
uv run modal deploy deploy/modal/modal_cua_server.py
```

This creates the executor functions (`run_holo3`, `run_fara`, `run_claude_cua`, `run_cua_*`, `run_gemma4_cua`) plus the web endpoint `api`. Each executor is its own image (Chrome + xdotool + the brain's runtime); the api image is lightweight (FastAPI + pydantic only). Modal scales each independently.

`run_fara` serves Microsoft's Fara-7B (Qwen2.5-VL based, MIT-licensed) via vLLM on a single A100-40GB. Pick it with `cua_model="fara"` — the brain implements the same interface as Holo3, so Hybrid and passthrough flows both work without further wiring. See [CUA models](../reference/cua-models.md) for the action-space caveats (no `double_click` / `right_click` / `drag`).

### Warm-container caveat after a code change

`modal deploy` registers a new revision (deploys take ~2-3 s when only Python source changed) but the previous container stays warm until `scaledown_window` elapses — 10 minutes by default. New requests can land on the old container, which is still serving from `sys.modules` baked at the previous import, so a freshly-deployed code change may not appear to take effect for up to 10 minutes.

When verifying a fix end-to-end (e.g. redeploy + plan rerun), force a cold start so the new image is loaded immediately:

```bash
uv run modal app stop mantis-server --yes
uv run modal deploy deploy/modal/modal_mantis_server.py
```

Then submit the verification request. Without this, you may see the **pre-fix** behaviour despite a green deploy — caught during the [#392](https://github.com/mercurialsolo/mantis/pull/392) recorder-race verification, where three back-to-back deploys appeared to do nothing until the warm container was killed.

## Submit a plan over HTTP (recommended)

```bash
curl -X POST https://workspace--mantis-cua-server-api.modal.run/v1/predict \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "detached": true,
    "micro": "plans/example/extract_listings.json",
    "profile_id":  "marketplace-prod",
    "workflow_id": "marketplace-listings-v1",
    "cua_model":   "holo3",
    "max_cost": 2,
    "max_time_minutes": 20
  }'
```

You get back `{"run_id": "...", "status": "queued", ...}` immediately. Poll with `{"action": "status", "run_id": "..."}` until terminal. See [API / Pause / resume](../api.md#pause-resume) for the `paused` / `action=resume` shape ([#347](https://github.com/mercurialsolo/mantis/issues/347) wires it through the Modal endpoint).

`profile_id` and `workflow_id` are split fields since [#341](https://github.com/mercurialsolo/mantis/issues/341) — see [Concepts](../getting-started/concepts.md#profile_id-workflow_id-the-resume-primitives-341). The Modal endpoint enforces a **per-profile lock**: two concurrent runs against the same `profile_id` get a 409 with the held `run_id` instead of silently corrupting Chrome's user-data-dir ([#342](https://github.com/mercurialsolo/mantis/issues/342)).

## Submit via the `local_entrypoint` CLI (one-off / debugging)

```bash
uv run modal run --detach deploy/modal/modal_cua_server.py \
  --micro plans/example/extract_listings.json \
  --model holo3 \
  --max-cost 2 \
  --max-time-minutes 20 \
  --profile-id marketplace-prod \
  --workflow-id marketplace-listings-v1 \
  --session-name marketplace-smoke
```

`--detach` returns immediately; the run continues in Modal. Each run gets its own GPU container, scales to zero when done. Useful when you don't want to mint a tenant token just to kick off a debugging run.

## Inspecting a running app

```bash
modal app list
modal app logs <app-id>
```

Results land on the `osworld-data` Modal volume:

```bash
modal volume ls osworld-data results
modal volume get osworld-data results/holo3_results_*.json local_results/
```

Per-run state for HTTP submissions lives at `/data/tenants/<tenant_id>/runs/<run_id>/`:
- `status.json` — terminal status, prompt/reason if paused, `modal_call_id` for hydration
- `result.json` — final result envelope
- `pause_state.json` — opaque PauseState blob (when paused)
- `task_suite.json` — snapshot of the dispatched task_suite (used on resume)
- `viewer.json` — MJPEG tunnel URL when `live_viewer: true` was set; merged into `action=status` responses as `viewer_url` (kept as a side-channel file so the executor never races the API on `status.json`)
- `events.log` — append-only event stream

Chrome user-data-dirs are kept per tenant + profile at
`/data/tenants/<tenant_id>/chrome-profile/<profile_id>/`. Cookies,
localStorage and IndexedDB persist across runs that share a
`profile_id` (this is what makes `profile_id` "sticky") and are
fully isolated between different `profile_id`s — including across
runs that happen to land on the same warm Modal container.

## Smoke test

```bash
bash scripts/verify_modal_luma_curl.sh
```

Eight-check curl-only E2E — health, parallel-dispatch via distinct `profile_id`s, 409 on collision, status polling, cancel + lock-release, plus three `action=resume` validation paths. No Python deps — drop-in for CI or live debugging.

## When to choose Modal over Baseten

| Pick Modal if… | Pick Baseten if… |
|---|---|
| You want true scale-to-zero (no idle GPU) | You want a stable HTTPS endpoint other services can call |
| Bursty / batch workloads | Steady traffic |
| You want a tenant-keyed HTTP surface AND scale-to-zero | You need Prometheus metrics, idempotency-key cache, webhook fan-out on terminal status |
| You're comfortable with Modal's cold-start (~30-60s) | You need consistently sub-second submit latency |

The Modal HTTP endpoint covers `POST /v1/predict` (submit, `action=status|result|cancel|resume`), `GET /v1/health`, and `GET /v1/models`. **Not yet ported from Baseten**: Idempotency-Key cache, webhook delivery, Prometheus `/metrics`, video streaming, rate limits, concurrency caps — all tracked under [#342](https://github.com/mercurialsolo/mantis/issues/342) Phase 2.

Pause / resume coverage caveat: the Modal endpoint supports `action=resume` on the **Holo3** executor only (it's the only Modal executor that uses `MicroPlanRunner` directly). Claude / EvoCUA / OpenCUA / Gemma4-CUA on Modal go through `task_loop.run_executor_lifecycle` and don't currently surface paused state. Baseten supports pause/resume across all brains. See [#347](https://github.com/mercurialsolo/mantis/issues/347).

## Browser-side runner: `mantis-plan-runner`

The `modal_cua_server.py` app above hosts the **brain** (the Holo3
GGUF + the FastAPI surface). For plans that need the **browser** to
run inside Modal too — the canonical case is a Cloudflare-protected
target where the local headless browser fails detection — there's a
sister app at `deploy/modal/modal_plan_runner.py`:

```
uv run modal deploy deploy/modal/modal_plan_runner.py
```

That image bakes in `xvfb`, `xdotool`, `scrot`, `chromium`, plus
`openbox` + `tint2` for a normal-desktop window-manager context, plus
the `mantis_agent` package and a `tinyproxy` auth-holder. The
deployed app exposes a single `run_plan` function that constructs
`XdotoolGymEnv` + `Holo3Brain` + `ClaudeGrounding` + `ClaudeExtractor`
internally, runs `MicroPlanRunner`, and returns the same result
payload the local `mantis plan run` CLI writes to `result.json`.

### Modal Secret

Create `mantis-plan-runner-secrets` once. Required keys:

```
modal secret create mantis-plan-runner-secrets \
    ANTHROPIC_API_KEY=sk-ant-...
```

Optional keys (add when you need them):

| Key | When |
|---|---|
| `MANTIS_API_TOKEN` | If the brain endpoint enforces tenant auth via `X-Mantis-Token`. |
| `OXYLABS_USERNAME` / `_PASSWORD` / `_ENTRYPOINT` | When proxying the Modal browser through an upstream residential proxy. |
| `OXYLABS_COUNTRY` / `_STATE` / `_CITY` | Geo pinning, when the upstream plan supports it. |
| `PRIVATEPROXY_USERNAME` / `_PASSWORD` / `_ENTRYPOINT` | Alternate residential-proxy provider; the runner auto-picks PrivateProxy when both are set. |
| `MANTIS_PROXY_PROVIDER` | Force `oxylabs` or `privateproxy` when both creds are mounted. |

### Submit a plan

From the laptop CLI:

```
mantis plan run-modal plans/marketplace-listings.txt \
    --endpoint https://workspace--mantis-server-api.modal.run/v1 \
    --header "X-Mantis-Token=$MANTIS_API_TOKEN" \
    --start-url https://www.marketplace.example/listings/ \
    --use-proxy
```

Or ad-hoc via `modal run`:

```
uv run modal run deploy/modal/modal_plan_runner.py \
    --plan-path plans/marketplace-listings.txt \
    --endpoint https://workspace--mantis-server-api.modal.run/v1 \
    --header X-Mantis-Token=...
```

### Diagnose proxy issues

The app also ships a `diagnose_proxy` function for ad-hoc upstream
probing (useful when a smoke run hits `ERR_TUNNEL_CONNECTION_FAILED`
or Cloudflare 403 inside the container):

```
uv run python -c "
import modal, json
fn = modal.Function.from_name('mantis-plan-runner', 'diagnose_proxy')
print(json.dumps(fn.remote(session='diag-1'), indent=2))
"
```

It returns the Modal egress IP, the IP each upstream proxy resolves
to, the tinyproxy log tail, and CONNECT response codes — enough to
distinguish "container can't reach upstream" from "upstream rejected
auth" from "upstream OK but target site blocked the proxy IP".

## See also

- `deploy/modal/` — every Modal entry-point in the repo
- [`mantis plan run-modal`](../getting-started/cli.md#mantis-plan-run-modal-path)
- [Plan formats](../getting-started/plan-formats.md)
- [Hosting overview](index.md)
