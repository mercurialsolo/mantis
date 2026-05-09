# Modal

Modal is the easiest path for ad-hoc / batch runs that benefit from scale-to-zero. The full set of Modal entry-points lives at `deploy/modal/`. The most common one is `modal_cua_server.py` which exposes the same `/v1/predict` semantics via Modal's `local_entrypoint`.

## Prerequisites

```bash
pip install modal
modal token new   # authenticates this machine to your Modal workspace
```

You'll also need an `.env` file at the repo root with the same five secrets Baseten uses. Modal's `Secret.from_dotenv()` picks them up at deploy time.

## Submit a plan (one-off)

```bash
uv run modal run --detach deploy/modal/modal_cua_server.py \
  --micro plans/example/extract_listings.json \
  --model holo3 \
  --max-cost 2 \
  --max-time-minutes 20 \
  --session-name marketplace-smoke
```

`--detach` returns immediately; the run continues in Modal. Each run gets its own GPU container, scales to zero when done.

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

## When to choose Modal over Baseten

| Pick Modal if… | Pick Baseten if… |
|---|---|
| You want true scale-to-zero (no idle GPU) | You want a stable HTTPS endpoint other services can call |
| Bursty / batch workloads | Steady traffic |
| You're comfortable invoking via the Modal CLI / SDK | You need a tenant-keyed multi-caller surface |
| You want per-run cost attribution at the run level | You want per-run cost + Prometheus metrics + webhooks |

The `/v1/predict` Tier-1/Tier-2 features (rate limits, idempotency, webhooks, allowlist, metrics) are only exposed through the FastAPI server — i.e., on Baseten / EKS / GKE / local. Modal entry-points are direct, single-tenant.

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
