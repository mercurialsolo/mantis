# Baseten

The reference deployment lives on Baseten. It's the fastest path: one Truss push and you have a managed, autoscaled GPU endpoint. The full Truss runbook (with cost-cache `--no-cache` guidance) lives at [`deploy/baseten/README.md`](https://github.com/mercurialsolo/mantis/blob/main/deploy/baseten/README.md); this page is the operator's checklist.

## Prerequisites

- Baseten account with a project + an API key
- `uvx truss` on your dev machine (`pip install --upgrade truss`, requires ≥ 0.15.2 for `--no-cache`)
- A clone of the repo

## 1. Provision Baseten secrets

These are the named secrets the container reads from `/secrets/<name>`. Set them once via the Baseten dashboard (Workspace → Secrets) or via the API:

```bash
export BASETEN_API_KEY="..."

# Generate a tenant token (save it — this is what callers use)
TOK=$(openssl rand -hex 32)
echo "Save this token: $TOK"

# Create the secret
python3 -c "import json,os; print(json.dumps({'name':'mantis_api_token','value':os.environ['TOK']}))" > /tmp/payload.json
curl -sS -X POST -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/payload.json \
  https://api.baseten.co/v1/secrets

# Repeat for: anthropic_api_key, proxy_url, proxy_user, proxy_pass
```

For multi-tenant deployments, also create `mantis_tenant_keys` (a JSON keys file — see [Tenant keys](../operations/tenant-keys.md)).

## 2. Push the Truss

```bash
DEPLOY_NAME="mantis-$(date -u +%Y%m%d-%H%M)"
uvx truss push deploy/baseten/holo3 --no-cache \
  --promote \
  --deployment-name "$DEPLOY_NAME" \
  --include-git-info
```

To deploy Fara-7B instead, point the push at `deploy/baseten/fara`:

```bash
uvx truss push deploy/baseten/fara --no-cache \
  --promote --deployment-name "$DEPLOY_NAME" --include-git-info
```

Both directories ship the same FastAPI surface (`/v1/predict`, `/v1/cua`, `/v1/chat/completions`, etc.) — only the in-pod inference server differs (llama.cpp + GGUF for Holo3; vLLM + bf16 weights for Fara). Fara skips the llama.cpp compile step, so the first build is ~5 min instead of ~50.

`--no-cache` is required on the **first push** after any change to `src/mantis_agent/` (the package code is shipped via `external_package_dirs` and isn't always part of the image hash — without `--no-cache` Baseten can serve stale code). Subsequent pushes that only change `build_commands` / `requirements` / `environment_variables` can omit the flag.

The first Holo3 build does the full llama.cpp + CUDA compile (~50 min). Subsequent builds are ~5 min if you don't change `build_commands`. The Fara build skips llama.cpp entirely.

## 3. Wait for it to go ACTIVE

```bash
# Poll until terminal
while true; do
  STATE=$(curl -sS -H "Authorization: Api-Key $BASETEN_API_KEY" \
    "https://api.baseten.co/v1/models/$MODEL_ID/deployments/$DEPLOY_ID" \
    | jq -r .status)
  echo "$(date '+%H:%M:%S')  $STATE"
  case "$STATE" in ACTIVE|BUILD_FAILED|DEPLOY_FAILED) break ;; esac
  sleep 60
done
```

`MODEL_ID` and `DEPLOY_ID` come from the push output (`https://app.baseten.co/models/<MODEL_ID>/logs/<DEPLOY_ID>`).

## 4. Test the live endpoint

The Baseten gateway exposes two route families against every truss-server deployment:

| URL | What it serves |
|---|---|
| `https://model-${MODEL_ID}.api.baseten.co/production/predict` | The configured `predict_endpoint` (default `/predict`) — the orchestrated run/status/resume entry point |
| `https://model-${MODEL_ID}.api.baseten.co/production/sync/<any-path>` | Pass-through to arbitrary FastAPI routes in the container — `/v1/chat/completions`, `/v1/models`, `/v1/health`, `/v1/cua` |

For a canary (non-promoted) deployment, swap `production` for `deployment/<DEPLOY_ID>` in either form.

Quick orchestrated-mode smoke:

```bash
ENDPOINT="https://model-${MODEL_ID}.api.baseten.co/production"

curl -fsS -X POST "$ENDPOINT/predict" \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -H "X-Mantis-Token: $TOK" \
  -H "Content-Type: application/json" \
  -d '{
    "detached": true,
    "micro": "plans/example/extract_listings.json",
    "profile_id":  "smoke",
    "workflow_id": "smoke-test-v1",
    "max_cost": 2,
    "max_time_minutes": 20
  }'
```

Expected: a `queued` response with a `run_id`. Then poll with `{"action":"status","run_id":"..."}` until terminal, then `{"action":"result","run_id":"..."}` for the leads.

Raw-inference smoke (the StaffAI-shaped path — model returns OpenAI-format tool calls; the caller runs its own CUA loop):

```bash
curl -fsS -X POST "$ENDPOINT/sync/v1/chat/completions" \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -H "X-Mantis-Token: $TOK" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model",
    "messages": [{"role": "user", "content": "Say hi."}],
    "max_tokens": 32
  }'
```

## Auth model

Baseten requires **both** headers:

| Header | Layer | What |
|---|---|---|
| `Authorization: Api-Key <BASETEN_API_KEY>` | gateway | Authenticates the platform request |
| `X-Mantis-Token: <tenant_token>` | container | Authenticates the tenant |

The gateway passes `Authorization: Api-Key …` through to the container; we deliberately use a custom `X-Mantis-Token` header for the container layer so it doesn't collide with the platform header.

## Cost guardrails

| Knob | Default | Where to set |
|---|---|---|
| `MAX_COST_USD` | $25 per run | Container env (`MANTIS_MAX_COST_USD`) — clamps the request's `max_cost` |
| `MAX_RUNTIME_MINUTES` | 60 per run | `MANTIS_MAX_RUNTIME_MINUTES` |
| Per-tenant `max_cost_per_run` | tenant config | tenant keys file |
| Per-tenant `rate_limit_per_minute` | 30 | tenant keys file |
| Replica autoscale | min=0 max=1 | `runtime.health_checks` in `deploy/baseten/holo3/config.yaml` |

If your traffic is bursty, leave `min=0` and accept the ~50 min cold-start image build the first time the replica scales from zero. For low-latency production, set `min=1` (always-on GPU charge).

## Updating

| Change | What you need |
|---|---|
| Tenant keys | Update the `mantis_tenant_keys` Baseten secret. Hot reload (5 s cache) — no redeploy. |
| Anthropic key for a tenant | Update its `anthropic_api_key_<tenant>` secret. Hot reload via env. |
| Plan files (`plans/...`) | Push a new deployment. (External package dirs aren't hot-reloaded.) |
| `src/mantis_agent/` code | `truss push --no-cache`. |
| `build_commands` / system deps | `truss push` (no `--no-cache` needed). |

## Smoke test before promoting

If you want to canary, push without `--promote`:

```bash
uvx truss push deploy/baseten/holo3 --no-cache \
  --deployment-name "$DEPLOY_NAME-canary" \
  --include-git-info
```

You'll get a non-production environment URL. Run smoke tests there, then promote via the Baseten dashboard.

## See also

- [`deploy/baseten/README.md`](https://github.com/mercurialsolo/mantis/blob/main/deploy/baseten/README.md) — the source of truth for the Truss config
- [Tenant keys](../operations/tenant-keys.md) — how to provision per-tenant tokens
- [Metrics](../operations/metrics.md) — wiring Prometheus scrape from Baseten
