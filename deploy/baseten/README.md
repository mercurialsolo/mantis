# Baseten Deployment

This directory contains Baseten Truss artifacts for the current Mantis CUA
workload:

- `holo3/`: Holo3-35B-A3B GGUF + Chrome/Xvfb/xdotool workload runner on H100.
- `gemma4/`: Gemma4-31B vLLM OpenAI-compatible endpoint on 2x H100.
- `gemma4_26b/`: Gemma4-26B A4B vLLM OpenAI-compatible endpoint on 2x H100.
- `training/holo3_distill/`: Baseten Training job for the current BoatTrader
  Holo3 distillation data.

The inference Trusses use Baseten custom Docker servers because the workload is
more than a model endpoint: it starts llama.cpp, launches Chrome in Xvfb, uses
xdotool for screen-level actions, and calls Claude for extraction/grounding.

## Required Baseten Secrets

Create these in the Baseten secret manager before deploying:

- `anthropic_api_key`: required for Claude extraction and grounding.
- `proxy_url`, `proxy_user`, `proxy_pass`: optional but recommended for
  BoatTrader runs that need a residential proxy.

## Deploy

```bash
uvx truss push deploy/baseten/holo3 \
  --promote --wait --deployment-name baseten-holo3-workload \
  --include-git-info

uvx truss push deploy/baseten/gemma4 \
  --environment production --wait --deployment-name baseten-gemma4-31b \
  --include-git-info

uvx truss push deploy/baseten/gemma4_26b \
  --environment production --wait --deployment-name baseten-gemma4-26b \
  --include-git-info
```

### When to add `--no-cache`

Truss caches built images by build-context hash. **Code that ships through
`external_package_dirs` (`src/mantis_agent`, `plans/`) is not always part of
that hash** — meaning if you only change `mantis_agent/baseten_server.py` (or
any other shipped package code) and nothing in `build_commands` /
`requirements` / `environment_variables`, Truss can skip the rebuild and the
old code keeps running. Force a clean rebuild on the first push after such
changes:

```bash
uvx truss push deploy/baseten/holo3 --no-cache \
  --promote --wait --deployment-name baseten-holo3-workload --include-git-info
```

`--no-cache` was added in Truss 0.15.2; run `pip install --upgrade truss` if
your client is older.

## Multi-tenant mode (Tier 1)

By default the deployment is single-tenant — one shared `mantis_api_token`
that any caller with the secret can use. To open the endpoint to multiple
tenants with per-key isolation, mount a JSON keys file and set
`MANTIS_TENANT_KEYS_PATH`:

```yaml
# in deploy/baseten/holo3/config.yaml
secrets:
  anthropic_api_key: null
  proxy_url: null
  proxy_user: null
  proxy_pass: null
  mantis_api_token: null         # legacy single-tenant fallback
  mantis_tenant_keys: null       # NEW — JSON keys file
environment_variables:
  MANTIS_TENANT_KEYS_PATH: /secrets/mantis_tenant_keys
```

Keys-file shape:

```json
{
  "tenant_keys": {
    "<x-mantis-token-value>": {
      "tenant_id": "vision_claude_prod",
      "scopes": ["run", "status", "result", "logs"],
      "max_concurrent_runs": 3,
      "max_cost_per_run": 5.0,
      "max_time_minutes_per_run": 30,
      "anthropic_secret_name": "anthropic_api_key_vision_claude",
      "allowed_domains": ["*.boattrader.com", "staffai-test-crm.exe.xyz"]
    }
  }
}
```

What the server does for each request:
- Looks up the `X-Mantis-Token` in the keys file (5-second cache, hot reload
  on rotation — no pod restart needed).
- Clamps `max_cost` and `max_time_minutes` against the tenant's caps in
  addition to the global caps (`MANTIS_MAX_COST_USD`, `MANTIS_MAX_RUNTIME_MINUTES`).
- Server-prefixes `state_key` with the tenant id so callers cannot collide
  with each other or read each other's checkpoint state.
- Resolves the per-tenant Anthropic key from the secret named in
  `anthropic_secret_name` (each tenant can have its own Anthropic billing).
- Per-tenant Chrome profile dir at
  `$MANTIS_DATA_DIR/tenants/<tenant_id>/chrome-profile/<state_key>/` so
  cookies and sessions never bleed across tenants.

Server-side hard caps (env-overridable):

| Env var | Default | Purpose |
|---|---|---|
| `MANTIS_MAX_STEPS_PER_PLAN` | 200 | Reject oversized plans |
| `MANTIS_MAX_LOOP_ITERATIONS` | 50 | Clamp `loop_count` in micro-plans |
| `MANTIS_MAX_RUNTIME_MINUTES` | 60 | Hard wall-time cap |
| `MANTIS_MAX_COST_USD` | 25.0 | Hard cost cap |

The single-tenant `mantis_api_token` mode keeps working unchanged when
`MANTIS_TENANT_KEYS_PATH` is unset — useful for development and
backwards-compat with the v1 deployment.

## API endpoints

| Path | Status | Purpose |
|---|---|---|
| `POST /v1/predict` | Tier-1 | Run a plan / poll status / fetch result. Validated, per-tenant capped. |
| `POST /predict` | Legacy alias | Identical behavior to `/v1/predict`. Kept for back-compat. |
| `GET /v1/models` | Tier-1 | Lists `holo3`. |
| `GET /health` | Open | Liveness + readiness for the platform. |

## Trigger A Workload Run

The default request runs `plans/boattrader/extract_url_filtered.json`. For
Modal-like behavior, set `detached: true`; the Baseten server returns a
`run_id` immediately, continues work in the replica, and writes status, results,
and lead CSV files under `$MANTIS_DATA_DIR/runs/<run_id>/`.

```json
{
  "detached": true,
  "micro": "plans/boattrader/extract_url_filtered.json",
  "state_key": "boattrader-miami-private-v1",
  "resume_state": false,
  "max_cost": 10.0,
  "max_time_minutes": 180
}
```

Poll status through the same `/predict` endpoint:

```json
{"action": "status", "run_id": "20260424_123456_ab12cd34"}
```

Fetch the result or server-side run events:

```json
{"action": "result", "run_id": "20260424_123456_ab12cd34"}
{"action": "logs", "run_id": "20260424_123456_ab12cd34", "tail": 200}
```

Full model logs are still available from Baseten:

```bash
uvx truss model-logs --model-id <model_id> --deployment-id <deployment_id>
```

There is also a local helper that mirrors the Modal detach/poll flow and loads
the Baseten API key from `.env`:

```bash
uv run python scripts/baseten_workload.py run \
  --micro plans/boattrader/extract_url_filtered.json \
  --state-key boattrader-miami-private-v1 \
  --max-cost 10 \
  --max-time-minutes 180

uv run python scripts/baseten_workload.py status --run-id <run_id>
uv run python scripts/baseten_workload.py logs --run-id <run_id> --tail 200 --raw
uv run python scripts/baseten_workload.py watch --run-id <run_id>
uv run python scripts/baseten_workload.py result --run-id <run_id> --csv-out leads.csv
```

## Trigger Fine-Tuning

The training job packages `training/train_holo3_distill.py` plus the local
distillation JSONL/screenshots, then writes checkpoints under
`$BT_CHECKPOINT_DIR` so Baseten can deploy them later.

```bash
uvx truss train push deploy/baseten/training/holo3_distill/config.py \
  --job-name holo3-boattrader-distill
```

After completion:

```bash
uvx truss train deploy_checkpoints --job-id <job_id>
```
