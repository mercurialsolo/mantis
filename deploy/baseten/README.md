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
