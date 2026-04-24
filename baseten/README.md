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
uvx truss push baseten/holo3 \
  --promote --wait --deployment-name baseten-holo3-workload \
  --include-git-info

uvx truss push baseten/gemma4 \
  --environment production --wait --deployment-name baseten-gemma4-31b \
  --include-git-info

uvx truss push baseten/gemma4_26b \
  --environment production --wait --deployment-name baseten-gemma4-26b \
  --include-git-info
```

## Trigger A Workload Run

Use Baseten async inference for long runs. The default request runs
`plans/boattrader/extract_url_filtered.json`.

```json
{
  "micro": "plans/boattrader/extract_url_filtered.json",
  "state_key": "boattrader-miami-private-v1",
  "resume_state": false,
  "max_cost": 10.0,
  "max_time_minutes": 180
}
```

## Trigger Fine-Tuning

The training job packages `training/train_holo3_distill.py` plus the local
distillation JSONL/screenshots, then writes checkpoints under
`$BT_CHECKPOINT_DIR` so Baseten can deploy them later.

```bash
uvx truss train push baseten/training/holo3_distill/config.py \
  --job-name holo3-boattrader-distill
```

After completion:

```bash
uvx truss train deploy_checkpoints --job-id <job_id>
```
