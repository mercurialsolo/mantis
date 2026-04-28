# Mantis

A unified perception-reasoning-action agent for computer use. Given a structured plan, Mantis drives a real browser (or any Xvfb-rendered application), takes actions, extracts structured data, and produces both a JSON result and an optional polished video walkthrough.

[![Docs](https://img.shields.io/badge/docs-mkdocs--material-teal)](https://mercurialsolo.github.io/mantis/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

```
       ┌──────────────────────┐         ┌─────────────────────────┐
3p ──► │ Mantis CUA service   │ ──────► │ Target app (Chrome,     │
caller │ Holo3 + Claude       │         │ file manager, terminal, │
       │ /v1/predict          │         │ LibreOffice, …)         │
       └──────────┬───────────┘         └─────────────────────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │ Result + lead CSV +  │
       │ polished screencast  │
       └──────────────────────┘
```

## What you get

- **Reliable multi-step plans.** A structured `MicroPlanRunner` enforces section / gate / loop semantics so even small models behave on long workflows.
- **Cheap inference at click latency.** Holo3-35B-A3B (GGUF on a single GPU) for tactical click / scroll / type / drag actions; Anthropic Claude only for surgical reasoning steps (extract / verify / ground a click).
- **Real browser, real desktop.** Xvfb + Chrome + xdotool. No Playwright fingerprints. Works against sites with bot detection.
- **Cloud-portable.** Same image runs on Baseten, Modal, EKS, GKE, or your own Docker host.
- **Multi-tenant out of the box.** Per-key auth, per-tenant rate limits, idempotency keys, HMAC-signed webhooks, URL allowlists, Prometheus metrics.
- **Screencast included.** Every run can produce a title-card → captioned-run-with-action-overlays → outro video that's ready to share.

## Verified end-to-end

| Path | Run | Result |
|---|---|---|
| Modal | 3-listing BoatTrader extraction | 2 / 3 leads, 1 with phone, $0.42, 13 min |
| Baseten | 3-listing BoatTrader extraction | 3 / 3 leads, 1 with phone, $0.42 budget, 9.5 min |

Real lead row from the Baseten run:

> **1997 Caroff CHATAM 52** — $254,000 — phone +596696520959 — boattrader.com/boat/1997-caroff-chatam-52-10130796/

## Documentation

The full docs site is at **[mercurialsolo.github.io/mantis](https://mercurialsolo.github.io/mantis/)** (or `mkdocs serve` from this checkout).

| If you want to… | Go here |
|---|---|
| Try it in 5 minutes | [Quickstart](docs/getting-started/quickstart.md) |
| Understand the architecture | [Concepts](docs/getting-started/concepts.md) · [Architecture](docs/architecture.md) |
| Deploy your own instance | [Hosting](docs/hosting/index.md) — Baseten / Modal / EKS / GKE / local |
| Integrate from your app | [Client](docs/client/index.md) — auth, plans, polling, recordings |
| Run a multi-tenant fleet | [Operations](docs/operations/index.md) — tenant keys, rate limits, webhooks, metrics |
| Look up an HTTP endpoint | [API reference](docs/api.md) |
| Replace a Claude-CUA-style backend | [vision_claude integration](docs/integration-vision_claude.md) |

## Quick start (no deploy needed)

The reference deployment is live on Baseten. With a tenant token from your operator:

```bash
export ENDPOINT="https://model-qvvgkneq.api.baseten.co/production"
export BASETEN_API_KEY="..."
export MANTIS_API_TOKEN="..."

curl -fsS -X POST "$ENDPOINT/v1/predict" \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "detached": true,
    "micro": "plans/boattrader/extract_url_filtered_3listings.json",
    "state_key": "first-run",
    "max_cost": 2,
    "max_time_minutes": 20,
    "record_video": true
  }'
```

Poll for `{action: "status"}` until the run terminates, then `{action: "result"}` to fetch leads, then `GET /v1/runs/<run_id>/video` for the screencast. Full walkthrough in the [Quickstart](docs/getting-started/quickstart.md).

## Install footprint

The base install is intentionally slim. Pick the extras for your use case:

```bash
pip install -e .                  # ~5 MB — Pillow only
pip install -e ".[orchestrator]"  # ~10 MB — MicroPlanRunner + remote Holo3 client
pip install -e ".[server]"        # ~15 MB — FastAPI server + Prometheus
pip install -e ".[local-cua]"     # ~2 GB  — torch + transformers + pyautogui
pip install -e ".[full]"          # everything
pip install -e ".[docs]"          # mkdocs + material theme
```

## Repository layout

```
src/mantis_agent/             core library
  api_schemas.py              PredictRequest + plan validation + caps
  baseten_server.py           FastAPI: /v1/predict + /v1/chat/completions + /metrics
  brain_holo3.py              Holo3 inference client
  brain_claude.py             Claude inference client
  extraction.py               ClaudeExtractor (structured data)
  grounding.py                ClaudeGrounding (refine click coordinates)
  gym/
    base.py                   GymEnvironment ABC
    runner.py                 GymRunner — sync agent loop
    micro_runner.py           MicroPlanRunner — structured plan executor
    xdotool_env.py            Xvfb + Chrome + xdotool driver
  presentation.py             Cards, captions, action overlays for the polished video
  recorder.py                 ffmpeg x11grab wrapper
  rate_limit.py               Token bucket + concurrency gauge per tenant
  idempotency.py              Idempotency-Key cache (24h TTL, file-backed)
  webhooks.py                 HMAC-signed run-completion callbacks
  metrics.py                  Prometheus counters / gauges / histograms
  tenant_auth.py              JSON keys file → TenantConfig

deploy/                       cloud paths
  modal/                      Modal entry-points (modal_cua_server.py, ...)
  baseten/                    Baseten Truss configs (holo3, gemma4, ...)
  aws/                        EKS — Terraform + k8s manifests + runbook
  gke/                        GKE — Terraform + k8s manifests + runbook

docker/
  cua.Dockerfile              Local CUA loop (Xvfb + xdotool + Chromium)
  server.Dockerfile           Production FastAPI server (CUDA + llama.cpp + Holo3)

docs/                         MkDocs site source
  index.md  api.md  architecture.md
  getting-started/  hosting/  client/  operations/  integrations/  reference/  appendix/
  diagrams/                   FigJam-rendered architecture diagrams

scripts/                      CLI helpers (run_*.py, monitor_*.sh, baseten_workload.py)
plans/                        plan files (.txt, .json)
tasks/                        task descriptors
benchmarks/                   OSWorld / VWA benchmark harnesses
training/                     distillation + fine-tuning configs
tests/                        pytest suite
```

## Development

```bash
# Install dev + docs extras
pip install -e ".[dev,docs]"

# Run tests
pytest tests/ -q

# Lint
ruff check .

# Build the docs site locally
mkdocs serve   # → http://127.0.0.1:8000
```

## License

MIT. See [LICENSE](LICENSE).
