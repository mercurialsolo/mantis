# Mantis CUA Agent

A unified perception-reasoning-action agent for computer use. Unlike traditional decoupled CUA pipelines (grounding model -> reasoning model -> executor), Mantis uses a single forward pass for all three stages, with a rolling frame buffer for temporal context.

## Architecture

```
Traditional (Agent-S, etc.):
  screenshot -> grounding_model -> reasoning_model -> exec -> screenshot -> repeat
  (Serial. ~3-5s per cycle. Model never sees transitions.)

StreamingCUA / GymRunner (this repo):
  continuous_frames -> model(perceive+reason+act) -> execute -> model sees results
  (One model, one pass. Frame buffer gives temporal context. Tight feedback loop.)
```

### Core Modules

| Module              | Purpose                                                 |
|---------------------|---------------------------------------------------------|
| `agent.py`          | `StreamingCUA` — async perception-action loop (local)   |
| `gym/runner.py`     | `GymRunner` — sync agent loop for gym environments      |
| `brain.py`          | Gemma 4 inference via HuggingFace Transformers          |
| `brain_llamacpp.py` | llama.cpp backend for GGUF quantized models             |
| `brain_holo3.py`    | Holo3-35B-A3B via vLLM (tool calling + reasoning)       |
| `brain_opencua.py`  | EvoCUA / OpenCUA via vLLM                               |
| `brain_claude.py`   | Claude via Anthropic API                                |
| `actions.py`        | Action types and native tool-call schemas               |
| `executor.py`       | pyautogui-based action execution                        |
| `streamer.py`       | Background screen capture into rolling frame buffer     |
| `viewer.py`         | Live web viewer (local, async)                          |
| `viewer_modal.py`   | Live web viewer (Modal, thread-safe)                    |

### Supported Models

| Model | Backend | GPU | Notes |
|-------|---------|-----|-------|
| Gemma 4 E4B | HuggingFace / llama.cpp | 1x A100 | Default local model |
| Gemma4-31B-CUA | llama.cpp | 1x A100 | Fine-tuned on AgentNet |
| EvoCUA-8B | vLLM | 1x A100 | Meituan CUA model |
| EvoCUA-32B | vLLM | 2x A100 | Higher accuracy |
| OpenCUA-32B | vLLM | 4x A100 | xLang CUA model |
| OpenCUA-72B | vLLM | 8x A100 | Maximum open-source accuracy |
| Holo3-35B-A3B | vLLM | 2x A100 | Tool calling + reasoning parser |
| Claude Sonnet/Opus | Anthropic API | None | API-based, saves trajectories for distillation |

### Micro-Intent Pipeline (Production)

The validated production pipeline for web data extraction:

```
Plain text plan OR JSON micro-plan
  → MicroPlanRunner (micro_runner.py):
    SETUP:    navigate (URL with filters) → gate (Claude verifies page)
    EXTRACT:  click (Claude batch-finds all listings) → URL → scroll → extract → back → loop
    PAGINATE: URL-based /page-N/ → Claude-guided → Holo3 fallback → loop
```

Each step: 1 sentence, fresh GymRunner, max 3-8 actions. Claude (Sonnet API) plans + reads data. Holo3 (3B GPU) clicks + scrolls.

| Component | Role | Cost |
|-----------|------|------|
| Holo3-35B-A3B | Click, scroll, navigate (1× A100, llama.cpp GGUF) | ~$0.02/lead |
| Claude Sonnet | Find listings, extract data, verify gates, grounding | ~$0.02/lead |
| IPRoyal proxy | Residential, geo-targeted, sticky sessions | ~$0.07/lead |
| **Total** | | **~$0.12/lead** |

Key modules for micro-intent pipeline:

| Module | Purpose |
|--------|---------|
| `gym/micro_runner.py` | MicroPlanRunner — execute micro-plans with checkpoint/verify/reverse |
| `plan_decomposer.py` | Plain text → MicroIntents with sections, gates, loops (Claude, cached) |
| `extraction.py` | ClaudeExtractor — find_all_listings, extract data, verify_gate, find_filter_target |
| `grounding.py` | ClaudeGrounding — refine click coordinates (distance-scaled confidence) |

### Running the Micro-Intent Pipeline

```bash
# From a JSON micro-plan (pre-built, no decomposition needed):
uv run modal run --detach deploy/modal/modal_cua_server.py \
  --micro plans/boattrader/extract_url_filtered.json \
  --model holo3 --viewer

# Production run with externalized checkpoint state:
uv run modal run --detach deploy/modal/modal_cua_server.py \
  --micro plans/boattrader/extract_url_filtered.json \
  --model holo3 --viewer \
  --state-key boattrader-miami-private-v1

# Retry a later Modal run from the last checkpoint for that state key:
uv run modal run --detach deploy/modal/modal_cua_server.py \
  --micro plans/boattrader/extract_url_filtered.json \
  --model holo3 --viewer \
  --state-key boattrader-miami-private-v1 \
  --resume-state

# From a plain text plan (decomposed by Claude Sonnet, cached):
uv run modal run --detach deploy/modal/modal_cua_server.py \
  --micro plans/boattrader/extract_only.txt \
  --model holo3 --viewer

# Monitor:
tail -f /tmp/longrun_*.log

# Results on Modal volume 'osworld-data' at /results/holo3_results_*.json
```

Micro-runs save step-level checkpoints to the Modal volume under `/data/checkpoints/<state-key>.json`. The checkpoint stores logical execution state, including current step, current results page, seen URLs, extracted leads, loop counters, listing cursor/cache, costs, and a re-entry URL; `--resume-state` reconstructs browser state from that snapshot before continuing.

### BoatTrader URL Filter Format

BoatTrader encodes filters as URL path segments (no sidebar clicking needed):

```
https://www.boattrader.com/boats/state-fl/city-miami/zip-33101/by-owner/price-35000/
                                 ^^^^^^^^  ^^^^^^^^^  ^^^^^^^^  ^^^^^^^^  ^^^^^^^^^^
                                 state     city(auto) zip       seller    min price

Pagination: .../price-35000/page-2/
```

### Plan JSON Format

```json
[
  {"intent": "Navigate to https://...", "type": "navigate", "section": "setup", "required": true},
  {"intent": "Verify filters applied", "type": "extract_data", "claude_only": true, "section": "setup", "gate": true, "verify": "..."},
  {"intent": "Click listing title", "type": "click", "grounding": true, "section": "extraction"},
  {"intent": "Read URL", "type": "extract_url", "claude_only": true, "section": "extraction"},
  {"intent": "Scroll down", "type": "scroll", "budget": 10, "section": "extraction"},
  {"intent": "Extract data", "type": "extract_data", "claude_only": true, "section": "extraction"},
  {"intent": "Go back", "type": "navigate_back", "section": "extraction"},
  {"intent": "Loop", "type": "loop", "loop_target": 2, "loop_count": 50, "section": "extraction"},
  {"intent": "Paginate", "type": "paginate", "grounding": true, "section": "pagination"},
  {"intent": "Loop", "type": "loop", "loop_target": 2, "loop_count": 50, "section": "pagination"}
]
```

Step types: `navigate`, `filter`, `click`, `scroll`, `extract_url`, `extract_data`, `navigate_back`, `paginate`, `loop`

Key fields: `section` (setup/extraction/pagination), `required` (retry then halt), `gate` (Claude verifies, halt on fail), `claude_only` (no Holo3), `grounding` (ClaudeGrounding refines clicks)

## Quick Start

```bash
# Install core
pip install -e .

# With viewer support
pip install -e ".[viewer]"

# With GPU acceleration
pip install -e ".[gpu]"

# Run locally
mantis "Open Firefox and search for cats" --model google/gemma-4-E4B-it

# Run with live viewer
mantis --viewer "Open Firefox and search for cats"
```

### Development

```bash
# Lint Python sources
uv run --extra dev ruff check .
```

## Live Viewer

A web-based dashboard for watching what the agent sees and does in real-time. Shows the live screen feed (MJPEG), action log with click indicators, and agent thinking.

### Local

```bash
mantis --viewer "your task here"
#   Viewer: http://localhost:7860?token=<random>
```

### Modal (remote GPU)

```bash
uv run modal run deploy/modal/modal_cua_server.py \
    --task-file tasks/your_tasks.json \
    --model gemma4-cua \
    --viewer
#   Viewer: https://mantis-cua-server--xxx.modal.run?token=<random>
```

The `--viewer` flag works with every model. On Modal, it uses `modal.forward()` to tunnel the viewer to a public HTTPS URL. Authentication is via a random token in the URL query parameter.

### What the viewer shows

- **Screen feed** -- MJPEG stream of the agent's display (~10 FPS)
- **Action log** -- Each step with action type, parameters, and reasoning
- **Click overlay** -- Pulsing ring indicator at click coordinates
- **Thinking** -- Model's reasoning text (when available)
- **Status** -- Step counter, elapsed time, connection indicator

### Remote access

The viewer binds to `0.0.0.0`. For public internet access without Modal:

```bash
ngrok http 7860
# or
cloudflared tunnel --url http://localhost:7860
```

## Modal Deployment

```bash
# Deploy the planner (stays warm for 10 min)
uv run modal deploy deploy/modal/modal_cua_server.py

# Run with a pre-built task suite
uv run modal run deploy/modal/modal_cua_server.py \
    --task-file tasks/boattrader/dynamic_production.json \
    --model gemma4-cua

# Run with plan preprocessing (Gemma4 planner -> executor)
uv run modal run deploy/modal/modal_cua_server.py \
    --plan-file plans/boattrader/full_spec.txt \
    --model evocua-8b \
    --inputs "pop_password=SelfService38#,zip_code=33101"

# Parallel extraction (fan-out across GPUs)
uv run modal run deploy/modal/modal_cua_server.py \
    --task-file tasks/boattrader/dynamic_production.json \
    --model gemma4-cua \
    --workers 5

# Claude (API-based, no GPU)
uv run modal run deploy/modal/modal_cua_server.py \
    --task-file tasks/your_tasks.json \
    --model claude \
    --claude-model claude-sonnet-4-20250514 \
    --thinking-budget 2048
```

### Executor Architecture

Each executor runs inside a Modal container with:
- **Xvfb** -- Virtual display (1280x720)
- **Real Google Chrome** -- Zero automation fingerprints (not Playwright)
- **xdotool** -- Native X11 input events
- **mss** -- Screen capture from virtual display
- **Residential proxy** -- IPRoyal with geo-targeting (optional)

## Project Structure

```
src/mantis_agent/
  agent.py              # StreamingCUA — async local agent loop
  brain.py              # Gemma4Brain (HuggingFace)
  brain_llamacpp.py     # LlamaCppBrain (llama.cpp server)
  brain_holo3.py        # Holo3Brain (vLLM)
  brain_opencua.py      # OpenCUABrain (vLLM)
  brain_claude.py       # ClaudeBrain (Anthropic API)
  actions.py            # Action types + tool schemas
  executor.py           # pyautogui action execution
  streamer.py           # Rolling frame buffer capture
  viewer.py             # Live web viewer (local/async)
  viewer_modal.py       # Live web viewer (Modal/threaded)
  extraction.py         # ClaudeExtractor — screenshot → structured data
  grounding.py          # ClaudeGrounding — click coordinate refinement
  plan_decomposer.py    # PlanDecomposer — plain text → MicroIntents
  main.py               # CLI entry point
  gym/
    base.py             # GymEnvironment abstract base
    runner.py           # GymRunner — sync agent loop
    micro_runner.py     # MicroPlanRunner — micro-intent execution
    xdotool_env.py      # X11 environment (Xvfb + xdotool)
    workflow_runner.py   # Loop/pagination workflows (legacy)
  verification/         # Step verification + playbooks
  curriculum/           # Training curriculum techniques
  prompts/              # Prompt templates

plans/
  boattrader/
    extract_url_filtered.json  # Production plan (URL-filtered + extraction loop)
    extract_only.txt           # Plain text plan (decomposed by Claude)
    test_url_filters.json      # Filter-only test
    test_filters_only.json     # Sidebar filter test

deploy/
  modal/                # Modal entrypoints (modal_cua_server.py, modal_osworld_*.py, ...)
  baseten/              # Baseten Truss deployments (holo3, gemma4, gemma4_26b)
  aws/                  # EKS: Terraform + k8s manifests + runbook
  gke/                  # GKE: Terraform + k8s manifests + runbook
docker/
  cua.Dockerfile        # CUA container (Xvfb + xdotool + Chromium)
  hud.Dockerfile        # HUD environment container
  local.Dockerfile      # Local container with Playwright
  server.Dockerfile     # Production FastAPI server (CUDA + llama.cpp + Holo3) for AWS/GKE
scripts/
  run_osworld.py        # OSWorld benchmark runner
  run_gym_anything.py   # Generic gym environment runner
  run_web_tasks.py      # Web task automation
  run_local.py          # Local CUA loop
  export_leads.py       # CSV export with viability checks
  monitor_run.py        # Tail Modal/Baseten run state
  replay_test.py        # Replay cached screenshots
  baseten_workload.py   # Detached run/poll helper for Baseten
tests/                  # pytest suite
docs/
  API.md                                       # HTTP API reference (any caller)
  ARCHITECTURE.md
  PROPOSAL-mantis-cua-replaces-vision_claude.md  # stakeholder-shareable
  integration-vision_claude.md                 # vision_claude integration spec
  learnings.md
  diagrams/                                    # Figma renders
```

## Using the deployed service

If you just want to send plans to a deployed Mantis CUA service, see
[`docs/API.md`](docs/API.md) for the HTTP API reference — auth, endpoints,
plan shapes, errors, and end-to-end curl examples.

If you're integrating from `vision_claude`, see
[`docs/integration-vision_claude.md`](docs/integration-vision_claude.md)
for the orchestrated-Mantis backend code and migration plan.

## License

MIT
