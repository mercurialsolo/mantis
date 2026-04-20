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

## Live Viewer

A web-based dashboard for watching what the agent sees and does in real-time. Shows the live screen feed (MJPEG), action log with click indicators, and agent thinking.

### Local

```bash
mantis --viewer "your task here"
#   Viewer: http://localhost:7860?token=<random>
```

### Modal (remote GPU)

```bash
uv run modal run modal_cua_server.py \
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
uv run modal deploy modal_cua_server.py

# Run with a pre-built task suite
uv run modal run modal_cua_server.py \
    --task-file tasks/boattrader/dynamic_production.json \
    --model gemma4-cua

# Run with plan preprocessing (Gemma4 planner -> executor)
uv run modal run modal_cua_server.py \
    --plan-file plans/boattrader/full_spec.txt \
    --model evocua-8b \
    --inputs "pop_password=SelfService38#,zip_code=33101"

# Parallel extraction (fan-out across GPUs)
uv run modal run modal_cua_server.py \
    --task-file tasks/boattrader/dynamic_production.json \
    --model gemma4-cua \
    --workers 5

# Claude (API-based, no GPU)
uv run modal run modal_cua_server.py \
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
  grounding.py          # RegionGrounding for click refinement
  main.py               # CLI entry point
  gym/
    base.py             # GymEnvironment abstract base
    runner.py           # GymRunner — sync agent loop
    xdotool_env.py      # X11 environment (Xvfb + xdotool)
    workflow_runner.py   # Loop/pagination workflows
  curriculum/           # Training curriculum techniques
  prompts/              # Prompt templates

modal_cua_server.py     # Modal cloud deployment (all models)
run_osworld.py          # OSWorld benchmark runner
run_gym_anything.py     # Generic gym environment runner
run_web_tasks.py        # Web task automation
```

## License

MIT
