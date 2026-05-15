# CUA models

The brain is the vision-action policy that drives the perception-reasoning-action loop on a screenshot. Mantis ships several interchangeable brains; pick one with `cua_model="<name>"` on `/v1/predict`, `--model <name>` on the CLI, or `MANTIS_MODEL=<name>` for the Baseten container.

All brains implement the same `think(frames, task, action_history, screen_size) -> InferenceResult` contract, so the Hybrid plan-executor + vision-fallback flow (`gym/runner.py`) works identically regardless of which one is wired in. "Passthrough" mode (every step routed straight to the brain) is just `cua_model=<name>` flowing through `_executor_for_model()`.

## Selecting a backend

How you select a backend depends on where Mantis is hosted. The selection model differs:

| Surface | Selection | Notes |
|---|---|---|
| Modal `/v1/predict` | Per-request `cua_model` field | One deployment serves every backend; each request spawns the right GPU function. |
| Modal CLI (`modal run`) | `--model <name>` | Same dispatch as the HTTP API. |
| Baseten `/production/predict` and `/production/sync/v1/cua` | **Per-deployment** via `MANTIS_MODEL` env | Each pod loads one brain at startup. `cua_model` in the request body is **ignored**. To use Fara on Baseten you push `deploy/baseten/fara/` as its own model and hit that deployment's URL. See [the URL contract below](#baseten-url-contract) for `/production/predict` vs `/production/sync/<path>`. |

Default model is `holo3` on both surfaces — existing callers that don't pass `cua_model` keep getting Holo3.

### Modal — HTTP API

```bash
curl -X POST https://workspace--mantis-cua-server-api.modal.run/v1/predict \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "detached":     true,
    "micro":        "plans/example/extract_listings.json",
    "profile_id":   "marketplace-prod",
    "workflow_id":  "marketplace-listings-v1",
    "cua_model":    "fara",
    "max_cost":     2,
    "max_time_minutes": 20
  }'
```

`GET /v1/models` returns the list the dispatcher accepts:

```jsonc
{ "data": [
  { "id": "holo3" }, { "id": "fara" }, { "id": "gemma4-cua" },
  { "id": "evocua-8b" }, { "id": "evocua-32b" },
  { "id": "opencua-32b" }, { "id": "opencua-72b" }, { "id": "claude" }
]}
```

### Modal — CLI

```bash
uv run modal run --detach deploy/modal/modal_cua_server.py \
  --micro plans/example/extract_listings.json \
  --model fara \
  --max-cost 2 --max-time-minutes 20 \
  --profile-id marketplace-prod \
  --workflow-id marketplace-listings-v1
```

### Baseten — separate deployment per model

```bash
# Push Fara as its own Baseten model
uvx truss push deploy/baseten/fara --no-cache --promote \
  --deployment-name "mantis-fara-$(date -u +%Y%m%d-%H%M)"

# Call the orchestrated /predict route
curl -X POST "https://model-<FARA_MODEL_ID>.api.baseten.co/production/predict" \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -H "X-Mantis-Token: $TOK" \
  -H "Content-Type: application/json" \
  -d '{ "detached": true, "micro": "plans/example/extract_listings.json",
        "profile_id": "alice", "workflow_id": "v1",
        "max_cost": 2, "max_time_minutes": 20 }'
```

Notice the request body has no `cua_model` field — the Fara deployment was started with `MANTIS_MODEL=fara` (see `deploy/baseten/fara/config.yaml`), and that's what determines the brain. Holo3 stays on its own deployment URL.

#### Baseten URL contract

The Baseten gateway forwards **two** route families to every truss-server deployment:

| Public URL | Container path | What it serves |
|---|---|---|
| `/production/predict` | `predict_endpoint:` from `config.yaml` (default `/predict`) | Mantis orchestrated run / status / resume — the high-level entry point. |
| `/production/sync/<any-path>` | `<any-path>` on the in-pod FastAPI / nginx (port `server_port`) | Pass-through for arbitrary FastAPI routes — `/v1/chat/completions`, `/v1/models`, `/v1/health`, `/v1/cua`. |

For a non-promoted (canary) deployment, swap `production` for `deployment/<DEPLOYMENT_ID>` in either form.

This is how a remote-brain integration (caller runs the CUA loop locally, hits Mantis only for per-step inference) points a `FaraBrain` at the Baseten deployment:

```python
endpoint = "https://model-<FARA_MODEL_ID>.api.baseten.co/production/sync"

brain = FaraBrain(
    base_url=f"{endpoint}/v1",                 # → /v1/chat/completions
    extra_headers={
        "Authorization": f"Api-Key {BASETEN_API_KEY}",
        "X-Mantis-Token": MANTIS_API_TOKEN,
    },
    screen_size=(1280, 800),
)
```

The brain only hits `/v1/chat/completions` (single-step inference). The CUA loop (screenshot → think → action → step) runs in the caller's container against the caller's `XdotoolGymEnv` — the Mantis Baseten pod is a thin model server in that mode.

Use `/production/predict` instead when you want Mantis itself to run the CUA loop on its own headed Chrome inside Xvfb (the orchestrated path).



| Name | Repo | Base | Serve | GPU | License |
|---|---|---|---|---|---|
| `holo3` | `Hcompany/Holo3-35B-A3B` (GGUF: `mradermacher/Holo3-35B-A3B-GGUF`) | Qwen3.5-VL MoE | llama.cpp + Q8_0 GGUF | 1× A100-80GB / H100 / L40S 48GB | Hcompany terms |
| `fara` | `microsoft/Fara-7B` | Qwen2.5-VL | vLLM (bf16) | 1× A100-40GB / L40S / A10G | MIT |
| `gemma4-cua` | local fine-tune (Gemma4-31B-CUA) | Gemma4 | llama.cpp + GGUF | 1× A100-80GB | Gemma terms |
| `evocua-8b` / `evocua-32b` | `meituan/EvoCUA-*` | Qwen2-VL | vLLM | 1-2× A100-80GB | EvoCUA terms |
| `opencua-32b` / `opencua-72b` | `xlangai/OpenCUA-*` | Qwen2-VL | vLLM | 4-8× A100-80GB | OpenCUA terms |
| `claude` | Anthropic API | — | API | 0 | API |

## Fara-7B

Microsoft's compact 7B CUA, Qwen2.5-VL-based, MIT-licensed. Smaller and cheaper to serve than Holo3 (single A100-40GB instead of A100-80GB), faster cold start because vLLM serves Qwen2.5-VL natively (no llama.cpp / GGUF detour), and competitive on small-model CUA benchmarks.

### Action space

Fara emits one `computer_use` tool call per turn. The brain adapter maps it to Mantis actions as follows:

| Fara action | Mantis action | Notes |
|---|---|---|
| `left_click` | `CLICK` | Coords are raw screen pixels at the screenshot's input resolution; the brain scales back to actual viewport. |
| `type` | `TYPE` | — |
| `key` | `KEY_PRESS` | Lists are joined with `+` (`["ctrl", "a"]` → `"ctrl+a"`). |
| `scroll` | `SCROLL` | Carries `scroll_direction` + `scroll_amount`. Optional `coordinate` scales to viewport. |
| `wait` | `WAIT` | `duration` → `seconds`. |
| `visit_url` | `TYPE(url)` | Rides the env's URL auto-navigate path (`xdotool_env.py:955`) — `ctrl+l` → paste → Return. |
| `history_back` | `KEY_PRESS("alt+Left")` | — |
| `web_search` | `TYPE("https://www.google.com/search?q=…")` | Same URL auto-navigate path. |
| `mouse_move` | `WAIT(0.2s)` | No-op. xdotool clicks already `mousemove` before press; explicit mouse-only moves collapse harmlessly. |
| `pause_and_memorize_fact` | `WAIT(0.2s)` | The fact is stashed in `Action.reasoning` for the trajectory. |
| `terminate` | `DONE` | `status="success"` → `success=True`. |

### Regressions vs Holo3

Fara has no native `double_click`, `right_click`, or `drag` in its training. The brain will accept and forward those if a planner adapter emits them, but the underlying model won't volunteer them. If your plan needs any of these, prefer Holo3.

### Coordinate handling

Fara was trained at a fixed input resolution (default 1428×896). The brain resizes the screenshot to that resolution before sending and scales emitted `coordinate` values back to the real viewport. Override via the constructor `input_size` kwarg or the `MANTIS_FARA_INPUT_WH` env var (`"1366x768"` form).

### Serving

* **Modal** — `cua_model="fara"` routes through `run_fara` on a single A100-40GB. The shared `_run_executor` path downloads `microsoft/Fara-7B` to the volume on first run, starts vLLM, and instantiates `FaraBrain`. See [Modal hosting](../hosting/modal.md).
* **Baseten** — `truss push deploy/baseten/fara/` ships the same FastAPI surface as Holo3 with vLLM (no llama.cpp build). Set `MANTIS_MODEL=fara`; `MANTIS_FARA_MODEL_DIR` defaults to `/models/fara` where the `weights:` block mounts the HF snapshot. See [Baseten hosting](../hosting/baseten.md).

### System prompt

`src/mantis_agent/prompts/files/fara_system.txt` — Microsoft's recommended template with the "Critical Point" stop semantics (checkout / book / call / order). Override per-tenant via `MANTIS_PROMPTS_DIR/fara_system.txt`.

## Holo3-35B-A3B

Hcompany's Qwen3.5-VL MoE (35B total, 3B active). Strong OSWorld-Verified score (77.8%), but served via llama.cpp because vLLM doesn't yet support `qwen3_5_moe`. The Modal `run_holo3` executor uses a 1× A100-80GB; the Baseten `holo3` config pulls the Q8_0 GGUF (~34 GB) + mmproj (~0.8 GB) from `mradermacher/Holo3-35B-A3B-GGUF`.

See `src/mantis_agent/brain_holo3.py` for the five-strategy response parser (tool_calls → native text → JSON → pyautogui → keyword) and Qwen smart-resize coordinate handling.
