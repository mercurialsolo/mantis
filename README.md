# CUA Agent — Streaming Computer Use Agent with Gemma 4

A unified perception-reasoning-action agent for computer use, powered by Google's Gemma 4 model family. Unlike traditional decoupled CUA pipelines (grounding model -> reasoning model -> executor), this agent uses a single Gemma 4 forward pass for all three stages, with a rolling frame buffer for temporal context.

## Architecture

```
Traditional (Agent-S, etc.):
  screenshot -> grounding_model -> reasoning_model -> exec -> screenshot -> repeat
  (Serial. ~3-5s per cycle. Model never sees transitions.)

StreamingCUA (this repo):
  continuous_frames -> gemma4(perceive+reason+act) -> execute -> model sees results
  (One model, one pass. Frame buffer gives temporal context. Tight feedback loop.)
```

### Core Components

| Module              | Purpose                                               |
|---------------------|-------------------------------------------------------|
| `brain.py`          | Gemma 4 inference — perceive + reason + act in one pass |
| `brain_llamacpp.py` | llama.cpp backend for GGUF quantized models           |
| `agent.py`          | Streaming perception-action loop with loop detection  |
| `actions.py`        | Action types and Gemma 4 native tool-call schemas     |
| `executor.py`       | pyautogui-based action execution                      |
| `streamer.py`       | Background screen capture into rolling frame buffer   |
| `hud_mcp_agent.py`  | HUD benchmark integration via MCP                     |

### Deployment

- **Local**: `brain.py` with HuggingFace Transformers (GPU required)
- **Modal A100**: `modal_osworld_direct.py` — llama-server with CUDA + QEMU VM
- **HUD eval**: `run_hud_osworld.py` — OSWorld-Verified benchmark via HUD platform

## Current Baseline

- **Model**: Gemma 4 E4B-it (4.5B params), Q4_K_M quantization for deployment
- **Benchmark**: OSWorld-Verified via HUD
- **Inference**: llama.cpp on A100, OpenAI-compatible API
- **Action space**: pyautogui (click, type, scroll, drag, key_press, wait, done)

---

## Improving CUA Scores with Gemma

The sections below outline concrete strategies for improving OSWorld benchmark performance, organized from highest expected impact to lowest.

### 1. Model Selection and Scaling

**Current**: Gemma 4 E4B-it (4.5B params, Q4_K_M)

**Improvements**:
- **Upgrade to 26B-A4B MoE variant** — Only 4B params active per token (similar latency to E4B) but draws from 26B total capacity. Best accuracy/speed tradeoff for CUA tasks that need deeper reasoning (multi-step file operations, complex UI navigation).
- **Use 31B for hard tasks** — For tasks where the agent consistently fails, route to the full 31B dense model. Accept higher latency (~2-3x) for complex reasoning (e.g., spreadsheet formulas, multi-app workflows).
- **Adaptive model routing** — Classify task difficulty from the instruction text and route: E4B for simple tasks (open app, click button), 26B-A4B for medium (multi-step navigation), 31B for hard (complex state reasoning). This keeps average latency low while boosting hard-task scores.
- **Higher quantization for hard tasks** — Use Q8 or FP16 instead of Q4_K_M when accuracy matters more than throughput. Q4 loses precision on fine-grained coordinate prediction.

### 2. Prompt Engineering and System Prompt Optimization

**Current**: Generic system prompt with basic guidelines.

**Improvements**:
- **Task-type-specific system prompts** — Different prompts for different OSWorld domains (OS, browser, office, coding). Each prompt encodes domain-specific knowledge: "In LibreOffice Calc, cells are addressed as A1, B2..." or "In Firefox, the URL bar can be focused with Ctrl+L."
- **Few-shot action examples** — Include 2-3 worked examples in the prompt showing the full observe->reason->act chain for common patterns (opening apps, navigating menus, filling forms). Gemma 4's long context can handle this without truncation.
- **Structured observation format** — Instead of just "look at the screen", prompt the model to explicitly describe what it sees before acting: "1. I see [description]. 2. To accomplish [task], I need to [plan]. 3. I will [action]." This forces grounding before action.
- **Error recovery instructions** — Add explicit guidance: "If your last action didn't produce the expected result, describe what went wrong before trying again. If you've tried the same approach 3 times, try an alternative."

### 3. Multi-Frame Temporal Reasoning

**Current**: 5 frames per inference, basic temporal labeling (t-4, t-3, ..., CURRENT).

**Improvements**:
- **Adaptive frame selection** — Instead of the last N frames at fixed intervals, select frames that show meaningful state changes (diff-based). Drop near-identical frames. This gives the model more information per frame slot.
- **Increase frame budget for complex tasks** — Some tasks (watching a download progress, waiting for a compile) benefit from more temporal context. Dynamically increase `frames_per_inference` when the agent calls `wait()`.
- **Frame annotations** — Annotate each frame with what action was taken between it and the next frame: "[Frame t-2] -> click(500, 300) -> [Frame t-1] -> type('hello') -> [Frame CURRENT]". This creates an explicit visual-action chain.
- **Keyframe detection** — Only trigger inference when the screen changes meaningfully (pixel diff above threshold). Saves inference budget for tasks with long loading times.

### 4. Action Space and Execution

**Current**: 8 action types (click, double_click, type_text, key_press, scroll, drag, wait, done).

**Improvements**:
- **Compound actions** — Add `select_text(start_x, start_y, end_x, end_y)`, `copy()`, `paste()`, `open_app(name)` as higher-level primitives. Reduces the number of steps needed for common operations (copy-paste currently takes 4 actions: click, shift+click, ctrl+c, ctrl+v).
- **Coordinate refinement** — After Gemma 4 predicts approximate coordinates, use a second pass with a cropped region around the target for pixel-precise clicking. Small models like E2B are fast enough for this refinement step.
- **Action verification** — After each action, compare pre/post screenshots. If the screen didn't change as expected, automatically retry with adjusted coordinates before consuming a step.
- **Smart typing** — For long text inputs, use clipboard paste instead of character-by-character typing. Faster and avoids typo accumulation from key-press timing issues.

### 5. Planning and Decomposition

**Current**: Single-step reasoning per inference. Thinking mode enabled but not structured.

**Improvements**:
- **Explicit planning phase** — Before the first action, run an inference with "Plan the steps to complete this task" and cache the plan. Each subsequent inference includes the plan for context. This is especially valuable for multi-step tasks (the majority of OSWorld).
- **Subgoal tracking** — Break the cached plan into subgoals. After each action, check if the current subgoal is complete before moving to the next. This prevents the agent from losing track of where it is in a complex task.
- **Backtracking** — If the agent detects it's off-plan (current screen doesn't match any expected state), explicitly reason about recovery: undo, close dialog, or restart from a known state.
- **Budget allocation** — For a 15-step budget, plan how many steps to allocate per subgoal. If a subgoal is consuming too many steps, skip it or try an alternative approach.

### 6. Training and Fine-tuning

**Current**: Using Gemma 4 off-the-shelf with no task-specific fine-tuning.

**Improvements**:
- **SFT on CUA trajectories** — Fine-tune on successful OSWorld trajectories (from stronger models like GPT-4o or Claude). Focus on the observe->reason->act format with Gemma 4's tool-calling format. Even a small dataset (500-1000 trajectories) should significantly improve action accuracy.
- **Coordinate grounding fine-tuning** — Create a dataset of (screenshot, element_description, coordinates) pairs from web/desktop UIs. Fine-tune the model to precisely locate UI elements. This is the #1 failure mode for small models.
- **DPO on success/failure pairs** — Collect pairs of successful and failed trajectories for the same task. Use DPO to steer the model toward successful action patterns.
- **Domain-specific LoRA** — Train separate LoRA adapters for each OSWorld domain (OS, browser, office, coding). Activate the right adapter based on the task domain. Keeps base model unchanged while adding specialized knowledge.

### 7. Inference Optimization

**Current**: llama.cpp with Q4_K_M on A100, `do_sample=False`, `max_new_tokens=1024`.

**Improvements**:
- **Speculative decoding** — Use E2B as the draft model for 26B-A4B or 31B. Same model family means high acceptance rate. Can 2-3x throughput without accuracy loss.
- **KV cache reuse** — The system prompt and tool definitions are constant across steps. Cache their KV states and reuse across inference calls. Saves ~30% of prefill time.
- **Parallel prefill** — Process the frame images in parallel during prefill. The visual encoder can process frames independently before the language model attends to them.
- **Context pruning** — For long trajectories (>10 steps), summarize old action history instead of including raw text. Keeps context length manageable without losing important state.

### 8. Evaluation and Error Analysis

**Improvements**:
- **Per-domain error taxonomy** — Classify failures into categories: wrong element clicked, correct element but wrong action, correct action but wrong parameters, task misunderstood, stuck in loop, ran out of steps. This reveals which improvements to prioritize.
- **Failure replay** — Record full trajectories (screenshots + actions) for failed tasks. Replay them to identify the exact step where things went wrong.
- **Ablation pipeline** — Automate A/B testing of prompt variants, frame counts, model sizes, etc. against a fixed task subset. Track score deltas rigorously.
- **Difficulty stratification** — Report scores broken down by task difficulty (number of required steps, number of apps involved, domain). Improvements on easy tasks may not transfer to hard ones.

### 9. Ensemble and Verification

**Improvements**:
- **Self-consistency** — Run the same inference 3 times with `temperature=0.3`. If the actions disagree, re-run with a prompt asking the model to choose between the candidates. Reduces random coordinate errors.
- **Verification agent** — After the main agent calls `done(success=true)`, run a separate verification pass: "Look at the screen. Has the following task been completed? [task]". If the verifier says no, resume the agent. Catches premature completion.
- **Two-model pipeline** — Use Gemma 4 31B as a planner (generates step-by-step instructions) and E4B as an executor (follows instructions with low latency). The planner corrects course when the executor goes off-track.

## Quick Start

```bash
# Install
pip install -e ".[gpu,hud]"

# Run on Modal (A100)
modal run modal_osworld_direct.py --domain os --tasks 5

# Run via HUD
export HUD_API_KEY=<your-key>
python run_hud_osworld.py --model-url http://localhost:8080/v1 --max-tasks 1
```

## Gemma 4 Model Variants

| Variant    | Params    | Active | Best For                          |
|------------|-----------|--------|-----------------------------------|
| E2B        | 2.3B      | 2.3B   | Fast draft model, simple tasks    |
| **E4B**    | **4.5B**  | **4.5B** | **Default. Good speed/accuracy**  |
| 26B-A4B    | 26B (MoE) | 4B     | Deep reasoning, similar latency   |
| 31B        | 31B       | 31B    | Maximum accuracy, higher latency  |

## License

MIT
