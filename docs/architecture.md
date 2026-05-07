# Mantis CUA Architecture

Mantis is a Computer Use Agent that fuses perception, reasoning, and action into a single model. One brain consumes screen frames + task + history and outputs actions, observing consequences in real-time.

## System Overview

```
User objective (text)
  |
  v
ObjectiveSpec.parse()          Parse objective into structured spec
  |
  v
GraphLearner                   Probe site + generate dependency graph
  |  |
  |  +-> SiteProber            Screenshot-based page analysis (no brain needed)
  |  +-> WorkflowGraph         DAG of phases with loop semantics
  |
  v
GraphCompiler                  Compile graph -> flat MicroPlan
  |
  v
PlanValidator                  Structural checks + auto-fix
  |
  v
MicroPlanRunner                Execute with checkpoint/verify/reverse
  |  |
  |  +-> Brain.think()         Perception + reasoning + action
  |  +-> GymEnvironment.step() Execute action, capture screenshot
  |  +-> ClaudeExtractor       Read structured data from screenshots
  |  +-> ClaudeGrounding       Refine click coordinates
  |  +-> DynamicPlanVerifier   Track coverage per page
  |
  v
ExtractionResult               Structured output (fields, viability, spam check)
```

## Module Map

### Core (`src/mantis_agent/`)

| Module | Purpose |
|--------|---------|
| `brain.py` | Gemma4Brain -- unified vision-language model |
| `brain_holo3.py` | Holo3-35B via llama.cpp |
| `brain_llamacpp.py` | Generic llama.cpp bridge for local GGUF models |
| `brain_opencua.py` | OpenCUA 32B/72B via vLLM tensor-parallel |
| `brain_claude.py` | Claude Sonnet/Opus via Anthropic API |
| `actions.py` | Action enum (click, type, scroll, done) and tool schemas |
| `extraction.py` | ClaudeExtractor + ExtractionSchema -- schema-driven data extraction |
| `grounding.py` | ClaudeGrounding / LLMGrounding -- pixel-level click targeting (cache-aware via `cache=` kwarg) |
| `grounding_cache.py` | GroundingCache -- TTL+LRU cache keyed by `(perceptual_hash(crop), description_hash)` (#117) |
| `plan_decomposer.py` | Text plan -> MicroPlan via Claude Sonnet |
| `site_config.py` | SiteConfig -- URL patterns, pagination, gate prompts, `prefer_som_grounding` flag (#117) |
| `loop_detector.py` | LoopDetector -- byte-equal / drift / state-loop signals (#116) |
| `brain_ladder.py` | BrainLadder -- primary/fallback escalation, emits `mantis_brain_escalation_total` (#156) |
| `metrics.py` | Prometheus handles -- per-action / brain / loop / grounding-cache (#156, #117) |
| `server_utils.py` | Shared utilities -- proxy, plan signatures, result builders |
| `task_loop.py` | TaskLoopConfig + run_task_loop -- shared executor lifecycle |

### Graph Learning (`src/mantis_agent/graph/`)

| Module | Purpose |
|--------|---------|
| `objective.py` | ObjectiveSpec -- structured objective with fields, filters, completion |
| `graph.py` | WorkflowGraph, PhaseNode, PhaseEdge -- DAG with repeat modes |
| `probe.py` | SiteProber -- navigate + screenshot + Claude analysis (no brain) |
| `compiler.py` | GraphCompiler -- WorkflowGraph -> MicroPlan |
| `learner.py` | GraphLearner -- orchestrates probe + skeleton + sample + cache |
| `store.py` | GraphStore -- persist/load keyed by domain + objective hash |
| `plan_validator.py` | PlanValidator -- structural checks + auto-fix before execution |

### Execution (`src/mantis_agent/gym/`)

| Module | Purpose |
|--------|---------|
| `runner.py` | GymRunner -- step-level agent loop, SoM-promotion dispatch (#117), plan-step loop reset (#116) |
| `micro_runner.py` | MicroPlanRunner -- 200 LOC coordinator post-#161; delegates via `__getattr__` |
| `_runner_helpers.py` | Module-level helpers (dispatch, settle, observability, reverse, filters) -- EPIC #161 |
| `run_executor.py` | RunExecutor + RunState -- run loop body, emits `mantis_action_total` / `mantis_loop_termination_total` (#156) |
| `step_handlers/` | Per-type handlers (NavigateHandler, ClaudeGuidedClickHandler, PaginateHandler, ...) |
| `step_recovery.py` | StepRecoveryPolicy -- failure → RecoveryAction dispatch table |
| `step_context.py` | StepContext + HandlerRegistry -- mocking surface for handler unit tests |
| `listings_scanner.py` | ListingsScanner -- per-page card cache + dedup, `is_duplicate` / `mark_seen` / `on_page_change` |
| `browser_state.py` | BrowserState -- scroll position, viewport, results-page URL composition |
| `checkpoint_manager.py` | CheckpointManager -- persist/restore + plan-signature flow |
| `cost_meter.py` | CostMeter -- gpu/claude/proxy cost rollup, Prometheus inflight gauge |
| `run_reporter.py` | RunReporter -- final summary block + costs dict |
| `tool_channel.py` | ToolChannel -- pause/resume tool invocation |
| `trace_exporter.py` | TraceExporter -- per-run JSON trace dump for SFT/DPO labelling, gated on `MANTIS_TRACE_EXPORT_DIR` (#155) |
| `trace_labeller.py` | TraceLabeller -- heuristic labels (positive/negative/neutral) over exported traces (#155 step 2) |
| `shadow_router.py` | ShadowRouter -- deterministic A/B variant assignment for shadow-deploy (#155 step 5) |
| `workflow_runner.py` | WorkflowRunner -- dynamic loops and pagination over GymRunner |
| `learning_runner.py` | LearningRunner -- verified execution for building playbooks |
| `xdotool_env.py` | XdotoolGymEnv -- real Chrome + xdotool (zero automation fingerprints) |
| `playwright_env.py` | PlaywrightGymEnv -- headless Chromium via Playwright |
| `plan_executor.py` | PlanExecutor -- deterministic DOM-based step execution |
| `page_discovery.py` | PageDiscovery -- DOM inspection for SoM-style element selection |

### Verification (`src/mantis_agent/verification/`)

| Module | Purpose |
|--------|---------|
| `dynamic_plan_verifier.py` | Per-page coverage tracking (found/attempted/opened/completed) |
| `step_verifier.py` | Before/after screenshot comparison via Claude |
| `playbook.py` | PlaybookStore -- learned site-specific steps with confidence scores |

## Key Interfaces

### Brain Protocol

```python
class Brain(Protocol):
    def think(
        frames: list[Image.Image],
        task: str,
        action_history: list[Action] | None,
    ) -> InferenceResult
```

Implementations: Gemma4Brain, Holo3Brain, LlamaCppBrain, OpenCUABrain, ClaudeBrain

### GymEnvironment

```python
class GymEnvironment(ABC):
    def reset(task, start_url) -> GymObservation
    def step(action) -> GymResult
    def screenshot() -> Image.Image
    def close()
```

Implementations: XdotoolGymEnv (real Chrome), PlaywrightGymEnv (headless)

### ExtractionSchema

```python
@dataclass
class ExtractionSchema:
    entity_name: str              # "boat listing", "job posting"
    fields: list[OutputField]     # what to extract
    required_fields: list[str]    # viability check
    spam_indicators: list[str]    # what to reject
    allowed_controls: list[str]   # safe reveal buttons
    forbidden_controls: list[str] # lead-form traps
```

Drives ClaudeExtractor prompts dynamically. Default: marketplace-listings schema (`mantis_agent.recipes.marketplace_listings.schema.SCHEMA`).

### SiteConfig

```python
@dataclass
class SiteConfig:
    detail_page_pattern: str      # regex for detail URLs
    results_page_pattern: str     # regex for results URLs
    pagination_format: str        # "/page-{n}/" or "?page={n}"
    gate_verify_prompt: str       # what to check after filters
    filtered_results_url: str     # recovery URL if filters lost
    prefer_som_grounding: bool    # #117 — promote SoM dispatch over Claude grounding
```

Used by MicroPlanRunner for URL checks instead of hardcoded patterns. The
`prefer_som_grounding` flag is read by `GymRunner` to decide whether to try
DOM-discovery + brain choice **before** the direct executor; on SoM-friendly
sites this saves a Claude grounding round-trip per click.

### Grounding cache (#117)

```python
from mantis_agent.grounding import ClaudeGrounding
from mantis_agent.grounding_cache import GroundingCache

cache = GroundingCache()  # default: 1024 entries, 1 h TTL
grounder = ClaudeGrounding(cache=cache)
```

Identical `(perceptual_hash(crop), description_hash)` pairs short-circuit the
Claude API call. The `Runtime` class in `baseten_server/runtime.py` constructs
a single process-level cache and shares it across both `ClaudeGrounding`
instantiations so layouts hit cache across pages, runs, and tenants.

### RewardFn (#120, #155)

```python
class RewardFn(Protocol):
    def step(*, action, gym_result, state) -> RewardSignal: ...
    def episode(*, run_result, state, ground_truth=None) -> RewardSignal: ...
```

Per-step shaping + terminal grading. `PlanAdherenceReward` composes
`format_reward`, `off_site_penalty`, `loop_penalty`, `task_success_reward`,
and (post-#120) `world_model_accuracy_reward` — Jaccard similarity between
each step's `predicted_outcome` (parsed from the brain's `Predicted:` line)
and `observed_outcome` (the runtime's feedback string). Set
`world_model_weight=0.0` to disable; default `0.05`.

## Execution Flow

### Phase 1: Plan Generation

```
Text objective
  -> ObjectiveSpec.parse()    Extract domain, entity, filters, schema
  -> GraphLearner.learn()     Check cache -> probe site -> generate skeleton
  -> GraphCompiler.compile()  DAG -> flat MicroPlan
  -> PlanValidator.enhance()  Fix missing navigate/gate/loops
```

### Phase 2: Filter Application (Setup)

```
navigate -> filter_0 -> filter_1 -> ... -> gate_verification
```

Each filter is a separate required step. If any fails, pipeline halts.
Gate checks all filters are active before extraction begins.

### Phase 3: Extraction Loop

```
for each discovered item on page:
    click title -> extract URL -> scroll to details
    -> expand collapsed sections -> extract data -> go back

when page exhausted:
    paginate -> loop back to discovery
```

Coverage tracked by DynamicPlanVerifier:
- found items, attempted items, opened items, completed items
- 7 structural checks per page (filters, attempts, completions, exhaustion)

### Phase 4: Result

```
ExtractionResult per item:
  - extracted_fields: {name: value}
  - is_viable(): required fields present + not spam
  - to_summary(): "VIABLE | Year: 2020 | Make: Sea Ray | ..."

Run result:
  - leads count, phone lead count
  - costs (GPU, Claude API, proxy bandwidth)
  - dynamic_verification_summary with per-page checks
  - checkpoint for resume
```

## Deployment

### Modal

```
deploy/modal/modal_cua_server.py
  |
  +-> gemma4_planner()       Persistent T4, llama.cpp, /v1/chat/completions
  +-> run_holo3()            Per-run A100, llama.cpp GGUF
  +-> run_gemma4_cua()       Per-run A100, llama.cpp GGUF
  +-> run_cua_*gpu()         Per-run A100s, vLLM (EvoCUA/OpenCUA)
  +-> run_claude_cua()       Per-run CPU only, Anthropic API
```

All executors delegate to `task_loop.run_executor_lifecycle()`.
Executor-specific behavior via callbacks: `on_task_result`, `on_task_complete`, `on_loop_complete`.

### Baseten

```
baseten_server.py (FastAPI)
  |
  +-> POST /predict          Run micro-plan or task suite
  +-> action=graph_learn     Probe + graph (CPU only)
  +-> action=status/result   Poll detached runs
```

Uses same `task_loop.run_executor_lifecycle()` as Modal.

### Local

```python
set -a && source .env && set +a
uv run modal run deploy/modal/modal_cua_server.py \
  --micro plans/example/extract_listings.json \
  --model holo3 --max-cost 0.30
```

## Cost Model

| Component | Cost | When |
|-----------|------|------|
| Plan decomposition | ~$0.01 | Once per text plan (cached) |
| Graph skeleton | ~$0.01 | Once per objective (cached) |
| Site probing | ~$0.02 | Once per domain (4-6 screenshots) |
| Claude extraction | ~$0.003 | Per listing (1-2 screenshots) |
| Claude grounding | ~$0.003 | Per click target refinement |
| Gate verification | ~$0.003 | Per gate check |
| GPU (Holo3/Gemma4) | ~$3.25/hr | During execution |
| Proxy bandwidth | ~$5/GB | During browser sessions |

Typical run: ~$0.14/lead extracted (2 Claude calls + GPU time + proxy).

## Directory Structure

```
cua-agent/
  src/mantis_agent/
    graph/              Graph learning, compilation, validation
    gym/                Environments, runners, plan execution
    verification/       Coverage tracking, step verification, playbooks
    curriculum/         Domain-specific interaction techniques
    tools/              Utility helpers
    extraction.py       Schema-driven screenshot data extraction
    grounding.py        Claude-based click targeting
    site_config.py      Domain-specific URL patterns
    server_utils.py     Shared proxy, result builders, CSV
    task_loop.py        Shared executor lifecycle
    plan_decomposer.py  Text -> MicroPlan via Claude
  deploy/
    modal/              Modal CLI entrypoints (modal_cua_server.py, modal_osworld_*.py, ...)
    baseten/            Baseten Truss deployments (holo3, gemma4, gemma4_26b)
  docker/               Containerfiles (cua, hud, local)
  scripts/              CLI tools (run_*.py, check_*.sh, baseten_workload.py)
  plans/                Plan files (.txt, .json)
  tasks/                Task descriptors
  benchmarks/           OSWorld / VWA benchmark harnesses (per-domain Modal apps)
  training/             Distillation and fine-tuning configs
  tests/                pytest suite
  docs/                 Architecture documentation
```
