# Embedding `MicroPlanRunner` in a host application

This is the reference for **hosts that import the `mantis-agent` library and
drive `MicroPlanRunner` in their own process** — vision_claude is the
canonical example. If you only call the HTTP `/v1/predict` endpoint, you
don't need this doc; see [Sending plans](../client/plans.md) instead.

> Companion reading: [Integration: vision_claude → Mantis](../integration-vision_claude.md)
> (the architectural narrative + sample wiring) and [vision_claude parity
> addendum](../staffai-vision_claude-parity.md) (the seven staffai-side
> patches needed for full parity).

---

## Install

The orchestrator surface is gated behind a pip extra so heavy GPU /
browser deps stay out of host processes:

```bash
pip install "mantis-agent[orchestrator]"
```

The `[orchestrator]` extra pulls only `requests` and `pydantic`. There is
a CI test (`tests/test_orchestrator_surface.py`) that fails the build if a
PR sneaks `torch`, `vllm`, `transformers`, `pyautogui`, or `playwright`
into the orchestrator import chain — so pinning to a tagged release in your
host's `pyproject.toml` is safe.

## Public surface

Everything a host needs is re-exported at the top level:

```python
from mantis_agent import (
    MicroPlanRunner,
    RunnerResult,
    PauseRequested,
    PauseState,
    StepResult,
    MicroPlan,
    MicroIntent,
    PlanDecomposer,
    scale_brain_to_display,
)
```

Lazy-loaded — `import mantis_agent` itself is cheap and side-effect-free.
Submodules are pulled on first attribute access.

## Minimal end-to-end shape

```python
from mantis_agent import MicroPlanRunner, MicroPlan
from mantis_agent.brain_holo3 import Holo3Brain
from mantis_agent.extraction import ClaudeExtractor
from mantis_agent.grounding import ClaudeGrounding

# 1. Wire the brain. extra_headers is the integration knob hosts use to
#    talk to a deployed Mantis service through any host (Baseten / Modal /
#    EKS / GKE).
brain = Holo3Brain(
    base_url=f"{settings.mantis_endpoint}/v1",
    extra_headers={
        "X-Mantis-Token": settings.mantis_api_token,
        # Optional — only when an upstream gateway demands it. Sent verbatim:
        **({"Authorization": settings.mantis_gateway_authorization}
           if settings.mantis_gateway_authorization else {}),
    },
    timeout=180,
)

# 2. Claude helpers go DIRECT to Anthropic. The host already has a key.
extractor = ClaudeExtractor(api_key=settings.anthropic_api_key)
grounding = ClaudeGrounding(api_key=settings.anthropic_api_key)

# 3. Build the runner. The env is host-supplied — typically a
#    GymEnvironment subclass that adapts the host's existing Xvfb desktop.
runner = MicroPlanRunner(
    brain=brain,
    env=my_gym_env,
    grounding=grounding,
    extractor=extractor,
    session_name="my_workflow",
    max_cost=5.0,
    max_time_minutes=30,
)

# 4. Run.
plan = MicroPlan.from_dict(my_plan_payload)  # or PlanDecomposer().decompose_text(prompt)
result = runner.run_with_status(plan)
```

## The four host-integration knobs

These are the four primitives the staffai integration relies on. Each is
opt-in — runs that don't set them see no change in behaviour.

### 1. `step_callback` — per-step observability ([#74](https://github.com/mercurialsolo/mantis/issues/74))

```python
def on_step(idx: int, intent: str, action, ok: bool) -> None:
    log.info("step %d %s: %s", idx, "ok" if ok else "fail", intent)

runner = MicroPlanRunner(..., step_callback=on_step)
```

Every `StepResult` also carries `screenshot_png: bytes | None` — encoded
PNG of the post-step display. Hosts can feed those bytes into their own
sidecar (e.g. browser-context extraction) without parsing message
structure. `keep_screenshots=N` caps retention to the most-recent N runs
to bound memory on long plans.

### 2. `cancel_event` — clean SIGTERM exit ([#76](https://github.com/mercurialsolo/mantis/issues/76))

```python
shutdown = threading.Event()
runner = MicroPlanRunner(..., cancel_event=shutdown)
# ... another thread sets shutdown when SIGTERM arrives.

result = runner.run_with_status(plan)
if result.cancelled:
    # state already persisted in the checkpoint — host returns the
    # equivalent of CUALoopResult(shutdown_requested=True, ...)
    persist_state(result.steps[-1] if result.steps else None)
```

Accepts any object with `.is_set()` or a plain callable. Checked at every
step boundary.

### 3. `register_tool` — host tools to the brain ([#71](https://github.com/mercurialsolo/mantis/issues/71))

```python
for tool in extra_anthropic_tools:
    runner.register_tool(
        name=tool.name,
        schema=tool.to_params(),  # JSON-schema; matches GenericToolAdapter shape
        handler=lambda kwargs, _t=tool: _t(**kwargs),
    )
```

Errors raised by handlers surface as `success=False` step results with a
diagnostic data string (`"tool:NAME:error:TypeName:msg"`) — never silently
swallowed. The exception to swallow-on-error is `PauseRequested`, which
the runner catches and turns into a clean pause (next).

### 4. `PauseRequested` + `runner.resume()` — OTP / 2FA / human-in-the-loop ([#73](https://github.com/mercurialsolo/mantis/issues/73))

```python
def request_user_input(args):
    staged = runner.consume_pause_input(default=None)
    if staged is None:
        # First call — pause the run.
        raise PauseRequested(reason="user_input", prompt=args["prompt"])
    # Resumed call — staged is the user's reply.
    return staged

runner.register_tool(
    "request_user_input",
    {"type": "object", "properties": {"prompt": {"type": "string"}}},
    request_user_input,
)

result = runner.run_with_status(plan)
if result.paused:
    state_blob = result.pause_state.to_dict()  # JSON-safe; store anywhere
    save_to_db(state_blob)
    return  # release the worker; resume on a fresh request

# Later, when the user replies:
state = PauseState.from_dict(load_from_db())
result = runner.resume(state, user_input="123456", plan=plan)
```

`PauseState` round-trips through `json.dumps` — Postgres JSONB friendly.
Plan-signature mismatch on resume raises `ValueError`: you can't resume a
different plan than the one that paused.

## Coordinate-space contract

The brain emits `(x, y)` in the **same pixel space as the screenshot it
saw**. The env dispatches in display pixels. When those two spaces differ
(host pre-resizes screenshots before inference), the host adapter is
responsible for the scaling — never push it onto the brain.

Use the helper:

```python
from mantis_agent import scale_brain_to_display

x_disp, y_disp = scale_brain_to_display(
    x_brain=action.params["x"],
    y_brain=action.params["y"],
    brain_size=brain_image.size,         # (w, h) of the image fed to inference
    display_size=desktop.viewport_size,  # (w, h) of the dispatch target
)
```

Full contract + worked examples + the bug-class history: [`reference/coordinate-spaces.md`](../reference/coordinate-spaces.md).

## `LAUNCH_APP` action ([#72](https://github.com/mercurialsolo/mantis/issues/72))

A new `ActionType.LAUNCH_APP` lets a plan start a desktop binary
explicitly — symmetric with the Claude backend's `bash` tool. Hosts that
want browser launch on demand implement the dispatch in their
`GymEnvironment.step()`:

```python
case ActionType.LAUNCH_APP:
    subprocess.Popen(
        [params["name"], *params.get("args", [])],
        env={**self._desktop_env, **params.get("env", {})},
    )
```

Failure to launch surfaces as a step error rather than a runner crash —
the next screenshot is what the plan checks.

## Backwards-compat invariants

A host integration must preserve everything that worked before — these
invariants are non-negotiable:

- The orchestrator surface is **purely additive**. No constructor argument
  to `MicroPlanRunner` is required for the new behaviour; defaults are
  `None` everywhere.
- `StepResult.to_dict()` excludes `screenshot_png` and `last_action`, so
  the existing checkpoint JSON shape is byte-identical.
- `mantis_agent` import does not pull `torch`, `vllm`, `transformers`,
  `pyautogui`, or `playwright`. A CI test (`tests/test_orchestrator_surface.py`)
  enforces this.
- Existing `MicroPlanRunner.run(plan)` callers receive `list[StepResult]`
  unchanged. Use `run_with_status(plan)` only when you need the
  `RunnerResult` shape.

For the staffai-side invariants (don't refactor `ClaudeCUABackend`, don't
change `vision_claude_state` shape for Claude runs, etc.) see the
[parity addendum](../staffai-vision_claude-parity.md#backwards-compatibility-invariants).

## Sharing this with another host

If you're integrating Mantis into a host besides vision_claude, the
relevant docs in pull-this-order are:

1. This doc — what to import + the four host-integration knobs.
2. [`integration-vision_claude.md`](../integration-vision_claude.md) — the architectural narrative; replace "vision_claude" with your host.
3. [`reference/coordinate-spaces.md`](../reference/coordinate-spaces.md) — required reading before you write a `step()` method.
4. [`reference/glossary.md`](../reference/glossary.md) — quick definitions for terms used throughout.
5. [`reference/env-vars.md`](../reference/env-vars.md) — server-side env vars (only relevant if you self-host the Mantis service rather than calling Baseten).

The `staffai-vision_claude-parity.md` doc is staffai-specific — file paths
and line numbers reference vision_claude internals — but the *shapes* of
the seven sections (per-plan flag, tool layer, browser launch, pause/resume,
observability, SIGTERM, coords) generalize to any host.
