# Embedding `MicroPlanRunner` in a host application

This is the reference for **hosts that import the `mantis-agent` library and
drive `MicroPlanRunner` in their own process**. If you only call the HTTP
`/v1/predict` endpoint, you don't need this doc; see [Sending plans](../client/plans.md)
instead.

> Companion reading: the [any-agent integration playbook](any-agent.md)
> covers the runtime contract and pre-flight checklist your host's env
> wrapper must satisfy.

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

These are the four primitives a host integration typically reaches for.
Each is opt-in — runs that don't set them see no change in behaviour.

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

### 2. `cancel_event` — clean SIGTERM exit ([#76](https://github.com/mercurialsolo/mantis/issues/76), [#288](https://github.com/mercurialsolo/mantis/issues/288))

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

`GymRunner` accepts the same `cancel_event` kwarg with the same shape
(#288). On a positive check, `GymRunner` returns
`RunResult(paused=True, pause_state=..., termination_reason="cancelled")`
with `pause_state.pending_reason="cancelled"` — snapshot shape is
identical to the `PauseRequested` path, so `runner.resume(pause_state)`
rehydrates via the same code path. `user_input` is optional on a
cancellation resume (the host typically just continues; pass it only if
the cancellation was user-driven).

```python
shutdown = threading.Event()
runner = GymRunner(brain=brain, env=env, cancel_event=shutdown)

result = runner.run(task="onboard the new user")
if result.paused and result.termination_reason == "cancelled":
    save_to_db(result.pause_state.to_dict())
    return  # release the worker; resume later from the snapshot
```

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

## The skip-envelope family — advance vs. retry signals

Hosts that wrap mantis in a tool surface (e.g. an LLM orchestrator
calling `MantisBrowserTool.run_sub_goal(...)`) routinely face the same
question on every halted / rejected sub-goal: "should the orchestrator
retry, or advance to the next position?"

Plan-text rules like *"never re-issue Step 4 with the same
position_on_page"* don't bind LLM orchestrator behavior reliably — they
get treated as preferences and overridden. Tool-result envelopes that
change the orchestrator's downstream state are the durable mechanism.

`StepResult` carries two fields for this:

```python
@dataclass
class StepResult:
    ...
    skip: bool = False         # advance, don't retry — runner says this run is terminal-for-this-context
    skip_reason: str | None = None  # canonical token the host branches on
```

When `skip=True`, the host's tool wrapper promotes the result to
`status: skipped` (a successful tool result with an advance semantic),
which the orchestrator already handles cleanly. When `skip=False`,
today's success/failure semantics apply.

Five opt-in trigger sources populate the envelope. All defaults preserve
today's unbounded behavior — runs that don't opt in see no change:

### a) Recipe-rejection intents ([#246](https://github.com/mercurialsolo/mantis/pull/247))

Recipes annotate which rejection reasons mean *terminal-for-this-row*
(`"skip"`) vs *retryable / enrichable* (`"extract_more"` / `"retry"`):

```python
from mantis_agent.extraction import ExtractionSchema

ExtractionSchema(
    spam_indicators=[...],
    spam_label="dealer",
    rejection_intents={
        "dealer":              "skip",           # terminal — host advances
        "incomplete_required": "extract_more",   # may enrich on detail page
    },
)
```

When `ClaudeExtractor` flags a row as dealer/spam (recipe-defined), the
runner stamps `skip=True, skip_reason="dealer"` and the host advances
past the listing instead of treating the rejection as a step failure to
retry. `marketplace_listings` ships with this pre-wired.

### b) Navigation-primitive halts ([#250](https://github.com/mercurialsolo/mantis/pull/251))

When a click / submit / scroll / navigate / gate step exhausts retries
and halts, the host can opt to convert that into a skip signal so the
orchestrator advances rather than re-attempting the same intent:

```python
runner = MicroPlanRunner(
    ...,
    navigation_primitives_emit_skip={"click", "submit", "scroll", "navigate", "gate"},
)
```

On halt, if the failed step's type is in the set and the step doesn't
already carry a recipe-rejection `skip_reason` (those win on conflict),
the runner stamps `skip=True, skip_reason="navigation_failed"`. Default
`None` preserves today's halt-as-failure semantics.

### c) Per-context sub-goal budget ([#254](https://github.com/mercurialsolo/mantis/pull/256))

Bound how many sub-goals (= `run()` calls) can fire against the same
URL anchor before the runner short-circuits:

```python
from mantis_agent.gym.context_budget import ContextBudget

runner = MicroPlanRunner(
    ...,
    context_budget=ContextBudget(
        max_sub_goals_per_url=3,         # bound per detail-page URL
        max_sub_goals_per_iteration=10,  # global cap across all URLs
        on_exceeded="emit_skip",         # vs "halt" / "log_only"
    ),
)
```

When the (N+1)th `run()` would fire against a URL that already has N
sub-goals, the runner short-circuits *before* invoking the executor and
returns `StepResult(skip=True, skip_reason="listing_budget_exceeded")`.
Anchor resolution: navigate-step URL → `_last_known_url` → sentinel.

### d) Already-seen URL predicate ([#255](https://github.com/mercurialsolo/mantis/pull/257))

Cross-session dedup: pass a predicate the runner consults at the top
of `extract_data` to skip URLs the host has already processed in prior
runs:

```python
def seen(url: str) -> bool:
    return url in already_extracted_urls  # or content_hash / CRM lookup

runner = MicroPlanRunner(
    ...,
    seen_url_predicate=seen,
)
```

The predicate is consulted *after* navigate but *before*
ClaudeExtractor fires — saves the deep-extract Claude call entirely.
Applicability gated by `SiteConfig.is_detail_page(url)` so search /
results URLs aren't accidentally deduped. The host owns the dedup
policy (URL match, content hash, CRM lookup, anything); mantis just
provides the timing window.

### Putting it together — what the host sees

```python
result = runner.run_with_status(plan)

for step in result.steps:
    if step.skip:
        # advance: terminal-for-this-context; don't retry
        log.info("skipped step %d: %s", step.step_index, step.skip_reason)
        continue
    if step.success:
        process_lead(step.data)
    else:
        # legitimate failure — host's retry policy applies
        ...
```

The canonical `skip_reason` values are:

| `skip_reason` | Trigger | When |
|---|---|---|
| `"dealer"` (or any recipe key) | Recipe-rejection intents | Extractor flagged the row + recipe says `"skip"` |
| `"navigation_failed"` | Navigation-primitive halt | Step type in opt-in set + recovery exhausted |
| `"listing_budget_exceeded"` | Context budget | (N+1)th sub-goal against same URL anchor |
| `"already_seen"` | Already-seen predicate | Predicate returned True at top of `extract_data` |

A more-specific recipe reason (`"dealer"`) always wins over a generic
runner reason (`"navigation_failed"`) — once a `StepResult` carries a
`skip_reason`, later layers don't overwrite it.

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

When your host carries an existing CUA backend that you want to keep
working alongside Mantis, the typical invariants to preserve are:

- Don't refactor your existing backend's class — add Mantis as a sibling
  selected by an env flag.
- Don't change the agent-state blob shape for the existing backend's
  runs; just add a new key for Mantis runs alongside.
- Default the env flag to the existing backend so nothing flips without
  an explicit opt-in.

## Sharing this with another host

If you're integrating Mantis into a fresh host, the relevant docs in
pull-this-order are:

1. [Integrating any agent](any-agent.md) — runtime contract + pre-flight
   checklist + the integration mistakes to avoid.
2. This doc — what to import + the four host-integration knobs.
3. [`reference/coordinate-spaces.md`](../reference/coordinate-spaces.md) — required reading before you write a `step()` method.
4. [`reference/glossary.md`](../reference/glossary.md) — quick definitions for terms used throughout.
5. [`reference/env-vars.md`](../reference/env-vars.md) — server-side env vars (only relevant if you self-host the Mantis service rather than calling Baseten).
