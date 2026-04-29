# vision_claude / staffai — integration parity addendum

Companion to [`integration-vision_claude.md`](./integration-vision_claude.md) and
[`PROPOSAL-mantis-cua-replaces-vision_claude.md`](./PROPOSAL-mantis-cua-replaces-vision_claude.md).
The existing docs cover the **bring-up** shape (`VisionClaudeGymEnv`,
`MantisOrchestratedBackend`, settings, factory). This doc captures the
**staffai-side** changes still needed to reach feature parity with the Claude
CUA backend, so a single plan can choose either backend and behave the same.

> **Scope:** every change listed here lives in
> `staffai/tools/staffai_tools/vision_claude/`. None of it ships in this
> (`mercurialsolo/mantis`) repo. The corresponding library-side capabilities
> are tracked as separate issues on this repo and cross-referenced below.

---

## 1. Per-plan CUA backend selection (no library dependency)

**Goal.** Today the backend is selected by the process-wide env var
`VISION_CLAUDE_CUA_BACKEND` (read by `VisionClaudeSettings.cua_backend`).
Every plan running in the same vision instance gets the same backend. We want
each plan to choose its backend via a flag on the plan input (or
`MessageCorrelate.data`), so a single fleet can mix Claude and Mantis runs.

**Patch sites in `staffai_tools/vision_claude/`:**

1. `vision_claude_tools.py` — `VisionClaudeInputSchema` (around line 130):
   add an optional override field.

   ```python
   cua_backend: Literal["claude", "mantis-orchestrated"] | None = Field(
       default=None,
       exclude=True,
       description=(
           "Override CUA backend per plan. Falls back to settings.cua_backend "
           "when None. Not visible to the model."
       ),
   )
   ```

2. `vision_claude_tools.py` — `VisionClaudeTool.__init__` (line 282) and
   `execute()`. Replace the single cached `self._backend` with a small cache
   keyed by name:

   ```python
   self._backends: dict[str, CUABackend] = {}
   # warm-start the default
   default_name = self._settings.cua_backend
   self._backends[default_name] = create_cua_backend(self._settings, name=default_name)
   ```

   In `execute()`, resolve per call:

   ```python
   backend_name = params.cua_backend or self._settings.cua_backend
   backend = self._backends.get(backend_name)
   if backend is None:
       backend = create_cua_backend(self._settings, name=backend_name)
       self._backends[backend_name] = backend
   ```

   And replace `self._backend.run_loop(...)` (line 499) with
   `backend.run_loop(...)`. Also include `backend_name` in the Langfuse log
   line at line 333 (currently logs `self._settings.cua_backend`).

3. `backend_factory.py` — extend signature to accept an explicit name:

   ```python
   def create_cua_backend(
       settings: VisionClaudeSettings, *, name: str | None = None
   ) -> CUABackend:
       backend_name = name or settings.cua_backend
       backend_cls = _get_backend_class(backend_name)
       return backend_cls(settings=settings)
   ```

4. `vision_computer_use_handler.py` — read the override from correlate /
   payload data and pass it through. Patch the four `tool.input_schema(...)`
   build sites at lines 778, 1241, 1905, 2220.

   Around line 466 add the override variable next to the existing ones:

   ```python
   cua_backend_override: str | None = None
   ```

   Around line 520 (alongside `proxy_provider_override = data.get(...)`):

   ```python
   cua_backend_override = data.get("cua_backend") or None
   ```

   Around line 562 (payload fallback, mirroring `model_override`):

   ```python
   if cua_backend_override is None:
       cua_backend_override = payload_data.get("cua_backend")
   ```

   Pass into each `tool.input_schema(...)` call:

   ```python
   cua_backend=cua_backend_override,
   ```

   For the **resume paths** (lines 1241 user-input resume, 1905 deploy
   resume), the override must persist across pause/resume. Save it into
   `plan.agent_data["vision_claude_state"]` alongside `messages` /
   `current_iteration`, and re-read it in `_handle_user_input_resume` /
   `_handle_deploy_restart` before building the next `input_schema`.

**Library dependency:** none. This change ships independently of any
mantis-agent capability.

---

## 2. Tool-call parity layer

**Goal.** The Claude backend exposes a rich tool surface to the model via
`GenericToolAdapter`: `auth_credentials`, `user_input` (`request_user_input`),
`correspondent_message`, `text_editor`, `bash`, proxy reconfig, web search,
`facebook_messenger`, `create_initial_message`, `pop_totp_auth`. The Mantis
orchestrated backend cannot offer these today because `MicroPlanRunner` has
a fixed action vocabulary and no host-provided tool-call channel.

**Library dependency:** [mercurialsolo/mantis#71 — *MicroPlanRunner: host-provided
tool-call channel*]. Once that lands and exposes a `register_tool(name, schema,
handler)` API on `MicroPlanRunner`, the staffai-side wiring is straightforward.

**Patch sites in `staffai_tools/vision_claude/`:**

1. `mantis_backend.py` — `MantisOrchestratedBackend.run_loop`. Build the same
   `extra_anthropic_tools` list that `vision_claude_tools.execute()` already
   builds (lines 360–440), but instead of passing them as `extra_tools` to
   `sampling_loop`, register each with the runner:

   ```python
   for tool in extra_anthropic_tools:
       runner.register_tool(
           name=tool.name,
           schema=tool.to_params(),  # already JSON-schema compatible
           handler=lambda kwargs, _tool=tool: _tool(**kwargs),
       )
   ```

   `GenericToolAdapter` already produces the JSON-schema input definition the
   runner needs. The handler closure invokes the existing tool with the
   model-supplied args.

2. `vision_claude_tools.py` — refactor the tool-set construction (lines
   360–440) into a helper `_build_extra_tools(params, settings, ...)` so both
   `ClaudeCUABackend` and `MantisOrchestratedBackend` can call it without
   duplicating logic. Today the construction is inlined in `execute()`.

**No code change needed for individual tools** — they already implement the
`AbstractAgentTool` contract that `GenericToolAdapter` wraps.

---

## 3. On-demand browser launch

**Goal.** Not every plan is a browser plan. Even browser plans need an
explicit launch step instead of assuming Chromium is running. The Claude
backend handles this via `bash` (the model runs `chromium &` or clicks the
dock icon). The Mantis orchestrated backend has neither today.

**Library dependency:** [mercurialsolo/mantis#72 — *GymEnvironment:
`launch_app` action and Chromium-launch contract*]. Resolution shape: either
add `ActionType.LAUNCH_APP` with `(name, args)` params, or expose `bash` via
the tool-call layer from §2 and let plans call it.

**Patch sites in `staffai_tools/vision_claude/`:**

1. `vision_claude_gym_env.py` (new file from §F.1 of `integration-vision_claude.md`)
   — implement the `LAUNCH_APP` (or `BASH`) branch in `step()`. Reuse the
   existing Chromium wrapper script (`/usr/bin/chromium-wrapper.sh` or
   whatever it ends up being) so `--user-data-dir` and `--proxy-server`
   flags stay in one place:

   ```python
   elif action.action_type == ActionType.LAUNCH_APP:
       if params["name"] == "chromium":
           subprocess.Popen(
               ["/usr/bin/chromium"] + params.get("args", []),
               env=self._desktop.get_env(),
           )
   ```

2. `desktop.py` — **do not** auto-launch Chromium for Mantis plans, even
   though it's tempting. Symmetry with the Claude backend (which never
   auto-launches) keeps the contract simple: a plan that needs a browser
   either calls `LAUNCH_APP` or clicks the dock icon, just like a human.

---

## 4. Pause/resume parity (OTP / 2FA / confirmation flows)

**Goal.** When the model needs out-of-band info, the Claude backend raises
`UserInputRequested`, which `agent_loop.py` catches and propagates as a
`pause_request` on `LoopResult`. `vision_computer_use_handler` saves
conversation state to `plan.agent_data["vision_claude_state"]` and creates a
`MessageCorrelate` row. The Mantis backend currently has no equivalent.

**Library dependency:** [mercurialsolo/mantis#73 — *MicroPlanRunner: pause
primitive*]. Resolution shape: when the host-registered `request_user_input`
tool from §2 is invoked, the runner suspends, returns a serializable
`PauseState`, and accepts a `runner.resume(state, user_input=...)` call that
re-injects the value and continues.

**Patch sites in `staffai_tools/vision_claude/`:**

1. `mantis_backend.py` — surface the runner's pause as a `PauseRequest` on
   `CUALoopResult`. The existing `CUALoopResult.pause_request` and
   `messages` fields are already backend-agnostic enough to carry serialized
   `PauseState` instead of Anthropic message blocks; the handler uses these
   opaquely.

2. `vision_computer_use_handler.py` — when saving / restoring
   `plan.agent_data["vision_claude_state"]`, store enough data to
   reconstruct on resume. For Mantis runs the saved blob is a `PauseState`,
   not a list of Anthropic message dicts. The `_handle_user_input_resume`
   path must detect which backend produced the state (use the
   `cua_backend_override` from §1 or stamp the state blob with a
   `backend_kind` field) and route accordingly. Do **not** try to feed a
   Mantis `PauseState` into the Claude backend or vice versa.

3. `vision_claude_tools.py` — `VisionClaudeInputSchema.resume_messages` is
   currently typed as `list[dict[str, Any]]`. Either widen it to `Any` (with
   a `backend_kind` companion field) or add a parallel `resume_state: Any`
   field. The latter is cleaner.

---

## 5. Per-step observability parity

**Goal.** Today `ClaudeCUABackend.run_loop` calls
`extract_browser_contexts(result.messages, vision_model=...)` to populate
`current_url`, `page_header`, `business_object_*`, `capture_method` on
`VisionClaudeOutputSchema`. These power the per-step "where was the agent?"
panels in Shelter. Mantis returns `browser_contexts=[]` because the
screenshots live inside `ScreenStreamer` and never make it back out.

**Library dependency:** [mercurialsolo/mantis#74 — *MicroPlanRunner: expose
per-step screenshots and step-callback hook*].

**Patch sites in `staffai_tools/vision_claude/`:**

1. `mantis_backend.py` — once the runner exposes per-step screenshot bytes
   on its result, run the same `extract_browser_contexts` sidecar over them.
   It currently parses Anthropic message structure; refactor so it accepts
   `list[(turn_index, image_bytes)]` directly. The Anthropic-message-aware
   wrapper used by the Claude backend can build that list from messages; the
   Mantis backend builds it from the runner's result.

2. `mantis_backend.py` — wire `_log_callback` per step using the runner's
   step-callback hook:

   ```python
   def _on_step(idx, intent, action, ok):
       if log_callback:
           log_callback(f"[mantis] step {idx}: {action.action_type.value} {'ok' if ok else 'failed'}")
   runner.step_callback = _on_step
   ```

   Today the Mantis backend only emits a start line and an end line.

3. `execution_observer.py` — `build_execution_observation` already tolerates
   empty `browser_contexts`. No change needed once §5.1 lands.

---

## 6. Deploy SIGTERM handling

**Goal.** The Claude backend honours `shutdown_event` (a `threading.Event`)
and exits cleanly on deploy SIGTERM, so `_handle_deploy_shutdown_save` can
persist state and the plan resumes on a fresh instance after the deploy
finishes.

**Library dependency:** [mercurialsolo/mantis#76 — *MicroPlanRunner: external
cancellation token*]. Resolution shape: `runner.run(plan, cancel_event=...)`
or equivalent.

**Patch sites in `staffai_tools/vision_claude/`:**

1. `mantis_backend.py` — forward the `shutdown_event` kwarg from
   `run_loop`'s `**kwargs` into the runner's cancellation hook. On
   cancellation, return a `CUALoopResult` with `shutdown_requested=True`
   and a serialized `PauseState` in `messages` (or `pause_request`,
   depending on how §4 ends up shaped).

2. `vision_computer_use_handler.py` — `_handle_deploy_shutdown_save`
   already routes on `shutdown_requested`; no change beyond making sure the
   serialized state from §4 round-trips through it.

---

## 7. Coordinate-space alignment

**Goal.** Holo3 emits clicks in screenshot pixel coordinates (post-resize?
full-resolution? DPR-aware?). `VisionClaudeGymEnv.step()` dispatches via
`ComputerTool.click(x, y)`, which expects Xvfb display coordinates.
Mismatches here have already produced critical bugs upstream (see closed
issue #25 — 1.5× click offset).

**Library dependency:** [mercurialsolo/mantis#75 — *Coordinate-space
invariants: document and test*]. Pure documentation + test work; no API
change required.

**Patch sites in `staffai_tools/vision_claude/`:**

1. `vision_claude_gym_env.py` — implement and test the scaling that the
   library doc prescribes. Add a unit test that mocks `Desktop.viewport_size
   = (1280, 720)`, feeds the env an action with a known image-coordinate,
   and asserts `ComputerTool.click` was called with the expected
   Xvfb-coordinate.

2. `desktop.py` — surface the actual Xvfb size as
   `Desktop.viewport_size` if it isn't already.

---

## Cross-reference matrix

| Parity item | Library issue | staffai-side section | Library dep? | Library status |
|---|---|---|---|---|
| Per-plan backend flag | — | §1 | no | n/a |
| Tool-call layer | mantis#71 | §2 | yes | ✅ landed |
| On-demand browser launch | mantis#72 | §3 | yes | ✅ landed |
| Pause/resume | mantis#73 | §4 | yes | ✅ landed |
| Per-step observability | mantis#74 | §5 | yes | ✅ landed |
| SIGTERM handling | mantis#76 | §6 | yes | ✅ landed |
| Coordinate alignment | mantis#75 | §7 | yes (docs) | ✅ landed |

### Library API summary (post-#71/#72/#73/#74/#75/#76)

The runner now exposes everything the staffai-side patches above need:

```python
from mantis_agent.gym.micro_runner import (
    MicroPlanRunner, PauseRequested, PauseState, RunnerResult,
)

runner = MicroPlanRunner(
    brain=..., env=...,
    step_callback=on_step,            # #74 (idx, intent, action|None, ok)
    keep_screenshots=20,              # #74 cap retained PNG bytes
    cancel_event=shutdown_event,      # #76 threading.Event-like or callable
)

# #71 — register host tools (exact mirror of GenericToolAdapter.to_params())
runner.register_tool(
    name="auth_credentials",
    schema=tool.to_params(),
    handler=lambda kwargs, _t=tool: _t(**kwargs),
)

# #73 — request_user_input handler raises PauseRequested on first invocation,
# returns the staged input on resume.
def request_user_input(args):
    staged = runner.consume_pause_input(default=None)
    if staged is None:
        raise PauseRequested(reason="user_input", prompt=args["prompt"])
    return staged

# #74/#76 — RunnerResult is the rich return shape:
result: RunnerResult = runner.run_with_status(plan)
if result.cancelled:        # #76
    ...                     # caller persists state, returns shutdown_requested=True
if result.paused:           # #73
    state = result.pause_state.to_dict()  # JSON-safe; store on plan.agent_data

# #73 — resume after user supplies input
result2 = runner.resume(state, user_input="123456", plan=plan)

# Per-step screenshots for #74 sidecar context extraction:
for s in result.steps:
    if s.screenshot_png:   # PNG bytes
        ...                # feed extract_browser_contexts
```

`StepResult.to_dict()` deliberately excludes `screenshot_png` and
`last_action`, so the existing checkpoint JSON shape is unchanged.

### Coordinate-space invariants (#75)

See [`reference/coordinate-spaces.md`](reference/coordinate-spaces.md) for
the documented contract and the `scale_brain_to_display` helper. The
`VisionClaudeGymEnv.step()` implementation should call that helper when
`Brain.last_image_size != Desktop.viewport_size`. The library ships a unit
test (`tests/test_gym_coordinates.py`) that pins the contract — mirror it
on the staffai side with `Desktop` mocked to confirm the integration scales
correctly.

---

## Order of operations

§1 ships first and unblocks per-plan A/B in production with whichever
backend exists today. §2 unblocks §3 and §4 (both consume the tool-call
layer). §5 / §6 / §7 are independent and can ship in parallel once the
library issues land. Phase 2 of the migration (canary tenant) needs §1 + §5
at minimum so observability stays at parity; §2 + §3 + §4 are required
before any plan that touches login / OTP / correspondents can flip to
Mantis.

---

## Backwards-compatibility invariants

**Goal of this section.** Anyone implementing the changes above on the
staffai side must preserve current vision_claude behaviour exactly. No
existing plan, env-var, or stored state shape should change for callers
who don't opt in to the new backend.

The following invariants are non-negotiable. If a patch breaks any of
them, the patch is wrong — find a different shape.

### What stays exactly the same

- **`VISION_CLAUDE_CUA_BACKEND` defaults**: still `"claude"`. Nothing flips
  the default. Existing fleets see no change without an explicit env-var
  or per-plan override.
- **`ClaudeCUABackend.run_loop` behaviour**: byte-for-byte identical.
  No refactor inside it. Pass-through to `agent_loop.sampling_loop`
  remains untouched.
- **`MantisCUABackend` (streaming, today's `mantis_backend.py`)**: keep
  it. Some operators may still flip to it for read-only plans. Don't
  delete or rename the file in this PR.
- **`VisionClaudeInputSchema` existing fields**: types and defaults
  unchanged. New fields (`cua_backend`, `resume_state`) are optional with
  default `None`.
- **`vision_claude_state` blob shape for `claude` runs**: unchanged.
  Mantis runs add a `backend_kind: "mantis-orchestrated"` discriminator
  alongside the existing keys; absence of the key means Claude (current
  shape).
- **`_handle_user_input_resume` / `_handle_deploy_restart` Claude path**:
  unchanged. Mantis path branches *before* the existing logic; if
  `backend_kind` is missing or `"claude"`, fall through.
- **`extract_browser_contexts(messages, vision_model=...)`**: keep the
  existing signature. Add a new entry point (e.g.
  `extract_browser_contexts_from_images(images, ...)`) for the Mantis
  path. Both share the inner extraction prompt.
- **`Desktop.size` / Xvfb resolution defaults** (`1280×800`): unchanged.
  No new requirements on the desktop layer for the Mantis backend.
- **`ComputerTool` and `agent_loop.py`**: untouched. The Mantis backend
  uses `xdotool` directly via `VisionClaudeGymEnv` and never goes
  through `ComputerTool` — so its scaling, screenshot path, and output
  format stay isolated.
- **SQS queues, MessageCorrelate shape, Plan status transitions**:
  unchanged. The Mantis backend produces the same `CUALoopResult` shape
  as Claude; the handler routing logic doesn't see a new code path.

### Pre-flight checks before merging

For each patch:

1. Run `pytest` on the existing `claude` backend tests — they must pass
   without modification.
2. Search for all `cua_backend` literal sites with
   `rg "cua_backend\s*[=:]"` — every one must still accept `"claude"` as
   a valid value with no behaviour change.
3. Load a stored `vision_claude_state` blob from a Claude run *before*
   the patch and a Claude run *after* the patch. The dicts must be
   bit-identical (same keys, same types).
4. Confirm `mantis-agent[orchestrator]` does not pull GPU dependencies
   on import. Run `python -c "import mantis_agent.gym.micro_runner"` in
   a clean venv with only the `[orchestrator]` extra; the import must
   succeed without `torch`, `vllm`, or any CUDA libs available.

### What does change (acceptable)

These changes are additive and only affect the new code paths:

- New `"mantis-orchestrated"` literal value (extends, doesn't remove).
- New optional fields on `VisionClaudeInputSchema` (`cua_backend`,
  `resume_state`).
- New `mantis_endpoint`, `mantis_api_token`, `mantis_gateway_authorization`
  settings (default empty / None).
- New `backend_kind` key inside `vision_claude_state`. Code that reads
  the blob must `.get("backend_kind", "claude")` so old blobs still work.
- New file `vision_claude_gym_env.py` (no edits to existing files
  except `vision_claude_tools.py` for tool registration in
  `MantisOrchestrated`'s branch).
- New file `mantis_orchestrated_backend.py` alongside the existing
  `mantis_backend.py`. Old streaming path keeps working.

### Migration is an env-var flip, not a data migration

No backfill, no schema migration, no Plan status sweep. Setting
`VISION_CLAUDE_CUA_BACKEND=mantis-orchestrated` (or the per-plan field
from §1) on a fresh plan opts that plan into the new path. Plans
already mid-flight on Claude continue on Claude until they finish.
