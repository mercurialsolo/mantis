# Runtime form controller

The runner's form-filling state lives on a single :class:`FormController`
object instead of four parallel locals scattered across `GymRunner.run`.
This makes the runtime takeover after repeated clicks (the staff-crm
benchmark's most common P0 failure) a first-class capability rather than
ad-hoc substitutions.

Tracking issue: [#301](https://github.com/mercurialsolo/mantis/issues/301).

## Why a controller

Benchmark evidence (staff-crm runs 011–033):

- **Runs 011–013**: prompt / rule / runtime nudges produced **0** `type_text`
  actions; Holo3 kept clicking the same field.
- **Runs 014–020**: force-fill substitution moved the task forward, proving
  runtime control is the reliable lever.
- **Runs 029–033**: CDP `Input.insertText` became necessary for React
  controlled inputs — raw xdotool typing/paste did not reliably update
  app state.

Treating form filling as a runtime capability (rather than prompt tuning)
is the only mechanism that has shipped reliable wins on this failure class.

## Responsibilities

Per the issue's six-step spec:

1. **Detect focused/target input** — DOM when available, otherwise
   `holo3_detector.detect_focused_field` on the screenshot.
2. **Click/focus once** — substitution short-circuits a re-click on a
   field already in `used_regions`.
3. **Type via the strongest backend** — CDP `Input.insertText` first,
   paste second, raw xdotool last. Backend selection lives in
   `xdotool_env._cdp_insert_text`; the controller decides *when* to type,
   not *how*.
4. **Verify the value landed** — wired through `gym_result.info["type_verified"]`
   by the env adapter when DOM access exists.
5. **Submit with Enter** after the last credential / search field, unless
   a submit target is explicitly required.
6. **Update force-fill state** when an external director or fallback path
   moves focus — exposed as `mark_consumed_label(label)` so values are not
   typed twice.

## Object surface

```python
from mantis_agent.gym.form_controller import FormController

# Episode-level construction (the runner does this in .run() automatically):
controller = FormController.from_task(brain, task)

# Read-only views (used by the done-acceptance gate, the Claude director,
# and the /v1/cua telemetry surface):
controller.has_pending          # bool
controller.pending_count        # int
controller.pending_labels       # list[str]
controller.consumed_count       # initial_count - pending_count
controller.submitted            # bool

# Mutation hooks:
controller.mark_consumed_label("password")  # external director hook
controller.mark_used_region(x=200, y=300)   # geometric used-region marker
controller.mark_submitted()                 # latch the auto-submit flag

# Decision API (delegates to the existing GymRunner static helpers so
# back-compat tests stay green):
controller.maybe_substitute_click_with_type(action, history, brain, screenshot)
controller.maybe_substitute_repeated_click(action, history, task)
controller.should_finish_task(task)
controller.finish_task_actions(task)
```

## Runner integration

`GymRunner.run` constructs a controller per episode and exposes it as
`self.form_controller`. The legacy `force_fill_*` local variables alias
the controller's lists so the rest of `run()` reads/writes through the
same underlying state — the refactor is zero-behaviour-change.

## Ablation toggle

Per [#261](https://github.com/mercurialsolo/mantis/issues/261) discipline:

```bash
MANTIS_FORM_CONTROLLER=disabled
```

When disabled, `self.form_controller` is `None` and the runner falls back
to the legacy `holo3_detector.extract_form_values` code path. Useful for
A/B comparisons measuring whether the controller's surface itself adds
value (it shouldn't change behaviour today; future capability lifts —
mandatory CDP backend selection, DOM-aware focus, post-type retry — will
land behind this toggle so they're individually measurable).

## See also

- [Done-acceptance gate](done-gate.md) — `pending_form_values` rejection
  uses `controller.pending_labels` to detect "claimed success with
  credentials still pending".
- [Predicate grammar](predicates.md) — `field_focused[:<name>]` lets
  brains predict which field will gain focus after a click.
- [Coordinate spaces](coordinate-spaces.md) — viewport vs full-page
  contract, also referenced by `_model_coords_to_screen`.
