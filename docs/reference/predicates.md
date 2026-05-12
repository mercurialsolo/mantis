# Predicate grammar (world-model verification)

Brains may emit a structured prediction of what they expect to observe **after**
their action lands. The runner parses the prediction, evaluates each predicate
against the post-action observation, writes per-predicate booleans into the
trajectory, and derives a `world_model_error` reward contribution.

This closes the world-model loop: the schema (`TrajectoryStep.predicted_outcome`,
`observed_outcome`, `reward_components`), exporter, labeller, and reward weights
were already in place; predicates make the signal structured rather than
free-form.

Tracking issue: [#291](https://github.com/mercurialsolo/mantis/issues/291).

## Surface forms

A brain emits one of two forms in its response. The runner accepts both.

**Structured JSON (preferred):**

```text
{"expected": ["url_contains:/checkout", "title_changed", "field_focused:email"]}
```

**Back-compat free-form line (the original Holo3 surface, #120):**

```text
Predicted: url_contains:/checkout, title_changed, field_focused:email
```

Free-form prose tokens that don't match a known predicate kind are silently
dropped, so the brain can mix natural-language reasoning with predicate tokens
without breaking the parser.

If the brain genuinely doesn't know what will happen, it should **omit the line
entirely**. Guessing is worse than silence — it inflates measured error and
poisons the reward signal.

## Grammar

Each predicate is a `kind[:arg]` token. Recognised kinds:

| Kind | Arg | Meaning |
|---|---|---|
| `url_contains` | `<substr>` | observed URL contains substr |
| `url_equals` | `<full url>` | observed URL equals the URL exactly |
| `url_changed` | — | URL differs from previous step |
| `url_unchanged` | — | URL identical to previous step |
| `title_contains` | `<substr>` | page title contains substr |
| `title_changed` | — | title differs from previous step |
| `field_focused` | optional `<name>` | a field is focused (and contains the name in its id/name/label/selector/placeholder when given) |
| `field_unfocused` | — | no field is focused |
| `frame_changed` | — | post-action frame hash differs from previous |
| `frame_stable` | — | post-action frame hash identical to previous |

Best-effort kinds — recognised by the grammar so trajectories round-trip them,
but evaluators currently return `None` ("not measured") on every adapter:

| Kind | Arg | Status |
|---|---|---|
| `element_appears` | `<text or selector>` | Returns `None` until a DOM/OCR bridge lands |
| `element_disappears` | `<text or selector>` | Same |
| `modal_opens` | — | Same |
| `modal_closes` | — | Same |

A predicate evaluating to `None` is **excluded from the accuracy denominator** —
the world-model metric stays meaningful across adapter capability levels.

## Evaluation signals

Predicates are evaluated against the post-action observation:

- `url`, `title` — from `gym_result.info` (Playwright/CDP-backed envs)
- `focused_input` — from `gym_result.info` (DOM-introspecting envs)
- `frame_hash` — perceptual hash of the post-action screenshot, compared
  against the previous step's hash for `frame_changed` / `frame_stable`

The previous step's URL/title/frame_hash come from `last_url`, `last_title`,
and the last entry in `trajectory`.

## Reward contribution

When at least one predicate is evaluable, the runner adds a per-step
`world_model_error` component to `TrajectoryStep.reward_components`:

```text
world_model_error = -0.05 * (wrong_predicates / evaluated_predicates)
```

The 0.05 weight matches the existing `PlanAdherenceReward.world_model_weight`
(see `rewards/plan_adherence.py`). All-correct predictions contribute `0.0`;
all-wrong contribute `-0.05`. Steps with no parseable or no evaluable
predicates contribute nothing.

The structured per-step component coexists with the existing episode-level
Jaccard `world_model_accuracy` from `rewards/components.world_model_accuracy_reward` —
they measure different things (per-predicate exact match vs. token overlap
of free-form prose) and both stay in `reward_components` for downstream
consumers.

## Trajectory shape

Each `TrajectoryStep` carries the parsed-and-evaluated results:

```python
step.predicted_outcome       # raw string the brain emitted
step.predicate_results       # list[{"predicate": str, "result": bool|None, "reason": str}]
step.reward_components       # may include "world_model_error"
```

The full list round-trips through `_trajectory_step_to_dict` (pause snapshots,
trace exports, training data).

## Ablation toggle

To disable predicate evaluation entirely — useful when measuring whether the
verification step itself is pulling its weight (per [#261](https://github.com/mercurialsolo/mantis/issues/261)):

```bash
MANTIS_PREDICATE_VERIFY=disabled
```

When disabled the runner still records `predicted_outcome` (so the raw model
output is preserved for distillation and offline analysis), but skips parsing
and evaluation, and emits no `world_model_error` component.

## Brain support

| Brain | Emits structured `expected` | Emits `Predicted:` |
|---|---|---|
| Claude (`brain_claude.py`) | Yes (system prompt updated) | — |
| OpenCUA (`brain_opencua.py`) | Yes (system prompt updated) | — |
| Holo3 (`brain_holo3.py`) | — | Yes (since #120) |

All three flow through `extract_predicted_outcome` in `gym/predicates.py` and
the same runner evaluation path.
