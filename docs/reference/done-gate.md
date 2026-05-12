# Done-acceptance gate

Brains — Holo3 in particular — sometimes emit `done(success=True)` before
the workflow is actually complete. The `GymRunner` runs a deterministic
gate **before** the optional model-based `verify_done` call so cheap,
unambiguous failures are caught without spending another vision call.

Tracking issue: [#303](https://github.com/mercurialsolo/mantis/issues/303).

## Failure modes the gate catches

From the staff-crm benchmark reports:

- **Run 009 / 010** — `done(success=True, summary='')` after a string of
  waits, without ever engaging the workflow.
- **Run 023** — fabricated summary claiming a downstream outcome
  ("Updated lead industry to Space Exploration") after only the click loop
  in front of the login form.
- **Per-step "Done when" confusion** — model treats a step-local clause
  as whole-task completion.

## Predicates

First-rejection-wins. Each predicate uses signals the runner already has —
no vision call, no API spend, no token cost.

| Reason code | When it fires |
|---|---|
| `empty_summary` | Summary is `""` or whitespace |
| `plan_steps_incomplete` | A structured `Plan` is present and `plan_step_idx < len(plan.steps) - 1` |
| `pending_form_values` | `force_fill_values` still has unconsumed credentials extracted from the plan |
| `summary_missing_required_fields` | Plan declared output-schema fields that the summary doesn't mention (case-insensitive) |
| `no_observed_delta_after_waits` | Last 3 actions are `WAIT` and the frame hash hasn't changed |
| `no_progress_in_window` | Last 5 steps show no URL change and no frame change |

A rejection is **not** the end of the run — the runner substitutes the
`done` with a no-op `wait` and increments a per-reason counter. The brain
gets another shot on the next inference. After `max_done_rejections`
(currently 2) the gate stops firing and the next `done(success=True)` is
accepted (avoid infinite loops when the gate is wrong).

## Trajectory shape

A substituted `WAIT` carries the rejection reason on the trajectory step:

```python
step.done_rejected_reason  # one of REJECT_CODES; "" otherwise
```

Aggregate counts surface on the `RunResult` and the `/v1/cua` API response:

```json
{
  "done_rejections_by_reason": {
    "empty_summary": 1,
    "no_observed_delta_after_waits": 2
  }
}
```

This makes any production run double as an ablation data point — you can
see how often the gate is firing and on which reason codes.

## Ordering vs. the model-based verifier

The gate runs **first**, before the existing `holo3_detector.verify_done`
model call. If the gate rejects, the verifier never runs (saves the API
spend). If the gate passes, the verifier runs as a second opinion.

```
done(success=true) → gate (free, deterministic)
                       ├─ reject → substitute WAIT, record reason, continue
                       └─ accept → verify_done (model call)
                                       ├─ reject → substitute WAIT, continue
                                       └─ accept → terminate run as success
```

Both rejection paths share the same `max_done_rejections` budget, so the
total number of rejection cycles per run is bounded.

## Ablation toggle

To disable the gate entirely — useful for measuring whether it's pulling
its weight (per [#261](https://github.com/mercurialsolo/mantis/issues/261)):

```bash
MANTIS_DONE_GATE=disabled
```

When disabled the runner falls through to the existing model-based
verifier path; `done_rejections_by_reason` stays empty.

## See also

- [Predicate grammar](predicates.md) — sibling structured signal that
  scores the brain's per-step world model.
- [Environment variables](env-vars.md#runner--verification) — the full
  list of runner-side toggles.
