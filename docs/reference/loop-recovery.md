# Loop recovery policy

The `LoopDetector` already flags three loop shapes — byte-equal repeats,
coordinate-drift, frozen-state. Until now the runner responded with a
contextual nudge in the next inference's prompt ("you've clicked here
three times, try typing instead"). That works some of the time, but
benchmark reports show many runs where the brain reads the nudge and
emits the same click class again.

This policy returns a **forced action-class transition** — the runner
substitutes a different action class for the brain's stuck output.
Acts on the same loop signal that #293's perceptual-diff verifier
detects and converts it into a concrete recovery.

Tracking issue: [#302](https://github.com/mercurialsolo/issues/302).

## Initial rules

The policy is intentionally narrow — the soft-loop signal is noisy and
forcing the wrong action class is worse than another nudge cycle.

| Pattern | Forced action | Reason code |
|---|---|---|
| `CLICK` loop on focused input + pending plan value | `TYPE(text=value)` | `type_pending_value` |
| `CLICK` loop on focused input, no pending value | `KEY_PRESS("Tab")` | `tab_to_next_field` |
| `CLICK` loop with no focused field + frozen frame + submit-shaped reasoning | `KEY_PRESS("Return")` | `press_return_for_submit` |

The first rule overlaps with `FormController`'s existing label-match
substitution but covers the case where the focused field's label
doesn't match any extracted plan value (e.g., a generic placeholder
the controller can't pin down to a specific entry).

## When the policy fires

All three conditions must hold:

1. **No earlier substitution applied** — `force-fill`, `force-submit`,
   `claude-director`, and `top-click-guard` get first refusal in
   sequence; the recovery policy is the last gate before dispatch.
2. **`LoopDetector.is_any_loop(soft_loop_window)` returns True** — any
   of the three loop shapes (byte-equal / drift / state).
3. **A rule matches the current `(action, focused_input, …)` state.**

When all three hold, the policy substitutes the action and the runner
records the reason code on the trajectory step.

## Trajectory shape

```python
step.loop_recovery_reason  # "" or one of REASON_CODES
step.action                # the substituted action (TAB / TYPE / RETURN)
```

`RunResult.loop_recoveries_by_reason` aggregates per-reason counts:

```json
{
  "loop_recoveries_by_reason": {
    "tab_to_next_field": 2,
    "press_return_for_submit": 1
  }
}
```

Surfaced on `/v1/cua` so every run doubles as an ablation data point.

## Ablation toggle

Per [#261](https://github.com/mercurialsolo/mantis/issues/261) discipline:

```bash
MANTIS_LOOP_RECOVERY=disabled
```

When disabled the policy short-circuits to "no forced action" and the
runner falls through to its legacy nudge path (the soft-loop signal
still triggers a feedback line; just no action substitution).

## Quality guard

The narrow rule set + sequential gating (existing substitutions first)
keeps false-positive substitutions bounded:

- `type_pending_value` only fires when a plan value exists AND the
  focused field is non-empty AND no earlier substitution matched.
- `tab_to_next_field` is benign — Tab moves to the next form field;
  worst case the brain re-clicks the original field next turn (no
  destructive state change).
- `press_return_for_submit` is conditioned on **frozen frame** (last
  `soft_loop_window` hashes identical), the strictest condition. A
  click that's actually moving the page forward never trips it.

## See also

- [Perceptual diff verifier](perceptual-diff.md) — emits the
  `WARNING: no observed effect` signal that the runner records on the
  same trajectory step the loop-recovery policy may have already
  substituted.
- [Form controller](form-controller.md) — `type_pending_value` consults
  the controller's pending list; the rule is a fallback when the
  controller's own label-match couldn't pick the value.
- [Done-acceptance gate](done-gate.md) — `pending_form_values` rejection
  pairs naturally with this policy: the gate refuses to terminate
  while values remain pending, and the policy can keep typing them.
