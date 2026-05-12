# Perceptual diff verifier

The runner used to declare an action successful purely on syscall
success — `executor.execute()` returned without raising → step counted.
That misses an entire class of silent failures:

- Cookie / consent overlays absorb the click.
- Form-validation messages flash and disappear before the next screenshot.
- Modals are mounted into hidden DOM positions.
- Page repaints onto an identical viewport (drift loop).

The perceptual diff verifier compares the pre-action frame to the
post-settle frame on **high-risk actions** and emits
`action_effect_observed: bool` per step. The runner injects a
`WARNING: high-risk action had no observed effect` line into the next
inference's feedback when the predicate fires, so the brain doesn't
loop on the same useless click.

Tracking issue: [#293](https://github.com/mercurialsolo/mantis/issues/293).

## High-risk classifier

The verifier only fires on actions where a silent failure would
plausibly matter — narrow on purpose so the verifier doesn't run on
every click.

| Action shape | High-risk? |
|---|---|
| `KEY_PRESS` with `keys ∈ {Return, Enter, *+Return, *+Enter}` | yes (form submit) |
| `CLICK` whose `reasoning` contains submit / confirm / buy / purchase / send / delete / save / sign in / log in / login / register / checkout / place order | yes |
| `CLICK` without a high-risk keyword | no |
| `WAIT`, `DONE`, `SCROLL`, `TYPE`, `DOUBLE_CLICK` | no |

The classifier is heuristic — brains often emit these keywords verbatim
in their chain-of-thought. False negatives (a high-risk click missed)
just mean the verifier skips that step. False positives just add a hash
comparison; no behaviour change.

## Diff signals

Two pHash comparisons, both using the existing `phash_64` from
`loop_detector.py`:

1. **Global frame hash** — full screenshot pre vs post.
2. **Region hash** — 200×200 crop centred on the action's
   `(x, y)`. Catches the case where the global hash changed (a banner
   ticked over, a clock updated) but the action region itself didn't —
   the click landed on nothing.

`effect_observed = global_changed OR region_changed`. Only declared
`False` when **both** stayed pixel-equivalent.

> **CLIP cosine is not shipped yet.** The original issue mentions it as
> a future addition; the pHash path is free (already computed for loop
> detection) and avoids a model dependency. A future PR can add CLIP
> behind a separate toggle once the pHash baseline is measured.

## Surfaces

### Per step

`TrajectoryStep.action_effect_observed`: `True` / `False` / `None`.
`None` means the check was skipped (action wasn't high-risk, frames
missing, toggle off).

### Per run

`RunResult.perceptual_summary` and `/v1/cua` response field:

```json
{
  "perceptual_summary": {
    "checked": 5,
    "no_effect": 1
  }
}
```

Empty `{}` when the verifier never fired in the run (no high-risk
actions or toggle off).

### Feedback injection

When the verifier sees a `False`, the next step's feedback string carries
a warning so the brain gets a real signal:

```
clicked (no visible change); WARNING: high-risk action had no observed effect (global_and_region_stable)
```

The brain reads this on the next inference and can pivot to a different
action class (loop-recovery work in #302 will formalise this).

## Ablation toggle

Per [#261](https://github.com/mercurialsolo/mantis/issues/261) discipline:

```bash
MANTIS_PERCEPTUAL_VERIFY=disabled
```

When disabled the verifier short-circuits to `effect_observed=None`;
no WARNING, no aggregate, no trajectory annotation. Flip on a deployed
instance to A/B without redeploy.

## Quality guarantee

The verifier is **observational** — it never blocks an action, never
substitutes the action, never overrides the runner's termination
decisions. The only behavioural effect is the WARNING line in feedback.
Worst-case false-positive: an over-aggressive "no observed effect"
warning that the brain can correctly ignore.

The narrow high-risk classifier ensures the verifier doesn't fire on
benign clicks where a no-op result is expected (scrolling through
listings, opening a card, etc.).

## See also

- [Done-acceptance gate](done-gate.md) — `no_observed_delta_after_waits`
  uses the same frame-hash comparison for the done() path.
- [Predicate grammar](predicates.md) — `frame_changed` / `frame_stable`
  expose the same signal to the brain at prediction time.
- [#302](https://github.com/mercurialsolo/mantis/issues/302) loop-recovery
  policy — pairs with this verifier; the WARNING the runner injects is
  exactly the signal the recovery policy will branch on.
