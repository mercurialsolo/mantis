# Adaptive loop-detector windows

`LoopDetector` flags three loop shapes — byte-equal repeats,
coordinate-drift, and frozen-state. The runner responds with a
soft-nudge at three repeats and a hard-terminate at eight. Those
fixed thresholds work for the median case but hurt two patterns:

- **Pagination / drilldown.** The brain emits five identical
  `click("Next")` actions; the page advances each time. The legacy
  `is_repeat_loop` fires (identical params) even though the run is
  making progress.
- **Stuck loops with micro-changing pixels.** A captcha or animated
  spinner re-renders enough to defeat `is_state_loop`'s frame-hash
  check, and the brain takes seven dead steps before the hard window
  trips.

This module adapts the comparison window per call by recent **action
diversity** + observed **state progress**.

Tracking issue: [#298](https://github.com/mercurialsolo/mantis/issues/298).

## Surfaces

| Method | Returns |
|---|---|
| `LoopDetector.pattern_diversity(window)` | `float` in `[0, 1]` — unique action signatures / window. Click-like actions bucket coords by ~`click_tol_px` so micro-drift collapses to one signature. |
| `LoopDetector.state_progressed(window)` | `bool` — public mirror of the URL/frame-hash check already used by `is_repeat_loop`. |
| `LoopDetector.adaptive_window(base, max_extension=2, floor=2)` | `int` — `base` widened on diverse / progressing windows, tightened on low-diversity / frozen-state windows. Floor and max-extension clamps applied. |
| `LoopDetector.is_any_loop_adaptive(base, max_extension=2)` | `bool` — `is_any_loop` over the adaptive window with a pagination guard (state moved AND not a state-loop → don't fire). |

`GymRunner._is_loop` is the single internal hook the runner consults
for every soft-nudge / hard-terminate / loop-recovery / claude-director
gate. It dispatches to `is_any_loop_adaptive` by default and falls
back to the legacy `is_any_loop` when `MANTIS_LOOP_ADAPTIVE=disabled`.

## Threshold logic

For the default `soft=3` / `hard=8` runner windows:

| Pattern in last `base` samples | Effective window | Rationale |
|---|---|---|
| Diversity ≥ 0.6 (varied action types/coords) | `base + 2` | Productive exploration — defer the nudge. |
| State progressed (URL or frame moved) | `base + 2` | Pagination / drilldown — defer. |
| Diversity ≤ 0.25 AND state frozen | `max(floor, base − 1)` | Clear stuck signature — terminate sooner. |
| Buffer < `base` samples | `base` (unchanged) | Not enough history to judge. |
| Otherwise | `base` (unchanged) | No clear signal. |

The pagination guard inside `is_any_loop_adaptive` provides the second
half of the fix: even after extension, identical-click pagination
would still trip `is_repeat_loop`. The guard short-circuits to `False`
when state moved across the (effective) window AND `is_state_loop`
would not fire.

## Ablation toggle

| Var | Default | Effect when set |
|---|---|---|
| `MANTIS_LOOP_ADAPTIVE` | `enabled` | `disabled` falls back to `is_any_loop(base)` everywhere — legacy fixed windows. |

`/v1/cua` accepts `loop_adaptive: true|false` in the request payload
to flip the toggle for that single request (the runtime patches
`os.environ` under a `try/finally` since `/v1/cua` is serialized per
container). Pairs with the existing per-request override surface
established for `perceptual_verify`, `loop_recovery`, `done_gate`.

Use `scripts/ablate_v1_cua.py --toggle loop_adaptive` to run a paired
A/B against a warm container.
