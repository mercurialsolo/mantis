# Adaptive settle

The runner and `PlanExecutor` previously slept a fixed `settle_time`
(default 1.5-4 s) after every action. That paid the worst-case latency
on every step even when the DOM settled in 50 ms, and silently capped on
network-heavy submits that needed more than the budget.

Adaptive settle replaces the fixed sleep with a stability gate that
returns as soon as the page has stopped repainting, capped at the legacy
budget.

Tracking issue: [#294](https://github.com/mercurialsolo/mantis/issues/294).

## Gates

### `wait_until_stable` (xdotool / screenshot-only path)

Polls a screenshot supplier every 100 ms, returns when two consecutive
`phash_64` reads agree (the page has stopped repainting). Cap at
`max_seconds = settle_time`.

```python
from mantis_agent.gym.adaptive_settle import wait_until_stable

elapsed = wait_until_stable(
    capture=env._screenshot,
    max_seconds=settle_time,
    poll_interval=0.1,
)
```

Behaviour:

- A capture returning `None` (or raising) is treated as "no signal yet" —
  the poll continues; stability is never declared on a missing frame.
- The 100 ms poll interval balances precision against `mss` screenshot
  cost (~5-15 ms per capture on Xvfb).
- The hash is the same `phash_64` the loop detector already computes, so
  this adds no extra hashing on the hot path beyond the per-poll capture.

### `wait_for_networkidle` (CDP / Playwright path)

Wraps `page.wait_for_load_state("networkidle", timeout=cap*1000)`. Returns
seconds waited (capped at `max_seconds`). On any error — page closed,
navigating, playwright unavailable — falls back to a plain sleep of the
remaining cap rather than skipping the settle entirely.

### `settle_after_action` (step-handler shorthand)

Step handlers (`filter.py`, `form.py`, `paginate.py`, `click.py`,
`navigate.py`) drive the browser through `XdotoolGymEnv` and have no
Playwright page handle. They use this thin wrapper:

```python
adaptive_settle.settle_after_action(env, max_seconds=N)
```

which calls `wait_until_stable(env._screenshot, max_seconds=N)`. If the
env doesn't expose a screenshot attribute (defensive against alternate
adapters), it falls back to `time.sleep(N)` so we never silently skip a
settle.

## Where it fires

| Site | Gate used |
|---|---|
| `XdotoolGymEnv.step` post-action | `wait_until_stable` (no DOM available) |
| `PlanExecutor._settle` (navigate/type/click/key) | `wait_for_networkidle` when a Playwright page exists, else fixed sleep |
| `MicroPlanRunner` step handlers (`filter`, `form`, `paginate`, `click`, `navigate`) | `settle_after_action` — `wait_until_stable` on `env._screenshot`, with a fixed-sleep fallback when no capture is exposed |

The 10+ scattered `time.sleep(self._settle_time)` sites in `PlanExecutor`
now route through a single `_settle()` method, and the ~20 scattered
"settle after browser action" sites in the step handlers route through
`settle_after_action` — both consolidations mean future gate
improvements land in one place.

## Ablation toggle

Per [#261](https://github.com/mercurialsolo/mantis/issues/261) discipline:

```bash
MANTIS_ADAPTIVE_SETTLE=disabled
```

When disabled both gates short-circuit back to the original fixed
`time.sleep(settle_time)`. Flip it on a deployed instance to compare wall
time on the same workload without redeploying the binary.

## Expected wall-time impact

From the lu.ma click flow profiled in [#310](https://github.com/mercurialsolo/mantis/issues/310):

| Per-step phase | Pre-#294 | Post-#294 (static page) |
|---|---|---|
| Brain inference | 1.5-2.0 s | unchanged |
| Settle | **fixed 3.0 s** | **~0.2-0.5 s** |
| Screenshot + dispatch | <0.5 s | unchanged |
| **Total per step** | ~5.0 s | **~2.5 s** |

On dynamic pages with ongoing repaint (animations, lazy-load), the gate
hits the cap and behaves like the legacy fixed sleep. Net is "as fast or
faster than before, never slower".

## See also

- [Predicate grammar](predicates.md) — `frame_stable` / `frame_changed`
  predicates expose the same hash signal to brains.
- [Done-acceptance gate](done-gate.md) — `no_observed_delta_after_waits`
  predicate uses the same frame-hash equality test.
- [#118](https://github.com/mercurialsolo/mantis/issues/118) speculative
  inference — complementary; overlaps `brain.think()` with the settle.
