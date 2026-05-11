"""Tests for issue #254 — per-context sub-goal budget skip envelope.

Fourth tactical sibling to #246 (recipe-rejection skip) and #250
(navigation-primitive halt skip). Same mechanism, third trigger
source: the runner tracks how many sub-goals (= ``run()`` calls)
have executed against a given URL anchor and stamps
``StepResult.skip=True / skip_reason='listing_budget_exceeded'``
when the count would exceed the recipe-author's bound.

Three layers being pinned here:

1. ``ContextBudget`` dataclass — opt-in shape carried on the
   runner. Default ``None`` preserves today's behavior.
2. ``MicroPlanRunner.__init__`` accepts and stores it; tracks
   per-URL counter across ``run()`` calls.
3. ``run()`` short-circuits with a synthetic skip-envelope
   ``StepResult`` when the counter would exceed the bound. The
   host's tool surface (per the #246/#250 mechanism) promotes
   ``skip=True`` to a successful tool result and the orchestrator
   advances past the over-budgeted context.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.gym.context_budget import ContextBudget
from mantis_agent.plan_decomposer import MicroIntent, MicroPlan


# ── ContextBudget dataclass ─────────────────────────────────────────


def test_context_budget_defaults() -> None:
    """A bare ``ContextBudget()`` is harmless — every limit unset and
    the on_exceeded mode defaults to ``emit_skip`` (the production
    behavior aligning with #246 / #250). Callers turn on the bounds
    they care about explicitly."""
    b = ContextBudget()
    assert b.max_sub_goals_per_url is None
    assert b.max_sub_goals_per_iteration is None
    assert b.on_exceeded == "emit_skip"


def test_context_budget_accepts_overrides() -> None:
    b = ContextBudget(
        max_sub_goals_per_url=3,
        max_sub_goals_per_iteration=10,
        on_exceeded="halt",
    )
    assert b.max_sub_goals_per_url == 3
    assert b.max_sub_goals_per_iteration == 10
    assert b.on_exceeded == "halt"


def test_context_budget_rejects_invalid_on_exceeded() -> None:
    """Typo guard — the allowed values are ``emit_skip`` / ``halt`` /
    ``log_only``. Anything else is a constructor-time error so a
    typoed config doesn't silently fall through to ``emit_skip``."""
    with pytest.raises(ValueError, match="on_exceeded"):
        ContextBudget(on_exceeded="warn")  # type: ignore[arg-type]


# ── MicroPlanRunner constructor opt-in ──────────────────────────────


def test_runner_default_context_budget_is_none() -> None:
    """Default preserves today's behavior — no opt-in means no
    counter, no envelope."""
    from mantis_agent.gym.micro_runner import MicroPlanRunner
    runner = MicroPlanRunner(brain=MagicMock(), env=MagicMock())
    assert runner.context_budget is None


def test_runner_accepts_context_budget_kwarg() -> None:
    from mantis_agent.gym.micro_runner import MicroPlanRunner
    cb = ContextBudget(max_sub_goals_per_url=3)
    runner = MicroPlanRunner(
        brain=MagicMock(), env=MagicMock(), context_budget=cb,
    )
    assert runner.context_budget is cb
    # Per-URL counter starts empty.
    assert runner._sub_goal_count_by_url == {}
    assert runner._sub_goal_count_total == 0


# ── Skip-envelope emission ──────────────────────────────────────────


def _navigate_plan(url: str, intent: str = "Open detail page") -> MicroPlan:
    return MicroPlan(steps=[
        MicroIntent(
            intent=intent, type="navigate",
            params={"url": url},
        ),
    ])


def _runner_with_budget(budget: ContextBudget) -> Any:
    """Build a real MicroPlanRunner and stub its underlying
    ``_executor.execute`` so we exercise the budget gate without
    needing a real env / brain. Returns a tuple of
    ``(runner, executor_call_count_ref)`` so tests can assert on how
    many times the inner execute was invoked."""
    from mantis_agent.gym.micro_runner import MicroPlanRunner
    runner = MicroPlanRunner(
        brain=MagicMock(), env=MagicMock(), context_budget=budget,
    )
    calls = {"n": 0}

    def fake_execute(plan, state, resume=False):
        calls["n"] += 1
        state.results.append(StepResult(
            step_index=0, intent=plan.steps[0].intent if plan.steps else "",
            success=True,
        ))

    runner._executor.execute = fake_execute
    return runner, calls


def test_first_n_runs_pass_through_normally() -> None:
    """A budget of 2 should let the first two ``run()`` calls
    against the same URL execute normally (counter increments but
    doesn't yet exceed)."""
    cb = ContextBudget(max_sub_goals_per_url=2)
    runner, calls = _runner_with_budget(cb)

    url = "https://www.example.com/boat/abc"
    r1 = runner.run(_navigate_plan(url))
    r2 = runner.run(_navigate_plan(url))

    assert calls["n"] == 2
    assert r1[-1].success is True
    assert r1[-1].skip is False
    assert r2[-1].success is True
    assert r2[-1].skip is False
    assert runner._sub_goal_count_by_url[url] == 2


def test_run_past_budget_returns_skip_envelope() -> None:
    """The (N+1)th ``run()`` against the same URL must short-circuit
    without invoking the executor and return a single synthetic
    StepResult carrying the skip envelope."""
    cb = ContextBudget(max_sub_goals_per_url=2)
    runner, calls = _runner_with_budget(cb)

    url = "https://www.example.com/boat/abc"
    runner.run(_navigate_plan(url))
    runner.run(_navigate_plan(url))
    r3 = runner.run(_navigate_plan(url))

    # Executor was called twice; the third call short-circuited.
    assert calls["n"] == 2
    assert len(r3) == 1
    assert r3[0].success is False
    assert r3[0].skip is True
    assert r3[0].skip_reason == "listing_budget_exceeded"
    # The envelope's data carries the URL + count so a host can
    # surface a useful message instead of just "skipped".
    assert "listing_budget_exceeded" in r3[0].data
    assert url in r3[0].data


def test_different_urls_get_independent_counters() -> None:
    """A budget of 2 must not leak across URLs — each anchor gets
    its own counter."""
    cb = ContextBudget(max_sub_goals_per_url=2)
    runner, calls = _runner_with_budget(cb)

    url_a = "https://www.example.com/boat/a"
    url_b = "https://www.example.com/boat/b"
    runner.run(_navigate_plan(url_a))
    runner.run(_navigate_plan(url_a))
    runner.run(_navigate_plan(url_b))
    runner.run(_navigate_plan(url_b))

    # All four executor calls landed — neither anchor hit the bound.
    assert calls["n"] == 4
    assert runner._sub_goal_count_by_url[url_a] == 2
    assert runner._sub_goal_count_by_url[url_b] == 2

    # Now exceed each one independently.
    r_a3 = runner.run(_navigate_plan(url_a))
    assert r_a3[0].skip is True
    # URL B is still under budget.
    r_b3 = runner.run(_navigate_plan(url_b))
    assert r_b3[0].skip is True


def test_max_sub_goals_per_iteration_bounds_across_urls() -> None:
    """The total-across-all-URLs cap applies on top of the per-URL
    cap. With ``max_sub_goals_per_iteration=3``, the 4th run skips
    even if each URL is individually under its per-URL cap."""
    cb = ContextBudget(
        max_sub_goals_per_url=100,        # effectively unset
        max_sub_goals_per_iteration=3,
    )
    runner, calls = _runner_with_budget(cb)

    runner.run(_navigate_plan("https://www.example.com/boat/a"))
    runner.run(_navigate_plan("https://www.example.com/boat/b"))
    runner.run(_navigate_plan("https://www.example.com/boat/c"))
    r4 = runner.run(_navigate_plan("https://www.example.com/boat/d"))

    assert calls["n"] == 3
    assert r4[0].skip is True
    assert r4[0].skip_reason == "listing_budget_exceeded"


def test_on_exceeded_halt_returns_halted_status() -> None:
    """``on_exceeded='halt'`` returns an empty results list and
    marks the runner's ``_final_status`` halted. No skip envelope
    is emitted — the host's halt path applies as usual."""
    cb = ContextBudget(max_sub_goals_per_url=1, on_exceeded="halt")
    runner, calls = _runner_with_budget(cb)

    url = "https://www.example.com/boat/abc"
    runner.run(_navigate_plan(url))
    r2 = runner.run(_navigate_plan(url))

    assert calls["n"] == 1
    assert r2 == []
    assert runner._final_status == "halted"


def test_on_exceeded_log_only_lets_run_continue() -> None:
    """``on_exceeded='log_only'`` runs every sub-goal normally but
    logs a warning. Useful for shadow-mode evaluation: see what
    bounds we'd hit without actually changing production behavior."""
    cb = ContextBudget(max_sub_goals_per_url=1, on_exceeded="log_only")
    runner, calls = _runner_with_budget(cb)

    url = "https://www.example.com/boat/abc"
    runner.run(_navigate_plan(url))
    r2 = runner.run(_navigate_plan(url))

    # Both executed.
    assert calls["n"] == 2
    assert r2[-1].skip is False


def test_anchor_resolution_prefers_navigate_step_url() -> None:
    """The anchor URL is read from the first navigate step's
    ``params.url``. That's the canonical source — the URL the
    sub-goal is *targeted at*, not whatever happened to be in the
    address bar before the plan started."""
    cb = ContextBudget(max_sub_goals_per_url=1)
    runner, calls = _runner_with_budget(cb)
    runner._last_known_url = "https://stale.example.com/old"

    target = "https://www.example.com/boat/xyz"
    runner.run(_navigate_plan(target))
    r2 = runner.run(_navigate_plan(target))

    # The counter tracked under the navigate URL, not the stale one.
    # Counter increments on every attempt — the second run trips the
    # budget but its anchor still got counted (==2 after two attempts).
    assert runner._sub_goal_count_by_url[target] == 2
    assert "stale.example.com" not in runner._sub_goal_count_by_url
    assert r2[0].skip is True


def test_anchor_resolution_falls_back_to_last_known_url() -> None:
    """When a plan has no navigate step (e.g. a sub-plan that
    operates on the current page only), the anchor is the runner's
    ``_last_known_url`` at run-time. Lets the host increment the
    counter for a detail page that was opened by a prior sub-goal."""
    cb = ContextBudget(max_sub_goals_per_url=1)
    runner, calls = _runner_with_budget(cb)
    runner._last_known_url = "https://www.example.com/boat/xyz"

    no_nav = MicroPlan(steps=[
        MicroIntent(intent="Click Show Phone", type="submit",
                    params={"label": "Show Phone"}),
    ])
    runner.run(no_nav)
    r2 = runner.run(no_nav)

    # Counter incremented on each attempt (==2 after both runs);
    # second hit the budget.
    assert runner._sub_goal_count_by_url["https://www.example.com/boat/xyz"] == 2
    assert r2[0].skip is True


def test_no_anchor_no_counter() -> None:
    """A plan with no navigate step AND no ``_last_known_url`` (cold
    runner) gets bucketed under a stable sentinel — the counter
    still works, all such plans share one bucket. Better than
    crashing or silently disabling the bound."""
    cb = ContextBudget(max_sub_goals_per_url=1)
    runner, calls = _runner_with_budget(cb)
    runner._last_known_url = ""

    no_nav = MicroPlan(steps=[
        MicroIntent(intent="Click Show Phone", type="submit"),
    ])
    runner.run(no_nav)
    r2 = runner.run(no_nav)

    # Both plans bucketed under the sentinel.
    assert r2[0].skip is True


def test_default_runner_no_budget_no_envelope() -> None:
    """A runner constructed without ``context_budget`` runs forever
    against the same URL. Preserves today's behavior."""
    from mantis_agent.gym.micro_runner import MicroPlanRunner

    runner = MicroPlanRunner(brain=MagicMock(), env=MagicMock())
    calls = {"n": 0}

    def fake_execute(plan, state, resume=False):
        calls["n"] += 1
        state.results.append(StepResult(
            step_index=0, intent="x", success=True,
        ))

    runner._executor.execute = fake_execute

    url = "https://www.example.com/boat/abc"
    for _ in range(50):
        runner.run(_navigate_plan(url))

    assert calls["n"] == 50  # all ran; no budget gate


def test_skip_envelope_carries_url_anchor_in_data() -> None:
    """The data field on the skip envelope must surface the URL
    and the count so a refinement agent / dashboard can attribute
    skipped runs to the right context anchor."""
    cb = ContextBudget(max_sub_goals_per_url=1)
    runner, calls = _runner_with_budget(cb)

    url = "https://www.example.com/boat/abc"
    runner.run(_navigate_plan(url))
    r2 = runner.run(_navigate_plan(url))

    assert "url=" in r2[0].data
    assert "count=" in r2[0].data
    assert url in r2[0].data
    # The synthetic step_result is at step_index=0 (the run never
    # entered the plan's actual steps).
    assert r2[0].step_index == 0


def test_run_with_status_propagates_skip_envelope() -> None:
    """``run_with_status`` wraps ``run`` — the skip envelope must
    survive the wrap so callers consuming ``RunnerResult.steps``
    see the same signal."""
    cb = ContextBudget(max_sub_goals_per_url=1)
    runner, calls = _runner_with_budget(cb)

    url = "https://www.example.com/boat/abc"
    runner.run_with_status(_navigate_plan(url))
    result = runner.run_with_status(_navigate_plan(url))

    assert len(result.steps) == 1
    assert result.steps[0].skip is True
    assert result.steps[0].skip_reason == "listing_budget_exceeded"
