"""MicroPlanRunner hooks: step_callback (#74), cancel_event (#76),
register_tool (#71), launch_app (#72), pause/resume (#73).

These tests use a minimal fake env so the runner can be exercised without a
real Xvfb/Holo3 stack.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest
from PIL import Image

from mantis_agent.actions import Action
from mantis_agent.gym.base import GymEnvironment, GymObservation, GymResult
from mantis_agent.gym.micro_runner import (
    MicroPlanRunner,
    StepResult,
)
from mantis_agent.plan_decomposer import MicroIntent, MicroPlan


# ── Fake env / brain ────────────────────────────────────────────────────


class _FakeEnv(GymEnvironment):
    """Minimal GymEnvironment that records actions and serves a tiny screenshot."""

    def __init__(self, viewport: tuple[int, int] = (320, 200)):
        self._viewport = viewport
        self.actions: list[Action] = []
        self.reset_calls: list[dict[str, Any]] = []

    def reset(self, task: str, **kwargs: Any) -> GymObservation:
        self.reset_calls.append(dict(kwargs))
        return GymObservation(screenshot=self._image())

    def step(self, action: Action) -> GymResult:
        self.actions.append(action)
        return GymResult(GymObservation(screenshot=self._image()), 0.0, False, {})

    def screenshot(self) -> Image.Image:
        return self._image()

    def close(self) -> None:
        pass

    @property
    def screen_size(self) -> tuple[int, int]:
        return self._viewport

    def _image(self) -> Image.Image:
        return Image.new("RGB", self._viewport, "white")


def _runner(env: _FakeEnv, **kwargs: Any) -> MicroPlanRunner:
    """Build a runner with extractor+grounding stubbed out."""
    return MicroPlanRunner(
        brain=None,
        env=env,
        grounding=None,
        extractor=None,
        checkpoint_path="/tmp/_micro_runner_test_ckpt.json",
        run_key="test",
        session_name="test",
        max_cost=999.0,
        max_time_minutes=999,
        **kwargs,
    )


def _trivial_plan() -> MicroPlan:
    """Three navigate steps — fully synchronous, no Holo3/Claude needed."""
    plan = MicroPlan(domain="test")
    for i in range(3):
        plan.steps.append(
            MicroIntent(
                intent=f"Step {i}: navigate to https://example.test/{i}",
                type="navigate", budget=3, section="setup", required=False,
            )
        )
    return plan


# ── #74 step_callback + screenshot retention ────────────────────────────


def test_step_callback_fires_per_step_with_signature():
    env = _FakeEnv()
    seen: list[tuple[int, str, Any, bool]] = []

    def cb(idx: int, intent: str, action: Any, ok: bool) -> None:
        seen.append((idx, intent, action, ok))

    r = _runner(env, step_callback=cb)
    r.run(_trivial_plan())

    assert len(seen) == 3
    indices = [s[0] for s in seen]
    assert indices == [0, 1, 2]
    intents = [s[1] for s in seen]
    assert all("navigate" in s for s in intents)
    # action arg may be None for some step types — that's fine for the contract.


def test_step_callback_failure_does_not_break_run():
    env = _FakeEnv()

    def cb(*_args: Any, **_kw: Any) -> None:
        raise RuntimeError("observability bug")

    r = _runner(env, step_callback=cb)
    # Run must complete despite the broken callback.
    steps = r.run(_trivial_plan())
    assert len(steps) == 3


def test_step_results_carry_screenshot_bytes_by_default():
    env = _FakeEnv()
    r = _runner(env)
    steps = r.run(_trivial_plan())
    assert len(steps) == 3
    for s in steps:
        # PNG header check — shows the helper actually encoded something.
        assert s.screenshot_png is not None and s.screenshot_png[:8] == b"\x89PNG\r\n\x1a\n"


def test_keep_screenshots_caps_retained_bytes():
    env = _FakeEnv()
    r = _runner(env, keep_screenshots=1)
    steps = r.run(_trivial_plan())
    # With cap=1 and 3 steps, only the last result keeps its PNG.
    has_png = [s.screenshot_png is not None for s in steps]
    assert has_png == [False, False, True]


def test_to_dict_excludes_screenshot_and_action():
    s = StepResult(step_index=0, intent="x", success=True, screenshot_png=b"fake-bytes")
    d = s.to_dict()
    assert "screenshot_png" not in d
    assert "last_action" not in d
    # Round-trip ignores ignored fields.
    s2 = StepResult.from_dict(d)
    assert s2.screenshot_png is None
    assert s2.last_action is None


def test_run_with_status_returns_completed_runner_result():
    env = _FakeEnv()
    r = _runner(env)
    result = r.run_with_status(_trivial_plan())
    assert result.status == "completed"
    assert len(result.steps) == 3


# ── #76 cancel_event ────────────────────────────────────────────────────


def test_cancel_event_stops_run_at_step_boundary():
    env = _FakeEnv()
    cancel = threading.Event()
    cb_calls: list[int] = []

    def cb(idx: int, *_args: Any) -> None:
        cb_calls.append(idx)
        # Trip cancel after the first step boundary completes.
        if idx == 0:
            cancel.set()

    r = _runner(env, step_callback=cb, cancel_event=cancel)
    result = r.run_with_status(_trivial_plan())

    assert result.cancelled is True
    assert result.status == "cancelled"
    # First step ran, second was preempted at the boundary.
    assert len(result.steps) == 1


def test_cancel_event_callable_is_supported():
    env = _FakeEnv()
    flips = {"hit": False}

    def cancel_fn() -> bool:
        return flips["hit"]

    def cb(idx: int, *_args: Any) -> None:
        if idx == 1:
            flips["hit"] = True

    r = _runner(env, step_callback=cb, cancel_event=cancel_fn)
    result = r.run_with_status(_trivial_plan())
    assert result.cancelled is True
    assert len(result.steps) == 2


def test_no_cancel_runs_all_steps():
    env = _FakeEnv()
    r = _runner(env)
    result = r.run_with_status(_trivial_plan())
    assert result.cancelled is False
    assert result.status == "completed"
    assert len(result.steps) == 3


# ── #71 register_tool ───────────────────────────────────────────────────


def test_register_tool_and_call_tool():
    env = _FakeEnv()
    r = _runner(env)
    captured: dict[str, Any] = {}

    def handler(args: dict[str, Any]) -> str:
        captured.update(args)
        return f"hi {args.get('name', 'anon')}"

    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    r.register_tool(name="greet", schema=schema, handler=handler)

    listed = r.list_tools()
    assert listed == [{"name": "greet", "schema": schema}]

    out = r.call_tool("greet", {"name": "Mantis"})
    assert out == "hi Mantis"
    assert captured == {"name": "Mantis"}


def test_register_tool_rejects_non_callable():
    env = _FakeEnv()
    r = _runner(env)
    with pytest.raises(TypeError):
        r.register_tool("bad", schema={}, handler="not-callable")  # type: ignore[arg-type]


def test_call_tool_unknown_raises_keyerror():
    env = _FakeEnv()
    r = _runner(env)
    with pytest.raises(KeyError):
        r.call_tool("missing")


def test_invoke_tool_surfaces_handler_errors_as_data():
    env = _FakeEnv()
    r = _runner(env)

    def handler(_args: dict[str, Any]) -> None:
        raise RuntimeError("nope")

    r.register_tool("kapow", {}, handler)
    ok, data = r._invoke_tool("kapow", {"x": 1})
    assert ok is False
    assert "RuntimeError" in data and "nope" in data
