"""GymRunner routing-backend telemetry (#295 + foundation for #300).

Asserts that every classified trajectory step carries an
``executor_backend`` tag and that the per-backend aggregate on
:class:`RunResult.executor_backend_counts` adds up.

The runner exposes three classified backends:

* ``plan``   — :class:`PlanExecutor.execute` ran a structured plan step
                deterministically.
* ``som``    — :class:`PageDiscovery` (Set-of-Mark) picked an element
                and the runner executed via DOM.
* ``vision`` — brain emitted raw coordinates / keystrokes that the env
                executed.

Routing decisions are gated by :class:`RoutingPolicy`, which is also
covered here (env-toggle parsing + default behaviour).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from PIL import Image

from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.base import GymEnvironment, GymObservation, GymResult
from mantis_agent.gym.plans import Plan, PlanStep
from mantis_agent.gym.runner import (
    GymRunner,
    RoutingPolicy,
    TrajectoryStep,
    _trajectory_step_from_dict,
    _trajectory_step_to_dict,
)


# ── Fakes ───────────────────────────────────────────────────────────────


@dataclass
class _BrainResult:
    action: Action
    thinking: str = ""
    predicted_outcome: str = ""


class _ScriptedBrain:
    """Emits a fixed sequence of actions then auto-DONE."""

    def __init__(self, script: list[Action]) -> None:
        self.script = list(script)
        self.calls = 0

    def think(
        self,
        frames: Any,
        task: str,
        action_history: Any = None,
        screen_size: tuple[int, int] = (100, 100),
    ) -> _BrainResult:
        if self.calls >= len(self.script):
            self.calls += 1
            return _BrainResult(Action(
                ActionType.DONE, {"success": True, "summary": "done"},
            ))
        action = self.script[self.calls]
        self.calls += 1
        return _BrainResult(action)


class _NoopEnv(GymEnvironment):
    """Minimal env that returns the same screenshot every step."""

    def __init__(self) -> None:
        self._frame = Image.new("RGB", (100, 100), color=(128, 128, 128))

    @property
    def screen_size(self) -> tuple[int, int]:
        return (100, 100)

    def reset(self, task: str, **kwargs: Any) -> GymObservation:
        return GymObservation(screenshot=self._frame)

    def step(self, action: Action) -> GymResult:
        return GymResult(
            GymObservation(screenshot=self._frame),
            reward=0.0, done=False, info={"url": "https://x.test/a", "title": "A"},
        )

    def close(self) -> None:
        pass


class _FakeStepResult:
    def __init__(self, success: bool, detail: str = "", url_after: str = "") -> None:
        self.success = success
        self.method = "direct"
        self.detail = detail
        self.url_after = url_after


class _FakePlanExecutor:
    """Returns success on every can_execute / execute call."""

    def __init__(self, will_succeed: bool = True) -> None:
        self._will_succeed = will_succeed
        self.calls = 0

    def can_execute(self, step: PlanStep) -> bool:
        return True

    def execute(
        self, step: PlanStep, plan_inputs: dict[str, str],
    ) -> _FakeStepResult:
        self.calls += 1
        return _FakeStepResult(
            success=self._will_succeed,
            detail=f"executed {step.action}: {step.target}",
            url_after="https://x.test/after",
        )


# ── Vision-only run ─────────────────────────────────────────────────────


def test_vision_only_run_tags_every_step_as_vision() -> None:
    """No plan + no executors → every classified step is ``vision``."""
    brain = _ScriptedBrain([
        Action(ActionType.CLICK, {"x": 1, "y": 1}),
        Action(ActionType.WAIT, {"seconds": 0.1}),
    ])
    env = _NoopEnv()
    runner = GymRunner(brain, env, max_steps=4)
    result = runner.run("vision-only task")

    backends = [s.executor_backend for s in result.trajectory]
    assert backends, "should produce at least one step"
    assert all(b == "vision" for b in backends), backends
    assert result.executor_backend_counts == {"vision": len(backends)}


# ── PlanExecutor branch ─────────────────────────────────────────────────


def test_plan_executor_success_tags_steps_as_plan() -> None:
    """Plan + PlanExecutor returning success → executor_backend=plan."""
    plan = Plan(
        name="t", description="", url="https://x.test",
        steps=[
            PlanStep(action="click", target="Login button"),
            PlanStep(action="click", target="Continue"),
        ],
    )
    brain = _ScriptedBrain([
        Action(ActionType.DONE, {"success": True, "summary": "done"}),
    ])
    env = _NoopEnv()
    plan_executor = _FakePlanExecutor(will_succeed=True)
    runner = GymRunner(
        brain=brain, env=env, max_steps=5, plan_executor=plan_executor,
    )
    result = runner.run("plan-driven task", plan=plan)

    # Both plan steps should execute deterministically.
    assert plan_executor.calls == 2
    plan_steps = [s for s in result.trajectory if s.executor_backend == "plan"]
    assert len(plan_steps) == 2, [s.executor_backend for s in result.trajectory]
    assert result.executor_backend_counts.get("plan") == 2


def test_plan_executor_disabled_falls_through_to_vision() -> None:
    """``RoutingPolicy.plan_executor_enabled=False`` skips the plan branch.

    The plan exists but the runner ignores PlanExecutor and lets the
    brain drive every step — every classified step ends up tagged
    ``vision`` even though a PlanExecutor is wired.
    """
    plan = Plan(
        name="t", description="", url="https://x.test",
        steps=[PlanStep(action="click", target="Login button")],
    )
    brain = _ScriptedBrain([
        Action(ActionType.CLICK, {"x": 1, "y": 1}),
        Action(ActionType.DONE, {"success": True, "summary": "done"}),
    ])
    plan_executor = _FakePlanExecutor(will_succeed=True)
    runner = GymRunner(
        brain=brain, env=_NoopEnv(), max_steps=4,
        plan_executor=plan_executor,
        routing_policy=RoutingPolicy(plan_executor_enabled=False),
    )
    result = runner.run("plan-driven task", plan=plan)

    # PlanExecutor must not have run.
    assert plan_executor.calls == 0
    backends = [s.executor_backend for s in result.trajectory]
    assert all(b == "vision" for b in backends if b), backends
    assert "plan" not in result.executor_backend_counts


# ── RoutingPolicy.from_env ──────────────────────────────────────────────


def test_routing_policy_from_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty env yields the dataclass defaults."""
    for v in (
        "MANTIS_ROUTE_PLAN_EXECUTOR",
        "MANTIS_ROUTE_SOM",
        "MANTIS_ROUTE_SOM_CLICKS",
    ):
        monkeypatch.delenv(v, raising=False)

    policy = RoutingPolicy.from_env()
    assert policy.plan_executor_enabled is True
    assert policy.som_enabled is True
    assert policy.som_for_unstructured_clicks is False


def test_routing_policy_from_env_disabled_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``disabled`` flips each toggle off."""
    monkeypatch.setenv("MANTIS_ROUTE_PLAN_EXECUTOR", "disabled")
    monkeypatch.setenv("MANTIS_ROUTE_SOM", "disabled")
    monkeypatch.setenv("MANTIS_ROUTE_SOM_CLICKS", "enabled")

    policy = RoutingPolicy.from_env()
    assert policy.plan_executor_enabled is False
    assert policy.som_enabled is False
    assert policy.som_for_unstructured_clicks is True


def test_routing_policy_from_env_ignores_garbage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unrecognized values keep the default — fail-closed on typos."""
    monkeypatch.setenv("MANTIS_ROUTE_PLAN_EXECUTOR", "yes")  # typo for enabled
    monkeypatch.setenv("MANTIS_ROUTE_SOM", "off")  # typo for disabled

    policy = RoutingPolicy.from_env()
    assert policy.plan_executor_enabled is True  # default kept
    assert policy.som_enabled is True  # default kept


# ── Trajectory round-trip ───────────────────────────────────────────────


def test_trajectory_step_round_trip_carries_executor_backend() -> None:
    """:func:`_trajectory_step_to_dict` ↔ :func:`_trajectory_step_from_dict`
    preserve the new field across a pause/resume snapshot."""
    step = TrajectoryStep(
        step=1,
        action=Action(ActionType.CLICK, {"x": 5, "y": 5}),
        thinking="t", reward=0.0, done=False, inference_time=0.0,
        executor_backend="plan",
    )
    payload = _trajectory_step_to_dict(step)
    assert payload["executor_backend"] == "plan"
    restored = _trajectory_step_from_dict(payload)
    assert restored.executor_backend == "plan"


def test_trajectory_step_default_executor_backend_empty() -> None:
    """Existing pause snapshots (pre-#295) decode without crashing — the
    missing key resolves to the empty-string default."""
    payload = {
        "step": 1,
        "action": {"action_type": "click", "params": {"x": 0, "y": 0}, "reasoning": ""},
        "thinking": "", "reward": 0.0, "done": False, "inference_time": 0.0,
    }
    restored = _trajectory_step_from_dict(payload)
    assert restored.executor_backend == ""
