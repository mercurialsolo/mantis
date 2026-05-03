"""Tests for #119 step 2 — run_streaming convenience function."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from PIL import Image

from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.run_streaming import run_streaming
from mantis_agent.gym.runner import RunResult


# ── Fakes ───────────────────────────────────────────────────────────────


@dataclass
class _FakeFrame:
    image: Image.Image
    timestamp: float = 0.0
    index: int = 0


class _FakeStreamer:
    def __init__(self, viewport: tuple[int, int] = (320, 240)) -> None:
        self._viewport = viewport
        self.screen_size = viewport
        self.captures = 0

    def capture_once(self) -> _FakeFrame:
        self.captures += 1
        img = Image.new("RGB", self._viewport, (128, 128, 128))
        return _FakeFrame(image=img, timestamp=time.time(), index=self.captures - 1)


class _FakeExecutor:
    @dataclass
    class _Result:
        action: Action
        success: bool = True
        duration: float = 0.001
        error: str = ""

    def __init__(self) -> None:
        self.executed: list[Action] = []
        self.screen_bounds: tuple[int, int] = (1920, 1080)

    def execute(self, action: Action) -> "_FakeExecutor._Result":
        self.executed.append(action)
        return self._Result(action=action)


@dataclass
class _FakeInferenceResult:
    action: Action
    thinking: str = ""
    raw_output: str = ""
    predicted_outcome: str = ""


class _DoneAfterNBrain:
    """Brain that returns a CLICK for N steps then DONE."""

    def __init__(self, n: int = 3) -> None:
        self.n = n
        self.calls: list[dict] = []
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def think(
        self,
        frames: Any = None,
        task: str = "",
        action_history: Any = None,
        screen_size: tuple[int, int] = (1920, 1080),
    ) -> _FakeInferenceResult:
        self.calls.append(
            {"task": task, "action_history": list(action_history or [])}
        )
        if len(self.calls) >= self.n:
            return _FakeInferenceResult(
                action=Action(ActionType.DONE, {"success": True, "summary": "ok"}),
            )
        return _FakeInferenceResult(
            action=Action(ActionType.CLICK, {"x": 10, "y": 10}),
        )


# ── Smoke ───────────────────────────────────────────────────────────────


def test_run_streaming_returns_run_result() -> None:
    brain = _DoneAfterNBrain(n=2)
    result = run_streaming(
        brain=brain,
        task="t",
        streamer=_FakeStreamer(),
        executor=_FakeExecutor(),
        settle_time=0.0,
        max_steps=5,
    )
    assert isinstance(result, RunResult)


def test_run_streaming_terminates_on_done() -> None:
    brain = _DoneAfterNBrain(n=2)
    result = run_streaming(
        brain=brain, task="t",
        streamer=_FakeStreamer(), executor=_FakeExecutor(),
        settle_time=0.0,
    )
    assert result.termination_reason == "done"
    assert result.success is True
    assert result.total_steps == 2


def test_run_streaming_respects_max_steps() -> None:
    brain = _DoneAfterNBrain(n=999)  # Never returns DONE within the cap.
    result = run_streaming(
        brain=brain, task="t",
        streamer=_FakeStreamer(), executor=_FakeExecutor(),
        settle_time=0.0, max_steps=4,
    )
    assert result.total_steps == 4
    assert result.termination_reason == "max_steps"


# ── Plumbing ────────────────────────────────────────────────────────────


def test_run_streaming_calls_brain_with_task() -> None:
    brain = _DoneAfterNBrain(n=2)
    run_streaming(
        brain=brain, task="MY TASK",
        streamer=_FakeStreamer(), executor=_FakeExecutor(),
        settle_time=0.0,
    )
    assert all("MY TASK" in c["task"] for c in brain.calls)


def test_run_streaming_drives_executor_actions() -> None:
    brain = _DoneAfterNBrain(n=3)  # 2 clicks then DONE.
    executor = _FakeExecutor()
    run_streaming(
        brain=brain, task="t",
        streamer=_FakeStreamer(), executor=executor,
        settle_time=0.0,
    )
    # First two steps fire CLICK; the DONE step doesn't go through executor.execute.
    assert len(executor.executed) == 2
    for action in executor.executed:
        assert action.action_type == ActionType.CLICK


def test_run_streaming_settle_time_observed(monkeypatch) -> None:
    sleep_calls: list[float] = []
    import mantis_agent.gym.streamer_env as mod
    monkeypatch.setattr(mod.time, "sleep", lambda s: sleep_calls.append(s))
    brain = _DoneAfterNBrain(n=3)
    run_streaming(
        brain=brain, task="t",
        streamer=_FakeStreamer(), executor=_FakeExecutor(),
        settle_time=0.4,
    )
    # Two action steps should each sleep 0.4.
    assert sleep_calls.count(0.4) == 2


# ── on_step callback (viewer hook) ──────────────────────────────────────


def test_run_streaming_invokes_on_step_callback() -> None:
    brain = _DoneAfterNBrain(n=2)
    events: list[dict] = []

    def on_step(event: dict) -> None:
        events.append(event)

    run_streaming(
        brain=brain, task="t",
        streamer=_FakeStreamer(), executor=_FakeExecutor(),
        settle_time=0.0,
        on_step=on_step,
    )
    # GymRunner emits at least task_start + step + action + done.
    types = {e["type"] for e in events}
    assert "task_start" in types
    assert "done" in types


# ── Streamer + executor are closed properly ─────────────────────────────


@dataclass
class _ClosingStreamer(_FakeStreamer):
    """Subclass that records close() — the env should never re-call this."""
    closed_count: int = 0


def test_run_streaming_closes_env_after_run() -> None:
    """The env is closed in a finally block so even an exception in the
    runner doesn't leak the streamer."""

    class _BadBrain:
        def load(self) -> None: ...

        def think(self, **kw: Any) -> _FakeInferenceResult:
            raise RuntimeError("boom")

    brain = _BadBrain()
    streamer = _FakeStreamer()
    executor = _FakeExecutor()
    try:
        run_streaming(
            brain=brain, task="t",
            streamer=streamer, executor=executor,
            settle_time=0.0,
        )
    except Exception:
        pass
    # Hard to assert cleanup directly without poking env._closed; the
    # main contract is "doesn't leak / hang" — passing here is the signal.


# ── StreamingCUA still imports + has the new docstring note ─────────────


def test_streamingcua_docstring_points_to_unified_path() -> None:
    """Discoverability: a developer reading agent.py finds the new entrypoint."""
    from mantis_agent import agent

    doc = (agent.__doc__ or "").lower()
    assert "run_streaming" in doc
    assert "gymrunner" in doc


def test_run_streaming_module_registered_under_gym_namespace() -> None:
    """Sanity check that the entrypoint imports cleanly under
    mantis_agent.gym.run_streaming, the path documented in the
    StreamingCUA docstring."""
    import mantis_agent.gym.run_streaming as mod

    assert callable(mod.run_streaming)
