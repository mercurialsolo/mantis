"""Regression test for #125 — AgentResult.total_time must be a duration, not a Unix timestamp.

Three early-return paths in StreamingCUA._run_loop previously stored time.time()
into total_time, leaking a wall-clock epoch into the result.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import pytest

from mantis_agent.actions import Action, ActionType
from mantis_agent.agent import StreamingCUA


@dataclass
class _StubInference:
    action: Action
    raw_output: str = ""
    thinking: str = ""
    tokens_used: int = 0


class _DoneBrain:
    """Brain that immediately calls done()."""

    def think(self, frames, task, action_history=None, screen_size=(1920, 1080)):
        return _StubInference(
            action=Action(ActionType.DONE, {"success": True, "summary": "ok"}),
        )


class _LoopBrain:
    """Brain that emits the same click forever — triggers loop detector."""

    def think(self, frames, task, action_history=None, screen_size=(1920, 1080)):
        return _StubInference(action=Action(ActionType.CLICK, {"x": 100, "y": 100}))


class _MaxStepsBrain:
    """Brain that emits a fresh wait each step — runs until max_steps."""

    def __init__(self) -> None:
        self.calls = 0

    def think(self, frames, task, action_history=None, screen_size=(1920, 1080)):
        self.calls += 1
        # Each call is unique so loop detector never fires.
        return _StubInference(action=Action(ActionType.WAIT, {"seconds": self.calls}))


class _StubStreamer:
    fps = 30.0  # high so asyncio.sleep is short
    screen_size = (1920, 1080)

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def get_recent_frames(self, n: int) -> list[Any]:
        return [object()] * max(n, 1)
    def capture_once(self) -> None: ...


class _StubExecutor:
    screen_bounds: tuple[int, int] = (1920, 1080)

    def execute(self, action: Action) -> Any:
        from mantis_agent.executor import ExecutionResult
        return ExecutionResult(action=action, success=True)


def _make_agent(brain, **kwargs) -> StreamingCUA:
    return StreamingCUA(
        brain=brain,
        streamer=_StubStreamer(),
        executor=_StubExecutor(),
        settle_time=0.0,
        **kwargs,
    )


def _assert_duration(total_time: float) -> None:
    """A duration is small; a Unix timestamp from time.time() is ~1.7e9."""
    assert 0.0 <= total_time < 60.0, (
        f"total_time={total_time!r} looks like a Unix timestamp, not a duration"
    )


def test_total_time_is_duration_on_done() -> None:
    agent = _make_agent(_DoneBrain(), max_steps=5)
    result = asyncio.run(agent.run("noop"))
    _assert_duration(result.total_time)
    assert result.success is True


def test_total_time_is_duration_on_loop_break() -> None:
    agent = _make_agent(_LoopBrain(), max_steps=20)
    result = asyncio.run(agent.run("noop"))
    _assert_duration(result.total_time)
    assert result.success is False
    assert "loop" in result.summary.lower()


def test_total_time_is_duration_on_max_steps() -> None:
    agent = _make_agent(_MaxStepsBrain(), max_steps=3)
    result = asyncio.run(agent.run("noop"))
    _assert_duration(result.total_time)
    assert result.total_steps == 3


def test_total_time_monotonic_with_wallclock() -> None:
    agent = _make_agent(_DoneBrain(), max_steps=5)
    t0 = time.time()
    result = asyncio.run(agent.run("noop"))
    elapsed = time.time() - t0
    assert result.total_time <= elapsed + 0.5
