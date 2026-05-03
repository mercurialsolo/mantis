"""Tests for #119 step 1 — StreamerGymEnv adapter."""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest
from PIL import Image

from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.base import GymEnvironment, GymObservation, GymResult
from mantis_agent.gym.streamer_env import StreamerGymEnv


# ── Fakes (avoid pulling in mss / pyautogui from the [local-cua] extras) ──


@dataclass
class _FakeFrame:
    image: Image.Image
    timestamp: float = 0.0
    index: int = 0


class _FakeStreamer:
    """Stand-in for :class:`ScreenStreamer` — produces deterministic frames."""

    def __init__(self, viewport: tuple[int, int] = (320, 240)) -> None:
        self._viewport = viewport
        self.screen_size = viewport
        self.captures = 0

    def capture_once(self) -> _FakeFrame:
        self.captures += 1
        # Vary color so tests can distinguish frames if needed.
        shade = (self.captures * 17) % 256
        img = Image.new("RGB", self._viewport, (shade, shade, shade))
        return _FakeFrame(image=img, timestamp=time.time(), index=self.captures - 1)


class _FakeExecutor:
    """Stand-in for :class:`ActionExecutor`."""

    @dataclass
    class _Result:
        action: Action
        success: bool = True
        duration: float = 0.0
        error: str = ""

    def __init__(self, fail_with: str = "") -> None:
        self.executed: list[Action] = []
        self.screen_bounds: tuple[int, int] = (1920, 1080)
        self._fail_with = fail_with

    def execute(self, action: Action) -> "_FakeExecutor._Result":
        self.executed.append(action)
        if self._fail_with:
            return self._Result(action=action, success=False, error=self._fail_with)
        return self._Result(action=action, success=True, duration=0.001)


def _env(streamer: _FakeStreamer | None = None, executor: _FakeExecutor | None = None,
         settle_time: float = 0.0) -> StreamerGymEnv:
    """Construct an env with sensible test defaults — settle_time=0
    keeps tests fast."""
    return StreamerGymEnv(
        streamer=streamer or _FakeStreamer(),
        executor=executor or _FakeExecutor(),
        settle_time=settle_time,
    )


# ── Protocol conformance ────────────────────────────────────────────────


def test_streamer_env_is_a_gym_environment() -> None:
    env = _env()
    assert isinstance(env, GymEnvironment)


# ── reset() ─────────────────────────────────────────────────────────────


def test_reset_returns_initial_observation() -> None:
    env = _env()
    obs = env.reset(task="any task")
    assert isinstance(obs, GymObservation)
    assert obs.screenshot is not None


def test_reset_captures_one_frame() -> None:
    streamer = _FakeStreamer()
    env = _env(streamer=streamer)
    env.reset(task="t")
    assert streamer.captures == 1


def test_reset_syncs_screen_size_from_streamer() -> None:
    streamer = _FakeStreamer(viewport=(1280, 720))
    env = _env(streamer=streamer)
    env.reset(task="t")
    assert env.screen_size == (1280, 720)


def test_reset_propagates_screen_size_to_executor_bounds() -> None:
    streamer = _FakeStreamer(viewport=(1280, 720))
    executor = _FakeExecutor()
    env = _env(streamer=streamer, executor=executor)
    env.reset(task="t")
    assert executor.screen_bounds == (1280, 720)


def test_reset_accepts_kwargs_without_complaint() -> None:
    """Protocol passes task_id / start_url / seed — adapter ignores them."""
    env = _env()
    env.reset(task="t", task_id="id", start_url="https://x", seed=42)


def test_reset_after_close_raises() -> None:
    env = _env()
    env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.reset(task="t")


# ── step() ──────────────────────────────────────────────────────────────


def test_step_executes_the_action() -> None:
    executor = _FakeExecutor()
    env = _env(executor=executor)
    env.reset(task="t")
    action = Action(ActionType.CLICK, {"x": 50, "y": 50})
    env.step(action)
    assert executor.executed == [action]


def test_step_returns_result_with_post_action_screenshot() -> None:
    streamer = _FakeStreamer()
    env = _env(streamer=streamer)
    env.reset(task="t")
    captures_after_reset = streamer.captures
    result = env.step(Action(ActionType.CLICK, {"x": 1, "y": 1}))
    assert isinstance(result, GymResult)
    assert result.observation.screenshot is not None
    # One capture per step on top of reset's capture.
    assert streamer.captures == captures_after_reset + 1


def test_step_records_execution_metadata_in_info() -> None:
    env = _env()
    env.reset(task="t")
    result = env.step(Action(ActionType.WAIT, {"seconds": 0.1}))
    assert result.info["execution_success"] is True
    assert isinstance(result.info["execution_duration"], float)


def test_step_records_execution_error_in_info() -> None:
    executor = _FakeExecutor(fail_with="permission denied")
    env = _env(executor=executor)
    env.reset(task="t")
    result = env.step(Action(ActionType.CLICK, {"x": 1, "y": 1}))
    assert result.info["execution_success"] is False
    assert result.info["execution_error"] == "permission denied"


def test_step_reward_is_zero_done_is_false_by_default() -> None:
    """The streamer adapter has no per-task reward signal — runner
    injects rewards via RewardFn. Same for done — only ActionType.DONE
    terminates a host-desktop session."""
    env = _env()
    env.reset(task="t")
    result = env.step(Action(ActionType.CLICK, {"x": 1, "y": 1}))
    assert result.reward == 0.0
    assert result.done is False


def test_step_settle_time_is_observed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-zero settle_time should result in a sleep between execute
    and capture. We mock time.sleep to verify."""
    sleep_calls: list[float] = []
    import mantis_agent.gym.streamer_env as mod
    monkeypatch.setattr(mod.time, "sleep", lambda s: sleep_calls.append(s))
    env = StreamerGymEnv(
        streamer=_FakeStreamer(), executor=_FakeExecutor(), settle_time=0.5,
    )
    env.reset(task="t")
    env.step(Action(ActionType.CLICK, {"x": 1, "y": 1}))
    assert 0.5 in sleep_calls


def test_step_settle_time_zero_skips_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    sleep_calls: list[float] = []
    import mantis_agent.gym.streamer_env as mod
    monkeypatch.setattr(mod.time, "sleep", lambda s: sleep_calls.append(s))
    env = _env(settle_time=0.0)
    env.reset(task="t")
    env.step(Action(ActionType.CLICK, {"x": 1, "y": 1}))
    assert sleep_calls == []


def test_step_after_close_raises() -> None:
    env = _env()
    env.reset(task="t")
    env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.step(Action(ActionType.CLICK, {"x": 1, "y": 1}))


# ── close() ─────────────────────────────────────────────────────────────


def test_close_is_idempotent() -> None:
    env = _env()
    env.close()
    env.close()  # Should not raise


def test_close_marks_env_unusable() -> None:
    env = _env()
    env.reset(task="t")
    env.close()
    with pytest.raises(RuntimeError):
        env.step(Action(ActionType.CLICK, {"x": 1, "y": 1}))


# ── screen_size ─────────────────────────────────────────────────────────


def test_screen_size_default_before_reset() -> None:
    env = _env()
    # Default reported before any capture happens. Sane fallback so
    # callers that read screen_size pre-reset don't crash.
    assert env.screen_size == (1920, 1080)


def test_screen_size_updates_after_reset() -> None:
    streamer = _FakeStreamer(viewport=(640, 480))
    env = _env(streamer=streamer)
    env.reset(task="t")
    assert env.screen_size == (640, 480)


# ── screenshot() (#74 observability convenience) ────────────────────────


def test_screenshot_returns_pil_image() -> None:
    env = _env()
    env.reset(task="t")
    img = env.screenshot()
    assert isinstance(img, Image.Image)


def test_screenshot_after_close_raises() -> None:
    env = _env()
    env.close()
    with pytest.raises(RuntimeError):
        env.screenshot()


# ── Lazy construction (no [local-cua] extras needed) ────────────────────


def test_default_construction_does_not_create_streamer_or_executor() -> None:
    """Lazy: until reset/step/screenshot is called, no real streamer or
    executor is constructed. Important so tests in the slim install
    don't pull in mss/pyautogui."""
    env = StreamerGymEnv()
    # Internals are still None.
    assert env._streamer is None
    assert env._executor is None


# ── Integration: drives a full episode ──────────────────────────────────


def test_full_episode_drives_executor_and_collects_frames() -> None:
    """Sanity: 5-action episode produces 5 distinct frames + 5 executions."""
    streamer = _FakeStreamer()
    executor = _FakeExecutor()
    env = _env(streamer=streamer, executor=executor)
    env.reset(task="t")
    for i in range(5):
        env.step(Action(ActionType.CLICK, {"x": i, "y": i}))
    # 1 reset capture + 5 step captures = 6 captures.
    assert streamer.captures == 6
    # 5 executions (no execute() on reset).
    assert len(executor.executed) == 5
