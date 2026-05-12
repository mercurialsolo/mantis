"""Tests for ``GymRunner`` host-tool registration + pause/resume (#285).

Three scenarios pinned (matching the three test cases in the issue):

1. ``register_tool`` + a brain that emits ``TOOL_CALL`` → handler returns
   a value → loop continues, trajectory carries the tool-call step.
2. Same setup but the handler raises ``PauseRequested`` → ``run()``
   returns ``RunResult(paused=True, pause_state=...)`` with the prompt
   propagated through.
3. ``resume(pause_state, user_input=...)`` continues the run; the next
   handler can read the supplied reply via ``consume_pause_input`` exactly
   once.

The brain and env are stubs — no real perception, no real screenshot
mutation. The point is the host-tool plumbing, not the agent loop's
content.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.base import GymEnvironment, GymObservation, GymResult
from mantis_agent.gym.checkpoint import PauseRequested, PauseState
from mantis_agent.gym.runner import GymRunner, RunResult


# ── Stubs ─────────────────────────────────────────────────────────────────


def _blank_image(width: int = 32, height: int = 24) -> Image.Image:
    return Image.new("RGB", (width, height), color="white")


class _StubEnv(GymEnvironment):
    """Minimal env — every step returns a blank frame + reward 0.0."""

    def __init__(self, screen: tuple[int, int] = (320, 200)) -> None:
        self._screen = screen
        self.reset_calls = 0
        self.step_calls = 0

    def reset(self, task: str, **kwargs: Any) -> GymObservation:
        self.reset_calls += 1
        return GymObservation(screenshot=_blank_image(*self._screen), extras={})

    def step(self, action: Action) -> GymResult:
        self.step_calls += 1
        return GymResult(
            observation=GymObservation(
                screenshot=_blank_image(*self._screen), extras={},
            ),
            reward=0.0,
            done=False,
            info={"url": "", "title": ""},
        )

    def close(self) -> None:
        return None

    @property
    def screen_size(self) -> tuple[int, int]:
        return self._screen

    def _capture(self) -> GymObservation:
        """Used by the resume path so the brain has a fresh frame."""
        return GymObservation(
            screenshot=_blank_image(*self._screen), extras={},
        )


class _ScriptedBrain:
    """Brain that emits a scripted sequence of actions.

    Each call to ``think()`` pops the next action off the script. Once
    empty, returns a ``DONE`` action so the runner terminates cleanly.
    """

    def __init__(self, actions: list[Action]) -> None:
        self._actions = list(actions)
        self.think_calls = 0
        self.tasks_seen: list[str] = []

    def load(self) -> None:
        return None

    def think(
        self,
        frames: list[Image.Image],
        task: str,
        action_history: list[Action] | None = None,
        screen_size: tuple[int, int] = (1920, 1080),
    ) -> Any:
        self.think_calls += 1
        self.tasks_seen.append(task)
        if self._actions:
            action = self._actions.pop(0)
        else:
            action = Action(
                action_type=ActionType.DONE,
                params={"success": True, "summary": "scripted brain exhausted"},
            )

        class _Result:
            pass

        result = _Result()
        result.action = action
        result.thinking = ""
        return result


def _tool_call(name: str, args: dict[str, Any] | None = None) -> Action:
    return Action(
        action_type=ActionType.TOOL_CALL,
        params={"name": name, "args": args or {}},
    )


def _runner(actions: list[Action], **kwargs: Any) -> tuple[GymRunner, _StubEnv, _ScriptedBrain]:
    env = _StubEnv()
    brain = _ScriptedBrain(actions)
    runner = GymRunner(brain=brain, env=env, max_steps=10, **kwargs)
    return runner, env, brain


# ── 1. Handler returns value → loop continues ─────────────────────────────


def test_tool_call_handler_returns_value_loop_continues() -> None:
    """A registered handler returning a value records a trajectory step
    and the loop continues to the next brain.think call."""
    runner, env, brain = _runner([
        _tool_call("greet", {"who": "world"}),
        # Brain falls off the script → returns DONE → run terminates cleanly.
    ])
    invocations: list[dict[str, Any]] = []

    def greet(args: dict[str, Any]) -> str:
        invocations.append(args)
        return f"hello {args.get('who', '')}".strip()

    runner.register_tool("greet", {"type": "object"}, greet)

    result = runner.run(task="say hi", task_id="t1")

    assert isinstance(result, RunResult)
    assert result.paused is False
    assert result.pause_state is None
    assert invocations == [{"who": "world"}]
    # Tool-call step is in the trajectory with the handler's return in feedback.
    tool_steps = [s for s in result.trajectory if s.action.action_type is ActionType.TOOL_CALL]
    assert len(tool_steps) == 1
    assert tool_steps[0].feedback == "tool:greet:ok:hello world"
    # env.step was never called for the tool-call step.
    assert env.step_calls == 0


def test_list_tools_returns_registered_names_and_schemas() -> None:
    runner, _env, _brain = _runner([])
    runner.register_tool("a", {"type": "object", "properties": {"x": {}}}, lambda _: "ok")
    runner.register_tool("b", {}, lambda _: "ok")
    listed = runner.list_tools()
    names = {t["name"] for t in listed}
    assert names == {"a", "b"}


def test_unregistered_tool_call_surfaces_error_in_feedback() -> None:
    """Brain emits TOOL_CALL for a name that wasn't registered — the
    runner records the step with a `tool:<n>:not_registered:...` feedback
    string and keeps going (mirrors MicroPlanRunner's non-swallow rule)."""
    runner, _env, _brain = _runner([_tool_call("missing", {})])
    result = runner.run(task="t", task_id="t1")
    tool_steps = [s for s in result.trajectory if s.action.action_type is ActionType.TOOL_CALL]
    assert len(tool_steps) == 1
    assert tool_steps[0].feedback.startswith("tool:missing:not_registered")


# ── 2. Handler raises PauseRequested → RunResult(paused=True) ─────────────


def test_handler_pause_returns_paused_run_result() -> None:
    runner, _env, _brain = _runner([
        _tool_call("ask_user", {"why": "need clarification"}),
    ])

    def ask_user(_args: dict[str, Any]) -> Any:
        raise PauseRequested(
            reason="user_input",
            prompt="What's your email?",
        )

    runner.register_tool("ask_user", {"type": "object"}, ask_user)

    result = runner.run(task="onboard", task_id="t-onboard")

    assert result.paused is True
    assert isinstance(result.pause_state, PauseState)
    assert result.pause_state.pending_tool == "ask_user"
    assert result.pause_state.pending_reason == "user_input"
    assert result.pause_state.prompt == "What's your email?"
    assert result.pause_state.task == "onboard"
    assert result.pause_state.task_id == "t-onboard"
    # The paused tool-call step is in both the runtime trajectory AND
    # the snapshot, so resume can replay.
    assert len(result.trajectory) >= 1
    assert result.termination_reason == "paused"


def test_pause_state_round_trips_through_dict() -> None:
    """PauseState must be JSON-serializable so the host can store it on
    its own state object across processes / restarts."""
    runner, _env, _brain = _runner([_tool_call("ask", {})])
    runner.register_tool(
        "ask", {},
        lambda _a: (_ for _ in ()).throw(
            PauseRequested(reason="user_input", prompt="?"),
        ),
    )
    result = runner.run(task="t", task_id="t1")
    assert result.pause_state is not None
    payload = result.pause_state.to_dict()
    # Round-trip via from_dict — exercise the JSON-friendly path.
    restored = PauseState.from_dict(payload)
    assert restored.pending_tool == "ask"
    assert restored.task == "t"
    assert restored.trajectory_steps == result.pause_state.trajectory_steps


# ── 3. resume() continues with consume_pause_input visible to handler ─────


def test_resume_makes_user_input_visible_to_consume_pause_input() -> None:
    """The handler that paused asks again on resume; this time it reads
    the user reply via ``consume_pause_input`` instead of raising.
    Read-once: a second call returns the default."""
    seen_inputs: list[Any] = []

    def ask_user(_args: dict[str, Any]) -> Any:
        # On first call: raise to pause. On resume: read the input.
        reply = runner.consume_pause_input(default=None)
        if reply is None:
            raise PauseRequested(reason="user_input", prompt="email?")
        seen_inputs.append(reply)
        return f"got:{reply}"

    runner, _env, _brain = _runner([
        # Two TOOL_CALL actions in the script: first pauses, second runs.
        # On resume the brain re-emits the tool call (host-driven re-prompt).
        _tool_call("ask_user", {}),
        _tool_call("ask_user", {}),
    ])
    runner.register_tool("ask_user", {}, ask_user)

    first = runner.run(task="onboard", task_id="t-onboard")
    assert first.paused is True
    assert first.pause_state is not None
    assert seen_inputs == []

    second = runner.resume(
        first.pause_state, user_input="user@example.com",
    )
    # Resume completes (brain script falls off → DONE).
    assert second.paused is False
    assert seen_inputs == ["user@example.com"]
    # The combined trajectory carries both the paused tool-call step and
    # the successful resumed tool-call step.
    tool_steps = [
        s for s in second.trajectory
        if s.action.action_type is ActionType.TOOL_CALL
    ]
    assert len(tool_steps) >= 2
    assert tool_steps[-1].feedback == "tool:ask_user:ok:got:user@example.com"


def test_resume_skips_env_reset() -> None:
    """The env is owned by the host across pause/resume — resume() must
    NOT call ``env.reset`` again, otherwise the URL / Chrome profile
    state the host is holding would be wiped."""
    runner, env, _brain = _runner([
        _tool_call("ask", {}),
        _tool_call("ask", {}),
    ])

    def ask(_a: dict[str, Any]) -> Any:
        reply = runner.consume_pause_input()
        if reply is None:
            raise PauseRequested(reason="user_input", prompt="?")
        return "ok"

    runner.register_tool("ask", {}, ask)

    runner.run(task="t", task_id="t1")
    pre_resume_resets = env.reset_calls
    assert pre_resume_resets == 1  # one for the initial run

    paused = runner.run(task="t", task_id="t1")
    # Second run() (without going through resume()) reset the env again.
    assert env.reset_calls == 2

    # Explicit resume path: NO new env.reset.
    runner.resume(paused.pause_state, user_input="hi")
    assert env.reset_calls == 2


def test_consume_pause_input_is_read_once() -> None:
    runner, _env, _brain = _runner([])
    runner._pause_input = "secret"
    assert runner.consume_pause_input() == "secret"
    # Second read sees the default.
    assert runner.consume_pause_input(default="default-value") == "default-value"
