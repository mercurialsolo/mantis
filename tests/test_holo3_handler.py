"""Holo3StepHandler unit tests — Phase 2 of EPIC #161.

Holo3 is the tactical brain — used as a last resort when Claude can't
identify a clear coordinate. The handler spins up a fresh GymRunner
bound to the same env / brain / grounding.

- Success path: GymRunner.run returns success=True, no verify clause →
  StepResult(success=True) with steps_used + duration forwarded
- Verify success: extractor.extract returns viable data, _check_verify
  returns True → success preserved
- Verify failure: _check_verify returns False → success demoted to
  False, claude_extract billed
- env.current_url updates _last_known_url
- step_type property
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from mantis_agent.gym.step_context import StepContext
from mantis_agent.gym.step_handlers.holo3 import Holo3StepHandler
from mantis_agent.plan_decomposer import MicroIntent


class _FakeRunner:
    def __init__(self) -> None:
        self.costs: dict[str, float] = {"claude_extract": 0}
        self._last_known_url = ""
        self.on_step = None
        self.scroll_traj_calls: list[tuple] = []
        self.verify_call_args: list = []
        self.verify_return = True

    def _update_scroll_state_from_trajectory(self, result, *, context: str) -> None:
        self.scroll_traj_calls.append((result, context))

    def _check_verify(self, condition: str, data, screenshot) -> bool:
        self.verify_call_args.append((condition, data, screenshot))
        return self.verify_return


def _ctx(runner, *, env=None, brain=None, extractor=None, grounding=None) -> StepContext:
    return StepContext(
        env=env or MagicMock(),
        brain=brain or MagicMock(),
        extractor=extractor,
        grounding=grounding,
        cost_meter=None,
        dynamic_verifier=None,
        scanner=None,
        site_config=None,
        tool_channel=None,
        extraction_cache=None,
        state={"index": 11},
    )


def _step(*, verify: str = "") -> MicroIntent:
    return MicroIntent(
        intent="Click the Help link", type="click",
        budget=8, grounding=True, verify=verify,
    )


@patch("mantis_agent.gym.step_handlers.holo3.GymRunner")
def test_holo3_success_no_verify(mock_runner_cls):
    mock_inner = MagicMock()
    mock_inner.run.return_value = MagicMock(success=True, total_steps=4, total_time=2.5)
    mock_runner_cls.return_value = mock_inner

    runner = _FakeRunner()
    env = MagicMock()
    env.current_url = "https://example.com/help"
    ctx = _ctx(runner, env=env)

    result = Holo3StepHandler(runner).execute(_step(), ctx)

    assert result.success is True
    assert result.steps_used == 4
    assert result.duration == 2.5
    assert runner._last_known_url == "https://example.com/help"
    # GymRunner constructed with the right kwargs
    mock_runner_cls.assert_called_once()
    kwargs = mock_runner_cls.call_args.kwargs
    assert kwargs["max_steps"] == 8
    assert kwargs["frames_per_inference"] == 1
    # No verify → no claude_extract
    assert runner.costs["claude_extract"] == 0


@patch("mantis_agent.gym.step_handlers.holo3.GymRunner")
def test_holo3_verify_passes_preserves_success(mock_runner_cls):
    mock_inner = MagicMock()
    mock_inner.run.return_value = MagicMock(success=True, total_steps=2, total_time=1.0)
    mock_runner_cls.return_value = mock_inner

    runner = _FakeRunner()
    runner.verify_return = True
    extractor = MagicMock()
    extract_result = MagicMock(url="https://example.com/help/topic-42")
    extractor.extract.return_value = extract_result
    env = MagicMock()
    env.current_url = "https://example.com/initial"
    ctx = _ctx(runner, env=env, extractor=extractor)

    result = Holo3StepHandler(runner).execute(_step(verify="page contains help text"), ctx)

    assert result.success is True
    assert runner.costs["claude_extract"] == 1
    assert runner.verify_call_args[0][0] == "page contains help text"
    # extract_result.url overrides env.current_url for last_known_url
    assert runner._last_known_url == "https://example.com/help/topic-42"


@patch("mantis_agent.gym.step_handlers.holo3.GymRunner")
def test_holo3_verify_fails_demotes_success(mock_runner_cls):
    mock_inner = MagicMock()
    mock_inner.run.return_value = MagicMock(success=True, total_steps=2, total_time=1.0)
    mock_runner_cls.return_value = mock_inner

    runner = _FakeRunner()
    runner.verify_return = False
    extractor = MagicMock()
    extractor.extract.return_value = MagicMock(url="https://example.com/wrong-page")
    ctx = _ctx(runner, env=MagicMock(), extractor=extractor)

    result = Holo3StepHandler(runner).execute(_step(verify="page contains help text"), ctx)

    assert result.success is False  # demoted
    assert runner.costs["claude_extract"] == 1


@patch("mantis_agent.gym.step_handlers.holo3.GymRunner")
def test_holo3_failure_short_circuits_verify(mock_runner_cls):
    """When the inner GymRunner reports failure, verify never runs."""
    mock_inner = MagicMock()
    mock_inner.run.return_value = MagicMock(success=False, total_steps=8, total_time=12.0)
    mock_runner_cls.return_value = mock_inner

    runner = _FakeRunner()
    extractor = MagicMock()
    ctx = _ctx(runner, env=MagicMock(), extractor=extractor)

    result = Holo3StepHandler(runner).execute(_step(verify="something"), ctx)

    assert result.success is False
    assert runner.costs["claude_extract"] == 0
    extractor.extract.assert_not_called()


@patch("mantis_agent.gym.step_handlers.holo3.GymRunner")
def test_holo3_passes_grounding_only_when_step_grounding_true(mock_runner_cls):
    mock_inner = MagicMock()
    mock_inner.run.return_value = MagicMock(success=True, total_steps=1, total_time=0.5)
    mock_runner_cls.return_value = mock_inner

    runner = _FakeRunner()
    grounding = MagicMock()
    ctx = _ctx(runner, env=MagicMock(), grounding=grounding)

    # step.grounding=True
    Holo3StepHandler(runner).execute(_step(), ctx)
    assert mock_runner_cls.call_args.kwargs["grounding"] is grounding

    mock_runner_cls.reset_mock()
    # step.grounding=False
    step_no_grounding = MicroIntent(intent="x", type="click", budget=4, grounding=False)
    Holo3StepHandler(runner).execute(step_no_grounding, ctx)
    assert mock_runner_cls.call_args.kwargs["grounding"] is None


def test_step_type_property():
    handler = Holo3StepHandler(_FakeRunner())
    assert handler.step_type == "holo3"
