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
from mantis_agent.gym.step_handlers.holo3 import (
    Holo3StepHandler,
    _build_scoped_task,
    _format_prior_attempt,
)
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


# ── Epic #377 Phase A.2: brain_loop_exhausted stamping ──────────────


@patch("mantis_agent.gym.step_handlers.holo3.GymRunner")
def test_brain_loop_exhausted_stamps_failure_class_on_max_steps(mock_runner_cls):
    """When the inner GymRunner exits at the step budget without
    success, the handler stamps ``failure_class=brain_loop_exhausted``
    so the critic / dashboard can route on a stable signal rather than
    parsing prose."""
    mock_inner = MagicMock()
    mock_inner.run.return_value = MagicMock(
        success=False,
        total_steps=10,
        total_time=180.0,
        termination_reason="max_steps",
    )
    mock_runner_cls.return_value = mock_inner

    runner = _FakeRunner()
    ctx = _ctx(runner, env=MagicMock(), extractor=MagicMock())

    result = Holo3StepHandler(runner).execute(_step(), ctx)

    assert result.success is False
    assert result.failure_class == "brain_loop_exhausted"
    assert result.steps_used == 10


@patch("mantis_agent.gym.step_handlers.holo3.GymRunner")
def test_brain_loop_exhausted_stamps_on_loop_detector_trip(mock_runner_cls):
    """Same class for loop-detector trips — both signal the brain went
    round-and-round without finishing. The critic treats them
    identically (rewrite the intent rather than retry it)."""
    mock_inner = MagicMock()
    mock_inner.run.return_value = MagicMock(
        success=False,
        total_steps=6,
        total_time=72.0,
        termination_reason="loop",
    )
    mock_runner_cls.return_value = mock_inner

    runner = _FakeRunner()
    ctx = _ctx(runner, env=MagicMock())

    result = Holo3StepHandler(runner).execute(_step(), ctx)

    assert result.success is False
    assert result.failure_class == "brain_loop_exhausted"


@patch("mantis_agent.gym.step_handlers.holo3.GymRunner")
def test_brain_loop_exhausted_not_stamped_when_success(mock_runner_cls):
    """A successful inner run at the budget cap (rare but possible) is
    NOT brain_loop_exhausted — the brain just used its full budget to
    finish."""
    mock_inner = MagicMock()
    mock_inner.run.return_value = MagicMock(
        success=True,
        total_steps=8,
        total_time=60.0,
        termination_reason="done",
    )
    mock_runner_cls.return_value = mock_inner

    runner = _FakeRunner()
    ctx = _ctx(runner, env=MagicMock())

    result = Holo3StepHandler(runner).execute(_step(), ctx)

    assert result.success is True
    assert result.failure_class == ""


# ── _build_scoped_task / _format_prior_attempt (P1 #6) ─────────────


def test_scoped_task_intent_only_when_step_is_bare():
    """A step with only ``intent`` set produces a one-line scope —
    no extra sections, no failure history. Keeps the prompt tight
    for the common "fresh step, no retries" case."""
    runner = _FakeRunner()
    step = MicroIntent(intent="Click Help", type="click")
    out = _build_scoped_task(step, runner, step_index=3)
    assert out == "Sub-goal: Click Help"


def test_scoped_task_includes_verify_when_set():
    """``step.verify`` lands under ``Success criterion:`` — Holo3
    needs to check this before reporting ``done(success=True)`` so
    a partial completion doesn't pose as a finished sub-goal."""
    runner = _FakeRunner()
    step = MicroIntent(
        intent="Filter by Contacted", type="submit",
        verify="URL includes status=Contacted",
    )
    out = _build_scoped_task(step, runner, step_index=6)
    assert "Sub-goal: Filter by Contacted" in out
    assert "Success criterion: URL includes status=Contacted" in out


def test_scoped_task_surfaces_target_hints_from_params_and_hints():
    """``params.label`` / ``kind`` / ``aliases`` and ``hints.region`` /
    ``near`` / ``layout`` all surface under ``Target hints:`` so the
    brain doesn't have to re-derive semantics the decomposer already
    extracted. This is the canonical staff-crm case: a submit step
    with a known button label that vision-only mode would otherwise
    have to recognise from raw pixels."""
    runner = _FakeRunner()
    step = MicroIntent(
        intent="Click the Update Lead button", type="submit",
        params={"label": "Update Lead", "kind": "button",
                "aliases": ["Save", "Save Changes"]},
        hints={"region": "form-footer", "near": "Cancel"},
    )
    out = _build_scoped_task(step, runner, step_index=18)
    assert "Target hints:" in out
    assert "label: Update Lead" in out
    assert "kind: button" in out
    assert "aliases: Save, Save Changes" in out
    assert "region: form-footer" in out
    assert "near: Cancel" in out


def test_scoped_task_includes_outcome_tagged_prior_failures():
    """``_step_failure_history`` entries surface as ``Previous attempts``
    annotated with their failure class (no_state_change /
    wrong_target / brain_loop_exhausted). The brain reads these
    and is expected to refute the same coordinates / patterns —
    THIS is what the pre-#hub ``Recent actions:`` block of bare
    ``click(x,y)`` strings failed to communicate."""
    runner = _FakeRunner()
    runner._step_failure_history = {8: [
        {"x": 531, "y": 442, "label": "Apply",
         "matched_label": "Apply", "kind": "no_state_change",
         "reason": "no observable state change"},
        {"x": 687, "y": 320, "label": "Apply",
         "matched_label": "Apply Filter", "kind": "wrong_target",
         "reason": "click landed on lead detail row"},
    ]}
    step = MicroIntent(
        intent="Click Apply", type="submit",
        params={"label": "Apply", "kind": "button"},
    )
    out = _build_scoped_task(step, runner, step_index=8)
    assert "Previous attempts at THIS sub-goal" in out
    # Both entries appear; both carry their outcomes.
    assert "(531, 442)" in out
    assert "no observable state change" in out
    assert "(687, 320)" in out
    assert "wrong target" in out
    # Coordinates aren't bare — the matched_label tag is woven in.
    assert "Apply Filter" in out


def test_scoped_task_caps_prior_failures_to_window():
    """When >3 prior attempts are stored, only the most-recent 3
    appear (window size). Older failures add noise without signal —
    Holo3 context is the scarcer resource than runner memory."""
    runner = _FakeRunner()
    runner._step_failure_history = {2: [
        {"x": i, "y": i, "label": "x", "kind": "no_state_change",
         "reason": f"attempt {i}"} for i in range(5)
    ]}
    step = MicroIntent(intent="x", type="submit",
                       params={"label": "x", "kind": "button"})
    out = _build_scoped_task(step, runner, step_index=2)
    # Most recent three are i=2, 3, 4. Older (i=0, 1) absent.
    assert "(2, 2)" in out
    assert "(3, 3)" in out
    assert "(4, 4)" in out
    assert "(0, 0)" not in out
    assert "(1, 1)" not in out


def test_scoped_task_appends_recovery_hint_block_when_present():
    """Agentic-recovery hints written to ``_recovery_hints`` flow
    through unchanged — they're typically the most specific signal
    (Claude analysing the failure screenshot) and the brain should
    see them last so they aren't truncated."""
    runner = _FakeRunner()
    runner._recovery_hints = {5: [
        "The Apply button is to the RIGHT of the Clear Filters button"
    ]}
    step = MicroIntent(intent="Click Apply", type="submit",
                       params={"label": "Apply", "kind": "button"})
    out = _build_scoped_task(step, runner, step_index=5)
    assert "to the RIGHT of the Clear Filters button" in out
    # The recovery hint block lives at the END so a token budget cap
    # doesn't truncate Claude's specific analysis in favour of the
    # boilerplate target hints.
    hint_pos = out.find("to the RIGHT")
    target_pos = out.find("Target hints:")
    assert hint_pos > target_pos


def test_format_prior_attempt_maps_known_failure_classes():
    """Each ``kind`` value the recovery loop stamps maps to a
    human-readable outcome verb. Pin the strings — a refactor that
    drops a mapping would silently change the prompt phrasing the
    brain learned to interpret."""
    base = {"x": 10, "y": 20, "label": "X", "matched_label": "X"}
    nsc = _format_prior_attempt({**base, "kind": "no_state_change"})
    assert "no observable state change" in nsc
    wt = _format_prior_attempt({**base, "kind": "wrong_target"})
    assert "wrong target" in wt
    ble = _format_prior_attempt({**base, "kind": "brain_loop_exhausted"})
    assert "brain ran out of moves" in ble
    sm = _format_prior_attempt({**base, "kind": "selector_miss"})
    assert "didn't hit an interactive element" in sm
    # Unknown kind → generic "failed (<kind>)" fallback.
    unk = _format_prior_attempt({**base, "kind": "novel_class"})
    assert "novel_class" in unk and "failed" in unk


@patch("mantis_agent.gym.step_handlers.holo3.GymRunner")
def test_holo3_handler_uses_scoped_task(mock_runner_cls):
    """End-to-end: a step with params + verify produces a scoped
    task string that the inner GymRunner receives. Locks the wire
    so the new scoping doesn't silently revert to the old bare-
    intent path on a future refactor."""
    mock_inner = MagicMock()
    mock_inner.run.return_value = MagicMock(
        success=True, total_steps=1, total_time=0.5,
        termination_reason="done",
    )
    mock_runner_cls.return_value = mock_inner

    runner = _FakeRunner()
    ctx = _ctx(runner, env=MagicMock())
    step = MicroIntent(
        intent="Click Apply", type="submit",
        verify="URL includes priority=Critical",
        params={"label": "Apply", "kind": "button"},
    )
    Holo3StepHandler(runner).execute(step, ctx)

    task_arg = mock_inner.run.call_args.kwargs["task"]
    assert "Sub-goal: Click Apply" in task_arg
    assert "Success criterion: URL includes priority=Critical" in task_arg
    assert "label: Apply" in task_arg


@patch("mantis_agent.gym.step_handlers.holo3.GymRunner")
def test_brain_loop_exhausted_not_stamped_on_env_done_failure(mock_runner_cls):
    """A failure terminated by ``env_done`` (env-side signal) or
    ``done`` (model emitted DONE but with success=False) is NOT a
    budget-burn — different bucket. The class stays empty so the
    classifier's fallback rules (or unknown) pick it up."""
    mock_inner = MagicMock()
    mock_inner.run.return_value = MagicMock(
        success=False,
        total_steps=3,
        total_time=10.0,
        termination_reason="env_done",
    )
    mock_runner_cls.return_value = mock_inner

    runner = _FakeRunner()
    ctx = _ctx(runner, env=MagicMock())

    result = Holo3StepHandler(runner).execute(_step(), ctx)

    assert result.success is False
    assert result.failure_class == ""
