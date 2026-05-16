"""Audit batch — server contract hardening.

Three independent guards on the /v1/predict surface:

* **Raw-long-plan guard** — long ``plan_text`` with ``decompose=False``
  used to silently route into a single-task suite. The brain has no
  scaffolding to decompose internally, so long raw plans oscillate
  or terminate prematurely. The guard rejects the request unless
  the caller explicitly opts in via ``allow_raw_long_plan=True``.

* **Honest terminal_status** — ``build_micro_result`` now emits a
  top-level ``terminal_status`` derived from ``runner._final_status``
  and ``runner._final_halt_reason``. The detached-status writer
  reads it instead of stamping ``"succeeded"`` on any non-exception
  result. A REQUIRED step that halted now surfaces as ``halted`` /
  ``budget_exceeded`` / ``time_exceeded``, not as a success.

* **Wire validate_micro_steps()** — the validator runs after
  decomposition / file-load and BEFORE ``build_micro_suite``. A
  1000-step plan from a runaway decomposer or a JSON file missing
  ``intent`` / ``type`` fields now fails fast instead of consuming
  Modal time.
"""

from __future__ import annotations

import pytest

from mantis_agent.baseten_server.runtime import BasetenCUARuntime


# ── Raw-long-plan guard ────────────────────────────────────────────


def _runtime() -> BasetenCUARuntime:
    return BasetenCUARuntime.__new__(BasetenCUARuntime)  # bypass __init__


def test_raw_text_suite_accepts_short_plan() -> None:
    """Short plan_text continues to work — the guard only kicks in
    on plans that genuinely exceed the executor's single-goal
    assumption."""
    rt = _runtime()
    suite = rt._raw_text_suite("Click the Login button.", {"start_url": "https://x"})
    assert suite["tasks"][0]["intent"] == "Click the Login button."


def test_raw_text_suite_rejects_long_plan_without_opt_in() -> None:
    """A 100-word plan_text with decompose=False must error — the
    server cap is 80 words. The error message tells the caller how
    to either decompose (drop decompose=False) or override
    (allow_raw_long_plan=true)."""
    rt = _runtime()
    long_text = " ".join(["word"] * 100)
    with pytest.raises(ValueError, match=r"100 words.*server cap.*80"):
        rt._raw_text_suite(long_text, {})


def test_raw_text_suite_accepts_long_plan_with_opt_in() -> None:
    """Benchmark scaffolding / ablation harnesses that intentionally
    want the brain to run a long text as a single task can opt in
    via ``allow_raw_long_plan=True``. The guard then steps aside."""
    rt = _runtime()
    long_text = " ".join(["word"] * 200)
    suite = rt._raw_text_suite(long_text, {"allow_raw_long_plan": True})
    assert long_text in suite["tasks"][0]["intent"]


def test_raw_text_suite_error_message_names_the_two_escape_hatches() -> None:
    """The error message must tell the caller BOTH paths forward —
    decompose (the recommended default) and allow_raw_long_plan
    (the expert escape hatch). Without that, callers have to read
    server source to recover."""
    rt = _runtime()
    long_text = " ".join(["word"] * 100)
    with pytest.raises(ValueError) as excinfo:
        rt._raw_text_suite(long_text, {})
    msg = str(excinfo.value).lower()
    assert "decompose" in msg
    assert "allow_raw_long_plan" in msg


# ── Honest terminal_status in build_micro_result ───────────────────


def test_terminal_status_completed_when_runner_completed_all_success() -> None:
    """``runner._final_status == "completed"`` + every step result
    success → terminal_status = "completed". The happy path; same
    semantics as the prior "succeeded" stamp."""
    from mantis_agent.gym.checkpoint import StepResult
    from mantis_agent.server_utils import build_micro_result

    runner = _build_minimal_runner(final_status="completed")
    step_results = [
        StepResult(step_index=0, intent="x", success=True),
        StepResult(step_index=1, intent="y", success=True),
    ]
    result = build_micro_result(
        runner=runner, run_id="r1", provider="modal",
        session_name="s", model_name="m", elapsed_seconds=10.0,
        step_results=step_results, state_key="", profile_id="",
        workflow_id="", checkpoint_path="", plan_signature="",
        resume_state=False,
    )
    assert result["terminal_status"] == "completed"


def test_terminal_status_completed_with_failures_when_some_steps_failed() -> None:
    """``runner._final_status == "completed"`` BUT some step.success
    is False → terminal_status = "completed_with_failures". This is
    the case for non-required steps that failed; the runner kept
    going but the run isn't a clean success."""
    from mantis_agent.gym.checkpoint import StepResult
    from mantis_agent.server_utils import build_micro_result

    runner = _build_minimal_runner(final_status="completed")
    step_results = [
        StepResult(step_index=0, intent="x", success=True),
        StepResult(step_index=1, intent="y", success=False),
        StepResult(step_index=2, intent="z", success=True),
    ]
    result = build_micro_result(
        runner=runner, run_id="r1", provider="modal",
        session_name="s", model_name="m", elapsed_seconds=10.0,
        step_results=step_results, state_key="", profile_id="",
        workflow_id="", checkpoint_path="", plan_signature="",
        resume_state=False,
    )
    assert result["terminal_status"] == "completed_with_failures"


def test_terminal_status_budget_exceeded_when_halted_on_budget_cap() -> None:
    """halt_reason=budget_cap maps to terminal_status=budget_exceeded.
    Distinct from generic halt so dashboards / alerts can branch."""
    from mantis_agent.server_utils import build_micro_result

    runner = _build_minimal_runner(final_status="halted", halt_reason="budget_cap")
    result = build_micro_result(
        runner=runner, run_id="r1", provider="modal",
        session_name="s", model_name="m", elapsed_seconds=10.0,
        step_results=[], state_key="", profile_id="",
        workflow_id="", checkpoint_path="", plan_signature="",
        resume_state=False,
    )
    assert result["terminal_status"] == "budget_exceeded"
    assert result["halt_reason"] == "budget_cap"


def test_terminal_status_time_exceeded_when_halted_on_time_cap() -> None:
    from mantis_agent.server_utils import build_micro_result

    runner = _build_minimal_runner(final_status="halted", halt_reason="time_cap")
    result = build_micro_result(
        runner=runner, run_id="r1", provider="modal",
        session_name="s", model_name="m", elapsed_seconds=10.0,
        step_results=[], state_key="", profile_id="",
        workflow_id="", checkpoint_path="", plan_signature="",
        resume_state=False,
    )
    assert result["terminal_status"] == "time_exceeded"


def test_terminal_status_halted_when_halt_reason_is_step_halt() -> None:
    """Generic halt (a REQUIRED step exhausted retries) → terminal_status
    = "halted". Distinct from budget/time so consumers can prioritize."""
    from mantis_agent.server_utils import build_micro_result

    runner = _build_minimal_runner(final_status="halted", halt_reason="step_halt")
    result = build_micro_result(
        runner=runner, run_id="r1", provider="modal",
        session_name="s", model_name="m", elapsed_seconds=10.0,
        step_results=[], state_key="", profile_id="",
        workflow_id="", checkpoint_path="", plan_signature="",
        resume_state=False,
    )
    assert result["terminal_status"] == "halted"


# ── validate_micro_steps wiring (smoke) ────────────────────────────


def test_validate_micro_steps_is_invoked_in_micro_suite_from_path(tmp_path) -> None:
    """``_micro_suite_from_path`` runs the validator on the loaded
    JSON plan. A plan missing required fields raises ValueError
    BEFORE the run starts — catches malformed plans cheaply."""
    import json

    bad_plan = tmp_path / "bad.json"
    bad_plan.write_text(json.dumps([{"type": "navigate"}]))  # missing intent

    rt = _runtime()
    with pytest.raises(ValueError, match=r"missing required field 'intent'"):
        rt._micro_suite_from_path(str(bad_plan), {})


# ── Helpers ────────────────────────────────────────────────────────


def _build_minimal_runner(
    *,
    final_status: str = "completed",
    halt_reason: str = "",
):
    """Minimal MicroPlanRunner-shaped object for build_micro_result.

    The function reads many attributes; we wire only the ones it
    actually consumes for the terminal_status path."""
    from unittest.mock import MagicMock

    runner = MagicMock()
    runner._final_status = final_status
    runner._final_halt_reason = halt_reason
    runner._final_costs = {}
    runner._successful_lead_data = lambda _: []
    runner._lead_key = lambda x: id(x)
    runner._lead_has_phone = lambda x: False
    runner.dynamic_verification_report = lambda **kw: {
        "status": kw.get("status", final_status),
        "verdict": "ok",
        "totals": {}, "checks": [],
    }
    runner.time_meter = None
    return runner
