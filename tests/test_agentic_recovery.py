"""Tests for the agentic failure-recovery loop (#224 follow-up).

Surfaced by the staff-crm post-PR-#235 rerun: the runner made it
through 10/13 steps end-to-end (login → leads → filter → click
lead → edit → set Industry Vertical → ...) but failed at step 10
``Click the Update Lead button`` with ``form_target_not_found`` —
the literal label "Update Lead" doesn't exist on the page; the
real button is labelled "Save".

The recovery loop closes this gap. When a required step terminally
fails (after retries + handler escalation), Claude analyses the
failure (step + last screenshot + failure data) and decides
whether to:

- ``add_hint`` — append a clarifying instruction to the next
  retry's search prompt
- ``edit_step`` — mutate the failed step (intent / type / params)
- ``insert_steps`` — splice helper steps before the failed step
- ``halt`` — surrender (current behavior)

Bounded by per-step + per-run budgets so a pathological page can't
spend the whole run on recovery alone.

These tests pin:

- :func:`analyse_failure_and_recover` — the Claude tool_use call
  shape, response coercion, fallback paths
- :class:`RecoveryDecision` — the four-mode dataclass
- :func:`splice_inserted_steps` — plan-splice with loop_target
  renumbering
- ``StepRecoveryPolicy._try_agentic_recovery`` integration —
  applies each mode correctly, enforces budgets, falls through to
  halt on missing key / unknown mode
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mantis_agent.agentic_recovery import (
    DEFAULT_MAX_RECOVERIES_PER_RUN,
    DEFAULT_MAX_RECOVERIES_PER_STEP,
    _build_decision,
    analyse_failure_and_recover,
    splice_inserted_steps,
)
from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.gym.step_recovery import StepRecoveryPolicy
from mantis_agent.plan_decomposer import MicroIntent, MicroPlan


def _tool_response(payload: dict) -> MagicMock:
    """Build a stubbed Anthropic /v1/messages response in tool_use shape."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "content": [
            {
                "type": "tool_use",
                "name": "record_recovery",
                "input": payload,
            }
        ]
    }
    return resp


# ── analyse_failure_and_recover ─────────────────────────────────────────


def test_analyse_returns_none_without_api_key(monkeypatch) -> None:
    """No ``ANTHROPIC_API_KEY`` → return None so the caller falls
    through to the legacy halt path."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    step = MicroIntent(intent="Click Save", type="submit")
    with patch("requests.post") as mock_post:
        result = analyse_failure_and_recover(
            step=step, failure_data="form_target_not_found",
            screenshot=None, plan_context=[], attempts=2,
        )
    assert result is None
    mock_post.assert_not_called()


def test_analyse_returns_none_on_api_error() -> None:
    """API non-200 → return None; don't propagate."""
    err = MagicMock(status_code=500, text="upstream timeout")
    step = MicroIntent(intent="Click Save", type="submit")
    with patch("requests.post", return_value=err):
        result = analyse_failure_and_recover(
            step=step, failure_data="form_target_not_found",
            screenshot=None, plan_context=[], attempts=2,
            api_key="k",
        )
    assert result is None


def test_analyse_returns_none_on_missing_tool_block() -> None:
    """200 OK but the response has no tool_use block (model emitted
    prose). Return None rather than crashing."""
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"content": [{"type": "text", "text": "no"}]}
    step = MicroIntent(intent="Click Save", type="submit")
    with patch("requests.post", return_value=resp):
        result = analyse_failure_and_recover(
            step=step, failure_data="form_target_not_found",
            screenshot=None, plan_context=[], attempts=2,
            api_key="k",
        )
    assert result is None


def test_prompt_includes_progress_evidence_guidance_for_halt() -> None:
    """The prompt must teach Claude that lack of visible progress
    across multiple retries is evidence the target may not exist —
    so it picks halt over insert_steps when scrolling/clicking
    haven't moved the page. This was the priority-field gap: Claude
    chose add_hint when halt was the right call because the prompt
    didn't enumerate "page hasn't moved" as a halt signal."""
    from mantis_agent.agentic_recovery import _ANALYSIS_PROMPT_TEMPLATE

    rendered = _ANALYSIS_PROMPT_TEMPLATE.format(
        intent="x", step_type="submit", params="{}",
        failure_data="x", attempts=3, plan_context="",
    )
    # Halt mode mentions progress evidence.
    assert "PROGRESS EVIDENCE" in rendered or "page state" in rendered.lower()
    # Explicit guidance to prefer halt when retries didn't progress.
    assert "halt" in rendered.lower()
    assert "loop" in rendered.lower() or "didn't" in rendered.lower() or "did not" in rendered.lower()


def test_analyse_calls_with_correct_tool_schema() -> None:
    """The ``tool_choice`` must force the ``record_recovery`` tool and
    the input_schema must enumerate the four recovery modes."""
    captured: dict = {}

    def _capture(url, **kwargs):
        captured["json"] = kwargs.get("json")
        return _tool_response({"mode": "halt", "reasoning": "x"})

    step = MicroIntent(intent="Click Save", type="submit")
    with patch("requests.post", side_effect=_capture):
        analyse_failure_and_recover(
            step=step, failure_data="x",
            screenshot=None, plan_context=[], attempts=1,
            api_key="k",
        )

    body = captured["json"]
    assert body["tool_choice"] == {"type": "tool", "name": "record_recovery"}
    schema = body["tools"][0]["input_schema"]
    assert set(schema["properties"]["mode"]["enum"]) == {
        "add_hint", "edit_step", "insert_steps", "halt",
    }
    assert "mode" in schema["required"]
    assert "reasoning" in schema["required"]


# ── RecoveryDecision coercion ───────────────────────────────────────────


def test_build_decision_add_hint() -> None:
    decision = _build_decision({
        "mode": "add_hint",
        "reasoning": "Button labelled Save",
        "hint": "the button is labeled 'Save', not 'Update Lead'",
    })
    assert decision.mode == "add_hint"
    assert "Save" in decision.hint


def test_build_decision_edit_step() -> None:
    decision = _build_decision({
        "mode": "edit_step",
        "reasoning": "Wrong step type",
        "edited_step": {
            "type": "select_option",
            "params": {"label": "Status", "option_label": "Contacted"},
        },
    })
    assert decision.mode == "edit_step"
    assert decision.edited_step["type"] == "select_option"
    assert decision.edited_step["params"]["option_label"] == "Contacted"


def test_build_decision_insert_steps_drops_malformed_entries() -> None:
    """Defensive: tool_use schema requires intent + type but we
    drop entries that violate that rather than letting them through."""
    decision = _build_decision({
        "mode": "insert_steps",
        "reasoning": "Need to dismiss modal first",
        "inserted_steps": [
            {"intent": "Press Escape to dismiss modal", "type": "scroll"},
            {"intent": "", "type": "scroll"},  # empty intent — drop
            {"type": "scroll"},  # missing intent — drop
            "not a dict",  # wrong shape — drop
        ],
    })
    assert decision.mode == "insert_steps"
    assert len(decision.inserted_steps) == 1
    assert "Escape" in decision.inserted_steps[0]["intent"]


def test_build_decision_halt() -> None:
    decision = _build_decision({"mode": "halt", "reasoning": "no qualified leads"})
    assert decision.mode == "halt"
    assert decision.hint == ""
    assert decision.edited_step == {}
    assert decision.inserted_steps == []


def test_build_decision_treats_unknown_mode_as_halt() -> None:
    """LLM occasionally produces a typo despite the enum schema —
    coerce defensively rather than letting it propagate."""
    decision = _build_decision({"mode": "wat", "reasoning": "?"})
    assert decision.mode == "halt"


# ── splice_inserted_steps ───────────────────────────────────────────────


def test_splice_inserts_at_index() -> None:
    """Splicing 2 steps before index 3 must produce a list with
    the originals at indices 0,1,2 then inserted at 3,4 then
    originals 3,4,... pushed to 5,6,..."""
    a = MicroIntent(intent="A", type="navigate")
    b = MicroIntent(intent="B", type="click")
    c = MicroIntent(intent="C", type="submit")
    d = MicroIntent(intent="D", type="extract_data")
    e = MicroIntent(intent="E", type="extract_data")
    helper1 = MicroIntent(intent="helper1", type="scroll")
    helper2 = MicroIntent(intent="helper2", type="scroll")

    new = splice_inserted_steps([a, b, c, d, e], 3, [helper1, helper2])

    assert [s.intent for s in new] == ["A", "B", "C", "helper1", "helper2", "D", "E"]


def test_splice_renumbers_loop_targets_pointing_at_or_after_insertion() -> None:
    """Loop steps reference target indices absolutely — splicing
    helpers in must shift any loop_target >= insertion_index by
    +len(inserted)."""
    nav = MicroIntent(intent="Open", type="navigate")
    click = MicroIntent(intent="Click", type="click")
    extract = MicroIntent(intent="Extract", type="extract_data")
    loop = MicroIntent(
        intent="Loop", type="loop", loop_target=1, loop_count=5,
    )
    helper = MicroIntent(intent="helper", type="scroll")

    # Insert 1 helper at index 1 — loop_target=1 must become 2.
    new = splice_inserted_steps([nav, click, extract, loop], 1, [helper])

    loop_after = [s for s in new if s.type == "loop"][0]
    assert loop_after.loop_target == 2  # was 1, shifted by +1


def test_splice_does_not_renumber_targets_before_insertion() -> None:
    """A loop pointing at an earlier-than-insertion step must not
    shift — only targets >= insertion index move."""
    a = MicroIntent(intent="A", type="navigate")
    loop = MicroIntent(
        intent="Loop back", type="loop", loop_target=0, loop_count=3,
    )
    c = MicroIntent(intent="C", type="extract_data")
    helper = MicroIntent(intent="helper", type="scroll")

    # Insert at index 2 — loop_target=0 stays 0.
    new = splice_inserted_steps([a, loop, c], 2, [helper])
    loop_after = [s for s in new if s.type == "loop"][0]
    assert loop_after.loop_target == 0


def test_splice_with_empty_inserted_is_no_op() -> None:
    a = MicroIntent(intent="A", type="navigate")
    b = MicroIntent(intent="B", type="extract_data")
    new = splice_inserted_steps([a, b], 1, [])
    assert [s.intent for s in new] == ["A", "B"]


# ── StepRecoveryPolicy._try_agentic_recovery integration ────────────────


@pytest.fixture(autouse=True)
def _ensure_api_key(monkeypatch):
    """Most policy-integration tests need ``ANTHROPIC_API_KEY`` set so
    the recovery call reaches the (mocked) Claude path. The two
    no-key tests opt out by deleting the env var inside the test."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def _make_runner_with_recovery_state() -> MagicMock:
    runner = MagicMock()
    runner._recovery_attempts_per_step = {}
    runner._total_recovery_attempts = 0
    runner._recovery_hints = {}
    runner._safe_screenshot.return_value = None
    return runner


def _make_plan() -> MicroPlan:
    return MicroPlan(steps=[
        MicroIntent(intent="A", type="navigate"),
        MicroIntent(intent="B", type="fill_field"),
        MicroIntent(intent="Click Update Lead", type="submit",
                    params={"label": "Update Lead"}),
    ])


def test_recovery_returns_none_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    runner = _make_runner_with_recovery_state()
    policy = StepRecoveryPolicy(parent=runner)
    plan = _make_plan()
    step = plan.steps[2]
    result = StepResult(
        step_index=2, intent=step.intent, success=False,
        data="form_target_not_found",
    )
    outcome = policy._try_agentic_recovery(
        step=step, step_result=result, step_index=2, plan=plan,
        step_retry_counts={2: 2}, attempts=2,
    )
    assert outcome is None  # legacy halt path


def test_recovery_applies_add_hint_mode() -> None:
    """Successful add_hint recovery: the hint lands on the runner's
    ``_recovery_hints`` and the outcome re-runs the same step."""
    runner = _make_runner_with_recovery_state()
    policy = StepRecoveryPolicy(parent=runner)
    plan = _make_plan()
    step = plan.steps[2]
    result = StepResult(
        step_index=2, intent=step.intent, success=False,
        data="form_target_not_found",
    )
    payload = {
        "mode": "add_hint",
        "reasoning": "Real button is labeled Save",
        "hint": "The button is labeled 'Save', not 'Update Lead'",
    }
    retry_counts = {2: 2}
    with patch("requests.post", return_value=_tool_response(payload)):
        outcome = policy._try_agentic_recovery(
            step=step, step_result=result, step_index=2, plan=plan,
            step_retry_counts=retry_counts, attempts=2,
        )

    assert outcome is not None
    assert outcome.halt is False
    assert outcome.step_index == 2
    assert "recovery_hint" in outcome.halt_reason
    # Hint stashed for the next form-handler invocation to read.
    assert "Save" in runner._recovery_hints[2][0]
    # Retry count reset so the step gets fresh attempts WITH the hint.
    assert retry_counts[2] == 0
    # Recovery budget bumped.
    assert runner._recovery_attempts_per_step[2] == 1
    assert runner._total_recovery_attempts == 1


def test_recovery_applies_edit_step_mode() -> None:
    """edit_step replaces the failed step's intent / type / params
    in place; the runner re-runs the (now edited) step."""
    runner = _make_runner_with_recovery_state()
    policy = StepRecoveryPolicy(parent=runner)
    plan = _make_plan()
    step = plan.steps[2]
    result = StepResult(
        step_index=2, intent=step.intent, success=False,
        data="form_target_not_found",
    )
    payload = {
        "mode": "edit_step",
        "reasoning": "Different label",
        "edited_step": {
            "params": {"label": "Save", "aliases": ["Update", "Submit"]},
        },
    }
    with patch("requests.post", return_value=_tool_response(payload)):
        outcome = policy._try_agentic_recovery(
            step=step, step_result=result, step_index=2, plan=plan,
            step_retry_counts={2: 2}, attempts=2,
        )

    assert outcome is not None
    assert outcome.halt is False
    assert "recovery_edit" in outcome.halt_reason
    # Step mutated in place.
    assert plan.steps[2].params["label"] == "Save"
    assert "Update" in plan.steps[2].params["aliases"]


def test_recovery_applies_insert_steps_mode() -> None:
    """insert_steps splices helper steps BEFORE the failed step
    via splice_inserted_steps. Plan length grows; the failed step
    still re-runs but now after the helpers."""
    runner = _make_runner_with_recovery_state()
    policy = StepRecoveryPolicy(parent=runner)
    plan = _make_plan()
    step = plan.steps[2]
    result = StepResult(
        step_index=2, intent=step.intent, success=False,
        data="form_target_not_found",
    )
    payload = {
        "mode": "insert_steps",
        "reasoning": "Need to dismiss modal first",
        "inserted_steps": [
            {"intent": "Press Escape to dismiss any open dialog",
             "type": "scroll", "params": {}},
        ],
    }
    with patch("requests.post", return_value=_tool_response(payload)):
        outcome = policy._try_agentic_recovery(
            step=step, step_result=result, step_index=2, plan=plan,
            step_retry_counts={2: 2}, attempts=2,
        )

    assert outcome is not None
    assert outcome.halt is False
    # Plan now has 4 steps (3 + 1 helper).
    assert len(plan.steps) == 4
    # Helper is at index 2; the original failed step is at 3.
    assert plan.steps[2].intent.startswith("Press Escape")
    assert plan.steps[3].params.get("label") == "Update Lead"


def test_recovery_halt_mode_falls_through() -> None:
    """When the LLM picks ``halt`` recovery, the policy returns None
    so the caller surfaces the legacy halt path."""
    runner = _make_runner_with_recovery_state()
    policy = StepRecoveryPolicy(parent=runner)
    plan = _make_plan()
    step = plan.steps[2]
    result = StepResult(
        step_index=2, intent=step.intent, success=False,
        data="form_target_not_found",
    )
    payload = {"mode": "halt", "reasoning": "Target genuinely missing"}
    with patch("requests.post", return_value=_tool_response(payload)):
        outcome = policy._try_agentic_recovery(
            step=step, step_result=result, step_index=2, plan=plan,
            step_retry_counts={2: 2}, attempts=2,
        )
    assert outcome is None  # legacy halt
    # But the budget was still bumped — the call happened.
    assert runner._total_recovery_attempts == 1


def test_recovery_per_step_budget_blocks_after_max() -> None:
    """A step that has already used its per-step recovery budget
    skips the LLM call and returns None directly. Prevents
    infinite recovery loops on a single bad step."""
    runner = _make_runner_with_recovery_state()
    runner._recovery_attempts_per_step[2] = DEFAULT_MAX_RECOVERIES_PER_STEP
    policy = StepRecoveryPolicy(parent=runner)
    plan = _make_plan()
    step = plan.steps[2]
    result = StepResult(
        step_index=2, intent=step.intent, success=False,
        data="form_target_not_found",
    )
    with patch("requests.post") as mock_post:
        outcome = policy._try_agentic_recovery(
            step=step, step_result=result, step_index=2, plan=plan,
            step_retry_counts={2: 2}, attempts=2,
        )
    assert outcome is None
    mock_post.assert_not_called()


def test_recovery_per_run_budget_blocks_after_max() -> None:
    """Total recoveries across all steps capped — once exhausted,
    no further recovery attempts."""
    runner = _make_runner_with_recovery_state()
    runner._total_recovery_attempts = DEFAULT_MAX_RECOVERIES_PER_RUN
    policy = StepRecoveryPolicy(parent=runner)
    plan = _make_plan()
    step = plan.steps[2]
    result = StepResult(
        step_index=2, intent=step.intent, success=False, data="x",
    )
    with patch("requests.post") as mock_post:
        outcome = policy._try_agentic_recovery(
            step=step, step_result=result, step_index=2, plan=plan,
            step_retry_counts={2: 2}, attempts=2,
        )
    assert outcome is None
    mock_post.assert_not_called()
