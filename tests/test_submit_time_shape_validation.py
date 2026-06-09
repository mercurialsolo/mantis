"""Submit-time shape validation — reject (cua_model, plan_shape) mismatches.

Pre-this PR, submitting a `task_suite._micro_plan` with `cua_model=claude`
produced HTTP 200 + a silent 0/0/0 success (Claude executor reads
`tasks[]`, saw zero tasks). Pre-this PR, submitting `task_suite.tasks`
with `cua_model=holo3` produced the same silent zero (Holo3 reads
`_micro_plan`, saw none). Both surface as `Tasks: 0` in Modal logs.

This module pins the 400-at-submit behavior so neither failure mode
recurs.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mantis_agent.api_schemas import PredictRequest


# ── Canonical pairings pass ───────────────────────────────────────


def test_claude_with_tasks_array_validates():
    """The canonical claude shape: `tasks` array on task_suite."""
    req = PredictRequest(
        task_suite={
            "session_name": "x",
            "tasks": [{"task_id": "a", "intent": "do x"}],
        },
        cua_model="claude",
    )
    assert req.task_suite is not None


def test_holo3_with_micro_plan_validates():
    """The canonical holo3 shape: `_micro_plan` list on task_suite."""
    req = PredictRequest(
        task_suite={
            "session_name": "x",
            "_micro_plan": [{"intent": "Go", "type": "navigate"}],
        },
        cua_model="holo3",
    )
    assert req.task_suite is not None


def test_fara_with_micro_plan_validates():
    PredictRequest(
        task_suite={"_micro_plan": [{"intent": "x", "type": "navigate"}]},
        cua_model="fara",
    )


def test_evocua_with_micro_plan_validates():
    PredictRequest(
        task_suite={"_micro_plan": [{"intent": "x", "type": "navigate"}]},
        cua_model="evocua-8b",
    )


# ── Mismatches reject ────────────────────────────────────────────


def test_claude_with_micro_plan_only_rejects():
    """Real production bug — the trap that fired during #785 verification.
    Claude executor reads tasks[], saw nothing because the payload
    only had _micro_plan."""
    with pytest.raises(ValidationError) as exc_info:
        PredictRequest(
            task_suite={
                "_micro_plan": [{"intent": "Go", "type": "navigate"}],
            },
            cua_model="claude",
        )
    msg = str(exc_info.value)
    assert "cua_model='claude' requires task_suite.tasks" in msg
    assert "cua_model='holo3'" in msg, "should suggest the holo3 fix"


def test_holo3_with_tasks_only_rejects():
    """The reverse trap — holo3 sees no _micro_plan."""
    with pytest.raises(ValidationError) as exc_info:
        PredictRequest(
            task_suite={
                "tasks": [{"task_id": "a", "intent": "x"}],
            },
            cua_model="holo3",
        )
    msg = str(exc_info.value)
    assert "cua_model='holo3'" in msg
    assert "tasks[]" in msg
    assert "cua_model='claude'" in msg, "should suggest the claude fix"


def test_holo3_with_neither_rejects():
    """Empty task_suite shape — most common decomposer-output error."""
    with pytest.raises(ValidationError) as exc_info:
        PredictRequest(
            task_suite={"session_name": "x"},
            cua_model="holo3",
        )
    msg = str(exc_info.value)
    assert "cua_model='holo3'" in msg
    assert "_micro_plan" in msg
    assert "build_micro_suite" in msg, "should suggest the canonical helper"


def test_evocua_with_neither_rejects():
    """All micro-plan executors share the same error."""
    with pytest.raises(ValidationError) as exc_info:
        PredictRequest(
            task_suite={"session_name": "x"},
            cua_model="evocua-32b",
        )
    assert "cua_model='evocua-32b'" in str(exc_info.value)


# ── Edge cases ──────────────────────────────────────────────────


def test_no_cua_model_does_not_enforce_shape():
    """Absent cua_model = server-side default — no validation. Lets
    operators / older clients keep submitting without the field."""
    req = PredictRequest(
        task_suite={"_micro_plan": [{"intent": "x", "type": "navigate"}]},
    )
    assert req.task_suite is not None


def test_unknown_cua_model_does_not_enforce_shape():
    """Future / custom executors not in _MICRO_PLAN_EXECUTORS or
    'claude' fall through — the check is conservative, only firing
    on KNOWN mismatches."""
    req = PredictRequest(
        task_suite={"_micro_plan": [{"intent": "x", "type": "navigate"}]},
        cua_model="my-future-executor",
    )
    assert req.task_suite is not None


def test_plan_text_skips_task_suite_shape_check():
    """plan_text reconstructs the suite server-side later — no shape
    to validate at this layer."""
    req = PredictRequest(plan_text="Go and extract things", cua_model="claude")
    assert req.plan_text == "Go and extract things"


def test_micro_path_skips_task_suite_shape_check():
    """`micro` is a file path — also reconstructed later."""
    req = PredictRequest(
        micro="plans/example/extract.json",
        cua_model="holo3",
    )
    assert req.micro == "plans/example/extract.json"


def test_action_modes_skip_validation_entirely():
    """status / result / cancel / resume / logs don't need a plan
    body; the shape check must not block them."""
    PredictRequest(action="status", run_id="r1")
    PredictRequest(action="result", run_id="r1")
    PredictRequest(action="cancel", run_id="r1")
    PredictRequest(action="resume", run_id="r1", user_input="123")


# ── Case insensitivity ──────────────────────────────────────────


def test_cua_model_case_insensitive():
    """`CLAUDE` and `claude` should behave the same — the check
    normalizes case before matching against the known model set."""
    with pytest.raises(ValidationError, match="requires task_suite.tasks"):
        PredictRequest(
            task_suite={"_micro_plan": [{"intent": "x", "type": "navigate"}]},
            cua_model="CLAUDE",
        )


def test_cua_model_strip_whitespace():
    """Leading / trailing whitespace shouldn't bypass the check."""
    with pytest.raises(ValidationError, match="cua_model="):
        PredictRequest(
            task_suite={"_micro_plan": [{"intent": "x", "type": "navigate"}]},
            cua_model="  claude  ",
        )
