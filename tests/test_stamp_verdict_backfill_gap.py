"""Gap 3 — backfill failure_reason on a stale-optimistic verdict.

``_stamp_verdict`` used to early-return whenever ``step_result.verdict``
was already set. Handlers that pre-stamp an *optimistic* ``OK`` verdict
(``reason=""``) before the outcome is known (request_user_input,
shadow) therefore masked the real failure: when the step later failed,
the empty-reason OK verdict survived, and the run-level
``failure_reason`` Augur derives from it showed ``null`` / ``unknown``
even though ``failure_class`` was populated.

The fix re-projects from ``failure_class`` in exactly one case —
existing verdict is ``OK`` but ``success`` is False. Every other
precedence rule is preserved:

* No verdict yet → project normally (unchanged).
* Existing OK verdict on a *successful* step → keep (happy path).
* Existing *failure* verdict (RECOVERABLE / NON_RECOVERABLE) → keep
  (a handler with richer evidence wins).
"""

from __future__ import annotations

from mantis_agent.cua_contracts import SCHEMA_VERSION, Verdict, VerdictKind
from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.gym.run_executor import _stamp_verdict


def _failed_step(failure_class: str = "no_state_change") -> StepResult:
    return StepResult(
        step_index=0, intent="click", success=False,
        data="click ok, no nav", duration=1.0, failure_class=failure_class,
    )


def _ok_step() -> StepResult:
    return StepResult(
        step_index=0, intent="extract", success=True,
        data="extracted 7 leads", duration=1.0,
    )


def _optimistic_ok_verdict() -> Verdict:
    return Verdict(
        schema_version=SCHEMA_VERSION, kind=VerdictKind.OK,
        reason="", evidence="", confidence=1.0,
    )


# ── the bug this closes ─────────────────────────────────────────────


def test_optimistic_ok_on_failed_step_is_backfilled():
    """Pre-stamped OK verdict + failed step → re-projected so the
    failure_class surfaces as the verdict reason."""
    step = _failed_step("no_state_change")
    step.verdict = _optimistic_ok_verdict()  # handler stamped this early

    _stamp_verdict(step)

    assert step.verdict is not None
    assert step.verdict.kind != VerdictKind.OK
    assert step.verdict.reason == "no_state_change"


def test_backfill_falls_back_to_unknown_when_no_class():
    """Optimistic OK + failed step with no failure_class still gets a
    non-empty reason (the validator requires one on failure verdicts)."""
    step = _failed_step("")
    step.verdict = _optimistic_ok_verdict()

    _stamp_verdict(step)

    assert step.verdict.kind != VerdictKind.OK
    assert step.verdict.reason == "unknown"


# ── precedence rules preserved ──────────────────────────────────────


def test_no_existing_verdict_projects_normally():
    step = _failed_step("selector_miss")
    assert step.verdict is None
    _stamp_verdict(step)
    assert step.verdict is not None
    assert step.verdict.reason == "selector_miss"


def test_ok_verdict_on_successful_step_is_kept():
    """Happy path: optimistic OK that turned out correct is untouched."""
    step = _ok_step()
    v = _optimistic_ok_verdict()
    step.verdict = v
    _stamp_verdict(step)
    assert step.verdict is v  # same object, not re-projected


def test_existing_failure_verdict_wins():
    """A handler that stamped a real failure verdict (richer evidence)
    is not overwritten."""
    step = _failed_step("no_state_change")
    rich = Verdict(
        schema_version=SCHEMA_VERSION, kind=VerdictKind.NON_RECOVERABLE,
        reason="cf_challenge", evidence="cloudflare interstitial detected",
        confidence=0.9,
    )
    step.verdict = rich
    _stamp_verdict(step)
    assert step.verdict is rich
    assert step.verdict.reason == "cf_challenge"
