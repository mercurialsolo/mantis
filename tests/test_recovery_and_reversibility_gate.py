"""Tests for #483 (typed RecoveryDecision) + #482 (reversibility gate wiring).

Covers:

* The verdict → decision adapter: every (kind, attempt, required)
  combination maps to a deterministic ``RecoveryDecision``.
* The executor's ``_stamp_recovery_decision`` populates the field
  from the runner's per-step failure history.
* ``pack_step`` surfaces both ``recovery_decision`` and
  ``preview_gate`` on every step where they're set; absent fields
  stay out so legacy callers don't see breaking key additions.
* The canonical event emitter forwards ``recovery_decision`` through
  to the trajectory event.
* The pre-execution reversibility gate hook is a no-op when:
  env disabled / no verifier wired / non-IRREVERSIBLE action / no
  screenshot. When enabled + verifier + IRREVERSIBLE, runs the
  preview gate and stashes the result on the runner. The post-step
  attach helper moves it onto the StepResult and clears the stash.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from mantis_agent.cua_contracts import (
    DEFAULT_RETRY_BUDGET,
    JSONL_FILENAME,
    RecoveryDecision,
    TrajectoryEmitter,
    Verdict,
    VerdictKind,
    decide_recovery,
)
from mantis_agent.cua_contracts.types import SCHEMA_VERSION
from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.gym.preview_gate import PreviewResult
from mantis_agent.gym.result_payload import pack_step
from mantis_agent.gym.run_executor import (
    _attach_preview_result,
    _maybe_run_reversibility_gate,
    _stamp_recovery_decision,
    _stamp_verdict,
)
from mantis_agent.plan_decomposer import MicroIntent


# ── Helpers ─────────────────────────────────────────────────────────────


def _ok_result(index: int = 0) -> StepResult:
    return StepResult(
        step_index=index, intent="x", success=True,
        data="ok", duration=1.0,
    )


def _fail_result(
    *, index: int = 0, failure_class: str = "selector_miss",
) -> StepResult:
    return StepResult(
        step_index=index, intent="x", success=False,
        data="fail", duration=1.0, failure_class=failure_class,
    )


def _intent(step_type: str = "click", required: bool = False) -> MicroIntent:
    return MicroIntent(intent="x", type=step_type, required=required)


def _verdict(kind: VerdictKind, reason: str = "") -> Verdict:
    return Verdict(
        schema_version=SCHEMA_VERSION, kind=kind,
        reason=reason or ("" if kind is VerdictKind.OK else "unknown"),
        evidence="", confidence=1.0,
    )


# ── #483: verdict → decision adapter ───────────────────────────────────


def test_decide_recovery_ok_always_advances() -> None:
    """OK verdicts advance regardless of attempt count or required."""
    for attempt in (0, 1, 5):
        for required in (False, True):
            d = decide_recovery(
                _verdict(VerdictKind.OK),
                attempt_index=attempt, required=required,
            )
            assert d is RecoveryDecision.ADVANCE


def test_decide_recovery_non_recoverable_always_terminates() -> None:
    """NON_RECOVERABLE verdicts terminate regardless of context —
    no retry / replan would converge."""
    for attempt in (0, 1, 5):
        for required in (False, True):
            d = decide_recovery(
                _verdict(VerdictKind.NON_RECOVERABLE, reason="cf_challenge"),
                attempt_index=attempt, required=required,
            )
            assert d is RecoveryDecision.TERMINATE


def test_decide_recovery_recoverable_in_budget_retries() -> None:
    """First few attempts on a recoverable verdict get a retry —
    IntentRewriter / agentic_recovery / preview-gate hints get
    another shot."""
    v = _verdict(VerdictKind.RECOVERABLE, reason="selector_miss")
    for attempt in range(DEFAULT_RETRY_BUDGET - 1):
        assert decide_recovery(v, attempt_index=attempt) is RecoveryDecision.RETRY


def test_decide_recovery_recoverable_exhausted_required_terminates() -> None:
    """A required step that's exhausted its retry budget terminates
    — mirrors the existing ``REQUIRED step failed after N
    retries — HALTING`` runner behaviour."""
    v = _verdict(VerdictKind.RECOVERABLE, reason="brain_loop_exhausted")
    d = decide_recovery(
        v, attempt_index=DEFAULT_RETRY_BUDGET, required=True,
    )
    assert d is RecoveryDecision.TERMINATE


def test_decide_recovery_recoverable_exhausted_not_required_advances() -> None:
    """A non-required step that's exhausted its budget advances —
    matches the existing runner's skip-past-non-required behaviour."""
    v = _verdict(VerdictKind.RECOVERABLE, reason="selector_miss")
    d = decide_recovery(
        v, attempt_index=DEFAULT_RETRY_BUDGET, required=False,
    )
    assert d is RecoveryDecision.ADVANCE


def test_decide_recovery_respects_custom_retry_budget() -> None:
    """The budget arg is honoured — handlers / step types that
    deserve a different budget can pass their own."""
    v = _verdict(VerdictKind.RECOVERABLE, reason="x")
    # Budget of 1 means no retry — the first failure exhausts.
    assert decide_recovery(v, attempt_index=0, required=True, retry_budget=1) is RecoveryDecision.TERMINATE
    # Budget of 5 means attempts 0..3 retry, 4 exhausts.
    assert decide_recovery(v, attempt_index=3, required=True, retry_budget=5) is RecoveryDecision.RETRY
    assert decide_recovery(v, attempt_index=4, required=True, retry_budget=5) is RecoveryDecision.TERMINATE


# ── #483: executor stamps the recovery decision ───────────────────────


def test_stamp_recovery_decision_fills_field_from_verdict() -> None:
    """After ``_stamp_verdict`` populates the verdict,
    ``_stamp_recovery_decision`` derives + stamps the decision so
    pack_step / emit hooks see both."""
    r = _ok_result()
    _stamp_verdict(r)
    assert r.recovery_decision is None  # precondition

    class _Runner:
        _step_failure_history = {}

    _stamp_recovery_decision(_Runner(), _intent(), r, step_index=0)
    assert r.recovery_decision is RecoveryDecision.ADVANCE


def test_stamp_recovery_decision_preserves_handler_stamp() -> None:
    """A handler that already stamped a richer decision (e.g. an
    explicit ROLLBACK) wins — adapter only fills when None."""
    r = _ok_result()
    _stamp_verdict(r)
    r.recovery_decision = RecoveryDecision.ROLLBACK

    class _Runner:
        _step_failure_history = {}

    _stamp_recovery_decision(_Runner(), _intent(), r, step_index=0)
    assert r.recovery_decision is RecoveryDecision.ROLLBACK


def test_stamp_recovery_decision_terminates_required_at_budget() -> None:
    """Failed required step at attempt budget → TERMINATE. Tracks
    the existing 'REQUIRED step failed after 2 retries — HALTING'
    runner behaviour."""
    r = _fail_result(failure_class="brain_loop_exhausted")
    _stamp_verdict(r)
    # Two prior attempts already in history — this third one exhausts.
    class _Runner:
        _step_failure_history = {0: ["attempt 1", "attempt 2"]}

    _stamp_recovery_decision(_Runner(), _intent(required=True), r, step_index=0)
    assert r.recovery_decision is RecoveryDecision.TERMINATE


def test_stamp_recovery_decision_no_op_when_verdict_missing() -> None:
    """If the verdict stamp got skipped (test bypass), the recovery
    stamp doesn't make one up — it just leaves the field None."""
    r = _ok_result()
    assert r.verdict is None

    class _Runner:
        _step_failure_history = {}

    _stamp_recovery_decision(_Runner(), _intent(), r, step_index=0)
    assert r.recovery_decision is None


# ── #483: result_payload surfaces the typed recovery decision ──────────


def test_pack_step_surfaces_recovery_decision_on_success() -> None:
    r = _ok_result()
    r.verdict = _verdict(VerdictKind.OK)
    r.recovery_decision = RecoveryDecision.ADVANCE
    out = pack_step(r)
    assert out["recovery_decision"] == "advance"


def test_pack_step_surfaces_recovery_decision_on_failure() -> None:
    r = _fail_result(failure_class="cf_challenge")
    r.verdict = _verdict(VerdictKind.NON_RECOVERABLE, reason="cf_challenge")
    r.recovery_decision = RecoveryDecision.TERMINATE
    out = pack_step(r)
    assert out["recovery_decision"] == "terminate"
    # Verdict is also in the payload (regression — the recovery
    # field shouldn't suppress the verdict surfacing).
    assert out["verdict"]["kind"] == "non_recoverable"


def test_pack_step_omits_recovery_decision_when_none() -> None:
    r = _ok_result()
    assert r.recovery_decision is None
    out = pack_step(r)
    assert "recovery_decision" not in out


# ── #483: canonical event carries the decision ─────────────────────────


def test_emitter_forwards_recovery_decision(tmp_path: Path) -> None:
    emitter = TrajectoryEmitter(run_id="r1", store_dir=str(tmp_path))
    r = _fail_result(failure_class="selector_miss")
    _stamp_verdict(r)
    r.recovery_decision = RecoveryDecision.RETRY
    emitter.emit(_intent(), r)

    record = json.loads(
        (tmp_path / JSONL_FILENAME).read_text(encoding="utf-8").splitlines()[0],
    )
    assert record["recovery_decision"] == "retry"


def test_emitter_emits_null_recovery_decision_when_unstamped(tmp_path: Path) -> None:
    """Callers that bypass the executor stamp leave the field None —
    the event still emits cleanly (validator doesn't require it)."""
    emitter = TrajectoryEmitter(run_id="r1", store_dir=str(tmp_path))
    r = _ok_result()
    _stamp_verdict(r)
    emitter.emit(_intent(), r)
    record = json.loads(
        (tmp_path / JSONL_FILENAME).read_text(encoding="utf-8").splitlines()[0],
    )
    assert record["recovery_decision"] is None


# ── #482: reversibility gate wiring (advisory mode) ────────────────────


def _make_runner_with_verifier(
    verifier=None, *, screenshot_returns=object(),
):
    """Build a minimal runner that has the pre-step gate hook's
    prerequisites: env.screenshot() + _preview_verifier."""
    env = MagicMock()
    env.screenshot = MagicMock(return_value=screenshot_returns)

    class _Runner:
        pass

    runner = _Runner()
    runner.env = env  # type: ignore[attr-defined]
    runner._preview_verifier = verifier  # type: ignore[attr-defined]
    return runner


def test_gate_hook_no_op_when_env_disabled(monkeypatch) -> None:
    monkeypatch.delenv("MANTIS_PREVIEW_GATE", raising=False)
    runner = _make_runner_with_verifier(verifier=lambda **_kw: (True, 0.9, "ok"))
    _maybe_run_reversibility_gate(runner, _intent("submit"), 0)
    assert getattr(runner, "_latest_preview_result", None) is None
    # env.screenshot must NOT be called — pre-screenshot is the
    # cheap thing the gate skips when disabled.
    runner.env.screenshot.assert_not_called()


def test_gate_hook_no_op_when_no_verifier_wired(monkeypatch) -> None:
    monkeypatch.setenv("MANTIS_PREVIEW_GATE", "on")
    runner = _make_runner_with_verifier(verifier=None)
    _maybe_run_reversibility_gate(runner, _intent("submit"), 0)
    assert getattr(runner, "_latest_preview_result", None) is None


def test_gate_hook_no_op_for_reversible_actions(monkeypatch) -> None:
    monkeypatch.setenv("MANTIS_PREVIEW_GATE", "on")
    verifier = MagicMock(return_value=(True, 0.9, "ok"))
    runner = _make_runner_with_verifier(verifier=verifier)
    _maybe_run_reversibility_gate(runner, _intent("click"), 0)
    # Click is REVERSIBLE — the gate's evaluate path returns "skipped"
    # without calling the verifier.
    verifier.assert_not_called()
    # Stash gets a "skipped" PreviewResult since evaluate returned one.
    stashed = getattr(runner, "_latest_preview_result", None)
    assert stashed is not None
    assert stashed.passed is True
    assert stashed.reason == "skipped"


def test_gate_hook_fires_for_irreversible_actions(monkeypatch) -> None:
    monkeypatch.setenv("MANTIS_PREVIEW_GATE", "on")
    verifier = MagicMock(return_value=(True, 0.9, "Submit button visible"))
    runner = _make_runner_with_verifier(verifier=verifier)
    _maybe_run_reversibility_gate(
        runner, _intent("submit", required=True), step_index=5,
    )
    verifier.assert_called_once()
    stashed = getattr(runner, "_latest_preview_result", None)
    assert stashed is not None
    assert stashed.passed is True
    assert stashed.reason == "verifier_accepted"


def test_gate_hook_advisory_logs_rejection_without_blocking(
    monkeypatch, caplog,
) -> None:
    """v1 contract: a rejected preview is logged at WARNING but does
    NOT block dispatch (the hook returns; the handler still runs).
    Operators can grep prod logs for the advisory line to assess
    impact before flipping the blocking-mode env."""
    monkeypatch.setenv("MANTIS_PREVIEW_GATE", "on")
    verifier = MagicMock(return_value=(False, 0.2, "checkbox not Submit"))
    runner = _make_runner_with_verifier(verifier=verifier)
    with caplog.at_level("WARNING"):
        _maybe_run_reversibility_gate(runner, _intent("submit"), 0)
    assert any("ADVISORY" in r.message for r in caplog.records)
    stashed = getattr(runner, "_latest_preview_result", None)
    assert stashed.passed is False
    assert stashed.reason == "verifier_rejected"


def test_gate_hook_swallows_verifier_exceptions(monkeypatch) -> None:
    """The gate must never crash the runner — a buggy verifier
    raises, the gate catches and returns a fail-closed result."""
    monkeypatch.setenv("MANTIS_PREVIEW_GATE", "on")

    def _raises(**_kw):
        raise RuntimeError("verifier broke")

    runner = _make_runner_with_verifier(verifier=_raises)
    # Must not raise.
    _maybe_run_reversibility_gate(runner, _intent("submit"), 0)
    stashed = getattr(runner, "_latest_preview_result", None)
    assert stashed.passed is False
    assert stashed.reason == "verifier_error"


def test_gate_hook_swallows_screenshot_exceptions(monkeypatch) -> None:
    """env.screenshot() raising must not crash the gate — log + skip."""
    monkeypatch.setenv("MANTIS_PREVIEW_GATE", "on")
    verifier = MagicMock(return_value=(True, 0.9, "ok"))
    runner = _make_runner_with_verifier(verifier=verifier)
    runner.env.screenshot = MagicMock(side_effect=RuntimeError("display gone"))
    _maybe_run_reversibility_gate(runner, _intent("submit"), 0)
    # No result stashed — but no exception propagated either.
    assert getattr(runner, "_latest_preview_result", None) is None
    verifier.assert_not_called()


# ── #482: attach helper moves result onto StepResult ───────────────────


def test_attach_preview_result_moves_from_stash_to_step_result() -> None:
    class _Runner:
        _latest_preview_result = PreviewResult(
            passed=False, confidence=0.3,
            reason="verifier_rejected", evidence="not the Submit",
        )

    runner = _Runner()
    r = _fail_result()
    _attach_preview_result(runner, r)
    assert r.preview_result is not None
    assert r.preview_result.passed is False
    assert r.preview_result.reason == "verifier_rejected"
    # Stash cleared so next step can't inherit it.
    assert runner._latest_preview_result is None


def test_attach_preview_result_noop_when_no_stash() -> None:
    """A step that didn't gate (REVERSIBLE / env-disabled) shouldn't
    have anything attached — preview_result stays None."""
    class _Runner:
        pass

    runner = _Runner()
    r = _ok_result()
    _attach_preview_result(runner, r)
    assert r.preview_result is None


# ── #482: pack_step surfaces preview_gate ──────────────────────────────


def test_pack_step_surfaces_preview_gate_result() -> None:
    r = _ok_result()
    r.preview_result = PreviewResult(
        passed=True, confidence=0.92,
        reason="verifier_accepted",
        evidence="Submit button matches plan label",
    )
    out = pack_step(r)
    assert out["preview_gate"] == {
        "passed": True,
        "confidence": 0.92,
        "reason": "verifier_accepted",
        "evidence": "Submit button matches plan label",
    }


def test_pack_step_omits_preview_gate_when_none() -> None:
    out = pack_step(_ok_result())
    assert "preview_gate" not in out
