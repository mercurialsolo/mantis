"""Tests for #530 — verdict.status/score regression on Mantis bundles.

Two-part fix:

1. ``AugurAdapter._build_step_trace`` was reading ``getattr(v_obj,
   "status", "")`` but the Mantis Verdict dataclass field is
   ``kind``, not ``status``. The typo defaulted every step to
   ``status="unknown"`` regardless of outcome.

2. Even after fixing the typo, handlers that optimistically stamp
   ``Verdict(kind=OK, confidence=1.0)`` before failure detection
   left a misleading verdict on later-failed steps. The defensive
   fix derives ``status`` from ``sr.success`` (canonical truth) and
   uses ``verdict.kind`` only to distinguish recoverable from
   non_recoverable on the failure branch. The companion
   ``set_score`` wire in ``run_executor`` was updated to derive
   the score from ``sr.success`` for the same reason.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("augur_sdk")

from mantis_agent.observability.augur import AugurAdapter


def _make_step_result(*, success: bool, kind_value: str,
                     confidence: float = 0.0,
                     skip: bool = False,
                     evidence: str = "",
                     failure_class: str = "",
                     step_index: int = 0):
    """Build a StepResult-shaped MagicMock with a Verdict object."""
    class _Kind:
        def __init__(self, v): self.value = v

    class _V:
        pass
    v = _V()
    v.kind = _Kind(kind_value) if kind_value else ""
    v.reason = failure_class
    v.evidence = evidence
    v.confidence = confidence

    sr = MagicMock(
        step_index=step_index, intent="x", success=success,
        skip=skip, reversed=False, duration=0.1,
        failure_class=failure_class, executor_backend="",
        last_action=None, verdict=v, recovery_decision=None,
        screenshot_png=None, data="", page_title="",
    )
    return sr


def _emit_and_read(tmp_path: Path, sr) -> dict:
    """Drive record_step + close, return the on-disk step dict."""
    a = AugurAdapter(
        run_id="v530", tenant_id="t", session_name="s", out_dir=tmp_path,
    )
    a.record_step(
        step_result=sr,
        started_at="2026-05-20T18:33:56Z",
        ended_at="2026-05-20T18:33:57Z",
    )
    a.close(status="completed")
    # SDK writes the step under steps/<augur_index:04d>.json — Mantis
    # 0-based → Augur 1-based bump, so step_index=20 → 0021.json.
    idx = int(getattr(sr, "step_index", 0)) + 1
    return json.loads((tmp_path / "steps" / f"{idx:04d}.json").read_text())


# ── status mapping ────────────────────────────────────────────────────────


def test_failed_step_with_optimistic_kind_ok_still_maps_to_failed(
    monkeypatch, tmp_path: Path,
):
    """The exact #530 reproducer: a handler pre-stamped
    Verdict(kind=OK) but sr.success=False (the runner detected
    failure later). Adapter must trust sr.success and emit
    status=failed, NOT status=passed."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    sr = _make_step_result(
        success=False, kind_value="ok", confidence=1.0,
        failure_class="no_state_change", step_index=20,
    )
    step = _emit_and_read(tmp_path, sr)
    assert step["verdict"]["status"] == "failed", (
        f"Optimistic kind=OK on a failed step must NOT bypass sr.success "
        f"(got {step['verdict']['status']!r}). This is the #530 regression."
    )


def test_failed_step_with_recoverable_kind_maps_to_recoverable(
    monkeypatch, tmp_path: Path,
):
    """When sr.success=False AND verdict.kind=recoverable, emit
    status=recoverable — the verdict's distinction is honored on
    the failure branch."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    sr = _make_step_result(
        success=False, kind_value="recoverable", confidence=0.5,
        failure_class="selector_miss",
    )
    step = _emit_and_read(tmp_path, sr)
    assert step["verdict"]["status"] == "recoverable"


def test_failed_step_with_non_recoverable_kind_maps_to_failed(
    monkeypatch, tmp_path: Path,
):
    """sr.success=False + kind=non_recoverable → status=failed
    (not 'non_recoverable' — that's a Mantis-internal label)."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    sr = _make_step_result(
        success=False, kind_value="non_recoverable", confidence=0.9,
        failure_class="cf_challenge",
    )
    step = _emit_and_read(tmp_path, sr)
    assert step["verdict"]["status"] == "failed"


def test_successful_step_maps_to_passed(monkeypatch, tmp_path: Path):
    """sr.success=True → status=passed regardless of verdict.kind."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    sr = _make_step_result(success=True, kind_value="ok", confidence=1.0)
    step = _emit_and_read(tmp_path, sr)
    assert step["verdict"]["status"] == "passed"


def test_skipped_step_maps_to_skipped(monkeypatch, tmp_path: Path):
    """sr.skip=True → status=skipped even when success=False."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    sr = _make_step_result(
        success=False, skip=True, kind_value="recoverable", confidence=0.5,
    )
    step = _emit_and_read(tmp_path, sr)
    assert step["verdict"]["status"] == "skipped"


def test_verdict_missing_falls_back_to_success_boolean(
    monkeypatch, tmp_path: Path,
):
    """When sr.verdict is None entirely, use sr.success."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    sr = MagicMock(
        step_index=0, intent="x", success=False,
        skip=False, reversed=False, duration=0.1,
        failure_class="cf_challenge", executor_backend="",
        last_action=None, verdict=None, recovery_decision=None,
        screenshot_png=None, data="", page_title="",
    )
    step = _emit_and_read(tmp_path, sr)
    assert step["verdict"]["status"] == "failed"


# ── evidence_refs population ─────────────────────────────────────────────


def test_verdict_evidence_string_lands_as_evidence_ref(
    monkeypatch, tmp_path: Path,
):
    """Mantis Verdict.evidence is a free-form string the verifier
    writes. Surface it as a single evidence_ref entry instead of
    dropping it (the previous behavior left evidence_refs empty)."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    sr = _make_step_result(
        success=False, kind_value="non_recoverable",
        confidence=0.9, evidence="URL unchanged after click",
        failure_class="no_state_change",
    )
    step = _emit_and_read(tmp_path, sr)
    assert step["verdict"]["evidence_refs"] == ["URL unchanged after click"]


def test_verdict_empty_evidence_leaves_evidence_refs_empty(
    monkeypatch, tmp_path: Path,
):
    """No evidence string → no evidence_refs entries (don't fabricate)."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    sr = _make_step_result(
        success=True, kind_value="ok", confidence=1.0, evidence="",
    )
    step = _emit_and_read(tmp_path, sr)
    assert step["verdict"]["evidence_refs"] == []


# ── reason field still flows correctly ───────────────────────────────────


def test_verdict_reason_threads_from_verdict_reason_when_present(
    monkeypatch, tmp_path: Path,
):
    """Verdict.reason populates verdict.reason on the bundle —
    confirms the field-name fix didn't accidentally break the
    reason path (different field, but adjacent code)."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    sr = _make_step_result(
        success=False, kind_value="non_recoverable", confidence=0.9,
        failure_class="cf_challenge",
    )
    step = _emit_and_read(tmp_path, sr)
    # Verdict.reason mirrors failure_class in _make_step_result.
    assert step["verdict"]["reason"] == "cf_challenge"


# ── set_score wire (run_executor) ────────────────────────────────────────


def test_set_score_zeros_on_failed_step_even_with_optimistic_confidence(
    monkeypatch, tmp_path: Path,
):
    """The companion run_executor wire: a failed step with
    optimistically-stamped Verdict(confidence=1.0) must NOT pass
    1.0 through to set_score. Fix derives score from sr.success."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    from mantis_agent.gym.run_executor import RunExecutor

    augur = AugurAdapter(
        run_id="exec_530", tenant_id="t", session_name="s", out_dir=tmp_path,
    )
    augur_spy = MagicMock(wraps=augur)
    runner = MagicMock()
    runner._augur = augur_spy
    runner._augur_capture_upgraded = False
    runner._healing_events = []
    runner.time_meter = None
    runner._invoke_step_callback = lambda *a, **kw: None
    runner.cost_meter = MagicMock()
    runner.cost_meter.costs = {}

    # The #530 reproducer: failed step + optimistic verdict
    sr = _make_step_result(
        success=False, kind_value="ok", confidence=1.0,
        failure_class="no_state_change", step_index=20,
    )
    sr.reasoning = ""

    executor = RunExecutor.__new__(RunExecutor)
    executor.parent = runner
    executor._emit_augur_step(sr, "2026-05-20T18:33:56Z")
    augur.close(status="completed")

    # set_score must have been called with 0.0, not the 1.0 the
    # verdict.confidence carried.
    augur_spy.set_score.assert_called_once()
    score = augur_spy.set_score.call_args.args[1]
    assert score == 0.0, (
        f"Failed step's set_score must use 0.0 (canonical truth from "
        f"sr.success=False), not the optimistic verdict.confidence "
        f"({sr.verdict.confidence}). Got score={score}."
    )


def test_set_score_uses_05_for_recoverable_failure(
    monkeypatch, tmp_path: Path,
):
    """Failed + verdict.kind=recoverable → score=0.5 (partial signal)."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    from mantis_agent.gym.run_executor import RunExecutor

    augur = AugurAdapter(
        run_id="exec_recoverable", tenant_id="t", session_name="s", out_dir=tmp_path,
    )
    augur_spy = MagicMock(wraps=augur)
    runner = MagicMock()
    runner._augur = augur_spy
    runner._augur_capture_upgraded = False
    runner._healing_events = []
    runner.time_meter = None
    runner._invoke_step_callback = lambda *a, **kw: None
    runner.cost_meter = MagicMock()
    runner.cost_meter.costs = {}

    sr = _make_step_result(
        success=False, kind_value="recoverable", confidence=0.5,
    )
    sr.reasoning = ""

    executor = RunExecutor.__new__(RunExecutor)
    executor.parent = runner
    executor._emit_augur_step(sr, "2026-05-20T18:33:56Z")
    augur.close(status="completed")

    augur_spy.set_score.assert_called_once()
    score = augur_spy.set_score.call_args.args[1]
    assert score == 0.5


def test_set_score_uses_confidence_when_available_on_success(
    monkeypatch, tmp_path: Path,
):
    """Successful step + non-zero verdict.confidence → pass the
    confidence through (the original #524 contract)."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    from mantis_agent.gym.run_executor import RunExecutor

    augur = AugurAdapter(
        run_id="exec_conf", tenant_id="t", session_name="s", out_dir=tmp_path,
    )
    augur_spy = MagicMock(wraps=augur)
    runner = MagicMock()
    runner._augur = augur_spy
    runner._augur_capture_upgraded = False
    runner._healing_events = []
    runner.time_meter = None
    runner._invoke_step_callback = lambda *a, **kw: None
    runner.cost_meter = MagicMock()
    runner.cost_meter.costs = {}

    sr = _make_step_result(success=True, kind_value="ok", confidence=0.83)
    sr.reasoning = ""

    executor = RunExecutor.__new__(RunExecutor)
    executor.parent = runner
    executor._emit_augur_step(sr, "2026-05-20T18:33:56Z")
    augur.close(status="completed")

    augur_spy.set_score.assert_called_once()
    score = augur_spy.set_score.call_args.args[1]
    assert score == 0.83


def test_set_score_skipped_step_does_not_emit_score(
    monkeypatch, tmp_path: Path,
):
    """Skipped steps (sr.skip=True) must NOT emit a score — there's
    no verdict to score for steps that didn't run."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    from mantis_agent.gym.run_executor import RunExecutor

    augur = AugurAdapter(
        run_id="exec_skipped", tenant_id="t", session_name="s", out_dir=tmp_path,
    )
    augur_spy = MagicMock(wraps=augur)
    runner = MagicMock()
    runner._augur = augur_spy
    runner._augur_capture_upgraded = False
    runner._healing_events = []
    runner.time_meter = None
    runner._invoke_step_callback = lambda *a, **kw: None
    runner.cost_meter = MagicMock()
    runner.cost_meter.costs = {}

    sr = _make_step_result(
        success=False, skip=True, kind_value="recoverable", confidence=0.5,
    )
    sr.reasoning = ""

    executor = RunExecutor.__new__(RunExecutor)
    executor.parent = runner
    executor._emit_augur_step(sr, "2026-05-20T18:33:56Z")
    augur.close(status="completed")

    augur_spy.set_score.assert_not_called()
