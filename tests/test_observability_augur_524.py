"""Tests for #524 ergonomic wires (set_score + set_capture_mode).

Two adapter wrappers + their run_executor wiring:

* ``AugurAdapter.set_score(...)`` — pass verifier confidence through to
  Augur instead of relying on the default binary status→score map.
* ``AugurAdapter.set_capture_mode(...)`` — upgrade from metadata
  capture to screenshots on first failure (auto-evidence-collection).

The third sub-task (``append_log`` for runner stdout streaming) is
deferred — the adapter wrapper already ships (since SDK 0.1.3); the
stdout-redirect piece is a separate scope.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("augur_sdk")

from mantis_agent.observability.augur import AugurAdapter


# ── set_score wrapper ────────────────────────────────────────────────────


def _record_step_minimal(adapter: AugurAdapter, step_index: int = 0) -> None:
    """Same helper pattern as test_observability_augur_518.py — drive
    record_step with just enough fields for the SDK to accept."""
    sr = MagicMock()
    sr.step_index = step_index
    sr.intent = "x"
    sr.success = True
    sr.skip = False
    sr.reversed = False
    sr.duration = 0.1
    sr.failure_class = ""
    sr.executor_backend = ""
    sr.last_action = None
    sr.verdict = None
    sr.recovery_decision = None
    sr.screenshot_png = None
    sr.data = ""
    sr.page_title = ""
    adapter.record_step(
        step_result=sr,
        started_at="2026-05-20T10:00:00Z",
        ended_at="2026-05-20T10:00:01Z",
    )


def test_set_score_patches_step_verdict_score(monkeypatch, tmp_path: Path):
    """set_score(step_index=0, score=0.83) lands on the recorded step's
    verdict — replaces Augur's default binary status mapping with the
    verifier's actual confidence. Adapter bumps Mantis 0-based →
    Augur 1-based at the boundary."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(
        run_id="score_v1", tenant_id="t", session_name="s", out_dir=tmp_path,
    )
    _record_step_minimal(a, step_index=0)
    a.set_score(0, 0.83, comparator="verifier")
    a.close(status="completed")
    step = json.loads((tmp_path / "steps" / "0001.json").read_text())
    # Scores land under verdict.score per the SDK 0.1.7 contract.
    verdict = step.get("verdict", {})
    assert verdict.get("score") == 0.83 or (
        # Some SDKs store under reward / components; accept either canonical key.
        verdict.get("reward") == 0.83
    )


def test_set_score_clamps_via_sdk(monkeypatch, tmp_path: Path):
    """SDK clamps score to [0.0, 1.0]; out-of-band values either get
    clamped or raise ValueError (which we swallow). Either way, the
    wrapper must not raise and the recorded score must land in the
    valid range."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(
        run_id="score_clamp", tenant_id="t", session_name="s", out_dir=tmp_path,
    )
    _record_step_minimal(a, step_index=0)
    # SDK 0.1.8 clamps internally — MUST NOT raise either way.
    a.set_score(0, 1.5)  # over
    a.set_score(0, -0.2)  # under
    a.close(status="completed")
    step = json.loads((tmp_path / "steps" / "0001.json").read_text())
    verdict = step.get("verdict", {})
    score = verdict.get("score", verdict.get("reward"))
    if score is not None:
        assert 0.0 <= float(score) <= 1.0


def test_set_score_noop_when_disabled(monkeypatch, tmp_path: Path):
    """Disabled adapter → silent no-op. Telemetry never breaks runs."""
    monkeypatch.setenv("MANTIS_AUGUR_DISABLED", "1")
    a = AugurAdapter(
        run_id="score_noop", tenant_id="t", session_name="s", out_dir=tmp_path,
    )
    assert not a.active
    # Must not raise.
    a.set_score(0, 0.5)
    a.set_score(0, 0.5, comparator="x", components={"a": 0.5})


# ── set_capture_mode wrapper ─────────────────────────────────────────────


def test_set_capture_mode_stamps_next_step(monkeypatch, tmp_path: Path):
    """set_capture_mode('screenshots') stamps the per-step capture
    override on the NEXT recorded step (SDK 0.1.8 behavior — manifest
    keeps its baseline mode). Used to upgrade healthy/metadata runs
    to screenshot evidence collection on first failure."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(
        run_id="cap_mode_v1", tenant_id="t", session_name="s",
        out_dir=tmp_path, capture_mode="metadata",
    )
    a.set_capture_mode("screenshots")
    _record_step_minimal(a, step_index=0)
    a.close(status="completed")
    step = json.loads((tmp_path / "steps" / "0001.json").read_text())
    # Step trace carries the override; manifest keeps the baseline.
    assert step.get("capture_mode") == "screenshots"
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["capture_mode"] == "metadata"  # baseline preserved


def test_set_capture_mode_unknown_value_does_not_raise(
    monkeypatch, tmp_path: Path,
):
    """Unknown capture-mode strings → SDK ValueError, which we swallow.
    Bundle proceeds with the prior mode intact."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(
        run_id="cap_mode_bad", tenant_id="t", session_name="s",
        out_dir=tmp_path, capture_mode="screenshots",
    )
    # Must not raise.
    a.set_capture_mode("not-a-real-mode")
    a.close(status="completed")
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    # Original mode preserved.
    assert manifest["capture_mode"] == "screenshots"


def test_set_capture_mode_noop_when_disabled(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MANTIS_AUGUR_DISABLED", "1")
    a = AugurAdapter(
        run_id="cap_mode_noop", tenant_id="t", session_name="s", out_dir=tmp_path,
    )
    assert not a.active
    a.set_capture_mode("full")  # Must not raise


# ── _emit_augur_step wiring ──────────────────────────────────────────────


def test_run_executor_passes_verdict_confidence_through_set_score(
    monkeypatch, tmp_path: Path,
):
    """When step_result.verdict.confidence > 0, _emit_augur_step should
    call augur.set_score with the actual confidence value. Catches
    regressions where the wire drops the verdict reading."""
    from mantis_agent.gym.run_executor import RunExecutor

    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    augur = AugurAdapter(
        run_id="exec_set_score", tenant_id="t", session_name="s", out_dir=tmp_path,
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

    step_result = MagicMock()
    step_result.step_index = 2
    step_result.success = True
    step_result.skip = False
    step_result.reversed = False
    step_result.failure_class = ""
    step_result.executor_backend = ""
    step_result.last_action = None
    step_result.screenshot_png = None
    step_result.reasoning = ""
    step_result.data = ""
    step_result.page_title = ""
    # Fake Verdict with a non-trivial confidence.
    step_result.verdict = MagicMock()
    step_result.verdict.confidence = 0.73
    step_result.recovery_decision = None
    step_result.executor_backend = ""

    executor = RunExecutor.__new__(RunExecutor)
    executor.parent = runner
    executor._emit_augur_step(step_result, "2026-05-20T10:00:00Z")

    augur_spy.set_score.assert_called_once()
    args = augur_spy.set_score.call_args
    assert args.args[0] == 2  # step_index
    assert args.args[1] == 0.73  # confidence
    assert args.kwargs.get("comparator") == "verifier"
    augur.close(status="completed")


def test_run_executor_falls_back_to_one_on_success_when_confidence_zero(
    monkeypatch, tmp_path: Path,
):
    """When verdict.confidence is 0.0 on a successful step, use the
    derived score 1.0 (canonical from sr.success). Updated from the
    prior #524-only contract per #530's emission-side defense: the
    score must always reflect sr.success so downstream RLHF/DPO
    consumers get a usable signal even when the verifier didn't
    publish a confidence value."""
    from mantis_agent.gym.run_executor import RunExecutor

    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    augur = AugurAdapter(
        run_id="exec_zero_score", tenant_id="t", session_name="s", out_dir=tmp_path,
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

    step_result = MagicMock()
    step_result.step_index = 1
    step_result.success = True
    step_result.skip = False
    step_result.failure_class = ""
    step_result.screenshot_png = None
    step_result.reasoning = ""
    step_result.data = ""
    step_result.verdict = MagicMock()
    step_result.verdict.confidence = 0.0  # planner didn't set one
    step_result.recovery_decision = None
    step_result.executor_backend = ""

    executor = RunExecutor.__new__(RunExecutor)
    executor.parent = runner
    executor._emit_augur_step(step_result, "2026-05-20T10:00:00Z")
    augur_spy.set_score.assert_called_once()
    score = augur_spy.set_score.call_args.args[1]
    assert score == 1.0, (
        "Successful step with no confidence must default to score=1.0 "
        f"(got {score}). Per #530's emission-side defense."
    )
    augur.close(status="completed")


def test_run_executor_upgrades_capture_mode_on_first_failure(
    monkeypatch, tmp_path: Path,
):
    """First failed step → upgrade capture mode to screenshots.
    Subsequent failures are idempotent (don't re-trigger)."""
    from mantis_agent.gym.run_executor import RunExecutor

    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    augur = AugurAdapter(
        run_id="exec_cap_upgrade", tenant_id="t", session_name="s",
        out_dir=tmp_path, capture_mode="metadata",
    )
    augur_spy = MagicMock(wraps=augur)
    runner = MagicMock()
    runner._augur = augur_spy
    runner._healing_events = []
    runner.time_meter = None
    runner._invoke_step_callback = lambda *a, **kw: None
    runner.cost_meter = MagicMock()
    runner.cost_meter.costs = {}
    # No prior upgrade flag.
    runner._augur_capture_upgraded = False

    def _make_failure(idx: int):
        sr = MagicMock()
        sr.step_index = idx
        sr.success = False
        sr.skip = False
        sr.failure_class = "no_state_change"
        sr.screenshot_png = None
        sr.reasoning = ""
        sr.data = ""
        sr.verdict = None
        sr.recovery_decision = None
        sr.executor_backend = ""
        return sr

    executor = RunExecutor.__new__(RunExecutor)
    executor.parent = runner

    # First failure → upgrades.
    executor._emit_augur_step(_make_failure(0), "2026-05-20T10:00:00Z")
    augur_spy.set_capture_mode.assert_called_once_with("screenshots")
    assert runner._augur_capture_upgraded is True

    # Second failure → no second upgrade.
    augur_spy.set_capture_mode.reset_mock()
    executor._emit_augur_step(_make_failure(1), "2026-05-20T10:00:01Z")
    augur_spy.set_capture_mode.assert_not_called()
    augur.close(status="completed")


def test_run_executor_does_not_upgrade_on_success(
    monkeypatch, tmp_path: Path,
):
    """Healthy runs (all success) stay on the original capture mode —
    no upgrade, no extra storage."""
    from mantis_agent.gym.run_executor import RunExecutor

    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    augur = AugurAdapter(
        run_id="exec_no_upgrade", tenant_id="t", session_name="s",
        out_dir=tmp_path, capture_mode="metadata",
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

    sr = MagicMock()
    sr.step_index = 0
    sr.success = True
    sr.skip = False
    sr.failure_class = ""
    sr.screenshot_png = None
    sr.reasoning = ""
    sr.data = ""
    sr.verdict = None
    sr.recovery_decision = None
    sr.executor_backend = ""

    executor = RunExecutor.__new__(RunExecutor)
    executor.parent = runner
    executor._emit_augur_step(sr, "2026-05-20T10:00:00Z")
    augur_spy.set_capture_mode.assert_not_called()
    augur.close(status="completed")
