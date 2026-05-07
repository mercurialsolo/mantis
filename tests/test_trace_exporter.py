"""Tests for #155 step 1 — TraceExporter.

Pins the schema, the feature-flag-gating, and the tenant-isolation
contract. The exporter is wired into ``RunExecutor._finalize``; tests
exercise the exporter directly via a stub runner so they don't need
Xvfb / Chrome.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.gym.trace_exporter import (
    ENV_DIR,
    ENV_INCLUDE_SCREENSHOTS,
    SCHEMA_VERSION,
    TraceExporter,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _runner_stub(
    *,
    run_key: str = "rid_abc",
    tenant_id: str = "t-one",
    session_name: str = "sess",
    plan_signature: str = "sig",
    final_status: str = "completed",
) -> MagicMock:
    runner = MagicMock()
    runner.run_key = run_key
    runner.tenant_id = tenant_id
    runner.session_name = session_name
    runner.plan_signature = plan_signature
    runner._final_status = final_status
    runner._run_start = 100.0
    runner._cost_totals.return_value = (1.0, 2.0, 0.5, 3.5)
    return runner


def _step(idx: int, **overrides) -> StepResult:
    base = dict(
        step_index=idx, intent=f"step {idx}", success=True,
        data="", steps_used=1, duration=0.42, reversed=False,
    )
    base.update(overrides)
    return StepResult(**base)


# ── from_env / feature flag ────────────────────────────────────────────


def test_from_env_disabled_by_default(monkeypatch):
    monkeypatch.delenv(ENV_DIR, raising=False)
    exp = TraceExporter.from_env()
    assert exp.enabled is False


def test_from_env_enabled_when_dir_set(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_DIR, str(tmp_path))
    exp = TraceExporter.from_env()
    assert exp.enabled is True
    assert exp.export_dir == str(tmp_path)


def test_from_env_include_screenshots_truthy(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_DIR, str(tmp_path))
    monkeypatch.setenv(ENV_INCLUDE_SCREENSHOTS, "true")
    assert TraceExporter.from_env().include_screenshots is True


def test_from_env_include_screenshots_falsy(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_DIR, str(tmp_path))
    monkeypatch.setenv(ENV_INCLUDE_SCREENSHOTS, "no")
    assert TraceExporter.from_env().include_screenshots is False


def test_maybe_export_noop_when_disabled():
    exp = TraceExporter(export_dir="")
    runner = _runner_stub()
    assert exp.maybe_export(runner, [_step(0)], status="completed") is None


# ── Schema ─────────────────────────────────────────────────────────────


def test_export_writes_json_at_tenant_path(tmp_path):
    exp = TraceExporter(export_dir=str(tmp_path))
    runner = _runner_stub(tenant_id="acme", run_key="run123")
    out = exp.maybe_export(runner, [_step(0), _step(1)], status="completed")
    assert out is not None
    out_path = Path(out)
    # Tenant-scoped path layout
    assert out_path.parent.name == "acme"
    assert out_path.name == "run123.json"


def test_export_payload_schema_top_level_fields(tmp_path):
    exp = TraceExporter(export_dir=str(tmp_path))
    runner = _runner_stub()
    out = exp.maybe_export(runner, [_step(0)], status="completed")
    payload = json.loads(Path(out).read_text())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["status"] == "completed"
    assert payload["run_id"] == "rid_abc"
    assert payload["tenant_id"] == "t-one"
    assert payload["plan_signature"] == "sig"
    assert payload["step_count"] == 1
    assert payload["costs"] == {"gpu": 1.0, "claude": 2.0, "proxy": 0.5, "total": 3.5}
    # Steps round-trip with the expected fields
    step = payload["steps"][0]
    for field in ("step_index", "intent", "success", "data", "duration", "reversed",
                  "predicted_outcome", "observed_outcome", "last_action"):
        assert field in step


def test_export_serializes_action(tmp_path):
    exp = TraceExporter(export_dir=str(tmp_path))
    runner = _runner_stub()
    step = _step(0)
    step.last_action = Action(ActionType.CLICK, {"x": 100, "y": 200})
    out = exp.maybe_export(runner, [step], status="completed")
    payload = json.loads(Path(out).read_text())
    action_dict = payload["steps"][0]["last_action"]
    assert action_dict["action_type"] == "click"
    assert action_dict["params"] == {"x": 100, "y": 200}


def test_export_handles_predicted_observed_outcomes(tmp_path):
    exp = TraceExporter(export_dir=str(tmp_path))
    runner = _runner_stub()
    step = _step(0)
    # SimpleNamespace lets us tack predicted_outcome on without subclassing
    step.predicted_outcome = "URL navigates to /detail"  # type: ignore[attr-defined]
    step.observed_outcome = "page navigated"  # type: ignore[attr-defined]
    out = exp.maybe_export(runner, [step], status="completed")
    payload = json.loads(Path(out).read_text())
    assert payload["steps"][0]["predicted_outcome"] == "URL navigates to /detail"
    assert payload["steps"][0]["observed_outcome"] == "page navigated"


def test_export_records_shadow_variant_when_set(tmp_path):
    """#155 step 5: when ``runner.shadow_variant`` is set, the trace
    JSON gains a top-level ``variant`` field. Lets shadow-deploy
    analytics attribute escalation rate per variant."""
    exp = TraceExporter(export_dir=str(tmp_path))
    runner = _runner_stub()
    runner.shadow_variant = "candidate"
    out = exp.maybe_export(runner, [_step(0)], status="completed")
    payload = json.loads(Path(out).read_text())
    assert payload["variant"] == "candidate"


def test_export_variant_field_empty_when_runner_has_no_attr(tmp_path):
    """Legacy runners without the shadow_variant attribute still emit
    a (blank) ``variant`` field — analytics treats blank as
    ``__unassigned__``."""
    exp = TraceExporter(export_dir=str(tmp_path))
    runner = _runner_stub()
    # Set shadow_variant to empty string, mirroring the legacy default
    # (a real MicroPlanRunner without the shadow router would have ``""``;
    # MagicMock would otherwise auto-vivify a Mock object).
    runner.shadow_variant = ""
    out = exp.maybe_export(runner, [_step(0)], status="completed")
    payload = json.loads(Path(out).read_text())
    assert payload["variant"] == ""


def test_export_empty_tenant_id_falls_back_to_shared(tmp_path):
    exp = TraceExporter(export_dir=str(tmp_path))
    runner = _runner_stub(tenant_id="")
    out = exp.maybe_export(runner, [_step(0)], status="completed")
    assert Path(out).parent.name == "__shared__"
    payload = json.loads(Path(out).read_text())
    # Top-level tenant_id is the empty string for legacy single-tenant runs
    assert payload["tenant_id"] == ""


# ── Screenshot persistence ─────────────────────────────────────────────


def test_export_does_not_write_screenshots_by_default(tmp_path):
    """Without the include flag, screenshot bytes never hit disk."""
    exp = TraceExporter(export_dir=str(tmp_path), include_screenshots=False)
    runner = _runner_stub()
    step = _step(0)
    step.screenshot_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32  # tiny stub
    exp.maybe_export(runner, [step], status="completed")
    # No companion screens dir
    assert not (tmp_path / "t-one" / "rid_abc_screens").exists()


def test_export_writes_screenshots_when_flag_set(tmp_path):
    exp = TraceExporter(export_dir=str(tmp_path), include_screenshots=True)
    runner = _runner_stub()
    step = _step(0)
    step.screenshot_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    exp.maybe_export(runner, [step], status="completed")
    shots_dir = tmp_path / "t-one" / "rid_abc_screens"
    assert shots_dir.exists()
    assert (shots_dir / "0000.png").exists()


# ── Telemetry safety ───────────────────────────────────────────────────


def test_export_swallows_io_failures(tmp_path):
    """If the exporter raises, the runtime must not crash."""
    # Pass a path under a non-writable location to force OSError on mkdir.
    exp = TraceExporter(export_dir="/proc/cannot-write")
    runner = _runner_stub()
    # Must not raise. Returns None on failure.
    assert exp.maybe_export(runner, [_step(0)], status="completed") is None
