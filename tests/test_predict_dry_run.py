"""Tests for /v1/predict `dry_run: true` (#785 DX-3).

The endpoint glue lives in modal_cua_server.py; the response builder
is the testable unit. End-to-end coverage via TestClient lives in
tests/test_modal_endpoint.py (the existing live-server style suite).
"""

from __future__ import annotations

import json

from mantis_agent.api_schemas import PredictRequest
from mantis_agent.server.dry_run import build_dry_run_response


# ── Schema field ────────────────────────────────────────────────


def test_dry_run_defaults_false():
    req = PredictRequest(plan_text="x")
    assert req.dry_run is False


def test_dry_run_accepts_true():
    req = PredictRequest(plan_text="x", dry_run=True)
    assert req.dry_run is True


# ── Builder shape ───────────────────────────────────────────────


def _suite(**overrides):
    base = {
        "session_name": "test",
        "_micro_plan": [
            {"intent": "Go", "type": "navigate"},
            {"intent": "Extract", "type": "extract_data", "claude_only": True},
        ],
    }
    base.update(overrides)
    return base


def test_builder_emits_dry_run_envelope():
    suite = _suite()
    resp = build_dry_run_response(json.dumps(suite), {"cua_model": "holo3"}, "t1")
    assert resp["dry_run"] is True
    assert resp["tenant_id"] == "t1"
    assert "task_suite" in resp
    assert "cost_estimate" in resp
    assert "plan_summary" in resp
    assert "next_step" in resp


def test_builder_summarizes_steps_by_type():
    suite = _suite(
        _micro_plan=[
            {"intent": "x", "type": "navigate"},
            {"intent": "x", "type": "click"},
            {"intent": "x", "type": "extract_data"},
            {"intent": "x", "type": "extract_data"},
            {"intent": "x", "type": "loop"},
        ]
    )
    resp = build_dry_run_response(json.dumps(suite), {"cua_model": "holo3"})
    s = resp["plan_summary"]
    assert s["step_count"] == 5
    assert s["by_type"] == {"navigate": 1, "click": 1, "extract_data": 2, "loop": 1}


def test_builder_counts_inline_extract_schemas():
    suite = _suite(
        _micro_plan=[
            {"intent": "x", "type": "extract_data",
             "extract": {"fields": [{"name": "title", "type": "str", "required": True}]}},
            {"intent": "x", "type": "extract_data"},  # no extract block
        ]
    )
    resp = build_dry_run_response(json.dumps(suite), {"cua_model": "holo3"})
    assert resp["plan_summary"]["with_inline_extract_schema"] == 1


def test_builder_counts_gate_and_required_steps():
    suite = _suite(
        _micro_plan=[
            {"intent": "x", "type": "extract_data", "gate": True, "required": True},
            {"intent": "x", "type": "navigate", "required": True},
            {"intent": "x", "type": "click"},
        ]
    )
    resp = build_dry_run_response(json.dumps(suite), {"cua_model": "holo3"})
    s = resp["plan_summary"]
    assert s["gate_steps"] == 1
    assert s["required_steps"] == 2


# ── compute_backend resolution ──────────────────────────────────


def test_builder_resolves_computer_plane_when_unset():
    suite = _suite()
    resp = build_dry_run_response(json.dumps(suite), {"cua_model": "holo3"})
    assert resp["compute_backend"] == "computer_plane"


def test_builder_resolves_browser_use_plane_from_runtime_block():
    suite = _suite(runtime={"compute_backend": "browser_use_plane"})
    resp = build_dry_run_response(json.dumps(suite), {"cua_model": "holo3"})
    assert resp["compute_backend"] == "browser_use_plane"


def test_builder_ignores_unknown_compute_backend():
    """Forward-compat: unknown future backend names fall through to
    the default, don't crash."""
    suite = _suite(runtime={"compute_backend": "future_plane_xyz"})
    resp = build_dry_run_response(json.dumps(suite), {"cua_model": "holo3"})
    assert resp["compute_backend"] == "computer_plane"


# ── Cost estimate ───────────────────────────────────────────────


def test_cost_estimate_scales_with_step_count():
    """More steps → more GPU dollars."""
    short = _suite(
        _micro_plan=[{"intent": "x", "type": "navigate"}],
    )
    long = _suite(
        _micro_plan=[{"intent": "x", "type": "navigate"}] * 20,
    )
    short_resp = build_dry_run_response(json.dumps(short), {"cua_model": "holo3"})
    long_resp = build_dry_run_response(json.dumps(long), {"cua_model": "holo3"})
    assert long_resp["cost_estimate"]["gpu_usd"] > short_resp["cost_estimate"]["gpu_usd"]


def test_cost_estimate_includes_decomposer_when_plan_text():
    """plan_text payload adds ~$0.015 (one Claude decomposer call)."""
    suite = _suite()
    pt = build_dry_run_response(json.dumps(suite), {
        "cua_model": "holo3", "plan_text": "Go and extract"
    })
    no_pt = build_dry_run_response(json.dumps(suite), {"cua_model": "holo3"})
    assert pt["cost_estimate"]["claude_usd"] > no_pt["cost_estimate"]["claude_usd"]


def test_cost_estimate_falls_back_for_unknown_model():
    """Unknown cua_model uses the holo3 rate card as a sensible default."""
    suite = _suite()
    resp = build_dry_run_response(json.dumps(suite), {"cua_model": "future-model-xyz"})
    assert resp["cost_estimate"]["estimated_total_usd"] > 0


def test_cost_estimate_carries_caveat_note():
    suite = _suite()
    resp = build_dry_run_response(json.dumps(suite), {"cua_model": "holo3"})
    assert "Rough order-of-magnitude" in resp["cost_estimate"]["note"]


# ── Error paths ─────────────────────────────────────────────────


def test_builder_handles_malformed_json_gracefully():
    """Should never crash the endpoint — return a structured error
    in the envelope instead."""
    resp = build_dry_run_response("not valid json", {"cua_model": "holo3"})
    assert resp["dry_run"] is True
    assert "error" in resp


def test_builder_handles_empty_micro_plan():
    """Edge case — task_suite with no steps. Don't crash."""
    suite = _suite(_micro_plan=[])
    resp = build_dry_run_response(json.dumps(suite), {"cua_model": "holo3"})
    assert resp["plan_summary"]["step_count"] == 0


def test_builder_passes_through_profile_and_workflow():
    """Caller can verify their profile_id / workflow_id resolved
    correctly even in dry-run."""
    resp = build_dry_run_response(
        json.dumps(_suite()),
        {
            "cua_model": "holo3",
            "profile_id": "tenant1__hn-prod",
            "workflow_id": "tenant1__hn-1",
        },
    )
    assert resp["profile_id"] == "tenant1__hn-prod"
    assert resp["workflow_id"] == "tenant1__hn-1"
