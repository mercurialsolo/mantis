"""Baseten ↔ Modal action-response shape parity.

PR #862 brought Baseten to functional parity with Modal's API
surfaces. Live verification afterwards surfaced two cosmetic shape
divergences:

  1. ``action=result`` — Modal nests the executor result under a
     ``result`` key on the same status envelope; Baseten was
     returning the raw result.json body at top-level.

  2. ``action=status`` identity fields — Modal surfaces
     ``profile_id`` / ``workflow_id`` / ``state_key`` / ``tenant_id``
     / ``max_steps`` at top-level; Baseten kept them nested inside
     ``status['payload']['task_suite']``.

These tests pin both fixes so the surfaces don't drift again.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def runtime(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MANTIS_BRAIN", "holo3")
    monkeypatch.setenv("MANTIS_SKIP_BRAIN_LOAD", "1")
    from mantis_agent.baseten_server.runtime import BasetenCUARuntime
    rt = BasetenCUARuntime()
    rt.load = lambda: None  # type: ignore[assignment]
    return rt


def _seed_run_with_payload(rt, run_id: str, *, status: str = "succeeded",
                           profile_id: str = "alice",
                           workflow_id: str = "wf-1",
                           state_key: str = "wf-1",
                           max_steps: int = 25) -> Path:
    """Stamp a status.json with a payload + task_suite envelope so the
    identity-lift fields are exercised."""
    rt._write_detached_status(run_id, {
        "status": status,
        "mode": "detached",
        "model": "holo3",
        "payload": {
            "max_steps": max_steps,
            "task_suite": {
                "_profile_id": profile_id,
                "_workflow_id": workflow_id,
                "_state_key": state_key,
            },
        },
    })
    return rt._run_path(run_id) / "status.json"


def _seed_result_json(rt, run_id: str, body: dict) -> Path:
    """Drop a result.json the way a finished worker would."""
    path = rt._run_path(run_id, create=True) / "result.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


# ── action=result wraps under `result` ─────────────────────────────────


def test_action_result_wraps_executor_payload_under_result_key(runtime) -> None:
    """Modal returns {<status fields>, result: {<executor payload>}}.
    Baseten now matches."""
    _seed_run_with_payload(runtime, "r-wrap")
    _seed_result_json(runtime, "r-wrap", {
        "leads": [{"rank": 1, "title": "x"}],
        "artifacts": ["leads.csv"],
        "viable": 1,
    })
    out = runtime.run({
        "action": "result", "run_id": "r-wrap", "detached": True,
    })
    assert "result" in out, "executor payload must be nested under 'result'"
    assert out["result"]["leads"] == [{"rank": 1, "title": "x"}]
    assert out["result"]["artifacts"] == ["leads.csv"]
    # And the status envelope is on the SAME object — same fields the
    # status response would surface.
    assert out["status"] == "succeeded"
    assert out["run_id"] == "r-wrap"


def test_action_result_when_no_result_yet_returns_result_null(runtime) -> None:
    """When result.json doesn't exist yet (run still pending), the
    envelope still nests — ``result: None`` + ``result_ready: False``.
    Mirrors Modal's shape; lets clients branch on a single key."""
    _seed_run_with_payload(runtime, "r-pending", status="running")
    out = runtime.run({
        "action": "result", "run_id": "r-pending", "detached": True,
    })
    assert out["result"] is None
    assert out["result_ready"] is False
    assert out["status"] == "running"


def test_action_result_envelope_carries_identity_fields(runtime) -> None:
    """The wrapped envelope also passes through ``_enrich_status``, so
    the lifted identity fields show up alongside the executor payload."""
    _seed_run_with_payload(runtime, "r-ids",
                           profile_id="alice", workflow_id="hn-top5",
                           state_key="hn-top5", max_steps=42)
    _seed_result_json(runtime, "r-ids", {"leads": []})
    out = runtime.run({
        "action": "result", "run_id": "r-ids", "detached": True,
    })
    assert out["profile_id"] == "alice"
    assert out["workflow_id"] == "hn-top5"
    assert out["state_key"] == "hn-top5"
    assert out["max_steps"] == 42


# ── action=status lifts identity fields to top-level ──────────────────


def test_action_status_surfaces_profile_id_at_top_level(runtime) -> None:
    _seed_run_with_payload(runtime, "r-pid", profile_id="alice")
    out = runtime.run({
        "action": "status", "run_id": "r-pid", "detached": True,
    })
    assert out["profile_id"] == "alice"


def test_action_status_surfaces_workflow_id_at_top_level(runtime) -> None:
    _seed_run_with_payload(runtime, "r-wid", workflow_id="hn-top5")
    out = runtime.run({
        "action": "status", "run_id": "r-wid", "detached": True,
    })
    assert out["workflow_id"] == "hn-top5"


def test_action_status_surfaces_state_key_at_top_level(runtime) -> None:
    _seed_run_with_payload(runtime, "r-sk", state_key="acme-prod-v3")
    out = runtime.run({
        "action": "status", "run_id": "r-sk", "detached": True,
    })
    assert out["state_key"] == "acme-prod-v3"


def test_action_status_surfaces_max_steps_at_top_level(runtime) -> None:
    _seed_run_with_payload(runtime, "r-ms", max_steps=42)
    out = runtime.run({
        "action": "status", "run_id": "r-ms", "detached": True,
    })
    assert out["max_steps"] == 42


def test_action_status_top_level_identity_wins_over_suite(runtime) -> None:
    """If the worker already wrote an identity field at top-level
    (e.g. from an in-flight update), the suite-lift must not overwrite
    it. Top-level is authoritative."""
    rt = runtime
    rt._write_detached_status("r-top", {
        "status": "running",
        "profile_id": "TOP_LEVEL",  # already authoritative
        "payload": {
            "task_suite": {"_profile_id": "FROM_SUITE"},
        },
    })
    out = rt.run({"action": "status", "run_id": "r-top", "detached": True})
    assert out["profile_id"] == "TOP_LEVEL"


def test_action_status_missing_payload_does_not_crash(runtime) -> None:
    """A status.json with no payload/suite must not crash the lift —
    just skip the identity fields silently."""
    rt = runtime
    rt._write_detached_status("r-bare", {
        "status": "queued",
        "mode": "detached",
    })
    out = rt.run({"action": "status", "run_id": "r-bare", "detached": True})
    assert out["status"] == "queued"
    # Lift fields absent; that's fine.
    assert "profile_id" not in out or out.get("profile_id") in {None, ""}


# ── REST shorthand /v1/runs/{id}/result also wraps ────────────────────


def test_rest_get_result_wraps_executor_payload_under_result(runtime, monkeypatch) -> None:
    """REST shorthand should preserve the same shape as
    POST /v1/predict with action=result. Direct-runtime invocation
    suffices to pin the contract — the FastAPI layer is a thin
    delegating wrapper covered in tests/test_baseten_parity_audit.py."""
    _seed_run_with_payload(runtime, "r-rest")
    _seed_result_json(runtime, "r-rest", {"leads": [{"rank": 1}]})

    # Invoke through the same code path the REST shorthand uses, but
    # without the FastAPI auth gate (the gate is covered separately).
    out = runtime.run({
        "action": "result", "run_id": "r-rest", "detached": True,
    })
    assert out["result"] == {"leads": [{"rank": 1}]}
    assert out["status"] == "succeeded"
