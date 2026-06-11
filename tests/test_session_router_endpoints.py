"""Tests for the session-router endpoints in build_api_app (Phase 1.5,
#846, PR 2).

Drives ``POST/DELETE/GET /v1/computer_sessions`` against the FastAPI
app via TestClient, with stubbed ``session_spawner`` and a plain-dict
``session_dict``. Covers the happy path, idempotent re-create,
tunnel-timeout 504, tenant mismatch 403, and the close/list paths.
"""

from __future__ import annotations

import importlib
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("modal")

from fastapi.testclient import TestClient

from mantis_agent import tenant_auth as ta_mod
from mantis_agent.session_wire import (
    SessionCreateResponse,
    SessionRecord,
)
from mantis_agent.run_state_store import KIND_SESSION, RunStateStore


_DEPLOY_MODAL = Path(__file__).resolve().parent.parent / "deploy" / "modal"
if str(_DEPLOY_MODAL) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_MODAL))


class _StubCall:
    """Stand-in for the FunctionCall handle returned by ``.spawn()``."""

    def __init__(self) -> None:
        self.object_id = "fc-session-stub-1"
        self.cancel_called = False

    def cancel(self) -> None:
        self.cancel_called = True


def _h() -> dict:
    return {"X-Mantis-Token": "test-token", "Content-Type": "application/json"}


def _create_payload(**overrides) -> dict:
    base = {
        "tenant_id": "default",
        "profile_id": "alice",
        "run_id": "run_1",
        "start_url": "https://example.com",
        "viewport": [1280, 720],
        "ttl_seconds": 3600,
    }
    base.update(overrides)
    return base


@pytest.fixture
def app_with_stubs(monkeypatch, tmp_path):
    """FastAPI app wired with a stub session spawner.

    The stub records the spawn payload AND simulates the orchestrator
    by writing the tunnel URL into the supplied ``session_dict`` on a
    background thread — so the router's poll loop finds it within
    a second.
    """
    monkeypatch.setenv("MANTIS_API_TOKEN", "test-token")
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path / "mantis-data"))
    monkeypatch.delenv("MANTIS_TENANT_KEYS_PATH", raising=False)
    monkeypatch.delenv("MODAL_TASK_ID", raising=False)
    ta_mod.reset_key_store()

    import modal_cua_server as mcs  # type: ignore[import-not-found]
    importlib.reload(mcs)

    session_dict: dict[str, Any] = {}
    spawn_calls: list[dict] = []

    def fake_spawner(payload: dict) -> _StubCall:
        spawn_calls.append(payload)
        call = _StubCall()
        # Simulate the orchestrator writing its tunnel URL after a
        # short delay — exercises the router's wait loop.
        def _publish() -> None:
            time.sleep(0.05)
            session_dict[payload["session_id"]] = {
                "session_id": payload["session_id"],
                "session_token": payload["session_token"],
                "tunnel_url": f"https://stub-{payload['session_id'][:8]}.modal.run",
                "status": "ready",
                "started_at_ms": int(time.time() * 1000),
            }
        threading.Thread(target=_publish, daemon=True).start()
        return call

    store = RunStateStore(backing={})

    app = mcs.build_api_app(
        executor_resolver=lambda model: None,
        function_call_lookup=lambda call_id: _StubCall(),
        run_state_store=store,
        session_spawner=fake_spawner,
        session_dict=session_dict,
    )
    return TestClient(app), session_dict, spawn_calls, store, mcs


# ── Happy path ────────────────────────────────────────────────────────


def test_create_session_returns_tunnel_url(app_with_stubs) -> None:
    client, session_dict, spawn_calls, _store, _mcs = app_with_stubs
    r = client.post(
        "/v1/computer_sessions",
        headers=_h(),
        data=json.dumps(_create_payload()),
    )
    assert r.status_code == 200, r.text
    body = SessionCreateResponse.model_validate(r.json())
    assert body.session_id.startswith("sess_")
    assert body.base_url.startswith("https://stub-")
    assert body.session_token
    assert body.expires_at_ms > int(time.time() * 1000)
    assert body.reused is False

    # One spawn happened with the expected payload shape.
    assert len(spawn_calls) == 1
    spawn = spawn_calls[0]
    assert spawn["session_id"] == body.session_id
    assert spawn["session_token"] == body.session_token
    assert spawn["ttl_seconds"] == 3600
    init = spawn["init_payload"]
    assert init["tenant_id"] == "default"
    assert init["profile_id"] == "alice"
    assert init["run_id"] == "run_1"
    # ttl_seconds is router-only; not forwarded into the init payload.
    assert "ttl_seconds" not in init


def test_create_session_persists_record(app_with_stubs) -> None:
    client, _sd, _calls, store, _mcs = app_with_stubs
    r = client.post(
        "/v1/computer_sessions",
        headers=_h(),
        data=json.dumps(_create_payload()),
    )
    sid = r.json()["session_id"]
    blob = store.get("default", sid, KIND_SESSION)
    assert blob is not None
    record = SessionRecord.model_validate(blob)
    assert record.session_id == sid
    assert record.status == "active"
    assert record.base_url.startswith("https://stub-")
    assert record.sandbox_id == "fc-session-stub-1"


def test_idempotent_recreate_returns_existing_session(app_with_stubs) -> None:
    client, _sd, spawn_calls, _store, _mcs = app_with_stubs
    payload = _create_payload(run_id="dup_run")
    r1 = client.post("/v1/computer_sessions", headers=_h(), data=json.dumps(payload))
    r2 = client.post("/v1/computer_sessions", headers=_h(), data=json.dumps(payload))
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["session_id"] == r2.json()["session_id"]
    assert r2.json()["reused"] is True
    # Only one spawn — the second hit the idempotent fast-path.
    assert len(spawn_calls) == 1


# ── Failure modes ─────────────────────────────────────────────────────


def test_tenant_id_normalized_to_token_tenant(app_with_stubs) -> None:
    """Brain-side hint for tenant_id is overridden by the auth token.
    The record + spawn payload carry the token's tenant, regardless
    of what the brain sent (brains often don't know their tenant
    outside the token)."""
    client, _sd, spawn_calls, store, _mcs = app_with_stubs
    r = client.post(
        "/v1/computer_sessions", headers=_h(),
        data=json.dumps(_create_payload(tenant_id="someone-else")),
    )
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]
    # Record stored under the TOKEN's tenant, not what the brain claimed.
    blob = store.get("default", sid, KIND_SESSION)
    assert blob is not None
    record = SessionRecord.model_validate(blob)
    assert record.tenant_id == "default"
    # And spawn payload's init_payload also carries the normalized tenant.
    assert spawn_calls[0]["init_payload"]["tenant_id"] == "default"


def test_unauthorized_request_rejected(app_with_stubs) -> None:
    client, _sd, _calls, _store, _mcs = app_with_stubs
    r = client.post(
        "/v1/computer_sessions",
        headers={"Content-Type": "application/json"},  # no token
        data=json.dumps(_create_payload()),
    )
    assert r.status_code == 401


def test_create_session_tunnel_timeout_returns_504(monkeypatch, tmp_path) -> None:
    """When the orchestrator never publishes a tunnel URL, the router
    times out and 504s — and persists a terminal error record."""
    monkeypatch.setenv("MANTIS_API_TOKEN", "test-token")
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.delenv("MODAL_TASK_ID", raising=False)
    ta_mod.reset_key_store()

    import modal_cua_server as mcs  # type: ignore[import-not-found]
    importlib.reload(mcs)

    # Patch the router-side polling deadline to a single short tick so
    # the test doesn't actually wait 120s.
    real_time = time.time
    base_time = real_time()
    monkeypatch.setattr("modal_cua_server.time.time",
                        lambda: real_time() + 130 if real_time() - base_time > 0.05 else real_time())

    session_dict: dict[str, Any] = {}
    captured_call = _StubCall()

    def silent_spawner(_payload: dict) -> _StubCall:
        # Doesn't publish — simulates orchestrator hang.
        return captured_call

    store = RunStateStore(backing={})
    app = mcs.build_api_app(
        executor_resolver=lambda model: None,
        function_call_lookup=lambda call_id: captured_call,
        run_state_store=store,
        session_spawner=silent_spawner,
        session_dict=session_dict,
    )
    client = TestClient(app)
    r = client.post(
        "/v1/computer_sessions", headers=_h(),
        data=json.dumps(_create_payload()),
    )
    assert r.status_code == 504, r.text
    assert "did not publish tunnel URL" in r.json()["detail"]
    # FunctionCall cancel was called.
    assert captured_call.cancel_called

    # And a terminal error record was persisted so an operator can
    # grep what happened.
    records = list(store._d.keys())  # noqa: SLF001
    sess_keys = [k for k in records if k.endswith("/session")]
    assert len(sess_keys) == 1
    tenant_seg, sid, _ = sess_keys[0].rsplit("/", 2)
    blob = store.get(tenant_seg, sid, KIND_SESSION)
    record = SessionRecord.model_validate(blob)
    assert record.status == "error"
    assert "did not publish" in record.error


# ── Close ─────────────────────────────────────────────────────────────


def test_close_session_signals_orchestrator_and_marks_record(app_with_stubs) -> None:
    client, session_dict, _calls, store, _mcs = app_with_stubs
    r = client.post(
        "/v1/computer_sessions", headers=_h(),
        data=json.dumps(_create_payload(run_id="close_test")),
    )
    sid = r.json()["session_id"]

    r2 = client.delete(f"/v1/computer_sessions/{sid}", headers=_h())
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["closed"] is True
    assert body["terminal_state"] == "closed"

    # session_dict has the close signal.
    assert session_dict[sid]["close_requested"] is True
    assert session_dict[sid]["close_reason"] == "brain_closed"

    # Record is marked closed.
    blob = store.get("default", sid, KIND_SESSION)
    assert SessionRecord.model_validate(blob).status == "closed"


def test_close_unknown_session_404(app_with_stubs) -> None:
    client, _sd, _calls, _store, _mcs = app_with_stubs
    r = client.delete(
        "/v1/computer_sessions/sess_does_not_exist", headers=_h(),
    )
    assert r.status_code == 404
    assert "unknown session_id" in r.json()["detail"]


# ── GET / LIST ────────────────────────────────────────────────────────


def test_get_session_returns_record(app_with_stubs) -> None:
    client, _sd, _calls, _store, _mcs = app_with_stubs
    r = client.post(
        "/v1/computer_sessions", headers=_h(),
        data=json.dumps(_create_payload(run_id="get_test")),
    )
    sid = r.json()["session_id"]
    g = client.get(f"/v1/computer_sessions/{sid}", headers=_h())
    assert g.status_code == 200, g.text
    body = SessionRecord.model_validate(g.json())
    assert body.session_id == sid
    assert body.run_id == "get_test"


def test_list_sessions_returns_only_tenant_records(app_with_stubs) -> None:
    client, _sd, _calls, store, _mcs = app_with_stubs
    # Create one session for the default tenant via the endpoint.
    r = client.post(
        "/v1/computer_sessions", headers=_h(),
        data=json.dumps(_create_payload(run_id="list_test")),
    )
    sid = r.json()["session_id"]
    # And inject a "different tenant" record directly into the store.
    from mantis_agent.session_wire import serialize_session_record
    other = SessionRecord(
        session_id="sess_other",
        tenant_id="other-tenant",
        profile_id="x",
        run_id="x",
        base_url="https://x",
        created_at_ms=0, expires_at_ms=0,
    )
    store.put("other-tenant", "sess_other", KIND_SESSION,
              serialize_session_record(other))

    listed = client.get("/v1/computer_sessions", headers=_h())
    assert listed.status_code == 200, listed.text
    sessions = listed.json()["sessions"]
    ids = [s["session_id"] for s in sessions]
    assert sid in ids
    assert "sess_other" not in ids, "cross-tenant session leaked into list"
    assert listed.json()["count"] == len(sessions)
