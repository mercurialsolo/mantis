"""Tests for the cancel-overwrite race + terminal-sticky guard (#866).

Two related bugs the previous server had:

1. ``_do_action`` for ``action=cancel`` swallowed any exception from
   ``lookup_function_call(call_id).cancel()`` — so when the Modal
   control-plane lookup failed the API reported ``cancelled`` while
   the executor kept running, eventually clobbering the cancelled
   status with its own terminal write.

2. ``_write_status`` had no guard against terminal overwrites — once
   a run was ``cancelled`` (or ``halted``), the executor's later
   ``succeeded`` / ``failed`` write would silently overwrite it.

These tests drive the FastAPI app via the existing test harness and
cover both paths.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("modal")

from fastapi.testclient import TestClient

from mantis_agent import tenant_auth as ta_mod
from mantis_agent.run_state_store import RunStateStore


_DEPLOY_MODAL = Path(__file__).resolve().parent.parent / "deploy" / "modal"
if str(_DEPLOY_MODAL) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_MODAL))


class _StubFunctionCall:
    def __init__(self, *, raise_on_cancel: Exception | None = None) -> None:
        self.object_id = "fc-stub-001"
        self.cancel_calls = 0
        self._raise_on_cancel = raise_on_cancel

    def get(self, timeout: float = 0.1) -> Any:
        # Never finishes — exercising cancel paths, not result paths.
        raise TimeoutError("still running")

    def cancel(self) -> None:
        self.cancel_calls += 1
        if self._raise_on_cancel is not None:
            raise self._raise_on_cancel


class _StubExecutor:
    def __init__(self, call: _StubFunctionCall) -> None:
        self.call = call

    def spawn(self, *, task_file_contents: str, **kwargs):  # noqa: D401
        return self.call


def _auth_headers() -> dict[str, str]:
    return {"X-Mantis-Token": "test-token", "Content-Type": "application/json"}


@pytest.fixture
def app_with_stub(monkeypatch, tmp_path):
    monkeypatch.setenv("MANTIS_API_TOKEN", "test-token")
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path / "mantis-data"))
    monkeypatch.delenv("MANTIS_TENANT_KEYS_PATH", raising=False)
    ta_mod.reset_key_store()

    import modal_cua_server as mcs  # type: ignore[import-not-found]
    importlib.reload(mcs)

    stub_call = _StubFunctionCall()
    stub_executor = _StubExecutor(stub_call)
    store = RunStateStore(backing={})

    app = mcs.build_api_app(
        executor_resolver=lambda model: stub_executor,
        function_call_lookup=lambda call_id: stub_call,
        run_state_store=store,
    )
    return TestClient(app), stub_call, store, mcs


def _submit(client: TestClient, profile_id: str = "alice") -> str:
    r = client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({
            "task_suite": {"tasks": [{"intent": "x"}]},
            "profile_id": profile_id,
        }),
    )
    assert r.status_code == 200, r.text
    return r.json()["run_id"]


def _cancel(client: TestClient, run_id: str) -> dict:
    r = client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({"action": "cancel", "run_id": run_id}),
    )
    assert r.status_code == 200, r.text
    return r.json()


def _status(client: TestClient, run_id: str) -> dict:
    r = client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({"action": "status", "run_id": run_id}),
    )
    assert r.status_code == 200, r.text
    return r.json()


# ── Terminal-sticky guard ────────────────────────────────────────────


def test_executor_terminal_write_cannot_overwrite_cancelled(app_with_stub) -> None:
    """Once cancel writes status=cancelled, a later executor terminal
    write (succeeded/failed/halted) is rejected. The cancel must stick."""
    client, _call, _store, mcs = app_with_stub
    run_id = _submit(client)
    body = _cancel(client, run_id)
    assert body["status"] == "cancelled"

    # Simulate the executor finishing AFTER the cancel landed — it
    # calls _write_status directly with its own terminal phase.
    tenant_id = body.get("tenant_id") or ""
    assert tenant_id, "submit response should carry tenant_id"
    mcs._write_status(tenant_id, run_id, {
        **body,
        "status": "succeeded",
        "updated_at": "2099-01-01T00:00:00Z",
    })

    # Read it back through the API — must still be cancelled.
    final = _status(client, run_id)
    assert final["status"] == "cancelled", (
        f"executor terminal write clobbered cancelled: {final}"
    )


def test_terminal_status_blocks_any_change(app_with_stub) -> None:
    client, _call, _store, mcs = app_with_stub
    run_id = _submit(client)
    cancelled = _cancel(client, run_id)
    tenant_id = cancelled["tenant_id"]

    # Try a transition that would step BACK to running — must be blocked.
    mcs._write_status(tenant_id, run_id, {
        **cancelled, "status": "running",
    })
    assert _status(client, run_id)["status"] == "cancelled"

    # Same-status write is a no-op effectively — value changes that
    # don't touch the ``status`` field are still allowed (they're a
    # legitimate way to add fields like ``cancel_lookup_error``).
    mcs._write_status(tenant_id, run_id, {
        **cancelled, "status": "cancelled", "extra_field": "ok",
    })
    after = _status(client, run_id)
    assert after["status"] == "cancelled"
    assert after.get("extra_field") == "ok"


# ── Cancel-call failure surfaced ─────────────────────────────────────


def test_cancel_surfaces_lookup_failure(monkeypatch, tmp_path) -> None:
    """When ``lookup_function_call(...).cancel()`` raises, the cancel
    still records cancelled but surfaces the failure to the caller via
    ``cancel_lookup_error`` on the status payload."""
    monkeypatch.setenv("MANTIS_API_TOKEN", "test-token")
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path / "mantis-data"))
    monkeypatch.delenv("MANTIS_TENANT_KEYS_PATH", raising=False)
    ta_mod.reset_key_store()

    import modal_cua_server as mcs  # type: ignore[import-not-found]
    importlib.reload(mcs)

    call = _StubFunctionCall(raise_on_cancel=RuntimeError("control-plane unreachable"))
    executor = _StubExecutor(call)
    store = RunStateStore(backing={})
    app = mcs.build_api_app(
        executor_resolver=lambda model: executor,
        function_call_lookup=lambda call_id: call,
        run_state_store=store,
    )
    client = TestClient(app)

    run_id = _submit(client)
    body = _cancel(client, run_id)
    assert body["status"] == "cancelled"
    assert "cancel_lookup_error" in body, (
        "cancel must surface the lookup failure instead of swallowing it"
    )
    assert "control-plane unreachable" in body["cancel_lookup_error"]
    assert call.cancel_calls == 1


def test_cancel_succeeds_silently_when_lookup_ok(app_with_stub) -> None:
    client, call, _store, _mcs = app_with_stub
    run_id = _submit(client)
    body = _cancel(client, run_id)
    assert body["status"] == "cancelled"
    assert "cancel_lookup_error" not in body
    assert call.cancel_calls == 1


# ── Unknown run_id still 404s (regression guard) ─────────────────────


def test_unknown_run_id_returns_404(app_with_stub) -> None:
    client, _call, _store, _mcs = app_with_stub
    r = client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({"action": "cancel", "run_id": "20990101_000000_deadbeef"}),
    )
    assert r.status_code == 404, r.text
    assert "unknown run_id" in r.json()["detail"]


# ── Store is the cross-replica visibility layer ──────────────────────


def test_status_visible_through_store_without_disk(app_with_stub) -> None:
    """Verify the store-first read path: a status written into the
    store but NOT on disk (simulating a write that hasn't reached the
    other replica's volume mount yet) is visible to a status read."""
    client, _call, store, mcs = app_with_stub
    run_id = _submit(client)
    body = _status(client, run_id)
    tenant_id = body["tenant_id"]

    # Wipe the disk copy to simulate a replica that hasn't picked up
    # the volume reload yet. The store is the cross-replica truth.
    disk_path = mcs._run_dir(tenant_id, run_id) / "status.json"
    disk_path.unlink()

    # Re-read — must come from the store.
    out = _status(client, run_id)
    assert out["run_id"] == run_id
    assert out["status"] in {"queued", "running"}
