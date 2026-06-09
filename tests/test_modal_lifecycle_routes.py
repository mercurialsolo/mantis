"""Tests for lifecycle routes mounted on the Modal HTTP endpoint (#806).

PR #792 shipped the lifecycle data layer (RunPhase enum + Pydantic
schemas + RunLifecycleStore). PR #806 mounts the three end-user
routes on the existing FastAPI surface, deriving phase from the
file-backed ``status.json`` the executor already writes (no in-memory
store — Modal runs the API + executor in different containers).

This file exercises:

- ``GET /v1/runs/{run_id}`` → ``RunPhaseResponse``
- ``GET /v1/queue`` → ``QueueStatusResponse`` scoped to the caller's tenant
- Phase mapping (queued / running / paused / succeeded / failed / etc.)
- Backoff hint scaling
- 404 / auth shapes
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

pytest.importorskip("modal")

from fastapi.testclient import TestClient

from mantis_agent import tenant_auth as ta_mod


_DEPLOY_MODAL = Path(__file__).resolve().parent.parent / "deploy" / "modal"
if str(_DEPLOY_MODAL) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_MODAL))


class _StubFunctionCall:
    def __init__(self) -> None:
        self.object_id = "fc-stub-lifecycle"

    def get(self, timeout: float = 0.1):
        return {}

    def cancel(self) -> None:
        return None


class _StubExecutor:
    def __init__(self, call: _StubFunctionCall) -> None:
        self.call = call
        self.spawn_kwargs: dict = {}

    def spawn(self, *, task_file_contents: str, **kwargs) -> _StubFunctionCall:
        self.spawn_kwargs = {"task_file_contents": task_file_contents, **kwargs}
        return self.call


@pytest.fixture
def app_ctx(monkeypatch, tmp_path):
    monkeypatch.setenv("MANTIS_API_TOKEN", "test-token")
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path / "mantis-data"))
    monkeypatch.delenv("MANTIS_TENANT_KEYS_PATH", raising=False)
    ta_mod.reset_key_store()

    import modal_cua_server as mcs  # type: ignore[import-not-found]
    importlib.reload(mcs)

    stub_call = _StubFunctionCall()
    stub_executor = _StubExecutor(stub_call)

    app = mcs.build_api_app(
        executor_resolver=lambda model: stub_executor,
        function_call_lookup=lambda call_id: stub_call,
    )
    return TestClient(app), mcs, tmp_path


def _headers() -> dict[str, str]:
    return {"X-Mantis-Token": "test-token", "Content-Type": "application/json"}


def _write_status(mcs, tenant_id: str, run_id: str, status: dict) -> None:
    """Write a status.json file under the synthetic data dir.

    Mirrors what the executor would write; bypasses /v1/predict so each
    test can pin the exact (phase, timestamps) input.
    """
    mcs._write_status(tenant_id, run_id, status)


def _seed_run(
    mcs,
    *,
    tenant_id: str = "default",
    run_id: str,
    status: str,
    created_at: str = "2026-06-08T00:00:00+00:00",
    updated_at: str = "2026-06-08T00:00:00+00:00",
    extra: dict | None = None,
) -> None:
    blob = {
        "run_id": run_id,
        "status": status,
        "tenant_id": tenant_id,
        "created_at": created_at,
        "updated_at": updated_at,
    }
    if extra:
        blob.update(extra)
    _write_status(mcs, tenant_id, run_id, blob)


# ── GET /v1/runs/{id} ──────────────────────────────────────────────


def test_phase_returns_404_for_unknown_run(app_ctx):
    client, _mcs, _tmp = app_ctx
    r = client.get("/v1/runs/missing-run", headers=_headers())
    assert r.status_code == 404


def test_phase_running_emits_runphase_response(app_ctx):
    client, mcs, _tmp = app_ctx
    _seed_run(mcs, run_id="run-running", status="running")
    r = client.get("/v1/runs/run-running", headers=_headers())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["run_id"] == "run-running"
    assert body["phase"] == "running"
    assert body["finished_at"] is None
    # Halt class only set on terminal phases.
    assert body["halt_class"] is None


def test_phase_queued(app_ctx):
    client, mcs, _tmp = app_ctx
    _seed_run(mcs, run_id="run-q", status="queued")
    body = client.get("/v1/runs/run-q", headers=_headers()).json()
    assert body["phase"] == "queued"


def test_phase_paused_maps_to_running(app_ctx):
    """Paused is operator state; from the lifecycle perspective the run
    is still working, so it shows as RUNNING for clients deciding
    whether to keep polling."""
    client, mcs, _tmp = app_ctx
    _seed_run(mcs, run_id="run-paused", status="paused")
    body = client.get("/v1/runs/run-paused", headers=_headers()).json()
    assert body["phase"] == "running"


def test_phase_succeeded_is_complete(app_ctx):
    client, mcs, _tmp = app_ctx
    _seed_run(mcs, run_id="run-ok", status="succeeded")
    body = client.get("/v1/runs/run-ok", headers=_headers()).json()
    assert body["phase"] == "complete"
    assert body["finished_at"] is not None


def test_phase_cancelled_carries_halt_class(app_ctx):
    client, mcs, _tmp = app_ctx
    _seed_run(mcs, run_id="run-x", status="cancelled")
    body = client.get("/v1/runs/run-x", headers=_headers()).json()
    assert body["phase"] == "cancelled"
    assert body["halt_class"] == "cancelled"


def test_phase_completed_with_failures_is_complete(app_ctx):
    """The executor surfaces a wider taxonomy than RunPhase carries —
    partial-success runs are still terminal-complete from a lifecycle
    perspective."""
    client, mcs, _tmp = app_ctx
    _seed_run(mcs, run_id="run-cwf", status="completed_with_failures")
    body = client.get("/v1/runs/run-cwf", headers=_headers()).json()
    assert body["phase"] == "complete"


def test_phase_halted_with_explicit_halt_class(app_ctx):
    client, mcs, _tmp = app_ctx
    _seed_run(
        mcs,
        run_id="run-h",
        status="halted",
        extra={"halt_class": "cf_challenge"},
    )
    body = client.get("/v1/runs/run-h", headers=_headers()).json()
    assert body["phase"] == "halted"
    assert body["halt_class"] == "cf_challenge"


# ── Backoff hint ───────────────────────────────────────────────────


def test_backoff_hint_terminal_is_long(app_ctx):
    """Terminal phases don't change — clients should back off a lot."""
    client, mcs, _tmp = app_ctx
    _seed_run(mcs, run_id="run-done", status="succeeded")
    body = client.get("/v1/runs/run-done", headers=_headers()).json()
    assert body["polling_backoff_ms_hint"] >= 10_000


def test_backoff_hint_present_and_positive_for_active(app_ctx):
    client, mcs, _tmp = app_ctx
    _seed_run(mcs, run_id="run-r", status="running")
    body = client.get("/v1/runs/run-r", headers=_headers()).json()
    assert body["polling_backoff_ms_hint"] > 0


# ── GET /v1/queue ──────────────────────────────────────────────────


def test_queue_empty_for_fresh_tenant(app_ctx):
    client, _mcs, _tmp = app_ctx
    r = client.get("/v1/queue", headers=_headers())
    assert r.status_code == 200
    body = r.json()
    assert body["queued"] == 0
    assert body["running"] == 0
    assert body["recovering"] == 0


def test_queue_counts_active_excludes_terminal(app_ctx):
    client, mcs, _tmp = app_ctx
    _seed_run(mcs, run_id="r-q1", status="queued")
    _seed_run(mcs, run_id="r-q2", status="queued")
    _seed_run(mcs, run_id="r-r1", status="running")
    _seed_run(mcs, run_id="r-rec", status="recovering")
    # Terminal runs MUST NOT be counted.
    _seed_run(mcs, run_id="r-done", status="succeeded")
    _seed_run(mcs, run_id="r-fail", status="failed")
    _seed_run(mcs, run_id="r-cancel", status="cancelled")

    body = client.get("/v1/queue", headers=_headers()).json()
    assert body["queued"] == 2
    assert body["running"] == 1
    assert body["recovering"] == 1


def test_queue_response_is_pydantic_shaped(app_ctx):
    """Validates the response matches QueueStatusResponse keys exactly."""
    client, mcs, _tmp = app_ctx
    _seed_run(mcs, run_id="r1", status="running")
    body = client.get("/v1/queue", headers=_headers()).json()
    assert set(body.keys()) == {"tenant_id", "queued", "running", "recovering", "eta_ms"}


def test_queue_tolerates_corrupt_status_files(app_ctx, tmp_path):
    """A garbage status.json must not 500 the queue endpoint —
    it should be skipped silently and other runs still counted."""
    client, mcs, _tmp = app_ctx
    _seed_run(mcs, run_id="r-ok", status="running")
    # Plant a broken status.json next to the valid ones.
    bad_dir = mcs._run_dir("default", "r-bad")
    bad_dir.mkdir(parents=True, exist_ok=True)
    (bad_dir / "status.json").write_text("{this is not valid json")
    body = client.get("/v1/queue", headers=_headers()).json()
    assert body["running"] == 1


# ── Auth ───────────────────────────────────────────────────────────


def test_phase_requires_auth(app_ctx):
    client, _mcs, _tmp = app_ctx
    r = client.get("/v1/runs/anything")
    assert r.status_code in {401, 403}


def test_queue_requires_auth(app_ctx):
    client, _mcs, _tmp = app_ctx
    r = client.get("/v1/queue")
    assert r.status_code in {401, 403}
