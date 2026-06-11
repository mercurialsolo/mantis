"""Tests for the submit→poll race condition fix.

User feedback (HN sample): "A run can return a ``run_id``, then status
says ``unknown run_id``, then it comes back as ``running/succeeded``."

Root cause: Modal Volume commit + reload has a small eventual-
consistency window. If the client polls immediately after submit and
lands on a different container, the volume mount on that container
might not have seen the ``status.json`` we just wrote — so the poll
returns 404.

Fix (#866): a cross-replica run-state store (``modal.Dict`` in prod,
plain ``dict`` in tests) queried ahead of the file-backed read. When
the executor (different container) later writes a fresher status to
the volume, the file read trumps the cache.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

pytest.importorskip("modal")

from fastapi.testclient import TestClient  # noqa: E402 — guarded by importorskip

from mantis_agent import tenant_auth as ta_mod
from mantis_agent.run_state_store import KIND_STATUS, RunStateStore


_DEPLOY_MODAL = Path(__file__).resolve().parent.parent / "deploy" / "modal"
if str(_DEPLOY_MODAL) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_MODAL))


@pytest.fixture
def mcs(monkeypatch, tmp_path):
    monkeypatch.setenv("MANTIS_API_TOKEN", "test-token")
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path / "mantis-data"))
    monkeypatch.delenv("MANTIS_TENANT_KEYS_PATH", raising=False)
    ta_mod.reset_key_store()

    import modal_cua_server as mod  # type: ignore[import-not-found]
    importlib.reload(mod)
    # Install a fresh per-test plain-dict store so leakage between
    # cases doesn't make a regression look like a pass.
    store = RunStateStore(backing={})
    mod._RUN_STATE_STORE = store
    mod._RUN_STATE_STORE_INIT = True
    return mod


def test_write_status_seeds_store(mcs):
    """``_write_status`` must put the status in the cross-replica
    store so the same container's poll-immediately-after path hits it."""
    status = {"run_id": "r1", "status": "queued", "updated_at": "2026-06-09T20:00:00+00:00"}
    mcs._write_status("default", "r1", status)
    store = mcs._get_run_state_store()
    cached = store.get("default", "r1", KIND_STATUS)
    assert cached is not None
    assert cached["status"] == "queued"


def test_read_status_returns_cached_when_volume_empty(mcs, tmp_path):
    """The cache must satisfy a poll even when the on-disk status.json
    isn't there yet (volume eventual-consistency window)."""
    status = {"run_id": "r-fresh", "status": "queued", "updated_at": "2026-06-09T20:00:00+00:00"}
    mcs._get_run_state_store().put("default", "r-fresh", KIND_STATUS, status)
    # Confirm the file doesn't exist.
    assert not (mcs._run_dir("default", "r-fresh") / "status.json").exists()
    got = mcs._read_status("default", "r-fresh")
    assert got is not None
    assert got["status"] == "queued"


def test_read_status_returns_none_for_truly_unknown(mcs):
    """A run id with no cache entry and no file → None, not a fake
    cache hit from a previous run."""
    assert mcs._read_status("default", "never-submitted") is None


def test_on_disk_newer_wins_over_cache(mcs):
    """The executor writes the status from a different container; on
    next read the file should win and refresh the cache."""
    store = mcs._get_run_state_store()
    # Seed cache with an older queued state.
    store.put("default", "r1", KIND_STATUS, {
        "run_id": "r1",
        "status": "queued",
        "updated_at": "2026-06-09T20:00:00+00:00",
    })
    # Executor flips it to running via the file (simulating a
    # different-container write that the API hasn't cached).
    _direct_file_write(
        mcs, "default", "r1", {
            "run_id": "r1",
            "status": "running",
            "updated_at": "2026-06-09T20:00:30+00:00",
        },
    )
    got = mcs._read_status("default", "r1")
    assert got is not None
    assert got["status"] == "running"
    # Cache should have been refreshed.
    refreshed = store.get("default", "r1", KIND_STATUS)
    assert refreshed is not None
    assert refreshed["status"] == "running"


def _direct_file_write(mcs, tenant_id: str, run_id: str, payload: dict) -> None:
    """Bypass _write_status so the cache isn't pre-warmed — simulates
    an executor-container write that the API didn't see."""
    import json as _json
    run_dir = mcs._run_dir(tenant_id, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.json").write_text(_json.dumps(payload, indent=2))


# ── End-to-end: submit→immediate-poll never returns 404 ────────────


class _StubCall:
    object_id = "fc-stub-race"

    def get(self, timeout: float = 0.1):
        return {}

    def cancel(self) -> None:
        return None


class _StubExecutor:
    def __init__(self) -> None:
        self.call = _StubCall()

    def spawn(self, *, task_file_contents, **kwargs):
        return self.call


def test_immediate_poll_after_submit_does_not_return_404(mcs):
    """End-to-end: submit, then poll the cheap-poll lifecycle endpoint
    in the very next request — must NOT see "unknown run_id" even
    though the volume commit may not have fully propagated."""
    import json

    executor = _StubExecutor()
    store = RunStateStore(backing={})
    app = mcs.build_api_app(
        executor_resolver=lambda model: executor,
        function_call_lookup=lambda call_id: executor.call,
        run_state_store=store,
    )
    client = TestClient(app)
    h = {"X-Mantis-Token": "test-token", "Content-Type": "application/json"}

    r = client.post(
        "/v1/predict",
        headers=h,
        data=json.dumps({
            "task_suite": {"_micro_plan": [{"intent": "x", "type": "navigate"}]},
            "profile_id": "alice",
            "workflow_id": "plan_v1",
        }),
    )
    assert r.status_code == 200, r.text
    rid = r.json()["run_id"]

    # Immediately poll lifecycle.
    poll = client.get(f"/v1/runs/{rid}", headers=h)
    assert poll.status_code == 200, poll.text
    body = poll.json()
    assert body["run_id"] == rid
    assert body["phase"] in {"queued", "running"}
