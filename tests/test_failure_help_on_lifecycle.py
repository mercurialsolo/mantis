"""Tests for the lifecycle endpoint surfacing ``failure_help`` (#841).

The Modal lifecycle endpoint (``GET /v1/runs/{id}``) should attach the
``failure_help`` dict to its response on terminal halted / cancelled
phases. Two paths:

1. The status file already has ``failure_help`` written by the
   failure path. Use it verbatim.
2. The status file has only a ``halt_class`` (older failure paths or
   manually-written status). Synthesize the help from the taxonomy.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

pytest.importorskip("modal")

from fastapi.testclient import TestClient  # noqa: E402 — guarded by importorskip

from mantis_agent import tenant_auth as ta_mod


_DEPLOY_MODAL = Path(__file__).resolve().parent.parent / "deploy" / "modal"
if str(_DEPLOY_MODAL) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_MODAL))


@pytest.fixture
def mcs(monkeypatch, tmp_path):
    monkeypatch.setenv("MANTIS_API_TOKEN", "test-token")
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path / "mantis-data"))
    monkeypatch.delenv("MANTIS_TENANT_KEYS_PATH", raising=False)
    monkeypatch.delenv("MANTIS_DEPLOY_AGE_WARN", raising=False)
    ta_mod.reset_key_store()

    import modal_cua_server as mod  # type: ignore[import-not-found]
    importlib.reload(mod)
    # #866: per-test plain-dict cross-replica store so leakage between
    # cases can't make a regression look like a pass.
    from mantis_agent.run_state_store import RunStateStore
    mod._RUN_STATE_STORE = RunStateStore(backing={})
    mod._RUN_STATE_STORE_INIT = True
    return mod


def _h() -> dict:
    return {"X-Mantis-Token": "test-token", "Content-Type": "application/json"}


class _StubCall:
    object_id = "fc-stub"

    def get(self, timeout: float = 0.1):
        return {}

    def cancel(self) -> None:
        return None


class _StubExecutor:
    def spawn(self, *, task_file_contents, **kwargs):
        return _StubCall()


def test_lifecycle_response_includes_failure_help_when_status_has_it(mcs):
    """The failure path stamped ``failure_help`` on status.json directly —
    the lifecycle endpoint must surface it verbatim."""
    executor = _StubExecutor()
    app = mcs.build_api_app(
        executor_resolver=lambda model: executor,
        function_call_lookup=lambda call_id: executor,
    )
    client = TestClient(app)

    mcs._write_status("default", "r-fh", {
        "run_id": "r-fh",
        "status": "failed",
        "halt_class": "anthropic_unreachable",
        "error": "ConnectionResetError(104, 'Connection reset by peer')",
        "failure_help": {
            "halt_class": "anthropic_unreachable",
            "summary": "Could not reach the Anthropic API",
            "likely_causes": ["Anthropic 5xx", "rate limit"],
            "next_steps": ["wait 60s and retry"],
            "debug_surfaces": {"events": "/v1/runs/r-fh/events?sse=true"},
        },
        "updated_at": "2026-06-10T20:00:00+00:00",
    })

    r = client.get("/v1/runs/r-fh", headers=_h())
    assert r.status_code == 200
    body = r.json()
    # halted phase derived from `failed` status string.
    assert body["phase"] == "halted"
    assert body["failure_help"]["halt_class"] == "anthropic_unreachable"
    assert "Could not reach" in body["failure_help"]["summary"]


def test_lifecycle_synthesizes_failure_help_from_halt_class(mcs):
    """Older failure paths may have written only halt_class. The
    endpoint must synthesize the help dict from the taxonomy so the
    caller still gets actionable text."""
    executor = _StubExecutor()
    app = mcs.build_api_app(
        executor_resolver=lambda model: executor,
        function_call_lookup=lambda call_id: executor,
    )
    client = TestClient(app)

    mcs._write_status("default", "r-cf", {
        "run_id": "r-cf",
        "status": "halted",
        "halt_class": "cf_challenge",
        "updated_at": "2026-06-10T20:00:00+00:00",
    })

    r = client.get("/v1/runs/r-cf", headers=_h())
    assert r.status_code == 200
    body = r.json()
    assert body["phase"] == "halted"
    fh = body["failure_help"]
    assert fh["halt_class"] == "cf_challenge"
    assert fh["next_steps"]
    assert "viewer" in " ".join(fh["next_steps"]).lower()


def test_lifecycle_does_not_attach_failure_help_on_running(mcs):
    """failure_help only on terminal halted / cancelled — running
    runs don't need it."""
    executor = _StubExecutor()
    app = mcs.build_api_app(
        executor_resolver=lambda model: executor,
        function_call_lookup=lambda call_id: executor,
    )
    client = TestClient(app)
    mcs._write_status("default", "r-run", {
        "run_id": "r-run", "status": "running",
        "halt_class": "cf_challenge",  # irrelevant on running
        "updated_at": "2026-06-10T20:00:00+00:00",
    })
    body = client.get("/v1/runs/r-run", headers=_h()).json()
    assert body["phase"] == "running"
    assert "failure_help" not in body


def test_lifecycle_no_failure_help_when_no_halt_class(mcs):
    """A halted run with no halt_class and no pre-stamped help → no
    failure_help in the response (rather than the fallback default)."""
    executor = _StubExecutor()
    app = mcs.build_api_app(
        executor_resolver=lambda model: executor,
        function_call_lookup=lambda call_id: executor,
    )
    client = TestClient(app)
    mcs._write_status("default", "r-noclass", {
        "run_id": "r-noclass",
        "status": "succeeded",  # complete phase
        "updated_at": "2026-06-10T20:00:00+00:00",
    })
    body = client.get("/v1/runs/r-noclass", headers=_h()).json()
    assert body["phase"] == "complete"
    assert "failure_help" not in body


def test_lifecycle_carries_failure_help_on_complete_with_error(mcs):
    """``completed_with_failures`` runs that still have an error
    string deserve the help payload — they're partial-failures
    from the caller's perspective."""
    executor = _StubExecutor()
    app = mcs.build_api_app(
        executor_resolver=lambda model: executor,
        function_call_lookup=lambda call_id: executor,
    )
    client = TestClient(app)
    mcs._write_status("default", "r-cwf", {
        "run_id": "r-cwf",
        "status": "completed_with_failures",
        "halt_class": "extract_data_failed",
        "error": "some steps failed",
        "updated_at": "2026-06-10T20:00:00+00:00",
    })
    body = client.get("/v1/runs/r-cwf", headers=_h()).json()
    assert body["phase"] == "complete"
    assert "failure_help" in body
    assert body["failure_help"]["halt_class"] == "extract_data_failed"
