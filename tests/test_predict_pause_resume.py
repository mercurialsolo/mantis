"""End-to-end tests for HTTP pause/resume via /v1/predict (#344).

The library-side pause primitives (`PauseRequested`, `PauseState`,
`runner.resume`) are already covered by `tests/test_micro_runner_hooks.py`
and `tests/test_gym_runner_tools.py`. These tests cover the wire shape
on top of them:

* `action=status` returns `status="paused"` with `prompt` + `reason` +
  `pause_state` for a paused detached run.
* `action=resume` rehydrates the stored PauseState, kicks the worker
  back into the runner, and flips status to `running` → `succeeded`.
* Negative paths: resume on a non-paused run is 400; missing
  ``user_input`` is 400; plan-signature mismatch on resume is 400.

We mock at the ``BasetenCUARuntime._run_micro`` boundary — the runner +
brain + Chrome stack is all covered by upstream library tests, and
the goal here is the route → runtime → disk → route lifecycle.
"""

from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")

from fastapi.testclient import TestClient

from mantis_agent import tenant_auth as ta_mod
from mantis_agent.baseten_server.runtime import BasetenCUARuntime


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Single-tenant TestClient + isolated MANTIS_DATA_DIR."""
    monkeypatch.setenv("MANTIS_API_TOKEN", "test-token")
    monkeypatch.setenv("MANTIS_LLAMA_PORT", "18080")
    monkeypatch.delenv("MANTIS_TENANT_KEYS_PATH", raising=False)
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path / "mantis-data"))
    ta_mod.reset_key_store()

    # ``baseten_server/__init__.py`` exposes ``app`` and ``runtime`` via
    # PEP-562 lazy attributes; importing ``routes`` materializes both and
    # forces the singleton ``runtime`` into the lazy cache so the rest of
    # the test can reach it via ``bs.app`` / ``bs.routes.runtime``.
    from mantis_agent import baseten_server as bs
    from mantis_agent.baseten_server import routes as bs_routes

    importlib.reload(bs_routes)
    importlib.reload(bs)

    # Real ``load()`` boots Holo3 / Gemma4 — out of reach in unit tests.
    # We're mocking ``_run_micro`` directly, so the brain is never used.
    bs_routes.runtime.loaded = True
    monkeypatch.setattr(BasetenCUARuntime, "load", lambda self: None, raising=True)

    return TestClient(bs.app), bs_routes


def _auth_headers() -> dict[str, str]:
    return {"X-Mantis-Token": "test-token", "Content-Type": "application/json"}


def _minimal_payload() -> dict[str, Any]:
    """A payload that bypasses everything — pre-built suite, no decompose."""
    return {
        "detached": True,
        "task_suite": {
            "session_name": "pause-test",
            "_micro_plan": [
                {"intent": "navigate to example.com", "type": "navigate", "budget": 1},
            ],
            "_plan_signature": "fixed-sig-for-tests",
            "_max_cost": 1.0,
            "_max_time_minutes": 5,
            "tasks": [],
        },
        "max_cost": 1.0,
        "max_time_minutes": 5,
        "profile_id": "test-profile",
        "workflow_id": "test-workflow",
    }


def _stub_run_micro_first_call(self, task_suite, payload, run_id=None):
    """Replacement for BasetenCUARuntime._run_micro that always pauses."""
    return {
        "_paused": True,
        "pause_state": {
            "version": 1,
            "run_key": "test-workflow",
            "plan_signature": task_suite.get("_plan_signature", ""),
            "session_name": task_suite.get("session_name", ""),
            "step_index": 0,
            "pending_tool": "request_user_input",
            "pending_arguments": {"prompt": "Enter the 6-digit code"},
            "pending_reason": "user_input",
            "prompt": "Enter the 6-digit code",
            "step_results": [],
            "loop_counters": {},
            "listings_on_page": 0,
            "checkpoint_path": "",
            "timestamp": time.time(),
        },
        "prompt": "Enter the 6-digit code",
        "reason": "user_input",
        "leads": [],
        "run_id": run_id or "stub",
    }


def _stub_run_micro_resume(self, task_suite, payload, run_id=None):
    """Replacement that succeeds; asserts the resume path carried user_input."""
    assert payload.get("_resume_user_input") == "123456", (
        f"expected user_input='123456', got {payload.get('_resume_user_input')!r}"
    )
    assert payload.get("_resume_pause_state"), (
        "expected _resume_pause_state to be layered onto the payload"
    )
    return {
        "viable": 1,
        "leads": [{"url": "https://example.com/a"}],
        "run_id": run_id or "stub",
    }


def _wait_for_status(client, run_id: str, *, want: set[str], timeout_s: float = 5.0):
    """Spin on action=status until status ∈ want (or timeout)."""
    t0 = time.monotonic()
    last = None
    while time.monotonic() - t0 < timeout_s:
        r = client.post(
            "/v1/predict",
            headers=_auth_headers(),
            json={"action": "status", "run_id": run_id},
        )
        assert r.status_code == 200, r.text
        last = r.json()
        if last.get("status") in want:
            return last
        time.sleep(0.05)
    pytest.fail(
        f"status never reached {want} for run_id={run_id}; "
        f"last={last!r} (waited {timeout_s}s)"
    )


# ── Happy path: pause → status → resume → succeed ─────────────────


def test_status_returns_paused_with_pause_state_and_prompt(client, monkeypatch):
    test_client, bs = client
    monkeypatch.setattr(
        BasetenCUARuntime, "_run_micro", _stub_run_micro_first_call, raising=True
    )

    submit = test_client.post(
        "/v1/predict", headers=_auth_headers(), json=_minimal_payload(),
    )
    assert submit.status_code == 200, submit.text
    run_id = submit.json()["run_id"]

    status = _wait_for_status(test_client, run_id, want={"paused"})
    assert status["status"] == "paused"
    assert status["prompt"] == "Enter the 6-digit code"
    assert status["reason"] == "user_input"
    assert status["pause_state"]["pending_tool"] == "request_user_input"
    assert status["pause_state"]["prompt"] == "Enter the 6-digit code"
    assert status["pause_state"]["plan_signature"] == "fixed-sig-for-tests"


def test_resume_flips_to_running_and_passes_user_input(client, monkeypatch):
    test_client, bs = client

    # First call: pauses. Second call (after resume): asserts user_input and succeeds.
    call_count = {"n": 0}

    def _switching_stub(self, task_suite, payload, run_id=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _stub_run_micro_first_call(self, task_suite, payload, run_id)
        return _stub_run_micro_resume(self, task_suite, payload, run_id)

    monkeypatch.setattr(
        BasetenCUARuntime, "_run_micro", _switching_stub, raising=True
    )

    # Submit + wait for paused.
    submit = test_client.post(
        "/v1/predict", headers=_auth_headers(), json=_minimal_payload(),
    )
    assert submit.status_code == 200, submit.text
    run_id = submit.json()["run_id"]
    _wait_for_status(test_client, run_id, want={"paused"})

    # Resume — synchronous response returns running.
    resume = test_client.post(
        "/v1/predict",
        headers=_auth_headers(),
        json={"action": "resume", "run_id": run_id, "user_input": "123456"},
    )
    assert resume.status_code == 200, resume.text
    body = resume.json()
    assert body["status"] == "running"
    assert body["run_id"] == run_id
    assert "resumed_at" in body

    # Wait for the resumed worker to finish.
    final = _wait_for_status(test_client, run_id, want={"succeeded"})
    assert final["status"] == "succeeded"
    # Old pause surface is wiped on the resumed status.
    assert final.get("prompt", "") == ""
    assert call_count["n"] == 2


# ── Negative paths ────────────────────────────────────────────────


def test_resume_on_non_paused_run_returns_400(client, monkeypatch):
    test_client, bs = client

    # Run that succeeds immediately (no pause).
    def _ok_stub(self, task_suite, payload, run_id=None):
        return {"viable": 0, "leads": [], "run_id": run_id or "stub"}

    monkeypatch.setattr(BasetenCUARuntime, "_run_micro", _ok_stub, raising=True)
    submit = test_client.post(
        "/v1/predict", headers=_auth_headers(), json=_minimal_payload(),
    )
    run_id = submit.json()["run_id"]
    _wait_for_status(test_client, run_id, want={"succeeded"})

    resume = test_client.post(
        "/v1/predict",
        headers=_auth_headers(),
        json={"action": "resume", "run_id": run_id, "user_input": "x"},
    )
    assert resume.status_code == 400, resume.text
    assert "paused" in resume.json()["detail"].lower()


def test_resume_requires_user_input(client):
    test_client, _bs = client
    # Pydantic catches this at schema level — no need to set up a real run.
    r = test_client.post(
        "/v1/predict",
        headers=_auth_headers(),
        json={"action": "resume", "run_id": "anything"},
    )
    assert r.status_code == 400, r.text
    assert "user_input" in r.json()["detail"]


def test_resume_plan_signature_mismatch_returns_400(client, monkeypatch, tmp_path):
    test_client, bs = client
    monkeypatch.setattr(
        BasetenCUARuntime, "_run_micro", _stub_run_micro_first_call, raising=True
    )

    submit = test_client.post(
        "/v1/predict", headers=_auth_headers(), json=_minimal_payload(),
    )
    run_id = submit.json()["run_id"]
    _wait_for_status(test_client, run_id, want={"paused"})

    # Tamper with pause_state.json — change the plan signature so the
    # synchronous guard in _detached_action("resume", ...) fires. The guard
    # only applies on the LEGACY path (no checkpointed suite), so drop
    # resolved_task_suite to exercise the re-derive + guard.
    data_root = Path(bs.runtime._run_path(run_id))  # bs is bs_routes here — the singleton lives on it
    pause_path = data_root / "pause_state.json"
    blob = json.loads(pause_path.read_text())
    blob.pop("resolved_task_suite", None)
    blob["plan_signature"] = "some-other-signature-from-an-edited-plan"
    pause_path.write_text(json.dumps(blob))

    resume = test_client.post(
        "/v1/predict",
        headers=_auth_headers(),
        json={"action": "resume", "run_id": run_id, "user_input": "123456"},
    )
    assert resume.status_code == 400, resume.text
    detail = resume.json()["detail"]
    assert "signature mismatch" in detail.lower()


def test_resume_restores_checkpoint_suite_ignoring_redecompose_drift(client, monkeypatch):
    """End-user bug fix: when the run paused with a resolved micro-suite in its
    checkpoint, resume restores it VERBATIM — so a plan that would re-decompose
    to a different signature (cache miss across replicas) no longer wedges the
    run in 'paused'. The signature guard is bypassed because there's no
    re-derivation."""
    test_client, bs = client
    call_count = {"n": 0}

    def _switching_stub(self, task_suite, payload, run_id=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _stub_run_micro_first_call(self, task_suite, payload, run_id)
        # On resume, assert we got the checkpointed suite verbatim.
        assert task_suite.get("_plan_signature") == "fixed-sig-for-tests"
        return _stub_run_micro_resume(self, task_suite, payload, run_id)

    monkeypatch.setattr(BasetenCUARuntime, "_run_micro", _switching_stub, raising=True)

    submit = test_client.post(
        "/v1/predict", headers=_auth_headers(), json=_minimal_payload(),
    )
    run_id = submit.json()["run_id"]
    _wait_for_status(test_client, run_id, want={"paused"})

    # The checkpoint persisted the resolved suite at pause.
    pause_path = Path(bs.runtime._run_path(run_id)) / "pause_state.json"
    blob = json.loads(pause_path.read_text())
    assert blob.get("resolved_task_suite", {}).get("_plan_signature") == "fixed-sig-for-tests"

    # Simulate non-deterministic re-decompose drift: tamper the stored signature
    # AND the saved payload so a re-derive WOULD mismatch. With the checkpointed
    # suite present, resume must ignore both and succeed.
    blob["plan_signature"] = "would-mismatch-if-rederived"
    pause_path.write_text(json.dumps(blob))

    resume = test_client.post(
        "/v1/predict",
        headers=_auth_headers(),
        json={"action": "resume", "run_id": run_id, "user_input": "123456"},
    )
    assert resume.status_code == 200, resume.text  # not 400 — no wedge
    final = _wait_for_status(test_client, run_id, want={"succeeded"})
    assert final["status"] == "succeeded"


def test_resume_unknown_run_returns_404(client):
    test_client, _bs = client
    r = test_client.post(
        "/v1/predict",
        headers=_auth_headers(),
        json={"action": "resume", "run_id": "does-not-exist", "user_input": "x"},
    )
    # Reading status.json on a missing run dir raises FileNotFoundError →
    # the route handler maps that to 404.
    assert r.status_code == 404, r.text
