"""Tests for the Modal HTTP endpoint (#342).

Drives the FastAPI app through ``TestClient`` with a mocked executor
spawner, so we exercise the full request lifecycle (validate → lock →
spawn → return run_id) without going through Modal's runtime.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

# ``deploy/modal/modal_cua_server.py`` imports the ``modal`` SDK at module
# scope. It's a deploy-time dependency, not a test dependency, so CI
# environments often don't have it installed. Skip cleanly rather than
# blowing up at collection.
pytest.importorskip("modal")

from fastapi.testclient import TestClient

from mantis_agent import tenant_auth as ta_mod


# Modal app modules under deploy/modal/ aren't on sys.path by default.
_DEPLOY_MODAL = Path(__file__).resolve().parent.parent / "deploy" / "modal"
if str(_DEPLOY_MODAL) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_MODAL))


class _StubFunctionCall:
    """Stand-in for ``modal.Function.spawn(...)`` and ``FunctionCall``."""

    def __init__(self, result: Any = None, raise_on_get: Exception | None = None) -> None:
        self.object_id = "fc-stub-123"
        self._result = result
        self._raise = raise_on_get

    def get(self, timeout: float = 0.1) -> Any:
        if self._raise is not None:
            raise self._raise
        return self._result

    def cancel(self) -> None:
        return None


class _StubExecutor:
    """Stand-in for an @app.function() executor."""

    def __init__(self, call: _StubFunctionCall) -> None:
        self.call = call
        self.spawn_kwargs: dict[str, Any] = {}

    def spawn(self, *, task_file_contents: str, **kwargs) -> _StubFunctionCall:
        self.spawn_kwargs = {"task_file_contents": task_file_contents, **kwargs}
        return self.call


@pytest.fixture
def app_with_stub(monkeypatch, tmp_path):
    """FastAPI app wired with a stub executor + isolated data dir."""
    monkeypatch.setenv("MANTIS_API_TOKEN", "test-token")
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path / "mantis-data"))
    monkeypatch.delenv("MANTIS_TENANT_KEYS_PATH", raising=False)
    ta_mod.reset_key_store()

    import modal_cua_server as mcs  # type: ignore[import-not-found]
    importlib.reload(mcs)

    stub_call = _StubFunctionCall(result={"viable": 3, "status": "succeeded"})
    stub_executor = _StubExecutor(stub_call)

    app = mcs.build_api_app(
        executor_resolver=lambda model: stub_executor,
        function_call_lookup=lambda call_id: stub_call,
    )
    return TestClient(app), stub_executor, stub_call


def _auth_headers() -> dict[str, str]:
    return {"X-Mantis-Token": "test-token", "Content-Type": "application/json"}


# ── Happy path ──────────────────────────────────────────────────────


def test_health_ok(app_with_stub) -> None:
    client, _exec, _call = app_with_stub
    r = client.get("/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_run_id(app_with_stub) -> None:
    client, executor, _call = app_with_stub
    r = client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({
            "task_suite": {"tasks": [{"intent": "x"}]},
            "profile_id": "alice",
            "workflow_id": "plan_v1",
        }),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "queued"
    assert body["mode"] == "detached"
    assert body["run_id"]
    # Identity round-trips through the response envelope.
    assert body["payload"]["profile_id"].endswith("__alice")
    assert body["payload"]["workflow_id"].endswith("__plan_v1")
    # Stub executor was actually spawned with the JSON-encoded suite.
    assert "task_file_contents" in executor.spawn_kwargs
    parsed = json.loads(executor.spawn_kwargs["task_file_contents"])
    assert parsed["tasks"] == [{"intent": "x"}]


def test_predict_rejects_missing_token(app_with_stub) -> None:
    client, _executor, _call = app_with_stub
    r = client.post(
        "/v1/predict",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"task_suite": {"tasks": []}}),
    )
    assert r.status_code == 401


# ── Profile-lock 409 (#342 core motivation) ─────────────────────────


def test_concurrent_same_profile_returns_409(app_with_stub) -> None:
    client, _executor, _call = app_with_stub
    # First submission acquires the lock.
    r1 = client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({
            "task_suite": {"tasks": []},
            "profile_id": "alice",
            "workflow_id": "plan_v1",
        }),
    )
    assert r1.status_code == 200, r1.text
    held_run_id = r1.json()["run_id"]

    # Second submission with the same profile_id hits the lock → 409.
    r2 = client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({
            "task_suite": {"tasks": []},
            "profile_id": "alice",  # same profile
            "workflow_id": "plan_v2",  # different workflow doesn't matter — Chrome locks at the profile level
        }),
    )
    assert r2.status_code == 409
    # The 409 surfaces the conflicting run_id so the caller can poll it.
    assert held_run_id in r2.text


def test_different_profiles_run_in_parallel(app_with_stub) -> None:
    """Distinct profile_ids must not serialize each other."""
    client, _executor, _call = app_with_stub
    r1 = client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({"task_suite": {"tasks": []}, "profile_id": "alice"}),
    )
    r2 = client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({"task_suite": {"tasks": []}, "profile_id": "bob"}),
    )
    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text
    assert r1.json()["run_id"] != r2.json()["run_id"]


# ── Status / result polling ────────────────────────────────────────


def test_status_action_returns_succeeded_after_get(app_with_stub) -> None:
    client, _executor, _call = app_with_stub
    r = client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({"task_suite": {"tasks": []}, "profile_id": "alice"}),
    )
    run_id = r.json()["run_id"]

    # Poll via action=status — stub .get() returns the result immediately.
    poll = client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({"action": "status", "run_id": run_id}),
    )
    assert poll.status_code == 200, poll.text
    body = poll.json()
    assert body["status"] == "succeeded"
    assert body["run_id"] == run_id


def test_result_action_returns_executor_result(app_with_stub) -> None:
    client, _executor, call = app_with_stub
    call._result = {"viable": 42, "leads": ["alice"]}
    r = client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({"task_suite": {"tasks": []}, "profile_id": "alice"}),
    )
    run_id = r.json()["run_id"]
    poll = client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({"action": "result", "run_id": run_id}),
    )
    assert poll.status_code == 200, poll.text
    body = poll.json()
    assert body["status"] == "succeeded"
    assert body["result"] == {"viable": 42, "leads": ["alice"]}


def test_viewer_url_side_channel_merges_into_status(app_with_stub) -> None:
    """#416 follow-up: the executor writes ``viewer.json`` (not status.json),
    and ``action=status`` merges it in without dropping API-owned fields.
    """
    import modal_cua_server as mcs  # type: ignore[import-not-found]

    client, _executor, _call = app_with_stub
    r = client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({
            "task_suite": {"tasks": []},
            "profile_id": "alice",
            "live_viewer": True,
        }),
    )
    run_id = r.json()["run_id"]

    fake_url = "https://viewer.modal.host?token=abc"
    mcs._write_viewer_url("default", run_id, fake_url)

    poll = client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({"action": "status", "run_id": run_id}),
    )
    assert poll.status_code == 200, poll.text
    body = poll.json()
    # Viewer URL surfaces …
    assert body["viewer_url"] == fake_url
    # … without clobbering the API-owned status fields.
    assert body["run_id"] == run_id
    assert body["modal_call_id"] == "fc-stub-123"
    assert body["profile_id"].endswith("__alice")
    assert body["status"] in {"queued", "running", "succeeded"}


def test_status_unknown_run_returns_404(app_with_stub) -> None:
    client, _executor, _call = app_with_stub
    r = client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({"action": "status", "run_id": "does-not-exist"}),
    )
    assert r.status_code == 404


# ── #347: pause / resume over the Modal endpoint ──────────────────


def _paused_envelope(prompt: str = "Enter the 6-digit code") -> dict:
    """The shape the Holo3 executor returns when PauseRequested fires."""
    return {
        "_paused": True,
        "pause_state": {
            "version": 1,
            "run_key": "alice-test",
            "plan_signature": "fixed-sig-for-tests",
            "session_name": "modal-pause-test",
            "step_index": 0,
            "pending_tool": "request_user_input",
            "pending_arguments": {"prompt": prompt},
            "pending_reason": "user_input",
            "prompt": prompt,
            "step_results": [],
            "loop_counters": {},
            "listings_on_page": 0,
            "checkpoint_path": "",
            "timestamp": 1234.5,
        },
        "prompt": prompt,
        "reason": "user_input",
        "mode": "micro",
        "viable": 0,
    }


def test_status_returns_paused_with_pause_state(app_with_stub) -> None:
    """Status poll on a paused executor returns the full pause envelope."""
    client, _executor, call = app_with_stub
    call._result = _paused_envelope("Enter the 6-digit code")

    r = client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({"task_suite": {"tasks": []}, "profile_id": "alice"}),
    )
    assert r.status_code == 200, r.text
    run_id = r.json()["run_id"]

    poll = client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({"action": "status", "run_id": run_id}),
    )
    assert poll.status_code == 200, poll.text
    body = poll.json()
    assert body["status"] == "paused"
    assert body["prompt"] == "Enter the 6-digit code"
    assert body["reason"] == "user_input"
    assert body["pause_state"]["pending_tool"] == "request_user_input"
    assert body["pause_state"]["plan_signature"] == "fixed-sig-for-tests"


def test_resume_layers_user_input_into_respawn(app_with_stub) -> None:
    """action=resume re-spawns the executor with pause_state + user_input baked into the suite."""
    client, executor, call = app_with_stub
    call._result = _paused_envelope("Enter the 6-digit code")

    submit = client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({"task_suite": {"tasks": []}, "profile_id": "alice"}),
    )
    run_id = submit.json()["run_id"]
    # Drive the FunctionCall to terminal-paused.
    client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({"action": "status", "run_id": run_id}),
    )

    # Now flip the stub to return success on the next .get() — the resume
    # path's fresh .spawn() returns the same stub call (executor reuses
    # the same _StubFunctionCall instance), so the next status poll lands
    # on succeeded.
    call._result = {"viable": 1, "leads": [{"url": "https://example.com/a"}]}

    resume = client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({"action": "resume", "run_id": run_id, "user_input": "123456"}),
    )
    assert resume.status_code == 200, resume.text
    body = resume.json()
    assert body["status"] == "running"
    assert body["run_id"] == run_id
    assert "resumed_at" in body

    # The fresh spawn must have layered the resume hints onto the suite.
    spawned_suite = json.loads(executor.spawn_kwargs["task_file_contents"])
    assert spawned_suite["_resume_user_input"] == "123456"
    assert spawned_suite["_resume_pause_state"]["pending_tool"] == "request_user_input"

    # Status poll picks up the new succeeded result.
    final = client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({"action": "status", "run_id": run_id}),
    )
    assert final.json()["status"] == "succeeded"


def test_resume_on_non_paused_run_returns_400(app_with_stub) -> None:
    client, _executor, _call = app_with_stub  # default stub returns success immediately
    r = client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({"task_suite": {"tasks": []}, "profile_id": "alice"}),
    )
    run_id = r.json()["run_id"]
    # Drive to succeeded.
    client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({"action": "status", "run_id": run_id}),
    )

    resume = client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({"action": "resume", "run_id": run_id, "user_input": "x"}),
    )
    assert resume.status_code == 400, resume.text
    assert "paused" in resume.json()["detail"].lower()


def test_resume_requires_user_input(app_with_stub) -> None:
    client, _executor, _call = app_with_stub
    # Pydantic catches the missing user_input — no real run setup needed.
    r = client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({"action": "resume", "run_id": "anything"}),
    )
    assert r.status_code == 400, r.text
    assert "user_input" in r.json()["detail"]


def test_paused_run_releases_lock_so_resume_can_reacquire(app_with_stub) -> None:
    """A paused run releases the profile lock; resume re-acquires it."""
    client, _executor, call = app_with_stub
    call._result = _paused_envelope("OTP?")
    submit = client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({"task_suite": {"tasks": []}, "profile_id": "alice"}),
    )
    run_id = submit.json()["run_id"]
    client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({"action": "status", "run_id": run_id}),
    )

    # While paused, the profile lock is released — a different run_id
    # under the same profile_id can grab it.
    call._result = {"viable": 0, "leads": []}
    new_run = client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({"task_suite": {"tasks": []}, "profile_id": "alice"}),
    )
    assert new_run.status_code == 200, new_run.text
    # And resuming the PAUSED run can no longer re-acquire the lock now
    # that a fresh run holds it — caller gets 409.
    resume = client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({"action": "resume", "run_id": run_id, "user_input": "123"}),
    )
    assert resume.status_code == 409, resume.text


def test_succeeded_run_releases_profile_lock(app_with_stub) -> None:
    """Once a run finishes, its profile_id should be free for re-use."""
    client, _executor, _call = app_with_stub
    r1 = client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({"task_suite": {"tasks": []}, "profile_id": "alice"}),
    )
    run_id_1 = r1.json()["run_id"]
    # Poll to terminal — stub returns immediately, lock released.
    client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({"action": "status", "run_id": run_id_1}),
    )
    # Same profile_id can now start a new run.
    r2 = client.post(
        "/v1/predict",
        headers=_auth_headers(),
        data=json.dumps({"task_suite": {"tasks": []}, "profile_id": "alice"}),
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["run_id"] != run_id_1
