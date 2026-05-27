"""End-to-end tests for the `ComputerAgent` FastAPI server (#698).

Uses FastAPI's `TestClient`. The Chrome / Xvfb stack is monkeypatched
out — these tests verify the wire contract layer (session binding, LRU
dedup, header gating) without spawning real subprocesses. A separate
integration test (`tests/integration/test_phase1_e2e.py`) covers the
real-Chrome path in a staging Modal app.
"""

from __future__ import annotations

import base64
import io
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture()
def fake_env() -> MagicMock:
    """A drop-in replacement for the `XdotoolGymEnv` instance the
    ComputerAgent owns. Provides only the surface the handlers touch."""

    env = MagicMock()
    env._env = {"DISPLAY": ":99"}
    env._browser_proc = MagicMock(pid=4242)

    # Screenshot — return a tiny PNG.
    img = Image.new("RGB", (8, 8), "white")
    env.screenshot.return_value = img

    # CDP scrollY readback.
    env._cdp_call.return_value = (True, {"result": {"value": 0}})

    return env


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, fake_env: MagicMock) -> TestClient:
    """Fresh app + state per test (otherwise sessions persist across cases)."""
    from mantis_agent.server import computer_agent as ca

    # Reset module state before each test.
    ca._state.session = None

    # Stop the handler from constructing a real XdotoolGymEnv.
    monkeypatch.setattr(ca, "_new_xdotool_env", lambda req: fake_env)

    # Build a fresh app so /openapi etc. cache nothing stale.
    app = ca.build_app()
    return TestClient(app)


@pytest.fixture()
def init_payload() -> dict[str, Any]:
    return {
        "tenant_id": "acme",
        "profile_id": "p1",
        "run_id": "r1",
        "enable_cdp": False,
        "viewport": [1280, 720],
    }


# ── health ────────────────────────────────────────────────────────────


def test_health_returns_ok_no_session(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["session_token"] is None


# ── session/init ──────────────────────────────────────────────────────


def test_session_init_returns_token_and_pid(
    client: TestClient, init_payload: dict[str, Any]
) -> None:
    r = client.post("/session/init", json=init_payload)
    assert r.status_code == 200
    body = r.json()
    assert body["session_token"]
    assert body["chrome_pid"] == 4242
    assert body["xvfb_display"] == ":99"


def test_session_init_idempotent_on_run_id(
    client: TestClient, init_payload: dict[str, Any]
) -> None:
    r1 = client.post("/session/init", json=init_payload)
    r2 = client.post("/session/init", json=init_payload)
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["session_token"] == r2.json()["session_token"]


def test_session_init_409_on_different_run_id(
    client: TestClient, init_payload: dict[str, Any]
) -> None:
    client.post("/session/init", json=init_payload)
    other = dict(init_payload, run_id="r2")
    r = client.post("/session/init", json=other)
    assert r.status_code == 409
    assert "r1" in r.json()["detail"]
    assert "r2" in r.json()["detail"]


# ── session/close ─────────────────────────────────────────────────────


def test_session_close_clears_state(
    client: TestClient, init_payload: dict[str, Any]
) -> None:
    init = client.post("/session/init", json=init_payload).json()
    r = client.post(
        "/session/close", json={"session_token": init["session_token"]}
    )
    assert r.status_code == 200 and r.json() == {"closed": True}
    # Subsequent screenshot with the stale token must 401.
    r2 = client.post(
        "/screenshot",
        json={"format": "png"},
        headers={"X-Mantis-Session": init["session_token"]},
    )
    assert r2.status_code == 401


def test_session_close_with_bad_token_is_noop(
    client: TestClient, init_payload: dict[str, Any]
) -> None:
    client.post("/session/init", json=init_payload)
    r = client.post("/session/close", json={"session_token": "wrong"})
    assert r.status_code == 200 and r.json() == {"closed": False}


# ── auth ──────────────────────────────────────────────────────────────


def test_screenshot_requires_session_header(client: TestClient) -> None:
    r = client.post("/screenshot", json={"format": "png"})
    assert r.status_code == 401


def test_xdotool_requires_session_header(client: TestClient) -> None:
    r = client.post(
        "/xdotool", json={"argv": ["mousemove", "0", "0"], "step_id": "s"}
    )
    assert r.status_code == 401


# ── screenshot ────────────────────────────────────────────────────────


def test_screenshot_returns_b64_png(
    client: TestClient, init_payload: dict[str, Any]
) -> None:
    tok = client.post("/session/init", json=init_payload).json()["session_token"]
    r = client.post(
        "/screenshot",
        json={"format": "png"},
        headers={"X-Mantis-Session": tok},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["width"] == 8 and body["height"] == 8
    img = Image.open(io.BytesIO(base64.b64decode(body["image_b64"])))
    assert img.size == (8, 8)


# ── xdotool dedup ─────────────────────────────────────────────────────


def test_xdotool_dispatches_subprocess(
    client: TestClient,
    init_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    class _Proc:
        returncode = 0
        stdout = b""
        stderr = b""

    def _run(argv: list[str], **_kw: Any) -> _Proc:
        captured.append(argv)
        return _Proc()

    monkeypatch.setattr(
        "mantis_agent.server.computer_agent.subprocess.run", _run
    )

    tok = client.post("/session/init", json=init_payload).json()["session_token"]
    r = client.post(
        "/xdotool",
        json={"argv": ["mousemove", "100", "200"], "step_id": "step-1"},
        headers={"X-Mantis-Session": tok},
    )
    assert r.status_code == 200
    assert r.json()["returncode"] == 0
    assert r.json()["deduplicated"] is False
    assert captured == [["xdotool", "mousemove", "100", "200"]]


def test_xdotool_dedup_returns_cached_on_repeated_step_id(
    client: TestClient,
    init_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second call with the same step_id returns the cached result and
    does NOT spawn a second subprocess — this is what makes retries safe."""
    spawn_count = 0

    class _Proc:
        returncode = 0
        stdout = b""
        stderr = b""

    def _run(_argv: list[str], **_kw: Any) -> _Proc:
        nonlocal spawn_count
        spawn_count += 1
        return _Proc()

    monkeypatch.setattr(
        "mantis_agent.server.computer_agent.subprocess.run", _run
    )

    tok = client.post("/session/init", json=init_payload).json()["session_token"]
    headers = {"X-Mantis-Session": tok}
    body = {"argv": ["key", "Return"], "step_id": "step-X"}

    r1 = client.post("/xdotool", json=body, headers=headers)
    r2 = client.post("/xdotool", json=body, headers=headers)

    assert spawn_count == 1
    assert r1.json()["deduplicated"] is False
    assert r2.json()["deduplicated"] is True


def test_xdotool_new_step_id_spawns_again(
    client: TestClient,
    init_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawn_count = 0

    class _Proc:
        returncode = 0
        stdout = b""
        stderr = b""

    def _run(_argv: list[str], **_kw: Any) -> _Proc:
        nonlocal spawn_count
        spawn_count += 1
        return _Proc()

    monkeypatch.setattr(
        "mantis_agent.server.computer_agent.subprocess.run", _run
    )

    tok = client.post("/session/init", json=init_payload).json()["session_token"]
    headers = {"X-Mantis-Session": tok}
    client.post(
        "/xdotool", json={"argv": ["key", "a"], "step_id": "s1"}, headers=headers
    )
    client.post(
        "/xdotool", json={"argv": ["key", "b"], "step_id": "s2"}, headers=headers
    )
    assert spawn_count == 2


# ── cdp gating ────────────────────────────────────────────────────────


def test_cdp_403_when_disabled(
    client: TestClient, init_payload: dict[str, Any]
) -> None:
    tok = client.post("/session/init", json=init_payload).json()["session_token"]
    r = client.post(
        "/cdp",
        json={"expression": "window.scrollY", "await_promise": False, "step_id": "x"},
        headers={"X-Mantis-Session": tok},
    )
    assert r.status_code == 403


def test_cdp_200_when_enabled(
    client: TestClient,
    init_payload: dict[str, Any],
    fake_env: MagicMock,
) -> None:
    init_payload = dict(init_payload, enable_cdp=True)
    fake_env._cdp_call.return_value = (True, {"result": {"value": 100}})
    tok = client.post("/session/init", json=init_payload).json()["session_token"]
    r = client.post(
        "/cdp",
        json={"expression": "window.scrollY", "await_promise": False, "step_id": "x"},
        headers={"X-Mantis-Session": tok},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["returncode"] == 0
    assert '"value": 100' in body["result_json"]
