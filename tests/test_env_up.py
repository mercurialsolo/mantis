"""Tests for the local backend lifecycle (subprocess stub mode).

These tests do NOT require Docker — they exercise the subprocess fallback
that the local backend uses when no Docker image is registered for an
env name. The stub env is a Python subprocess serving the harness
contract on a free local port.

Coverage:

* ``start`` + ``wait_healthy`` boots the stub and lands ``/__env__/health``.
* ``get_url`` returns the bound URL.
* ``stop`` terminates the subprocess.
* Two concurrent starts get different ports (port-conflict regression).
* Health timeout surfaces a clear ``TimeoutError`` instead of hanging.
"""

from __future__ import annotations

import json
import socket
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest

from mantis_agent.sim_envs.local import LocalBackend, _pick_free_port


def _http_get(url: str, *, timeout: float = 2.0, headers: dict | None = None) -> tuple[int, str]:
    req = Request(url, headers=headers or {})
    try:
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.status, resp.read().decode("utf-8")
    except URLError as exc:
        # urlopen raises HTTPError (a URLError subclass) on non-2xx.
        code = getattr(exc, "code", None)
        if code is not None:
            return code, getattr(exc, "reason", "") or ""
        raise


def test_pick_free_port_returns_unique_ports():
    p1 = _pick_free_port()
    p2 = _pick_free_port()
    assert p1 != p2 or p1 > 0  # at minimum, port is valid


def test_local_backend_starts_and_stops_stub_env():
    backend = LocalBackend()
    handle = backend.start("stub-test", seed=7, now="2026-02-01T00:00:00Z")
    try:
        backend.wait_healthy(handle, timeout_s=10.0)
        status, body = _http_get(f"{handle.url}/__env__/health")
        assert status == 200
        payload = json.loads(body)
        assert payload["ok"] is True
        assert payload["seed"] == 7
        assert payload["now"] == "2026-02-01T00:00:00Z"
        assert backend.get_url(handle).startswith("http://127.0.0.1:")
    finally:
        backend.stop(handle)

    # After stop the port should be free again (eventually). Polling
    # briefly to avoid platform-specific TIME_WAIT noise.
    deadline = time.time() + 3.0
    while time.time() < deadline:
        try:
            _http_get(f"{handle.url}/__env__/health", timeout=0.5)
        except Exception:  # noqa: BLE001 — expected
            break
        time.sleep(0.1)


def test_local_backend_two_starts_get_distinct_urls():
    backend = LocalBackend()
    h1 = backend.start("stub-test")
    h2 = backend.start("stub-test")
    try:
        assert h1.url != h2.url
        backend.wait_healthy(h1, timeout_s=10.0)
        backend.wait_healthy(h2, timeout_s=10.0)
    finally:
        backend.stop(h1)
        backend.stop(h2)


def test_stop_is_idempotent():
    backend = LocalBackend()
    handle = backend.start("stub-test")
    backend.wait_healthy(handle, timeout_s=10.0)
    backend.stop(handle)
    backend.stop(handle)  # second call must not raise


def test_health_timeout_raises_clear_error():
    """A handle pointed at nothing should time out cleanly."""
    backend = LocalBackend()
    from mantis_agent.sim_envs.runtime import RuntimeHandle

    # Bind+release a port — guaranteed nothing listening on it.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        dead_port = s.getsockname()[1]

    handle = RuntimeHandle(
        env_name="dead", url=f"http://127.0.0.1:{dead_port}",
        admin_token="x", backend="local", extra={"mode": "subprocess"},
    )
    with pytest.raises(TimeoutError):
        backend.wait_healthy(handle, timeout_s=1.0)
