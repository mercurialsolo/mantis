"""Tests for the brain-side ``SessionRouterClient`` (Phase 1.5, #846, PR 3).

Drives the client against a stubbed HTTP session — no Modal, no
network. Covers happy path, 404, 429, 504, network failure, and the
quiet-close semantics.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import requests

from mantis_agent.session_wire import (
    SessionCreateRequest,
    SessionCreateResponse,
    SessionNotFoundError,
    SessionQuotaExceededError,
    SessionRouterError,
    SessionUnreachableError,
)
from mantis_agent.server.session_router_client import SessionRouterClient


class _StubResp:
    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> Any:
        if isinstance(self._body, str):
            return json.loads(self._body)
        return self._body

    @property
    def text(self) -> str:
        if isinstance(self._body, str):
            return self._body
        return json.dumps(self._body)


class _StubHttp:
    """Captures the post/delete calls so tests can assert on them."""

    def __init__(self, post_resp: _StubResp | Exception, delete_resp: _StubResp | Exception | None = None) -> None:
        self._post_resp = post_resp
        self._delete_resp = delete_resp
        self.post_calls: list[dict] = []
        self.delete_calls: list[dict] = []

    def post(self, url, *, json=None, headers=None, timeout=None) -> _StubResp:
        self.post_calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if isinstance(self._post_resp, Exception):
            raise self._post_resp
        return self._post_resp

    def delete(self, url, *, headers=None, timeout=None) -> _StubResp:
        self.delete_calls.append({"url": url, "headers": headers, "timeout": timeout})
        if isinstance(self._delete_resp, Exception):
            raise self._delete_resp
        return self._delete_resp


def _req() -> SessionCreateRequest:
    return SessionCreateRequest(
        tenant_id="t1", profile_id="alice", run_id="r1",
    )


# ── Construction ──────────────────────────────────────────────────────


def test_requires_router_url() -> None:
    with pytest.raises(ValueError):
        SessionRouterClient(router_url="", auth_token="t")


def test_requires_auth_token() -> None:
    with pytest.raises(ValueError):
        SessionRouterClient(router_url="https://x", auth_token="")


# ── create_session ────────────────────────────────────────────────────


def test_create_session_happy_path() -> None:
    body = {
        "session_id": "sess_abc",
        "base_url": "https://sess.modal.run",
        "session_token": "tok_xyz",
        "expires_at_ms": 1_700_000_000_000,
        "sandbox_id": "fc-1",
        "reused": False,
    }
    http = _StubHttp(_StubResp(200, body))
    client = SessionRouterClient(
        router_url="https://api.modal.run/", auth_token="my-token", session=http,
    )
    resp = client.create_session(_req())
    assert isinstance(resp, SessionCreateResponse)
    assert resp.session_id == "sess_abc"
    assert resp.base_url == "https://sess.modal.run"
    # URL trailing slash got stripped.
    assert http.post_calls[0]["url"] == "https://api.modal.run/v1/computer_sessions"
    # Token forwarded.
    assert http.post_calls[0]["headers"]["X-Mantis-Token"] == "my-token"


def test_create_session_504_raises_unreachable() -> None:
    http = _StubHttp(_StubResp(504, {"detail": "did not publish tunnel URL"}))
    client = SessionRouterClient(
        router_url="https://x", auth_token="t", session=http,
    )
    with pytest.raises(SessionUnreachableError) as exc:
        client.create_session(_req())
    assert "did not publish tunnel URL" in str(exc.value)


def test_create_session_429_raises_quota() -> None:
    http = _StubHttp(_StubResp(429, {"detail": "tenant quota exceeded"}))
    client = SessionRouterClient(router_url="https://x", auth_token="t", session=http)
    with pytest.raises(SessionQuotaExceededError):
        client.create_session(_req())


def test_create_session_other_4xx_raises_base_error() -> None:
    http = _StubHttp(_StubResp(403, {"detail": "tenant_id mismatch"}))
    client = SessionRouterClient(router_url="https://x", auth_token="t", session=http)
    with pytest.raises(SessionRouterError) as exc:
        client.create_session(_req())
    assert "403" in str(exc.value)


def test_create_session_network_error_raises_unreachable() -> None:
    http = _StubHttp(requests.ConnectionError("dns fail"))
    client = SessionRouterClient(router_url="https://x", auth_token="t", session=http)
    with pytest.raises(SessionUnreachableError):
        client.create_session(_req())


# ── close_session ─────────────────────────────────────────────────────


def test_close_session_happy_path() -> None:
    http = _StubHttp(
        _StubResp(200, {"session_id": "x"}),
        delete_resp=_StubResp(200, {"closed": True, "terminal_state": "closed"}),
    )
    client = SessionRouterClient(router_url="https://x", auth_token="t", session=http)
    resp = client.close_session("sess_abc")
    assert resp is not None
    assert resp.closed is True
    assert http.delete_calls[0]["url"] == "https://x/v1/computer_sessions/sess_abc"
    assert http.delete_calls[0]["headers"]["X-Mantis-Reason"] == "brain_closed"


def test_close_session_404_is_quiet_by_default() -> None:
    """Reaper already cleaned up — close should not raise."""
    http = _StubHttp(
        _StubResp(200, {}),
        delete_resp=_StubResp(404, {"detail": "unknown session_id"}),
    )
    client = SessionRouterClient(router_url="https://x", auth_token="t", session=http)
    result = client.close_session("sess_gone")
    assert result is None


def test_close_session_404_raises_when_not_quiet() -> None:
    http = _StubHttp(
        _StubResp(200, {}),
        delete_resp=_StubResp(404, {"detail": "unknown session_id"}),
    )
    client = SessionRouterClient(router_url="https://x", auth_token="t", session=http)
    with pytest.raises(SessionNotFoundError):
        client.close_session("sess_gone", quiet=False)


def test_close_session_network_error_is_quiet_by_default() -> None:
    http = _StubHttp(
        _StubResp(200, {}),
        delete_resp=requests.ConnectionError("split brain"),
    )
    client = SessionRouterClient(router_url="https://x", auth_token="t", session=http)
    assert client.close_session("sess_x") is None
