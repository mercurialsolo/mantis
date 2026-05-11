"""Unit tests for ``mantis_agent.client.MantisClient``.

Uses a stub session (no live HTTP) so the test stays fast, deterministic,
and works offline. The Mantis FastAPI app is exercised separately in
``tests/test_video_endpoint.py`` and ``tests/test_chat_completions_proxy.py``
via ``fastapi.testclient.TestClient``; this file only tests the client
side of the contract.
"""

from __future__ import annotations

from typing import Any, Callable, Optional
from unittest.mock import MagicMock

import pytest

from mantis_agent.client import (
    MantisAPIError,
    MantisAuthError,
    MantisClient,
    MantisRateLimitError,
    MantisRunFailed,
    MantisTimeoutError,
    PredictRequest,
    RunStatus,
)


class _StubResponse:
    """Minimal ``requests.Response`` lookalike for the stub session."""

    def __init__(
        self,
        status_code: int = 200,
        json_body: Optional[dict] = None,
        text: str = "",
        headers: Optional[dict] = None,
        content_chunks: Optional[list[bytes]] = None,
    ) -> None:
        self.status_code = status_code
        self._json = json_body
        self.text = text
        self.headers = headers or {}
        self.url = ""  # populated by the stub session
        self._chunks = content_chunks or []

    def json(self) -> Any:
        if self._json is None:
            raise ValueError("no json body")
        return self._json

    def iter_content(self, chunk_size: int = 1024):  # noqa: ARG002 — match requests sig
        yield from self._chunks

    def __enter__(self) -> "_StubResponse":
        return self

    def __exit__(self, *_exc: Any) -> None:
        return None


class _StubSession:
    """Stand-in for ``requests.Session`` that records calls and replays
    canned responses. Configure with a ``handler(method, url, **kw)`` callable
    or a ``queue`` of pre-built :class:`_StubResponse` objects.
    """

    def __init__(
        self,
        handler: Optional[Callable[..., _StubResponse]] = None,
        queue: Optional[list[_StubResponse]] = None,
    ) -> None:
        self.handler = handler
        self.queue: list[_StubResponse] = list(queue or [])
        self.calls: list[tuple[str, str, dict]] = []

    def _next(self, method: str, url: str, **kw: Any) -> _StubResponse:
        self.calls.append((method, url, kw))
        if self.handler is not None:
            resp = self.handler(method, url, **kw)
        elif self.queue:
            resp = self.queue.pop(0)
        else:
            raise AssertionError(
                f"unexpected {method} {url} — stub has no more responses",
            )
        resp.url = url
        return resp

    def post(self, url: str, **kw: Any) -> _StubResponse:
        return self._next("POST", url, **kw)

    def get(self, url: str, **kw: Any) -> _StubResponse:
        return self._next("GET", url, **kw)

    def close(self) -> None:
        pass


def _client(session: _StubSession) -> MantisClient:
    return MantisClient(
        endpoint="https://mantis.example/production/sync",
        api_key="api-key-x",
        mantis_token="tenant-x",
        session=session,  # type: ignore[arg-type]
    )


# ── constructor + headers ─────────────────────────────────────────────────


def test_endpoint_strips_trailing_v1_and_slash() -> None:
    session = _StubSession()
    c = MantisClient(endpoint="https://m.example/v1/", session=session)  # type: ignore[arg-type]
    assert c.endpoint == "https://m.example"


def test_empty_endpoint_rejected() -> None:
    with pytest.raises(ValueError, match="endpoint is required"):
        MantisClient(endpoint="")


def test_from_env_reads_three_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANTIS_ENDPOINT", "https://e.example")
    monkeypatch.setenv("BASETEN_API_KEY", "key-1")
    monkeypatch.setenv("MANTIS_API_TOKEN", "tok-1")
    c = MantisClient.from_env(session=_StubSession())  # type: ignore[arg-type]
    assert c.endpoint == "https://e.example"
    assert c.api_key == "key-1"
    assert c.mantis_token == "tok-1"


def test_from_env_missing_endpoint_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MANTIS_ENDPOINT", raising=False)
    with pytest.raises(ValueError, match="MANTIS_ENDPOINT"):
        MantisClient.from_env()


# ── predict / submit ──────────────────────────────────────────────────────


def test_predict_sends_typed_request_and_returns_handle() -> None:
    session = _StubSession(queue=[
        _StubResponse(json_body={
            "status": "queued",
            "created_at": "2026-04-28T00:00:00Z",
            "model": "holo3",
            "mode": "detached",
            "run_id": "run-abc",
            "payload": {},
            "updated_at": "2026-04-28T00:00:00Z",
            "status_path": "/x/status.json",
            "result_path": "/x/result.json",
            "csv_path": "/x/leads.csv",
            "events_path": "/x/events.log",
        }),
    ])
    c = _client(session)
    handle = c.predict(
        PredictRequest(
            micro="plans/example/extract_listings.json",
            state_key="abc",
            max_cost=2,
        ),
    )
    assert handle.run_id == "run-abc"
    assert handle.status == "queued"
    method, url, kw = session.calls[0]
    assert method == "POST"
    assert url.endswith("/v1/predict")
    assert kw["headers"]["Authorization"] == "Api-Key api-key-x"
    assert kw["headers"]["X-Mantis-Token"] == "tenant-x"
    assert kw["json"]["micro"] == "plans/example/extract_listings.json"
    assert kw["json"]["state_key"] == "abc"
    assert kw["json"]["detached"] is True


def test_predict_accepts_raw_dict() -> None:
    session = _StubSession(queue=[
        _StubResponse(json_body={
            "status": "queued",
            "created_at": "x",
            "model": "holo3",
            "mode": "detached",
            "run_id": "run-2",
            "payload": {},
            "updated_at": "x",
            "status_path": "x",
            "result_path": "x",
            "csv_path": "x",
            "events_path": "x",
        }),
    ])
    c = _client(session)
    handle = c.predict({"micro": "p.json"})
    assert handle.run_id == "run-2"


def test_predict_rejects_garbage_dict() -> None:
    # A dict missing all the plan-shape fields fails PredictRequest validation.
    c = _client(_StubSession())
    with pytest.raises(Exception):  # pydantic ValidationError
        c.predict({})


def test_idempotency_key_added_to_headers() -> None:
    session = _StubSession(queue=[
        _StubResponse(json_body={
            "status": "queued",
            "created_at": "x",
            "model": "m",
            "mode": "detached",
            "run_id": "r",
            "payload": {},
            "updated_at": "x",
            "status_path": "x",
            "result_path": "x",
            "csv_path": "x",
            "events_path": "x",
        }),
    ])
    c = MantisClient(
        endpoint="https://m.example",
        api_key="k",
        mantis_token="t",
        idempotency_key="order-7afc3b91",
        session=session,  # type: ignore[arg-type]
    )
    c.predict({"micro": "p.json"})
    assert session.calls[0][2]["headers"]["Idempotency-Key"] == "order-7afc3b91"


# ── action variants ───────────────────────────────────────────────────────


def test_status_parses_run_status() -> None:
    session = _StubSession(queue=[
        _StubResponse(json_body={
            "status": "running",
            "run_id": "r1",
            "started_at": "2026-04-28T00:00:00Z",
        }),
    ])
    c = _client(session)
    s = c.status("r1")
    assert isinstance(s, RunStatus)
    assert s.status == "running"
    assert s.run_id == "r1"


def test_result_returns_dict() -> None:
    session = _StubSession(queue=[
        _StubResponse(json_body={"result": {"leads": ["A", "B"]}}),
    ])
    c = _client(session)
    out = c.result("r1")
    assert out["result"]["leads"] == ["A", "B"]


def test_logs_returns_event_strings() -> None:
    session = _StubSession(queue=[
        _StubResponse(json_body={"events": ["event-a", "event-b", "event-c"]}),
    ])
    c = _client(session)
    events = c.logs("r1", tail=3)
    assert events == ["event-a", "event-b", "event-c"]
    assert session.calls[0][2]["json"] == {
        "action": "logs", "run_id": "r1", "tail": 3,
    }


def test_logs_rejects_out_of_range_tail() -> None:
    c = _client(_StubSession())
    with pytest.raises(ValueError, match=r"tail must be in"):
        c.logs("r1", tail=0)
    with pytest.raises(ValueError, match=r"tail must be in"):
        c.logs("r1", tail=10_001)


def test_cancel_sends_action() -> None:
    session = _StubSession(queue=[
        _StubResponse(json_body={"status": "cancelling"}),
    ])
    c = _client(session)
    c.cancel("r1")
    assert session.calls[0][2]["json"] == {"action": "cancel", "run_id": "r1"}


# ── errors ────────────────────────────────────────────────────────────────


def test_auth_error_on_401() -> None:
    session = _StubSession(queue=[
        _StubResponse(status_code=401, json_body={"detail": "bad token"}),
    ])
    c = _client(session)
    with pytest.raises(MantisAuthError) as ei:
        c.status("r1")
    assert ei.value.status_code == 401
    assert ei.value.response_body == {"detail": "bad token"}


def test_rate_limit_carries_retry_after() -> None:
    session = _StubSession(queue=[
        _StubResponse(
            status_code=429,
            json_body={"detail": "slow down"},
            headers={"Retry-After": "12.5"},
        ),
    ])
    c = _client(session)
    with pytest.raises(MantisRateLimitError) as ei:
        c.status("r1")
    assert ei.value.retry_after_seconds == 12.5


def test_generic_api_error_on_500() -> None:
    session = _StubSession(queue=[
        _StubResponse(status_code=500, text="kaboom"),
    ])
    c = _client(session)
    with pytest.raises(MantisAPIError) as ei:
        c.status("r1")
    assert ei.value.status_code == 500
    assert ei.value.response_body == "kaboom"


# ── wait_for_completion + run_to_completion ───────────────────────────────


def test_wait_for_completion_returns_on_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Three status calls — running, running, succeeded.
    session = _StubSession(queue=[
        _StubResponse(json_body={"status": "running", "run_id": "r1"}),
        _StubResponse(json_body={"status": "running", "run_id": "r1"}),
        _StubResponse(json_body={
            "status": "succeeded",
            "run_id": "r1",
            "summary": {"viable": 3},
        }),
    ])
    sleeps: list[float] = []
    monkeypatch.setattr(
        "mantis_agent.client.client.time.sleep", lambda s: sleeps.append(s),
    )
    c = _client(session)
    final = c.wait_for_completion("r1", poll_interval_s=0.1)
    assert final.status == "succeeded"
    assert len(session.calls) == 3
    assert sleeps == [0.1, 0.1]  # one sleep per non-terminal poll


def test_wait_for_completion_raises_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _StubSession(handler=lambda *_a, **_kw: _StubResponse(
        json_body={"status": "running", "run_id": "r1"},
    ))
    # Drive the monotonic clock so we hit the timeout deterministically.
    ticks = iter([0.0, 0.5, 1.1])
    monkeypatch.setattr(
        "mantis_agent.client.client.time.monotonic", lambda: next(ticks),
    )
    monkeypatch.setattr(
        "mantis_agent.client.client.time.sleep", lambda _s: None,
    )
    c = _client(session)
    with pytest.raises(MantisTimeoutError) as ei:
        c.wait_for_completion("r1", poll_interval_s=0.1, timeout_s=1.0)
    assert ei.value.run_id == "r1"


def test_wait_for_completion_calls_on_status_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _StubSession(queue=[
        _StubResponse(json_body={"status": "running", "run_id": "r1"}),
        _StubResponse(json_body={"status": "succeeded", "run_id": "r1"}),
    ])
    monkeypatch.setattr(
        "mantis_agent.client.client.time.sleep", lambda _s: None,
    )
    seen: list[str] = []
    c = _client(session)
    c.wait_for_completion(
        "r1", poll_interval_s=0.01, on_status=lambda s: seen.append(s.status),
    )
    assert seen == ["running", "succeeded"]


def test_run_to_completion_raises_on_failed_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _StubSession(queue=[
        # predict
        _StubResponse(json_body={
            "status": "queued",
            "created_at": "x",
            "model": "m",
            "mode": "detached",
            "run_id": "r1",
            "payload": {},
            "updated_at": "x",
            "status_path": "x",
            "result_path": "x",
            "csv_path": "x",
            "events_path": "x",
        }),
        # status: failed (terminal)
        _StubResponse(json_body={
            "status": "failed",
            "run_id": "r1",
            "error": "kaboom",
        }),
    ])
    monkeypatch.setattr(
        "mantis_agent.client.client.time.sleep", lambda _s: None,
    )
    c = _client(session)
    with pytest.raises(MantisRunFailed) as ei:
        c.run_to_completion({"micro": "p.json"}, poll_interval_s=0.01)
    assert ei.value.status.status == "failed"


def test_run_to_completion_returns_result_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _StubSession(queue=[
        _StubResponse(json_body={
            "status": "queued",
            "created_at": "x",
            "model": "m",
            "mode": "detached",
            "run_id": "r1",
            "payload": {},
            "updated_at": "x",
            "status_path": "x",
            "result_path": "x",
            "csv_path": "x",
            "events_path": "x",
        }),
        _StubResponse(json_body={"status": "succeeded", "run_id": "r1"}),
        _StubResponse(json_body={"result": {"leads": ["A"]}}),
    ])
    monkeypatch.setattr(
        "mantis_agent.client.client.time.sleep", lambda _s: None,
    )
    c = _client(session)
    out = c.run_to_completion({"micro": "p.json"}, poll_interval_s=0.01)
    assert out["result"]["leads"] == ["A"]


# ── cua / video / health ──────────────────────────────────────────────────


def test_cua_run_builds_pure_cua_request() -> None:
    session = _StubSession(queue=[
        _StubResponse(json_body={"success": True, "steps": 4}),
    ])
    c = _client(session)
    out = c.cua_run("Click the docs link on example.com", max_steps=5)
    assert out["success"] is True
    method, url, kw = session.calls[0]
    assert method == "POST"
    assert url.endswith("/v1/cua")
    assert kw["json"]["instruction"] == "Click the docs link on example.com"
    assert kw["json"]["max_steps"] == 5


def test_fetch_video_streams_to_disk(tmp_path: Any) -> None:
    chunks = [b"abc", b"defgh", b"ij"]
    session = _StubSession(queue=[
        _StubResponse(content_chunks=chunks),
    ])
    c = _client(session)
    dest = c.fetch_video("r1", tmp_path / "out.mp4")
    assert dest.exists()
    assert dest.read_bytes() == b"abcdefghij"
    # Default fetch sends no query params — server defaults to polished.
    method, url, kw = session.calls[0]
    assert method == "GET"
    assert url.endswith("/v1/runs/r1/video")
    assert not (kw.get("params") or {})


def test_fetch_video_raw_passes_raw_one(tmp_path: Any) -> None:
    """polished=False must send ?raw=1 — that's the query param the server
    actually reads (`request.query_params.get("raw", "")` in
    baseten_server/routes.py). Pre-fix the client sent ?polished=false,
    which the server silently ignored, so callers always got the polished
    version even when they asked for raw."""
    session = _StubSession(queue=[_StubResponse(content_chunks=[b"raw"])])
    c = _client(session)
    c.fetch_video("r1", tmp_path / "raw.mp4", polished=False)
    assert session.calls[0][2]["params"] == {"raw": "1"}


def test_health_returns_payload() -> None:
    session = _StubSession(queue=[
        _StubResponse(json_body={"status": "ok", "version": "0.1.0"}),
    ])
    c = _client(session)
    out = c.health()
    assert out == {"status": "ok", "version": "0.1.0"}
    assert session.calls[0][1].endswith("/v1/health")


# ── context manager ───────────────────────────────────────────────────────


def test_context_manager_closes_session() -> None:
    session = MagicMock()
    with MantisClient(
        endpoint="https://m.example", session=session,
    ) as c:
        assert c.endpoint == "https://m.example"
    session.close.assert_called_once()
