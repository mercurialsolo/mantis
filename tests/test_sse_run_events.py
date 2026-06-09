"""Tests for SSE event stream — GET /v1/runs/{id}/events?sse=true (#808).

The endpoint wraps the file-backed reasoning.jsonl as a server-sent
event stream. These tests drive the FastAPI app through TestClient
and exercise:

- Non-SSE fallback returns the existing reasoning-trace shape.
- SSE response Content-Type + initial phase event.
- Each reasoning.jsonl row → one SSE event with the right name + id.
- Phase transitions emit phase events between data events.
- Reaching a terminal status emits a ``terminal`` event and closes.
- Last-Event-ID resume skips already-delivered rows.
- 404 for unknown run_id, auth for unauthenticated callers.
"""

from __future__ import annotations

import importlib
import json
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
        self.object_id = "fc-stub-sse"

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
    return TestClient(app), mcs


def _headers() -> dict:
    return {"X-Mantis-Token": "test-token"}


def _seed(mcs, *, run_id: str, status: str = "running", events: list[dict] | None = None) -> None:
    mcs._write_status("default", run_id, {
        "run_id": run_id,
        "status": status,
        "tenant_id": "default",
        "created_at": "2026-06-08T00:00:00+00:00",
        "updated_at": "2026-06-08T00:00:01+00:00",
    })
    if events is not None:
        run_dir = mcs._run_dir("default", run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "reasoning.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events) + ("\n" if events else "")
        )


def _parse_sse(body: str) -> list[dict]:
    """Tiny SSE parser for test assertions. Returns one dict per event."""
    out: list[dict] = []
    cur_event = "message"
    cur_id = ""
    data_lines: list[str] = []
    for line in body.split("\n"):
        if line.startswith(":"):
            # Heartbeat comment — track separately by appending a marker.
            out.append({"_heartbeat": True, "raw": line})
            continue
        if line == "":
            if data_lines:
                try:
                    payload = json.loads("\n".join(data_lines))
                except json.JSONDecodeError:
                    payload = {"_raw": "\n".join(data_lines)}
                out.append({"event": cur_event, "id": cur_id, "data": payload})
            cur_event = "message"
            cur_id = ""
            data_lines = []
            continue
        if line.startswith("event:"):
            cur_event = line[len("event:"):].lstrip()
        elif line.startswith("id:"):
            cur_id = line[len("id:"):].lstrip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())
    return out


# ── Non-SSE fallback ───────────────────────────────────────────────


def test_non_sse_returns_json_with_events_array(app_ctx):
    client, mcs = app_ctx
    _seed(mcs, run_id="r1", status="running", events=[
        {"ts": "2026-06-08T00:00:01", "kind": "step", "step_index": 0},
        {"ts": "2026-06-08T00:00:02", "kind": "step", "step_index": 1},
    ])
    r = client.get("/v1/runs/r1/events", headers=_headers())
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert body["events"][0]["step_index"] == 0
    assert body["events"][1]["step_index"] == 1


def test_non_sse_honors_since_cursor(app_ctx):
    client, mcs = app_ctx
    _seed(mcs, run_id="r2", status="running", events=[
        {"ts": "2026-06-08T00:00:01", "kind": "step", "step_index": 0},
        {"ts": "2026-06-08T00:00:02", "kind": "step", "step_index": 1},
        {"ts": "2026-06-08T00:00:03", "kind": "step", "step_index": 2},
    ])
    r = client.get(
        "/v1/runs/r2/events?since=2026-06-08T00:00:01",
        headers=_headers(),
    )
    body = r.json()
    assert body["count"] == 2
    assert body["events"][0]["step_index"] == 1


# ── SSE shape ──────────────────────────────────────────────────────


def test_sse_terminal_run_emits_initial_phase_then_terminal(app_ctx):
    """Terminal runs short-circuit: client sees the initial phase event
    + the terminal event and the stream closes immediately."""
    client, mcs = app_ctx
    _seed(mcs, run_id="r-done", status="succeeded", events=[
        {"ts": "2026-06-08T00:00:01", "kind": "step", "step_index": 0},
    ])
    with client.stream("GET", "/v1/runs/r-done/events?sse=true", headers=_headers()) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join(resp.iter_text())
    parsed = [e for e in _parse_sse(body) if not e.get("_heartbeat")]
    event_names = [e["event"] for e in parsed]
    assert "phase" in event_names
    assert "terminal" in event_names
    # Phase event names the current state of the run.
    phase_events = [e for e in parsed if e["event"] == "phase"]
    assert phase_events[0]["data"]["phase"] == "complete"


def test_sse_emits_reasoning_events_using_kind_as_event_name(app_ctx):
    client, mcs = app_ctx
    _seed(mcs, run_id="r-ev", status="succeeded", events=[
        {"ts": "2026-06-08T00:00:01", "kind": "step", "step_index": 0},
        {"ts": "2026-06-08T00:00:02", "kind": "extract", "row": {"title": "x"}},
    ])
    with client.stream("GET", "/v1/runs/r-ev/events?sse=true", headers=_headers()) as resp:
        body = "".join(resp.iter_text())
    parsed = [e for e in _parse_sse(body) if not e.get("_heartbeat")]
    names = [e["event"] for e in parsed]
    assert "step" in names
    assert "extract" in names


def test_sse_event_id_uses_ts_for_resume_cursor(app_ctx):
    client, mcs = app_ctx
    _seed(mcs, run_id="r-id", status="succeeded", events=[
        {"ts": "2026-06-08T00:00:01", "kind": "step"},
    ])
    with client.stream("GET", "/v1/runs/r-id/events?sse=true", headers=_headers()) as resp:
        body = "".join(resp.iter_text())
    parsed = [e for e in _parse_sse(body) if not e.get("_heartbeat")]
    step_events = [e for e in parsed if e["event"] == "step"]
    assert step_events[0]["id"] == "2026-06-08T00:00:01"


def test_sse_last_event_id_header_skips_delivered(app_ctx):
    client, mcs = app_ctx
    _seed(mcs, run_id="r-resume", status="succeeded", events=[
        {"ts": "2026-06-08T00:00:01", "kind": "step", "i": 0},
        {"ts": "2026-06-08T00:00:02", "kind": "step", "i": 1},
        {"ts": "2026-06-08T00:00:03", "kind": "step", "i": 2},
    ])
    with client.stream(
        "GET",
        "/v1/runs/r-resume/events?sse=true",
        headers={**_headers(), "Last-Event-ID": "2026-06-08T00:00:01"},
    ) as resp:
        body = "".join(resp.iter_text())
    parsed = [e for e in _parse_sse(body) if not e.get("_heartbeat")]
    step_events = [e for e in parsed if e["event"] == "step"]
    # Only the two newer rows should be delivered.
    assert [e["data"]["i"] for e in step_events] == [1, 2]


def test_sse_since_query_param_skips_delivered(app_ctx):
    client, mcs = app_ctx
    _seed(mcs, run_id="r-since", status="succeeded", events=[
        {"ts": "2026-06-08T00:00:01", "kind": "step", "i": 0},
        {"ts": "2026-06-08T00:00:02", "kind": "step", "i": 1},
    ])
    with client.stream(
        "GET",
        "/v1/runs/r-since/events?sse=true&since=2026-06-08T00:00:01",
        headers=_headers(),
    ) as resp:
        body = "".join(resp.iter_text())
    parsed = [e for e in _parse_sse(body) if not e.get("_heartbeat")]
    step_events = [e for e in parsed if e["event"] == "step"]
    assert [e["data"]["i"] for e in step_events] == [1]


def test_sse_tolerates_missing_reasoning_jsonl(app_ctx):
    """A run that hasn't started writing reasoning events yet must
    still produce a valid SSE response — initial phase + terminal."""
    client, mcs = app_ctx
    _seed(mcs, run_id="r-norjsonl", status="succeeded")
    with client.stream("GET", "/v1/runs/r-norjsonl/events?sse=true", headers=_headers()) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    parsed = [e for e in _parse_sse(body) if not e.get("_heartbeat")]
    assert any(e["event"] == "phase" for e in parsed)
    assert any(e["event"] == "terminal" for e in parsed)


def test_sse_skips_malformed_jsonl_lines(app_ctx):
    """A garbage line in reasoning.jsonl shouldn't 500 the stream;
    the well-formed rows around it must still deliver."""
    client, mcs = app_ctx
    _seed(mcs, run_id="r-bad", status="succeeded")
    run_dir = mcs._run_dir("default", "r-bad")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "reasoning.jsonl").write_text(
        json.dumps({"ts": "2026-06-08T00:00:01", "kind": "step", "i": 0}) + "\n"
        + "{this is not valid json\n"
        + json.dumps({"ts": "2026-06-08T00:00:02", "kind": "step", "i": 1}) + "\n"
    )
    with client.stream("GET", "/v1/runs/r-bad/events?sse=true", headers=_headers()) as resp:
        body = "".join(resp.iter_text())
    parsed = [e for e in _parse_sse(body) if not e.get("_heartbeat")]
    step_events = [e for e in parsed if e["event"] == "step"]
    assert [e["data"]["i"] for e in step_events] == [0, 1]


# ── 404 + auth ─────────────────────────────────────────────────────


def test_sse_returns_404_for_unknown_run(app_ctx):
    client, _mcs = app_ctx
    r = client.get("/v1/runs/missing/events?sse=true", headers=_headers())
    assert r.status_code == 404


def test_sse_requires_auth(app_ctx):
    client, _mcs = app_ctx
    r = client.get("/v1/runs/anything/events?sse=true")
    assert r.status_code in {401, 403}
