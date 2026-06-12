"""Parity audit for the Baseten runtime + routes — 12 Modal-only behaviours.

Each test covers one of the items shipped to bring the Baseten side
to feature parity with the Modal CUA server:

A. Status enrichment in ``_detached_action``:
   1. ``failure_help`` synthesized from ``halt_class`` on terminal runs
   2. ``action=reasoning_trace`` returns events with optional ``since``
      cursor filter
   3. ``cancel_lookup_error`` surfaced on cancel when the sentinel write
      fails or the worker thread can't be signalled
   4. ``viewer_url`` surfaced when a sidecar ``viewer.json`` exists

B. New REST endpoints in ``routes.py``:
   5. ``GET /v1/runs/{id}/status``
   6. ``GET /v1/runs/{id}/result``
   7. ``POST /v1/runs/{id}/cancel``
   8. ``GET /v1/runs/{id}`` lifecycle phase + backoff hint + failure_help
   9. ``GET /v1/queue`` queue snapshot
  10. ``GET /v1/runs/{id}/augur`` augur metadata envelope
  11. ``GET /v1/runs/{id}/events`` reasoning trace JSON fallback

C. Worker integration:
  12. ``_write_augur_metadata`` round-trips augur_run_id + bundle_dir

The runtime fixture matches ``tests/test_baseten_cancel_pause_dispatch.py``
so we exercise the same isolated tmp_path data root and stub out the
heavy ``load`` step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def runtime(tmp_path: Path, monkeypatch):
    """Construct a ``BasetenCUARuntime`` against an isolated data dir."""
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MANTIS_BRAIN", "holo3")
    monkeypatch.setenv("MANTIS_SKIP_BRAIN_LOAD", "1")
    from mantis_agent.baseten_server.runtime import BasetenCUARuntime
    rt = BasetenCUARuntime()
    rt.load = lambda: None  # type: ignore[assignment]
    return rt


def _seed_run(rt: Any, run_id: str, status: str, **extra: Any) -> Path:
    payload: dict[str, Any] = {
        "status": status,
        "mode": "detached",
        "model": "holo3",
        **extra,
    }
    rt._write_detached_status(run_id, payload)
    return rt._run_path(run_id) / "status.json"


# ── A1. failure_help on terminal status ────────────────────────────────


def test_failure_help_synthesized_on_halted(runtime) -> None:
    _seed_run(
        runtime, "r-halted", "halted",
        halt_class="cf_challenge",
    )
    out = runtime.run({"action": "status", "run_id": "r-halted"})
    assert out["status"] == "halted"
    fh = out.get("failure_help")
    assert fh is not None, "failure_help missing on halted run"
    assert fh.get("halt_class") == "cf_challenge"
    assert "summary" in fh
    assert "next_steps" in fh


def test_failure_help_absent_on_running(runtime) -> None:
    _seed_run(runtime, "r-live", "running")
    out = runtime.run({"action": "status", "run_id": "r-live"})
    assert out["status"] == "running"
    assert "failure_help" not in out


def test_failure_help_preserves_existing_dict(runtime) -> None:
    """When the runner already attached a failure_help, we don't synthesize."""
    existing = {"halt_class": "custom", "summary": "runner-attached"}
    _seed_run(
        runtime, "r-custom", "halted",
        halt_class="cf_challenge",
        failure_help=existing,
    )
    out = runtime.run({"action": "status", "run_id": "r-custom"})
    assert out["failure_help"] == existing


# ── A2. action=reasoning_trace ─────────────────────────────────────────


def test_reasoning_trace_returns_events_and_count(runtime) -> None:
    _seed_run(runtime, "r-trace", "running")
    jsonl = runtime._run_path("r-trace") / "reasoning.jsonl"
    jsonl.write_text(
        '\n'.join([
            json.dumps({"ts": "2026-06-11T10:00:00Z", "kind": "step", "i": 1}),
            json.dumps({"ts": "2026-06-11T10:00:01Z", "kind": "step", "i": 2}),
            json.dumps({"ts": "2026-06-11T10:00:02Z", "kind": "step", "i": 3}),
        ])
    )
    out = runtime.run({"action": "reasoning_trace", "run_id": "r-trace"})
    assert out["count"] == 3
    assert [e["i"] for e in out["events"]] == [1, 2, 3]
    # status fields preserved on the envelope
    assert out["status"] == "running"


def test_reasoning_trace_since_cursor_filters(runtime) -> None:
    _seed_run(runtime, "r-since", "running")
    jsonl = runtime._run_path("r-since") / "reasoning.jsonl"
    jsonl.write_text(
        '\n'.join([
            json.dumps({"ts": "2026-06-11T10:00:00Z", "i": 1}),
            json.dumps({"ts": "2026-06-11T10:00:01Z", "i": 2}),
            json.dumps({"ts": "2026-06-11T10:00:02Z", "i": 3}),
        ])
    )
    out = runtime.run({
        "action": "reasoning_trace",
        "run_id": "r-since",
        "since": "2026-06-11T10:00:01Z",
    })
    assert out["count"] == 1
    assert out["events"][0]["i"] == 3


def test_reasoning_trace_empty_when_no_log(runtime) -> None:
    _seed_run(runtime, "r-empty", "running")
    out = runtime.run({"action": "reasoning_trace", "run_id": "r-empty"})
    assert out["count"] == 0
    assert out["events"] == []


def test_reasoning_trace_skips_malformed_lines(runtime) -> None:
    _seed_run(runtime, "r-mal", "running")
    jsonl = runtime._run_path("r-mal") / "reasoning.jsonl"
    jsonl.write_text(
        '\n'.join([
            'not json',
            json.dumps({"ts": "2026-06-11T10:00:00Z", "i": 1}),
            '',
            json.dumps(["not", "a", "dict"]),
            json.dumps({"ts": "2026-06-11T10:00:01Z", "i": 2}),
        ])
    )
    out = runtime.run({"action": "reasoning_trace", "run_id": "r-mal"})
    assert out["count"] == 2


# ── A3. cancel_lookup_error surfaces but cancel still succeeds ─────────


def test_cancel_lookup_error_when_sentinel_write_fails(
    runtime, monkeypatch,
) -> None:
    _seed_run(runtime, "r-werr", "running")
    # Monkeypatch Path.write_text to fail for cancel_request.json.
    real_write_text = Path.write_text

    def _raising(self: Path, *args: Any, **kwargs: Any) -> int:
        if self.name == "cancel_request.json":
            raise OSError("simulated EROFS")
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _raising)
    out = runtime.run({"action": "cancel", "run_id": "r-werr"})
    # The cancel still wins — operator intent is sticky.
    assert out["status"] == "cancelled"
    # But the diagnostic is surfaced so the caller knows the worker
    # won't observe the sentinel.
    assert "cancel_lookup_error" in out
    assert "sentinel_write_failed" in out["cancel_lookup_error"]


def test_cancel_no_lookup_error_on_happy_path(runtime) -> None:
    _seed_run(runtime, "r-ok", "running")
    out = runtime.run({"action": "cancel", "run_id": "r-ok"})
    assert out["status"] == "cancelled"
    assert "cancel_lookup_error" not in out


# ── A4. viewer_url surface ─────────────────────────────────────────────


def test_viewer_url_surfaced_from_sidecar(runtime) -> None:
    _seed_run(runtime, "r-view", "running")
    viewer_path = runtime._run_path("r-view") / "viewer.json"
    viewer_path.write_text(json.dumps({"viewer_url": "https://viewer.example/x"}))
    out = runtime.run({"action": "status", "run_id": "r-view"})
    assert out["viewer_url"] == "https://viewer.example/x"


def test_viewer_url_status_field_wins_over_sidecar(runtime) -> None:
    """The runner already merges viewer_url into status.json; sidecar
    is a fallback for tooling that drops the file separately."""
    _seed_run(
        runtime, "r-view2", "running",
        viewer_url="https://from-status/",
    )
    viewer_path = runtime._run_path("r-view2") / "viewer.json"
    viewer_path.write_text(json.dumps({"viewer_url": "https://sidecar/"}))
    out = runtime.run({"action": "status", "run_id": "r-view2"})
    # status.json shape wins; sidecar only fills the gap when absent.
    assert out["viewer_url"] == "https://from-status/"


def test_viewer_url_absent_when_no_sidecar(runtime) -> None:
    _seed_run(runtime, "r-noview", "running")
    out = runtime.run({"action": "status", "run_id": "r-noview"})
    assert "viewer_url" not in out


# ── A12 (worker integration). Augur metadata round-trip ────────────────


def test_augur_metadata_round_trips(runtime) -> None:
    runtime._write_augur_metadata("r-aug", "workflow-abc123")
    meta = runtime._read_augur_metadata("r-aug")
    assert meta is not None
    assert meta["augur_run_id"] == "workflow-abc123"
    assert meta["bundle_dir"].endswith("workflow-abc123")


def test_augur_metadata_empty_id_is_noop(runtime) -> None:
    runtime._write_augur_metadata("r-aug-empty", "")
    assert runtime._read_augur_metadata("r-aug-empty") is None


def test_augur_metadata_missing_returns_none(runtime) -> None:
    assert runtime._read_augur_metadata("r-never-written") is None


# ── B5–B7. REST shorthand endpoints (status, result, cancel) ───────────
#
# Use the in-process TestClient against the FastAPI ``app`` (auth is
# stubbed via dependency override the way the existing routes tests do).


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    """FastAPI TestClient + tenant auth stub."""
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MANTIS_BRAIN", "holo3")
    monkeypatch.setenv("MANTIS_SKIP_BRAIN_LOAD", "1")
    from fastapi.testclient import TestClient
    from mantis_agent.baseten_server.routes import app, runtime as routes_runtime
    from mantis_agent.baseten_server.middleware import require_run_scope
    from mantis_agent.tenant_auth import TenantConfig

    # Make sure the module-level runtime in routes.py points at this
    # tmp_path-rooted instance — _run_path is computed from data_root().
    # Routes use the runtime singleton already; just force a load no-op.
    routes_runtime.load = lambda: None  # type: ignore[assignment]

    tenant = TenantConfig(
        tenant_id="test-tenant",
        scopes=("run",),
    )

    def _override_run_scope() -> TenantConfig:
        return tenant

    app.dependency_overrides[require_run_scope] = _override_run_scope
    yield TestClient(app), routes_runtime
    app.dependency_overrides.pop(require_run_scope, None)


def _client_seed(rt: Any, run_id: str, status: str, **extra: Any) -> None:
    rt._write_detached_status(
        run_id,
        {"status": status, "mode": "detached", "model": "holo3", **extra},
    )


def test_rest_get_status_matches_action_status(client) -> None:
    api, rt = client
    _client_seed(rt, "r-rest-status", "running")
    rest = api.get("/v1/runs/r-rest-status/status").json()
    action = rt.run({"action": "status", "run_id": "r-rest-status"})
    # Both forms surface the same status string + run_id.
    assert rest["status"] == action["status"] == "running"
    assert rest["run_id"] == action["run_id"] == "r-rest-status"


def test_rest_get_result_returns_envelope(client) -> None:
    api, rt = client
    _client_seed(rt, "r-rest-result", "running")
    # No result.json yet — endpoint returns result_ready=False shape.
    rest = api.get("/v1/runs/r-rest-result/result").json()
    assert rest["result_ready"] is False
    assert rest["status"] == "running"


def test_rest_post_cancel_flips_to_cancelled(client) -> None:
    api, rt = client
    _client_seed(rt, "r-rest-cancel", "running")
    rest = api.post("/v1/runs/r-rest-cancel/cancel").json()
    assert rest["status"] == "cancelled"
    # And a status read confirms the sticky write.
    again = rt.run({"action": "status", "run_id": "r-rest-cancel"})
    assert again["status"] == "cancelled"


def test_rest_get_status_unknown_run_404s(client) -> None:
    api, _ = client
    resp = api.get("/v1/runs/r-missing/status")
    assert resp.status_code == 404


# ── B8. GET /v1/runs/{id} lifecycle endpoint ───────────────────────────


def test_lifecycle_endpoint_returns_phase_and_backoff(client) -> None:
    api, rt = client
    _client_seed(rt, "r-life", "running")
    body = api.get("/v1/runs/r-life").json()
    assert body["phase"] == "running"
    assert body["run_id"] == "r-life"
    assert "polling_backoff_ms_hint" in body


def test_lifecycle_endpoint_surfaces_failure_help_on_halted(client) -> None:
    api, rt = client
    _client_seed(
        rt, "r-life-halt", "halted",
        halt_class="proxy_unreachable",
    )
    body = api.get("/v1/runs/r-life-halt").json()
    assert body["phase"] == "halted"
    assert "failure_help" in body
    assert body["failure_help"]["halt_class"] == "proxy_unreachable"


def test_lifecycle_endpoint_surfaces_augur_run_id(client) -> None:
    api, rt = client
    _client_seed(rt, "r-life-aug", "running")
    rt._write_augur_metadata("r-life-aug", "wf-xyz12345")
    body = api.get("/v1/runs/r-life-aug").json()
    assert body.get("augur_run_id") == "wf-xyz12345"
    assert body.get("augur_bundle_url") == "/v1/runs/r-life-aug/augur"


def test_lifecycle_endpoint_unknown_run_404s(client) -> None:
    api, _ = client
    resp = api.get("/v1/runs/r-missing-life")
    assert resp.status_code == 404


# ── B9. GET /v1/queue ──────────────────────────────────────────────────


def test_queue_endpoint_counts_active_phases(client) -> None:
    api, rt = client
    _client_seed(rt, "q-a", "queued")
    _client_seed(rt, "q-b", "running")
    _client_seed(rt, "q-c", "running")
    _client_seed(rt, "q-d", "succeeded")  # terminal — excluded
    _client_seed(rt, "q-e", "halted")  # terminal — excluded
    body = api.get("/v1/queue").json()
    assert body["queued"] == 1
    assert body["running"] == 2
    # Terminal runs don't contribute to recovering / queued / running.
    assert body["recovering"] == 0
    assert body["tenant_id"] == "test-tenant"


def test_queue_endpoint_empty_on_no_runs(client, tmp_path: Path) -> None:
    api, _ = client
    body = api.get("/v1/queue").json()
    assert body["queued"] == 0
    assert body["running"] == 0
    assert body["recovering"] == 0


# ── B10. GET /v1/runs/{id}/augur ───────────────────────────────────────


def test_augur_envelope_endpoint_returns_metadata(client) -> None:
    api, rt = client
    _client_seed(rt, "r-aug-route", "running")
    rt._write_augur_metadata("r-aug-route", "wf-aug-route")
    body = api.get("/v1/runs/r-aug-route/augur").json()
    assert body["run_id"] == "r-aug-route"
    assert body["augur_run_id"] == "wf-aug-route"
    assert body["bundle_present"] is False  # no files yet on the bundle dir
    assert body["files"] == []


def test_augur_envelope_endpoint_404s_when_no_metadata(client) -> None:
    api, rt = client
    _client_seed(rt, "r-aug-bare", "running")
    resp = api.get("/v1/runs/r-aug-bare/augur")
    assert resp.status_code == 404


# ── B11. GET /v1/runs/{id}/events (JSON fallback) ──────────────────────


def test_events_endpoint_returns_reasoning_trace_envelope(client) -> None:
    api, rt = client
    _client_seed(rt, "r-events", "running")
    jsonl = rt._run_path("r-events") / "reasoning.jsonl"
    jsonl.write_text(
        '\n'.join([
            json.dumps({"ts": "2026-06-11T11:00:00Z", "kind": "navigate"}),
            json.dumps({"ts": "2026-06-11T11:00:01Z", "kind": "click"}),
        ])
    )
    body = api.get("/v1/runs/r-events/events").json()
    assert body["count"] == 2
    assert body["status"] == "running"
    assert [e["kind"] for e in body["events"]] == ["navigate", "click"]


def test_events_endpoint_unknown_run_404s(client) -> None:
    api, _ = client
    resp = api.get("/v1/runs/r-events-missing/events")
    assert resp.status_code == 404


def test_events_endpoint_honors_since_cursor(client) -> None:
    api, rt = client
    _client_seed(rt, "r-events-since", "running")
    jsonl = rt._run_path("r-events-since") / "reasoning.jsonl"
    jsonl.write_text(
        '\n'.join([
            json.dumps({"ts": "2026-06-11T11:00:00Z", "i": 1}),
            json.dumps({"ts": "2026-06-11T11:00:02Z", "i": 2}),
        ])
    )
    body = api.get(
        "/v1/runs/r-events-since/events",
        params={"since": "2026-06-11T11:00:00Z"},
    ).json()
    assert body["count"] == 1
    assert body["events"][0]["i"] == 2


# ── D. Artifact path parity with Modal ────────────────────────────────
#
# Pre-fix, ``_run_path`` returned ``<root>/runs/<id>/`` but the
# ``/v1/runs/{id}/artifacts/{name}`` reader at
# ``routes.get_run_artifact`` looked under
# ``<root>/tenants/<tenant>/runs/<id>/`` — every artifact request 404'd
# even when ``persist_run_artifacts`` had materialized the file. The
# fix aligns ``_run_path`` to the tenant-scoped shape that Modal's
# ``_run_dir`` already uses; legacy un-scoped runs still resolve via a
# read-side fallback.


def test_run_path_is_tenant_scoped(runtime, tmp_path: Path) -> None:
    """``_run_path(create=True)`` lands under ``tenants/<tenant>/runs/<id>/``."""
    p = runtime._run_path("r-scoped", create=True)
    # The path under tmp_path must include the tenant segment.
    rel = p.relative_to(tmp_path)
    parts = rel.parts
    assert parts[0] == "tenants", f"missing tenants segment in {parts}"
    # MANTIS_TENANT_ID isn't set in the fixture → defaults to DEFAULT_TENANT.
    from mantis_agent.tenant_auth import DEFAULT_TENANT
    assert parts[1] == DEFAULT_TENANT.tenant_id
    assert parts[2] == "runs"
    assert parts[3] == "r-scoped"


def test_extracted_rows_artifact_served_when_runtime_persisted_it(
    client, tmp_path: Path, monkeypatch,
) -> None:
    """End-to-end: rows persisted by ``_save_detached_result`` show up
    at ``GET /v1/runs/{id}/artifacts/extracted_rows.json``."""
    api, rt = client
    # The fixture stubs the tenant to "test-tenant"; mirror that here so
    # the writer (which reads MANTIS_TENANT_ID via os.environ) lands the
    # files in the same dir the reader scans.
    monkeypatch.setenv("MANTIS_TENANT_ID", "test-tenant")
    # The artifact endpoint takes the unscoped ``require_mantis_token``
    # dep directly; the fixture only overrides ``require_run_scope``,
    # so wire the token dep through too.
    from mantis_agent.baseten_server.routes import app
    from mantis_agent.baseten_server.middleware import require_mantis_token
    from mantis_agent.tenant_auth import TenantConfig
    tenant = TenantConfig(tenant_id="test-tenant", scopes=("run",))
    app.dependency_overrides[require_mantis_token] = lambda: tenant
    try:
        run_id = "r-rows-parity"
        # Seed a status so the lifecycle endpoints don't 404 the run.
        _client_seed(rt, run_id, "succeeded")
        # Drive the same write path the worker uses post-completion.
        rt._save_detached_result(run_id, {
            "run_id": run_id,
            "artifacts": [
                {
                    "kind": "structured_data",
                    "name": "extracted_rows",
                    "schema": {"fields": ["rank", "title"]},
                    "data": [
                        {"rank": "1", "title": "first"},
                        {"rank": "2", "title": "second"},
                    ],
                },
            ],
        })
        # GET the artifact via the public endpoint.
        resp = api.get(f"/v1/runs/{run_id}/artifacts/extracted_rows.json")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert isinstance(body, list)
        assert len(body) == 2
        assert body[0]["rank"] == "1"
    finally:
        app.dependency_overrides.pop(require_mantis_token, None)


def test_run_path_falls_back_to_legacy_unscoped_layout(
    runtime, tmp_path: Path,
) -> None:
    """Runs persisted before the fix landed (un-scoped
    ``<root>/runs/<id>/``) remain readable via the fallback branch
    until the container restarts."""
    legacy_dir = tmp_path / "runs" / "r-legacy"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "status.json").write_text(json.dumps({"status": "succeeded"}))
    # Read-path (create=False): no scoped dir exists, so the fallback
    # returns the legacy un-scoped one.
    p = runtime._run_path("r-legacy")
    assert p == legacy_dir
    assert (p / "status.json").exists()


# ── E. Runtime concurrency limiter ────────────────────────────────────
#
# Pre-fix, ``self.lock`` was a ``threading.Lock`` and the detached
# worker held it across the entire run. The replica could only run
# ONE detached run at a time, regardless of ``max_concurrent_runs``.
# A trivial 3-min run could sit queued for 30+ minutes behind a slow
# active run. Fix: counting Semaphore sized via
# ``MANTIS_RUNTIME_CONCURRENCY`` (default 1 preserves legacy
# behaviour, operator bumps to enable parallelism).


def test_runtime_lock_defaults_to_serialized_single_slot(
    runtime,
) -> None:
    """Default lock is a 1-slot semaphore — preserves legacy behaviour
    where the operator hasn't explicitly opted into parallelism."""
    import threading
    assert isinstance(runtime.lock, type(threading.Semaphore()))
    assert runtime._runtime_concurrency == 1


def test_runtime_lock_honours_env_concurrency(
    tmp_path: Path, monkeypatch,
) -> None:
    """``MANTIS_RUNTIME_CONCURRENCY=4`` → semaphore allows 4 concurrent
    holders. Demonstrates the limiter actually unblocks queued runs
    when slots are free."""
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MANTIS_BRAIN", "holo3")
    monkeypatch.setenv("MANTIS_SKIP_BRAIN_LOAD", "1")
    monkeypatch.setenv("MANTIS_RUNTIME_CONCURRENCY", "4")
    from mantis_agent.baseten_server.runtime import BasetenCUARuntime
    rt = BasetenCUARuntime()
    assert rt._runtime_concurrency == 4
    # Acquire 4 in a row — each must succeed without blocking.
    held = []
    for _ in range(4):
        ok = rt.lock.acquire(blocking=False)
        assert ok is True
        held.append(ok)
    # The 5th must be refused — limiter respects the cap.
    assert rt.lock.acquire(blocking=False) is False
    # Release one and the 6th attempt succeeds again.
    rt.lock.release()
    assert rt.lock.acquire(blocking=False) is True


def test_runtime_lock_invalid_env_falls_back_to_one(
    tmp_path: Path, monkeypatch,
) -> None:
    """A non-int value in MANTIS_RUNTIME_CONCURRENCY doesn't crash the
    runtime — falls back to the safe default of 1."""
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MANTIS_BRAIN", "holo3")
    monkeypatch.setenv("MANTIS_SKIP_BRAIN_LOAD", "1")
    monkeypatch.setenv("MANTIS_RUNTIME_CONCURRENCY", "not-an-int")
    from mantis_agent.baseten_server.runtime import BasetenCUARuntime
    rt = BasetenCUARuntime()
    assert rt._runtime_concurrency == 1


def test_runtime_lock_zero_or_negative_clamped_to_one(
    tmp_path: Path, monkeypatch,
) -> None:
    """0 or -5 in the env shouldn't deadlock the runtime — clamp to 1."""
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MANTIS_BRAIN", "holo3")
    monkeypatch.setenv("MANTIS_SKIP_BRAIN_LOAD", "1")
    monkeypatch.setenv("MANTIS_RUNTIME_CONCURRENCY", "-3")
    from mantis_agent.baseten_server.runtime import BasetenCUARuntime
    rt = BasetenCUARuntime()
    assert rt._runtime_concurrency == 1


def test_runtime_lock_doesnt_serialize_when_two_runs_overlap(
    tmp_path: Path, monkeypatch,
) -> None:
    """Regression for the user-reported wedge: a fast run shouldn't
    sit behind a slow one when ``MANTIS_RUNTIME_CONCURRENCY=2``."""
    import threading
    import time
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MANTIS_BRAIN", "holo3")
    monkeypatch.setenv("MANTIS_SKIP_BRAIN_LOAD", "1")
    monkeypatch.setenv("MANTIS_RUNTIME_CONCURRENCY", "2")
    from mantis_agent.baseten_server.runtime import BasetenCUARuntime
    rt = BasetenCUARuntime()

    started: dict[str, float] = {}
    finished: dict[str, float] = {}

    def _slow():
        with rt.lock:
            started["slow"] = time.monotonic()
            time.sleep(0.5)
            finished["slow"] = time.monotonic()

    def _fast():
        # Tiny stagger so we measure the slow run's hold, not start order.
        time.sleep(0.05)
        with rt.lock:
            started["fast"] = time.monotonic()
            time.sleep(0.05)
            finished["fast"] = time.monotonic()

    t_slow = threading.Thread(target=_slow)
    t_fast = threading.Thread(target=_fast)
    t_slow.start()
    t_fast.start()
    t_slow.join()
    t_fast.join()

    # The fast run must start BEFORE the slow run finishes — proving
    # the two slots run concurrently rather than serializing on a
    # single global mutex (the pre-fix behaviour).
    assert "fast" in started and "slow" in started
    assert started["fast"] < finished["slow"]
