"""Unit tests for the session-router wire contract (Phase 1.5, #846).

Foundation PR — wire models + KIND_SESSION on RunStateStore. No Modal,
no FastAPI, no HTTP.
"""

from __future__ import annotations

import time

import pytest

from mantis_agent.session_wire import (
    SessionCloseRequest,
    SessionCloseResponse,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionNotFoundError,
    SessionQuotaExceededError,
    SessionRecord,
    SessionRouterError,
    SessionUnreachableError,
    iter_session_records,
    parse_session_record,
    serialize_session_record,
)
from mantis_agent.run_state_store import KIND_SESSION, RunStateStore


# ── Wire models ───────────────────────────────────────────────────────


def test_session_create_request_defaults() -> None:
    req = SessionCreateRequest(
        tenant_id="t1", profile_id="alice", run_id="r1",
    )
    assert req.start_url == "about:blank"
    assert req.viewport == (1280, 720)
    assert req.enable_cdp is False
    assert req.ttl_seconds == 3600


def test_session_create_request_clamps_ttl() -> None:
    # Below floor → ValidationError
    with pytest.raises(Exception):
        SessionCreateRequest(
            tenant_id="t1", profile_id="p", run_id="r", ttl_seconds=30,
        )
    # Above ceiling → ValidationError
    with pytest.raises(Exception):
        SessionCreateRequest(
            tenant_id="t1", profile_id="p", run_id="r", ttl_seconds=999999,
        )


def test_session_create_request_round_trips_through_json() -> None:
    req = SessionCreateRequest(
        tenant_id="t1", profile_id="alice", run_id="r1",
        start_url="https://example.com",
        viewport=(1920, 1080),
        proxy_server="http://proxy:8080",
        profile_dir="/data/chrome-profile/x",
        extra_http_headers={"X-Bypass-Warning": "1"},
        enable_cdp=True,
        ttl_seconds=7200,
    )
    blob = req.model_dump(mode="json")
    same = SessionCreateRequest.model_validate(blob)
    assert same == req


def test_session_create_response_carries_required_fields() -> None:
    resp = SessionCreateResponse(
        session_id="sess_abc",
        base_url="https://t-abc.modal.run",
        session_token="tok_xyz",
        expires_at_ms=int(time.time() * 1000) + 3_600_000,
    )
    assert resp.reused is False
    assert resp.sandbox_id == ""


def test_session_close_request_default_reason() -> None:
    req = SessionCloseRequest(session_id="sess_1")
    assert req.reason == "brain_closed"


def test_session_close_response_defaults() -> None:
    resp = SessionCloseResponse(closed=True)
    assert resp.terminal_state == ""


# ── Error subclasses keep their http_status ───────────────────────────


def test_router_errors_carry_http_status() -> None:
    assert SessionNotFoundError.http_status == 404
    assert SessionUnreachableError.http_status == 504
    assert SessionQuotaExceededError.http_status == 429
    assert SessionRouterError.http_status == 500
    # All subclass the base.
    assert issubclass(SessionNotFoundError, SessionRouterError)
    assert issubclass(SessionUnreachableError, SessionRouterError)
    assert issubclass(SessionQuotaExceededError, SessionRouterError)


# ── parse / serialize round-trip ──────────────────────────────────────


def test_parse_session_record_round_trip() -> None:
    record = SessionRecord(
        session_id="sess_1",
        tenant_id="t1",
        profile_id="alice",
        run_id="r1",
        base_url="https://x.modal.run",
        sandbox_id="sb_1",
        session_token="tok_1",
        created_at_ms=1_000_000,
        expires_at_ms=2_000_000,
    )
    blob = serialize_session_record(record)
    assert blob["session_id"] == "sess_1"
    parsed = parse_session_record(blob)
    assert parsed == record


def test_parse_session_record_handles_none_and_garbage() -> None:
    assert parse_session_record(None) is None
    assert parse_session_record({}) is None
    assert parse_session_record({"session_id": 1}) is None
    assert parse_session_record({"unexpected": "shape"}) is None


def test_parse_session_record_tolerates_extra_fields() -> None:
    """Forward compat — a newer router can add fields without breaking
    an older reaper that reads existing records."""
    record = SessionRecord(
        session_id="sess_1", tenant_id="t1", profile_id="p", run_id="r",
        base_url="x", created_at_ms=0, expires_at_ms=0,
    )
    blob = serialize_session_record(record) | {"future_field": "ignored"}
    parsed = parse_session_record(blob)
    assert parsed is not None
    assert parsed.session_id == "sess_1"


# ── iter_session_records: list across tenants + filter by tenant ─────


def _store_with_sessions(records: list[SessionRecord]) -> RunStateStore:
    """Plain-dict-backed store with KIND_SESSION blobs for each record.

    Session blobs are stored under the (tenant_id, session_id,
    KIND_SESSION) tuple — session_id occupies the run_id slot of the
    canonical key, because sessions outlive runs sometimes (reaper
    sweep recreates).
    """
    store = RunStateStore(backing={})
    for r in records:
        store.put(r.tenant_id, r.session_id, KIND_SESSION,
                  serialize_session_record(r))
    return store


def _mk(session_id: str, tenant_id: str = "t1") -> SessionRecord:
    return SessionRecord(
        session_id=session_id, tenant_id=tenant_id,
        profile_id="p", run_id="r",
        base_url=f"https://{session_id}.x",
        created_at_ms=0, expires_at_ms=0,
    )


def test_iter_session_records_returns_all_across_tenants() -> None:
    store = _store_with_sessions([_mk("a"), _mk("b", "t2"), _mk("c", "t3")])
    out = list(iter_session_records(store))
    ids = sorted(r.session_id for r in out)
    assert ids == ["a", "b", "c"]


def test_iter_session_records_filters_by_tenant() -> None:
    store = _store_with_sessions([_mk("a", "alice"), _mk("b", "bob"), _mk("c", "alice")])
    out = sorted(r.session_id for r in iter_session_records(store, tenant_id="alice"))
    assert out == ["a", "c"]


def test_iter_session_records_skips_unrelated_kinds() -> None:
    """The store is shared with KIND_STATUS / KIND_VIEWER / etc.;
    listing must not pick those up."""
    from mantis_agent.run_state_store import (
        KIND_AUGUR,
        KIND_STATUS,
        KIND_VIEWER,
    )
    store = _store_with_sessions([_mk("a")])
    store.put("t1", "run-1", KIND_STATUS, {"status": "running"})
    store.put("t1", "run-1", KIND_VIEWER, {"viewer_url": "x"})
    store.put("t1", "run-1", KIND_AUGUR, {"augur_run_id": "y"})
    out = [r.session_id for r in iter_session_records(store)]
    assert out == ["a"]


def test_iter_session_records_handles_corrupt_blob_gracefully() -> None:
    """A SessionRecord blob written by a newer/older shape that fails
    validation must be skipped, not raised."""
    store = _store_with_sessions([_mk("ok")])
    # Inject a corrupt blob under a session-shaped key by reaching
    # into the backing dict directly.
    store._d["t1/corrupt/session"] = {"not_a_session": True}  # noqa: SLF001
    out = [r.session_id for r in iter_session_records(store)]
    assert out == ["ok"]


def test_iter_session_records_with_unknown_tenant_returns_nothing() -> None:
    store = _store_with_sessions([_mk("a", "alice")])
    assert list(iter_session_records(store, tenant_id="bob")) == []


# ── KIND_SESSION registered ───────────────────────────────────────────


def test_kind_session_round_trip_through_store() -> None:
    """Smoke: KIND_SESSION is wired into the store's validator set so
    put/get with it works without raising."""
    store = RunStateStore(backing={})
    blob = {"hello": "world"}
    store.put("t1", "sess_x", KIND_SESSION, blob)
    assert store.get("t1", "sess_x", KIND_SESSION) == blob


def test_kind_session_typo_still_rejected() -> None:
    """Sanity — the validator wasn't loosened when KIND_SESSION was added."""
    store = RunStateStore(backing={})
    with pytest.raises(ValueError):
        store.put("t1", "sess_x", "sesion", {})  # type: ignore[arg-type]
