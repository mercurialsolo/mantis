"""Tests for the session reaper (Phase 1.5, #846, PR 4)."""

from __future__ import annotations

from mantis_agent.session_wire import (
    SessionRecord,
    serialize_session_record,
)
from mantis_agent.run_state_store import KIND_SESSION, RunStateStore
from mantis_agent.server.session_reaper import (
    apply_reap_decisions,
    find_reapable_sessions,
)


def _mk(
    session_id: str,
    *,
    tenant_id: str = "t1",
    expires_at_ms: int = 1_000_000,
    status: str = "active",
    base_url: str = "https://x.modal.run",
    sandbox_id: str = "fc-1",
) -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        tenant_id=tenant_id,
        profile_id="p",
        run_id="r",
        base_url=base_url,
        sandbox_id=sandbox_id,
        created_at_ms=0,
        expires_at_ms=expires_at_ms,
        status=status,
    )


def _seed(store: RunStateStore, records: list[SessionRecord]) -> None:
    for r in records:
        store.put(r.tenant_id, r.session_id, KIND_SESSION,
                  serialize_session_record(r))


# ── find_reapable_sessions ────────────────────────────────────────────


def test_finds_expired_sessions() -> None:
    store = RunStateStore(backing={})
    _seed(store, [
        _mk("expired_1", expires_at_ms=500),    # before now=1000
        _mk("expired_2", expires_at_ms=999),
        _mk("alive",     expires_at_ms=5000),
    ])
    out = find_reapable_sessions(store, now_ms=1000)
    ids = sorted(d.session_id for d in out)
    assert ids == ["expired_1", "expired_2"]
    assert all(d.reason == "ttl_expired" for d in out)


def test_skips_brain_terminal_states() -> None:
    """Records the brain already closed (or the orchestrator marked
    error) must not be reaped a second time."""
    store = RunStateStore(backing={})
    _seed(store, [
        _mk("alive_then_closed", expires_at_ms=500, status="closed"),
        _mk("alive_then_reaped", expires_at_ms=500, status="reaped"),
        _mk("alive_then_error",  expires_at_ms=500, status="error"),
        _mk("alive_then_exp",    expires_at_ms=500, status="expired"),
        _mk("still_active",      expires_at_ms=500, status="active"),
    ])
    out = find_reapable_sessions(store, now_ms=1000)
    assert [d.session_id for d in out] == ["still_active"]


def test_flags_active_without_base_url_as_reapable() -> None:
    """Router got partway through then crashed — recorded as active
    with no base_url. Reaper should clean these up too."""
    store = RunStateStore(backing={})
    _seed(store, [
        _mk("orphan", expires_at_ms=99_999, status="active", base_url=""),
        _mk("good",   expires_at_ms=99_999, status="active"),
    ])
    out = find_reapable_sessions(store, now_ms=1000)
    assert [d.session_id for d in out] == ["orphan"]
    assert out[0].reason == "active_without_base_url"


def test_tenant_filter_scope() -> None:
    store = RunStateStore(backing={})
    _seed(store, [
        _mk("alice_x", tenant_id="alice", expires_at_ms=500),
        _mk("bob_y",   tenant_id="bob",   expires_at_ms=500),
    ])
    out = find_reapable_sessions(store, now_ms=1000, tenant_id="alice")
    assert [d.session_id for d in out] == ["alice_x"]


# ── apply_reap_decisions ──────────────────────────────────────────────


def test_apply_marks_records_as_reaped() -> None:
    store = RunStateStore(backing={})
    _seed(store, [_mk("victim", expires_at_ms=500, sandbox_id="fc-victim")])
    decisions = find_reapable_sessions(store, now_ms=1000)

    cancel_calls: list[str] = []
    signal_calls: list[str] = []
    summary = apply_reap_decisions(
        decisions, store=store,
        cancel_function_call=cancel_calls.append,
        signal_close_in_session_dict=signal_calls.append,
        now_ms=1000,
    )
    assert summary == {"reaped": 1, "skipped": 0, "errors": []}
    assert cancel_calls == ["fc-victim"]
    assert signal_calls == ["victim"]

    # Record now status="reaped" with the reason annotation.
    blob = store.get("t1", "victim", KIND_SESSION)
    assert blob is not None
    assert blob["status"] == "reaped"
    assert "ttl_expired" in blob["error"]
    assert blob["last_action_ms"] == 1000


def test_apply_tolerates_callback_failures() -> None:
    """A misbehaving cancel/signal callback must not abort the whole
    sweep — the record still gets marked, and the next decision still
    runs."""
    store = RunStateStore(backing={})
    _seed(store, [
        _mk("a", expires_at_ms=500, sandbox_id="fc-a"),
        _mk("b", expires_at_ms=500, sandbox_id="fc-b"),
    ])

    def _bad_cancel(_sid: str) -> None:
        raise RuntimeError("modal control plane unreachable")

    decisions = find_reapable_sessions(store, now_ms=1000)
    summary = apply_reap_decisions(
        decisions, store=store,
        cancel_function_call=_bad_cancel,
        signal_close_in_session_dict=lambda _sid: None,
        now_ms=1000,
    )
    # Both records still reaped — cancel failures are observability, not blockers.
    assert summary["reaped"] == 2
    assert summary["errors"] == []


def test_apply_records_summary_on_store_failure() -> None:
    """If the store write itself fails for a single decision, the
    failure is recorded but the remaining decisions still process."""

    class _FlakyStore(RunStateStore):
        def __init__(self) -> None:
            super().__init__(backing={})
            self._fail_for: str | None = None  # set after seed

        def put(self, tenant_id, run_id, kind, value):
            if self._fail_for is not None and run_id == self._fail_for:
                raise RuntimeError("disk full")
            super().put(tenant_id, run_id, kind, value)

    store = _FlakyStore()
    _seed(store, [
        _mk("a", expires_at_ms=500),
        _mk("b", expires_at_ms=500),
        _mk("c", expires_at_ms=500),
    ])
    # Now arm the failure so the apply phase trips on "b".
    store._fail_for = "b"  # noqa: SLF001
    decisions = find_reapable_sessions(store, now_ms=1000)
    summary = apply_reap_decisions(
        decisions, store=store, now_ms=1000,
    )
    assert summary["reaped"] == 2
    assert summary["skipped"] == 1
    assert summary["errors"][0]["session_id"] == "b"
    assert "disk full" in summary["errors"][0]["error"]


def test_apply_handles_empty_decision_list() -> None:
    summary = apply_reap_decisions([], store=RunStateStore(backing={}), now_ms=1000)
    assert summary == {"reaped": 0, "skipped": 0, "errors": []}


# ── full sweep integration ────────────────────────────────────────────


def test_full_sweep_end_to_end() -> None:
    """Realistic mix: some expired, some healthy, some already-closed.
    Final state: only the expired ones get reaped, others untouched."""
    store = RunStateStore(backing={})
    now_ms = 5_000_000
    _seed(store, [
        _mk("expired_a", expires_at_ms=now_ms - 1000),
        _mk("expired_b", expires_at_ms=now_ms - 500),
        _mk("alive_a",   expires_at_ms=now_ms + 10_000),
        _mk("closed",    expires_at_ms=now_ms - 1000, status="closed"),
    ])
    decisions = find_reapable_sessions(store, now_ms=now_ms)
    apply_reap_decisions(decisions, store=store, now_ms=now_ms)

    assert store.get("t1", "expired_a", KIND_SESSION)["status"] == "reaped"
    assert store.get("t1", "expired_b", KIND_SESSION)["status"] == "reaped"
    assert store.get("t1", "alive_a", KIND_SESSION)["status"] == "active"
    assert store.get("t1", "closed", KIND_SESSION)["status"] == "closed"
