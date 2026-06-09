"""Tests for run lifecycle data layer (#784, PR 7)."""

from __future__ import annotations

import time


from mantis_agent.run_lifecycle import (
    HALT_CANCELLED,
    HALT_TIMEOUT,
    CancelRunRequest,
    CancelRunResponse,
    QueueStatusResponse,
    RunLifecycleStore,
    RunPhase,
    RunPhaseResponse,
    build_phase_response,
)


# ── RunPhase ────────────────────────────────────────────────────


def test_run_phase_terminal_set():
    assert RunPhase.HALTED.is_terminal
    assert RunPhase.CANCELLED.is_terminal
    assert RunPhase.COMPLETE.is_terminal
    assert not RunPhase.QUEUED.is_terminal
    assert not RunPhase.RUNNING.is_terminal
    assert not RunPhase.RECOVERING.is_terminal


def test_run_phase_string_values_stable():
    # Wire constants — don't change without bumping a schema version.
    assert RunPhase.QUEUED.value == "queued"
    assert RunPhase.RUNNING.value == "running"
    assert RunPhase.RECOVERING.value == "recovering"
    assert RunPhase.HALTED.value == "halted"
    assert RunPhase.CANCELLED.value == "cancelled"
    assert RunPhase.COMPLETE.value == "complete"


# ── Halt class constants ────────────────────────────────────────


def test_halt_class_constants():
    assert HALT_CANCELLED == "cancelled"
    assert HALT_TIMEOUT == "halt_timeout"


# ── RunLifecycleStore: register + transition ───────────────────


def test_register_initializes_queued():
    s = RunLifecycleStore()
    st = s.register("r1", "tenant-a", timeout_seconds=300)
    assert st.phase is RunPhase.QUEUED
    assert st.tenant_id == "tenant-a"
    assert st.timeout_seconds == 300
    assert st.created_at > 0
    assert st.started_at is None
    assert st.finished_at is None


def test_transition_sets_started_at_on_running():
    s = RunLifecycleStore()
    s.register("r1", "t1")
    st = s.transition("r1", RunPhase.RUNNING)
    assert st is not None
    assert st.phase is RunPhase.RUNNING
    assert st.started_at is not None
    assert st.last_event_at is not None


def test_transition_to_terminal_sets_finished_at():
    s = RunLifecycleStore()
    s.register("r1", "t1")
    s.transition("r1", RunPhase.RUNNING)
    st = s.transition("r1", RunPhase.COMPLETE)
    assert st is not None
    assert st.finished_at is not None


def test_transition_idempotent_keeps_phase_change_time():
    s = RunLifecycleStore()
    s.register("r1", "t1")
    s.transition("r1", RunPhase.RUNNING)
    first_change = s.get("r1").last_phase_change_ms  # type: ignore[union-attr]
    time.sleep(0.01)
    s.transition("r1", RunPhase.RUNNING)  # same phase
    second_change = s.get("r1").last_phase_change_ms  # type: ignore[union-attr]
    assert first_change == second_change


def test_transition_to_new_phase_bumps_change_time():
    s = RunLifecycleStore()
    s.register("r1", "t1")
    s.transition("r1", RunPhase.RUNNING)
    first_change = s.get("r1").last_phase_change_ms  # type: ignore[union-attr]
    time.sleep(0.01)
    s.transition("r1", RunPhase.RECOVERING)
    second_change = s.get("r1").last_phase_change_ms  # type: ignore[union-attr]
    assert second_change > first_change


def test_transition_missing_run_returns_none():
    s = RunLifecycleStore()
    assert s.transition("nonexistent", RunPhase.RUNNING) is None


# ── Cancellation ───────────────────────────────────────────────


def test_cancel_idempotent():
    s = RunLifecycleStore()
    s.register("r1", "t1")
    s.request_cancel("r1", reason="first")
    s.request_cancel("r1", reason="second")
    st = s.get("r1")
    assert st is not None
    assert st.cancel_requested is True
    # First reason sticks; second is a no-op.
    assert st.cancel_reason == "first"


def test_should_cancel_polls_correctly():
    s = RunLifecycleStore()
    s.register("r1", "t1")
    assert s.should_cancel("r1") is False
    s.request_cancel("r1", reason="op")
    assert s.should_cancel("r1") is True


def test_should_cancel_unknown_run():
    s = RunLifecycleStore()
    assert s.should_cancel("ghost") is False


def test_mark_cancelled_sets_halt_class():
    s = RunLifecycleStore()
    s.register("r1", "t1")
    s.transition("r1", RunPhase.RUNNING)
    st = s.mark_cancelled("r1")
    assert st is not None
    assert st.phase is RunPhase.CANCELLED
    assert st.halt_class == HALT_CANCELLED
    assert st.finished_at is not None


# ── Timeout ────────────────────────────────────────────────────


def test_is_timed_out_false_without_timeout():
    s = RunLifecycleStore()
    s.register("r1", "t1", timeout_seconds=None)
    s.transition("r1", RunPhase.RUNNING)
    assert s.is_timed_out("r1") is False


def test_is_timed_out_false_before_start():
    s = RunLifecycleStore()
    s.register("r1", "t1", timeout_seconds=10)
    # Queued; not started.
    assert s.is_timed_out("r1") is False


def test_is_timed_out_when_elapsed_exceeds():
    s = RunLifecycleStore()
    s.register("r1", "t1", timeout_seconds=10)
    s.transition("r1", RunPhase.RUNNING)
    started = s.get("r1").started_at  # type: ignore[union-attr]
    # Simulate elapsed = 15s by passing future `now`.
    assert s.is_timed_out("r1", now=started + 15) is True
    assert s.is_timed_out("r1", now=started + 5) is False


def test_mark_timeout_halt_sets_halt_class():
    s = RunLifecycleStore()
    s.register("r1", "t1", timeout_seconds=10)
    s.transition("r1", RunPhase.RUNNING)
    st = s.mark_timeout_halt("r1")
    assert st is not None
    assert st.phase is RunPhase.HALTED
    assert st.halt_class == HALT_TIMEOUT


# ── Polling backoff hint ───────────────────────────────────────


def test_polling_backoff_short_on_fresh_transition():
    s = RunLifecycleStore()
    s.register("r1", "t1")
    s.transition("r1", RunPhase.RUNNING)
    # Just transitioned — 500ms hint.
    hint = s.polling_backoff_ms("r1")
    assert hint == 500


def test_polling_backoff_grows_on_stale_running():
    s = RunLifecycleStore()
    s.register("r1", "t1")
    s.transition("r1", RunPhase.RUNNING)
    started = s.get("r1").last_phase_change_ms  # type: ignore[union-attr]
    # 5s since last change — 1500ms tier.
    assert s.polling_backoff_ms("r1", now=(started + 5_000) / 1000) == 1_500
    # 20s — 5000ms tier.
    assert s.polling_backoff_ms("r1", now=(started + 20_000) / 1000) == 5_000
    # 60s — 10000ms tier.
    assert s.polling_backoff_ms("r1", now=(started + 60_000) / 1000) == 10_000


def test_polling_backoff_terminal_phase_is_long():
    s = RunLifecycleStore()
    s.register("r1", "t1")
    s.transition("r1", RunPhase.RUNNING)
    s.transition("r1", RunPhase.COMPLETE)
    assert s.polling_backoff_ms("r1") == 30_000


def test_polling_backoff_unknown_run_is_default():
    s = RunLifecycleStore()
    assert s.polling_backoff_ms("ghost") == 5_000


# ── Queue snapshot ─────────────────────────────────────────────


def test_queue_snapshot_counts_by_phase():
    s = RunLifecycleStore()
    s.register("a", "tenant1")
    s.register("b", "tenant1")
    s.transition("b", RunPhase.RUNNING)
    s.register("c", "tenant1")
    s.transition("c", RunPhase.RUNNING)
    s.transition("c", RunPhase.RECOVERING)
    s.register("d", "tenant2")  # different tenant
    snap = s.queue_snapshot("tenant1")
    assert snap.queued == 1   # a
    assert snap.running == 1  # b
    assert snap.recovering == 1  # c
    # d isn't in tenant1's snapshot.


def test_queue_snapshot_for_empty_tenant():
    s = RunLifecycleStore()
    snap = s.queue_snapshot("nobody")
    assert snap.queued == 0
    assert snap.running == 0
    assert snap.recovering == 0


# ── build_phase_response ───────────────────────────────────────


def test_build_phase_response_carries_polling_hint():
    s = RunLifecycleStore()
    s.register("r1", "t1", timeout_seconds=60)
    s.transition("r1", RunPhase.RUNNING)
    resp = build_phase_response(s.get("r1"), s)  # type: ignore[arg-type]
    assert resp.run_id == "r1"
    assert resp.phase is RunPhase.RUNNING
    assert resp.polling_backoff_ms_hint == 500
    assert resp.started_at is not None


def test_build_phase_response_for_terminal_carries_halt_class():
    s = RunLifecycleStore()
    s.register("r1", "t1")
    s.transition("r1", RunPhase.RUNNING)
    s.mark_cancelled("r1")
    resp = build_phase_response(s.get("r1"), s)  # type: ignore[arg-type]
    assert resp.phase is RunPhase.CANCELLED
    assert resp.halt_class == HALT_CANCELLED
    assert resp.polling_backoff_ms_hint == 30_000


# ── GC ─────────────────────────────────────────────────────────


def test_gc_drops_terminal_runs_past_ttl():
    s = RunLifecycleStore(ttl_seconds=10)
    s.register("r1", "t1")
    s.transition("r1", RunPhase.COMPLETE)
    s.register("r2", "t1")
    s.transition("r2", RunPhase.COMPLETE)
    # Simulate clock running forward by 30s.
    finished = s.get("r1").finished_at  # type: ignore[union-attr]
    dropped = s.gc(now=finished + 30)
    assert dropped == 2
    assert s.get("r1") is None
    assert s.get("r2") is None


def test_gc_keeps_non_terminal_runs():
    s = RunLifecycleStore(ttl_seconds=1)
    s.register("r1", "t1")
    s.transition("r1", RunPhase.RUNNING)
    # No finished_at — gc has no signal to drop.
    dropped = s.gc(now=time.time() + 1000)
    assert dropped == 0
    assert s.get("r1") is not None


def test_gc_keeps_recently_finished_runs():
    s = RunLifecycleStore(ttl_seconds=60)
    s.register("r1", "t1")
    s.transition("r1", RunPhase.COMPLETE)
    finished = s.get("r1").finished_at  # type: ignore[union-attr]
    # Only 10s past finished — within TTL.
    dropped = s.gc(now=finished + 10)
    assert dropped == 0


# ── Pydantic schemas ───────────────────────────────────────────


def test_cancel_request_default_reason():
    req = CancelRunRequest()
    assert req.reason == "operator_cancel"


def test_cancel_response_shape():
    resp = CancelRunResponse(cancelled=True, phase=RunPhase.CANCELLED, halt_class=HALT_CANCELLED)
    assert resp.cancelled is True
    assert resp.phase is RunPhase.CANCELLED


def test_queue_status_shape():
    resp = QueueStatusResponse(tenant_id="t", queued=3, running=1, recovering=0)
    assert resp.queued == 3


def test_run_phase_response_shape():
    resp = RunPhaseResponse(run_id="r1", phase=RunPhase.RUNNING, polling_backoff_ms_hint=500)
    assert resp.phase is RunPhase.RUNNING


# ── File-backed phase derivation (#806) ─────────────────────────


def test_phase_from_status_string_known_values():
    from mantis_agent.run_lifecycle import phase_from_status_string

    assert phase_from_status_string("queued") is RunPhase.QUEUED
    assert phase_from_status_string("running") is RunPhase.RUNNING
    assert phase_from_status_string("paused") is RunPhase.RUNNING
    assert phase_from_status_string("recovering") is RunPhase.RECOVERING
    assert phase_from_status_string("succeeded") is RunPhase.COMPLETE
    assert phase_from_status_string("completed_with_failures") is RunPhase.COMPLETE
    assert phase_from_status_string("cancelled") is RunPhase.CANCELLED
    assert phase_from_status_string("failed") is RunPhase.HALTED
    assert phase_from_status_string("halted") is RunPhase.HALTED
    assert phase_from_status_string("timeout") is RunPhase.HALTED


def test_phase_from_status_string_handles_empty():
    from mantis_agent.run_lifecycle import phase_from_status_string

    assert phase_from_status_string(None) is RunPhase.QUEUED
    assert phase_from_status_string("") is RunPhase.QUEUED


def test_phase_from_status_string_unknown_defaults_running():
    """An executor that surfaces an unfamiliar status string shouldn't
    crash the lifecycle endpoint — treat it as RUNNING (caller keeps
    polling) rather than QUEUED (would falsely imply pre-start)."""
    from mantis_agent.run_lifecycle import phase_from_status_string

    assert phase_from_status_string("waiting_for_proxy") is RunPhase.RUNNING


def test_build_phase_response_from_status_complete_carries_finished_at():
    from mantis_agent.run_lifecycle import build_phase_response_from_status

    resp = build_phase_response_from_status({
        "run_id": "r-done",
        "status": "succeeded",
        "created_at": "2026-06-08T00:00:00+00:00",
        "updated_at": "2026-06-08T00:01:00+00:00",
    })
    assert resp.phase is RunPhase.COMPLETE
    assert resp.finished_at is not None
    assert resp.started_at is not None
    assert resp.started_at < resp.finished_at


def test_build_phase_response_from_status_running_no_finished_at():
    from mantis_agent.run_lifecycle import build_phase_response_from_status

    resp = build_phase_response_from_status({
        "run_id": "r-r",
        "status": "running",
        "created_at": "2026-06-08T00:00:00+00:00",
        "updated_at": "2026-06-08T00:00:30+00:00",
    })
    assert resp.phase is RunPhase.RUNNING
    assert resp.finished_at is None


def test_build_phase_response_from_status_halt_class_for_timeout():
    from mantis_agent.run_lifecycle import build_phase_response_from_status

    resp = build_phase_response_from_status({
        "run_id": "r-t",
        "status": "timeout",
        "created_at": "2026-06-08T00:00:00+00:00",
        "updated_at": "2026-06-08T00:30:00+00:00",
    })
    assert resp.phase is RunPhase.HALTED
    assert resp.halt_class == HALT_TIMEOUT


def test_build_phase_response_from_status_tolerates_missing_timestamps():
    from mantis_agent.run_lifecycle import build_phase_response_from_status

    resp = build_phase_response_from_status({"run_id": "r-bare", "status": "running"})
    assert resp.phase is RunPhase.RUNNING
    # No started_at → no finished_at, but the response builds without error.
