"""Run lifecycle — cancel, timeout, queue status, stable polling (#784).

Data layer for the lifecycle controls the HN URL-collection user
report flagged as missing (#785). PR 7 ships:

- `RunPhase` enum — six first-class phases.
- Pydantic schemas for the new endpoints (cancel / status / queue).
- `RunLifecycleStore` — in-memory state with TTL + cancel-token
  semantics. Operators can subclass / swap for Redis / Modal Dict.
- Halt-class constants `cancelled` + `halt_timeout` (first-class so
  dashboards and Augur tags can group separately from runtime errors).
- CLI helpers used by `mantis runs cancel` / `mantis runs watch`.

Integration into the existing `/v1/predict` FastAPI surface
(`modal_cua_server.py`, `baseten_server/routes.py`) is the focused
follow-up — this PR ships the data structures + a router module that
operators can mount.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel


# ── Phase taxonomy ──────────────────────────────────────────────────


class RunPhase(str, Enum):
    """Distinct phase a run is in.

    Surfaced on `GET /v1/runs/{id}`. Phases are first-class — separate
    from `success / failure / cancelled` outcome attribution so
    dashboards can group "still working" (queued / running /
    recovering) vs "done" (halted / cancelled / complete) without
    string-matching.
    """

    QUEUED = "queued"
    RUNNING = "running"
    RECOVERING = "recovering"  # in step_recovery loop after a failure
    HALTED = "halted"          # finished with a halt class
    CANCELLED = "cancelled"    # explicit POST /cancel
    COMPLETE = "complete"      # finished successfully

    @property
    def is_terminal(self) -> bool:
        return self in (RunPhase.HALTED, RunPhase.CANCELLED, RunPhase.COMPLETE)


# Halt classes added by lifecycle controls. These are wire constants —
# stable strings on the `summary.halt_class` field of terminal runs.
HALT_CANCELLED: str = "cancelled"
HALT_TIMEOUT: str = "halt_timeout"


# ── Wire schemas ────────────────────────────────────────────────────


class CancelRunRequest(BaseModel):
    """Body of `POST /v1/runs/{id}/cancel`.

    `reason` is logged for operator triage. Cancellation is idempotent
    on `run_id` — a second cancel returns the same final status.
    """

    reason: str = "operator_cancel"


class CancelRunResponse(BaseModel):
    cancelled: bool
    phase: RunPhase
    halt_class: str | None = None
    reason: str | None = None
    final_status_url: str | None = None


class RunPhaseResponse(BaseModel):
    """Response shape of `GET /v1/runs/{id}` lifecycle view.

    Backwards-compat: existing `RunStatus` (api_schemas.py) still
    works for full-detail polling. This response is the "phase only"
    cheap-poll variant that returns the polling backoff hint so
    clients stop hammering when nothing's changing.
    """

    run_id: str
    phase: RunPhase
    last_event_at: float | None = None
    polling_backoff_ms_hint: int = 1000
    started_at: float | None = None
    finished_at: float | None = None
    halt_class: str | None = None


class QueueStatusResponse(BaseModel):
    """Response shape of `GET /v1/queue`.

    Per-tenant view (operators authenticate via existing tenant
    middleware; the response is scoped to the caller's tenant).
    `eta_ms` is best-effort; null when no recent dispatch latency
    samples exist.
    """

    tenant_id: str
    queued: int
    running: int
    recovering: int
    eta_ms: int | None = None


# ── Lifecycle store ─────────────────────────────────────────────────


@dataclass
class RunState:
    """In-memory state for one run.

    Operators can swap this for a Redis-backed implementation by
    subclassing `RunLifecycleStore` and overriding `_get` / `_put` /
    `_iter_tenant`. The dataclass shape is the contract.
    """

    run_id: str
    tenant_id: str
    phase: RunPhase = RunPhase.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    last_event_at: float | None = None
    timeout_seconds: int | None = None  # set at submit
    cancel_requested: bool = False
    cancel_reason: str | None = None
    halt_class: str | None = None
    # Adaptive polling: tracks how fast state actually changes. A run
    # that hasn't moved in 10s gets a longer backoff hint.
    last_phase_change_ms: float = field(default_factory=lambda: time.time() * 1000)


@dataclass
class RunLifecycleStore:
    """In-memory store + helpers. Operators can subclass for durability.

    Single-process by design — production deploys with N replicas need
    a shared backend (Redis / Modal Dict). The store carries TTL-bound
    state so cancel tokens don't grow unbounded.
    """

    ttl_seconds: int = 3600  # entries older than this are eligible for GC
    _runs: dict[str, RunState] = field(default_factory=dict)

    # ── CRUD ──

    def register(
        self,
        run_id: str,
        tenant_id: str,
        timeout_seconds: int | None = None,
    ) -> RunState:
        st = RunState(
            run_id=run_id, tenant_id=tenant_id, timeout_seconds=timeout_seconds
        )
        self._runs[run_id] = st
        return st

    def get(self, run_id: str) -> RunState | None:
        return self._runs.get(run_id)

    def transition(self, run_id: str, phase: RunPhase) -> RunState | None:
        st = self._runs.get(run_id)
        if st is None:
            return None
        # Idempotent — re-entering the same phase doesn't bump
        # last_phase_change so polling backoffs grow correctly.
        if st.phase is not phase:
            st.phase = phase
            st.last_phase_change_ms = time.time() * 1000
        st.last_event_at = time.time()
        if phase is RunPhase.RUNNING and st.started_at is None:
            st.started_at = time.time()
        if phase.is_terminal and st.finished_at is None:
            st.finished_at = time.time()
        return st

    # ── Cancellation ──

    def request_cancel(self, run_id: str, reason: str) -> RunState | None:
        """Mark for cancellation. Idempotent — second call returns the
        same state. Runner polls `should_cancel` to honor."""
        st = self._runs.get(run_id)
        if st is None:
            return None
        if not st.cancel_requested:
            st.cancel_requested = True
            st.cancel_reason = reason
        return st

    def should_cancel(self, run_id: str) -> bool:
        st = self._runs.get(run_id)
        return bool(st and st.cancel_requested)

    def mark_cancelled(self, run_id: str) -> RunState | None:
        st = self.transition(run_id, RunPhase.CANCELLED)
        if st is not None:
            st.halt_class = HALT_CANCELLED
        return st

    # ── Timeout ──

    def is_timed_out(self, run_id: str, *, now: float | None = None) -> bool:
        """Pure check — caller invokes per heartbeat. Server-side
        enforcement transitions to HALTED with `halt_class=halt_timeout`."""
        st = self._runs.get(run_id)
        if st is None or st.timeout_seconds is None or st.started_at is None:
            return False
        t = now if now is not None else time.time()
        return (t - st.started_at) >= st.timeout_seconds

    def mark_timeout_halt(self, run_id: str) -> RunState | None:
        st = self.transition(run_id, RunPhase.HALTED)
        if st is not None:
            st.halt_class = HALT_TIMEOUT
        return st

    # ── Polling backoff hint ──

    def polling_backoff_ms(self, run_id: str, *, now: float | None = None) -> int:
        """Adaptive hint: how long should a client wait before polling
        this run again?

        Heuristic: terminal phases give a long hint (state won't
        change). Active phases that haven't transitioned in 10s get
        progressively longer hints. Fresh transitions (<2s ago) get
        the short hint.
        """
        st = self._runs.get(run_id)
        if st is None:
            return 5000
        if st.phase.is_terminal:
            return 30_000
        t_ms = (now if now is not None else time.time()) * 1000
        since_change_ms = max(0.0, t_ms - st.last_phase_change_ms)
        if since_change_ms < 2_000:
            return 500
        if since_change_ms < 10_000:
            return 1_500
        if since_change_ms < 30_000:
            return 5_000
        return 10_000

    # ── Queue snapshot ──

    def queue_snapshot(self, tenant_id: str) -> QueueStatusResponse:
        queued = running = recovering = 0
        for st in self._runs.values():
            if st.tenant_id != tenant_id:
                continue
            if st.phase is RunPhase.QUEUED:
                queued += 1
            elif st.phase is RunPhase.RUNNING:
                running += 1
            elif st.phase is RunPhase.RECOVERING:
                recovering += 1
        return QueueStatusResponse(
            tenant_id=tenant_id,
            queued=queued,
            running=running,
            recovering=recovering,
            eta_ms=None,
        )

    # ── GC ──

    def gc(self, *, now: float | None = None) -> int:
        """Drop terminal runs older than `ttl_seconds`. Returns count."""
        t = now if now is not None else time.time()
        to_drop = [
            rid
            for rid, st in self._runs.items()
            if st.phase.is_terminal
            and st.finished_at is not None
            and (t - st.finished_at) >= self.ttl_seconds
        ]
        for rid in to_drop:
            self._runs.pop(rid, None)
        return len(to_drop)


# ── Helper for FastAPI integration ──────────────────────────────────


def build_phase_response(st: RunState, store: RunLifecycleStore) -> RunPhaseResponse:
    return RunPhaseResponse(
        run_id=st.run_id,
        phase=st.phase,
        last_event_at=st.last_event_at,
        polling_backoff_ms_hint=store.polling_backoff_ms(st.run_id),
        started_at=st.started_at,
        finished_at=st.finished_at,
        halt_class=st.halt_class,
    )


__all__ = [
    "CancelRunRequest",
    "CancelRunResponse",
    "HALT_CANCELLED",
    "HALT_TIMEOUT",
    "QueueStatusResponse",
    "RunLifecycleStore",
    "RunPhase",
    "RunPhaseResponse",
    "RunState",
    "build_phase_response",
]
