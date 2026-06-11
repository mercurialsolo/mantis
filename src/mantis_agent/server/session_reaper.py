"""Reaper logic for orphaned computer-plane sessions (Phase 1.5, #846, PR 4).

The brain's ``SessionRoutedComputerImpl.close()`` is the happy-path
teardown for a per-session container. The reaper is the safety net
when:

* The brain crashes mid-run (no close ever fires).
* The router crashes after spawning but before persisting the record
  cleanly.
* The session's TTL expires (long-running brain didn't renew).
* The orchestrator container crashed without writing its terminal
  status.

This module ships the pure-logic ``find_reapable_sessions`` function
that returns a structured plan; the Modal scheduled wrapper applies
the plan. Splitting them lets the test suite exercise reaper decisions
against a plain-dict :class:`RunStateStore` without involving Modal.

Decisions are deliberately conservative: only sessions whose
``expires_at_ms`` is in the past — or whose RunStateStore record is
already in a terminal status the brain did not produce — are reaped.
Idle-but-still-healthy sessions never get killed by this loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from ..session_wire import (
    SessionRecord,
    iter_session_records,
    serialize_session_record,
)
from ..run_state_store import KIND_SESSION

logger = logging.getLogger(__name__)


# Statuses the brain owns. Reaper never touches a session that's
# already in one of these — they're either explicitly closed by
# the brain or marked terminal by the orchestrator.
_BRAIN_TERMINAL_STATES = frozenset({"closed", "reaped", "error", "expired"})


@dataclass(frozen=True)
class ReapDecision:
    """One reaper action: terminate ``session_id`` because ``reason``."""

    session_id: str
    tenant_id: str
    sandbox_id: str
    reason: str           # ttl_expired | already_terminal | hint_field
    record: SessionRecord


def find_reapable_sessions(
    store: Any,
    *,
    now_ms: int,
    tenant_id: str | None = None,
) -> list[ReapDecision]:
    """Scan the RunStateStore and return sessions that should be reaped.

    ``now_ms`` — wall-clock at scan time. Splitting it out (not
    ``time.time``-based) keeps the function pure for tests.

    ``tenant_id`` — optional filter. ``None`` (default) scans all
    tenants the store knows about.

    Pure: doesn't mutate the store, doesn't issue terminations.
    Caller decides what to do with the list — typically pass it to
    :func:`apply_reap_decisions`.
    """
    decisions: list[ReapDecision] = []
    for record in iter_session_records(store, tenant_id=tenant_id):
        if record.status in _BRAIN_TERMINAL_STATES:
            continue
        if record.expires_at_ms and record.expires_at_ms < now_ms:
            decisions.append(ReapDecision(
                session_id=record.session_id,
                tenant_id=record.tenant_id,
                sandbox_id=record.sandbox_id,
                reason="ttl_expired",
                record=record,
            ))
            continue
        # An orchestrator that died mid-publish stamps status="error"
        # before exiting (or never updates from "active"). The
        # status=="error" case is caught by _BRAIN_TERMINAL_STATES
        # above. status=="active" with a missing base_url means the
        # router's wait loop bailed but the record was still
        # persisted — treat that as reapable so it doesn't clog the
        # active list.
        if record.status == "active" and not record.base_url:
            decisions.append(ReapDecision(
                session_id=record.session_id,
                tenant_id=record.tenant_id,
                sandbox_id=record.sandbox_id,
                reason="active_without_base_url",
                record=record,
            ))
    return decisions


def apply_reap_decisions(
    decisions: list[ReapDecision],
    *,
    store: Any,
    cancel_function_call: Callable[[str], None] | None = None,
    signal_close_in_session_dict: Callable[[str], None] | None = None,
    now_ms: int,
) -> dict:
    """Execute a reap plan against the store + Modal.

    Steps per decision:

    1. Cancel the Modal FunctionCall (so the container exits). Best-
       effort — a missing handle or already-terminated function is
       a no-op.
    2. Signal ``close_requested=True`` in the session_dict. Belt-and-
       braces — if the cancel succeeded the orchestrator already
       exited, but a long-tail container reading the dict will exit
       sooner.
    3. Update the RunStateStore record to ``status="reaped"`` with
       a small ``reaped_at_ms`` field appended via the model dump so
       the record stays forward-compatible.

    Returns a summary dict:

        {
          "reaped": <count>,
          "skipped": <count of decisions whose store update failed>,
          "errors": [{session_id, error}, ...]
        }
    """
    summary: dict[str, Any] = {"reaped": 0, "skipped": 0, "errors": []}
    for d in decisions:
        try:
            if cancel_function_call is not None and d.sandbox_id:
                try:
                    cancel_function_call(d.sandbox_id)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "reaper: cancel failed sandbox=%s (%s)",
                        d.sandbox_id, exc,
                    )
            if signal_close_in_session_dict is not None:
                try:
                    signal_close_in_session_dict(d.session_id)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "reaper: signal failed session=%s (%s)",
                        d.session_id, exc,
                    )
            updated = d.record.model_copy(update={
                "status": "reaped",
                "error": f"reaper: {d.reason}",
                "last_action_ms": now_ms,
            })
            store.put(
                d.tenant_id, d.session_id, KIND_SESSION,
                serialize_session_record(updated),
            )
            summary["reaped"] += 1
        except Exception as exc:  # noqa: BLE001
            summary["skipped"] += 1
            summary["errors"].append({
                "session_id": d.session_id, "error": f"{type(exc).__name__}: {exc}",
            })
            logger.warning(
                "reaper: apply failed session=%s (%s)", d.session_id, exc,
            )
    return summary
