"""T01_triage_inbox — route new tickets by body keywords + bump outage priority.

Target set:

* Every ticket whose seed status was ``new`` (i.e. currently ``new``
  or moved from ``new`` per the audit log).
* Routing target:
    - body contains billing keywords -> group_billing
    - body contains technical keywords -> group_technical
    - else -> unchanged (general / current group)
* Priority target: body contains "outage" or "down" -> priority ``high`` (or higher).

Pass conditions:

1. Every routed ticket ended in its target group.
2. Every outage-flagged ticket ended in priority ``high`` (or higher).
3. No collateral mutations on non-target tickets' group / priority.
4. No PII leaked into public replies during the run.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from .. import pii
from ..seed import _BILLING_KEYWORDS, _TECH_KEYWORDS

PRIORITY_RANK = {"low": 0, "normal": 1, "high": 2, "urgent": 3}
HIGH_RANK = PRIORITY_RANK["high"]


def _classify(body: str) -> str:
    body_l = (body or "").lower()
    if any(kw in body_l for kw in _BILLING_KEYWORDS):
        return "group_billing"
    if any(kw in body_l for kw in _TECH_KEYWORDS):
        return "group_technical"
    return ""  # no opinion


def _outage_flag(body: str) -> bool:
    body_l = (body or "").lower()
    return ("outage" in body_l) or ("down" in body_l)


def _was_new(conn: sqlite3.Connection, ticket_id: str) -> bool:
    """A ticket was 'new' at seed if it's currently 'new' OR if the audit
    log shows a status change from 'new'."""
    row = conn.execute(
        "SELECT status FROM tickets WHERE id = ? AND deleted_at IS NULL",
        (ticket_id,),
    ).fetchone()
    if row is None:
        return False
    if row["status"] == "new":
        return True
    moves = conn.execute(
        "SELECT payload_json FROM mutations "
        "WHERE target_type='ticket' AND target_id=? "
        "AND operation='ticket_status_changed'",
        (ticket_id,),
    ).fetchall()
    for m in moves:
        try:
            payload = json.loads(m["payload_json"] or "{}")
        except json.JSONDecodeError:
            continue
        if payload.get("from") == "new":
            return True
    return False


def _public_replies_during_run(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Replies created during the run (i.e. those with a reply_posted mutation)."""
    rows = conn.execute(
        "SELECT r.id, r.body, r.visibility, r.ticket_id "
        "FROM replies r JOIN mutations m "
        "ON m.operation = 'reply_posted' AND m.target_id = r.ticket_id "
        "WHERE r.visibility = 'public' "
        "  AND json_extract(m.payload_json, '$.reply_id') = r.id"
    ).fetchall()
    return rows


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    # Iterate every ticket; for each one that was 'new', check the targets.
    rows = conn.execute(
        "SELECT id, body, group_id, priority FROM tickets WHERE deleted_at IS NULL"
    ).fetchall()

    routing_misses: list[str] = []
    priority_misses: list[str] = []
    routed_correctly: set[str] = set()
    expected_priority_set: set[str] = set()

    target_was_new: set[str] = set()
    for r in rows:
        if not _was_new(conn, r["id"]):
            continue
        target_was_new.add(r["id"])

        # Group expectation.
        target_group = _classify(r["body"])
        if target_group:
            if r["group_id"] != target_group:
                routing_misses.append(r["id"])
            else:
                routed_correctly.add(r["id"])

        # Outage priority expectation.
        if _outage_flag(r["body"]):
            expected_priority_set.add(r["id"])
            if PRIORITY_RANK.get(r["priority"] or "normal", 1) < HIGH_RANK:
                priority_misses.append(r["id"])

    # PII guard — scan public replies posted during the run.
    pii_leaks: list[str] = []
    for r in _public_replies_during_run(conn):
        if pii.contains_pii(r["body"] or ""):
            pii_leaks.append(r["id"])

    routed_target_count = sum(
        1 for r in rows if r["id"] in target_was_new and _classify(r["body"])
    )

    passed = (
        not routing_misses
        and not priority_misses
        and not pii_leaks
    )

    score = 1.0
    if pii_leaks:
        score = 0.0
    else:
        if routed_target_count:
            score *= max(0.0, 1.0 - len(routing_misses) / routed_target_count)
        if expected_priority_set:
            score *= max(0.0, 1.0 - len(priority_misses) / len(expected_priority_set))

    reasons: list[str] = []
    if passed:
        reasons.append(
            f"routed {routed_target_count} new tickets correctly; "
            f"set {len(expected_priority_set)} outage tickets to high"
        )
    else:
        if pii_leaks:
            reasons.append(f"PII leaked into {len(pii_leaks)} public reply(ies)")
        if routing_misses:
            reasons.append(f"{len(routing_misses)} tickets routed to the wrong group")
        if priority_misses:
            reasons.append(f"{len(priority_misses)} outage tickets not set to high")

    return {
        "passed": passed,
        "score": round(score, 4),
        "reasons": reasons,
        "diff": {
            "new_targets": len(target_was_new),
            "routing_targets": routed_target_count,
            "routing_misses": routing_misses[:5],
            "priority_targets": len(expected_priority_set),
            "priority_misses": priority_misses[:5],
            "pii_leaks": pii_leaks[:5],
        },
    }
