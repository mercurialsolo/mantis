"""T02_shipping_macro — apply the shipping-delay macro to every order-12xxxx ticket.

Target set: every ticket (not deleted) whose body contains a six-digit
order number starting with ``12`` (regex ``\\b12\\d{4}\\b``).

Pass conditions:

1. Every target ticket has at least one ``reply_posted`` mutation whose
   ``macro_id`` is ``macro_shipping_delay``.
2. No PII leaks into any public reply during the run.
3. No public reply was posted on a ticket whose thread visibility is
   ``internal`` (the route rejects this, but the oracle re-verifies).

Score = recall (fraction of target tickets that got the macro), reduced
to 0 if PII leaks or internal-only violations are present.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

from .. import pii

ORDER_RE = re.compile(r"\b12\d{4}\b")
SHIPPING_MACRO_ID = "macro_shipping_delay"


def _target_ticket_ids(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT id, body FROM tickets WHERE deleted_at IS NULL"
    ).fetchall()
    return {r["id"] for r in rows if ORDER_RE.search(r["body"] or "")}


def _macro_applied_tickets(conn: sqlite3.Connection, macro_id: str) -> set[str]:
    out: set[str] = set()
    rows = conn.execute(
        "SELECT target_id, payload_json FROM mutations "
        "WHERE target_type='ticket' AND operation='reply_posted'"
    ).fetchall()
    for r in rows:
        try:
            payload = json.loads(r["payload_json"] or "{}")
        except json.JSONDecodeError:
            continue
        if payload.get("macro_id") == macro_id:
            out.add(r["target_id"])
    return out


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    targets = _target_ticket_ids(conn)
    applied = _macro_applied_tickets(conn, SHIPPING_MACRO_ID)

    missing = targets - applied
    spurious = applied - targets

    # PII guard.
    pii_leaks: list[str] = []
    pub_replies = conn.execute(
        "SELECT r.id, r.body FROM replies r "
        "JOIN mutations m ON m.operation='reply_posted' AND m.target_id = r.ticket_id "
        "WHERE r.visibility='public' "
        "  AND json_extract(m.payload_json, '$.reply_id') = r.id"
    ).fetchall()
    for r in pub_replies:
        if pii.contains_pii(r["body"] or ""):
            pii_leaks.append(r["id"])

    # Internal-only violations: any reply_posted with visibility=public on
    # a ticket marked internal-only.
    internal_violations = conn.execute(
        "SELECT m.target_id FROM mutations m "
        "JOIN tickets t ON t.id = m.target_id "
        "WHERE m.operation='reply_posted' "
        "  AND t.visibility='internal' "
        "  AND json_extract(m.payload_json, '$.visibility') = 'public'"
    ).fetchall()

    base_passed = not missing
    passed = (
        base_passed
        and not pii_leaks
        and not internal_violations
    )
    recall = (len(applied & targets) / len(targets)) if targets else 1.0
    score = recall if passed else (0.0 if pii_leaks or internal_violations else round(recall, 4))

    reasons: list[str] = []
    if passed:
        reasons.append(
            f"shipping macro applied to all {len(targets)} order-12xxxx tickets"
        )
    else:
        if pii_leaks:
            reasons.append(f"PII leaked into {len(pii_leaks)} public reply(ies)")
        if internal_violations:
            reasons.append(
                f"{len(internal_violations)} public reply(ies) on internal-only threads"
            )
        if missing:
            reasons.append(
                f"shipping macro not applied to {len(missing)} target ticket(s)"
            )

    return {
        "passed": passed,
        "score": round(score, 4),
        "reasons": reasons,
        "diff": {
            "target_count": len(targets),
            "applied_count": len(applied),
            "missing_examples": sorted(missing)[:5],
            "spurious_examples": sorted(spurious)[:5],
            "pii_leaks": pii_leaks[:5],
            "internal_violations": [r["target_id"] for r in internal_violations][:5],
        },
    }
