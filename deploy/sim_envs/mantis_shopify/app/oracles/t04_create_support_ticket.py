"""t04 — Create a support ticket via Support → Contact Partner Support."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

MIN_DESC_LEN = 5


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    reasons: list[str] = []

    audit_rows = conn.execute(
        "SELECT id, target_id, payload_json FROM audit_log "
        "WHERE operation='support_ticket_created' ORDER BY id DESC"
    ).fetchall()
    if not audit_rows:
        return _fail(["no support_ticket_created audit row"], {})

    matched = None
    for r in audit_rows:
        payload = json.loads(r["payload_json"] or "{}")
        if not payload.get("subject", "").strip():
            continue
        if not payload.get("category", "").strip():
            continue
        if int(payload.get("description_len") or 0) < MIN_DESC_LEN:
            continue
        ticket = conn.execute(
            "SELECT * FROM tickets WHERE id=?", (r["target_id"],),
        ).fetchone()
        if ticket is None:
            continue
        if not (ticket["description"] or "").strip():
            continue
        matched = (dict(ticket), payload)
        break

    if matched is None:
        return _fail(
            ["no support_ticket_created audit row has subject + category + "
             f"description≥{MIN_DESC_LEN} chars resolving to a tickets row"],
            {"audit_rows_seen": len(audit_rows)},
        )

    ticket, payload = matched
    return {
        "passed": True, "score": 1.0,
        "reasons": [f"ticket {ticket['id']} created with category="
                    f"{ticket['category']!r}"],
        "diff": {
            "ticket_id": ticket["id"],
            "subject": ticket["subject"],
            "category": ticket["category"],
            "status": ticket["status"],
        },
    }


def _fail(reasons, diff):
    return {"passed": False, "score": 0.0, "reasons": reasons, "diff": diff}
