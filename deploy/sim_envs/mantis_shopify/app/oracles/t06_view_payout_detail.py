"""t06 — Open a payout detail page from the Payouts history list."""

from __future__ import annotations

import sqlite3
from typing import Any


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    rows = conn.execute(
        "SELECT id, target_id FROM audit_log WHERE operation='payout_viewed' "
        "ORDER BY id DESC"
    ).fetchall()
    if not rows:
        return _fail(["no payout_viewed audit row"], {})

    matched = None
    for r in rows:
        payout = conn.execute(
            "SELECT * FROM payouts WHERE id=?", (r["target_id"],),
        ).fetchone()
        if payout is None:
            continue
        matched = dict(payout)
        break

    if matched is None:
        return _fail(
            ["payout_viewed audit row(s) found but none reference a valid "
             "payouts row"],
            {"audit_rows_seen": len(rows)},
        )

    return {
        "passed": True, "score": 1.0,
        "reasons": [f"payout {matched['id']} detail page was opened"],
        "diff": {
            "payout_id": matched["id"],
            "amount_cents": matched["amount_cents"],
            "status": matched["status"],
        },
    }


def _fail(reasons, diff):
    return {"passed": False, "score": 0.0, "reasons": reasons, "diff": diff}
