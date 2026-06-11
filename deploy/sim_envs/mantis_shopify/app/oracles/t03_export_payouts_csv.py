"""t03 — Export the payouts list as CSV (Payouts → Export CSV)."""

from __future__ import annotations

import sqlite3
from typing import Any


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    row = conn.execute(
        "SELECT id FROM audit_log WHERE operation='payouts_export_requested' "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return {
            "passed": False, "score": 0.0,
            "reasons": ["no payouts_export_requested audit row"],
            "diff": {},
        }
    return {
        "passed": True, "score": 1.0,
        "reasons": ["payouts export was triggered"],
        "diff": {"audit_id": row["id"]},
    }
