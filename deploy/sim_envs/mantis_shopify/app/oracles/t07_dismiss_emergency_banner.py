"""t07 — Dismiss the emergency-contact banner."""

from __future__ import annotations

import sqlite3
from typing import Any


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    row = conn.execute(
        "SELECT id FROM audit_log WHERE operation='banner_dismissed' "
        "AND target_id='emergency_contact' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return {
            "passed": False, "score": 0.0,
            "reasons": ["no banner_dismissed audit row for emergency_contact"],
            "diff": {},
        }
    return {
        "passed": True, "score": 1.0,
        "reasons": ["emergency-contact banner was dismissed"],
        "diff": {"audit_id": row["id"]},
    }
