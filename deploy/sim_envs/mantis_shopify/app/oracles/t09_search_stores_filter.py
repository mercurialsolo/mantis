"""t09 — Filter or search the Stores list."""

from __future__ import annotations

import json
import sqlite3
from typing import Any


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    rows = conn.execute(
        "SELECT id, payload_json FROM audit_log "
        "WHERE operation='stores_filter_applied' ORDER BY id DESC LIMIT 5"
    ).fetchall()
    if not rows:
        return {
            "passed": False, "score": 0.0,
            "reasons": ["no stores_filter_applied audit row"],
            "diff": {},
        }
    for r in rows:
        payload = json.loads(r["payload_json"] or "{}")
        q = (payload.get("q") or "").strip()
        st = (payload.get("status") or "").strip()
        if q or st:
            return {
                "passed": True, "score": 1.0,
                "reasons": [
                    f"stores filter applied with q={q!r} status={st!r} "
                    f"matched={payload.get('matched')}"
                ],
                "diff": {"q": q, "status": st,
                         "matched": payload.get("matched")},
            }
    return {
        "passed": False, "score": 0.0,
        "reasons": ["stores_filter_applied rows exist but every payload "
                    "had empty q and status"],
        "diff": {"rows": len(rows)},
    }
