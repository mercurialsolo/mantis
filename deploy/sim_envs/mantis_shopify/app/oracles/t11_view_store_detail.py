"""t11 — Open a store detail page from the Stores list."""

from __future__ import annotations

import sqlite3
from typing import Any


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    rows = conn.execute(
        "SELECT id, target_id FROM audit_log WHERE operation='store_viewed' "
        "ORDER BY id DESC"
    ).fetchall()
    if not rows:
        return _fail(["no store_viewed audit row"], {})
    for r in rows:
        store = conn.execute(
            "SELECT * FROM stores WHERE id=?", (r["target_id"],),
        ).fetchone()
        if store is None:
            continue
        return {
            "passed": True, "score": 1.0,
            "reasons": [f"store {store['id']} detail page was opened"],
            "diff": {
                "store_id": store["id"],
                "name": store["name"],
                "kind": store["kind"],
            },
        }
    return _fail(["store_viewed audit row(s) found but none reference a "
                  "valid stores row"], {"audit_rows_seen": len(rows)})


def _fail(reasons, diff):
    return {"passed": False, "score": 0.0, "reasons": reasons, "diff": diff}
