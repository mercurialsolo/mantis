"""t12 — Open a catalog detail page from the Catalogs list."""

from __future__ import annotations

import sqlite3
from typing import Any


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    rows = conn.execute(
        "SELECT id, target_id FROM audit_log WHERE operation='catalog_viewed' "
        "ORDER BY id DESC"
    ).fetchall()
    if not rows:
        return _fail(["no catalog_viewed audit row"], {})
    for r in rows:
        catalog = conn.execute(
            "SELECT * FROM catalogs WHERE id=?", (r["target_id"],),
        ).fetchone()
        if catalog is None:
            continue
        return {
            "passed": True, "score": 1.0,
            "reasons": [f"catalog {catalog['id']} detail page was opened"],
            "diff": {
                "catalog_id": catalog["id"],
                "name": catalog["name"],
                "status": catalog["status"],
            },
        }
    return _fail(["catalog_viewed audit row(s) found but none reference a "
                  "valid catalogs row"], {"audit_rows_seen": len(rows)})


def _fail(reasons, diff):
    return {"passed": False, "score": 0.0, "reasons": reasons, "diff": diff}
