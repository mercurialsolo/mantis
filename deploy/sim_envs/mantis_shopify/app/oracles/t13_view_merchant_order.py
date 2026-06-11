"""t13 — Open a merchant order detail page from a store's Admin."""

from __future__ import annotations

import sqlite3
from typing import Any


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    rows = conn.execute(
        "SELECT id, target_id FROM audit_log WHERE operation='order_viewed' "
        "ORDER BY id DESC"
    ).fetchall()
    if not rows:
        return _fail(["no order_viewed audit row"], {})
    for r in rows:
        order = conn.execute(
            "SELECT * FROM merchant_orders WHERE id=?", (r["target_id"],),
        ).fetchone()
        if order is None:
            continue
        return {
            "passed": True, "score": 1.0,
            "reasons": [f"order {order['order_number']!r} detail page opened"],
            "diff": {
                "order_id": order["id"],
                "order_number": order["order_number"],
                "store_id": order["store_id"],
                "total_cents": order["total_cents"],
            },
        }
    return _fail(["order_viewed audit row(s) found but none reference a "
                  "valid merchant_orders row"], {"audit_rows_seen": len(rows)})


def _fail(reasons, diff):
    return {"passed": False, "score": 0.0, "reasons": reasons, "diff": diff}
