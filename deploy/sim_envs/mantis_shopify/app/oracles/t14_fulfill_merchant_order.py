"""t14 — Fulfill an unfulfilled merchant order from store Admin."""

from __future__ import annotations

import sqlite3
from typing import Any


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    rows = conn.execute(
        "SELECT id, target_id FROM audit_log WHERE operation='order_fulfilled' "
        "ORDER BY id DESC"
    ).fetchall()
    if not rows:
        return _fail(["no order_fulfilled audit row"], {})
    for r in rows:
        order = conn.execute(
            "SELECT * FROM merchant_orders WHERE id=?", (r["target_id"],),
        ).fetchone()
        if order is None:
            continue
        if order["fulfillment_status"] != "fulfilled":
            continue
        return {
            "passed": True, "score": 1.0,
            "reasons": [f"order {order['order_number']!r} fulfilled"],
            "diff": {
                "order_id": order["id"],
                "order_number": order["order_number"],
                "fulfillment_status": order["fulfillment_status"],
            },
        }
    return _fail(["order_fulfilled audit row(s) found but none reference a "
                  "valid merchant_orders row in 'fulfilled' state"],
                 {"audit_rows_seen": len(rows)})


def _fail(reasons, diff):
    return {"passed": False, "score": 0.0, "reasons": reasons, "diff": diff}
