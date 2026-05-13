"""T04_export_bogo_orders — populate ``saved_view_bogo_recent`` with
exactly the orders that used coupon BOGO in the last 7 days
(relative to FAKE_NOW = 2026-01-15T09:00:00Z).

Pass conditions:

1. ``saved_view_members`` for ``saved_view_bogo_recent`` equals (no
   missing, no extras) the set of orders whose:
     - coupon_codes contains 'BOGO'
     - placed_at >= (FAKE_NOW - 7 days)
2. The seeded target set is the 5 pinned BOGO recent orders
   (``order_04801`` … ``order_04805``).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import timedelta
from typing import Any

VIEW_ID = "saved_view_bogo_recent"


def _parse_iso(value: str):
    from datetime import datetime
    if value and value.endswith("Z"):
        value = value[:-1]
    return datetime.fromisoformat(value)


def _target_order_ids(conn: sqlite3.Connection, *, now: str) -> set[str]:
    now_dt = _parse_iso(now)
    cutoff = (now_dt - timedelta(days=7)).isoformat()
    rows = conn.execute(
        "SELECT id, coupon_codes FROM orders WHERE placed_at >= ?",
        (cutoff,),
    ).fetchall()
    out: set[str] = set()
    for r in rows:
        try:
            codes = json.loads(r["coupon_codes"] or "[]")
        except json.JSONDecodeError:
            codes = []
        if "BOGO" in codes:
            out.add(r["id"])
    return out


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    targets = _target_order_ids(conn, now=now)
    members = {
        r["order_id"] for r in conn.execute(
            "SELECT order_id FROM saved_view_members WHERE view_id = ?",
            (VIEW_ID,),
        ).fetchall()
    }
    missing = targets - members
    spurious = members - targets

    passed = not missing and not spurious

    reasons: list[str] = []
    if passed:
        reasons.append(
            f"saved view {VIEW_ID} contains exactly the {len(targets)} target orders"
        )
    else:
        if missing:
            reasons.append(f"missing {len(missing)} target orders")
        if spurious:
            reasons.append(f"spurious {len(spurious)} non-target orders in the view")

    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reasons": reasons,
        "diff": {
            "target_count": len(targets),
            "member_count": len(members),
            "missing_examples": sorted(missing)[:5],
            "spurious_examples": sorted(spurious)[:5],
        },
    }
