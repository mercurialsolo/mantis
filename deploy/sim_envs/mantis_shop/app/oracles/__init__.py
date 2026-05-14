"""Oracles — server-side graders for the 5 mantis-shop plans.

Each oracle reads DB state + audit log. Never the agent's transcript.
Mirror mantis-crm's pattern: same return shape, same dispatch table.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Callable

from . import (
    t01_buy_jacket,
    t02_refund_line_item,
    t03_create_coupon,
    t04_export_bogo_orders,
    t05_inventory_adjust,
    t06_login_then_buy_jacket,
)

GraderFn = Callable[..., dict[str, Any]]


GRADERS: dict[str, GraderFn] = {
    "T01_buy_jacket": t01_buy_jacket.grade,
    "T02_refund_line_item": t02_refund_line_item.grade,
    "T03_create_coupon": t03_create_coupon.grade,
    "T04_export_bogo_orders": t04_export_bogo_orders.grade,
    "T05_inventory_adjust": t05_inventory_adjust.grade,
    "T06_login_then_buy_jacket": t06_login_then_buy_jacket.grade,
}


def grade(task_id: str, conn: sqlite3.Connection, *,
          now: str, seed_val: int) -> dict[str, Any]:
    fn = GRADERS.get(task_id)
    if fn is None:
        return {
            "passed": False,
            "score": 0.0,
            "task_id": task_id,
            "reasons": [f"no oracle registered for task_id={task_id!r}"],
            "diff": {},
        }
    result = fn(conn, now=now, seed_val=seed_val)
    result.setdefault("task_id", task_id)
    return result
