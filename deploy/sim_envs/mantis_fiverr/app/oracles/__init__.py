"""Oracles — server-side graders for mantis-fiverr tasks.

Each oracle module exports a single ``grade(conn, *, now, seed_val) ->
{"passed": bool, "score": float, "reasons": list, "diff": dict}``
function that reads SQLite + audit_log and returns the harness oracle
response.

Same dispatcher shape as mantis_shop. New plans land their grader as
a new file + an entry in :data:`GRADERS`.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Callable

from . import (
    t01_order_basic_logo,
    t02_message_seller_then_order,
    t03_leave_5star_review,
)

GraderFn = Callable[..., dict[str, Any]]


GRADERS: dict[str, GraderFn] = {
    "t01_order_basic_logo": t01_order_basic_logo.grade,
    "t02_message_seller_then_order": t02_message_seller_then_order.grade,
    "t03_leave_5star_review": t03_leave_5star_review.grade,
}


def grade(
    task_id: str,
    conn: sqlite3.Connection,
    *,
    now: str,
    seed_val: int,
) -> dict[str, Any]:
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


__all__ = ["GRADERS", "GraderFn", "grade"]
