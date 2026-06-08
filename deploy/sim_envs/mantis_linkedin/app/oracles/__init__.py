"""Oracles — server-side graders for mantis-linkedin tasks.

Each oracle reads the DB + audit log only (never the agent transcript).
Return shape mirrors mantis-shop / mantis-boattrader:

    {"passed": bool, "score": float, "task_id": str,
     "reasons": [str, ...], "diff": {...}}
"""

from __future__ import annotations

import sqlite3
from typing import Any, Callable

from . import (
    t01_connect_with_user,
    t02_post_text_update,
    t03_easy_apply_to_job,
)

GraderFn = Callable[..., dict[str, Any]]

GRADERS: dict[str, GraderFn] = {
    "t01_connect_with_user": t01_connect_with_user.grade,
    "t02_post_text_update": t02_post_text_update.grade,
    "t03_easy_apply_to_job": t03_easy_apply_to_job.grade,
    # Convenience aliases — match the README's tNN shorthand.
    "t01": t01_connect_with_user.grade,
    "t02": t02_post_text_update.grade,
    "t03": t03_easy_apply_to_job.grade,
}


def grade(task_id: str, conn: sqlite3.Connection, *, now: str,
          seed_val: int) -> dict[str, Any]:
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
