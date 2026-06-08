"""Oracles — server-side graders for mantis-mercor.

Each oracle reads DB state + audit_log. Never the agent transcript.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Callable

from . import t01_apply_to_ml_engineer, t02_shortlist_candidate, t03_decline_application

GraderFn = Callable[..., dict[str, Any]]

GRADERS: dict[str, GraderFn] = {
    "t01": t01_apply_to_ml_engineer.grade,
    "t01_apply_to_ml_engineer": t01_apply_to_ml_engineer.grade,
    "t02": t02_shortlist_candidate.grade,
    "t02_shortlist_candidate": t02_shortlist_candidate.grade,
    "t03": t03_decline_application.grade,
    "t03_decline_application": t03_decline_application.grade,
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
