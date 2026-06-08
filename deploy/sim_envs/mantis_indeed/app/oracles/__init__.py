"""mantis-indeed oracles — server-side graders.

Each oracle reads DB state + audit_log. Never the agent's transcript.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Callable

from . import t01_search_save_remote, t02_easy_apply, t03_employer_review_applicant

GraderFn = Callable[..., dict[str, Any]]


GRADERS: dict[str, GraderFn] = {
    "t01_search_save_remote": t01_search_save_remote.grade,
    "t02_easy_apply": t02_easy_apply.grade,
    "t03_employer_review_applicant": t03_employer_review_applicant.grade,
    # Convenience aliases (UPPER variants like mantis-shop)
    "T01_search_save_remote": t01_search_save_remote.grade,
    "T02_easy_apply": t02_easy_apply.grade,
    "T03_employer_review_applicant": t03_employer_review_applicant.grade,
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
