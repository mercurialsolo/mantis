"""Oracles — server-side ground-truth graders for the 5 mantis-crm plans.

Each oracle module exports a single ``grade(conn, *, now, seed_val)``
function that reads the DB plus the ``mutations`` audit log and
returns a dict shaped like the harness oracle response:

    {"passed": bool, "score": float, "reasons": [...], "diff": {...}}

The dispatch table at the bottom maps ``task_id`` → grader. New plans
land their grader as a new file + an entry in :data:`GRADERS`.

Design notes
------------

* Oracles are deterministic — they read DB state, not the agent's
  transcript. Same end state → same verdict.
* Each oracle computes both the "right set" (rows that should have
  been touched) and the "wrong set" (rows that shouldn't have).
  Touching a row outside the target set is a fail, even if every
  target was also touched.
* The 5 reruns / 5 identical scores acceptance criterion falls out of
  the determinism guarantee automatically: same agent → same DB end
  state → same oracle reads → same score.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Callable

from . import (
    t01_tag_reengage,
    t02_merge_acme_dupes,
    t03_at_risk_deals,
    t04_add_meeting_note,
    t05_pipeline_review,
)

GraderFn = Callable[..., dict[str, Any]]


GRADERS: dict[str, GraderFn] = {
    "T01_tag_reengage": t01_tag_reengage.grade,
    "T02_merge_acme_dupes": t02_merge_acme_dupes.grade,
    "T03_at_risk_deals": t03_at_risk_deals.grade,
    "T04_add_meeting_note": t04_add_meeting_note.grade,
    "T05_pipeline_review": t05_pipeline_review.grade,
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
