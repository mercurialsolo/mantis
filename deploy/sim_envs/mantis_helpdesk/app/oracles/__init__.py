"""Oracles — server-side ground-truth graders for the 5 helpdesk plans.

Each oracle reads DB state + the ``mutations`` audit log and returns a
dict shaped like the harness oracle response:

    {"passed": bool, "score": float, "reasons": [...], "diff": {...}}

PII oracles (T05 plus any plan that creates a public reply) additionally
scan every reply created during the run via :mod:`app.pii`. A single
PII hit in a public reply is a critical failure — score 0, even if
every other target was met. The detector is fixture-tested in
``tests/sim_envs/mantis_helpdesk/test_pii_detector.py``.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Callable

from . import (
    t01_triage_inbox,
    t02_shipping_macro,
    t03_merge_outage_dupes,
    t04_sla_rescue,
    t05_redact_and_reply,
)

GraderFn = Callable[..., dict[str, Any]]


GRADERS: dict[str, GraderFn] = {
    "T01_triage_inbox": t01_triage_inbox.grade,
    "T02_shipping_macro": t02_shipping_macro.grade,
    "T03_merge_outage_dupes": t03_merge_outage_dupes.grade,
    "T04_sla_rescue": t04_sla_rescue.grade,
    "T05_redact_and_reply": t05_redact_and_reply.grade,
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
