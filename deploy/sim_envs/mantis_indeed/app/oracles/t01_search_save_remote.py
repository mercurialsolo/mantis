"""t01_search_save_remote — seeker searches "software engineer" in "Austin,
TX" with remote=1, then saves job_00007.

Pass conditions:

1. A `search_submitted` audit row exists with payload `q` containing
   "software engineer", `l` containing "austin", and `remote` non-empty.
2. A `job_saved` audit row exists for `user_00001` (the default
   acting seeker) targeting `job_00007`.
3. The saved_jobs table reflects that row.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

EXPECTED_USER_ID = "user_00001"
EXPECTED_JOB_ID = "job_00007"


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    reasons: list[str] = []

    # 1. search_submitted with the right filters.
    rows = conn.execute(
        "SELECT payload_json FROM audit_log "
        "WHERE operation = 'search_submitted' ORDER BY id"
    ).fetchall()
    matching_search = None
    for r in rows:
        try:
            p = json.loads(r["payload_json"] or "{}")
        except json.JSONDecodeError:
            p = {}
        q = (p.get("q") or "").lower()
        l = (p.get("l") or "").lower()  # noqa: E741 -- Indeed query-param convention
        remote = (p.get("remote") or "").lower()
        if "software engineer" in q and "austin" in l and remote in {"1", "true", "on"}:
            matching_search = p
            break
    if matching_search is None:
        reasons.append(
            "no search_submitted audit row with q~'software engineer', "
            "l~'austin', remote=1"
        )

    # 2. job_saved audit row for the target seeker + job.
    saved_rows = conn.execute(
        "SELECT payload_json FROM audit_log "
        "WHERE operation = 'job_saved' ORDER BY id"
    ).fetchall()
    saved_match = None
    for r in saved_rows:
        try:
            p = json.loads(r["payload_json"] or "{}")
        except json.JSONDecodeError:
            p = {}
        if p.get("user_id") == EXPECTED_USER_ID and p.get("job_id") == EXPECTED_JOB_ID:
            saved_match = p
            break
    if saved_match is None:
        reasons.append(
            f"no job_saved audit row for user={EXPECTED_USER_ID}, "
            f"job={EXPECTED_JOB_ID}"
        )

    # 3. saved_jobs row exists.
    row = conn.execute(
        "SELECT 1 FROM saved_jobs WHERE user_id = ? AND job_id = ?",
        (EXPECTED_USER_ID, EXPECTED_JOB_ID),
    ).fetchone()
    if row is None:
        reasons.append(
            f"saved_jobs has no row for {EXPECTED_USER_ID} + {EXPECTED_JOB_ID}"
        )

    passed = not reasons
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reasons": reasons or [
            f"search performed + {EXPECTED_JOB_ID} saved by {EXPECTED_USER_ID}"
        ],
        "diff": {
            "matching_search": matching_search,
            "saved_match": saved_match,
        },
    }
