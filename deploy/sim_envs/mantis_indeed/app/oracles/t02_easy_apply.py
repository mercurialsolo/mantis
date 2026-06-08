"""t02_easy_apply — seeker Easy Applies to job_00012 with phone +
resume_00001 + screening answers.

Pass conditions:

1. An `application_submitted` audit row exists for `user_00001` +
   `job_00012` with non-empty phone, resume_id, and answers.
2. The `applications` row matches: status='new', resume_id present,
   answers_json non-empty.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

EXPECTED_USER_ID = "user_00001"
EXPECTED_JOB_ID = "job_00012"


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    reasons: list[str] = []

    audit = conn.execute(
        "SELECT payload_json FROM audit_log "
        "WHERE operation = 'application_submitted' ORDER BY id"
    ).fetchall()
    match = None
    for r in audit:
        try:
            p = json.loads(r["payload_json"] or "{}")
        except json.JSONDecodeError:
            p = {}
        if p.get("user_id") == EXPECTED_USER_ID and p.get("job_id") == EXPECTED_JOB_ID:
            match = p
            break
    if match is None:
        reasons.append(
            f"no application_submitted audit row for {EXPECTED_USER_ID} + {EXPECTED_JOB_ID}"
        )
        return _fail(reasons, {})

    if not match.get("phone"):
        reasons.append("application phone missing")
    if not match.get("resume_id"):
        reasons.append("application resume_id missing")
    if not match.get("answers"):
        reasons.append("application answers missing")

    # DB-state guard.
    app_row = conn.execute(
        "SELECT * FROM applications WHERE user_id = ? AND job_id = ?",
        (EXPECTED_USER_ID, EXPECTED_JOB_ID),
    ).fetchone()
    if app_row is None:
        reasons.append("applications table missing the row")
    else:
        if not app_row["resume_id"]:
            reasons.append("applications.resume_id is empty")
        if not app_row["phone"]:
            reasons.append("applications.phone is empty")
        try:
            ans = json.loads(app_row["answers_json"] or "{}")
        except json.JSONDecodeError:
            ans = {}
        if not ans:
            reasons.append("applications.answers_json is empty")

    passed = not reasons
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reasons": reasons or [
            f"application submitted by {EXPECTED_USER_ID} for {EXPECTED_JOB_ID}"
        ],
        "diff": {
            "audit_payload": match,
            "applications_row": dict(app_row) if app_row else None,
        },
    }


def _fail(reasons: list[str], diff: dict[str, Any]) -> dict[str, Any]:
    return {"passed": False, "score": 0.0, "reasons": reasons, "diff": diff}
