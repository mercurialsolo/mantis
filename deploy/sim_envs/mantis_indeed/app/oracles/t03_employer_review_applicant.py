"""t03_employer_review_applicant — employer (`user_emp_00003`) moves an
applicant on `job_00003` from "new" to "reviewed".

Pass conditions:

1. At least one `application_status_changed` audit row exists with
   `previous_status='new'` and `new_status='reviewed'` targeting an
   application whose `job_id = job_00003`.
2. The applications row reflects status='reviewed' + reviewed_at not
   NULL.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

EXPECTED_JOB_ID = "job_00003"


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    reasons: list[str] = []

    audit = conn.execute(
        "SELECT target_id, payload_json FROM audit_log "
        "WHERE operation = 'application_status_changed' ORDER BY id"
    ).fetchall()
    match = None
    for r in audit:
        try:
            p = json.loads(r["payload_json"] or "{}")
        except json.JSONDecodeError:
            p = {}
        if p.get("previous_status") == "new" and p.get("new_status") == "reviewed":
            # Confirm the target application is on job_00003.
            app_row = conn.execute(
                "SELECT job_id FROM applications WHERE id = ?",
                (r["target_id"],),
            ).fetchone()
            if app_row and app_row["job_id"] == EXPECTED_JOB_ID:
                match = {
                    "target_id": r["target_id"],
                    "payload": p,
                }
                break
    if match is None:
        reasons.append(
            "no application_status_changed audit row for an application on "
            f"{EXPECTED_JOB_ID} moving 'new' → 'reviewed'"
        )
        return _fail(reasons, {})

    app_row = conn.execute(
        "SELECT * FROM applications WHERE id = ?",
        (match["target_id"],),
    ).fetchone()
    if app_row is None:
        reasons.append("application row missing post-update")
    else:
        if app_row["status"] != "reviewed":
            reasons.append(
                f"applications.status is {app_row['status']!r}, expected 'reviewed'"
            )
        if not app_row["reviewed_at"]:
            reasons.append("applications.reviewed_at is empty")

    passed = not reasons
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reasons": reasons or [
            f"application {match['target_id']} moved new→reviewed on {EXPECTED_JOB_ID}"
        ],
        "diff": {
            "match": match,
            "applications_row": dict(app_row) if app_row else None,
        },
    }


def _fail(reasons: list[str], diff: dict[str, Any]) -> dict[str, Any]:
    return {"passed": False, "score": 0.0, "reasons": reasons, "diff": diff}
