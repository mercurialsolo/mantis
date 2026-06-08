"""t03_easy_apply_to_job — user Easy Applies on job_00003 with phone + resume confirm.

Pass criteria:
- Row in job_applications with user_00001 + job_00003, status='submitted',
  non-empty phone.
- audit_log row operation='job_application_submitted' for the same id.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any


def grade(conn: sqlite3.Connection, *, now: str,
          seed_val: int) -> dict[str, Any]:
    reasons: list[str] = []
    user_id = "user_00001"
    job_id = "job_00003"

    row = conn.execute(
        "SELECT id, status, phone, resume_label, answers_json "
        "FROM job_applications WHERE user_id = ? AND job_id = ?",
        (user_id, job_id),
    ).fetchone()

    if not row:
        return {
            "passed": False,
            "score": 0.0,
            "reasons": [
                f"no job_application row for {user_id} × {job_id}"
            ],
            "diff": {"expected": {
                "user_id": user_id, "job_id": job_id,
                "status": "submitted", "phone": "<non-empty>",
            }, "actual": None},
        }

    if row["status"] != "submitted":
        reasons.append(
            f"application status is {row['status']!r}, expected 'submitted'"
        )
    if not (row["phone"] or "").strip():
        reasons.append("application phone is empty")

    audit_row = conn.execute(
        "SELECT id, payload_json FROM audit_log "
        "WHERE operation = 'job_application_submitted' "
        "AND target_id = ? ORDER BY id DESC LIMIT 1",
        (row["id"],),
    ).fetchone()
    if not audit_row:
        reasons.append("no audit_log row for job_application_submitted")

    audit_payload: dict[str, Any] = {}
    if audit_row:
        try:
            audit_payload = json.loads(audit_row["payload_json"])
        except json.JSONDecodeError:
            audit_payload = {}

    passed = not reasons
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reasons": reasons or ["all checks passed"],
        "diff": {
            "application": {
                "id": row["id"], "status": row["status"],
                "phone": row["phone"], "resume_label": row["resume_label"],
            },
            "audit_payload": audit_payload,
        },
    }
