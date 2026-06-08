"""t01 — Candidate applies to ``job_00001`` (Internal Medicine Expert).

The task is named "apply_to_ml_engineer" historically per the task
suite naming convention; in this env the canonical ``job_00001`` is
the Internal Medicine Expert role (the first role visible on home).
The grader works against ``job_00001`` regardless of title.

Pass conditions:

1. An ``application_submitted`` audit row exists with
   ``target_id`` referencing a row in ``applications``.
2. The application's ``job_id == 'job_00001'``.
3. The application's ``candidate_id == 'candidate_00001'`` (or any
   candidate when ENV_REQUIRE_AUTH=0 and effective_user is canonical).
4. ``status == 'submitted'`` after the run.
5. ``screening_answers`` is a JSON array with the same length as
   the job's ``screening_qs`` and EVERY answer is non-empty.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

EXPECTED_JOB_ID = "job_00001"
EXPECTED_CANDIDATE = "candidate_00001"


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    reasons: list[str] = []

    audit_rows = conn.execute(
        "SELECT target_id, payload_json FROM audit_log "
        "WHERE operation = 'application_submitted' ORDER BY id"
    ).fetchall()

    job_row = conn.execute(
        "SELECT screening_qs FROM jobs WHERE id=?", (EXPECTED_JOB_ID,),
    ).fetchone()
    if job_row is None:
        return _fail([f"job {EXPECTED_JOB_ID} missing in seed"], {})
    expected_n_qs = len(json.loads(job_row["screening_qs"] or "[]"))

    matched_app: dict[str, Any] | None = None
    for r in audit_rows:
        payload = json.loads(r["payload_json"] or "{}")
        if payload.get("job_id") != EXPECTED_JOB_ID:
            continue
        if payload.get("candidate_id") != EXPECTED_CANDIDATE:
            continue
        app_row = conn.execute(
            "SELECT * FROM applications WHERE id=?", (r["target_id"],),
        ).fetchone()
        if app_row is None:
            continue
        matched_app = dict(app_row)
        break

    if matched_app is None:
        reasons.append(
            f"no application_submitted audit row found for "
            f"candidate={EXPECTED_CANDIDATE} job={EXPECTED_JOB_ID}"
        )
        return _fail(reasons, {"audit_rows_seen": len(audit_rows)})

    if matched_app["status"] != "submitted":
        reasons.append(f"status {matched_app['status']!r} ≠ 'submitted'")

    answers = json.loads(matched_app.get("screening_answers") or "[]")
    if not isinstance(answers, list):
        reasons.append("screening_answers is not a list")
    elif len(answers) != expected_n_qs:
        reasons.append(
            f"screening_answers has {len(answers)} entries; job has "
            f"{expected_n_qs} screening questions"
        )
    else:
        for i, a in enumerate(answers):
            if not (isinstance(a, str) and a.strip()):
                reasons.append(f"screening_answers[{i}] is empty")

    passed = not reasons
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reasons": reasons or [
            f"application {matched_app['id']} submitted with complete "
            f"screening answers"
        ],
        "diff": {
            "application_id": matched_app["id"],
            "status": matched_app["status"],
            "n_answers": len(answers) if isinstance(answers, list) else 0,
        },
    }


def _fail(reasons: list[str], diff: dict[str, Any]) -> dict[str, Any]:
    return {"passed": False, "score": 0.0, "reasons": reasons, "diff": diff}
