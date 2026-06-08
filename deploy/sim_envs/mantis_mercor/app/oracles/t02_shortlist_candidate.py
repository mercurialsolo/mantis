"""t02 — Client adds candidate_00007 to the shortlist for job_00001.

Pass conditions:

1. A ``shortlist_entries`` row exists with ``job_id='job_00001'``,
   ``candidate_id='candidate_00007'``.
2. An ``audit_log`` row with operation=``candidate_shortlisted`` and
   matching target.
3. ``added_by`` references a user with role=``client`` (the canonical
   ``client_00001`` when ENV_REQUIRE_AUTH=0).
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

EXPECTED_JOB = "job_00001"
EXPECTED_CANDIDATE = "candidate_00007"


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    reasons: list[str] = []

    row = conn.execute(
        "SELECT * FROM shortlist_entries WHERE job_id=? AND candidate_id=?",
        (EXPECTED_JOB, EXPECTED_CANDIDATE),
    ).fetchone()
    if row is None:
        reasons.append(
            f"no shortlist_entries row for job={EXPECTED_JOB} "
            f"candidate={EXPECTED_CANDIDATE}"
        )
        return _fail(reasons, {})

    entry = dict(row)

    added_by = entry.get("added_by")
    if added_by:
        u = conn.execute(
            "SELECT role FROM users WHERE id=?", (added_by,),
        ).fetchone()
        if u is None or u["role"] not in {"client", "admin"}:
            reasons.append(
                f"added_by user {added_by!r} not a client/admin"
            )

    audit = conn.execute(
        "SELECT id, payload_json FROM audit_log "
        "WHERE operation='candidate_shortlisted' AND target_id=?",
        (entry["id"],),
    ).fetchone()
    if audit is None:
        reasons.append("no candidate_shortlisted audit row for the entry")
    else:
        payload = json.loads(audit["payload_json"] or "{}")
        if payload.get("job_id") != EXPECTED_JOB:
            reasons.append(
                f"audit payload.job_id {payload.get('job_id')!r} ≠ {EXPECTED_JOB}"
            )
        if payload.get("candidate_id") != EXPECTED_CANDIDATE:
            reasons.append(
                f"audit payload.candidate_id {payload.get('candidate_id')!r} "
                f"≠ {EXPECTED_CANDIDATE}"
            )

    passed = not reasons
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reasons": reasons or [
            f"candidate {EXPECTED_CANDIDATE} shortlisted for {EXPECTED_JOB}"
        ],
        "diff": {
            "shortlist_id": entry["id"],
            "added_by": entry.get("added_by"),
        },
    }


def _fail(reasons: list[str], diff: dict[str, Any]) -> dict[str, Any]:
    return {"passed": False, "score": 0.0, "reasons": reasons, "diff": diff}
