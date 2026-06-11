"""t02 — Invite a new staff member via Team → Invite staff."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

ACCEPTABLE_ROLES = {
    "owner", "staff_business", "staff_dev", "staff_marketing",
    "staff_support", "staff_finance",
}


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    reasons: list[str] = []

    audit_rows = conn.execute(
        "SELECT id, target_id, payload_json FROM audit_log "
        "WHERE operation='staff_invited' ORDER BY id DESC"
    ).fetchall()

    if not audit_rows:
        return _fail(["no staff_invited audit row"], {})

    matched = None
    for r in audit_rows:
        payload = json.loads(r["payload_json"] or "{}")
        email = (payload.get("email") or "").strip()
        role = (payload.get("role") or "").strip()
        if "@" not in email:
            continue
        if role not in ACCEPTABLE_ROLES:
            continue
        user = conn.execute(
            "SELECT * FROM users WHERE id=?", (r["target_id"],),
        ).fetchone()
        if user is None:
            continue
        if user["status"] != "invited":
            continue
        matched = (dict(user), payload)
        break

    if matched is None:
        return _fail(
            ["no staff_invited audit row produced an invited users row "
             "with a valid role and email"],
            {"audit_rows_seen": len(audit_rows)},
        )

    user_row, payload = matched
    passed = True
    return {
        "passed": passed,
        "score": 1.0,
        "reasons": [
            f"user {user_row['id']} invited with role={payload['role']!r}"
        ],
        "diff": {
            "user_id": user_row["id"],
            "email": user_row["email"],
            "role": user_row["role"],
            "status": user_row["status"],
        },
    }


def _fail(reasons, diff):
    return {"passed": False, "score": 0.0, "reasons": reasons, "diff": diff}
