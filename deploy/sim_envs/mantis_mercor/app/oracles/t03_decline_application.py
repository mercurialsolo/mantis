"""t03 — Client declines the canonical pending application ``app_00001``.

Pass conditions:

1. The seeded ``app_00001`` row now has ``status='rejected'`` and a
   non-empty ``reject_reason``.
2. An ``audit_log`` row with operation=``application_declined`` and
   target_id=``app_00001`` exists, and its payload carries the reason
   + previous_status='submitted'.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

TARGET_APP = "app_00001"


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    reasons: list[str] = []

    app_row = conn.execute(
        "SELECT * FROM applications WHERE id=?", (TARGET_APP,),
    ).fetchone()
    if app_row is None:
        return _fail(
            [f"application {TARGET_APP} missing — seed did not initialise"],
            {},
        )
    app = dict(app_row)

    if app["status"] != "rejected":
        reasons.append(f"status {app['status']!r} ≠ 'rejected'")
    if not (app.get("reject_reason") or "").strip():
        reasons.append("reject_reason is empty")

    audit = conn.execute(
        "SELECT payload_json FROM audit_log "
        "WHERE operation='application_declined' AND target_id=?",
        (TARGET_APP,),
    ).fetchone()
    if audit is None:
        reasons.append("no application_declined audit row")
    else:
        payload = json.loads(audit["payload_json"] or "{}")
        if not (payload.get("reason") or "").strip():
            reasons.append("audit payload.reason is empty")
        if payload.get("previous_status") not in {"submitted", "under_review"}:
            reasons.append(
                f"audit payload.previous_status "
                f"{payload.get('previous_status')!r} not in submitted/under_review"
            )

    passed = not reasons
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reasons": reasons or [
            f"application {TARGET_APP} declined with reason "
            f"{app.get('reject_reason')!r}"
        ],
        "diff": {
            "application_id": TARGET_APP,
            "status": app["status"],
            "reject_reason": app.get("reject_reason"),
        },
    }


def _fail(reasons: list[str], diff: dict[str, Any]) -> dict[str, Any]:
    return {"passed": False, "score": 0.0, "reasons": reasons, "diff": diff}
