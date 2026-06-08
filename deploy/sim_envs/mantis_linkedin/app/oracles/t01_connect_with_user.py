"""t01_connect_with_user — current user sends a connection request to user_00042 with a note.

Pass criteria:
- A ``connections`` row exists with from_user_id='user_00001',
  to_user_id='user_00042', status='pending'.
- ``note`` column is non-empty (the modal's textarea was populated).
- An audit_log row with operation='connection_requested' covers the same
  edge.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any


def grade(conn: sqlite3.Connection, *, now: str,
          seed_val: int) -> dict[str, Any]:
    reasons: list[str] = []
    target_from = "user_00001"
    target_to = "user_00042"

    row = conn.execute(
        "SELECT id, status, note FROM connections "
        "WHERE from_user_id = ? AND to_user_id = ?",
        (target_from, target_to),
    ).fetchone()

    if not row:
        return {
            "passed": False,
            "score": 0.0,
            "reasons": [
                f"no connection row from {target_from} → {target_to}",
            ],
            "diff": {"expected": {
                "from_user_id": target_from, "to_user_id": target_to,
                "status": "pending", "note": "<non-empty>",
            }, "actual": None},
        }

    if row["status"] != "pending":
        reasons.append(
            f"connection status is {row['status']!r}, expected 'pending'"
        )
    if not (row["note"] or "").strip():
        reasons.append("connection note is empty — modal textarea expected")

    audit_row = conn.execute(
        "SELECT id, payload_json FROM audit_log "
        "WHERE operation = 'connection_requested' "
        "AND target_id = ? ORDER BY id DESC LIMIT 1",
        (row["id"],),
    ).fetchone()
    if not audit_row:
        reasons.append("no audit_log row for connection_requested")

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
            "connection": {
                "id": row["id"], "status": row["status"], "note": row["note"],
            },
            "audit_payload": audit_payload,
        },
    }
