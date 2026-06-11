"""t05 — Update the partner's business email in Settings → Business details."""

from __future__ import annotations

import json
import sqlite3
from typing import Any


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    rows = conn.execute(
        "SELECT id, target_id, payload_json FROM audit_log "
        "WHERE operation='settings_updated' ORDER BY id DESC"
    ).fetchall()
    if not rows:
        return _fail(["no settings_updated audit row"], {})

    matched = None
    for r in rows:
        payload = json.loads(r["payload_json"] or "{}")
        if payload.get("field") != "business_email":
            continue
        if "@" not in (payload.get("value") or ""):
            continue
        partner = conn.execute(
            "SELECT * FROM partners WHERE id=?", (r["target_id"],),
        ).fetchone()
        if partner is None:
            continue
        if partner["business_email"] != payload["value"]:
            continue
        matched = (dict(partner), payload)
        break

    if matched is None:
        return _fail(
            ["no settings_updated row for business_email matches the current "
             "partners.business_email"],
            {"audit_rows_seen": len(rows)},
        )

    partner, payload = matched
    return {
        "passed": True, "score": 1.0,
        "reasons": [
            f"partner {partner['id']} business_email now "
            f"{partner['business_email']!r}"
        ],
        "diff": {
            "partner_id": partner["id"],
            "business_email": partner["business_email"],
        },
    }


def _fail(reasons, diff):
    return {"passed": False, "score": 0.0, "reasons": reasons, "diff": diff}
