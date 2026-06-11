"""t01 — Submit a Plus lead via Sales → Submit a Plus lead."""

from __future__ import annotations

import json
import sqlite3
from typing import Any


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    reasons: list[str] = []

    audit_rows = conn.execute(
        "SELECT target_id, payload_json FROM audit_log "
        "WHERE operation='lead_submitted' ORDER BY id DESC"
    ).fetchall()

    matched: dict[str, Any] | None = None
    for r in audit_rows:
        payload = json.loads(r["payload_json"] or "{}")
        if payload.get("product") != "plus":
            continue
        lead = conn.execute(
            "SELECT * FROM leads WHERE id=?", (r["target_id"],),
        ).fetchone()
        if lead is None:
            continue
        matched = dict(lead)
        break

    if matched is None:
        return _fail(
            ["no lead_submitted audit row with product='plus' found"],
            {"audit_rows": len(audit_rows)},
        )

    if not matched.get("merchant_name", "").strip():
        reasons.append("merchant_name is empty on the matched lead")
    if "@" not in (matched.get("contact_email") or ""):
        reasons.append("contact_email missing or invalid on the matched lead")

    passed = not reasons
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reasons": reasons or [
            f"lead {matched['id']} submitted as Plus for "
            f"{matched['merchant_name']!r}"
        ],
        "diff": {
            "lead_id": matched["id"],
            "merchant_name": matched["merchant_name"],
            "contact_email": matched["contact_email"],
        },
    }


def _fail(reasons, diff):
    return {"passed": False, "score": 0.0, "reasons": reasons, "diff": diff}
