"""t10 — Submit a Shopify POS Pro referral."""

from __future__ import annotations

import json
import sqlite3
from typing import Any


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    rows = conn.execute(
        "SELECT id, target_id, payload_json FROM audit_log "
        "WHERE operation='lead_submitted' ORDER BY id DESC"
    ).fetchall()

    for r in rows:
        payload = json.loads(r["payload_json"] or "{}")
        if payload.get("product") != "pos":
            continue
        lead = conn.execute(
            "SELECT * FROM leads WHERE id=?", (r["target_id"],),
        ).fetchone()
        if lead is None:
            continue
        if not (lead["merchant_name"] or "").strip():
            continue
        if "@" not in (lead["contact_email"] or ""):
            continue
        return {
            "passed": True, "score": 1.0,
            "reasons": [f"POS Pro referral submitted for "
                        f"{lead['merchant_name']!r}"],
            "diff": {
                "lead_id": lead["id"],
                "merchant_name": lead["merchant_name"],
                "contact_email": lead["contact_email"],
            },
        }
    return {
        "passed": False, "score": 0.0,
        "reasons": ["no lead_submitted audit row with product='pos' "
                    "references a valid leads row with merchant + email"],
        "diff": {"audit_rows": len(rows)},
    }
