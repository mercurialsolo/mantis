"""t08 — Request a directory review for a listing in the Partner Directory."""

from __future__ import annotations

import json
import sqlite3
from typing import Any


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    rows = conn.execute(
        "SELECT id, target_id, payload_json FROM audit_log "
        "WHERE operation='directory_review_requested' ORDER BY id DESC"
    ).fetchall()
    if not rows:
        return _fail(["no directory_review_requested audit row"], {})

    matched = None
    for r in rows:
        payload = json.loads(r["payload_json"] or "{}")
        listing = conn.execute(
            "SELECT * FROM directory_listings WHERE id=?", (r["target_id"],),
        ).fetchone()
        if listing is None:
            continue
        if listing["review_status"] not in {"requested", "received"}:
            continue
        if not payload.get("business_name", "").strip():
            continue
        matched = (dict(listing), payload)
        break

    if matched is None:
        return _fail(
            ["directory_review_requested rows present but none reference a "
             "listing with review_status in {requested, received}"],
            {"audit_rows_seen": len(rows)},
        )

    listing, payload = matched
    return {
        "passed": True, "score": 1.0,
        "reasons": [f"review requested for {listing['business_name']!r}"],
        "diff": {
            "listing_id": listing["id"],
            "business_name": listing["business_name"],
            "review_status": listing["review_status"],
        },
    }


def _fail(reasons, diff):
    return {"passed": False, "score": 0.0, "reasons": reasons, "diff": diff}
