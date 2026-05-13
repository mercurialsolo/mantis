"""T04_add_meeting_note — log a meeting note on Sarah Chen for "yesterday".

Pass conditions:

1. A new activity exists on the (singular) Sarah Chen contact.
2. activity_type == 'meeting'.
3. body contains "discussed Q3 expansion" (case-insensitive substring).
4. occurred_at is the day before FAKE_NOW (date match only — we accept
   any time-of-day).

The oracle is intentionally lenient on phrasing; the spec specifies
the body verbatim but a few words of paraphrase from the agent are
acceptable so long as the substring is intact.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import Any

REQUIRED_PHRASE = "discussed q3 expansion"


def _parse_iso(value: str) -> datetime:
    if value and value.endswith("Z"):
        value = value[:-1]
    return datetime.fromisoformat(value)


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    sarah_ids = [
        r["id"] for r in conn.execute(
            "SELECT id FROM contacts WHERE name='Sarah Chen' AND deleted_at IS NULL"
        ).fetchall()
    ]
    if len(sarah_ids) != 1:
        return {
            "passed": False,
            "score": 0.0,
            "reasons": [f"expected exactly one Sarah Chen; got {len(sarah_ids)}"],
            "diff": {"sarah_ids": sarah_ids},
        }
    sarah_id = sarah_ids[0]

    yesterday = (_parse_iso(now) - timedelta(days=1)).date().isoformat()

    rows = conn.execute(
        "SELECT * FROM activities WHERE target_type='contact' AND target_id=? "
        "AND activity_type='meeting' AND occurred_at LIKE ?",
        (sarah_id, f"{yesterday}%"),
    ).fetchall()

    matching = [
        dict(r) for r in rows
        if REQUIRED_PHRASE in (r["body"] or "").lower()
    ]
    passed = len(matching) >= 1

    reasons = (
        [f"meeting note found on {sarah_id} for {yesterday}"]
        if passed
        else [f"no meeting activity on {sarah_id} for {yesterday} with body containing {REQUIRED_PHRASE!r}"]
    )
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reasons": reasons,
        "diff": {
            "sarah_contact_id": sarah_id,
            "expected_day": yesterday,
            "matching_activity_count": len(matching),
            "candidate_activities_on_day": len(rows),
        },
    }
