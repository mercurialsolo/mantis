"""T02_merge_acme_dupes — merge @acme.com duplicates, keep most-active survivor.

The seed plants four contacts with email ``alice.lead@acme.com``:

* contact_00001 — 12 activities (winner — most active)
* contact_00002 — 4 activities
* contact_00003 — 2 activities
* contact_00004 — 1 activity

Pass conditions:
1. Exactly one non-deleted contact remains with email ``alice.lead@acme.com``.
2. The survivor is ``contact_00001`` (most-active).
3. All activities that were on the losers re-point to the survivor —
   the survivor now has at least 12 + 4 + 2 + 1 = 19 activities.
4. No activity is orphaned (every original target-id of the losers is
   now ``contact_00001``).
"""

from __future__ import annotations

import sqlite3
from typing import Any

ACME_EMAIL = "alice.lead@acme.com"
EXPECTED_SURVIVOR = "contact_00001"
EXPECTED_LOSERS = ["contact_00002", "contact_00003", "contact_00004"]
EXPECTED_TOTAL_ACTIVITIES = 12 + 4 + 2 + 1


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    reasons: list[str] = []

    survivors = conn.execute(
        "SELECT id FROM contacts WHERE email = ? AND deleted_at IS NULL ORDER BY id",
        (ACME_EMAIL,),
    ).fetchall()
    survivor_ids = [r["id"] for r in survivors]

    if len(survivor_ids) != 1:
        reasons.append(
            f"expected exactly 1 live @acme.com contact; found {len(survivor_ids)}: "
            f"{survivor_ids}"
        )

    survivor_correct = survivor_ids == [EXPECTED_SURVIVOR]
    if survivor_ids and not survivor_correct:
        reasons.append(
            f"survivor should be {EXPECTED_SURVIVOR} (most activities); "
            f"got {survivor_ids[0]}"
        )

    # Activities now on the survivor (anything pointing at it)
    survivor_activity_count = conn.execute(
        "SELECT COUNT(*) FROM activities "
        "WHERE target_type='contact' AND target_id=?",
        (EXPECTED_SURVIVOR,),
    ).fetchone()[0]

    enough_activities = survivor_activity_count >= EXPECTED_TOTAL_ACTIVITIES
    if not enough_activities:
        reasons.append(
            f"survivor has {survivor_activity_count} activities; "
            f"expected ≥ {EXPECTED_TOTAL_ACTIVITIES} (sum of pre-merge counts)"
        )

    # No orphans — none of the loser contact ids should still be the
    # ``target_id`` of any activity.
    placeholders = ",".join("?" for _ in EXPECTED_LOSERS)
    orphan_count = conn.execute(
        f"SELECT COUNT(*) FROM activities "
        f"WHERE target_type='contact' AND target_id IN ({placeholders})",
        EXPECTED_LOSERS,
    ).fetchone()[0]
    if orphan_count > 0:
        reasons.append(
            f"{orphan_count} activities still point at loser contacts "
            f"({EXPECTED_LOSERS})"
        )

    passed = (len(survivor_ids) == 1 and survivor_correct
              and enough_activities and orphan_count == 0)

    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reasons": reasons or ["merge correct: 1 survivor, all activities preserved"],
        "diff": {
            "survivor_ids": survivor_ids,
            "expected_survivor": EXPECTED_SURVIVOR,
            "survivor_activity_count": survivor_activity_count,
            "expected_activities_min": EXPECTED_TOTAL_ACTIVITIES,
            "orphaned_activities": orphan_count,
        },
    }
