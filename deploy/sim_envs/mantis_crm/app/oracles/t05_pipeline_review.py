"""T05_pipeline_review — export every "user_05 big deal closing this quarter" to a list.

Target deals:

* owner_id == 'user_00005'
* amount > 50_000
* expected_close BETWEEN FAKE_NOW and end-of-quarter (FAKE_NOW is 2026-01-15,
  so Q1 2026 ends 2026-03-31)

Pass conditions:

1. The ``list_pipeline_review`` list exists (seeded by the env).
2. Every target deal id is a member of that list.
3. No non-target deal is in the list (precision == 1.0).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any

LIST_ID = "list_pipeline_review"
OWNER_ID = "user_00005"
AMOUNT_FLOOR = 50_000.0


def _parse_iso(value: str) -> datetime:
    if value and value.endswith("Z"):
        value = value[:-1]
    return datetime.fromisoformat(value)


def _end_of_quarter(now_dt: datetime) -> datetime:
    q = (now_dt.month - 1) // 3
    end_month = (q + 1) * 3
    if end_month == 12:
        return now_dt.replace(month=12, day=31, hour=23, minute=59, second=59)
    return now_dt.replace(month=end_month + 1, day=1, hour=0, minute=0, second=0)


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    now_dt = _parse_iso(now)
    eoq = _end_of_quarter(now_dt).isoformat()

    target_rows = conn.execute(
        "SELECT id FROM deals WHERE owner_id = ? AND amount > ? "
        "AND expected_close >= ? AND expected_close <= ?",
        (OWNER_ID, AMOUNT_FLOOR, now_dt.isoformat(), eoq),
    ).fetchall()
    target_ids = {r["id"] for r in target_rows}

    in_list = {
        r["member_id"] for r in conn.execute(
            "SELECT member_id FROM list_members "
            "WHERE list_id = ? AND member_type = 'deal'",
            (LIST_ID,),
        ).fetchall()
    }

    missing = target_ids - in_list
    spurious = in_list - target_ids
    passed = not missing and not spurious

    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reasons": (
            [f"list contains exactly the {len(target_ids)} target deals"]
            if passed
            else [
                f"missing {len(missing)} target deals",
                f"spurious {len(spurious)} non-target deals",
            ]
        ),
        "diff": {
            "target_count": len(target_ids),
            "in_list_count": len(in_list),
            "missing_examples": sorted(missing)[:5],
            "spurious_examples": sorted(spurious)[:5],
        },
    }
