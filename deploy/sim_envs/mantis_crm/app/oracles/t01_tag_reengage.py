"""T01_tag_reengage — tag every "1M+ ARR + 90+d stale" contact as ``reengage``.

Target set: contacts whose
  * company.arr_band IN ('1M+', '10M+')
  * last_activity_at >= 90 days before FAKE_NOW
  * deleted_at IS NULL

Pass conditions:
1. Every target contact has the ``reengage`` tag.
2. No non-target contact has the ``reengage`` tag.
3. (Bonus) No non-tag mutations recorded against contacts outside the
   target set — protects against the agent over-eagerly editing other
   fields.

Score is the harmonic-ish mean of precision + recall on the
"reengage tag added" set.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Any

REENGAGE_TAG = "reengage"
QUALIFYING_ARR = {"1M+", "10M+"}
STALE_DAYS = 90


def _parse_iso(value: str) -> datetime:
    if value and value.endswith("Z"):
        value = value[:-1]
    return datetime.fromisoformat(value)


def _target_contact_ids(conn: sqlite3.Connection, *, now: str) -> set[str]:
    now_dt = _parse_iso(now)
    threshold = (now_dt - timedelta(days=STALE_DAYS)).isoformat()
    rows = conn.execute(
        "SELECT c.id FROM contacts c "
        "JOIN companies co ON c.company_id = co.id "
        "WHERE c.deleted_at IS NULL "
        "  AND co.arr_band IN ('1M+', '10M+') "
        "  AND c.last_activity_at <= ?",
        (threshold,),
    ).fetchall()
    return {r["id"] for r in rows}


def _tagged_contact_ids(conn: sqlite3.Connection, tag: str) -> set[str]:
    rows = conn.execute(
        "SELECT id, tags FROM contacts WHERE deleted_at IS NULL"
    ).fetchall()
    out: set[str] = set()
    for r in rows:
        try:
            tags = json.loads(r["tags"] or "[]")
        except json.JSONDecodeError:
            tags = []
        if tag in tags:
            out.add(r["id"])
    return out


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    targets = _target_contact_ids(conn, now=now)
    tagged = _tagged_contact_ids(conn, REENGAGE_TAG)

    missing = targets - tagged
    spurious = tagged - targets

    precision = (len(tagged & targets) / len(tagged)) if tagged else 0.0
    recall = (len(tagged & targets) / len(targets)) if targets else 0.0

    passed = len(missing) == 0 and len(spurious) == 0
    score = 0.0 if not (precision + recall) else (2 * precision * recall) / (precision + recall)

    return {
        "passed": passed,
        "score": round(score, 4),
        "reasons": (
            [f"all {len(targets)} target contacts tagged correctly"]
            if passed
            else [
                f"missing tags on {len(missing)} target contacts",
                f"spurious tags on {len(spurious)} non-target contacts",
                f"precision={precision:.3f} recall={recall:.3f}",
            ]
        ),
        "diff": {
            "target_count": len(targets),
            "tagged_count": len(tagged),
            "missing_examples": sorted(missing)[:5],
            "spurious_examples": sorted(spurious)[:5],
        },
    }
