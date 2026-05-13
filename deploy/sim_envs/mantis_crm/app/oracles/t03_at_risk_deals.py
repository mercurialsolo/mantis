"""T03_at_risk_deals — move stale Proposal-stage deals into ``At Risk``.

Target set, evaluated against the *current* DB state read together
with the audit log:

* Every deal whose seed-time stage was ``Proposal`` AND whose
  ``expected_close`` is in the past (relative to FAKE_NOW) by >30 days
  should now be ``At Risk``.

Because the oracle reads the snapshot after the run, we can't see the
pre-run stage directly — but we can recover it from the ``mutations``
log: any deal whose ``deal_stage_changed`` entry shows ``from='Proposal'``
counts toward the touched set, and the ``to`` value tells us whether
the agent put it in the right destination.

Strict pass: every qualifying deal ends in ``At Risk`` AND no
non-qualifying deal ends with a stage change recorded.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Any


def _parse_iso(value: str) -> datetime:
    if value and value.endswith("Z"):
        value = value[:-1]
    return datetime.fromisoformat(value)


def _qualifying_deal_ids(conn: sqlite3.Connection, *, now: str) -> set[str]:
    now_dt = _parse_iso(now)
    cutoff = (now_dt - timedelta(days=30)).isoformat()
    # We approximate the seed-time set as deals currently either Proposal
    # (untouched by the agent) or At Risk (moved by the agent) whose
    # expected_close is older than 30 days. This is the smallest
    # over-approximation that doesn't require pre-run snapshotting.
    rows = conn.execute(
        "SELECT id FROM deals "
        "WHERE stage IN ('Proposal', 'At Risk') "
        "  AND expected_close <= ?",
        (cutoff,),
    ).fetchall()
    return {r["id"] for r in rows}


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    qualifying = _qualifying_deal_ids(conn, now=now)

    # Movers: every deal_stage_changed mutation
    mutations = conn.execute(
        "SELECT target_id, payload_json FROM mutations "
        "WHERE operation = 'deal_stage_changed' "
        "ORDER BY id"
    ).fetchall()

    moved_to_at_risk: set[str] = set()
    moved_elsewhere: set[str] = set()
    for m in mutations:
        try:
            payload = json.loads(m["payload_json"] or "{}")
        except json.JSONDecodeError:
            payload = {}
        target = m["target_id"]
        to_stage = payload.get("to")
        from_stage = payload.get("from")
        if from_stage == "Proposal" and to_stage == "At Risk":
            moved_to_at_risk.add(target)
        elif from_stage == "Proposal":
            moved_elsewhere.add(target)

    # Every qualifying deal must end as At Risk (either it was already there or
    # got moved). We check current stage.
    current_at_risk = {
        r["id"] for r in conn.execute(
            "SELECT id FROM deals WHERE stage = 'At Risk'"
        ).fetchall()
    }

    missing = qualifying - current_at_risk
    spurious = current_at_risk - qualifying

    passed = not missing and not spurious and not moved_elsewhere

    reasons: list[str] = []
    if passed:
        reasons.append(f"{len(qualifying)} qualifying deals now At Risk; no collateral")
    else:
        if missing:
            reasons.append(f"{len(missing)} qualifying deals NOT moved to At Risk")
        if spurious:
            reasons.append(f"{len(spurious)} non-qualifying deals erroneously At Risk")
        if moved_elsewhere:
            reasons.append(
                f"{len(moved_elsewhere)} Proposal deals moved to a wrong stage"
            )

    precision = (len(qualifying & current_at_risk) / len(current_at_risk)) if current_at_risk else 0.0
    recall = (len(qualifying & current_at_risk) / len(qualifying)) if qualifying else 0.0
    f1 = 0.0 if not (precision + recall) else (2 * precision * recall) / (precision + recall)

    return {
        "passed": passed,
        "score": round(f1, 4),
        "reasons": reasons,
        "diff": {
            "qualifying_count": len(qualifying),
            "current_at_risk": len(current_at_risk),
            "missing_examples": sorted(missing)[:5],
            "spurious_examples": sorted(spurious)[:5],
            "moved_to_wrong_stage": sorted(moved_elsewhere)[:5],
        },
    }
