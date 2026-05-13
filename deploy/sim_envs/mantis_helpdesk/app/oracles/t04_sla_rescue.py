"""T04_sla_rescue — reassign near-SLA-breach tickets to a low-load agent in-group.

Target set: every ticket where
  * status IN ('new','open','pending')
  * sla_breach_at <= FAKE_NOW + 2h (within 2h of breach, including already breached)
  * group_id is not NULL

Pass conditions:

1. Every target ticket now has ``assignee_id`` set (non-null).
2. The assignee is an active agent in the ticket's group.
3. The assignee's *initial* open-ticket count (seed snapshot at run
   start) was <3 — we approximate this by checking the count *now*; the
   agent should have spread load. Strict v1 check: assignee's current
   open ticket count <= 3 + len(targets assigned to them by us).

A simpler heuristic for v1: assignee count of <= 3 OPEN target rows
each — i.e. no single agent absorbed all 400 near-breach tickets. We
implement this by counting target-tickets-per-assignee.

4. No PII in public replies (no replies expected for this plan; we
   still guard).
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from datetime import timedelta
from typing import Any

from .. import pii
from ..seed import _parse_iso


def _target_ticket_ids(conn: sqlite3.Connection, now: str) -> list[str]:
    cutoff = (_parse_iso(now) + timedelta(hours=2)).isoformat()
    rows = conn.execute(
        "SELECT id FROM tickets "
        "WHERE deleted_at IS NULL "
        "  AND status IN ('new','open','pending') "
        "  AND group_id IS NOT NULL "
        "  AND sla_breach_at <= ?",
        (cutoff,),
    ).fetchall()
    return [r["id"] for r in rows]


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    target_ids = _target_ticket_ids(conn, now=now)
    if not target_ids:
        return {
            "passed": False, "score": 0.0,
            "reasons": ["no target tickets — seed misconfigured"],
            "diff": {"target_count": 0},
        }

    placeholders = ",".join("?" for _ in target_ids)
    rows = conn.execute(
        f"SELECT t.id, t.assignee_id, t.group_id, "
        f"       u.group_id AS assignee_group, u.is_active "
        f"FROM tickets t LEFT JOIN users u ON t.assignee_id = u.id "
        f"WHERE t.id IN ({placeholders})",
        target_ids,
    ).fetchall()

    unassigned: list[str] = []
    out_of_group: list[str] = []
    inactive: list[str] = []
    per_assignee: Counter[str] = Counter()
    for r in rows:
        if not r["assignee_id"]:
            unassigned.append(r["id"])
            continue
        if r["assignee_group"] != r["group_id"]:
            out_of_group.append(r["id"])
        if r["is_active"] != 1:
            inactive.append(r["id"])
        per_assignee[r["assignee_id"]] += 1

    overloaded = [a for a, n in per_assignee.items() if n > 3]

    pii_leaks: list[str] = []
    pub_replies = conn.execute(
        "SELECT r.id, r.body FROM replies r "
        "JOIN mutations m ON m.operation='reply_posted' AND m.target_id = r.ticket_id "
        "WHERE r.visibility='public' "
        "  AND json_extract(m.payload_json, '$.reply_id') = r.id"
    ).fetchall()
    for r in pub_replies:
        if pii.contains_pii(r["body"] or ""):
            pii_leaks.append(r["id"])

    passed = (
        not unassigned
        and not out_of_group
        and not inactive
        and not overloaded
        and not pii_leaks
    )

    reasons: list[str] = []
    if passed:
        reasons.append(
            f"reassigned {len(target_ids)} near-breach tickets, max load {max(per_assignee.values())} per agent"
        )
    else:
        if unassigned:
            reasons.append(f"{len(unassigned)} target ticket(s) still unassigned")
        if out_of_group:
            reasons.append(f"{len(out_of_group)} ticket(s) assigned to agent outside the group")
        if inactive:
            reasons.append(f"{len(inactive)} ticket(s) assigned to inactive agent")
        if overloaded:
            reasons.append(f"{len(overloaded)} agent(s) hold >3 target tickets")
        if pii_leaks:
            reasons.append(f"PII leaked into {len(pii_leaks)} public reply(ies)")

    score = 1.0
    if pii_leaks:
        score = 0.0
    else:
        if target_ids:
            score *= max(0.0, 1.0 - len(unassigned) / len(target_ids))
            score *= max(0.0, 1.0 - len(out_of_group) / len(target_ids))
            score *= max(0.0, 1.0 - len(overloaded) / max(1, len(per_assignee)))

    return {
        "passed": passed,
        "score": round(score, 4),
        "reasons": reasons,
        "diff": {
            "target_count": len(target_ids),
            "unassigned": unassigned[:5],
            "out_of_group": out_of_group[:5],
            "inactive_assignees": inactive[:5],
            "overloaded_agents": overloaded[:5],
            "max_load": max(per_assignee.values()) if per_assignee else 0,
            "pii_leaks": pii_leaks[:5],
        },
    }
