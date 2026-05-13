"""T03_merge_outage_dupes — merge 30 outage duplicates into one survivor.

Seed:
* ``ticket_07000`` is the planted survivor (priority high, group_technical).
* ``ticket_07001 .. ticket_07030`` are 30 duplicate reports of the same
  login outage from different requesters.

Pass conditions:

1. The 30 loser tickets are all soft-deleted.
2. The survivor's replies count rose by at least 1 (the status-page
   reply the agent posted).
3. No orphaned replies — every reply that pointed at a loser now
   points at the survivor.
4. A public reply on the survivor includes a status page URL
   (``https://status.`` substring) — the spec asks the agent to share
   the status-page link.
5. No PII leaks in any public reply created during the run.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from .. import pii
from ..seed import OUTAGE_LOSER_RANGE, OUTAGE_SURVIVOR_ID


def _loser_ids() -> list[str]:
    return [f"ticket_{i:05d}" for i in OUTAGE_LOSER_RANGE]


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    losers = _loser_ids()
    reasons: list[str] = []

    # 1. Soft-delete check.
    placeholders = ",".join("?" for _ in losers)
    live_losers = conn.execute(
        f"SELECT id FROM tickets WHERE id IN ({placeholders}) AND deleted_at IS NULL",
        losers,
    ).fetchall()
    soft_deleted_ok = not live_losers
    if not soft_deleted_ok:
        reasons.append(f"{len(live_losers)} duplicate(s) still live; expected all soft-deleted")

    # 2. Survivor still alive + carries the merged replies.
    survivor = conn.execute(
        "SELECT id, status FROM tickets WHERE id = ? AND deleted_at IS NULL",
        (OUTAGE_SURVIVOR_ID,),
    ).fetchone()
    survivor_alive = survivor is not None
    if not survivor_alive:
        reasons.append(f"survivor {OUTAGE_SURVIVOR_ID} not found / deleted")

    # 3. No orphan replies.
    orphan_count = conn.execute(
        f"SELECT COUNT(*) FROM replies WHERE ticket_id IN ({placeholders})",
        losers,
    ).fetchone()[0]
    if orphan_count > 0:
        reasons.append(f"{orphan_count} replies still point at loser tickets")

    # 4. Status-page reply on the survivor (created during the run).
    statuspage_reply = conn.execute(
        "SELECT r.id, r.body FROM replies r "
        "JOIN mutations m ON m.operation='reply_posted' AND m.target_id = r.ticket_id "
        "WHERE r.ticket_id = ? "
        "  AND r.visibility = 'public' "
        "  AND json_extract(m.payload_json, '$.reply_id') = r.id "
        "  AND LOWER(r.body) LIKE '%status.%'",
        (OUTAGE_SURVIVOR_ID,),
    ).fetchall()
    has_statuspage_reply = bool(statuspage_reply)
    if not has_statuspage_reply:
        reasons.append("no public reply with a status-page URL on the survivor")

    # 5. PII guard.
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
    if pii_leaks:
        reasons.append(f"PII leaked into {len(pii_leaks)} public reply(ies)")

    # 6. Merge audit log shows the absorbed set.
    merge_log = conn.execute(
        "SELECT payload_json FROM mutations WHERE operation='ticket_merge_completed' "
        "AND target_id = ?",
        (OUTAGE_SURVIVOR_ID,),
    ).fetchall()
    absorbed_ids: set[str] = set()
    for m in merge_log:
        try:
            payload = json.loads(m["payload_json"] or "{}")
        except json.JSONDecodeError:
            continue
        for tid in payload.get("absorbed") or []:
            absorbed_ids.add(tid)
    merge_covers_losers = set(losers).issubset(absorbed_ids)
    if not merge_covers_losers:
        missing = sorted(set(losers) - absorbed_ids)
        reasons.append(f"merge audit missing {len(missing)} of the 30 losers")

    passed = (
        soft_deleted_ok and survivor_alive and orphan_count == 0
        and has_statuspage_reply and not pii_leaks
        and merge_covers_losers
    )
    score = 1.0 if passed else 0.0

    return {
        "passed": passed,
        "score": score,
        "reasons": reasons or [
            f"all 30 outage duplicates merged into {OUTAGE_SURVIVOR_ID} with status-page reply"
        ],
        "diff": {
            "survivor_id": OUTAGE_SURVIVOR_ID,
            "live_loser_count": len(live_losers),
            "orphan_replies": orphan_count,
            "status_page_reply_count": len(statuspage_reply),
            "absorbed_count": len(absorbed_ids & set(losers)),
            "pii_leaks": pii_leaks[:5],
        },
    }
