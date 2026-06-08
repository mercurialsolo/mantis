"""t02_message_seller_then_order — buyer messages a seller, then orders.

Sequence verification:
1. A ``message_sent`` audit row exists with sender_id == buyer_00001
   addressed to the seller of gig_00001.
2. An ``order_placed`` audit row exists for buyer_00001 on gig_00001.
3. The message audit row has a strictly earlier ``id`` than the order
   row (ordering is the whole point of the task).
4. The conversations row for that (buyer, seller) pair has last_msg_at
   updated.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

EXPECTED_BUYER = "buyer_00001"
EXPECTED_GIG = "gig_00001"


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    reasons: list[str] = []

    gig = conn.execute(
        "SELECT seller_id FROM gigs WHERE id = ?", (EXPECTED_GIG,),
    ).fetchone()
    if gig is None:
        return _fail([f"missing gig {EXPECTED_GIG}"], {})
    expected_seller = gig["seller_id"]

    rows = conn.execute(
        "SELECT id, operation, target_id, payload_json FROM audit_log "
        "WHERE operation IN ('message_sent', 'order_placed') "
        "ORDER BY id"
    ).fetchall()

    msg_id: int | None = None
    order_id: int | None = None
    conv_id: str | None = None
    order_target: str | None = None
    for r in rows:
        try:
            p = json.loads(r["payload_json"] or "{}")
        except json.JSONDecodeError:
            p = {}
        if r["operation"] == "message_sent" and msg_id is None:
            if p.get("sender_id") == EXPECTED_BUYER and p.get("recipient_id") == expected_seller:
                msg_id = int(r["id"])
                conv_id = r["target_id"]
        if r["operation"] == "order_placed" and order_id is None:
            if (p.get("buyer_id") == EXPECTED_BUYER
                    and p.get("gig_id") == EXPECTED_GIG
                    and p.get("seller_id") == expected_seller):
                order_id = int(r["id"])
                order_target = r["target_id"]

    if msg_id is None:
        reasons.append(
            f"no message_sent audit row from {EXPECTED_BUYER} to seller "
            f"{expected_seller}"
        )
    if order_id is None:
        reasons.append(
            f"no order_placed audit row for {EXPECTED_BUYER} on {EXPECTED_GIG}"
        )
    if msg_id is not None and order_id is not None and msg_id >= order_id:
        reasons.append(
            "message audit row must precede the order row "
            f"(msg id={msg_id} ≥ order id={order_id})"
        )

    # Conversation row updated?
    if conv_id is not None:
        c = conn.execute(
            "SELECT last_msg_at FROM conversations WHERE id = ?",
            (conv_id,),
        ).fetchone()
        if c is None:
            reasons.append(f"conversation {conv_id} missing")

    passed = not reasons
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reasons": reasons or [
            f"message {msg_id} preceded order {order_id} on {order_target}"
        ],
        "diff": {
            "message_audit_id": msg_id,
            "order_audit_id": order_id,
            "conversation_id": conv_id,
            "order_id": order_target,
        },
    }


def _fail(reasons: list[str], diff: dict[str, Any]) -> dict[str, Any]:
    return {"passed": False, "score": 0.0, "reasons": reasons, "diff": diff}
