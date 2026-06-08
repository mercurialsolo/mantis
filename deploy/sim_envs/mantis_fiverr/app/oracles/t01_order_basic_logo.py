"""t01_order_basic_logo — buyer_00001 orders gig_00001 at Basic tier.

Pass conditions:

1. An order_placed audit row exists for buyer_00001 + gig_00001 with
   tier='basic'.
2. The corresponding orders row has the right buyer/gig/tier and
   subtotal == gig.pkg_basic_price.
3. service_fee and total are consistent with the canonical formula
   (service_fee = subtotal * 0.055 + 2.0; total = subtotal + service_fee).
4. Exactly one order_items row for this order.
5. Status starts at 'active' (we don't require completion).
6. No collateral: no other order_placed rows fired for this buyer during
   the run.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

EXPECTED_BUYER = "buyer_00001"
EXPECTED_GIG = "gig_00001"
EXPECTED_TIER = "basic"


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    reasons: list[str] = []

    rows = conn.execute(
        "SELECT id, target_id, payload_json FROM audit_log "
        "WHERE operation = 'order_placed' ORDER BY id"
    ).fetchall()

    buyer_orders: list[tuple[str, dict[str, Any]]] = []
    for r in rows:
        try:
            payload = json.loads(r["payload_json"] or "{}")
        except json.JSONDecodeError:
            payload = {}
        if payload.get("buyer_id") == EXPECTED_BUYER:
            buyer_orders.append((r["target_id"], payload))

    if not buyer_orders:
        reasons.append(
            f"no order_placed audit row for {EXPECTED_BUYER}; agent never "
            "completed checkout"
        )
        return _fail(reasons, {"audit_count": len(rows)})

    # Look for the matching one (gig + tier). Extra orders for this
    # buyer count as collateral.
    target_id, target_payload = next(
        ((tid, p) for tid, p in buyer_orders
         if p.get("gig_id") == EXPECTED_GIG and p.get("tier") == EXPECTED_TIER),
        (None, None),
    )
    if target_id is None:
        reasons.append(
            f"buyer placed an order but not gig={EXPECTED_GIG} tier={EXPECTED_TIER}; "
            f"got: {[(p.get('gig_id'), p.get('tier')) for _, p in buyer_orders]}"
        )
        return _fail(reasons, {"buyer_orders": [tid for tid, _ in buyer_orders]})

    other_buyer_orders = [tid for tid, _ in buyer_orders if tid != target_id]
    if other_buyer_orders:
        reasons.append(
            f"collateral: buyer placed extra orders {other_buyer_orders}"
        )

    # Row check
    order_row = conn.execute(
        "SELECT * FROM orders WHERE id = ?", (target_id,),
    ).fetchone()
    if order_row is None:
        reasons.append(f"audit row points at missing order {target_id}")
        return _fail(reasons, {"target_id": target_id})
    order = dict(order_row)

    if order["buyer_id"] != EXPECTED_BUYER:
        reasons.append(
            f"orders.buyer_id={order['buyer_id']!r} ≠ {EXPECTED_BUYER!r}"
        )
    if order["gig_id"] != EXPECTED_GIG:
        reasons.append(f"orders.gig_id={order['gig_id']!r} ≠ {EXPECTED_GIG!r}")
    if order["tier"] != EXPECTED_TIER:
        reasons.append(f"orders.tier={order['tier']!r} ≠ {EXPECTED_TIER!r}")

    gig_row = conn.execute(
        "SELECT pkg_basic_price FROM gigs WHERE id = ?", (EXPECTED_GIG,),
    ).fetchone()
    expected_subtotal = float(gig_row["pkg_basic_price"]) if gig_row else None
    if expected_subtotal is None:
        reasons.append(f"missing gig {EXPECTED_GIG}")
        return _fail(reasons, {})
    if abs(order["subtotal"] - expected_subtotal) > 0.01:
        reasons.append(
            f"subtotal {order['subtotal']} ≠ Basic price {expected_subtotal}"
        )
    expected_fee = round(expected_subtotal * 0.055 + 2.0, 2)
    if abs(order["service_fee"] - expected_fee) > 0.05:
        reasons.append(
            f"service_fee {order['service_fee']} ≠ expected {expected_fee}"
        )
    expected_total = round(expected_subtotal + expected_fee, 2)
    if abs(order["total"] - expected_total) > 0.05:
        reasons.append(
            f"total {order['total']} ≠ expected {expected_total}"
        )

    items = conn.execute(
        "SELECT * FROM order_items WHERE order_id = ?", (target_id,),
    ).fetchall()
    if len(items) != 1:
        reasons.append(f"expected 1 order_items row, got {len(items)}")

    if order["status"] not in ("active", "delivered", "completed"):
        reasons.append(f"unexpected status {order['status']!r}")

    passed = not reasons
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reasons": reasons or [f"order {target_id} placed correctly"],
        "diff": {
            "order_id": target_id,
            "subtotal": order["subtotal"],
            "service_fee": order["service_fee"],
            "total": order["total"],
            "tier": order["tier"],
        },
    }


def _fail(reasons: list[str], diff: dict[str, Any]) -> dict[str, Any]:
    return {"passed": False, "score": 0.0, "reasons": reasons, "diff": diff}
