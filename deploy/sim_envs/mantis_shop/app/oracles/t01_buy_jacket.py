"""T01_buy_jacket — buy a size-M blue jacket under $100, ship to the seeded
Brooklyn address, apply coupon SPRING15.

Pass conditions:

1. An order exists (placed AT-OR-AFTER FAKE_NOW i.e. created during the
   run, not a pre-seeded historical order) for ``customer_00001``
   shipping to ``addr_brooklyn_001``.
2. The order has exactly one line item for variant
   ``variant_p00001_M_BLUE`` (size M, color BLUE), product_00001
   (Heritage Field Jacket, under $100), unit_price 89.00.
3. The order's coupon_codes list contains ``SPRING15``.
4. discount_total reflects 15% off the line subtotal
   (89 × 1 × 0.15 = 13.35).
5. status == 'paid'.
6. The variant's inventory is decremented by exactly the order quantity.
   No other variant's inventory was touched (audit_log "no collateral"
   guard via the ``inventory_adjusted`` operation count being 0 — this
   surface mutates inventory only via order_placed, not via direct
   adjustments).
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

EXPECTED_VARIANT_ID = "variant_00001_M_BLUE"
EXPECTED_VARIANT_SKU = "JACKET-001-M-BLUE"
EXPECTED_PRODUCT_ID = "product_00001"
EXPECTED_CUSTOMER_ID = "customer_00001"
EXPECTED_ADDRESS_ID = "addr_brooklyn_001"
EXPECTED_COUPON = "SPRING15"
EXPECTED_UNIT_PRICE = 89.00
EXPECTED_DISCOUNT_RATE = 0.15
# Per the seed, M-BLUE starts at exactly 25 units.
SEED_INVENTORY_M_BLUE = 25


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    reasons: list[str] = []

    # Find candidate orders created during the run. We look at the
    # audit_log for ``order_placed`` rows on this customer.
    placed_rows = conn.execute(
        "SELECT target_id, payload_json FROM audit_log "
        "WHERE operation = 'order_placed' "
        "ORDER BY id"
    ).fetchall()
    candidate_ids: list[str] = []
    for r in placed_rows:
        try:
            payload = json.loads(r["payload_json"] or "{}")
        except json.JSONDecodeError:
            payload = {}
        if payload.get("customer_id") == EXPECTED_CUSTOMER_ID:
            candidate_ids.append(r["target_id"])

    if not candidate_ids:
        reasons.append("no order_placed audit row found for the seeded customer")
        return _fail(reasons, {})

    # Of those, find the one with the right shape.
    best_order: dict[str, Any] | None = None
    for oid in candidate_ids:
        row = conn.execute(
            "SELECT * FROM orders WHERE id = ?", (oid,),
        ).fetchone()
        if row is None:
            continue
        order = dict(row)
        if order["shipping_address_id"] != EXPECTED_ADDRESS_ID:
            continue
        items = [
            dict(r) for r in conn.execute(
                "SELECT * FROM order_items WHERE order_id = ?", (oid,),
            ).fetchall()
        ]
        if len(items) != 1:
            continue
        line = items[0]
        if line["variant_id"] != EXPECTED_VARIANT_ID:
            continue
        if abs(line["unit_price"] - EXPECTED_UNIT_PRICE) > 0.01:
            continue
        try:
            codes = json.loads(order["coupon_codes"] or "[]")
        except json.JSONDecodeError:
            codes = []
        if EXPECTED_COUPON not in codes:
            continue
        best_order = order | {"items": items, "codes": codes}
        break

    if best_order is None:
        reasons.append(
            f"no matching order: need single line of {EXPECTED_VARIANT_SKU} "
            f"at ${EXPECTED_UNIT_PRICE:.2f} shipping to {EXPECTED_ADDRESS_ID} "
            f"with coupon {EXPECTED_COUPON}"
        )
        return _fail(reasons, {"candidate_order_ids": candidate_ids})

    # Coupon discount math
    qty = best_order["items"][0]["quantity"]
    expected_discount = round(
        EXPECTED_UNIT_PRICE * qty * EXPECTED_DISCOUNT_RATE, 2,
    )
    if abs(best_order["discount_total"] - expected_discount) > 0.05:
        reasons.append(
            f"discount_total {best_order['discount_total']} ≠ expected "
            f"{expected_discount} ({EXPECTED_DISCOUNT_RATE * 100:.0f}% off "
            f"${EXPECTED_UNIT_PRICE:.2f} × {qty})"
        )

    if best_order["status"] != "paid":
        reasons.append(f"order status {best_order['status']!r} ≠ 'paid'")

    # Inventory check on the target variant
    inv_row = conn.execute(
        "SELECT quantity FROM inventory WHERE variant_id = ?",
        (EXPECTED_VARIANT_ID,),
    ).fetchone()
    if inv_row is None:
        reasons.append("inventory row for the variant is missing")
        current_inv = None
    else:
        current_inv = inv_row["quantity"]
        expected_inv = SEED_INVENTORY_M_BLUE - qty
        if current_inv != expected_inv:
            reasons.append(
                f"variant inventory now {current_inv}; expected "
                f"{expected_inv} (was {SEED_INVENTORY_M_BLUE}, ordered {qty})"
            )

    # Collateral guard — no inventory_adjusted ops should exist on any
    # variant. The order_placed path handles inventory via direct
    # UPDATE; the audit_log records the adjustment only if an admin
    # touched it explicitly.
    extra_inv_adjusts = conn.execute(
        "SELECT COUNT(*) FROM audit_log "
        "WHERE operation = 'inventory_adjusted'"
    ).fetchone()[0]
    if extra_inv_adjusts > 0:
        reasons.append(
            f"unexpected inventory_adjusted audit rows: {extra_inv_adjusts} "
            "(T01 should not touch admin inventory)"
        )

    passed = not reasons
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reasons": reasons or [
            f"order {best_order['id']} placed with correct variant + coupon + total"
        ],
        "diff": {
            "order_id": best_order["id"],
            "order_total": best_order["total"],
            "discount_total": best_order["discount_total"],
            "variant_inventory_now": current_inv,
        },
    }


def _fail(reasons: list[str], diff: dict[str, Any]) -> dict[str, Any]:
    return {"passed": False, "score": 0.0, "reasons": reasons, "diff": diff}
