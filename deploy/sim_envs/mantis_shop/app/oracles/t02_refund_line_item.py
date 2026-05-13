"""T02_refund_line_item — refund line item 2 of order #4421 with reason
"damaged"; notify-customer true.

Pass conditions:

1. An ``order_refunds`` row exists for ``order_04421`` with
   ``line_no = 2`` and ``reason`` containing 'damaged' (case-insensitive).
2. ``notify_customer = 1`` on the refund row.
3. The amount equals line 2's unit_price × quantity (full line refund).
4. The order's customer-visible status reflects the refund: the
   ``orders.notify_customer`` flag is set to 1 (the field the admin
   surface and the customer-facing receipt both read).
"""

from __future__ import annotations

import sqlite3
from typing import Any

EXPECTED_ORDER_ID = "order_04421"
EXPECTED_LINE_NO = 2
REQUIRED_REASON = "damaged"


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    reasons: list[str] = []

    order_row = conn.execute(
        "SELECT * FROM orders WHERE id = ?", (EXPECTED_ORDER_ID,),
    ).fetchone()
    if order_row is None:
        return _fail([f"order {EXPECTED_ORDER_ID} not found"], {})

    line_row = conn.execute(
        "SELECT * FROM order_items WHERE order_id = ? AND line_no = ?",
        (EXPECTED_ORDER_ID, EXPECTED_LINE_NO),
    ).fetchone()
    if line_row is None:
        return _fail(
            [f"line {EXPECTED_LINE_NO} not found on {EXPECTED_ORDER_ID}"], {},
        )
    expected_amount = round(line_row["unit_price"] * line_row["quantity"], 2)

    refunds = conn.execute(
        "SELECT * FROM order_refunds WHERE order_id = ? AND line_no = ? "
        "ORDER BY id",
        (EXPECTED_ORDER_ID, EXPECTED_LINE_NO),
    ).fetchall()
    if not refunds:
        return _fail(
            [f"no refund recorded for line {EXPECTED_LINE_NO} of "
             f"{EXPECTED_ORDER_ID}"],
            {"expected_amount": expected_amount},
        )

    # Pick the matching refund (any with the right reason + amount).
    match: dict[str, Any] | None = None
    for r in refunds:
        rd = dict(r)
        if REQUIRED_REASON not in (rd["reason"] or "").lower():
            continue
        if abs(rd["amount"] - expected_amount) > 0.05:
            continue
        if not rd["notify_customer"]:
            continue
        match = rd
        break

    if match is None:
        for r in refunds:
            rd = dict(r)
            sub_reasons: list[str] = []
            if REQUIRED_REASON not in (rd["reason"] or "").lower():
                sub_reasons.append(
                    f"reason {rd['reason']!r} doesn't contain {REQUIRED_REASON!r}"
                )
            if abs(rd["amount"] - expected_amount) > 0.05:
                sub_reasons.append(
                    f"amount {rd['amount']} ≠ expected {expected_amount}"
                )
            if not rd["notify_customer"]:
                sub_reasons.append("notify_customer flag not set")
            reasons.append("refund id=" + str(rd["id"]) + ": " + "; ".join(sub_reasons))
        return _fail(reasons, {"expected_amount": expected_amount,
                              "refund_count": len(refunds)})

    if not order_row["notify_customer"]:
        reasons.append(
            "orders.notify_customer flag not propagated; customer-visible "
            "status wouldn't reflect the refund"
        )

    passed = not reasons
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reasons": reasons or [
            f"refund_id={match['id']} on line {EXPECTED_LINE_NO} of "
            f"{EXPECTED_ORDER_ID} with reason 'damaged' + notify_customer"
        ],
        "diff": {
            "refund_id": match["id"],
            "refund_amount": match["amount"],
            "order_notify_customer": bool(order_row["notify_customer"]),
        },
    }


def _fail(reasons: list[str], diff: dict[str, Any]) -> dict[str, Any]:
    return {"passed": False, "score": 0.0, "reasons": reasons, "diff": diff}
