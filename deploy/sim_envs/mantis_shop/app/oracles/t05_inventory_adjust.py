"""T05_inventory_adjust — bump inventory of ``sku=TEE-BLK-M`` by 50
with reason "restock from warehouse".

Pass conditions:

1. The variant ``TEE-BLK-M`` exists, its inventory is now exactly
   seed + 50 (= 150 — seed pins 100).
2. The audit log contains an ``inventory_adjusted`` row with payload
   ``sku='TEE-BLK-M'``, ``delta=50`` (the net change recorded), and a
   reason containing the substring "restock" (case-insensitive).
3. No other variant's inventory was touched via the admin path
   (precision: no extra ``inventory_adjusted`` rows on other variants).
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

TARGET_SKU = "TEE-BLK-M"
EXPECTED_DELTA = 50
SEED_QUANTITY = 100  # per seed.py
EXPECTED_REASON_SUBSTR = "restock"


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    reasons: list[str] = []

    variant = conn.execute(
        "SELECT id FROM variants WHERE sku = ?", (TARGET_SKU,),
    ).fetchone()
    if variant is None:
        return _fail([f"variant with sku {TARGET_SKU!r} not found"], {})
    variant_id = variant["id"]

    inv_row = conn.execute(
        "SELECT quantity FROM inventory WHERE variant_id = ?",
        (variant_id,),
    ).fetchone()
    current = inv_row["quantity"] if inv_row else 0
    expected = SEED_QUANTITY + EXPECTED_DELTA
    if current != expected:
        reasons.append(
            f"inventory for {TARGET_SKU} is {current}; expected {expected} "
            f"(seed {SEED_QUANTITY} + {EXPECTED_DELTA})"
        )

    # Audit log check
    audit_rows = conn.execute(
        "SELECT * FROM audit_log WHERE operation = 'inventory_adjusted' "
        "ORDER BY id"
    ).fetchall()
    matching: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []
    for r in audit_rows:
        try:
            payload = json.loads(r["payload_json"] or "{}")
        except json.JSONDecodeError:
            payload = {}
        if payload.get("sku") == TARGET_SKU:
            matching.append(payload)
        else:
            other.append({"target_id": r["target_id"], "payload": payload})

    if not matching:
        reasons.append("no inventory_adjusted audit row recorded for "
                       f"{TARGET_SKU}")
    else:
        net_delta = sum(int(p.get("delta") or 0) for p in matching)
        if net_delta != EXPECTED_DELTA:
            reasons.append(
                f"net audit delta {net_delta} ≠ expected {EXPECTED_DELTA}"
            )
        # At least one matching row must have the expected reason substring.
        reason_ok = any(
            EXPECTED_REASON_SUBSTR in (p.get("reason") or "").lower()
            for p in matching
        )
        if not reason_ok:
            reasons.append(
                f"no audit row's reason contains {EXPECTED_REASON_SUBSTR!r}; "
                f"got reasons: {[p.get('reason') for p in matching]}"
            )

    if other:
        reasons.append(
            f"{len(other)} unrelated inventory_adjusted rows on other "
            f"variants (precision violation): {other[:3]}"
        )

    passed = not reasons
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reasons": reasons or [
            f"inventory for {TARGET_SKU} bumped by {EXPECTED_DELTA} with "
            f"reason recorded"
        ],
        "diff": {
            "variant_id": variant_id,
            "current_quantity": current,
            "expected_quantity": expected,
            "matching_audit_count": len(matching),
        },
    }


def _fail(reasons: list[str], diff: dict[str, Any]) -> dict[str, Any]:
    return {"passed": False, "score": 0.0, "reasons": reasons, "diff": diff}
