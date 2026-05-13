"""T03_create_coupon — create a 20%-off coupon for category
``outerwear-women`` expiring 2026-02-15, max 100 uses.

The agent picks the code. The oracle accepts ANY new code (one that
doesn't appear in the seed) so long as the terms match exactly.

Pass conditions:

1. Exactly one coupon row exists that:
   * is NOT one of the seeded codes
   * kind == 'pct', value == 20
   * scope_category == 'outerwear-women'
   * expires_at starts with '2026-02-15'
   * max_uses == 100
   * disabled_at IS NULL
2. The corresponding ``coupon_created`` audit log row exists.

There must be exactly one matching coupon (we don't accept the agent
creating multiple variants — that's collateral).
"""

from __future__ import annotations

import sqlite3
from typing import Any

# Coupons present in the seed. Any newly-created coupon must NOT be in
# this set.
SEEDED_CODES = {
    "SPRING15", "BOGO", "STACK_TRAP_A", "STACK_TRAP_B", "VIPONLY",
    "WINTER10", "WINTER20", "TEE5", "DENIM15", "BLACKFRIDAY",
    "OLDSALE", "FREESHIP", "SUMMER25", "REFER20", "LOYAL10",
}

EXPECTED_KIND = "pct"
EXPECTED_VALUE = 20.0
EXPECTED_CATEGORY = "outerwear-women"
EXPECTED_EXPIRY_PREFIX = "2026-02-15"
EXPECTED_MAX_USES = 100


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    rows = conn.execute(
        "SELECT * FROM coupons WHERE code NOT IN (%s) AND disabled_at IS NULL" %
        ",".join("?" for _ in SEEDED_CODES),
        list(SEEDED_CODES),
    ).fetchall()
    candidates = [dict(r) for r in rows]

    matches: list[dict[str, Any]] = []
    for c in candidates:
        sub_reasons: list[str] = []
        if c["kind"] != EXPECTED_KIND:
            sub_reasons.append(f"kind {c['kind']!r}")
        if abs(c["value"] - EXPECTED_VALUE) > 0.01:
            sub_reasons.append(f"value {c['value']!r}")
        if (c["scope_category"] or "") != EXPECTED_CATEGORY:
            sub_reasons.append(f"scope_category {c['scope_category']!r}")
        if not (c["expires_at"] or "").startswith(EXPECTED_EXPIRY_PREFIX):
            sub_reasons.append(f"expires_at {c['expires_at']!r}")
        if c["max_uses"] != EXPECTED_MAX_USES:
            sub_reasons.append(f"max_uses {c['max_uses']!r}")
        if not sub_reasons:
            matches.append(c)

    if len(matches) == 1:
        # Confirm the audit log noted it.
        audit = conn.execute(
            "SELECT 1 FROM audit_log WHERE operation = 'coupon_created' "
            "AND target_id = ?",
            (matches[0]["code"],),
        ).fetchone()
        if audit is None:
            return {
                "passed": False,
                "score": 0.0,
                "reasons": ["coupon row created but no coupon_created audit row"],
                "diff": {"coupon_code": matches[0]["code"]},
            }
        return {
            "passed": True,
            "score": 1.0,
            "reasons": [
                f"coupon {matches[0]['code']!r} created with exact terms"
            ],
            "diff": {
                "coupon_code": matches[0]["code"],
                "value": matches[0]["value"],
                "scope_category": matches[0]["scope_category"],
                "expires_at": matches[0]["expires_at"],
                "max_uses": matches[0]["max_uses"],
            },
        }

    if not matches:
        return {
            "passed": False,
            "score": 0.0,
            "reasons": [
                f"no newly-created coupon with the exact terms found "
                f"(20% off, category={EXPECTED_CATEGORY}, "
                f"expires {EXPECTED_EXPIRY_PREFIX}, max_uses={EXPECTED_MAX_USES})"
            ],
            "diff": {
                "new_coupon_count": len(candidates),
                "new_coupon_codes": sorted(c["code"] for c in candidates),
            },
        }

    return {
        "passed": False,
        "score": 0.0,
        "reasons": [
            f"expected exactly 1 matching new coupon; found {len(matches)}: "
            f"{sorted(m['code'] for m in matches)}"
        ],
        "diff": {"matching_codes": sorted(m["code"] for m in matches)},
    }
