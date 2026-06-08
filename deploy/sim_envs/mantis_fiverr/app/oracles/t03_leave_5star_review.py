"""t03_leave_5star_review — buyer leaves a 5-star review on order_00007.

Pass conditions:

1. A reviews row exists for order_00007 with stars == 5.
2. A ``review_submitted`` audit row exists with stars == 5 and the right
   order_id.
3. The gig's avg_rating + review_count were recomputed (review_count
   incremented by 1; avg_rating shifted toward the new 5-star value).
4. Buyer matches the buyer on order_00007 (you can only review your own
   order).
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

EXPECTED_ORDER = "order_00007"
EXPECTED_STARS = 5


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    reasons: list[str] = []

    order = conn.execute(
        "SELECT id, buyer_id, gig_id FROM orders WHERE id = ?",
        (EXPECTED_ORDER,),
    ).fetchone()
    if order is None:
        return _fail([f"missing order {EXPECTED_ORDER}"], {})

    review = conn.execute(
        "SELECT * FROM reviews WHERE order_id = ?", (EXPECTED_ORDER,),
    ).fetchone()
    if review is None:
        reasons.append(
            f"no review row for {EXPECTED_ORDER}; agent did not publish"
        )
        return _fail(reasons, {})

    if review["stars"] != EXPECTED_STARS:
        reasons.append(
            f"review stars={review['stars']} ≠ {EXPECTED_STARS}"
        )
    if review["buyer_id"] != order["buyer_id"]:
        reasons.append(
            f"review buyer_id={review['buyer_id']!r} ≠ order buyer "
            f"{order['buyer_id']!r}"
        )
    if review["gig_id"] != order["gig_id"]:
        reasons.append(
            f"review gig_id={review['gig_id']!r} ≠ order gig "
            f"{order['gig_id']!r}"
        )

    # Audit log row
    audit_rows = conn.execute(
        "SELECT payload_json FROM audit_log "
        "WHERE operation = 'review_submitted' AND target_id = ?",
        (review["id"],),
    ).fetchall()
    if not audit_rows:
        reasons.append("no review_submitted audit row")
    else:
        try:
            p = json.loads(audit_rows[-1]["payload_json"] or "{}")
        except json.JSONDecodeError:
            p = {}
        if p.get("stars") != EXPECTED_STARS:
            reasons.append(
                f"audit row stars={p.get('stars')} ≠ {EXPECTED_STARS}"
            )

    # Gig avg + review_count recomputed
    gig_row = conn.execute(
        "SELECT avg_rating, review_count FROM gigs WHERE id = ?",
        (order["gig_id"],),
    ).fetchone()
    # Compare to what we'd compute from reviews directly.
    expected = conn.execute(
        "SELECT AVG(stars) AS avg, COUNT(*) AS cnt FROM reviews WHERE gig_id = ?",
        (order["gig_id"],),
    ).fetchone()
    if gig_row["review_count"] != int(expected["cnt"]):
        reasons.append(
            f"gigs.review_count={gig_row['review_count']} ≠ counted "
            f"{int(expected['cnt'])}"
        )
    if abs(gig_row["avg_rating"] - float(expected["avg"] or 0.0)) > 0.06:
        reasons.append(
            f"gigs.avg_rating={gig_row['avg_rating']} not consistent with "
            f"reviews avg {float(expected['avg'] or 0.0):.2f}"
        )

    passed = not reasons
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reasons": reasons or [
            f"5-star review {review['id']} published on {EXPECTED_ORDER}; "
            f"gig avg now {gig_row['avg_rating']}, {gig_row['review_count']} reviews"
        ],
        "diff": {
            "review_id": review["id"],
            "stars": review["stars"],
            "gig_avg_rating": gig_row["avg_rating"],
            "gig_review_count": gig_row["review_count"],
        },
    }


def _fail(reasons: list[str], diff: dict[str, Any]) -> dict[str, Any]:
    return {"passed": False, "score": 0.0, "reasons": reasons, "diff": diff}
