"""T06_login_then_buy_jacket — adds the auth gate on top of T01.

Pass conditions
---------------

1. ``audit_log`` contains a ``login_succeeded`` row with target_id
   ``user_demo`` *before* the ``order_placed`` row. (Audit ids
   auto-increment, so id-ordering is the proof of "logged in first".)
2. The login row's payload was minted via the real flow — either
   ``via='password'`` or ``via='oauth'``. Synthetic seeds / harness
   bypasses don't write this row, so they fail the gate.
3. All of T01's existing pass conditions still hold (order shape +
   coupon + inventory + no collateral inventory mutations).

This is the gradable surface for issue #387: it asserts the agent
actually navigated the login screen, not that the harness pre-authed
a cookie.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from . import t01_buy_jacket

EXPECTED_USER_ID = "user_demo"
VALID_LOGIN_VIA = {"password", "oauth"}


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    reasons: list[str] = []

    # 1+2: login row exists and is from a real flow.
    login_row = conn.execute(
        "SELECT id, occurred_at, payload_json FROM audit_log "
        "WHERE operation = 'login_succeeded' AND target_id = ? "
        "ORDER BY id ASC LIMIT 1",
        (EXPECTED_USER_ID,),
    ).fetchone()
    login_audit_id = None
    if login_row is None:
        reasons.append(
            f"no login_succeeded audit row for {EXPECTED_USER_ID}; "
            "agent must navigate /login (password) or /oauth/authorize first"
        )
    else:
        login_audit_id = login_row["id"]
        # payload_json is a json string; we don't need to deserialize to
        # check a simple substring on ``via=``.
        raw = login_row["payload_json"] or ""
        if not any(f'"via": "{v}"' in raw or f'"via":"{v}"' in raw
                   for v in VALID_LOGIN_VIA):
            reasons.append(
                "login_succeeded audit row missing valid `via` "
                f"(expected one of {sorted(VALID_LOGIN_VIA)})"
            )

    # 1 (cont): login precedes order_placed.
    order_row = conn.execute(
        "SELECT id FROM audit_log "
        "WHERE operation = 'order_placed' ORDER BY id ASC LIMIT 1"
    ).fetchone()
    if login_audit_id is not None and order_row is not None:
        if order_row["id"] <= login_audit_id:
            reasons.append(
                "order_placed audit id precedes login_succeeded — "
                "the order must be placed *after* logging in"
            )

    # 3: T01 still grades the order shape.
    inner = t01_buy_jacket.grade(conn, now=now, seed_val=seed_val)
    if not inner["passed"]:
        reasons.extend(f"[T01] {r}" for r in inner["reasons"])

    passed = not reasons
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reasons": reasons or [
            f"logged in as {EXPECTED_USER_ID} then placed {inner['diff'].get('order_id')}"
        ],
        "diff": {
            "login_audit_id": login_audit_id,
            "order_audit_id": order_row["id"] if order_row else None,
            **inner.get("diff", {}),
        },
    }
