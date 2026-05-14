"""T06_login_then_triage_inbox — adds the auth gate on top of T01.

Pass conditions
---------------

1. ``mutations`` has a ``login_succeeded`` row with target_id
   ``agent_001`` (the seeded demo agent — note helpdesk uses 3-digit
   padding for agents, unlike CRM's 5-digit) *before* the first ticket
   mutation T01 grades on.
2. Login row's payload was minted via the real flow
   (``via='password'`` or ``via='oauth'``); harness shortcuts don't
   write that row.
3. All of T01_triage_inbox's existing pass conditions still hold.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from . import t01_triage_inbox

EXPECTED_USER_ID = "agent_001"
VALID_LOGIN_VIA = {"password", "oauth"}


def grade(conn: sqlite3.Connection, *, now: str, seed_val: int) -> dict[str, Any]:
    reasons: list[str] = []

    login_row = conn.execute(
        "SELECT id, occurred_at, payload_json FROM mutations "
        "WHERE operation = 'login_succeeded' AND target_id = ? "
        "ORDER BY id ASC LIMIT 1",
        (EXPECTED_USER_ID,),
    ).fetchone()
    login_mut_id = None
    if login_row is None:
        reasons.append(
            f"no login_succeeded mutation for {EXPECTED_USER_ID}; "
            "agent must navigate /login (password) or /oauth/authorize first"
        )
    else:
        login_mut_id = login_row["id"]
        raw = login_row["payload_json"] or ""
        if not any(f'"via": "{v}"' in raw or f'"via":"{v}"' in raw
                   for v in VALID_LOGIN_VIA):
            reasons.append(
                "login_succeeded mutation missing valid `via` "
                f"(expected one of {sorted(VALID_LOGIN_VIA)})"
            )

    # At least one ticket mutation must come after login.
    if login_mut_id is not None:
        post = conn.execute(
            "SELECT COUNT(*) FROM mutations "
            "WHERE target_type = 'ticket' AND id > ?",
            (login_mut_id,),
        ).fetchone()[0]
        if post == 0:
            reasons.append(
                "no ticket mutations recorded after login_succeeded"
            )

    inner = t01_triage_inbox.grade(conn, now=now, seed_val=seed_val)
    if not inner["passed"]:
        reasons.extend(f"[T01] {r}" for r in inner["reasons"])

    passed = not reasons
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reasons": reasons or [
            f"logged in as {EXPECTED_USER_ID} then completed T01_triage_inbox"
        ],
        "diff": {
            "login_mutation_id": login_mut_id,
            **inner.get("diff", {}),
        },
    }
