"""Shared helpers for the mantis-auth oracles.

Every scenario grades on the same ground truth: the ``mutations`` audit
log the auth flow writes. The terminal fact is a ``login_succeeded`` row
tagged with ``via`` (and ``provider`` for OAuth); method scenarios add a
proof-of-ceremony check (token consumed, code verified, passkey asserted)
so a shortcut that mints a session without walking the flow still fails.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

# The canonical demo account every scenario targets — enrolled in every
# method. See ``seed.py``.
DEMO_USER_ID = "user_00001"


def _parse(payload_json: str | None) -> dict[str, Any]:
    try:
        out = json.loads(payload_json or "{}")
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        return {}


def login_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """All ``login_succeeded`` mutations, oldest first, payload unpacked."""
    rows = conn.execute(
        "SELECT id, target_id, payload_json FROM mutations "
        "WHERE operation = 'login_succeeded' ORDER BY id ASC").fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        p = _parse(r["payload_json"])
        out.append({"id": r["id"], "user_id": r["target_id"],
                    "via": p.get("via"), "provider": p.get("provider"),
                    "email": p.get("email")})
    return out


def first_event(conn: sqlite3.Connection, operation: str) -> dict[str, Any] | None:
    r = conn.execute(
        "SELECT id, target_id, payload_json FROM mutations "
        "WHERE operation = ? ORDER BY id ASC LIMIT 1", (operation,)).fetchone()
    if r is None:
        return None
    return {"id": r["id"], "user_id": r["target_id"],
            **_parse(r["payload_json"])}


def verdict(passed: bool, reasons: list[str],
            diff: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reasons": reasons,
        "diff": diff,
    }


def grade_login_via(conn: sqlite3.Connection, *, via: str,
                    provider: str | None = None,
                    user_id: str = DEMO_USER_ID) -> dict[str, Any]:
    """Generic grader: a ``login_succeeded`` for ``user_id`` via ``via``
    (and ``provider`` when given). Used directly by the simpler scenarios
    and as the terminal check by the method-specific ones."""
    matches = [
        r for r in login_rows(conn)
        if r["user_id"] == user_id and r["via"] == via
        and (provider is None or r["provider"] == provider)
    ]
    label = f"{via}" + (f"/{provider}" if provider else "")
    if not matches:
        return verdict(False, [
            f"no login_succeeded for {user_id} via {label}; the agent must "
            "complete the sign-in flow (a session minted by other means "
            "won't write this audit row)",
        ], {"login_mutation_id": None, "via": via, "provider": provider})
    return verdict(True, [f"signed in as {user_id} via {label}"],
                   {"login_mutation_id": matches[0]["id"], "via": via,
                    "provider": provider})
