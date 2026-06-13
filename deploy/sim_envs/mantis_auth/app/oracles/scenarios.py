"""Per-method graders, built on the shared helpers in ``_common``.

Kept in one module because the scenarios are close variants of a single
contract — adding a method is a few lines, not a new file. Each public
``grade_*`` matches the oracle signature ``(conn, *, now, seed_val)``.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from . import _common as C


def grade_password(conn: sqlite3.Connection, *, now: str,
                   seed_val: int) -> dict[str, Any]:
    return C.grade_login_via(conn, via="password")


def make_oauth_grader(provider: str):
    def grade(conn: sqlite3.Connection, *, now: str,
              seed_val: int) -> dict[str, Any]:
        return C.grade_login_via(conn, via="oauth", provider=provider)
    grade.__name__ = f"grade_oauth_{provider}"
    grade.__doc__ = f"Login via OAuth with the {provider} provider."
    return grade


def grade_magic_link(conn: sqlite3.Connection, *, now: str,
                     seed_val: int) -> dict[str, Any]:
    reasons: list[str] = []
    consumed = C.first_event(conn, "magic_link_consumed")
    if consumed is None:
        reasons.append(
            "no magic_link_consumed event; the agent must open the link "
            "delivered to /inbox rather than guessing the verify URL")
    # Proof the actual emailed token was burned.
    used = conn.execute(
        "SELECT COUNT(*) FROM emails "
        "WHERE kind = 'magic_link' AND consumed_at IS NOT NULL").fetchone()[0]
    if used == 0:
        reasons.append("no magic-link email was marked consumed")

    inner = C.grade_login_via(conn, via="magic_link")
    if not inner["passed"]:
        reasons.extend(inner["reasons"])
    passed = not reasons
    return C.verdict(
        passed,
        reasons or ["read the magic link from the inbox and signed in"],
        {"magic_links_consumed": used, **inner["diff"]})


def grade_email_otp(conn: sqlite3.Connection, *, now: str,
                    seed_val: int) -> dict[str, Any]:
    reasons: list[str] = []
    verified = C.first_event(conn, "otp_verified")
    if verified is None:
        reasons.append(
            "no otp_verified event; the agent must read the code from "
            "/inbox and submit it")
    used = conn.execute(
        "SELECT COUNT(*) FROM emails "
        "WHERE kind = 'otp' AND consumed_at IS NOT NULL").fetchone()[0]
    if used == 0:
        reasons.append("no OTP email was marked consumed")

    inner = C.grade_login_via(conn, via="email_otp")
    if not inner["passed"]:
        reasons.extend(inner["reasons"])
    passed = not reasons
    return C.verdict(
        passed,
        reasons or ["read the OTP from the inbox and signed in"],
        {"otp_codes_consumed": used, **inner["diff"]})


def grade_passkey(conn: sqlite3.Connection, *, now: str,
                  seed_val: int) -> dict[str, Any]:
    reasons: list[str] = []
    asserted = C.first_event(conn, "passkey_asserted")
    if asserted is None:
        reasons.append("no passkey_asserted event; the agent must complete "
                       "the passkey ceremony at /auth/passkey")
    bumped = conn.execute(
        "SELECT COUNT(*) FROM passkey_credentials WHERE sign_count > 0"
    ).fetchone()[0]
    if bumped == 0:
        reasons.append("no passkey credential sign_count was incremented")

    inner = C.grade_login_via(conn, via="passkey")
    if not inner["passed"]:
        reasons.extend(inner["reasons"])
    passed = not reasons
    return C.verdict(
        passed,
        reasons or ["completed the passkey assertion and signed in"],
        {"passkeys_used": bumped, **inner["diff"]})
