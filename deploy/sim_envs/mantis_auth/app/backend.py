"""Concrete :class:`authflow.AuthBackend` over this env's SQLite store.

This is the only file that bridges the portable ``authflow`` package to
``mantis_auth``'s ``db``. Another env embedding the flow would write its
own equivalent against its own user store.

Audit semantics
---------------

Every method-specific event (``magic_link_consumed``, ``otp_verified``,
``passkey_asserted``, ``oauth_consent`` …) and the single terminal
``login_succeeded`` row are written to the ``mutations`` table — the
ground truth the oracles grade on — and mirrored to the in-memory event
log via the injected ``emit`` callback.
"""

from __future__ import annotations

from typing import Any, Callable

from . import db


def _row(cur) -> dict[str, Any] | None:
    r = cur.fetchone()
    return dict(r) if r else None


class SqliteAuthBackend:
    def __init__(self, *, now_fn: Callable[[], str],
                 emit_fn: Callable[[str, dict[str, Any]], None]) -> None:
        self._now = now_fn
        self._emit = emit_fn

    # clock ----------------------------------------------------------------
    def now(self) -> str:
        return self._now()

    # users ----------------------------------------------------------------
    def lookup_user_by_id(self, user_id: str) -> dict[str, Any] | None:
        conn = db.connect()
        return _row(conn.execute(
            "SELECT id, name, email, role, is_active FROM users WHERE id = ?",
            (user_id,)))

    def lookup_user_by_email(self, email: str) -> dict[str, Any] | None:
        conn = db.connect()
        return _row(conn.execute(
            "SELECT id, name, email, role, is_active, password_hash "
            "FROM users WHERE LOWER(email) = ?",
            (email.strip().lower(),)))

    # oauth ----------------------------------------------------------------
    def lookup_oauth_accounts(self, provider: str) -> list[dict[str, Any]]:
        conn = db.connect()
        rows = conn.execute(
            "SELECT oi.subject AS subject, oi.email AS email, "
            "       u.id AS user_id, u.name AS name "
            "FROM oauth_identities oi JOIN users u ON u.id = oi.user_id "
            "WHERE oi.provider = ? ORDER BY oi.email",
            (provider,)).fetchall()
        return [dict(r) for r in rows]

    def lookup_user_by_oauth(self, provider: str,
                             subject: str) -> dict[str, Any] | None:
        conn = db.connect()
        row = _row(conn.execute(
            "SELECT u.id AS id, u.name AS name, u.email AS email, "
            "       u.role AS role, oi.subject AS oauth_subject "
            "FROM oauth_identities oi JOIN users u ON u.id = oi.user_id "
            "WHERE oi.provider = ? AND oi.subject = ?",
            (provider, subject)))
        return row

    # passkeys -------------------------------------------------------------
    def list_passkeys(self) -> list[dict[str, Any]]:
        conn = db.connect()
        rows = conn.execute(
            "SELECT pc.id AS cred_id, pc.label AS label, pc.user_id AS user_id, "
            "       u.email AS email, u.name AS name "
            "FROM passkey_credentials pc JOIN users u ON u.id = pc.user_id "
            "ORDER BY pc.label").fetchall()
        return [dict(r) for r in rows]

    def get_passkey(self, cred_id: str) -> dict[str, Any] | None:
        conn = db.connect()
        return _row(conn.execute(
            "SELECT id AS cred_id, user_id, label, sign_count "
            "FROM passkey_credentials WHERE id = ?", (cred_id,)))

    def bump_passkey_sign_count(self, cred_id: str) -> None:
        with db.transaction() as conn:
            conn.execute(
                "UPDATE passkey_credentials SET sign_count = sign_count + 1 "
                "WHERE id = ?", (cred_id,))

    # email (magic link + OTP) --------------------------------------------
    def deliver_email(self, *, to_email: str, kind: str, subject: str,
                      body: str, token: str | None = None,
                      code: str | None = None) -> None:
        with db.transaction() as conn:
            n = conn.execute("SELECT COUNT(*) FROM emails").fetchone()[0]
            email_id = f"email_{n + 1:05d}"
            conn.execute(
                "INSERT INTO emails (id, to_email, subject, body, kind, "
                "token, code, created_at, consumed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                (email_id, to_email, subject, body, kind, token, code,
                 self._now()))

    def consume_magic_token(self, token: str) -> dict[str, Any] | None:
        with db.transaction() as conn:
            row = _row(conn.execute(
                "SELECT id, to_email FROM emails "
                "WHERE kind = 'magic_link' AND token = ? AND consumed_at IS NULL",
                (token,)))
            if row is None:
                return None
            conn.execute("UPDATE emails SET consumed_at = ? WHERE id = ?",
                         (self._now(), row["id"]))
        return self.lookup_user_by_email(row["to_email"])

    def consume_otp(self, email: str, code: str) -> dict[str, Any] | None:
        if not email or not code:
            return None
        with db.transaction() as conn:
            row = _row(conn.execute(
                "SELECT id, to_email FROM emails "
                "WHERE kind = 'otp' AND LOWER(to_email) = ? AND code = ? "
                "AND consumed_at IS NULL ORDER BY id DESC LIMIT 1",
                (email.strip().lower(), code.strip())))
            if row is None:
                return None
            conn.execute("UPDATE emails SET consumed_at = ? WHERE id = ?",
                         (self._now(), row["id"]))
        return self.lookup_user_by_email(row["to_email"])

    # audit ----------------------------------------------------------------
    def record_login(self, *, user_id: str, via: str, email: str,
                      provider: str | None = None) -> None:
        payload = {"email": email, "via": via}
        if provider:
            payload["provider"] = provider
        with db.transaction() as conn:
            db.log_mutation(conn, occurred_at=self._now(),
                            operation="login_succeeded", target_type="user",
                            target_id=user_id, payload=payload)
        self._emit("login_succeeded", {"target_id": user_id, **payload})

    def record_event(self, *, operation: str, target_id: str,
                     payload: dict[str, Any]) -> None:
        with db.transaction() as conn:
            db.log_mutation(conn, occurred_at=self._now(), operation=operation,
                            target_type="user", target_id=target_id,
                            payload=payload)
        self._emit(operation, {"target_id": target_id, **payload})
