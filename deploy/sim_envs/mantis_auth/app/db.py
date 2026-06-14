"""SQLite schema + thin query helpers for mantis-auth.

This env's whole purpose is *authentication*: a minimal SaaS console
("Mantis Console") sits behind an auth wall, and the interesting surface
is the many ways an agent can get *through* that wall — password,
OAuth (multiple providers), email magic-link, email OTP, and passkey.

The DB lives in memory by default (``DB_PATH=:memory:``) — same seed
re-runs from scratch in well under a second and isolates plan runs
perfectly. Pass ``DB_PATH=/var/lib/mantis-auth/db.sqlite`` for dev
iteration if you want state to persist across restarts.

Schema notes
------------

* All IDs are deterministic strings (``user_00001``, ``cred_00001``) so
  a plan that references a row always finds the same one.
* ``oauth_identities`` is a separate table so one user can sign in via
  several providers (Google *and* GitHub map to the same ``user_id``).
* ``emails`` is the in-env mock mailbox. Magic-link / OTP flows write a
  row here; the agent navigates ``/inbox`` to read it (just like real
  CUA email flows). The oracle reads ``consumed_at`` to prove the agent
  actually used the token rather than shortcutting.
* ``mutations`` is the append-only audit log every oracle grades on.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Iterator

# One module-level connection guarded by a lock. SQLite ``:memory:`` DBs
# are per-connection, so we keep exactly one for the process lifetime;
# the lock makes single-connection use thread-safe (FastAPI runs request
# handlers on a thread pool).
_DB_LOCK = threading.RLock()
_DB: sqlite3.Connection | None = None


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    email           TEXT NOT NULL,
    is_active       INTEGER NOT NULL DEFAULT 1,
    -- Plain sha256 hex — sim env, not prod. NULL means "no password set"
    -- (the account is OAuth/passkey-only, mirroring SSO-only users).
    password_hash   TEXT,
    role            TEXT NOT NULL DEFAULT 'member',  -- 'member' | 'admin'
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

CREATE TABLE IF NOT EXISTS oauth_identities (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES users(id),
    provider        TEXT NOT NULL,   -- 'google' | 'github' | 'microsoft' | 'okta'
    subject         TEXT NOT NULL,   -- provider-side stable id ("sub")
    email           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_oauth_subject
    ON oauth_identities(provider, subject);
CREATE INDEX IF NOT EXISTS idx_oauth_user ON oauth_identities(user_id);

CREATE TABLE IF NOT EXISTS passkey_credentials (
    id              TEXT PRIMARY KEY,   -- doubles as the WebAuthn credential id
    user_id         TEXT NOT NULL REFERENCES users(id),
    label           TEXT NOT NULL,      -- "MacBook Touch ID", "YubiKey 5C"
    sign_count      INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_passkey_user ON passkey_credentials(user_id);

CREATE TABLE IF NOT EXISTS emails (
    -- In-env mock mailbox. Magic-link + OTP flows deposit a message
    -- here; the agent reads it at ``/inbox``.
    id              TEXT PRIMARY KEY,
    to_email        TEXT NOT NULL,
    subject         TEXT NOT NULL,
    body            TEXT NOT NULL,
    kind            TEXT NOT NULL,      -- 'magic_link' | 'otp'
    token           TEXT,               -- magic-link single-use token
    code            TEXT,               -- 6-digit OTP
    created_at      TEXT NOT NULL,
    consumed_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_emails_to ON emails(to_email);

CREATE TABLE IF NOT EXISTS mutations (
    -- Audit log of every auth-relevant event after seed. Oracles read
    -- this to assert "the agent authenticated via method X".
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    occurred_at     TEXT NOT NULL,
    operation       TEXT NOT NULL,      -- e.g. 'login_succeeded'
    target_type     TEXT NOT NULL,
    target_id       TEXT NOT NULL,
    payload_json    TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_mutations_op ON mutations(operation);
"""


def db_path() -> str:
    return os.environ.get("DB_PATH") or ":memory:"


def connect() -> sqlite3.Connection:
    """Return the shared SQLite connection, creating it on first use."""
    global _DB
    with _DB_LOCK:
        if _DB is None:
            conn = sqlite3.connect(db_path(), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.executescript(SCHEMA)
            _DB = conn
        return _DB


def reset_connection() -> None:
    """Drop the in-process connection — used by tests + ``/__env__/reset``."""
    global _DB
    with _DB_LOCK:
        if _DB is not None:
            _DB.close()
            _DB = None


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    """Acquire the lock + start a transaction; commit on success."""
    conn = connect()
    with _DB_LOCK:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


# ── audit log ───────────────────────────────────────────────────────────


def log_mutation(
    conn: sqlite3.Connection,
    *,
    occurred_at: str,
    operation: str,
    target_type: str,
    target_id: str,
    payload: dict[str, Any] | None = None,
) -> None:
    """Append a mutation record. Oracles read this to grade runs."""
    conn.execute(
        "INSERT INTO mutations (occurred_at, operation, target_type, "
        "target_id, payload_json) VALUES (?, ?, ?, ?, ?)",
        (occurred_at, operation, target_type, target_id,
         json.dumps(payload or {}, sort_keys=True)),
    )
