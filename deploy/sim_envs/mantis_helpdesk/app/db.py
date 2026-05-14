"""SQLite schema + thin query helpers for mantis-helpdesk.

Mirrors the mantis-crm DB module (#332): a single in-memory SQLite
connection guarded by a module-level lock, JSON helpers for ``tags``
and ``cc``/``bcc`` arrays stored as TEXT, and an audit-log table
(``mutations``) that every oracle reads.

Schema notes
------------

* IDs are deterministic strings (``ticket_00042``, ``reply_00123``)
  driven by the seed generator's row index. Same SEED → identical ids
  across boots.
* ``tags`` on ``tickets`` is a JSON array stored as TEXT. Helpers in
  this module pack/unpack.
* ``replies.visibility`` is ``'public'`` or ``'internal'`` — the
  oracle treats a public reply on an internal-only thread as a
  critical failure.
* ``tickets.deleted_at`` is the soft-delete sentinel; the merge path
  flips it on losers.
* ``escalation_links`` stores symmetric "this ticket relates to that
  one" pairs for T03 + future escalation plans.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Iterator

_DB_LOCK = threading.RLock()
_DB: sqlite3.Connection | None = None


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    email           TEXT NOT NULL,
    role            TEXT NOT NULL DEFAULT 'requester',  -- requester | agent | admin
    group_id        TEXT,                                -- agents only
    locale          TEXT NOT NULL DEFAULT 'en',
    is_active       INTEGER NOT NULL DEFAULT 1,
    -- Mock auth surface (#387). Plain sha256 hex — sim env, not prod.
    -- Only agents + admins have a password_hash; requesters are external
    -- and never sign in to this env.
    password_hash   TEXT,
    oauth_subject   TEXT
);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_group ON users(group_id);

CREATE TABLE IF NOT EXISTS groups (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    slug            TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS tickets (
    id              TEXT PRIMARY KEY,
    subject         TEXT NOT NULL,
    body            TEXT NOT NULL DEFAULT '',
    requester_id    TEXT NOT NULL REFERENCES users(id),
    assignee_id     TEXT,                                -- nullable; unassigned
    group_id        TEXT REFERENCES groups(id),
    status          TEXT NOT NULL DEFAULT 'new',         -- new | open | pending | solved | closed
    priority        TEXT NOT NULL DEFAULT 'normal',      -- low | normal | high | urgent
    channel         TEXT NOT NULL DEFAULT 'email',       -- email | chat | form
    tags            TEXT NOT NULL DEFAULT '[]',          -- JSON array
    locale          TEXT NOT NULL DEFAULT 'en',
    visibility      TEXT NOT NULL DEFAULT 'public',      -- public | internal-only thread marker
    sla_breach_at   TEXT,                                -- ISO time when SLA breaches
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    deleted_at      TEXT                                  -- soft-delete for merged losers
);
CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);
CREATE INDEX IF NOT EXISTS idx_tickets_assignee ON tickets(assignee_id);
CREATE INDEX IF NOT EXISTS idx_tickets_group ON tickets(group_id);
CREATE INDEX IF NOT EXISTS idx_tickets_priority ON tickets(priority);
CREATE INDEX IF NOT EXISTS idx_tickets_sla ON tickets(sla_breach_at);

CREATE TABLE IF NOT EXISTS replies (
    id              TEXT PRIMARY KEY,
    ticket_id       TEXT NOT NULL REFERENCES tickets(id),
    author_id       TEXT NOT NULL REFERENCES users(id),
    body            TEXT NOT NULL DEFAULT '',
    visibility      TEXT NOT NULL DEFAULT 'public',      -- public | internal
    cc              TEXT NOT NULL DEFAULT '[]',          -- JSON array of email strings
    bcc             TEXT NOT NULL DEFAULT '[]',
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_replies_ticket ON replies(ticket_id);

CREATE TABLE IF NOT EXISTS macros (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    body            TEXT NOT NULL,                       -- Body with {{requester.first_name}} etc.
    folder          TEXT NOT NULL DEFAULT 'general'
);

CREATE TABLE IF NOT EXISTS triggers (
    -- Read-only routing rules. The bulk-assign path applies them as a
    -- post-mutation hook so e.g. assigning a billing-group ticket to an
    -- agent outside the billing group auto-reverts. We mirror Zendesk's
    -- "trigger" UI surface — agents see them, can't edit them in v1.
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    condition_json  TEXT NOT NULL DEFAULT '{}',          -- e.g. {"group_id": "group_billing"}
    action_json     TEXT NOT NULL DEFAULT '{}',          -- e.g. {"revert_assignee_to_group": true}
    is_active       INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS escalation_links (
    -- Symmetric pair store for T03 + future escalation plans. We always
    -- write both rows so a SELECT on either side surfaces the link.
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id       TEXT NOT NULL REFERENCES tickets(id),
    related_ticket_id TEXT NOT NULL REFERENCES tickets(id),
    relation        TEXT NOT NULL DEFAULT 'related',     -- related | escalates_to | duplicate_of
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_escalations_ticket ON escalation_links(ticket_id);

CREATE TABLE IF NOT EXISTS mutations (
    -- Audit log. Every state-changing route appends; oracles read.
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    occurred_at     TEXT NOT NULL,
    operation       TEXT NOT NULL,                       -- e.g. 'ticket_priority_changed'
    target_type     TEXT NOT NULL,                       -- ticket | reply | macro
    target_id       TEXT NOT NULL,
    payload_json    TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_mutations_target ON mutations(target_type, target_id);
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
    """Drop the in-process connection — used by ``/__env__/reset``."""
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


# ── JSON helpers ────────────────────────────────────────────────────────


def pack_tags(tags: list[str]) -> str:
    return json.dumps(sorted(set(tags)))


def unpack_tags(raw: str) -> list[str]:
    try:
        out = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    return list(out) if isinstance(out, list) else []


def pack_emails(emails: list[str]) -> str:
    return json.dumps([e.strip() for e in emails if e and e.strip()])


def unpack_emails(raw: str) -> list[str]:
    try:
        out = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    return [str(x) for x in out] if isinstance(out, list) else []


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
    """Append a mutation record. The oracle reads this to grade runs."""
    conn.execute(
        "INSERT INTO mutations (occurred_at, operation, target_type, target_id, payload_json) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            occurred_at, operation, target_type, target_id,
            json.dumps(payload or {}, sort_keys=True),
        ),
    )
