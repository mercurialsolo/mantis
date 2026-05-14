"""SQLite schema + thin query helpers for mantis-crm.

The DB lives in memory by default — same seed re-runs from scratch in
<2s and isolates plan runs perfectly. ``DB_PATH=:memory:`` is the
container default; pass ``DB_PATH=/var/lib/mantis-crm/db.sqlite`` for
dev iteration if you want state to persist across restarts.

Schema notes
------------

* All IDs are *deterministic strings* (``contact_00001``, ``deal_00042``)
  not autoincrement ints. The seed generator derives them from row
  index so a plan that references ``contact_01234`` always finds the
  same contact.
* ``tags`` on ``contacts`` is a JSON array stored as TEXT — SQLite
  doesn't have arrays. Helpers in this module pack/unpack.
* ``custom_fields`` on ``contacts`` is JSON TEXT. Treat as
  schema-less.
* ``activities.target_type`` + ``activities.target_id`` is the
  polymorphic FK (``contact`` | ``deal``).

Two indexes:

* ``contacts(email)`` — for duplicate detection + search.
* ``activities(target_type, target_id)`` — for "load activity timeline".

Real CRMs have a hundred more indexes; we add them as the agent gets
better and table-scans start to matter.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Iterator

# A module-level connection guarded by a lock. SQLite's ``:memory:`` DBs
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
    -- Mock auth surface (#387). Plain sha256 hex — sim env, not prod.
    password_hash   TEXT,
    role            TEXT NOT NULL DEFAULT 'agent',  -- 'agent' | 'admin'
    oauth_subject   TEXT
);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

CREATE TABLE IF NOT EXISTS companies (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    domain          TEXT,
    size_band       TEXT,
    industry        TEXT,
    arr_band        TEXT,
    parent_company_id TEXT REFERENCES companies(id)
);

CREATE TABLE IF NOT EXISTS contacts (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    email           TEXT,
    phone           TEXT,
    company_id      TEXT REFERENCES companies(id),
    lifecycle_stage TEXT,
    owner_id        TEXT,
    tags            TEXT NOT NULL DEFAULT '[]',
    created_at      TEXT NOT NULL,
    last_activity_at TEXT,
    custom_fields   TEXT NOT NULL DEFAULT '{}',
    source          TEXT,
    deleted_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_contacts_email ON contacts(email);
CREATE INDEX IF NOT EXISTS idx_contacts_company ON contacts(company_id);

CREATE TABLE IF NOT EXISTS deals (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    contact_id      TEXT REFERENCES contacts(id),
    company_id      TEXT REFERENCES companies(id),
    stage           TEXT NOT NULL,
    amount          REAL NOT NULL,
    expected_close  TEXT,
    owner_id        TEXT
);
CREATE INDEX IF NOT EXISTS idx_deals_stage ON deals(stage);

CREATE TABLE IF NOT EXISTS activities (
    id              TEXT PRIMARY KEY,
    target_type     TEXT NOT NULL,
    target_id       TEXT NOT NULL,
    activity_type   TEXT NOT NULL,
    body            TEXT NOT NULL DEFAULT '',
    actor_id        TEXT,
    occurred_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_activities_target
    ON activities(target_type, target_id);

CREATE TABLE IF NOT EXISTS lists (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    kind            TEXT NOT NULL,            -- 'filter' | 'manual'
    filter_json     TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS list_members (
    list_id         TEXT NOT NULL REFERENCES lists(id),
    member_type     TEXT NOT NULL,            -- 'contact' | 'deal'
    member_id       TEXT NOT NULL,
    PRIMARY KEY (list_id, member_type, member_id)
);

CREATE TABLE IF NOT EXISTS mutations (
    -- Audit log of every mutation that happens after seed. The oracle
    -- reads this to assert "the agent touched exactly the right rows".
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    occurred_at     TEXT NOT NULL,
    operation       TEXT NOT NULL,            -- e.g. 'tag_added'
    target_type     TEXT NOT NULL,
    target_id       TEXT NOT NULL,
    payload_json    TEXT NOT NULL DEFAULT '{}'
);

-- ── advanced CRM surfaces ────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS tasks (
    -- Work items distinct from activities. Activities are "what
    -- happened"; tasks are "what should happen". Real CRMs split these
    -- and we mirror that — agents have to navigate two separate panels.
    id              TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    body            TEXT NOT NULL DEFAULT '',
    target_type     TEXT,                    -- 'contact' | 'deal' | NULL (standalone)
    target_id       TEXT,
    assignee_id     TEXT,
    due_date        TEXT,
    priority        TEXT NOT NULL DEFAULT 'normal', -- low | normal | high
    completed_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_tasks_target
    ON tasks(target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_tasks_due ON tasks(due_date);

CREATE TABLE IF NOT EXISTS notes (
    -- Persistent free-form notes anchored to a record. Distinct from
    -- activities: notes get pinned, activities scroll. Used in real
    -- CRMs for "background on this account" rather than "we called Tuesday".
    id              TEXT PRIMARY KEY,
    target_type     TEXT NOT NULL,           -- 'contact' | 'deal' | 'company'
    target_id       TEXT NOT NULL,
    body_md         TEXT NOT NULL,
    pinned          INTEGER NOT NULL DEFAULT 0,
    author_id       TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notes_target
    ON notes(target_type, target_id);

CREATE TABLE IF NOT EXISTS email_templates (
    -- Shared reply templates with merge fields like {{contact.first_name}}.
    -- Agents in real CRMs spend a lot of clicks on these.
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    subject         TEXT NOT NULL,
    body            TEXT NOT NULL,
    folder          TEXT NOT NULL DEFAULT 'general'
);

CREATE TABLE IF NOT EXISTS lifecycle_transitions (
    -- Materialised stage-change log so the contact detail page can show
    -- "lead → mql at 2025-08-04T…". Most CRMs compute this off the
    -- mutations log; we keep it explicit for cheaper queries.
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    contact_id      TEXT NOT NULL,
    from_stage      TEXT,
    to_stage        TEXT NOT NULL,
    occurred_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_lc_contact ON lifecycle_transitions(contact_id);
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


def pack_custom_fields(fields: dict[str, Any]) -> str:
    return json.dumps(fields, sort_keys=True)


def unpack_custom_fields(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}


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
        (occurred_at, operation, target_type, target_id,
         json.dumps(payload or {}, sort_keys=True)),
    )
