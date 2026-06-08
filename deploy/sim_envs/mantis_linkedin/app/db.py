"""SQLite schema + helpers for mantis-linkedin.

In-memory by default. Single module-level connection guarded by a
re-entrant lock, schema applied at first connect.

Tables
------

* ``users`` — profile rows (handle, name, headline, about, location).
* ``experience``, ``education``, ``skills`` — per-user list rows.
* ``connections`` — directed: from_user_id → to_user_id; status
  ``pending`` | ``accepted``. The ``note`` column carries the
  Connect-modal note text.
* ``posts``, ``comments``, ``reactions`` — feed graph.
* ``threads`` + ``messages`` — messaging.
* ``jobs`` + ``job_applications`` — jobs + Easy Apply submissions.
* ``audit_log`` — the source of truth for oracles. Every
  state-changing route writes to it.

All IDs are deterministic strings (``user_00042``, ``job_00003`` etc.)
so the same SEED produces the same surface.
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
    id              TEXT PRIMARY KEY,        -- 'user_00042'
    handle          TEXT NOT NULL UNIQUE,    -- 'jane-doe-9a2f'
    name            TEXT NOT NULL,
    headline        TEXT NOT NULL DEFAULT '',
    about           TEXT NOT NULL DEFAULT '',
    location        TEXT NOT NULL DEFAULT '',
    email           TEXT NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL,
    avatar_color    TEXT NOT NULL DEFAULT '#0a66c2',
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_users_handle ON users(handle);

CREATE TABLE IF NOT EXISTS experience (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES users(id),
    title           TEXT NOT NULL,
    company         TEXT NOT NULL,
    location        TEXT NOT NULL DEFAULT '',
    start_date      TEXT NOT NULL,
    end_date        TEXT,                    -- NULL = current
    description     TEXT NOT NULL DEFAULT '',
    sort_idx        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_exp_user ON experience(user_id);

CREATE TABLE IF NOT EXISTS education (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES users(id),
    school          TEXT NOT NULL,
    degree          TEXT NOT NULL DEFAULT '',
    field           TEXT NOT NULL DEFAULT '',
    start_year      INTEGER,
    end_year        INTEGER,
    sort_idx        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_edu_user ON education(user_id);

CREATE TABLE IF NOT EXISTS skills (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES users(id),
    name            TEXT NOT NULL,
    endorsements    INTEGER NOT NULL DEFAULT 0,
    sort_idx        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_skills_user ON skills(user_id);

CREATE TABLE IF NOT EXISTS connections (
    id              TEXT PRIMARY KEY,        -- 'conn_00001'
    from_user_id    TEXT NOT NULL REFERENCES users(id),
    to_user_id      TEXT NOT NULL REFERENCES users(id),
    status          TEXT NOT NULL,           -- 'pending' | 'accepted'
    note            TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL,
    accepted_at     TEXT,
    UNIQUE (from_user_id, to_user_id)
);
CREATE INDEX IF NOT EXISTS idx_conn_to ON connections(to_user_id, status);
CREATE INDEX IF NOT EXISTS idx_conn_from ON connections(from_user_id, status);

CREATE TABLE IF NOT EXISTS posts (
    id              TEXT PRIMARY KEY,        -- 'post_00012'
    author_id       TEXT NOT NULL REFERENCES users(id),
    body            TEXT NOT NULL,
    hashtags        TEXT NOT NULL DEFAULT '[]',  -- JSON list of strings (no leading #)
    visibility      TEXT NOT NULL DEFAULT 'public',
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_posts_author ON posts(author_id);
CREATE INDEX IF NOT EXISTS idx_posts_created ON posts(created_at);

CREATE TABLE IF NOT EXISTS comments (
    id              TEXT PRIMARY KEY,
    post_id         TEXT NOT NULL REFERENCES posts(id),
    author_id       TEXT NOT NULL REFERENCES users(id),
    body            TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_comments_post ON comments(post_id);

CREATE TABLE IF NOT EXISTS reactions (
    post_id         TEXT NOT NULL REFERENCES posts(id),
    user_id         TEXT NOT NULL REFERENCES users(id),
    kind            TEXT NOT NULL,           -- 'like' | 'celebrate' | 'support' | 'love' | 'insightful' | 'funny'
    created_at      TEXT NOT NULL,
    PRIMARY KEY (post_id, user_id)
);
CREATE INDEX IF NOT EXISTS idx_reactions_post ON reactions(post_id);

CREATE TABLE IF NOT EXISTS threads (
    id              TEXT PRIMARY KEY,        -- 'thread_00003'
    participants    TEXT NOT NULL,           -- JSON list of user_ids
    last_message_at TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    thread_id       TEXT NOT NULL REFERENCES threads(id),
    sender_id       TEXT NOT NULL REFERENCES users(id),
    body            TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    read_at         TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id);

CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,        -- 'job_00003'
    title           TEXT NOT NULL,
    company         TEXT NOT NULL,
    location        TEXT NOT NULL,
    description_md  TEXT NOT NULL DEFAULT '',
    easy_apply      INTEGER NOT NULL DEFAULT 1,
    promoted        INTEGER NOT NULL DEFAULT 0,
    applicants      INTEGER NOT NULL DEFAULT 0,
    posted_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_title ON jobs(title);

CREATE TABLE IF NOT EXISTS job_applications (
    id              TEXT PRIMARY KEY,        -- 'app_00001'
    user_id         TEXT NOT NULL REFERENCES users(id),
    job_id          TEXT NOT NULL REFERENCES jobs(id),
    status          TEXT NOT NULL DEFAULT 'submitted',
    phone           TEXT NOT NULL DEFAULT '',
    resume_label    TEXT NOT NULL DEFAULT '',
    answers_json    TEXT NOT NULL DEFAULT '{}',
    submitted_at    TEXT NOT NULL,
    UNIQUE (user_id, job_id)
);

CREATE TABLE IF NOT EXISTS audit_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    occurred_at     TEXT NOT NULL,
    operation       TEXT NOT NULL,
    target_type     TEXT NOT NULL,
    target_id       TEXT NOT NULL,
    payload_json    TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_audit_op ON audit_log(operation);
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
    global _DB
    with _DB_LOCK:
        if _DB is not None:
            _DB.close()
            _DB = None


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    conn = connect()
    with _DB_LOCK:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def pack_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def unpack_json(raw: str) -> Any:
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}


def log_audit(
    conn: sqlite3.Connection,
    *,
    occurred_at: str,
    operation: str,
    target_type: str,
    target_id: str,
    payload: dict[str, Any] | None = None,
) -> None:
    """Append a row to ``audit_log``. Oracles read this to grade runs."""
    conn.execute(
        "INSERT INTO audit_log (occurred_at, operation, target_type, target_id, payload_json) "
        "VALUES (?, ?, ?, ?, ?)",
        (occurred_at, operation, target_type, target_id,
         json.dumps(payload or {}, sort_keys=True)),
    )
