"""SQLite schema + helpers for mantis-indeed.

Same pattern as mantis-shop: in-memory by default, a single module-level
connection guarded by a re-entrant lock, schema applied at first connect.

Schema highlights
-----------------

* ``companies`` — employer companies (logo letter is auto-derived from
  company name).
* ``users`` — seekers + employers. ``role`` is ``seeker`` or ``employer``.
* ``jobs`` — title, company_id, location, salary_low/high, remote_flag,
  job_type, posted_date, description, ``jk`` (Indeed-style 16-hex job
  key — what URLs reference).
* ``resumes`` — per-seeker resumes.
* ``applications`` — seeker submitting to a job. Status field tracks the
  employer's review workflow.
* ``saved_jobs`` — seeker bookmarks.
* ``audit_log`` — append-only mutation record. Oracle reads this +
  current DB state to grade runs.

IDs are deterministic strings: ``user_00007``, ``job_00012``, etc.
Job-key ``jk`` is the URL-facing 16-hex token, separate from ``id``
since Indeed's URLs use ``jk=...``.
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
CREATE TABLE IF NOT EXISTS companies (
    id              TEXT PRIMARY KEY,           -- 'company_00001'
    name            TEXT NOT NULL UNIQUE,
    rating          REAL NOT NULL DEFAULT 4.0,
    review_count    INTEGER NOT NULL DEFAULT 0,
    industry        TEXT NOT NULL DEFAULT '',
    headquarters    TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_companies_name ON companies(name);

CREATE TABLE IF NOT EXISTS users (
    id              TEXT PRIMARY KEY,           -- 'user_00007'
    email           TEXT NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL,
    role            TEXT NOT NULL,              -- 'seeker' | 'employer'
    name            TEXT NOT NULL DEFAULT '',
    phone           TEXT NOT NULL DEFAULT '',
    city            TEXT NOT NULL DEFAULT '',
    state           TEXT NOT NULL DEFAULT '',
    zip             TEXT NOT NULL DEFAULT '',
    company_id      TEXT REFERENCES companies(id),    -- employers only
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_company ON users(company_id);

CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,           -- 'job_00012'
    jk              TEXT NOT NULL UNIQUE,       -- 16-hex URL key
    title           TEXT NOT NULL,
    company_id      TEXT NOT NULL REFERENCES companies(id),
    location        TEXT NOT NULL,              -- 'Austin, TX'
    salary_low      INTEGER NOT NULL DEFAULT 0,
    salary_high     INTEGER NOT NULL DEFAULT 0,
    salary_period   TEXT NOT NULL DEFAULT 'year', -- 'year' | 'hour'
    remote_flag     INTEGER NOT NULL DEFAULT 0,
    job_type        TEXT NOT NULL DEFAULT 'Full-time',
    experience_level TEXT NOT NULL DEFAULT 'Mid level',
    posted_at       TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    snippet         TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'active'   -- 'active' | 'draft' | 'closed'
);
CREATE INDEX IF NOT EXISTS idx_jobs_jk ON jobs(jk);
CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company_id);
CREATE INDEX IF NOT EXISTS idx_jobs_location ON jobs(location);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);

CREATE TABLE IF NOT EXISTS resumes (
    id              TEXT PRIMARY KEY,           -- 'resume_00001'
    user_id         TEXT NOT NULL REFERENCES users(id),
    title           TEXT NOT NULL,
    summary         TEXT NOT NULL DEFAULT '',
    skills          TEXT NOT NULL DEFAULT '[]',   -- JSON list
    experience      TEXT NOT NULL DEFAULT '',
    updated_at      TEXT NOT NULL,
    deleted_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_resumes_user ON resumes(user_id);

CREATE TABLE IF NOT EXISTS applications (
    id              TEXT PRIMARY KEY,           -- 'application_00001'
    user_id         TEXT NOT NULL REFERENCES users(id),
    job_id          TEXT NOT NULL REFERENCES jobs(id),
    resume_id       TEXT REFERENCES resumes(id),
    phone           TEXT NOT NULL DEFAULT '',
    answers_json    TEXT NOT NULL DEFAULT '{}',   -- JSON dict of screening Q&A
    status          TEXT NOT NULL DEFAULT 'new',  -- 'new' | 'reviewed' | 'rejected' | 'hired'
    applied_at      TEXT NOT NULL,
    reviewed_at     TEXT,
    UNIQUE (user_id, job_id)
);
CREATE INDEX IF NOT EXISTS idx_apps_user ON applications(user_id);
CREATE INDEX IF NOT EXISTS idx_apps_job ON applications(job_id);
CREATE INDEX IF NOT EXISTS idx_apps_status ON applications(status);

CREATE TABLE IF NOT EXISTS saved_jobs (
    user_id         TEXT NOT NULL REFERENCES users(id),
    job_id          TEXT NOT NULL REFERENCES jobs(id),
    saved_at        TEXT NOT NULL,
    PRIMARY KEY (user_id, job_id)
);

CREATE TABLE IF NOT EXISTS audit_log (
    -- Append-only mutation record. The oracle reads this + the DB
    -- snapshot to grade runs.
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    occurred_at     TEXT NOT NULL,
    operation       TEXT NOT NULL,
    target_type     TEXT NOT NULL,
    target_id       TEXT NOT NULL,
    payload_json    TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_audit_op ON audit_log(operation);
CREATE INDEX IF NOT EXISTS idx_audit_target ON audit_log(target_id);
"""


def db_path() -> str:
    return os.environ.get("DB_PATH") or ":memory:"


def connect() -> sqlite3.Connection:
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


# ── JSON helpers ──────────────────────────────────────────────────────


def pack_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def unpack_json(raw: str) -> Any:
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}


# ── audit log ─────────────────────────────────────────────────────────


def log_audit(
    conn: sqlite3.Connection,
    *,
    occurred_at: str,
    operation: str,
    target_type: str,
    target_id: str,
    payload: dict[str, Any] | None = None,
) -> None:
    """Append a row to ``audit_log``. The oracle reads this to grade runs."""
    conn.execute(
        "INSERT INTO audit_log (occurred_at, operation, target_type, target_id, payload_json) "
        "VALUES (?, ?, ?, ?, ?)",
        (occurred_at, operation, target_type, target_id,
         json.dumps(payload or {}, sort_keys=True)),
    )
