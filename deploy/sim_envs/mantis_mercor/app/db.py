"""SQLite schema + helpers for mantis-mercor.

Mirrors mantis-shop / mantis-crm: in-memory by default, single shared
connection, schema applied at first connect.

Entities
--------
* ``users`` — candidates + clients (role column).
* ``companies`` — client orgs (owner = client user).
* ``jobs`` — open roles posted by a company.
* ``candidate_profiles`` — resume / skills / hourly_rate / availability
  per candidate user.
* ``applications`` — a candidate applying to a job. Tracks status +
  screening_answers.
* ``shortlist_entries`` — a client adding a candidate to a job-level
  shortlist.
* ``interviews``, ``reviews``, ``payments`` — historical / read-only.
* ``audit_log`` — the SOURCE OF TRUTH oracles grade against.
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
    email           TEXT NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL,
    role            TEXT NOT NULL,            -- 'candidate' | 'client' | 'admin'
    name            TEXT NOT NULL DEFAULT '',
    company_id      TEXT,                     -- nullable; set for clients
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

CREATE TABLE IF NOT EXISTS companies (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    domain          TEXT NOT NULL DEFAULT '',
    owner_user_id   TEXT REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS candidate_profiles (
    user_id         TEXT PRIMARY KEY REFERENCES users(id),
    headline        TEXT NOT NULL DEFAULT '',
    skills_json     TEXT NOT NULL DEFAULT '[]',
    hourly_rate     REAL NOT NULL DEFAULT 0,
    availability    TEXT NOT NULL DEFAULT 'Part-time',
    resume_text     TEXT NOT NULL DEFAULT '',
    updated_at      TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    company_id      TEXT NOT NULL REFERENCES companies(id),
    title           TEXT NOT NULL,
    category        TEXT NOT NULL,            -- 'Medical' | 'Legal' | 'Finance' | 'Software' | 'Consulting' | 'Office'
    description_md  TEXT NOT NULL DEFAULT '',
    skills_json     TEXT NOT NULL DEFAULT '[]',
    rate_min        REAL NOT NULL,
    rate_max        REAL NOT NULL,
    engagement      TEXT NOT NULL,            -- 'Hourly' | 'Project' | 'Full-time'
    hours_per_week  INTEGER NOT NULL DEFAULT 20,
    hires_recently  INTEGER NOT NULL DEFAULT 0,
    screening_qs    TEXT NOT NULL DEFAULT '[]',
    status          TEXT NOT NULL DEFAULT 'open',
    posted_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_category ON jobs(category);
CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company_id);

CREATE TABLE IF NOT EXISTS applications (
    id              TEXT PRIMARY KEY,
    job_id          TEXT NOT NULL REFERENCES jobs(id),
    candidate_id    TEXT NOT NULL REFERENCES users(id),
    status          TEXT NOT NULL,            -- 'draft' | 'submitted' | 'under_review' | 'interview' | 'rejected' | 'hired'
    headline        TEXT NOT NULL DEFAULT '',
    skills_json     TEXT NOT NULL DEFAULT '[]',
    hourly_rate     REAL NOT NULL DEFAULT 0,
    resume_text     TEXT NOT NULL DEFAULT '',
    screening_answers TEXT NOT NULL DEFAULT '[]',
    reject_reason   TEXT NOT NULL DEFAULT '',
    submitted_at    TEXT,
    updated_at      TEXT NOT NULL,
    UNIQUE (job_id, candidate_id)
);
CREATE INDEX IF NOT EXISTS idx_app_status ON applications(status);
CREATE INDEX IF NOT EXISTS idx_app_job ON applications(job_id);
CREATE INDEX IF NOT EXISTS idx_app_cand ON applications(candidate_id);

CREATE TABLE IF NOT EXISTS shortlist_entries (
    id              TEXT PRIMARY KEY,
    job_id          TEXT NOT NULL REFERENCES jobs(id),
    candidate_id    TEXT NOT NULL REFERENCES users(id),
    added_by        TEXT NOT NULL REFERENCES users(id),
    created_at      TEXT NOT NULL,
    UNIQUE (job_id, candidate_id)
);
CREATE INDEX IF NOT EXISTS idx_shortlist_job ON shortlist_entries(job_id);

CREATE TABLE IF NOT EXISTS interviews (
    id              TEXT PRIMARY KEY,
    application_id  TEXT NOT NULL REFERENCES applications(id),
    scheduled_at    TEXT NOT NULL,
    interviewer     TEXT NOT NULL,
    outcome         TEXT NOT NULL DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS reviews (
    id              TEXT PRIMARY KEY,
    application_id  TEXT NOT NULL REFERENCES applications(id),
    rating          INTEGER NOT NULL,
    comments        TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS payments (
    id              TEXT PRIMARY KEY,
    application_id  TEXT NOT NULL REFERENCES applications(id),
    amount          REAL NOT NULL,
    occurred_at     TEXT NOT NULL
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


def unpack_json(raw: str, default: Any = None) -> Any:
    try:
        return json.loads(raw or ("[]" if default is None else json.dumps(default)))
    except json.JSONDecodeError:
        return default if default is not None else []


def log_audit(
    conn: sqlite3.Connection,
    *,
    occurred_at: str,
    operation: str,
    target_type: str,
    target_id: str,
    payload: dict[str, Any] | None = None,
) -> None:
    conn.execute(
        "INSERT INTO audit_log (occurred_at, operation, target_type, target_id, payload_json) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            occurred_at,
            operation,
            target_type,
            target_id,
            json.dumps(payload or {}, sort_keys=True),
        ),
    )
