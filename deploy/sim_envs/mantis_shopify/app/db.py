"""SQLite schema for mantis-shopify.

Mirrors mantis-mercor's pattern: in-memory by default, single shared
connection, schema applied on first connect. audit_log is the
oracle source-of-truth.
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
CREATE TABLE IF NOT EXISTS partners (
    id              TEXT PRIMARY KEY,
    partner_id      TEXT NOT NULL,
    business_name   TEXT NOT NULL DEFAULT '',
    website         TEXT NOT NULL DEFAULT '',
    business_email  TEXT NOT NULL DEFAULT '',
    support_email   TEXT NOT NULL DEFAULT '',
    phone           TEXT NOT NULL DEFAULT '',
    address1        TEXT NOT NULL DEFAULT '',
    address2        TEXT NOT NULL DEFAULT '',
    city            TEXT NOT NULL DEFAULT '',
    zip             TEXT NOT NULL DEFAULT '',
    state           TEXT NOT NULL DEFAULT '',
    country         TEXT NOT NULL DEFAULT '',
    emergency_name  TEXT NOT NULL DEFAULT '',
    emergency_email TEXT NOT NULL DEFAULT '',
    emergency_phone TEXT NOT NULL DEFAULT '',
    payout_method   TEXT NOT NULL DEFAULT '',
    updated_at      TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS users (
    id              TEXT PRIMARY KEY,
    email           TEXT NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL DEFAULT '',
    name            TEXT NOT NULL DEFAULT '',
    role            TEXT NOT NULL DEFAULT 'staff',  -- 'owner' | 'staff'
    status          TEXT NOT NULL DEFAULT 'active', -- 'active' | 'invited'
    last_login_at   TEXT NOT NULL DEFAULT '',
    avatar_color    TEXT NOT NULL DEFAULT '#5c6ac4',
    created_at      TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

CREATE TABLE IF NOT EXISTS stores (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    slug            TEXT NOT NULL,
    kind            TEXT NOT NULL,                  -- 'client_transfer' | 'collaborator' | 'active_development'
    status          TEXT NOT NULL DEFAULT 'active', -- 'active' | 'archived' | 'inactive'
    last_login_at   TEXT NOT NULL DEFAULT '',
    plan            TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_stores_kind ON stores(kind);
CREATE INDEX IF NOT EXISTS idx_stores_status ON stores(status);

CREATE TABLE IF NOT EXISTS payouts (
    id              TEXT PRIMARY KEY,
    period_start    TEXT NOT NULL,
    period_end      TEXT NOT NULL,
    sent_at         TEXT NOT NULL DEFAULT '',
    amount_cents    INTEGER NOT NULL,
    currency        TEXT NOT NULL DEFAULT 'USD',
    method          TEXT NOT NULL DEFAULT 'bank account (***05)',
    status          TEXT NOT NULL DEFAULT 'paid'    -- 'paid' | 'pending'
);

CREATE TABLE IF NOT EXISTS payout_line_items (
    id              TEXT PRIMARY KEY,
    payout_id       TEXT NOT NULL REFERENCES payouts(id),
    occurred_at     TEXT NOT NULL,
    description     TEXT NOT NULL,
    amount_cents    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pli_payout ON payout_line_items(payout_id);

CREATE TABLE IF NOT EXISTS leads (
    id              TEXT PRIMARY KEY,
    product         TEXT NOT NULL,                  -- 'plus' | 'pos' | 'plus_b2b'
    merchant_name   TEXT NOT NULL,
    contact_email   TEXT NOT NULL DEFAULT '',
    contact_name    TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'submitted', -- 'submitted' | 'qualified' | 'won' | 'lost'
    earnings_cents  INTEGER NOT NULL DEFAULT 0,
    submitted_at    TEXT NOT NULL,
    notes           TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS referrals (
    id              TEXT PRIMARY KEY,
    merchant_name   TEXT NOT NULL,
    plan            TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'active',
    earnings_cents  INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS directory_listings (
    id              TEXT PRIMARY KEY,
    business_name   TEXT NOT NULL,
    plan            TEXT NOT NULL,
    review_status   TEXT NOT NULL DEFAULT 'available' -- 'available' | 'requested' | 'received'
);

CREATE TABLE IF NOT EXISTS tickets (
    id              TEXT PRIMARY KEY,
    subject         TEXT NOT NULL,
    category        TEXT NOT NULL,
    description     TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'open',
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS themes (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'draft',  -- 'draft' | 'in_review' | 'approved' | 'rejected'
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS catalogs (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    products_count  INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'draft',  -- 'draft' | 'published'
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS changelog (
    id              TEXT PRIMARY KEY,
    published_at    TEXT NOT NULL,
    category        TEXT NOT NULL,
    title           TEXT NOT NULL,
    summary         TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS notifications (
    id              TEXT PRIMARY KEY,
    occurred_at     TEXT NOT NULL,
    title           TEXT NOT NULL,
    body            TEXT NOT NULL DEFAULT '',
    read            INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS merchant_orders (
    id              TEXT PRIMARY KEY,
    store_id        TEXT NOT NULL REFERENCES stores(id),
    order_number    TEXT NOT NULL,                  -- '#1042'
    customer_name   TEXT NOT NULL,
    customer_email  TEXT NOT NULL DEFAULT '',
    total_cents     INTEGER NOT NULL,
    currency        TEXT NOT NULL DEFAULT 'USD',
    financial_status TEXT NOT NULL DEFAULT 'paid',  -- 'paid' | 'pending' | 'refunded'
    fulfillment_status TEXT NOT NULL DEFAULT 'unfulfilled', -- 'unfulfilled' | 'fulfilled' | 'partial'
    items_count     INTEGER NOT NULL DEFAULT 1,
    items_json      TEXT NOT NULL DEFAULT '[]',
    ordered_at      TEXT NOT NULL,
    delivery_method TEXT NOT NULL DEFAULT 'Standard'
);
CREATE INDEX IF NOT EXISTS idx_mo_store ON merchant_orders(store_id);

CREATE TABLE IF NOT EXISTS merchant_products (
    id              TEXT PRIMARY KEY,
    store_id        TEXT NOT NULL REFERENCES stores(id),
    title           TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'active', -- 'active' | 'draft' | 'archived'
    inventory       INTEGER NOT NULL DEFAULT 0,
    vendor          TEXT NOT NULL DEFAULT '',
    product_type    TEXT NOT NULL DEFAULT '',
    price_cents     INTEGER NOT NULL DEFAULT 0,
    sku             TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mp_store ON merchant_products(store_id);

CREATE TABLE IF NOT EXISTS merchant_customers (
    id              TEXT PRIMARY KEY,
    store_id        TEXT NOT NULL REFERENCES stores(id),
    name            TEXT NOT NULL,
    email           TEXT NOT NULL DEFAULT '',
    orders_count    INTEGER NOT NULL DEFAULT 0,
    total_spent_cents INTEGER NOT NULL DEFAULT 0,
    location        TEXT NOT NULL DEFAULT '',
    last_order_at   TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mc_store ON merchant_customers(store_id);

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
