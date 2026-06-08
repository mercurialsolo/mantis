"""SQLite schema + thin query helpers for mantis-fiverr.

Shape mirrors mantis_shop / mantis_crm: in-memory by default, one
module-level connection guarded by a re-entrant lock, schema applied
at first connect.

Schema overview
---------------

* ``users`` — auth surface. ``role`` ∈ ``buyer``, ``seller``.
* ``sellers`` — extended seller-profile attached to a user (level,
  avatar palette, response time, country).
* ``categories`` — top-level + subcategory rows (flat for simplicity).
* ``gigs`` — one row per gig. Three package tiers are stored on the
  same row for now (basic/standard/premium each have price, delivery
  days, revisions, description, features JSON).
* ``orders`` — buyer-placed orders. ``status`` ∈ ``active``,
  ``delivered``, ``completed``, ``cancelled``.
* ``order_items`` — line items (usually 1 per order; future-proof).
* ``conversations`` + ``messages`` — inbox storage.
* ``reviews`` — per-order review row + stars + text.
* ``audit_log`` — append-only mutation record. Single source of truth
  for oracles.
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
    username        TEXT NOT NULL UNIQUE,
    email           TEXT NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL,
    role            TEXT NOT NULL,            -- 'buyer' | 'seller'
    display_name    TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

CREATE TABLE IF NOT EXISTS sellers (
    user_id         TEXT PRIMARY KEY REFERENCES users(id),
    level           TEXT NOT NULL,            -- 'new' | 'level_one' | 'level_two' | 'top_rated'
    country         TEXT NOT NULL,
    languages       TEXT NOT NULL DEFAULT '[]',
    response_time_h INTEGER NOT NULL DEFAULT 24,
    member_since    TEXT NOT NULL,
    avg_rating      REAL NOT NULL DEFAULT 0.0,
    review_count    INTEGER NOT NULL DEFAULT 0,
    avatar_palette  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS categories (
    slug            TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    parent_slug     TEXT,
    icon            TEXT NOT NULL DEFAULT 'square',
    sort_order      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_categories_parent ON categories(parent_slug);

CREATE TABLE IF NOT EXISTS gigs (
    id              TEXT PRIMARY KEY,
    seller_id       TEXT NOT NULL REFERENCES users(id),
    slug            TEXT NOT NULL UNIQUE,
    title           TEXT NOT NULL,
    description_md  TEXT NOT NULL DEFAULT '',
    category_slug   TEXT NOT NULL REFERENCES categories(slug),
    image_palette   INTEGER NOT NULL DEFAULT 0,
    -- package tiers stored inline; cleaner than a 4-row JOIN per page
    pkg_basic_title       TEXT NOT NULL DEFAULT 'Basic',
    pkg_basic_desc        TEXT NOT NULL DEFAULT '',
    pkg_basic_price       REAL NOT NULL DEFAULT 5.0,
    pkg_basic_delivery_d  INTEGER NOT NULL DEFAULT 3,
    pkg_basic_revisions   INTEGER NOT NULL DEFAULT 1,
    pkg_basic_features    TEXT NOT NULL DEFAULT '[]',
    pkg_standard_title    TEXT NOT NULL DEFAULT 'Standard',
    pkg_standard_desc     TEXT NOT NULL DEFAULT '',
    pkg_standard_price    REAL NOT NULL DEFAULT 25.0,
    pkg_standard_delivery_d INTEGER NOT NULL DEFAULT 5,
    pkg_standard_revisions INTEGER NOT NULL DEFAULT 2,
    pkg_standard_features TEXT NOT NULL DEFAULT '[]',
    pkg_premium_title     TEXT NOT NULL DEFAULT 'Premium',
    pkg_premium_desc      TEXT NOT NULL DEFAULT '',
    pkg_premium_price     REAL NOT NULL DEFAULT 75.0,
    pkg_premium_delivery_d INTEGER NOT NULL DEFAULT 7,
    pkg_premium_revisions INTEGER NOT NULL DEFAULT 4,
    pkg_premium_features  TEXT NOT NULL DEFAULT '[]',
    avg_rating      REAL NOT NULL DEFAULT 0.0,
    review_count    INTEGER NOT NULL DEFAULT 0,
    orders_count    INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_gigs_category ON gigs(category_slug);
CREATE INDEX IF NOT EXISTS idx_gigs_seller ON gigs(seller_id);

CREATE TABLE IF NOT EXISTS orders (
    id              TEXT PRIMARY KEY,
    number          TEXT NOT NULL UNIQUE,        -- '#FO0001'
    buyer_id        TEXT NOT NULL REFERENCES users(id),
    seller_id       TEXT NOT NULL REFERENCES users(id),
    gig_id          TEXT NOT NULL REFERENCES gigs(id),
    tier            TEXT NOT NULL,               -- 'basic' | 'standard' | 'premium'
    subtotal        REAL NOT NULL,
    service_fee     REAL NOT NULL DEFAULT 0.0,
    total           REAL NOT NULL,
    status          TEXT NOT NULL,               -- 'active' | 'delivered' | 'completed' | 'cancelled'
    requirements    TEXT NOT NULL DEFAULT '',
    placed_at       TEXT NOT NULL,
    due_at          TEXT NOT NULL,
    delivered_at    TEXT,
    completed_at    TEXT,
    cancelled_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_orders_buyer ON orders(buyer_id);
CREATE INDEX IF NOT EXISTS idx_orders_seller ON orders(seller_id);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);

CREATE TABLE IF NOT EXISTS order_items (
    id              TEXT PRIMARY KEY,
    order_id        TEXT NOT NULL REFERENCES orders(id),
    line_no         INTEGER NOT NULL,
    description     TEXT NOT NULL,
    unit_price      REAL NOT NULL,
    quantity        INTEGER NOT NULL DEFAULT 1,
    UNIQUE (order_id, line_no)
);
CREATE INDEX IF NOT EXISTS idx_order_items_order ON order_items(order_id);

CREATE TABLE IF NOT EXISTS conversations (
    id              TEXT PRIMARY KEY,
    buyer_id        TEXT NOT NULL REFERENCES users(id),
    seller_id       TEXT NOT NULL REFERENCES users(id),
    last_msg_at     TEXT NOT NULL,
    UNIQUE (buyer_id, seller_id)
);
CREATE INDEX IF NOT EXISTS idx_conv_buyer ON conversations(buyer_id);
CREATE INDEX IF NOT EXISTS idx_conv_seller ON conversations(seller_id);

CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    sender_id       TEXT NOT NULL REFERENCES users(id),
    body            TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_msgs_conv ON messages(conversation_id);

CREATE TABLE IF NOT EXISTS reviews (
    id              TEXT PRIMARY KEY,
    order_id        TEXT NOT NULL UNIQUE REFERENCES orders(id),
    gig_id          TEXT NOT NULL REFERENCES gigs(id),
    buyer_id        TEXT NOT NULL REFERENCES users(id),
    seller_id       TEXT NOT NULL REFERENCES users(id),
    stars           INTEGER NOT NULL,
    body            TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reviews_gig ON reviews(gig_id);

CREATE TABLE IF NOT EXISTS favorites (
    user_id         TEXT NOT NULL REFERENCES users(id),
    gig_id          TEXT NOT NULL REFERENCES gigs(id),
    created_at      TEXT NOT NULL,
    PRIMARY KEY (user_id, gig_id)
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
    """Drop the in-process connection — used by ``/__env__/reset``."""
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
    return json.dumps(value, sort_keys=False)


def unpack_json(raw: str) -> Any:
    try:
        return json.loads(raw or "null")
    except json.JSONDecodeError:
        return None


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
        (occurred_at, operation, target_type, target_id,
         json.dumps(payload or {}, sort_keys=True)),
    )
