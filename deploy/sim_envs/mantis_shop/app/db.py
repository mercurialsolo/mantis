"""SQLite schema + thin query helpers for mantis-shop.

Mirrors the shape of ``mantis-crm/app/db.py``: in-memory by default, a
single module-level connection guarded by a re-entrant lock, schema
applied at first connect.

Schema highlights
-----------------

* ``products`` — base SKU. Variants live in ``variants`` keyed by
  ``product_id``; per-variant inventory in ``inventory``; per-variant
  region-locks (won't ship to ZIPs in ``region_lock``) in
  ``variant_region_locks``.
* ``orders`` — paid/fulfilled/refunded/cancelled. Line items in
  ``order_items``; refunds in ``order_refunds``.
* ``carts`` — one row per session cookie; ``cart_items`` holds the
  per-variant qty. The cart load on every page render is what the
  spec calls "cart persistence after nav-away".
* ``coupons`` — pct off / $ off / BOGO. ``stackable`` is the *claimed*
  stackable flag (UI shows it), ``stacking_exclusions`` is the JSON
  list of coupon codes the server actually rejects together. That
  divergence is the deliberate "looks stackable but server rejects"
  trap described in the spec.
* ``audit_log`` — append-only mutation record (same shape as crm's
  ``mutations``); oracles read this + DB state, never the agent's
  transcript.
* ``saved_views`` — admin "saved searches" (T04 export target).

All IDs are deterministic strings: ``product_00001``, ``order_04421``,
``variant_p00012_M_BLUE``, etc. Same SEED → identical IDs + identical
column values.
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
CREATE TABLE IF NOT EXISTS customers (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    email           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS customer_addresses (
    id              TEXT PRIMARY KEY,
    customer_id     TEXT NOT NULL REFERENCES customers(id),
    line1           TEXT NOT NULL,
    city            TEXT NOT NULL,
    region          TEXT NOT NULL,    -- 'NY', 'CA', etc.
    zip             TEXT NOT NULL,
    is_default      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_addr_customer ON customer_addresses(customer_id);

CREATE TABLE IF NOT EXISTS customer_payment_methods (
    -- Mocked. Stored as a vendor-neutral id string; no real PAN/PII.
    id              TEXT PRIMARY KEY,
    customer_id     TEXT NOT NULL REFERENCES customers(id),
    brand           TEXT NOT NULL,   -- 'visa' | 'mc' | 'amex'
    last4           TEXT NOT NULL,
    label           TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS collections (
    id              TEXT PRIMARY KEY,
    slug            TEXT NOT NULL UNIQUE,
    title           TEXT NOT NULL,
    kind            TEXT NOT NULL,   -- 'rule' | 'manual'
    rule_json       TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS products (
    id              TEXT PRIMARY KEY,
    sku             TEXT NOT NULL UNIQUE,
    title           TEXT NOT NULL,
    description_md  TEXT NOT NULL DEFAULT '',
    category        TEXT NOT NULL,
    base_price      REAL NOT NULL,
    sale_price      REAL,            -- NULL when not on sale
    image_url       TEXT,            -- may be NULL/broken (10 intentionally)
    rating          REAL NOT NULL DEFAULT 4.0
);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
CREATE INDEX IF NOT EXISTS idx_products_sku ON products(sku);

CREATE TABLE IF NOT EXISTS variants (
    id              TEXT PRIMARY KEY,        -- 'variant_p00012_M_BLUE'
    product_id      TEXT NOT NULL REFERENCES products(id),
    sku             TEXT NOT NULL UNIQUE,    -- 'TEE-BLK-M'
    size            TEXT NOT NULL,
    color           TEXT NOT NULL,
    price_override  REAL                     -- usually NULL; rarely a per-variant price
);
CREATE INDEX IF NOT EXISTS idx_variants_product ON variants(product_id);

CREATE TABLE IF NOT EXISTS inventory (
    variant_id      TEXT PRIMARY KEY REFERENCES variants(id),
    quantity        INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS variant_region_locks (
    -- A variant present here cannot ship to addresses whose ``region``
    -- column matches. Empty table for the common case; ~3% of variants
    -- have at least one row here per seed messiness.
    variant_id      TEXT NOT NULL REFERENCES variants(id),
    region          TEXT NOT NULL,
    PRIMARY KEY (variant_id, region)
);

CREATE TABLE IF NOT EXISTS collection_members (
    collection_id   TEXT NOT NULL REFERENCES collections(id),
    product_id      TEXT NOT NULL REFERENCES products(id),
    PRIMARY KEY (collection_id, product_id)
);

CREATE TABLE IF NOT EXISTS coupons (
    code            TEXT PRIMARY KEY,
    kind            TEXT NOT NULL,   -- 'pct' | 'amount' | 'bogo'
    value           REAL NOT NULL,   -- 15 (pct) | 10.00 ($) | 1.0 (bogo flag)
    scope_category  TEXT,            -- optional: limit to one category
    stackable       INTEGER NOT NULL DEFAULT 0,    -- UI claim
    stacking_exclusions TEXT NOT NULL DEFAULT '[]', -- JSON list of codes the server rejects together
    expires_at      TEXT,
    max_uses        INTEGER,
    uses_count      INTEGER NOT NULL DEFAULT 0,
    disabled_at     TEXT
);

CREATE TABLE IF NOT EXISTS carts (
    -- One cart per session. ``session_id`` is set on a cookie at first
    -- page render. Persists across nav.
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL UNIQUE,
    coupon_codes    TEXT NOT NULL DEFAULT '[]',  -- applied codes (JSON)
    shipping_address_id TEXT,
    shipping_method TEXT,                         -- 'standard' | 'express' | 'overnight'
    payment_method_id TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cart_items (
    cart_id         TEXT NOT NULL REFERENCES carts(id),
    variant_id      TEXT NOT NULL REFERENCES variants(id),
    quantity        INTEGER NOT NULL,
    PRIMARY KEY (cart_id, variant_id)
);

CREATE TABLE IF NOT EXISTS orders (
    id              TEXT PRIMARY KEY,      -- 'order_04421'
    number          TEXT NOT NULL UNIQUE,  -- '#4421' — what the UI shows
    customer_id     TEXT REFERENCES customers(id),
    shipping_address_id TEXT REFERENCES customer_addresses(id),
    payment_method_id TEXT,                 -- mocked id; no real PAN
    status          TEXT NOT NULL,          -- 'paid' | 'fulfilled' | 'refunded' | 'cancelled'
    subtotal        REAL NOT NULL,
    discount_total  REAL NOT NULL DEFAULT 0,
    shipping_total  REAL NOT NULL DEFAULT 0,
    total           REAL NOT NULL,
    coupon_codes    TEXT NOT NULL DEFAULT '[]',  -- JSON
    notify_customer INTEGER NOT NULL DEFAULT 0,
    placed_at       TEXT NOT NULL,
    fulfilled_at    TEXT,
    cancelled_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_placed_at ON orders(placed_at);
CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);

CREATE TABLE IF NOT EXISTS order_items (
    -- Line items are 1-indexed *within an order* on the customer-visible
    -- surface (the UI says "line 2"). Internally we still use a synthetic
    -- PK + ``line_no`` for stable references.
    id              TEXT PRIMARY KEY,
    order_id        TEXT NOT NULL REFERENCES orders(id),
    line_no         INTEGER NOT NULL,
    variant_id      TEXT NOT NULL REFERENCES variants(id),
    sku             TEXT NOT NULL,
    title           TEXT NOT NULL,
    unit_price      REAL NOT NULL,
    quantity        INTEGER NOT NULL,
    UNIQUE (order_id, line_no)
);
CREATE INDEX IF NOT EXISTS idx_order_items_order ON order_items(order_id);

CREATE TABLE IF NOT EXISTS order_refunds (
    id              TEXT PRIMARY KEY,
    order_id        TEXT NOT NULL REFERENCES orders(id),
    line_no         INTEGER,        -- NULL for full-order refund
    amount          REAL NOT NULL,
    reason          TEXT NOT NULL DEFAULT '',
    notify_customer INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_refunds_order ON order_refunds(order_id);

CREATE TABLE IF NOT EXISTS saved_views (
    -- Admin "saved search" surface. T04 exports into the seeded
    -- ``saved_view_bogo_recent`` row.
    id              TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    filter_json     TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS saved_view_members (
    view_id         TEXT NOT NULL REFERENCES saved_views(id),
    order_id        TEXT NOT NULL REFERENCES orders(id),
    PRIMARY KEY (view_id, order_id)
);

CREATE TABLE IF NOT EXISTS audit_log (
    -- Append-only mutation record. The oracle reads this + the DB
    -- snapshot to grade runs. Mirrors mantis-crm's ``mutations`` table.
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
    """Acquire the lock + start a transaction; commit on success."""
    conn = connect()
    with _DB_LOCK:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


# ── JSON helpers ──────────────────────────────────────────────────────


def pack_codes(codes: list[str]) -> str:
    return json.dumps(sorted(set(codes)))


def unpack_codes(raw: str) -> list[str]:
    try:
        out = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    return list(out) if isinstance(out, list) else []


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
