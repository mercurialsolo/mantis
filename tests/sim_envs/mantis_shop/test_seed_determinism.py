"""Seed determinism — same SEED → identical row IDs + counts.

Same shape as the mantis-crm tests. Pins the seeded landmarks the five
oracles depend on (Brooklyn address, JACKET-001-M-BLUE variant,
TEE-BLK-M variant, SPRING15/BOGO/STACK_TRAP_*, order_04421).
"""

from __future__ import annotations

import sqlite3

import pytest


@pytest.fixture
def fresh_db():
    def _build(seed_val: int = 42) -> sqlite3.Connection:
        from app import db, seed as seed_mod  # noqa: PLC0415

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(db.SCHEMA)
        seed_mod.seed(conn, seed_val=seed_val,
                      fake_now="2026-01-15T09:00:00Z")
        return conn

    return _build


def _snapshot(conn: sqlite3.Connection, table: str, order_col: str = "id",
              *, limit: int = 50) -> list[tuple]:
    return [tuple(r) for r in conn.execute(
        f"SELECT * FROM {table} ORDER BY {order_col} LIMIT ?", (limit,)
    ).fetchall()]


def _count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def test_row_counts_match_spec(fresh_db):
    conn = fresh_db(42)
    assert _count(conn, "customers") == 3_000
    assert _count(conn, "products") == 2_000
    assert _count(conn, "orders") == 5_000
    assert _count(conn, "coupons") == 15
    # Variants vary by per-product matrix; sanity-check the range.
    n_variants = _count(conn, "variants")
    assert 5_000 < n_variants < 30_000
    # Saved views: 1 pinned view (T04 target).
    assert _count(conn, "saved_views") == 1


def test_same_seed_same_snapshot(fresh_db):
    a = fresh_db(42)
    b = fresh_db(42)
    cases = [
        ("customers", "id"),
        ("products", "id"),
        ("variants", "id"),
        ("coupons", "code"),
        ("orders", "id"),
    ]
    for table, order_col in cases:
        assert _snapshot(a, table, order_col) == _snapshot(b, table, order_col), (
            f"snapshot mismatch in {table}"
        )


def test_different_seeds_differ(fresh_db):
    a = fresh_db(42)
    b = fresh_db(43)
    # Same row count, different content.
    sa = _snapshot(a, "products", limit=20)
    sb = _snapshot(b, "products", limit=20)
    assert len(sa) == len(sb)
    assert sa != sb


def test_brooklyn_address_pinned(fresh_db):
    """T01 ships to addr_brooklyn_001 on customer_00001."""
    conn = fresh_db(42)
    row = conn.execute(
        "SELECT * FROM customer_addresses WHERE id = 'addr_brooklyn_001'"
    ).fetchone()
    assert row is not None
    assert row["customer_id"] == "customer_00001"
    assert row["city"] == "Brooklyn"
    assert row["region"] == "NY"
    assert row["zip"] == "11201"
    assert row["is_default"] == 1


def test_jacket_m_blue_in_stock_and_unlocked(fresh_db):
    """T01 needs JACKET-001-M-BLUE to be in stock and ship to Brooklyn (NY)."""
    conn = fresh_db(42)
    variant = conn.execute(
        "SELECT v.*, i.quantity FROM variants v "
        "LEFT JOIN inventory i ON v.id = i.variant_id "
        "WHERE v.sku = 'JACKET-001-M-BLUE'"
    ).fetchone()
    assert variant is not None
    assert variant["product_id"] == "product_00001"
    assert variant["size"] == "M"
    assert variant["color"] == "BLUE"
    assert variant["quantity"] == 25  # pinned in seed
    locks = conn.execute(
        "SELECT region FROM variant_region_locks WHERE variant_id = ?",
        (variant["id"],),
    ).fetchall()
    assert locks == [], "M-BLUE jacket must not be region-locked"


def test_jacket_under_100(fresh_db):
    conn = fresh_db(42)
    p = conn.execute(
        "SELECT base_price, sale_price, category FROM products WHERE id = 'product_00001'"
    ).fetchone()
    assert p["base_price"] < 100
    assert p["category"] == "outerwear-women"


def test_tee_blk_m_pinned(fresh_db):
    """T05 needs TEE-BLK-M with exactly 100 starting inventory."""
    conn = fresh_db(42)
    row = conn.execute(
        "SELECT v.id, i.quantity FROM variants v "
        "LEFT JOIN inventory i ON v.id = i.variant_id "
        "WHERE v.sku = 'TEE-BLK-M'"
    ).fetchone()
    assert row is not None
    assert row["quantity"] == 100


def test_order_4421_has_3_line_items_paid(fresh_db):
    """T02 refunds line 2 of order_04421 — that line must exist + status paid."""
    conn = fresh_db(42)
    order = conn.execute(
        "SELECT * FROM orders WHERE id = 'order_04421'"
    ).fetchone()
    assert order is not None
    assert order["status"] == "paid"
    items = conn.execute(
        "SELECT line_no FROM order_items WHERE order_id = 'order_04421' "
        "ORDER BY line_no"
    ).fetchall()
    line_nos = [r["line_no"] for r in items]
    assert line_nos == [1, 2, 3]


def test_spring15_and_bogo_present(fresh_db):
    conn = fresh_db(42)
    codes = {r["code"] for r in conn.execute("SELECT code FROM coupons").fetchall()}
    assert {"SPRING15", "BOGO", "STACK_TRAP_A", "STACK_TRAP_B"} <= codes


def test_stack_trap_pair_marked_stackable_but_exclude_each_other(fresh_db):
    """The deliberate UI-lies trap: both stackable=1 but exclude each other."""
    conn = fresh_db(42)
    a = conn.execute(
        "SELECT * FROM coupons WHERE code = 'STACK_TRAP_A'"
    ).fetchone()
    b = conn.execute(
        "SELECT * FROM coupons WHERE code = 'STACK_TRAP_B'"
    ).fetchone()
    assert a["stackable"] == 1 and b["stackable"] == 1
    import json
    assert "STACK_TRAP_B" in json.loads(a["stacking_exclusions"])
    assert "STACK_TRAP_A" in json.loads(b["stacking_exclusions"])


def test_bogo_recent_orders_pinned(fresh_db):
    """T04 — the seeded BOGO orders fall within the last 7 days."""
    from datetime import timedelta

    from app.seed import _parse_iso  # noqa: PLC0415

    conn = fresh_db(42)
    now = _parse_iso("2026-01-15T09:00:00Z")
    cutoff = (now - timedelta(days=7)).isoformat()
    rows = conn.execute(
        "SELECT id, placed_at, coupon_codes FROM orders "
        "WHERE id LIKE 'order_048%' AND placed_at >= ?",
        (cutoff,),
    ).fetchall()
    bogo_ids = {r["id"] for r in rows if "BOGO" in (r["coupon_codes"] or "")}
    assert {"order_04801", "order_04802", "order_04803",
            "order_04804", "order_04805"} <= bogo_ids


def test_saved_view_bogo_recent_exists_empty(fresh_db):
    """T04 expects the view to exist and start empty (agent populates it)."""
    conn = fresh_db(42)
    view = conn.execute(
        "SELECT * FROM saved_views WHERE id = 'saved_view_bogo_recent'"
    ).fetchone()
    assert view is not None
    members = conn.execute(
        "SELECT COUNT(*) AS n FROM saved_view_members "
        "WHERE view_id = 'saved_view_bogo_recent'"
    ).fetchone()
    assert members["n"] == 0
