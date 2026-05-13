"""Each oracle returns the same verdict twice on identical DB state."""

from __future__ import annotations

import sqlite3

import pytest


@pytest.fixture
def seeded_db():
    from app import db, seed as seed_mod  # noqa: PLC0415

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(db.SCHEMA)
    seed_mod.seed(conn, seed_val=42, fake_now="2026-01-15T09:00:00Z")
    return conn


@pytest.mark.parametrize("task_id", [
    "T01_buy_jacket",
    "T02_refund_line_item",
    "T03_create_coupon",
    "T04_export_bogo_orders",
    "T05_inventory_adjust",
])
def test_oracle_is_deterministic(seeded_db, task_id):
    from app.oracles import grade  # noqa: PLC0415

    r1 = grade(task_id, seeded_db, now="2026-01-15T09:00:00Z", seed_val=42)
    r2 = grade(task_id, seeded_db, now="2026-01-15T09:00:00Z", seed_val=42)
    assert r1 == r2, f"{task_id}: oracle non-deterministic"


def test_oracle_unknown_task(seeded_db):
    from app.oracles import grade  # noqa: PLC0415

    r = grade("DOES_NOT_EXIST", seeded_db,
              now="2026-01-15T09:00:00Z", seed_val=42)
    assert r["passed"] is False
    assert "no oracle" in " ".join(r["reasons"]).lower()


def test_oracle_t01_fails_on_seed(seeded_db):
    """Out of the box, no order_placed for customer_00001 — T01 must fail."""
    from app.oracles import grade  # noqa: PLC0415

    r = grade("T01_buy_jacket", seeded_db,
              now="2026-01-15T09:00:00Z", seed_val=42)
    assert r["passed"] is False


def test_oracle_t02_fails_without_refund(seeded_db):
    from app.oracles import grade  # noqa: PLC0415

    r = grade("T02_refund_line_item", seeded_db,
              now="2026-01-15T09:00:00Z", seed_val=42)
    assert r["passed"] is False


def test_oracle_t03_fails_without_creation(seeded_db):
    """No new coupons created → T03 fails."""
    from app.oracles import grade  # noqa: PLC0415

    r = grade("T03_create_coupon", seeded_db,
              now="2026-01-15T09:00:00Z", seed_val=42)
    assert r["passed"] is False


def test_oracle_t04_fails_when_view_empty(seeded_db):
    """The view starts empty → T04 must fail (targets missing)."""
    from app.oracles import grade  # noqa: PLC0415

    r = grade("T04_export_bogo_orders", seeded_db,
              now="2026-01-15T09:00:00Z", seed_val=42)
    assert r["passed"] is False
    assert r["diff"]["target_count"] >= 1


def test_oracle_t04_passes_when_correct_orders_added(seeded_db):
    """Synthesize the export: insert all target orders into the view."""
    from app.oracles import grade  # noqa: PLC0415
    from app.oracles.t04_export_bogo_orders import _target_order_ids

    targets = sorted(_target_order_ids(seeded_db, now="2026-01-15T09:00:00Z"))
    assert targets, "seed must produce BOGO recent orders"
    for oid in targets:
        seeded_db.execute(
            "INSERT INTO saved_view_members (view_id, order_id) VALUES (?, ?)",
            ("saved_view_bogo_recent", oid),
        )
    seeded_db.commit()

    r = grade("T04_export_bogo_orders", seeded_db,
              now="2026-01-15T09:00:00Z", seed_val=42)
    assert r["passed"], r["reasons"]
    assert r["score"] == 1.0


def test_oracle_t05_fails_on_seed(seeded_db):
    from app.oracles import grade  # noqa: PLC0415

    r = grade("T05_inventory_adjust", seeded_db,
              now="2026-01-15T09:00:00Z", seed_val=42)
    assert r["passed"] is False
