"""Server-side enforcement: variant constraints + coupon stacking.

These are the failure modes the spec deliberately plants. The agent
must navigate around them; an env that lets the agent bypass them via
JS-only or client-only checks would let bad attempts pass undetected.
"""

from __future__ import annotations

import pytest


pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("jinja2")
pytest.importorskip("multipart")


@pytest.fixture
def client():
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from app.main import app  # noqa: PLC0415

    with TestClient(app) as c:
        yield c


# ── variant constraint enforcement ─────────────────────────────────────


def test_add_to_cart_rejects_out_of_stock_variant(client):
    """Find any 0-stock variant in the seed and confirm the add-to-cart
    handler bounces the agent with an error message."""
    from app import db as env_db  # noqa: PLC0415

    conn = env_db.connect()
    row = conn.execute(
        "SELECT v.id, v.product_id FROM variants v "
        "JOIN inventory i ON v.id = i.variant_id WHERE i.quantity = 0 LIMIT 1"
    ).fetchone()
    assert row is not None, "seed should produce at least one OOS variant"

    r = client.post(
        "/cart/add",
        data={"variant_id": row["id"], "quantity": "1"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    location = r.headers["location"]
    assert location.startswith(f"/products/{row['product_id']}")
    assert "out%20of%20stock" in location or "out of stock" in location

    # Cart did NOT receive the variant.
    cart_count = conn.execute(
        "SELECT COUNT(*) FROM cart_items WHERE variant_id = ?",
        (row["id"],),
    ).fetchone()[0]
    assert cart_count == 0


def test_checkout_rejects_region_locked_variant(client):
    """When a region-locked variant is in the cart and the agent picks an
    address in the locked region, the address step must re-render with
    an error and NOT advance to shipping."""
    from app import db as env_db  # noqa: PLC0415

    conn = env_db.connect()
    locked = conn.execute(
        "SELECT v.id, v.product_id, vrl.region "
        "FROM variant_region_locks vrl "
        "JOIN variants v ON v.id = vrl.variant_id "
        "JOIN inventory i ON i.variant_id = v.id "
        "WHERE i.quantity > 0 LIMIT 1"
    ).fetchone()
    assert locked is not None, "seed should produce at least one region-locked variant"

    locked_region = locked["region"]

    # Add the locked variant first.
    r = client.post(
        "/cart/add",
        data={"variant_id": locked["id"], "quantity": "1"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/cart"

    # Submit a free-form address in the locked region.
    r = client.post(
        "/checkout/address",
        data={
            "saved_address_id": "",
            "line1": "1 Test St",
            "city": "Testville",
            "region": locked_region,
            "zip": "12345",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    location = r.headers["location"]
    assert location.startswith("/checkout/address")
    assert "error" in location.lower()
    assert f"ship%20to%20{locked_region}" in location or f"ship to {locked_region}" in location

    # Cart's shipping_address_id is still NULL — the step was rejected.
    cart_row = conn.execute(
        "SELECT shipping_address_id FROM carts ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert cart_row["shipping_address_id"] is None


def test_address_rejects_invalid_zip(client):
    """ZIP validation must run server-side; bad ZIP re-renders the
    address step instead of advancing to shipping."""
    r = client.post(
        "/checkout/address",
        data={
            "saved_address_id": "",
            "line1": "1 Test St",
            "city": "Test City",
            "region": "NY",
            "zip": "not-a-zip",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    location = r.headers["location"]
    assert location.startswith("/checkout/address")
    assert "invalid%20zip" in location.lower() or "invalid zip" in location.lower()


# ── coupon stacking enforcement ────────────────────────────────────────


def test_stack_trap_pair_rejected_together(client):
    """The deliberate UI-lies pair STACK_TRAP_A + STACK_TRAP_B both show
    stackable=1 but the server rejects them together via
    ``stacking_exclusions``."""
    # First coupon applies cleanly.
    r = client.post(
        "/cart/apply-coupon",
        data={"code": "STACK_TRAP_A"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/cart"

    # Second coupon is rejected.
    r = client.post(
        "/cart/apply-coupon",
        data={"code": "STACK_TRAP_B"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    location = r.headers["location"]
    assert location.startswith("/cart")
    assert "exclusion" in location.lower() or "stacking" in location.lower()

    # Cart state: only STACK_TRAP_A is applied.
    from app import db as env_db  # noqa: PLC0415
    conn = env_db.connect()
    cart = conn.execute(
        "SELECT coupon_codes FROM carts ORDER BY id DESC LIMIT 1"
    ).fetchone()
    import json
    applied = json.loads(cart["coupon_codes"])
    assert "STACK_TRAP_A" in applied
    assert "STACK_TRAP_B" not in applied


def test_non_stackable_blocks_a_second_coupon(client):
    """Applying a non-stackable coupon (VIPONLY) then any second coupon
    must reject."""
    r = client.post(
        "/cart/apply-coupon",
        data={"code": "VIPONLY"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/cart"

    r = client.post(
        "/cart/apply-coupon",
        data={"code": "SPRING15"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert "not%20stackable" in r.headers["location"] or "not stackable" in r.headers["location"]


def test_unknown_coupon_rejected(client):
    r = client.post(
        "/cart/apply-coupon",
        data={"code": "NOSUCHCODE"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    location = r.headers["location"]
    assert "unknown" in location.lower()


def test_disabled_coupon_rejected(client):
    """The seed plants OLDSALE with ``disabled_at`` set — must reject."""
    r = client.post(
        "/cart/apply-coupon",
        data={"code": "OLDSALE"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert "disabled" in r.headers["location"].lower()


def test_expired_coupon_rejected(client):
    """The seed plants BLACKFRIDAY with expires_at in the past."""
    r = client.post(
        "/cart/apply-coupon",
        data={"code": "BLACKFRIDAY"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert "expired" in r.headers["location"].lower()


# ── T01 round-trip via the storefront forms ────────────────────────────


def test_t01_round_trip(client):
    """Drive the full T01 buy-jacket flow through the agent-facing forms
    and confirm the oracle approves."""
    # Add the M-BLUE jacket to cart.
    r = client.post(
        "/cart/add",
        data={"variant_id": "variant_00001_M_BLUE", "quantity": "1"},
        follow_redirects=False,
    )
    assert r.status_code == 303 and r.headers["location"] == "/cart"

    # Apply SPRING15.
    r = client.post(
        "/cart/apply-coupon",
        data={"code": "SPRING15"},
        follow_redirects=False,
    )
    assert r.status_code == 303 and r.headers["location"] == "/cart"

    # Address: pick Brooklyn.
    r = client.post(
        "/checkout/address",
        data={"saved_address_id": "addr_brooklyn_001"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/checkout/shipping"

    # Shipping: standard.
    r = client.post(
        "/checkout/shipping",
        data={"method": "standard"},
        follow_redirects=False,
    )
    assert r.status_code == 303

    # Payment: pay_00001.
    r = client.post(
        "/checkout/payment",
        data={"payment_method_id": "pay_00001"},
        follow_redirects=False,
    )
    assert r.status_code == 303

    # Place order.
    r = client.post(
        "/checkout/place_order",
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"].startswith("/orders/confirm/")

    verdict = client.get(
        "/__env__/oracle?task_id=T01_buy_jacket",
        headers={"X-Env-Admin": "test-admin-token"},
    ).json()
    assert verdict["passed"], verdict["reasons"]
    assert verdict["score"] == 1.0
