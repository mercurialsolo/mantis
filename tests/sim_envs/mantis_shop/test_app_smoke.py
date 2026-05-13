"""End-to-end smoke via httpx TestClient.

Covers:

* Boot — startup hook seeds the DB; counts match the spec.
* Harness gating — ``/__env__/*`` 401s without ``X-Env-Admin``.
* Health is open.
* The major storefront + admin surfaces render 200.
* T05 oracle round-trip: adjust TEE-BLK-M by +50 → oracle passes.
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


def _admin_headers():
    return {"X-Env-Admin": "test-admin-token"}


# ── harness gating ────────────────────────────────────────────────────


@pytest.mark.parametrize("path", [
    "/__env__/state",
    "/__env__/events",
    "/__env__/oracle?task_id=T01_buy_jacket",
])
def test_admin_routes_401_without_token(client, path):
    r = client.get(path)
    assert r.status_code == 401


def test_health_is_open(client):
    r = client.get("/__env__/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["seed"] == 42


def test_admin_oracle_200_with_token(client):
    r = client.get("/__env__/oracle?task_id=T01_buy_jacket",
                   headers=_admin_headers())
    assert r.status_code == 200
    body = r.json()
    assert body["task_id"] == "T01_buy_jacket"
    assert "passed" in body and "score" in body


# ── agent-facing surfaces ─────────────────────────────────────────────


def test_root_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "mantis-shop" in r.text


def test_catalog_renders(client):
    r = client.get("/catalog")
    assert r.status_code == 200
    assert "Heritage Field Jacket" in r.text or "Catalog" in r.text


def test_product_detail_renders(client):
    r = client.get("/products/product_00001")
    assert r.status_code == 200
    assert "Heritage Field Jacket" in r.text
    assert "JACKET-001-M-BLUE" in r.text


def test_cart_renders(client):
    r = client.get("/cart")
    assert r.status_code == 200
    # First load mints a session + cart; cart is empty.
    assert "Your cart" in r.text


def test_admin_orders_renders(client):
    r = client.get("/admin/orders")
    assert r.status_code == 200
    assert "order_" in r.text


def test_admin_order_detail_renders(client):
    r = client.get("/admin/orders/order_04421")
    assert r.status_code == 200
    assert "order_04421" in r.text


def test_admin_products_renders(client):
    r = client.get("/admin/products")
    assert r.status_code == 200
    assert "Heritage Field Jacket" in r.text


def test_admin_coupons_renders(client):
    r = client.get("/admin/coupons")
    assert r.status_code == 200
    assert "SPRING15" in r.text


def test_admin_saved_view_renders(client):
    r = client.get("/admin/saved_views/saved_view_bogo_recent")
    assert r.status_code == 200
    assert "BOGO" in r.text


# ── T05 round-trip ─────────────────────────────────────────────────────


def test_t05_oracle_passes_after_adjust(client):
    """Adjust TEE-BLK-M inventory + verify the oracle approves."""
    r = client.post(
        "/admin/products/inventory/adjust_by_sku",
        data={"sku": "TEE-BLK-M", "delta": "50",
              "reason": "restock from warehouse"},
        follow_redirects=False,
    )
    assert r.status_code in (200, 303)

    verdict = client.get("/__env__/oracle?task_id=T05_inventory_adjust",
                         headers=_admin_headers()).json()
    assert verdict["passed"], verdict["reasons"]
    assert verdict["score"] == 1.0


# ── reset round-trip ──────────────────────────────────────────────────


def test_reset_clears_audit(client):
    # First, mutate something so audit_log gains a row.
    client.post(
        "/admin/products/inventory/adjust_by_sku",
        data={"sku": "TEE-BLK-M", "delta": "5", "reason": "test"},
        follow_redirects=False,
    )
    state = client.get("/__env__/state", headers=_admin_headers()).json()
    assert state["counts"]["audit_log"] >= 1

    r = client.post("/__env__/reset", headers=_admin_headers())
    assert r.status_code == 200

    state = client.get("/__env__/state", headers=_admin_headers()).json()
    assert state["counts"]["audit_log"] == 0
