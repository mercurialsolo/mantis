"""Cart: persistent per-session line items + coupon application.

One cart per ``session_id`` cookie. Cart loads on every page render so
agent nav-aways recover automatically.

Variant-constraint and coupon-stacking enforcement live here — both are
server-side, both fail the agent's request with a re-rendered page and
an error message (no JS-only validation).
"""

from __future__ import annotations

import secrets
import time
from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import db, main as app_main

router = APIRouter()

SESSION_COOKIE = "shop_session"


def _now_iso() -> str:
    return app_main.now_value()


def _log_audit(conn, *, operation: str, target_type: str, target_id: str,
               payload: dict[str, Any] | None = None) -> None:
    db.log_audit(
        conn,
        occurred_at=_now_iso(),
        operation=operation,
        target_type=target_type,
        target_id=target_id,
        payload=payload or {},
    )
    app_main.emit(f"mutation.{operation}", {
        "target_type": target_type, "target_id": target_id,
        **(payload or {}),
    })


def _get_or_create_cart(conn, session_id: str) -> dict[str, Any]:
    row = conn.execute(
        "SELECT * FROM carts WHERE session_id = ?", (session_id,),
    ).fetchone()
    if row is not None:
        return dict(row)
    cart_id = f"cart_{int(time.time() * 1000)}_{secrets.token_hex(3)}"
    conn.execute(
        "INSERT INTO carts (id, session_id, created_at) VALUES (?, ?, ?)",
        (cart_id, session_id, _now_iso()),
    )
    conn.commit()
    return {
        "id": cart_id, "session_id": session_id, "coupon_codes": "[]",
        "shipping_address_id": None, "shipping_method": None,
        "payment_method_id": None, "created_at": _now_iso(),
    }


def ensure_session(request: Request) -> str:
    """Read or mint the session cookie. The cart is keyed on this."""
    sid = request.cookies.get(SESSION_COOKIE)
    if sid:
        return sid
    return f"sess_{secrets.token_hex(8)}"


def load_cart_items(conn, cart_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT ci.cart_id, ci.variant_id, ci.quantity, v.sku, v.size, v.color, "
        "       p.id AS product_id, p.title, "
        "       COALESCE(p.sale_price, p.base_price) AS unit_price, "
        "       p.image_url "
        "FROM cart_items ci "
        "JOIN variants v ON ci.variant_id = v.id "
        "JOIN products p ON v.product_id = p.id "
        "WHERE ci.cart_id = ? ORDER BY ci.variant_id",
        (cart_id,),
    ).fetchall()
    return [dict(r) for r in rows]


# ── coupon evaluation ─────────────────────────────────────────────────


def _coupon_row(conn, code: str) -> dict[str, Any] | None:
    r = conn.execute("SELECT * FROM coupons WHERE code = ?", (code,)).fetchone()
    return dict(r) if r else None


def _coupon_active(coupon: dict[str, Any], now: str) -> tuple[bool, str]:
    """Return (active, reason). Used by both apply_coupon and oracle helpers."""
    if coupon["disabled_at"]:
        return False, "coupon is disabled"
    if coupon["expires_at"] and coupon["expires_at"] < now:
        return False, "coupon has expired"
    if (coupon["max_uses"] is not None
            and coupon["uses_count"] >= coupon["max_uses"]):
        return False, "coupon has reached its max-uses cap"
    return True, ""


def _check_stacking(conn, applied: list[str], new_code: str) -> str:
    """Return an error string if applying ``new_code`` on top of ``applied``
    violates server-side stacking rules. Empty string ⇒ ok.

    Rules:

    1. A non-stackable coupon cannot coexist with anything else.
    2. Two stackable coupons that name each other in
       ``stacking_exclusions`` are rejected (the deliberate UI-lies trap).
    """
    new_row = _coupon_row(conn, new_code)
    if new_row is None:
        return f"unknown coupon code: {new_code}"

    for existing in applied:
        if existing == new_code:
            return "coupon already applied"
        existing_row = _coupon_row(conn, existing)
        if existing_row is None:
            continue
        if not new_row["stackable"] or not existing_row["stackable"]:
            return (
                f"{new_code!r} cannot be combined with {existing!r} "
                "(one or both are not stackable)"
            )
        ex_excl = db.unpack_codes(new_row["stacking_exclusions"] or "[]")
        if existing in ex_excl:
            return (
                f"{new_code!r} cannot be combined with {existing!r} "
                "(stacking exclusion)"
            )
        their_excl = db.unpack_codes(existing_row["stacking_exclusions"] or "[]")
        if new_code in their_excl:
            return (
                f"{new_code!r} cannot be combined with {existing!r} "
                "(stacking exclusion)"
            )
    return ""


def compute_totals(items: list[dict[str, Any]],
                   coupon_codes: list[str],
                   coupons: list[dict[str, Any]]) -> dict[str, float]:
    """Cart subtotal + discount math. Shared by cart view + checkout review."""
    subtotal = sum(it["unit_price"] * it["quantity"] for it in items)
    discount = 0.0
    code_to_row = {c["code"]: c for c in coupons}
    for code in coupon_codes:
        row = code_to_row.get(code)
        if row is None:
            continue
        kind = row["kind"]
        if kind == "pct":
            discount += subtotal * (row["value"] / 100.0)
        elif kind == "amount":
            discount += float(row["value"])
        elif kind == "bogo":
            if items:
                # cheapest line free (one unit)
                cheapest = min(items, key=lambda it: it["unit_price"])
                discount += cheapest["unit_price"]
    shipping = 8.00 if (subtotal > 0 and subtotal < 75) else 0.0
    total = max(0.0, subtotal - discount + shipping)
    return {
        "subtotal": round(subtotal, 2),
        "discount": round(discount, 2),
        "shipping": round(shipping, 2),
        "total": round(total, 2),
    }


# ── cart view ─────────────────────────────────────────────────────────


@router.get("/cart", response_class=HTMLResponse)
async def view_cart(request: Request) -> HTMLResponse:
    """Render the cart. Mints a session cookie if missing — that's
    what gives us the 'cart persists across nav' behaviour."""
    sid = ensure_session(request)
    conn = db.connect()
    cart = _get_or_create_cart(conn, sid)
    items = load_cart_items(conn, cart["id"])
    coupon_codes = db.unpack_codes(cart["coupon_codes"])
    coupon_rows = [
        _coupon_row(conn, c) for c in coupon_codes
        if _coupon_row(conn, c) is not None
    ]
    totals = compute_totals(items, coupon_codes, coupon_rows)
    error = request.query_params.get("error") or ""

    resp = app_main.app.state.templates.TemplateResponse(
        "cart.html",
        {
            "request": request,
            "cart": cart,
            "items": items,
            "coupons": coupon_codes,
            "totals": totals,
            "error": error,
        },
    )
    resp.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax")
    return resp


# ── add/update/remove ─────────────────────────────────────────────────


@router.post("/cart/add")
async def add_to_cart(
    request: Request,
    variant_id: str = Form(...),
    quantity: int = Form(1),
) -> RedirectResponse:
    """Add a variant to the cart. Server-side checks:

    * variant exists
    * variant has ``quantity`` units of stock
    * if the cart already has a shipping_address_id set, the variant is
      not region-locked against that address's region.
    """
    sid = ensure_session(request)
    conn = db.connect()
    cart = _get_or_create_cart(conn, sid)

    variant_row = conn.execute(
        "SELECT v.*, p.id AS product_id, "
        "       COALESCE(i.quantity, 0) AS stock "
        "FROM variants v JOIN products p ON v.product_id = p.id "
        "LEFT JOIN inventory i ON v.id = i.variant_id "
        "WHERE v.id = ?",
        (variant_id,),
    ).fetchone()
    if variant_row is None:
        return _redirect_with_cookie("/cart?error=Unknown%20variant", sid)
    product_id = variant_row["product_id"]
    if quantity < 1:
        return _redirect_with_cookie(
            f"/products/{product_id}?error=Quantity%20must%20be%20at%20least%201",
            sid,
        )
    if variant_row["stock"] < quantity:
        return _redirect_with_cookie(
            f"/products/{product_id}?error=Variant%20is%20out%20of%20stock",
            sid,
        )

    # Region lock check against cart's selected shipping address if any.
    if cart["shipping_address_id"]:
        addr = conn.execute(
            "SELECT region FROM customer_addresses WHERE id = ?",
            (cart["shipping_address_id"],),
        ).fetchone()
        if addr:
            lock = conn.execute(
                "SELECT 1 FROM variant_region_locks "
                "WHERE variant_id = ? AND region = ?",
                (variant_id, addr["region"]),
            ).fetchone()
            if lock:
                return _redirect_with_cookie(
                    f"/products/{product_id}?error=Variant%20cannot%20ship%20to%20{addr['region']}",
                    sid,
                )

    with db.transaction() as txn:
        existing = txn.execute(
            "SELECT quantity FROM cart_items WHERE cart_id = ? AND variant_id = ?",
            (cart["id"], variant_id),
        ).fetchone()
        if existing:
            new_qty = existing["quantity"] + quantity
            txn.execute(
                "UPDATE cart_items SET quantity = ? "
                "WHERE cart_id = ? AND variant_id = ?",
                (new_qty, cart["id"], variant_id),
            )
        else:
            txn.execute(
                "INSERT INTO cart_items (cart_id, variant_id, quantity) "
                "VALUES (?, ?, ?)",
                (cart["id"], variant_id, quantity),
            )
        _log_audit(txn, operation="cart_add", target_type="cart",
                   target_id=cart["id"],
                   payload={"variant_id": variant_id, "quantity": quantity,
                            "session_id": sid})
    return _redirect_with_cookie("/cart", sid)


@router.post("/cart/update")
async def update_cart(
    request: Request,
    variant_id: str = Form(...),
    quantity: int = Form(...),
) -> RedirectResponse:
    sid = ensure_session(request)
    conn = db.connect()
    cart = _get_or_create_cart(conn, sid)
    if quantity <= 0:
        # treat as remove
        with db.transaction() as txn:
            txn.execute(
                "DELETE FROM cart_items WHERE cart_id = ? AND variant_id = ?",
                (cart["id"], variant_id),
            )
            _log_audit(txn, operation="cart_remove", target_type="cart",
                       target_id=cart["id"],
                       payload={"variant_id": variant_id, "session_id": sid})
        return _redirect_with_cookie("/cart", sid)

    stock_row = conn.execute(
        "SELECT COALESCE(quantity, 0) AS stock FROM inventory WHERE variant_id = ?",
        (variant_id,),
    ).fetchone()
    if stock_row and stock_row["stock"] < quantity:
        return _redirect_with_cookie(
            f"/cart?error=Only%20{stock_row['stock']}%20in%20stock", sid,
        )

    with db.transaction() as txn:
        txn.execute(
            "UPDATE cart_items SET quantity = ? "
            "WHERE cart_id = ? AND variant_id = ?",
            (quantity, cart["id"], variant_id),
        )
        _log_audit(txn, operation="cart_update", target_type="cart",
                   target_id=cart["id"],
                   payload={"variant_id": variant_id, "quantity": quantity,
                            "session_id": sid})
    return _redirect_with_cookie("/cart", sid)


@router.post("/cart/remove")
async def remove_from_cart(
    request: Request,
    variant_id: str = Form(...),
) -> RedirectResponse:
    sid = ensure_session(request)
    conn = db.connect()
    cart = _get_or_create_cart(conn, sid)
    with db.transaction() as txn:
        txn.execute(
            "DELETE FROM cart_items WHERE cart_id = ? AND variant_id = ?",
            (cart["id"], variant_id),
        )
        _log_audit(txn, operation="cart_remove", target_type="cart",
                   target_id=cart["id"],
                   payload={"variant_id": variant_id, "session_id": sid})
    return _redirect_with_cookie("/cart", sid)


# ── coupon application ───────────────────────────────────────────────


@router.post("/cart/apply-coupon")
async def apply_coupon(
    request: Request,
    code: str = Form(...),
) -> RedirectResponse:
    """Apply a coupon. Server validates expiry, max-uses, and stacking.

    The 'looks stackable but server-rejects' trap fires here: codes
    STACK_TRAP_A + STACK_TRAP_B each carry the other in their
    ``stacking_exclusions`` list and the check below rejects.
    """
    sid = ensure_session(request)
    conn = db.connect()
    cart = _get_or_create_cart(conn, sid)
    code = code.strip().upper()
    if not code:
        return _redirect_with_cookie("/cart?error=Coupon%20code%20required", sid)

    coupon = _coupon_row(conn, code)
    if coupon is None:
        return _redirect_with_cookie(f"/cart?error=Unknown%20coupon%20{code}", sid)

    active, reason = _coupon_active(coupon, app_main.now_value())
    if not active:
        return _redirect_with_cookie(
            f"/cart?error={reason.replace(' ', '%20')}", sid,
        )

    applied = db.unpack_codes(cart["coupon_codes"])
    stack_err = _check_stacking(conn, applied, code)
    if stack_err:
        return _redirect_with_cookie(
            f"/cart?error={stack_err.replace(' ', '%20')}", sid,
        )

    applied.append(code)
    with db.transaction() as txn:
        txn.execute(
            "UPDATE carts SET coupon_codes = ? WHERE id = ?",
            (db.pack_codes(applied), cart["id"]),
        )
        _log_audit(txn, operation="coupon_applied", target_type="cart",
                   target_id=cart["id"],
                   payload={"code": code, "session_id": sid})
    return _redirect_with_cookie("/cart", sid)


def _redirect_with_cookie(location: str, sid: str) -> RedirectResponse:
    resp = RedirectResponse(location, status_code=303)
    resp.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax")
    return resp
