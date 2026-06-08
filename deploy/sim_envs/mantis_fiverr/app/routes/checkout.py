"""Checkout — confirm package + write order + redirect."""

from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import auth, db, main as app_main

router = APIRouter()


_TIER_PRICE_COL = {
    "basic": ("pkg_basic_price", "pkg_basic_delivery_d", "pkg_basic_title"),
    "standard": ("pkg_standard_price", "pkg_standard_delivery_d", "pkg_standard_title"),
    "premium": ("pkg_premium_price", "pkg_premium_delivery_d", "pkg_premium_title"),
}


def _templates(request: Request):
    return request.app.state.templates


def _resolve_gig_full(conn, gig_id: str):
    return conn.execute(
        """SELECT g.*, u.username AS seller_username, u.display_name AS seller_display
           FROM gigs g JOIN users u ON g.seller_id = u.id
           WHERE g.id = ?""",
        (gig_id,),
    ).fetchone()


def _tier_or_400(tier: str) -> str:
    if tier not in _TIER_PRICE_COL:
        raise HTTPException(status_code=400, detail="invalid tier")
    return tier


@router.get("/checkout/{gig_id}", response_class=HTMLResponse)
async def checkout(request: Request, gig_id: str):
    tier = _tier_or_400(request.query_params.get("tier", "basic"))
    conn = db.connect()
    gig = _resolve_gig_full(conn, gig_id)
    if gig is None:
        raise HTTPException(status_code=404, detail="gig not found")
    price_col, delivery_col, title_col = _TIER_PRICE_COL[tier]
    unit_price = gig[price_col]
    service_fee = round(unit_price * 0.055 + 2.0, 2)
    total = round(unit_price + service_fee, 2)
    return _templates(request).TemplateResponse(
        "checkout.html",
        {
            "request": request,
            "gig": dict(gig),
            "tier": tier,
            "tier_title": gig[title_col],
            "tier_delivery": gig[delivery_col],
            "unit_price": unit_price,
            "service_fee": service_fee,
            "total": total,
        },
    )


@router.post("/checkout/{gig_id}")
async def submit(
    request: Request,
    gig_id: str,
    tier: str = Form("basic"),
    requirements: str = Form(""),
    payment_method: str = Form("card"),
):
    tier = _tier_or_400(tier)
    conn = db.connect()
    gig = _resolve_gig_full(conn, gig_id)
    if gig is None:
        raise HTTPException(status_code=404, detail="gig not found")

    price_col, delivery_col, title_col = _TIER_PRICE_COL[tier]
    unit_price = float(gig[price_col])
    service_fee = round(unit_price * 0.055 + 2.0, 2)
    total = round(unit_price + service_fee, 2)
    buyer_id = auth.effective_buyer_id(request)
    seller_id = gig["seller_id"]

    # Next order number — string concat for deterministic display id.
    count = int(conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0])
    order_id = f"order_{count+1:05d}"
    number = f"#FO{count+1:04d}"

    placed_at = app_main.now_value()
    try:
        placed_dt = datetime.fromisoformat(placed_at.replace("Z", "+00:00"))
    except Exception:  # noqa: BLE001
        placed_dt = datetime.utcnow()
    due_at = (placed_dt + timedelta(days=int(gig[delivery_col]))).isoformat()

    with db.transaction() as tx:
        tx.execute(
            """INSERT INTO orders (
                id, number, buyer_id, seller_id, gig_id, tier,
                subtotal, service_fee, total, status, requirements,
                placed_at, due_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?)""",
            (order_id, number, buyer_id, seller_id, gig_id, tier,
             unit_price, service_fee, total, requirements,
             placed_at, due_at),
        )
        tx.execute(
            "INSERT INTO order_items (id, order_id, line_no, description, "
            "unit_price, quantity) VALUES (?, ?, 1, ?, ?, 1)",
            (f"oi_{count+1:05d}", order_id,
             f"{gig[title_col]} package",
             unit_price),
        )
        tx.execute(
            "UPDATE gigs SET orders_count = orders_count + 1 WHERE id = ?",
            (gig_id,),
        )
        db.log_audit(
            tx,
            occurred_at=placed_at,
            operation="order_placed",
            target_type="order",
            target_id=order_id,
            payload={
                "buyer_id": buyer_id,
                "seller_id": seller_id,
                "gig_id": gig_id,
                "tier": tier,
                "subtotal": unit_price,
                "service_fee": service_fee,
                "total": total,
                "payment_method": payment_method,
            },
        )

    return RedirectResponse(url=f"/orders/{order_id}", status_code=303)
