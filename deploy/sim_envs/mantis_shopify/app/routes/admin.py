"""Merchant Admin shell — `/store/<store_id>/admin/*`.

Mimics the post-login Shopify Admin experience: Home dashboard,
Orders, Products, Customers, Settings. Wired from the "Log in"
CTA on the Partners /stores list.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import db, main as app_main

router = APIRouter()


def _store_or_404(store_id: str):
    conn = db.connect()
    row = conn.execute(
        "SELECT * FROM stores WHERE id=?", (store_id,),
    ).fetchone()
    return dict(row) if row else None


def _admin_ctx(request: Request, store_id: str, active: str,
               extra: dict | None = None) -> dict | HTMLResponse:
    store = _store_or_404(store_id)
    if store is None:
        return HTMLResponse("Store not found", status_code=404)
    ctx = {
        "request": request,
        "store": store,
        "admin_section": active,
        "store_id": store_id,
    }
    if extra:
        ctx.update(extra)
    return ctx


@router.get("/store/{store_id}/admin", response_class=HTMLResponse)
async def admin_home(store_id: str, request: Request):
    ctx = _admin_ctx(request, store_id, "home")
    if isinstance(ctx, HTMLResponse):
        return ctx
    conn = db.connect()
    # Today/recent stats
    total_sales = int(conn.execute(
        "SELECT COALESCE(SUM(total_cents),0) FROM merchant_orders WHERE store_id=?",
        (store_id,),
    ).fetchone()[0])
    orders_total = int(conn.execute(
        "SELECT COUNT(*) FROM merchant_orders WHERE store_id=?",
        (store_id,),
    ).fetchone()[0])
    products_total = int(conn.execute(
        "SELECT COUNT(*) FROM merchant_products WHERE store_id=?",
        (store_id,),
    ).fetchone()[0])
    customers_total = int(conn.execute(
        "SELECT COUNT(*) FROM merchant_customers WHERE store_id=?",
        (store_id,),
    ).fetchone()[0])
    recent_orders = [dict(r) for r in conn.execute(
        "SELECT * FROM merchant_orders WHERE store_id=? "
        "ORDER BY ordered_at DESC LIMIT 5",
        (store_id,),
    ).fetchall()]
    unfulfilled = int(conn.execute(
        "SELECT COUNT(*) FROM merchant_orders WHERE store_id=? "
        "AND fulfillment_status='unfulfilled'",
        (store_id,),
    ).fetchone()[0])
    ctx.update({
        "total_sales": total_sales,
        "orders_total": orders_total,
        "products_total": products_total,
        "customers_total": customers_total,
        "recent_orders": recent_orders,
        "unfulfilled": unfulfilled,
        "conversion_rate": "1.8%",
        "sessions": 423,
    })
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="admin_home_viewed",
        target_type="store",
        target_id=store_id,
        payload={},
    )
    conn.commit()
    app_main.emit("admin_home_viewed", {"store_id": store_id})
    templates = request.app.state.templates
    return templates.TemplateResponse("admin_home.html", ctx)


@router.get("/store/{store_id}/admin/orders", response_class=HTMLResponse)
async def admin_orders(store_id: str, request: Request,
                       tab: str = "all", q: str = "",
                       sort: str = "newest"):
    ctx = _admin_ctx(request, store_id, "orders")
    if isinstance(ctx, HTMLResponse):
        return ctx
    conn = db.connect()
    where = ["store_id=?"]
    params: list = [store_id]
    if tab == "unfulfilled":
        where.append("fulfillment_status='unfulfilled'")
    elif tab == "unpaid":
        where.append("financial_status='pending'")
    elif tab == "open":
        where.append("fulfillment_status != 'fulfilled'")
    elif tab == "closed":
        where.append("fulfillment_status='fulfilled'")
    if q.strip():
        where.append("(order_number LIKE ? OR customer_name LIKE ?)")
        params.extend([f"%{q.strip()}%", f"%{q.strip()}%"])
    order_by = {
        "newest":         "ordered_at DESC",
        "oldest":         "ordered_at ASC",
        "total_desc":     "total_cents DESC",
        "total_asc":      "total_cents ASC",
        "customer":       "customer_name COLLATE NOCASE ASC",
    }.get(sort, "ordered_at DESC")
    rows = [dict(r) for r in conn.execute(
        f"SELECT * FROM merchant_orders WHERE {' AND '.join(where)} "
        f"ORDER BY {order_by} LIMIT 50",
        params,
    ).fetchall()]
    tab_counts = {}
    for t, w in [("all", "1=1"),
                 ("unfulfilled", "fulfillment_status='unfulfilled'"),
                 ("unpaid", "financial_status='pending'"),
                 ("open", "fulfillment_status != 'fulfilled'"),
                 ("closed", "fulfillment_status='fulfilled'")]:
        tab_counts[t] = int(conn.execute(
            f"SELECT COUNT(*) FROM merchant_orders WHERE store_id=? AND {w}",
            (store_id,),
        ).fetchone()[0])
    ctx.update({
        "orders": rows,
        "tab": tab,
        "q": q,
        "sort": sort,
        "tab_counts": tab_counts,
    })
    templates = request.app.state.templates
    return templates.TemplateResponse("admin_orders.html", ctx)


@router.get("/store/{store_id}/admin/orders/create",
            response_class=HTMLResponse)
async def admin_order_create_form(store_id: str, request: Request):
    """Form to create a new merchant order. MUST be declared before
    /orders/{order_id} so 'create' isn't matched as an order_id."""
    ctx = _admin_ctx(request, store_id, "orders")
    if isinstance(ctx, HTMLResponse):
        return ctx
    templates = request.app.state.templates
    return templates.TemplateResponse("admin_order_new.html", ctx)


@router.post("/store/{store_id}/admin/orders/create")
async def admin_order_create_submit(
    store_id: str, request: Request,
    customer_name: str = Form(...),
    customer_email: str = Form(""),
    item_title: str = Form(...),
    item_qty: int = Form(1),
    item_price: int = Form(0),
    delivery_method: str = Form("Standard"),
):
    conn = db.connect()
    oid = _make_id(conn, "merchant_orders", "mo_")
    order_number = f"#{1000 + int(conn.execute('SELECT COUNT(*) FROM merchant_orders').fetchone()[0]) + 1}"
    total = int(item_price) * int(item_qty) * 100
    items = [{"title": item_title, "qty": int(item_qty),
              "price_cents": int(item_price) * 100}]
    conn.execute(
        "INSERT INTO merchant_orders (id, store_id, order_number, "
        "customer_name, customer_email, total_cents, currency, "
        "financial_status, fulfillment_status, items_count, items_json, "
        "ordered_at, delivery_method) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (oid, store_id, order_number, customer_name.strip(),
         customer_email.strip(), total, "USD", "paid", "unfulfilled",
         1, json.dumps(items), app_main.now_value(), delivery_method),
    )
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="order_created",
        target_type="merchant_order",
        target_id=oid,
        payload={"store_id": store_id, "order_number": order_number,
                 "customer_name": customer_name.strip()},
    )
    conn.commit()
    app_main.emit("order_created", {"order_id": oid})
    return RedirectResponse(
        f"/store/{store_id}/admin/orders/{oid}?created=order",
        status_code=303,
    )


@router.get("/store/{store_id}/admin/orders/{order_id}",
            response_class=HTMLResponse)
async def admin_order_detail(store_id: str, order_id: str, request: Request):
    ctx = _admin_ctx(request, store_id, "orders")
    if isinstance(ctx, HTMLResponse):
        return ctx
    conn = db.connect()
    row = conn.execute(
        "SELECT * FROM merchant_orders WHERE id=? AND store_id=?",
        (order_id, store_id),
    ).fetchone()
    if row is None:
        return HTMLResponse("Order not found", status_code=404)
    order = dict(row)
    items = json.loads(order["items_json"] or "[]")
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="order_viewed",
        target_type="merchant_order",
        target_id=order_id,
        payload={"store_id": store_id, "order_number": order["order_number"]},
    )
    conn.commit()
    app_main.emit("order_viewed", {"order_id": order_id})
    ctx.update({"order": order, "items": items})
    templates = request.app.state.templates
    return templates.TemplateResponse("admin_order_detail.html", ctx)


@router.post("/store/{store_id}/admin/orders/{order_id}/fulfill")
async def admin_order_fulfill(store_id: str, order_id: str, request: Request):
    conn = db.connect()
    row = conn.execute(
        "SELECT * FROM merchant_orders WHERE id=? AND store_id=?",
        (order_id, store_id),
    ).fetchone()
    if row is None:
        return HTMLResponse("Order not found", status_code=404)
    conn.execute(
        "UPDATE merchant_orders SET fulfillment_status='fulfilled' WHERE id=?",
        (order_id,),
    )
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="order_fulfilled",
        target_type="merchant_order",
        target_id=order_id,
        payload={"store_id": store_id, "order_number": row["order_number"]},
    )
    conn.commit()
    app_main.emit("order_fulfilled", {"order_id": order_id})
    return RedirectResponse(
        f"/store/{store_id}/admin/orders/{order_id}?fulfilled=1",
        status_code=303,
    )


@router.get("/store/{store_id}/admin/products", response_class=HTMLResponse)
async def admin_products(store_id: str, request: Request,
                          q: str = "", sort: str = "newest"):
    ctx = _admin_ctx(request, store_id, "products")
    if isinstance(ctx, HTMLResponse):
        return ctx
    conn = db.connect()
    where = ["store_id=?"]
    params: list = [store_id]
    if q.strip():
        where.append("title LIKE ?")
        params.append(f"%{q.strip()}%")
    order_by = {
        "newest":     "created_at DESC",
        "oldest":     "created_at ASC",
        "title_asc":  "title COLLATE NOCASE ASC",
        "title_desc": "title COLLATE NOCASE DESC",
        "price_high": "price_cents DESC",
        "price_low":  "price_cents ASC",
        "inv_low":    "inventory ASC",
    }.get(sort, "created_at DESC")
    rows = [dict(r) for r in conn.execute(
        f"SELECT * FROM merchant_products WHERE {' AND '.join(where)} "
        f"ORDER BY {order_by} LIMIT 100",
        params,
    ).fetchall()]
    ctx.update({"products": rows, "q": q, "sort": sort})
    templates = request.app.state.templates
    return templates.TemplateResponse("admin_products.html", ctx)


@router.get("/store/{store_id}/admin/customers", response_class=HTMLResponse)
async def admin_customers(store_id: str, request: Request,
                           q: str = "", sort: str = "spent_desc"):
    ctx = _admin_ctx(request, store_id, "customers")
    if isinstance(ctx, HTMLResponse):
        return ctx
    conn = db.connect()
    where = ["store_id=?"]
    params: list = [store_id]
    if q.strip():
        where.append("(name LIKE ? OR email LIKE ?)")
        params.extend([f"%{q.strip()}%", f"%{q.strip()}%"])
    order_by = {
        "spent_desc":  "total_spent_cents DESC",
        "spent_asc":   "total_spent_cents ASC",
        "name_asc":    "name COLLATE NOCASE ASC",
        "name_desc":   "name COLLATE NOCASE DESC",
        "orders_desc": "orders_count DESC",
        "recent":      "last_order_at DESC",
    }.get(sort, "total_spent_cents DESC")
    rows = [dict(r) for r in conn.execute(
        f"SELECT * FROM merchant_customers WHERE {' AND '.join(where)} "
        f"ORDER BY {order_by} LIMIT 100",
        params,
    ).fetchall()]
    ctx.update({"customers": rows, "q": q, "sort": sort})
    templates = request.app.state.templates
    return templates.TemplateResponse("admin_customers.html", ctx)


@router.get("/store/{store_id}/admin/customers/create",
            response_class=HTMLResponse)
async def admin_customer_create_form(store_id: str, request: Request):
    """Must precede /customers/{customer_id}."""
    ctx = _admin_ctx(request, store_id, "customers")
    if isinstance(ctx, HTMLResponse):
        return ctx
    templates = request.app.state.templates
    return templates.TemplateResponse("admin_customer_new.html", ctx)


@router.post("/store/{store_id}/admin/customers/create")
async def admin_customer_create_submit(
    store_id: str, request: Request,
    name: str = Form(...),
    email: str = Form(...),
    location: str = Form(""),
):
    conn = db.connect()
    cid = _make_id(conn, "merchant_customers", "mc_")
    conn.execute(
        "INSERT INTO merchant_customers (id, store_id, name, email, "
        "orders_count, total_spent_cents, location, last_order_at, "
        "created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (cid, store_id, name.strip(), email.strip(), 0, 0,
         location.strip(), "", app_main.now_value()),
    )
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="customer_created",
        target_type="merchant_customer",
        target_id=cid,
        payload={"store_id": store_id, "name": name.strip(),
                 "email": email.strip()},
    )
    conn.commit()
    app_main.emit("customer_created", {"customer_id": cid})
    return RedirectResponse(
        f"/store/{store_id}/admin/customers/{cid}?created=customer",
        status_code=303,
    )


@router.get("/store/{store_id}/admin/customers/{customer_id}",
            response_class=HTMLResponse)
async def admin_customer_detail(store_id: str, customer_id: str,
                                 request: Request):
    ctx = _admin_ctx(request, store_id, "customers")
    if isinstance(ctx, HTMLResponse):
        return ctx
    conn = db.connect()
    row = conn.execute(
        "SELECT * FROM merchant_customers WHERE id=? AND store_id=?",
        (customer_id, store_id),
    ).fetchone()
    if row is None:
        return HTMLResponse("Customer not found", status_code=404)
    customer = dict(row)
    # Recent orders for this customer (match on customer_email)
    recent = [dict(r) for r in conn.execute(
        "SELECT * FROM merchant_orders WHERE store_id=? AND customer_email=? "
        "ORDER BY ordered_at DESC LIMIT 8",
        (store_id, customer["email"]),
    ).fetchall()]
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="customer_viewed",
        target_type="merchant_customer",
        target_id=customer_id,
        payload={"store_id": store_id, "name": customer["name"]},
    )
    conn.commit()
    app_main.emit("customer_viewed", {"customer_id": customer_id})
    ctx.update({"customer": customer, "recent_orders": recent})
    templates = request.app.state.templates
    return templates.TemplateResponse("admin_customer_detail.html", ctx)


# ── Create / Add form routes ──────────────────────────────────

def _make_id(conn, table, prefix):
    n = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) + 1
    return f"{prefix}{n:06d}"


@router.get("/store/{store_id}/admin/orders/create", response_class=HTMLResponse)
async def admin_order_new_form(store_id: str, request: Request):
    ctx = _admin_ctx(request, store_id, "orders")
    if isinstance(ctx, HTMLResponse):
        return ctx
    templates = request.app.state.templates
    return templates.TemplateResponse("admin_order_new.html", ctx)


@router.post("/store/{store_id}/admin/orders/create")
async def admin_order_new_submit(
    store_id: str, request: Request,
    customer_name: str = Form(...),
    customer_email: str = Form(""),
    item_title: str = Form(...),
    item_qty: int = Form(1),
    item_price: int = Form(0),
    delivery_method: str = Form("Standard"),
):
    conn = db.connect()
    oid = _make_id(conn, "merchant_orders", "mo_")
    order_number = f"#{1000 + int(conn.execute('SELECT COUNT(*) FROM merchant_orders').fetchone()[0]) + 1}"
    total = int(item_price) * int(item_qty) * 100  # price entered as dollars
    items = [{"title": item_title, "qty": int(item_qty),
              "price_cents": int(item_price) * 100}]
    conn.execute(
        "INSERT INTO merchant_orders (id, store_id, order_number, "
        "customer_name, customer_email, total_cents, currency, "
        "financial_status, fulfillment_status, items_count, items_json, "
        "ordered_at, delivery_method) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (oid, store_id, order_number, customer_name.strip(),
         customer_email.strip(), total, "USD", "paid", "unfulfilled",
         1, json.dumps(items), app_main.now_value(), delivery_method),
    )
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="order_created",
        target_type="merchant_order",
        target_id=oid,
        payload={"store_id": store_id, "order_number": order_number,
                 "customer_name": customer_name.strip()},
    )
    conn.commit()
    app_main.emit("order_created", {"order_id": oid})
    return RedirectResponse(
        f"/store/{store_id}/admin/orders/{oid}?created=order",
        status_code=303,
    )


@router.get("/store/{store_id}/admin/products/create", response_class=HTMLResponse)
async def admin_product_new_form(store_id: str, request: Request):
    ctx = _admin_ctx(request, store_id, "products")
    if isinstance(ctx, HTMLResponse):
        return ctx
    templates = request.app.state.templates
    return templates.TemplateResponse("admin_product_new.html", ctx)


@router.post("/store/{store_id}/admin/products/create")
async def admin_product_new_submit(
    store_id: str, request: Request,
    title: str = Form(...),
    price: int = Form(0),
    inventory: int = Form(0),
    vendor: str = Form(""),
    product_type: str = Form(""),
    status: str = Form("draft"),
):
    conn = db.connect()
    pid = _make_id(conn, "merchant_products", "mp_")
    conn.execute(
        "INSERT INTO merchant_products (id, store_id, title, status, "
        "inventory, vendor, product_type, price_cents, sku, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (pid, store_id, title.strip(), status, int(inventory),
         vendor.strip(), product_type.strip(), int(price) * 100,
         f"SKU-{pid[3:]}", app_main.now_value()),
    )
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="product_created",
        target_type="merchant_product",
        target_id=pid,
        payload={"store_id": store_id, "title": title.strip(),
                 "status": status},
    )
    conn.commit()
    app_main.emit("product_created", {"product_id": pid})
    return RedirectResponse(
        f"/store/{store_id}/admin/products/{pid}?created=product",
        status_code=303,
    )


@router.get("/store/{store_id}/admin/customers/create",
            response_class=HTMLResponse)
async def admin_customer_new_form(store_id: str, request: Request):
    ctx = _admin_ctx(request, store_id, "customers")
    if isinstance(ctx, HTMLResponse):
        return ctx
    templates = request.app.state.templates
    return templates.TemplateResponse("admin_customer_new.html", ctx)


@router.post("/store/{store_id}/admin/customers/create")
async def admin_customer_new_submit(
    store_id: str, request: Request,
    name: str = Form(...),
    email: str = Form(...),
    location: str = Form(""),
):
    conn = db.connect()
    cid = _make_id(conn, "merchant_customers", "mc_")
    conn.execute(
        "INSERT INTO merchant_customers (id, store_id, name, email, "
        "orders_count, total_spent_cents, location, last_order_at, "
        "created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (cid, store_id, name.strip(), email.strip(), 0, 0,
         location.strip(), "", app_main.now_value()),
    )
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="customer_created",
        target_type="merchant_customer",
        target_id=cid,
        payload={"store_id": store_id, "name": name.strip(),
                 "email": email.strip()},
    )
    conn.commit()
    app_main.emit("customer_created", {"customer_id": cid})
    return RedirectResponse(
        f"/store/{store_id}/admin/customers/{cid}?created=customer",
        status_code=303,
    )


@router.get("/store/{store_id}/admin/discounts/new",
            response_class=HTMLResponse)
async def admin_discount_new_form(store_id: str, request: Request):
    ctx = _admin_ctx(request, store_id, "discounts")
    if isinstance(ctx, HTMLResponse):
        return ctx
    templates = request.app.state.templates
    return templates.TemplateResponse("admin_discount_new.html", ctx)


@router.post("/store/{store_id}/admin/discounts/new")
async def admin_discount_new_submit(
    store_id: str, request: Request,
    code: str = Form(...),
    discount_type: str = Form("Amount off products"),
    value: str = Form(""),
):
    conn = db.connect()
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="discount_created",
        target_type="discount_code",
        target_id=code.strip().upper(),
        payload={"store_id": store_id, "code": code.strip().upper(),
                 "type": discount_type, "value": value.strip()},
    )
    conn.commit()
    app_main.emit("discount_created", {"code": code.strip().upper()})
    return RedirectResponse(
        f"/store/{store_id}/admin/discounts?created=discount",
        status_code=303,
    )


@router.post("/store/{store_id}/admin/orders/{order_id}/refund")
async def admin_order_refund(store_id: str, order_id: str, request: Request):
    conn = db.connect()
    row = conn.execute(
        "SELECT * FROM merchant_orders WHERE id=? AND store_id=?",
        (order_id, store_id),
    ).fetchone()
    if row is None:
        return HTMLResponse("Order not found", status_code=404)
    conn.execute(
        "UPDATE merchant_orders SET financial_status='refunded' WHERE id=?",
        (order_id,),
    )
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="order_refunded",
        target_type="merchant_order",
        target_id=order_id,
        payload={"store_id": store_id, "order_number": row["order_number"]},
    )
    conn.commit()
    app_main.emit("order_refunded", {"order_id": order_id})
    return RedirectResponse(
        f"/store/{store_id}/admin/orders/{order_id}?refunded=1",
        status_code=303,
    )


@router.get("/store/{store_id}/admin/marketing", response_class=HTMLResponse)
async def admin_marketing(store_id: str, request: Request):
    ctx = _admin_ctx(request, store_id, "marketing")
    if isinstance(ctx, HTMLResponse):
        return ctx
    # Synthetic campaigns derived from store id for determinism
    base = [
        ("Spring Awakening", "Email", "Active", 4200, "Apr 2 – May 16"),
        ("Mother's Day Drop", "Facebook + Instagram", "Completed", 6800, "Apr 28 – May 12"),
        ("New Arrivals Boost", "Google Ads", "Active", 3120, "May 5 – Jun 5"),
        ("Lapsed-Customer Win-Back", "Email", "Scheduled", 1500, "Jun 10 – Jun 24"),
        ("Summer Solstice Tease", "TikTok", "Draft", 0, "—"),
    ]
    campaigns = [
        {"id": f"camp_{i:04d}", "name": n, "channel": ch, "status": st,
         "spend_cents": sp * 100, "window": w}
        for i, (n, ch, st, sp, w) in enumerate(base, start=1)
    ]
    ctx["campaigns"] = campaigns
    templates = request.app.state.templates
    return templates.TemplateResponse("admin_marketing.html", ctx)


@router.get("/store/{store_id}/admin/discounts", response_class=HTMLResponse)
async def admin_discounts(store_id: str, request: Request):
    ctx = _admin_ctx(request, store_id, "discounts")
    if isinstance(ctx, HTMLResponse):
        return ctx
    base = [
        ("SPRING25",     "Amount off products", "25% off all", "Active",     142, "Expires Jun 15, 2026"),
        ("FREESHIP",     "Free shipping",       "Free shipping ≥ $50", "Active", 286, "No end date"),
        ("WELCOME10",    "Amount off order",    "$10 off first order", "Active", 1342, "No end date"),
        ("LOYAL-LOY",    "Amount off products", "15% loyalty",        "Scheduled", 0,   "Starts Jun 12, 2026"),
        ("VIP-Holiday-24", "Amount off order",  "$30 off",            "Expired", 89,   "Expired Jan 6, 2026"),
        ("BUNDLE-3",     "Buy X get Y",         "Buy 2 get 1 free",   "Active", 56,    "No end date"),
    ]
    discounts = [
        {"code": c, "type": t, "value": v, "status": s, "uses": u, "schedule": sc}
        for c, t, v, s, u, sc in base
    ]
    ctx["discounts"] = discounts
    templates = request.app.state.templates
    return templates.TemplateResponse("admin_discounts.html", ctx)


@router.get("/store/{store_id}/admin/analytics", response_class=HTMLResponse)
async def admin_analytics(store_id: str, request: Request):
    ctx = _admin_ctx(request, store_id, "analytics")
    if isinstance(ctx, HTMLResponse):
        return ctx
    conn = db.connect()
    total_sales = int(conn.execute(
        "SELECT COALESCE(SUM(total_cents),0) FROM merchant_orders WHERE store_id=?",
        (store_id,),
    ).fetchone()[0])
    orders_total = int(conn.execute(
        "SELECT COUNT(*) FROM merchant_orders WHERE store_id=?",
        (store_id,),
    ).fetchone()[0])
    # Top products by occurrence in orders
    top_products = [dict(r) for r in conn.execute(
        "SELECT title, COUNT(*) as units, SUM(price_cents) as revenue "
        "FROM merchant_products WHERE store_id=? "
        "GROUP BY title ORDER BY revenue DESC LIMIT 5",
        (store_id,),
    ).fetchall()]
    ctx.update({
        "total_sales": total_sales,
        "orders_total": orders_total,
        "aov_cents": total_sales // max(1, orders_total),
        "returning_rate": "31%",
        "top_products": top_products,
    })
    templates = request.app.state.templates
    return templates.TemplateResponse("admin_analytics.html", ctx)


@router.get("/store/{store_id}/admin/content", response_class=HTMLResponse)
async def admin_content(store_id: str, request: Request,
                         tab: str = "pages"):
    ctx = _admin_ctx(request, store_id, "content")
    if isinstance(ctx, HTMLResponse):
        return ctx
    pages = [
        {"id": "pg_about",    "title": "About us",         "status": "Visible", "updated": "May 28, 2026"},
        {"id": "pg_contact",  "title": "Contact",          "status": "Visible", "updated": "May 10, 2026"},
        {"id": "pg_returns",  "title": "Returns policy",   "status": "Visible", "updated": "Apr 22, 2026"},
        {"id": "pg_privacy",  "title": "Privacy policy",   "status": "Visible", "updated": "Apr 22, 2026"},
        {"id": "pg_holiday",  "title": "Holiday lookbook", "status": "Hidden",  "updated": "Dec 1, 2025"},
    ]
    blog_posts = [
        {"id": "bp_journal_01", "title": "Inside the studio: spring drop",
         "author": "Mason", "status": "Published",  "published": "May 24, 2026"},
        {"id": "bp_journal_02", "title": "Sourcing notes: small batch ceramics",
         "author": "Mason", "status": "Published",  "published": "May 8, 2026"},
        {"id": "bp_journal_03", "title": "Behind the new linen line",
         "author": "Mason", "status": "Scheduled",  "published": "Jun 18, 2026"},
        {"id": "bp_journal_04", "title": "Customer spotlight: Lena's living room",
         "author": "Mason", "status": "Draft",      "published": "—"},
    ]
    files = [
        {"name": "spring-drop-hero.jpg", "size": "2.4 MB", "kind": "Image"},
        {"name": "size-guide-2026.pdf",  "size": "412 KB", "kind": "PDF"},
        {"name": "studio-walkthrough.mp4", "size": "18.6 MB", "kind": "Video"},
        {"name": "lookbook-summer.pdf",   "size": "1.1 MB", "kind": "PDF"},
    ]
    metaobjects = [
        {"name": "Studio location", "entries": 3,  "updated": "May 30, 2026"},
        {"name": "Press feature",   "entries": 12, "updated": "May 18, 2026"},
        {"name": "Founder bio",     "entries": 1,  "updated": "Apr 11, 2026"},
    ]
    if tab not in {"pages", "blog_posts", "files", "metaobjects"}:
        tab = "pages"
    ctx.update({
        "tab": tab, "pages": pages, "blog_posts": blog_posts,
        "files": files, "metaobjects": metaobjects,
    })
    templates = request.app.state.templates
    return templates.TemplateResponse("admin_content.html", ctx)


@router.get("/store/{store_id}/admin/online-store", response_class=HTMLResponse)
async def admin_online_store(store_id: str, request: Request):
    ctx = _admin_ctx(request, store_id, "online_store")
    if isinstance(ctx, HTMLResponse):
        return ctx
    templates_list = [
        {"name": "Linen — Spring 2026", "role": "Live theme",
         "updated": "May 24, 2026", "status": "Published"},
        {"name": "Skeleton — testing",  "role": "Unpublished",
         "updated": "May 12, 2026", "status": "Draft"},
        {"name": "Marble — holiday",    "role": "Unpublished",
         "updated": "Dec 5, 2025",   "status": "Draft"},
    ]
    pages = [
        {"name": "About",    "type": "Page"},
        {"name": "Contact",  "type": "Page"},
        {"name": "Journal",  "type": "Blog"},
    ]
    menus = [
        {"name": "Main menu",   "item_count": 6},
        {"name": "Footer menu", "item_count": 4},
    ]
    ctx.update({
        "themes": templates_list,
        "pages": pages, "menus": menus,
        "domain": f"{ctx['store']['slug']}.myshopify.com",
        "custom_domain": "mason.example",
    })
    templates_ = request.app.state.templates
    return templates_.TemplateResponse("admin_online_store.html", ctx)


@router.get("/store/{store_id}/admin/pos", response_class=HTMLResponse)
async def admin_pos(store_id: str, request: Request):
    ctx = _admin_ctx(request, store_id, "pos")
    if isinstance(ctx, HTMLResponse):
        return ctx
    locations = [
        {"name": "Boylston St flagship",   "city": "Boston, MA",
         "staff": 4, "devices": 3, "status": "Open"},
        {"name": "Studio + Café (popup)",  "city": "Brooklyn, NY",
         "staff": 2, "devices": 1, "status": "Open"},
        {"name": "Holiday pop-in",          "city": "Portland, OR",
         "staff": 1, "devices": 1, "status": "Inactive"},
    ]
    ctx["locations"] = locations
    templates = request.app.state.templates
    return templates.TemplateResponse("admin_pos.html", ctx)


@router.get("/store/{store_id}/admin/apps", response_class=HTMLResponse)
async def admin_apps(store_id: str, request: Request):
    ctx = _admin_ctx(request, store_id, "apps")
    if isinstance(ctx, HTMLResponse):
        return ctx
    installed = [
        {"name": "Klaviyo: Email Marketing", "category": "Marketing",
         "rating": 4.6, "installed": True},
        {"name": "Recharge Subscriptions",   "category": "Selling",
         "rating": 4.8, "installed": True},
        {"name": "Judge.me Product Reviews", "category": "Reviews",
         "rating": 4.9, "installed": True},
    ]
    suggested = [
        {"name": "Shopify Inbox", "category": "Customer support", "rating": 4.7},
        {"name": "Loox Photo Reviews", "category": "Reviews", "rating": 4.9},
        {"name": "PageFly Page Builder", "category": "Store design", "rating": 4.9},
        {"name": "SEO Manager", "category": "Marketing", "rating": 4.8},
    ]
    ctx.update({"installed": installed, "suggested": suggested})
    templates = request.app.state.templates
    return templates.TemplateResponse("admin_apps.html", ctx)


@router.get("/store/{store_id}/admin/settings", response_class=HTMLResponse)
async def admin_settings(store_id: str, request: Request):
    ctx = _admin_ctx(request, store_id, "settings")
    if isinstance(ctx, HTMLResponse):
        return ctx
    groups = [
        ("Store details", [
            ("General",            "Store name, address, and contact details"),
            ("Plan",               "Manage subscription and billing"),
            ("Domains",            "Manage URL and routing for your store"),
        ]),
        ("Selling", [
            ("Payments",           "Accept payments and manage providers"),
            ("Checkout",           "Customize checkout, post-purchase, and policies"),
            ("Shipping & delivery","Rates and zones"),
            ("Taxes & duties",     "Tax overrides, exemptions, and duties"),
        ]),
        ("Customer experience", [
            ("Locations",          "Manage retail and fulfillment locations"),
            ("Markets",            "Sell into new geographies"),
            ("Notifications",      "Edit transactional emails and SMS"),
            ("Gift cards",         "Issue and refund balances"),
        ]),
        ("Operations", [
            ("Users and permissions",  "Invite staff and configure roles"),
            ("Apps and sales channels","Manage installed integrations"),
            ("Files",                  "Storage for uploaded files"),
            ("Custom data",            "Metafields, metaobjects, schemas"),
        ]),
    ]
    ctx["groups"] = groups
    templates = request.app.state.templates
    return templates.TemplateResponse("admin_settings.html", ctx)


@router.get("/store/{store_id}/admin/products/{product_id}",
            response_class=HTMLResponse)
async def admin_product_detail(store_id: str, product_id: str,
                                request: Request):
    ctx = _admin_ctx(request, store_id, "products")
    if isinstance(ctx, HTMLResponse):
        return ctx
    conn = db.connect()
    row = conn.execute(
        "SELECT * FROM merchant_products WHERE id=? AND store_id=?",
        (product_id, store_id),
    ).fetchone()
    if row is None:
        return HTMLResponse("Product not found", status_code=404)
    product = dict(row)
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="product_viewed",
        target_type="merchant_product",
        target_id=product_id,
        payload={"store_id": store_id, "title": product["title"]},
    )
    conn.commit()
    app_main.emit("product_viewed", {"product_id": product_id})
    ctx.update({"product": product})
    templates = request.app.state.templates
    return templates.TemplateResponse("admin_product_detail.html", ctx)
