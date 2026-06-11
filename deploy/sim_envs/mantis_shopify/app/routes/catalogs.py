"""Catalogs — `/catalogs`, `/catalogs/new`, `/catalogs/<id>/publish`."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import db, main as app_main

router = APIRouter()


@router.get("/catalogs", response_class=HTMLResponse)
async def catalogs(request: Request):
    templates = request.app.state.templates
    conn = db.connect()
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM catalogs ORDER BY created_at DESC"
    ).fetchall()]
    return templates.TemplateResponse(
        "catalogs.html",
        {
            "request": request,
            "active_section": "catalogs",
            "rows": rows,
        },
    )


@router.get("/catalogs/new", response_class=HTMLResponse)
async def catalog_new_form(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "catalog_new.html",
        {"request": request, "active_section": "catalogs"},
    )


@router.post("/catalogs/new")
async def catalog_new_submit(
    request: Request,
    name: str = Form(...),
    products_count: int = Form(0),
):
    conn = db.connect()
    cid = f"catalog_user_{int(conn.execute('SELECT COUNT(*) FROM catalogs').fetchone()[0]) + 1:05d}"
    conn.execute(
        "INSERT INTO catalogs (id, name, products_count, status, created_at) "
        "VALUES (?,?,?,?,?)",
        (cid, name.strip(), int(products_count), "draft", app_main.now_value()),
    )
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="catalog_created",
        target_type="catalog",
        target_id=cid,
        payload={"name": name.strip(), "products_count": int(products_count)},
    )
    conn.commit()
    app_main.emit("catalog_created", {"catalog_id": cid})
    return RedirectResponse("/catalogs", status_code=303)


@router.post("/catalogs/{catalog_id}/publish")
async def catalog_publish(catalog_id: str, request: Request):
    conn = db.connect()
    conn.execute(
        "UPDATE catalogs SET status='published' WHERE id=?",
        (catalog_id,),
    )
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="catalog_published",
        target_type="catalog",
        target_id=catalog_id,
        payload={},
    )
    conn.commit()
    app_main.emit("catalog_published", {"catalog_id": catalog_id})
    return RedirectResponse(f"/catalogs/{catalog_id}", status_code=303)


@router.get("/catalogs/{catalog_id}", response_class=HTMLResponse)
async def catalog_detail(catalog_id: str, request: Request, tab: str = "products"):
    templates = request.app.state.templates
    conn = db.connect()
    row = conn.execute(
        "SELECT * FROM catalogs WHERE id=?", (catalog_id,),
    ).fetchone()
    if row is None:
        return HTMLResponse("Catalog not found", status_code=404)
    catalog = dict(row)
    if tab not in {"products", "stores", "settings"}:
        tab = "products"

    # Synthetic product list — derived deterministically from the
    # catalog name so it's stable but varied.
    base_products = [
        ("Hand-poured candle, juniper", 24, 1280, "Active"),
        ("Linen apron, oat", 36, 4200, "Active"),
        ("Brass desk lamp, matte", 142, 950, "Active"),
        ("Riverstone tumbler set (4)", 58, 220, "Active"),
        ("Felted wool throw, sage", 145, 410, "Draft"),
        ("Ceramic pour-over kit", 42, 1830, "Active"),
        ("Cedar cutting board, large", 64, 745, "Active"),
        ("Walnut wall mirror, round", 198, 86, "Draft"),
    ]
    rng = catalog["products_count"] or 1
    products = [
        {
            "sku": f"sku-{i:04d}",
            "title": p[0],
            "price_cents": p[1] * 100,
            "inventory": p[2] + (i * 11) % 30,
            "status": p[3],
        }
        for i, p in enumerate(base_products[:min(rng, len(base_products))])
    ]

    # Stores this catalog is published to (subset of stores table).
    target_stores = [dict(r) for r in conn.execute(
        "SELECT id, name, slug, plan FROM stores ORDER BY last_login_at DESC "
        "LIMIT 4"
    ).fetchall()]

    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="catalog_viewed",
        target_type="catalog",
        target_id=catalog_id,
        payload={"name": catalog["name"], "status": catalog["status"]},
    )
    conn.commit()
    app_main.emit("catalog_viewed", {"catalog_id": catalog_id})

    return templates.TemplateResponse(
        "catalog_detail.html",
        {
            "request": request,
            "active_section": "catalogs",
            "catalog": catalog,
            "products": products,
            "target_stores": target_stores,
            "current_tab": tab,
        },
    )
