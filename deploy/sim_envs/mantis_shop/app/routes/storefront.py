"""Storefront: catalog list, search, product detail (PDP)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from .. import db, main as app_main

router = APIRouter()

PAGE_SIZE = 24


@router.get("/catalog", response_class=HTMLResponse)
async def catalog(request: Request) -> HTMLResponse:
    """Paginated grid of products. Query params:

    * ``page`` — 1-based
    * ``category`` — e.g. ``outerwear-women``
    * ``q`` — substring on title
    * ``max_price`` — float ceiling
    """
    page = max(1, int(request.query_params.get("page") or 1))
    category = (request.query_params.get("category") or "").strip()
    q = (request.query_params.get("q") or "").strip()
    max_price_raw = (request.query_params.get("max_price") or "").strip()

    conn = db.connect()
    where = ["1=1"]
    args: list[Any] = []
    if category:
        where.append("p.category = ?")
        args.append(category)
    if q:
        where.append("p.title LIKE ?")
        args.append(f"%{q}%")
    if max_price_raw:
        try:
            mp = float(max_price_raw)
            where.append("COALESCE(p.sale_price, p.base_price) <= ?")
            args.append(mp)
        except ValueError:
            pass

    where_sql = " AND ".join(where)
    total = conn.execute(
        f"SELECT COUNT(*) FROM products p WHERE {where_sql}", args,
    ).fetchone()[0]
    offset = (page - 1) * PAGE_SIZE
    rows = conn.execute(
        f"SELECT p.* FROM products p WHERE {where_sql} "
        f"ORDER BY p.id LIMIT ? OFFSET ?",
        args + [PAGE_SIZE, offset],
    ).fetchall()
    products = [dict(r) for r in rows]
    pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)

    # Categories for the sidebar.
    categories = [
        r["slug"] for r in conn.execute(
            "SELECT slug FROM collections WHERE kind = 'rule' "
            "AND slug NOT IN ('on-sale', 'under-50') ORDER BY slug"
        ).fetchall()
    ]

    return app_main.app.state.templates.TemplateResponse(
        "catalog.html",
        {
            "request": request,
            "products": products,
            "categories": categories,
            "page": page,
            "pages": pages,
            "total": total,
            "filters": {"category": category, "q": q, "max_price": max_price_raw},
        },
    )


@router.get("/search", response_class=HTMLResponse)
async def search(request: Request) -> HTMLResponse:
    """Top-bar fuzzy search → catalog redirect, but render directly so
    the result is a real page the agent can read."""
    q = (request.query_params.get("q") or "").strip()
    if not q:
        return await catalog(request)
    return await catalog(request)


@router.get("/products/{product_id}", response_class=HTMLResponse)
async def product_detail(request: Request, product_id: str) -> HTMLResponse:
    """PDP. Variant picker is server-side aware — disabled options for
    out-of-stock are marked in the rendered DOM. The agent must still
    pick a valid (in-stock, region-permitted) variant; the add-to-cart
    handler rejects invalid combos with a re-rendered PDP and an error."""
    conn = db.connect()
    row = conn.execute(
        "SELECT * FROM products WHERE id = ?", (product_id,),
    ).fetchone()
    if row is None:
        return HTMLResponse("Not found", status_code=404)
    product = dict(row)
    variants = [
        dict(v) for v in conn.execute(
            "SELECT v.*, COALESCE(i.quantity, 0) AS stock "
            "FROM variants v LEFT JOIN inventory i ON v.id = i.variant_id "
            "WHERE v.product_id = ? ORDER BY v.size, v.color",
            (product_id,),
        ).fetchall()
    ]
    # Region locks per variant
    locks_rows = conn.execute(
        "SELECT variant_id, region FROM variant_region_locks "
        "WHERE variant_id IN (SELECT id FROM variants WHERE product_id = ?)",
        (product_id,),
    ).fetchall()
    locked: dict[str, list[str]] = {}
    for r in locks_rows:
        locked.setdefault(r["variant_id"], []).append(r["region"])
    for v in variants:
        v["locked_regions"] = locked.get(v["id"], [])

    sizes = sorted({v["size"] for v in variants})
    colors = sorted({v["color"] for v in variants})

    error = request.query_params.get("error") or ""

    return app_main.app.state.templates.TemplateResponse(
        "product_detail.html",
        {
            "request": request,
            "product": product,
            "variants": variants,
            "sizes": sizes,
            "colors": colors,
            "error": error,
        },
    )
