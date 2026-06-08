"""Home, search, category landing — the discovery surface."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from .. import db, main as app_main

router = APIRouter()


def _templates(request: Request):
    return request.app.state.templates


def _gig_card_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "slug": row["slug"],
        "title": row["title"],
        "seller_id": row["seller_id"],
        "seller_username": row["seller_username"],
        "seller_display": row["seller_display"],
        "seller_level": row["seller_level"],
        "seller_avatar_palette": row["seller_avatar_palette"],
        "basic_price": row["pkg_basic_price"],
        "avg_rating": row["avg_rating"],
        "review_count": row["review_count"],
        "image_palette": row["image_palette"],
        "category_slug": row["category_slug"],
    }


GIG_CARD_SELECT = """
    SELECT g.id, g.slug, g.title, g.seller_id, g.pkg_basic_price,
           g.avg_rating, g.review_count, g.image_palette, g.category_slug,
           u.username AS seller_username,
           u.display_name AS seller_display,
           s.level AS seller_level,
           s.avatar_palette AS seller_avatar_palette
    FROM gigs g
    JOIN users u ON g.seller_id = u.id
    JOIN sellers s ON g.seller_id = s.user_id
"""


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    conn = db.connect()
    # Popular categories — top-level only (parent_slug IS NULL).
    cats = [dict(r) for r in conn.execute(
        "SELECT slug, title, icon FROM categories WHERE parent_slug IS NULL "
        "ORDER BY sort_order"
    ).fetchall()]
    # Featured 8 gigs — ordered by orders_count desc.
    featured = [
        _gig_card_row(r) for r in conn.execute(
            GIG_CARD_SELECT + " ORDER BY g.orders_count DESC LIMIT 8"
        ).fetchall()
    ]
    popular_chips = ["Website Design", "WordPress", "Logo Design", "AI Services"]
    return _templates(request).TemplateResponse(
        "home.html",
        {
            "request": request,
            "categories": cats,
            "featured": featured,
            "popular_chips": popular_chips,
        },
    )


@router.get("/search/gigs", response_class=HTMLResponse)
async def search(request: Request):
    conn = db.connect()
    qp = request.query_params
    query = qp.get("query", "").strip()
    sort = qp.get("sort", "relevance")
    level_filters = qp.getlist("level")
    budget = qp.get("budget", "")
    delivery = qp.get("delivery", "")
    try:
        page = max(1, int(qp.get("page", 1)))
    except (TypeError, ValueError):
        page = 1
    per_page = 12

    where: list[str] = []
    params: list[Any] = []
    if query:
        where.append("(LOWER(g.title) LIKE ? OR LOWER(g.category_slug) LIKE ?)")
        ql = f"%{query.lower()}%"
        params.extend([ql, ql])
    if level_filters:
        where.append("s.level IN (" + ",".join("?" * len(level_filters)) + ")")
        params.extend(level_filters)
    if budget == "value":
        where.append("g.pkg_basic_price <= 50")
    elif budget == "mid":
        where.append("g.pkg_basic_price > 50 AND g.pkg_basic_price <= 200")
    elif budget == "high":
        where.append("g.pkg_basic_price > 200")
    if delivery == "express":
        where.append("g.pkg_basic_delivery_d <= 1")
    elif delivery == "3d":
        where.append("g.pkg_basic_delivery_d <= 3")
    elif delivery == "7d":
        where.append("g.pkg_basic_delivery_d <= 7")
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""

    if sort == "best_selling":
        order_sql = " ORDER BY g.orders_count DESC"
    elif sort == "newest":
        order_sql = " ORDER BY g.created_at DESC"
    elif sort == "rating":
        order_sql = " ORDER BY g.avg_rating DESC"
    else:  # relevance
        order_sql = " ORDER BY g.review_count DESC"

    total = int(conn.execute(
        "SELECT COUNT(*) FROM gigs g JOIN sellers s ON g.seller_id = s.user_id"
        + where_sql, params
    ).fetchone()[0])

    offset = (page - 1) * per_page
    rows = conn.execute(
        GIG_CARD_SELECT + where_sql + order_sql + f" LIMIT {per_page} OFFSET {offset}",
        params,
    ).fetchall()
    results = [_gig_card_row(r) for r in rows]
    pages = max(1, (total + per_page - 1) // per_page)

    return _templates(request).TemplateResponse(
        "search.html",
        {
            "request": request,
            "query": query,
            "results": results,
            "total": total,
            "page": page,
            "pages": pages,
            "sort": sort,
            "level_filters": level_filters,
            "budget": budget,
            "delivery": delivery,
        },
    )


@router.get("/categories/{slug}", response_class=HTMLResponse)
async def category(request: Request, slug: str):
    conn = db.connect()
    cat = conn.execute(
        "SELECT slug, title, parent_slug, icon FROM categories WHERE slug = ?",
        (slug,),
    ).fetchone()
    if cat is None:
        return HTMLResponse("Category not found", status_code=404)
    cat = dict(cat)
    subs = [dict(r) for r in conn.execute(
        "SELECT slug, title, icon FROM categories WHERE parent_slug = ? "
        "ORDER BY sort_order",
        (slug,),
    ).fetchall()]
    rows = conn.execute(
        GIG_CARD_SELECT + " WHERE g.category_slug = ? OR g.category_slug IN ("
        "SELECT slug FROM categories WHERE parent_slug = ?) "
        "ORDER BY g.review_count DESC LIMIT 12",
        (slug, slug),
    ).fetchall()
    return _templates(request).TemplateResponse(
        "category.html",
        {
            "request": request,
            "category": cat,
            "subcategories": subs,
            "results": [_gig_card_row(r) for r in rows],
        },
    )
