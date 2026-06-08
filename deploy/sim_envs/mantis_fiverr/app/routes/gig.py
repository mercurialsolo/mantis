"""Gig detail (the canonical /<username>/<gig-slug> URL) + favorites."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import auth, db, main as app_main

router = APIRouter()


def _templates(request: Request):
    return request.app.state.templates


def _resolve_gig(conn, username: str, slug: str) -> dict[str, Any] | None:
    row = conn.execute(
        """SELECT g.*, u.username AS seller_username, u.display_name AS seller_display,
                  s.level AS seller_level, s.country AS seller_country,
                  s.languages AS seller_languages, s.response_time_h AS seller_response_h,
                  s.member_since AS seller_member_since,
                  s.avg_rating AS seller_avg_rating, s.review_count AS seller_review_count,
                  s.avatar_palette AS seller_avatar_palette
           FROM gigs g
           JOIN users u ON g.seller_id = u.id
           JOIN sellers s ON g.seller_id = s.user_id
           WHERE u.username = ? AND g.slug = ?""",
        (username, slug),
    ).fetchone()
    if row is None:
        return None
    out = dict(row)
    out["pkg_basic_features"] = db.unpack_json(out["pkg_basic_features"]) or []
    out["pkg_standard_features"] = db.unpack_json(out["pkg_standard_features"]) or []
    out["pkg_premium_features"] = db.unpack_json(out["pkg_premium_features"]) or []
    out["seller_languages"] = db.unpack_json(out["seller_languages"]) or []
    return out


@router.get("/{username}/{slug}", response_class=HTMLResponse)
async def gig_detail(request: Request, username: str, slug: str):
    # Avoid swallowing the /static, /search, /categories, /checkout,
    # /inbox, /orders, /login, /signup top-level prefixes.
    if username in {
        "static", "assets", "search", "categories", "checkout",
        "inbox", "orders", "login", "signup", "logout", "favorite",
        "__env__",
    }:
        raise HTTPException(status_code=404)
    conn = db.connect()
    gig = _resolve_gig(conn, username, slug)
    if gig is None:
        raise HTTPException(status_code=404, detail="gig not found")

    reviews = [dict(r) for r in conn.execute(
        """SELECT r.*, u.display_name AS buyer_display, u.username AS buyer_username
           FROM reviews r
           JOIN users u ON r.buyer_id = u.id
           WHERE r.gig_id = ?
           ORDER BY r.created_at DESC LIMIT 8""",
        (gig["id"],),
    ).fetchall()]

    category_row = conn.execute(
        "SELECT slug, title, parent_slug FROM categories WHERE slug = ?",
        (gig["category_slug"],),
    ).fetchone()
    crumbs = [{"slug": "/", "title": "Home"}]
    if category_row is not None:
        if category_row["parent_slug"]:
            parent = conn.execute(
                "SELECT slug, title FROM categories WHERE slug = ?",
                (category_row["parent_slug"],),
            ).fetchone()
            if parent:
                crumbs.append({"slug": f"/categories/{parent['slug']}",
                               "title": parent["title"]})
        crumbs.append({"slug": f"/categories/{category_row['slug']}",
                       "title": category_row["title"]})

    return _templates(request).TemplateResponse(
        "gig_detail.html",
        {
            "request": request,
            "gig": gig,
            "reviews": reviews,
            "crumbs": crumbs,
        },
    )


@router.post("/{username}/{slug}/favorite")
async def favorite_gig(request: Request, username: str, slug: str):
    conn = db.connect()
    gig = _resolve_gig(conn, username, slug)
    if gig is None:
        raise HTTPException(status_code=404, detail="gig not found")
    user_id = auth.effective_buyer_id(request)
    with db.transaction() as tx:
        try:
            tx.execute(
                "INSERT INTO favorites (user_id, gig_id, created_at) "
                "VALUES (?, ?, ?)",
                (user_id, gig["id"], app_main.now_value()),
            )
        except Exception:  # noqa: BLE001
            tx.execute(
                "DELETE FROM favorites WHERE user_id=? AND gig_id=?",
                (user_id, gig["id"]),
            )
        db.log_audit(
            tx,
            occurred_at=app_main.now_value(),
            operation="favorite_toggled",
            target_type="gig",
            target_id=gig["id"],
            payload={"user_id": user_id},
        )
    return RedirectResponse(url=f"/{username}/{slug}", status_code=303)
