"""Feed surface: /feed/, post create, react, comment."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import db, main as app_main
from ..util import extract_hashtags, truncate

router = APIRouter()


def _author_chip(conn, user_id: str) -> dict[str, Any]:
    r = conn.execute(
        "SELECT id, name, headline, handle, avatar_color FROM users WHERE id=?",
        (user_id,),
    ).fetchone()
    return dict(r) if r else {
        "id": user_id, "name": "Unknown", "headline": "",
        "handle": "unknown", "avatar_color": "#666",
    }


_ADVERTISER_COLORS = {
    "Vercel": "#000000",
    "Stripe": "#635bff",
    "Notion": "#191919",
    "AWS":    "#ff9900",
    "Figma":  "#a259ff",
}


def _post_view(conn, row) -> dict[str, Any]:
    keys = row.keys() if hasattr(row, "keys") else []
    sponsored = bool(row["sponsored"]) if "sponsored" in keys else False
    if sponsored:
        adv = row["advertiser"]
        author = {
            "id": f"adv_{adv.lower()}",
            "name": adv,
            "handle": f"company-{adv.lower()}",
            "headline": "Promoted",
            "avatar_color": _ADVERTISER_COLORS.get(adv, "#666"),
        }
    else:
        author = _author_chip(conn, row["author_id"])
    reactions = conn.execute(
        "SELECT COUNT(*) AS n FROM reactions WHERE post_id = ?",
        (row["id"],),
    ).fetchone()["n"]
    comments = conn.execute(
        "SELECT COUNT(*) AS n FROM comments WHERE post_id = ?",
        (row["id"],),
    ).fetchone()["n"]
    media = db.unpack_json(row["media"]) if "media" in keys else []
    return {
        "id": row["id"],
        "author": author,
        "body": row["body"],
        "body_short": truncate(row["body"], max_chars=320),
        "hashtags": db.unpack_json(row["hashtags"]),
        "created_at": row["created_at"],
        "reactions": reactions,
        "comments": comments,
        "media": media,
        "sponsored": sponsored,
        "advertiser": row["advertiser"] if "advertiser" in keys else "",
        "cta_label": row["cta_label"] if "cta_label" in keys else "",
        "cta_url": row["cta_url"] if "cta_url" in keys else "",
    }


def _suggested_connections(conn, current_user_id: str, limit: int = 3):
    """Users we don't have a connection edge to."""
    rows = conn.execute(
        "SELECT id, name, headline, handle, avatar_color FROM users "
        "WHERE id != ? "
        "AND id NOT IN (SELECT to_user_id FROM connections WHERE from_user_id = ?) "
        "AND id NOT IN (SELECT from_user_id FROM connections WHERE to_user_id = ?) "
        "ORDER BY id LIMIT ?",
        (current_user_id, current_user_id, current_user_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


@router.get("/feed/", response_class=HTMLResponse)
async def feed(request: Request):
    conn = db.connect()
    user = request.state.current_user or {}
    user_id = user.get("id", "user_00001")

    organic_rows = conn.execute(
        "SELECT id, author_id, body, hashtags, created_at, media, sponsored, "
        "advertiser, cta_label, cta_url FROM posts "
        "WHERE sponsored = 0 ORDER BY created_at DESC, id DESC LIMIT 25"
    ).fetchall()
    sponsored_rows = conn.execute(
        "SELECT id, author_id, body, hashtags, created_at, media, sponsored, "
        "advertiser, cta_label, cta_url FROM posts "
        "WHERE sponsored = 1 ORDER BY id"
    ).fetchall()
    organic = [_post_view(conn, r) for r in organic_rows]
    sponsored_views = [_post_view(conn, r) for r in sponsored_rows]
    # Interleave sponsored posts at fixed indexes (mirrors LinkedIn's
    # ~every-4-posts cadence). Insert from the end so earlier indexes
    # stay valid as we splice.
    posts = list(organic)
    sponsored_slots = [2, 6, 10, 14, 18]
    for slot, sp in zip(reversed(sponsored_slots), reversed(sponsored_views)):
        if slot <= len(posts):
            posts.insert(slot, sp)
        else:
            posts.append(sp)
    connection_count = conn.execute(
        "SELECT COUNT(*) AS n FROM connections "
        "WHERE (from_user_id = ? OR to_user_id = ?) AND status = 'accepted'",
        (user_id, user_id),
    ).fetchone()["n"]

    suggested = _suggested_connections(conn, user_id)
    me = _author_chip(conn, user_id)

    return request.app.state.templates.TemplateResponse(
        "feed.html",
        {
            "request": request,
            "me": me,
            "posts": posts,
            "connection_count": connection_count,
            "suggested": suggested,
            "news_items": [
                ("New funding rounds for AI infra startups", "2h ago • 4,201 readers"),
                ("Remote work trends — Q2 2026 report", "5h ago • 2,810 readers"),
                ("Why postmortem culture matters more than ever", "1d ago • 6,144 readers"),
                ("Hiring slowed in May; tech roles steady", "1d ago • 1,902 readers"),
                ("Open-source funding models that actually work", "2d ago • 980 readers"),
            ],
        },
    )


@router.post("/feed/post")
async def create_post(request: Request, body: str = Form(...)):
    conn = db.connect()
    user = request.state.current_user or {}
    user_id = user.get("id", "user_00001")

    text = (body or "").strip()
    if not text:
        return RedirectResponse("/feed/", status_code=303)

    # Generate a fresh deterministic-ish ID.
    next_idx = conn.execute(
        "SELECT COALESCE(MAX(CAST(SUBSTR(id, 6) AS INTEGER)), 0) + 1 AS n "
        "FROM posts"
    ).fetchone()["n"]
    pid = f"post_{next_idx:05d}"
    tags = extract_hashtags(text)
    created_at = app_main.now_value()

    conn.execute(
        "INSERT INTO posts (id, author_id, body, hashtags, visibility, "
        "created_at) VALUES (?,?,?,?,?,?)",
        (pid, user_id, text, db.pack_json(tags), "public", created_at),
    )
    db.log_audit(
        conn,
        occurred_at=created_at,
        operation="post_created",
        target_type="post",
        target_id=pid,
        payload={"author_id": user_id, "body": text, "hashtags": tags},
    )
    conn.commit()
    app_main.emit("post_created", {"post_id": pid, "user_id": user_id})
    app_main.emit_mutation(
        op="post_created", target_type="post", target_id=pid,
        payload={"author_id": user_id, "body": text, "hashtags": tags},
    )
    return RedirectResponse("/feed/", status_code=303)


@router.post("/feed/posts/{post_id}/react")
async def react_to_post(
    request: Request, post_id: str,
    kind: str = Form("like"),
):
    conn = db.connect()
    user = request.state.current_user or {}
    user_id = user.get("id", "user_00001")
    now = app_main.now_value()

    # Toggle: if existing row matches kind, remove. Else upsert.
    existing = conn.execute(
        "SELECT kind FROM reactions WHERE post_id = ? AND user_id = ?",
        (post_id, user_id),
    ).fetchone()
    if existing and existing["kind"] == kind:
        conn.execute(
            "DELETE FROM reactions WHERE post_id = ? AND user_id = ?",
            (post_id, user_id),
        )
        op = "reaction_removed"
    else:
        if existing:
            conn.execute(
                "UPDATE reactions SET kind = ?, created_at = ? "
                "WHERE post_id = ? AND user_id = ?",
                (kind, now, post_id, user_id),
            )
        else:
            conn.execute(
                "INSERT INTO reactions (post_id, user_id, kind, created_at) "
                "VALUES (?,?,?,?)",
                (post_id, user_id, kind, now),
            )
        op = "reaction_added"

    db.log_audit(
        conn,
        occurred_at=now,
        operation=op,
        target_type="post",
        target_id=post_id,
        payload={"user_id": user_id, "kind": kind},
    )
    conn.commit()
    app_main.emit(op, {"post_id": post_id, "user_id": user_id, "kind": kind})
    app_main.emit_mutation(
        op=op, target_type="post", target_id=post_id,
        payload={"user_id": user_id, "kind": kind},
    )
    return RedirectResponse("/feed/", status_code=303)


@router.post("/feed/posts/{post_id}/comment")
async def comment_on_post(
    request: Request, post_id: str,
    body: str = Form(...),
):
    conn = db.connect()
    user = request.state.current_user or {}
    user_id = user.get("id", "user_00001")
    text = (body or "").strip()
    if not text:
        return RedirectResponse("/feed/", status_code=303)
    now = app_main.now_value()

    next_idx = conn.execute(
        "SELECT COALESCE(MAX(CAST(SUBSTR(id, 5) AS INTEGER)), 0) + 1 AS n "
        "FROM comments"
    ).fetchone()["n"]
    cid = f"com_{next_idx:06d}"

    conn.execute(
        "INSERT INTO comments (id, post_id, author_id, body, created_at) "
        "VALUES (?,?,?,?,?)",
        (cid, post_id, user_id, text, now),
    )
    db.log_audit(
        conn,
        occurred_at=now,
        operation="comment_added",
        target_type="post",
        target_id=post_id,
        payload={"comment_id": cid, "author_id": user_id, "body": text},
    )
    conn.commit()
    app_main.emit("comment_added", {"post_id": post_id, "comment_id": cid})
    app_main.emit_mutation(
        op="comment_added", target_type="post", target_id=post_id,
        payload={"comment_id": cid, "author_id": user_id, "body": text},
    )
    return RedirectResponse("/feed/", status_code=303)
