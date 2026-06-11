"""Stores — `/stores`."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import db, main as app_main

router = APIRouter()


TAB_KIND_MAP = {
    "all": None,
    "client_transfer": "client_transfer",
    "collaborator": "collaborator",
    "archived": "archived",
    "inactive": "inactive",
}


@router.get("/stores", response_class=HTMLResponse)
async def stores(request: Request, tab: str = "all", q: str = "",
                 status: str = "", sort: str = "last_login"):
    templates = request.app.state.templates
    conn = db.connect()

    where = []
    params: list = []

    tab_norm = tab if tab in TAB_KIND_MAP else "all"
    kind_filter = TAB_KIND_MAP[tab_norm]
    if tab_norm == "archived":
        where.append("status = ?"); params.append("archived")
    elif tab_norm == "inactive":
        where.append("status = ?"); params.append("inactive")
    elif kind_filter:
        where.append("kind = ? AND status='active'"); params.append(kind_filter)
    else:
        where.append("status='active'")

    if q.strip():
        where.append("(name LIKE ? OR slug LIKE ?)")
        params.extend([f"%{q.strip()}%", f"%{q.strip()}%"])
    if status.strip():
        where.append("plan = ?")
        params.append(status.strip())

    where_clause = " AND ".join(where) or "1=1"
    order = "last_login_at DESC" if sort != "name" else "name COLLATE NOCASE ASC"
    rows = [dict(r) for r in conn.execute(
        f"SELECT * FROM stores WHERE {where_clause} ORDER BY {order}",
        params,
    ).fetchall()]

    if q.strip() or status.strip():
        db.log_audit(
            conn,
            occurred_at=app_main.now_value(),
            operation="stores_filter_applied",
            target_type="stores",
            target_id="search",
            payload={"q": q.strip(), "status": status.strip(),
                     "tab": tab_norm, "matched": len(rows)},
        )
        conn.commit()
        app_main.emit("stores_filter_applied", {"q": q, "status": status})

    return templates.TemplateResponse(
        "stores.html",
        {
            "request": request,
            "active_section": "stores",
            "rows": rows,
            "current_tab": tab_norm,
            "q": q,
            "status_filter": status,
            "sort": sort,
            "n": len(rows),
        },
    )


@router.post("/stores/{store_id}/login")
async def store_login(store_id: str, request: Request):
    conn = db.connect()
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="store_login_followed",
        target_type="store",
        target_id=store_id,
        payload={},
    )
    conn.commit()
    app_main.emit("store_login_followed", {"store_id": store_id})
    return RedirectResponse(f"/store/{store_id}/admin", status_code=303)


@router.post("/stores/{store_id}/archive")
async def store_archive(store_id: str, request: Request):
    conn = db.connect()
    conn.execute(
        "UPDATE stores SET status='archived' WHERE id=?", (store_id,),
    )
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="store_archived",
        target_type="store",
        target_id=store_id,
        payload={},
    )
    conn.commit()
    app_main.emit("store_archived", {"store_id": store_id})
    return RedirectResponse("/stores", status_code=303)


@router.get("/stores/{store_id}", response_class=HTMLResponse)
async def store_detail(store_id: str, request: Request):
    templates = request.app.state.templates
    conn = db.connect()
    row = conn.execute(
        "SELECT * FROM stores WHERE id=?", (store_id,),
    ).fetchone()
    if row is None:
        return HTMLResponse("Store not found", status_code=404)
    store = dict(row)

    # Synthesise a deterministic activity feed from audit_log + a few
    # store-anchored derived rows so the page isn't bare for fresh data.
    activity = [
        {
            "ts": r["occurred_at"],
            "operation": r["operation"],
            "payload": r["payload_json"],
        }
        for r in conn.execute(
            "SELECT occurred_at, operation, payload_json FROM audit_log "
            "WHERE target_id=? ORDER BY id DESC LIMIT 25",
            (store_id,),
        ).fetchall()
    ]

    # Team members with access — owner first, then a small slice of staff.
    members = [dict(r) for r in conn.execute(
        "SELECT id, name, avatar_color FROM users WHERE role='owner' "
        "ORDER BY name LIMIT 1"
    ).fetchall()]
    members += [dict(r) for r in conn.execute(
        "SELECT id, name, avatar_color FROM users WHERE role != 'owner' "
        "AND status='active' ORDER BY name LIMIT 2"
    ).fetchall()]

    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="store_viewed",
        target_type="store",
        target_id=store_id,
        payload={"name": store["name"], "kind": store["kind"]},
    )
    conn.commit()
    app_main.emit("store_viewed", {"store_id": store_id})

    return templates.TemplateResponse(
        "store_detail.html",
        {
            "request": request,
            "active_section": "stores",
            "store": store,
            "activity": activity,
            "members": members,
            "support_email": "support@" + store["slug"].replace("-", "") + ".example",
            "country": "United States",
            "created_human": "May 26, 2022",
        },
    )


@router.get("/stores/new", response_class=HTMLResponse)
async def store_new_form(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "store_new.html",
        {"request": request, "active_section": "stores"},
    )


@router.post("/stores/new")
async def store_new_submit(
    request: Request,
    name: str = Form(...),
    slug: str = Form(""),
    kind: str = Form("client_transfer"),
):
    conn = db.connect()
    sid = f"store_user_{int(conn.execute('SELECT COUNT(*) FROM stores').fetchone()[0]) + 1:05d}"
    eff_slug = slug.strip() or name.lower().replace(' ', '-')[:32]
    conn.execute(
        "INSERT INTO stores (id, name, slug, kind, status, last_login_at, plan) "
        "VALUES (?,?,?,?,?,?,?)",
        (sid, name.strip(), eff_slug, kind, "active",
         app_main.now_value(), "Basic"),
    )
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="store_added",
        target_type="store",
        target_id=sid,
        payload={"name": name, "slug": eff_slug, "kind": kind},
    )
    conn.commit()
    app_main.emit("store_added", {"store_id": sid})
    return RedirectResponse("/stores", status_code=303)
