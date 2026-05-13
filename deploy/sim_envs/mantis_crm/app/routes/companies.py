"""Company list + detail."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from .. import db, main as app_main

router = APIRouter()

PAGE_SIZE = 50


@router.get("/companies", response_class=HTMLResponse)
async def list_companies(request: Request) -> HTMLResponse:
    page = max(1, int(request.query_params.get("page") or 1))
    q = (request.query_params.get("q") or "").strip()
    industry = (request.query_params.get("industry") or "").strip()

    conn = db.connect()
    where = ["1=1"]
    args: list = []
    if q:
        where.append("(name LIKE ? OR domain LIKE ?)")
        args += [f"%{q}%", f"%{q}%"]
    if industry:
        where.append("industry = ?")
        args.append(industry)
    where_clause = " AND ".join(where)

    total = conn.execute(
        f"SELECT COUNT(*) FROM companies WHERE {where_clause}", args
    ).fetchone()[0]
    rows = conn.execute(
        f"SELECT * FROM companies WHERE {where_clause} ORDER BY id "
        f"LIMIT ? OFFSET ?",
        args + [PAGE_SIZE, (page - 1) * PAGE_SIZE],
    ).fetchall()

    pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    return app_main.app.state.templates.TemplateResponse(
        "companies_list.html",
        {
            "request": request,
            "companies": [dict(r) for r in rows],
            "total": total, "page": page, "pages": pages,
            "filters": {"q": q, "industry": industry},
        },
    )


@router.get("/companies/{company_id}", response_class=HTMLResponse)
async def company_detail(request: Request, company_id: str) -> HTMLResponse:
    conn = db.connect()
    row = conn.execute("SELECT * FROM companies WHERE id = ?", (company_id,)).fetchone()
    if row is None:
        return HTMLResponse("Not found", status_code=404)
    contacts = [
        dict(c) for c in conn.execute(
            "SELECT id, name, email, lifecycle_stage FROM contacts "
            "WHERE company_id = ? AND deleted_at IS NULL ORDER BY id LIMIT 200",
            (company_id,),
        ).fetchall()
    ]
    deals = [
        dict(d) for d in conn.execute(
            "SELECT id, name, stage, amount, expected_close FROM deals "
            "WHERE company_id = ? ORDER BY id LIMIT 200",
            (company_id,),
        ).fetchall()
    ]
    return app_main.app.state.templates.TemplateResponse(
        "company_detail.html",
        {
            "request": request,
            "company": dict(row),
            "contacts": contacts,
            "deals": deals,
        },
    )
