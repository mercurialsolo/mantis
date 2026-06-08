"""Jobs list + detail routes."""

from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from .. import db

router = APIRouter()


@router.get("/jobs", response_class=HTMLResponse)
async def jobs_list(
    request: Request,
    category: str | None = None,
    engagement: str | None = None,
    rate_min: int | None = None,
    rate_max: int | None = None,
):
    templates = request.app.state.templates
    conn = db.connect()

    sql = (
        "SELECT j.*, c.name AS company_name "
        "FROM jobs j JOIN companies c ON c.id = j.company_id "
        "WHERE j.status = 'open'"
    )
    params: list = []
    if category:
        sql += " AND j.category = ?"
        params.append(category)
    if engagement:
        sql += " AND j.engagement = ?"
        params.append(engagement)
    if rate_min is not None:
        sql += " AND j.rate_max >= ?"
        params.append(rate_min)
    if rate_max is not None:
        sql += " AND j.rate_min <= ?"
        params.append(rate_max)
    sql += " ORDER BY j.id"

    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    categories = ["Medical", "Legal", "Finance", "Software", "Consulting", "Office"]

    return templates.TemplateResponse(
        "jobs_list.html",
        {
            "request": request,
            "jobs": rows,
            "categories": categories,
            "filter_category": category or "",
            "filter_engagement": engagement or "",
            "filter_rate_min": rate_min,
            "filter_rate_max": rate_max,
        },
    )


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_detail(request: Request, job_id: str):
    templates = request.app.state.templates
    conn = db.connect()
    row = conn.execute(
        "SELECT j.*, c.name AS company_name "
        "FROM jobs j JOIN companies c ON c.id = j.company_id "
        "WHERE j.id = ?",
        (job_id,),
    ).fetchone()
    if row is None:
        return HTMLResponse("<h1>Role not found</h1>", status_code=404)
    job = dict(row)
    job["skills"] = json.loads(job.get("skills_json") or "[]")
    job["screening_questions"] = json.loads(job.get("screening_qs") or "[]")
    return templates.TemplateResponse(
        "job_detail.html",
        {"request": request, "job": job},
    )
