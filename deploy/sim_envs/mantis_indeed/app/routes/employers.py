"""/employers/* — dashboard + posting review + applicant status change."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from .. import auth, db, main as app_main

router = APIRouter()


def _format_salary(low: int, high: int, period: str) -> str:
    if not low and not high:
        return ""
    suffix = "a year" if period == "year" else "an hour"
    if low and high and low != high:
        return f"${low:,} - ${high:,} {suffix}"
    return f"${low or high:,} {suffix}"


@router.get("/employers/dashboard", response_class=HTMLResponse)
async def employer_dashboard(request: Request) -> Response:
    eid = auth.effective_employer_id(request)
    conn = db.connect()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (eid,)).fetchone()
    if user is None:
        return HTMLResponse("<h1>Not found</h1>", status_code=404)
    cid = user["company_id"]
    jobs = conn.execute(
        "SELECT * FROM jobs WHERE company_id = ? ORDER BY posted_at DESC",
        (cid,),
    ).fetchall()
    job_dicts = []
    for j in jobs:
        d = dict(j)
        d["app_counts"] = {
            "new": 0,
            "reviewed": 0,
            "rejected": 0,
            "hired": 0,
        }
        for r in conn.execute(
            "SELECT status, COUNT(*) AS n FROM applications "
            "WHERE job_id = ? GROUP BY status",
            (j["id"],),
        ).fetchall():
            d["app_counts"][r["status"]] = r["n"]
        d["app_total"] = sum(d["app_counts"].values())
        job_dicts.append(d)
    new_this_week = sum(d["app_counts"]["new"] for d in job_dicts)
    return request.app.state.templates.TemplateResponse(
        "employer_dashboard.html",
        {
            "request": request,
            "user": dict(user),
            "jobs": job_dicts,
            "kpi_active": len([j for j in job_dicts if j["status"] == "active"]),
            "kpi_new_apps": new_this_week,
            "kpi_views": 1234,  # synthetic
        },
    )


@router.get("/employers/jobs/{job_id}", response_class=HTMLResponse)
async def employer_posting_detail(job_id: str, request: Request) -> Response:
    conn = db.connect()
    job = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if job is None:
        return HTMLResponse("<h1>Posting not found</h1>", status_code=404)
    status_filter = request.query_params.get("status") or ""
    sql = (
        "SELECT a.*, u.name AS user_name, u.email AS user_email, "
        "r.title AS resume_title FROM applications a "
        "JOIN users u ON u.id = a.user_id "
        "LEFT JOIN resumes r ON r.id = a.resume_id "
        "WHERE a.job_id = ?"
    )
    params = [job_id]
    if status_filter in {"new", "reviewed", "rejected", "hired"}:
        sql += " AND a.status = ?"
        params.append(status_filter)
    sql += " ORDER BY a.applied_at DESC"
    apps = [dict(r) for r in conn.execute(sql, params).fetchall()]
    return request.app.state.templates.TemplateResponse(
        "employer_posting.html",
        {
            "request": request,
            "job": dict(job),
            "applications": apps,
            "status_filter": status_filter,
        },
    )


@router.post("/employers/applications/{app_id}/status")
async def employer_application_status(
    app_id: str,
    request: Request,
    status: str = Form(...),
) -> Response:
    if status not in {"new", "reviewed", "rejected", "hired"}:
        return HTMLResponse("invalid status", status_code=400)
    conn = db.connect()
    row = conn.execute(
        "SELECT job_id, status FROM applications WHERE id = ?", (app_id,),
    ).fetchone()
    if row is None:
        return HTMLResponse("application not found", status_code=404)
    previous = row["status"]
    now = app_main.now_value()
    reviewed_at = now if status != "new" else None
    conn.execute(
        "UPDATE applications SET status = ?, reviewed_at = ? WHERE id = ?",
        (status, reviewed_at, app_id),
    )
    db.log_audit(
        conn,
        occurred_at=now,
        operation="application_status_changed",
        target_type="application",
        target_id=app_id,
        payload={
            "previous_status": previous,
            "new_status": status,
            "reviewed_at": reviewed_at,
        },
    )
    conn.commit()
    next_url = request.headers.get("referer") or f"/employers/jobs/{row['job_id']}"
    return RedirectResponse(next_url, status_code=303)


@router.get("/employers/jobs/new", response_class=HTMLResponse)
async def employer_new_posting_form(request: Request) -> Response:
    return request.app.state.templates.TemplateResponse(
        "employer_new_posting.html",
        {"request": request},
    )


@router.post("/employers/jobs/new")
async def employer_new_posting_submit(
    request: Request,
    title: str = Form(...),
    location: str = Form(...),
    salary_low: int = Form(0),
    salary_high: int = Form(0),
    remote: str = Form(""),
    job_type: str = Form("Full-time"),
    description: str = Form(""),
) -> Response:
    eid = auth.effective_employer_id(request)
    conn = db.connect()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (eid,)).fetchone()
    if user is None:
        return HTMLResponse("not authorized", status_code=403)
    n = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    jid = f"job_new_{n + 1:05d}"
    jk = f"{n + 1:016x}"
    conn.execute(
        "INSERT INTO jobs (id, jk, title, company_id, location, salary_low, "
        "salary_high, salary_period, remote_flag, job_type, experience_level, "
        "posted_at, description, snippet, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 'year', ?, ?, 'Mid level', ?, ?, ?, 'active')",
        (jid, jk, title, user["company_id"], location, salary_low, salary_high,
         1 if remote in {"on", "1", "true"} else 0, job_type,
         app_main.now_value(), description, description[:240]),
    )
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="job_created",
        target_type="job",
        target_id=jid,
        payload={"title": title, "company_id": user["company_id"]},
    )
    conn.commit()
    return RedirectResponse(f"/employers/jobs/{jid}", status_code=303)
