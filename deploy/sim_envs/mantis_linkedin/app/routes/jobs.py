"""Jobs surface: home, search, detail, Easy Apply."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import db, main as app_main

router = APIRouter()


def _me_id(request: Request) -> str:
    me = request.state.current_user or {"id": "user_00001"}
    return me["id"]


def _job_view(row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "title": row["title"],
        "company": row["company"],
        "location": row["location"],
        "description_md": row["description_md"],
        "easy_apply": bool(row["easy_apply"]),
        "promoted": bool(row["promoted"]),
        "applicants": row["applicants"],
        "posted_at": row["posted_at"],
    }


@router.get("/jobs/", response_class=HTMLResponse)
async def jobs_home(request: Request):
    conn = db.connect()
    rows = conn.execute(
        "SELECT * FROM jobs ORDER BY promoted DESC, id LIMIT 8"
    ).fetchall()
    top_picks = [_job_view(r) for r in rows]
    recommended = [
        _job_view(r) for r in conn.execute(
            "SELECT * FROM jobs ORDER BY id DESC LIMIT 5"
        ).fetchall()
    ]
    return request.app.state.templates.TemplateResponse(
        "jobs.html",
        {
            "request": request,
            "top_picks": top_picks,
            "recommended": recommended,
        },
    )


@router.get("/jobs/search/", response_class=HTMLResponse)
async def jobs_search(
    request: Request,
    keywords: str = "",
    location: str = "",
    easy_apply: int = 0,
    id: str | None = None,
):
    conn = db.connect()

    sql = "SELECT * FROM jobs WHERE 1=1"
    args: list[Any] = []
    if keywords.strip():
        like = f"%{keywords.strip().lower()}%"
        sql += " AND (lower(title) LIKE ? OR lower(company) LIKE ?)"
        args.extend([like, like])
    if location.strip():
        sql += " AND lower(location) LIKE ?"
        args.append(f"%{location.strip().lower()}%")
    if easy_apply:
        sql += " AND easy_apply = 1"
    sql += " ORDER BY promoted DESC, id"
    rows = conn.execute(sql, args).fetchall()
    results = [_job_view(r) for r in rows]

    selected = None
    if id:
        for r in results:
            if r["id"] == id:
                selected = r
                break
    if selected is None and results:
        selected = results[0]

    return request.app.state.templates.TemplateResponse(
        "jobs_search.html",
        {
            "request": request,
            "results": results,
            "keywords": keywords,
            "location": location,
            "easy_apply": easy_apply,
            "selected": selected,
        },
    )


@router.get("/jobs/view/{job_id}/", response_class=HTMLResponse)
@router.get("/jobs/view/{job_id}", response_class=HTMLResponse)
async def job_view(request: Request, job_id: str):
    conn = db.connect()
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        return HTMLResponse("Job not found", status_code=404)
    me_id = _me_id(request)
    applied = conn.execute(
        "SELECT id FROM job_applications WHERE user_id = ? AND job_id = ?",
        (me_id, job_id),
    ).fetchone() is not None
    return request.app.state.templates.TemplateResponse(
        "job_view.html",
        {
            "request": request,
            "job": _job_view(row),
            "applied": applied,
        },
    )


@router.post("/jobs/{job_id}/apply")
async def apply_to_job(
    request: Request, job_id: str,
    phone: str = Form(...),
    resume_label: str = Form("Use my profile"),
    answers: str = Form("{}"),
):
    conn = db.connect()
    me_id = _me_id(request)
    now = app_main.now_value()

    job = conn.execute(
        "SELECT id, title, company, easy_apply FROM jobs WHERE id = ?",
        (job_id,),
    ).fetchone()
    if not job:
        return RedirectResponse(f"/jobs/view/{job_id}/", status_code=303)
    if not job["easy_apply"]:
        return RedirectResponse(f"/jobs/view/{job_id}/?error=external_apply",
                                status_code=303)
    if not phone.strip():
        return RedirectResponse(f"/jobs/view/{job_id}/?error=phone_required",
                                status_code=303)

    try:
        answers_obj = json.loads(answers or "{}")
        if not isinstance(answers_obj, dict):
            answers_obj = {"raw": str(answers)}
    except json.JSONDecodeError:
        answers_obj = {"raw": str(answers)}

    next_idx = conn.execute(
        "SELECT COALESCE(MAX(CAST(SUBSTR(id, 5) AS INTEGER)), 0) + 1 AS n "
        "FROM job_applications"
    ).fetchone()["n"]
    aid = f"app_{next_idx:05d}"

    try:
        conn.execute(
            "INSERT INTO job_applications (id, user_id, job_id, status, phone, "
            "resume_label, answers_json, submitted_at) VALUES (?,?,?,?,?,?,?,?)",
            (aid, me_id, job_id, "submitted", phone.strip(),
             resume_label.strip(), db.pack_json(answers_obj), now),
        )
    except Exception:
        # Already applied — surface the existing record but skip new audit.
        conn.commit()
        return RedirectResponse(f"/jobs/view/{job_id}/?applied=1",
                                status_code=303)

    db.log_audit(
        conn,
        occurred_at=now,
        operation="job_application_submitted",
        target_type="job_application",
        target_id=aid,
        payload={
            "user_id": me_id, "job_id": job_id,
            "phone": phone.strip(), "resume_label": resume_label.strip(),
            "answers": answers_obj,
        },
    )
    conn.commit()
    app_main.emit("job_application_submitted",
                  {"application_id": aid, "job_id": job_id, "user_id": me_id})
    app_main.emit_mutation(
        op="job_application_submitted", target_type="job_application",
        target_id=aid,
        payload={
            "user_id": me_id, "job_id": job_id, "phone": phone.strip(),
            "answers": answers_obj,
        },
    )
    return RedirectResponse(f"/jobs/view/{job_id}/?applied=1",
                            status_code=303)
