"""/resumes — resume manager."""

from __future__ import annotations

import json

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from .. import auth, db, main as app_main

router = APIRouter()


@router.get("/resumes", response_class=HTMLResponse)
async def resumes_list(request: Request) -> Response:
    conn = db.connect()
    uid = auth.effective_user_id(request)
    rows = conn.execute(
        "SELECT * FROM resumes WHERE user_id = ? AND deleted_at IS NULL "
        "ORDER BY id",
        (uid,),
    ).fetchall()
    resumes = []
    for r in rows:
        d = dict(r)
        try:
            d["skills_list"] = json.loads(d.get("skills") or "[]")
        except json.JSONDecodeError:
            d["skills_list"] = []
        resumes.append(d)
    return request.app.state.templates.TemplateResponse(
        "resumes.html",
        {"request": request, "resumes": resumes},
    )


@router.post("/resumes/new")
async def resumes_new(
    request: Request,
    title: str = Form("New resume"),
    summary: str = Form(""),
    skills: str = Form(""),
    experience: str = Form(""),
) -> Response:
    conn = db.connect()
    uid = auth.effective_user_id(request)
    n = conn.execute("SELECT COUNT(*) FROM resumes").fetchone()[0]
    rid = f"resume_run_{n + 1:05d}"
    skills_list = [s.strip() for s in skills.split(",") if s.strip()]
    conn.execute(
        "INSERT INTO resumes (id, user_id, title, summary, skills, experience, "
        "updated_at, deleted_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
        (rid, uid, title, summary, json.dumps(skills_list), experience,
         app_main.now_value()),
    )
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="resume_created",
        target_type="resume",
        target_id=rid,
        payload={"user_id": uid, "title": title},
    )
    conn.commit()
    return RedirectResponse("/resumes", status_code=303)


@router.post("/resumes/{rid}/delete")
async def resumes_delete(rid: str, request: Request) -> Response:
    conn = db.connect()
    uid = auth.effective_user_id(request)
    conn.execute(
        "UPDATE resumes SET deleted_at = ? WHERE id = ? AND user_id = ?",
        (app_main.now_value(), rid, uid),
    )
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="resume_deleted",
        target_type="resume",
        target_id=rid,
        payload={"user_id": uid},
    )
    conn.commit()
    return RedirectResponse("/resumes", status_code=303)
