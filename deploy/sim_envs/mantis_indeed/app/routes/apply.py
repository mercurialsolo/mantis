"""Easy Apply (`/apply/<jk>`) — 3 steps + review + submit."""

from __future__ import annotations

import json

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from .. import auth, db, main as app_main

router = APIRouter()


def _job_by_jk(conn, jk: str):
    return conn.execute("SELECT * FROM jobs WHERE jk = ?", (jk,)).fetchone()


def _user_resumes(conn, uid: str):
    return [dict(r) for r in conn.execute(
        "SELECT * FROM resumes WHERE user_id = ? AND deleted_at IS NULL "
        "ORDER BY id",
        (uid,),
    ).fetchall()]


@router.get("/apply/{jk}", response_class=HTMLResponse)
async def apply_step1(jk: str, request: Request) -> Response:
    """Step 1 — contact information."""
    conn = db.connect()
    job_row = _job_by_jk(conn, jk)
    if job_row is None:
        return HTMLResponse("<h1>Job not found</h1>", status_code=404)
    uid = auth.effective_user_id(request)
    user = conn.execute(
        "SELECT * FROM users WHERE id = ?", (uid,),
    ).fetchone()
    user = dict(user) if user else {}
    return request.app.state.templates.TemplateResponse(
        "apply_step1.html",
        {
            "request": request,
            "job": dict(job_row),
            "user": user,
            "step": 1,
        },
    )


@router.post("/apply/{jk}/step2")
async def apply_step2_submit(
    jk: str,
    request: Request,
    full_name: str = Form(""),
    email: str = Form(""),
    phone: str = Form(""),
    city: str = Form(""),
    state: str = Form(""),
    zip: str = Form(""),
) -> Response:
    """Persist step 1 to session-like in-memory store keyed by user+jk."""
    state_dict = request.app.state
    if not hasattr(state_dict, "_apply_drafts"):
        state_dict._apply_drafts = {}
    uid = auth.effective_user_id(request)
    state_dict._apply_drafts[(uid, jk)] = {
        "full_name": full_name,
        "email": email,
        "phone": phone,
        "city": city,
        "state": state,
        "zip": zip,
    }
    return RedirectResponse(f"/apply/{jk}/resume", status_code=303)


@router.get("/apply/{jk}/resume", response_class=HTMLResponse)
async def apply_resume(jk: str, request: Request) -> Response:
    conn = db.connect()
    job_row = _job_by_jk(conn, jk)
    if job_row is None:
        return HTMLResponse("<h1>Job not found</h1>", status_code=404)
    uid = auth.effective_user_id(request)
    return request.app.state.templates.TemplateResponse(
        "apply_step2.html",
        {
            "request": request,
            "job": dict(job_row),
            "resumes": _user_resumes(conn, uid),
            "step": 2,
        },
    )


@router.post("/apply/{jk}/questions")
async def apply_questions_submit(
    jk: str,
    request: Request,
    resume_id: str = Form(""),
) -> Response:
    state_dict = request.app.state
    if not hasattr(state_dict, "_apply_drafts"):
        state_dict._apply_drafts = {}
    uid = auth.effective_user_id(request)
    draft = state_dict._apply_drafts.setdefault((uid, jk), {})
    draft["resume_id"] = resume_id
    return RedirectResponse(f"/apply/{jk}/questions", status_code=303)


@router.get("/apply/{jk}/questions", response_class=HTMLResponse)
async def apply_questions(jk: str, request: Request) -> Response:
    conn = db.connect()
    job_row = _job_by_jk(conn, jk)
    if job_row is None:
        return HTMLResponse("<h1>Job not found</h1>", status_code=404)
    return request.app.state.templates.TemplateResponse(
        "apply_step3.html",
        {
            "request": request,
            "job": dict(job_row),
            "step": 3,
        },
    )


@router.post("/apply/{jk}/review")
async def apply_review_submit(
    jk: str,
    request: Request,
    yrs_experience: str = Form("0"),
    auth_to_work: str = Form("Yes"),
    sponsorship: str = Form("No"),
) -> Response:
    state_dict = request.app.state
    if not hasattr(state_dict, "_apply_drafts"):
        state_dict._apply_drafts = {}
    uid = auth.effective_user_id(request)
    draft = state_dict._apply_drafts.setdefault((uid, jk), {})
    draft["answers"] = {
        "yrs_experience": yrs_experience,
        "auth_to_work": auth_to_work,
        "sponsorship": sponsorship,
    }
    return RedirectResponse(f"/apply/{jk}/review", status_code=303)


@router.get("/apply/{jk}/review", response_class=HTMLResponse)
async def apply_review(jk: str, request: Request) -> Response:
    conn = db.connect()
    job_row = _job_by_jk(conn, jk)
    if job_row is None:
        return HTMLResponse("<h1>Job not found</h1>", status_code=404)
    uid = auth.effective_user_id(request)
    state_dict = request.app.state
    draft = getattr(state_dict, "_apply_drafts", {}).get((uid, jk), {})
    return request.app.state.templates.TemplateResponse(
        "apply_review.html",
        {
            "request": request,
            "job": dict(job_row),
            "draft": draft,
            "step": 4,
        },
    )


@router.post("/apply/{jk}/submit")
async def apply_submit(jk: str, request: Request) -> Response:
    conn = db.connect()
    job_row = _job_by_jk(conn, jk)
    if job_row is None:
        return HTMLResponse("<h1>Job not found</h1>", status_code=404)
    uid = auth.effective_user_id(request)
    state_dict = request.app.state
    draft = getattr(state_dict, "_apply_drafts", {}).get((uid, jk), {})

    job_id = job_row["id"]
    existing = conn.execute(
        "SELECT id FROM applications WHERE user_id = ? AND job_id = ?",
        (uid, job_id),
    ).fetchone()
    if existing:
        # Update phone+answers idempotently.
        conn.execute(
            "UPDATE applications SET phone = ?, answers_json = ?, "
            "resume_id = ?, applied_at = ? WHERE id = ?",
            (draft.get("phone", ""),
             json.dumps(draft.get("answers", {}), sort_keys=True),
             draft.get("resume_id") or None,
             app_main.now_value(),
             existing["id"]),
        )
        app_id = existing["id"]
    else:
        # Generate a fresh id by counting.
        n = conn.execute("SELECT COUNT(*) FROM applications").fetchone()[0]
        app_id = f"application_run_{n + 1:05d}"
        conn.execute(
            "INSERT INTO applications (id, user_id, job_id, resume_id, "
            "phone, answers_json, status, applied_at, reviewed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'new', ?, NULL)",
            (app_id, uid, job_id, draft.get("resume_id") or None,
             draft.get("phone", ""),
             json.dumps(draft.get("answers", {}), sort_keys=True),
             app_main.now_value()),
        )
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="application_submitted",
        target_type="application",
        target_id=app_id,
        payload={
            "user_id": uid,
            "job_id": job_id,
            "jk": jk,
            "resume_id": draft.get("resume_id"),
            "phone": draft.get("phone", ""),
            "answers": draft.get("answers", {}),
        },
    )
    conn.commit()
    return RedirectResponse(f"/apply/{jk}/confirm", status_code=303)


@router.get("/apply/{jk}/confirm", response_class=HTMLResponse)
async def apply_confirm(jk: str, request: Request) -> Response:
    conn = db.connect()
    job_row = _job_by_jk(conn, jk)
    if job_row is None:
        return HTMLResponse("<h1>Job not found</h1>", status_code=404)
    return request.app.state.templates.TemplateResponse(
        "apply_confirm.html",
        {"request": request, "job": dict(job_row)},
    )
