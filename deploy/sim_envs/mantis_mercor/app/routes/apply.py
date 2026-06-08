"""Multi-step apply flow — 5 steps, draft persisted in-memory.

Routes:

* ``GET /apply/<job_id>`` — redirects to step 1.
* ``GET/POST /apply/<job_id>/step/<n>`` — render + persist each step.
* ``POST /apply/<job_id>/submit`` — write the application row + audit.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import auth as auth_mod, db, main as app_main

router = APIRouter()


def _job(conn, job_id: str) -> dict | None:
    row = conn.execute(
        "SELECT j.*, c.name AS company_name FROM jobs j "
        "JOIN companies c ON c.id = j.company_id WHERE j.id = ?",
        (job_id,),
    ).fetchone()
    return dict(row) if row else None


@router.get("/apply/{job_id}")
async def apply_root(job_id: str):
    return RedirectResponse(f"/apply/{job_id}/step/1", status_code=303)


@router.get("/apply/{job_id}/step/{n}", response_class=HTMLResponse)
async def apply_step(request: Request, job_id: str, n: int):
    conn = db.connect()
    job = _job(conn, job_id)
    if job is None:
        return HTMLResponse("<h1>Role not found</h1>", status_code=404)

    user = auth_mod.effective_user(request, default_role="candidate")
    draft = app_main.get_draft(user["id"], job_id)
    screening_qs = json.loads(job.get("screening_qs") or "[]")

    return request.app.state.templates.TemplateResponse(
        "apply_step.html",
        {
            "request": request,
            "job": job,
            "draft": draft,
            "step": n,
            "screening_qs": screening_qs,
            "total_steps": 5,
        },
    )


@router.post("/apply/{job_id}/step/{n}")
async def apply_step_submit(
    request: Request,
    job_id: str,
    n: int,
):
    form = await request.form()
    user = auth_mod.effective_user(request, default_role="candidate")
    updates: dict = {}

    if n == 1:
        updates["headline"] = (form.get("headline") or "").strip()
        updates["skills"] = (form.get("skills") or "").strip()
        try:
            updates["hourly_rate"] = float(form.get("hourly_rate") or 0)
        except ValueError:
            updates["hourly_rate"] = 0.0
    elif n == 2:
        updates["resume_text"] = (form.get("resume_text") or "").strip()
    elif n == 3:
        # Screening answers — fields named answer_0, answer_1, ...
        conn = db.connect()
        job = _job(conn, job_id)
        n_qs = len(json.loads(job.get("screening_qs") or "[]")) if job else 0
        answers = []
        for i in range(n_qs):
            answers.append((form.get(f"answer_{i}") or "").strip())
        updates["answers"] = answers

    if updates:
        app_main.set_draft(user["id"], job_id, **updates)

    nav = (form.get("nav") or "next").strip()
    if nav == "back" and n > 1:
        return RedirectResponse(f"/apply/{job_id}/step/{n - 1}", status_code=303)
    if n >= 5:
        return RedirectResponse(
            f"/apply/{job_id}/submit", status_code=303,
        )
    return RedirectResponse(f"/apply/{job_id}/step/{n + 1}", status_code=303)


@router.get("/apply/{job_id}/submit", response_class=HTMLResponse)
async def apply_submit_get(request: Request, job_id: str):
    # Render the review screen with a Submit button.
    conn = db.connect()
    job = _job(conn, job_id)
    if job is None:
        return HTMLResponse("<h1>Role not found</h1>", status_code=404)
    user = auth_mod.effective_user(request, default_role="candidate")
    draft = app_main.get_draft(user["id"], job_id)
    screening_qs = json.loads(job.get("screening_qs") or "[]")
    return request.app.state.templates.TemplateResponse(
        "apply_review.html",
        {
            "request": request,
            "job": job,
            "draft": draft,
            "screening_qs": screening_qs,
        },
    )


@router.post("/apply/{job_id}/submit", response_class=HTMLResponse)
async def apply_submit_post(request: Request, job_id: str):
    conn_check = db.connect()
    job = _job(conn_check, job_id)
    if job is None:
        return HTMLResponse("<h1>Role not found</h1>", status_code=404)

    user = auth_mod.effective_user(request, default_role="candidate")
    draft = app_main.get_draft(user["id"], job_id)
    now = app_main.now_value()

    skills_list = [s.strip() for s in (draft.get("skills") or "").split(",") if s.strip()]
    answers = draft.get("answers") or []
    headline = draft.get("headline") or ""
    hourly_rate = float(draft.get("hourly_rate") or 0)
    resume_text = draft.get("resume_text") or ""

    app_id = f"app_run_{abs(hash((user['id'], job_id))) % 1_000_000:06d}"

    with db.transaction() as conn:
        existing = conn.execute(
            "SELECT id FROM applications WHERE job_id=? AND candidate_id=?",
            (job_id, user["id"]),
        ).fetchone()
        if existing:
            # Upsert: update existing application to submitted with fresh fields.
            app_id = existing["id"]
            conn.execute(
                "UPDATE applications SET status='submitted', headline=?, "
                "skills_json=?, hourly_rate=?, resume_text=?, screening_answers=?, "
                "submitted_at=?, updated_at=? WHERE id=?",
                (
                    headline,
                    db.pack_json(skills_list),
                    hourly_rate,
                    resume_text,
                    db.pack_json(answers),
                    now,
                    now,
                    app_id,
                ),
            )
        else:
            conn.execute(
                "INSERT INTO applications (id, job_id, candidate_id, status, "
                "headline, skills_json, hourly_rate, resume_text, "
                "screening_answers, reject_reason, submitted_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    app_id, job_id, user["id"], "submitted",
                    headline,
                    db.pack_json(skills_list),
                    hourly_rate,
                    resume_text,
                    db.pack_json(answers),
                    "",
                    now,
                    now,
                ),
            )

        db.log_audit(
            conn,
            occurred_at=now,
            operation="application_submitted",
            target_type="application",
            target_id=app_id,
            payload={
                "job_id": job_id,
                "candidate_id": user["id"],
                "screening_answers": answers,
                "headline": headline,
            },
        )

    app_main.clear_draft(user["id"], job_id)
    app_main.emit("application_submitted",
                  {"application_id": app_id, "job_id": job_id})

    return request.app.state.templates.TemplateResponse(
        "apply_confirm.html",
        {"request": request, "job": job, "application_id": app_id},
    )
