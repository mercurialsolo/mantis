"""Dashboard — role-sensitive (candidate vs client)."""

from __future__ import annotations

import json

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import auth as auth_mod, db, main as app_main

router = APIRouter()


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    conn = db.connect()
    user = auth_mod.effective_user(request, default_role="candidate")
    templates = request.app.state.templates

    if user.get("role") == "client":
        company_id = user.get("company_id") or ""
        # Pending applications for jobs owned by my company.
        pending = [
            dict(r) for r in conn.execute(
                "SELECT a.*, j.title AS job_title, u.name AS candidate_name "
                "FROM applications a "
                "JOIN jobs j ON j.id = a.job_id "
                "JOIN users u ON u.id = a.candidate_id "
                "WHERE j.company_id = ? AND a.status IN ('submitted', 'under_review') "
                "ORDER BY a.updated_at DESC LIMIT 25",
                (company_id,),
            ).fetchall()
        ]
        shortlist = [
            dict(r) for r in conn.execute(
                "SELECT s.*, j.title AS job_title, u.name AS candidate_name "
                "FROM shortlist_entries s "
                "JOIN jobs j ON j.id = s.job_id "
                "JOIN users u ON u.id = s.candidate_id "
                "WHERE s.added_by = ? "
                "ORDER BY s.created_at DESC LIMIT 5",
                (user["id"],),
            ).fetchall()
        ]
        return templates.TemplateResponse(
            "dashboard_client.html",
            {"request": request, "user": user, "pending": pending,
             "shortlist": shortlist},
        )

    # Candidate view
    my_apps = [
        dict(r) for r in conn.execute(
            "SELECT a.*, j.title AS job_title, c.name AS company_name "
            "FROM applications a "
            "JOIN jobs j ON j.id = a.job_id "
            "JOIN companies c ON c.id = j.company_id "
            "WHERE a.candidate_id = ? "
            "ORDER BY a.updated_at DESC",
            (user["id"],),
        ).fetchall()
    ]
    return templates.TemplateResponse(
        "dashboard_candidate.html",
        {"request": request, "user": user, "applications": my_apps},
    )


@router.post("/dashboard/shortlist")
async def add_shortlist(
    request: Request,
    job_id: str = Form(...),
    candidate_id: str = Form(...),
):
    user = auth_mod.effective_user(request, default_role="client")
    if user.get("role") not in {"client", "admin"}:
        return RedirectResponse("/dashboard", status_code=303)
    now = app_main.now_value()
    entry_id = f"shortlist_run_{abs(hash((job_id, candidate_id))) % 1_000_000:06d}"

    with db.transaction() as conn:
        existing = conn.execute(
            "SELECT id FROM shortlist_entries WHERE job_id=? AND candidate_id=?",
            (job_id, candidate_id),
        ).fetchone()
        if existing:
            entry_id = existing["id"]
        else:
            conn.execute(
                "INSERT INTO shortlist_entries (id, job_id, candidate_id, "
                "added_by, created_at) VALUES (?,?,?,?,?)",
                (entry_id, job_id, candidate_id, user["id"], now),
            )
        db.log_audit(
            conn,
            occurred_at=now,
            operation="candidate_shortlisted",
            target_type="shortlist_entry",
            target_id=entry_id,
            payload={
                "job_id": job_id,
                "candidate_id": candidate_id,
                "added_by": user["id"],
            },
        )
    app_main.emit("candidate_shortlisted",
                  {"job_id": job_id, "candidate_id": candidate_id})
    return RedirectResponse("/dashboard", status_code=303)


@router.post("/dashboard/decline")
async def decline_application(
    request: Request,
    application_id: str = Form(...),
    reason: str = Form(...),
):
    user = auth_mod.effective_user(request, default_role="client")
    if user.get("role") not in {"client", "admin"}:
        return RedirectResponse("/dashboard", status_code=303)
    reason = (reason or "").strip()
    if not reason:
        return RedirectResponse(
            "/dashboard?error=Reason+required", status_code=303,
        )
    now = app_main.now_value()
    with db.transaction() as conn:
        row = conn.execute(
            "SELECT id, status FROM applications WHERE id=?",
            (application_id,),
        ).fetchone()
        if row is None:
            return RedirectResponse("/dashboard?error=Not+found", status_code=303)
        if row["status"] not in {"submitted", "under_review"}:
            return RedirectResponse(
                "/dashboard?error=Not+pending", status_code=303,
            )
        conn.execute(
            "UPDATE applications SET status='rejected', reject_reason=?, "
            "updated_at=? WHERE id=?",
            (reason, now, application_id),
        )
        db.log_audit(
            conn,
            occurred_at=now,
            operation="application_declined",
            target_type="application",
            target_id=application_id,
            payload={
                "reason": reason,
                "declined_by": user["id"],
                "previous_status": row["status"],
            },
        )
    app_main.emit("application_declined",
                  {"application_id": application_id})
    return RedirectResponse("/dashboard", status_code=303)
