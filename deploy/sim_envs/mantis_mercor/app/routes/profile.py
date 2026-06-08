"""Candidate profile editor."""

from __future__ import annotations

import json

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import auth as auth_mod, db, main as app_main

router = APIRouter()


@router.get("/profile", response_class=HTMLResponse)
async def profile_view(request: Request):
    conn = db.connect()
    user = auth_mod.effective_user(request, default_role="candidate")
    row = conn.execute(
        "SELECT * FROM candidate_profiles WHERE user_id=?", (user["id"],),
    ).fetchone()
    profile = dict(row) if row else {
        "user_id": user["id"], "headline": "", "skills_json": "[]",
        "hourly_rate": 0.0, "availability": "Part-time",
        "resume_text": "", "updated_at": "",
    }
    profile["skills"] = json.loads(profile.get("skills_json") or "[]")
    return request.app.state.templates.TemplateResponse(
        "profile.html",
        {"request": request, "user": user, "profile": profile},
    )


@router.post("/profile")
async def profile_save(
    request: Request,
    headline: str = Form(""),
    skills: str = Form(""),
    hourly_rate: str = Form("0"),
    availability: str = Form("Part-time"),
):
    user = auth_mod.effective_user(request, default_role="candidate")
    try:
        rate = float(hourly_rate or 0)
    except ValueError:
        rate = 0.0
    skills_list = [s.strip() for s in skills.split(",") if s.strip()]
    if availability not in {"Full-time", "Part-time", "Project-based"}:
        availability = "Part-time"

    now = app_main.now_value()
    with db.transaction() as conn:
        existing = conn.execute(
            "SELECT user_id FROM candidate_profiles WHERE user_id=?",
            (user["id"],),
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE candidate_profiles SET headline=?, skills_json=?, "
                "hourly_rate=?, availability=?, updated_at=? WHERE user_id=?",
                (headline, db.pack_json(skills_list), rate, availability, now,
                 user["id"]),
            )
        else:
            conn.execute(
                "INSERT INTO candidate_profiles (user_id, headline, skills_json, "
                "hourly_rate, availability, resume_text, updated_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (user["id"], headline, db.pack_json(skills_list), rate,
                 availability, "", now),
            )
        db.log_audit(
            conn,
            occurred_at=now,
            operation="profile_updated",
            target_type="candidate_profile",
            target_id=user["id"],
            payload={
                "headline": headline,
                "skills": skills_list,
                "hourly_rate": rate,
                "availability": availability,
            },
        )

    app_main.emit("profile_updated", {"user_id": user["id"]})
    return RedirectResponse("/profile", status_code=303)
