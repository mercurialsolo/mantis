"""Partner Directory — `/partner_directory`, profile, review request."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import db, main as app_main

router = APIRouter()


@router.get("/partner_directory", response_class=HTMLResponse)
async def directory(request: Request):
    templates = request.app.state.templates
    conn = db.connect()
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM directory_listings ORDER BY id"
    ).fetchall()]
    return templates.TemplateResponse(
        "directory.html",
        {
            "request": request,
            "active_section": "directory",
            "sub_section": "",
            "rows": rows,
        },
    )


@router.get("/partner_directory/profile", response_class=HTMLResponse)
async def directory_profile(request: Request):
    templates = request.app.state.templates
    conn = db.connect()
    partner = dict(conn.execute("SELECT * FROM partners LIMIT 1").fetchone())
    return templates.TemplateResponse(
        "directory_profile.html",
        {
            "request": request,
            "active_section": "directory",
            "sub_section": "profile",
            "partner": partner,
        },
    )


@router.post("/partner_directory/profile")
async def directory_profile_save(
    request: Request,
    studio_name: str = Form(""),
    bio: str = Form(""),
    services: str = Form(""),
    languages: str = Form(""),
    locations: str = Form(""),
    hourly_rate: str = Form(""),
):
    conn = db.connect()
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="directory_profile_updated",
        target_type="directory_profile",
        target_id="self",
        payload={
            "studio_name": studio_name.strip(),
            "bio": bio.strip()[:300],
            "services": services.strip(),
            "languages": languages.strip(),
            "locations": locations.strip(),
            "hourly_rate": hourly_rate.strip(),
        },
    )
    conn.commit()
    app_main.emit("directory_profile_updated", {})
    return RedirectResponse("/partner_directory/profile", status_code=303)


@router.post("/partner_directory/{listing_id}/request_review")
async def directory_request_review(listing_id: str, request: Request):
    conn = db.connect()
    row = conn.execute(
        "SELECT * FROM directory_listings WHERE id=?", (listing_id,),
    ).fetchone()
    if row is None:
        return RedirectResponse("/partner_directory", status_code=303)
    conn.execute(
        "UPDATE directory_listings SET review_status='requested' WHERE id=?",
        (listing_id,),
    )
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="directory_review_requested",
        target_type="directory_listing",
        target_id=listing_id,
        payload={"business_name": row["business_name"]},
    )
    conn.commit()
    app_main.emit("directory_review_requested", {"listing_id": listing_id})
    return RedirectResponse("/partner_directory", status_code=303)
