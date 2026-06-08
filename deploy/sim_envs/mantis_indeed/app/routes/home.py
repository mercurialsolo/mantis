"""Home page (`/`) — the two-input hero search."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, Response

from .. import db

router = APIRouter()


TRENDING_SEARCHES = [
    "software engineer",
    "remote",
    "warehouse",
    "registered nurse",
    "data analyst",
    "customer service",
]

EMPLOYER_RESOURCES = [
    "Post a job",
    "Employer reviews",
    "Pricing",
    "Hiring lab",
    "Resume search",
]


@router.get("/", response_class=HTMLResponse)
async def home(request: Request) -> Response:
    conn = db.connect()
    job_count = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE status = 'active'"
    ).fetchone()[0]
    return request.app.state.templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "job_count": job_count,
            "trending": TRENDING_SEARCHES,
            "resources": EMPLOYER_RESOURCES,
        },
    )
