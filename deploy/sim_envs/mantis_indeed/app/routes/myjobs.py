"""/myjobs — Saved + Applied tabs."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, Response

from .. import auth, db

router = APIRouter()


def _format_salary(low: int, high: int, period: str) -> str:
    if not low and not high:
        return ""
    suffix = "a year" if period == "year" else "an hour"
    if low and high and low != high:
        return f"${low:,} - ${high:,} {suffix}"
    return f"${low or high:,} {suffix}"


@router.get("/myjobs", response_class=HTMLResponse)
async def myjobs(request: Request) -> Response:
    tab = request.query_params.get("tab", "saved")
    if tab not in {"saved", "applied"}:
        tab = "saved"
    conn = db.connect()
    uid = auth.effective_user_id(request)

    if tab == "saved":
        rows = conn.execute(
            "SELECT j.*, c.name AS company_name FROM saved_jobs s "
            "JOIN jobs j ON j.id = s.job_id "
            "JOIN companies c ON c.id = j.company_id "
            "WHERE s.user_id = ? "
            "ORDER BY s.saved_at DESC",
            (uid,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT j.*, c.name AS company_name, a.status AS app_status, "
            "a.applied_at AS app_applied_at FROM applications a "
            "JOIN jobs j ON j.id = a.job_id "
            "JOIN companies c ON c.id = j.company_id "
            "WHERE a.user_id = ? "
            "ORDER BY a.applied_at DESC",
            (uid,),
        ).fetchall()

    items = []
    for r in rows:
        d = dict(r)
        d["salary_display"] = _format_salary(d["salary_low"], d["salary_high"],
                                              d["salary_period"])
        items.append(d)

    return request.app.state.templates.TemplateResponse(
        "myjobs.html",
        {"request": request, "tab": tab, "items": items},
    )
