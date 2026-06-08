"""Search results (`/jobs`), detail HTMX fragment, viewjob, save-toggle."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .. import auth, db, main as app_main

router = APIRouter()


FILTER_CHIPS = [
    "Date posted",
    "Remote",
    "Developer skill",
    "Job Type",
    "Experience level",
    "Pay",
    "Education",
    "Clearance type",
    "Developer type",
    "Compensation package",
    "Distance",
]


def _job_row(row) -> dict[str, Any]:
    out = dict(row)
    return out


def _join_company(conn, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not jobs:
        return jobs
    cids = {j["company_id"] for j in jobs}
    placeholders = ",".join(["?"] * len(cids))
    rows = conn.execute(
        f"SELECT * FROM companies WHERE id IN ({placeholders})",
        list(cids),
    ).fetchall()
    by_id = {r["id"]: dict(r) for r in rows}
    for j in jobs:
        j["company"] = by_id.get(j["company_id"], {"name": "Unknown",
                                                   "rating": 0,
                                                   "review_count": 0})
    return jobs


def _format_salary(low: int, high: int, period: str) -> str:
    if not low and not high:
        return ""
    suffix = "a year" if period == "year" else "an hour"
    if low and high and low != high:
        return f"${low:,} - ${high:,} {suffix}"
    return f"${low or high:,} {suffix}"


def _relative_posted(now_iso: str, posted_iso: str) -> str:
    from datetime import datetime
    try:
        n = datetime.fromisoformat(now_iso.rstrip("Z"))
        p = datetime.fromisoformat(posted_iso.rstrip("Z"))
    except ValueError:
        return ""
    delta = n - p
    days = delta.days
    if days <= 0:
        return "Just posted"
    if days == 1:
        return "Posted 1 day ago"
    if days < 30:
        return f"Posted {days} days ago"
    return "Posted 30+ days ago"


def _search_jobs(
    conn,
    q: str,
    l: str,  # noqa: E741 -- `l` mirrors Indeed's location query param
    remote: bool,
    date_max_days: int | None,
    job_type: str | None,
) -> list[dict[str, Any]]:
    """Substring + filter search. Trivial — no fancy ranking."""
    sql = "SELECT * FROM jobs WHERE status = 'active'"
    params: list[Any] = []
    if q.strip():
        sql += " AND (LOWER(title) LIKE ? OR LOWER(snippet) LIKE ?)"
        pat = f"%{q.strip().lower()}%"
        params.extend([pat, pat])
    if l.strip():
        # Treat `remote` location as the remote-only filter.
        ll = l.strip().lower()
        if ll == "remote":
            sql += " AND remote_flag = 1"
        else:
            sql += " AND LOWER(location) LIKE ?"
            params.append(f"%{ll}%")
    if remote:
        sql += " AND remote_flag = 1"
    if job_type:
        sql += " AND job_type = ?"
        params.append(job_type)
    sql += " ORDER BY posted_at DESC"
    rows = conn.execute(sql, params).fetchall()
    return [_job_row(r) for r in rows]


@router.get("/jobs", response_class=HTMLResponse)
async def jobs_search(request: Request) -> Response:
    qp = request.query_params
    q = qp.get("q", "")
    l = qp.get("l", "")  # noqa: E741 -- Indeed query-param convention
    vjk = qp.get("vjk", "")
    remote = qp.get("remote", "") in {"1", "true", "on"}
    job_type = qp.get("job_type") or None

    conn = db.connect()
    jobs = _search_jobs(conn, q, l, remote, None, job_type)
    jobs = _join_company(conn, jobs)

    # Decorate cards with display-friendly salary + relative date.
    for j in jobs:
        j["salary_display"] = _format_salary(j["salary_low"], j["salary_high"],
                                              j["salary_period"])
        j["posted_display"] = _relative_posted(app_main.now_value(),
                                                j["posted_at"])
        j["snippet_short"] = (j["snippet"][:240]
                              + ("…" if len(j["snippet"]) > 240 else ""))

    # Selected job: vjk in URL, else first job.
    selected: dict[str, Any] | None = None
    if vjk:
        selected = next((j for j in jobs if j["jk"] == vjk), None)
    if selected is None and jobs:
        selected = jobs[0]

    # T01-style saved jobs lookup (anonymous default seeker).
    uid = auth.effective_user_id(request)
    saved_ids = {
        r["job_id"] for r in conn.execute(
            "SELECT job_id FROM saved_jobs WHERE user_id = ?",
            (uid,),
        ).fetchall()
    }

    # Active filter chips for visual state.
    active_chips = set()
    if remote:
        active_chips.add("Remote")

    h1 = f"{q.lower()} jobs in {l}" if q and l else (
        f"{q.lower()} jobs" if q else "All jobs"
    )

    return request.app.state.templates.TemplateResponse(
        "jobs_search.html",
        {
            "request": request,
            "q": q,
            "l": l,
            "vjk": vjk,
            "remote": remote,
            "job_type": job_type,
            "jobs": jobs,
            "selected": selected,
            "saved_ids": saved_ids,
            "filter_chips": FILTER_CHIPS,
            "active_chips": active_chips,
            "h1": h1,
            "total_count": len(jobs),
        },
    )


@router.get("/jobs/_detail", response_class=HTMLResponse)
async def jobs_detail_fragment(request: Request) -> Response:
    """HTMX-style fragment to swap into the right pane without nav."""
    jk = request.query_params.get("jk", "")
    conn = db.connect()
    row = conn.execute("SELECT * FROM jobs WHERE jk = ?", (jk,)).fetchone()
    if row is None:
        return HTMLResponse("<div class='empty-pane'>Job not found.</div>",
                            status_code=404)
    job = _join_company(conn, [_job_row(row)])[0]
    job["salary_display"] = _format_salary(job["salary_low"], job["salary_high"],
                                            job["salary_period"])
    job["posted_display"] = _relative_posted(app_main.now_value(),
                                              job["posted_at"])
    uid = auth.effective_user_id(request)
    is_saved = bool(conn.execute(
        "SELECT 1 FROM saved_jobs WHERE user_id = ? AND job_id = ?",
        (uid, job["id"]),
    ).fetchone())
    return request.app.state.templates.TemplateResponse(
        "_detail_pane.html",
        {"request": request, "job": job, "is_saved": is_saved},
    )


@router.get("/viewjob", response_class=HTMLResponse)
async def viewjob(request: Request) -> Response:
    jk = request.query_params.get("jk", "")
    conn = db.connect()
    row = conn.execute("SELECT * FROM jobs WHERE jk = ?", (jk,)).fetchone()
    if row is None:
        return HTMLResponse("<h1>Job not found</h1>", status_code=404)
    job = _join_company(conn, [_job_row(row)])[0]
    job["salary_display"] = _format_salary(job["salary_low"], job["salary_high"],
                                            job["salary_period"])
    job["posted_display"] = _relative_posted(app_main.now_value(),
                                              job["posted_at"])
    uid = auth.effective_user_id(request)
    is_saved = bool(conn.execute(
        "SELECT 1 FROM saved_jobs WHERE user_id = ? AND job_id = ?",
        (uid, job["id"]),
    ).fetchone())
    return request.app.state.templates.TemplateResponse(
        "viewjob.html",
        {"request": request, "job": job, "is_saved": is_saved},
    )


@router.post("/jobs/{jk}/save")
async def jobs_save_toggle(jk: str, request: Request) -> JSONResponse:
    """Toggle saved_jobs row. Returns {saved: bool}."""
    conn = db.connect()
    row = conn.execute("SELECT id FROM jobs WHERE jk = ?", (jk,)).fetchone()
    if row is None:
        return JSONResponse({"error": "job not found"}, status_code=404)
    job_id = row["id"]
    uid = auth.effective_user_id(request)

    existing = conn.execute(
        "SELECT 1 FROM saved_jobs WHERE user_id = ? AND job_id = ?",
        (uid, job_id),
    ).fetchone()
    if existing:
        conn.execute(
            "DELETE FROM saved_jobs WHERE user_id = ? AND job_id = ?",
            (uid, job_id),
        )
        db.log_audit(
            conn,
            occurred_at=app_main.now_value(),
            operation="job_unsaved",
            target_type="saved_job",
            target_id=f"{uid}:{job_id}",
            payload={"user_id": uid, "job_id": job_id, "jk": jk},
        )
        conn.commit()
        return JSONResponse({"saved": False, "job_id": job_id})
    conn.execute(
        "INSERT INTO saved_jobs (user_id, job_id, saved_at) VALUES (?, ?, ?)",
        (uid, job_id, app_main.now_value()),
    )
    # Capture the search filters at save time (so T01 can verify
    # "saved with the right filters in scope").
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="job_saved",
        target_type="saved_job",
        target_id=f"{uid}:{job_id}",
        payload={
            "user_id": uid,
            "job_id": job_id,
            "jk": jk,
            "filters": {
                "q": request.query_params.get("q") or "",
                "l": request.query_params.get("l") or "",
                "remote": request.query_params.get("remote") or "",
            },
        },
    )
    conn.commit()
    return JSONResponse({"saved": True, "job_id": job_id})


@router.get("/jobs/_search_audit")
async def jobs_search_audit(request: Request) -> JSONResponse:
    """Lightweight audit-emitter the front-end calls on search submit so
    the oracle can verify the agent actually performed the search with
    the right filters. Idempotent — emits once per unique (q,l,remote)
    tuple per second.
    """
    qp = request.query_params
    payload = {
        "q": qp.get("q") or "",
        "l": qp.get("l") or "",
        "remote": qp.get("remote") or "",
    }
    conn = db.connect()
    db.log_audit(
        conn,
        occurred_at=app_main.now_value(),
        operation="search_submitted",
        target_type="search",
        target_id=f"{payload['q']}|{payload['l']}|{payload['remote']}",
        payload=payload,
    )
    conn.commit()
    return JSONResponse({"ok": True})
