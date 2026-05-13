"""Tasks — work items distinct from activities.

CRM convention: an "activity" is "what happened" (a logged call); a
"task" is "what should happen" (call them by Tuesday). Real tools
split these; we mirror that so the agent has a separate panel to
navigate.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import db, main as app_main

router = APIRouter()
PAGE_SIZE = 50


@router.get("/tasks", response_class=HTMLResponse)
async def list_tasks(request: Request) -> HTMLResponse:
    page = max(1, int(request.query_params.get("page") or 1))
    status = (request.query_params.get("status") or "open").strip()
    assignee = (request.query_params.get("assignee") or "").strip()
    overdue_only = request.query_params.get("overdue") == "1"

    where: list = []
    args: list = []
    if status == "open":
        where.append("completed_at IS NULL")
    elif status == "completed":
        where.append("completed_at IS NOT NULL")
    if assignee:
        where.append("assignee_id = ?")
        args.append(assignee)
    if overdue_only:
        where.append("due_date < ? AND completed_at IS NULL")
        args.append(app_main.now_value())

    where_clause = " AND ".join(where) or "1=1"
    conn = db.connect()
    total = conn.execute(
        f"SELECT COUNT(*) FROM tasks WHERE {where_clause}", args
    ).fetchone()[0]
    rows = conn.execute(
        f"SELECT * FROM tasks WHERE {where_clause} ORDER BY "
        f"CASE WHEN due_date IS NULL THEN 1 ELSE 0 END, due_date, id "
        f"LIMIT ? OFFSET ?",
        args + [PAGE_SIZE, (page - 1) * PAGE_SIZE],
    ).fetchall()
    pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    return app_main.app.state.templates.TemplateResponse(
        "tasks_list.html",
        {
            "request": request,
            "tasks": [dict(r) for r in rows],
            "total": total, "page": page, "pages": pages,
            "filters": {"status": status, "assignee": assignee,
                        "overdue": overdue_only},
            "now": app_main.now_value(),
        },
    )


@router.post("/tasks/{task_id}/complete")
async def complete_task(task_id: str) -> RedirectResponse:
    with db.transaction() as txn:
        row = txn.execute("SELECT completed_at FROM tasks WHERE id = ?",
                          (task_id,)).fetchone()
        if row is None or row["completed_at"]:
            return RedirectResponse("/tasks", status_code=303)
        txn.execute(
            "UPDATE tasks SET completed_at = ? WHERE id = ?",
            (app_main.now_value(), task_id),
        )
        db.log_mutation(
            txn, occurred_at=app_main.now_value(),
            operation="task_completed", target_type="task", target_id=task_id,
        )
    app_main.emit("mutation.task_completed", {"task_id": task_id})
    return RedirectResponse("/tasks", status_code=303)


@router.post("/tasks/create")
async def create_task(
    title: str = Form(...),
    target_type: str = Form(""),
    target_id: str = Form(""),
    assignee_id: str = Form(""),
    due_date: str = Form(""),
    priority: str = Form("normal"),
    body: str = Form(""),
) -> RedirectResponse:
    new_id = f"task_user_{int(time.time() * 1000):013d}"
    with db.transaction() as txn:
        txn.execute(
            "INSERT INTO tasks (id, title, body, target_type, target_id, "
            "assignee_id, due_date, priority, completed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)",
            (
                new_id, title, body,
                target_type or None, target_id or None,
                assignee_id or None, due_date or None, priority,
            ),
        )
        db.log_mutation(
            txn, occurred_at=app_main.now_value(),
            operation="task_created", target_type="task", target_id=new_id,
            payload={"title": title, "target_type": target_type or None,
                     "target_id": target_id or None, "due_date": due_date or None},
        )
    app_main.emit("mutation.task_created", {"task_id": new_id})
    redirect = "/tasks"
    if target_type in ("contact", "deal") and target_id:
        redirect = f"/{target_type}s/{target_id}"
    return RedirectResponse(redirect, status_code=303)
