"""Email templates with merge fields — agents pick one + a contact to preview."""

from __future__ import annotations

import re

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from .. import db, main as app_main

router = APIRouter()


_MERGE_PATTERN = re.compile(r"\{\{\s*([\w.]+)\s*\}\}")


def render_merge(text: str, *, contact: dict, company: dict | None, user: dict | None) -> str:
    """Replace ``{{contact.first_name}}`` style tokens with values."""
    def first_name(name: str) -> str:
        return (name or "").split(" ")[0] if name else ""

    def last_name(name: str) -> str:
        parts = (name or "").split(" ")
        return parts[-1] if len(parts) > 1 else ""

    ctx = {
        "contact.first_name": first_name(contact.get("name", "")),
        "contact.last_name": last_name(contact.get("name", "")),
        "contact.name": contact.get("name", ""),
        "contact.email": contact.get("email", "") or "",
        "company.name": (company or {}).get("name", "") if company else "",
        "company.domain": (company or {}).get("domain", "") if company else "",
        "user.name": (user or {}).get("name", "") if user else "",
        "user.email": (user or {}).get("email", "") if user else "",
    }
    return _MERGE_PATTERN.sub(lambda m: ctx.get(m.group(1), m.group(0)), text)


@router.get("/templates", response_class=HTMLResponse)
async def list_templates(request: Request) -> HTMLResponse:
    folder = (request.query_params.get("folder") or "").strip()
    conn = db.connect()
    where = []
    args: list = []
    if folder:
        where.append("folder = ?")
        args.append(folder)
    where_clause = " AND ".join(where) or "1=1"
    rows = conn.execute(
        f"SELECT id, name, subject, folder FROM email_templates "
        f"WHERE {where_clause} ORDER BY id",
        args,
    ).fetchall()
    folders = [r["folder"] for r in conn.execute(
        "SELECT DISTINCT folder FROM email_templates ORDER BY folder"
    ).fetchall()]
    return app_main.app.state.templates.TemplateResponse(
        "templates_list.html",
        {
            "request": request,
            "templates": [dict(r) for r in rows],
            "folders": folders,
            "active_folder": folder,
        },
    )


@router.get("/templates/{template_id}", response_class=HTMLResponse)
async def template_detail(request: Request, template_id: str) -> HTMLResponse:
    conn = db.connect()
    row = conn.execute(
        "SELECT * FROM email_templates WHERE id = ?", (template_id,),
    ).fetchone()
    if row is None:
        return HTMLResponse("Not found", status_code=404)

    preview_subject = row["subject"]
    preview_body = row["body"]
    contact_id = (request.query_params.get("preview_contact") or "").strip()
    contact_obj: dict | None = None
    company_obj: dict | None = None
    user_obj: dict | None = None
    if contact_id:
        c = conn.execute(
            "SELECT c.*, co.name AS company_name, co.domain AS company_domain "
            "FROM contacts c LEFT JOIN companies co ON c.company_id = co.id "
            "WHERE c.id = ?",
            (contact_id,),
        ).fetchone()
        if c is not None:
            contact_obj = {"name": c["name"], "email": c["email"]}
            company_obj = {"name": c["company_name"] or "",
                           "domain": c["company_domain"] or ""}
            owner = conn.execute(
                "SELECT name, email FROM users WHERE id = ?",
                (c["owner_id"],),
            ).fetchone()
            if owner is not None:
                user_obj = {"name": owner["name"], "email": owner["email"]}
            preview_subject = render_merge(row["subject"], contact=contact_obj,
                                           company=company_obj, user=user_obj)
            preview_body = render_merge(row["body"], contact=contact_obj,
                                        company=company_obj, user=user_obj)

    return app_main.app.state.templates.TemplateResponse(
        "template_detail.html",
        {
            "request": request,
            "template": dict(row),
            "preview_contact_id": contact_id,
            "preview_subject": preview_subject,
            "preview_body": preview_body,
        },
    )
