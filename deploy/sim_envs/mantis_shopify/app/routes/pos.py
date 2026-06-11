"""Shopify POS — `/pos`. Renders the marketing surface + a shortcut to
submit a POS referral that funnels through /sales/leads/new?product=pos.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/pos", response_class=HTMLResponse)
async def pos(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "pos.html",
        {"request": request, "active_section": "pos"},
    )
