"""Partner / Product docs — `/docs/partner`, `/docs/product`."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()


PARTNER_TOC = [
    "Getting started",
    "Build apps",
    "Build themes",
    "Earn revenue",
    "App listings",
    "Partner Directory",
    "Compliance",
]

PRODUCT_TOC = [
    "Admin API",
    "Storefront API",
    "Webhooks",
    "Functions",
    "Polaris (UI)",
    "Hydrogen",
    "Liquid",
]

PARTNER_ARTICLES = [
    ("Getting started as a Shopify Partner",
     "Set up your Partners account, install the Shopify CLI, and create your first development store."),
    ("Build your first app",
     "Walk through scaffolding, OAuth, and webhooks. Bring an app from idea to listing in a day."),
    ("Earn revenue through referrals",
     "Plus, POS, and B2B referral programs explained — payouts, eligibility, and lifecycle."),
]

PRODUCT_ARTICLES = [
    ("Admin API: GraphQL vs REST",
     "When to pick which surface, paginate large queries, and stay under rate limits."),
    ("Hydrogen + Oxygen: production setup",
     "Deploy a custom storefront on Oxygen with Hydrogen's data primitives."),
    ("Functions: replace deprecated apps",
     "Build server-side commerce logic with no app sidecar."),
]


@router.get("/docs/partner", response_class=HTMLResponse)
async def partner_docs(request: Request, q: str = ""):
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "docs.html",
        {
            "request": request,
            "active_section": "partner_docs",
            "title": "Partner docs",
            "subtitle": "Everything you need to build and grow with Shopify.",
            "toc": PARTNER_TOC,
            "articles": PARTNER_ARTICLES,
            "q": q,
        },
    )


@router.get("/docs/product", response_class=HTMLResponse)
async def product_docs(request: Request, q: str = ""):
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "docs.html",
        {
            "request": request,
            "active_section": "product_docs",
            "title": "Product docs",
            "subtitle": "Reference for every Shopify developer surface.",
            "toc": PRODUCT_TOC,
            "articles": PRODUCT_ARTICLES,
            "q": q,
        },
    )
