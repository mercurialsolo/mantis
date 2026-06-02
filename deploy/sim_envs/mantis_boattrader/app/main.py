"""mantis-boattrader FastAPI app — high-fidelity boattrader.com clone.

Two route surfaces (mirrors mantis-shop):

* ``/`` — public marketplace pages (home, /boats, /boat/<slug>, ad clicks,
  cookie consent endpoints, lead submissions, ad creative SVGs).
* ``/__env__/*`` — harness-only, gated on ``X-Env-Admin`` header. Provides
  ``/health``, ``/reset``, ``/leads``, ``/state`` for verification.

Configurable latency is enforced by middleware: every public request
sleeps a uniform-random duration between ``LATENCY_MS_MIN`` and
``LATENCY_MS_MAX``. ``LATENCY_FAILURE_RATE`` (0..1) injects 503s.
"""

from __future__ import annotations

import asyncio
import os
import random
import secrets
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from fastapi import FastAPI, Form, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import db, seed

APP_DIR = Path(__file__).parent
ADMIN_TOKEN_ENV = "ENV_ADMIN_TOKEN"
ADMIN_HEADER = "X-Env-Admin"
CONSENT_COOKIE = "bt_cookie_consent"


def _admin_token() -> str:
    token = os.environ.get(ADMIN_TOKEN_ENV, "").strip()
    if not token:
        # Generate a per-process token if none provided — keeps local
        # ``docker run`` ergonomic; the harness still injects its own.
        token = secrets.token_urlsafe(32)
        os.environ[ADMIN_TOKEN_ENV] = token
    return token


def _latency_bounds() -> tuple[float, float]:
    lo = float(os.environ.get("LATENCY_MS_MIN", 0)) / 1000.0
    hi = float(os.environ.get("LATENCY_MS_MAX", 0)) / 1000.0
    if hi < lo:
        hi = lo
    return lo, hi


def _failure_rate() -> float:
    try:
        return max(0.0, min(1.0, float(os.environ.get("LATENCY_FAILURE_RATE", 0))))
    except ValueError:
        return 0.0


# Per-process ad rotation seed — keeps each container's rotation order
# stable across requests (so screenshots are reproducible).
_ad_seed = int(os.environ.get("SEED", 42)) ^ 0xAD5
_ad_rng = random.Random(_ad_seed)


def _now_iso() -> str:
    return os.environ.get("FAKE_NOW") or seed.FAKE_NOW_DEFAULT


# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(title="mantis-boattrader", docs_url=None, redoc_url=None)

    templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
    # Expose helpers to every template.
    templates.env.globals["env_admin_required"] = False
    templates.env.globals["site_name"] = "Boat Trader"
    templates.env.globals["wide_ad_id"] = lambda: _ad_rng.choice([a.id for a in db.store().ads])
    templates.env.globals["pick_ad"] = lambda slot: _pick_ad_for(slot)
    templates.env.globals["dealers_by_id"] = lambda: db.store().dealers_by_id

    app.state.templates = templates
    app.state.admin_token = _admin_token()

    # Boot — force seed eagerly so /__env__/health works as soon as the
    # process is alive.
    db.store()

    # ── Latency middleware ───────────────────────────────────────────
    @app.middleware("http")
    async def _latency_gate(request: Request, call_next):  # noqa: ANN202
        path = request.url.path
        # Static + harness routes bypass latency injection.
        if path.startswith("/__env__") or path.startswith("/static") or path.startswith("/ad/"):
            return await call_next(request)
        lo, hi = _latency_bounds()
        if hi > 0:
            await asyncio.sleep(random.uniform(lo, hi))
        if _failure_rate() > 0 and random.random() < _failure_rate():
            return JSONResponse(
                {"error": "service_unavailable", "retry_after_ms": int(hi * 1000)},
                status_code=503,
            )
        return await call_next(request)

    # ── Cookie consent state passed to every template ────────────────
    @app.middleware("http")
    async def _consent_state(request: Request, call_next):  # noqa: ANN202
        request.state.consent = request.cookies.get(CONSENT_COOKIE)
        return await call_next(request)

    # ── Routes ───────────────────────────────────────────────────────
    _mount_static(app)
    _mount_public(app, templates)
    _mount_harness(app)
    return app


def _mount_static(app: FastAPI) -> None:
    static_dir = APP_DIR / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ---------------------------------------------------------------------------
# Public routes
# ---------------------------------------------------------------------------


_BOAT_TYPE_SLOTS = [
    "leaderboard_top",
    "leaderboard_mid",
    "rail_right",
    "rail_left",
    "footer_banner",
]


def _pick_ad_for(slot: str) -> Any:
    ads = db.store().ads
    if not ads:
        return None
    # Stable per-slot rotation so different slots get different creatives.
    idx = (hash(slot) ^ _ad_seed) % len(ads)
    return ads[idx]


def _mount_public(app: FastAPI, templates: Jinja2Templates) -> None:

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> Any:
        return templates.TemplateResponse(
            request,
            "home.html",
            {
                "featured": db.featured_boats(limit=12),
                "near": db.boats_near(_zip=None, limit=8),
                "popular_types": db.popular_types(),
                "top_makes": db.list_makes()[:10],
                "popular_zip": "95109",
            },
        )

    @app.get("/boats/", response_class=HTMLResponse)
    @app.get("/boats", response_class=HTMLResponse)
    async def boats(request: Request, page: int = Query(1, ge=1)) -> Any:
        params: dict[str, Any] = dict(request.query_params)
        params.pop("page", None)
        result = db.query_boats(params, page=page, per_page=24)
        return templates.TemplateResponse(
            request,
            "boats.html",
            {
                "result": result,
                "querystring": urlencode([(k, v) for k, v in request.query_params.items() if k != "page"]),
                "sorts": db.SORTS,
            },
        )

    @app.get("/boats/{filters:path}", response_class=HTMLResponse)
    async def boats_filtered(
        request: Request, filters: str, page: int = Query(1, ge=1)
    ) -> Any:
        """BoatTrader-style path filters: ``/boats/state-fl/by-owner/``,
        ``/boats/by-owner/``, ``/boats/zip-33316/by-owner/``.

        We honour the seller (``by-owner`` / ``by-dealer``) and ``state``
        segments — the ones the by-owner plan drives — and ignore any
        other or unknown segment (zip, make, …) so a slightly-off path
        (e.g. ``state-all``) still renders listings instead of 404-ing.
        Declared after the exact ``/boats`` routes so the no-filter index
        is unaffected.
        """
        params: dict[str, Any] = {}
        for seg in filters.split("/"):
            seg = seg.strip().lower()
            if seg == "by-owner":
                params["listing_type"] = "owner"
            elif seg == "by-dealer":
                params["listing_type"] = "dealer"
            elif seg.startswith("state-"):
                code = seg[len("state-"):].upper()
                if code and code != "ALL":
                    params["state"] = code
        # Query-string params (if any) override path-derived ones.
        for k, v in request.query_params.items():
            if k != "page":
                params[k] = v
        result = db.query_boats(params, page=page, per_page=24)
        return templates.TemplateResponse(
            request,
            "boats.html",
            {
                "result": result,
                "querystring": urlencode(
                    [(k, v) for k, v in request.query_params.items() if k != "page"]
                ),
                "sorts": db.SORTS,
            },
        )

    @app.get("/boat/{slug}/", response_class=HTMLResponse)
    @app.get("/boat/{slug}", response_class=HTMLResponse)
    async def boat_detail(request: Request, slug: str) -> Any:
        boat = db.boat_by_slug(slug)
        if boat is None:
            raise HTTPException(status_code=404, detail="boat not found")
        dealer = db.dealer_for(boat)
        prev_b, next_b = db.adjacent_boats(boat)
        return templates.TemplateResponse(
            request,
            "boat_detail.html",
            {
                "boat": boat,
                "dealer": dealer,
                "prev_boat": prev_b,
                "next_boat": next_b,
                "similar": db.similar_boats(boat, limit=6),
            },
        )

    @app.post("/boat/{slug}/contact")
    async def contact_dealer(
        request: Request,
        slug: str,
        name: str = Form(...),
        email: str = Form(...),
        phone: str = Form(""),
        message: str = Form(""),
    ) -> Any:
        boat = db.boat_by_slug(slug)
        if boat is None:
            raise HTTPException(status_code=404, detail="boat not found")
        lead = db.record_lead({
            "boat_id": boat.id,
            "boat_title": boat.title,
            "dealer_id": boat.dealer_id,
            "name": name,
            "email": email,
            "phone": phone,
            "message": message,
            "ts": time.time(),
        })
        # Render the detail page with a banner.
        dealer = db.dealer_for(boat)
        prev_b, next_b = db.adjacent_boats(boat)
        return templates.TemplateResponse(
            request,
            "boat_detail.html",
            {
                "boat": boat,
                "dealer": dealer,
                "prev_boat": prev_b,
                "next_boat": next_b,
                "similar": db.similar_boats(boat, limit=6),
                "lead_confirmation": lead,
            },
        )

    # ── Cookie consent endpoints ─────────────────────────────────────
    @app.get("/__site/clear-consent")
    async def clear_consent(request: Request) -> Any:
        resp = RedirectResponse(url="/", status_code=303)
        resp.delete_cookie(CONSENT_COOKIE, path="/")
        return resp

    @app.post("/__site/consent")
    async def set_consent(
        request: Request,
        choice: str = Form(...),
        next_url: str = Form("/"),
    ) -> Any:
        choice = "accept" if choice == "accept" else "decline"
        # Only allow local redirects.
        if not next_url.startswith("/"):
            next_url = "/"
        resp = RedirectResponse(url=next_url, status_code=303)
        # 180-day cookie like a typical CMP.
        resp.set_cookie(CONSENT_COOKIE, choice, max_age=60 * 60 * 24 * 180, samesite="lax")
        db.emit_mutation(
            operation="consent_set",
            target_type="session",
            target_id="",
            payload={"choice": choice, "next_url": next_url},
        )
        return resp

    # ── Ad creatives ─────────────────────────────────────────────────
    @app.get("/ad/{ad_id}.svg")
    async def ad_svg(ad_id: str, w: int = 728, h: int = 90) -> Any:
        ad = next((a for a in db.store().ads if a.id == ad_id), None)
        if ad is None:
            raise HTTPException(status_code=404, detail="ad not found")
        svg = _render_ad_svg(ad, w=w, h=h)
        return Response(content=svg, media_type="image/svg+xml")

    @app.get("/ad/{ad_id}/click")
    async def ad_click(ad_id: str) -> Any:
        ad = next((a for a in db.store().ads if a.id == ad_id), None)
        if ad is None:
            raise HTTPException(status_code=404, detail="ad not found")
        return RedirectResponse(url=ad.href, status_code=303)

    # ── Procedural boat image SVGs (served from /assets/ so the
    #    StaticFiles mount at /static/ doesn't 404 them). ──────────
    @app.get("/assets/img/v/{type_slug}_{boat_palette:int}_{img_idx:int}.svg")
    async def boat_img_v(type_slug: str, boat_palette: int, img_idx: int) -> Any:
        svg = _render_boat_svg_typed(type_slug, boat_palette, img_idx)
        return Response(content=svg, media_type="image/svg+xml")

    @app.get("/assets/img/boat_{boat_palette:int}_{img_idx:int}.svg")
    async def boat_img(boat_palette: int, img_idx: int) -> Any:
        # Legacy fallback (early seeds before the type-aware path landed).
        svg = _render_boat_svg(boat_palette, img_idx)
        return Response(content=svg, media_type="image/svg+xml")

    @app.get("/assets/img/type_{slug}.svg")
    async def type_img(slug: str) -> Any:
        return Response(content=_render_type_svg(slug), media_type="image/svg+xml")

    @app.get("/assets/logo.svg")
    async def logo_svg() -> Any:
        return Response(content=_render_logo_svg(), media_type="image/svg+xml")

    # Phone number rendered as an SVG — used for private-seller listings
    # so the digits aren't sitting as plain text in the HTML markup
    # (which is how scrapers / regex usually harvest contact info).
    # The phone is gated behind ``/boat/<slug>/show-phone`` which sets a
    # session-style cookie before rendering the SVG.
    @app.get("/assets/phone/{slug}.svg")
    async def phone_svg(request: Request, slug: str) -> Any:
        boat = db.boat_by_slug(slug)
        if boat is None or not boat.owner_phone:
            raise HTTPException(status_code=404, detail="no private phone for boat")
        # Only render when the visitor has clicked "Show Phone Number".
        if request.cookies.get(f"bt_show_phone_{boat.id}") != "1":
            raise HTTPException(status_code=403, detail="phone hidden")
        return Response(
            content=_render_phone_svg(boat.owner_phone),
            media_type="image/svg+xml",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/boat/{slug}/show-phone")
    async def show_phone(request: Request, slug: str) -> Any:
        boat = db.boat_by_slug(slug)
        if boat is None:
            raise HTTPException(status_code=404, detail="boat not found")
        resp = RedirectResponse(url=f"/boat/{slug}/#contact", status_code=303)
        resp.set_cookie(f"bt_show_phone_{boat.id}", "1", max_age=60 * 60, samesite="lax")
        db.emit_mutation(
            operation="phone_revealed",
            target_type="boat",
            target_id=boat.id,
            payload={"slug": slug},
        )
        return resp


# ---------------------------------------------------------------------------
# Harness routes
# ---------------------------------------------------------------------------


def _mount_harness(app: FastAPI) -> None:

    def _require_admin(request: Request) -> None:
        token = request.headers.get(ADMIN_HEADER, "")
        if not token or token != app.state.admin_token:
            raise HTTPException(status_code=403, detail="admin token required")

    @app.get("/__env__/health")
    async def health() -> Any:
        return {
            "ok": True,
            "boats": len(db.store().boats),
            "dealers": len(db.store().dealers),
            "now": _now_iso(),
            "latency_ms_min": int(_latency_bounds()[0] * 1000),
            "latency_ms_max": int(_latency_bounds()[1] * 1000),
            "failure_rate": _failure_rate(),
        }

    @app.post("/__env__/reset")
    async def reset(request: Request) -> Any:
        _require_admin(request)
        db.reset()
        # ``db.reset()`` rebuilds the store from scratch, so the
        # mutations log was already cleared. Stamp ``env_reset`` so
        # tests can see the boundary between runs.
        db.emit_mutation(
            operation="env_reset",
            target_type="env",
            target_id="",
            payload={},
        )
        return {"ok": True}

    @app.get("/__env__/state")
    async def state(request: Request) -> Any:
        _require_admin(request)
        s = db.store()
        return {
            "boats": len(s.boats),
            "dealers": len(s.dealers),
            "leads": len(s.leads),
            "mutations": len(s.mutations),
            "recent_mutations": db.list_mutations(limit=50),
            "facets": s.facets,
        }

    @app.get("/__env__/leads")
    async def leads(request: Request) -> Any:
        _require_admin(request)
        return {"leads": db.list_leads()}

    @app.get("/__env__/mutations")
    async def mutations(request: Request) -> Any:
        _require_admin(request)
        try:
            since = int(request.query_params.get("since") or 0)
        except (TypeError, ValueError):
            since = 0
        muts = db.list_mutations()
        if since > 0:
            muts = [m for m in muts if m.get("id", 0) > since]
        return {"mutations": muts}

    @app.get("/__env__/oracle")
    async def oracle(request: Request) -> Any:
        _require_admin(request)
        task_id = (request.query_params.get("task_id") or "").strip()
        if not task_id:
            return {
                "passed": False,
                "score": 0.0,
                "task_id": "",
                "reasons": ["task_id is required"],
                "diff": {},
            }
        from .oracles import grade as oracle_grade
        return oracle_grade(
            task_id,
            db.store(),
            now=_now_iso(),
            seed_val=int(os.environ.get("SEED", 42)),
        )

    @app.post("/__env__/config")
    async def config(
        request: Request,
        latency_ms_min: int | None = None,
        latency_ms_max: int | None = None,
        failure_rate: float | None = None,
    ) -> Any:
        _require_admin(request)
        if latency_ms_min is not None:
            os.environ["LATENCY_MS_MIN"] = str(max(0, latency_ms_min))
        if latency_ms_max is not None:
            os.environ["LATENCY_MS_MAX"] = str(max(0, latency_ms_max))
        if failure_rate is not None:
            os.environ["LATENCY_FAILURE_RATE"] = str(failure_rate)
        return {
            "latency_ms_min": int(os.environ.get("LATENCY_MS_MIN", 0)),
            "latency_ms_max": int(os.environ.get("LATENCY_MS_MAX", 0)),
            "failure_rate": _failure_rate(),
        }


# ---------------------------------------------------------------------------
# SVG renderers — all assets generated procedurally so the container has
# zero outbound network at runtime.
# ---------------------------------------------------------------------------


def _render_phone_svg(digits: str) -> str:
    """Render a 10-digit phone number as an SVG image.

    Each digit is drawn as a path inside its own <g>, with non-standard
    spacing and bezier curves so OCR / digit-segmentation has to do real
    work. Output is intentionally not regex-parseable from the markup —
    the actual digit characters never appear as text in the SVG.
    """
    if len(digits) != 10:
        digits = (digits + "0000000000")[:10]
    # Stylised glyphs — paths approximate the digits but the SVG never
    # contains "0"..."9" as text characters.
    glyph_paths = {
        "0": "M 8 0 Q 0 0 0 14 Q 0 28 8 28 Q 16 28 16 14 Q 16 0 8 0 M 8 4 Q 12 4 12 14 Q 12 24 8 24 Q 4 24 4 14 Q 4 4 8 4",
        "1": "M 4 4 L 8 0 L 8 28 L 12 28 L 4 28",
        "2": "M 0 6 Q 0 0 8 0 Q 16 0 16 8 Q 16 14 8 18 L 0 28 L 16 28",
        "3": "M 0 4 Q 4 0 8 0 Q 16 0 16 7 Q 16 12 10 14 Q 16 16 16 21 Q 16 28 8 28 Q 4 28 0 24",
        "4": "M 12 0 L 0 18 L 16 18 M 12 0 L 12 28",
        "5": "M 16 0 L 0 0 L 0 14 Q 4 12 8 12 Q 16 12 16 20 Q 16 28 8 28 Q 4 28 0 24",
        "6": "M 14 2 Q 8 -2 4 6 Q 0 14 0 20 Q 0 28 8 28 Q 16 28 16 20 Q 16 12 8 12 Q 4 12 0 16",
        "7": "M 0 0 L 16 0 L 8 28",
        "8": "M 8 0 Q 1 0 1 7 Q 1 14 8 14 Q 15 14 15 7 Q 15 0 8 0 M 8 14 Q 0 14 0 21 Q 0 28 8 28 Q 16 28 16 21 Q 16 14 8 14",
        "9": "M 16 14 Q 14 28 8 28 Q 0 28 0 14 Q 0 6 8 6 Q 16 6 16 14 Q 16 22 8 22 Q 0 22 0 14",
    }
    out: list[str] = []
    x = 12
    for i, d in enumerate(digits):
        # Punctuation/spacing matching the (NNN) NNN-NNNN format.
        if i == 0:
            out.append(f'<text x="{x}" y="22" font-family="Arial" font-size="22" fill="#0a3d62">(</text>')
            x += 12
        elif i == 3:
            out.append(f'<text x="{x}" y="22" font-family="Arial" font-size="22" fill="#0a3d62">)</text>')
            x += 12
        elif i == 6:
            out.append(f'<rect x="{x}" y="13" width="10" height="2" fill="#0a3d62"/>')
            x += 14
        out.append(f'<g transform="translate({x},2)"><path d="{glyph_paths[d]}" fill="none" stroke="#0a3d62" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></g>')
        x += 20
    width = x + 6
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} 32" width="{width}" height="32" aria-label="phone number">'
        + "".join(out)
        + '</svg>'
    )


def _render_logo_svg() -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 38">'
        '<style>text{font:700 28px Georgia,serif;fill:#0a3d62;letter-spacing:.5px}'
        '.tag{font:italic 700 28px Georgia,serif;fill:#0a3d62}</style>'
        '<text x="0" y="28" class="tag">Boat </text>'
        '<text x="78" y="28">Trader</text>'
        '</svg>'
    )


def _render_ad_svg(ad: Any, w: int, h: int) -> str:
    is_hero = h >= 400  # hero carousel uses 520; banner slots use ~120.
    # Hero variant: dramatic ocean + boat silhouette behind text.
    if is_hero:
        boat_x = int(w * 0.18)
        boat_y = int(h * 0.62)
        return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid slice">
  <defs>
    <linearGradient id="sky" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0" stop-color="{ad.bg_a}"/>
      <stop offset="1" stop-color="{ad.bg_b}"/>
    </linearGradient>
    <linearGradient id="sea" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0" stop-color="{ad.bg_b}"/>
      <stop offset="1" stop-color="#000000" stop-opacity=".7"/>
    </linearGradient>
    <radialGradient id="sun" cx="0.75" cy="0.30" r="0.55">
      <stop offset="0" stop-color="#ffffff" stop-opacity=".35"/>
      <stop offset="1" stop-color="#ffffff" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect width="{w}" height="{h}" fill="url(#sky)"/>
  <rect y="{int(h*0.68)}" width="{w}" height="{int(h*0.32)}" fill="url(#sea)"/>
  <rect width="{w}" height="{h}" fill="url(#sun)"/>
  <g transform="translate({boat_x},{boat_y})" fill="rgba(255,255,255,0.22)" stroke="rgba(255,255,255,0.45)" stroke-width="1">
    <path d="M0 0 L 320 0 L 286 28 L 22 28 Z"/>
    <path d="M44 -52 L 248 -52 L 280 0 L 18 0 Z"/>
    <rect x="84" y="-90" width="172" height="34" rx="4"/>
    <rect x="118" y="-126" width="84" height="32" rx="4"/>
    <line x1="160" y1="-180" x2="160" y2="-126" stroke-width="3"/>
    <rect x="156" y="-180" width="40" height="3" rx="1"/>
  </g>
  <ellipse cx="{boat_x-30}" cy="{boat_y+34}" rx="80" ry="6" fill="#ffffff" opacity=".25"/>
  <ellipse cx="{boat_x+320}" cy="{boat_y+34}" rx="80" ry="6" fill="#ffffff" opacity=".25"/>
  <g font-family="Georgia, 'Times New Roman', serif">
    <text x="{int(w*0.55)}" y="{int(h*0.40)}" font-size="{max(28, h*0.13):.0f}" font-weight="700" fill="{ad.accent}" text-anchor="middle" letter-spacing="2">{ad.headline}</text>
    <text x="{int(w*0.55)}" y="{int(h*0.55)}" font-size="{max(20, h*0.08):.0f}" font-style="italic" fill="{ad.accent}" text-anchor="middle">{ad.subline}</text>
  </g>
  <g font-family="Arial, sans-serif">
    <rect x="{int(w*0.42)}" y="{int(h*0.78)}" width="{int(w*0.26)}" height="48" rx="24" fill="{ad.accent}" opacity=".95"/>
    <text x="{int(w*0.55)}" y="{int(h*0.78)+30}" font-size="{max(12, h*0.04):.0f}" font-weight="700" fill="{ad.bg_a}" text-anchor="middle" letter-spacing="3">{ad.cta}</text>
  </g>
</svg>'''
    # Banner / footer variant: horizontal layout with small silhouette.
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid slice">
  <defs>
    <linearGradient id="g" x1="0" x2="1" y1="0" y2="0">
      <stop offset="0" stop-color="{ad.bg_a}"/>
      <stop offset="1" stop-color="{ad.bg_b}"/>
    </linearGradient>
    <linearGradient id="hg" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0" stop-color="#ffffff" stop-opacity="0.0"/>
      <stop offset="1" stop-color="#000000" stop-opacity="0.25"/>
    </linearGradient>
  </defs>
  <rect width="{w}" height="{h}" fill="url(#g)"/>
  <rect width="{w}" height="{h}" fill="url(#hg)"/>
  <g transform="translate({int(w*0.08)},{int(h*0.62)})" fill="rgba(255,255,255,0.18)">
    <path d="M0 0 L 180 0 L 195 -22 L 18 -22 Z"/>
    <rect x="35" y="-44" width="120" height="22" rx="4"/>
    <rect x="55" y="-58" width="64" height="14" rx="3"/>
  </g>
  <text x="{int(w*0.5)}" y="{int(h*0.42)}" font-family="Georgia, serif" font-size="{max(14, h*0.30):.0f}" font-weight="700" fill="{ad.accent}" text-anchor="middle">{ad.headline}</text>
  <text x="{int(w*0.5)}" y="{int(h*0.66)}" font-family="Georgia, serif" font-size="{max(11, h*0.20):.0f}" fill="{ad.accent}" text-anchor="middle">{ad.subline}</text>
  <text x="{int(w*0.5)}" y="{int(h*0.90)}" font-family="Arial, sans-serif" font-size="{max(9, h*0.13):.0f}" letter-spacing="2" fill="{ad.accent}" text-anchor="middle">{ad.cta}</text>
</svg>'''


_BOAT_PALETTES = [
    ("#67b6e8", "#0f4c75", "#f3f6f9"),  # bright midday
    ("#5a8db1", "#0a3d62", "#eef1f4"),  # dusk
    ("#8ab9d9", "#1b4965", "#f7f7f7"),  # high overcast
    ("#7da0b2", "#324354", "#e8eef2"),  # storm
    ("#9bc6e3", "#214a72", "#f4f7fa"),  # tropical
    ("#487c9c", "#082135", "#dbe5ec"),  # twilight
    ("#a4cbe6", "#173a5d", "#fafafa"),  # cool spring
    ("#7eb1d7", "#0e3756", "#eef3f7"),  # offshore
]


def _render_boat_svg(palette: int, idx: int) -> str:
    return _render_boat_svg_typed("powerboat", palette, idx)


# Type-aware silhouette. Different boat types get different cabin /
# rigging shapes so the listing-card images don't all look identical.
def _render_boat_svg_typed(type_slug: str, palette: int, idx: int) -> str:
    sky, water, hull = _BOAT_PALETTES[palette % len(_BOAT_PALETTES)]
    cam_tilt = (idx - 3) * 2.0
    boat_y = 280 + (idx % 3) * 12

    def hull_block() -> str:
        return (
            '<path d="M-180 0 L 180 0 L 150 36 L -150 36 Z" fill="url(#hull)" stroke="#5e6b75" stroke-width="1"/>'
            '<rect x="-170" y="28" width="340" height="4" fill="#0c4ea1"/>'
        )

    if type_slug == "yacht":
        sup = (
            hull_block()
            + '<path d="M-130 -38 L 110 -38 L 130 0 L -150 0 Z" fill="#dfe7ec" stroke="#5e6b75"/>'
            + '<path d="M-100 -72 L 84 -72 L 110 -38 L -116 -38 Z" fill="#eef3f7" stroke="#5e6b75"/>'
            + '<rect x="-72" y="-102" width="148" height="28" rx="3" fill="#dfe7ec" stroke="#5e6b75"/>'
            + '<rect x="-32" y="-130" width="64" height="22" rx="3" fill="#dfe7ec" stroke="#5e6b75"/>'
            + ''.join(f'<rect x="{-110 + j*32}" y="-60" width="22" height="14" fill="#1c2c3a" rx="2"/>' for j in range(7))
            + ''.join(f'<rect x="{-58 + j*30}" y="-92" width="20" height="12" fill="#1c2c3a" rx="2"/>' for j in range(5))
        )
    elif type_slug == "sailboat":
        sup = (
            hull_block()
            + '<path d="M-22 -200 L -10 -8 L 18 -8 Z" fill="#ffffff" stroke="#5e6b75"/>'
            + '<path d="M2 -190 L 90 -20 L 6 -20 Z" fill="#f5f7fa" stroke="#5e6b75"/>'
            + '<line x1="0" y1="-200" x2="0" y2="0" stroke="#5e6b75" stroke-width="2"/>'
            + '<rect x="-40" y="-30" width="80" height="22" rx="3" fill="#dfe7ec" stroke="#5e6b75"/>'
        )
    elif type_slug == "pontoon":
        sup = (
            '<rect x="-180" y="-6" width="360" height="22" rx="4" fill="url(#hull)" stroke="#5e6b75"/>'
            + '<rect x="-180" y="-2" width="360" height="3" fill="#0c4ea1"/>'
            + '<ellipse cx="-160" cy="22" rx="20" ry="8" fill="#7d8893" opacity=".7"/>'
            + '<ellipse cx="160" cy="22" rx="20" ry="8" fill="#7d8893" opacity=".7"/>'
            + '<rect x="-150" y="-58" width="300" height="52" rx="6" fill="#eef3f7" stroke="#5e6b75"/>'
            + '<rect x="-46" y="-100" width="92" height="42" rx="5" fill="#dfe7ec" stroke="#5e6b75"/>'
            + ''.join(f'<rect x="{-130 + j*40}" y="-46" width="28" height="22" fill="#1c2c3a" rx="2"/>' for j in range(7))
        )
    elif type_slug == "pwc":
        sup = (
            '<path d="M-90 0 Q -110 -30 0 -34 Q 110 -30 90 0 L 70 16 L -70 16 Z" fill="url(#hull)" stroke="#5e6b75"/>'
            + '<path d="M-28 -54 L 28 -54 L 36 -34 L -36 -34 Z" fill="#dfe7ec" stroke="#5e6b75"/>'
            + '<rect x="-18" y="-72" width="36" height="20" rx="6" fill="#5b6671"/>'
            + '<line x1="-30" y1="-72" x2="-46" y2="-92" stroke="#5b6671" stroke-width="4" stroke-linecap="round"/>'
            + '<line x1="30" y1="-72" x2="46" y2="-92" stroke="#5b6671" stroke-width="4" stroke-linecap="round"/>'
        )
    elif type_slug == "centerconsole":
        sup = (
            hull_block()
            + '<rect x="-22" y="-72" width="56" height="56" rx="4" fill="#eef3f7" stroke="#5e6b75"/>'
            + '<rect x="-18" y="-66" width="48" height="20" fill="#1c2c3a" rx="2"/>'
            + '<rect x="-34" y="-106" width="80" height="6" rx="2" fill="#5b6671"/>'
            + '<rect x="-30" y="-106" width="6" height="34" fill="#5b6671"/>'
            + '<rect x="38" y="-106" width="6" height="34" fill="#5b6671"/>'
            + '<rect x="150" y="-12" width="14" height="22" rx="2" fill="#1c2c3a"/>'  # outboard
        )
    elif type_slug == "bass":
        sup = (
            hull_block()
            + '<rect x="-150" y="-32" width="300" height="32" fill="#dfe7ec" stroke="#5e6b75"/>'
            + '<rect x="-30" y="-58" width="58" height="26" fill="#1c2c3a" rx="2"/>'
            + '<circle cx="-100" cy="-10" r="6" fill="#2c8a3e"/>'
            + '<circle cx="100" cy="-10" r="6" fill="#2c8a3e"/>'
        )
    elif type_slug == "trawler":
        sup = (
            hull_block()
            + '<path d="M-120 -40 L 110 -40 L 130 0 L -140 0 Z" fill="#dfe7ec" stroke="#5e6b75"/>'
            + '<rect x="-80" y="-92" width="160" height="52" rx="3" fill="#eef3f7" stroke="#5e6b75"/>'
            + '<rect x="-20" y="-138" width="40" height="46" rx="3" fill="#dfe7ec" stroke="#5e6b75"/>'
            + ''.join(f'<rect x="{-72 + j*28}" y="-78" width="20" height="14" fill="#1c2c3a" rx="2"/>' for j in range(6))
        )
    elif type_slug == "cruiser":
        sup = (
            hull_block()
            + '<path d="M-130 -58 L 100 -58 L 130 0 L -150 0 Z" fill="#dfe7ec" stroke="#5e6b75"/>'
            + '<rect x="-44" y="-92" width="98" height="34" rx="3" fill="#eef3f7" stroke="#5e6b75"/>'
            + ''.join(f'<rect x="{-110 + j*38}" y="-46" width="28" height="14" fill="#1c2c3a" rx="2"/>' for j in range(6))
        )
    else:  # bowrider / runabout / generic powerboat
        sup = (
            hull_block()
            + '<path d="M-110 -54 L 90 -54 L 110 0 L -130 0 Z" fill="#dfe7ec" stroke="#5e6b75"/>'
            + ''.join(f'<rect x="{-100 + j*40}" y="-44" width="32" height="16" fill="#1c2c3a" rx="2"/>' for j in range(5))
            + '<rect x="-30" y="-86" width="60" height="6" rx="2" fill="#5b6671"/>'
            + '<rect x="-26" y="-86" width="6" height="32" fill="#5b6671"/>'
            + '<rect x="20" y="-86" width="6" height="32" fill="#5b6671"/>'
        )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 480" preserveAspectRatio="xMidYMid slice">
  <defs>
    <linearGradient id="sky" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0" stop-color="#e9f4fb"/><stop offset="1" stop-color="{sky}"/>
    </linearGradient>
    <linearGradient id="sea" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0" stop-color="{sky}"/><stop offset="1" stop-color="{water}"/>
    </linearGradient>
    <linearGradient id="hull" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0" stop-color="{hull}"/><stop offset="1" stop-color="#c3ccd2"/>
    </linearGradient>
  </defs>
  <rect width="640" height="260" fill="url(#sky)"/>
  <rect y="260" width="640" height="220" fill="url(#sea)"/>
  <!-- distant landmass strip -->
  <path d="M0 252 L 90 240 L 180 250 L 280 244 L 380 250 L 480 246 L 580 254 L 640 248 L 640 264 L 0 264 Z" fill="#314c63" opacity=".5"/>
  <g transform="translate({320 + cam_tilt*4}, {boat_y}) rotate({cam_tilt})">{sup}
    <ellipse cx="-200" cy="40" rx="40" ry="6" fill="#ffffff" opacity="0.5"/>
    <ellipse cx="200" cy="40" rx="40" ry="6" fill="#ffffff" opacity="0.5"/>
  </g>
  <text x="624" y="468" font-family="Arial" font-size="11" fill="#ffffff" opacity="0.55" text-anchor="end">Boat Trader photo {idx + 1}/9</text>
</svg>'''


_TYPE_VARIANT = {
    "runabout":       ("#bcd6ea", "#4189c6", "runabout"),
    "centerconsole":  ("#a8cce6", "#36739f", "centerconsole"),
    "pontoon":        ("#cfdde9", "#5a8db1", "pontoon"),
    "bowrider":       ("#a3c4e1", "#1f6cd1", "bowrider"),
    "cruiser":        ("#9ec0dd", "#2a5d8a", "cruiser"),
    "sailboat":       ("#c0d4e3", "#3b6f97", "sailboat"),
    "yacht":          ("#94b8d9", "#0e3a5f", "yacht"),
    "trawler":        ("#a5c6df", "#34688f", "trawler"),
    "bass":           ("#b4d0e6", "#2b5a82", "bass"),
    "pwc":            ("#c3dbef", "#3d77a4", "pwc"),
}


def _render_type_svg(slug: str) -> str:
    sky, water, variant = _TYPE_VARIANT.get(slug, ("#bcd6ea", "#4189c6", "runabout"))
    # Per-variant silhouettes — sized to fit a 300x180 viewBox without
    # leaking outside, and rendered against a sky+water gradient so the
    # tile's white label area below stays clean.
    if variant == "sailboat":
        boat = '''<polygon points="0,-12 0,-78 38,-12" fill="#ffffff" stroke="#1d3a5c"/>
                  <path d="M-50 6 L 50 6 L 38 24 L -38 24 Z" fill="#ffffff" stroke="#1d3a5c"/>'''
    elif variant == "yacht":
        boat = '''<path d="M-80 0 L 78 0 L 62 22 L -64 22 Z" fill="#ffffff" stroke="#1d3a5c"/>
                  <rect x="-58" y="-26" width="116" height="22" rx="3" fill="#ffffff" stroke="#1d3a5c"/>
                  <rect x="-32" y="-44" width="64" height="18" rx="3" fill="#ffffff" stroke="#1d3a5c"/>
                  <rect x="-14" y="-58" width="28" height="14" rx="3" fill="#ffffff" stroke="#1d3a5c"/>'''
    elif variant == "pontoon":
        boat = '''<rect x="-78" y="-10" width="156" height="14" rx="2" fill="#ffffff" stroke="#1d3a5c"/>
                  <rect x="-60" y="-32" width="120" height="22" rx="3" fill="#e7eef2" stroke="#1d3a5c"/>
                  <line x1="-78" y1="4" x2="-78" y2="14" stroke="#1d3a5c" stroke-width="2"/>
                  <line x1="78" y1="4" x2="78" y2="14" stroke="#1d3a5c" stroke-width="2"/>'''
    elif variant == "pwc":
        boat = '''<path d="M-44 0 q -8 -22 44 -22 q 50 0 44 22 z" fill="#ffffff" stroke="#1d3a5c"/>
                  <rect x="-18" y="-32" width="36" height="14" rx="6" fill="#ffffff" stroke="#1d3a5c"/>'''
    else:  # generic powerboat
        boat = '''<path d="M-72 0 L 72 0 L 58 18 L -58 18 Z" fill="#ffffff" stroke="#1d3a5c"/>
                  <rect x="-44" y="-22" width="88" height="22" rx="3" fill="#ffffff" stroke="#1d3a5c"/>
                  <rect x="-20" y="-36" width="40" height="14" rx="3" fill="#ffffff" stroke="#1d3a5c"/>'''
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 180" preserveAspectRatio="xMidYMid slice">
  <defs>
    <linearGradient id="g" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0" stop-color="{sky}"/>
      <stop offset="1" stop-color="{water}"/>
    </linearGradient>
  </defs>
  <rect width="300" height="180" fill="url(#g)"/>
  <ellipse cx="80" cy="160" rx="120" ry="12" fill="rgba(0,0,0,0.08)"/>
  <ellipse cx="220" cy="166" rx="100" ry="10" fill="rgba(0,0,0,0.08)"/>
  <g transform="translate(150,118)">{boat}</g>
</svg>'''


# ---------------------------------------------------------------------------
# Module-level FastAPI app — uvicorn entrypoint.
# ---------------------------------------------------------------------------


app = create_app()
