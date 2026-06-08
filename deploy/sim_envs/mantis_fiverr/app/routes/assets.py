"""Procedural SVG assets — placeholder gig thumbnails, avatars, logo, icons.

Zero outbound network at runtime. All graphics generated from
deterministic seed-aware palettes.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import Response

router = APIRouter(prefix="/assets")

# ──────────────────────────────────────────────────────────────────────
# Palettes
# ──────────────────────────────────────────────────────────────────────

GIG_PALETTES = [
    ("#1dbf73", "#0f7048"),
    ("#5b3eff", "#3a26b3"),
    ("#ff7640", "#cf4b18"),
    ("#0aaff1", "#0271a4"),
    ("#ffb33e", "#c87a16"),
    ("#e84393", "#a02666"),
    ("#26c5b1", "#0e7d6f"),
    ("#444444", "#222222"),
]


AVATAR_PALETTES = [
    ("#1dbf73", "#fff"),
    ("#5b3eff", "#fff"),
    ("#ff7640", "#fff"),
    ("#0aaff1", "#fff"),
    ("#ffb33e", "#222"),
    ("#e84393", "#fff"),
    ("#26c5b1", "#fff"),
    ("#74767e", "#fff"),
]


# ──────────────────────────────────────────────────────────────────────


@router.get("/logo.svg")
async def logo_svg() -> Response:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 140 38" '
        'width="140" height="38" aria-label="Fiverr">'
        '<style>.f{font:700 30px Helvetica,Arial,sans-serif;fill:#222325}'
        '.g{fill:#1dbf73}</style>'
        '<text x="0" y="30" class="f">fiv</text>'
        '<text x="50" y="30" class="f">err</text>'
        '<circle cx="49" cy="9" r="4" class="g"/>'
        '<rect x="46" y="9" width="6" height="8" rx="1" class="g" transform="rotate(15 49 13)"/>'
        '<text x="115" y="30" class="f g">.</text>'
        '</svg>'
    )
    return Response(svg, media_type="image/svg+xml")


@router.get("/gig/{palette:int}_{idx:int}.svg")
async def gig_thumb(palette: int, idx: int) -> Response:
    a, b = GIG_PALETTES[palette % len(GIG_PALETTES)]
    # Solid gradient + simple geometric mark + caption strip at bottom.
    motifs = [
        '<circle cx="160" cy="120" r="56" fill="rgba(255,255,255,0.18)"/>'
        '<circle cx="160" cy="120" r="32" fill="rgba(255,255,255,0.35)"/>',
        '<path d="M40 160 L160 40 L280 160 Z" fill="rgba(255,255,255,0.22)"/>',
        '<rect x="80" y="60" width="160" height="120" rx="12" fill="rgba(255,255,255,0.2)"/>'
        '<rect x="100" y="80" width="120" height="20" rx="4" fill="rgba(255,255,255,0.45)"/>'
        '<rect x="100" y="110" width="80" height="14" rx="3" fill="rgba(255,255,255,0.35)"/>',
        '<polygon points="80,60 240,60 280,120 240,180 80,180 40,120" fill="rgba(255,255,255,0.2)"/>',
        '<g fill="rgba(255,255,255,0.3)"><circle cx="100" cy="100" r="22"/>'
        '<circle cx="180" cy="80" r="14"/><circle cx="220" cy="150" r="32"/></g>',
        '<g stroke="rgba(255,255,255,0.4)" stroke-width="3" fill="none">'
        '<path d="M30 150 Q160 30 290 150"/>'
        '<path d="M30 110 Q160 -10 290 110"/></g>',
        '<g fill="rgba(255,255,255,0.25)"><rect x="60" y="60" width="50" height="120" rx="4"/>'
        '<rect x="130" y="80" width="50" height="100" rx="4"/>'
        '<rect x="200" y="40" width="50" height="140" rx="4"/></g>',
        '<g fill="rgba(255,255,255,0.3)"><polygon points="160,40 200,140 100,140"/>'
        '<polygon points="220,80 260,180 180,180"/></g>',
    ]
    motif = motifs[idx % len(motifs)]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 220" preserveAspectRatio="xMidYMid slice">
  <defs>
    <linearGradient id="g" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0" stop-color="{a}"/><stop offset="1" stop-color="{b}"/>
    </linearGradient>
  </defs>
  <rect width="320" height="220" fill="url(#g)"/>
  {motif}
</svg>'''
    return Response(svg, media_type="image/svg+xml")


@router.get("/avatar/{palette:int}_{letter}.svg")
async def avatar(palette: int, letter: str) -> Response:
    a, fg = AVATAR_PALETTES[palette % len(AVATAR_PALETTES)]
    initial = (letter[:1] or "?").upper()
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 56 56" width="56" height="56">
  <circle cx="28" cy="28" r="28" fill="{a}"/>
  <text x="28" y="36" text-anchor="middle" font-family="Helvetica,Arial,sans-serif"
        font-size="22" font-weight="700" fill="{fg}">{initial}</text>
</svg>'''
    return Response(svg, media_type="image/svg+xml")


@router.get("/category/{slug}.svg")
async def category_icon(slug: str) -> Response:
    # 12 simple line icons keyed by slug, falling back to a generic square.
    color = "#1dbf73"
    icons = {
        "graphics-design": '<path d="M16 16 L40 16 L40 40 L16 40 Z" stroke-width="3" fill="none"/>'
                            '<path d="M22 22 L34 34 M34 22 L22 34" stroke-width="3"/>',
        "programming-tech": '<path d="M14 18 L8 28 L14 38" stroke-width="3" fill="none"/>'
                            '<path d="M42 18 L48 28 L42 38" stroke-width="3" fill="none"/>'
                            '<line x1="26" y1="14" x2="30" y2="42" stroke-width="3"/>',
        "digital-marketing": '<path d="M14 28 L42 16 L42 40 L14 28" stroke-width="3" fill="none"/>'
                              '<path d="M16 28 L16 38" stroke-width="3"/>',
        "video-animation": '<rect x="12" y="18" width="32" height="20" rx="3" fill="none" stroke-width="3"/>'
                            '<polygon points="26,22 26,34 38,28" />',
        "writing-translation": '<path d="M12 40 L24 16 L36 40 M16 32 L32 32" stroke-width="3" fill="none"/>',
        "music-audio": '<circle cx="20" cy="38" r="6" stroke-width="3" fill="none"/>'
                       '<circle cx="40" cy="34" r="6" stroke-width="3" fill="none"/>'
                       '<path d="M26 38 L26 16 L46 12 L46 34" stroke-width="3" fill="none"/>',
        "business": '<rect x="10" y="18" width="36" height="24" rx="3" stroke-width="3" fill="none"/>'
                    '<path d="M22 18 L22 12 L34 12 L34 18" stroke-width="3" fill="none"/>',
        "data": '<ellipse cx="28" cy="14" rx="16" ry="6" stroke-width="3" fill="none"/>'
                '<path d="M12 14 L12 42 Q12 48 28 48 Q44 48 44 42 L44 14" stroke-width="3" fill="none"/>'
                '<path d="M12 28 Q12 34 28 34 Q44 34 44 28" stroke-width="3" fill="none"/>',
        "photography": '<rect x="8" y="18" width="40" height="26" rx="3" stroke-width="3" fill="none"/>'
                       '<circle cx="28" cy="31" r="8" stroke-width="3" fill="none"/>'
                       '<rect x="18" y="14" width="10" height="6" stroke-width="3" fill="none"/>',
        "ai-services": '<path d="M28 8 L32 22 L46 22 L34 30 L38 44 L28 36 L18 44 L22 30 L10 22 L24 22 Z" stroke-width="3" fill="none"/>',
        "lifestyle": '<path d="M28 42 Q12 32 12 22 Q12 14 20 14 Q26 14 28 20 Q30 14 36 14 Q44 14 44 22 Q44 32 28 42 Z" stroke-width="3" fill="none"/>',
        "consulting": '<path d="M8 12 L48 12 L48 36 L20 36 L8 44 Z" stroke-width="3" fill="none"/>'
                       '<line x1="16" y1="20" x2="40" y2="20" stroke-width="3"/>'
                       '<line x1="16" y1="28" x2="32" y2="28" stroke-width="3"/>',
    }
    g = icons.get(slug, '<rect x="14" y="14" width="28" height="28" stroke-width="3" fill="none"/>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 56 56" width="48" height="48"
                  stroke="{color}" stroke-linecap="round" stroke-linejoin="round">
  {g}
</svg>'''
    return Response(svg, media_type="image/svg+xml")


@router.get("/hero.svg")
async def hero_bg() -> Response:
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 600" preserveAspectRatio="xMidYMid slice">
  <defs>
    <linearGradient id="g" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0" stop-color="#1a3e30"/>
      <stop offset="1" stop-color="#0a1410"/>
    </linearGradient>
    <radialGradient id="r" cx="0.7" cy="0.3" r="0.7">
      <stop offset="0" stop-color="#1dbf73" stop-opacity="0.45"/>
      <stop offset="1" stop-color="#0a1410" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect width="1440" height="600" fill="url(#g)"/>
  <rect width="1440" height="600" fill="url(#r)"/>
  <g fill="rgba(255,255,255,0.04)">
    <circle cx="200" cy="120" r="50"/>
    <circle cx="1100" cy="480" r="80"/>
    <circle cx="1300" cy="160" r="32"/>
  </g>
</svg>'''
    return Response(svg, media_type="image/svg+xml")
