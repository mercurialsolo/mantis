"""Regression: the BDP stats-strip 'Engine' card must show the boat's true
engine MAKE as its leading token and must never pair a make with a competing
manufacturer's model.

The pre-fix bug: ``boat_detail.html`` built the Engine stat-card value from a
single shared series pool::

    boat.engine_make ~ ' ' ~ (['ZR6','V12','V8','MerCruiser','F300','VX1100',
                              'XS500'])[boat.year % 7]

so a Caterpillar boat rendered cross-brand nonsense like ``Caterpillar
MerCruiser`` (MerCruiser is a Mercury brand) / ``Caterpillar F300`` (a Yamaha
outboard) / ``Caterpillar VX1100`` (a Yamaha WaveRunner). The BT02
``detect_visible`` engine-make guard reads exactly this card, saw a competing
brand, and false-negated real Caterpillar boats — so a frozen agent walked
right past the Caterpillar listings and ``leads_total`` stayed 0. Surfaced on
the BT02 frozen smoke 2026-06-01 (both tasks scored 0.0 at $0.08 despite 60
qualifying boats; at the time two Caterpillars sat at page-1 ranks 0 and 6).

The later reach-asymmetry reseed (``seed._apply_deep_caterpillar``, exercised
by ``test_bt02_reach_asymmetry``) deliberately buries the Caterpillars and
re-pins only a couple onto mid page-1 ranks, so this test no longer assumes any
particular rank — it walks the engine card of *every* Caterpillar by slug.

The fix keys the series pool by make so the card reads e.g. ``Caterpillar
C32`` — make-consistent, no competing brand, ``detect_visible`` keys cleanly on
the first token. Display-only: the oracle grades off ``boat.engine_make``, not
this string.
"""

from __future__ import annotations

import re

import pytest


pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("jinja2")

# Model designations that belong to a *specific* manufacturer. None of these may
# follow a different make on the Engine card (e.g. "Caterpillar MerCruiser").
CROSS_BRAND_TOKENS = ("MerCruiser", "F300", "VX1100", "VX 1100")

_ENGINE_CARD = re.compile(
    r'stat-label">Engine</span><span class="stat-value">([^<]+)</span>'
)


def _cross_brand_hit(value: str) -> str | None:
    """Return the first retired cross-brand token present as a *whole token* in
    ``value``, else None. Whole-token (word-boundary) matching avoids false
    positives from legitimate same-brand series that merely embed the substring
    — e.g. Suzuki's real ``DF300B`` contains ``F300`` but is not the Yamaha
    ``F300`` outboard, and is make-consistent."""
    for bad in CROSS_BRAND_TOKENS:
        if re.search(rf"\b{re.escape(bad)}\b", value):
            return bad
    return None


@pytest.fixture
def client():
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from app.main import create_app  # noqa: PLC0415

    app = create_app()
    with TestClient(app) as c:
        yield c


def _engine_card_value(client, slug: str) -> str:
    r = client.get(f"/boat/{slug}/")
    assert r.status_code == 200, (slug, r.status_code)
    m = _ENGINE_CARD.search(r.text)
    assert m, f"no Engine stat-card on /boat/{slug}/"
    return m.group(1).strip()


def _boats_by_engine():
    from app import db  # noqa: PLC0415

    cats, others = [], []
    for b in db.store().boats:
        (cats if (b.engine_make or "").lower() == "caterpillar" else others).append(b)
    return cats, others


def test_caterpillar_card_leads_with_make_and_has_no_cross_brand(client):
    """Every Caterpillar boat's Engine card starts with 'Caterpillar ' and
    carries no competing-manufacturer model token."""
    cats, _ = _boats_by_engine()
    assert cats, "seed should contain Caterpillar boats"

    for b in cats:
        value = _engine_card_value(client, b.slug)
        assert value.startswith("Caterpillar "), (b.slug, value)
        hit = _cross_brand_hit(value)
        assert hit is None, f"{b.slug}: cross-brand '{hit}' in '{value}'"


def test_engine_card_first_token_matches_true_make(client):
    """The card's leading token(s) equal the boat's real make for a sample of
    non-Caterpillar boats too — so the guard reads False on them, not a
    coincidental 'Caterpillar' bleed-through from a shared pool."""
    _, others = _boats_by_engine()
    # One representative per distinct make keeps the HTTP round-trips bounded.
    seen: dict[str, object] = {}
    for b in others:
        seen.setdefault(b.engine_make, b)
    assert seen, "seed should contain non-Caterpillar boats"

    for make, b in seen.items():
        value = _engine_card_value(client, b.slug)
        assert value.startswith(f"{make} "), (b.slug, make, value)
        assert "Caterpillar" not in value, (b.slug, value)


def test_no_engine_card_uses_the_retired_shared_pool(client):
    """Belt-and-suspenders: the retired cross-brand tokens must not appear on
    ANY boat's Engine card, regardless of make."""
    cats, others = _boats_by_engine()
    # Sample broadly but bounded: all Caterpillar + one per other make.
    sample = list(cats)
    by_make: dict[str, object] = {}
    for b in others:
        by_make.setdefault(b.engine_make, b)
    sample.extend(by_make.values())

    for b in sample:
        value = _engine_card_value(client, b.slug)
        hit = _cross_brand_hit(value)
        assert hit is None, f"{b.slug} ({b.engine_make}): '{hit}' in '{value}'"
