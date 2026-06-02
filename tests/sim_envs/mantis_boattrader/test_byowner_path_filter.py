"""BoatTrader-style ``/boats/<path-filters>/`` routing must resolve, not 404.

Real BoatTrader narrows listings through URL *path* segments
(``/boats/state-fl/by-owner/``) rather than a querystring, and the
BT03 by-owner plan was hardened to drive that scheme. The sim env
originally served only ``/boats/?query`` — so the plan's
``/boats/state-<code>/by-owner/`` navigation 404'd before the agent
ever reached an owner detail page, turning a genuine reveal-under-drift
failure into a navigation artifact.

These tests pin the path-filter route: the by-owner segment selects
exactly the private-seller listings, the state segment narrows by
location, a slightly-off ``state-all`` still renders (graceful) instead
of 404-ing, and the bare ``/boats/`` index is unaffected.
"""

from __future__ import annotations

import re


def _client():
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from app.main import create_app  # noqa: PLC0415

    return TestClient(create_app())


def _slugs(html: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"/boat/([a-z0-9\-]+)/", html)))


def _listing_types(slugs: list[str]) -> set[str]:
    from app import db  # noqa: PLC0415

    out: set[str] = set()
    for sl in slugs:
        b = db.boat_by_slug(sl)
        if b is not None:
            out.add(b.listing_type)
    return out


def test_state_all_by_owner_resolves(monkeypatch):
    """The exact path the live frozen run hit (``state-all`` → all states,
    by-owner) must return 200, not the 404 that stranded the agent at step 4."""
    monkeypatch.setenv("SEED", "42")
    with _client() as c:
        r = c.get("/boats/state-all/by-owner/")
    assert r.status_code == 200, f"state-all/by-owner 404'd again: {r.status_code}"
    assert _slugs(r.text), "by-owner page rendered no listings"


def test_by_owner_returns_only_owner_listings(monkeypatch):
    """``by-owner`` selects exactly the private-seller listings — a dealer or
    sponsored card leaking in would let the agent mis-target a non-reveal page."""
    monkeypatch.setenv("SEED", "42")
    with _client() as c:
        slugs = _slugs(c.get("/boats/by-owner/").text)
    assert slugs, "by-owner page rendered no listings"
    assert _listing_types(slugs) == {"owner"}, "non-owner listing leaked onto by-owner"


def test_by_dealer_excludes_owner_listings(monkeypatch):
    """The mirror filter: ``by-dealer`` must never surface an owner listing."""
    monkeypatch.setenv("SEED", "42")
    with _client() as c:
        slugs = _slugs(c.get("/boats/by-dealer/").text)
    assert slugs, "by-dealer page rendered no listings"
    assert "owner" not in _listing_types(slugs), "owner listing leaked onto by-dealer"


def test_state_segment_narrows_by_location(monkeypatch):
    """``state-fl`` constrains to Florida — the page must carry FL owner boats
    and no out-of-state listing."""
    monkeypatch.setenv("SEED", "42")
    from app import db  # noqa: PLC0415

    with _client() as c:
        slugs = _slugs(c.get("/boats/state-fl/by-owner/").text)
    assert slugs, "FL by-owner page rendered no listings"
    states = {db.boat_by_slug(s).state for s in slugs}
    assert states == {"FL"}, f"state-fl leaked other states: {states}"


def test_bare_boats_index_unaffected(monkeypatch):
    """The exact ``/boats/`` index must still render the unfiltered catalog
    (owner + dealer + sponsored) — the path route is declared after it."""
    monkeypatch.setenv("SEED", "42")
    with _client() as c:
        slugs = _slugs(c.get("/boats/").text)
    assert slugs, "index rendered no listings"
    # The unfiltered first page should not be owner-only.
    assert _listing_types(slugs) - {"owner"}, "index collapsed to owner-only"
