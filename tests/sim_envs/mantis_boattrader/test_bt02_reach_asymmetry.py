"""The BT02 reach-asymmetry reseed must create a frozen-vs-S0 discriminator.

BT02 (Caterpillar-engine lead lookup) only separates a frozen agent from an
S0-retrieval agent if the Caterpillar listings sit *out of a frozen agent's
sequential reach* yet *inside the page-1 DOM* an S0 click-target hint can bias
toward. The stock seed defeats this by ranking a Caterpillar at slot 0 (a
frozen agent hits it for free); ``seed._apply_deep_caterpillar`` buries every
Caterpillar and re-pins a couple onto mid page-1 ranks.

These tests pin that behaviour and — critically — prove the reseed leaves the
other oracles' answer sets and run-to-run determinism intact, since it only
mutates ``views`` / ``badges`` on existing Caterpillar boats.
"""

from __future__ import annotations


# A frozen agent grinds ~2-3 boats before the budget cap. Anything at a rank
# below this is "free reach"; the discriminator requires the Caterpillars to
# sit beyond it. Kept loose (4 > observed ~2.4) so the test asserts the
# *property*, not a brittle exact cost.
_FROZEN_REACH = 4
_PER_PAGE = 24


def _recommended(boats):
    """Mirror of the env's default 'recommended' sort (db.py): Featured first,
    then by views descending."""
    return sorted(boats, key=lambda b: (0 if "Featured" in b.badges else 1, -b.views))


def _is_cat(b) -> bool:
    return (b.engine_make or "").strip().lower() == "caterpillar"


def _cat_ranks(boats) -> list[int]:
    return [i for i, b in enumerate(_recommended(boats)) if _is_cat(b)]


def _build(monkeypatch, *, deep: bool):
    """Fresh ``seed.build()`` with the reseed gate forced on/off."""
    from app import seed  # noqa: PLC0415

    monkeypatch.setenv("BT02_DEEP_CATERPILLAR", "1" if deep else "0")
    return seed.build()["boats"]


def test_caterpillar_count_and_make_preserved(monkeypatch):
    """The reseed must not drop or relabel a single Caterpillar — the oracle
    grades off ``engine_make`` and its answer set has to survive untouched."""
    deep = _build(monkeypatch, deep=True)
    stock = _build(monkeypatch, deep=False)

    deep_cats = [b for b in deep if _is_cat(b)]
    stock_cats = [b for b in stock if _is_cat(b)]
    assert deep_cats, "reseed produced zero Caterpillars"
    assert len(deep_cats) == len(stock_cats), (
        "reseed changed the Caterpillar count "
        f"({len(stock_cats)} → {len(deep_cats)})"
    )
    # Same boats, identified by id — only views/badges may differ.
    assert {b.id for b in deep_cats} == {b.id for b in stock_cats}


def test_no_caterpillar_within_frozen_reach(monkeypatch):
    """No Caterpillar may sit in the first few ranks a frozen agent walks for
    free — otherwise frozen passes BT02 and the discriminator collapses."""
    boats = _build(monkeypatch, deep=True)
    ranks = _cat_ranks(boats)
    assert ranks, "no Caterpillars in the catalog"
    assert min(ranks) >= _FROZEN_REACH, (
        f"a Caterpillar is at rank {min(ranks)} (< frozen reach "
        f"{_FROZEN_REACH}); frozen would hit it for free"
    )


def test_caterpillar_reachable_on_page_one(monkeypatch):
    """At least one Caterpillar must remain inside page 1 (past frozen's reach)
    so an S0 click-target hint has a same-viewport target to bias toward."""
    boats = _build(monkeypatch, deep=True)
    ranks = _cat_ranks(boats)
    page1 = [r for r in ranks if _FROZEN_REACH <= r < _PER_PAGE]
    assert page1, (
        "no Caterpillar in the S0-reachable band "
        f"[{_FROZEN_REACH}, {_PER_PAGE}); ranks were {ranks[:5]}…"
    )


def test_reseed_is_deterministic(monkeypatch):
    """Two builds at the same seed produce identical Caterpillar ranks — the
    reseed's RNG must not depend on dict/set iteration order."""
    r1 = _cat_ranks(_build(monkeypatch, deep=True))
    r2 = _cat_ranks(_build(monkeypatch, deep=True))
    assert r1 == r2, f"non-deterministic reseed: {r1[:5]}… != {r2[:5]}…"


def test_gate_off_restores_stock_order(monkeypatch):
    """``BT02_DEEP_CATERPILLAR=0`` is a clean off-switch: the stock seed ranks a
    Caterpillar inside the frozen-reach band, so gate-off must too."""
    ranks = _cat_ranks(_build(monkeypatch, deep=False))
    assert ranks, "no Caterpillars in the catalog"
    assert min(ranks) < _FROZEN_REACH, (
        "gate-off should leave the stock order (a Caterpillar within frozen "
        f"reach), but the shallowest was rank {min(ranks)}"
    )


def test_bt01_answer_set_unchanged(monkeypatch):
    """The reseed touches only Caterpillar views/badges, so BT01's qualifying
    set (used Sea Rays under a price cap) must be byte-identical gate-on vs
    gate-off — no collateral on a sibling oracle."""
    from app.oracles.bt01_lead_capture_filtered_search import (  # noqa: PLC0415
        _boat_matches_filters,
    )

    deep = _build(monkeypatch, deep=True)
    stock = _build(monkeypatch, deep=False)
    deep_ids = {b.id for b in deep if _boat_matches_filters(b)}
    stock_ids = {b.id for b in stock if _boat_matches_filters(b)}
    assert deep_ids, "expected BT01 to have a non-empty qualifying set"
    assert deep_ids == stock_ids, "reseed perturbed BT01's qualifying set"
