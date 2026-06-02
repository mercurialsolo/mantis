"""The BT03 reveal-drift fixture must create a frozen-vs-S1 discriminator.

BT03 (by-owner phone reveal) only separates a frozen agent from an S1-exemplar
agent if revealing a private seller's phone is *not* the obvious move. The stock
private-seller card pairs a prominent "Contact Seller" lead form with an explicit
"Show Phone Number" button — a frozen agent clicks that button for free and the
discriminator collapses.

``seed._apply_byowner_reveal_drift`` flags every owner listing so the detail page
runs the *action-omission* drift: it shows a masked teaser of the number inline,
so the contact reads as already-populated. A frozen agent grabs the visible
number and (per the plan's "reveal the phone if not already") omits the reveal
click — a recall miss — while an S1 worked-reveal exemplar, primed that a click
makes the FULL number appear (which the masked teaser does not satisfy), still
seeks out the reveal control and fires it. The reveal target keeps the verbatim
"Show Phone Number" label (matching the plan + exemplar); the *only* drift is the
inline masked teaser plus de-emphasised (muted-link) styling. The lever is recall,
not target-identification. (Two earlier *relabel* variants — "View seller details"
and "Show full number" — each dropped the label and each made a live matrix
collapse, defeating BOTH arms equally: the S1 exemplar replays an action, not a
label, so a relabel it can't ground breaks S1 just as it breaks frozen.)

These tests pin the fixture's data shape (the flag lands only on owner boats, the
gate toggles it, it's deterministic, the reveal target survives) and prove the
rendered detail page actually drifts — without ever asserting a per-seed boat id.
"""

from __future__ import annotations


def _build(monkeypatch, *, drift: bool):
    """Fresh ``seed.build()`` with the reveal-drift gate forced on/off."""
    from app import seed  # noqa: PLC0415

    monkeypatch.setenv("BT03_REVEAL_DRIFT", "1" if drift else "0")
    return seed.build()["boats"]


# ── fixture data shape ───────────────────────────────────────────────────


def test_drift_flags_every_owner_and_only_owners(monkeypatch):
    """Gate-on flags exactly the owner listings — the reveal control only exists
    on owner cards, so flagging a dealer/sponsored boat would be dead collateral."""
    boats = _build(monkeypatch, drift=True)
    owners = [b for b in boats if b.is_owner_listed]
    assert owners, "seed produced zero owner listings"
    assert all(b.reveal_drift for b in owners), "an owner listing was left undrifted"
    assert not any(b.reveal_drift for b in boats if not b.is_owner_listed), (
        "a non-owner boat was flagged for reveal drift"
    )


def test_gate_off_leaves_no_drift(monkeypatch):
    """``BT03_REVEAL_DRIFT=0`` is a clean off-switch: nothing is flagged and the
    stock "Show Phone Number" layout renders for every owner listing."""
    boats = _build(monkeypatch, drift=False)
    assert any(b.is_owner_listed for b in boats), "seed produced zero owner listings"
    assert not any(b.reveal_drift for b in boats), "gate-off should flag nothing"


def test_drift_preserves_reveal_target(monkeypatch):
    """The drift changes how the reveal control *looks*, never whether there is a
    phone to reveal — every flagged owner must still carry ``owner_phone``."""
    flagged = [b for b in _build(monkeypatch, drift=True) if b.reveal_drift]
    assert flagged, "no owner listings were flagged"
    assert all(b.owner_phone for b in flagged), "a drifted owner lost its phone"


def test_drift_is_deterministic(monkeypatch):
    """Same seed → same flagged id set (the pass must not depend on iteration
    order), so the discriminator is reproducible across runs."""
    a = {b.id for b in _build(monkeypatch, drift=True) if b.reveal_drift}
    b = {x.id for x in _build(monkeypatch, drift=True) if x.reveal_drift}
    assert a and a == b, f"non-deterministic drift set ({len(a)} vs {len(b)})"


def test_bt03_qualifying_set_unchanged(monkeypatch):
    """The BT03 oracle grades by-owner reveals; its qualifying owner-id set must
    be identical gate-on vs gate-off — the flag never touches ``listing_type``."""
    on = {b.id for b in _build(monkeypatch, drift=True) if b.is_owner_listed}
    off = {b.id for b in _build(monkeypatch, drift=False) if b.is_owner_listed}
    assert on and on == off, "reveal drift perturbed BT03's qualifying set"


# ── rendered layout drift (TestClient) ───────────────────────────────────


def _client():
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from app.main import create_app  # noqa: PLC0415

    return TestClient(create_app())


def _an_owner_slug(monkeypatch, *, drift: bool) -> str:
    owners = [b for b in _build(monkeypatch, drift=drift) if b.is_owner_listed]
    assert owners, "no owner listing to render"
    return owners[0].slug


def test_drifted_owner_page_shows_inline_decoy_keeps_reveal(monkeypatch):
    """Gate-on (action-omission): a masked teaser of the number renders inline so
    the contact reads as already-populated, while the reveal endpoint + testid
    survive (mechanism + oracle mutation unchanged) behind a muted but *verbatim*-
    labelled control. The inline decoy is the reason a frozen agent omits the
    reveal; the surviving control is what an S1 exemplar re-finds and fires."""
    slug = _an_owner_slug(monkeypatch, drift=True)
    with _client() as c:
        html = c.get(f"/boat/{slug}/").text
    assert 'data-testid="private-seller-card"' in html, "not an owner detail page"
    # The inline masked teaser — deterministic mask infix, no per-seed digits.
    assert 'data-testid="seller-phone-inline"' in html
    assert ") •••-••" in html
    # The reveal mechanism the oracle grades survives — verbatim label retained,
    # only the styling is de-emphasised (muted link, not a prominent button), so
    # grounding is NOT a confound. The lever is purely the teaser + de-emphasis.
    assert 'data-testid="show-phone-btn"' in html
    assert f"/boat/{slug}/show-phone" in html
    assert "Show Phone Number" in html
    assert "bdp-reveal-link" in html
    # Neither failed relabel variant is present — the label is NOT the lever.
    assert "Show full number" not in html
    assert "View seller details" not in html


def test_stock_owner_page_shows_phone_label(monkeypatch):
    """Gate-off: the owner detail page renders the explicit "Show Phone Number"
    button with no inline decoy — the easy path a frozen agent takes for free."""
    slug = _an_owner_slug(monkeypatch, drift=False)
    with _client() as c:
        html = c.get(f"/boat/{slug}/").text
    assert 'data-testid="private-seller-card"' in html
    assert "Show Phone Number" in html
    assert "View seller details" not in html
    assert 'data-testid="seller-phone-inline"' not in html
    # Stock uses the prominent button styling, never the muted drift-link.
    assert "bdp-reveal-link" not in html
