"""The BT03 reveal-drift fixture must create a frozen-vs-S1 discriminator.

BT03 (by-owner phone reveal) only separates a frozen agent from an S1-exemplar
agent if revealing a private seller's phone is *not* the obvious move. The stock
private-seller card pairs a prominent "Contact Seller" lead form with an explicit
"Show Phone Number" button — a frozen agent clicks that button for free and the
discriminator collapses. ``seed._apply_byowner_reveal_drift`` flags every owner
listing so the detail page renders the reveal control de-emphasised + relabelled
(no "phone" keyword); a frozen agent then favours the lead form (recall miss)
while an S1 worked-reveal exemplar re-finds the control by its action.

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


def test_drifted_owner_page_hides_phone_label(monkeypatch):
    """Gate-on: the reveal endpoint + testid survive (mechanism + oracle mutation
    unchanged) but the phone-labelled affordance is replaced by a muted link —
    that relabel is what makes a frozen agent mis-target the lead form."""
    slug = _an_owner_slug(monkeypatch, drift=True)
    with _client() as c:
        html = c.get(f"/boat/{slug}/").text
    assert 'data-testid="private-seller-card"' in html, "not an owner detail page"
    assert 'data-testid="show-phone-btn"' in html
    assert f"/boat/{slug}/show-phone" in html
    assert "Show Phone Number" not in html
    assert "View seller details" in html


def test_stock_owner_page_shows_phone_label(monkeypatch):
    """Gate-off: the owner detail page renders the explicit "Show Phone Number"
    button — the easy path a frozen agent takes for free."""
    slug = _an_owner_slug(monkeypatch, drift=False)
    with _client() as c:
        html = c.get(f"/boat/{slug}/").text
    assert 'data-testid="private-seller-card"' in html
    assert "Show Phone Number" in html
    assert "View seller details" not in html
