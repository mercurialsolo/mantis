"""The BT03 reveal-GATE fixture must create a frozen-vs-S1 *injection* discriminator.

The sibling ``reveal_drift`` fixture de-emphasises an UNCHANGED reveal so a frozen
agent omits a click it already has in plan. The gate is a different lever: it adds
a PREREQUISITE the base plan does not contain at all. ``seed._apply_byowner_reveal_gate``
(env ``BT03_REVEAL_GATE``) flags every owner listing so the seller requires a
"start contact request" before the number unlocks — ``/show-phone`` only emits the
``phone_revealed`` mutation once ``bt_contact_start_<id>`` is set.

A frozen agent runs the plan as-authored (navigate → Show Phone Number) and the
server, seeing no contact-start cookie, reveals nothing → recall miss. The S1
injection seam (``apply_exemplar_overlay`` with an ``inject_before`` exemplar)
inserts the missing "start contact request" step ahead of the reveal, so S1
satisfies the gate and the same unchanged reveal fires.

These tests pin the fixture's data shape (the flag lands only on owner boats, it's
deterministic, BT03's qualifying set survives) and prove the gate's *behaviour*:
show-phone without the prerequisite emits no mutation; with it, exactly one — and
gate-off keeps the stock direct-reveal path intact.
"""

from __future__ import annotations


def _build(monkeypatch, *, gate: bool):
    """Fresh ``seed.build()`` with the reveal-gate forced on/off."""
    from app import seed  # noqa: PLC0415

    monkeypatch.setenv("BT03_REVEAL_GATE", "1" if gate else "0")
    return seed.build()["boats"]


def _an_owner(monkeypatch, *, gate: bool):
    owners = [b for b in _build(monkeypatch, gate=gate) if b.is_owner_listed]
    assert owners, "seed produced zero owner listings"
    return owners[0]


def _client():
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from app.main import create_app  # noqa: PLC0415

    return TestClient(create_app())


def _reveals():
    from app import db  # noqa: PLC0415

    return [m for m in db.list_mutations() if m.get("operation") == "phone_revealed"]


# ── fixture data shape ───────────────────────────────────────────────────


def test_gate_flags_every_owner_and_only_owners(monkeypatch):
    """Gate-on flags exactly the owner listings — the reveal control only exists
    on owner cards, so flagging a dealer boat would be dead collateral."""
    boats = _build(monkeypatch, gate=True)
    owners = [b for b in boats if b.is_owner_listed]
    assert owners, "seed produced zero owner listings"
    assert all(b.reveal_gated for b in owners), "an owner listing was left ungated"
    assert not any(b.reveal_gated for b in boats if not b.is_owner_listed), (
        "a non-owner boat was flagged for the reveal gate"
    )


def test_gate_off_leaves_no_gate(monkeypatch):
    """``BT03_REVEAL_GATE=0`` is a clean off-switch: nothing is flagged and the
    stock direct-reveal layout renders for every owner listing."""
    boats = _build(monkeypatch, gate=False)
    assert any(b.is_owner_listed for b in boats), "seed produced zero owner listings"
    assert not any(b.reveal_gated for b in boats), "gate-off should flag nothing"


def test_gate_preserves_reveal_target(monkeypatch):
    """The gate changes *when* the reveal fires, never whether there is a phone to
    reveal — every flagged owner must still carry ``owner_phone``."""
    flagged = [b for b in _build(monkeypatch, gate=True) if b.reveal_gated]
    assert flagged, "no owner listings were flagged"
    assert all(b.owner_phone for b in flagged), "a gated owner lost its phone"


def test_gate_is_deterministic(monkeypatch):
    """Same seed → same flagged id set, so the discriminator is reproducible."""
    a = {b.id for b in _build(monkeypatch, gate=True) if b.reveal_gated}
    b = {x.id for x in _build(monkeypatch, gate=True) if x.reveal_gated}
    assert a and a == b, f"non-deterministic gate set ({len(a)} vs {len(b)})"


def test_bt03_qualifying_set_unchanged(monkeypatch):
    """The BT03 oracle grades by-owner reveals; its qualifying owner-id set must be
    identical gate-on vs gate-off — the flag never touches ``listing_type``."""
    on = {b.id for b in _build(monkeypatch, gate=True) if b.is_owner_listed}
    off = {b.id for b in _build(monkeypatch, gate=False) if b.is_owner_listed}
    assert on and on == off, "the reveal gate perturbed BT03's qualifying set"


# ── rendered layout (TestClient) ─────────────────────────────────────────


def test_gated_owner_page_leads_with_prereq_and_suppresses_lead_form(monkeypatch):
    """Gate-on: the owner page LEADS with the "start contact request" prerequisite as a
    prominent primary button and renders the unchanged, verbatim-labelled reveal control
    — no masked teaser (the gate is not the drift). While the gate is pending the
    Name/Email/Phone lead form is SUPPRESSED: its prominent "Contact Seller" submit is the
    attractor that out-competed the injected prerequisite click and collapsed a live 2-arm
    smoke (S1 filled the form instead of starting the contact request). The prerequisite is
    the step the base plan omits; the reveal is what fires once the gate is satisfied."""
    boat = _an_owner(monkeypatch, gate=True)
    with _client() as c:
        html = c.get(f"/boat/{boat.slug}/").text
    assert 'data-testid="private-seller-card"' in html, "not an owner detail page"
    # The omitted prerequisite — present, groundable, POSTs to contact-start…
    assert 'data-testid="contact-start-btn"' in html
    assert f"/boat/{boat.slug}/contact-start" in html
    assert "Start contact request" in html
    # …and it LEADS as a prominent primary button, not a muted link, so the injected
    # S1 click has an unambiguous dominant target.
    assert "btn-primary btn-block contact-start-btn" in html
    # The competing lead form is suppressed while the gate is pending — removing the
    # attractor is the whole point of this layout.
    assert 'data-testid="dealer-lead-form"' not in html, "lead form not suppressed while gate pending"
    assert 'data-testid="contact-seller-submit"' not in html, "lead-form submit leaked while gate pending"
    # The reveal survives verbatim; gate is orthogonal to the masked-teaser drift.
    assert 'data-testid="show-phone-btn"' in html
    assert "Show Phone Number" in html
    assert 'data-testid="seller-phone-inline"' not in html


def test_started_contact_hides_prereq_restores_lead_form_keeps_reveal(monkeypatch):
    """Once the contact request is started the prerequisite control disappears so it
    never blocks grounding the reveal — the reveal itself stays present, and the standard
    Name/Email/Phone lead form RETURNS (it only hid while the gate was pending, so this
    layout is invisible to ungated and post-gate states)."""
    boat = _an_owner(monkeypatch, gate=True)
    with _client() as c:
        c.post(f"/boat/{boat.slug}/contact-start", follow_redirects=False)
        html = c.get(f"/boat/{boat.slug}/").text
    assert 'data-testid="contact-start-btn"' not in html, "prereq not hidden once started"
    assert 'data-testid="show-phone-btn"' in html, "reveal control vanished"
    assert "Show Phone Number" in html
    # The lead form returns once the gate clears.
    assert 'data-testid="dealer-lead-form"' in html, "lead form did not return after contact-start"


# ── gate behaviour: the mutation is what the oracle grades ────────────────


def test_show_phone_without_prereq_emits_no_mutation(monkeypatch):
    """Gate-on, no prerequisite: clicking Show Phone Number is a no-op — no
    ``phone_revealed`` mutation, phone SVG stays hidden. This is the frozen miss."""
    boat = _an_owner(monkeypatch, gate=True)
    with _client() as c:
        r = c.post(f"/boat/{boat.slug}/show-phone", follow_redirects=False)
        svg = c.get(f"/assets/phone/{boat.slug}.svg")
    assert r.status_code == 303
    assert _reveals() == [], "gated reveal fired without the prerequisite"
    assert svg.status_code == 403, "phone unlocked without the prerequisite"


def test_prereq_then_show_phone_emits_one_reveal(monkeypatch):
    """Gate-on, prerequisite satisfied: the contact-start step unlocks the reveal,
    which then emits exactly one ``phone_revealed`` mutation for this owner. This is
    the S1 path (the injected step + the unchanged reveal)."""
    boat = _an_owner(monkeypatch, gate=True)
    with _client() as c:
        c.post(f"/boat/{boat.slug}/contact-start", follow_redirects=False)
        c.post(f"/boat/{boat.slug}/show-phone", follow_redirects=False)
        svg = c.get(f"/assets/phone/{boat.slug}.svg")
    reveals = _reveals()
    assert len(reveals) == 1, f"expected one reveal, got {len(reveals)}"
    assert reveals[0]["target_id"] == boat.id
    assert svg.status_code == 200, "phone stayed hidden after the prerequisite"


def test_gate_off_show_phone_reveals_directly(monkeypatch):
    """Gate-off regression: the stock direct-reveal path is untouched — Show Phone
    Number emits the mutation with no prerequisite, and no prereq control renders."""
    boat = _an_owner(monkeypatch, gate=False)
    with _client() as c:
        html = c.get(f"/boat/{boat.slug}/").text
        c.post(f"/boat/{boat.slug}/show-phone", follow_redirects=False)
        svg = c.get(f"/assets/phone/{boat.slug}.svg")
    assert 'data-testid="contact-start-btn"' not in html, "prereq leaked into gate-off"
    # No gate pending → the standard lead form renders (the suppression is gate-only).
    assert 'data-testid="dealer-lead-form"' in html, "lead form missing on an ungated owner page"
    reveals = _reveals()
    assert len(reveals) == 1, f"stock reveal did not fire (got {len(reveals)})"
    assert reveals[0]["target_id"] == boat.id
    assert svg.status_code == 200
