"""Structural fidelity tests for the SRP filter panel.

These tests don't pixel-diff against real boattrader.com — that's the
job of ``deploy/sim_envs/mantis_boattrader/scripts/perceptual_diff.py``
(developer-local). Instead they assert the DOM has the structural
anchors that the v=82..v=84 fidelity passes locked in: the toggle
switch, the search-as-you-type filter list, the explicit Zip<br>Code
wrap, default-closed Boat Type / Make, the cache-buster pin, etc.

The idea is to catch silent regressions when someone refactors the
template or CSS — e.g. accidentally putting back the old ``<select>``
or removing the ``.switch`` wrapper.

Per ``FIDELITY_BUILD_FROM_SCRATCH_PROMPT.md`` Phase 5, this is the CI
gate that runs on every PR. The expensive perceptual harness stays
opt-in for local fidelity work.
"""

from __future__ import annotations

import re

import pytest


pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("jinja2")
pytest.importorskip("multipart")


@pytest.fixture
def client():
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from app.main import create_app  # noqa: PLC0415

    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def srp_html(client) -> str:
    r = client.get("/boats/")
    assert r.status_code == 200, r.text[:400]
    return r.text


@pytest.fixture
def base_css(client) -> str:
    r = client.get("/static/app.css")
    assert r.status_code == 200
    return r.text


# ── Save Search + outer card chrome (v=82) ────────────────────────────


def test_srp_renders(srp_html):
    assert 'class="srp"' in srp_html
    assert 'data-testid="save-search"' in srp_html


def test_filter_card_uses_canonical_shadow(base_css):
    """Outer filter card chrome — 1px outline + 6px radius + project
    shadow var. Set in v=82 to pair with the loan-calc card."""
    assert ".filters-form {" in base_css
    # Pull the .filters-form rule block.
    block = _rule_block(base_css, ".filters-form {")
    assert "background: #fff" in block.replace(" ", "").replace("\n", "") or \
           "background:#fff" in block.replace(" ", "")
    assert "border: 1px solid #e0e0e0" in block
    assert "border-radius: 6px" in block
    assert "var(--bt-shadow)" in block


def test_section_divider_is_2px(base_css):
    """v=82: section dividers bumped from 1px to 2px to match real BT."""
    block = _rule_block(base_css, ".filter-group {")
    assert "border-bottom: 2px solid #ededed" in block


# ── Location segmented control (v=82) ─────────────────────────────────


def test_zip_code_label_wraps_on_two_lines(srp_html):
    """v=82: 'Zip Code' label carries an explicit <br> so it wraps to two
    lines like real BT's narrower Roboto rendering. Plain 'Zip Code' text
    in the active tab would mean the wrap regressed."""
    assert 'class="zip-tab active" data-tab="zip">Zip<br>Code<' in srp_html


def test_zip_row_has_fixed_widths(base_css):
    """v=82: Zip input pinned to 100px (was flex:1 stretch), miles 106px."""
    zip_input_block = _rule_block(base_css, ".zip-row .zip-input {")
    assert "width: 100px" in zip_input_block
    assert "flex: 0 0 100px" in zip_input_block


def test_use_my_location_underlined(base_css):
    block = _rule_block(base_css, ".zip-use-location {")
    assert "text-decoration: underline" in block
    assert "font-size: 14px" in block


def test_zip_input_focus_state_is_blue_border(base_css):
    """v=82: focus uses a blue border + 0.5px ring (was 2px outline)."""
    block = _rule_block(base_css, ".filter-select:focus, .filter-input:focus {")
    assert "border-color: var(--bt-blue)" in block
    assert "outline: none" in block


# ── 5-digit zip auto-submit (v=82) ────────────────────────────────────


def test_zip_auto_submit_js_present(client):
    """The auto-submit JS lives in base.html. If it gets ripped out,
    typing a 5-digit zip on the live site stops navigating."""
    r = client.get("/")
    assert r.status_code == 200
    # The auto-submit regex is the unique signature.
    assert "/^\\d{5}$/.test(zip.value)" in r.text
    assert "form.submit()" in r.text


# ── Price Drop toggle (v=83) ──────────────────────────────────────────


def test_price_drop_is_toggle_not_checkbox(srp_html):
    """v=83: replaced <label class="checkbox-row"> with a div containing
    the new .switch toggle. The old checkbox-row in the price section
    would be a regression."""
    # The new structure
    assert 'class="price-drop-row"' in srp_html
    assert 'class="price-drop-label"' in srp_html
    assert 'class="switch"' in srp_html
    assert 'class="switch-toggle"' in srp_html
    # The label text is "Price Drop" (not "Price Drop only" — v=83 fix)
    assert ">Price Drop<" in srp_html
    assert "Price Drop only" not in srp_html


def test_price_drop_info_icon_present(srp_html):
    """Inline-SVG info glyph next to the label."""
    # Must contain the info-icon span + svg
    assert 'class="info-icon"' in srp_html


def test_switch_uses_has_input_checked(base_css):
    """v=83: :has(input:checked) toggles bg → blue + slides thumb 24px."""
    assert ".switch:has(input:checked)" in base_css
    assert ".switch:has(input:checked) .switch-toggle" in base_css


def test_switch_geometry_50x26(base_css):
    """50×26 grey track matches real BT exactly."""
    block = _rule_block(base_css, ".switch {")
    assert "width: 50px" in block
    assert "height: 26px" in block
    assert "background: #cccccc" in block
    assert "border-radius: 24px" in block


def test_switch_thumb_22x22(base_css):
    block = _rule_block(base_css, ".switch-toggle {")
    assert "width: 22px" in block
    assert "height: 22px" in block
    assert "background: #ffffff" in block
    assert "transition: left 0.2s" in block


# ── Search-as-you-type dropdowns (v=84) ───────────────────────────────


def test_boat_type_uses_search_list_not_select(srp_html):
    """v=84: <select name="type"> was replaced with .filter-search input
    + ul.filter-options. A regressed <select name="type"> would mean
    the rework was reverted."""
    assert '<select class="filter-select" name="type">' not in srp_html
    assert 'data-search-target="type-list"' in srp_html
    assert 'id="type-list"' in srp_html
    assert 'placeholder="Search Boat Type"' in srp_html


def test_make_uses_search_list_not_select(srp_html):
    assert '<select class="filter-select" name="make">' not in srp_html
    assert 'data-search-target="make-list"' in srp_html
    assert 'id="make-list"' in srp_html
    assert 'placeholder="Search Make"' in srp_html


def test_fuel_uses_filter_options_not_select(srp_html):
    assert '<select class="filter-select" name="fuel">' not in srp_html
    assert 'data-filter-name="fuel"' in srp_html


def test_hull_uses_filter_options_not_select(srp_html):
    assert '<select class="filter-select" name="hull">' not in srp_html
    assert 'data-filter-name="hull"' in srp_html


def test_filter_options_scrollable_with_max_height(base_css):
    """ul.filter-options must cap at 270px with overflow-y: auto to
    match real BT's `<ul.opts>` spec."""
    block = _rule_block(base_css, ".filter-options {")
    assert "max-height: 270px" in block
    assert "overflow-y: auto" in block


def test_filter_opt_checkbox_styling(base_css):
    """18×18 custom-styled checkbox with checkmark::after that flips to
    blue when the underlying <input> is :checked."""
    block = _rule_block(base_css, ".filter-opt-checkbox {")
    assert "width: 18px" in block
    assert "height: 18px" in block
    after_block = _rule_block(base_css, ".filter-opt-checkbox::after {")
    assert "transform: rotate(-45deg)" in after_block
    checked_block = _rule_block(
        base_css,
        '.filter-options input[type="checkbox"]:checked + .filter-opt-checkbox {',
    )
    assert "background: var(--bt-blue)" in checked_block


def test_boat_type_and_make_default_closed(srp_html):
    """v=84: real BT renders these sections collapsed by default. The
    `open` attribute on <details> would mean the regression came back."""
    # Re-locate the filter-type / filter-make blocks and confirm no `open`.
    type_block = _details_block(srp_html, 'data-testid="filter-type"')
    make_block = _details_block(srp_html, 'data-testid="filter-make"')
    assert "data-testid=\"filter-type\" open" not in type_block
    assert "data-testid=\"filter-make\" open" not in make_block


# ── SRP search box rotator (v=86) ─────────────────────────────────────


def test_srp_search_uses_rotator_not_static_prefix(srp_html):
    """v=86: replaced the static `<span>Try</span>` + single-suggestion
    placeholder with `.ai-search-v2__rotator` cycling through 3 example
    queries — matches real BT's vertical translate animation."""
    assert 'class="ai-search-v2__rotator"' in srp_html
    assert 'class="ai-search-v2__rotator-inner"' in srp_html
    # Three suggestion lines
    assert srp_html.count('class="ai-search-v2__try-text"') == 3
    # All three example queries present
    for needle in ("fishing boats under $80k", "Sea Ray under 40 feet", "pontoon boats near me"):
        assert needle in srp_html


def test_srp_search_input_placeholder_is_blank(srp_html):
    """v=86: the visible 'Try …' text comes from the span overlay, so
    the native input placeholder must be empty (real BT uses a single
    space). A non-empty placeholder would double-render with the span."""
    import re
    m = re.search(r'<input class="ai-search-v2__input" type="search" name="q" placeholder="([^"]*)"', srp_html)
    assert m, "ai-search-v2 input not found"
    placeholder = m.group(1).strip()
    assert placeholder == "", (
        f"input placeholder should be blank (rotator handles the visible text); got {placeholder!r}"
    )


def test_rotator_animation_in_css(base_css):
    """v=86: `@keyframes ai-try-rotate` with 9s ease-in-out alternate
    animation on `.ai-search-v2__rotator-inner` — matches real BT."""
    assert "@keyframes ai-try-rotate" in base_css
    block = _rule_block(base_css, ".ai-search-v2__rotator-inner {")
    assert "animation: ai-try-rotate 9s ease-in-out infinite alternate" in block
    rotator_block = _rule_block(base_css, ".ai-search-v2__rotator {")
    assert "overflow: hidden" in rotator_block
    assert "height: 18px" in rotator_block


def test_sparkle_icon_is_16x16(srp_html, base_css):
    """v=88: sparkle icon resized from sandbox's old 20×20 inline SVG to
    match real BT's 16×16 ai.svg. Wrapper uses line-height:0 +
    inline-flex so the 18px parent font-size doesn't add vertical
    padding (was rendering at 20×26 instead of 16×16)."""
    # SVG width/height attrs must be 16
    assert 'width="16" height="16" viewBox="0 0 16 16"' in srp_html, (
        "sparkle SVG must be 16x16 (real BT uses ai.svg at this size); "
        "if you see width=\"20\", v=87's 20x20 inline SVG regressed back in"
    )
    # Wrapper rule pins to 16x16 with line-height:0
    block = _rule_block(base_css, ".ai-search-v2__icon {")
    assert "width: 16px" in block
    assert "height: 16px" in block
    assert "line-height: 0" in block


def test_search_form_focuses_input_on_any_click(client, base_css):
    """v=89: clicks anywhere in the .ai-search-v2__form (sparkle, "Try …"
    overlay, empty area) must focus the input. Without this fix, clicks
    on the .ai-search-v2__try span just selected its text. The fix is a
    `mousedown` handler in base.html that calls `inp.focus()` for any
    non-submit-button target, plus `cursor: text` on the span so the
    affordance is visible."""
    r = client.get("/")
    assert r.status_code == 200
    html = r.text
    # JS handler must exist
    assert ".ai-search-v2__form" in html
    assert "form.addEventListener('mousedown'" in html
    assert "inp.focus()" in html
    # The span gets `cursor: text` so users don't see the I-beam → text
    # cursor mismatch (CSS).
    try_block = _rule_block(base_css, ".ai-search-v2__try {")
    assert "cursor: text" in try_block
    assert "user-select: none" in try_block


def test_rotator_hides_on_focus(base_css):
    """v=87: real BT clears the 'Try …' prefix the moment you click
    into the input (before any typing). Sandbox uses `:focus-within`
    on the form to set `display: none` on `.ai-search-v2__try`."""
    # Find the rule that hides the prefix; must include :focus-within
    idx = base_css.find(":focus-within .ai-search-v2__try")
    assert idx >= 0, "missing :focus-within rule for the Try prefix"
    # Also must hide on typed value (covers the blur-with-text case)
    idx2 = base_css.find(":has(input:not(:placeholder-shown):not([value=\"\"])) .ai-search-v2__try")
    assert idx2 >= 0, "missing :has(typed-value) rule for the Try prefix"


# ── Cache-buster pin ──────────────────────────────────────────────────


def test_app_css_cache_buster_is_current(srp_html):
    """The base template pins ?v=NN on app.css so a CSS edit invalidates
    the browser cache. v=84 is the current pin. If a contributor adds CSS
    without bumping this, deployed browsers will serve the stale file."""
    m = re.search(r"/static/app\.css\?v=(\d+)", srp_html)
    assert m, "app.css cache-buster not found in template"
    version = int(m.group(1))
    assert version >= 84, (
        f"app.css cache-buster is at v={version}; bump it when shipping CSS edits"
    )


# ── v=94 typography diff fixes ────────────────────────────────────────


def test_section_heading_is_16_500_404040(base_css):
    """v=94: all section headings (Location / Condition / Length / …)
    on real BT use 16/500/#404040. Sandbox was 15/400 #333."""
    block = _rule_block(base_css, ".filter-group-label {")
    assert "font-weight: 500" in block
    assert "font-size: 16px" in block
    assert "color: #404040" in block


def test_use_my_location_uses_negative_top_margin(base_css):
    """v=94: real BT uses `margin: -16px 0 15px` on `.search-user-location`
    to tuck the link up against the zip-row above. Sandbox was
    `margin-top: 8px` (pushed below the row instead)."""
    block = _rule_block(base_css, ".zip-use-location {")
    assert "margin: -16px 0 15px" in block


def test_filter_options_has_grey_backdrop(base_css):
    """v=94: real BT's ul.opts uses `background: #f7f7f7; padding: 8px`
    instead of a 1px outer border. Sandbox was bordered with no fill."""
    block = _rule_block(base_css, ".filter-options {")
    assert "background: #f7f7f7" in block
    assert "padding: 8px" in block
    # Outer border should be removed (rules don't mention `border:` line)
    # but keep the radius
    assert "border-radius: 8px" in block


def test_filter_options_label_is_15px(base_css):
    """v=94: list-item label text 14px → 15px to match real BT."""
    block = _rule_block(base_css, ".filter-options label {")
    assert "font-size: 15px" in block


# ── v=95 color shade pass ─────────────────────────────────────────────


def test_filter_inputs_are_404040(base_css):
    """v=95: `.filter-select` + `.filter-input` color → #404040 (was
    #333). Real BT inputs/selects render at this slightly darker shade."""
    block = _rule_block(base_css, ".filter-select, .filter-input {")
    assert "color: #404040" in block


def test_switcher_options_are_near_black(base_css):
    """v=95: `.zip-tab` + `.seg` color → #0a0a0a (was var(--bt-text)).
    Real BT renders switcher labels noticeably darker than the panel
    body — verified rgb(10,10,10) on `.switcher-option-label`."""
    zip_block = _rule_block(base_css, ".zip-tab {")
    assert "color: #0a0a0a" in zip_block
    seg_block = _rule_block(base_css, ".seg {")
    assert "color: #0a0a0a" in seg_block


# ── v=96 sort row + pagination typography ─────────────────────────────


def test_sort_row_is_12_400(base_css):
    """v=96: real BT renders Sort: at 12/400/#333 — sandbox had 14/500/#404040
    (oversized + bolded by the <strong> tag). Override sets the right weight."""
    block = _rule_block(base_css, ".sort-row {")
    assert "font-size: 12px" in block
    assert "font-weight: 400" in block
    strong = _rule_block(base_css, ".sort-label strong {")
    assert "font-weight: 400" in strong


def test_pagination_is_15_400(base_css):
    """v=96: real BT pagination wrapper is 15/400 with margin 15px 0
    (was 14/400 m=26px in sandbox)."""
    block = _rule_block(base_css, ".pagination {")
    assert "font-size: 15px" in block
    assert "margin: 15px 0" in block


def test_cookie_consent_banner_renders_on_first_visit(client):
    """v=103: cookie consent banner appears bottom-right when the
    `bt_cookie_consent` cookie isn't set. Has Customize / Reject /
    Accept buttons. Reject + Accept POST to /__site/consent with the
    matching `choice`; Customize dismisses in JS only."""
    r = client.get("/")
    assert r.status_code == 200
    assert 'id="cookie-consent"' in r.text
    assert 'aria-label="Cookie consent"' in r.text
    # All three buttons present
    assert '>Customize<' in r.text
    assert '>Reject<' in r.text
    assert '>Accept<' in r.text
    # Customize is a button with data-cookie-action; Reject/Accept are
    # forms posting to /__site/consent.
    assert 'data-cookie-action="customize"' in r.text
    assert r.text.count('action="/__site/consent"') >= 2
    assert 'name="choice" value="decline"' in r.text
    assert 'name="choice" value="accept"' in r.text


def test_cookie_consent_hidden_once_accepted(client):
    """After Accept (sets bt_cookie_consent cookie), the banner should
    disappear on subsequent loads."""
    # First load: banner present
    r1 = client.get("/")
    assert 'id="cookie-consent"' in r1.text
    # POST accept
    r2 = client.post("/__site/consent", data={"choice": "accept", "next_url": "/"}, follow_redirects=False)
    assert r2.status_code == 303
    # Now the test client has the cookie; re-fetch home — banner gone
    r3 = client.get("/")
    assert 'id="cookie-consent"' not in r3.text


def test_cookie_consent_card_styling(base_css):
    """v=103 CSS: banner is `position: fixed; right: 24px; bottom: 24px`
    bottom-right floating card with rounded corners + drop shadow.
    3 pill buttons in a row with min-width 108px."""
    block = _rule_block(base_css, ".cookie-consent {")
    assert "position: fixed" in block
    assert "right: 24px" in block
    assert "bottom: 24px" in block
    assert "border-radius: 12px" in block
    btn_block = _rule_block(base_css, ".cookie-consent-btn {")
    assert "border-radius: 50px" in btn_block
    assert "min-width: 108px" in btn_block


def test_ribbon_is_sticky_on_scroll(base_css):
    """v=102: blue pre-qualify ribbon uses `position: sticky; top: 0`
    so it stays in flow at scroll=0 (right under the nav) and sticks
    flush to the viewport top once the nav scrolls past. User
    wanted top:0 on scroll, not v=100's 44px-gap pattern.
    Position:sticky stays in flow so .bt-main needs no compensation
    margin."""
    block = _rule_block(base_css, ".ribbon-prequal {")
    assert "position: sticky" in block
    assert "top: 0" in block
    # .bt-main should NOT have margin-top compensation (ribbon in flow).
    # Check the literal property line — `margin: 0 auto;` — not the
    # historical mention in the comment.
    main_block = _rule_block(base_css, ".bt-main {")
    assert "  margin: 0 auto;" in main_block


def test_pagination_link_color_is_a5a5a5(base_css):
    """v=96: real BT page-link color is rgb(165,165,165) = #A5A5A5 for
    both active and inactive — they differ only by font-weight (700)
    and a 2px bottom border on active."""
    block = _rule_block(base_css, ".pagination a {")
    assert "color: #A5A5A5" in block
    active_block = _rule_block(base_css, ".pagination a.active {")
    assert "color: #A5A5A5" in active_block


# ── BDP fidelity anchors (v=91) ───────────────────────────────────────


@pytest.fixture
def bdp_html(client) -> str:
    # Pick any boat detail page — TestClient hits the same handler.
    r = client.get("/")
    # Find a BDP slug from a listing card on the home page.
    import re
    m = re.search(r'href="(/boat/[^"]+)"', r.text)
    assert m, "no BDP link found on home"
    slug_url = m.group(1)
    r2 = client.get(slug_url)
    assert r2.status_code == 200, r2.text[:300]
    return r2.text


def test_bdp_h1_has_length_suffix_span(bdp_html):
    """v=107: real BT BDP renders the length as a sibling SPAN right
    after H1 (not appended inside H1), styled 20/700 #333. v=104's
    removal was based on a stale measurement — re-probed real BT
    dealer + private listings both show the span. Sandbox restored
    it as a SPAN child of H1 (visually identical, simpler markup)."""
    assert 'class="bdp-length"' in bdp_html
    # Length suffix should match the pattern "| <N>'"
    import re
    assert re.search(r'<span class="bdp-length">\s*\|\s*\d+\'', bdp_html), \
        "bdp-length span missing the '| <N>'' pattern"


def test_bdp_has_no_meet_your_seller_heading(bdp_html):
    """v=107 (was v=104): real BT BDP does NOT render a "Meet Your
    Seller" H2 above the contact card on either dealer or private
    listings — re-probed against 2024 Catalina 355 (dealer) and 2015
    Pioneer 197 Sportfish (private) and both go straight to the
    contact heading. v=104 added it based on a stale measurement;
    v=107 removed it again."""
    assert 'meet-your-seller-h2' not in bdp_html
    assert 'Meet Your Seller' not in bdp_html


def test_owner_highlights_is_h2(bdp_html):
    """v=91: real BT renders 'Owner Highlights' as H2 20/700 alongside
    'Boat Details' and 'What Owners Say'. Sandbox had it as H3 — flipped."""
    # The heading uses the class `owners-card-tags-heading` regardless of tag,
    # so check both that the class is on an H2 and not on an H3.
    assert '<h2 class="owners-card-tags-heading">Owner Highlights</h2>' in bdp_html
    assert '<h3 class="owners-card-tags-heading">' not in bdp_html


def test_bdp_grid_capped_at_real_bt_width(base_css):
    """v=91: real BT renders BDP content at ~1319px (left col 890 + gap
    79 + right col 334). Cap .bdp-grid at max-width: 1336px so sandbox
    matches and the thumbnail strip auto-scales to 175×116."""
    block = _rule_block(base_css, ".bdp-grid {")
    assert "max-width: 1336px" in block


# ── SCOPE.md + prompt docs ────────────────────────────────────────────


def test_scope_md_exists():
    """Phase 0 doc per FIDELITY_BUILD_FROM_SCRATCH_PROMPT.md."""
    from pathlib import Path
    p = Path(__file__).resolve().parents[3] / "deploy" / "sim_envs" / "mantis_boattrader" / "SCOPE.md"
    assert p.exists(), f"missing {p}"
    text = p.read_text()
    # Should list at least the SRP route + filter interactions.
    assert "/boats/" in text
    assert "Out-of-scope" in text
    assert "Done bar" in text


def test_fidelity_agent_prompt_exists():
    """Gap-fix playbook present (build-from-scratch lives in a separate
    PR branch until it lands on main; once merged, add an assert for
    FIDELITY_BUILD_FROM_SCRATCH_PROMPT.md too)."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[3] / "deploy" / "sim_envs" / "mantis_boattrader"
    assert (root / "FIDELITY_AGENT_PROMPT.md").exists()


# ── helpers ───────────────────────────────────────────────────────────


def _rule_block(css: str, selector_line: str) -> str:
    """Pull the body of a single CSS rule. selector_line should include
    the trailing '{', e.g. '.filters-form {'. Returns everything from
    that selector through the matching '}'.

    Anchors to start-of-line so '.filter-group {' doesn't match the
    sibling-combinator rule '.search-alerts-button + .filter-group {'.
    """
    anchor = "\n" + selector_line
    idx = css.find(anchor)
    if idx < 0 and css.startswith(selector_line):
        idx = 0
    else:
        idx = idx + 1  # skip the leading \n
    assert idx >= 0, f"selector {selector_line!r} not found in CSS"
    end = css.find("}", idx)
    assert end > idx
    return css[idx:end + 1]


# ── v=105..v=107 BDP exact-mirror tests ────────────────────────────
# Each test below pins one of the structural changes from the
# 2026-05-24 BDP re-probe of real boattrader (dealer 2024 Catalina
# 355 + private 2015 Pioneer 197 Sportfish at 1440x900).

def test_bdp_has_next_previous_sticky_bar(bdp_html):
    """v=105: `.next-previous` always-sticky bar replaces the old
    320px-scroll-triggered `.bdp-sticky-bar`. Real BT renders it at
    position:sticky top:0, 54px tall, bg #f7f7f7, z:110."""
    assert 'class="next-previous"' in bdp_html
    assert 'class="next-previous-button next-previous-back"' in bdp_html


def test_bdp_next_previous_has_breadcrumb(bdp_html):
    """v=108: sticky bar contains the breadcrumb nav only. (v=105/106
    initially included a listing-name/price/location row beneath the
    breadcrumb, but re-comparison with the real BT screenshot showed
    that data is NOT visible in the sticky-bar row — only the
    breadcrumb sits in the middle.)"""
    assert 'class="breadcrumb next-previous-breadcrumb"' in bdp_html
    # The listing-info row from v=105 should NOT be present anymore
    assert 'next-previous-info"' not in bdp_html
    assert 'next-previous-listing-name' not in bdp_html


def test_bdp_next_previous_actions_only_prev_next(bdp_html):
    """v=108: real BT sticky bar's right action group has ONLY
    "Previous Boat" + "Next Boat" links. No Save button, no
    "Offered By" text."""
    assert 'next-previous-prev' in bdp_html or 'next-previous-next' in bdp_html
    # Removed in v=108: Save button and Offered-By tag
    assert 'next-previous-save' not in bdp_html
    assert 'next-previous-offered-by' not in bdp_html


def test_bdp_no_legacy_nav_row(base_css):
    """v=105: `.bdp-nav-row` (the duplicate breadcrumb + Back/Next
    block above the gallery) is hidden — its contents now live inside
    `.next-previous`."""
    assert ".bdp-nav-row { display: none" in base_css


def test_bdp_more_details_is_h4(bdp_html):
    """v=105: real BT renders "More Details" as H4 16/700 (NOT H3)."""
    assert '<h4 class="accordion-title">More Details</h4>' in bdp_html
    assert '<h3 class="accordion-title">More Details</h3>' not in bdp_html


def test_bdp_location_accordion_is_h4(bdp_html):
    """v=105: real BT renders "Location" as H4 16/700 (NOT H3)."""
    assert '<h4 class="accordion-title">Location</h4>' in bdp_html
    assert '<h3 class="accordion-title">Location</h3>' not in bdp_html


def test_bdp_has_accommodations_subhead(bdp_html):
    """v=105: real BT renders an "Accommodations" H4 14/700 subhead
    under the Measurements accordion alongside Dimensions/Weights/Tanks."""
    assert '<h4 class="accordion-subhead">Accommodations</h4>' in bdp_html


def test_bdp_still_have_a_question_is_h3(bdp_html):
    """v=105: real BT renders "Still have a question?" as H3 24/700
    #303030 (NOT H2 as sandbox previously had)."""
    assert '<h3 class="question-card-h2">Still have a question?</h3>' in bdp_html
    assert '<h2 class="question-card-h2">Still have a question?</h2>' not in bdp_html


def test_bdp_more_from_dealer_is_h4(bdp_html):
    """v=105: real BT renders "More From This Dealer" as H4 20/700
    with the carousel-special color #414d4a (NOT H2)."""
    # The page only includes this section for dealer listings, but
    # the fixture used by `bdp_html` may pick either type. Skip if
    # the section isn't rendered (private listing path)."""
    if "more-from-dealer-h2" not in bdp_html:
        return  # private listing — section absent by design
    assert '<h4 class="more-from-dealer-h2">More From This Dealer</h4>' in bdp_html
    assert '<h2 class="more-from-dealer-h2">' not in bdp_html


def test_bdp_no_listed_by_section(bdp_html):
    """v=105: the entire "Listed By" section deleted — real BT does
    not render a separate Listed By block on either listing type."""
    assert 'bdp-listed-by' not in bdp_html
    assert 'bdp-listed-by-h2' not in bdp_html


def test_bdp_has_sparkle_on_what_owners_say(bdp_html):
    """v=112: REVERTED v=105's removal. Real BT renders "What Owners
    Say" WITH the ✦ sparkle prefix (per user screenshot 2026-05-24).
    v=105 wrongly removed it; v=112 restored it."""
    import re
    m = re.search(
        r'<h2 class="owners-card-heading">.*?</h2>',
        bdp_html,
        re.DOTALL,
    )
    assert m, "owners-card-heading H2 missing"
    assert 'sparkle-icon' in m.group(0), "sparkle-icon span missing from heading"
    assert '✦' in m.group(0), "✦ glyph missing from heading"


def test_bdp_no_key_features_heading(bdp_html):
    """v=105: "Key Features" H3 removed — real BT doesn't render this
    heading on probed listings. The feature list <ul> kept as part
    of the Description body."""
    assert 'bdp-feature-h3' not in bdp_html
    assert '>Key Features<' not in bdp_html


def test_bdp_dealer_card_heading_is_h3(bdp_html):
    """v=105/107: contact card heading is H3 18/500 (was H2 20/700).
    Dealer cards use the salesperson name, private cards use
    "Contact Private Seller"."""
    import re
    h3_match = re.search(
        r'<h3 class="dealer-card-heading">(Contact [^<]+)</h3>',
        bdp_html,
    )
    assert h3_match, "dealer-card-heading H3 not found"
    heading = h3_match.group(1)
    # Either "Contact Private Seller" or "Contact <FirstName LastName>"
    assert (
        heading == "Contact Private Seller"
        or re.match(r"^Contact [A-Z][a-z]+ [A-Z][a-z]+$", heading)
    ), f"unexpected heading: {heading!r}"
    # H2 dealer-card-heading should not exist anymore
    assert '<h2 class="dealer-card-heading">' not in bdp_html


def test_bdp_dealer_card_has_inline_sublines(bdp_html):
    """v=107: dealer card renders dealer name + city as a subline
    paragraph and dealer phone as a subline link directly under the
    salesperson H3 (matches real BT pattern). Skip on private listing
    fixture (those listings don't have a dealer)."""
    if "bdp-private-seller-card" in bdp_html:
        return  # private listing path doesn't have dealer sublines
    assert 'class="dealer-card-subline dealer-card-name"' in bdp_html
    assert 'class="dealer-card-subline dealer-card-phone"' in bdp_html


def test_bdp_engagement_row_simplified_to_views_saves(bdp_html):
    """v=117: engagement row simplified to plain "Views | Saves"
    matching real BT's `.listing-engagement-indicators`. Removed
    icons, removed "Listed days ago" row, added 1×12px #dee2e3
    vertical divider span."""
    assert 'listing-engagement-indicators' in bdp_html
    assert 'listing-engagement-indicators__item' in bdp_html
    assert 'listing-engagement-indicators__divider' in bdp_html
    # Pull engagement row from <div class="bdp-engagement-row…"> to
    # the matching </div> by counting open/close tags.
    import re
    start_match = re.search(r'<div class="bdp-engagement-row[^"]*"', bdp_html)
    assert start_match, "engagement row container not found"
    start = start_match.start()
    # Naive depth-counter from `start`:
    depth = 0
    i = start
    while i < len(bdp_html):
        if bdp_html[i:i+4] == '<div':
            depth += 1
            i += 4
        elif bdp_html[i:i+6] == '</div>':
            depth -= 1
            i += 6
            if depth == 0:
                break
        else:
            i += 1
    block = bdp_html[start:i]
    assert '<svg' not in block, "engagement row should have no icons"
    assert 'Listed ' not in block and 'New to Market' not in block, \
        "Listed/New-to-Market row should be removed"
    assert 'Views' in block
    assert 'Save' in block


def test_bdp_dealership_card_not_rendered(bdp_html):
    """v=107: the separate `.bdp-dealership-card` (right-rail logo +
    active/sold stats block) is wrapped in `{% if false %}` and not
    rendered — real BT does not have a separate dealership card in
    the right rail."""
    assert 'class="bdp-dealership-card"' not in bdp_html
    assert 'dealership-stat-num' not in bdp_html


def _details_block(html: str, anchor: str) -> str:
    """Pull the <details ...anchor...> block up to (and including) its
    </details>. Anchor is a substring like 'data-testid="filter-type"'."""
    idx = html.find(anchor)
    assert idx >= 0, f"anchor {anchor!r} not found in HTML"
    # Find the surrounding <details ...> opening
    start = html.rfind("<details", 0, idx)
    assert start >= 0
    end = html.find("</details>", idx)
    assert end > start
    return html[start:end + len("</details>")]
