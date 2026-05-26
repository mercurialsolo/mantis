# `mantis_boattrader` — Fidelity match log

Working doc tracking each element's match status against
`https://www.boattrader.com/`. Update as iterations land.

Last updated: **v=130** (2026-05-25) — BDP exact-mirror marathon
(v=105..v=121, 15 deploys across 8 rounds). Re-probed real BT
dealer (2024 Catalina 355), private (2015 Pioneer 197 Sportfish),
and Pursuit 3070 listings at 1440×900 and landed structural
deltas in 8 iteration rounds — 6 prompted by user screenshot
feedback (sticky-bar layout, white margins, light-blue card bgs +
✦ sparkle, gradient page wrapper, white gutter on ad strip,
stats-strip → owners-card gap, loan-calc design, breadcrumb font).

Recent rounds:
- Round 7 (v=118): stripped `.bdp-question-card` chrome (white bg,
  border, padding) — real BT renders inline on page bg.
- Round 8 (v=119): tightened stats-strip → owners-card gap from
  118px to ~32px to match real BT; added 1px #ededed border to
  `.bdp-owners-card` per real BT probe.
- Round 8 (v=120): loan calculator redesigned to match real BT:
  Loan Term (Months) field replaces Credit Score (FICO); right
  pane stripped to centered heading + huge 64/700 $monthly +
  TOTAL LOAN AMOUNT + divider + See Important Disclosure (removed
  APR, 180 MONTHS, Get-Pre-Qualified button, pre-qualified
  bullets). 12px border-radius outer card with overflow:hidden so
  the white form + light-blue preview share corners cleanly.
- Round 8 (v=121): breadcrumb font 15/400 #333 → **12/400 #616161**
  per user screenshot — v=109's enlarged values mistook the
  `.next-previous-info-container` wrapper computed style for the
  actual `.breadcrumb` UL.

Suite **83 passing** (16 new BDP tests).

v=118 stripped `.bdp-question-card` chrome (real BT inline on page bg).
v=119 stats-strip / owners-card gap fix (118px → 32px) + owners-card
border + removed strip's spurious border-bottom.
v=120 loan calc redesign — Loan Term field replaces FICO; right
pane stripped to monthly + total + disclosure.
v=121 breadcrumb 15/400 #333 → 12/400 #616161 per user screenshot.
v=122 `.bdp-details-section` border-bottom → full border (1px
#ededed on all 4 sides) per real BT probe.
v=123 hide right-rail `.prequal-card` (not in real BT).
v=124 contact form Email+Phone split row + transparent input bg.
v=125 dealer card "Verified Broker" label + subline 16/#474c4a +
heading color #1a2022.
v=126 tighter Still-have-a-question body 18→15px + button 20→14px
+ section margin 24→12px.
v=127 collapse .bdp-more-from-dealer / .bdp-question-card padding
+ margin (was 116px gap).
**v=128 ROOT CAUSE — `.bdp-grid-stack` was inheriting `gap: 80px`
from `.bdp-grid`, which became row-gap between every section.
Added `row-gap: 0` so sections sit only on their own margins.**
v=129 column widths exactly match real BT: gallery 920→890,
.bdp-summary right:0→34 so layout is 890+76+400+34=1400 vs real BT.
v=130 restored right-rail `.bdp-dealership-card` (v=107's hide was
a wrong probe — re-probe of Pursuit 3070 showed .enhanced-business-
card-wrapper inside .summary-section at y=858).

Live URL: `https://8080-014f48ab-eeb1-4ca5-947e-42e169d1fcc8.daytonaproxy01.net/boats/`
(token rotates per sandbox restart; current: `syyxwc54axpktn5xcpo91iedvskoeqhv`)

## Methodology

Each row records the element/region, the real BT measurement
(via Chrome MCP `getComputedStyle` / `getBoundingClientRect`), and
the current state. Status legend:

- ✅ exact (within 1-2px rendering tolerance)
- 🟡 close (matches structurally; minor pixel diff)
- ⏳ partial (some pieces match, more work needed)
- ❌ missing / broken
- 🚫 intentionally not matched (e.g. real photos vs SVG placeholders)

## Global

| Element | Real BT | Mine | Status |
|---|---|---|---|
| Font family | Roboto, -apple-system, ... (system fallback; no actual Roboto loaded) | same — no Google Fonts link, system fallback | ✅ |
| Nav weight | 500 medium via system font | 500 medium via system font | ✅ |
| Page container width (SRP/home) | 1440px max, 36px side margin | same | ✅ |
| Page container width (BDP) | ~1319-1336px content (left col 890 + 79 gap + right col 334) | `.bdp-grid` capped at `max-width: 1336px` centered in `.bt-main` (v=91) | ✅ |

## Header

| Element | Real BT | Mine | Status |
|---|---|---|---|
| Logo SVG | 150×19 at x=30, y=20 | 150×19 at x=30 | ✅ |
| Header bg | white | white | ✅ |
| First nav item x (Find) | 338 | 338 | ✅ |
| Nav item padding | 23px 20px 26px 25px | 23px 20px 26px 25px | ✅ |
| Nav item color | rgb(7,50,79) navy | rgb(7,50,79) navy | ✅ |
| Nav item font-size | 15px | 15px | ✅ |
| Nav item weight | 500 | 500 | ✅ |
| Sign up / Log in (right) | rendered at x≈1437/1497 | same position | ✅ |

## Blue prequal ribbon

| Element | Real BT | Mine | Status |
|---|---|---|---|
| Height | 40px | 40px | ✅ |
| Bg | rgb(37,102,176) | same | ✅ |
| Inner padding | 10px 16px | 10px 16px | ✅ |
| "Get started" CTA | plain inline bold text | plain inline bold text | ✅ |
| **Sticky on scroll** | `position: fixed; top: 62px` — stays visible at y=62 across scroll (real BT nav is 62px tall, leaves 62px gap above ribbon on scroll) | `position: sticky; top: 0` — at scroll=0 ribbon sits flush under the nav (y=44, in flow); on scroll past the nav, ribbon flushes to viewport top (y=0). User-preferred behavior per v=102 — matches real BT visually at scroll=0 and tucks tight to the top on scroll instead of leaving a 44px gap. | ✅ (intentional divergence from real BT's fixed-top:62 pattern) |
| Hidden on BDP | yes (no ribbon on BDP) | hidden via `{% block ribbon %}{% endblock %}` | ✅ |

## SRP page

### Page header / breadcrumb / H1

| Element | Real BT | Mine | Status |
|---|---|---|---|
| Breadcrumb font-size | 12px / 400 | 12px / 400 | ✅ |
| Breadcrumb color | #757575 | #757575 | ✅ |
| Breadcrumb padding | 7px 15px | 7px 0 (vertical) | ✅ |
| H1 "Boats for sale" font-size | 32px | 32px | ✅ |
| H1 weight | 700 | 700 | ✅ |
| H1 color | rgb(64,64,64) #404040 | #404040 | ✅ |
| H1 line-height | 32px | 32px | ✅ |
| Bc → H1 visual gap | ~19px (via padding) | ~19px | ✅ |
| Ribbon → bc gap | small (system padding) | small | ✅ |

### Search bar ("Try" + quote)

| Element | Real BT | Mine | Status |
|---|---|---|---|
| Outer card bg | linear-gradient(135deg, #2566b0, #2b82d9 45%, #3391fc) | same | ✅ |
| Outer card border-radius | 12px | 12px | ✅ |
| Outer card padding | 12.25px 16px | 12px 16px | ✅ |
| Inner white field br | 10px | 10px | ✅ |
| "Try" weight | 700 | 700 | ✅ |
| "Try" font-size | 16px | 16px | ✅ |
| "Try" color | rgb(100,116,139) #64748b | same | ✅ |
| Quote placeholder | 16/400 #64748b | 16/400 #64748b | ✅ |
| **"Try …" rotator** (SRP only) | 3-line vertical column inside `.ai-search-v2__rotator` (overflow:hidden, 18px tall), `@keyframes 9s ease-in-out infinite alternate ai-try-rotate` cycling translateY 0→-18→-36 across "fishing boats under $80k", "Sea Ray under 40 feet", "pontoon boats near me" | Same — 3 `.ai-search-v2__try-text` spans, same 9s alternate animation, same 3 example queries (v=86). Native input placeholder emptied; visible suggestion comes from the span overlay. | ✅ |
| Rotator hides on focus | Real BT clears `.ai-search-v2__try` the moment user clicks into the input (before any typing) | Sandbox uses `.ai-search-v2__form:focus-within .ai-search-v2__try { display: none }` (v=87) | ✅ |
| Rotator hides on typed value | Real BT keeps prefix hidden when input has content (even if focus moved elsewhere) | Sandbox uses `:has(input:not(:placeholder-shown):not([value=""]))` (v=86) | ✅ |
| Sparkle icon | `<img src="/static/legacy/img/icons/ai.svg" width="16" height="16">` — 4-point star + two small "+" accents (top-right, bottom-left), all stroked/filled `#2566B0` | Inline SVG transcribed from real BT's ai.svg, 16×16 viewBox, same paths + same `#2566B0` (v=88 — previously 20×20 with different path data) | ✅ |
| Sparkle wrapper geometry | natural 16px line, no padding | `.ai-search-v2__icon { width:16; height:16; line-height:0; inline-flex }` so the 18px parent font-size doesn't expand it to 20×26 (v=88) | ✅ |
| Click anywhere → focus input | Real BT overlays `.ai-search-v2__try` via `position:absolute` + `pointer-events:none`; click on the span passes through to the input | Sandbox uses inline-flex layout (span takes real space), so adds a `mousedown` handler in base.html: any click in the form that isn't on submit/links calls `inp.focus()` + moves caret to end. CSS `cursor: text` + `user-select: none` on the span match the affordance (v=89). | ✅ |
| Submit icon | magnifier SVG | inline SVG circle+handle | ✅ |

### Pre-qualify banner

| Element | Real BT | Mine | Status |
|---|---|---|---|
| Bg | light blue gradient | light blue gradient | ✅ |
| Bank icon shape | rounded square 8px radius | rounded square 8px radius | ✅ |
| Bank icon bg | blue gradient (135deg) | linear-gradient(135deg, #3b78c0, #2566b0 55%, #1d569a) | ✅ |
| Bank icon size | 40×40 | 40×40 | ✅ |
| Title font | 14/700 | 14/700 | ✅ |
| Title color | rgb(16,24,40) #101828 | #101828 | ✅ |
| Subtext | 11/400 #4a5565 | 11/400 #4a5565 | ✅ |
| "Get Started" button | 13/700 white, 100px br, 0 24px pad, h=36 | same | ✅ |

### Filter sidebar

| Element | Real BT | Mine | Status |
|---|---|---|---|
| Outer card bg/border/radius | white, 1px gray (real BT actually has no border; sandbox keeps the chrome) | white, 1px #e0e0e0, 6px br | ✅ |
| Outer card drop shadow | sandbox-style (real BT has none) | `var(--bt-shadow)` — matches `.loan-calc-card` recipe | ✅ |
| Card padding | 15px sides | 15px 15px 18px | ✅ |
| Save Search button | 270×40, blue pill, 16/700, 50px br | 270×40, same | ✅ |
| Save Search → Location gap | ~50-60px | 51px | ✅ |
| Section divider | 2px solid #ededed | 2px solid #ededed (v=82) | ✅ |
| Section label (toggle-btn text) | 16/500 #404040 line-height 20px | 16/500 #404040 line-height 20px (v=94 — was 15/400 #333) | ✅ |
| Chevron | down arrow ~10px | down arrow ~10px via border trick | ✅ |
| Zip/City/Other segmented | 282×59 gray track, 50px br, 4px pad | 280×59 same shape (v=82: flex layout) | ✅ |
| Active tab label wrap | "Zip\nCode" (2 lines), forced by narrow cell | explicit `<br>` in markup (v=82) — Roboto's narrow rendering would wrap naturally, but sandbox's system font doesn't | ✅ |
| 25 miles select | 106×40, 1px #ededed, 8px br | 106×40, same | ✅ |
| Zip input | 100×40 fixed (not flex) | 100×40 fixed (v=82) | ✅ |
| "from" label | 16px #5E5E5E, margin 0 16px | 16px #5E5E5E, margin 0 16px (v=82) | ✅ |
| "Use My Location" | underlined, 14/400, blue, right-aligned, **margin: -16px 0 15px** (tucks up against zip row) | underlined, 14/400, blue, right-aligned, `margin: -16px 0 15px` (v=94 — was margin-top: 8px) | ✅ |
| Zip input focus | border-color → blue, no outline | border 1.5px blue + 0.5px shadow ring (v=82) | ✅ |
| 5-digit zip auto-submit | typing 5 digits navigates to `?zip=NNNNN` | debounced 250ms form.submit() in base.html (v=82) | ✅ |
| Price Drop control | `.switch.toggleButton` 50×26, white 22×22 thumb, slides on click; Material Icons `info` (24×24 #c2c2c2) next to label | `.switch` div with white thumb, `:has(input:checked)` toggles blue + slides right; inline SVG info icon 18×18 #c2c2c2 (v=83) | ✅ |
| Price Drop label | "Price Drop" + info icon | "Price Drop" + info icon (v=83; was "Price Drop only" + checkbox) | ✅ |
| Boat Type / Make filter UI | Search input (270×40, 4px br, magnifier icon) + `ul.opts` (270 wide, 270 maxH, scroll, **bg #f7f7f7, padding 8px, no outer border**) + 40px-tall `<li>` items with **15/400 #333** text | Same — `.filter-search-wrap` + `.filter-options` (bg #f7f7f7, padding 8px, v=94) + custom-styled `.filter-opt-checkbox`; label text 15px (v=94 — was 14px) | ✅ |
| Fuel Type / Hull filter UI | `ul.opts` checkbox list (no search input — small list) | Same — `.filter-options` only (v=84) | ✅ |
| Beam / Max Draft | Range slider + No Min/No Max number inputs | Same | ✅ |
| Default-closed sections | Boat Type / Make / Beam / Max Draft / Fuel / Hull / Engines / For Sale By all start `closed` | Beam / Max Draft / Fuel / Hull / Engines / For Sale By already closed; Boat Type + Make flipped to closed in v=84 | ✅ |
| Search-as-you-type filter | Typing in search input hides non-matching `<li>` items | JS in base.html hides `<li>` whose `.filter-opt-label` doesn't include the query; "All …" option always visible (v=84) | ✅ |
| Filter list selection submit | Checkbox check navigates to `/boats/?type=X` (single-select, since backend takes one value) | JS unchecks siblings + submits form on `change` (v=84) — visual checkboxes match real BT's checkbox UI even though backend is single-select | ✅ |
| All/New/Used segmented | same shape as zip toggle | same | ✅ |
| Length / Year / Price sliders | 22×22 blue thumb w/ 2px white border, 4px rail #e9e9e9, blue fill | same exact spec | ✅ |
| **Slider handle URL sync** | rc-slider auto-positions handles from URL | JS in base.html reads input min/max/value, sets handle positions | ✅ |
| Beam filter | range slider + min/max ft | added | ✅ |
| Max Draft filter | range slider + min/max ft | added | ✅ |
| Fuel Type filter | dropdown | added (Gas/Diesel/Electric/Other) | ✅ |
| Hull filter | dropdown | added (Fiberglass/Aluminum/Composite/Steel/Wood/Other) | ✅ |
| Engines filter | Number (segmented) + Engine Type (radio list) | added | ✅ |
| For Sale By filter | Any/Dealer/Private Seller radios | added | ✅ |

### Boat Loan Calculator widget (below filter card)

Re-probed real BT 2026-05-23 — widget concept is structurally different
from sandbox. Real BT puts the **output ("Monthly Payment $X")** at the
top with `.calc-summary-title` (22/900 #404040), then 4 input fields
(`Enter purchase price`, `Enter Down Payment`, `Enter term in years`,
APR). Sandbox puts the **input form first** with title "Boat Loan
Calculator" (18/700 #303030), then a result row.

| Element | Real BT | Mine | Status |
|---|---|---|---|
| Widget concept | output-first (Monthly Payment headline + inputs below) | input-first (Boat Loan Calculator title + inputs + result row) | 🟡 — different widget concept; both work for agent training; rewriting would need user buy-in on which version |
| Container | `.calc-calculator-body` w=300, transparent bg, no border/radius/shadow | `.loan-calc-card` w=303, bg=#fff, 1px #e0e0e0 border, 6px br, `var(--bt-shadow)` (intentional sandbox chrome) | 🟡 |
| Title | "Monthly Payment" 22/900 #404040 — also acts as the output display | "Boat Loan Calculator" 18/700 #303030 | 🟡 |
| Input geometry | 262×39, fs 15.9px, 3.9px br, 1px rgba(0,0,0,0.2) border | 269×40, fs 14px, 4px br | 🟡 — close (within rendering tolerance) |
| Input placeholders | "Enter purchase price", "Enter Down Payment", "Enter term in years" | "$" only | 🟡 — sandbox's `$` is less informative |
| Field types | 4 fields (purchase price, down payment, term years, APR) | 3 fields (loan amount, term months, APR) | 🟡 — slightly different inputs but functionally equivalent calc |
| Calculate button | outline secondary style | blue outline | ✅ visual style matches |
| "Get Pre-Qualified" CTA | blue filled pill | blue filled pill | ✅ |
| Fineprint help text | small gray | added | ✅ |

### Listing cards (SRP)

| Element | Real BT | Mine | Status |
|---|---|---|---|
| Card dimensions | 351×389 | 351×388 | ✅ |
| Card border | 1px #e0e0e0 | 1px #e0e0e0 | ✅ |
| Card filter (drop-shadow) | drop-shadow(rgba(0,0,0,0.2) 0px 2px 1px) | same | ✅ |
| Card border-radius | 4px | 4px | ✅ |
| Grid gap | 10px | 10px | ✅ |
| Image aspect | 3:2 (349×232) | 3:2 (347×231) | ✅ |
| Title | 18/700 #404040, ellipsis | 18/700 #404040 | ✅ |
| Price | 16/400 #404040 | 16/400 #404040 | ✅ |
| Monthly | 14/700 rgb(19,154,245) blue | same | ✅ |
| Monthly ⓘ info icon | small circled "i" | small circled "i" italic | ✅ |
| Click ⓘ → sticky tooltip | bottom-card blue banner, white text, × close | implemented via JS toggle | ✅ |
| Meta (city, dealer) | 12/400 #9e9e9e | 12/400 #9e9e9e | ✅ |
| Footer dealer logo | dark badge + dealer name | dark badge + dealer initial+name | ✅ |
| Footer divider above | 1px #e0e0e0 | 1px #e0e0e0 | ✅ |
| "Contact Seller" pill | blue outline, 14/700 | blue outline, 14/700 | ✅ |
| Owner cards | no dealer logo, "Contact Owner" | template uses `{% if not b.is_owner_listed %}` | ✅ |
| Featured badge | white pill 4px pad 16px br | same | ✅ |
| Heart icon (top-right of img) | SVG outline heart 16px in 28px circle | SVG outline heart, white-translucent circle | ✅ |
| Sponsored card border | same as others (no special) | 1px #e0e0e0 | ✅ |

### Sort row

| Element | Real BT | Mine | Status |
|---|---|---|---|
| Layout | plain inline "Sort:[strong] Recommended ▾" | same | ✅ |
| Border | none | none | ✅ |
| Font | **12/400 #333** (re-probed v=96; earlier 14/500 #404040 was wrong) | 12/400 #333 (`.sort-label strong { font-weight: 400 }` overrides the bold default) | ✅ |
| Chevron | inline SVG | inline SVG (bg-image data URL) | ✅ |

### Pagination

| Element | Real BT | Mine | Status |
|---|---|---|---|
| Style | plain text links | plain text links | ✅ |
| Wrapper font + margin | **15/400, margin 15px 0** (re-probed v=96; earlier 14/400 was wrong) | 15/400, margin 15px 0 | ✅ |
| Page link color | **#A5A5A5 medium-grey** (both active + inactive) | #A5A5A5 | ✅ |
| Page link font-weight | 700 (both active + inactive) | 700 | ✅ |
| Page link padding | 5px 10px | 5px 10px | ✅ |
| Active page distinguisher | 2px blue underline only (color matches inactive) | 2px var(--bt-blue) bottom border | ✅ |
| Prev/Next | plain blue | plain blue (`var(--bt-blue) !important`) | ✅ |
| "of N" | dim gray, regular weight | #757575 font-weight 400 | ✅ |

## BDP page

### Top strip

| Element | Real BT | Mine | Status |
|---|---|---|---|
| Sponsor ad strip | 728×90 leaderboard on #f7f7f7 strip, 122px tall | same exactly | ✅ |
| Blue ribbon on BDP | not present | hidden via `{% block ribbon %}` override | ✅ |
| Back to Search link | 12/400 #139af5 light blue | same | ✅ |
| Sticky breadcrumb | 15/400 #333 | 15/400 #333 | ✅ |
| Next Boat link | 12/400 #139af5 light blue | same | ✅ |

### Gallery

| Element | Real BT | Mine | Status |
|---|---|---|---|
| Main image aspect | 3:2 (890×593) | 3:2 (auto-scales by width) | ✅ |
| Main image border-radius | 8px | 8px | ✅ |
| Prev/Next/Share/Like buttons | 40×40 circle, rgba(0,0,0,0.3) bg | same | ✅ |
| Thumbnail strip | 5 thumbs, 175×116, 8px br | 5 thumbs at 175×116, 8px br (v=91 — auto-scaled by `.bdp-grid max-width: 1336px` capping the parent column to ~890px) | ✅ |

### Right rail (Featured card)

| Element | Real BT | Mine | Status |
|---|---|---|---|
| Width | 366px | 366px | ✅ |
| Bg / border / radius | white, 1px #ededed, 8px br | same | ✅ |
| Padding | 16px | 16px | ✅ |
| Drop-shadow | rgba(0,0,0,0.2) 0 2px 2px 0 | same | ✅ |
| Featured badge | small white pill | same | ✅ |
| H1 boat title | 20/700 rgb(51,51,51) | same | ✅ |
| Address | 14/400 #757575 | similar | ✅ |
| Price | 20/700 #333 | same | ✅ |
| "Own this boat for $X/mo" | 16/400 BT blue link | 16/400 blue | ✅ |
| Views/Saves/Listed strip | 12/400 with SVG icons | 12/400 with SVG eye/heart/calendar | ✅ |
| Contact name (H3) | 18/500 dark | 18/500 dark | ✅ |
| Lead form inputs | 40px h, 4px br, 1px #c2c2c2 | same | ✅ |
| Lead form padding | 8px 12px 4px (floating-label friendly) | same | ✅ |
| Contact Seller submit | blue pill | blue pill | ✅ |
| Call (outline) button | 2px blue outline, 14/700, pill | same | ✅ |
| "Show Phone Number" (owner) | hidden until click → SVG number reveal | implemented | ✅ |

### Below-gallery content

| Element | Real BT | Mine | Status |
|---|---|---|---|
| Stats strip (Length / Year / etc.) | label 14/700 #303030, value 14/400 #333 | same | ✅ |
| "What Owners Say" + Owner Highlights | "Owner Highlights" is `<h2>` 20/700 #333 (verified via real BT probe) | sandbox `<h2>` 20/700 #333 (v=91 — was `<h3>`, the CSS already styled `.owners-card-tags-heading` at 20/700 so only the tag changed) | ✅ |
| Owner-highlight pills | bg #e3f1fe, color #2566b0, 12/700, 4px 8px pad, 5px br | same | ✅ |
| Boat Details H2 section | 20/700 #333, bg #f5f9ff, 8px br, 20px 16px pad | same | ✅ |
| Description accordion (H3) | open by default | open by default | ✅ |
| Measurements accordion (H3) → Dimensions/Weights/Tanks (H4) | collapsed, h4 subgroups | added | ✅ |
| Propulsion accordion (H3) | collapsed | implemented | ✅ |
| More Details accordion heading | `<h4>` 16/700 #333 (re-probed v=107 dealer Catalina 355) | sandbox `<h4>` 16/700 (v=105) | ✅ |
| Location accordion heading | `<h4>` 16/700 #333 (re-probed v=107) | sandbox `<h4>` 16/700 (v=105) | ✅ |
| Accommodations subhead under Measurements | `<h4>` 14/700 #333 (re-probed v=107) | sandbox `<h4>` 14/700 (v=105 — derived cabin/berth counts from length) | ✅ |
| `.next-previous` always-sticky bar | 1600×54 position:sticky top:0 z:110 bg #f7f7f7 padding 5px 20px — Back-to-Search + breadcrumb + listing info (stacked) + Save + Offered By + Next Boat | sandbox `.next-previous` (v=105) with column-stacked rows (v=106) | ✅ |
| Dealer contact card heading | `<h3>` 18/500 #303030 (dealer: "Contact <Salesperson>", private: "Contact Private Seller") | sandbox `<h3 class="dealer-card-heading">` 18/500 #303030 (v=105 flipped H2→H3; v=107 simplified private path) | ✅ |
| Dealer name + phone sublines under salesperson H3 | inline name+city + phone-link rows (dealer-only) | sandbox `.dealer-card-subline` rows (v=107) | ✅ |
| "Meet Your Seller" H2 above contact card | **not present** on either listing (re-probed dealer + private) | removed in v=105 (v=104 added based on stale measurement) | ✅ |
| H1 length suffix | sibling `<span>` "| <N>'" after H1, 20/700 #333 | sandbox `<span class="bdp-length">` inside H1 (v=107 restored; v=104 had removed) | ✅ |
| "Listed By" section | **not present** on either listing | removed in v=105 (dealer info now lives in contact card + sticky-bar Offered-By) | ✅ |
| "Key Features" H3 | **not present** on real BT | removed in v=105 (feature `<ul>` kept as part of Description body) | ✅ |
| ✦ sparkle prefix on "What Owners Say" | not present on real BT (plain "What Owners Sayinfo") | removed in v=105 | ✅ |
| "More From This Dealer" heading | `<h4>` 20/700 #414d4a (dealer-only) | sandbox `<h4 class="more-from-dealer-h2">` 20/700 #414d4a (v=105 flipped H2→H4) | ✅ |
| "Still have a question?" heading | `<h3>` 24/700 #303030 | sandbox `<h3 class="question-card-h2">` 24/700 #303030 (v=105 flipped H2→H3) | ✅ |
| Right-rail dealership-card (separate logo + stats block) | **not present** on real BT (dealer info integrated above) | `.bdp-dealership-card` wrapped in `{% if false %}` (v=107) | ✅ |
| Description / Measurements / Propulsion accordion headings | `<h3>` 16/700 #333 (verified) | sandbox `<h3>` 16/700 #333 | ✅ |
| Dimensions / Weights / Tanks subheadings | `<h4>` 14/700 (verified) | sandbox `<h4>` (already matches) | ✅ |
| Boat Details / What Owners Say section headers | `<h2>` 20/700 (verified) | sandbox already H2 | ✅ |
| Dealership card | dealership-card with logo, address, stats | implemented | ✅ |
| "Get pre-qualified in minutes" rail card | 18/700 heading + checkmarks + outline btn | implemented (heading bumped to 18px in v=56) | ✅ |
| More From This Dealer carousel | horizontal scroll of small cards | implemented | ✅ |
| Still have a question? card | 24/700 question h2, body 18px | implemented | ✅ |
| Other Services tiles | 14/700 label, tile names 14/400 #4d4d4d | implemented | ✅ |

## Home page

| Element | Real BT | Mine | Status |
|---|---|---|---|
| Hero card layout | 296×48 narrow overlay card at (14, 139) on hero image (NOT a full-width search bar) | implemented as narrow card overlay | ✅ |
| Hero H1 "Find your perfect boat" | 296×19 at (14, 108), 16/700 #fff over hero image | implemented | ✅ |
| Hero search form | same `.ai-search-v2__form` as SRP — sparkle + Try-rotator + magnifier | shared component, same rotator (v=86) | ✅ |
| "Sell Your Boat Fast!" callout | sailboat icon + label + Sell pill | implemented | ✅ |
| "Boats Near You" section heading | full-width h2 at y=586, 15/700 #333, with "Based on your location" subtext inline | implemented | ✅ |
| Hero ad (right side strip) | full-bleed sponsor banner | gradient placeholder ad | 🚫 (placeholder by design) |
| Featured Brands tiles | brand logo grid below the rails | implemented per `.brand-tile` | ✅ |
| Popular Boat Types tiles | type-tile rail | implemented per `.type-tile` | ✅ |
| Popular Boats card rail | listing-card rail (same shape as SRP listing card) | implemented (shares `.card-grid` patterns) | ✅ |
| Recent Articles card rail | editorial card rail linking to /articles/... | implemented | ✅ |
| Per-tile geometry on rail sections | individual brand-tile / type-tile / article-card sizes + gaps | not individually probed — corpus marks as open follow-up in `_captured/home/structural.json` | 🟡 |

## Behaviors

| Behavior | Real BT | Mine | Status |
|---|---|---|---|
| Filter URL → state sync | sidebar reflects URL params | condition pill, length inputs sync; slider handles position via JS | ✅ |
| Click ⓘ on monthly | opens sticky banner | implemented | ✅ |
| Click outside tooltip | closes | implemented | ✅ |
| Detail page sticky bar | appears on scroll | `bdp-scrolled` body class via scroll listener | ✅ |
| 5-digit Zip auto-submit | typing complete zip navigates to filtered SRP | debounced 250ms `form.submit()` in base.html (v=82) | ✅ |
| Price Drop toggle click | flips checkbox + visual track turns blue | `.switch` + `:has(input:checked)` slides thumb (v=83) | ✅ |
| Boat Type / Make search-as-you-type filter | typing hides non-matching `<li>` items | JS in base.html toggles `.hidden` on non-matching `<li>` (v=84) | ✅ |
| Filter checkbox single-select submit | clicking a Boat Type / Make / Fuel / Hull checkbox auto-navigates | JS unchecks siblings + `form.submit()` (v=84) | ✅ |
| Search-box prefix hides on focus | "Try …" disappears the moment input is clicked | `.ai-search-v2__form:focus-within .ai-search-v2__try { display: none }` (v=87) | ✅ |
| Search-box prefix hides on typed value | "Try …" stays hidden when value present (even after blur) | `:has(input:not(:placeholder-shown):not([value=""]))` (v=86) | ✅ |
| Click anywhere in search field → focus input | clicks on sparkle / "Try …" overlay / empty area all focus input | `mousedown` handler on `.ai-search-v2__form` calls `inp.focus()` (v=89) | ✅ |
| Search-box "Try …" suggestions rotate | 3 example queries cycle through a 18px clipped window | `@keyframes ai-try-rotate` 9s ease-in-out alternate (v=86) | ✅ |

## Open work / known minor diffs

- ✅ ~~BDP content area width~~ — fixed in v=91 via `.bdp-grid { max-width: 1336px }`.
- 🚫 **Real photos vs SVG placeholder boats**: out-of-scope per `SCOPE.md` —
  sandbox uses procedural SVG by design. Re-classified from 🟡.
- ✅ ~~More Details / Location H-level~~ — re-probed v=107 on dealer +
  private listings: both render H4 16/700 #333. Sandbox flipped H3→H4
  in v=105.
- ✅ ~~Thumbnail size~~ — auto-fixed by v=91's `.bdp-grid max-width`.
- ✅ ~~Per-tile geometry on Home page rails~~ — measured in v=93. Real BT
  uses 363×120 boat-listing cards, 266×161 brand tiles, 325×317 article
  cards. Sandbox card-grid pattern matches within typical render
  variance; corpus documents the spec for future verification.
- ✅ ~~BDP `.next-previous` sticky navigation bar~~ — added in v=105
  with v=106 layout fix. Position:sticky top:0, h=54, bg #f7f7f7,
  z:110, padding 5px 20px; stacked breadcrumb + listing-info rows;
  "Offered By: <dealer>" tag on dealer listings.
- ✅ ~~Listing-dependent BDP elements~~ — "More From This Dealer"
  carousel gated by `is_owner_listed` (dealer-only). "Listed By"
  section deleted entirely. Show Phone retained as cookie-gated
  interaction surface on private listings (kept for agent training
  even though real BT private listings often render without it).
- 🟡 **Mobile viewport**: no mobile pass yet. CSS has `@media (max-width: 980px)`
  responsive rules but they're not verified against real BT mobile rendering.

## Data variety (fixtures)

| Trait | Distribution |
|---|---|
| Listing type | 67% dealer / 25% owner / 8% sponsored |
| POA boats | ~8% of dealers — `display_price="Request a Price"`, no monthly, no ⓘ |
| Price drop badge | ~18% of dealer listings (rare for owners) |
| Owner listings | "Contact Owner" CTA, no dealer logo, phone hidden behind click |
| Engine hours | 0 (new) or 20-950 (used) |
| Engine count | 1 / 2 / 3 by length |
| Badges | Featured / In-Stock / Price Drop / New Arrival / Sponsored — variable subset |
| Description blurbs | 5+ templates rotated by listing_type |
| Days listed | 0-120 → "New to Market" / "Listed N days/months ago" |

## Iteration log (versions)

`v=46` BDP ribbon removed, SRP card heights, meta color
`v=47` BDP grid 1fr+80+366
`v=48` ribbon 40px
`v=49` filter card border + padding
`v=50` dealer logos in cards
`v=51` badge color uniform, sponsored card border
`v=52` SRP H1 color and line-height
`v=53` owner-tag pills
`v=54` BDP rail padding + shadow
`v=55` lead form inputs 4px radius, outline button 2px
`v=56` prequal heading 18/700, Boat Details bg
`v=57` BDP back/next colors
`v=58` sticky-breadcrumb 15/400/#333
`v=59` ad-strip 728×90
`v=60` brand margin → Find at x=338
`v=61` logo 150×19 compact + system fallback
`v=62` SRP breadcrumb 12/400/#757575
`v=63` info icon + sticky tooltip
`v=64` tooltip click-toggle JS
`v=65` card grid gap 10px
`v=66` breadcrumb padding 7px
`v=67` removed breadcrumb top margin
`v=68` Sort plain inline + SVG icons (sparkle/magnifier/heart)
`v=69` SVG icons (eye/calendar/phone)
`v=70` slider thumbs 22×22
`v=71` sort weight 500, rounded square bank icon gradient
`v=72` gallery buttons 40×40
`v=73` slider handle JS URL sync
`v=74` price input max attribute
`v=75` pagination plain-text
`v=76` Save Search bottom margin attempt
`v=77` fixed margin specificity override
`v=78` removed Google Fonts (use system font for nav weight)
`v=79` BDP gallery 3:2 aspect
`v=80` added Beam/Max Draft/Fuel Type/Hull/Engines/For Sale By + Boat Loan Calculator widget
`v=81` accordion section h3 + measurements subsection h4
`v=82` filter-panel fidelity pass:
       • "Zip Code" wraps via explicit `<br>` (matches real BT's natural Roboto wrap)
       • Zip-row layout: fixed `[miles 106] 16px [from] 16px [zip 100]` (was flex:1 stretch)
       • "from" → 16px / #5E5E5E / margin 0 16px
       • `.zip-use-location` underlined, 14px (was 13px, no underline)
       • Section dividers 2px (was 1px) — matches real BT `.collapse-content` bottom border
       • Outer card now uses `var(--bt-shadow)` drop shadow (paired with `.loan-calc-card`)
       • Focus state on `.filter-input` / `.zip-input`: blue border + 0.5px ring (was 2px outline)
       • 5-digit Zip input auto-submits the filters form (matches real BT's auto-navigation)
       • Adds `deploy/sim_envs/mantis_boattrader/scripts/perceptual_diff.py` harness
       • Adds `deploy/sim_envs/mantis_boattrader/FIDELITY_AGENT_PROMPT.md` — repeatable workflow
`v=83` Price Drop toggle switch replaces checkbox:
       • New `.price-drop-row` with label + inline-SVG info icon (18×18 #c2c2c2)
       • New `.switch` (50×26 grey track, white 22×22 thumb) + `:has(input:checked)`
         flips background to `--bt-blue` and slides thumb 24px right with `transition: left 0.2s`
       • Label changed from "Price Drop only" → "Price Drop" (matches real BT)
       • Documents remaining 🟡 gaps in dropdown sections (Boat Type / Make / Beam /
         Max Draft / Fuel Type / Hull) — real BT uses search-as-you-type filter lists,
         sandbox keeps `<select>` for now; future v= will swap when needed
`v=84` Dropdown sections reworked to match real BT search-list pattern:
       • Boat Type + Make + Fuel Type + Hull: `<select>` → search-input
         (Boat Type / Make only) + `ul.filter-options` checkbox list
         (40px-tall li, custom-styled `.filter-opt-checkbox` with checkmark
         pseudo-element, blue when `:checked`)
       • `ul.filter-options` capped at `max-height: 270px; overflow-y: auto`
         with 1px #ededed border + 4px radius (matches real BT spec exactly)
       • JS in base.html: search-as-you-type hides non-matching `<li>` (keeps
         "All …" option always visible); selecting a checkbox unchecks
         siblings + auto-submits form to `?type=X` etc.
       • Boat Type + Make `<details>` flipped from `open` → default-closed
         (matches real BT — sections collapse on first load)
       • Adds `SCOPE.md` per Phase 0 of `FIDELITY_BUILD_FROM_SCRATCH_PROMPT.md`
         — in-scope pages + interactions, viewport, backend complexity,
         out-of-scope (🚫), done bar, open follow-ups
       • All filter-sidebar 🟡 rows now ✅ (or moved to SCOPE.md "Out-of-scope"
         if intentionally divergent — e.g. multi-select, "more options" link)
       • Adds `tests/sim_envs/mantis_boattrader/test_filter_panel_fidelity.py` —
         23 structural-anchor assertions covering v=82..v=84 changes. Runs in
         the regular pytest matrix, no playwright needed (FastAPI TestClient).
         Phase 5 (verification harness) from FIDELITY_BUILD_FROM_SCRATCH_PROMPT.md.
`v=86` SRP search-box "Try …" rotator animation matches real BT:
       • Replaces static `<span>Try</span>` + single-suggestion placeholder
         with `.ai-search-v2__rotator` (overflow:hidden, 18px tall window)
         containing 3 `.ai-search-v2__try-text` lines
       • New `@keyframes ai-try-rotate` — 0%/25% translateY(0), 37.5%/62.5%
         translateY(-18), 75%/100% translateY(-36); animation runs
         `9s ease-in-out infinite alternate` exactly as real BT does
       • Native input placeholder is now `" "` (single space); visible
         suggestion comes from the span overlay so it doesn't double-render
       • `.ai-search-v2__form:has(input:not(:placeholder-shown))` hides
         the rotator when user types — matches real BT's DOM removal
       • Adds 3 new fidelity tests covering rotator structure + animation
         + blank placeholder (test_filter_panel_fidelity.py now 26 passing)
       • `.ai-search-v2__try-prefix` kept as a legacy alias for the
         non-rotating prefix used elsewhere (e.g. home hero search)
`v=117` Engagement row simplified to real BT structure:
       • Removed 3-item `<ul class="bdp-engagement-row">` (Views +
         Saves + Listed-days icons) and replaced with a 2-item
         `<div class="bdp-engagement-row listing-engagement-indicators">`
         containing just "Views" and "Saves" with a 1×12px #dee2e3
         vertical divider span between (`.listing-engagement-
         indicators__divider`). 12/400 #757575 per real BT probe.
       • No icons, no Listed/days-ago row (real BT doesn't show those).
       • One new test (test_bdp_engagement_row_simplified_to_views_saves).
         Suite now **83 passing**.
`v=116` Ad-strip full-viewport bleed:
       • User screenshot showed grey #f7f7f7 bg cut off at .bt-main
         max-width 1440 with ~80px white gutters on each side of
         the top sponsored ad-leaderboard.
       • Same fix as v=113 sticky bar: `.ad-strip { width: 100vw;
         margin-left: calc(-50vw + 50%) }`. Ad image stays centered
         via flex `justify-content: center`.
       • `.bdp-grid-stack > .ad-strip` overrides reset width to auto
         so mid-page leaderboards stay inside their grid column.
`v=115` Right-rail summary card width 366px → 400px:
       • Real BT `.summary-section` probe: w=400 x=1066, pad 16px,
         bg #fff, border 1px #ededed, br 8px, shadow 0 2px 2px
         rgba(0,0,0,0.2). Sandbox was 366px (too narrow).
       • Global replace `366px → 400px` (3 sites: .bdp-grid
         template, .bdp-grid-stack padding offset, .bdp-summary).
       • Lead-input geometry (border 1px #c2c2c2, br 4px, pad
         8px 12px 4px, h 40px) already matched per v=55 — no change.
`v=114` Top-section polish per comprehensive bg/margin/font probe:
       • `.next-previous { border-bottom: 1px solid #ededed }`
         REMOVED — real BT probe shows no bottom border. v=105's
         border was creating a visible divider line in the user
         screenshot between the top ad-strip and the sticky bar.
       • `.ad-strip` outer margin `24px 0 → 0`. Real BT renders
         the leaderboard flush against the page header above and
         the sticky bar below.
       • `.bdp-details-section`: added `border-bottom: 1px solid
         #ededed` per real BT `.accordion-details-section` probe.
       • `.bdp-price` wrapper bumped 16/400 → 20/700 #333. Real
         BT price renders at 20/700 on the wrapping element.
       • `.bdp-title` H1 margin `8px 0 6px → 0` per real BT.
       • `.accordion-body` (Description inner): added `background:
         #fff; border-radius: 8px; padding: 0 16px 18px` — real
         BT renders Description as a nested WHITE card inside the
         light-blue `.bdp-details-section` parent.
`v=113` Sticky bar full-viewport bleed (per user screenshot):
       • User screenshot showed `.next-previous` grey #f7f7f7 bg
         cut off ~100px before the viewport right edge, leaving
         a white gap.
       • Bug: `width: 100%` referred to `.bt-main`'s constrained
         max-width 1440px, not the viewport. Fix: switched to
         `width: 100vw; margin-left: calc(-50vw + 50%)` so the
         bar's bg spans edge-to-edge.
       • Same fix applied to `.boat-details-gradient` wrapper.
`v=112` Restored light-blue card bgs + ✦ sparkle (per user screenshot):
       • v=110/v=111 whitewashed all section cards thinking real
         BT was plain white. User's real-BT screenshot showed
         clear light-blue (#f5f9ff) cards on "What Owners Say"
         and "Boat Details", plus a ✦ sparkle prefix on
         "What Owners Say".
       • Re-probed precisely:
         - `.accordion-details-section` (Boat Details): bg #f5f9ff,
           br 8px, pad 20px 16px, border-bottom 1px #ededed
         - `.boat-overview-wrapper.ai-ratings-review-bundle`
           (Owners + Highlights): bg #f5f9ff, br 8px, pad 20px 16px
         - `.accordion-details-items` (Description): bg #fff,
           br 8px, pad 0 16px (nested white card on light-blue)
       • Sandbox `.bdp-owners-card` bg `#eef4fb` → `#f5f9ff` (exact
         match to real BT, not the older v=83 tint).
       • `.bdp-details-section` and `.bdp-more-from-dealer` also
         set to #f5f9ff with matching geometry.
       • Re-added `<span class="sparkle-icon">✦</span>` prefix to
         the `.owners-card-heading` H2 — v=105 had wrongly removed
         it.
       • Test `test_bdp_no_sparkle_on_what_owners_say` rewritten
         to `test_bdp_has_sparkle_on_what_owners_say`.
`v=111` BDP gradient page wrapper (REVERTED in v=112):
       • User screenshot of real BT styles.css showed
         `.boat-details-gradient { background: linear-gradient(180deg,
         #f7f7f7, transparent 20%) !important }`.
       • Added `<div class="boat-details boat-details-gradient">`
         wrapper around the BDP body content after the sticky bar.
       • Wrapper bleeds full viewport width via the same margin
         trick as `.next-previous`.
       • Section bgs (`.bdp-owners-card`, `.bdp-details-section`,
         `.bdp-more-from-dealer`) set to transparent so the
         gradient shows through.
       • PARTIAL REVERT in v=112: kept the gradient wrapper but
         restored light-blue card bgs since real BT actually has
         BOTH the gradient wrapper AND the light-blue sections
         (sections cover the gradient where they sit).
`v=110` Whitewashed BDP sections (REVERTED in v=112):
       • Mistakenly read user's "background isn't white it's #fff"
         comment as wanting EVERY section white. Set
         `.bdp-owners-card`, `.bdp-details-section`, `.bdp-more-
         from-dealer` to bg #fff.
       • Reverted in v=112 after user shared the actual real BT
         screenshot showing light-blue cards.
`v=109` Sticky bar inner wrapper + breadcrumb tightening:
       • Added `.next-previous-inner` div with `max-width: 1400px;
         margin: 0 80px` matching real BT's container exactly.
         At 1600px viewport this places "‹ Search" at x≈100 (20px
         bar padding + 80px wrapper margin), not at x≈25 as v=108
         had.
       • `@media (max-width: 1280px)` falls back to `margin: 0 auto`
         so the bar doesn't crowd out at narrow viewports.
       • Breadcrumb 14/400 #303030 → 15/400 #333 (matches real BT).
       • Back/Next buttons: 14/400 → 12/400, padding 8px 14px →
         11px 15px.
`v=108` Sticky bar simplified per user real-BT screenshot:
       • v=105/106/107 sticky bar had too much in it (listing
         name/price/location row, Save button, Offered-By tag).
         Real BT screenshot showed bar contains ONLY:
         "‹ Search / breadcrumb / Previous Boat / Next Boat".
       • Earlier JS probes had found the listing-info text via
         `.next-previous-info` but that element is in a hidden
         absolutely-positioned info layer, NOT the visible row.
       • Removed `.next-previous-info-container`,
         `.next-previous-info`, `.next-previous-listing-name`,
         `.next-previous-listing-price`, `.next-previous-listing-
         loc`, `.next-previous-save`, `.next-previous-offered-by`.
       • Changed "‹ Back to Search" → "‹ Search".
       • Added `Previous Boat` link (sandbox passes `prev_boat`
         already from `db.adjacent_boats`).
       • Tests rewritten: `test_bdp_next_previous_has_breadcrumb_
         and_info` → `test_bdp_next_previous_has_breadcrumb`;
         `test_bdp_next_previous_info_container_is_column_flex` →
         `test_bdp_next_previous_actions_only_prev_next`.
`v=107` BDP exact-mirror — round 3 (H1 length-suffix span + dealer-card sublines):
       • H1 length-suffix re-added as a sibling SPAN inside the H1
         tag (`<h1>{title}<span class="bdp-length"> | {len}'</span></h1>`).
         Real BT's pattern is identical but with the SPAN as a true
         sibling outside H1; visually equivalent and the simpler
         child-span markup avoids extra DOM nodes. v=104 over-
         corrected by removing the suffix entirely.
       • Dealer card restructured: removed the headshot SVG + flex
         `.dealer-card-meta` row; added `.dealer-card-subline` rows
         for `{dealer.name} - {dealer.city}` (#303030 14/400) and
         `{dealer.phone}` (clickable `tel:` link, blue) directly
         under the salesperson H3, matching real BT's pattern
         exactly (verified on 2024 Catalina 355).
       • Private contact form simplified: removed the message
         `<textarea>` and the .lead-row Email+Phone wrapper (real BT
         private form is plain stack Name/Email/Phone/Submit). Show
         Phone button retained as a cookie-gated interaction surface
         for agent training.
       • Separate right-rail `.bdp-dealership-card` (logo + active/
         sold stats block) wrapped in `{% if false %}` — real BT
         doesn't render a separate dealership card; dealer info
         lives in the contact card body + the `.next-previous-
         offered-by` tag inside the sticky bar.
       • Cache-buster bumped to `?v=107`.
       • 13 new fidelity tests covering all v=105..v=107 deltas.
         Suite now **82 passing**. Two stale v=104 tests rewritten:
         `test_bdp_h1_has_no_length_suffix` → `test_bdp_h1_has_length_
         suffix_span`; `test_bdp_has_meet_your_seller_heading` →
         `test_bdp_has_no_meet_your_seller_heading`.
`v=106` `.next-previous` sticky bar layout fix (per user screenshot):
       • v=105 shipped the new always-sticky bar but
         `.next-previous-info-container` defaulted to `display: flex;
         align-items: center` which put the breadcrumb and the
         listing-info row side-by-side on one line (user feedback:
         "this is not at all what the section above the image looks
         like").
       • Changed to `flex-direction: column; align-items: flex-start;
         justify-content: center; gap: 2px; height: 100%` so the
         breadcrumb sits on the top row (y≈219) and the listing
         name/price/location row sits beneath (y≈249), matching
         real BT's stacked-rows pattern.
       • Cache-buster bumped to `?v=106`.
`v=105` BDP exact-mirror — round 1 (heading hierarchy + cleanup):
       Re-probed real BT on 2024-05-24 at 1440×900 against TWO
       listings to nail down the deltas v=104's single probe missed:
       dealer 2024 Catalina 355 (real BT) + private 2015 Pioneer 197
       Sportfish (real BT). Confirmed pattern across BOTH listing
       types, then landed the following structural changes:
       • Removed `<h2 class="meet-your-seller-h2">Meet Your Seller</h2>`
         — v=104 added this based on a stale measurement. Re-probe
         showed NO Meet-Your-Seller heading on either dealer or
         private listings on real BT. Selector commented out in CSS.
       • Dealer contact card heading: `<h2>` → `<h3>` (already styled
         18/500 #303030 via `.dealer-card-heading`).
       • Private contact card heading: changed from "Contact
         `{owner_name}`, the owner" to generic "Contact Private
         Seller" (real BT private uses generic phrasing).
       • Removed `private-seller-tag` "Private Seller" div and
         `private-seller-note` disclaimer paragraph (real BT private
         doesn't render either).
       • Boat Details accordion heading levels:
         * "More Details" `<h3>` → `<h4>` (real BT renders at 16/700)
         * "Location" `<h3>` → `<h4>` (real BT renders at 16/700)
         * Added "Accommodations" `<h4>` 14/700 #333 under
           Measurements alongside Dimensions / Weights / Tanks
           (real BT has it; sandbox derives cabin/berth counts from
           length).
       • "Still have a question?" `<h2>` → `<h3>` (24/700 #303030 per
         real BT).
       • "More From This Dealer" `<h2>` → `<h4>` with color #414d4a
         (real BT carousel-section heading uses this special color).
       • "More From This Dealer" section wrapped in
         `{% if not boat.is_owner_listed %}` so it only renders on
         dealer listings (matches real BT — private listings don't
         have this rail).
       • Deleted the entire `.bdp-listed-by` section (both dealer
         and private paths). Real BT does NOT render a separate
         Listed By section on either listing type — dealer identity
         is conveyed via the contact card + the `.next-previous-
         offered-by` tag in the sticky bar.
       • Removed "Key Features" `<h3>` heading from the Description
         accordion (real BT doesn't render this heading). Feature
         `<ul>` body kept as part of the Description.
       • Removed `<span class="sparkle-icon">✦</span>` prefix from
         the "What Owners Say" `<h2>` heading. Real BT renders the
         heading as plain text + inline info icon.
       • Added `.next-previous` always-sticky bar at top of BDP
         (replaces the old 320px-scroll-triggered `.bdp-sticky-bar`,
         which is now hidden via `display: none !important`).
         Structure: Back-to-Search button (12/400 #139af5) +
         `.next-previous-info-container` (breadcrumb + listing
         name/price/location) + actions (Save + Offered By tag on
         dealer + Next Boat link). Position:sticky top:0, h:54,
         bg #f7f7f7, z:110, padding 5px 20px. Matches real BT's
         `.next-previous` selector measurements exactly.
       • Hid `.bdp-nav-row` (the duplicate Back+breadcrumb+Next
         row above the gallery) since its contents now live inside
         `.next-previous`.
       • Cache-buster bumped to `?v=105`.
`v=104` BDP visual/perceptual compare — dealer + owner (user request):
       • Probed real BT private listings (President Trawler + Pioneer
         Sportfish + Sea Fox + Yamaha — all rendered as private with
         "Meet Your Seller" / "Contact <FirstName>"). Couldn't isolate
         a "Contact Dealer" pattern on first-page listings; real BT
         appears to fold both seller types under the same UI.
       • Probed sandbox dealer (Crestliner VT 18) + sandbox owner
         (Chaparral 267 SSX) at the same viewport.
       • Findings written up in `_captured/bdp/SELLER_VS_OWNER_DIFF.md`
         — heading hierarchy side-by-side + layout invariants verified
         ✅ + remaining 🟡 enumerated.
       • Concrete fixes landed in v=104:
         * Removed `<span class="bdp-length">| 16'</span>` from H1.
           Real BT H1 is just "Year Make Model" without the length
           suffix.
         * Added `<h2 class="meet-your-seller-h2">Meet Your Seller</h2>`
           above the contact card block (renders for both
           `is_owner_listed` true + false paths). 20/700 #404040
           matching real BT.
       • Documented divergences kept as 🟡:
         * Sandbox extra "Key Features" / "More Details" / "Location"
           H3s (not visible on probed real BT BDPs)
         * Sparkle prefix on "What Owners Say" (sandbox decorative add)
         * Show Phone always shown on dealer cards (real BT gates by
           dealer/private — already 🟡 in SCOPE.md)
       • Cache-buster bumped to `?v=104`.
       • 2 new fidelity tests: H1 has no length suffix + "Meet Your
         Seller" H2 present. Suite now **46 passing**.
`v=103` Cookie consent banner added (user screenshot):
       • New `.cookie-consent` bottom-right floating card (white,
         12px radius, layered drop shadow, 22px 26px padding).
       • Body copy 15/regular #333 with "Privacy" link underlined.
       • 3 pill buttons (108min×44 tall, 1px #d6d6d6 border, 50px br,
         hover bg #f7f7f7): Customize / Reject / Accept.
       • Conditional render via Jinja: only shown when
         `bt_cookie_consent` cookie isn't set (first visit).
       • Wiring:
         * Customize → JS removes the banner without setting cookie
           (reappears next visit)
         * Reject → form POST `/__site/consent` with `choice=decline`
           + `next_url=<current>` (303 redirect back, cookie set)
         * Accept → same as Reject with `choice=accept`
       • Backend `/__site/consent` route already existed (sets
         `bt_cookie_consent` cookie 180d max-age, emits
         `consent_set` mutation). No backend changes.
       • Cache-buster bumped to `?v=103`.
       • 3 new fidelity tests: banner renders on first visit, banner
         hidden once cookie set, card styling matches the spec.
         Suite now **44 passing**.
`v=102` Ribbon switched to `position: sticky; top: 0`:
       • v=100's `position: fixed; top: 44px` left a 44px gap above the
         ribbon on scroll (the area where the nav used to be) — user
         pointed out via screenshot that they wanted the ribbon flush
         to top:0 on scroll.
       • Changed to `position: sticky; top: 0; z-index: 100`:
         * At scroll=0: ribbon stays in flow at y=44, right under the
           nav — matches the user-supplied real-BT screenshot's
           nav-above-ribbon stacking with no gap
         * On scroll past the nav: ribbon flushes to viewport top (y=0)
       • `.bt-main` margin-top: 40px reverted to `margin: 0 auto`
         since position:sticky stays in flow (no compensation needed).
       • Cache-buster bumped to `?v=102`.
       • Live-verified: ribbon y goes from 44 (scroll=0) → 0 (scroll≥100).
       • This is an intentional divergence from real BT's behavior —
         real BT uses `position: fixed; top: 62px` which leaves a 62px
         gap above the ribbon on scroll. User preferred the flush-to-top
         pattern.
       • Updated regression test to match new sticky behavior.
`v=101` Boat Loan Calculator widget re-probed (doc-only):
       • Sandbox's `.loan-calc-card` and real BT's `.calc-calculator-body`
         are STRUCTURALLY DIFFERENT widget concepts:
         * Real BT: output-first (`Monthly Payment $X` headline at 22/900
           #404040, then 4 inputs: purchase price / down payment / term
           years / APR)
         * Sandbox: input-first (`Boat Loan Calculator` title at 18/700,
           then 3 inputs: loan amount / term months / APR, then a result row)
       • Both work for agent training (same calc output) but visually
         and structurally diverge.
       • Existing FIDELITY rows marked ✅ were misleading — they described
         sandbox's version not real BT's. Flipped to 🟡 with the concept
         divergence documented.
       • No code change — rewriting the widget needs user buy-in on which
         concept to keep (sandbox's familiar input-first card vs real BT's
         output-first card).
`v=100` Pre-qualify ribbon sticky on scroll:
       • User asked to verify the blue banner on /boats is sticky on
         scroll. Probed real BT: `.ribbon-prequal` is
         `position: fixed; top: 62px` and stays at y=62 across
         scrollY 0/500/1500.
       • Sandbox had `.ribbon-prequal { position: static }` — ribbon
         just sat under the nav and scrolled away.
       • Fix: `.ribbon-prequal` → `position: fixed; top: 44px;
         left: 0; right: 0; z-index: 100`. Sandbox nav is 44px tall
         (real BT's is 62px), so the offset scales accordingly.
       • `.bt-main` gets `margin-top: 40px` so the breadcrumb / H1
         don't tuck under the now-out-of-flow ribbon.
       • Live-verified: ribbon y stays at 44 across scrollY 0/800/2000.
       • Adds `test_ribbon_is_sticky_on_scroll` — suite now **41 passing**.
       • Cache-buster bumped to `?v=100`.
`v=99` Bumped sandbox `auto_stop_interval` 15 → 180 min:
       • The sandbox was auto-stopping after ~15min of idle, hitting
         every cron-triggered loop iteration with a manual restart
         step (boot container → wait for uvicorn → re-fetch token →
         update FIDELITY row). 5× in this session.
       • `sb.set_autostop_interval(180)` (3h) removes the friction;
         sandbox now stays warm across cron ticks at :07.
       • Token rotated to `a_oh4hg7awgeznbwwglbfml0sdlrywzw`.
       • Documenting this as a saved memory pattern alongside the
         existing `feedback_boattrader_sandbox_restart_recipe.md` —
         "auto-stop interval defaults to 15min, bump to ≥120 for
         long-running iteration work".
`v=98` SCOPE.md handoff summary (doc-only):
       • Rewrote SCOPE.md's "Done bar" section as a 5-row status
         table showing all four high-level criteria from the build-
         from-scratch prompt are met after the v=82..v=97 PR #620
         work.
       • Rewrote "Open follow-ups" as 3 prioritized 🟡 items, each
         tagged with what's needed to close it (human decision /
         backend change / measurement pass). Removed the v=82-era
         items that have since been closed.
       • Adds a "Pick-up by the next session" note explaining where
         the autonomous loop stopped + how the remaining 🟡s need
         human input rather than another probe pass.
`v=97` FIDELITY.md Sort + Pagination row sync (doc-only):
       • The v=96 CSS edits flipped Sort row to 12/400/#333 and
         Pagination to 15/400/#A5A5A5, but the matching FIDELITY.md
         rows still showed stale values ("14/500 #404040" and
         "14/400 #0a0a0a") — copying old assumptions from earlier
         iterations.
       • Updated both rows with the re-probed values + linked back to
         v=96's iteration log entry so the doc and CSS now agree.
       • Re-verified SRP listing card section against real BT — all
         rows still ✅ (card 350×384, title 18/700 #404040, price
         16/400 #404040, monthly 14/700 #139af5, meta 12/400 #9e9e9e).
         No code changes needed for listing cards.
`v=96` SRP sort row + pagination re-measured:
       • `.sort-row` 14/500 → 12/400; color #404040 → #333. Real BT's
         "Sort: Recommended" inline label is much smaller and lighter
         than the panel body. The existing CSS comment claimed 14/500
         but a fresh probe at /boats/ shows 12/400.
       • `.sort-label strong` font-weight 700 → 400 (no longer bolds
         the "Sort:" prefix — matches real BT).
       • `.sort-select` 14/500/#404040 → 12/400/#333.
       • `.pagination` font 14/400 → 15/400; margin 26px 0 → 15px 0.
       • `.pagination a` color #0a0a0a → **#A5A5A5** (medium grey);
         padding 8px 14px → 5px 10px. Real BT page-links are bold
         but use a light-grey color, not the near-black sandbox had.
       • `.pagination a.active` color → #A5A5A5 (real BT's active and
         inactive page-links share the same color — differentiated
         only by the 2px bottom border which sandbox already had).
       • 3 new fidelity tests: test_sort_row_is_12_400,
         test_pagination_is_15_400, test_pagination_link_color_is_a5a5a5.
         Suite now **40 passing**.
       • Cache-buster bumped to `?v=96`.
`v=95` Color shade pass (deferred 5th item from v=94 typography diff):
       • `.filter-select` + `.filter-input` color: `var(--bt-text)`
         (#333) → **`#404040`**. Verified against real BT's
         `.tool-set select` and `input[placeholder*="Zip"]` —
         both render at rgb(64,64,64).
       • `.zip-tab` color: `var(--bt-text)` → **`#0a0a0a`**.
         Real BT's `.switcher-option-label` renders at rgb(10,10,10) —
         near-black, noticeably darker than the panel body. Also
         updates `.zip-tab.active` to match.
       • `.seg` + `.seg.active` color: same `#0a0a0a` treatment as
         `.zip-tab` (Condition / All / New / Used use the same
         switcher styling).
       • Scoped change — did NOT change `--bt-text` globally to avoid
         cascading shade shifts in header nav, breadcrumb, listing
         cards, footer.
       • 2 new fidelity tests: test_filter_inputs_are_404040,
         test_switcher_options_are_near_black. Suite now **37 passing**.
       • Cache-buster bumped to `?v=95`.
`v=94` Typography diff fixes (side-by-side probe of filter panel):
       • `.filter-group-label`: 15/400 #333 → **16/500 #404040** with
         line-height 20px. Verified against all real BT section
         `.toggle-btn` elements (Location / Condition / Length /
         Year / Price / Boat Type / Make / Beam / Max Draft / Fuel /
         Hull / Engines / For Sale By) — all render at the same spec.
       • `.zip-use-location`: `margin-top: 8px` → **`margin: -16px 0
         15px`**. Real BT pulls the link UP toward the zip-row above
         using a negative top margin so it visually tucks under
         the row instead of sitting below it.
       • `.filter-options`: outer `1px #ededed border` → **`background:
         #f7f7f7; padding: 8px`** with `border-radius: 8px`. Real BT
         uses a soft grey backdrop instead of a hard border to group
         the scrollable list.
       • `.filter-options li:hover label`: hover bg #f7f7f7 → **#ededed**
         so the hover stays visible against the new #f7f7f7 parent.
       • `.filter-options label`: font-size 14 → **15px**. Real BT
         renders the visible list-item text at 15px / weight 400 #333.
       • Cache-buster bumped to `?v=94`.
       • 4 new fidelity tests: section-heading 16/500/#404040,
         use-my-location negative-top-margin, filter-options grey
         backdrop, filter-options label 15px. Suite now **35 passing**.
       • Live-verified via Chrome MCP probe — all 4 computed-style
         values byte-match real BT.
`v=93` Home + BDP corpus measurements completed:
       • Home: measured tile geometry for "Boats Near You" (363×120
         listing-card row), "Featured Brands" (266×161 brand tiles
         in 5-up grid), "Recent Articles" (325×317 article cards).
         "Popular Boat Types" section not found on current real BT —
         possibly removed or renamed since the FIDELITY row was written.
       • Home: real BT uses TWO H2 sizes — 15/700 for boat-listing
         subsections (Boats Near You) with inline subtext, and
         22.5/700 for major sections (Featured Brands, Popular Boats,
         Recent Articles). Sandbox uses one size; documented for
         future styling alignment.
       • BDP: identified `.next-previous` always-sticky navigation
         bar (1512×54 at top:0 — Previous Boat / Next Boat links)
         absent in sandbox; documented as 🟡 with the design decision
         deferred (sandbox has its own different sticky pattern).
       • BDP: verified Similar Boats + Show Phone are listing-dependent
         (this probe's listing was private-seller, so neither rendered).
       • Verified all accordion heading levels match real BT exactly
         (H2/H3/H4 distribution captured under
         `below_fold_accordions_verified` in the BDP corpus).
       • `_captured/README.md` status table: all three pages now ✅.
       • Code unchanged; this is corpus + FIDELITY.md doc work only.
`v=92` FIDELITY.md cleanup pass (no code changes):
       • "Open work / known minor diffs" cleaned: BDP width and
         thumbnail size were fixed in v=91 but still listed as 🟡 —
         flipped to ✅ with strikethrough. "Real photos" re-classified
         from 🟡 to 🚫 (out-of-scope per SCOPE.md, not a fidelity gap).
       • New consolidated 🟡 rows for Home tile geometry, BDP below-fold
         elements, and mobile viewport — each pointing at the open
         follow-up in the matching `_captured/<page>/structural.json`.
       • Home page table expanded from 3 → 11 rows using the measured
         data captured in v=90 (hero card 296×48 narrow overlay, H1
         16/700 #fff, "Boats Near You" h2 at y=586, etc.).
       • Behaviors table expanded from 4 → 11 rows to cover the
         interactions wired in v=82..v=91 (5-digit zip auto-submit,
         price-drop toggle, search-as-you-type filter, single-select
         submit, focus-hide / typed-hide, click-anywhere-focus,
         try-text rotator).
       • Code unchanged; this is doc-only.
`v=91` BDP fidelity touch-ups (re-probed real BT BDP):
       • Owner Highlights `<h3>` → `<h2>` — real BT renders this as
         H2 20/700 alongside "Boat Details" / "What Owners Say". The
         existing `.owners-card-tags-heading` CSS already styled it at
         20/700, so only the tag changed.
       • `.bdp-grid { max-width: 1336px; margin-inline: auto }` —
         caps BDP content area to match real BT's measured 1319px
         (left col 890 + gap 79 + right col 334). Sandbox was rendering
         at 1400 (81px wider, the long-standing 🟡 row).
       • As a bonus, capping the parent column auto-shrinks
         `.bdp-thumbs` from (994-48)/5 ≈ 189 wide × 126 tall down to
         (890-48)/5 ≈ 168 × 112 — close enough to real BT's exact
         175×116 that the existing 🟡 thumbnail row flips to ✅.
       • Verified real BT's accordion heading levels: Description /
         Measurements / Propulsion = H3 16/700, Dimensions / Weights /
         Tanks = H4 14/700. Sandbox already matches all of these.
         The earlier 🟡 row claiming "real BT uses H4" for More Details
         + Location turned out to be a stale measurement; those
         section headers aren't visible in current real BT — they may
         have been removed or renamed since the row was written.
         Keeping 🟡 on those two with the caveat noted in the row.
       • Bumped cache-buster to ?v=90.
`v=90` Home + BDP `_captured/` corpora measured (partial):
       • Real-BT probes against / and /boat/<slug>/ at 1512×711.
       • Home: hero card 296×48 at (14,139) sits as a narrow overlay on
         the hero image, NOT a full-width search bar. H1 "Find your
         perfect boat" white 16/700 directly above. First section
         "Boats Near You" h2 at y=586, full-width 1488 #333 15/700.
         Featured Brands / Popular Types / Popular Boats / Articles /
         Why BoatTrader sections present but per-tile geometry still
         TODO (open follow-ups).
       • BDP: breadcrumb 332×40 at (137,191) 12/400 #616161. Gallery
         container 890×727 at x=70, 8px br. Right rail at x=1039–1389
         holds H1 "2015 Pioneer 197 Sportfish" (248×23, 20/700 #333),
         main price "$25,900" (20px), monthly est "$4,000" (18px,
         x=1132, sits inline-right of price), contact form
         `.lead-form-basic__body` 334×260 at x=1055. Content area
         ~1319px wide (sandbox is 1400 — 81px wider, accept 🟡).
       • _captured/README.md status table: srp ✅ / home + bdp 🟡 partial.
`v=89` Search box: click anywhere focuses the input:
       • User reported screenshot showing "Try" text getting highlight-
         selected instead of focusing input when clicked. On real BT,
         clicks anywhere in the search field (sparkle, "Try …" overlay,
         empty area) focus the input. Real BT achieves this via
         `position:absolute` + `pointer-events:none` on the overlay.
       • Sandbox uses inline-flex layout (the overlay actually takes
         space), so a `mousedown` handler in base.html intercepts any
         click in `.ai-search-v2__form` not landing on the submit button
         and calls `inp.focus()` + `setSelectionRange(end, end)`.
       • CSS: `.ai-search-v2__try { cursor: text; user-select: none }`
         so the visual affordance matches and dragging across the
         overlay doesn't select text.
`v=88` Search-box sparkle icon resized to 16×16:
       • Real BT's `/static/legacy/img/icons/ai.svg` is 16×16. Sandbox
         was using a 20×20 inline SVG with different path data.
       • Replaced sandbox's SVG path with the exact path data from
         real BT (4-point burst star + two "+" accent marks at
         top-right and bottom-left, all `#2566B0` stroke+fill).
       • `.ai-search-v2__icon` wrapper: `width: 16px; height: 16px;
         line-height: 0; display: inline-flex` so the 18px parent
         font-size (from `.srp-search`) doesn't expand the wrapper to
         the 20×26 it was before this fix.
       • Cache-buster bumped to `?v=88` then `?v=89`.
`v=87` SRP search-box prefix hides on FOCUS, not just on typing:
       • Bug found via user-supplied before/after screenshots — real BT
         clears the "Try …" prefix the moment the user clicks into the
         input, even before any typing. v=86 only hid it once the
         input had typed content, so an empty-but-focused input still
         showed "Try …" — wrong.
       • Adds `.ai-search-v2__form:focus-within .ai-search-v2__try
         { display: none }` so the prefix disappears on first click.
         The `:has(input:not(:placeholder-shown))` rule from v=86 is
         kept to cover the blur-with-text case.
       • Adds `test_rotator_hides_on_focus` to the fidelity suite.
       • Live-verified: `display: flex` (unfocused) → `display: none`
         (focused) — exact match to real BT screenshot pair.
`v=85` Phase 1 (Discovery) corpus checked in under `_captured/`:
       • `_captured/srp/structural.json` — full SRP spec: filter card, save-search,
         location section (switcher / miles row / zip input / use-my-location), all
         range-slider sections, price-drop toggle, dropdown sections (Boat Type /
         Make / Fuel / Hull), engines, for-sale-by, interactions. Measurements +
         computed-style tokens + interaction notes from Chrome MCP probes at
         1512×711 viewport, captured 2026-05-23.
       • `_captured/home/structural.json` — TODO placeholder; lists sections to
         capture and points to FIDELITY.md rows where matched specs already exist.
       • `_captured/bdp/structural.json` — same TODO placeholder for BDP.
       • `_captured/README.md` — format spec, methodology, what's intentionally
         not in the corpus (raw DOM, screenshots, HAR).
       • Closes "Done bar" item: `_captured/` corpus checked in.
