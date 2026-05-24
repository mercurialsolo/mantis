# `mantis_boattrader` — Fidelity match log

Working doc tracking each element's match status against
`https://www.boattrader.com/`. Update as iterations land.

Last updated: **v=97** (2026-05-23) — FIDELITY.md Sort row + Pagination rows updated with the re-measured values from v=96 (were still showing stale pre-fix values). SRP listing card section re-verified against real BT probes — all rows ✅.

Live URL: `https://8080-014f48ab-eeb1-4ca5-947e-42e169d1fcc8.daytonaproxy01.net/boats/`
(token rotates per sandbox restart; current: `rzsrm967ibbct7vgbgpzdzudgofezmmv`)

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

| Element | Real BT | Mine | Status |
|---|---|---|---|
| Title | "Boat Loan Calculator" 18/700 | same | ✅ |
| Loan Amount field | input with $ | added | ✅ |
| Loan Term dropdown | 240/180/120/60 months | added | ✅ |
| Interest Rate (APR) | 6.49% default | added | ✅ |
| Calculate button | blue outline | added | ✅ |
| Monthly Payment result | "$0.00" centered | added | ✅ |
| "Get Pre-Qualified" CTA | blue filled pill | added | ✅ |
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
| More Details accordion heading | not visible in v=91 probe (real BT may have removed or renamed this section); existing FIDELITY note claimed H4 but unverifiable today | sandbox uses H3 — leave as-is until real BT shows it again | 🟡 |
| Location accordion heading | not visible in v=91 probe — same caveat as More Details | sandbox uses H3 — leave as-is | 🟡 |
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
- 🟡 **More Details / Location H-level in Boat Details accordion**: real BT
  doesn't currently render these headings on probed listings — earlier
  claim that real BT uses H4 is unverifiable today. Sandbox uses H3.
  Re-probe + decide once real BT shows these sections again.
- ✅ ~~Thumbnail size~~ — auto-fixed by v=91's `.bdp-grid max-width`.
- ✅ ~~Per-tile geometry on Home page rails~~ — measured in v=93. Real BT
  uses 363×120 boat-listing cards, 266×161 brand tiles, 325×317 article
  cards. Sandbox card-grid pattern matches within typical render
  variance; corpus documents the spec for future verification.
- 🟡 **BDP `.next-previous` sticky navigation bar**: Real BT renders a
  1512×54 ALWAYS-sticky bar at top:0 with Previous Boat / Next Boat
  links. Sandbox doesn't have this widget; instead it has a different
  sticky pattern (`.bdp-scrolled` body class after 320px scroll for a
  title+price+contact bar). Different concept — decide whether to add
  the next-previous widget or accept the divergence.
- 🟡 **Listing-dependent BDP elements**: Similar Boats rail and
  Show-Phone button are present on some listings (dealer) and absent on
  others (private seller). Need a dedicated dealer-listing probe pass
  to measure these. Sandbox always renders both — divergence from
  real BT's conditional rendering is functional (agent training fine)
  but visually divergent.
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
