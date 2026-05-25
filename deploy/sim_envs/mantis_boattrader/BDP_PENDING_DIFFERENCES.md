# `mantis_boattrader` — BDP pending differences vs real boattrader.com

Snapshot of remaining perceptual deltas between the sandbox BDP and
real `boattrader.com` after the iter1..iter18 fidelity pass (PR
`feat/bdp-fidelity-iter1`, 26 commits). Captured 2026-05-24.

Read `BDP_FIDELITY_PLAN.md` for the original section-by-section diff
and `FIDELITY.md` for the version-by-version closure log. This doc
only lists what is **not yet matched**.

## Categorization

- 🚫 **Intentionally not matched** (per `SCOPE.md`) — placeholders
- 🟡 **Cosmetic** — minor visual diff, no functional impact on the
  agent's screenshot-grounded perception
- 🟠 **Data-shape** — sandbox catalog generates different value shapes
  than real BT (e.g. always 10-digit phone, no "Engine Model" field)
- 🔴 **Structural** — markup or layout diff that's visible at first
  glance; should be next priorities

## Gallery

| Element | Real BT | Sandbox | Severity | Notes |
|---|---|---|---|---|
| Main image | Real boat photos | Procedural SVG placeholder | 🚫 | Per SCOPE.md §Out-of-scope |
| Thumbnail strip | Real thumbs, 5 visible at 175×116, blue outline on active | Procedural SVG, 5 visible, blue outline | 🚫 | Photos are placeholder |
| `View N Photos` overlay | First thumb darkened, white centered text | Matched | ✅ | |
| Share / Like / Prev / Next buttons | 40×40 circle, rgba(0,0,0,0.3) | Matched | ✅ | |

## Top navigation (nav row + sticky bar)

| Element | Real BT | Sandbox | Severity | Notes |
|---|---|---|---|---|
| Sticky bar | `next-previous show-info next-previous-top` class, two scroll states (breadcrumb / title+price+save) | Matched: `bdp-scrolled` (320px) + `bdp-scrolled-deep` (700px) toggles | ✅ | |
| Save icon in sticky bar | CSS-class heart on `<button class="heart">` | Inline SVG heart + "Save" text inside `.sticky-save` | 🟡 | Visually similar at small size |
| Breadcrumb category 2 | `Sail` (sailboats) / `Power` (everything else) | Matched (Jinja `'Sail' in boat.boat_type` gate) | ✅ | |
| `< Search` link | Plain text link, no padding/border | Matched | ✅ | |
| Previous / Next button | 104×34, padding 11/15 (next 11/35/11/15), 4px radius, chevrons via ::before/::after | Matched | ✅ | |

## Right-rail summary card

| Element | Real BT | Sandbox | Severity | Notes |
|---|---|---|---|---|
| Outer card | 366×, 1px #ededed, 8px radius, 16px pad, soft shadow | Matched | ✅ | |
| Title | 20/700 #333, no length suffix in H1 | Matched (length is sibling span) | ✅ | |
| Length suffix | Sibling `<span>` with `\| XX'` | Matched (`.bdp-length` inside `.bdp-title-row`) | ✅ | |
| Price + drop arrow | Bold price + ↓ + drop amount | Matched (Unicode ↓) | 🟡 | Real BT may use SVG arrow on some listings — not yet probed |
| Engagement row | `views \| saves \| listed` w/ 1×12 hairline dividers, gap 10px | Matched | ✅ | |
| Views K-format | `1.5k Views` for >999 | Matched | ✅ | |
| Dealer card heading | `Contact <dealership-name>` (NOT salesperson) | Matched | ✅ | |
| Headshot avatar | NOT rendered in side-panel | Matched (dropped) | ✅ | |
| Pin icon + address | 14×14 SVG pin + street, city, state, zip | Matched | ✅ | |
| Phone | E.164 `+1XXXXXXXXXX` (no parens/dashes) | Matched (`seed.py` regenerated) | ✅ | |
| Form fields | First & Last Name / Email + Phone (split) / Message | Matched | ✅ | |
| Submit button | FILLED `rgb(37,102,176)` blue, "Contact Seller" (both variants) | Matched | ✅ | |
| Verified Broker badge | Present on dealer cards on some real BT listings | Not in sandbox | 🟠 | Data-shape: sandbox dealers don't carry verification flag |
| Salesperson name pattern | Real BT shows e.g. "Contact Bill Adams" on some listings | Sandbox always uses dealership-name heading | 🟠 | Could gate on a dealer flag — not yet done |
| Private-seller card | NO pill, NO disclaimer, generic "Contact Private Seller" heading | Matched | ✅ | |

## Stats strip (below gallery)

| Element | Real BT | Sandbox | Severity | Notes |
|---|---|---|---|---|
| Card dims | 166×39, 4-col × 2-row grid | Matched | ✅ | |
| Label/value typography | Label 14/700 #303030, value 14/400 #333 | Matched | ✅ | |
| Engine(s) Hours stat | Present on power boats, absent on sailboats | Matched (Jinja gate) | ✅ | |
| Engine(s) Hours ⓘ tooltip | Inline ⓘ next to label | Dropped (would wrap label on 166px) | 🟡 | Acceptable for now |
| YEAR text inside icon | Not present | Dropped from sandbox SVG | ✅ | |

## "What Owners Say" card

| Element | Real BT | Sandbox | Severity | Notes |
|---|---|---|---|---|
| Sparkle ✦ icon | 24×24 SVG with fill #2566B0 (AI label) | 18px Unicode `✦` with `color: var(--bt-blue)` (#2566b0) | 🟡 | Visually similar |
| Info ⓘ tooltip | Material-Icons font `info` glyph | Unicode circled-i | 🟡 | Visually similar |
| H2 styling | 20/700 #333 | Matched | ✅ | |
| Owner Highlights pill | bg #e3f1fe, color #2566b0, 12/700, 4×8 pad, 5px radius | Matched | ✅ | |

## Boat Details accordion section

| Element | Real BT | Sandbox | Severity | Notes |
|---|---|---|---|---|
| Section bg | `#f5f9ff` tint, 20×16 pad, 8px radius | Matched | ✅ | |
| H2 "Boat Details" | 20/700, margin 0 | Matched | ✅ | |
| 5 accordions (Description / Measurements / Propulsion / More Details / Location) | Present on power boats | Matched | ✅ | |
| Accordion item padding | `<details>` 18px 0, `<summary>` 0, height ~28px | Matched (sandbox ~23px) | ✅ | Close to real |
| Accordion border-bottom | 1px #ededed between items | Matched | ✅ | |
| Description body | 15/19.5 (lh 1.3), padding 8px 0 0 8px | Matched | ✅ | |
| Seller blurb | `<strong>` topic line at 15/700 | Matched | ✅ | |
| DESCRIPTION sub-label | `<strong>DESCRIPTION</strong>` at 15/700 (NO letter-spacing) | Matched | ✅ | |
| Show More link | Right-aligned, 13/400 BT-blue button, expands clamp | Matched (clamp at 168px, JS toggle) | ✅ | |
| Propulsion Engine Hours line | Absent on sailboats | Matched (Jinja gate) | ✅ | |
| Engine Model / Engine Year fields | Present on real BT propulsion | Not in sandbox data model | 🟠 | seed.py would need to expose engine_model + engine_year |

## Mid-page ad strip

| Element | Real BT | Sandbox | Severity | Notes |
|---|---|---|---|---|
| Strip bg #f7f7f7 spans full viewport | Edge-to-edge 1600px | Matched (full-bleed via `margin: 24px calc(50% - 50vw); width: 100vw`) | ✅ | |
| Banner | 970×120 centered creative | Matched (970×120 with `Sponsored` corner tag) | ✅ | |
| Strip height | 122px | Matched | ✅ | |
| Ad creative | Real advertiser image | Procedural SVG placeholder | 🚫 | Per SCOPE.md |

## Loan Calculator

| Element | Real BT | Sandbox | Severity | Notes |
|---|---|---|---|---|
| Form heading | 20/700 "Boat Loan Payment Calculator" | Matched | ✅ | |
| Form fields (4) | Year / Purchase Price / Down Payment / Loan Term (Months) | Matched | ✅ | |
| Input style | 348×40, 4px radius, 1px #ededed border, 16px font | Matched | ✅ | |
| Preview pane bg | Light-blue tint (#f5f9ff or close) | Matched | ✅ | |
| Monthly payment display | 50px / 700 centered | Matched | ✅ | |
| Total Loan Amount line | 13/normal, centered | Matched | ✅ | |
| Disclosure link | "See Important Disclosure³" | Matched | ✅ | |

## Listed By + More From This Dealer

| Element | Real BT | Sandbox | Severity | Notes |
|---|---|---|---|---|
| Section bg | `#f5f9ff` tint matching Boat Details | Matched | ✅ | |
| "Listed By" heading | 20/700 (DIV in real BT, H2 in sandbox) | Visually matched | 🟡 | Tag-level diff, no visual impact |
| Logo + dealer name | Logo mark + uppercase dealer name | Matched | ✅ | |
| Trusted Boat Trader Partner label | Plain text at bottom (no `\| <years>` suffix) | Matched | ✅ | |
| Stat boxes (Active / Sold listings) | Bold num 16/600 + small label 14/400 | Matched | ✅ | |
| Visit Seller Website link | Plain text, NO ↗ arrow | Matched | ✅ | |
| More From This Dealer carousel | Horizontal rail of image-only cards, "More Boats from this Dealer" link below | Matched | ✅ | |
| MFD link color | #0051AD (darker than main BT-blue) | Matched | ✅ | |

## Footer / page-level

| Element | Real BT | Sandbox | Severity | Notes |
|---|---|---|---|---|
| Body width | `<main class="page-container">` spans full viewport (1600px) | Matched (`.bt-main { max-width: none }`) | ✅ | |
| Footer container | Dark navy band full-width, inner 1440px centered | Matched (`.site-footer + .footer-cols`) | ✅ | |
| Ribbon (prequal) | Top blue band, full-width | Matched | ✅ | |

## Variant coverage smoke

| Variant | URL pattern | Status |
|---|---|---|
| Dealer (yacht) | `2018-princess-f55-10000418` | ✅ all anchors match |
| Private seller | `2025-chaparral-267-ssx-10000008` | ✅ all anchors match |
| Sponsored | `2026-robalo-r230-10000209` | ✅ SPONSORED LISTING pill renders |
| POA ("Request a Price") | `2022-boston-whaler-210-dauntless-10000425` | ✅ no monthly link, no ⓘ |
| Price drop | `2021-crestliner-vt-18-10000396` | ✅ price + ↓ + drop amount |
| Sailboat | `2021-catalina-385-10000363` | ✅ Sail breadcrumb, 7 stats, no Engine Hours in Propulsion |

## Known remaining gaps (priority order)

### 🟠 Data-shape (sandbox catalog doesn't model)

1. **Verified Broker badge** — Real BT renders a "Verified Broker"
   row on some dealer cards. Sandbox dealers don't carry a
   `verified_broker` flag. Add one + render conditionally.
2. **Engine Model / Engine Year fields** — Real BT Propulsion accordion
   surfaces these as separate rows (e.g. `Engine Model: 3YM30AE`,
   `Engine Year: 2023`). Sandbox `Boat` dataclass doesn't have them.
   Add fields + render.
3. **Salesperson-name dealer pattern** — Some real BT dealer cards
   use `Contact <salesperson>` instead of `Contact <dealership>`.
   Sandbox previously rendered the salesperson pattern; iter10 swapped
   to the dealership pattern (matching the user-shared Westchester
   example). Restore conditional rendering driven by a `dealer.salesperson`
   flag if needed.

### 🟡 Cosmetic (acceptable for now)

4. **Sticky Save heart** — Real BT uses CSS-class heart on a
   `<button class="heart">`; sandbox uses inline SVG + text. Visually
   similar at small size; would need a CSS background-image swap.
5. **Sparkle ✦ in "What Owners Say"** — Real BT uses a 24×24 inline
   SVG with `fill="#2566B0"` and `aria-label="AI"`. Sandbox uses
   Unicode `✦` at 18px in `var(--bt-blue)`. Same color, slightly
   different glyph weight.
6. **Info ⓘ tooltips** — Real BT uses Material-Icons font `info`
   glyph; sandbox uses Unicode `ⓘ`. Same intent, slightly different
   render.
7. **Engine(s) Hours stat ⓘ** — Dropped in iter5 because the icon
   wrapped the label on a 166px card. Real BT's narrower Roboto
   rendering fits both. Add back if/when the font baseline is closer.
8. **Dealer-rail card dim** — Sandbox 220×271 vs real 235×260
   (carousel card inside More From This Dealer). Marginal.
9. **Price drop arrow** — Sandbox uses Unicode `↓`; real BT may use
   a custom small SVG. Not yet probed on a real-BT price-drop listing
   (the ones we tried didn't surface drops live).

### 🚫 Intentionally not matched (per SCOPE.md)

10. **Real ad creatives** — sandbox uses 6 procedural SVG ads in
    fixed rotation.
11. **Real boat photos** — sandbox uses procedural SVG boats.
12. **Dealer logos** — sandbox uses initial-letter rectangles.
13. **Analytics scripts** — sandbox keeps class names matching real
    BT but doesn't run gtag / FullStory / Kameleoon.
14. **OneTrust cookie banner** — sandbox renders a generic accept/
    reject pair.

## How to extend this doc

When you find a NEW delta in a future iter, add a row under the
relevant section. When a delta is closed (matched), move it from
"Known remaining gaps" up into the matched-row table and mark ✅.
Don't delete the row — its presence in the matched section is the
proof that it was once a gap.

When adding a 🟠 data-shape fix, you usually need to:
1. Add the field to `seed.py` `Boat` or `Dealer` dataclass
2. Generate it in `_seed_boats` / `_seed_dealers`
3. Render it in `boat_detail.html`
4. Hit `/__env__/reset` (or restart uvicorn) to regenerate the catalog
5. Verify via Chrome MCP probe
