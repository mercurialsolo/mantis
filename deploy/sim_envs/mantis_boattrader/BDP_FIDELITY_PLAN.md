# `mantis_boattrader` — BDP (boat detail page) fidelity plan

Companion to `FIDELITY.md`. This doc is a perceptual diff of our sandbox
BDP vs the real `boattrader.com` BDP, plus an ordered fix plan.

- **Real BDP probed**: `https://www.boattrader.com/boat/2010-contender-32-st-10169273/` — private seller
- **Sandbox BDP probed**: `https://8080-…daytonaproxy01.net/boat/2025-chaparral-267-ssx-10000008/` — private seller
- **Viewport**: 1512×711 (canonical 1440×900 — Retina render at 1512)
- **Date**: 2026-05-23
- **Source of truth**: real-BT screenshots + `getBoundingClientRect` + `getComputedStyle` probes (raw numbers in §Appendix)

> Placeholder policy reminder: images (gallery photos, sponsor ad
> creatives, dealer/headshot logos) stay as SVG procedural placeholders
> per `SCOPE.md` §"Out-of-scope". Everything else has to mirror exactly.

## Severity legend

- 🔴 **block** — perception-breaking, agent will see a different page
- 🟠 **major** — clearly visible delta, fix in this pass
- 🟡 **minor** — pixel/wording delta, fix when convenient
- 🚫 **placeholder** — intentional (per SCOPE.md), no action

## TL;DR — what's wrong vs the real BDP

The sandbox BDP already matches real-BT on the **bulk-layout level**
(2-col grid, gallery width, stats strip count, accordion pattern, loan
calc 2-col, footer cards). The actionable deltas are concentrated in
**two regions**:

1. **Sticky top bar** — sandbox shows the breadcrumb on scroll; real BT
   replaces the breadcrumb with `Title · Price · Location · ♡ Save` once
   the user scrolls past the gallery. Also missing: `Previous Boat`
   link, the `♡ Save` icon, and the position/styling of the contact
   pill differ.
2. **Right-rail contact card (private seller path)** — sandbox adds a
   "Private Seller" pill, addresses the owner by name ("Contact Ravi,
   the owner"), shows a verbose disclaimer, puts "Show Phone Number"
   ABOVE the lead form, and renders the submit as an **outline** pill
   labelled "Email Seller". Real BT shows NO pill, uses generic
   "Contact Private Seller" heading, has no disclaimer paragraph in the
   card body, doesn't surface phone-reveal in the card on most
   listings, and renders the submit as a **filled** blue pill labelled
   "Contact Seller".

Plus a scatter of small wording/heading/icon deltas (§Section diff).

## Section-by-section diff

### 1. Top sponsor ad strip
| Element | Real | Sandbox | Status |
|---|---|---|---|
| Ad block | 970×120 real creative on `#f7f7f7` strip | 970×120 SVG creative w/ `SPONSORED` tag | 🚫 placeholder |
| Strip total height | ~122px including pad | ~122px | ✅ |

No action.

### 2. Top nav row (above gallery)
| Element | Real | Sandbox | Status |
|---|---|---|---|
| Left | `< Search` (light blue 12/400) | `‹ Search` | ✅ |
| Center | breadcrumb (Home / Boats For Sale / Power / Center Console / Contender) 12/400 #616161 | same shape, color #757575 | 🟡 color diff #757575 vs #616161 |
| Right | `Previous Boat` + `Next Boat` (both BT-blue 12/400) | only `Next Boat ›` | 🟠 missing `Previous Boat` (or "Previous Boat" should appear when there's a prior boat in session) |
| Chevron style | plain word link, no chevron | sandbox adds `‹`/`›` glyphs | 🟡 wording diff |

**Fix**: drop the `‹`/`›` glyphs from the back/next links (plain word
links per real); add a `prev_boat` template var symmetrical to
`next_boat` so a `< Previous Boat` link is shown when one exists; nudge
breadcrumb text color #757575 → #616161.

### 3. Sticky bar (appears on scroll)
| Element | Real | Sandbox | Status |
|---|---|---|---|
| Sticky bar height | 54px | 54px | ✅ |
| Bar bg | `rgba(255,255,255,0.98)` w/ subtle bottom shadow | same | ✅ |
| Content (top half scroll) | `< Search · breadcrumb · Previous Boat · Next Boat · [Contact Seller]` | `‹ Search · breadcrumb · Next Boat ›` (no Contact Seller pill in sticky, no Previous) | 🔴 missing pill + Previous |
| Content (mid scroll, past gallery) | `< Search · Title · $Price · City, ST ZIP · ♡ Save · Previous Boat · Next Boat · [Contact Seller]` (breadcrumb REPLACED by title summary) | breadcrumb unchanged at any scroll | 🔴 missing the second display state |
| Sticky bar Contact Seller pill | `bg rgb(37,102,176)` filled, 152×36, white 15/500 | filled pill present visually but no Title/Price summary surfacing it | 🟠 |
| Save (♡) icon in sticky bar | outline heart + "Save" label | not present | 🟠 |

Real BT class is `next-previous show-info next-previous-top` — there
are two visual states (`next-previous-top` and a deeper-scroll variant)
toggled by scroll position. The "show-info" modifier is what flips the
content from breadcrumb-mode to title-mode.

**Fix**: add a second scroll threshold (~gallery height, ~700px) that
flips a class like `bdp-sticky-info` on `body`; CSS then hides the
breadcrumb and shows a `Title · Price · Location · ♡ Save` row. Move
the Contact Seller pill into the sticky bar markup (right-aligned).

### 4. Right rail summary card — common section (before contact form)
| Element | Real | Sandbox | Status |
|---|---|---|---|
| Card width | 366px | 332px | 🟠 32px narrower than real (right rail grid column 366/100/auto) — needs `1fr 80px 366px` grid |
| Card padding | 16px | 16px | ✅ |
| Card border | 1px #ededed | 1px #ededed | ✅ |
| Border radius | 8px | 8px | ✅ |
| Drop shadow | rgba(0,0,0,0.2) 0 2px 2px 0 | same | ✅ |
| Badge above title | NONE (badges live in engagement strip) | `New Arrival` pill at top of card | 🟠 remove `.bdp-badges` above title; merge badge into engagement strip / move below |
| H1 | `2010 Contender 32 ST` — 20/700 #333 (no `| 32'` length appended) | `2025 Chaparral 267 SSX \| 27'` — 20/700 #333 | 🟡 sandbox appends `\| length`; real puts length only in stats strip / sub-text |
| Location line | `Jupiter, FL 33478` — 14/400 #757575 | `Naples, FL 34102` — same | ✅ |
| Price line | `$177,500   ↓$10,000` — 20/700 #333, drop arrow ▼ in light style | `$103,900` (no drop here, but same shape when present) | ✅ |
| "Own this boat for $X/mo" link | 16/400 BT-blue, then `(ⓘ) Customize` | same | ✅ |
| Engagement row | `1.5k Views | 44 Saves | New to Market` with eye/heart/cal icons | `71 Views | 53 Saves | New to Market` | 🟡 sandbox doesn't K-format views >999; pipe separators visible in real (light gray vertical `|`) — verify sandbox renders the dividers (currently row uses gap-only) |
| Engagement font | 12/400 #666 | 12/400 #666 | ✅ |
| Divider below engagement | 1px #ededed bottom-divider | sandbox doesn't render | 🟡 add `border-top` to the contact section beneath |

### 5. Right rail — Private seller variant
| Element | Real | Sandbox | Status |
|---|---|---|---|
| "Private Seller" tag/pill | NOT shown in card | rendered as 105×22 dark-navy pill | 🟠 remove |
| Heading | `Contact Private Seller` (generic) — 20/700 #333 | `Contact Ravi, the owner` (owner name) — 18/500 dark | 🔴 wording + weight wrong |
| Disclaimer paragraph | Not present in card body (BT has a smaller `Tips` block elsewhere) | "This boat is being sold directly by the owner. Boat Trader does not verify…" — multi-line gray paragraph | 🟠 remove from main card (or shrink to a 2-line caption) |
| Show Phone button position | not rendered in card on the listings we tested (BT phone-reveal lives elsewhere or absent for owners w/ no phone) | rendered ABOVE the form as a 100% blue outline pill | 🟠 move to AFTER the form, OR hide on cards where no phone is known |
| Phone-reveal note | not present | "The seller's number is hidden until you confirm…" gray small-text | 🟠 remove or shrink |
| Lead form fields | Name (full) · Email + Phone (split) · Message (full) — 40px h, 4px br, 1px #c2c2c2 | identical structure + dims (✅) | ✅ |
| Submit button | `Contact Seller` (NOT "Email Seller"), FILLED blue pill bg `rgb(37,102,176)` white, 100% width, 40px h | `Email Seller` text, BLUE OUTLINE pill | 🔴 text + filled-vs-outline |
| Email subject prefill in message | `Hi, I'm interested…` (no owner name) | `Hi Ravi, I'm interested in your 2025 Chaparral 267 SSX. Is it still available?` | 🟡 remove owner name from prefill |

### 6. Right rail — Dealer variant (not directly compared this session)
The sandbox currently renders, in this order: headshot SVG + name +
dealer address/phone, then lead form, then a separate `bdp-dealership-card`
with logo + active/sold stats + Visit Seller Website link. Real BT's
dealer variant is shaped similarly but the dealership-card content lives
inside the same outer card (one card, not two stacked cards) for many
listings, and "Active listings / Sold listings" sit in a 2-stat box row.
Re-probe a real BT dealer BDP before changing — this is a follow-up.

| Element | Real (TBD) | Sandbox | Status |
|---|---|---|---|
| Headshot vs dealer logo position | TBD | both rendered | 🟡 needs real-BT confirm |
| `Visit Seller Website ↗` link | plain blue text link | sandbox renders w/ arrow ↗ — ✅ shape | 🟡 |
| `Trusted Boat Trader Partner` row | small caption + `Premier` etc. | identical text | ✅ |
| Active/Sold listings stat boxes | 2 stat cards, num 20/700 + label 12/400 | 2 stat boxes — ✅ shape | 🟡 dim/color verify |

### 7. Below-gallery: stats strip (8 cards 4×2)
| Element | Real | Sandbox | Status |
|---|---|---|---|
| Container | ~870×170 (gallery-column width), 4-col grid | 890×147, 4-col grid w/ 205px tracks | 🟡 minor width diff; sandbox is 20px wider |
| Card dims | 166×39 (smaller cards, less padding) | 205×44 | 🟠 sandbox cards are 39px wider + 5px taller |
| Card grid gap | small (~12-14px) | 14×18px | ✅ close |
| Label | 14/700 #303030 | 14/700 #303030 | ✅ |
| Value | 14/400 #333 | 14/400 #333 | ✅ |
| Icon | circle ring 19r + simple dark blue glyph | circle ring 19r + dark blue glyph | ✅ |
| **Label text "Engine(s) Hours"** | with parens | "Engine Hours" no parens | 🟠 wording |
| Info ⓘ on "Engine(s) Hours" | small gray (i) tooltip trigger | not present | 🟡 add tooltip stub |
| Card label `YEAR` inside icon SVG | not present | sandbox YEAR icon has inline `<text>YEAR</text>` that shows up in innerText | 🟡 visual fine, but extra text contaminates DOM grep — drop the inline label |
| Order | Engine / Total Power / Engine(s) Hours / Class // Length / Year / Model / Capacity | same | ✅ |

### 8. "What Owners Say" card + Owner Highlights
| Element | Real | Sandbox | Status |
|---|---|---|---|
| Heading | `✦ What Owners Say ⓘ` 20/700 #333 | same | ✅ |
| Heading icon | sparkle + info ⓘ | sparkle + info ⓘ | ✅ |
| Paragraph | 14/400 #333, ~5 lines | 14/400 #333 | ✅ |
| Owner Highlights heading | 16/700 #333 | 16/700 #333 | ✅ |
| Tag pills | bg #e3f1fe, color #2566b0, 12/700, 4px 8px pad, 5px br | same per FIDELITY.md | ✅ |

### 9. Boat Details accordions
| Element | Real | Sandbox | Status |
|---|---|---|---|
| Section heading `Boat Details` | 20/700 #333 — no card background or only very subtle | sandbox wraps the section in `#f5f9ff` 8px-radius card with 20px 16px pad | 🟠 real BT does NOT wrap Boat Details in a tinted card (per latest screenshots — looks like a white card with only top/bottom dividers); verify and remove `background-color: #f5f9ff` if so |
| Accordions present | Description (open), Measurements, Propulsion | adds More Details + Location (not in real BT for the listings we tested) | 🟠 remove More Details + Location, OR demote both to H4 within an existing accordion |
| Accordion title | 16/700 #333 | 16/700 #333 | ✅ |
| Chevron | down/up | down/up via border trick | ✅ |
| Open-by-default | only Description | only Description ✅ | ✅ |
| `Show More` link inside Description | bold BT-blue, right-aligned at clamp boundary | sandbox renders FULL description (no clamp + Show More) | 🟠 add CSS line-clamp + JS toggle |
| Key Features sub-list inside Description | not present in real BT | sandbox renders an extra "Key Features" H3 + bullet list inside Description body | 🟠 remove or demote, OR move into its own accordion |

### 10. Mid-page sponsor ad
| Element | Real | Sandbox | Status |
|---|---|---|---|
| Position | between Boat Details accordion and Loan Calculator | same | ✅ |
| Dimensions / strip | 970×120 | same | ✅ |
| Sponsored tag | top-right | top-right | ✅ |

### 11. Boat Loan Calculator (full-width card)
| Element | Real | Sandbox | Status |
|---|---|---|---|
| 2-col grid | left form / right preview w/ light-blue bg on right | same | ✅ |
| `Boat Loan Payment Calculator` heading | 20/700 #333 left | sandbox h3 16/700 — verify weight | 🟡 verify font-size |
| Right "Get pre-qualified" sub-card | bullets + outline button | sandbox uses `<hr>` then bullets — pattern OK | ✅ |
| `See Important Disclosure³` link | small gray, with superscript | same | ✅ |

### 12. "Listed By" / "More From This Dealer" (dealer listings only)
Both omitted from real-BT private-seller BDP. Sandbox correctly hides
`More From This Dealer` for private sellers but still renders a
"Listed By" private-seller card. Real BT does NOT render any "Listed
By" for private sellers — the private-seller info already lives in the
right-rail card. **🟠 hide the `bdp-listed-by` block when
`boat.is_owner_listed`.**

### 13. "Still have a question?" card
| Element | Real | Sandbox | Status |
|---|---|---|---|
| H2 | `Still have a question?` 24/700 #333 | same | ✅ |
| Rule (blue 32×2 line below H2) | present | present | ✅ |
| Body | "Get answers, schedule a visit..." 14/400 / 16/400 | 14/400 ✅ | ✅ |
| `Contact seller` button | blue outline pill, 14/700 | same | ✅ |

### 14. "OTHER SERVICES" tiles
| Element | Real | Sandbox | Status |
|---|---|---|---|
| Label | `OTHER SERVICES` 14/700 ucase | same | ✅ |
| Tiles | logo circle + product name (14/400 #4d4d4d) | same | 🚫 logos are placeholder SVG |
| Tile count | 6+ in real (Surveyor, Documentation, Boat Loans, Insurance, Warranty, Trailers …) | 2 (Elite Marine Surveyors + Boat Documentation) | 🟠 sandbox shows only 2; bump to 6 to match real's row density |

### 15. Footer (out of BDP scope but renders below)
Footer is global — matches real BT structurally per existing
`base.html`. Not in this plan.

## Fix plan — ordered

The ordering goes outside-in: things that move on every scroll first
(sticky bar / nav row), then the highest-attention region
(right-rail card), then below-gallery content. Within each block, the
🔴 deltas land first.

### Pass 1 — sticky bar + nav row  *(small-medium template edit)*

1. **Template** `boat_detail.html`:
   - Remove `‹`/`›` glyphs from `.bdp-back` and `.bdp-next` link
     text → `Search` and `Next Boat` (plain words, link styling).
   - Add a `{% if prev_boat %}<a … class="bdp-prev">Previous Boat</a>{% endif %}`
     symmetrical to `next_boat`; wire `prev_boat` in the route handler
     (`main.py` BDP view) — pick the previous boat in the dealer/seed
     list, mirroring how `next_boat` is computed.
   - Inside `.bdp-sticky-bar`, render BOTH variants:
     - `.bdp-sticky-breadcrumb-state` — current breadcrumb mode
     - `.bdp-sticky-info-state` — Title · `|` · `$Price` · `|` ·
       `City, ST ZIP` · `|` · `♡ Save` (heart SVG + "Save" text)
   - Add a Contact Seller pill at the right end of the sticky bar
     (always visible whichever state is active).
2. **CSS** `app.css`:
   - `body.bdp-scrolled` continues to show the sticky bar.
   - Add `body.bdp-scrolled-deep` (set when `window.scrollY > 700`) →
     `.bdp-sticky-breadcrumb-state { display:none }` +
     `.bdp-sticky-info-state { display:flex }`.
3. **JS** in `base.html`:
   - Add second scroll threshold (700px) toggling `bdp-scrolled-deep`
     class on body.
4. **Wording / color**:
   - Update breadcrumb `color: #616161` (was #757575).

### Pass 2 — right rail private-seller card  *(highest visual impact)*

1. **Template** `boat_detail.html` (`{% if boat.is_owner_listed %}` branch):
   - Delete `.private-seller-tag` pill.
   - Change heading: `<h2>Contact Private Seller</h2>` (generic), use
     `dealer-card-heading` class (same 20/700 #333 as dealer variant).
   - Delete the `.private-seller-note` paragraph.
   - Move the `phone-row` block to AFTER the lead form (last item
     before the closing card), OR gate behind `{% if boat.owner_phone %}`
     so cards w/o a known phone don't show the reveal.
   - Change form submit text: `Email Seller` → `Contact Seller`.
   - Change form submit class from outline to filled — drop
     `btn-block` outline variant and use the same `.btn-primary
     .contact-seller-btn` that the dealer card uses.
   - Remove the `Hi {{ boat.owner_name }},` prefix from the message
     placeholder; use the dealer-card-style generic copy.
2. **Common right-rail tweaks**:
   - Delete `.bdp-badges` block above title (badges already in
     engagement strip; if a badge needs surfacing in the card, prefix
     it inside the engagement row instead).
   - In `.bdp-title`, drop the ` | XX'` length suffix — length lives
     in the stats strip.
   - Add K-formatting filter for `boat.views` (`> 999 → "1.5k"`).
   - Add `border-top: 1px solid #ededed` to the contact-form section
     to create the visible divider real BT shows.
3. **CSS**:
   - Card column width: grid column 3 should be **366px** (matches
     `_captured/srp` recipe). Sandbox currently lands at 332px because
     the `.bdp-grid` template-columns are off. Update
     `grid-template-columns: 1fr 366px` (drop the empty 80px middle
     column or merge into the gap).

### Pass 3 — stats strip + Boat Details accordion

1. **Template** `boat_detail.html`:
   - Stats strip: rename label `Engine Hours` → `Engine(s) Hours`.
   - Add `<span class="info-tip" title="Reported by the seller">ⓘ</span>`
     next to the Engine(s) Hours label.
   - YEAR icon SVG: drop the inline `<text>YEAR</text>` (visual fine
     without it; cleaner DOM).
   - Boat Details accordions: remove `More Details` and `Location`
     accordions OR demote them inside `Measurements`/`Description` as
     H4 sub-sections. Real BT has only **Description / Measurements /
     Propulsion** at this level for the listings we tested.
   - Inside Description: line-clamp the long paragraphs (CSS clamp at
     4 lines) and add a `Show More` toggle button (right-aligned BT-blue
     bold link). Remove the "Key Features" h3+ul render — that block
     isn't in real BT's Description body.
2. **CSS**:
   - Stats card target dims: ~166×39 (currently 205×44). Either drop
     `padding` or tighten the icon-text gap. Real has 4 columns at
     ~166px each ≈ 664+gap < 870 total, so a small `gap: 16-24px`
     with smaller cards is the shape.
   - `.bdp-details-section { background: none; padding: 0; }` — remove
     the `#f5f9ff` tinted bg if real BT renders the section on a plain
     white surface (verify with one more probe on a different real BT
     BDP — could be listing-type dependent).

### Pass 4 — listed-by, services, ad+loan calc cosmetics

1. **Template**:
   - Wrap the `bdp-listed-by` private-seller block in
     `{% if not boat.is_owner_listed %}` — real BT shows nothing here
     for private sellers (Contender 32 ST renders no "Listed By").
   - "Other Services" tiles: expand from 2 → 6 (Marine Surveyors,
     Boat Documentation, Boat Loans, Boat Insurance, Boat Warranty,
     Boat Transport). Logos remain placeholder SVGs (`🚫`).
   - Loan Calc heading: verify `loan-form-heading` is set to 20/700
     (real BT) — currently `font-size: 18px` per `.loan-form-heading`
     CSS; bump to 20px.

### Pass 5 — verification

1. Re-probe both pages with `scripts/perceptual_diff.py` after each
   pass and update `FIDELITY.md` rows.
2. Fill in the now-empty `_captured/bdp/structural.json` with the
   real numbers captured in this session (the appendix below).
3. Add a pytest in `tests/sim_envs/mantis_boattrader/` that asserts
   the BDP-specific structural anchors (sticky-bar two-state markup,
   right-rail card width = 366px, stats-strip label="Engine(s) Hours",
   private-seller heading text = "Contact Private Seller").

## Appendix — raw probes (2026-05-23)

### Real boattrader.com (Contender 32 ST — private seller)

```
viewport: 1512×711
main_image: 890×593 at x=70 (3:2)
breadcrumb: 12/400 rgb(97,97,97) padding 12px 15px 12px 0
H1 (right-rail): 20/700 #333 — "2010 Contender 32 ST" (NO length suffix)
sticky bar: class="next-previous show-info next-previous-top", w=1512 h=54
sticky bar text: "Search 2010 Contender 32 ST $177,500 Jupiter, FL 33478 Previous Boat Next Boat Contact Seller"
sticky Contact Seller pill: bg rgb(37,102,176) filled, 152×36, white
lead Name input: 334×40, 1px #c2c2c2, 4px br, 14/400 padding 8px 12px 4px
submit btn class: style-module_action__ + variants — FILLED blue, label "Contact Seller"
Boat Details H2: 20/700 #333 at x=87 (left col)
Stats labels: "Engine | Total Power | Engine(s) Hours | Class | Length | Year | Model | Capacity"
Stats label style: 14/700 rgb(48,48,48)
Stats card dims: ~166×39
Phone-reveal button: NOT rendered on this listing (private seller w/o phone in card)
Private-seller note paragraph: NOT rendered
```

### Sandbox (Chaparral 267 SSX — private seller)

```
viewport: 1512×711
H1: 20/700 #333 — "2025 Chaparral 267 SSX | 27'" (sandbox appends length)
sticky bar (.bdp-sticky-bar): "‹ Search Home / Boats For Sale / Power / Bowrider / Chaparral Next Boat Contact S…"
back/next: 12/400 rgb(19,154,245) "‹ Search" / "Next Boat ›" (no Previous)
right-rail badge pill: rendered (`New Arrival`) above title
right-rail width: 332px (TARGET 366)
right-rail bg/border: white, 1px #ededed, 8px br, padding 16px — ✅
price: 20/700 #333 — ✅
engagement row: "71 Views" — 12/400 rgb(102,102,102)
private-seller pill: 105×22, 11/700 dark navy
contact heading: "Contact Ravi, the owner" — wrong wording + weight
submit button: outline pill labelled "Email Seller" — wrong style + label
stats-strip: 8 cards, 205×44, 14×18 gap — too wide vs real's 166×39
accordions: Description, Measurements, Propulsion, More Details, Location — 2 extras
```

## Open follow-ups (out of this plan's scope)

- **Dealer BDP** — only probed a private-seller listing this session.
  Re-run the probe against a real BT *dealer* BDP to validate the
  "Listed By" + "More From This Dealer" layout before changing.
- **Mobile** — `SCOPE.md` lists mobile as out-of-scope; skip.
- **Real ad creatives** — placeholder per scope.
- **Photo gallery real images** — placeholder per scope.
- **Map iframe in Location accordion** — sandbox has a placeholder div;
  real BT uses a Google Maps embed. Keep placeholder per scope, but if
  Location accordion is removed in Pass 3 the question is moot.
