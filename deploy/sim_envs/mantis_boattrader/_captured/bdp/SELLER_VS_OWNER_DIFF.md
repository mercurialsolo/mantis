# BDP visual/perceptual compare — dealer vs owner vs real BT

Probed 2026-05-23 at 1512×711 viewport.

## Listings probed

| Source | Type | URL |
|---|---|---|
| Real BT private | private seller | `/boat/2003-president-performance-trawler-9856489/` |
| Real BT private | private seller | `/boat/2015-pioneer-197-sportfish-10079002/` |
| Real BT (no dealer found) | n/a | Real BT's first-page listings under `?for-sale-by=dealer` all rendered as private (Meet Your Seller + first-name pattern). Couldn't isolate a "Contact Dealer" pattern in this probe pass — real BT may have folded dealer + private under the same "Meet Your Seller" UI. |
| Sandbox dealer | dealer | `/boat/2021-crestliner-vt-18-10000396/` |
| Sandbox owner | owner | `/boat/2025-chaparral-267-ssx-10000008/` |

## Heading hierarchy side-by-side

| Section | Real BT private | Sandbox dealer (v=103) | Sandbox owner (v=103) | Status |
|---|---|---|---|---|
| H1 page title | "2003 President Performance Trawler" (no length) | "2021 Crestliner VT 18 \| 16'" (length suffix) | "2025 Chaparral 267 SSX \| 27'" (length suffix) | 🟡 → ✅ (v=104 removes the length suffix) |
| Seller intro | H2 "Meet Your Seller" + H3 "Bill Adams" + H3 "Contact Bill Adams" | H2 "Contact Elena Patel" | H2 "Contact Ravi, the owner" | 🟡 → ✅ (v=104 adds H2 "Meet Your Seller" above the contact card) |
| Sparkle prefix on "What Owners Say" | plain "What Owners Sayinfo" | ✦ + ⓘ inline | ✦ + ⓘ inline | 🚫 sandbox decorative addition (kept) |
| H2 Owner Highlights | yes 20/700 | yes 20/700 (v=91) | yes 20/700 | ✅ |
| H2 Boat Details | yes 20/700 | yes 20/700 | yes 20/700 | ✅ |
| H3 Description | yes 16/700 | yes 16/700 | yes 16/700 | ✅ |
| H3 Key Features | not visible on real BT | present | present | 🟡 — sandbox extra subhead (kept; could be conditional) |
| H3 Measurements | yes 16/700 | yes 16/700 | yes 16/700 | ✅ |
| H4 Dimensions / Weights / Tanks | yes 14/700 | yes 14/700 | yes 14/700 | ✅ |
| H3 Propulsion | yes 16/700 | yes 16/700 | yes 16/700 | ✅ |
| H3 More Details | not visible on real BT (couldn't verify if present) | present | present | 🟡 |
| H3 Location | not visible on real BT (couldn't verify if present) | present | present | 🟡 |
| "Still have a question?" | H3 24/700 | H2 (sandbox `question-card-h2`) | H2 | 🟡 — different tag, same visual prominence |
| Show Phone button | dealer-only (not visible on private listings) | always shown | always shown | 🟡 (open follow-up per SCOPE.md — needs `is_dealer` gating) |

## Layout invariants (verified ✅)

- Content width: 1319px (real BT) vs 1336px (sandbox `.bdp-grid max-width`) ✅
- Gallery container: 890×727 at x=70 ✅
- Right rail starts at x=1039, width 334 ✅
- H1 in right rail: 248×23 at y=263, 20/700 #333 ✅
- Main price `$25,900` at y=322, 20px ✅
- Monthly est `$4,000` inline-right of price at x=1132, 18px ✅
- Contact form `.lead-form-basic__body` 334×260 at x=1055 ✅
- Breadcrumb "Home / Boats For Sale / Power / Center Console / Pioneer" 12/400 #616161 at y=191 ✅
- Thumbnail strip 175×116 (auto-scaled via .bdp-grid max-width capping in v=91) ✅
- Accordion heading levels (H2 / H3 / H4 distribution) ✅

## Dealer vs owner divergences (sandbox-internal)

Sandbox renders these differently per `boat.is_owner_listed`:

| Element | Owner listing | Dealer listing |
|---|---|---|
| Card class | `.bdp-private-seller-card` with `.private-seller-tag` "Private Seller" | `.bdp-dealer-card` |
| Contact heading | "Contact `<owner_name>`, the owner" | "Contact `<salesperson>`" (rotates through 6 names) |
| Disclaimer | "This boat is being sold directly by the owner. Boat Trader does not verify private-seller listings — please ask for a survey, title, and registration before purchase." | (none) |
| Show Phone | gated by cookie `bt_show_phone_<id>`; opens a phone-svg reveal | (button always shown — no real dealer phone surface yet) |
| Dealer logo in similar-boats footer | hidden (`{% if not boat.is_owner_listed %}`) | shown |

## v=104 fixes applied

1. ✅ Removed H1 length suffix `<span class="bdp-length">| 16'</span>` from `boat_detail.html:96`. H1 now matches real BT format: "Year Make Model".
2. ✅ Added `<h2 class="meet-your-seller-h2">Meet Your Seller</h2>` above the contact card block — appears for both owner and dealer paths, matching real BT's pattern.

## Remaining 🟡 (need user input)

1. **Show Phone gating** — sandbox always shows on dealer cards too. Real BT only shows on dealer listings. Needs `is_dealer` flag on dealer card OR remove from dealer path entirely.
2. **Extra H3s "Key Features" / "More Details" / "Location"** — sandbox shows them on every BDP. Real BT doesn't show these specific headings on probed listings. Could keep (sandbox affordance) or hide.
3. **"Still have a question?" H2 vs H3** — sandbox uses H2 with a custom class. Real BT uses H3 24/700. Semantic-only diff; not visible.
4. **Real BT dealer-listing pattern not isolated** — couldn't find a "Contact Dealer" page in this probe; real BT may have unified dealer + private under "Meet Your Seller". Worth a separate probe pass against a known-dealer URL when one surfaces.
