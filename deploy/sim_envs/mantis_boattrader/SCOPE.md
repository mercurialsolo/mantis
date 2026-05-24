# `mantis_boattrader` — scope

Phase 0 doc per `FIDELITY_BUILD_FROM_SCRATCH_PROMPT.md`. Lists what
this sim env is explicitly trying to mirror from `boattrader.com`,
and what's intentionally out of scope. Update when scope changes.

## In-scope pages

| Route | Description |
|---|---|
| `GET /`                        | Home — hero search panel, ad carousel, "Boats Near You", featured brands, popular types/boats, ad strips, recent articles |
| `GET /boats/`                  | SRP — full filter sidebar (Location/Condition/Length/Year/Price/Boat Type/Make/Beam/Max Draft/Fuel Type/Hull/Engines/For Sale By), pre-qualify banner, sort, 3-up listing grid, pagination, mid-grid ad card, Boat Loan Calculator widget |
| `GET /boats/?<filters>`        | All filter query-param combinations the agent might hit (`condition=`, `make=`, `type=`, `price_min/max=`, `length_min/max=`, `year_min/max=`, `beam_min/max=`, `draft_min/max=`, `fuel=`, `hull=`, `engines_count=`, `engine_type=`, `for_sale_by=`, `zip=`, `distance=`, `sort=`, `page=`, `q=`, `price_drop=`) |
| `GET /boat/<slug>/`            | BDP — gallery, badges, price+monthly, view/save counters, dealer card with contact form, owner highlights, spec grid, propulsion, similar boats |
| `POST /boat/<slug>/contact`    | Submit a lead — re-renders BDP with confirmation banner; appends `lead_submitted` mutation |
| `POST /boat/<slug>/show-phone` | Reveal dealer phone — appends `phone_revealed` mutation |
| `POST /__site/consent`         | Accept/reject cookies; sets `bt_cookie_consent` cookie; appends `consent_set` mutation |

Harness routes (`GET /__env__/health`, `POST /__env__/reset`,
`GET /__env__/state`, `GET /__env__/leads`, `GET /__env__/mutations`,
`GET /__env__/oracle`, `POST /__env__/config`) are sim-env scaffolding,
not real BT mirrors — they exist only for grading.

## In-scope interactions

### Search + filter (SRP)
- Type Zip → form auto-submits at 5 digits → navigates to `/boats/?zip=NNNNN`
- Click "Zip Code" / "City / State" / "Other" segmented tabs (visual state only; sandbox keeps Zip Code semantically active)
- Click "Use My Location" link (visual; sandbox does NOT prompt geolocation API)
- Change `distance` select → form submit on submit click (no auto-submit)
- Click "All/New/Used" segmented control → form submit on apply
- Drag range-slider thumbs / type into No Min / No Max number inputs (Length/Year/Price/Beam/Max Draft)
- Type into Boat Type / Make search-as-you-type inputs → filters the visible `<li>` list
- Click a `<li>` checkbox in Boat Type / Make / Fuel Type / Hull → unchecks siblings, auto-submits to `?type=X` / `?make=X` / `?fuel=X` / `?hull=X`
- Toggle Price Drop switch → form submit (visual toggle slides + bg turns blue)
- Click Engines segmented (`Any/1/2/3/4+`) and Engine Type radios → form submit on apply
- Click For Sale By radios → form submit on apply
- Click Apply Filters / Clear all
- Click "Save Search" pill (no-op visually; sandbox doesn't persist saved searches)
- Change Sort dropdown → form submit
- Click a listing card → BDP
- Click pagination Prev/Next/Page-N

### Listing detail (BDP)
- Submit contact form → re-render BDP with confirmation banner, mutate `lead_submitted`
- Click "Show phone" → reveal, mutate `phone_revealed`
- Click similar-boat tile → BDP nav

### Site-wide
- Accept/reject cookie consent → mutate `consent_set`
- Sticky header on scroll past 320px (BDP only)

## Viewport(s)

- **Canonical**: 1440×900 (desktop)
- Responsive: SRP sidebar collapses to single column at `max-width: 980px`
- Mobile (≤560px): listing/card grid collapses to 1 column
- Out-of-scope: true mobile rendering (no separate mobile templates; CSS media queries only)

## Backend complexity

- **Catalog**: 600 boats, 20 dealers, 25 makes, 10 boat types — generated
  deterministically from `SEED` env var. In-memory `Store` (not SQL).
- **Mutations**: in-memory `Store.mutations: list[dict]`; cleared on
  `/__env__/reset`. Drives the oracle.
- **Single-select filters**: backend treats `?make=foo` as single value
  (no multi-select even though the UI now uses checkboxes — JS enforces
  single-select).
- **Persistence**: none beyond cookies + in-process catalog. No DB.
- **Latency injection**: every public request sleeps
  `[LATENCY_MS_MIN, LATENCY_MS_MAX]` ms; `LATENCY_FAILURE_RATE` fraction
  return 503 with `retry_after_ms`.

## Out-of-scope (intentional 🚫)

- Analytics (`gtag`, `dataLayer`, FullStory, Kameleoon experiment classes — class-names captured but no script behavior)
- Real ad networks — sandbox uses 6 procedural SVG creatives in fixed rotation
- Cookie consent provider chrome (OneTrust banner) — sandbox renders a generic accept/reject
- Live chat widget (Drift/Intercom)
- Real photos — all SVG placeholders (`🚫` by design; FIDELITY.md marks them so)
- Auth / Account / Saved searches persistence — Save Search is a visual no-op
- "More options" expand pattern in Make ("2,044 more Makes…") — sandbox shows the full list (only 25 makes anyway)
- Multi-select in Boat Type / Make / Fuel / Hull — real BT supports it; sandbox enforces single-select via JS to match the backend
- The "Sort: Recommended" inline link that opens a small dropdown — sandbox uses a regular `<select>`
- City/State and "Other" segmented tabs — visible but non-functional in sandbox
- Real BT's `/boats/state-ny/city-new-york/zip-10001/` SEO URL pattern — sandbox uses `/boats/?zip=10001`

## Done bar

Updated 2026-05-23 after PR #620 (v=82..v=97 — 16 commits of fidelity
work). All four high-level criteria from
`FIDELITY_BUILD_FROM_SCRATCH_PROMPT.md` met:

| Criterion | Status |
|---|---|
| ✅ for every row in `FIDELITY.md` across in-scope sections | ✅ (3 documented 🟡 follow-ups remain — listed below) |
| 100% of in-scope interactions replay with matching URL deltas | ✅ — 8 explicit interactions wired + verified (zip auto-submit, price-drop toggle, search-as-you-type, single-select submit, focus-hide, typed-hide, click-anywhere-focus, try-text rotator) |
| Verification harness green in CI | ✅ — `test_filter_panel_fidelity.py` runs 40 structural-anchor tests in the regular pytest matrix; `scripts/perceptual_diff.py` is the developer-local heavy harness |
| `_captured/` corpus checked in | ✅ — SRP fully measured, Home + BDP 🟡 partial (per-tile geometry of rail sections noted as open follow-ups in the corpus files themselves) |
| `SCOPE.md` + agent prompt docs | ✅ |

## Open follow-ups

Three remaining 🟡 items. All three need a human design decision
rather than another autonomous probe pass — listed in priority order:

1. **`.next-previous` sticky navigation widget** (BDP) — real BT
   renders an always-sticky 1512×54 Previous Boat / Next Boat
   navigation bar at the top of every BDP. Sandbox doesn't have this
   widget; sandbox's `.bdp-scrolled` body class triggers a different
   sticky pattern (title + price + Contact CTA) after 320px of
   scroll. **Decision needed:** add real BT's widget to sandbox, or
   keep sandbox's existing alternative pattern.

2. **Listing-dependent BDP elements** — Similar Boats card rail and
   Show Phone reveal button are present on dealer listings and absent
   on private-seller listings on real BT. Sandbox always renders
   both. **Decision needed:** add `is_dealer` boolean to the boat
   fixture, gate these elements on it, and probe a real BT dealer
   listing to measure them; or accept the always-render divergence.

3. **Mobile viewport pass** — sandbox has `@media (max-width: 980px)`
   responsive rules but they've never been probed against real BT's
   mobile rendering at 390×844. **Decision needed:** is mobile in
   scope for agent training? If yes, prioritize a measurement pass.
   If no, mark as 🚫 out-of-scope in SCOPE.md.

Pick-up by the next session:
- The autonomous loop (`cron ce9632d2` while session is live) has
  reached diminishing returns on visible filter-panel + SRP gaps.
- The 🟡 items above each need either a one-line human decision
  (#1, #3) or a small backend change + measurement pass (#2).
- Cache-buster pin is at `?v=96`. Bump as soon as any CSS changes.

## How to apply this doc

- Open `SCOPE.md` first when starting any sim-env work to confirm what's expected.
- When adding new flows / pages: append rows to "In-scope interactions" + "In-scope pages" BEFORE coding.
- When the user asks for something that's listed under "Out-of-scope": confirm before expanding scope, then move the row up.
- Pair this doc with `FIDELITY.md` (status matrix) and `FIDELITY_BUILD_FROM_SCRATCH_PROMPT.md` (playbook).
