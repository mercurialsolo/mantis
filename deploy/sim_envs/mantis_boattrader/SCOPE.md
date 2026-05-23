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

- ✅ for every row in `FIDELITY.md` across all in-scope sections (or 🟡 / 🚫 with a one-line reason)
- 100% of in-scope interactions replay against sandbox with matching URL deltas (DOM/network/timing tracked in FIDELITY.md when divergent)
- `scripts/perceptual_diff.py` runs green for the regions it covers (filter-panel today; SRP listing card + BDP planned)
- Sandbox is reachable via the Daytona preview URL with the current token in `FIDELITY.md`

## Open follow-ups

- Wire `perceptual_diff.py` into CI as a regression gate (currently scaffolded but not invoked)
- Capture the `_captured/` corpus checked-in for every page (currently only in agent memory + FIDELITY.md rows)
- Mobile viewport pass (currently desktop-only verified)
- Sticky-header on SRP scroll (BDP-only today)

## How to apply this doc

- Open `SCOPE.md` first when starting any sim-env work to confirm what's expected.
- When adding new flows / pages: append rows to "In-scope interactions" + "In-scope pages" BEFORE coding.
- When the user asks for something that's listed under "Out-of-scope": confirm before expanding scope, then move the row up.
- Pair this doc with `FIDELITY.md` (status matrix) and `FIDELITY_BUILD_FROM_SCRATCH_PROMPT.md` (playbook).
