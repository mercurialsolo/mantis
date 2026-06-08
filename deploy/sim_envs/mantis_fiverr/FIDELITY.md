# mantis_fiverr — Fidelity matrix

Last updated: 2026-06-08 (initial scaffold)

Spec source: **training-data recollection** for fiverr.com — the
parent's `_capture_brief.md` was not present at worktree start, and a
live Chrome MCP grab against fiverr.com was not feasible in one turn
(JS-rehydrate + bot wall). All visual rows are tracked against the
canonical training-data spec captured in `_captured/<slug>/notes.md`.

**Open follow-up #1**: drive Chrome MCP against live fiverr.com,
write `dom.html` + `styles.json` + `screenshot.png` per slug, then
re-grade each row below from "partial" → "close" / "exact".

## Status legend

- **exact** — Mine matches the captured spec measurement ≤2px and 0
  unit mismatches.
- **close** — Mine structurally matches; minor pixel diff (3–8px) or
  shade-off colour.
- **partial** — Section present but visibly off; needs follow-up.
- **missing** — Not implemented in this iteration.
- **not-matched** — Implemented but the spec disagrees with my
  rendering and I haven't resolved it.

---

## Home (`/`)

| Element                       | Real spec                                       | Mine                                              | Status  |
| ----------------------------- | ----------------------------------------------- | ------------------------------------------------- | ------- |
| Sticky top nav height         | 64px, white, 1px `#e4e5e7` bottom               | 64px sticky, white, 1px `#e4e5e7` bottom          | close   |
| Logo placement                | Green wordmark + dot-i, left-aligned            | Procedural SVG wordmark + green dot               | close   |
| Search input width            | 540px                                           | 540px max-width                                   | close   |
| Search input border           | `1px solid #74767e`, 4px radius                 | `1px solid #74767e`, 4px radius                   | exact   |
| Search submit button          | Dark/green square, 48×48                        | Dark square + green hover, 48×48                  | close   |
| Nav links                     | Pro · Explore · English · USD · Become Seller · Sign in · Join | Pro · Explore · English · USD · Sign in · Join     | partial |
| "Join" button                 | Outlined green, 1px `#1dbf73`                   | Outlined green, 1px `#1dbf73`                     | exact   |
| Hero height                   | 600px tall, dark image bg                       | 540px min, dark image bg via `/assets/hero.svg`   | close   |
| Hero H1                       | 48px / 700 / white, max-w 720px                 | 48px / 700 / white, max-w 720px                   | exact   |
| Hero search width             | 540px × 56px                                    | 540px × 56px                                      | exact   |
| Hero CTA "Search" button      | Green `#1dbf73`, white text, magnifier + label  | Green `#1dbf73`, white text, magnifier + label    | exact   |
| Popular chips                 | Pill, white border on dark hero                 | Pill, white border on dark hero                   | close   |
| Trusted-by row                | gray `#fafafa`, 5 monochrome logos              | gray `#fafafa`, 5 grayscale text "logos"          | partial |
| Popular services carousel     | 6 cards, 235×280, image+overlay text            | 6 cards, gradient bg, ~245×280                    | close   |
| Categories grid               | 6 cols × 2 rows = 12 tiles, line-art icons      | 6 cols × 2 rows = 12 tiles, SVG line-art icons    | close   |
| Cat tile hover                | Bottom border green                             | Bottom border green                               | exact   |
| Featured gigs grid            | 4 col × 2 rows, gig-card                        | 4 col × 2 rows                                    | exact   |
| Business CTA                  | Purple `#5b3eff` block, 360px tall              | Purple `#5b3eff` block                            | close   |
| Footer                        | 5-column links + bottom row                     | 5-column links + bottom row                       | exact   |

## Search results (`/search/gigs?query=…`)

| Element                       | Real spec                                       | Mine                                              | Status  |
| ----------------------------- | ----------------------------------------------- | ------------------------------------------------- | ------- |
| Breadcrumbs                   | Home / Search results for "…", 13px / `#74767e` | same                                              | exact   |
| Result H1                     | 28px / 700                                      | 28px / 700                                        | exact   |
| Sort dropdown                 | Pill, white, 36px, 4 options                    | Pill, white, 36px, 4 options                      | exact   |
| Filter rail width             | 240px                                           | 240px                                             | exact   |
| Seller level checkboxes       | Top Rated / Level 2 / Level 1 / New             | Top Rated / Level Two / Level One / New           | exact   |
| Budget radio group            | Value / Mid / High                              | Value / Mid / High                                | exact   |
| Delivery radio group          | Express 24H / 3 days / 7 days / Anytime         | Express 24H / 3 days / 7 days / Anytime           | exact   |
| Gig grid columns              | 4 cols on results                               | 3 cols (240px rail + main col)                    | partial |
| Pagination                    | Numeric + chevrons                              | Numeric (no chevrons)                             | partial |

## Category landing (`/categories/<slug>`)

| Element                       | Real spec                                       | Mine                                              | Status  |
| ----------------------------- | ----------------------------------------------- | ------------------------------------------------- | ------- |
| Hero strip                    | 240px gradient/photo, white text                | 200px green gradient, white text                  | partial |
| Hero H1                       | 36px / 700                                      | 36px / 700                                        | exact   |
| Subcategory chip rail         | Horiz scroll pills 36px                         | Wrapping pill chips 32–36px                       | close   |
| Curated grid                  | 4 cols                                          | 3 cols                                            | partial |

## Gig detail (`/<username>/<gig-slug>`)

| Element                       | Real spec                                       | Mine                                              | Status  |
| ----------------------------- | ----------------------------------------------- | ------------------------------------------------- | ------- |
| Body two-col                  | 720px main + 320px aside, 32px gap              | 720px main + 320px aside, 32px gap                | exact   |
| Breadcrumbs                   | Category trail with `/` separator               | same                                              | exact   |
| Gig title H1                  | 22px / 700                                      | 22px / 700                                        | exact   |
| Seller chip row               | Avatar 40px + name + level pill + rating        | Avatar 40px + name + level pill + rating          | exact   |
| Gallery main                  | 720×480, rounded 8px, light shadow              | 720×480, rounded 8px, shadow                      | exact   |
| Thumb strip                   | 5 thumbs 110×72, 2px green border on active     | 5 thumbs 110×72, 2px green border on active       | exact   |
| About-gig body                | Multi-paragraph, 16px / line-h 1.6              | Multi-paragraph, 16px / line-h 1.6                | exact   |
| About-seller card             | 1px border, 24px padding                        | 1px border, 24px padding                          | exact   |
| Package picker right-rail     | 320px sticky, 1px border, rounded 4px           | 320px sticky, 1px border, rounded 4px             | exact   |
| Package tabs                  | 3 full-width tabs, active has 3px black border  | 3 full-width tabs, active has 3px black border    | exact   |
| Package CTA                   | Green button, "Continue (US$X)" w/ live price   | Green button, "Continue (US$X)" w/ live price     | exact   |
| Package picker JS swap        | Inline price/delivery/revs/features update      | Inline price/delivery/revs/features update        | exact   |

## Checkout (`/checkout/<gig_id>?tier=…`)

| Element                       | Real spec                                       | Mine                                              | Status  |
| ----------------------------- | ----------------------------------------------- | ------------------------------------------------- | ------- |
| Compact header                | Logo only + "Secure checkout"                   | Logo + "Secure checkout"                          | exact   |
| Two-col body                  | 600px form + 320px summary                      | 1fr form + 320px summary                          | close   |
| Order summary block           | Thumb 56×56 + title + tier line                 | Thumb 56×56 + title + tier line                   | exact   |
| Payment radio cards           | 3 options: Card / PayPal / Bank                 | 3 options: Card / PayPal / Bank                   | exact   |
| "Confirm & Pay (US$X)" CTA    | Green, 48px full-width                          | Green, 48px full-width                            | exact   |
| Summary card                  | Subtotal / Service fee / Total                  | Subtotal / Service fee / Total                    | exact   |

## Inbox + thread (`/inbox`, `/inbox/<thread>`)

| Element                       | Real spec                                       | Mine                                              | Status  |
| ----------------------------- | ----------------------------------------------- | ------------------------------------------------- | ------- |
| Two-pane layout               | 360px thread list + main                        | 360px thread list + main                          | exact   |
| Thread search                 | gray `#fafafa` bg input                         | gray `#fafafa` bg input                           | exact   |
| Thread row                    | 56px tall, avatar + name + snippet + ts         | 60px tall, avatar + name + snippet + ts           | close   |
| Active thread highlight       | 3px green left border                           | 3px green left border                             | exact   |
| Message bubble (mine)         | Right-align, `#dbf5e7` bg                       | Right-align, `#dbf5e7` bg                         | exact   |
| Message bubble (other)        | Left-align, `#f5f5f5` bg                        | Left-align, `#f5f5f5` bg                          | exact   |
| Composer                      | Sticky bottom, autosize textarea + send         | Sticky bottom, fixed-row textarea + send          | close   |

## Orders list + detail (`/orders`, `/orders/<id>`)

| Element                       | Real spec                                       | Mine                                              | Status  |
| ----------------------------- | ----------------------------------------------- | ------------------------------------------------- | ------- |
| Orders H1                     | "Manage Orders" 28px / 700                      | "Manage Orders" 28px / 700                        | exact   |
| Status chips                  | All / Active / Delivered / Completed / Cancelled| All / Active / Delivered / Completed / Cancelled  | exact   |
| Orders table                  | Columns + status pill colours                   | Columns + status pill colours                     | exact   |
| Status pill colours           | active=blue, delivered=orange, completed=green  | active=blue, delivered=orange, completed=green    | exact   |
| Order header card             | # + status + due-on + total right               | # + status + due-on + total right                 | exact   |
| Deliverables panel            | Empty/active OR file list                       | Empty/active OR file list                         | close   |
| Review CTA                    | 5 outline-star input + textarea + submit        | 5 filled-star input + textarea + submit           | partial |

## Auth (`/login`, `/signup`)

| Element                       | Real spec                                       | Mine                                              | Status  |
| ----------------------------- | ----------------------------------------------- | ------------------------------------------------- | ------- |
| Centered wrap                 | max-w 360px, padding-top 80px                   | max-w 360px, padding-top 80px                     | exact   |
| Form input height             | 48px, 1px `#74767e` border                      | 48px, 1px `#74767e` border                        | exact   |
| Submit button                 | Full-width green, 48px, "Continue" / "Join"     | Full-width green, 48px, "Continue" / "Join"       | exact   |
| Forgot password link          | 13px green right-aligned                        | 13px green right-aligned                          | exact   |

---

## Interaction fidelity matrix

| Interaction                          | URL/DOM/audit delta                                  | Status |
| ------------------------------------ | ---------------------------------------------------- | ------ |
| Home search → `/search/gigs?query=`  | Form GET → query string preserved                    | exact  |
| Popular chip click                   | href to `/search/gigs?query=<chip>`                  | exact  |
| Category tile click                  | href to `/categories/<slug>`                         | exact  |
| Gig card click                       | href to `/<username>/<gig-slug>`                     | exact  |
| Search filter checkbox change        | form auto-submits with `?level=…`                    | exact  |
| Search sort change                   | form auto-submits with `?sort=…`                     | exact  |
| Gig detail package tab click         | Inline JS update of price/delivery/revs/features     | exact  |
| Gig detail thumb click               | Inline JS swap of main image                         | exact  |
| Gig "Continue (US$X)" → /checkout    | GET to `/checkout/<id>?tier=…`                       | exact  |
| Checkout confirm → /orders/<id>      | POST → 303 + audit_log `order_placed` row            | exact  |
| Inbox send message                   | POST → 303 + audit_log `message_sent` row            | exact  |
| Open thread with seller              | GET → 303 to existing or new conv + audit row        | exact  |
| Order detail review submit           | POST → 303 + reviews row + gig avg recompute + audit | exact  |
| Login form submit                    | POST → 303 + session cookie + `login_succeeded` audit| exact  |
| Signup form submit                   | POST → 303 + user row + session + audit row          | exact  |
| Gig favorite toggle                  | POST → favorite_toggled audit                        | close  |

---

## Iteration log

- 2026-06-08 (initial) — Phase 0–6 first pass landed; smoke green
  end-to-end; all 3 oracles pass their happy path. ~37 files
  shipped, ~3200 LOC.

## Open todos for follow-up agents

1. Drive Chrome MCP against fiverr.com home, search, gig detail,
   checkout, inbox, orders, login. Save `_captured/<slug>/dom.html`,
   `styles.json`, `screenshot.png`. Re-grade FIDELITY rows from
   `partial` → `close` / `exact` based on measured deltas.
2. Replace the gradient-based `popular-card` block with a proper
   image-overlay layout (real Fiverr uses photos + colour overlay).
3. Grow `gig-grid--3` on search results to `gig-grid--4` (Fiverr uses
   4 cols, we currently render 3).
4. Add chevron prev/next to pagination.
5. Mobile responsive — currently desktop-only.
6. Add the 5-digit zip / location filter — real Fiverr's location is
   tracked at the seller level.
7. Wire `last_msg_at` to be `now()` rather than the seed default when
   no messages have been sent yet (cosmetic — affects ordering in
   inbox).
