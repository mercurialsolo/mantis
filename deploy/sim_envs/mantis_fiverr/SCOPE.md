# mantis_fiverr — SCOPE

Synthetic high-fidelity mirror of **fiverr.com** — freelance gig
marketplace. Sellers list "gigs" with three package tiers
(Basic / Standard / Premium); buyers browse, order, message, and
review.

## In-scope pages

| URL pattern                             | Surface                                 |
| --------------------------------------- | --------------------------------------- |
| `/`                                     | Marketing home — search bar, popular categories carousel, featured gig grid |
| `/search/gigs?query=`                   | Search results — filter rail + gig card grid + sort dropdown |
| `/categories/<slug>`                    | Category landing — subcategories + curated grid |
| `/<seller-username>/<gig-slug>`         | Gig detail — gallery, packages picker, description, seller card, reviews |
| `/checkout/<gig-id>?tier=`              | Package + add-ons confirm + payment placeholder |
| `/inbox`                                | Conversations list |
| `/inbox/<thread>`                       | Thread detail |
| `/orders`                               | Buyer order list |
| `/orders/<id>`                          | Order detail — status, deliverables, messages, review CTA |
| `/login`, `/signup`                     | Auth |

## In-scope interactions

- **Search**: type query → `/search/gigs?query=…`. Sort dropdown +
  filter checkboxes update `?sort=` and `?level=` query params.
- **Gig detail**: switch Basic/Standard/Premium tabs → price + delivery
  time + revision count + features update inline (vanilla JS, no nav).
  "Continue ($X)" CTA reflects current tier price.
- **Checkout**: confirm → order row + `audit_log` row + redirect to
  `/orders/<id>`.
- **Messaging**: send a message in a thread → message row + thread
  `updated_at` bumps.
- **Review**: leave star rating + text on a completed order → review row
  + gig avg-rating recompute.

## Out-of-scope (NOT mirrored)

- Real photography / brand assets → deterministic SVG placeholders
- Analytics / advertising / tracking scripts → stripped
- Third-party widgets (intercom, hotjar, FullStory) → stripped
- OAuth / social login → username + password only
- Real payment processing → "Pay" button writes audit_log + flips
  order status. No Stripe.
- Real-time websockets / SSE → polling or static
- A/B variants → one canonical variant per page (the anonymous-default)
- Pro / Business / Logo Maker upsells — out
- Currency switcher → USD only
- Locale switcher → en-US only

## Viewports

Desktop 1440×900 only. Mobile responsive is out-of-scope for this
iteration.

## Backend

SQLite, deterministic seed. Read+write surface. Every state-changing
route writes an `audit_log` row. Oracles read `audit_log`, not derived
state.

## Done bar

- ✅ Structural deltas ≤2px per FIDELITY.md row
- ✅ 100% in-scope interactions replay with matching URL+DOM+audit_log
- ✅ `scripts/smoke.py` green
- ✅ `_captured/` corpus checked in
- ✅ All four bookkeeping files present (SCOPE, FIDELITY,
  FIDELITY_AGENT_PROMPT, README)

## Oracle tasks

- **t01_order_basic_logo** — buyer orders gig `gig_00001` Basic tier;
  oracle verifies order row total = Basic price with line items
- **t02_message_seller_then_order** — buyer sends a message in a thread,
  then orders; oracle verifies conversation + order linkage
- **t03_leave_5star_review** — buyer leaves a 5-star review on completed
  `order_00007`; oracle verifies review row + gig avg-rating update
