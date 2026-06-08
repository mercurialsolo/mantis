# /<seller-username>/<gig-slug> — Fiverr gig detail

The most interaction-heavy page in scope. The package picker is the
key interaction-fidelity surface.

## Layout

Two-column body: left content 720px wide, right sticky package picker
panel 320px wide, gap 32px. Wraps in `max-width: 1108px`, padding `0
80px`.

### Breadcrumbs row
`Home / <category> / <subcategory> / <gig title>` — 13px / 400 /
`#74767e`.

### H1 gig title

22px / 700 / `#222325`, 2 lines max.

### Seller chip row (below title)
- 40px circular avatar + seller name (14px / 600) + Level badge
  `Level 2 Seller` pill (orange `#ff7640` bg, white text, 11px / 600,
  rounded 2px) + star rating with `5.0 (124)` reviews count + green
  contact button "Contact me"

### Gallery (left)
- Main image 720×480, rounded 8px, light shadow
- Thumb strip below: 5 thumbs 110×72 each, gap 8px, selected one has
  2px `#1dbf73` border, others 1px `#e4e5e7`

### "About this gig" section
H3 "About this gig" 20px / 700. Body paragraphs 16px / 400, line-height
1.6, color `#222325`. Multi-paragraph plain text.

### "About the seller" card
Boxed `#e4e5e7` 1px border, padding 24px, rounded 4px. 80px avatar +
seller details: name, location, member since, response time, last
delivery. CTA "Contact me" outlined green button.

### Reviews section
H3 "Reviews" + star summary "5.0 (124)". Filter dropdowns by rating.
List of review cards: avatar, name, country flag, rating, date, body
text.

## Right sticky package picker

Card 320px wide, sticky-top with offset 88px (under nav). Border 1px
`#e4e5e7`, rounded 4px, no shadow.

### Package tabs (top of card)

Three tabs full-width: `Basic` · `Standard` · `Premium`. Active tab:
black bottom border 3px, text `#222325` / 600. Inactive: text
`#74767e` / 400.

### Tab body

- Top row: package title (16px / 700) ← left + price (24px / 700) ← right
  `US$<price>` with `$` superscript-ish
- Description: 14px / 400 / `#74767e` (2 lines)
- Features list (4-6 items):
  - Each row: checkmark icon (green `#1dbf73`) + label 14px
  - Examples: `3-day delivery`, `2 Revisions`, `Source file`, `Logo
    transparency`, `Vector file`, `Printable file`
- Delivery row: clock icon + "3-day delivery" 14px / 600
- Revisions row: refresh icon + "2 Revisions" 14px / 600
- CTA: solid green button `#1dbf73`, full-width, 48px tall, white text
  16px / 600, text `Continue (US$<price>)`
- Below CTA: secondary outline "Compare packages" 14px / 600

## Interactions captured

- Clicking a package tab swaps active state, updates price, features,
  delivery, revisions, and CTA price text **inline, no nav** —
  small JS handler reading from `data-package-N-*` attributes on the
  container.
- Clicking "Continue (US$X)" → `/checkout/<gig_id>?tier=<basic|std|prem>`
- Thumb-strip click on gallery swaps main image, no nav (small JS)
- "Contact me" → `/inbox/<thread_id>` if existing thread else
  `/inbox/new?to=<seller>`
- Heart icon top-right of gallery toggles favorite (POST
  `/gig/<id>/favorite`, audit_log row)
