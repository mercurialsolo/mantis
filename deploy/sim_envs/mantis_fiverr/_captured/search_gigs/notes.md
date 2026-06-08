# /search/gigs?query=… — Fiverr search results

## Layout

Same sticky top nav. Body wraps in `max-width: 1400px`, padding `0
80px`.

### Breadcrumbs row (28px tall)
`Home / Search results for "<query>"` — 13px / 400 / `#74767e`,
slash separator `#95979d`.

### Result header
- H1: `Results for "<query>"` — 28px / 700 / `#222325`
- "X results" subtitle 14px / `#74767e`
- Right side: **Sort by ▼** dropdown — pill, white bg, 1px
  `#e4e5e7` border, 36px tall, label "Sort by: Relevance" 13px / 600.
  Options: Relevance · Best Selling · Newest Arrivals · Average rating

### Filter rail (sticky, left, 240px wide)

Each filter group: label 14px / 700 bold, options 14px / 400.

1. **Seller details**
   - Seller level checkbox group: `Top Rated Seller`, `Level Two`,
     `Level One`, `New Seller`
2. **Budget**
   - `Value · Up to US$50`, `Mid-range · US$50-200`, `High-end · US$200+`
3. **Delivery time**
   - Radio group: `Express 24H`, `Up to 3 days`, `Up to 7 days`, `Anytime`

Filters update `?level=`, `?budget=`, `?delivery=` query params on
click (no page reload via small JS handler that sets `location.search`).

### Result grid

Right of filter rail. Same gig card shape as home (4 columns × N rows,
282×340 each, 24px gap). Pagination at bottom: numeric pages 1 2 3 …
N, prev/next chevrons.

## Interactions captured

- Sort dropdown change → `?sort=<value>` reload
- Filter checkbox toggle → `?level=top_rated&level=level_two&…`
- Card click → `/<username>/<gig-slug>`
- Pagination click → `?page=N`
