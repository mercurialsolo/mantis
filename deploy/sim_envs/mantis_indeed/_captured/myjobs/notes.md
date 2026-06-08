# `/myjobs` — Saved + Applied tabs

## Layout

- Sticky top nav.
- H1 `My jobs`.
- Tab strip: `Saved` (default) | `Applied`. Active tab has brand-blue
  underline 2px.
- Tab body shows a list of job cards with the same card shape as the
  `/jobs` left list (slightly compacted — no snippet).
- Empty states:
  - Saved: `You haven't saved any jobs yet. Save jobs from search results
    to find them here.`
  - Applied: `You haven't applied to any jobs yet.`

## Interaction

- Click tab → `?tab=saved` or `?tab=applied`.
- Click a card title → `/viewjob?jk=<jk>`.
- Bookmark icon on saved card → toggles off → POST `/jobs/<jk>/save`
  removes the saved_jobs row.
- On Applied: each card shows status badge (New/Reviewed/Rejected/Hired)
  and `Applied 3 days ago` microcopy.
