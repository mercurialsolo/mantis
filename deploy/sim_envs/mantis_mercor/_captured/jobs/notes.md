# `/jobs` — Open roles list

Captured from `/experts/` shape in capture brief (similar list
surface). Direct `/jobs` re-capture flagged ⏳ in FIDELITY.md.

## Layout

- Slim sticky header (same as home).
- Left filter rail (~260px wide):
    - "Category" multi-select chips (Medical, Legal, Finance,
      Software, Consulting, Office)
    - "Rate" range filter (min/max numeric inputs)
    - "Engagement type" radio (Hourly / Project / Full-time)
- Right results pane (~880px wide), 3-column card grid.
- Cards use the canonical role-card component (see home/notes.md).
- Sort dropdown top-right of results pane: `Sort by: Latest`.

## Interaction triggers

- Card click → `GET /jobs/<id>`.
- Apply button on card → `GET /apply/<id>` (skipping detail page).
- Filter chip toggle → updates URL query params (`?category=Medical`).

## Mirror priorities

- 3-col grid at desktop, stacking responsive at narrow.
- Filter rail labels must match: "Category", "Rate", "Engagement
  type".
- Sort label: `Sort by: Latest`.
