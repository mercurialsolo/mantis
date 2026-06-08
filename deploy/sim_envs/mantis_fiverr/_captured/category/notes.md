# /categories/<slug> — Fiverr category landing

## Layout
Max-width 1400px, padding 0 80px.

### Hero strip
- 240px tall, gradient or photo, dark overlay
- H1 white "Graphics & Design" 36px / 700
- Subtitle 16px / `#ffffff` opacity .8

### Subcategory chip rail
Horizontal scroll, 12 chips. Each chip = pill, 1px `#e4e5e7` border,
14px / 600, height 36px, padding 0 16px. Examples for Graphics &
Design: `Logo Design`, `Brand Style Guide`, `Business Cards &
Stationery`, `Illustration`, `Web & Mobile Design`, …

### Curated grid
H2 "Most popular in <category>" 24px / 700. Same gig-card grid as
search results (4 cols).

## Interactions
- Subcategory chip click → `/search/gigs?query=<slug>&category=<slug>`
- Card click → gig detail
