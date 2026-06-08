# `/jobs?q=software+engineer&l=Austin%2C+TX` — search results

## Layout (1440×900 desktop)

- Sticky top nav same as `/`.
- Just below: a secondary search bar that mirrors hero (compact 40px
  height) so the agent can re-search without scrolling up. Two inputs +
  Search CTA. Pre-fills with `q` + `l`.
- Underneath the secondary search: H1 `software engineer jobs in Austin,
  TX` — lowercase echo of query + location. font-size 20px / weight 700.
- Filter chip rail beneath the H1. 11 chips, horizontal, scrollable
  overflow on narrow viewports. Exact order:
  1. `Date posted`
  2. `Remote`
  3. `Developer skill`
  4. `Job Type`
  5. `Experience level`
  6. `Pay`
  7. `Education`
  8. `Clearance type`
  9. `Developer type`
  10. `Compensation package`
  11. `Distance`
  - Each chip ~32px tall, 1px border `#D4D2D0`, padding 6px 12px,
    border-radius 16px, font 14px / weight 400. Active chip: brand blue
    border + `#E4F1FF` bg.
- Sub-strip with sort dropdown `Sort by: relevance` (right-aligned),
  and result-count caption left: `1-15 of 1,234 jobs`.

## 3-pane body

- Container max-width 1440px. Two columns: results list (480px) +
  detail pane (680px). Combined `grid-template-columns: 480px 680px`
  with 20px gap, centred.
- **Left list — result card**:
  - 1 px border `#D4D2D0`, padding 16px, border-radius 8px, gap 12px
    between cards, white bg. Selected card bg `#E4F1FF`, 2px brand-blue
    left-border accent (4px wide actually) on the selected card.
  - Inner shape:
    - Row 1: title `h2.jobTitle` 18px bold brand blue (acts as link).
    - Row 2: company name in `#2D2D2D` 14px + rating stars + review
      count `(N reviews)` 12px grey.
    - Row 3: location 14px `#595959`.
    - Row 4: salary range `$130,000 - $170,000 a year` 14px `#2D2D2D`.
    - Row 5: snippet 14px line-clamped to 3 lines `#595959`.
    - Row 6: footer with posted-date `Posted 3 days ago` 12px grey +
      bookmark-icon button right-aligned (filled when saved).
  - Card is wholly clickable — clicking anywhere selects it (right pane
    re-renders, URL gets `?vjk=<jk>` appended).
- **Right detail pane**:
  - 680px wide, sticky-positioned top:60px so it stays visible as the
    user scrolls the left list.
  - Header block:
    - Big title H2 `{Job Title} - job post` (the suffix is canonical) —
      24px bold.
    - Company name 16px + location 14px grey.
    - `Apply now` primary CTA button (blue, 40px tall, white text) +
      `Save job` secondary outline button.
  - Tab strip: `Job details`, `Company`, `Reviews`. Default `Job details`.
  - Body: salary range strip with `$130K - $170K a year` + remote flag
    chip + job-type chip; full description (10 paragraphs); benefits
    bullets; apply-CTA footer.

## Interaction signals

- Clicking a left card → JS toggles `data-selected="true"` on that
  card + clears it on siblings; updates `?vjk=<jk>` in URL via
  `history.replaceState`; XHR/fetch GETs `/jobs/_detail?jk=<jk>` and
  swaps the right pane HTML.
- Clicking the bookmark icon → fetch POST `/jobs/<jk>/save` (toggle);
  icon swaps optimistically.
- Clicking `Apply now` → navigate to `/apply/<jk>`.
- Clicking the title link → navigate to `/viewjob?jk=<jk>`.
- Filter chip click → opens a popover with options; selecting one
  updates URL params and reloads.

## Capture status

- Notes derived from `_capture_brief.md`. Indeed's anonymous traffic
  faces Cloudflare; live MCP capture not attempted this turn. Follow-up
  agents should run the per-element probe + capture `dom.html`,
  `styles.json`, `screenshot.png`.
