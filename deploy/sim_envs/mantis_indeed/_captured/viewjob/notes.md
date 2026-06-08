# `/viewjob?jk=<id>` — full job detail page

## Layout

- Sticky top nav.
- Container max-width 1024px, centred.
- Job header card (white bg, 1px border `#D4D2D0`, padding 24px,
  border-radius 8px):
  - H1 `{Job Title} - job post` (same canonical suffix).
  - Company name + location row.
  - Salary range chip + job-type chip + remote chip.
  - `Apply now` blue primary CTA + `Save job` outline.
- Tab strip: `Job details`, `Company`, `Reviews`.
- Body: salary strip, benefits, full description, requirements, footer
  CTA.

## Interaction

- `Apply now` → `/apply/<jk>`.
- `Save job` → POST `/jobs/<jk>/save` toggle.

Same visual rules as the right detail pane on `/jobs`. Reused template
fragment.
