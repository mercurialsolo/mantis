# `/jobs/<id>` — Role detail page

## Layout

- Slim sticky header.
- Two-column body:
    - Main (~880px): role title H1, rate range, "N hired recently",
      Description (Markdown rendered), Skills required (chip list),
      Screening questions preview list.
    - Right rail (~320px sticky): summary card (rate, engagement type,
      hours/week), big `Apply` button (indigo-600 bg, white text).

## Interaction triggers

- Apply CTA → `GET /apply/<id>` (or `POST /apply/<id>/start` if logged
  out, redirects to login).

## Mirror priorities

- The right-rail summary card sticky behaviour matters for layout
  realism; can be regular block in v1 (no sticky JS).
- H1 must be the role title (no marketing tagline above).
