# Captured corpus — mantis-linkedin

This directory holds the ground-truth spec for each in-scope page of the
live linkedin.com surface. Each slug is built from offline notes
(training-data recollection) because LinkedIn requires auth and
challenges automated browsers on most surfaces, so live MCP-driven
captures were not viable.

When a Chrome-MCP–driven capture pass is later possible (e.g. via a
human-supplied authenticated session), regenerate each slug's
`dom.html`/`styles.json`/`screenshot.png` and update FIDELITY.md.

## Per-slug contents

- `notes.md` — page layout, typography, palette, key element labels, interactions
- `styles.json` — per-element computed-style observations (offline)
- `dom.html` — best-effort structural HTML (offline)
- `screenshot.png` — NOT captured (auth-gated). See `notes.md` for the visual spec.

## Brand spec (canonical, applied site-wide)

- Primary palette
  - LinkedIn blue: `#0a66c2` (button, link, brand)
  - Hover blue: `#004182`
  - Text near-black: `#000000e6` (rgba 0,0,0,0.9)
  - Text grey: `#00000099` (rgba 0,0,0,0.6)
  - Surface white: `#ffffff`
  - Page bg: `#f3f2ef`
  - Card border: `#e0dfdc` (1px solid #00000022)
  - Success green: `#057642`
  - Danger red: `#cc1016`
- Font stack: `-apple-system, system-ui, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif`
- Type scale
  - Display H1: 24px / 600
  - Card H2: 20px / 600
  - Body: 14px / 400, line-height 1.42857
  - Caption: 12px / 400, color rgba(0,0,0,0.6)
- Layout
  - Global top nav: 52px fixed, 1128px content max-width
  - Card radius: 8px; box-shadow `0 0 0 1px rgb(0 0 0 / 8%), 0 2px 3px rgb(0 0 0 / 8%)`
  - Three-column grid on /feed/: 225px / 555px / 300px (gaps 24px)
