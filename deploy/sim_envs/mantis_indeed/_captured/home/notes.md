# `/` — home

## Layout (1440×900 desktop)

- Sticky top nav, slim (~52px). White bg, 1px bottom border `#E4E2E0`.
- Top nav left: brand wordmark "indeed" lowercased in brand blue `#2557A7`,
  large size (`~32px` height including descender).
- Top nav right items (anonymous home): `Sign in` link, `Employers / Post Job`
  link. Each ~14px Indeed Sans.
- Below the top bar a secondary strip with: `Home`, `Company reviews`,
  `Find salaries`, `Employers`, `Create your resume`, `Change country
  🇺🇸 United States`. Underline on hover.
- Hero block ~280px tall, centred max-width ~1080px.
  - H1: blank in the literal sense — the hero composes from heading +
    search form. In the captured live snapshot the heading reads
    `Find your next job` (this exact phrase varies slightly across
    A/B variants; we pick this canonical wording).
  - Two-input search bar SIDE BY SIDE within a rounded border container,
    1px solid `#D4D2D0`, height 56px, separator between inputs.
    - Left: name=`q`, placeholder `Job title, keywords, or company`,
      aria-label `search: Job title, keywords, or company`,
      icon prefix 🔍 (Indeed actually renders a magnifier glyph).
    - Right: name=`l`, placeholder `City, state, zip code, or "remote"`,
      aria-label `Edit location`, icon prefix 📍.
    - CTA: blue button `Search`, brand blue bg `#2557A7`, white text,
      height 40px, border-radius 4px, padding 0 24px.
- Below hero, two panels side by side:
  - Left: `What's trending on Indeed` (H2 ~20px bold).
    - Bullet list of 4-6 trending searches as chip-style links.
  - Right: `Employer Resources` (H2 ~20px bold).
    - Bullet list of resource links.

## Typography

- Body font: `"Indeed Sans", "Noto Sans", "Helvetica Neue", Helvetica,
  Arial, "Liberation Sans", Roboto, Noto, sans-serif`.
- H1: 32px / 40px line-height, weight 700.
- H2: 20px / 28px line-height, weight 700.
- Body: 16px / 24px.
- Small body: 14px / 20px.

## Colour palette

- Brand blue: `#2557A7`. Hover-darken: `#164081`.
- Text primary: `#2D2D2D`.
- Text secondary: `#595959`.
- Border / dividers: `#D4D2D0`.
- Light grey bg surface: `#F3F2F1`.
- Link visited: same as primary (no purple).
- White: `#FFFFFF`.

## Interaction signals

- Tabbing through hero inputs uses focus-outline 2px solid `#2557A7`.
- Search submits to `/jobs?q=...&l=...` (GET form).
- The right input does NOT auto-submit on 5-digit zip in this canonical
  capture (see capture brief — convention only).

## Capture status

- Notes derived from `_capture_brief.md` § indeed + training-data
  recollection. Live Chrome MCP capture not attempted in this turn
  (Indeed presents Cloudflare interstitials anonymously); fidelity
  follow-up agents should run the per-element probe and update
  `dom.html` / `styles.json` / `screenshot.png`.
