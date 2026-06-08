# / (home) — Fiverr marketing home

## Top nav (sticky, white, 1px bottom border `#e4e5e7`, height 64px)

Left → right:
- Fiverr wordmark logo (29px tall, green `#1dbf73`, dot over the
  "i" is a slight rotated comma) — links to `/`
- Search bar: 540px wide, 48px tall, rounded 4px, `1px solid #74767e`,
  placeholder "Search for any service…", inline green search button
  (square, 48×48, white magnifier icon)
- Right nav items (gap 24px, font 16px / 600):
  Fiverr Pro · Explore · English (▼) · US$ USD (▼) · Become a Seller ·
  Sign in (text) · Join (outlined green button, padding 6px 16px,
  rounded 4px, border 1px `#1dbf73`, text green)

## Hero (full-bleed dark image, 600px tall)

- Title H1: "Find the right freelance service, right away" — white,
  44px, 700, max-width 720px, left-aligned, padding-left 80px
- Search composite, 540px wide × 56px tall, white bg, rounded 4px,
  inset 1px shadow: placeholder "Search for any service…" + green
  submit button "[magnifier] Search" 100px wide, 56px tall,
  border-radius 0 4px 4px 0
- "Popular:" chips row beneath search: chip pills, white border,
  transparent bg, white text. Default chips: `Website Design`,
  `WordPress`, `Logo Design`, `AI Services`

## Trusted by row

Centered grayscale logo strip "Trusted by:" + logo placeholders
(Meta, Google, Netflix, P&G, PayPal). 80px tall, `#fafafa` bg.

## Popular services carousel

Section H2 "Popular services", 32px / 700 / `#222325`.
6 wide cards, horizontally scrollable:
- 235px × 280px each, gap 16px
- Card: gradient/photo top (170px), title bottom (60px, white text on
  card-color block), e.g. "AI Artists | Add Promise" gradient blue,
  "Logo Design | Build your brand", "WordPress | Customize your site",
  "Voice Over | Share your message", "Video Editing | Bring your story
  to life", "Social Media | Reach more customers"

## "You need it, we've got it" — category tiles grid

H2 32px / 700. 6 columns × 2 rows = 12 tiles. Each tile 188×112:
- Square icon top (48px, line-art, green), label below 14px / 600
- Examples: Graphics & Design, Programming & Tech, Digital Marketing,
  Video & Animation, Writing & Translation, Music & Audio, Business,
  Data, Photography, AI Services, Lifestyle, Consulting

## Featured gigs grid

H2 "Recommended for you" 32px / 700. Grid 4 columns × 2 rows = 8 cards.
Each gig card 282×340:
- Image area 282×190 (4:3 ratio approximation) — solid color/gradient placeholder
- Heart icon top-right (favorite)
- Seller row: 26px avatar circle + seller name (14px / 600) + Level
  badge (small pill text "Level 2 Seller" 11px / 600 / `#95979d`)
- Title: 14px / 400 / `#222325`, 2 lines clamp, "I will <verb>
  <object>" sentence case
- Star row: filled yellow stars 12px + rating digit "5.0" 12px / 600 +
  "(reviews_count)" 12px / `#74767e`
- Bottom row: heart icon (left if not pinned) + "Starting at US$X"
  right-aligned, 13px / 600

## Fiverr Business CTA block

Dark purple `#5b3eff` block, full-width, 360px tall. Left: H2 white "A
whole world of freelance talent at your fingertips". Sub-copy 16px.
CTA white outline "Get Started". Right: composed illustration.

## Footer

Light gray `#fafafa`, 6 columns of links (Categories, About, Support,
Community, More, Mobile apps), wordmark + social row bottom.

## Interactions captured (default state)

- Search submit → `/search/gigs?query=<encoded>`
- Popular chip click → `/search/gigs?query=<chip-text>`
- Category tile click → `/categories/<slug>`
- Gig card click → `/<seller-username>/<gig-slug>`
- "Join" button → `/signup`
- "Sign in" → `/login`
