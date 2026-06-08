# Capture brief — observations from live sites (2026-06-08)

Captured by Claude Code via Chrome MCP at 1512×677 viewport while the parent
session was deciding scope for the four new high-fidelity mirror envs
(mantis_mercor / mantis_fiverr / mantis_linkedin / mantis_indeed).

These are NOT a substitute for each agent doing its own Phase-1 capture into
`_captured/`. They are a ground-truth anchor: identifiers, palette, font, and
structural hints captured directly from the live sites that each agent should
mirror exactly. If your captured corpus disagrees with this file, trust the
corpus (it's later) — but the live-site observations below were direct, so
they're a high-confidence baseline.

---

## 1. mercor.com

**URL captured**: `https://www.mercor.com/`, `https://www.mercor.com/experts/`

**Brand & typography**
- Font family: `Inter, "Inter Fallback"` — load Inter via local @font-face or system fallback; the look is plain Inter.
- Body background: transparent (page falls through to white).
- Body color: `rgb(0, 0, 0)` pure black for primary text.
- Accent / brand color: indigo `rgb(79, 70, 229)` (Tailwind indigo-600). Used for primary CTAs and links.
- Surface greys (in priority order): `rgb(243, 244, 246)` light grey panel (gray-100), `rgb(229, 231, 235)` (gray-200), `rgb(156, 163, 175)` (gray-400), `rgb(107, 114, 128)` (gray-500), `rgb(75, 85, 99)` (gray-600), `rgb(31, 41, 55)` (gray-800).
- This is unmistakably a Tailwind palette — match it exactly.

**Top nav** (anonymous home `/`)
- Slim header, no large logo brand bar.
- Nav items, in order: `APEX`, `APEX-Agents`, `APEX-SWE`, `Research`, `Enterprise`, `Experts`.
- Header nav width was 374px (compact left-side nav, NOT full-bleed).

**Home page (`/`)**
- H1: `Shape the frontier of AI`
- Sole H2 in fold: `Latest roles`
- Stats strip near top with three pseudo-stats: `Average pay $/hr`, `Roles created (k)`, `Daily payouts ($)`.
- "Latest roles" card grid below — each card shape:
  - Role title (e.g. `Internal Medicine Expert`)
  - Rate range (`$130-$180/hr`)
  - 3-letter avatar badge (`LFA`, `JDT`, `GBR` — generated initials, not photos)
  - "N hired recently" microcopy (e.g. `75 hired recently`)
  - `Apply` CTA button

**Experts landing (`/experts/`)**
- H1: `Get paid to work on AI projects`
- Numbered "how it works" three-step strip: `1. Create a profile`, `2. Take AI assessments`, `3. Find opportunities → Apply to current li[stings]`.
- "Latest roles" section followed by per-role sections, each with its own H2 (Internal Medicine Expert, Hematology/Oncology Expert, Biology PhD Expert, Legal Expert — Litigation, Private Equity Expert, Management & Strategy Consultants (MBB/Big 5), Office-Suite Experts).
- FAQ section with H2s: `What is Mercor?`, `What is AI training work?`.
- 18 role-card links (`a[href*="/jobs"]`) visible at viewport.

**Interaction signals**
- Apply button on each role card — clicking goes to a per-role apply flow (form-based, multi-step).
- No login banner blocking the public site — the apply CTA is the conversion point.

**Mirror priorities**
- Tailwind-ish grey + indigo palette must match.
- The 3-letter-initial avatar pattern (deterministic per role) is a distinctive visual element — replicate.
- The "$130-$180/hr" + "N hired recently" + "Apply" card layout is the canonical role card — implement this exact shape.

---

## 2. fiverr.com

**URLs captured**: `https://www.fiverr.com/`, `https://www.fiverr.com/search/gigs?query=logo%20design`

**Brand & typography**
- Font family: `Macan, "Helvetica Neue", Helvetica, Arial, sans-serif` — Macan is a Fiverr custom font; system Helvetica fallback is acceptable for the mirror.
- Body background: `rgb(255, 255, 255)` pure white.
- Brand green: Fiverr's signature green (#1DBF73 / `rgb(29, 191, 115)`) — confirm via spot-check in your agent's capture; use for primary CTAs and accent.

**Top nav**
- Items in order: `Fiverr Pro`, `Explore`, `EN`, `Become a Seller`, `Sign in`, `Join`.
- Secondary category strip directly under: `Trending 🔥`, `Graphics & Design`, `Programming & Tech`, `Digital Marketing`, `Video & Animation`, `Writing & Translation` (more truncated).
- Top search bar — placeholder text: `What service are you looking for today?`

**Home page (`/`)**
- H1: `Our freelancers will take it from here`
- H2 strip: `Popular services`
- Popular services tile grid: `Vibe Coding`, `Website Development`, `Video Editing`, `Software Development`, `Book Publishing`, `Architecture & Interior Design`, `Book Design`, `UGC Videos`, `Voice Over` — each is a card with image + label.
- 53 category-related links visible.

**Search results page (`/search/gigs?query=logo%20design`)**
- H1: `Results for logo design`
- Filter rail with labels: `Logo services / Find a logo designer`, `Logo maker / Customize pre-made logos`, `Category`, `Logo options`, `Seller details`, `Budget`, `Delivery time`.
- Sort dropdown: `Sort by: Relevance` (top-right of result list area).
- 157 gig cards in viewport — gig card includes: gig title, seller avatar+username, star rating + review count, "Starting at $X".

**Interaction signals**
- Search bar in top nav: typing + submit navigates to `/search/gigs?query=…`.
- Filter chips toggle URL params.
- Gig detail page has the canonical package picker (Basic / Standard / Premium) that re-prices CTA inline.

**Mirror priorities**
- Hero phrasing "Our freelancers will take it from here" + "Popular services" H2 — exact text.
- The placeholder copy `What service are you looking for today?` must be exact.
- The filter rail labels above must be exact.
- "Sort by: Relevance" exact label.

---

## 3. linkedin.com

**URL captured**: `https://www.linkedin.com/` (resolved to `/feed/` — session was already logged in), `https://www.linkedin.com/jobs/`

**Brand & typography**
- Font family: `system-ui, -apple-system, "system-ui", "Segoe UI", Roboto, Ubuntu, Oxygen, Cantarell, "Fira Sans", "Droid Sans", "Helvetica Neue", Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol"` — system stack, no custom font.
- Body background: `rgb(244, 242, 238)` — LinkedIn's signature warm-beige neutral. THIS IS A DISTINCTIVE COLOUR and must be exact.
- Top nav header height: 52px.
- Primary text: `rgba(0, 0, 0, 0.9)`. Secondary text: `rgba(0, 0, 0, 0.6)`. Muted: `rgba(0, 0, 0, 0.75)`.
- White panel background `rgb(255, 255, 255)` for cards (profile card, post card, etc.).
- Dark navy text `rgb(43, 51, 63)` used as a heading colour.
- LinkedIn brand blue is `#0a66c2` (NOT captured directly because the user's session deep-linked but use this — it's the canonical brand colour).

**Top nav (signed-in)**
- Items in order: `Home`, `My Network`, `Jobs`, `Messaging`, `Notifications`, `Me`, `For Business`, `Advertise`.
- 52px tall sticky header. White background, items as icon+label stack.
- Search bar to the left of the nav items (omitted from the structural dump but present).

**Feed (`/feed/`)**
- Three-column layout: left rail (profile-card + saved items) + centre feed + right rail (people you may know + news).
- LinkedIn uses heavily hashed CSS class names (e.g. `f5526673 _7e38840f`) — DO NOT try to match class names. Match by ROLE / structure.
- Each feed post is a card on white background with: author avatar + headline + post text + image/video + reaction bar (Like / Comment / Repost / Send) + reaction counts above.

**Jobs page (`/jobs/`)**
- Top input placeholder: `Describe the job you want` — exact text.
- H1 not rendered before async load — page is heavily lazy-loaded; mirror it as a server-rendered approximation.

**Mirror priorities**
- The warm-beige `#F4F2EE` (244, 242, 238) page background is signature — MUST be exact.
- 52px nav with the eight items above in order.
- System-ui font stack — no custom font.
- White card panels on beige page background.
- Reaction bar order: `Like / Comment / Repost / Send` (LinkedIn's canonical order).
- Brand accent: `#0a66c2` blue.

---

## 4. indeed.com

**URLs captured**: `https://www.indeed.com/`, `https://www.indeed.com/jobs?q=software+engineer&l=Austin%2C+TX`

**Brand & typography**
- Font family: `"Indeed Sans", "Noto Sans", "Helvetica Neue", Helvetica, Arial, "Liberation Sans", Roboto, Noto, sans-serif` — Indeed Sans is a custom font; Helvetica fallback is fine for the mirror.
- Brand blue: `#2557A7` (Indeed brand — canonical, confirm via spot check).
- Body background: transparent/white.

**Top nav (home page)**
- Items in order: `Home`, `Company reviews`, `Find salaries`, `Sign in` / `Employers / Post Job`.
- Secondary strip below: `Home`, `Company reviews`, `Find salaries`, `Employers`, `Create your resume`, `Change country 🇺🇸 United States`.

**Home page (`/`)**
- Hero: two-input search bar side by side
  - Left input: placeholder `Job title, keywords, or company`, name=`q`, aria-label `search: Job title, keywords, or company`.
  - Right input: placeholder `City, state, zip code, or "remote"`, name=`l`, aria-label `Edit location`.
  - CTA: `Search`.
- Below hero: panels for `What's trending on Indeed`, `Employer Resources`.

**Search results page (`/jobs?q=…&l=…`)**
- H1: `software engineer jobs in Austin, TX` (lowercased query echoed)
- 3-pane layout:
  - Top horizontal filter chip rail (Date posted | Remote | Developer skill | Job Type | Experience level | Pay | Education | Clearance type | Developer type | Compensation package | Distance) — these are the EXACT chip labels and order. 11 chips.
  - Left = results list (each card: title, company, location, salary, snippet, posted date, Apply button)
  - Right = detail pane width 680px showing the selected job in full (header "{Title} - job post" pattern, full description, Apply button)
- Detail pane H2: `{Job Title} - job post` (e.g. `Java Microservices Developer with Reactive Programming - job post`) — the `- job post` suffix is the canonical pattern.
- Apply button labelled `Apply` (sometimes `loading` mid-state, sometimes `Apply now on company site`).

**Interaction signals**
- Clicking a result card on the left updates the right detail pane WITHOUT navigation — the URL changes to add `?vjk=<jobkey>`.
- Each chip opens a popover; selecting a value re-runs search and updates result list.
- 5-digit zip in the `Where` input is conventional but not strictly auto-submit on this site.

**Mirror priorities**
- The two hero placeholders (`Job title, keywords, or company` + `City, state, zip code, or "remote"`) must be EXACT.
- The chip rail labels in the exact order above must be EXACT.
- The detail-pane "- job post" suffix on the job title H2 must be EXACT.
- The right-pane-no-nav interaction (clicking a left card → URL ?vjk=… + right pane re-renders) must work in the mirror — this is THE canonical Indeed interaction.
- Brand blue `#2557A7` accent.

---

## Cross-site conventions worth replicating

- **Initials avatar pattern (mercor)** — when you don't have photos, render 2-3-letter initial blocks on neutral bg. Mercor does this in production.
- **Card with badge + price + microcopy CTA** — common to all four sites for their primary list units (mercor role cards, fiverr gig cards, linkedin job cards, indeed job cards). Standardise this component.
- **Sticky top nav, slim** — all four sites have a sticky thin top nav (mercor 37–52px, linkedin 52px, indeed varies, fiverr ~80px). Keep yours sticky.
- **NO Google Fonts at runtime** — all four sites either ship a custom font from their own CDN or use system stack. For the mirror, use a local font (if you ship one) or system stack only. Outbound fetches at runtime are out of scope per CLAUDE.md.

---

## Anti-patterns observed (avoid)

- **Heavy SPA with hashed class names** (LinkedIn). DO NOT try to mirror class-name hashes. Match by ROLE, semantics, layout, and label text. The CUA reads pixels and labels, not class names.
- **Lazy-load delays** on heavily dynamic pages. Mirror as a server-rendered static approximation — the agent gets the same visible surface deterministically.
