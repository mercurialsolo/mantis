# mantis_indeed — SCOPE

This sim env mirrors **indeed.com** as a server-rendered FastAPI surface that
the mantis CUA can train against. High-fidelity means pixel-parity visual
replica + interaction-parity behavioural replica vs the live site.

Source: `https://www.indeed.com/`

## In-scope pages

- `/` — marketing home with the canonical two-input hero (What + Where).
- `/jobs?q=&l=&sort=&radius=&remote=&date=` — 3-pane search results
  (top filter chip rail, left results list, right detail pane).
- `/viewjob?jk=<id>` — full job detail page (direct-link variant of the
  right pane).
- `/apply/<id>` — Indeed Easy Apply, 3 steps: contact → resume → screening
  Q&A, then review → submit.
- `/myjobs` — Saved + Applied tabs.
- `/resumes` — resume manager (upload + edit).
- `/employers/dashboard` — employer posting board + applicant review.
- `/employers/jobs/new` — employer create posting.
- `/login`, `/signup` — username + password.

## In-scope entities

- `users` — seekers + employers (`role` column).
- `companies`
- `jobs` (title, company, location, salary_low/high, remote_flag, posted_date,
  description, jk job-key).
- `resumes` (per-user).
- `applications` (user + job + status: new/reviewed/rejected/hired).
- `saved_jobs` (user + job + saved_at).
- `audit_log` — append-only mutation record (oracle source of truth).

## In-scope interactions

- **Search** — type What + Where in hero → submit → `/jobs?q=&l=`.
- **Result selection** — click a result card on the left of `/jobs` →
  detail pane on the right updates **without page nav** (URL adds
  `?vjk=<jk>`). This is the canonical Indeed interaction.
- **View Job** — click "View job" → `/viewjob?jk=<jk>` full page.
- **Filter** — click a chip on the filter rail → URL param toggles, results
  re-render. Chips: Date posted, Remote, Pay, Job Type, Experience level,
  Education, Company.
- **Save** — bookmark icon on a result toggles `saved_jobs` row + bookmark
  visual state.
- **Easy Apply** — from the detail pane click "Apply now" → 3-step form →
  submit → application row + redirect to confirmation.
- **Employer review** — from `/employers/dashboard`, click a posting →
  applicant list → move applicant from "New" to "Reviewed" → audit row.

## Out of scope (explicitly NOT mirrored)

- Real images / brand photography → deterministic SVG placeholders.
- Advertising / analytics / tracking scripts → stripped.
- Third-party widgets → stripped.
- OAuth / social login → username + password only.
- Payment processing → no real Stripe; audit_log row only.
- Real-time websockets / SSE → polling or static.
- A/B variants → pick the anonymous-visitor canonical.

## Brand & typography (ground truth — capture brief)

- Font: `"Indeed Sans", "Noto Sans", "Helvetica Neue", Helvetica, Arial,
  "Liberation Sans", Roboto, Noto, sans-serif`. Helvetica fallback is fine.
- Brand blue: `#2557A7`.
- Body bg: white `#FFFFFF`.
- Primary CTA: brand blue background, white text.
- Top nav: slim, sticky, white bg with bottom border.

## Hero / nav copy (must be exact)

- Hero H1: depends on slug — `/` shows the search form; `/jobs?q=software
  engineer&l=Austin, TX` shows lowercase-echo H1
  `software engineer jobs in Austin, TX`.
- Left hero input placeholder: `Job title, keywords, or company`,
  aria-label `search: Job title, keywords, or company`, name `q`.
- Right hero input placeholder: `City, state, zip code, or "remote"`,
  aria-label `Edit location`, name `l`.
- CTA: `Search`.
- Top nav (anon home): `Home`, `Company reviews`, `Find salaries`,
  `Sign in`, `Employers / Post Job`.
- Secondary strip: `Home`, `Company reviews`, `Find salaries`, `Employers`,
  `Create your resume`, `Change country 🇺🇸 United States`.

## Search results layout (must match)

- Three panes. Top chip rail with 11 chips in this exact order:
  `Date posted`, `Remote`, `Developer skill`, `Job Type`, `Experience level`,
  `Pay`, `Education`, `Clearance type`, `Developer type`,
  `Compensation package`, `Distance`.
- Left = result list (each card: title, company, location, salary
  range, snippet, posted-relative-date, Apply button, bookmark icon).
- Right = detail pane width 680px. Detail-pane H2 pattern:
  `{Job Title} - job post` — the `- job post` suffix is canonical.
- Apply button label: `Apply now`.

## Done bar

Same as `mantis_boattrader/SCOPE.md`:

- Structural deltas vs captured spec ≤ 2 px in every visible section.
- Perceptual diff vs captured screenshot < 0.5%.
- Every in-scope interaction replays with matching URL + DOM + audit_log
  deltas — verified by the three oracle tasks.

The first-pass iteration ships partial-fidelity sections (marked in
FIDELITY.md). Subsequent agents close the gaps using `FIDELITY_AGENT_PROMPT.md`.
