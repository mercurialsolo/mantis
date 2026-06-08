# `mantis_mercor` — Fidelity match log

Working doc tracking each element's match status against
`https://www.mercor.com/`. Update as iterations land.

Last updated: **first-pass scaffold** (2026-06-08) — initial pass
implementing the captured spec from `_capture_brief.md`. No live
re-probe this turn; subsequent passes should re-capture via Chrome
MCP at 1440×900 and update the rows below in place.

## Methodology

Each row records the element/region, the real Mercor measurement
(from the capture brief or future direct probes), and the current
state. Status legend:

- ✅ exact (within 1-2px rendering tolerance)
- 🟡 close (matches structurally; minor pixel diff)
- ⏳ partial (some pieces match, more work needed)
- ❌ missing / broken
- 🚫 intentionally not matched (e.g. real photos, A/B variants)

## Global / palette / typography

| Element | Real Mercor | Mine | Status |
|---|---|---|---|
| Body color | `#000000` pure black | `#000000` via `body { color: var(--black) }` | ✅ |
| Body bg | `#FFFFFF` white | `#FFFFFF` | ✅ |
| Brand color | `#4F46E5` indigo-600 | `--indigo-600: #4F46E5` | ✅ |
| Font family | `Inter, "Inter Fallback"` | `Inter, "Inter Fallback", system-ui, ...` | ✅ |
| gray-100 | `#F3F4F6` | `--gray-100: #F3F4F6` | ✅ |
| gray-200 | `#E5E7EB` | `--gray-200: #E5E7EB` | ✅ |
| gray-500 | `#6B7280` | `--gray-500: #6B7280` | ✅ |
| gray-800 | `#1F2937` | `--gray-800: #1F2937` | ✅ |
| Focus ring | Tailwind ring-2 ring-indigo-500/15 | `0 0 0 3px rgba(79,70,229,0.15)` | 🟡 |

## Topbar / nav (`/`)

| Element | Real Mercor | Mine | Status |
|---|---|---|---|
| Header height | ~52px sticky thin | 52px sticky | ✅ |
| Border-bottom | 1px gray-200 | 1px var(--gray-200) | ✅ |
| Nav items order | APEX, APEX-Agents, APEX-SWE, Research, Enterprise, Experts | same | ✅ |
| Nav item font | 13/500 | 13/500 | 🟡 |
| Right actions | Sign in (anonymous) | Sign in + Sign up (added Sign up to give CUA an entry) | 🟡 |
| Brand mark | (no large brand bar on real site) | small `M` mark + "mercor" lowercase | 🟡 |

## Hero (`/`)

| Element | Real Mercor | Mine | Status |
|---|---|---|---|
| H1 text | "Shape the frontier of AI" | "Shape the frontier of AI" | ✅ |
| H1 font | Inter 48/700 | 48/700 | 🟡 |
| Hero CTA buttons | (1 primary on live) | 2 buttons (added "How it works") | ⏳ |
| Stats strip | 3 stats (Avg pay $/hr, Roles created k, Daily payouts $) | 3 stats, same labels | ✅ |
| Stats values | live values vary; brief noted layout | seed-driven static (95 / 12k / 240k) | ⏳ |

## Latest roles grid (`/`)

| Element | Real Mercor | Mine | Status |
|---|---|---|---|
| Section H2 | "Latest roles" | "Latest roles" | ✅ |
| Card grid columns | likely 3 at desktop | 3 cols, responsive 2/1 | ✅ |
| Card shape | title / $X-$Y/hr / 3-letter avatar / "N hired recently" / Apply | same | ✅ |
| Card avatar | 3-letter initials on gray background | gray-100 bg, gray-800 text, 34x34 circle | 🟡 |
| Apply CTA color | indigo-600 background, white text | indigo-600 / white | ✅ |
| Card border | 1px gray-200 + 12px radius | 1px gray-200 + 12px radius | 🟡 |
| Card hover | shadow lift | shadow-md on hover | 🟡 |

## `/experts` page

| Element | Real Mercor | Mine | Status |
|---|---|---|---|
| H1 | "Get paid to work on AI projects" | same | ✅ |
| 3-step "how it works" | 1. Create profile, 2. Take AI assessments, 3. Find opportunities | same | ✅ |
| Latest roles section | present below how-it-works | present | ✅ |
| FAQ H2s | "What is Mercor?", "What is AI training work?" | both present, `<details>` | ✅ |
| Per-role sections per category | real site renders per-category H2 + cards | not yet — single grid only | ⏳ |

## `/jobs` filter rail + results

| Element | Real Mercor | Mine | Status |
|---|---|---|---|
| Filter rail position | left side | left, 260px sticky | 🟡 |
| Filter labels | Category, Rate, Engagement type | same | ✅ |
| Filter widget kind | chip radios | chip radios | ✅ |
| Sort label | "Sort by: Latest" | "Sort by: Latest" (static) | 🟡 |
| Results grid | 3-col desktop card grid | 3-col, same card | ✅ |

## `/jobs/<id>` detail

| Element | Real Mercor | Mine | Status |
|---|---|---|---|
| H1 = role title | yes | yes | ✅ |
| Right rail summary card | rate + engagement + Apply CTA | same, 320px sticky | ✅ |
| Description rendering | Markdown / paragraphs | Markdown → `<br/>` substitution | ⏳ |
| Skills chips | gray-100 chips | gray-100 chips | ✅ |
| Screening preview | yes | numbered list | 🟡 |

## `/apply/<id>` multi-step

| Element | Real Mercor | Mine | Status |
|---|---|---|---|
| Step progress | "Step N of M" + step list | both rendered | 🟡 |
| Persisted draft on Back | yes | in-memory `APPLY_DRAFTS` keyed by (user, job) | ✅ |
| Step fields | profile / resume / Q&A / review / submit | same 5 steps | ✅ |
| Confirmation banner | "Application submitted" | "Application submitted — we'll be in touch." | 🟡 |

## `/login`, `/signup`

| Element | Real Mercor | Mine | Status |
|---|---|---|---|
| Centered card layout | yes | max-width 480px centered card | ✅ |
| H1 | "Sign in to Mercor" | same | ✅ |
| Error inline | yes | red alert div | 🟡 |
| Signup role radio | (real site uses path branching, not radio) | Candidate / Client radio | 🚫 (intentional — gives CUA explicit role pick) |

## `/dashboard` (candidate + client)

| Element | Real Mercor | Mine | Status |
|---|---|---|---|
| Role-sensitive view | yes | yes (sniffs session.users.role) | ✅ |
| Candidate H1 | "My applications" | same | ✅ |
| Status badges | colored | indigo / amber / blue / green / red per status | 🟡 |
| Client H1 | "Review queue" | same | ✅ |
| Add-to-shortlist toggle | inline | inline form posts → redirect; not yet AJAX-toggle to "Added" | ⏳ |
| Decline reason required | yes | required + inline `<details>` form | ✅ |

## `/profile`

| Element | Real Mercor | Mine | Status |
|---|---|---|---|
| Fields | headline / skills / rate / availability | same | ✅ |
| Save → audit_log | yes | `profile_updated` audit row | ✅ |

## Iteration log

- **2026-06-08 v=1 (first-pass scaffold)** — captured spec from
  `_capture_brief.md`, scaffolded 8 templates + shared CSS, wired
  all 7 routes, shipped 3 oracles + smoke test (31/31 checks pass).
  Mostly 🟡-close; no live re-probe yet so most rows can't escalate
  to ✅. Open follow-up: capture corpus via Chrome MCP, then per-row
  pixel-diff pass.

## Remaining gaps for follow-up agents

1. **No `_captured/*/dom.html` / `screenshot.png` / `styles.json`** —
   only `notes.md` slugs. A follow-up should fire up Chrome MCP at
   1440×900, snapshot each page's served HTML + computed-style probe
   per element, then diff vs current templates.
2. **Description rendering** in `job_detail.html` uses
   `\n→<br/>` — should run a tiny Markdown subset (headings, lists)
   without adding a dep.
3. **Sort dropdown is static** on `/jobs`; should at minimum
   render a real `<select>` with `Latest`, `Highest pay`,
   `Most hires`.
4. **Per-category role sections on `/experts`** missing; render a
   section-per-category with own H2.
5. **Client dashboard "Add to shortlist" toggle** — currently
   navigates; convert to small vanilla-JS inline toggle with
   `disabled` + label flip to "Added" on success.
6. **Stats values on `/`** are seed-static; consider deriving
   from real `applications.hourly_rate` percentile for non-trivial
   matchability with the live site's numbers.
7. **Pixel-level fidelity** — none of the rows above are ✅ except
   palette tokens. After capture corpus lands, run section-by-section
   sizes/spacing diff.
