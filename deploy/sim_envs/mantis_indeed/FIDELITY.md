# mantis_indeed — FIDELITY

**Last updated**: 2026-06-08 (first-pass build)

## Status legend

- ✅ **exact** — structural delta ≤2px, palette + text match the capture.
- 🟢 **close** — matches structurally; minor pixel diff or small spacing
  drift that doesn't break the agent's perception.
- 🟡 **partial** — element exists with correct shape but not yet tuned
  to capture-grade precision (font, spacing, or border drift).
- 🔴 **missing** — element from capture not yet implemented.
- ⚪ **not-matched** — captured spec ambiguous; followup needed to
  pin down the canonical look.

## In-scope page coverage

### `/` — home

| Element             | Real spec                                                        | Mine                                           | Status |
|---------------------|------------------------------------------------------------------|------------------------------------------------|--------|
| Sticky top nav      | 52px tall, white, 1px border-bottom `#E4E2E0`                   | 52px, white, 1px border-bottom `#D4D2D0`       | 🟢     |
| Brand wordmark      | `indeed` brand-blue, 28-32px weight 700                          | `indeed` `#2557A7` 28px weight 700             | ✅     |
| Right nav items     | `Sign in`, `Employers / Post Job`                                | exact strings                                  | ✅     |
| Secondary strip     | Home, Company reviews, Find salaries, Employers, Create your resume, Change country | exact strings + country pill        | ✅     |
| Hero H1             | `Find your next job` (canonical anon variant)                    | exact                                          | ✅     |
| Hero search bar     | 56px, 880px wide, 1px border, 2 inputs + CTA, separator          | 56px, 880px max, 1px border, 2 inputs + CTA   | 🟢     |
| Left input          | placeholder `Job title, keywords, or company`, aria + name=q     | exact                                          | ✅     |
| Right input         | placeholder `City, state, zip code, or "remote"`, aria + name=l  | exact                                          | ✅     |
| Search CTA          | brand-blue, label `Search`, 40px tall                            | brand-blue `Search` 40px                       | ✅     |
| Trending panel H2   | `What's trending on Indeed`                                      | exact                                          | ✅     |
| Resources panel H2  | `Employer Resources`                                             | exact                                          | ✅     |
| Hero bg surface     | light grey panel                                                  | `var(--surface-grey)` `#F3F2F1`                | 🟢     |

### `/jobs?q=&l=` — search results

| Element                       | Real spec                                                                 | Mine                                                        | Status |
|-------------------------------|---------------------------------------------------------------------------|-------------------------------------------------------------|--------|
| Sticky top nav (reused)       | same as `/`                                                              | same as `/`                                                  | ✅     |
| Secondary search bar          | 44px compact, prefills q + l                                              | 44px compact, prefills q + l                                 | 🟢     |
| H1                            | `software engineer jobs in Austin, TX` (lowercase echo)                   | exact templated as `{q.lower()} jobs in {l}`                 | ✅     |
| Filter chip rail              | 11 chips exact order: Date posted, Remote, Developer skill, Job Type, …  | 11 chips, exact order, exact labels                          | ✅     |
| Chip styling                  | 32px tall, 1px border, 16px radius, 14px, active = blue tint + blue border | 32px tall, 1px border, radius 16px, active state matches    | 🟢     |
| Sort row                      | `Sort by: relevance` right-aligned + result count left                    | matches                                                      | 🟢     |
| 3-pane layout                 | grid 480px + 680px + 20px gap, container 1440px                           | exact grid template                                          | ✅     |
| Result card                   | 1px border, 8px radius, 16px padding, white bg                            | exact                                                        | ✅     |
| Card title                    | 18px bold brand-blue link                                                 | matches                                                      | ✅     |
| Selected card                 | blue tint bg + 4px brand-blue left accent                                 | tint + `::before` 4px left accent                            | 🟢     |
| Bookmark icon                 | filled star when saved, outline otherwise                                 | ☆/★ swap                                                     | 🟡     |
| Right detail pane             | 680px wide, sticky                                                        | 680px, sticky top:64px                                       | 🟢     |
| Detail-pane H2 suffix         | `- job post` canonical                                                    | exact                                                        | ✅     |
| Detail tabs                   | `Job details`, `Company`, `Reviews`                                       | exact                                                        | ✅     |
| Apply CTA label               | `Apply now`                                                               | exact                                                        | ✅     |
| Result selection w/o nav      | click left card → right pane swaps + URL `?vjk=<jk>`                      | implemented via vanilla JS + `/jobs/_detail` fragment        | ✅     |

### `/viewjob?jk=<id>` — full job detail

| Element                  | Real spec                                                        | Mine                       | Status |
|--------------------------|------------------------------------------------------------------|----------------------------|--------|
| Job header card          | bordered card with title, company, location, salary chips        | matches                    | 🟢     |
| H2 `… - job post` suffix | canonical                                                        | exact                      | ✅     |
| Apply / Save CTAs        | primary + outline                                                | exact                      | ✅     |
| Tabs                     | Job details / Company / Reviews                                  | exact                      | ✅     |
| Description rendering    | paragraphs from markdown-ish source                              | split on \n\n              | 🟡     |

### `/apply/<jk>` — Easy Apply (4 screens)

| Element                  | Real spec                                                        | Mine                                  | Status |
|--------------------------|------------------------------------------------------------------|---------------------------------------|--------|
| Stepper                  | 4 steps: Contact → Resume → Questions → Review                   | exact 4-step stepper                  | ✅     |
| Step 1 H1                | `Add your contact information`                                   | exact                                 | ✅     |
| Step 2 H1                | `Add your resume`                                                | exact                                 | ✅     |
| Step 3 H1                | `Answer screening questions`                                     | exact                                 | ✅     |
| Step 4 H1                | `Review your application`                                        | exact                                 | ✅     |
| Submit CTA               | `Submit application`                                             | exact                                 | ✅     |
| Success confirmation     | green check + `Application submitted` + next-step CTAs           | matches                               | 🟢     |
| Auto-form-validation     | required + HTML5                                                 | HTML5 required attrs                  | 🟡     |

### `/myjobs` — Saved + Applied

| Element            | Real spec                                | Mine                              | Status |
|--------------------|------------------------------------------|-----------------------------------|--------|
| H1                 | `My jobs`                                | exact                             | ✅     |
| Tabs               | `Saved` / `Applied`                      | exact tabs with active underline  | ✅     |
| Empty state Saved  | `You haven't saved any jobs yet. …`      | exact                             | ✅     |
| Empty state Applied| `You haven't applied to any jobs yet.`   | exact                             | ✅     |
| Card shape         | reused result-card                       | reused                            | 🟢     |
| Applied status badge | `New / Reviewed / Rejected / Hired`    | exact + colour-coded              | 🟢     |

### `/resumes` — resume manager

| Element             | Real spec                                          | Mine                                | Status |
|---------------------|----------------------------------------------------|-------------------------------------|--------|
| H1                  | `Your resumes`                                     | exact                               | ✅     |
| Resume row layout   | title + meta + Edit/Delete buttons                 | matches                             | 🟢     |
| Add resume form     | inline disclosure → title/summary/skills/exp       | matches                             | 🟡     |

### `/employers/dashboard`

| Element               | Real spec                                              | Mine                                | Status |
|-----------------------|--------------------------------------------------------|-------------------------------------|--------|
| H1                    | `Employer dashboard`                                   | exact                               | ✅     |
| KPI strip             | 3 cards: Active postings, New applicants this week, Total views | matches                  | ✅     |
| Postings table        | title, location, applicants breakdown, posted, status  | matches                             | 🟢     |
| Row hover             | tint                                                   | tint + cursor                       | 🟢     |

### `/employers/jobs/<id>` — posting / applicant review

| Element                  | Real spec                                       | Mine                              | Status |
|--------------------------|-------------------------------------------------|-----------------------------------|--------|
| H1                       | `{Job Title}`                                  | exact                             | ✅     |
| Status filter chip row   | All / New / Reviewed / Rejected / Hired         | matches                           | ✅     |
| Applicant table          | Name / Resume / Status / Applied + actions     | matches                           | 🟢     |
| Status select            | dropdown w/ auto-submit on change               | matches                           | ✅     |
| Mark-reviewed shortcut   | inline button for `new` applicants              | shipped                           | 🟡     |

### `/employers/jobs/new`

| Element            | Real spec                                            | Mine                            | Status |
|--------------------|------------------------------------------------------|---------------------------------|--------|
| H1                 | `Post a job`                                         | exact                           | ✅     |
| Form               | title + location + salary low/high + remote + type + desc | matches                  | ✅     |
| Submit CTAs        | `Publish` + `Save as draft`                          | matches                         | 🟢     |

### `/login`, `/signup`

| Element            | Real spec                                | Mine                       | Status |
|--------------------|------------------------------------------|----------------------------|--------|
| Card width         | 360px centred                            | 360px centred              | ✅     |
| Login H1           | `Sign in`                                | exact                      | ✅     |
| Signup H1          | `Create your account`                    | exact                      | ✅     |
| Error banner       | red                                      | red banner                 | 🟢     |

## Open gaps (for follow-up agents)

- **Pixel-perfect pass** — overall sections are structurally correct but
  haven't been per-pixel diff-verified against a live screenshot. Run
  the per-element probe + screenshot diff once Chrome MCP can capture
  through Cloudflare.
- **Font** — site uses system Helvetica fallback rather than Indeed
  Sans. Acceptable per the build prompt (no Google Fonts at runtime),
  but tracking as 🟡.
- **Hero background** — live site uses a subtle gradient; mirror uses
  flat grey panel. 🟡.
- **Detail tabs interactivity** — visual only, no content swap. 🟡.
- **Reviews / Company tabs in detail pane** — placeholder only. 🔴 for
  full content fidelity.
- **Filter chip popovers** — chips are visual-only except for `Remote`
  which actually toggles. Date-posted/Pay/etc. need popover impl. 🟡.
- **Captured corpus** — only `notes.md` shipped for each slug. Follow-up
  agents should populate `dom.html`, `styles.json`, `screenshot.png`
  once Cloudflare-friendly capture is feasible.
- **Search ranking** — substring match only; live Indeed has fuzzy +
  relevance ranking. Acceptable per build prompt.
- **Easy Apply auto-fill** — currently pulls from user record; live
  site prefills from past applications. 🟡.

## Iteration log

- **2026-06-08 (initial)** — Phases 0–6 first-pass complete. All three
  oracles pass via smoke.py. Captured corpus is `notes.md`-only;
  fidelity loop has not yet pixel-diffed against a captured screenshot
  (Cloudflare-gated). All major in-scope pages render; interactions
  wired to the audit_log.
