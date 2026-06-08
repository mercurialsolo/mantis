# mantis_indeed — captured corpus

This corpus is the ground-truth spec the mantis_indeed mirror reproduces.
Each subdirectory corresponds to one in-scope page slug.

## Capture method

- Primary: `_capture_brief.md` at the sim_envs root — contains palette,
  typography, exact nav-item order, exact placeholder text, hero copy
  and chip-rail labels captured live from indeed.com by the parent
  session.
- Augmented with training-data recollection (flagged in `notes.md`).
- Live Chrome MCP capture against indeed.com is auth-walled / Cloudflare-
  challenged on most surfaces — captured what we can without auth and
  flagged auth-gated surfaces as ⏳ in `FIDELITY.md` for follow-up.

## Slugs

- `home/` — `/`
- `jobs_search/` — `/jobs?q=software+engineer&l=Austin%2C+TX`
- `viewjob/` — `/viewjob?jk=...`
- `apply/` — `/apply/<id>`
- `myjobs/` — `/myjobs`
- `resumes/` — `/resumes`
- `employers_dashboard/` — `/employers/dashboard`
- `employers_jobs_new/` — `/employers/jobs/new`
- `login/`, `signup/` — auth surfaces

Each slug has `notes.md` minimum; `dom.html`/`styles.json` populated
where capture was possible.
