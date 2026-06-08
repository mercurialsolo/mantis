# mantis_indeed

Server-rendered synthetic mirror of `indeed.com` for mantis CUA training.

## What it mirrors

- Two-input hero search (`/`)
- Canonical Indeed 3-pane search results (`/jobs`)
- Full job detail page (`/viewjob`)
- 4-step Easy Apply (`/apply/<jk>`)
- Saved + Applied tabs (`/myjobs`)
- Resume manager (`/resumes`)
- Employer dashboard + posting / applicant review (`/employers/*`)
- Username + password auth (`/login`, `/signup`)

See `SCOPE.md` for the full in-scope spec and `FIDELITY.md` for
section-by-section match status.

## Run

```bash
cd deploy/sim_envs/mantis_indeed
docker build -t mantis/sim-env-mantis-indeed:latest .
docker run --rm -p 8031:8080 \
    -e SEED=42 \
    -e FAKE_NOW=2026-01-15T09:00:00Z \
    -e ENV_ADMIN_TOKEN=$(python -c 'import secrets;print(secrets.token_urlsafe(32))') \
    mantis/sim-env-mantis-indeed:latest
```

Or run locally without Docker:

```bash
pip install -r requirements.txt
ENV_ADMIN_TOKEN=test FAKE_NOW=2026-01-15T09:00:00Z \
  python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Smoke test

```bash
ENV_ADMIN_TOKEN=test python scripts/smoke.py
```

Exits 0 when all in-scope pages render + all three oracles pass via
their happy-path HTTP drives.

## Oracle tasks

- `t01_search_save_remote` — seeker searches "software engineer" in
  "Austin, TX" with `remote=1`, then saves `job_00007`. Oracle reads
  `search_submitted` + `job_saved` audit rows and the `saved_jobs`
  table.
- `t02_easy_apply` — seeker Easy Applies to `job_00012` with phone +
  `resume_00001` + screening answers. Oracle reads
  `application_submitted` audit row + `applications` row.
- `t03_employer_review_applicant` — employer moves the seeded
  `application_seed_t03` row on `job_00003` from `new` → `reviewed`.
  Oracle reads `application_status_changed` audit row +
  `applications.status` + `applications.reviewed_at`.

Oracles are dispatched via `GET /__env__/oracle?task_id=<id>` (admin
token required).

## Fidelity tracker

`FIDELITY.md` holds the section-by-section match status with the legend
`exact / close / partial / missing / not-matched`. First-pass build
gets every in-scope row to at least 🟢 close. Pixel-perfect ≤2px is the
north star, not a one-turn deliverable.

`FIDELITY_AGENT_PROMPT.md` is the playbook follow-up agents use to
close one gap at a time.

## Auth defaults

Without `ENV_REQUIRE_AUTH=1`, the env operates with a default acting
seeker `user_00001` and default acting employer `user_emp_00003` so
oracles can grade without first going through `/login`. With
`ENV_REQUIRE_AUTH=1`, protected paths 303 → `/login`.

Pinned credentials (per seed):

- Seeker `user_00001` — email `seeker1@example.com`, password
  `seekerpass`.
- Employer `user_emp_00003` — email `employer3@example.com`, password
  `emppass`.
