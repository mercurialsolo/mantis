# mantis_mercor

Synthetic mirror of `mercor.com` — an AI talent marketplace. Built
to the same fidelity bar as `mantis_boattrader` and `mantis_shop`:
server-rendered FastAPI + Jinja approximation that the mantis CUA
agent drives end-to-end and the oracle grades.

## Scope

See `SCOPE.md`. In-scope pages: `/`, `/jobs`, `/jobs/<id>`,
`/apply/<id>/...`, `/login`, `/signup`, `/dashboard`, `/profile`,
`/experts`. Three oracle tasks (`t01`, `t02`, `t03`).

## Run

### Local (Python)

```bash
cd deploy/sim_envs/mantis_mercor
pip install -r requirements.txt
ENV_ADMIN_TOKEN=test python -m uvicorn app.main:app --port 8090
```

Open `http://127.0.0.1:8090/` for the home page.
`http://127.0.0.1:8090/__env__/health` is public; everything else
under `/__env__/` requires header `X-Env-Admin: test`.

### Docker

```bash
docker build -t mantis/sim-env-mantis-mercor:latest deploy/sim_envs/mantis_mercor
docker run --rm -p 8090:8080 \
    -e SEED=42 \
    -e FAKE_NOW=2026-06-08T09:00:00Z \
    -e ENV_ADMIN_TOKEN=$(python -c 'import secrets;print(secrets.token_urlsafe(32))') \
    mantis/sim-env-mantis-mercor:latest
```

### Smoke test

```bash
cd deploy/sim_envs/mantis_mercor
ENV_ADMIN_TOKEN=test python scripts/smoke.py
```

Boots the app in-process via `fastapi.testclient.TestClient`, drives
each oracle's happy path, and asserts all three pass.

## Oracle tasks

| ID | Description | Pass condition |
|----|-------------|----------------|
| `t01` (`t01_apply_to_ml_engineer`) | Candidate applies to `job_00001` | application row submitted with all screening answers filled |
| `t02` (`t02_shortlist_candidate`) | Client shortlists `candidate_00007` for `job_00001` | shortlist row + audit_log `candidate_shortlisted` |
| `t03` (`t03_decline_application`) | Client declines `app_00001` with reason | applications.status='rejected' + reason text + audit_log |

Dispatch via `GET /__env__/oracle?task_id=t01` (with admin header).

## Fidelity tracker

`FIDELITY.md` — section-by-section structural match log vs the live
`mercor.com`. Status legend matches `mantis_boattrader/FIDELITY.md`.

## Layout

```
deploy/sim_envs/mantis_mercor/
├── Dockerfile
├── README.md
├── SCOPE.md
├── FIDELITY.md
├── FIDELITY_AGENT_PROMPT.md
├── requirements.txt
├── _captured/              # Phase-1 discovery corpus (one slug per page).
├── scripts/
│   └── smoke.py
└── app/
    ├── __init__.py
    ├── main.py             # FastAPI factory + helpers
    ├── auth.py             # email + password sessions
    ├── db.py               # SQLite schema + audit_log
    ├── seed.py             # deterministic per-seed fixture
    ├── routes/
    │   ├── env_admin.py    # /__env__/{health,reset,seed,clock,oracle,state,events,mutations}
    │   ├── auth.py
    │   ├── marketing.py    # / and /experts
    │   ├── jobs.py
    │   ├── apply.py        # 5-step apply flow
    │   ├── dashboard.py    # candidate + client surfaces + shortlist/decline
    │   └── profile.py
    ├── oracles/
    │   ├── __init__.py
    │   ├── t01_apply_to_ml_engineer.py
    │   ├── t02_shortlist_candidate.py
    │   └── t03_decline_application.py
    ├── static/
    │   └── site.css        # hand-rolled, grouped by component
    └── templates/
        ├── base.html, home.html, experts.html
        ├── jobs_list.html, job_detail.html
        ├── apply_step.html, apply_review.html, apply_confirm.html
        ├── login.html, signup.html
        ├── dashboard_candidate.html, dashboard_client.html
        ├── profile.html
        └── _role_card.html (Jinja macro)
```
