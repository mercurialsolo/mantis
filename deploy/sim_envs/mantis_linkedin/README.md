# mantis-linkedin — server-rendered LinkedIn mirror

High-fidelity sim env mirroring **linkedin.com** for the mantis CUA harness.
Covers the professional-network surface: profile, feed, jobs, connections,
messaging. The agent drives this env via vision in Chrome; the oracle
harness grades actions against `audit_log`.

## Surfaces shipped

- `/feed/` — main feed (left rail + centre + right rail)
- `/in/<handle>/` — profile (hero, About, Experience, Education, Skills, Activity)
- `/mynetwork/` — connections home + People you may know
- `/mynetwork/invitation-manager/` — received + sent invitations
- `/messaging/` — two-pane messaging (thread list + thread pane + composer)
- `/jobs/`, `/jobs/search/?keywords=`, `/jobs/view/<id>/` — jobs flow + Easy Apply
- `/login`, `/signup`, `/`

See `SCOPE.md` for the full in-scope spec and `FIDELITY.md` for the
per-section match matrix vs the offline ground-truth captures in
`_captured/`.

## Run locally

```
cd deploy/sim_envs/mantis_linkedin
ENV_ADMIN_TOKEN=test python -m uvicorn app.main:app --port 8090
```

Visit:
- http://127.0.0.1:8090/ — anonymous splash
- http://127.0.0.1:8090/feed/ — feed (acts as `user_00001` "Demo Mantis"
  when `ENV_REQUIRE_AUTH` is unset)
- http://127.0.0.1:8090/__env__/health — public; returns
  `{"ok": true, "seed": 42, "now": …, "boot_time": …}`

When `ENV_REQUIRE_AUTH=1`, log in as `demo@mantis.example` /
`mantis-demo` from `/login`.

## Docker

```
docker build -t mantis/sim-env-mantis-linkedin:latest deploy/sim_envs/mantis_linkedin
docker run --rm -p 8090:8080 \
    -e SEED=42 \
    -e FAKE_NOW=2026-06-08T09:00:00Z \
    -e ENV_ADMIN_TOKEN=$(python -c 'import secrets;print(secrets.token_urlsafe(32))') \
    mantis/sim-env-mantis-linkedin:latest
```

## Harness contract — `/__env__/*`

All routes (except `/health`) are gated on `X-Env-Admin: <ENV_ADMIN_TOKEN>`.

| Route              | Method | Purpose                                          |
| ------------------ | ------ | ------------------------------------------------ |
| `/health`          | GET    | Public; returns `{ok, seed, now, boot_time}`     |
| `/reset`           | POST   | Reseed DB to baseline + clear events             |
| `/seed`            | POST   | Reseed with `{seed: int}` body                   |
| `/clock`           | POST   | Advance FAKE_NOW with `{now: ISO8601}`           |
| `/oracle?task_id=` | GET    | Run grader for a task_id                         |
| `/state`           | GET    | Structured snapshot (counts + recent audit rows) |
| `/events`          | GET    | Process-level event log since `?since=`          |
| `/mutations`       | GET    | Mutation records since `?since=`                 |

## Oracles

| task_id                       | Goal                                                                                                |
| ----------------------------- | --------------------------------------------------------------------------------------------------- |
| `t01_connect_with_user`       | Demo user sends a Connect request to `user_00042` **with a note**.                                  |
| `t02_post_text_update`        | Demo user posts a text update containing at least one `#hashtag`.                                   |
| `t03_easy_apply_to_job`       | Demo user Easy-Applies to `job_00003` with phone + resume confirm.                                  |

Each oracle reads `audit_log` + DB only — never the agent transcript.
Aliases `t01`, `t02`, `t03` are accepted as a convenience.

## Smoke

```
ENV_ADMIN_TOKEN=test python scripts/smoke.py
```

Boots the app in-process, hits every in-scope page, dispatches every
oracle, then drives each oracle's happy path and confirms `passed=true`.

## Fidelity tracker

`FIDELITY.md` is the canonical source of truth for per-section match
status. Open it BEFORE touching templates / CSS so you don't relitigate
known close/exact rows.

The follow-up agent prompt lives in `FIDELITY_AGENT_PROMPT.md`.
