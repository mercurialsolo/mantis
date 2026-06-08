# mantis_fiverr

High-fidelity synthetic mirror of **fiverr.com** — the agent-training
sim env for the freelance gig marketplace surface.

## What this mirrors

A buyer-facing slice of Fiverr:

- Discovery: home, search, category landing
- Gig detail with three-tier package picker (Basic/Standard/Premium)
- Checkout (no real payment)
- Messaging (inbox + thread)
- Orders (list + detail + review submission)
- Auth (email/password + signup)

Out-of-scope: real images/brand assets, ads/analytics, OAuth/social
login, real payment processing, websockets, A/B variants, mobile
responsive, multi-currency, multi-locale.

See `SCOPE.md` for the full list.

## How to run

### Local (Docker)

```bash
cd deploy/sim_envs/mantis_fiverr
docker build -t mantis-fiverr .
docker run --rm -p 8080:8080 -e ENV_ADMIN_TOKEN=test mantis-fiverr
```

Then `curl http://127.0.0.1:8080/__env__/health` and browse to
`http://127.0.0.1:8080/`.

### Local (uvicorn, dev)

```bash
cd deploy/sim_envs/mantis_fiverr
pip install -r requirements.txt
ENV_ADMIN_TOKEN=test python -m uvicorn app.main:app --port 8080 --reload
```

### Smoke test

```bash
cd deploy/sim_envs/mantis_fiverr
ENV_ADMIN_TOKEN=test python scripts/smoke.py
```

Drives every in-scope page + happy-path for each of the three
oracle tasks. Green = the harness contract is intact.

## Harness contract

- `GET /__env__/health` — public, `{ok, seed, now, boot_time, gigs}`
- `POST /__env__/reset` — admin; reseeds DB to baseline
- `POST /__env__/seed` — admin; body `{seed:int}`
- `POST /__env__/clock` — admin; body `{now:ISO8601}`
- `GET /__env__/oracle?task_id=…` — admin; dispatches to the matching
  grader. Returns canonical `{passed, score, task_id, reasons, diff}`
- `GET /__env__/state` — admin; counts + recent audit_log
- `GET /__env__/events?since=ts` — admin; in-process event log
- `GET /__env__/mutations?since=N` — admin; the audit_log slice since
  the given id

All admin routes gate on `X-Env-Admin: <ENV_ADMIN_TOKEN>`.

`ENV_REQUIRE_AUTH=1` flips authentication on for `/inbox`, `/orders`,
`/checkout`. Default off so oracle contracts grade against the
canonical buyer (`buyer_00001`).

## Oracle tasks

- **`t01_order_basic_logo`** — buyer_00001 orders gig_00001 at Basic
  tier. Verifies subtotal + service_fee + total + order_placed audit.
- **`t02_message_seller_then_order`** — buyer_00001 messages
  gig_00001's seller, THEN orders. Verifies the audit-log ordering.
- **`t03_leave_5star_review`** — buyer reviews `order_00007` with 5
  stars. Verifies the reviews row + gig avg-rating recompute + audit
  row.

## Fidelity tracker

See `FIDELITY.md` for the section-by-section match matrix and open
follow-ups. The build methodology is in
`../mantis_boattrader/FIDELITY_BUILD_FROM_SCRATCH_PROMPT.md`; the
gap-fix playbook for follow-up turns is in
`FIDELITY_AGENT_PROMPT.md`.

## Repo layout

```
mantis_fiverr/
├── Dockerfile               # python:3.11-slim + uvicorn :8080
├── requirements.txt
├── README.md
├── SCOPE.md                 # in-scope pages + interactions
├── FIDELITY.md              # section match matrix + iteration log
├── FIDELITY_AGENT_PROMPT.md # gap-fix playbook
├── _captured/               # ground-truth corpus per page
│   ├── home/{notes.md, …}
│   ├── search_gigs/, gig_detail/, checkout/, inbox/, orders/,
│   ├── category/, login/, signup/
├── app/
│   ├── main.py              # FastAPI factory
│   ├── db.py                # SQLite schema + audit_log helpers
│   ├── seed.py              # deterministic seed
│   ├── auth.py              # signed-cookie sessions
│   ├── routes/
│   │   ├── env_admin.py     # /__env__/*
│   │   ├── auth_routes.py   # /login, /signup, /logout
│   │   ├── storefront.py    # /, /search/gigs, /categories/<slug>
│   │   ├── gig.py           # /<username>/<gig-slug> (catch-all)
│   │   ├── checkout.py      # /checkout/<gig-id>
│   │   ├── inbox.py         # /inbox + /inbox/<thread>
│   │   ├── orders.py        # /orders + /orders/<id>
│   │   └── assets.py        # /assets/*.svg (procedural)
│   ├── oracles/
│   │   ├── t01_order_basic_logo.py
│   │   ├── t02_message_seller_then_order.py
│   │   └── t03_leave_5star_review.py
│   ├── templates/           # Jinja2
│   │   ├── base.html, _gig_card.html
│   │   ├── home.html, search.html, category.html
│   │   ├── gig_detail.html, checkout.html
│   │   ├── inbox.html, orders.html, order_detail.html
│   │   ├── login.html, signup.html
│   └── static/site.css      # hand-rolled (~700 lines)
└── scripts/smoke.py         # in-process end-to-end smoke
```

## Deterministic IDs

Same `SEED` → identical IDs across runs:

- `buyer_00001`..`buyer_00020`
- `seller_00001`..`seller_00030`
- `gig_00001`..`gig_00030`
- `order_00001`..`order_00015` (1–8 completed, 9–11 delivered,
  12–15 active)
- `review_00001`..`review_00005` (skips `order_00007` deliberately
  so `t03` has a target)
- `conv_00001`..`conv_00004`
