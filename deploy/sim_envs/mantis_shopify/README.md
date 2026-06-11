# mantis-shopify — Shopify Partners back-office mirror

Functional, high-fidelity mirror of the Shopify Partners dashboard
post-login, for use as a CUA training sim env. Mirrors every
sub-section EXCEPT `App distribution / /apps` (per scope).

## Sections mirrored

| Route                         | Section                |
| ----------------------------- | ---------------------- |
| `/`                           | Home (stores + KPIs + changelog) |
| `/stores`                     | Stores                 |
| `/sales`                      | Sales → Leads          |
| `/sales/referrals`            | Sales → Referrals      |
| `/sales/leads/new`            | Submit a lead          |
| `/catalogs`                   | Catalogs               |
| `/themes`                     | Themes                 |
| `/partner_directory`          | Partner Directory      |
| `/partner_directory/profile`  | Directory profile edit |
| `/pos`                        | Shopify POS marketing surface |
| `/docs/partner`               | Partner docs           |
| `/docs/product`               | Product docs           |
| `/support`                    | Support landing        |
| `/support/contact`            | Submit ticket          |
| `/payouts`                    | Payouts list           |
| `/payouts/<id>`               | Payout detail          |
| `/team`                       | Team (owners + staff)  |
| `/team/invite`                | Invite team member     |
| `/settings`                   | Partner account settings |
| `/notifications`              | Notifications list     |

## Oracles

10 programmable verifiers live in `app/oracles/`, each reading
DB state + `audit_log` (never the agent transcript). All run in
<100 ms.

| Task | Pass criteria summary |
| ---- | -------------------- |
| `t01_submit_plus_lead` | `lead_submitted` audit row with `product=plus` resolves to a `leads` row with merchant + email |
| `t02_invite_staff_member` | `staff_invited` audit row resolves to a users row with `status=invited` and a valid role |
| `t03_export_payouts_csv` | `payouts_export_requested` audit row exists |
| `t04_create_support_ticket` | `support_ticket_created` audit row + `tickets` row with subject + category + non-empty description |
| `t05_update_business_email` | `settings_updated` audit row w/ `field=business_email` and the partners row's `business_email` matches |
| `t06_view_payout_detail` | `payout_viewed` audit row that resolves to a `payouts` row |
| `t07_dismiss_emergency_banner` | `banner_dismissed` audit row with `target_id=emergency_contact` |
| `t08_directory_request_review` | `directory_review_requested` audit row + listing's `review_status` is `requested` or `received` |
| `t09_search_stores_filter` | `stores_filter_applied` audit row with non-empty `q` or `status` |
| `t10_submit_pos_referral` | `lead_submitted` audit row with `product=pos` resolves to a valid `leads` row |

Run them locally:

```bash
cd deploy/sim_envs/mantis_shopify
ENV_ADMIN_TOKEN=test python scripts/oracle_smoke.py
```

## Local dev

```bash
cd deploy/sim_envs/mantis_shopify
pip install -r requirements.txt
ENV_ADMIN_TOKEN=test python -m uvicorn app.main:app --port 8080
```

## Docker

```bash
docker build -t mantis/sim-env-mantis-shopify:latest .
docker run --rm -p 8080:8080 \
  -e ENV_ADMIN_TOKEN=$(python -c 'import secrets;print(secrets.token_urlsafe(32))') \
  mantis/sim-env-mantis-shopify:latest
```

## Daytona deploy

```bash
uv run python deploy/sim_envs/daytona_mantis_shopify.py
```

## Auth model

By default `ENV_REQUIRE_AUTH=0` — the canonical owner
(`owner@example.com` / `barada@example.com`) is silently treated as
the signed-in user. Set `ENV_REQUIRE_AUTH=1` to require the cookie
login (demo creds: `barada@example.com` / `password`).

## Harness contract

* `GET /__env__/health`
* `POST /__env__/reset`
* `POST /__env__/seed`
* `POST /__env__/clock`
* `GET /__env__/oracle?task_id=tNN`
* `GET /__env__/state`
* `GET /__env__/events?since=<ts>`
* `GET /__env__/mutations?since=<id>`

All non-health admin routes require header `X-Env-Admin: <ENV_ADMIN_TOKEN>`.
