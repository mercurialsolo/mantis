# Sim-env harness — wiring plans to simulated environments

Status: v1 (#336 — landed by this PR)
Reference contract: [SPEC.md](SPEC.md) §"Shared harness contract"

The harness is the integration layer between `mantis plan run` and the
per-env containers that ship under #332+. It lets one plan JSON run
against a local Docker env or a Modal-hosted env with no plan edits —
the agent's view is always "navigate to a URL".

## Quick start

Boot the stub env locally, run a smoke plan against it, and read the
oracle verdict:

```bash
# Optional: get a handle file you can keep across shells.
uv run python scripts/env_up.py start --env stub

# One-shot plan-run-against-env (handles start/stop for you):
uv run mantis plan run examples/sim_envs/stub_T01_passthrough.json \
    --env stub --runtime local \
    --endpoint <BRAIN_URL> \
    --output-dir outputs/harness-smoke
cat outputs/harness-smoke/oracle.json
```

## What `--env <name>` actually does

When `--env` is set, the CLI does five things on top of the normal run:

1. **Boot** — picks a runtime backend (`local`, `modal`, or the
   reserved `e2b`), calls `backend.start(<env>, seed=…, now=…)`,
   waits for `/__env__/health` to return `{"ok": true}`. Health-poll
   timeouts surface a clear `TimeoutError` rather than hanging.
2. **Template** — substitutes `{{ENV_URL}}` in the plan JSON for the
   env's base URL. Every string-shaped field is touched (intents,
   `params.url`, nested params).
3. **Run** — hands the templated plan to `MicroPlanRunner` as usual.
   The runner doesn't know it's running against a sim env.
4. **Grade** — after the runner exits, calls
   `GET /__env__/oracle?task_id=<id>` with the admin token. The
   verdict lands as `oracle.json` next to `result.json`; `result.json`
   gets an additive `grading` block.
5. **Tear down** — calls `backend.stop(handle)` in a `finally`. Pass
   `--keep` to leave the env running for debugging.

`task_id` comes from a top-level `"task_id"` field in the plan JSON.
Plans without one fall back to the plan filename's stem.

## Two route surfaces, one container

Every env exposes:

| Surface | Path | Audience | Auth |
| --- | --- | --- | --- |
| Web UI | `:PORT/` | Agent | Pre-seeded session cookie (single user) |
| Harness | `:PORT/__env__/*` | Harness only | `X-Env-Admin: <token>` |

The admin token is generated per run by the harness and passed to the
container as an env var (`ENV_ADMIN_TOKEN`). The agent's browser
context never sees it. `/__env__/health` is intentionally open so
health checks can run un-authenticated; everything else 401s without
the header.

See `tests/test_admin_token_isolation.py` for the contract assertions.

## Runtimes

| Runtime | Cold start | Use it when | Default for |
| --- | --- | --- | --- |
| `local`  | n/a       | Laptop dev, debugging | Local CLI |
| `modal`  | 5–15 s    | CI / benchmark runs   | Modal-resident CLI |
| `e2b`    | reserved  | High-volume batch (#336 §"Hosting") | — |

`--runtime` default is `local` unless `MODAL_TASK_ID` /
`MODAL_FUNCTION_NAME` is set in the environment, in which case it's
`modal`. Override explicitly when needed.

The `local` backend falls back to a Python subprocess running the
in-package stub env when no Docker image is registered for the env
name. Real envs land an entry in
`mantis_agent.sim_envs.local._image_for_env` (or override via
`MANTIS_SIM_ENV_IMAGE_<NAME>`).

## Adding a real env

Each env PR (#332+) lands four things on top of this harness:

1. A Dockerfile + container image registered in `_image_for_env`.
2. A `deploy/sim_envs/<env>.py` exporting a `modal.App` named
   `mantis-sim-env-<env>`.
3. A `SiteConfig.default_<env>()` classmethod.
4. ≥5 plans under `plans/<env>/` with explicit `task_id` fields.

Nothing in the harness needs to change for those PRs.

## Batch eval

```bash
uv run python -m benchmarks.sim_envs --env mantis-crm --runtime modal \
    --endpoint <BRAIN_URL> --max-workers 4
```

Runs every plan in `plans/<env>/` against a fresh env instance per
plan, writes per-plan dirs + `bench_summary.json`, prints a one-line
oracle pass/fail table. Exit code 0 iff every plan passed the oracle.

## mantis-crm (first real env)

The first env landed on this harness is **mantis-crm** (#332) — a
Salesforce/HubSpot-shape CRM with 50k contacts, 12k deals, 200k
activities, tasks, notes, email templates with merge fields, lifecycle
transitions, forecast and audit log.

```bash
# Local Docker
docker build -t mantis/sim-env-mantis-crm:latest deploy/sim_envs/mantis_crm
uv run mantis plan run examples/sim_envs/mantis_crm/T01_tag_reengage.json \
    --env mantis-crm --runtime local --endpoint <BRAIN_URL>

# Modal
modal secret create mantis-sim-env-mantis-crm-secrets \
    ENV_ADMIN_TOKEN=$(python -c 'import secrets;print(secrets.token_urlsafe(32))')
uv run modal deploy deploy/sim_envs/modal_mantis_crm.py
uv run mantis plan run examples/sim_envs/mantis_crm/T01_tag_reengage.json \
    --env mantis-crm --runtime modal --endpoint <BRAIN_URL>
```

The env ships with five plans + per-task oracles (`T01_tag_reengage`,
`T02_merge_acme_dupes`, `T03_at_risk_deals`, `T04_add_meeting_note`,
`T05_pipeline_review`). Each oracle reads server-side state directly,
not the agent's transcript.

## mantis-helpdesk (queue triage + threaded reply + PII-strict oracle)

The second real env on the harness is **mantis-helpdesk** (#333) — a
Zendesk/Intercom-shape helpdesk with 15k tickets (4k open), 60k
threaded replies, 30 agents across 6 groups, 40 macros, and 12
read-only triggers. The PII detector in `app/pii.py` strictly enforces
no SSN-shape and no Luhn-valid credit-card-shape strings in any public
reply created during the run; oracles return score 0 on any leak.

```bash
# Local Docker
docker build -t mantis/sim-env-mantis-helpdesk:latest deploy/sim_envs/mantis_helpdesk
uv run mantis plan run examples/sim_envs/mantis_helpdesk/T01_triage_inbox.json \
    --env mantis-helpdesk --runtime local --endpoint <BRAIN_URL>

# Modal
modal secret create mantis-sim-env-mantis-helpdesk-secrets \
    ENV_ADMIN_TOKEN=$(python -c 'import secrets;print(secrets.token_urlsafe(32))')
uv run modal deploy deploy/sim_envs/modal_mantis_helpdesk.py
uv run mantis plan run examples/sim_envs/mantis_helpdesk/T01_triage_inbox.json \
    --env mantis-helpdesk --runtime modal --endpoint <BRAIN_URL>
```

The env ships with five plans + per-task oracles (`T01_triage_inbox`,
`T02_shipping_macro`, `T03_merge_outage_dupes`, `T04_sla_rescue`,
`T05_redact_and_reply`). The composer route rejects public replies on
internal-only threads; the bulk-assign route applies a billing-group
trigger that auto-reverts assignees outside the billing group.

## mantis-shop (Shopify-shape storefront + admin)

The third env on the harness is **mantis-shop** (#334) — a
Shopify-shape storefront with PDP variant pickers, multi-step
checkout, and an admin surface for orders/products/coupons. 2,000
products with a size/color variant matrix, 5,000 historical orders,
3,000 customers + saved addresses + mocked payment methods, 40
collections, 15 coupons (pct / amount / BOGO), and a deliberate
"looks-stackable-but-server-rejects" coupon pair the agent must
respect.

```bash
# Local Docker
docker build -t mantis/sim-env-mantis-shop:latest deploy/sim_envs/mantis_shop
uv run mantis plan run examples/sim_envs/mantis_shop/T01_buy_jacket.json \
    --env mantis-shop --runtime local --endpoint <BRAIN_URL>

# Modal
modal secret create mantis-sim-env-mantis-shop-secrets \
    ENV_ADMIN_TOKEN=$(python -c 'import secrets;print(secrets.token_urlsafe(32))')
uv run modal deploy deploy/sim_envs/modal_mantis_shop.py
uv run mantis plan run examples/sim_envs/mantis_shop/T01_buy_jacket.json \
    --env mantis-shop --runtime modal --endpoint <BRAIN_URL>
```

The env ships with five plans + per-task oracles (`T01_buy_jacket`,
`T02_refund_line_item`, `T03_create_coupon`, `T04_export_bogo_orders`,
`T05_inventory_adjust`). Server-side constraints (variant stock,
region locks, coupon stacking, ZIP validation) are all enforced in
FastAPI handlers — the agent cannot bypass them via JS-only checks
or DOM manipulation.

## Modal deploy of the stub

```bash
# One-time: create the secret holding the admin token.
modal secret create mantis-sim-env-stub-secrets \
    ENV_ADMIN_TOKEN=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')

# Deploy the stub app.
uv run modal deploy deploy/sim_envs/modal_stub.py

# Use it.
export MANTIS_STUB_ENV_ADMIN_TOKEN=<the-value-you-just-generated>
uv run mantis plan run examples/sim_envs/stub_T01_passthrough.json \
    --env stub --runtime modal --endpoint <BRAIN_URL>
```
