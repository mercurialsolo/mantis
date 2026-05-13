# Simulated Environments — Specification

Status: proposal, v1
Owner: TBD
Tracking: see GitHub epic linked from this PR

## Why

Today, mantis plans target live external sites (boattrader, fb, zillow, staff-crm). That gives realistic DOM but two problems for benchmarking:

1. **Non-determinism** — sites change, anti-bot defenses fluctuate, seed data drifts.
2. **No oracle** — we can't reliably grade "did the agent complete the task?" without a ground-truth state we control.

Self-hosted simulated environments give us deterministic DOM + seeded state + a server-side oracle. They run locally (or in CI) and let us measure agent progress over time on the same task with the same starting conditions.

These are not toys: each env carries enough seed data and edge-case messiness to expose real failure modes (duplicate records, stale info, partial form state, paginated tables, modal stacking, etc.).

## Scope (v1)

Four environments, one per major interaction archetype:

1. **mantis-crm** — dense tables, record detail, bulk operations on dirty data
2. **mantis-helpdesk** — queue triage, threaded reply composition, macros
3. **mantis-shop** — catalog/funnel navigation, cart/checkout, admin order ops
4. **mantis-docs** — rich-text editing, comments, search-then-cite

Deferred to v2: travel booking, banking, calendar/email, BI dashboard, spreadsheet.

## Shared harness contract

Every env is a single Docker image exposing:

| Surface | Path | Purpose |
| --- | --- | --- |
| Web UI | `:PORT/` | The SPA the agent navigates |
| Reset | `POST /__env__/reset` | Restore DB to seed; idempotent |
| Seed | `POST /__env__/seed` body `{seed: int}` | Re-seed with a specific deterministic seed |
| Clock | `POST /__env__/clock` body `{now: iso8601}` | Freeze the in-app clock |
| Oracle | `GET /__env__/oracle?task_id=<id>` | Returns `{passed: bool, reasons: [...], diff: {...}}` |
| State dump | `GET /__env__/state` | Full DB snapshot as JSON (for debugging) |
| Health | `GET /__env__/health` | Liveness + seed version |

All `__env__/*` endpoints are bound to localhost-only and stripped at build time for any future public deployment. They MUST NOT be visible in the agent's DOM.

### Determinism rules

- Every env boots from `SEED` env var (default `42`); same seed → identical DB rows + IDs.
- Clock is frozen by default at `FAKE_NOW=2026-01-15T09:00:00Z`; relative-time plans ("yesterday") work without flake.
- No outbound network from the container.
- No background jobs that mutate state (no schedulers, no auto-decay).
- IDs are deterministic: `contact_00001`, not UUIDs.

### Auth

Single pre-authenticated user (cookie pre-set). Login flow is not what we're testing; it's a covered case in other plans (staff-crm). Multi-user / RBAC plans land in v2.

### DOM contract

- All interactive elements carry stable `data-testid` attributes.
- `data-testid` is used by the **oracle** for state assertions, NOT by the agent — the agent must navigate visually (or via accessibility tree). Screenshots and the DOM-without-testids tree are what the agent perceives.
- Real-world messiness is preserved: a11y labels are sometimes missing, button text is sometimes ambiguous, modals overlap. We are NOT cleaning up the UI to make agents pass; we are reproducing realistic mess.

### Oracle interface

Each plan declares a `task_id` (string). When the plan finishes, the harness calls `GET /__env__/oracle?task_id=<id>`. The oracle returns:

```json
{
  "passed": true,
  "score": 1.0,
  "reasons": ["contact_01234 has tag 'reengage'", "no other contacts mutated"],
  "diff": { "contacts_changed": ["contact_01234"], ... }
}
```

Oracles read **server-side state**, not the agent's transcript. This is the whole point: we grade the outcome the user actually wanted, not whether the agent's narration matched.

### Telemetry

The env emits one event-log line per HTTP request to `/__env__/events.jsonl`, captured by mantis's existing trace exporter (`src/mantis_agent/gym/trace_exporter.py`). Combined with the agent's screenshot trajectory, this gives us full post-hoc replay.

---

## How envs wire into mantis (today's runtime)

A new env hooks into existing mantis primitives — no executor or planner changes required for v1.

```
┌──────────────────────────┐         ┌────────────────────────────┐
│  plans/<env>/*.json      │  ───→   │  MicroPlanRunner           │
│  (existing step schema)  │         │  (gym/micro_runner.py)     │
└──────────────────────────┘         └──────────────┬─────────────┘
                                                    │
                              ┌─────────────────────┼──────────────────────┐
                              ▼                     ▼                      ▼
                     ┌──────────────────┐  ┌────────────────┐    ┌──────────────────┐
                     │  PlaywrightEnv   │  │  SiteConfig    │    │  trace_exporter  │
                     │  (gym/           │  │  (per-env URL  │    │  → events.jsonl  │
                     │   playwright_    │  │   patterns)    │    └─────────┬────────┘
                     │   env.py)        │  └────────────────┘              │
                     └────────┬─────────┘                                   │
                              │ HTTP                                        │
                              ▼                                             ▼
                     ┌──────────────────────────────────────────────────────────┐
                     │  Docker: mantis-<env>  (SPA + DB + oracle endpoints)     │
                     │                                                          │
                     │   /__env__/reset, /__env__/seed, /__env__/oracle, etc.   │
                     └──────────────────────────────────────────────────────────┘
```

### Concrete wiring points

1. **Plan files** — `plans/<env>/<task>.json`, same step schema as `plans/example/extract_listings.json` (`navigate`, `click`, `fill_field`, `extract_data`, etc.). The first `navigate` step points at `http://localhost:<PORT>/` (the env container) instead of an external URL. New step types are NOT required for v1.

2. **SiteConfig** — each env contributes one `SiteConfig` (`src/mantis_agent/site_config.py`) describing its URL shape (list vs detail patterns) so `MicroPlanRunner` heuristics (pagination, detail-page detection, listing dedup) still work. Pattern: add a `default_<env>()` classmethod next to `default_boattrader()`.

3. **Env runner** — a new `scripts/env_up.py` (small wrapper) takes `--env mantis-crm --seed 42 --task <id>`, brings the container up with the right `SEED`/`FAKE_NOW`, waits for `/__env__/health`, returns the URL. The existing `MicroPlanRunner` invocation gets that URL via the plan's first navigate step (templated).

4. **Grading hook** — after `MicroPlanRunner` finishes, a new `grade_run()` calls `GET /__env__/oracle?task_id=<id>` and writes `oracle.json` next to the existing `outputs/<run_id>/trace.jsonl`. This adds a `grading` block to `RunReport` (`gym/run_reporter.py`) — purely additive.

5. **Trace correlation** — the env's `events.jsonl` is pulled at teardown and merged into the trace exporter output keyed by timestamp. Lets us see "the agent clicked X at t=12.4s, the server recorded mutation Y at t=12.5s".

6. **CLI** — `mantis plan run plans/mantis-crm/T01_tag_reengage.json --env mantis-crm` (dispatch already in `main.py:119` → `cli.py`). The `--env` flag triggers steps 3-5.

### What does NOT change

- Brain / planner / executor — unchanged. The agent sees a webpage; it doesn't know it's simulated.
- Step schema — unchanged. Existing step types cover everything in v1.
- Trace format — unchanged shape; oracle results are an additive sibling file.
- Routing policy / SoM dispatch — unchanged.

### What DOES need a small change

- `RunReport` gains an optional `grading: { passed: bool, score: float, reasons: [...], task_id: str }` block.
- `cli.py` gains a `--env <name>` flag that triggers env_up/teardown + grading.
- `SiteConfig.default_<env>()` per environment.
- A `benchmarks/sim_envs.py` runner for batch eval across all plans in an env (parallel to `benchmarks/osworld_chrome.py`).

That's it. The bulk of the work is the envs themselves; the mantis integration is small and additive.

---

## 1. mantis-crm

**Purpose:** dense tables + record detail + bulk operations on dirty data. Target archetype: Salesforce / HubSpot CRM.

**Entities (seed):**
- `contacts` — 50,000 rows. Fields: name, email, phone, company, lifecycle_stage, owner_id, tags (multi), created_at, last_activity_at, custom_fields (jsonb), source.
- `companies` — 8,000. Fields: name, domain, size_band, industry, arr_band, parent_company_id (some self-joins).
- `deals` — 12,000. Fields: contact_id, company_id, stage, amount, expected_close, owner_id.
- `activities` — 200,000. Polymorphic to contact/deal. Types: call, email, note, meeting. Body text. occurred_at.
- `lists` — 50 saved views (filter definitions + manual lists).
- `users` — 12 sales reps.

**Seed messiness (deliberate):**
- 8% duplicate contacts (same email, slight name variant, different company assignment).
- 12% missing phone, 3% malformed email.
- 200 contacts assigned to a child company when activity history points at the parent.
- ~5% of deals have `expected_close` in the past but still in active stages.
- Owner_id occasionally points at a deactivated user.

**Screens:**
- Contact list — sortable virtualized table, 50/page, filter sidebar, saved views dropdown.
- Contact detail — profile pane, activity timeline (paginated), related deals, related companies, inline edit on every field.
- Company detail — mirror of contact detail.
- Deal pipeline — kanban by stage, drag-drop between columns.
- Bulk edit modal — multi-select rows → assign owner, add/remove tag, change stage, delete.
- Search — global, fuzzy on name/email/company.
- Reports — 3 prebuilt charts (deals by stage, contacts by source, activity by user) with click-through to filtered list.

**Mutations:** edit any field, add activity, change deal stage, merge contacts (one survivor, history preserved), bulk-assign, soft-delete with 24h undo window.

**Hard cases:**
- Find contact by partial info ("VP Eng at Acme, lives in NYC") with 50k rows and no perfect match.
- Merge 3 duplicates correctly — survivor must inherit all activities, no orphan rows.
- Move 20 deals across stages by criteria expressed in natural language.
- State compounding: filter + sort + pagination + bulk-select interactions.

**Oracle:** DB diff on target rows; activity-log assertions for merges; "no unintended row mutated outside the target set".

**Example plans (task_ids):**

- `T01_tag_reengage` — Tag every contact with no `last_activity_at` in 90 days at companies with `arr_band >= "1M+"` as `reengage`.
- `T02_merge_acme_dupes` — Merge duplicate contacts for `@acme.com`; keep the contact with the most activities; lose no history.
- `T03_at_risk_deals` — Move all deals in "Proposal" stage with `expected_close` older than 30 days to "At Risk".
- `T04_add_meeting_note` — Add a meeting note dated yesterday to Sarah Chen's record with body "discussed Q3 expansion".
- `T05_pipeline_review` — Export to a saved list every deal owned by `user_05` over $50k closing this quarter.

---

## 2. mantis-helpdesk

**Purpose:** queue triage + threaded reply composition + macros / escalation chains. Target archetype: Zendesk / Intercom.

**Entities (seed):**
- `tickets` — 15,000 total, 4,000 open. Fields: subject, body, requester_id, assignee_id, status (new/open/pending/solved/closed), priority, channel (email/chat/form), tags (multi), sla_breach_at, created_at, group_id.
- `replies` — 60,000. ticket_id, author_id, body, visibility (public/internal), attachments.
- `users` — 5,000 requesters + 30 agents.
- `macros` — 40 saved replies with template vars (`{{requester.first_name}}`, etc.).
- `triggers` — 12 conditional routing rules. Read-only in v1 — the agent must respect them (e.g., changing assignee on a billing ticket may auto-revert).
- `groups` — 6 (billing, technical, success, ops, sales, ext-vendor).

**Seed messiness:**
- 6% mislabeled priorities (urgent flagged low, vice versa).
- ~200 tickets contain PII (SSN-shaped strings, credit-card-shaped) in the body — must be redacted before any public reply.
- ~400 tickets within 2h of SLA breach; ~150 already breached.
- 30 tickets are duplicate reports of the same outage from different requesters (must merge).
- Several multi-language tickets (es, fr, de) — macros are English only.

**Screens:**
- Inbox — queue with filters (status, assignee, group, priority, tag, SLA-soon). Saved views.
- Ticket detail — thread + composer + side panel (requester profile, requester's other tickets, suggested related tickets).
- Composer — rich text, attach, macro insert (search + apply, vars get substituted), internal-note toggle, CC/BCC.
- Bulk actions — multi-select → assign, tag, change status, close, merge.
- Reports — open by group, SLA breach trend.

**Mutations:** reply (public/internal), change status/priority/assignee/group/tags, apply macro, merge tickets, escalate (links to another ticket).

**Hard cases:**
- Pick the right macro for a multi-issue ticket — applying the wrong one fails the oracle.
- Avoid replying publicly on an internal-only thread (oracle treats this as a critical failure).
- Identify a duplicate cluster and merge correctly; survivor inherits all replies.
- Triage 50 new tickets by SLA + priority + content (not just by created_at).

**Oracle:** ticket state diff; macro substitution correctness; **PII regex check on every public reply created during the run** (any leak = automatic fail); merge survivor / loser correctness.

**Example plans:**

- `T01_triage_inbox` — Route all `new` tickets: billing keywords → billing group, technical keywords → eng, others stay general. Set priority `high` if body mentions "outage" or "down".
- `T02_shipping_macro` — Apply the "shipping delay" macro to every ticket mentioning an order number `12xxxx`.
- `T03_merge_outage_dupes` — Merge duplicate reports of the login outage; reply on the survivor with the status-page link.
- `T04_sla_rescue` — Find open tickets within 2h of SLA breach and reassign to an agent in the same group with `<3` open tickets.
- `T05_redact_and_reply` — Reply to ticket #4421 (which contains PII) with the requested information, without leaking the PII into the public body.

---

## 3. mantis-shop

**Purpose:** catalog/funnel navigation + multi-step form state + cart/checkout. Target archetype: Shopify storefront + admin.

**Entities (seed):**
- `products` — 2,000. Fields: title, sku, base_price, variants (size/color matrix), images, description (markdown), category, inventory_per_variant, ratings, on_sale (bool).
- `collections` — 40. Rule-based ("under $50", "outerwear-women") + manual.
- `orders` — 5,000 historical. Line items, address, payment_method_id (mocked), status (paid/fulfilled/refunded/cancelled).
- `customers` — 3,000 with saved addresses + mocked payment methods.
- `coupons` — 15. Types: pct off, $ off, BOGO. Some exclusive, some stackable. Expiration dates.

**Seed messiness:**
- ~5% of variants out of stock.
- Some variants region-locked (won't ship to Brooklyn but will to LA).
- ~30% of products on sale (sale + original both displayed).
- ~10 products with broken/missing images (intentional realism).
- 2 coupons that look stackable in the UI but the server rejects together.

**Screens (storefront):**
- Home, category, search, PDP (variant picker enforces stock + region), cart, checkout (4 steps: address → shipping method → payment → review). Cart persists across nav.

**Screens (admin):**
- Orders list (filter by status, date, customer), order detail (refund/fulfill/cancel), product editor (variants, inventory adjustment), coupon creator/editor.

**Mutations:**
- Storefront: add to cart, update qty, remove, apply coupon, place order.
- Admin: refund (partial or full), fulfill (mark shipped), edit product (price, inventory, description), create/disable coupon.

**Hard cases:**
- PDP variant picker enforces constraint matrix — agent must pick a valid combination.
- Coupon stacking rules — applying an exclusive coupon disables a previously applied one.
- Address validation rejects invalid ZIPs (the form re-renders, focus moves).
- Cart recovery after the agent navigates away and returns.

**Oracle:** order row exists with expected items + variant + total + coupon; inventory decremented exactly; admin actions (refund, fulfill) reflected in order state and customer-visible status.

**Example plans:**

- `T01_buy_jacket` — Buy a size-M blue jacket under $100, ship to the seeded Brooklyn address, apply coupon `SPRING15`.
- `T02_refund_line_item` — Refund line item 2 of order `#4421` with reason "damaged"; mark notify-customer true.
- `T03_create_coupon` — Create a 20%-off coupon for category `outerwear-women` expiring 2026-02-15, max 100 uses.
- `T04_export_bogo_orders` — Find all orders in the last 7 days using coupon `BOGO`; export to a saved view.
- `T05_inventory_adjust` — Increase inventory of `sku=TEE-BLK-M` by 50 with reason "restock from warehouse".

---

## 4. mantis-docs

**Purpose:** rich-text editing + commenting/threading + search-then-cite. Target archetype: Notion / Confluence.

**Entities (seed):**
- `pages` — 500. Tree-structured. Fields: title, parent_id, owner_id, permissions (private/workspace/public), body (block-tree).
- `blocks` — ~15,000 atomic. Types: paragraph, heading (1-3), bulleted_list_item, numbered_list_item, table, code, image, embed, callout, divider. Every block has a stable id.
- `comments` — 3,000. Anchored to a block. Threaded (reply chains). Resolved / unresolved.
- `revisions` — autosave history per page, ~10 per page on average.
- `users` — 20.
- `workspaces` — 3.

**Seed messiness:**
- 12 pages contain stale info that newer pages contradict (e.g., `/handbook/expenses-v1` says "submit weekly", `/handbook/expenses` says "monthly").
- 30 pages have unresolved comments where the underlying ask has actually been addressed in the page body (oracle checks both).
- ~40 broken internal links.
- 5 pages have deeply nested block trees (lists of lists of tables).

**Screens:**
- Page-tree sidebar — collapsible, drag to reorder.
- Editor — block-based; click between blocks creates a new paragraph; slash menu (`/heading`, `/table`, `/code`) inserts typed blocks at cursor.
- Comments side panel — per-block thread, reply, resolve, reopen.
- Search — full-text + filters (in workspace, with mentions, by author).
- Revision history — timeline of autosave snapshots + diff view.
- Share / permissions modal — per-page.

**Mutations:** create/edit/move/delete blocks (block-tree level, not text-level); comment/reply/resolve/reopen; move/rename/delete pages; change permissions; revert to revision.

**Hard cases:**
- Insert a table at a specific position — block-level insertion semantics, not "type a tab-separated string into a paragraph".
- Preserve heading hierarchy when restructuring (H3 → H2 promotion must demote nothing accidentally).
- Resolve only comments whose underlying concern is addressed in the page body — oracle reads comment + current block content and confirms.
- Disambiguate between two contradicting pages by metadata (newer `updated_at`, or tag `canonical`).

**Oracle:** page block-tree JSON diff with semantic equivalence (paragraph text matches modulo whitespace); comment state diff with author + resolved-by; revision count delta; link validity post-edit.

**Example plans:**

- `T01_q3_one_pager` — Create a 1-pager on Q3 launch at `/launches/Q3` using bullets from `/planning/Q3-outline`; add a table of milestones (date, owner, status); share with the design workspace.
- `T02_resolve_addressed_comments` — On `/engineering/auth-rewrite`, resolve every unresolved comment whose ask is already reflected in the current page body; on the rest, reply asking for status.
- `T03_pick_canonical_policy` — Two pages describe remote-work expenses; pick the canonical one (tag `canonical` or, if absent, newer); add a callout summarizing it to `/onboarding/welcome`.
- `T04_restructure_handbook` — In `/handbook/sales`: promote the H3 `pricing` to H2; move all H3s under "discounts" beneath it; delete the section titled "legacy plans" (3 blocks).
- `T05_find_and_cite` — Find our current data-retention policy; quote the retention duration in a new block at the top of `/onboarding/security` with a backlink.

---

## Build order

We build in depth, not breadth. The CRM goes first and proves the harness (Docker shape, seed loader, oracle, grading hook, telemetry merge). Once the CRM has 5 plans passing end-to-end with reliable oracle scoring, helpdesk and shop follow in parallel. Docs is last because rich-text/block-tree oracles are the hardest to get right and we want the harness battle-tested first.

Milestones (rough):

1. mantis-crm container + harness contract + 5 plans + oracle + grading hook in `RunReport`.
2. CLI `--env` flag, batch runner in `benchmarks/sim_envs.py`.
3. mantis-helpdesk + mantis-shop (parallel).
4. mantis-docs.
5. v2 envs (travel, banking, calendar/email, BI).

Each env ships its own PR, references this spec, and includes seed-determinism tests + at least one plan that exercises every screen.
