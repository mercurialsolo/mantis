# Sending plans

Once you have an authenticated session, every plan submission lands on `POST /v1/predict` with one of the four plan-shape fields. Pick the one that fits your data:

```jsonc
{
  "detached": true,                                  // (default) async with run_id
  "task_suite":  { ... },                            // OR
  "task_file":   "tasks/crm/crm_tasks.json",     // OR
  "micro":       "plans/example/...json",            // OR
  "plan_text":   "Plain English description",        //

  "profile_id":        "alice-prod",                 // (#341) Chrome user-data-dir; sticky across plan revisions
  "workflow_id":       "marketplace-listings-v3",    // (#341) checkpoint id; rotate when the plan changes
  // "state_key": "legacy-single-field",            // pre-#341 callers; routes to BOTH on the server for back-compat
  "resume_state":      false,                        // continue from last checkpoint at workflow_id
  "max_cost":          2.0,                          // clamped to tenant cap
  "max_time_minutes":  20,
  "proxy_city":        "Miami",                      // optional IPRoyal geo override
  "proxy_state":       "FL",
  "record_video":      false,                        // see Recordings page
  "video_format":      "mp4",
  "video_fps":         5,
  "live_viewer":       false,                        // surfaces ``viewer_url`` on action=status (holo3 only)
  "callback_url":      "https://my-webhook",         // overrides tenant default
  "compute_backend":   "computer_plane"              // OR "browser_use_plane" — see Picking a compute plane below
}
```

Field reference is on the [Reference / HTTP API](../api.md#post-v1predict) page.

## Decision tree

```
Have a stable workflow you'll run many times?
  ├─ yes → micro-plan via `micro: "plans/<domain>/<workflow>.json"`
  └─ no  → one-shot or rapidly-changing?
            ├─ yes → `plan_text` — server decomposes via Claude
            └─ no  → multi-task suite style?
                     ├─ yes → `task_suite` (multi-task dict)
                     └─ no  → `task_suite` with a flat micro-plan list
```

## Examples

=== "micro (high-volume reliable)"

    ```bash
    curl -X POST "$ENDPOINT/v1/predict" \
      -H "Authorization: Api-Key $BASETEN_API_KEY" \
      -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
      -H "Content-Type: application/json" \
      -d '{
        "detached": true,
        "micro": "plans/example/extract_listings.json",
        "profile_id": "marketplace-prod",
        "workflow_id": "marketplace-listings-v1",
        "max_cost": 2,
        "max_time_minutes": 20
      }'
    ```

=== "plan_text (one-shot)"

    ```bash
    curl -X POST "$ENDPOINT/v1/predict" \
      -H "Authorization: Api-Key $BASETEN_API_KEY" \
      -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
      -H "Content-Type: application/json" \
      -d '{
        "detached": true,
        "plan_text": "Go to a marketplace listings site, filter to private sellers above $35,000 in Florida, extract listing details for the first 3 listings.",
        "profile_id": "ad-hoc",
        "workflow_id": "ad-hoc-marketplace-1",
        "max_cost": 2
      }'
    ```

=== "task_suite (multi-task)"

    ```bash
    curl -X POST "$ENDPOINT/v1/predict" \
      -H "Authorization: Api-Key $BASETEN_API_KEY" \
      -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
      -H "Content-Type: application/json" \
      -d "$(cat <<JSON
    {
      "detached": true,
      "task_suite": $(cat tasks/crm/crm_tasks.json),
      "profile_id": "crm-prod",
      "workflow_id": "crm-$(date +%s)",
      "max_cost": 5,
      "max_time_minutes": 30
    }
    JSON
    )"
    ```

=== "task_file (baked-in path)"

    ```bash
    curl -X POST "$ENDPOINT/v1/predict" \
      -H "Authorization: Api-Key $BASETEN_API_KEY" \
      -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
      -H "Content-Type: application/json" \
      -d '{
        "detached": true,
        "task_file": "tasks/crm/crm_tasks.json",
        "profile_id": "crm-prod",
        "workflow_id": "crm-tasks-v1"
      }'
    ```

## Caps and clamping

| Knob | Server hard cap (env-overridable) | Tenant cap (per-tenant config) | Request value |
|---|---|---|---|
| `max_cost` | `MANTIS_MAX_COST_USD` (default $25) | `max_cost_per_run` | `max_cost` |
| `max_time_minutes` | `MANTIS_MAX_RUNTIME_MINUTES` (60) | `max_time_minutes_per_run` | `max_time_minutes` |
| Plan size | `MANTIS_MAX_STEPS_PER_PLAN` (200) | n/a — rejects 400 | n/a |
| `loop_count` | `MANTIS_MAX_LOOP_ITERATIONS` (50) | n/a — silently clamps | n/a |

The effective value is `min(server_cap, tenant_cap, request_value)`. If you ask for $50, your tenant has $5 cap, and the server has $25 cap → you get $5.

## URL allowlist

If your tenant has `allowed_domains` configured, the server scans your plan's `navigate` steps + `task_suite.base_url` + each `task.start_url` and rejects 403 if any host is off-list. Wildcards like `*.example.com` work; exact matches like `crm.example.com` work.

If you need to add a domain, ask your operator to update your tenant config — see [URL allowlist](../operations/allowlist.md).

## Inline extraction schema

For plans that don't have a registered server-side recipe (the common case for ad-hoc plans), declare your extraction contract **inline on the step**. The validator then enforces the fields *you* asked for instead of falling through to the framework's `no_schema_configured` rejection.

Put an `extract` block at the top level of any `extract_data` (or `claude`) step:

```jsonc
{
  "intent": "Extract the top 5 stories",
  "type": "extract_data",
  "claude_only": true,
  "extract": {
    "schema_name": "hn_top5",                        // identifier — used as CSV filename + Augur tag
    "entity_name": "hn_story",                       // referenced in the extraction prompt
    "fields": [
      {"name": "rank",           "type": "int", "required": true},
      {"name": "title",          "type": "str", "required": true},
      {"name": "story_url",      "type": "str", "required": false},
      {"name": "points",         "type": "int", "required": false},
      {"name": "author",         "type": "str", "required": false},
      {"name": "age",            "type": "str", "required": false},
      {"name": "comments_count", "type": "int", "required": false}
    ],
    "max_items": 5                                    // optional cap on rows returned
  }
}
```

### Field reference

| Field | Required | Effect |
|---|---|---|
| `schema_name` | yes | Logged at runtime; CSV output filename; Augur tag |
| `entity_name` | optional (default `"item"`) | Used in the extractor prompt ("Extract from this {entity_name}…") |
| `fields[].name` | yes | Column name in the CSV / key in the JSON output row |
| `fields[].type` | yes | `int` / `str` / `bool` — passed to Claude's response schema |
| `fields[].required` | optional (default `true`) | Validator rejects rows missing `required: true` fields; `required: false` is allowed to be empty |
| `max_items` | optional | Cap on rows returned. Omit for single-row extraction |

The same shape is documented in [plan.schema.json](../reference/plan.schema.json) under `$defs.Step.properties.extract`.

### What if I don't include an `extract` block?

The step still runs, but the framework needs to find a schema somewhere:

1. **Inline `extract` block on the step** — described above. Wins when set.
2. **Recipe-bound schema on the extractor** — set at executor startup when your plan domain has a registered recipe (e.g. `marketplace_listings`, `job_listings`, `search_results`). Plan authors don't see this directly; it comes from `ExtractionSchema` defined alongside the recipe.
3. **No schema at all** — the validator rejects every extracted row with reason `no_schema_configured`. The runner emits a WARNING at step entry to surface the misconfig (see "Diagnostics" below).

For ad-hoc plans against arbitrary sites, declare the inline `extract` block. Recipe registration is only useful when you need domain-specific spam/control rules (forbidden buttons to skip, dealer-vs-private classification, etc.) in addition to a field schema.

### Diagnostics

When you submit an `extract_data` / `extract_url` step with no schema available from any source, the runner logs at WARNING level:

```
[claude_step] extract_data step has no extraction schema
  (no `extract` block on the step, no recipe-bound `extractor.schema`).
  The validator will reject every extracted row with `no_schema_configured`.
  Either add an inline `extract` block to this step
  (see docs/client/plans.md#inline-extraction-schema) or configure a recipe
  at executor startup.
```

This fires per step at WARNING level (visible in Modal app logs by default). The rejection itself still happens — the warning just makes the misconfig obvious in the trace instead of being buried in the trailing result envelope.

### What the schema enforces

`is_viable()` returns `True` only when every `required: true` field has a non-empty, non-placeholder value. The framework's `_UNKNOWN_PLACEHOLDERS` set treats these strings as empty: `"" / "unknown" / "<unknown>" / "none" / "n/a" / "na" / "not visible" / "not shown" / "not available" / "tbd"` (case + whitespace insensitive). So Claude returning `"<UNKNOWN>"` for `title` is treated the same as omitting the field.

Fields marked `required: false` are kept on the row even when empty — they just don't gate viability.

### What's NOT in the inline `extract` block (recipe-only)

The fields below live on `ExtractionSchema` but the inline path doesn't expose them. If you need them, register a recipe:

- `spam_indicators`, `spam_seller_indicators` — domain-specific text that marks a row as spam/dealer
- `forbidden_controls`, `allowed_controls` — reveal-button name filters
- `listing_card_exclusions` — listing-tile spam (financing CTAs, sponsored cards)
- `rejection_intents` — skip-envelope routing for downstream short-circuits

For straight "give me these N fields from each item" extraction — the inline block is enough.

### Multi-row from a single list page — `max_items`

If your target page has N items already visible on a single screen
(HN top stories, GitHub issue list, search-results-style pages), set
`extract.max_items > 1` on a single `extract_data` step. One Claude
call returns all N rows; each lands as a separate entry in
`extracted_rows.json` / `extracted_rows.csv` / `leads.csv`.

```jsonc
{
  "intent": "Extract the top 5 stories from HN",
  "type": "extract_data",
  "claude_only": true,
  "extract": {
    "schema_name": "hn_top5",
    "entity_name": "hn_story",
    "fields": [
      {"name": "rank", "type": "int", "required": true},
      {"name": "title", "type": "str", "required": true},
      {"name": "story_url", "type": "str", "required": false},
      {"name": "points", "type": "int", "required": false}
    ],
    "max_items": 5
  }
}
```

For pages where each item needs detail-page enrichment (seller phone
behind a "Show" button, full job description on a sub-page), the
single-row `extract_data` inside a `collect_urls` → `loop` chain is
still the right pattern.

**Where multi-row rows surface in the response.** The captured rows
land in three places, all consistent since [#842](https://github.com/mercurialsolo/mantis/pull/842):

- `extracted_rows.json` / `extracted_rows.csv` artifacts — one entry per row, canonical structured form
- `leads.csv` artifact — same rows in CSV summary form
- Lifecycle envelope — `viable: N` counts every row whose `required: true` fields are all populated, and `leads: [...]` carries one `"VIABLE | Field: value | …"` summary string per row

Pre-#842 the counter and summary array silently reported `viable: 0`
and `leads: []` even when the multi-row primitive captured rows
correctly — only the artifacts were trustworthy. They now agree.

## Control-flow primitives

For plans that need to vary their behaviour based on what the page
actually shows, three primitives compose:

| Step type | Effect |
|---|---|
| `detect_visible` | One vision call: "is X visible?" Writes a bool to `runner._state_vars[out_var]`. |
| `if_else` | Reads `condition_var`, jumps to `then_target` (truthy) or `else_target` (falsy). |
| `loop` | Jumps to `loop_target` up to `loop_count` times. Optional `stop_var` exits early. |

Worked example — skip a cookie-consent dance when the banner isn't shown:

```jsonc
[
  {"intent": "Navigate to the dashboard", "type": "navigate",
   "params": {"url": "https://example.com/dashboard"}},
  {"intent": "Cookie banner visible?", "type": "detect_visible",
   "out_var": "cookie_shown"},
  {"intent": "Branch", "type": "if_else",
   "condition_var": "cookie_shown",
   "then_target": 3,        // dismiss
   "else_target": 5},       // skip dismiss
  {"intent": "Click Accept", "type": "submit",
   "params": {"label": "Accept"}},
  {"intent": "Wait briefly", "type": "scroll", "budget": 1},
  {"intent": "Extract dashboard data", "type": "extract_data", "claude_only": true,
   "extract": {"fields": [{"name": "title", "required": true}]}}
]
```

Safety: missing `condition_var`, unset target (`-1`), or out-of-range
index falls through to `step_index + 1` instead of teleporting or
hanging. The runner records a synthetic `StepResult` for every
branch so the trace shows `var=value→target`.

Full reference: [Plan formats → control-flow primitives](../getting-started/plan-formats.md#control-flow-primitives-loop-if_else-multi-row-extract_data).

## Picking a compute plane

Every run executes on one of two compute planes. Default is **Computer Plane** — the production-stealth path. **Browser-Use Plane** is opt-in for plans that need DOM-aware reads (current URL, anchor `href` peek before click, semantic role disambiguation on dense list pages, tab management).

| Plane | Driver | DOM-aware | Stealth (CF/Turnstile) | Pick this when… |
|---|---|---|---|---|
| `computer_plane` (default) | Xvfb + Chrome + xdotool | ✗ | ✓ | Production scraping, CF/Turnstile-protected sites, screenshot+click flows |
| `browser_use_plane` | Playwright + Chromium (headless) | ✓ | ✗ | Dense link lists where vision misclicks (HN, comment threads), tab-management plans, plans needing anchor-href reads |

Select via the `runtime.compute_backend` field in your plan or via the submission-level `compute_backend` field on the request. Precedence: plan `runtime.compute_backend` > submission `compute_backend` > global default (`computer_plane`).

### Computer Plane (default) — marketplace-style extraction

The standard production posture. CUA-pure: screenshot + xdotool only. Holo3 grounds vision; Claude extracts.

```bash
curl -X POST "$ENDPOINT/v1/predict" \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "detached": true,
    "cua_model": "holo3",
    "plan_text": "Search the boat marketplace for private-seller listings in Florida above $35,000, then for each listing extract year, make, model, price, seller phone, and listing URL.",
    "profile_id": "marketplace-prod",
    "workflow_id": "marketplace-fl-v1",
    "max_cost": 2,
    "max_time_minutes": 20
  }'
```

Decomposer runs the prose → MicroIntent steps with inline `extract` blocks (since #785), validator enforces the schema, Holo3 + Xvfb-driven Chrome do the actual scraping under stealth.

### Browser-Use Plane (opt-in) — link-list harvest

Use when vision struggles. Browser-Use Plane is the right path for the HN top-N use case the browser-use epic (#785) was filed for: link lists where title/comment/author rows are visually similar and the runtime needs anchor `href` reads to disambiguate.

```bash
curl -X POST "$ENDPOINT/v1/predict" \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "detached": true,
    "cua_model": "holo3",
    "compute_backend": "browser_use_plane",
    "plan_text": "Extract the top 5 stories from Hacker News including rank, title, story URL, points, author, age, and comment count. Do not click titles.",
    "profile_id": "hn-prod",
    "workflow_id": "hn-top5-v1",
    "max_cost": 0.50,
    "max_time_minutes": 6
  }'
```

Equivalently, declare it in the plan so the choice survives across submissions:

```jsonc
{
  "runtime": {
    "compute_backend": "browser_use_plane",
    "max_cost": 0.50
  },
  "steps": [
    {"type": "navigate", "intent": "Navigate to HN", "params": {"url": "https://news.ycombinator.com/"}},
    {
      "type": "extract_data",
      "intent": "Extract top 5 stories with rank, title, story_url, points, author, age, comments_count",
      "claude_only": true,
      "extract": {
        "schema_name": "hn_top5",
        "entity_name": "hn_story",
        "fields": [
          {"name": "rank",           "type": "int", "required": true},
          {"name": "title",          "type": "str", "required": true},
          {"name": "story_url",      "type": "str", "required": false},
          {"name": "points",         "type": "int", "required": false},
          {"name": "author",         "type": "str", "required": false},
          {"name": "age",            "type": "str", "required": false},
          {"name": "comments_count", "type": "int", "required": false}
        ],
        "max_items": 5
      }
    }
  ]
}
```

### What Browser-Use Plane gives you that Computer Plane doesn't

Once you opt in, the Browser-Use Plane handlers can consume these DOM-aware verbs (transparent to plan authors — they just work because the plan declares `browser_use_plane`):

- `state.current_url()` / `state.tabs()` / `state.focused_element()` — read-only browser state
- `tabs.open_in_new(url)` / `tabs.close(tab_id)` / `tabs.activate(tab_id)` — programmatic tab management
- `links.peek_target(selector)` — read an anchor `href` without clicking it (the HN list disambiguation primitive)
- `target_role` on `click` / `capture_link_in_new_tab` steps — semantic role lookup via per-site recipe (e.g. on HN: "title" → the story link, "comment_count" → the threads link)

These verbs are **capability-gated** behind `dom_aware`. Pure-CUA executors (Holo3, Claude vision) carry an allowlist that excludes them, so wiring a plan to the wrong plane fails loud at session start instead of silently mis-clicking.

### Non-goals for Browser-Use Plane at v1

- **CF / Turnstile parity** — Browser-Use Plane does NOT promise stealth against bot detectors. Use Computer Plane for stealth-sensitive targets.
- **Cross-plane profile sharing** — `(tenant_id, profile_id)` is the same identity on both planes but storage is independent. Switching plane mid-workflow re-authenticates. Cross-plane handoff is a deferred follow-up.

Full plane architecture, capability model, and host details live in:

- [Compute Client contract](../reference/compute-client.md) — umbrella spec for both planes
- [Computer Plane](../reference/computer-plane.md) — Xvfb + xdotool implementation
- [Browser-Use Plane](../reference/browser-use-plane.md) — Playwright implementation, deploy, wire contract

## What the server returns

For `detached: true` (the default) — a queued handle:

```jsonc
{
  "status": "queued",
  "model": "holo3",
  "mode": "detached",
  "run_id": "20260428_021432_076255ef",
  "created_at": "2026-04-28T02:14:32.331Z",
  "payload": { ...echoed input... },
  "status_path":  "/workspace/.../status.json",
  "result_path":  "/workspace/.../result.json",
  "csv_path":     "/workspace/.../leads.csv",
  "events_path":  "/workspace/.../events.log"
}
```

Use `run_id` for [Runs and polling](runs-and-polling.md).

For `detached: false` — the call blocks until the run completes (5-30+ min). Useful only for short plans; the response is the same shape you'd otherwise fetch with `{action: "result"}`.

## See also

- [Plan formats](../getting-started/plan-formats.md) — full schemas
- [Runs and polling](runs-and-polling.md) — what to do with a `run_id`
- [Errors](errors.md) — when something goes wrong
