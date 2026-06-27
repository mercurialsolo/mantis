# Plan formats

Three shapes are accepted. The runtime picks them up in priority order from the `/v1/predict` payload.

## When to use which

| Use case | Recommended shape |
|---|---|
| Recurring high-volume workflow with predictable steps | **Micro-plan JSON**, baked into the image at `plans/<domain>/<workflow>.json` |
| Arbitrary plain-English request | **plan_text** — server decomposes via Claude (cached) |
| Ad-hoc plan you don't want baked in | **task_suite** (inline JSON dict) |
| Multi-task suite with `task_id` + `verify` clauses | **task_suite** or **task_file** |

## Micro-plan (recommended for production extraction)

A flat JSON list of step objects executed by `MicroPlanRunner`. Best reliability — `sections`, `gates`, `loops`, and explicit `claude_only` markers all live here.

```jsonc
[
  {
    "intent": "Navigate to https://marketplace.example.com/listings/state-fl/by-owner/price-35000/",
    "type": "navigate",
    "budget": 3,
    "section": "setup",
    "required": true
  },
  {
    "intent": "Verify page shows private-seller listings near Florida above $35,000",
    "type": "extract_data",
    "claude_only": true,
    "section": "setup",
    "gate": true,
    "verify": "Page shows listings from private sellers near Florida with prices above $35,000"
  },
  {
    "intent": "Click only an organic private-seller listing title; skip sponsored cards",
    "type": "click",
    "budget": 8,
    "grounding": true,
    "section": "extraction"
  },
  {
    "intent": "Read the URL from the browser address bar",
    "type": "extract_url",
    "claude_only": true,
    "section": "extraction"
  },
  {
    "intent": "Scroll down toward the Description section",
    "type": "scroll",
    "budget": 10,
    "section": "extraction"
  },
  {
    "intent": "Extract Year, Make, Model, Price, Phone if visible, and Seller Name",
    "type": "extract_data",
    "claude_only": true,
    "section": "extraction"
  },
  {
    "intent": "Go back to the search results page",
    "type": "navigate_back",
    "budget": 3,
    "section": "extraction"
  },
  {
    "intent": "Loop back to click the next listing",
    "type": "loop",
    "loop_target": 2,
    "loop_count": 3,
    "section": "extraction"
  }
]
```

Field reference is on the [Concepts](concepts.md#step-types-micro-plan-shape) page.

### Control-flow primitives — `loop`, `if_else`, multi-row `extract_data`

The `loop` step is the basic iteration primitive — jump back to an
earlier step up to `loop_count` times. The `if_else` step branches on
a runtime variable. Both compose with `detect_visible` (which writes
a bool to `runner._state_vars`).

**`loop` — iterate over the same body N times.**

```jsonc
[
  {"intent": "Navigate to results page", "type": "navigate",
   "params": {"url": "https://example.com/search"}},

  // — loop body starts here (index 1)
  {"intent": "Click the next result", "type": "click", "section": "extraction"},
  {"intent": "Extract the detail page", "type": "extract_data",
   "claude_only": true, "section": "extraction"},
  {"intent": "Go back to the results", "type": "navigate_back", "section": "extraction"},
  // — body ends; loop step at index 4 jumps back to index 1 —

  {"intent": "Repeat next-result/extract/back", "type": "loop",
   "loop_target": 1, "loop_count": 5, "section": "extraction"}
]
```

`loop_target` is the **absolute** step index the runner jumps to.
`loop_count` caps the iterations (server clamps it to
`MANTIS_MAX_LOOP_ITERATIONS`, default 50). Optional `stop_var` reads
a runner state variable and exits the loop early when truthy — pair
it with a `detect_visible` body step to halt as soon as a target is
found, instead of walking every iteration.

**`if_else` — branch on a state variable.**

Composes with `detect_visible`: the prior step writes a bool to
`runner._state_vars[<var>]`, then `if_else` reads the same var and
jumps to `then_target` (truthy) or `else_target` (falsy / missing).
Step indices are absolute.

```jsonc
[
  {"intent": "Navigate to the dashboard", "type": "navigate",
   "params": {"url": "https://example.com/dashboard"}},

  {"intent": "Is a cookie consent banner visible?", "type": "detect_visible",
   "out_var": "cookie_banner_shown"},

  {"intent": "Branch on whether the banner is up", "type": "if_else",
   "condition_var": "cookie_banner_shown",
   "then_target": 3,   // → step 3: dismiss the banner first
   "else_target": 5},  // → step 5: skip the dismiss + proceed

  {"intent": "Click Accept on the cookie banner", "type": "submit",
   "params": {"label": "Accept"}},
  {"intent": "Wait briefly", "type": "scroll", "budget": 1},

  {"intent": "Extract the dashboard tiles", "type": "extract_data",
   "claude_only": true,
   "extract": {"fields": [{"name": "title", "required": true}]}}
]
```

Safety: when `condition_var` is empty, `then_target`/`else_target`
default to `-1`, or the index is out of range, `if_else` falls through
to `step_index + 1` instead of teleporting / hanging. A synthetic
`StepResult` records the decision (`var=value→target`) so the run
trace is grep-able.

**Post-action success contract — `hints` on click / submit steps.**

The runner verifies that a `click` / `submit` actually accomplished
something and **demotes a step that reported success but produced no
observable change** (so the next step doesn't run against a stale page).
By default the signal is the URL: a submit that navigates is success, a
same-URL submit gets a visual double-check. Declare the expected outcome
to make this precise:

| Hint | Meaning |
| --- | --- |
| `expect_url_contains` | str or list — every substring must appear in the post-action URL, else the step is demoted `wrong_target`. |
| `expect_url_excludes` | str or list — substrings that must **not** appear (e.g. "must not drift to a detail page"). |
| `expect_text_present` | str or list — for **in-place submits that don't navigate**. The success signal is a confirmation toast or a control flipping state. |

Use `expect_text_present` when success leaves the URL unchanged and
nothing new *appears* — the canonical case is a LinkedIn connection
request: clicking **Send** closes the invitation modal, the URL stays on
the profile, and the button flips to "Pending". Without the hint the
runner sees "nothing appeared" and falsely demotes the (already-sent)
invite to `submit_failed`; with it, a cheap vision check confirms the
phrase is visible and keeps the success.

```jsonc
{
  "intent": "Send the invitation",
  "type": "submit",
  "params": {"label": "Send", "aliases": ["Send", "Send without a note"]},
  "hints": {"expect_text_present": ["Pending", "Invitation sent"]}
}
```

The hint is opt-in: omit it and behaviour is unchanged. See
`plans/linkedin-connect.json` for the full reference plan.

**Multi-row `extract_data` / `extract_rows` — top-N from a list page.**

Set `extract.max_items > 1` on an `extract_data` step (or use step type
`extract_rows`) to extract up to N rows in **one** Claude call instead
of N navigate→extract→back round trips. Each row lands as a separate
entry in `extracted_rows.json` / `extracted_rows.csv` / `leads.csv`.

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
      {"name": "points", "type": "int", "required": false},
      {"name": "author", "type": "str", "required": false},
      {"name": "age", "type": "str", "required": false},
      {"name": "comments_count", "type": "int", "required": false}
    ],
    "max_items": 5
  }
}
```

When NOT to use multi-row: pages where each item needs detail-page
enrichment (e.g. seller phone behind a "Show" button). Stick with
`collect_urls` → `loop` → `navigate` → single-row `extract_data` →
`navigate_back`.

### Declaring runtime defaults inside the plan

A plan can carry a top-level `runtime` block so it's self-describing — no submitter has to remember the right proxy / cost / time flags. The bare-array form above stays valid; the wrapped form adds a sibling next to `steps`:

```jsonc
{
  "runtime": {
    "proxy_disabled": false,
    "proxy_provider": "privateproxy",
    "proxy_city": "miami",
    "max_cost": 3.0,
    "max_time_minutes": 10
  },
  "steps": [
    { "intent": "Navigate to https://lu.ma/discover", "type": "navigate" }
  ]
}
```

| Field | Type | Effect |
| --- | --- | --- |
| `proxy_disabled` | bool | Skip proxy setup, connect direct. Use for CF-protected SaaS that whitelists the test environment's IP. |
| `proxy_provider` | string | `privateproxy` (preferred residential), `oxylabs`, or `iproyal`. Defaults to the runtime's `MANTIS_PROXY_PROVIDER` env. |
| `proxy_city` | string | Preferred proxy exit city (passed to the provider's session API). |
| `proxy_state` | string | Preferred proxy exit state (US two-letter code). |
| `max_cost` | number | Per-run cost ceiling in USD; runner halts when exceeded. |
| `max_time_minutes` | integer | Per-run wall-clock ceiling in minutes. |

**Override precedence.** Whenever the caller passes one of these fields explicitly (CLI flag, HTTP body, kwarg), the caller's value beats the plan default. Passing `None` (or omitting the field entirely) falls back to the plan's declaration. This is what lets `--proxy-disabled` on the command line override a `proxy_disabled: false` plan without breaking the plan-as-default contract.

**Loader + merger** (Python — for embedded callers and one-shot submit scripts):

```python
from mantis_agent.server_utils import (
    load_plan_file, merge_runtime, build_micro_suite,
)

steps, plan_runtime = load_plan_file("plans/luma-extract.json")

# proxy_disabled=cli.proxy_disabled is None when the flag wasn't passed,
# so the plan default wins; pass a real bool to override.
runtime = merge_runtime(plan_runtime, proxy_disabled=cli.proxy_disabled)
suite = build_micro_suite(steps, "luma", **runtime)
```

Both shipped plan shapes work — `load_plan_file` returns `(steps, {})` for the bare-array form, and `(steps, runtime)` for the wrapped form. Unknown `runtime` keys are dropped silently so future schema additions don't leak into `build_micro_suite` as `TypeError`-causing kwargs.

**Reference plans** carried in this repo:

* [`examples/form_fill_with_runtime.json`](https://github.com/mercurialsolo/mantis/blob/main/examples/form_fill_with_runtime.json) — minimal demo of the wrapped form (proxy off, $0.50 cap, 5 min cap).
* `scripts/run_luma_extract.py` — submit helper that loads a wrapped plan, calls `merge_runtime`, splats into `build_micro_suite`, then POSTs `/v1/predict`. Use it as the template for new domain-specific submitters.

**End-to-end verification** (against deployed Modal):

| Plan | `runtime.proxy_provider` | Modal worker startup log |
|---|---|---|
| `plans/staff-crm-long.json` | unset (proxy disabled) | direct connection — no proxy line |
| `plans/luma-extract.json` (proxy on, `iproyal`) | unset → env default | `Proxy: iproyal via http://geo.iproyal.com:12321` |
| `plans/luma-extract.json` (proxy on, `privateproxy`) | `privateproxy` | `Proxy: privateproxy via http://edge1-us.privateproxy.me:8888` |

### Iterate on plan structure without GPU or API cost

Use `MANTIS_BRAIN=mock` to point the runner at a deterministic stub
brain that returns `DONE` on every `think()` call — every click / scroll
/ holo3 step succeeds immediately, so the run walks through your plan's
sections, gates, and loops without burning Holo3 inference or Anthropic
credits.

```bash
MANTIS_BRAIN=mock mantis plan dry-run plans/example/extract_listings.json
```

This catches the structural bugs — wrong loop_target index, gate
predicate in the wrong section, missing `navigate` URL — long before
you submit the plan to a paid deployment. ClaudeExtractor and
ClaudeGrounding are independent of the brain, so plans that lean on
extraction still need `ANTHROPIC_API_KEY` for those steps; everything
else runs free.

### IDE autocomplete via JSON Schema

A JSON Schema for the micro-plan format ships at
[`reference/plan.schema.json`](../reference/plan.schema.json) (also live at
`https://mercurialsolo.github.io/mantis/reference/plan.schema.json`). Point
your editor at it and you get autocomplete + inline validation for every
step field.

**VS Code** — add to `.vscode/settings.json` (or workspace settings):

```jsonc
{
  "json.schemas": [
    {
      "fileMatch": [
        "**/plans/**/*.json",
        "**/recipes/**/plan.json",
        "**/examples/*.json"
      ],
      "url": "https://mercurialsolo.github.io/mantis/reference/plan.schema.json"
    }
  ]
}
```

**Neovim / coc.nvim** — same shape under `coc-settings.json` (`json.schemas`).

**IntelliJ / PyCharm** — *Preferences → Languages & Frameworks → Schemas
and DTDs → JSON Schema Mappings*, then add a mapping with the URL above and
a file pattern matching your plan layout.

The schema is intentionally permissive (`additionalProperties: true`) so
plan-specific fields don't trip validation — IDE autocomplete still covers
the well-known step types and fields.


Submit it three ways:

```jsonc
// (a) reference a baked-in path
{ "micro": "plans/example/extract_listings.json" }

// (b) inline the raw step list
{ "task_suite": [ ...the array above... ] }

// (c) inline as a JSON string
{ "task_file_contents": "[\"intent\":\"...\", ...]" }
```

The server validates and **clamps** every plan against:

- `MANTIS_MAX_STEPS_PER_PLAN` (default 200) — oversized plans rejected with 400
- `MANTIS_MAX_LOOP_ITERATIONS` (default 50) — `loop_count` silently clamped

## Task suite (Claude-CUA-style autonomous-per-task)

Used by a multi-task CRM workflow. Each task is given to the runner with its own `max_steps` budget and a `verify` clause; Claude decides what to do per task.

```jsonc
{
  "session_name": "crm_demo",
  "base_url": "https://crm.example.com",
  "auth": { "user_id": "...", "password": "..." },
  "tasks": [
    {
      "task_id": "login",
      "intent": "Go to https://... and log in with user X and password Y",
      "save_session": true,
      "start_url": "https://crm.example.com",
      "verify": { "type": "url_not_contains", "value": "login" }
    },
    {
      "task_id": "update_lead_industry",
      "intent": "Go to the Leads Page. Update industry of qualified lead to 'Space Exploration'.",
      "require_session": true,
      "start_url": "https://crm.example.com",
      "verify": { "type": "page_contains_text", "value": "Space Exploration" }
    }
  ]
}
```

Verify predicates supported today:

| `verify.type` | Meaning |
|---|---|
| `url_not_contains` | After the task, the URL doesn't contain `value` |
| `url_contains` | After the task, the URL contains `value` |
| `page_contains_text` | The current page renders `value` somewhere |
| `download_exists` | A file matching the glob was created during the task |

Submit:

```jsonc
{ "task_suite": { ... full dict ... } }
// or
{ "task_file": "tasks/crm/crm_tasks.json" }
```

## plan_text (auto-decompose)

Plain English in, micro-plan out. The server runs `PlanDecomposer` (Claude, cached by the plan-text signature so subsequent runs of the same plan-text don't re-pay decomposition).

```jsonc
{
  "plan_text": "Go to a marketplace listings site, filter to private sellers above $35,000 in Florida, extract listing details for the first 3 listings, save year/make/model/price/phone."
}
```

Decomposition costs ~$0.10 per unique plan-text the first time, free after. Best for one-shot tasks or when you don't yet have a stable workflow definition. Once you've validated a decomposition, copy the resulting plan to a `plans/...json` file and start submitting via `micro` for cheaper, deterministic runs.

## Decision tree

```
Have a stable workflow you'll run many times?
  ├─ yes → write a micro-plan once and submit via `micro`
  └─ no  → is it a one-shot or rapidly-changing?
            ├─ yes → submit as `plan_text`, server decomposes via Claude
            └─ no  → is it a multi-task suite?
                     ├─ yes → submit as `task_suite`
                     └─ no  → micro-plan inline via `task_suite` field
```

## Per-tenant URL allowlist

If your tenant config has a non-empty `allowed_domains` list, the server scans your plan for `navigate` URLs / `task.start_url` / `task_suite.base_url` and rejects 403 if any host is off-list. This applies to all three plan shapes. See [URL allowlist](../operations/allowlist.md) for the matching rules.

## Next

- [Authentication](../client/auth.md) — getting a tenant token
- [Sending plans](../client/plans.md) — full request shape and curl recipes
- [Operations / tenant keys](../operations/tenant-keys.md) — operator-side provisioning
