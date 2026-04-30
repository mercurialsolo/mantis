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
    "intent": "Navigate to https://www.boattrader.com/boats/state-fl/by-owner/price-35000/",
    "type": "navigate",
    "budget": 3,
    "section": "setup",
    "required": true
  },
  {
    "intent": "Verify page shows private-seller listings near Miami above $35,000",
    "type": "extract_data",
    "claude_only": true,
    "section": "setup",
    "gate": true,
    "verify": "Page shows boat listings from private sellers near Miami with prices above $35,000"
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

Submit it three ways:

```jsonc
// (a) reference a baked-in path
{ "micro": "plans/boattrader/extract_url_filtered_3listings.json" }

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
  "plan_text": "Go to BoatTrader, filter to Miami private sellers above $35,000, extract listing details for the first 3 listings, save year/make/model/price/phone."
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
