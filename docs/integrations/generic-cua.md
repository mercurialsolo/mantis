# Generic CUA over HTTP — any site, any data shape

Use Mantis as a **language-agnostic extraction service**. No Python install,
no library embedding — just `/v1/predict` with a plan and your own data
shape. This is the right doc if you're building an extraction pipeline
for jobs, products, profiles, news, real-estate, or any site that follows
the **search → click → extract → loop** pattern.

> Companion: [Recipes](recipes.md) for copy-paste templates of common
> patterns. [Embedding MicroPlanRunner](embedding-microplanrunner.md) if
> you'd rather drive the runner in-process from Python.

---

## Onboarding in 5 steps

1. Get an `X-Mantis-Token` from the Mantis operator (one tenant key per app).
2. Set `allowed_domains` on the tenant (wildcards: `*.greenhouse.io`).
3. Author or generate a micro-plan for your target site.
4. Optionally: define a custom `ExtractionSchema` in the plan payload.
5. POST to `/v1/predict` with `detached: true`; poll `/predict` with
   `action: status` until the run completes; then `action: result`.

The three values you need:

```bash
export MANTIS_ENDPOINT="https://model-qvvgkneq.api.baseten.co/production/sync"
export MANTIS_API_TOKEN="..."        # the X-Mantis-Token your operator issued
export BASETEN_API_KEY="..."         # only needed when fronted by the Baseten gateway
```

For Modal / EKS / GKE deployments, drop the `BASETEN_API_KEY` — Mantis
self-hosted enforces auth purely via `X-Mantis-Token`.

---

## The four plan shapes — pick one

| Shape | Field | Best for |
|---|---|---|
| Plain English | `plan_text: "Extract the first 5 jobs from..."` | One-shots, prototyping. Server decomposes via Claude (cached after first call). |
| Hand-authored micro-plan | `micro: <inline-json-list-of-steps>` or `micro: "plans/path.json"` | Production extraction at high volume. Maximum reliability. |
| Multi-task batch | `task_suite: { tasks: [...] }` | Several independent tasks in one submission. |
| Pre-baked file | `task_file: "tasks/myorg/foo.json"` | The plan ships in the container image. |

Decision tree is in [Sending plans](../client/plans.md#decision-tree).

---

## Example 1 — One-shot extraction with `plan_text`

The fastest path. You hand it English; it figures out the structure.

```bash
curl -fsS -X POST "$MANTIS_ENDPOINT/v1/predict" \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "detached": true,
    "plan_text": "Go to https://news.ycombinator.com/ and extract the title, points, and comment count of the top 5 stories",
    "state_key": "hn-top5",
    "max_cost": 1,
    "max_time_minutes": 10
  }'
```

Returns:

```jsonc
{
  "run_id": "20260429_103045_a1b2c3d4",
  "status": "queued"
}
```

Poll `/predict` with `{"action":"status","run_id":"..."}` until status is
`succeeded` / `failed` / `cancelled`, then fetch results with
`{"action":"result","run_id":"..."}`. Full polling pattern in
[Runs and polling](../client/runs-and-polling.md).

---

## Example 2 — Hand-authored micro-plan with custom schema

When the same workflow runs many times, hand-author the plan once. Inline
JSON is supported in the `micro` field — you don't need to ship a file
in the image.

```bash
curl -fsS -X POST "$MANTIS_ENDPOINT/v1/predict" \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d @- <<'JSON'
{
  "detached": true,
  "state_key": "greenhouse-ml-sf-prod",
  "max_cost": 4,
  "max_time_minutes": 30,
  "extraction_schema": {
    "entity_name": "job posting",
    "fields": [
      {"name": "title",      "type": "str", "required": true,  "example": "ML Engineer"},
      {"name": "team",       "type": "str", "required": false, "example": "Search & Ranking"},
      {"name": "location",   "type": "str", "required": false, "example": "San Francisco"},
      {"name": "url",        "type": "str", "required": true,  "example": "boards.greenhouse.io/..."},
      {"name": "department", "type": "str", "required": false, "example": "Engineering"}
    ],
    "required_fields": ["title", "url"],
    "spam_indicators": ["recruiter", "staffing agency"],
    "spam_label": "recruiter spam"
  },
  "micro": [
    {"intent": "Navigate to https://boards.greenhouse.io/openai/jobs?department=Engineering&location=San+Francisco",
     "type": "navigate", "budget": 3, "section": "setup", "required": true},
    {"intent": "Verify page shows OpenAI engineering job listings filtered to San Francisco",
     "type": "extract_data", "claude_only": true, "budget": 0, "section": "setup",
     "gate": true,
     "verify": "Page is the OpenAI Greenhouse jobs board, filtered to Engineering+San Francisco, with at least one listing visible"},
    {"intent": "Click the next un-extracted job posting title",
     "type": "click", "budget": 8, "grounding": true, "section": "extraction"},
    {"intent": "Read the URL from the address bar",
     "type": "extract_url", "claude_only": true, "budget": 0, "section": "extraction"},
    {"intent": "Scroll down to read the full job description",
     "type": "scroll", "budget": 5, "section": "extraction"},
    {"intent": "Extract title, team, location, department, url",
     "type": "extract_data", "claude_only": true, "budget": 0, "section": "extraction"},
    {"intent": "Re-navigate to the search page at https://boards.greenhouse.io/openai/jobs?department=Engineering&location=San+Francisco",
     "type": "navigate", "budget": 3, "section": "extraction"},
    {"intent": "Loop back to click the next job",
     "type": "loop", "loop_target": 2, "loop_count": 10, "section": "extraction"}
  ]
}
JSON
```

Notes on the schema:

- **`extraction_schema`** is read by `ClaudeExtractor` to drive the
  per-listing JSON output and the spam detector. Field shape mirrors
  `mantis_agent.extraction.ExtractionSchema`.
- **`required_fields`** controls viability — listings missing any of these
  are dropped from the result. `["title", "url"]` is the minimum useful
  shape; relax it for noisier sites.
- **`spam_indicators`** and **`spam_label`** customize the detector. For
  jobs, recruiter spam is the analog of dealer spam in the boats domain.
- The plan uses `re-navigate` (a fresh `navigate` step) instead of
  `navigate_back` to return to the results page — this is more reliable
  than the browser back button when the model isn't sure.

---

## Example 3 — Custom site config for unusual URL patterns

`SiteConfig` tells the runner which URLs are detail pages vs. results
pages, and how pagination works. Most sites are auto-detected by the site prober;
for unusual sites,
ship a `site_config` block alongside the plan:

```jsonc
{
  "detached": true,
  "state_key": "redfin-sf-condos-v1",
  "site_config": {
    "domain": "redfin.com",
    "detail_page_pattern": "/CA/[\\w-]+/.*?/home/\\d+",
    "results_page_pattern": "/city/\\d+/CA/San-Francisco/filter/",
    "pagination_format": "page-{n}/",
    "pagination_type": "path_suffix",
    "pagination_strip_pattern": "page-\\d+/?$",
    "filtered_results_url": "https://www.redfin.com/city/17151/CA/San-Francisco/filter/property-type=condo,price-min=500k"
  },
  "micro": [ /* steps reference $RESULTS_URL = filtered_results_url */ ]
}
```

`site_config` is what tells the runner that going to `…/home/12345`
means "we're on a detail page" (so it knows to scroll + extract) and
that pagination is `…/page-2/`, not `?page=2`.

Reference: `mantis_agent.site_config.SiteConfig` (full field list).

---

## Tenant configuration the operator does once

Before any extraction starts, the Mantis operator issues a tenant key
with the shape:

```jsonc
{
  "tenant_keys": {
    "<your-x-mantis-token>": {
      "tenant_id": "your_org",
      "scopes": ["run", "status", "result", "logs"],
      "max_concurrent_runs": 3,
      "max_cost_per_run": 5.0,
      "max_time_minutes_per_run": 30,
      "anthropic_secret_name": "anthropic_api_key_your_org",
      "allowed_domains": [
        "*.greenhouse.io",
        "*.lever.co",
        "boards.example.com"
      ],
      "webhook_url": "https://your-app.example.com/mantis-webhook",
      "webhook_secret_name": "your_org_webhook_secret"
    }
  }
}
```

`allowed_domains` is enforced per-request — plans that try to navigate
outside the wildcard list are rejected before any GPU time is spent.
`webhook_url` is fired when each detached run completes. Full setup in
[Tenant keys](../operations/tenant-keys.md).

---

## What you get back

Every successful run produces:

```jsonc
{
  "run_id": "20260429_103045_a1b2c3d4",
  "status": "succeeded",
  "summary": {
    "viable": 7,                 // listings that passed the spam + required-fields gates
    "leads": [
      "VIABLE | title: ML Engineer | team: Search | location: San Francisco | url: ...",
      "VIABLE | title: ML Engineer Manager | ...",
      ...
    ],
    "leads_with_phone": 0,       // for schemas that include a phone field
    "cost_total": 0.42,
    "cost_breakdown": {
      "gpu":    0.12,
      "claude": 0.12,
      "proxy":  0.18
    },
    "dynamic_verification_summary": { /* coverage report — discovered vs attempted vs completed */ }
  },
  "csv_path": "/workspace/mantis-data/results/.../leads.csv",
  "events_path": "/workspace/mantis-data/runs/.../events.log",
  "video_path": "/workspace/mantis-data/runs/.../recording.mp4"  // when record_video: true
}
```

Leads come back as pipe-delimited strings keyed by your schema's field
names. CSV and recording are downloadable via dedicated endpoints — see
[Recordings](../client/recordings.md) and [Runs and polling](../client/runs-and-polling.md).

---

## What's NOT supported (today)

- **File uploads.** No `LAUNCH_APP`-equivalent for `<input type="file">`.
- **Multi-tab orchestration.** Plans run on one tab; opening new tabs via
  `Ctrl+T` works but isn't first-class.
- **CAPTCHA / hCaptcha.** Holo3 cannot solve these. Residential proxy
  + Cloudflare auto-pass usually works; hard challenges don't.
- **Plans >200 steps in one submission.** Server hard cap. Chunk into
  multiple runs sharing the same `state_key` to use checkpoint resume.

For these, [Library embedding](embedding-microplanrunner.md) gives you
escape hatches via `register_tool` (host-provided tools the model can
call) and `PauseRequested` (human-in-the-loop pause).

---

## Next

- [Recipes](recipes.md) — copy-paste plans for common patterns.
- [Sending plans](../client/plans.md) — full request schema.
- [Runs and polling](../client/runs-and-polling.md) — async lifecycle.
- [Tenant keys](../operations/tenant-keys.md) — operator setup.
