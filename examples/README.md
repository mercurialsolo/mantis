# Example plans + submission scripts

Two flavors of examples live here. Pick by integration shape:

- **JSON plans** (this README's existing list below) — feed to `/v1/predict` via curl. Best for shell pipelines, CI smoke tests, and operator one-shots.
- **Python submission scripts** (`*.py` files) — use the `mantis_agent.client.MantisClient` SDK. Best for application code that needs structured rows back.

## Python submission scripts (showcases the #785 chain)

| Script | Showcases | Cost (rough) | Plane |
|---|---|---|---|
| [`hn_top_5.py`](hn_top_5.py) | `plan_text` submission with decomposer-auto-emitted `extract` block (PR #801) | ~$0.20 | Browser-Use Plane |
| [`github_issues.py`](github_issues.py) | Explicit `micro` step list with inline `extract` block and `compute_backend` selection | ~$0.30 | Browser-Use Plane |
| [`pricing_page.py`](pricing_page.py) | `dry_run: true` preview before submitting for real (DX-3) | ~$0.08 | Computer Plane (default) |

Setup:

```bash
pip install -e ".[client]"
export MANTIS_API_ENDPOINT="https://your-deployment.modal.run"
export MANTIS_API_TOKEN="mantis_..."
python examples/hn_top_5.py
```

What if it fails:

- **`HTTP 400 + \"cua_model='claude' requires task_suite.tasks\"`** — submit-time shape validation (PR #812). Either switch `cua_model` or restructure the suite.
- **`status=succeeded` + `result.rows == []`** — your `extract` schema didn't match what the page actually has. Use `dry_run: true` (PR #813) to inspect the decomposed plan and verify the field list.
- **`status=completed_with_failures` + `halt_reason=\"extract_data_failed\"`** — Claude got the schema but couldn't find the fields. Check `action=logs` for the per-step trace; the page may have changed structure.
- **`profile_id` 409** — another run is holding the Chrome profile. Wait, or use a unique `profile_id` per call (typical: `f\"{user_id}-{job_id}\"`).

---

# JSON example plans (legacy curl-style)

These are committed, generic micro-plans you can run against a fresh Mantis
deployment. Unlike the customer-specific plans under `plans/` (gitignored),
everything here is public and works without proprietary data.

> Each plan here has a matching recipe under
> [`src/mantis_agent/recipes/<name>/plan.json`](../src/mantis_agent/recipes/)
> with the same JSON plus a `README.md` documenting tested sites and
> known limits. The recipe location is the canonical home; `examples/`
> is preserved for callers whose curl commands already point here.
>
> | Plan in `examples/` | Recipe |
> |---|---|
> | `extract_jobs.json` | [`recipes/job_listings`](../src/mantis_agent/recipes/job_listings/) |
> | `form_fill.json` | [`recipes/form_submit`](../src/mantis_agent/recipes/form_submit/) |
> | `google_search_extract.json` | [`recipes/search_results`](../src/mantis_agent/recipes/search_results/) |
> | `docs_lookup.json` | [`recipes/docs_lookup`](../src/mantis_agent/recipes/docs_lookup/) |

## Files

| Plan | What it does | Verified against |
|---|---|---|
| [`extract_jobs.json`](extract_jobs.json) | Open a public jobs board, filter by keyword, walk N listings, extract title / company / location / url | greenhouse.io listings |
| [`form_fill.json`](form_fill.json) | Open a public form, fill fields from a JSON payload, screenshot the result | httpbin.org/forms/post |
| [`google_search_extract.json`](google_search_extract.json) | Search Google, walk top N organic results, extract title + url + snippet | google.com (logged-out) |
| [`docs_lookup.json`](docs_lookup.json) | Navigate to a public docs site, search for a term, screenshot the result page | docs.python.org |

## Running

```bash
curl -fsS -X POST "$ENDPOINT/v1/predict" \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d @examples/form_fill.json
```

## Authoring your own

A plan is a JSON array of step objects. Each step has:

- `intent` — one human-readable sentence describing what this step accomplishes
- `type` — one of `holo3` (tactical action), `claude` (extract / verify / ground),
  `loop` (repeat a sub-sequence), `navigate` (URL change), `wait`
- type-specific fields (selectors, URLs, max steps, loop counts)

See `docs/getting-started/plan-formats.md` for the full schema and
`docs/integrations/recipes.md` for pattern-level guidance.

## A note on caps

Server-side caps (`MANTIS_MAX_STEPS_PER_PLAN=200`, `MAX_RUNTIME_MINUTES=60`,
`MAX_COST_USD=25`) apply to every plan. Examples here all fit comfortably
within the defaults — if you raise these for your tenant, document why.
