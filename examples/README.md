# Example plans

These are committed, generic micro-plans you can run against a fresh Mantis
deployment. Unlike the customer-specific plans under `plans/` (gitignored),
everything here is public and works without proprietary data.

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
