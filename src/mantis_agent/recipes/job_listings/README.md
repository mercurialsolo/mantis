# Recipe: job_listings

Walks a public jobs board and extracts the first N listings as a
structured row per role. Designed for Greenhouse-hosted careers pages
where each listing is a clickable title in a flat table.

## What it does

1. Open the careers URL (e.g. `https://boards.greenhouse.io/<company>`).
2. Wait for the listings table to render.
3. For the first 5 unvisited listings:
   - Click the title to open the role detail page.
   - Extract `title`, `team`, `location`, `url`.
   - Press `Alt+Left` to return to the listings table.

Rows are returned as a JSON array of `{title, team, location, url}` dicts.

## Tested against

- `boards.greenhouse.io/mozilla` (and equivalent open careers boards).
- Generic enough to work against Lever / Workday-style boards where each
  listing has a clickable title and a distinct detail URL. Sites that
  hide listings behind an OAuth gate or require login will fail at the
  navigate step.

## Known limits

- `loop_count` is hard-set to 5 inside the plan; for fewer or more
  listings, edit the value in the recipe's `plan.json` (or copy it into
  your own deployment-level plan and adjust).
- The `team` and `location` fields are best-effort — boards that bury
  them in non-standard locations may return empty strings.
- The recipe assumes browser-back via `Alt+Left` returns to the listings
  view. Sites that intercept history navigation can break the loop —
  switch to an explicit `navigate` step targeting the listings URL if
  you see the loop stall on the second iteration.

## Usage

```bash
curl -fsS -X POST "$ENDPOINT/v1/predict" \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d @src/mantis_agent/recipes/job_listings/plan.json
```

Or reference by name once the dispatcher is wired:

```jsonc
{ "micro": "src/mantis_agent/recipes/job_listings/plan.json", "max_cost": 2 }
```
