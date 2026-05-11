# Recipe: search_results

Pulls the top N organic results from a search-engine results page (SERP)
as a JSON array of `{title, url, snippet}` rows. Skips ads and "People
also ask" / "Featured snippet" blocks — only organic rows count.

## What it does

1. Open the SERP URL (the shipped plan uses
   `google.com/search?q=site%3Adocs.python.org+pathlib`).
2. Wait for organic results to render.
3. Single Claude extraction pass that reads the page screenshot and
   returns up to 5 organic rows.

Rows: `[{title, url, snippet?}, ...]`. `snippet` is optional — pages
that don't show snippets (e.g. heavily personalised SERPs) return
title + url only.

## Tested against

- `google.com/search?q=…` (logged-out, no personalisation cookies).
- Bing and DuckDuckGo SERPs work with the same plan — the extraction
  prompt asks for "the top N **organic** results" so the model
  generalises across SERP layouts.

## Known limits

- The query is baked into the `navigate.url` in `plan.json`. To make
  the recipe parametric, replace the URL with a `{{search_url}}`
  placeholder and substitute it caller-side (mirrors the
  `marketplace_listings` recipe's `{{search_url}}` convention).
- `max_items: 5` is the cap; raising it on rich SERPs gets diminishing
  returns because Google's "More results" pagination isn't followed.
  For full-page traversal, swap the single extract step for a loop +
  paginate pattern (see `job_listings` for the loop idiom).
- Sites that fingerprint headless browsers (Cloudflare on Bing's edge,
  reCAPTCHA on Google for some IPs) may serve a challenge page —
  enable the residential proxy and run headed if you see consistent
  empty results.

## Usage

```bash
curl -fsS -X POST "$ENDPOINT/v1/predict" \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d @src/mantis_agent/recipes/search_results/plan.json
```
