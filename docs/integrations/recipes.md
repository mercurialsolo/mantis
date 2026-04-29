# Recipes — copy-paste plans for common patterns

Working starting points for the four extraction patterns most teams
onboard with. Each recipe is a complete `/v1/predict` request body —
swap the URL and the schema fields and you have a working pipeline.

> Same pattern across all four: **navigate → gate → click → extract_url
> → scroll → extract_data → re-navigate → loop**. The structure is
> universal; what changes is the schema and the start URL.

For the full request envelope (auth, polling, results), see
[Generic CUA usage](generic-cua.md). For when each piece kicks in, see
[Concepts](../getting-started/concepts.md).

---

## Recipe 1 — Job listings (Greenhouse / Lever / Workday-style)

**Best for:** boards with `/jobs/<id>` detail URLs and filterable list pages.

```bash
curl -fsS -X POST "$MANTIS_ENDPOINT/v1/predict" \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d @- <<'JSON'
{
  "detached": true,
  "state_key": "greenhouse-openai-eng-sf",
  "max_cost": 4,
  "max_time_minutes": 30,
  "extraction_schema": {
    "entity_name": "job posting",
    "fields": [
      {"name": "title",      "type": "str", "required": true,  "example": "Software Engineer"},
      {"name": "team",       "type": "str", "required": false, "example": "Infrastructure"},
      {"name": "location",   "type": "str", "required": false, "example": "San Francisco, CA"},
      {"name": "department", "type": "str", "required": false, "example": "Engineering"},
      {"name": "url",        "type": "str", "required": true,  "example": "boards.greenhouse.io/..."}
    ],
    "required_fields": ["title", "url"],
    "spam_indicators": ["recruiter", "staffing agency", "third party"],
    "spam_label": "recruiter spam"
  },
  "micro": [
    {"intent": "Navigate to https://boards.greenhouse.io/openai/jobs?department=Engineering&location=San+Francisco",
     "type": "navigate", "budget": 3, "section": "setup", "required": true},
    {"intent": "Verify page is the OpenAI Greenhouse engineering board, San Francisco filter active, listings visible",
     "type": "extract_data", "claude_only": true, "budget": 0, "section": "setup",
     "gate": true,
     "verify": "Page shows engineering job listings filtered to San Francisco"},
    {"intent": "Click the next un-extracted job posting title",
     "type": "click", "budget": 8, "grounding": true, "section": "extraction"},
    {"intent": "Read URL from address bar",
     "type": "extract_url", "claude_only": true, "budget": 0, "section": "extraction"},
    {"intent": "Scroll down to read the description",
     "type": "scroll", "budget": 5, "section": "extraction"},
    {"intent": "Extract title, team, location, department, url",
     "type": "extract_data", "claude_only": true, "budget": 0, "section": "extraction"},
    {"intent": "Re-navigate to the search page at https://boards.greenhouse.io/openai/jobs?department=Engineering&location=San+Francisco",
     "type": "navigate", "budget": 3, "section": "extraction"},
    {"intent": "Loop to next listing",
     "type": "loop", "loop_target": 2, "loop_count": 10, "section": "extraction"}
  ]
}
JSON
```

Lever variant: same shape, replace start URL with
`https://jobs.lever.co/<company>/?team=Engineering&location=...` and
the gate verify text accordingly.

---

## Recipe 2 — E-commerce product catalog (Amazon / Shopify-style)

**Best for:** product detail pages with `price`, `rating`, `availability` fields.

```bash
curl -fsS -X POST "$MANTIS_ENDPOINT/v1/predict" \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d @- <<'JSON'
{
  "detached": true,
  "state_key": "amazon-mech-keyboards",
  "max_cost": 3,
  "max_time_minutes": 25,
  "extraction_schema": {
    "entity_name": "product",
    "fields": [
      {"name": "name",         "type": "str", "required": true,  "example": "Keychron K6 Wireless"},
      {"name": "price",        "type": "str", "required": true,  "example": "$89.99"},
      {"name": "rating",       "type": "str", "required": false, "example": "4.5/5 (1,234)"},
      {"name": "availability", "type": "str", "required": false, "example": "In Stock"},
      {"name": "brand",        "type": "str", "required": false, "example": "Keychron"},
      {"name": "url",          "type": "str", "required": true,  "example": "amazon.com/dp/..."}
    ],
    "required_fields": ["name", "price", "url"]
  },
  "site_config": {
    "domain": "amazon.com",
    "detail_page_pattern": "/dp/[A-Z0-9]+",
    "results_page_pattern": "/s\\?",
    "pagination_format": "page={n}",
    "pagination_type": "query_param",
    "pagination_strip_pattern": "[?&]page=\\d+"
  },
  "micro": [
    {"intent": "Navigate to https://www.amazon.com/s?k=mechanical+keyboard",
     "type": "navigate", "budget": 3, "section": "setup", "required": true},
    {"intent": "Verify page is Amazon search results for mechanical keyboards",
     "type": "extract_data", "claude_only": true, "budget": 0, "section": "setup",
     "gate": true, "verify": "Page is an Amazon search results page for mechanical keyboards with multiple products visible"},
    {"intent": "Click the next un-extracted product title (skip sponsored tiles)",
     "type": "click", "budget": 8, "grounding": true, "section": "extraction"},
    {"intent": "Read URL from address bar",
     "type": "extract_url", "claude_only": true, "budget": 0, "section": "extraction"},
    {"intent": "Scroll down to see price + availability + reviews",
     "type": "scroll", "budget": 5, "section": "extraction"},
    {"intent": "Extract name, price, rating, availability, brand, url",
     "type": "extract_data", "claude_only": true, "budget": 0, "section": "extraction"},
    {"intent": "Re-navigate to https://www.amazon.com/s?k=mechanical+keyboard",
     "type": "navigate", "budget": 3, "section": "extraction"},
    {"intent": "Loop to next product",
     "type": "loop", "loop_target": 2, "loop_count": 8, "section": "extraction"},
    {"intent": "Click Next page",
     "type": "paginate", "budget": 10, "grounding": true, "section": "pagination"},
    {"intent": "Loop back to extract products on the new page",
     "type": "loop", "loop_target": 2, "loop_count": 3, "section": "pagination"}
  ]
}
JSON
```

Note the **two-level loop**: the inner loop walks listings on a page;
the outer loop paginates and re-runs the inner loop.

---

## Recipe 3 — News articles (any newsroom site)

**Best for:** structured headline + byline + body extraction.

```bash
curl -fsS -X POST "$MANTIS_ENDPOINT/v1/predict" \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d @- <<'JSON'
{
  "detached": true,
  "state_key": "techcrunch-ai-recent",
  "max_cost": 2,
  "max_time_minutes": 20,
  "extraction_schema": {
    "entity_name": "news article",
    "fields": [
      {"name": "headline",  "type": "str", "required": true,  "example": "OpenAI launches..."},
      {"name": "byline",    "type": "str", "required": false, "example": "Jane Doe"},
      {"name": "published", "type": "str", "required": false, "example": "2026-04-29T12:00:00Z"},
      {"name": "summary",   "type": "str", "required": false, "example": "first paragraph"},
      {"name": "url",       "type": "str", "required": true,  "example": "techcrunch.com/2026/..."}
    ],
    "required_fields": ["headline", "url"]
  },
  "micro": [
    {"intent": "Navigate to https://techcrunch.com/category/artificial-intelligence/",
     "type": "navigate", "budget": 3, "section": "setup", "required": true},
    {"intent": "Verify page is the TechCrunch AI category index with headlines visible",
     "type": "extract_data", "claude_only": true, "budget": 0, "section": "setup",
     "gate": true, "verify": "Page is TechCrunch's AI category page with multiple article headlines"},
    {"intent": "Click the next un-extracted article headline",
     "type": "click", "budget": 8, "grounding": true, "section": "extraction"},
    {"intent": "Read URL from address bar",
     "type": "extract_url", "claude_only": true, "budget": 0, "section": "extraction"},
    {"intent": "Scroll past the lede paragraph",
     "type": "scroll", "budget": 3, "section": "extraction"},
    {"intent": "Extract headline, byline, published, summary (first paragraph), url",
     "type": "extract_data", "claude_only": true, "budget": 0, "section": "extraction"},
    {"intent": "Re-navigate to https://techcrunch.com/category/artificial-intelligence/",
     "type": "navigate", "budget": 3, "section": "extraction"},
    {"intent": "Loop to next article",
     "type": "loop", "loop_target": 2, "loop_count": 5, "section": "extraction"}
  ]
}
JSON
```

For paywalled sources, supply login via the embedding path
([`register_tool`](embedding-microplanrunner.md)) — the HTTP-only path
doesn't have a credential-injection primitive.

---

## Recipe 4 — Real estate listings (Zillow / Redfin / Trulia-style)

**Best for:** listings with `address`, `price`, `beds/baths/sqft`, `listing agent`.

```bash
curl -fsS -X POST "$MANTIS_ENDPOINT/v1/predict" \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d @- <<'JSON'
{
  "detached": true,
  "state_key": "zillow-sf-condos-1m-2m",
  "max_cost": 5,
  "max_time_minutes": 35,
  "extraction_schema": {
    "entity_name": "real estate listing",
    "fields": [
      {"name": "address",      "type": "str", "required": true,  "example": "123 Main St, San Francisco, CA 94105"},
      {"name": "price",        "type": "str", "required": true,  "example": "$1,250,000"},
      {"name": "beds",         "type": "str", "required": false, "example": "2"},
      {"name": "baths",        "type": "str", "required": false, "example": "2"},
      {"name": "sqft",         "type": "str", "required": false, "example": "1,100"},
      {"name": "agent",        "type": "str", "required": false, "example": "Jane Smith, Compass"},
      {"name": "url",          "type": "str", "required": true,  "example": "zillow.com/homedetails/..."}
    ],
    "required_fields": ["address", "price", "url"],
    "spam_indicators": ["3D Home", "Zestimate-only", "off-market"],
    "spam_label": "non-listing"
  },
  "site_config": {
    "domain": "zillow.com",
    "detail_page_pattern": "/homedetails/[\\w-]+/\\d+_zpid",
    "results_page_pattern": "/homes/for_sale/",
    "pagination_format": "/{n}_p/",
    "pagination_type": "path_suffix"
  },
  "micro": [
    {"intent": "Navigate to https://www.zillow.com/homes/for_sale/San-Francisco-CA/condos_dom/?price=1000000-2000000",
     "type": "navigate", "budget": 3, "section": "setup", "required": true},
    {"intent": "Verify page is the SF condos for sale, $1M-$2M price range, with listings visible",
     "type": "extract_data", "claude_only": true, "budget": 0, "section": "setup",
     "gate": true, "verify": "Zillow SF condos $1M-$2M, multiple property cards visible"},
    {"intent": "Click the next un-extracted listing card",
     "type": "click", "budget": 8, "grounding": true, "section": "extraction"},
    {"intent": "Read URL from address bar",
     "type": "extract_url", "claude_only": true, "budget": 0, "section": "extraction"},
    {"intent": "Scroll to property facts and listing agent",
     "type": "scroll", "budget": 6, "section": "extraction"},
    {"intent": "Extract address, price, beds, baths, sqft, agent, url",
     "type": "extract_data", "claude_only": true, "budget": 0, "section": "extraction"},
    {"intent": "Re-navigate to https://www.zillow.com/homes/for_sale/San-Francisco-CA/condos_dom/?price=1000000-2000000",
     "type": "navigate", "budget": 3, "section": "extraction"},
    {"intent": "Loop to next listing",
     "type": "loop", "loop_target": 2, "loop_count": 12, "section": "extraction"}
  ]
}
JSON
```

Real estate sites are aggressive about Cloudflare; if you see `gate_failed`
in the result, increase `proxy_city` / `proxy_state` to a residential
target close to the listing geography.

---

## Picking the right `loop_count` and `max_cost`

Rough calibration (Holo3 / Claude / IPRoyal proxy on the Baseten H100):

| Items wanted | Recommended `loop_count` | Wall time | Typical cost |
|---|---|---|---|
| 3-5 | 5-8 | 6-10 min | $0.50-$1.00 |
| 10 | 15 | 15-20 min | $1.50-$2.50 |
| 20 | 30 | 30-40 min | $3-$5 |
| 50+ | Chunk via state_key resume | n/a | n/a |

Bump `loop_count` higher than your target (1.5×) — it absorbs failures on
spam-skipped listings and dead detail pages. Hard caps:
`MANTIS_MAX_LOOP_ITERATIONS=50`, `MANTIS_MAX_STEPS_PER_PLAN=200`.

For >50 items, split into multiple `/v1/predict` runs sharing the same
`state_key` and pass `resume_state: true` — the runner picks up at the
last checkpoint.

---

## Anti-patterns to avoid

| Don't | Do instead |
|---|---|
| `navigate_back` (browser back button) — Holo3 misclicks it ~30% | Use a fresh `navigate` step to the search URL |
| Plans without a `gate` step after `navigate` | Always add a Claude-verified gate so you halt fast on Cloudflare/wrong-page |
| `max_cost: 0.50` | Hits the cap mid-extraction; budget at least $1.50 for a 10-listing run |
| Hardcoding `"Sponsored"` skip logic in `intent` strings | Set `spam_indicators` in the schema; the runner skips automatically |
| `claude_only: false` on extract_data | Holo3 isn't a vision-language model; always use Claude for extraction |

---

## Next

- [Generic CUA usage](generic-cua.md) — the request envelope + tenant setup.
- [Embedding MicroPlanRunner](embedding-microplanrunner.md) — when you
  need pause/resume, host tools, or library-level control.
- [Plan formats](../getting-started/plan-formats.md) — every micro-step
  field documented.
