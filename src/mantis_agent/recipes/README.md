# Recipes

A **recipe** is a self-contained, vertical-specific bundle of plan + extraction
schema + (optional) verifier that the generic Mantis core can compose.

Recipes are how Mantis stays a generic CUA agent while still shipping working
patterns for common tasks. The core never imports from `recipes/`. Plans
declare a recipe by name; the loader pulls in only what's needed.

## Shipped recipes

| Name | What it does | Sites tested |
|---|---|---|
| [`marketplace_listings`](marketplace_listings/) | Vehicle / boat / RV listings → structured row per item (year / make / model / price / phone / seller / spam) | BoatTrader and comparable marketplaces |
| [`marketplace_planner`](marketplace_planner/) | Legacy CLI: text marketplace plan → Claude-Opus micro-plan + browse summaries. Not on the default path; use `--micro` + `marketplace_listings` instead. | BoatTrader |
| [`staff_crm`](staff_crm/) | Staff / internal CRM admin console — supplies URL-filter encoding for sidebar / dropdown filters that aren't reliably clickable | Staff CRM-style consoles |
| [`job_listings`](job_listings/) | Public jobs board → first N roles with title / team / location / url | Greenhouse / Lever / Workday-style boards |
| [`form_submit`](form_submit/) | Open + fill + submit a form, capture the response | `httpbin.org/forms/post` and similar single-page forms |
| [`search_results`](search_results/) | SERP → top N organic results as `{title, url, snippet}` | Google / Bing / DuckDuckGo |
| [`docs_lookup`](docs_lookup/) | Search a docs site for a query, capture the top hit's title + first paragraph | `docs.python.org`, Sphinx / MkDocs / Docusaurus sites |

## Layout

```
recipes/
  marketplace_listings/      e.g., year/make/model/price/phone schema
    README.md
    plan.json
    schema.py                ExtractionSchema
    verifier.py              optional StepVerifier subclass
  job_listings/              README.md + plan.json
  form_submit/               README.md + plan.json
  search_results/            README.md + plan.json
  docs_lookup/               README.md + plan.json
```

## Contract

Each recipe directory must contain:

- `README.md` — what it does, what sites it has been tested against, known limits
- `plan.json` — a valid Mantis micro-plan
- `schema.py` *(optional)* — a module exposing `SCHEMA: ExtractionSchema`
- `verifier.py` *(optional)* — a module exposing `VERIFIER: StepVerifier`

The recipe must NOT:

- Import customer-specific configuration (selectors, URLs, account names)
- Mutate global state on import
- Pull heavyweight optional deps (torch / playwright) — those belong in
  the brain layer, not the recipe layer

## Status

This directory is a target structure. The current codebase still has
vertical-specific code embedded in the core (`extraction.py`, `rewards/`,
`verification/playbook.py`, `site_config.py`) — see the open issue for the
extraction work. New verticals should go here from day one; legacy ones
will be migrated incrementally.
