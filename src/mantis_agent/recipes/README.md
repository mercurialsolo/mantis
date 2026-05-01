# Recipes

A **recipe** is a self-contained, vertical-specific bundle of plan + extraction
schema + (optional) verifier that the generic Mantis core can compose.

Recipes are how Mantis stays a generic CUA agent while still shipping working
patterns for common tasks. The core never imports from `recipes/`. Plans
declare a recipe by name; the loader pulls in only what's needed.

## Layout

```
recipes/
  marketplace_listings/      e.g., year/make/model/price/phone schema
    README.md
    plan.json
    schema.py                ExtractionSchema
    verifier.py              optional StepVerifier subclass
  jobs_board/
    README.md
    plan.json
    schema.py
  ...
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
