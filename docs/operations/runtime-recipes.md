# Runtime recipe registration

Code-shipped recipes under `src/mantis_agent/recipes/<name>/` require
a fork + redeploy to add or change. **Runtime recipes** live on the
shared volume at `/data/tenants/<tenant>/recipes/<name>.json` and can
be registered, fetched, and deleted over HTTP — no redeploy.

Use runtime recipes for:

- Domain-specific `spam_indicators`, `forbidden_controls`,
  `listing_card_exclusions`, `rejection_intents` that don't make sense
  to ship with the core code.
- Per-tenant overrides of a shipped recipe (e.g. extend
  `marketplace_listings` with your own spam tokens).
- Quick experiments — add, run, observe, iterate without a deploy.

For one-off plans, the inline `extract` block (see
[Sending plans / inline extraction schema](../client/plans.md#inline-extraction-schema))
is usually enough. Reach for runtime recipes when the same schema
shape is reused across many plans.

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| `POST` | `/v1/recipes` | `{"name": "<slug>", "schema": {...}}` | Persisted `{name, schema}` envelope |
| `GET` | `/v1/recipes` | — | `{"recipes": [{"name": "..."}]}` |
| `GET` | `/v1/recipes/{name}` | — | `{name, schema}` or 404 |
| `DELETE` | `/v1/recipes/{name}` | — | `{name, deleted: bool}` |

All routes require the same `X-Mantis-Token` as `/v1/predict` and are
scoped to the caller's tenant — recipes registered under tenant A are
not visible to tenant B.

## Name rules

- Letters, digits, hyphen, underscore. No dots, slashes, or spaces.
- 1–64 chars.
- Cannot start with `-`.

These constraints exist so the name maps cleanly to a filesystem path
and can't be used for traversal.

## Schema payload

The `schema` field accepts the same shape as the inline `extract`
block — internally it's parsed by `ExtractionSchema.from_dict`.

```bash
curl -fsS -X POST "$ENDPOINT/v1/recipes" \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_marketplace",
    "schema": {
      "entity_name": "property",
      "fields": [
        {"name": "address", "type": "str", "required": true},
        {"name": "price",   "type": "str", "required": true},
        {"name": "beds",    "type": "str", "required": false},
        {"name": "url",     "type": "str", "required": true}
      ],
      "required_fields": ["address", "price", "url"],
      "spam_indicators": ["sponsored", "featured"],
      "forbidden_controls": ["Contact Agent", "Request Tour"],
      "rejection_intents": {"sponsored": "skip"}
    }
  }'
```

Validation runs at write time — malformed payloads return `400` at
registration, not at extract time. The recipe is then immediately
usable by any plan submitted under that tenant.

## Resolution precedence

When a plan references a recipe by name (e.g. `_recipe: "my_marketplace"`
in the task suite envelope), the resolver:

1. Looks for a runtime recipe under the caller's tenant directory.
2. Falls back to the code-shipped recipe with that name.

So a tenant can override a shipped recipe (`marketplace_listings`,
`job_listings`, …) by registering a runtime recipe with the same name —
the override only affects that tenant.

## Limits and caveats

- **Tenant-scoped.** No cross-tenant sharing; if you need a shared
  taxonomy across tenants, ship it as a code recipe.
- **No versioning.** A `POST` overwrites the existing recipe of the
  same name. Use distinct names (`my_marketplace_v2`) if you want side-
  by-side variants.
- **Best-effort persistence.** The store writes a single JSON file per
  recipe to the shared volume. Failed writes raise `500`; clients
  should treat the operation as not-yet-applied and retry.
- **Not yet:** plan / loop / rewards overlays. Only `ExtractionSchema`
  is registrable at runtime today. The shipped-recipe surface still
  carries `plan.json`, `loop_overrides.py`, etc.; runtime equivalents
  are deferred.

## Python SDK helpers

Coming in a follow-up. Use `requests`/`httpx` directly meanwhile —
the endpoints are deliberately small and the JSON shape is stable.

## See also

- [Sending plans](../client/plans.md) — when to use inline extract vs a recipe
- [Recipes (code-shipped)](../integrations/recipes.md) — the static
  recipe inventory
