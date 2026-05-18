# staff_crm

Recipe overlay for staff / internal CRM admin consoles — lead list with
sidebar filter views, Priority dropdown, row-link detail pages, edit
form, in-app messaging.

## What it provides

- **`SiteConfig.filter_url_strategies`** — maps sidebar filter keywords
  (`contacted`, `qualified`, `high`, `critical`, …) to URL query-string
  segments (`status=Contacted`, `priority=High`, …) so callers that
  resolve a recipe name (`GraphLearner` with `recipe_name="staff_crm"`,
  or any other caller threading the recipe through `PlanEnhancer`) can
  emit direct URL navigations for filter-change steps instead of
  relying on sidebar-link clicks.

  This is the workaround for two common fixture-friction shapes on
  staff CRMs:
  1. Sidebar filter links that have a broken `onclick` interceptor and
     don't actually navigate when clicked.
  2. Filter dropdown options that Holo3 can't ground visually inside an
     open menu.

  When the recipe is selected, the equivalent URL-encoded navigate
  bypasses both.
- **URL patterns** — `/leads/<id>` for detail, `/leads` (with optional
  query string) for the results list, `?page=N` for pagination.

## Layout

```
recipes/staff_crm/
  README.md
  __init__.py
  site_config.py     # exports SITE_CONFIG: SiteConfig
```

## Status

| | |
|---|---|
| On default path | No (overlay; selected by `recipe_name` argument) |
| Sites tested | Staff CRM-style internal admin consoles |
| Companion runtime piece | `gym/critic.py:_maybe_use_row_link_dom_href` — DOM-derived row-link href fallback for the table-row click step (Holo3 row-link grounding limitation) |
