# Recipe: docs_lookup

Looks up a query on a public documentation site and captures the title
+ first paragraph of the top hit. Useful as a verification step in a
larger workflow ("did the help-link land on the right page?") or as a
single-shot lookup against any docs site with an on-page search box.

## What it does

1. Open the docs root (the shipped plan uses `https://docs.python.org/3/`).
2. Click the on-page search input via Holo3 (label is the `hint` —
   `"Quick search"` for Python docs).
3. Type the query and press Enter.
4. Capture `{page_title, first_paragraph?, url}` from the top result
   via Claude.

The output is one structured row, not a list — this recipe is intentionally
a single-hit lookup. For top-N organic results across a search engine
SERP, use the `search_results` recipe instead.

## Tested against

- `docs.python.org/3` (Sphinx-rendered Python stdlib docs).
- Adapts to any Sphinx / MkDocs / Docusaurus-style docs site with an
  on-page search affordance. To retarget, edit `navigate.url` and the
  Holo3 hints in `plan.json` to match the site's search input label.

## Known limits

- The search input label is encoded in the Holo3 hint (`"Quick search"`).
  Sites that label the input differently (`"Search the docs"`,
  `"Filter"`, no label) need the hint updated.
- `first_paragraph` is best-effort — pages where the first paragraph
  is hidden behind a tab or accordion may return an empty string. The
  recipe drops to `page_title + url` in that case rather than failing.
- The recipe assumes the top search hit is the desired page. Multi-result
  search drawers (e.g. when typing surfaces a dropdown of candidate
  pages) may require an extra `holo3` click step before the Enter
  submission.

## Usage

```bash
curl -fsS -X POST "$ENDPOINT/v1/predict" \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d @src/mantis_agent/recipes/docs_lookup/plan.json
```
