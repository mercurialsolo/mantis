# Recipe — news.ycombinator.com

**Plane:** Browser-Use Plane (`runtime.compute_backend: browser_use_plane`)
**Reference plan use case:** [HN URL collection user-feedback report](https://github.com/mercurialsolo/mantis/issues/785)

## Why this recipe exists

HN's news list row has six clickable elements per item that look visually similar — title, domain badge, vote arrow, author, age, comment count. Vision-only grounding (`run_holo3`, `run_claude_cua`) confuses them routinely (#781). This recipe maps the plan-level `target_role` field to stable CSS selectors so click steps can disambiguate semantically without giving up the no-DOM-for-grounding rule on pure-CUA executors (`feedback_cua_no_dom_access.md`).

## target_role → selector mapping

Site detection: any URL under `https://news.ycombinator.com/`.

| `target_role` | Selector | Notes |
|---|---|---|
| `title` | `tr.athing td.title > span.titleline > a` | The story title link — external URL for most submissions. |
| `domain_badge` | `tr.athing td.title > span.titleline > span.sitebit > a` | The faded `(domain.com)` parenthetical after the title. |
| `vote_button` | `tr.athing td.votelinks center > a.clicky` | The upvote triangle — modifies user state; rarely a plan target. |
| `author` | `tr + tr td.subtext a.hnuser` | The submitter handle in the metadata row. |
| `age` | `tr + tr td.subtext span.age > a` | The "N hours ago" link → permalink view. |
| `comment_count` | `tr + tr td.subtext a:last-child` | The "N comments" link → discussion thread. |

The metadata row sits **immediately after** the title row; the `tr + tr td.subtext` adjacent-sibling selector targets it relative to the title row.

## Example plan — collect outbound article URLs (issue #785)

```yaml
runtime:
  compute_backend: browser_use_plane

steps:
  - intent: "Open Hacker News front page"
    type: navigate
    url: https://news.ycombinator.com

  - intent: "For each story title on this page, capture its outbound URL"
    type: loop
    loop_count: 30
    steps:
      - intent: "Open story title link in a new tab and read its URL"
        type: capture_link_in_new_tab
        target_role: title
        source_selector: "tr.athing td.title > span.titleline > a"
        emit: current_url
        on_no_navigation: skip
```

The `capture_link_in_new_tab` handler:

1. Calls `links.peek_target(source_selector)` — pre-flight read of the anchor `href` so the handler can short-circuit if the target is internal (`item?id=...`) when the plan only wants outbound URLs.
2. Calls `tabs.open_in_new(via_selector=source_selector)` — modifier-aware click that yields a popup.
3. Calls `state.current_url()` on the new tab (after activating it briefly).
4. Calls `tabs.close(tab_id)` — explicit teardown; the dispatch cache holds the response for retries.
5. Returns the captured URL as a row in the per-step output.

## Notes

- The `target_role` field is **ignored** by pure-CUA executors. Plans that rely on it must declare `runtime.compute_backend: browser_use_plane` (#785).
- Selectors above are observed at 2026-06-07. HN's markup is stable but not contractually so; if a selector starts missing, log a recipe update PR — don't silently fall back to vision.
- Vision fallback is intentional when `target_role` resolves to no selector for the current site — keeps the plan author's intent observable rather than crashing.
