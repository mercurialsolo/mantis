# Prompt: "Build a high-fidelity simulation of <site> autonomously"

This is the **build-from-scratch** variant of the fidelity prompt — it
covers everything an agent needs to clone a real site into a sim env
where the agent can capture *both UI and interactions autonomously*.

For the **gap-fix** variant (used when a sim env already exists and you
need to bring it to pixel parity with a specific real site), see the
sibling doc `FIDELITY_AGENT_PROMPT.md`.

---

You are building a complete sim env that mirrors `<target-url>` — same
pages, same layout, **same interactions**. The result must be drivable
end-to-end by an agent training pipeline without falling back to the
real site. Failure modes to avoid: superficial cloning that breaks on
edge cases, hidden gaps that surface only during agent training, and
"looks close enough" calls the user can't audit later.

## Mental model: four loops, not one

1. **Discovery** — walk every page + interaction on the real site,
   write everything to a captured-state corpus on disk
2. **Visual fidelity** — section-by-section CSS/template parity,
   driven by structural diffs + perceptual screenshots
3. **Interaction fidelity** — replay each captured interaction against
   both real and sim, diff results, plug gaps
4. **Verification harness** — automated tests that catch any
   regression

Each loop produces evidence the user can audit. **Don't compress
phases.** Discovery is the hardest to revisit later — capture
aggressively now even if you're not sure you need it.

## Phase 0 — Scoping (with the user, not autonomously)

Before any code, get sign-off on:

- **In-scope pages** — explicit list of URLs, including `?query=`
  variants that matter
- **In-scope interactions** — every flow (search, filter, navigate to
  detail, submit form, save preference, login)
- **Viewport(s)** — 1440×900 / 390×844 / both / responsive
- **Backend** — static catalog? read-only seeded? full CRUD with
  persistence? do interactions mutate visible state?
- **Out-of-scope** — ads, analytics, A/B variants, third-party widgets
  (match visual, stub behavior)
- **Done bar** — structural deltas ≤2px / interaction replay matches
  URL+DOM+network deltas / verification harness green

Save to `<sim-env>/SCOPE.md`.

## Phase 1 — Discovery (capture, don't write)

For each in-scope URL, capture the DOM, every element's computed
style, asset inventory, and event map into
`<sim-env>/_captured/<slug>/{dom.html, styles.json, assets.json,
events.json, screenshot.png, network.har}`.

```js
// Per-element structural probe — run in Chrome MCP on the target page
const probe = el => {
  const r = el.getBoundingClientRect(); const cs = getComputedStyle(el);
  return { selector: cssPath(el), tag: el.tagName, classes: [...el.classList],
    rect: {w:r.width|0, h:r.height|0, x:r.x|0, y:r.y|0},
    style: pick(cs, ['fontSize','fontWeight','color','backgroundColor',
      'border','borderRadius','padding','margin','boxShadow',
      'textDecoration','textAlign','display','position']),
    text: el.textContent?.slice(0,200) };
};
JSON.stringify([...document.querySelectorAll('*')].map(probe))
```

For each in-scope interaction, drive it via `computer.left_click` +
`type`. Record before/after DOM snapshots, URL, network calls, console
events as `_captured/_interactions/<flow>/step-N-{before,action,after}.json`.

**Do not write sim code yet.** This corpus is the spec. If you code
before the corpus is done, you'll miss things and find them only
during agent training.

## Phase 2 — Scaffolding (generate from corpus)

Pick the stack to match the target's complexity:

- Server-rendered, minimal JS → FastAPI + Jinja (like
  `mantis_boattrader`)
- SPA with client routing → Next.js or Vite + React **only** if the
  real site uses one — don't over-engineer

Generate:

- Templates from `dom.html` (strip third-party scripts; Jinja-loop
  data-driven sections)
- `app.css` from `styles.json`, deduped, grouped by component
- Fixtures matching the captured catalog shape, seeded by `SEED` env
  var for determinism
- One backend route per captured interaction

Stand up the sim. Verify the first page loads. **Don't iterate on
visual fidelity yet** — that's Phase 3.

## Phase 3 — Visual fidelity loop (per section, per page)

1. **Twin tabs** — real + sim in the **same Chrome window**, same
   viewport. Different windows give different `innerWidth` →
   measurements aren't comparable.
2. **Scroll-align** — both tabs to the same y-offset relative to the
   target section.
3. **Structural diff** — run the probe in both, side-by-side table.
   Any row off by ≥3px or with a unit mismatch (transparent vs `#fff`,
   1px vs 2px, underline vs none) is a fix.
4. **Perceptual diff** — `computer.zoom` with **matched
   `region=[x,y,x+w,y+h]`** on both tabs, `save_to_disk: true`. The
   structural table tells you *what* to fix; the screenshot tells you
   *whether the fix worked*.
5. **Edit CSS/HTML in worktree**. Batch all measured deltas into one
   edit.
6. **Hot-patch** — `_daytona_patch.py <sandbox-id>` (uploads files +
   restarts uvicorn). Faster than a redeploy.
7. **Re-measure**. Don't trust the screenshot until the numbers match
   too. Repeat until deltas ≤2px / 0 unit mismatches.
8. **Flip ✅ in FIDELITY.md**. Commit.

States to cover for each interactive element: `default`, `hover`,
`focus`, `focus-visible`, `active`, `disabled`, `error/invalid`,
`loading`.

## Phase 4 — Interaction fidelity loop (per flow)

For each captured interaction:

1. **Replay in real**: drive via Chrome MCP, observe URL/DOM/network
   change.
2. **Replay in sim**: drive same actions, observe same.
3. **Diff**:
   - **URL** — same path/query after action? Same canonical form?
     (e.g. real BT navigates to `/boats/state-ny/city-new-york/zip-10001/`
     on 5-digit zip — sandbox needs a matching debounced submit.)
   - **DOM** — same visible text/elements appear/disappear?
   - **Network** — equivalent backend mutation? (POST `/contact` etc.)
   - **Timing** — does the sim auto-fire at the same trigger?
4. **Plug the gap** — add JS handler or backend route. Keep behavior
   minimal; no copying analytics.
5. **Verify** — re-replay, confirm zero diff.

Behaviors agents tend to forget unless prompted:

- Focus state changes (input → blue border) — visible during typing
- Auto-submit on completion (5-digit zip, valid email)
- Optimistic UI updates (favorite toggles before backend confirms)
- Empty / error / loading / validation states
- Modal open/close + scroll lock
- Dropdown autocomplete with URL rewrites
- Cookie consent + dismissal persistence
- Pagination edge cases (page 1, last page, empty result)

## Phase 5 — Verification harness

Build an automated regression suite the user can run on every PR:

```python
# <sim-env>/scripts/fidelity_check.py
# For each (real_url, sim_url, region) in VISUAL_CASES:
#   capture both via playwright
#   probe each element listed in FIDELITY.md
#   assert structural deltas <= tolerance
#   assert perceptual diff (pixelmatch < 0.5%)
# For each (interaction_name, steps) in INTERACTION_CASES:
#   replay against both
#   assert URL / visible-text / network match
# Emit per-section pass/fail report.
```

This is the gate that catches future regressions. Wire into CI.

## Phase 6 — Bookkeeping (each is auditable separately)

- **`FIDELITY.md`** — section-by-section matrix `(Element | Real | Sim
  | Status)`. ✅ / 🟡 / ⏳ / ❌ / 🚫.
- **Iteration log** at bottom of FIDELITY.md — one line per version
  bump.
- **`SCOPE.md`** — current in-scope pages + interactions.
- **`FIDELITY_AGENT_PROMPT.md`** — gap-fix playbook.
- **`FIDELITY_BUILD_FROM_SCRATCH_PROMPT.md`** — this doc, the
  build-from-scratch playbook.
- **`_captured/` corpus** — checked in. It's the spec the agent diffs
  against if the real site changes.

## Anti-patterns

1. **Don't trust visual intuition.** Every claim is backed by a
   measurement.
2. **Don't pick the user's screenshot as canonical.** It may have been
   at a different viewport. Re-capture both at the agreed viewport
   before diagnosing.
3. **Don't compress discovery.** Capturing every page + interaction
   upfront is cheap; backfilling missed flows after writing code is
   expensive.
4. **Don't redeploy per property change.** Batch all deltas in a
   section into one edit + one push + one re-measure.
5. **Don't measure when `innerWidth: 0`.** That means the tab is in a
   hidden window. Re-open in the focused window first.
6. **Don't write JS for static layout.** Every visual fidelity fix is
   `selector { property: value }`. If you're writing JS to size an
   element, the CSS is wrong.
7. **Don't promise byte-identical screenshots.** Fonts and
   anti-aliasing differ. Aim for structural parity + perceptual parity
   (`pixelmatch < 0.5%`).
8. **Don't silently skip elements.** Every skip is a 🚫 or 🟡 row in
   FIDELITY.md with a one-line reason. The user can re-prioritize.
9. **Don't copy analytics / ads / tracking.** Stub visually, don't
   replicate the data plane.
10. **Don't trust "the user said it's fine" in conversation.** Capture
    the assumption in SCOPE.md.

## Done bar

- ✅ for every row in FIDELITY.md across all in-scope sections
- 100% of in-scope interactions replay with matching URL/DOM/network
  deltas
- Verification harness green in CI
- `_captured/` corpus checked in
- `SCOPE.md` + `FIDELITY_AGENT_PROMPT.md` present
