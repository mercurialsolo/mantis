# Mercor fidelity gap-fix prompt

You're a follow-up agent improving the `mantis_mercor` synthetic
mirror's match against `https://www.mercor.com/`. The first-pass
scaffold landed: all routes serve 200, three oracles pass via
`scripts/smoke.py`, and the section-by-section status lives in
`FIDELITY.md`. Your job is to convert 🟡-close rows to ✅-exact and
close any ⏳-partial / ❌-missing gaps.

## Read first

1. `SCOPE.md` — what's in / out of scope.
2. `FIDELITY.md` — the section-by-section match log.
3. `_captured/<slug>/notes.md` — captured spec per page.
4. `deploy/sim_envs/_capture_brief.md` — parent-session ground-truth
   palette + nav + typography for mercor.com.
5. The exemplar: `deploy/sim_envs/mantis_boattrader/FIDELITY.md`
   — the bar for pixel-parity, and the iteration log style.

## Workflow — one section per loop

For each section in `FIDELITY.md` ordered by impact (start at the
top — palette/typography/topbar; descend through hero, latest-roles,
jobs detail, apply, dashboard):

1. **Capture** the real Mercor section via Chrome MCP at 1440×900:
   - Load the page (`mcp__claude-in-chrome__navigate`).
   - Probe the section's bounding box + computed style via
     `mcp__claude-in-chrome__javascript_tool`, e.g.

     ```js
     (() => {
       const el = document.querySelector('h1');
       const r = el.getBoundingClientRect();
       const cs = getComputedStyle(el);
       return {
         text: el.innerText,
         rect: {x: r.x, y: r.y, w: r.width, h: r.height},
         font: cs.font,
         color: cs.color,
         padding: cs.padding,
         margin: cs.margin,
       };
     })()
     ```
   - Save the raw probe into
     `_captured/<slug>/styles.json` (append; don't overwrite past
     iterations).

2. **Diff** against current sim:
   - Boot the sim locally
     (`ENV_ADMIN_TOKEN=test python -m uvicorn app.main:app --port 8090`).
   - Probe the same section via Chrome MCP on
     `http://127.0.0.1:8090/<path>`.
   - Compute deltas. Anything > 2px structural or wrong-token
     palette is an action item.

3. **Patch** `app/static/site.css` or the matching template.
   Tweak component CSS in `site.css` (it's grouped by component —
   look for the matching `/* N. <component> */` banner). Don't bolt
   on inline styles or one-off classes; promote shared tokens to
   `:root` if you need new ones.

4. **Re-probe & verify** via Chrome MCP that the section now
   matches within 2px.

5. **Update `FIDELITY.md`** in place: flip the row to ✅ (or 🟡
   with a one-line note) and append a one-liner to the
   "Iteration log" section at the top with the section name +
   what landed.

6. **Re-run smoke** (`python scripts/smoke.py`) to confirm nothing
   regressed.

## Anti-patterns to avoid

- **Inline styles** on a template — keep all styling in
  `site.css`.
- **Page-scoped CSS** for shared components — if `.role-card`
  needs a tweak for `/jobs`, change `.role-card` once.
- **Touching the audit_log shape** — oracles are wired to the
  current payload keys. If you must add a key, append; never rename.
- **Adding Google Fonts at runtime** — the live site uses Inter
  but does not pull from Google Fonts at the path we capture.
  Keep the system-stack fallback.
- **Hardcoding seed-dependent values** — if a chart number is
  seed-driven, write a helper instead of hardcoding "75 hired
  recently" in the template.
- **Removing `data-testid` attributes** — the harness uses them
  to drive deterministic Chrome MCP-style locators.

## Suggested priority order

1. Hero H1 font size + line-height + letter-spacing (visible above
   the fold; biggest first-impression delta).
2. Role-card spacing — gap between rate, avatar, "N hired recently",
   Apply.
3. Stats strip layout — column widths, separator, label color.
4. Filter rail typography + chip spacing.
5. Job detail rail-card padding + sticky offset.
6. Apply step-list active-pill color + radius.
7. Dashboard table row padding + status badge size.

## What "done" looks like

- Every FIDELITY.md row that isn't 🚫 (intentionally not matched)
  reads ✅ or 🟡 with a 1-line justification.
- `scripts/smoke.py` still passes 100%.
- Iteration log at top of FIDELITY.md shows your work.
- No outbound network at runtime (verify with `--network none` if
  you spin up the container).
