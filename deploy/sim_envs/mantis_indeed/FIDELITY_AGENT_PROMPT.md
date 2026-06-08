# mantis_indeed — FIDELITY agent prompt

You're a follow-up agent closing a specific fidelity gap on the
`mantis_indeed` sim env. Treat this prompt as the playbook for every
iteration.

## 1. Read first

- `SCOPE.md` — what's in/out of scope.
- `FIDELITY.md` — current section-by-section status. Pick a 🟡 or 🔴
  row to close. Don't pick rows that are already ✅ unless you've found
  a regression.
- `_captured/<slug>/notes.md` — the ground truth.
- `app/templates/<page>.html` + `app/static/site.css` — current impl.

## 2. Pick exactly one gap per iteration

Don't fan out. Pick one row in FIDELITY.md (e.g. "filter-chip popovers
🟡") and close it to ✅ or 🟢.

## 3. The fidelity loop

For visual gaps:

1. Take a fresh capture of the live element (Chrome MCP javascript_tool
   probe — see `mantis_boattrader/FIDELITY_BUILD_FROM_SCRATCH_PROMPT.md`
   for the per-element probe).
2. Compare key declarations: width, height, padding, margin, border,
   border-radius, background, color, font-size, font-weight, line-height.
3. Update `app/static/site.css` to match. Avoid !important; group
   declarations by component.
4. Update FIDELITY.md row to ✅ or 🟢 with a one-line iteration log
   entry.

For interaction gaps:

1. Capture the live interaction's URL/DOM/network effect.
2. Wire the matching route in `app/routes/<surface>.py`.
3. Add a `db.log_audit(...)` call at the state-change boundary.
4. Run `python scripts/smoke.py`. Add a new check to smoke if relevant.
5. Update FIDELITY.md.

## 4. Forbidden

- Adding new pages outside `SCOPE.md`.
- Adding new pip deps without justification (slim Docker is a goal).
- Outbound network at runtime.
- Real brand assets (logos, photos).
- Plain `git add .` — stage individual files.

## 5. Done when

- The picked row is ✅ or 🟢.
- `python scripts/smoke.py` exits 0.
- FIDELITY.md is updated.
- No collateral regressions in adjacent rows.

## 6. Helpful patterns from `mantis_boattrader`

- Section-by-section CSS comments grouped by component name.
- One state per chip/button (default, hover, focus-visible, active,
  disabled).
- `data-testid` on every interactive node — the CUA harness can use
  these for grounding in offline tests (visually, the agent uses
  pixels; the testids are for our own e2e tests).
