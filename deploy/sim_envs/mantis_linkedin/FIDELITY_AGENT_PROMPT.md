# mantis-linkedin — gap-fix agent prompt

You are picking up a fidelity-iteration session against the mantis-linkedin
sim env. The end state bar is a high-fidelity replica of linkedin.com
(structural deltas ≤2px, perceptual diff <0.5%, every in-scope interaction
replays with matching URL/DOM/network deltas).

## Read first (in order)

1. `SCOPE.md` — in-scope pages, in-scope interactions, done bar.
2. `FIDELITY.md` — current match matrix + iteration log + open gaps.
3. `_captured/README.md` + each `_captured/<slug>/notes.md` — the ground-truth
   spec for the surface you're improving.
4. `app/main.py`, `app/routes/<surface>.py`, `app/templates/<surface>.html`,
   `app/static/site.css` — the actual implementation.

## Loop

For each row in `FIDELITY.md` not at `exact`:

1. **Verify the spec.** Re-read `_captured/<slug>/notes.md`. If a live
   Chrome-MCP capture for this slug becomes available, trust it over the
   offline notes — but flag any disagreement in `FIDELITY.md` so the user
   can audit.
2. **Identify the delta.** Open the rendered page locally
   (`ENV_ADMIN_TOKEN=test python -m uvicorn app.main:app --port 8090`) and
   measure. Use DevTools box-model + computed styles.
3. **Decide.** Cosmetic CSS in `app/static/site.css`. Structural fixes in
   templates. Behavioural fixes in routes. Never touch tests to mask
   regressions.
4. **Patch + re-verify.** Run `python scripts/smoke.py` after every patch.
   It must stay green.
5. **Update FIDELITY.md.** Bump the row status and append to the iteration
   log at the bottom — date stamp + 1-line summary of what landed.

## Hard rules

- Single CSS file (`site.css`). Group by component. Don't fragment.
- No JS framework. Small vanilla JS only where the interaction demands it
  (modals, multi-step Easy Apply, Send button enable/disable).
- No real brand assets. Placeholder SVGs / colour blocks only.
- Every state-changing route writes to `audit_log` with `(operation,
  target_type, target_id, payload_json, occurred_at)`. Oracles grade by
  reading audit_log + state — don't migrate to deriving from current state.
- Smoke MUST stay green. Don't ship a patch that breaks
  `scripts/smoke.py`.

## Anti-patterns

- **Drift on the brand palette.** The hex codes in
  `_captured/README.md` are canonical. Don't introduce a new shade of
  blue without verifying.
- **One-off "fix" to one page that breaks shared components.** A change
  to `.btn-primary` cascades — re-check every page.
- **Removing data-testid attributes.** Templates carry stable
  data-testid hooks (e.g. `post-card`, `connect-modal`,
  `easy-apply-modal`). Keep them — the CUA harness uses them indirectly
  via vision-derived element identification.

## Done check before handing off

- `python scripts/smoke.py` → SMOKE PASSED.
- `curl -fs http://127.0.0.1:8090/__env__/health` → `ok=true`.
- `FIDELITY.md` iteration log has a new dated entry.
- Any new `not-matched` or `partial` rows have a 1-line reason.
