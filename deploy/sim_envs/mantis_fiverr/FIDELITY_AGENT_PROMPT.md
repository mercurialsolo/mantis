# Prompt — "Close a fidelity gap in mantis_fiverr"

This is the **gap-fix** playbook (the build-from-scratch variant lives
in `../mantis_boattrader/FIDELITY_BUILD_FROM_SCRATCH_PROMPT.md`).

You're handed:

- `mantis_fiverr` — a working sim env of fiverr.com (`docker run`
  ready; `scripts/smoke.py` green; FIDELITY.md table populated).
- A row in `FIDELITY.md` marked `partial` / `close` / `missing`.

Your job is to close ONE row (or a tight cluster of related rows in
the same section) per turn, then push the status forward by one bucket.

## Working loop

1. **Read the captured spec** — open `_captured/<slug>/notes.md` for
   the page and find the spec line for the element. If the row is
   based on training-data recollection (flagged at top of FIDELITY),
   drive Chrome MCP first and write the real measurements to
   `_captured/<slug>/styles.json`. THIS is your ground truth.

2. **Twin tabs** — open the live fiverr.com page + our sim env page in
   the SAME Chrome window at 1440×900. Different windows give
   different `innerWidth` and measurements aren't comparable. Confirm
   `window.innerWidth === 1440` on both.

3. **Structural diff** — run the per-element probe (from
   FIDELITY_BUILD_FROM_SCRATCH_PROMPT.md) on both tabs, side-by-side.
   Any row off by ≥3px, any colour/border/spacing unit mismatch is a
   fix.

4. **Perceptual diff** — `computer.zoom` with **matched
   `region=[x,y,x+w,y+h]`** on both tabs, `save_to_disk: true`. The
   structural numbers tell you WHAT to fix; the screenshot tells you
   WHETHER the fix worked.

5. **Edit CSS/HTML** in the worktree. Batch all measured deltas in
   the section into ONE edit before re-measuring.

6. **Re-measure**. Don't trust a screenshot alone — confirm the
   measurements match too.

7. **Update FIDELITY.md** — flip the row's status forward
   (`partial` → `close` → `exact`). Append an iteration log entry at
   the bottom of FIDELITY.md.

8. **Run smoke.py** — it must still pass. Visual fidelity changes that
   break interaction routes are bugs, not improvements.

## Anti-patterns to avoid

1. **Don't measure from the wrong window.** `innerWidth: 0` →
   different window → measurements are garbage.
2. **Don't fix without measuring.** "Looks closer" is not a metric.
3. **Don't redeploy per property change.** Edit → measure → commit.
4. **Don't skip the `_captured/` update.** Future agents diff against
   that corpus; stale captures lock in stale fidelity.
5. **Don't promise pixel-perfect screenshots across fonts.** Aim for
   `pixelmatch < 0.5%`. Fonts and anti-aliasing differ.
6. **Don't introduce framework-y JS for a layout fix.** A `selector {
   property: value }` is always cheaper. If you find yourself
   reaching for React or a framework — STOP, the CSS is wrong.
7. **Don't reorder route registration** without re-running the smoke
   test. The gig detail catch-all (`/{username}/{slug}`) MUST stay
   last (see Anti-route-pattern comment in `app/main.py`).

## When to call done

The row's status is updated to `exact` and:

- `scripts/smoke.py` still passes
- The captured measurement in `_captured/<slug>/styles.json` matches
  ours within tolerance
- A new iteration line is appended to FIDELITY.md
