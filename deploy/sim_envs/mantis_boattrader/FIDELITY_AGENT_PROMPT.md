# Fidelity-pass prompt for a coding agent

How to instruct an agent (Claude Code, Cursor, etc.) to bring a sim-env
page to pixel parity with a real site. Tested on `mantis_boattrader`
vs `boattrader.com` — same shape applies to any sim env that mirrors a
public site.

> The agent has tools to drive Chrome via MCP (`mcp__claude-in-chrome__*`),
> can edit files, and can push code to the running Daytona sandbox via
> `_daytona_patch.py`. If your harness differs, adapt steps 1 and 4.

---

## Prompt body

You are fixing visual fidelity of a sim-env page against a real website.
Both must be open in Chrome MCP **in the same window**. Do not trust
visual intuition alone — every claim must be backed by a
`getBoundingClientRect` + `getComputedStyle` measurement.

### 0. Pre-flight

- Confirm the sandbox is up:
  ```
  curl -H 'x-daytona-preview-token: <tok>' <sandbox>/__env__/health
  ```
  Expect `{"ok":true,...}`. If 400 with `"Is the Sandbox started?"`, run
  `daytona.get(sid).start()` (see `feedback_boattrader_sandbox_restart_recipe.md`).
- Create a worktree on the current branch:
  ```
  git worktree add .claude/worktrees/<name> -b <branch> HEAD
  ```

### 1. Twin tabs, identical viewport

- Open real site + sandbox in **the same Chrome window**. Different
  windows give different `innerWidth` → measurements are not comparable.
- Resize to a canonical viewport (1440×900 unless otherwise specified).
- Scroll both tabs so the target region is at the same y-offset:
  ```js
  const el = document.querySelector('<target>');
  window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - 20);
  ```

### 2. Structural diff (measurements before pixels)

For each element in the target region, run this in both tabs:

```js
const el = document.querySelector(sel); // or find by text
const r = el.getBoundingClientRect();
const cs = getComputedStyle(el);
({
  w: r.width|0, h: r.height|0, x: r.x|0,
  fs: cs.fontSize, fw: cs.fontWeight, color: cs.color,
  bg: cs.backgroundColor, br: cs.borderRadius, b: cs.border,
  p: cs.padding, m: cs.margin,
  sh: cs.boxShadow.slice(0, 80),
  td: cs.textDecoration.slice(0, 40),
  ta: cs.textAlign,
})
```

Produce a side-by-side table. Anything off by ≥3px or with a different
unit (transparent vs `#fff`, 1px vs 2px, underline vs none) is a row to fix.

### 3. Perceptual diff (after measurements, not before)

- `computer.zoom` with `region=[x, y, x+w, y+h]` on **both** tabs, same
  rectangle.
- Save both with `save_to_disk: true`.
- Compare visually. The structural table tells you *what* to fix; the
  screenshot tells you *whether the fix worked.*

### 4. Iterate

- Edit CSS/HTML in the worktree.
- Push to the running sandbox:
  ```
  .venv/bin/python deploy/sim_envs/_daytona_patch.py <sandbox-id>
  ```
- Reload sandbox tab + re-measure. **Don't believe the screenshot until
  the numbers match too.**
- Each iteration of the loop is one CSS edit + one push + one re-measure.
  Don't push between every property change — batch into a single edit.

### 5. Interaction parity

- For each interactive element: click it in both tabs, type the same input,
  observe DOM/URL changes.
- Mirror the behavior in sim env JS (e.g., auto-submit on 5-digit zip).
- Re-screenshot the post-interaction state.

### 6. Land + book-keep

- Update `FIDELITY.md`: flip ✅ on each row you fixed, bump the
  `Last updated: v=N` line and the current preview token if it rotated.
- Commit on the worktree branch with one PR per fidelity pass.
- Use the perceptual-diff harness for final verification:
  ```
  .venv/bin/python deploy/sim_envs/mantis_boattrader/scripts/perceptual_diff.py \
      --sandbox <url> --token <tok> --region <region>
  ```
  Expect "✓ no structural deltas" before opening the PR.

---

## Anti-patterns (call these out explicitly so the agent doesn't fall in)

- **Don't trust the user's screenshot as canonical.** It may have been
  taken at a different viewport. Re-screenshot both yourself at the same
  viewport before diagnosing.
- **Don't fix one thing at a time and redeploy 12 times.** Batch all
  measured deltas into a single CSS edit, push once, re-verify.
- **Don't measure when `innerWidth: 0`.** That means the tab is in a
  hidden window; close and re-open in the focused window before trusting
  any rect data.
- **Don't add JS for static layout.** Every fidelity fix is
  `selector { property: value }`. If you find yourself writing JS to size
  an element, the CSS is wrong.
- **Don't strip the outer card.** Real boattrader.com renders sections
  borderless, but our sandbox intentionally keeps the 1px outline + 6px
  radius around `.filters-form`. Match it where the structure differs,
  keep it where it's an intentional sim-env affordance. Confirm with the
  user before removing visible chrome.
- **Don't `git add -f` adhoc verification scripts.** Save under
  `deploy/sim_envs/<env>/scripts/` if they're reusable; otherwise write
  to `/tmp/` (see `feedback_no_adhoc_submit_scripts_in_repo.md`).

---

## Quick checklist (copy into a TaskCreate at session start)

1. [ ] Sandbox health 200, token current in FIDELITY.md
2. [ ] Worktree created on current branch
3. [ ] Both tabs open in the same Chrome window at 1440×900
4. [ ] Structural delta table captured for target region
5. [ ] Perceptual screenshots saved at matched rectangle
6. [ ] CSS/HTML edits applied + pushed via `_daytona_patch.py`
7. [ ] Re-measure → numerical deltas ≤2px / 0 unit mismatches
8. [ ] Interaction parity tested (focus state, typing, submit)
9. [ ] FIDELITY.md updated (✅ rows + version bump + token)
10. [ ] Commit + PR open on worktree branch
