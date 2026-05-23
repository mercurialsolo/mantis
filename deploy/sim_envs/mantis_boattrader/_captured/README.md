# `_captured/` — boattrader spec corpus

Phase 1 (Discovery) output per
`../FIDELITY_BUILD_FROM_SCRATCH_PROMPT.md`. Each subdir holds the
structural spec for one in-scope page: per-element measurements,
computed styles, classes, interaction notes.

This is the **spec** the sandbox is diffed against. When real
boattrader.com changes structurally, the captured snapshot here is the
record of what it looked like at capture time — re-capture when the
real site changes, then re-diff against the sandbox.

## Contents

| Path | Status | Source |
|---|---|---|
| `srp/structural.json`   | ✅ measured 2026-05-23 | Chrome MCP probes from real boattrader.com (1512×711 viewport) |
| `home/structural.json`  | ⏳ TODO placeholder    | listing of sections to capture; sources from FIDELITY.md |
| `bdp/structural.json`   | ⏳ TODO placeholder    | listing of sections to capture; sources from FIDELITY.md |

## Format

Each `structural.json` has a `_meta` block at top:

```json
{
  "_meta": {
    "captured_at": "<ISO-8601>",
    "url": "<real BT URL captured>",
    "viewport": {"width": <px>, "height": <px>, "devicePixelRatio": <n>},
    "captured_via": "Chrome MCP getBoundingClientRect + getComputedStyle"
  },
  "<section>": {
    "<element>": {
      "_real_class": "<real BT class name>",
      "rect": {"width": <px>, "height": <px>, "x": <px>},
      "style": {"<computed-style-prop>": "<value>"},
      "note": "<anything important the diff needs to know>"
    }
  }
}
```

## How to update

1. Open the real BT page in Chrome MCP at the canonical viewport
   (1440×900 — note the actual viewport may render at 1512 on macOS
   Retina; see SCOPE.md).
2. Run a per-element probe that returns
   `{rect, computed-style-tokens, classes, text}` for each FIDELITY.md
   row in scope.
3. Save the JSON here. Bump `_meta.captured_at`.
4. If the spec changes (e.g. real BT changed their CSS), update the
   matching row in `FIDELITY.md` and either fix the sandbox or move
   the row to SCOPE.md "Out-of-scope".

## What's NOT here

- **Raw `dom.html`** — full page HTML is 1MB+ and includes analytics
  scripts the sandbox doesn't mirror. Out of scope for the corpus.
- **`screenshot.png`** — pixel baselines are captured ad-hoc by
  `scripts/perceptual_diff.py` for local fidelity work; not committed
  to keep PR diffs small.
- **`network.har`** — request waterfall is captured only when
  reproducing an interaction bug; not part of the steady-state spec.
- **`events.json`** — handler bindings on each element (`onclick`,
  React props) are documented inline in each `structural.json`'s
  `interactions` block where they matter.

## Companion docs

- **`../SCOPE.md`** — what's in / out of scope
- **`../FIDELITY.md`** — current ✅/🟡/⏳/❌/🚫 status per element
- **`../FIDELITY_AGENT_PROMPT.md`** — gap-fix playbook
- **`../FIDELITY_BUILD_FROM_SCRATCH_PROMPT.md`** — full clone playbook
  (this corpus is its Phase 1 output)
