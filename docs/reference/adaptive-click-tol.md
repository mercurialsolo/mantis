# Adaptive click tolerance

`LoopDetector` flags coordinate-drift loops by comparing successive
click coordinates against a fixed `click_tol_px = 8`. That single
constant is wrong on two ends of the viewport spectrum:

- **4K (3840 × 2160)**: an 8 px tolerance is one third of one
  percent of the diagonal; legitimate retries on the same UI target
  routinely drift by 15–20 px and get flagged as loops.
- **Phone-class (1080p portrait)**: 8 px is roughly correct, but
  could be tighter on dense layouts where 8 px straddles two adjacent
  controls.

Element semantics also matter: a `Submit` button covers ~150 × 50 px
and tolerates wider drift; a navigation link is one word wide and
needs tighter detection so a click drifting onto the next link
doesn't read as the same target.

Tracking issue: [#296](https://github.com/mercurialsolo/mantis/issues/296).

## Surfaces

| Symbol | Returns |
|---|---|
| `compute_click_tol_px(viewport, *, floor=8)` | `int` — drift tolerance baseline scaled as `max(floor, 0.4 % × diagonal)`. |
| `LoopDetector._effective_click_tol(action)` | `int` — `click_tol_px × class_multiplier` when the action's reasoning text classifies the target. |
| `MANTIS_ADAPTIVE_CLICK_TOL` env var | `enabled` (default) / `disabled`. When off, `compute_click_tol_px` returns `floor` and `_effective_click_tol` returns the raw attribute. |

`GymRunner.__init__` calls `compute_click_tol_px(env.screen_size)` once
and passes the result as `LoopDetector(click_tol_px=…)`. Env access is
guarded — a misconfigured env that raises on `screen_size` falls back
to the floor.

## Scaling table

| Viewport | Diagonal | Tolerance |
|---|---|---|
| 1280 × 800 (default Mantis) | 1430 px | 8 px (floor) |
| 1366 × 768 (laptop) | 1567 px | 8 px (floor) |
| 1920 × 1080 (FHD desktop) | 2202 px | 9 px |
| 2560 × 1440 (QHD) | 2937 px | 12 px |
| 3840 × 2160 (4K) | 4404 px | 18 px |
| 7680 × 4320 (8K) | 8810 px | 35 px |

## Element-class multipliers

Reasoning-text keywords on the first sample of a drift window scale
the effective tolerance:

| Keyword (substring) | Multiplier | Rationale |
|---|---|---|
| `submit button` / `button` / `submit` | 1.5× | Large hit areas, brain re-clicks the centroid. |
| `dropdown` / `select` / `menu item` | 0.75× | Adjacent options must be distinguished. |
| `link` / `anchor` | 0.5× | Text targets — small drift = different word. |
| `listing` / `card` / `input field` / `form field` / `text field` | 1.0× | Default — no widening or tightening. |
| (no match) | 1.0× | Falls back to base `click_tol_px`. |

The keyword tuple is checked in order; earlier entries win on a tie.
Multi-word phrases (`submit button`) are listed before their substrings
(`button`) so future divergence in multipliers stays correct.

## Ablation

| Var | Default | Effect when set |
|---|---|---|
| `MANTIS_ADAPTIVE_CLICK_TOL` | `enabled` | `disabled` reverts both the screen-DPI baseline and the per-class multiplier — `LoopDetector` runs with the legacy hardcoded `click_tol_px`. |

`/v1/cua` accepts `adaptive_click_tol: true|false` in the request
payload to flip the toggle for that single request via the runtime's
per-request override surface.

```bash
.venv/bin/python scripts/ablate_v1_cua.py \
  --toggle adaptive_click_tol \
  --instruction "Extract events from lu.ma SF and paginate" \
  --start-url https://lu.ma/sf \
  --pairs 3
```

## Follow-ups

- `GroundingResult` has no `element_class` field today; the per-class
  multiplier reads keywords out of the action's free-form reasoning.
  When grounding gets a structured class hint, swap the substring
  match for the structured value.
