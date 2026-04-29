# Coordinate-space invariants for `GymEnvironment`

> Resolves [#75](https://github.com/mercurialsolo/mantis/issues/75). Companion
> reading: closed [#25](https://github.com/mercurialsolo/mantis/issues/25)
> ("CRITICAL — Resolution Mismatch (1.5x click offset)") — that bug is the
> reason this contract is now written down.

A `GymEnvironment` subclass that misinterprets the click coordinate space will
produce silent off-target clicks that look like "the model is bad". This
document is the source of truth that integrators (e.g. `VisionClaudeGymEnv` on
the StaffAI side) must follow.

---

## The contract

For any `Action` produced by the brain whose `action_type` carries spatial
coordinates (`CLICK`, `DOUBLE_CLICK`, `SCROLL`, `DRAG`):

1. **`Action.params["x"]` and `Action.params["y"]` are raw pixel offsets in
   the same coordinate space as the screenshot the brain consumed for the
   inference that produced the action.** No normalization (0–1), no DPR
   multiplication, no Y-axis flip. Origin is the top-left.
2. **`GymEnvironment.screen_size` returns the `(width, height)` of the display
   the env will dispatch the action to** — i.e. the destination space.
3. The brain's input image and the env's display *should be the same size*.
   When they differ (a host resizes screenshots before inference), it is the
   adapter's job to scale coordinates back to the env's display space inside
   `step()`. Do not push the burden onto the brain.

This means a `step()` implementation can rely on the action being expressed
in screenshot pixels and dispatch directly when the screenshot size equals
`screen_size`. Otherwise it must apply the scaling formula below.

---

## Why this matters

Holo3 (and any CUA model trained on raw screenshots) emits clicks against the
exact image bytes it received. If a host resizes a 1920×1080 framebuffer to
1280×720 before sending it to the model and then dispatches the model's
`(640, 360)` click against the original 1920×1080 framebuffer, the click
lands at *(640, 360)* — not at the center of the screen. The visible UI will
look fine; the click target will be wrong by 1.5×.

That is exactly the bug class that produced [#25](https://github.com/mercurialsolo/mantis/issues/25)
(closed) — `Holo3Brain` was passing through screenshot pixels but the env was
dispatching against a different-sized display.

---

## The scaling formula

Let:

- `(brain_w, brain_h)` = size of the screenshot the brain saw (`Image.size`).
- `(display_w, display_h)` = `GymEnvironment.screen_size`.

Then:

```python
x_display = round(x_brain * display_w / brain_w)
y_display = round(y_brain * display_h / brain_h)
```

If `brain_w == display_w` and `brain_h == display_h` (the common case), the
formula collapses to identity and no scaling is needed.

Worked examples:

| Display (Xvfb)   | Brain image      | Scale (W,H)     | Brain (640, 360) → display |
|------------------|------------------|-----------------|----------------------------|
| 1280×720         | 1280×720         | 1.000, 1.000    | (640, 360)                 |
| 1280×720         | 768×432 (resized)| 1.667, 1.667    | (1067, 600)                |
| 1920×1080        | 1280×720         | 1.500, 1.500    | (960, 540)                 |
| 1280×800         | 1280×720         | 1.000, 1.111    | (640, 400)                 |

> Asymmetric scale (last row) only happens when the resize doesn't preserve
> aspect ratio — usually a sign of a bug upstream. Keep the brain image's
> aspect ratio matched to the display unless you know what you're doing.

---

## What `XdotoolGymEnv` does today

[`XdotoolGymEnv`](https://github.com/mercurialsolo/mantis/blob/main/src/mantis_agent/gym/xdotool_env.py)
implements the common case directly:

- Xvfb is launched at `viewport=(W,H)`. Screenshots are `W×H`.
- `screen_size` returns `(W, H)`.
- `step()` dispatches `(x, y)` straight to xdotool — no scaling.
- Out-of-bounds coordinates are clamped to `[0, W-1] × [0, H-1]` via
  `_clamp` (defensive — the brain should not emit them, but we don't crash if
  it does).

If you point a brain at this env and the brain resizes screenshots before
inference, you must scale coordinates back inside the brain (or wrap the env
with an adapter that does). `XdotoolGymEnv` itself is a pure passthrough.

---

## What `VisionClaudeGymEnv` (StaffAI) needs to do

`VisionClaudeGymEnv` adapts the StaffAI Xvfb desktop to Mantis. The contract
to implement:

1. Set `screen_size` to whatever `Desktop.viewport_size` reports — the *real*
   Xvfb framebuffer size. **Do not** report the resized brain-image size here.
2. Inside `step()`, compute the scale from the brain image's `.size` (the one
   passed to inference) to `screen_size`, then apply it before dispatching
   to `ComputerTool.click(x, y)`.
3. Add a unit test mirroring `tests/test_gym_coordinates.py` in this repo,
   using `Desktop.viewport_size = (1280, 720)`. Feed an action with
   `x=640, y=360`, mock `Brain.last_image_size = (768, 432)`, and assert
   `ComputerTool.click` is invoked with `(1067, 600)` (within ±1 px for
   rounding).

A reusable helper is exported on `XdotoolGymEnv` so integrators don't have
to re-derive the math:

```python
from mantis_agent.gym.xdotool_env import scale_brain_to_display

x_disp, y_disp = scale_brain_to_display(
    x_brain=action.params["x"],
    y_brain=action.params["y"],
    brain_size=brain_image.size,   # (w, h)
    display_size=desktop.viewport_size,
)
computer_tool.click(x_disp, y_disp)
```

The function is small and pure — call it from any env adapter.

---

## DPR, retina, and other distractions

Xvfb has no concept of device pixel ratio. The framebuffer pixel space is
the only space. Don't multiply by 2 because macOS/Retina would; Xvfb is
not a Retina display. If you ever run Mantis against a real macOS
screenshot pipeline, that's the moment to introduce DPR-aware scaling — and
that adapter is responsible for handling it, not the brain.
