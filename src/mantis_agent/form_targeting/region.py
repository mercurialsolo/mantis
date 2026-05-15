"""Region cropping for form-target grounding (#435 item 1).

Implements the *"planner localizes roughly, router crops the screenshot
to that region, executor outputs coordinates in the cropped frame
which get re-projected to screen space"* pattern documented in
``docs/cua_notes.md`` §6 "Patterns worth knowing".

The planner (plan author or the agentic_recovery's ``insert_steps``)
hints at a region via ``step.hints["region"]``. The form handler
calls :func:`crop_to_region` before passing the screenshot to
``find_form_target``, then re-projects the returned coordinates back
to full-screen space with :func:`reproject_coords`.

Two hint shapes are supported:

* **Explicit rectangle** —
  ``{"x": int, "y": int, "w": int, "h": int}``. Pixels at the
  screenshot's natural resolution. Coordinates outside the screen are
  clamped.
* **Named region** — a short string like ``"bottom"`` / ``"bottom-third"``
  / ``"top-half"``. Translated to a rectangle relative to the
  screenshot dimensions; lets plan authors avoid hard-coding pixels
  that drift with viewport changes.

If the hint is missing, malformed, or names a region the helper
doesn't recognise, :func:`crop_to_region` returns the original
screenshot with a ``(0, 0)`` offset — a no-op so existing callers
that don't set a region hint behave identically to the pre-#435
runtime.
"""

from __future__ import annotations

import logging
from typing import Any

from PIL import Image

logger = logging.getLogger("mantis_agent.form_targeting.region")


# Named regions → fractions of (left, top, right, bottom). Coordinates
# are 0..1 of the screenshot's width/height. Ordered roughly by
# expected utility on a typical web app layout.
_NAMED_REGIONS: dict[str, tuple[float, float, float, float]] = {
    "full":         (0.0, 0.0, 1.0, 1.0),
    "top":          (0.0, 0.0, 1.0, 1 / 3),
    "bottom":       (0.0, 2 / 3, 1.0, 1.0),
    "left":         (0.0, 0.0, 1 / 3, 1.0),
    "right":        (2 / 3, 0.0, 1.0, 1.0),
    "center":       (1 / 4, 1 / 4, 3 / 4, 3 / 4),
    "top-half":     (0.0, 0.0, 1.0, 0.5),
    "bottom-half":  (0.0, 0.5, 1.0, 1.0),
    "top-third":    (0.0, 0.0, 1.0, 1 / 3),
    "bottom-third": (0.0, 2 / 3, 1.0, 1.0),
    # Form-specific aliases — the most common case in practice. A
    # submit-button row lives at the bottom; the lead-detail header
    # lives at the top.
    "form-footer":  (0.0, 0.6, 1.0, 1.0),
    "form-header":  (0.0, 0.0, 1.0, 0.25),
}


def crop_to_region(
    screenshot: Image.Image,
    region: Any,
) -> tuple[Image.Image, tuple[int, int]]:
    """Crop ``screenshot`` to ``region``; return ``(cropped, (offset_x, offset_y))``.

    The offset is the top-left pixel of the crop in full-screen
    coordinates — callers re-project executor-emitted coordinates
    back to screen space via :func:`reproject_coords`.

    On any unknown / malformed region, returns the original
    screenshot with a ``(0, 0)`` offset. Logs a warning so the
    operator can spot a typo in a plan's ``hints.region``.
    """
    if not region:
        return screenshot, (0, 0)

    width, height = screenshot.size
    rect = _resolve_rect(region, width, height)
    if rect is None:
        logger.warning(
            "crop_to_region: unrecognised region %r — passing screenshot through unchanged",
            region,
        )
        return screenshot, (0, 0)

    left, top, right, bottom = rect
    # Clamp to screen bounds; protect against zero-area crops (which
    # would crash Pillow). A zero-area crop falls back to the full
    # screenshot too — same fail-open behaviour as an unknown region.
    left = max(0, min(width - 1, left))
    top = max(0, min(height - 1, top))
    right = max(left + 1, min(width, right))
    bottom = max(top + 1, min(height, bottom))

    cropped = screenshot.crop((left, top, right, bottom))
    logger.debug(
        "crop_to_region: %r → (%d,%d,%d,%d) [%dx%d → %dx%d]",
        region, left, top, right, bottom,
        width, height, cropped.width, cropped.height,
    )
    return cropped, (left, top)


def reproject_coords(
    coords: dict[str, Any] | None,
    offset: tuple[int, int],
) -> dict[str, Any] | None:
    """Add ``offset`` back to coordinate fields in a form-target result.

    ``coords`` is the dict returned by ``find_form_target`` — at
    minimum ``{"x": int, "y": int, ...}``. The form-target provider
    saw the cropped screenshot, so its ``x``/``y`` are in cropped
    space; this helper shifts them by the crop origin so the runner's
    click lands at the same pixel of the full screen.

    Returns ``None`` when ``coords`` is ``None`` so callers can chain
    ``reproject_coords(find_form_target(...), offset)``.
    """
    if coords is None:
        return None
    off_x, off_y = offset
    if off_x == 0 and off_y == 0:
        return coords
    out = dict(coords)
    if "x" in out and isinstance(out["x"], (int, float)):
        out["x"] = int(out["x"]) + off_x
    if "y" in out and isinstance(out["y"], (int, float)):
        out["y"] = int(out["y"]) + off_y
    return out


def _resolve_rect(
    region: Any, width: int, height: int,
) -> tuple[int, int, int, int] | None:
    """Translate ``region`` (string or dict) into a (left, top, right, bottom)
    pixel rectangle, or ``None`` if unrecognised.
    """
    if isinstance(region, str):
        frac = _NAMED_REGIONS.get(region.strip().lower())
        if frac is None:
            return None
        fleft, ftop, fright, fbottom = frac
        return (
            int(width * fleft), int(height * ftop),
            int(width * fright), int(height * fbottom),
        )
    if isinstance(region, dict):
        try:
            x = int(region["x"])
            y = int(region["y"])
            w = int(region["w"])
            h = int(region["h"])
        except (KeyError, TypeError, ValueError):
            return None
        if w <= 0 or h <= 0:
            return None
        return (x, y, x + w, y + h)
    return None


__all__ = ["crop_to_region", "reproject_coords"]
