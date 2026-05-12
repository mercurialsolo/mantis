"""Perceptual-diff verifier for high-risk actions (#293).

The runner used to declare an action successful purely on syscall
success: ``executor.execute()`` returned without raising → action
counted. That misses an entire class of silent failures:

* Overlays / consent banners absorb the click.
* Validation messages flash and disappear before the next screenshot.
* Modals are mounted into hidden DOM positions.
* Page repaints onto an identical viewport (drift loop).

This module compares the pre-action frame to the post-settle frame and
emits ``action_effect_observed: bool`` per high-risk step. The runner
injects a ``WARNING: action had no observed effect`` line into the
next inference's feedback when the predicate fires, so the brain gets
a real signal instead of looping on the same useless click.

Signals:

* **Global frame hash** — `phash_64` of the full screenshot. Cheap;
  detects any meaningful repaint.
* **Region hash** — `phash_64` of a 200×200 crop around the action's
  ``(x, y)``. Catches the case where the global hash changes (a banner
  ticked over, a clock updated) but the action region itself didn't —
  the click landed on nothing.

Both signals are short-circuited via ``MANTIS_PERCEPTUAL_VERIFY=disabled``
for the #261 ablation discipline.

CLIP cosine is intentionally NOT shipped here. The issue mentions it as
a future addition; we land the pHash path first since it's free
(already computed for loop detection) and avoids a model dep. A future
PR can add CLIP behind a separate toggle once the pHash baseline is
measured.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..actions import Action, ActionType
from ..loop_detector import phash_64

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)


_ENV_TOGGLE: str = "MANTIS_PERCEPTUAL_VERIFY"


def is_enabled() -> bool:
    """``MANTIS_PERCEPTUAL_VERIFY=disabled`` short-circuits the helper."""
    return os.environ.get(_ENV_TOGGLE, "enabled").lower() != "disabled"


# ── High-risk classifier ───────────────────────────────────────────────


# Keywords in an action's ``reasoning`` text that classify a click as
# high-risk (submit/buy/send/etc.). The brain often emits these verbatim
# in its CoT. Conservative list — false negatives (a high-risk click
# missed) just mean the verifier skips that step. False positives just
# add a hash comparison; no behaviour change.
_HIGH_RISK_REASONING_KEYWORDS: tuple[str, ...] = (
    "submit",
    "confirm",
    "buy",
    "purchase",
    "send",
    "delete",
    "save",
    "sign in",
    "log in",
    "login",
    "register",
    "checkout",
    "place order",
)


def is_high_risk(
    action: Action,
    *,
    thinking: str | None = None,
    task: str | None = None,
) -> bool:
    """Classify whether ``action`` warrants perceptual verification.

    Heuristic — kept narrow on purpose so the verifier doesn't fire on
    every click:

    1. ``KEY_PRESS`` with ``Return`` / ``Enter`` (incl. chord forms like
       ``ctrl+Return``) — typical form submit.
    2. ``CLICK`` whose ``Action.reasoning``, the brain's free-form
       ``thinking``, or the run-level ``task`` text contains any keyword
       in :data:`_HIGH_RISK_REASONING_KEYWORDS`.

    Returns ``False`` for ``WAIT`` / ``DONE`` / ``SCROLL`` / ``TYPE``
    actions — none of those need post-action effect verification (TYPE
    has its own ``type_verified`` flag from the env adapter).

    Brains differ in where they leave the high-risk signal: Claude
    populates ``Action.reasoning``; Holo3 emits a bare ``click()`` with
    the reasoning in its separate ``thinking`` channel; some flows
    encode the intent only in the task prompt. Check all three.
    """
    if action.action_type == ActionType.KEY_PRESS:
        keys = str(action.params.get("keys", "")).lower()
        return keys in {"return", "enter"} or keys.endswith("+return") or keys.endswith("+enter")

    if action.action_type == ActionType.CLICK:
        haystacks = [
            (action.reasoning or "").lower(),
            (thinking or "").lower(),
            (task or "").lower(),
        ]
        return any(
            kw in haystack
            for haystack in haystacks
            for kw in _HIGH_RISK_REASONING_KEYWORDS
        )

    return False


# ── Frame-region diff ──────────────────────────────────────────────────


@dataclass
class EffectCheck:
    """Outcome of a perceptual-diff comparison.

    * ``effect_observed`` — ``True`` when the global hash or the region
      hash changed between pre and post. ``False`` when both stayed
      identical (silent failure). ``None`` when the check was skipped
      (toggle disabled / no pre-frame / not high-risk).
    * ``global_changed`` / ``region_changed`` — individual signals, kept
      for telemetry. Either is enough to count as effect_observed.
    """

    effect_observed: bool | None
    global_changed: bool = False
    region_changed: bool = False
    reason: str = ""


def _crop_around(
    img: "Image.Image",
    x: int,
    y: int,
    *,
    size: int = 200,
) -> "Image.Image":
    """Crop a ``size × size`` square centred on ``(x, y)``, clamped to
    the image bounds. Returns the cropped sub-image.
    """
    w, h = img.size
    half = size // 2
    left = max(0, min(x - half, w - size))
    top = max(0, min(y - half, h - size))
    right = min(w, left + size)
    bottom = min(h, top + size)
    return img.crop((left, top, right, bottom))


def region_hash(
    img: "Image.Image",
    x: int,
    y: int,
    *,
    size: int = 200,
) -> str:
    """``phash_64`` of the action-region crop. Empty string on failure
    so the caller treats it as "no signal" without raising."""
    try:
        crop = _crop_around(img, int(x), int(y), size=size)
        return phash_64(crop)
    except Exception as exc:
        logger.debug("perceptual-diff: region_hash failed: %s", exc)
        return ""


def action_had_effect(
    pre_frame: "Image.Image | None",
    post_frame: "Image.Image | None",
    action: Action,
    *,
    thinking: str | None = None,
    task: str | None = None,
) -> EffectCheck:
    """Decide whether ``action`` produced an observable effect.

    Returns ``EffectCheck(effect_observed=None, ...)`` when:

    * The ablation toggle is off.
    * Either frame is missing (first step / capture failure).
    * The action isn't a high-risk class.

    Returns ``effect_observed=True`` when EITHER the global frame hash
    changed OR the 200×200 region around the click coords changed.
    ``False`` only when BOTH stayed pixel-equivalent — i.e. the action
    visibly did nothing.

    Action-region crop is centred on ``params["x"], params["y"]`` when
    present (CLICK / DOUBLE_CLICK / SCROLL). For ``KEY_PRESS`` and
    other coord-less actions, only the global hash is consulted.
    """
    if not is_enabled():
        return EffectCheck(effect_observed=None, reason="toggle disabled")
    if pre_frame is None or post_frame is None:
        return EffectCheck(effect_observed=None, reason="missing frame")
    if not is_high_risk(action, thinking=thinking, task=task):
        return EffectCheck(effect_observed=None, reason="not high-risk")

    try:
        pre_global = phash_64(pre_frame)
        post_global = phash_64(post_frame)
    except Exception as exc:
        logger.debug("perceptual-diff: global hash failed: %s", exc)
        return EffectCheck(effect_observed=None, reason=f"hash failed: {exc}")

    global_changed = pre_global != post_global

    region_changed = False
    if "x" in action.params and "y" in action.params:
        try:
            x = int(action.params["x"])
            y = int(action.params["y"])
        except (TypeError, ValueError):
            x = y = -1
        if x >= 0 and y >= 0:
            pre_region = region_hash(pre_frame, x, y)
            post_region = region_hash(post_frame, x, y)
            if pre_region and post_region:
                region_changed = pre_region != post_region

    effect = bool(global_changed or region_changed)
    reason = (
        "global_and_region_stable" if not effect
        else "global_changed" if global_changed and not region_changed
        else "region_changed" if region_changed and not global_changed
        else "global_and_region_changed"
    )
    return EffectCheck(
        effect_observed=effect,
        global_changed=global_changed,
        region_changed=region_changed,
        reason=reason,
    )
