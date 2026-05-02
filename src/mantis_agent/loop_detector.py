"""Loop detection for CUA agent loops.

The previous detectors (in `agent.py` and `gym/runner.py`) only flagged
*byte-equal* repeats — same action_type with bit-identical params. Real
agent loops show up two other ways:

1. **Coordinate drift** — `click(487,312) → click(489,310) → click(491,308)`.
   Each is technically distinct but they're all clicks at the same UI target.
2. **State loops** — diverse actions but the page never changes. The brain
   tries 5 different things on a stuck modal; URL + screenshot don't move.

Conversely, deliberate `Page_Down` repetition for long-scroll is *not* a
loop when the visible content changes between presses.

This module gives both runners one helper with all three signals:
``is_repeat_loop``, ``is_drift_loop``, ``is_state_loop``. Each takes the
same ``window`` so callers can keep their existing soft/hard thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING

from .actions import Action, ActionType

if TYPE_CHECKING:
    from PIL import Image


# Action types where deliberate repetition is normal usage and should not
# count toward loop detection unless state also stops changing.
_SCROLL_LIKE_ACTIONS: frozenset[ActionType] = frozenset(
    {ActionType.SCROLL, ActionType.WAIT}
)
_SCROLL_LIKE_KEYS: frozenset[str] = frozenset(
    {"page_down", "pagedown", "page_up", "pageup", "down", "up", "j", "k"}
)


@dataclass
class _Sample:
    action: Action
    url: str = ""
    frame_hash: str = ""


@dataclass
class LoopDetector:
    """Stateful loop detector. Append one sample per executed step.

    Args:
        click_tol_px: Two click actions are considered "the same target" if
            both x and y differ by no more than this many pixels.
    """

    click_tol_px: int = 8
    _samples: list[_Sample] = field(default_factory=list)

    def reset(self) -> None:
        self._samples.clear()

    def record(
        self,
        action: Action,
        *,
        url: str = "",
        frame: "Image.Image | None" = None,
    ) -> None:
        """Append the most recently executed action and its post-state.

        Pass ``frame`` for state-loop detection. URL alone is enough on most
        web tasks; the frame hash catches in-page modal/loader cases where
        the URL never changes.
        """
        frame_hash = phash_64(frame) if frame is not None else ""
        self._samples.append(_Sample(action=action, url=url, frame_hash=frame_hash))

    # ── Public predicates ────────────────────────────────────────────────

    def is_repeat_loop(self, window: int) -> bool:
        """The last ``window`` actions are byte-equal (legacy behavior)."""
        recent = self._tail(window)
        if recent is None:
            return False
        if self._is_scroll_like(recent[0].action) and self._state_changed(recent):
            return False
        first = recent[0].action
        return all(
            s.action.action_type == first.action_type and s.action.params == first.params
            for s in recent[1:]
        )

    def is_drift_loop(self, window: int) -> bool:
        """The last ``window`` actions hit the same target with small coordinate drift.

        Applies to CLICK / DOUBLE_CLICK / DRAG. Other action types fall back to
        byte-equality (handled by :meth:`is_repeat_loop`).
        """
        recent = self._tail(window)
        if recent is None:
            return False
        first = recent[0].action
        if first.action_type not in (
            ActionType.CLICK,
            ActionType.DOUBLE_CLICK,
            ActionType.DRAG,
        ):
            return False
        first_xy = (first.params.get("x"), first.params.get("y"))
        if first_xy[0] is None or first_xy[1] is None:
            return False
        for s in recent[1:]:
            if s.action.action_type != first.action_type:
                return False
            x, y = s.action.params.get("x"), s.action.params.get("y")
            if x is None or y is None:
                return False
            if abs(x - first_xy[0]) > self.click_tol_px:
                return False
            if abs(y - first_xy[1]) > self.click_tol_px:
                return False
        return True

    def is_state_loop(self, window: int) -> bool:
        """Last ``window`` samples share the same observed state.

        Uses URL when available; uses frame hash when URL is shared (or empty).
        Returns False if any sample is missing both URL and frame hash — we
        can't conclude a state loop without observed state.
        """
        recent = self._tail(window)
        if recent is None:
            return False
        first = recent[0]
        # Need at least one signal to compare.
        if not first.url and not first.frame_hash:
            return False
        for s in recent[1:]:
            if first.url and s.url and s.url != first.url:
                return False
            if first.frame_hash and s.frame_hash and s.frame_hash != first.frame_hash:
                return False
            if not s.url and not s.frame_hash:
                return False
        return True

    def is_any_loop(self, window: int) -> bool:
        """Convenience: any of the three signals fires."""
        return (
            self.is_repeat_loop(window)
            or self.is_drift_loop(window)
            or self.is_state_loop(window)
        )

    # ── Internals ────────────────────────────────────────────────────────

    def _tail(self, window: int) -> list[_Sample] | None:
        if window < 2 or len(self._samples) < window:
            return None
        return self._samples[-window:]

    @staticmethod
    def _is_scroll_like(action: Action) -> bool:
        if action.action_type in _SCROLL_LIKE_ACTIONS:
            return True
        if action.action_type == ActionType.KEY_PRESS:
            keys = str(action.params.get("keys") or action.params.get("key") or "").lower()
            return any(k in keys for k in _SCROLL_LIKE_KEYS)
        return False

    @staticmethod
    def _state_changed(samples: list[_Sample]) -> bool:
        """True if URL or frame_hash differs across the window."""
        urls = {s.url for s in samples if s.url}
        hashes = {s.frame_hash for s in samples if s.frame_hash}
        return len(urls) > 1 or len(hashes) > 1


# ── Tiny zero-dep perceptual hash ────────────────────────────────────────


def phash_64(img: "Image.Image") -> str:
    """9x8 difference-hash → 17-char hex (16 hex of dHash + 1 of brightness bucket).

    dHash compares each pixel to its right neighbor, producing 64 bits that are
    stable to small drift but distinguish images with different content. The
    brightness suffix differentiates uniform images of different luminance
    (a black vs white screen would otherwise both hash to all-zeros).

    Not collision-safe — only used for state-loop detection where two visually
    identical pages should hash equal even if file bytes differ slightly.
    """
    try:
        from PIL import Image as _Image  # noqa: F401  (import gate)
    except Exception:
        return ""
    try:
        small = img.convert("L").resize((9, 8))
    except Exception:
        # Defensive: corrupted frame → empty hash means "no signal".
        return ""
    pixels = list(small.getdata())
    if len(pixels) != 72:
        return ""
    bits = 0
    bit_idx = 0
    for row in range(8):
        base = row * 9
        for col in range(8):
            if pixels[base + col] > pixels[base + col + 1]:
                bits |= 1 << bit_idx
            bit_idx += 1
    # Brightness suffix in [0, 15] so uniform-different images don't collide.
    avg = sum(pixels) // len(pixels)
    bucket = min(avg // 16, 15)
    return f"{bits:016x}{bucket:x}"


def encode_png_hash(img: "Image.Image") -> str:
    """Alternative bytewise PNG hash. Stable across processes (unlike id())."""
    import hashlib

    buf = BytesIO()
    img.save(buf, format="PNG")
    return hashlib.sha1(buf.getvalue()).hexdigest()[:16]
