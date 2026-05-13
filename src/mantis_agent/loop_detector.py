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

#298: ``adaptive_window`` / ``is_any_loop_adaptive`` adjust those
thresholds by recent action diversity + observed state progress, so
pagination doesn't trigger spurious soft nudges and clear traps fire
sooner. Enabled by default; ``MANTIS_LOOP_ADAPTIVE=disabled`` falls
back to fixed windows for ablation.

#296: ``compute_click_tol_px`` derives the drift tolerance from screen
diagonal so a fixed 8 px isn't too tight on 4K or too loose on a phone.
``LoopDetector._effective_click_tol`` applies a per-action class
multiplier (button 1.5×, link 0.5×, …) when the action's reasoning
text classifies the target. ``MANTIS_ADAPTIVE_CLICK_TOL=disabled``
forces the hardcoded ``click_tol_px`` for ablation.
"""

from __future__ import annotations

import math
import os
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

# #296: per-class drift-tolerance multipliers applied when the action's
# reasoning text classifies the target. Buttons cluster widely (large
# hit areas; brain often re-clicks the centroid), links cluster tightly
# (text targets — even small drift means a different word).
_CLASS_TOL_MULTIPLIERS: tuple[tuple[str, float], ...] = (
    # Order matters: longer / more specific keywords first so the
    # ambiguous "submit" doesn't match before "submit button".
    ("submit button", 1.5),
    ("button", 1.5),
    ("submit", 1.5),
    ("dropdown", 0.75),
    ("select", 0.75),
    ("menu item", 0.75),
    ("link", 0.5),
    ("anchor", 0.5),
    ("listing", 1.0),
    ("card", 1.0),
    ("input field", 1.0),
    ("form field", 1.0),
    ("text field", 1.0),
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

        #296: tolerance is :meth:`_effective_click_tol` of the first sample's
        reasoning — a button gets 1.5×, a link gets 0.5×.
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
        tol = self._effective_click_tol(first)
        for s in recent[1:]:
            if s.action.action_type != first.action_type:
                return False
            x, y = s.action.params.get("x"), s.action.params.get("y")
            if x is None or y is None:
                return False
            if abs(x - first_xy[0]) > tol:
                return False
            if abs(y - first_xy[1]) > tol:
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

    # ── #298: adaptive thresholds ──────────────────────────────────────

    def pattern_diversity(self, window: int) -> float:
        """Ratio of unique action signatures to window size, in ``[0, 1]``.

        ``1.0`` means every recent action was distinct; ``0.0`` is empty
        history. Click / double-click coordinates are bucketed by
        :attr:`click_tol_px` so micro-drift on the same UI target counts
        as one signature (matches :meth:`is_drift_loop` semantics).

        Returns ``0.0`` when the buffer is shorter than ``window`` so the
        signal is conservative on cold starts.
        """
        if window < 2 or len(self._samples) < window:
            return 0.0
        recent = self._samples[-window:]
        sigs = {self._action_signature(s.action) for s in recent}
        return len(sigs) / window

    def state_progressed(self, window: int) -> bool:
        """True when URL or frame hash moved across the last ``window`` samples.

        Public-surface mirror of the internal ``_state_changed`` guard
        already used by :meth:`is_repeat_loop`. Returns ``False`` when the
        buffer is too short to draw a conclusion.
        """
        if window < 2 or len(self._samples) < window:
            return False
        return self._state_changed(self._samples[-window:])

    def adaptive_window(
        self,
        base_window: int,
        *,
        max_extension: int = 2,
        floor: int = 2,
    ) -> int:
        """Adjust ``base_window`` by recent action diversity + state progress.

        The fixed default windows (soft=3, hard=8 in :class:`GymRunner`)
        are correct for the median case but produce two failure modes:

        - **Spurious nudges on pagination**: the brain clicks the "Next"
          button five times, the page advances each time. ``is_repeat_loop``
          fires (identical click params) even though the run is making
          progress. The fix is to *extend* the window when state is
          progressing or actions are varied — diversity alone isn't
          enough because pagination has zero diversity.
        - **Late termination on real traps**: a modal/captcha keeps the
          page locked while the brain emits five identical clicks. The
          state-loop signal already catches this at the hard window;
          the soft window can fire earlier (recovery / nudge) when
          diversity is low *and* state is frozen.

        The returned window is clamped to ``[floor, base + max_extension]``.
        """
        if base_window <= floor:
            return base_window
        # Need at least ``base_window`` samples to draw any conclusion;
        # short-buffer reads from ``pattern_diversity`` / ``state_progressed``
        # both return zero/False which would otherwise tip the tightening
        # branch. Keep the legacy fixed window until history catches up.
        if len(self._samples) < base_window:
            return base_window
        diversity = self.pattern_diversity(base_window)
        progressed = self.state_progressed(base_window)
        # Extend when work looks productive (varied actions OR moving page).
        if diversity >= 0.6 or progressed:
            return base_window + max_extension
        # Tighten when we have a clear stuck signature (no state, no variety).
        if diversity <= 0.25 and not progressed:
            shrink = max(1, max_extension // 2)
            return max(floor, base_window - shrink)
        return base_window

    def is_any_loop_adaptive(
        self,
        base_window: int,
        *,
        max_extension: int = 2,
    ) -> bool:
        """Adaptive variant of :meth:`is_any_loop` (#298).

        - Computes the effective window via :meth:`adaptive_window`.
        - Adds a pagination guard: if observed state moved across the
          window AND the run isn't a state-loop (frozen state with
          diverse actions, which the legacy :meth:`is_state_loop` catches),
          the run is making progress — don't fire even on identical
          repeated actions. This is the case the legacy ``is_repeat_loop``
          gets wrong on drilldown / "Next" pagination, where every
          ``click("Next")`` is byte-equal but the URL/frame moves.
        """
        effective = self.adaptive_window(
            base_window, max_extension=max_extension
        )
        if self.state_progressed(effective) and not self.is_state_loop(effective):
            return False
        return self.is_any_loop(effective)

    def _effective_click_tol(self, action: Action) -> int:
        """#296: per-action drift tolerance.

        Defaults to :attr:`click_tol_px`. Scaled by class multiplier when
        the action's reasoning text classifies the target (button → 1.5×,
        link → 0.5×, …) AND ``MANTIS_ADAPTIVE_CLICK_TOL`` is on (default).
        Falls back to the raw attribute when the toggle is off so the
        ablation harness can A/B without redeploys.
        """
        if not adaptive_click_tol_enabled():
            return self.click_tol_px
        reasoning = (action.reasoning or "").lower()
        for keyword, multiplier in _CLASS_TOL_MULTIPLIERS:
            if keyword in reasoning:
                return max(1, int(round(self.click_tol_px * multiplier)))
        return self.click_tol_px

    @staticmethod
    def _action_signature(action: Action) -> tuple:
        """Stable signature used by :meth:`pattern_diversity`.

        Click-like actions bucket coordinates so micro-drift collapses to
        one signature (same UI target). Other action types use their
        primary discriminating parameter.
        """
        atype = action.action_type
        params = action.params or {}
        if atype in (ActionType.CLICK, ActionType.DOUBLE_CLICK):
            x = params.get("x")
            y = params.get("y")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                # Bucket to ``click_tol_px``-ish granularity; 8 px matches
                # the default ``click_tol_px`` so signatures track drift.
                return (atype, int(x) // 8, int(y) // 8)
            return (atype,)
        if atype == ActionType.DRAG:
            return (
                atype,
                int((params.get("end_x") or 0)) // 8,
                int((params.get("end_y") or 0)) // 8,
            )
        if atype == ActionType.KEY_PRESS:
            keys = str(params.get("keys") or params.get("key") or "").lower()
            return (atype, keys)
        if atype == ActionType.TYPE:
            text = str(params.get("text") or params.get("content") or "")
            return (atype, text[:32])
        if atype == ActionType.SCROLL:
            return (atype, str(params.get("direction") or "down"))
        return (atype,)

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


def adaptive_loop_enabled() -> bool:
    """#298: gate for adaptive loop windows.

    Default-on. ``MANTIS_LOOP_ADAPTIVE=disabled`` (or ``0`` / ``false``)
    forces fixed-window behaviour so the ablation harness can run an
    A/B without redeploys. Read once per call site so the per-request
    env-var override in ``BasetenCUARuntime._run_pure_cua`` flips
    behaviour mid-container.
    """
    raw = os.environ.get("MANTIS_LOOP_ADAPTIVE", "enabled").strip().lower()
    return raw not in {"disabled", "0", "false", "off", "no"}


def adaptive_click_tol_enabled() -> bool:
    """#296: gate for screen-DPI / element-class drift tolerance.

    Default-on. ``MANTIS_ADAPTIVE_CLICK_TOL=disabled`` (or ``0`` /
    ``false``) forces the hardcoded ``LoopDetector.click_tol_px`` for
    A/B comparison without redeploys.
    """
    raw = os.environ.get("MANTIS_ADAPTIVE_CLICK_TOL", "enabled").strip().lower()
    return raw not in {"disabled", "0", "false", "off", "no"}


def compute_click_tol_px(viewport: tuple[int, int], *, floor: int = 8) -> int:
    """#296: drift tolerance baseline scaled by screen diagonal.

    The legacy ``click_tol_px = 8`` is too tight on 4K (legitimate
    micro-retries flagged as drift loops) and approximately right on
    1080p. Scale ``0.4 %`` of the diagonal so 4K gets ~18 px and 8K
    gets ~37 px, with a ``floor`` so phone-class viewports keep the
    legacy value:

    | Viewport            | Diagonal | Tolerance |
    |---------------------|----------|-----------|
    | 1280 × 800 (default)| 1430 px  | 8 px (floor) |
    | 1366 × 768 (laptop) | 1567 px  | 8 px (floor) |
    | 1920 × 1080 (FHD)   | 2202 px  | 9 px |
    | 2560 × 1440 (QHD)   | 2937 px  | 12 px |
    | 3840 × 2160 (4K)    | 4404 px  | 18 px |

    The harness toggles tolerance via ``MANTIS_ADAPTIVE_CLICK_TOL``;
    when off, callers should pass the legacy ``8`` constructor default.
    """
    if not adaptive_click_tol_enabled():
        return floor
    w, h = viewport
    if w <= 0 or h <= 0:
        return floor
    diag = math.sqrt(w * w + h * h)
    return max(floor, int(round(0.004 * diag)))
