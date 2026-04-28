"""Polished recording: title card + step captions + result outro.

Turns the raw ``recording.mp4`` (Xvfb screencast) into something that
looks like a product feature walkthrough:

  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
  │ Title card  │→ │ Run footage │→ │ Outro card  │
  │ 3s          │  │ + captions  │  │ 5s          │
  └─────────────┘  └─────────────┘  └─────────────┘

* **Title card**: tenant + plan name + run id, rendered once via PIL.
* **Captions**: SRT timed against per-step elapsed seconds, burned in
  with ffmpeg's ``subtitles`` filter (libass).
* **Outro card**: viable leads, leads_with_phone, total cost, duration.

The raw recording is kept; the polished one is saved alongside as
``recording_polished.<fmt>`` and is what the download endpoint serves
when present.
"""

from __future__ import annotations

import dataclasses
import logging
import shutil
import subprocess
from datetime import timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont

from .actions import Action, ActionType
from .gym.base import GymEnvironment, GymObservation, GymResult

logger = logging.getLogger("mantis_agent.presentation")


# ── Action event capture ────────────────────────────────────────────────────
# Each agent action that has a useful visual maps to one of these events.
# ``t_seconds`` is the elapsed time since the recording started, NOT
# wall-clock — so it lines up with the raw video timeline regardless of
# how long the agent took to spin up.

@dataclasses.dataclass
class ClickEvent:
    """A CLICK / DOUBLE_CLICK / RIGHT_CLICK observation."""

    t_seconds: float
    x: int
    y: int
    button: str = "left"
    double: bool = False


@dataclasses.dataclass
class KeyPressEvent:
    """A KEY_PRESS observation — keyboard chord like Ctrl+S, Enter, alt+Left."""

    t_seconds: float
    keys: str


@dataclasses.dataclass
class TypeEvent:
    """A TYPE observation — text the agent typed."""

    t_seconds: float
    text: str


@dataclasses.dataclass
class ScrollEvent:
    """A SCROLL observation."""

    t_seconds: float
    direction: str = "down"   # "up" | "down" | "left" | "right"
    amount: int = 5


@dataclasses.dataclass
class DragEvent:
    """A DRAG observation — straight-line drag from one point to another."""

    t_seconds: float
    x1: int
    y1: int
    x2: int
    y2: int


class ActionEventLog:
    """Captures every visual-overlay-relevant action made during a run.

    Thread-safe (single ``threading.Lock``). The runtime hands it to
    :class:`ActionRecordingEnv`, which intercepts CLICK / KEY_PRESS / TYPE /
    SCROLL / DRAG actions and routes them into the matching list. After
    the run ends, the log is passed to :func:`render_action_overlay_pngs`.
    """

    def __init__(self, anchor_time: float | None = None) -> None:
        import threading
        import time as _time
        self._anchor = anchor_time if anchor_time is not None else _time.time()
        self._lock = threading.Lock()
        self._clicks: list[ClickEvent] = []
        self._keys: list[KeyPressEvent] = []
        self._types: list[TypeEvent] = []
        self._scrolls: list[ScrollEvent] = []
        self._drags: list[DragEvent] = []

    def _now(self) -> float:
        import time as _time
        return max(0.0, _time.time() - self._anchor)

    def record_click(self, x: int, y: int, button: str = "left", double: bool = False) -> None:
        with self._lock:
            self._clicks.append(
                ClickEvent(t_seconds=self._now(), x=int(x), y=int(y), button=button, double=double)
            )

    def record_key(self, keys: str) -> None:
        if not keys:
            return
        with self._lock:
            self._keys.append(KeyPressEvent(t_seconds=self._now(), keys=str(keys)))

    def record_type(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            self._types.append(TypeEvent(t_seconds=self._now(), text=str(text)))

    def record_scroll(self, direction: str = "down", amount: int = 5) -> None:
        with self._lock:
            self._scrolls.append(
                ScrollEvent(t_seconds=self._now(), direction=str(direction), amount=int(amount))
            )

    def record_drag(self, x1: int, y1: int, x2: int, y2: int) -> None:
        with self._lock:
            self._drags.append(
                DragEvent(t_seconds=self._now(), x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2))
            )

    @property
    def clicks(self) -> list[ClickEvent]:
        with self._lock:
            return list(self._clicks)

    @property
    def keys(self) -> list[KeyPressEvent]:
        with self._lock:
            return list(self._keys)

    @property
    def types(self) -> list[TypeEvent]:
        with self._lock:
            return list(self._types)

    @property
    def scrolls(self) -> list[ScrollEvent]:
        with self._lock:
            return list(self._scrolls)

    @property
    def drags(self) -> list[DragEvent]:
        with self._lock:
            return list(self._drags)

    @property
    def total(self) -> int:
        with self._lock:
            return (
                len(self._clicks) + len(self._keys) + len(self._types)
                + len(self._scrolls) + len(self._drags)
            )

    def __len__(self) -> int:
        return self.total

    # Backwards-compatibility shim for code/tests that still call .events / .record(x,y).
    @property
    def events(self) -> list[ClickEvent]:
        return self.clicks

    def record(self, x: int, y: int, button: str = "left") -> None:
        self.record_click(x, y, button=button)


# Backwards-compat alias for the old name.
ClickEventLog = ActionEventLog


class ActionRecordingEnv(GymEnvironment):
    """Pass-through wrapper that logs every visually-relevant action.

    Universal for any computer-use scenario — browser, file manager,
    terminal, dialogs, desktop apps. The env contract talks pixels and
    keys; overlays render at pixel coordinates / on-screen badges
    regardless of what application is in focus.
    """

    def __init__(self, inner: GymEnvironment, log: ActionEventLog) -> None:
        self._inner = inner
        self._log = log

    @property
    def screen_size(self) -> tuple[int, int]:
        return self._inner.screen_size

    def reset(self, task: str, **kwargs: Any) -> GymObservation:
        return self._inner.reset(task, **kwargs)

    def step(self, action: Action) -> GymResult:
        try:
            self._capture(action)
        except Exception:  # noqa: BLE001 — never break the run because of logging
            pass
        return self._inner.step(action)

    def _capture(self, action: Action) -> None:
        params = action.params or {}
        if action.action_type == ActionType.CLICK:
            self._log.record_click(
                params.get("x", 0), params.get("y", 0),
                button=str(params.get("button", "left")),
            )
        elif action.action_type == ActionType.DOUBLE_CLICK:
            self._log.record_click(
                params.get("x", 0), params.get("y", 0),
                button=str(params.get("button", "left")),
                double=True,
            )
        elif action.action_type == ActionType.KEY_PRESS:
            self._log.record_key(str(params.get("keys", "")))
        elif action.action_type == ActionType.TYPE:
            self._log.record_type(str(params.get("text", "")))
        elif action.action_type == ActionType.SCROLL:
            self._log.record_scroll(
                direction=str(params.get("direction", "down")),
                amount=int(params.get("amount", 5)),
            )
        elif action.action_type == ActionType.DRAG:
            self._log.record_drag(
                params.get("x1", 0), params.get("y1", 0),
                params.get("x2", 0), params.get("y2", 0),
            )

    def close(self) -> None:
        self._inner.close()

    # Forward attribute access so callers reading env.current_url etc. still work.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


# Backwards-compat alias.
ClickRecordingEnv = ActionRecordingEnv


# ── Cards (PIL-rendered, no external font deps) ─────────────────────────────
@dataclasses.dataclass
class CardConfig:
    """Description of one card (title or outro).

    ``body_lines`` are rendered below ``subtitle`` with smaller spacing.
    Colors default to a dark slate background with emerald accent — easy
    on the eye, recognizable as "ours."
    """

    title: str
    subtitle: str = ""
    body_lines: list[str] = dataclasses.field(default_factory=list)
    footer: str = ""
    bg_color: tuple[int, int, int] = (15, 23, 42)        # slate-900
    title_color: tuple[int, int, int] = (248, 250, 252)  # slate-50
    accent_color: tuple[int, int, int] = (16, 185, 129)  # emerald-500
    body_color: tuple[int, int, int] = (203, 213, 225)   # slate-300
    footer_color: tuple[int, int, int] = (100, 116, 139)  # slate-500


_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load a system font in approximate priority order; fall back to default."""
    for path in _FONT_CANDIDATES:
        if not Path(path).exists():
            continue
        if bold and "bold" not in path.lower() and "Bold" not in path:
            continue
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_card(width: int, height: int, config: CardConfig) -> bytes:
    """Render one card as a PNG byte string.

    Layout (centered):
        ┌────────────────────────────────────┐
        │                                    │
        │           TITLE (big)              │
        │           ───── (accent rule)      │
        │           subtitle                 │
        │                                    │
        │           body line 1              │
        │           body line 2              │
        │                                    │
        │                          footer    │
        └────────────────────────────────────┘
    """
    img = Image.new("RGB", (width, height), color=config.bg_color)
    draw = ImageDraw.Draw(img)

    title_size = max(36, height // 12)
    subtitle_size = max(20, height // 24)
    body_size = max(18, height // 28)
    footer_size = max(14, height // 36)

    f_title = _load_font(title_size, bold=True)
    f_subtitle = _load_font(subtitle_size)
    f_body = _load_font(body_size)
    f_footer = _load_font(footer_size)

    def _measure(text: str, font) -> tuple[int, int]:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Stack everything in a centered column
    cy = height // 5  # start ~20% from the top

    # Title
    tw, th = _measure(config.title, f_title)
    draw.text(((width - tw) // 2, cy), config.title, fill=config.title_color, font=f_title)
    cy += th + max(8, height // 64)

    # Accent rule (centered, ~width/4 wide)
    rule_w = width // 4
    rule_h = max(3, height // 240)
    rule_x = (width - rule_w) // 2
    draw.rectangle(
        (rule_x, cy, rule_x + rule_w, cy + rule_h),
        fill=config.accent_color,
    )
    cy += rule_h + max(16, height // 32)

    # Subtitle
    if config.subtitle:
        sw, sh = _measure(config.subtitle, f_subtitle)
        draw.text(
            ((width - sw) // 2, cy), config.subtitle,
            fill=config.body_color, font=f_subtitle,
        )
        cy += sh + max(8, height // 48)

    # Body lines
    for line in config.body_lines:
        bw, bh = _measure(line, f_body)
        draw.text(
            ((width - bw) // 2, cy), line,
            fill=config.body_color, font=f_body,
        )
        cy += bh + max(6, height // 96)

    # Footer (bottom-right)
    if config.footer:
        fw, fh = _measure(config.footer, f_footer)
        draw.text(
            (width - fw - max(20, width // 64), height - fh - max(20, height // 36)),
            config.footer, fill=config.footer_color, font=f_footer,
        )

    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def write_card(path: Path, width: int, height: int, config: CardConfig) -> Path:
    """Render a card and write to disk. Returns the same path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(render_card(width, height, config))
    return path


# ── Step captions (SRT) ────────────────────────────────────────────────────
def _format_srt_ts(t: float) -> str:
    """Convert seconds float → SRT timestamp ``HH:MM:SS,mmm``."""
    if t < 0:
        t = 0
    delta = timedelta(seconds=t)
    total_seconds = int(delta.total_seconds())
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    ms = int(round((delta.total_seconds() - total_seconds) * 1000))
    if ms >= 1000:
        s += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


@dataclasses.dataclass
class StepCaption:
    """One subtitle cue spanning ``[start_t, end_t]``."""

    start_t: float
    end_t: float
    text: str


def captions_from_step_timings(
    step_timings: list[tuple[float, str, str]],
    *,
    title_offset: float = 0.0,
) -> list[StepCaption]:
    """Build SRT cues from ``[(elapsed_seconds, intent, status), ...]``.

    Each caption runs from one step's elapsed time to the next; the last
    caption stretches until ``last_t + 3s`` so the final step is visible.
    Captions are prefixed with the status ("✓ ") for completed steps and
    ("✗ ") for failures so viewers can read pass/fail at a glance.
    """
    if not step_timings:
        return []
    captions: list[StepCaption] = []
    sorted_timings = sorted(step_timings, key=lambda x: x[0])
    for i, (t, intent, status) in enumerate(sorted_timings):
        start = t + title_offset
        if i + 1 < len(sorted_timings):
            end = sorted_timings[i + 1][0] + title_offset
        else:
            end = start + 3.0
        prefix = ""
        if status in {"completed", "ok", "succeeded"}:
            prefix = "[OK] "
        elif status in {"failed", "error"}:
            prefix = "[FAIL] "
        text = (prefix + intent).strip()
        # Wrap long lines for readability (40 chars per line max)
        text = _wrap_for_srt(text, max_chars=44)
        captions.append(StepCaption(start_t=start, end_t=end, text=text))
    return captions


def _wrap_for_srt(text: str, max_chars: int = 44) -> str:
    """Wrap into max-2 lines, breaking at word boundaries."""
    words = text.split()
    if not words:
        return ""
    lines: list[str] = []
    current = words[0]
    for w in words[1:]:
        if len(current) + 1 + len(w) <= max_chars:
            current += " " + w
        else:
            lines.append(current)
            current = w
            if len(lines) == 2:
                # Truncate any further words with an ellipsis.
                lines[-1] = lines[-1] + "…"
                return "\n".join(lines)
    lines.append(current)
    return "\n".join(lines[:2])


def captions_to_srt(captions: list[StepCaption]) -> str:
    """Serialize captions into the SRT text format ffmpeg expects."""
    out: list[str] = []
    for i, c in enumerate(captions, start=1):
        out.append(str(i))
        out.append(f"{_format_srt_ts(c.start_t)} --> {_format_srt_ts(c.end_t)}")
        out.append(c.text)
        out.append("")
    return "\n".join(out)


# ── Action overlay (PNG sequence) ──────────────────────────────────────────
# Visual feedback for every kind of action the agent can take. All overlays
# share the same PNG canvas so ffmpeg only needs one overlay filter.
RIPPLE_DURATION_SECONDS = 0.6
RIPPLE_RING_COUNT = 2
RIPPLE_MAX_RADIUS = 90
RIPPLE_MIN_RADIUS = 18
RIPPLE_COLOR = (56, 189, 248)        # sky-400
RIPPLE_INNER_COLOR = (255, 255, 255)
RIPPLE_PEAK_ALPHA = 220

KEY_BADGE_DURATION_SECONDS = 1.5
KEY_BADGE_BG = (15, 23, 42)          # slate-900
KEY_BADGE_FG = (248, 250, 252)       # slate-50
KEY_BADGE_ACCENT = (56, 189, 248)    # sky-400

SCROLL_DURATION_SECONDS = 0.8
SCROLL_ARROW_COLOR = (56, 189, 248)

TYPE_DURATION_SECONDS = 1.8
TYPE_CAPTION_BG = (15, 23, 42)
TYPE_CAPTION_FG = (248, 250, 252)

DRAG_TRAIL_COLOR = (56, 189, 248)
DRAG_TRAIL_DURATION_SECONDS = 0.9


def _draw_click_ripple(
    draw: ImageDraw.ImageDraw, x: int, y: int, progress: float, *, double: bool = False,
) -> None:
    """Concentric expanding rings + a crisp center dot. Used for both single
    and double clicks (double clicks get a second ring offset by 0.1 progress)."""
    ring_offsets = (0.0, 0.1) if double else (0.0,)
    for ring_offset in ring_offsets:
        for ring in range(RIPPLE_RING_COUNT):
            ring_progress = max(0.0, progress - 0.15 * ring - ring_offset)
            if ring_progress <= 0 or ring_progress > 1:
                continue
            radius = int(
                RIPPLE_MIN_RADIUS
                + (RIPPLE_MAX_RADIUS - RIPPLE_MIN_RADIUS) * ring_progress
            )
            alpha = int(RIPPLE_PEAK_ALPHA * (1.0 - ring_progress))
            if alpha <= 0:
                continue
            line_w = max(2, 6 - int(4 * ring_progress))
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                outline=(*RIPPLE_COLOR, alpha),
                width=line_w,
            )
    inner_r = max(2, int(8 * (1.0 - progress)))
    inner_alpha = int(255 * (1.0 - progress))
    if inner_alpha > 0 and inner_r > 0:
        draw.ellipse(
            [x - inner_r, y - inner_r, x + inner_r, y + inner_r],
            fill=(*RIPPLE_INNER_COLOR, inner_alpha),
        )


def _draw_key_badge(
    draw: ImageDraw.ImageDraw,
    keys: str,
    progress: float,
    canvas_w: int,
    canvas_h: int,
    slot: int = 0,
) -> None:
    """Bottom-right rounded badge showing the keyboard chord.

    ``progress`` is the animation phase (0 = just pressed, 1 = about to vanish).
    ``slot`` stacks multiple concurrent key badges vertically.
    """
    if progress >= 1.0:
        return
    label = keys.replace("+", " + ").upper()
    font = _load_font(max(20, canvas_h // 28), bold=True)
    pad_x = max(16, canvas_w // 80)
    pad_y = max(10, canvas_h // 72)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    badge_w = tw + pad_x * 2
    badge_h = th + pad_y * 2
    margin = max(20, canvas_h // 36)

    # Slide-in from the right (first ~30% of progress) then hold.
    slide_phase = min(progress / 0.2, 1.0) if progress < 0.2 else 1.0
    fade = 1.0 if progress < 0.7 else max(0.0, 1.0 - (progress - 0.7) / 0.3)

    x_target = canvas_w - badge_w - margin
    x = int(x_target + (1.0 - slide_phase) * badge_w * 1.5)
    y = canvas_h - badge_h - margin - slot * (badge_h + 8)

    bg_alpha = int(230 * fade)
    fg_alpha = int(255 * fade)
    accent_alpha = int(220 * fade)
    if bg_alpha <= 0:
        return

    radius = badge_h // 2
    # Rounded rectangle background
    draw.rounded_rectangle(
        [x, y, x + badge_w, y + badge_h],
        radius=radius,
        fill=(*KEY_BADGE_BG, bg_alpha),
        outline=(*KEY_BADGE_ACCENT, accent_alpha),
        width=2,
    )
    draw.text(
        (x + pad_x, y + pad_y - bbox[1]),
        label,
        font=font,
        fill=(*KEY_BADGE_FG, fg_alpha),
    )


def _draw_scroll_arrow(
    draw: ImageDraw.ImageDraw,
    direction: str,
    progress: float,
    canvas_w: int,
    canvas_h: int,
) -> None:
    """Edge-mounted directional arrow that slides + fades."""
    if progress >= 1.0 or progress < 0:
        return
    fade = 1.0 if progress < 0.6 else max(0.0, 1.0 - (progress - 0.6) / 0.4)
    alpha = int(200 * fade)
    if alpha <= 0:
        return

    arrow_size = max(60, canvas_h // 12)
    travel = arrow_size  # how far it slides during the animation
    color = (*SCROLL_ARROW_COLOR, alpha)

    # Direction → (anchor edge midpoint, slide offset, polygon points around 0,0)
    if direction == "down":
        cx = canvas_w - arrow_size - max(20, canvas_w // 64)
        cy = canvas_h // 2 - arrow_size // 2 + int(travel * progress)
        pts = [(0, 0), (arrow_size, 0), (arrow_size // 2, arrow_size)]
    elif direction == "up":
        cx = canvas_w - arrow_size - max(20, canvas_w // 64)
        cy = canvas_h // 2 - arrow_size // 2 - int(travel * progress)
        pts = [(0, arrow_size), (arrow_size, arrow_size), (arrow_size // 2, 0)]
    elif direction == "right":
        cx = canvas_w // 2 - arrow_size // 2 + int(travel * progress)
        cy = canvas_h - arrow_size - max(20, canvas_h // 36)
        pts = [(0, 0), (arrow_size, arrow_size // 2), (0, arrow_size)]
    elif direction == "left":
        cx = canvas_w // 2 - arrow_size // 2 - int(travel * progress)
        cy = canvas_h - arrow_size - max(20, canvas_h // 36)
        pts = [(arrow_size, 0), (0, arrow_size // 2), (arrow_size, arrow_size)]
    else:
        return

    polygon = [(cx + px, cy + py) for px, py in pts]
    draw.polygon(polygon, fill=color)


def _draw_type_caption(
    draw: ImageDraw.ImageDraw,
    text: str,
    progress: float,
    canvas_w: int,
    canvas_h: int,
) -> None:
    """Top-center caption showing what the agent is typing."""
    if progress >= 1.0 or not text:
        return
    fade = 1.0 if progress < 0.7 else max(0.0, 1.0 - (progress - 0.7) / 0.3)
    alpha = int(230 * fade)
    if alpha <= 0:
        return

    label = f"⌨  Typing: \"{text}\""
    if len(label) > 60:
        label = label[:57] + "…\""

    font = _load_font(max(20, canvas_h // 30), bold=False)
    pad_x = max(20, canvas_w // 64)
    pad_y = max(10, canvas_h // 64)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    box_w = tw + pad_x * 2
    box_h = th + pad_y * 2
    margin_top = max(40, canvas_h // 16)
    x = (canvas_w - box_w) // 2
    y = margin_top
    radius = box_h // 2

    draw.rounded_rectangle(
        [x, y, x + box_w, y + box_h],
        radius=radius,
        fill=(*TYPE_CAPTION_BG, alpha),
        outline=(*KEY_BADGE_ACCENT, int(180 * fade)),
        width=2,
    )
    draw.text(
        (x + pad_x, y + pad_y - bbox[1]),
        label,
        font=font,
        fill=(*TYPE_CAPTION_FG, int(255 * fade)),
    )


def _draw_drag_trail(
    draw: ImageDraw.ImageDraw,
    x1: int, y1: int, x2: int, y2: int,
    progress: float,
) -> None:
    """Animated line from (x1,y1) → (x2,y2) with a fading head dot."""
    if progress >= 1.0 or progress < 0:
        return
    head_x = int(x1 + (x2 - x1) * progress)
    head_y = int(y1 + (y2 - y1) * progress)
    fade = 1.0 if progress < 0.7 else max(0.0, 1.0 - (progress - 0.7) / 0.3)
    line_alpha = int(180 * fade)
    if line_alpha <= 0:
        return
    draw.line(
        [(x1, y1), (head_x, head_y)],
        fill=(*DRAG_TRAIL_COLOR, line_alpha),
        width=4,
    )
    head_r = max(4, 10)
    draw.ellipse(
        [head_x - head_r, head_y - head_r, head_x + head_r, head_y + head_r],
        fill=(*DRAG_TRAIL_COLOR, int(220 * fade)),
    )


def render_action_overlay_pngs(
    out_dir: Path,
    *,
    duration_seconds: float,
    fps: int,
    width: int,
    height: int,
    clicks: list[ClickEvent] | None = None,
    keys: list[KeyPressEvent] | None = None,
    types: list[TypeEvent] | None = None,
    scrolls: list[ScrollEvent] | None = None,
    drags: list[DragEvent] | None = None,
    title_offset_seconds: float = 0.0,
) -> Path | None:
    """Render the per-frame overlay PNG sequence for ffmpeg's image2 demuxer.

    All overlay types (click ripples + keyboard chord badges + scroll
    arrows + type captions + drag trails) composite onto the same
    transparent canvas so ffmpeg only needs one ``overlay`` filter.

    Most frames are blank (no active overlay); they're hardlinked to a
    single ``_blank.png`` to keep disk usage tight. Only frames with
    active overlays incur PIL render cost.

    Returns the directory containing ``frame_%06d.png``, or None when
    there are no overlay events at all (caller should skip overlay).
    """
    clicks = clicks or []
    keys = keys or []
    types = types or []
    scrolls = scrolls or []
    drags = drags or []
    if not (clicks or keys or types or scrolls or drags):
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    n_frames = max(1, int(round(duration_seconds * fps)))

    # Pre-render the empty frame once and reuse for blank slots.
    blank_path = out_dir / "_blank.png"
    Image.new("RGBA", (width, height), (0, 0, 0, 0)).save(blank_path)

    # Pre-compute event activation windows (already adjusted for the
    # title-offset).
    def _shift(t: float) -> float:
        return t + title_offset_seconds

    click_w = [(c, _shift(c.t_seconds), RIPPLE_DURATION_SECONDS) for c in clicks]
    key_w = [(k, _shift(k.t_seconds), KEY_BADGE_DURATION_SECONDS) for k in keys]
    type_w = [(ty, _shift(ty.t_seconds), TYPE_DURATION_SECONDS) for ty in types]
    scroll_w = [(s, _shift(s.t_seconds), SCROLL_DURATION_SECONDS) for s in scrolls]
    drag_w = [(d, _shift(d.t_seconds), DRAG_TRAIL_DURATION_SECONDS) for d in drags]

    written = 0
    for i in range(n_frames):
        t = i / fps

        active_clicks = [(e, (t - st) / dur)
                         for (e, st, dur) in click_w
                         if 0.0 <= t - st <= dur]
        active_keys = [(e, (t - st) / dur)
                       for (e, st, dur) in key_w
                       if 0.0 <= t - st <= dur]
        active_types = [(e, (t - st) / dur)
                        for (e, st, dur) in type_w
                        if 0.0 <= t - st <= dur]
        active_scrolls = [(e, (t - st) / dur)
                          for (e, st, dur) in scroll_w
                          if 0.0 <= t - st <= dur]
        active_drags = [(e, (t - st) / dur)
                        for (e, st, dur) in drag_w
                        if 0.0 <= t - st <= dur]

        any_active = (
            active_clicks or active_keys or active_types
            or active_scrolls or active_drags
        )

        out_path = out_dir / f"frame_{i:06d}.png"
        if not any_active:
            try:
                if not out_path.exists():
                    out_path.hardlink_to(blank_path)
            except (OSError, AttributeError):
                import shutil as _sh
                _sh.copyfile(blank_path, out_path)
            continue

        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        # Click ripples
        for click, progress in active_clicks:
            _draw_click_ripple(
                draw, click.x, click.y,
                progress=max(0.0, min(1.0, progress)),
                double=getattr(click, "double", False),
            )
        # Drag trails
        for drag, progress in active_drags:
            _draw_drag_trail(
                draw, drag.x1, drag.y1, drag.x2, drag.y2,
                progress=max(0.0, min(1.0, progress)),
            )
        # Scroll arrows
        for scroll, progress in active_scrolls:
            _draw_scroll_arrow(
                draw, scroll.direction,
                progress=max(0.0, min(1.0, progress)),
                canvas_w=width, canvas_h=height,
            )
        # Type captions (centered top — only show one at a time, last-wins)
        if active_types:
            tev, tprog = active_types[-1]
            _draw_type_caption(
                draw, tev.text,
                progress=max(0.0, min(1.0, tprog)),
                canvas_w=width, canvas_h=height,
            )
        # Key badges — stack from the bottom-right when concurrent
        for slot, (key, progress) in enumerate(active_keys):
            _draw_key_badge(
                draw, key.keys,
                progress=max(0.0, min(1.0, progress)),
                canvas_w=width, canvas_h=height,
                slot=slot,
            )

        canvas.save(out_path, optimize=True)
        written += 1

    logger.info(
        "action overlay: %d clicks / %d keys / %d types / %d scrolls / %d drags "
        "→ %d frames (%d non-blank) at %s",
        len(clicks), len(keys), len(types), len(scrolls), len(drags),
        n_frames, written, out_dir,
    )
    return out_dir


# Backwards-compat: the old name still works for callers that only pass clicks.
def render_ripple_overlay_pngs(
    out_dir: Path,
    *,
    duration_seconds: float,
    fps: int,
    width: int,
    height: int,
    clicks: list[ClickEvent],
    title_offset_seconds: float = 0.0,
) -> Path | None:
    return render_action_overlay_pngs(
        out_dir,
        duration_seconds=duration_seconds,
        fps=fps,
        width=width,
        height=height,
        clicks=clicks,
        title_offset_seconds=title_offset_seconds,
    )


# ── Composition (ffmpeg) ────────────────────────────────────────────────────
def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def compose_polished_video(
    raw_video: Path,
    title_card: Optional[Path],
    outro_card: Optional[Path],
    subtitles_srt: Optional[Path],
    output: Path,
    *,
    ripples_dir: Optional[Path] = None,
    ripples_fps: int = 30,
    title_duration: float = 3.0,
    outro_duration: float = 5.0,
    width: int = 1280,
    height: int = 720,
    fmt: str = "mp4",
) -> bool:
    """Concat title + raw (with burned-in subtitles) + outro into ``output``.

    Any of ``title_card``, ``outro_card``, ``subtitles_srt`` may be None;
    the filtergraph adapts. Returns True on a 0 exit code, False otherwise.
    The raw video is left intact regardless.
    """
    if not _ffmpeg_available():
        logger.warning("ffmpeg not on PATH; skipping polished compose")
        return False
    if not raw_video.exists() or raw_video.stat().st_size == 0:
        logger.warning("raw video missing/empty; cannot polish")
        return False

    output.parent.mkdir(parents=True, exist_ok=True)

    # Build the input list. Order matters for the [N:v] indices in the
    # filter_complex below.
    inputs: list[str] = []

    # 0: raw video
    inputs += ["-i", str(raw_video)]

    # Optional title (loop a single image as a video)
    title_idx: Optional[int] = None
    if title_card and title_card.exists():
        inputs += ["-loop", "1", "-t", str(title_duration), "-i", str(title_card)]
        title_idx = (len(inputs) // 2) - 1

    outro_idx: Optional[int] = None
    if outro_card and outro_card.exists():
        inputs += ["-loop", "1", "-t", str(outro_duration), "-i", str(outro_card)]
        outro_idx = (len(inputs) // 2) - 1

    # Optional ripple PNG sequence (image2 demuxer)
    ripples_idx: Optional[int] = None
    if ripples_dir and ripples_dir.exists():
        first = ripples_dir / "frame_000000.png"
        if first.exists():
            inputs += [
                "-framerate", str(ripples_fps),
                "-i", str(ripples_dir / "frame_%06d.png"),
            ]
            ripples_idx = (len(inputs) // 2) - 1

    # Build filter_complex
    parts: list[str] = []
    # Normalize raw to target size + 30 fps (so concat doesn't choke on
    # mismatched timebase / variable framerate).
    raw_chain = (
        f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30"
    )
    if subtitles_srt and subtitles_srt.exists():
        # Burn captions in. ffmpeg's subtitles= filter expects a single-quoted
        # path with backslash-escaped colons.
        srt_path = str(subtitles_srt).replace(":", r"\:").replace("'", r"\'")
        raw_chain += f",subtitles=filename='{srt_path}'"
    parts.append(f"{raw_chain}[v_run]")

    if ripples_idx is not None:
        # Normalize the PNG sequence to the same canvas + framerate, then
        # overlay onto the run. PNGs are RGBA; ffmpeg honors the alpha
        # channel so transparent regions pass through.
        parts.append(
            f"[{ripples_idx}:v]scale={width}:{height},setsar=1,fps=30,"
            f"format=rgba[v_ripples]"
        )
        parts.append("[v_run][v_ripples]overlay=0:0[v_main]")
    else:
        parts.append("[v_run]copy[v_main]")

    concat_inputs: list[str] = []

    if title_idx is not None:
        parts.append(
            f"[{title_idx}:v]scale={width}:{height},setsar=1,fps=30,"
            f"format=yuv420p[v_title]"
        )
        concat_inputs.append("[v_title]")

    concat_inputs.append("[v_main]")

    if outro_idx is not None:
        parts.append(
            f"[{outro_idx}:v]scale={width}:{height},setsar=1,fps=30,"
            f"format=yuv420p[v_outro]"
        )
        concat_inputs.append("[v_outro]")

    concat_n = len(concat_inputs)
    parts.append(
        "".join(concat_inputs) + f"concat=n={concat_n}:v=1:a=0[v_out]"
    )

    filter_complex = ";".join(parts)

    # Output codec choice mirrors the ScreenRecorder defaults.
    codec_args: list[str]
    if fmt == "webm":
        codec_args = ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32", "-row-mt", "1", "-cpu-used", "5"]
    elif fmt == "gif":
        codec_args = []  # ffmpeg picks the right encoder by extension
    else:
        codec_args = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "26", "-pix_fmt", "yuv420p"]

    cmd = (
        ["ffmpeg", "-y", "-loglevel", "error"]
        + inputs
        + ["-filter_complex", filter_complex, "-map", "[v_out]"]
        + codec_args
        + [str(output)]
    )

    logger.info(
        "compose polished video: %d inputs (raw=1, title=%s, outro=%s, srt=%s) -> %s",
        len(inputs) // 2,
        bool(title_idx is not None),
        bool(outro_idx is not None),
        bool(subtitles_srt and subtitles_srt.exists()),
        output,
    )
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=600)
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg compose timed out")
        return False
    if proc.returncode != 0:
        logger.error(
            "ffmpeg compose failed: %s",
            (proc.stderr or b"").decode(errors="replace")[:500],
        )
        return False
    return output.exists() and output.stat().st_size > 0


# ── Convenience: build cards from a result summary ──────────────────────────
def title_card_for_run(
    *,
    plan_label: str,
    tenant_id: str,
    run_id: str,
    started_at: str = "",
) -> CardConfig:
    return CardConfig(
        title="Mantis CUA",
        subtitle=plan_label,
        body_lines=[
            f"tenant: {tenant_id}",
            f"run: {run_id}",
        ] + ([f"started: {started_at}"] if started_at else []),
        footer="screencast",
    )


def outro_card_from_summary(
    summary: dict[str, Any],
    *,
    plan_label: str = "",
    cost_total: float | None = None,
    duration_seconds: float | None = None,
) -> CardConfig:
    """Build the post-run summary card from a result `summary` dict.

    Accepts the lead-extraction shape (`viable`, `leads_with_phone`) and
    the generic shape (`steps_executed`, `total_time_s`).
    """
    body: list[str] = []
    viable = summary.get("viable")
    phones = summary.get("leads_with_phone")
    if viable is not None or phones is not None:
        body.append(f"viable leads: {viable if viable is not None else '—'}")
        body.append(f"with phone:   {phones if phones is not None else '—'}")
    steps = summary.get("steps_executed")
    if steps is not None:
        body.append(f"steps:        {steps}")
    duration = summary.get("total_time_s") or duration_seconds
    if duration is not None:
        body.append(f"duration:     {int(duration)}s")
    if cost_total is None:
        cost_total = summary.get("cost_total")
    if cost_total is not None:
        body.append(f"cost:         ${float(cost_total):.2f}")

    return CardConfig(
        title="Run complete",
        subtitle=plan_label or "screencast",
        body_lines=body,
        footer="Mantis CUA",
    )
