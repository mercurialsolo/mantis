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

logger = logging.getLogger("mantis_agent.presentation")


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
    parts.append(f"{raw_chain}[v_main]")

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
