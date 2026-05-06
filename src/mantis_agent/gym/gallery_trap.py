"""Gallery/lightbox trap detection and bounded recovery helpers.

The detector is intentionally cheap and dependency-light. It combines
visual signals from the screenshot with optional text signals from model
reasoning or OCR-like sources when callers have them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from PIL import Image

from ..actions import Action, ActionType


_GALLERY_TEXT_RE = re.compile(
    r"\b\d+\s+of\s+\d+\b|"
    r"\bphoto\s+gallery\b|"
    r"\bimage\s+gallery\b|"
    r"\blightbox\b|"
    r"\bfullscreen\s+photo\b|"
    r"\bphoto\s+viewer\b|"
    r"\bimage\s+viewer\b|"
    r"\bclose\s*[x×]\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GalleryTrapDetection:
    """Result of gallery/lightbox trap detection."""

    detected: bool
    confidence: float
    reason: str
    signals: dict[str, float | bool] = field(default_factory=dict)


def detect_gallery_trap(
    screenshot: Image.Image,
    *,
    text: str = "",
    threshold: float = 0.65,
) -> GalleryTrapDetection:
    """Detect likely fullscreen gallery/lightbox state.

    Args:
        screenshot: Current post-action screenshot.
        text: Optional model reasoning / OCR text. Textual "1 of N",
            "lightbox", or "photo gallery" signals are strong evidence.
        threshold: Minimum confidence required for ``detected=True``.

    The visual heuristic looks for a dark full-screen overlay with a brighter
    central image area. It deliberately does not classify a uniformly dark page
    as a gallery unless a text signal is also present.
    """
    visual = _visual_gallery_signals(screenshot)
    text_hit = bool(text and _GALLERY_TEXT_RE.search(text))

    score = 0.0
    if text_hit:
        score += 0.70
    if visual["dark_ratio"] >= 0.45:
        score += 0.25
    if visual["border_dark_ratio"] >= 0.65:
        score += 0.20
    if visual["center_contrast"] >= 0.18 and visual["center_bright_ratio"] >= 0.12:
        score += 0.25
    if visual["uniform_dark"]:
        score -= 0.20

    score = max(0.0, min(1.0, score))
    detected = score >= threshold
    signals: dict[str, float | bool] = {
        **visual,
        "text_gallery_signal": text_hit,
    }
    reason = _reason(signals, detected)
    return GalleryTrapDetection(
        detected=detected,
        confidence=score,
        reason=reason,
        signals=signals,
    )


def gallery_recovery_actions() -> tuple[Action, ...]:
    """Bounded recovery sequence for an open gallery/lightbox."""
    return (
        Action(ActionType.KEY_PRESS, {"keys": "escape"}, reasoning="recover from gallery trap"),
        Action(ActionType.KEY_PRESS, {"keys": "alt+left"}, reasoning="recover from gallery trap"),
    )


def _visual_gallery_signals(screenshot: Image.Image) -> dict[str, float | bool]:
    img = screenshot.convert("L").resize((160, 90))
    pixels = list(img.getdata())
    total = len(pixels) or 1
    dark_ratio = sum(1 for p in pixels if p < 55) / total

    w, h = img.size
    border_pixels: list[int] = []
    border = max(4, min(w, h) // 10)
    for y in range(h):
        for x in range(w):
            if x < border or x >= w - border or y < border or y >= h - border:
                border_pixels.append(img.getpixel((x, y)))
    border_dark_ratio = sum(1 for p in border_pixels if p < 65) / (len(border_pixels) or 1)

    # Center crop approximates the photo pane in common fullscreen galleries.
    x1, x2 = int(w * 0.22), int(w * 0.78)
    y1, y2 = int(h * 0.20), int(h * 0.82)
    center_pixels = [
        img.getpixel((x, y))
        for y in range(y1, y2)
        for x in range(x1, x2)
    ]
    center_avg = sum(center_pixels) / (len(center_pixels) or 1)
    full_avg = sum(pixels) / total
    center_bright_ratio = sum(1 for p in center_pixels if p > 90) / (len(center_pixels) or 1)
    center_contrast = max(0.0, (center_avg - full_avg) / 255.0)

    return {
        "dark_ratio": dark_ratio,
        "border_dark_ratio": border_dark_ratio,
        "center_bright_ratio": center_bright_ratio,
        "center_contrast": center_contrast,
        "uniform_dark": dark_ratio > 0.95 and center_bright_ratio < 0.03,
    }


def _reason(signals: dict[str, float | bool], detected: bool) -> str:
    if not detected:
        return "no gallery trap detected"
    reasons: list[str] = []
    if signals.get("text_gallery_signal"):
        reasons.append("gallery text signal")
    if float(signals.get("dark_ratio", 0.0)) >= 0.45:
        reasons.append("dark overlay")
    if float(signals.get("center_bright_ratio", 0.0)) >= 0.12:
        reasons.append("central image pane")
    return ", ".join(reasons) or "gallery-like screenshot"
