"""Holo3-backed :class:`FormTargetProvider` (#406 Part 3).

Sends the screenshot + a grounding prompt to a :class:`Holo3Brain`
endpoint, parses click coordinates from the response. Uses the same
model coords → screen coords conversion the brain does for its own
action parsing, so the returned (x, y) is directly usable by the form
handler without further adjustment.

Why this exists alongside :class:`ClaudeFormTargetProvider`:

- Anthropic's API returned 529 Overloaded on every find_form_target
  call inside a single fill_field step's budget during the lu.ma
  smoke (run id ``20260515_004848_5b2ba5c8``) — even with the retry
  loop from #403/#404, an ~3 minute overload window can still halt a
  required step. Holo3 runs in the same Modal container with its own
  quota pool; routing the form-target calls there is independent
  recovery.
- Holo3 is the cheaper path. Each ``find_form_target`` Claude call
  costs ~$0.01–0.02 and ~2s. The Holo3 endpoint serves grounding at
  GPU-amortised cost with sub-second latency once warm.

Trade-off vs. Claude: Holo3 is purpose-built for *clickable element
location* on UI, but it's not tuned for reading prose (the
verify_dropdown_value "what does the dropdown say right now?" pattern
needs text-OCR-style reasoning). This provider keeps a *Claude
fallback* for ``verify_dropdown_value`` — calls go to the fallback
when present, return ``None`` otherwise. Production routing in
``StepContext`` wires Holo3 + a Claude fallback so the verifier never
silently degrades.

The smoke gate before flipping ``MANTIS_FORM_TARGET_PROVIDER`` to
``holo3`` by default is a separate piece of work (docs +
side-by-side comparison on three canonical forms).
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from PIL import Image

from .base import DropdownVerifyResult, FormTargetProvider, FormTargetResult

if TYPE_CHECKING:
    from ..brain_holo3 import Holo3Brain

logger = logging.getLogger(__name__)


# Holo3 returns text like ``Action: click({'x': 640, 'y': 360})``.
# This regex picks the (x, y) out of either the JSON-shaped args or
# the bare ``key=value`` form the model occasionally emits when its
# output gets truncated mid-token.
_CLICK_PATTERN = re.compile(
    r"(?:Action:\s*)?click\(\s*(\{.*?\}|\S.*?)\s*\)",
    re.DOTALL | re.IGNORECASE,
)


def _parse_click_coords(text: str) -> tuple[int, int] | None:
    """Pull (x, y) out of a Holo3 response.

    Accepts either ``click({'x': 640, 'y': 360})`` (JSON-ish) or
    ``click(x=640, y=360)`` (key=value) — both forms appear in
    practice depending on prompt phrasing. Returns ``None`` when no
    click action is present at all, so the caller can degrade to
    not-found rather than crashing the run.
    """
    m = _CLICK_PATTERN.search(text)
    if not m:
        return None
    args_blob = m.group(1).strip()
    x = y = None
    if args_blob.startswith("{"):
        # JSON-ish — try double-quote then single-quote variant
        # (Holo3 uses ' for Python-style dicts).
        for candidate in (args_blob, args_blob.replace("'", '"')):
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    x = obj.get("x")
                    y = obj.get("y")
                    break
            except json.JSONDecodeError:
                continue
    if x is None or y is None:
        # key=value form — picks ``x=640`` and ``y=360``.
        for k, v in re.findall(r"(\w+)\s*=\s*[\"']?(-?\d+)[\"']?", args_blob):
            if k.lower() == "x":
                x = v
            elif k.lower() == "y":
                y = v
    if x is None or y is None:
        return None
    try:
        return int(float(str(x))), int(float(str(y)))
    except (TypeError, ValueError):
        return None


_FORM_PROMPT_TEMPLATE = (
    "You are looking at a {width}x{height} screenshot of a web page.\n\n"
    "Locate the form control matching this description:\n"
    "  intent: {intent}\n"
    "  label: {label}\n"
    "  acceptable labels: {aliases}\n\n"
    "If you can see a matching input field, dropdown, or button, "
    "respond with a single click action at its center, in the form:\n"
    "  Action: click({{'x': X, 'y': Y}})\n\n"
    "If no visible element matches, respond with:\n"
    "  Action: done({{'success': false, 'summary': 'not found'}})"
)


_AFFORDANCE_PROMPT_TEMPLATE = (
    "You are looking at a {width}x{height} screenshot of a web page.\n\n"
    "Locate the element that best matches this intent by its visual "
    "shape and position (not by exact text — text may be missing, in "
    "another language, or icon-only):\n"
    "  {intent}\n\n"
    "Use these heuristics to pick the element type from the intent:\n"
    "  click/open/submit/save → button, link, or clickable card\n"
    "  enter/type/fill/input  → text input or textarea\n"
    "  pick/choose/select     → dropdown control\n"
    "  toggle/check/enable    → checkbox / switch / radio\n\n"
    "If you can see a matching element, respond with a single click "
    "action at its center, in the form:\n"
    "  Action: click({{'x': X, 'y': Y}})\n\n"
    "If no visible element matches, respond with:\n"
    "  Action: done({{'success': false, 'summary': 'not found'}})"
)


class Holo3FormTargetProvider(FormTargetProvider):
    """Holo3-backed form-target grounding.

    Sends a single screenshot + grounding prompt to the brain's vision
    endpoint via :meth:`Holo3Brain.detect_with_image`, parses the
    coordinates Holo3 emits, converts them through the brain's own
    coordinate system, and returns a :class:`FormTargetResult`.

    The brain instance is held by reference so a single Holo3 deploy
    can serve both the runner's CUA loop and the form-target provider
    — no second model spin-up.

    ``verify_dropdown_value`` is delegated to ``claude_fallback`` when
    provided. Holo3 isn't tuned for "read this displayed value" prose
    so the verifier would degrade silently if we let it route here.
    Wiring the fallback explicitly keeps the contract honest.
    """

    def __init__(
        self,
        brain: "Holo3Brain",
        *,
        claude_fallback: FormTargetProvider | None = None,
    ) -> None:
        self._brain = brain
        self._claude_fallback = claude_fallback

    def find_form_target(
        self,
        screenshot: Image.Image,
        intent: str,
        *,
        target_label: str = "",
        target_value: str = "",
        target_aliases: list[str] | None = None,
    ) -> FormTargetResult | None:
        aliases = [a for a in (target_aliases or []) if a]
        prompt = _FORM_PROMPT_TEMPLATE.format(
            width=screenshot.width,
            height=screenshot.height,
            intent=intent,
            label=target_label or "(unspecified)",
            aliases=", ".join(aliases) or "(none)",
        )
        text = self._brain.detect_with_image(prompt, screenshot, max_tokens=256)
        return self._parse_form_response(
            text, screenshot.size,
            fallback_action="type" if target_value else "click",
            label=target_label, value=target_value,
        )

    def find_target_by_affordance(
        self,
        screenshot: Image.Image,
        intent: str,
    ) -> FormTargetResult | None:
        prompt = _AFFORDANCE_PROMPT_TEMPLATE.format(
            width=screenshot.width,
            height=screenshot.height,
            intent=intent,
        )
        text = self._brain.detect_with_image(prompt, screenshot, max_tokens=256)
        # Affordance pass doesn't know the canonical value, leave it
        # blank — caller already has it from MicroIntent.params.
        return self._parse_form_response(
            text, screenshot.size,
            fallback_action=_infer_action_from_intent(intent),
            label="", value="",
        )

    def verify_dropdown_value(
        self,
        screenshot: Image.Image,
        dropdown_label: str,
        expected_value: str,
    ) -> DropdownVerifyResult | None:
        """Holo3 isn't tuned for text-reading — delegate to the
        Claude fallback when configured; return ``None`` (= "couldn't
        verify, trust the click") otherwise.

        Returning None matches the legacy "API blip" semantics in the
        form handler so callers don't need to special-case the
        Holo3 path.
        """
        if self._claude_fallback is not None:
            return self._claude_fallback.verify_dropdown_value(
                screenshot, dropdown_label, expected_value,
            )
        logger.info(
            "  [holo3-form] verify_dropdown_value skipped — no Claude "
            "fallback configured; returning None (treat as unverified)",
        )
        return None

    def _parse_form_response(
        self,
        text: str,
        screen_size: tuple[int, int],
        *,
        fallback_action: str,
        label: str,
        value: str,
    ) -> FormTargetResult | None:
        if not text:
            logger.warning("  [holo3-form] empty response from brain")
            return None
        coords = _parse_click_coords(text)
        if coords is None:
            # ``done(success=false)`` is the canonical "not found"
            # path; log at INFO so the trace shows Holo3 actually
            # answered (vs. a network blip that returned empty text).
            logger.info(
                "  [holo3-form] no click in response — treating as "
                "not_found: %s", text[:120],
            )
            return None
        from ..brain_holo3 import _model_coords_to_screen
        sx, sy = _model_coords_to_screen(
            coords[0], coords[1], screen_size[0], screen_size[1],
        )
        result: FormTargetResult = {
            "x": int(sx),
            "y": int(sy),
            "action": fallback_action,
            "value": value or "",
            "label": label or "",
        }
        logger.info(
            "  [holo3-form] '%s' at (%d, %d) action=%s",
            label[:40] or "(no label)", sx, sy, fallback_action,
        )
        return result


def _infer_action_from_intent(intent: str) -> str:
    """Map intent verbs to the action the runner should take.

    Mirrors the affordance prompt's heuristics so a caller that asks
    for "Fill the title field" gets ``action="type"`` even when the
    brain's response (``Action: click(...)``) doesn't carry that
    semantic. Without this nudge, every affordance result defaulted
    to ``click``, which the form handler then refused for fill
    intents under the #404 hardening.
    """
    lo = (intent or "").lower()
    if any(verb in lo for verb in ("fill", "enter", "type", "input")):
        return "type"
    if any(verb in lo for verb in ("pick", "choose", "select")):
        return "select"
    return "click"
