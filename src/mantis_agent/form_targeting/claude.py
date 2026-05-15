"""Claude-backed :class:`FormTargetProvider` (#406).

Owns the three methods that previously lived on ``ClaudeExtractor``:

- ``find_form_target`` — labelled-element lookup
- ``find_target_by_affordance`` — language-agnostic / icon fallback
- ``verify_dropdown_value`` — post-click dropdown verifier

The implementations are kept verbatim from the extractor so behaviour
is identical — same prompts, same tool_use schemas, same coord-coercion
logic, same logging prefix (``[claude-form]``) so existing log scrapers
keep working. The only changes are:

1. They now live in their own module, not stapled to an extractor class.
2. They reach the Anthropic API via a shared
   :class:`~mantis_agent._anthropic.client.AnthropicToolUseClient` —
   which means the 529-retry-with-backoff from #404 applies to these
   calls without re-implementation.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from PIL import Image

from .._anthropic.client import AnthropicToolUseClient
from .base import DropdownVerifyResult, FormTargetProvider, FormTargetResult

logger = logging.getLogger(__name__)


def _coerce_coord(value: Any) -> int | None:
    """Best-effort int coercion for click coordinates returned by the model.

    Tool_use ``input_schema`` requires ``"type": "integer"`` for
    coordinate fields, but the model occasionally emits values as
    strings with stray whitespace / trailing commas (canonical
    failure: ``"x": "296, "`` observed on long-prompt retries that
    fed failure-history into the search). Crashing the run on those
    cases — instead of treating them as ``not_found`` — was a sharp
    edge surfaced by the priority-field staff-crm rerun.

    Returns the parsed int, or ``None`` when the value can't be
    coerced. Caller treats ``None`` as the same not-found path as
    a zero-coordinate response.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        cleaned = value.strip().rstrip(",;]}").strip()
        if not cleaned:
            return None
        try:
            return int(float(cleaned))
        except (TypeError, ValueError):
            return None
    return None


class ClaudeFormTargetProvider(FormTargetProvider):
    """Default :class:`FormTargetProvider` — uses Claude vision.

    Holds an :class:`AnthropicToolUseClient`. Callers that already have
    an extractor can share its client via ``from_extractor`` so we
    don't allocate two Anthropic-client instances per runner.
    """

    def __init__(self, client: AnthropicToolUseClient) -> None:
        self._client = client
        # Debug-dump directory mirrors the extractor's behaviour so
        # operators can still inspect ``claude_form*.png`` / prompt /
        # response triples in the same location.
        self.debug_dir = os.environ.get(
            "MANTIS_DEBUG_DIR", "/data/screenshots/claude_debug",
        )

    @classmethod
    def from_extractor(cls, extractor: Any) -> "ClaudeFormTargetProvider":
        """Construct a provider sharing the extractor's Anthropic client.

        ``ClaudeExtractor`` exposes ``self._client`` (an
        :class:`AnthropicToolUseClient`) since #406 Part 1; reusing
        the api_key + retry policy keeps a single accounting per
        runner. The model is overridden to the cheaper
        ``form_target_model`` (default Haiku, see #434) when the
        extractor exposes it — grounding is a single visual lookup
        which doesn't need Opus / Sonnet quality.
        """
        client = getattr(extractor, "_client", None)
        if not isinstance(client, AnthropicToolUseClient):
            raise TypeError(
                "from_extractor: extractor must expose a "
                "_client: AnthropicToolUseClient (got %r)" % type(client).__name__
            )
        # #434: prefer the extractor's ``form_target_model`` when set,
        # falling back to the extractor's main model for back-compat
        # with callers that haven't migrated to the split.
        form_target_model = getattr(extractor, "form_target_model", "") or client.model
        new_client = AnthropicToolUseClient(
            api_key=client.api_key,
            model=form_target_model,
            log_prefix="ClaudeFormTarget",
        )
        return cls(new_client)

    def _debug_path(self, stem: str, suffix: str) -> str:
        os.makedirs(self.debug_dir, exist_ok=True)
        return os.path.join(self.debug_dir, f"{stem}{suffix}")

    def find_form_target(
        self,
        screenshot: Image.Image,
        intent: str,
        *,
        target_label: str = "",
        target_value: str = "",
        target_aliases: list[str] | None = None,
    ) -> FormTargetResult | None:
        target_clause = (
            f"\nThe target element label/text is: \"{target_label}\""
            if target_label else ""
        )
        value_clause = (
            f"\nThe value to type or option to select is: \"{target_value}\""
            if target_value else ""
        )
        aliases = [a for a in (target_aliases or []) if a]
        alias_clause = (
            "\nAcceptable equivalent labels (any of these is a valid match): "
            + ", ".join(f'"{a}"' for a in aliases)
            if aliases else ""
        )
        # Prompt body lives at
        # ``mantis_agent/prompts/files/find_form_target.txt`` so plan
        # authors / operators can A/B-test wording without forking
        # this module. Caller-supplied placeholders below.
        from ..prompts import load_prompt

        prompt = load_prompt(
            "find_form_target",
            screen_width=screenshot.width,
            screen_height=screenshot.height,
            intent=intent,
            target_clause=target_clause,
            value_clause=value_clause,
            alias_clause=alias_clause,
        )

        debug_stem = "claude_form"
        try:
            screenshot.save(self._debug_path(debug_stem, ".png"))
        except Exception:
            pass
        try:
            with open(self._debug_path(debug_stem, "_prompt.txt"), "w") as f:
                f.write(prompt)
        except Exception:
            pass

        parsed = self._client.call_with_tool_schema(
            screenshot,
            prompt,
            tool_name="report_form_target",
            tool_description=(
                "Report the form element matching the task — coordinates "
                "and interaction (click / type / select), or not_found "
                "if no matching element is visible on screen."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "Center X in pixels"},
                    "y": {"type": "integer", "description": "Center Y in pixels"},
                    "action": {
                        "type": "string",
                        "enum": ["click", "right_click", "type", "select", "not_found"],
                    },
                    "value": {"type": "string"},
                    "label": {"type": "string"},
                },
                "required": ["x", "y", "action", "value", "label"],
            },
        )

        try:
            with open(self._debug_path(debug_stem, "_response.txt"), "w") as f:
                f.write(json.dumps(parsed) if parsed is not None else "<no tool_use>")
        except Exception:
            pass

        if not parsed:
            logger.warning("  [claude-form] tool_use returned no usable result")
            return None

        if parsed.get("action") == "not_found":
            label = parsed.get("label", "unknown")
            logger.info(f"  [claude-form] target not visible: {intent[:60]}")
            logger.info(f"  [claude-form] What Claude sees: {label[:120]}")
            return None

        x = _coerce_coord(parsed.get("x"))
        y = _coerce_coord(parsed.get("y"))
        if x is None or y is None:
            logger.warning(
                "  [claude-form] non-integer coordinates returned: "
                "x=%r y=%r — treating as not found",
                parsed.get("x"), parsed.get("y"),
            )
            return None
        if x == 0 and y == 0:
            logger.warning("  [claude-form] zero coordinates")
            return None

        result: FormTargetResult = {
            "x": x,
            "y": y,
            "action": str(parsed.get("action", "click")),
            "value": str(parsed.get("value", target_value or "")),
            "label": str(parsed.get("label", target_label or "")),
        }
        logger.info(f"  [claude-form] '{result['label'][:40]}' at ({x},{y}) action={result['action']}")
        return result

    def find_target_by_affordance(
        self,
        screenshot: Image.Image,
        intent: str,
    ) -> FormTargetResult | None:
        prompt = (
            f"Look at this screenshot ({screenshot.width}x{screenshot.height} pixels).\n\n"
            f"INTENT (read this carefully — it tells you what kind of "
            f"element to find): {intent}\n\n"
            f"The label-driven search for this intent already failed: no "
            f"element matched the literal label / alias the plan "
            f"specified. The intent prose ABOVE is the source of truth "
            f"for what the runner is trying to do — read it, infer the "
            f"element type that fits, and locate that element by VISUAL "
            f"AFFORDANCE rather than text matching.\n\n"
            f"Element-type heuristics (use the intent verbs to choose):\n"
            f"- ``click`` / ``open`` / ``submit`` / ``save`` / ``confirm`` "
            f"/ ``select [a row / lead / item]`` — find a button, link, "
            f"or clickable card. Primary action buttons are typically "
            f"coloured / filled, in form footers or sticky toolbars; "
            f"secondary buttons (Cancel, Reset, Back) are typically "
            f"grey / outline.\n"
            f"- ``enter`` / ``type`` / ``fill`` / ``input`` — find a "
            f"text input or textarea. Match the intent's field name to "
            f"the visible label adjacent to (above / left of / "
            f"placeholder inside) the input box.\n"
            f"- ``pick`` / ``choose`` / ``select [option] from [dropdown]`` "
            f"— find a dropdown / select control (visible chevron or "
            f"current value).\n"
            f"- ``toggle`` / ``check`` / ``enable`` / ``disable`` — "
            f"find a checkbox / toggle / radio.\n\n"
            f"LANGUAGE-AGNOSTIC: the element's visible label may be in "
            f"any language (English, French, German, Japanese, etc) "
            f"or icon-only. Match by AFFORDANCE — shape, position, "
            f"styling — not by specific words. The intent's reference "
            f"to a field or button name is a HINT (semantic match) "
            f"rather than a literal text-match requirement.\n\n"
            f"NO HARDCODED ACTION ASSUMPTIONS: don't assume the target "
            f"is necessarily a submit button. If the intent is "
            f"``Enter the password``, the target is the password "
            f"input, NOT the Login button. The intent verb is the "
            f"final word.\n\n"
            f"If you see no element on screen that plausibly matches "
            f"the intent, return action=not_found and describe in "
            f"``label`` what is visible. The runner will route that "
            f"to its existing failure / recovery path.\n\n"
            f"Return CENTER coordinates of the chosen element along "
            f"with the EXACT TEXT or icon description visible on it. "
            f"Set ``action`` to one of ``click`` / ``type`` / "
            f"``select`` based on what the runner should do next "
            f"(matching the verb in the intent)."
        )
        parsed = self._client.call_with_tool_schema(
            screenshot,
            prompt,
            tool_name="report_target_by_affordance",
            tool_description=(
                "Report the location, observed label, and recommended "
                "action for the element best matching the intent — "
                "identified by visual affordance, not label match."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "x": {
                        "type": "integer",
                        "description": "Center x of the chosen element.",
                    },
                    "y": {
                        "type": "integer",
                        "description": "Center y of the chosen element.",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["click", "right_click", "type", "select", "not_found"],
                        "description": (
                            "What the runner should do at this element. "
                            "``click`` for buttons / links / row "
                            "containers; ``right_click`` only when the "
                            "task explicitly needs the browser's native "
                            "context menu on the element (e.g. \"Open "
                            "Link in New Tab\", \"Copy Link\"); ``type`` "
                            "for text inputs the runner should fill; "
                            "``select`` for an option already visible in "
                            "an open dropdown menu (click first if the "
                            "dropdown is closed). ``not_found`` if "
                            "the page has nothing matching the intent."
                        ),
                    },
                    "label": {
                        "type": "string",
                        "description": (
                            "EXACT text observed on / near the element "
                            "(any language) or an icon description for "
                            "icon-only elements. When action=not_found, "
                            "describe what IS visible instead."
                        ),
                    },
                },
                "required": ["action", "label"],
            },
            max_tokens=500,
        )
        if not parsed:
            logger.warning(
                "  [claude-form] affordance: tool_use returned no result"
            )
            return None
        if parsed.get("action") == "not_found":
            logger.info(
                "  [claude-form] affordance: not found — observed: %s",
                str(parsed.get("label", ""))[:120],
            )
            return None
        x = _coerce_coord(parsed.get("x"))
        y = _coerce_coord(parsed.get("y"))
        if x is None or y is None or (x == 0 and y == 0):
            logger.warning(
                "  [claude-form] affordance: invalid coords x=%r y=%r",
                parsed.get("x"), parsed.get("y"),
            )
            return None
        result: FormTargetResult = {
            "x": x,
            "y": y,
            "action": str(parsed.get("action", "click")),
            "value": "",
            "label": str(parsed.get("label", "")),
        }
        logger.info(
            "  [claude-form] affordance: '%s' at (%d, %d) action=%s — "
            "discovered via visual affordance (no alias match needed)",
            result["label"][:60], x, y, result["action"],
        )
        return result

    def verify_dropdown_value(
        self,
        screenshot: Image.Image,
        dropdown_label: str,
        expected_value: str,
    ) -> DropdownVerifyResult | None:
        prompt = (
            f"Look at this screenshot ({screenshot.width}x{screenshot.height} pixels).\n\n"
            f"A dropdown control labelled '{dropdown_label}' is visible on "
            f"the page. Read its CURRENT VALUE — the text displayed inside "
            f"the closed dropdown control, showing which option is currently "
            f"selected. Most dropdowns render the selected text on the left "
            f"of the control with a chevron/arrow on the right.\n\n"
            f"Important: report only what is *literally rendered* inside "
            f"the dropdown control right now — do not infer, normalise, or "
            f"guess. If the dropdown is empty / no value is visible, return "
            f"an empty string. If a menu is still open and partially "
            f"covering the control, report what the control itself shows "
            f"(not the highlighted menu item)."
        )
        parsed = self._client.call_with_tool_schema(
            screenshot,
            prompt,
            tool_name="report_dropdown_value",
            tool_description=(
                "Report the literal text displayed inside a dropdown "
                "control — the currently-selected value."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "observed": {
                        "type": "string",
                        "description": (
                            "The literal text displayed inside the "
                            "dropdown control right now. Empty string "
                            "if no value is visible."
                        ),
                    },
                },
                "required": ["observed"],
            },
            max_tokens=200,
        )
        if not parsed:
            return None
        observed = str(parsed.get("observed") or "")
        matches = _semantic_dropdown_match(observed, expected_value)
        return {"matches": matches, "observed": observed}


def _semantic_dropdown_match(observed: str, expected: str) -> bool:
    """Decide if a dropdown's observed value matches the expected
    option. Case-insensitive, whitespace-tolerant, substring on
    either side. ``"High"`` matches ``"High Priority"`` and vice
    versa; ``"Critical"`` does *not* match ``"High"``. Empty
    observed never matches a non-empty expected (post-click
    verifier blanked its read → caller should treat as not-matched
    rather than silently passing).

    Lifted verbatim from ``ClaudeExtractor._semantic_dropdown_match``
    so existing behaviour is identical.
    """
    a = (observed or "").strip().casefold()
    b = (expected or "").strip().casefold()
    if not a or not b:
        return False
    return a == b or a in b or b in a
