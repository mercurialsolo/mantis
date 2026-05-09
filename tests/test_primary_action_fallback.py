"""Tests for the vision-based primary-action fallback in find_form_target.

Surfaced by the staff-crm priority rerun on Modal: the form had an
``Update Lead`` button visible in the bottom-right, but Claude's
label-driven ``find_form_target`` search failed across all 9 scroll
probes. Diagnosis from the user's manually-captured screenshot:
the form footer contains Cancel | Reset Form | Update Lead — the
button IS labelled correctly, but the runner's End / Page_Down
keystrokes were absorbed by an open dropdown so the visible
viewport never reached the form footer where the buttons live.

Two layered fixes:

1. Defocus any active input (Tab key) before the vision fallback's
   final scroll-to-end, so the keystroke actually moves the page
   instead of acting on the focused dropdown.
2. Vision-based primary-action search — when label-match exhausts,
   ask Claude to identify the action button by visual affordance
   (colour, position, primary-CTA pattern) regardless of label.
   Language-agnostic: works for English (Save), French
   (Enregistrer), German (Speichern), icon-only buttons, etc.

These tests pin:

- :meth:`ClaudeExtractor.find_target_by_affordance` — tool_use
  shape, label-agnostic prompt, coercion of malformed coords,
  not_found path
- Form handler integration — the vision fallback fires after
  scroll-probe exhausts, defocuses + scrolls before searching,
  and reports the discovered label so the operator can update
  plan aliases.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from PIL import Image


# ── ClaudeExtractor.find_target_by_affordance ────────────────────────────


def _tool_resp(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "content": [
            {
                "type": "tool_use",
                "name": "report_target_by_affordance",
                "input": payload,
            }
        ]
    }
    return resp


def test_primary_action_returns_coords_and_label_when_found() -> None:
    """Successful vision pass: Claude identifies the button by
    affordance, returns coords + observed label."""
    from mantis_agent.extraction.extractor import ClaudeExtractor

    ex = ClaudeExtractor(api_key="k")
    img = Image.new("RGB", (1280, 720), color="white")
    payload = {
        "x": 1198, "y": 670,
        "action": "click",
        "label": "Update Lead",
    }
    with patch("requests.post", return_value=_tool_resp(payload)):
        result = ex.find_target_by_affordance(img, "save the form")

    assert result is not None
    assert result["x"] == 1198
    assert result["y"] == 670
    assert result["label"] == "Update Lead"
    assert result["action"] == "click"


def test_primary_action_returns_none_on_not_found() -> None:
    """When the page truly has no submit button (e.g. read-only
    detail page or auto-saving form), Claude returns
    ``action=not_found`` and the helper returns ``None``."""
    from mantis_agent.extraction.extractor import ClaudeExtractor

    ex = ClaudeExtractor(api_key="k")
    img = Image.new("RGB", (1280, 720))
    payload = {
        "x": 0, "y": 0, "action": "not_found",
        "label": "page is read-only — no submit / save / update visible",
    }
    with patch("requests.post", return_value=_tool_resp(payload)):
        result = ex.find_target_by_affordance(img, "save")

    assert result is None


def test_primary_action_handles_non_english_label() -> None:
    """The whole point: language-agnostic vision affordance. A
    French ``Enregistrer`` button must be returned with its
    actual label so the operator can see what the runner found."""
    from mantis_agent.extraction.extractor import ClaudeExtractor

    ex = ClaudeExtractor(api_key="k")
    img = Image.new("RGB", (1280, 720))
    payload = {
        "x": 1100, "y": 600,
        "action": "click",
        "label": "Enregistrer",
    }
    with patch("requests.post", return_value=_tool_resp(payload)):
        result = ex.find_target_by_affordance(img, "save changes")

    assert result is not None
    assert result["label"] == "Enregistrer"


def test_affordance_returns_input_for_fill_intent() -> None:
    """Critical contract: the affordance pass is intent-driven, NOT
    hardcoded to 'primary action button'. A fill_field intent must
    produce a ``type`` action targeting an INPUT, not a click on
    some random button. Hardcoding action types would break plans
    that need text entry."""
    from mantis_agent.extraction.extractor import ClaudeExtractor

    ex = ClaudeExtractor(api_key="k")
    img = Image.new("RGB", (1280, 720))
    payload = {
        "x": 600, "y": 240,
        "action": "type",
        "label": "Mot de passe",  # French "password" — language-agnostic
    }
    with patch("requests.post", return_value=_tool_resp(payload)):
        result = ex.find_target_by_affordance(
            img, "Enter the password into the password field",
        )

    assert result is not None
    assert result["action"] == "type"
    assert result["label"] == "Mot de passe"


def test_affordance_returns_select_for_dropdown_intent() -> None:
    """Same — for a select_option intent, affordance returns ``select``
    on the option / ``click`` on the dropdown control. NOT a button
    click hardcoded into the prompt."""
    from mantis_agent.extraction.extractor import ClaudeExtractor

    ex = ClaudeExtractor(api_key="k")
    img = Image.new("RGB", (1280, 720))
    payload = {
        "x": 800, "y": 400,
        "action": "click",
        "label": "Priority Level",
    }
    with patch("requests.post", return_value=_tool_resp(payload)):
        result = ex.find_target_by_affordance(
            img, "Pick High from the Priority Level dropdown",
        )

    assert result is not None
    assert result["action"] == "click"
    assert result["label"] == "Priority Level"


def test_primary_action_handles_icon_only_button() -> None:
    """Icon-only buttons (no text) — Claude returns a description
    of the icon as the label so logs surface what was clicked."""
    from mantis_agent.extraction.extractor import ClaudeExtractor

    ex = ClaudeExtractor(api_key="k")
    img = Image.new("RGB", (1280, 720))
    payload = {
        "x": 1200, "y": 50,
        "action": "click",
        "label": "checkmark icon (no text)",
    }
    with patch("requests.post", return_value=_tool_resp(payload)):
        result = ex.find_target_by_affordance(img, "submit form")

    assert result is not None
    assert "checkmark" in result["label"].lower()


def test_primary_action_coerces_malformed_coords() -> None:
    """Same coerce_coord defensiveness as find_form_target — handles
    ``"x": "1198, "`` and similar trailing-comma malformations."""
    from mantis_agent.extraction.extractor import ClaudeExtractor

    ex = ClaudeExtractor(api_key="k")
    img = Image.new("RGB", (1280, 720))
    payload = {
        "x": "1198, ", "y": 670,
        "action": "click",
        "label": "Update Lead",
    }
    with patch("requests.post", return_value=_tool_resp(payload)):
        result = ex.find_target_by_affordance(img, "save")

    assert result is not None
    assert result["x"] == 1198


def test_primary_action_returns_none_on_zero_coords() -> None:
    """``(0, 0)`` is the not-found sentinel even when action=click —
    same convention as find_form_target."""
    from mantis_agent.extraction.extractor import ClaudeExtractor

    ex = ClaudeExtractor(api_key="k")
    img = Image.new("RGB", (1280, 720))
    payload = {"x": 0, "y": 0, "action": "click", "label": "?"}
    with patch("requests.post", return_value=_tool_resp(payload)):
        assert ex.find_target_by_affordance(img, "save") is None


def test_primary_action_prompt_is_language_agnostic() -> None:
    """The prompt MUST tell Claude the label can be in any language /
    icon-only — pin this so future edits don't accidentally bias the
    search toward English-only matching."""
    from mantis_agent.extraction.extractor import ClaudeExtractor

    captured: dict = {}

    def _capture(url, **kwargs):
        captured["json"] = kwargs.get("json")
        return _tool_resp({"action": "not_found", "label": "x"})

    ex = ClaudeExtractor(api_key="k")
    img = Image.new("RGB", (1280, 720))
    with patch("requests.post", side_effect=_capture):
        ex.find_target_by_affordance(img, "save")

    body = captured["json"]
    # Find the text content block (image is first).
    content = body["messages"][0]["content"]
    text_block = next(c for c in content if c.get("type") == "text")
    prompt = text_block["text"].lower()
    # Language guidance present.
    assert "any language" in prompt or "language-agnostic" in prompt or "language" in prompt
    # Visual affordance cues mentioned (color, position).
    assert "colour" in prompt or "color" in prompt or "accent" in prompt
    assert "position" in prompt or "footer" in prompt or "bottom" in prompt
    # No hardcoded label expected.
    assert "must be 'save'" not in prompt and "must be 'submit'" not in prompt
    # Critical contract: the prompt is INTENT-driven, NOT hardcoded
    # to find a primary action button. Plans that need text entry,
    # dropdown selection, or toggles must route through the same
    # affordance pass with the right action type.
    assert "intent" in prompt
    assert "no hardcoded action" in prompt or "intent verb" in prompt or "don't assume" in prompt
    # Element-type heuristics for input / dropdown / button must all
    # be enumerated.
    assert "input" in prompt and "dropdown" in prompt and "button" in prompt


# ── Form handler integration ────────────────────────────────────────────


def test_form_handler_falls_back_to_vision_when_label_search_exhausts() -> None:
    """End-to-end: when the scroll-probe sweep returns None for every
    position, the form handler defocuses + scrolls to end + calls
    find_target_by_affordance. If THAT returns a target, the click
    proceeds and the step succeeds."""
    from mantis_agent.actions import Action, ActionType
    from mantis_agent.gym.step_context import StepContext
    from mantis_agent.gym.step_handlers.form import ClaudeGuidedFormHandler
    import mantis_agent.gym.step_handlers.form as form_mod
    from mantis_agent.plan_decomposer import MicroIntent

    env = MagicMock()
    img = Image.new("RGB", (10, 10))
    env.screenshot.return_value = img

    extractor = MagicMock()
    extractor.find_form_target.return_value = None  # every scroll probe misses
    extractor.find_target_by_affordance.return_value = {
        "x": 1198, "y": 670, "action": "click", "value": "",
        "label": "Update Lead",
    }

    runner = MagicMock()
    runner.extractor = extractor
    runner.env = env
    runner._best_effort_current_url = MagicMock(return_value="https://crm.test/")
    runner._adaptive_submit_settle = MagicMock(return_value=0.0)
    runner._safe_screenshot = MagicMock(return_value=None)
    runner._dump_debug_screenshot = MagicMock()
    runner._last_known_url = "https://crm.test/"
    runner._step_failure_history = {}
    runner._recovery_hints = {}
    runner.costs = {"claude_extract": 0, "gpu_steps": 0, "gpu_seconds": 0.0}

    step = MicroIntent(
        intent="Click the Update Lead button to save",
        type="submit", budget=4,
        params={"label": "Update Lead", "aliases": ["Save", "Submit"]},
    )

    orig = form_mod._wait_for_rendered_screenshot
    form_mod._wait_for_rendered_screenshot = lambda *_a, **_kw: img
    try:
        ctx = StepContext(env=env, brain=MagicMock(), extractor=extractor)
        ctx.state["index"] = 0
        result = ClaudeGuidedFormHandler(runner).execute(step, ctx)
    finally:
        form_mod._wait_for_rendered_screenshot = orig

    assert result.success is True
    # The vision fallback was invoked exactly once (label-match search
    # is independent of vision-affordance pass).
    extractor.find_target_by_affordance.assert_called_once()
    # Tab + End keypresses fired before the vision search to defocus
    # any active input and force scroll-to-bottom.
    keys_pressed = []
    for c in env.step.call_args_list:
        if not c.args:
            continue
        action = c.args[0]
        if (
            isinstance(action, Action)
            and action.action_type == ActionType.KEY_PRESS
        ):
            keys_pressed.append(action.params.get("keys"))
    assert "Tab" in keys_pressed
    assert "End" in keys_pressed


def test_form_handler_does_not_call_vision_when_label_match_succeeds() -> None:
    """Cost guard: when label-match finds the target on any probe,
    vision fallback must NOT fire. One Claude vision call per step
    is the right cost; firing the affordance pass too is wasteful."""
    from mantis_agent.gym.step_context import StepContext
    from mantis_agent.gym.step_handlers.form import ClaudeGuidedFormHandler
    import mantis_agent.gym.step_handlers.form as form_mod
    from mantis_agent.plan_decomposer import MicroIntent

    env = MagicMock()
    img = Image.new("RGB", (10, 10))
    env.screenshot.return_value = img

    extractor = MagicMock()
    # Initial probe finds it.
    extractor.find_form_target.return_value = {
        "x": 540, "y": 600, "action": "click", "value": "",
        "label": "Save",
    }

    runner = MagicMock()
    runner.extractor = extractor
    runner.env = env
    runner._best_effort_current_url = MagicMock(return_value="https://crm.test/")
    runner._adaptive_submit_settle = MagicMock(return_value=0.0)
    runner._safe_screenshot = MagicMock(return_value=None)
    runner._dump_debug_screenshot = MagicMock()
    runner._last_known_url = "https://crm.test/"
    runner._step_failure_history = {}
    runner._recovery_hints = {}
    runner.costs = {"claude_extract": 0, "gpu_steps": 0, "gpu_seconds": 0.0}

    step = MicroIntent(
        intent="Click Save", type="submit",
        params={"label": "Save"},
    )

    orig = form_mod._wait_for_rendered_screenshot
    form_mod._wait_for_rendered_screenshot = lambda *_a, **_kw: img
    try:
        ctx = StepContext(env=env, brain=MagicMock(), extractor=extractor)
        ctx.state["index"] = 0
        result = ClaudeGuidedFormHandler(runner).execute(step, ctx)
    finally:
        form_mod._wait_for_rendered_screenshot = orig

    assert result.success is True
    # Vision fallback skipped — label-match was sufficient.
    extractor.find_target_by_affordance.assert_not_called()
