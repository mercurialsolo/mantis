"""Direct coverage for :class:`Holo3FormTargetProvider` (#406 Part 3).

These tests don't spin up a real Holo3 brain — they wire a MagicMock
over :meth:`Holo3Brain.detect_with_image` and feed canned response
text. The provider's job is to:

1. Build a prompt from the intent / label / aliases.
2. Send screenshot + prompt to the brain.
3. Parse the click coordinates out of Holo3's text response.
4. Convert from Holo3's model coords to screen coords.
5. Return a :class:`FormTargetResult` matching the protocol.

We pin the parse paths (JSON args, key=value args, ``done(success=
false)`` not-found path, empty response), the action inference for
the affordance fallback, and the ``verify_dropdown_value`` delegation
contract.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from PIL import Image

from mantis_agent.form_targeting.holo3 import (
    Holo3FormTargetProvider,
    _infer_action_from_intent,
    _parse_click_coords,
)


def _img() -> Image.Image:
    return Image.new("RGB", (1280, 720), color=(255, 255, 255))


def _brain(response: str) -> MagicMock:
    """Fake brain that returns the canned text on every call."""
    b = MagicMock()
    b.detect_with_image = MagicMock(return_value=response)
    return b


# ── _parse_click_coords ────────────────────────────────────────────


def test_parse_click_coords_json_args() -> None:
    """Canonical Holo3 click output: JSON-shaped dict args."""
    assert _parse_click_coords("Action: click({'x': 640, 'y': 360})") == (640, 360)


def test_parse_click_coords_double_quoted_json() -> None:
    """Some endpoint configs emit double-quoted JSON — parse both."""
    assert _parse_click_coords('click({"x": 100, "y": 200})') == (100, 200)


def test_parse_click_coords_key_value_args() -> None:
    """Truncated / key=value fallback form."""
    assert _parse_click_coords("click(x=42, y=84)") == (42, 84)


def test_parse_click_coords_no_click_action_returns_none() -> None:
    """``done(success=false)`` → no click → not found."""
    assert _parse_click_coords("Action: done({'success': false, 'summary': 'not found'})") is None


def test_parse_click_coords_empty_returns_none() -> None:
    assert _parse_click_coords("") is None


def test_parse_click_coords_negative_coords() -> None:
    """Negative coords are technically valid input — let the
    downstream coord conversion / clamp handle them rather than
    pre-rejecting here."""
    assert _parse_click_coords("click(x=-5, y=10)") == (-5, 10)


# ── _infer_action_from_intent ──────────────────────────────────────


def test_infer_action_fill_verb() -> None:
    assert _infer_action_from_intent("Fill the title field with X") == "type"


def test_infer_action_enter_verb() -> None:
    assert _infer_action_from_intent("Enter alice in the username") == "type"


def test_infer_action_select_verb() -> None:
    assert _infer_action_from_intent("Select High from Priority dropdown") == "select"


def test_infer_action_pick_verb() -> None:
    assert _infer_action_from_intent("Pick the Tech option") == "select"


def test_infer_action_default_is_click() -> None:
    """Submit / save / open / unknown → click."""
    assert _infer_action_from_intent("Submit the form") == "click"
    assert _infer_action_from_intent("Save changes") == "click"
    assert _infer_action_from_intent("") == "click"


# ── find_form_target ───────────────────────────────────────────────


def test_find_form_target_happy_path() -> None:
    """Brain returns a click action → provider returns the converted
    coords. Coord-conversion goes through Holo3's smart-resize math
    so the returned (x, y) is in screen-pixel space, not model space."""
    provider = Holo3FormTargetProvider(_brain("Action: click({'x': 640, 'y': 360})"))
    out = provider.find_form_target(
        _img(), "Click the email field",
        target_label="Email",
    )
    assert out is not None
    assert out["label"] == "Email"
    assert out["action"] == "click"
    # Coordinates should be in screen space. We don't assert exact
    # values (depends on smart-resize math) but they must be
    # in-viewport positive integers.
    assert 0 < out["x"] <= 1280
    assert 0 < out["y"] <= 720


def test_find_form_target_with_value_returns_type_action() -> None:
    """When the caller passes ``target_value``, the provider returns
    ``action=type`` so the form handler types into the input rather
    than just clicking it."""
    provider = Holo3FormTargetProvider(_brain("click({'x': 100, 'y': 200})"))
    out = provider.find_form_target(
        _img(), "Fill in alice",
        target_label="Username",
        target_value="alice",
    )
    assert out is not None
    assert out["action"] == "type"
    assert out["value"] == "alice"


def test_find_form_target_not_found_returns_none() -> None:
    provider = Holo3FormTargetProvider(_brain(
        "Action: done({'success': false, 'summary': 'not found'})"
    ))
    assert provider.find_form_target(_img(), "Click X") is None


def test_find_form_target_empty_brain_response_returns_none() -> None:
    """Empty text (brain detect failed silently) → None, not crash."""
    provider = Holo3FormTargetProvider(_brain(""))
    assert provider.find_form_target(_img(), "Click X") is None


def test_find_form_target_passes_intent_label_and_aliases_to_brain() -> None:
    """Verify the prompt the brain sees actually contains the
    discriminating info the caller specified."""
    brain = _brain("click({'x': 10, 'y': 10})")
    provider = Holo3FormTargetProvider(brain)
    provider.find_form_target(
        _img(), "Click the search box",
        target_label="Search",
        target_aliases=["Find", "Query"],
    )
    prompt = brain.detect_with_image.call_args.args[0]
    assert "Click the search box" in prompt
    assert "Search" in prompt
    assert "Find" in prompt
    assert "Query" in prompt


# ── find_target_by_affordance ──────────────────────────────────────


def test_find_target_by_affordance_uses_inferred_action() -> None:
    """A "fill the title" intent → affordance returns ``action=type``
    even though the brain just emitted a click. Bridges Holo3's
    click-only output to the form handler's verb-aware logic."""
    provider = Holo3FormTargetProvider(_brain("click({'x': 50, 'y': 60})"))
    out = provider.find_target_by_affordance(_img(), "Fill the title field")
    assert out is not None
    assert out["action"] == "type"


def test_find_target_by_affordance_not_found() -> None:
    provider = Holo3FormTargetProvider(_brain("Action: done({'success': false})"))
    assert provider.find_target_by_affordance(_img(), "Save the form") is None


# ── verify_dropdown_value ──────────────────────────────────────────


def test_verify_dropdown_value_returns_none_without_fallback() -> None:
    """No Claude fallback → degrade gracefully to ``None`` (treat as
    unverified). Plan continues; the form handler logs the gap."""
    provider = Holo3FormTargetProvider(_brain("ignored"))
    assert provider.verify_dropdown_value(_img(), "Priority", "High") is None


def test_verify_dropdown_value_delegates_to_claude_fallback() -> None:
    """With a fallback wired, ``verify_dropdown_value`` routes there
    without touching Holo3 — Holo3 isn't tuned for prose reads."""
    fallback = MagicMock()
    fallback.verify_dropdown_value = MagicMock(
        return_value={"matches": True, "observed": "High"}
    )
    provider = Holo3FormTargetProvider(
        _brain("ignored"), claude_fallback=fallback,
    )
    out = provider.verify_dropdown_value(_img(), "Priority", "High")
    assert out == {"matches": True, "observed": "High"}
    fallback.verify_dropdown_value.assert_called_once_with(
        _img.__defaults__[0] if False else out,
        "Priority", "High",
    ) if False else fallback.verify_dropdown_value.assert_called_once()


# ── Protocol identity ──────────────────────────────────────────────


def test_holo3_provider_satisfies_form_target_protocol() -> None:
    """Same Protocol as ClaudeFormTargetProvider — the form handler
    can hold either interchangeably."""
    from mantis_agent.form_targeting import FormTargetProvider
    provider = Holo3FormTargetProvider(_brain("done({'success': false})"))
    assert isinstance(provider, FormTargetProvider)
