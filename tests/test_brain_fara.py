"""Unit tests for :mod:`mantis_agent.brain_fara`.

Covers the parser (tool_calls + text fallback), the Fara→Mantis action
mapping (including the visit_url / history_back / web_search adapters
and the double_click / right_click / drag regressions), coordinate
scaling, and the prompt registration plumbing.

No GPU and no live vLLM — every think() round-trip is faked via
:meth:`FaraBrain._parse_response`.
"""

from __future__ import annotations

from urllib.parse import quote_plus

import pytest

from mantis_agent.actions import Action, ActionType
from mantis_agent.brain_fara import (
    DEFAULT_INPUT_SIZE,
    FARA_TOOLS,
    FaraBrain,
    InferenceResult,
    _coerce_json,
    _extract_predicted_outcome,
    _resize_for_model,
    _resolve_input_size,
    _scale_coords,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _fake_response(content: str = "", tool_call: dict | None = None) -> dict:
    """Minimal OpenAI-style response shape ``_parse_response`` expects."""
    message: dict = {"content": content, "tool_calls": []}
    if tool_call is not None:
        message["tool_calls"] = [tool_call]
    return {
        "choices": [{"message": message}],
        "usage": {"total_tokens": 0},
    }


def _fara_tool_call(action: str, **fields) -> dict:
    """Build an OpenAI ``tool_calls[0]`` entry calling ``computer_use``."""
    import json
    arguments = {"action": action, **fields}
    return {
        "id": "call_0",
        "type": "function",
        "function": {
            "name": "computer_use",
            "arguments": json.dumps(arguments),
        },
    }


# ── Tool schema sanity ─────────────────────────────────────────────────


def test_fara_tools_has_single_computer_use_function() -> None:
    assert len(FARA_TOOLS) == 1
    fn = FARA_TOOLS[0]["function"]
    assert fn["name"] == "computer_use"
    actions = fn["parameters"]["properties"]["action"]["enum"]
    # Every action our adapter recognises must be advertised to the model.
    for required in (
        "left_click", "type", "key", "scroll", "wait",
        "visit_url", "history_back", "web_search",
        "mouse_move", "pause_and_memorize_fact", "terminate",
    ):
        assert required in actions, f"missing tool action {required!r}"


# ── Prompt registration ────────────────────────────────────────────────


def test_fara_system_prompt_registered() -> None:
    from mantis_agent.prompts import FARA_SYSTEM, list_prompts, prompt_version
    assert "fara_system" in list_prompts()
    assert "Fara" in FARA_SYSTEM or "computer_use" in FARA_SYSTEM
    sha = prompt_version("fara_system")
    assert len(sha) == 8


# ── Coordinate scaling ─────────────────────────────────────────────────


def test_scale_coords_passthrough_when_sizes_match() -> None:
    assert _scale_coords(640, 360, (1280, 720), (1280, 720)) == (640, 360)


def test_scale_coords_default_resolution() -> None:
    # Fara default is 1428x896. Midpoint should land at viewport midpoint.
    mw, mh = DEFAULT_INPUT_SIZE
    sx, sy = _scale_coords(mw // 2, mh // 2, DEFAULT_INPUT_SIZE, (1280, 720))
    assert abs(sx - 640) <= 1
    assert abs(sy - 360) <= 1


def test_scale_coords_extremes_clamp_via_round() -> None:
    # (0, 0) → (0, 0); top-right corner maps to (sw-1ish, 0)
    assert _scale_coords(0, 0, (1428, 896), (1280, 720)) == (0, 0)
    sx, sy = _scale_coords(1428, 0, (1428, 896), (1280, 720))
    assert sx == 1280 and sy == 0


def test_resolve_input_size_respects_explicit() -> None:
    assert _resolve_input_size((1024, 768)) == (1024, 768)


def test_resolve_input_size_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANTIS_FARA_INPUT_WH", "1366x768")
    assert _resolve_input_size(None) == (1366, 768)


def test_resolve_input_size_env_garbage_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MANTIS_FARA_INPUT_WH", "not-a-size")
    assert _resolve_input_size(None) == DEFAULT_INPUT_SIZE


# ── Image resize ───────────────────────────────────────────────────────


def test_resize_is_noop_when_already_target() -> None:
    from PIL import Image
    img = Image.new("RGB", (1428, 896), (0, 0, 0))
    out = _resize_for_model(img, (1428, 896))
    assert out is img  # short-circuits


def test_resize_changes_size_when_different() -> None:
    from PIL import Image
    img = Image.new("RGB", (1920, 1080), (255, 255, 255))
    out = _resize_for_model(img, (1428, 896))
    assert out.size == (1428, 896)
    assert out is not img


# ── _coerce_json ───────────────────────────────────────────────────────


def test_coerce_json_strict() -> None:
    assert _coerce_json('{"a": 1}') == {"a": 1}


def test_coerce_json_python_dict_repr() -> None:
    assert _coerce_json("{'a': True, 'b': None}") == {"a": True, "b": None}


def test_coerce_json_unrecoverable_returns_empty() -> None:
    assert _coerce_json("not json at all") == {}


# ── _extract_predicted_outcome ─────────────────────────────────────────


def test_predicted_outcome_basic() -> None:
    text = "do thing\nPredicted: url_changed, title_changed"
    assert _extract_predicted_outcome(text) == "url_changed, title_changed"


def test_predicted_outcome_missing() -> None:
    assert _extract_predicted_outcome("just text") == ""


def test_predicted_outcome_capped_at_240() -> None:
    body = "x" * 500
    assert len(_extract_predicted_outcome(f"Predicted: {body}")) == 240


# ── tool_calls path: each Fara action → Mantis Action ──────────────────


@pytest.fixture
def brain() -> FaraBrain:
    return FaraBrain(
        api_key="dummy",
        screen_size=(1280, 720),
        input_size=(1428, 896),
    )


def _parse_tool(brain: FaraBrain, tool_call: dict) -> Action:
    result = brain._parse_response(
        _fake_response(tool_call=tool_call),
        screen_size=brain.default_screen_size,
    )
    assert isinstance(result, InferenceResult)
    return result.action


def test_left_click_maps_to_click_with_scaled_coords(brain: FaraBrain) -> None:
    action = _parse_tool(
        brain, _fara_tool_call("left_click", coordinate=[714, 448]),
    )
    assert action.action_type == ActionType.CLICK
    # 714/1428 = 0.5 → 640 ; 448/896 = 0.5 → 360
    assert action.params["x"] == 640
    assert action.params["y"] == 360
    assert action.params["button"] == "left"


def test_type_maps_to_type_action(brain: FaraBrain) -> None:
    action = _parse_tool(brain, _fara_tool_call("type", text="hello"))
    assert action.action_type == ActionType.TYPE
    assert action.params == {"text": "hello"}


def test_key_maps_to_key_press(brain: FaraBrain) -> None:
    action = _parse_tool(brain, _fara_tool_call("key", text="Return"))
    assert action.action_type == ActionType.KEY_PRESS
    assert action.params == {"keys": "Return"}


def test_key_list_is_joined(brain: FaraBrain) -> None:
    action = _parse_tool(brain, _fara_tool_call("key", text=["ctrl", "a"]))
    assert action.params == {"keys": "ctrl+a"}


def test_scroll_maps_with_direction_and_amount(brain: FaraBrain) -> None:
    action = _parse_tool(
        brain,
        _fara_tool_call(
            "scroll", coordinate=[714, 448],
            scroll_direction="down", scroll_amount=5,
        ),
    )
    assert action.action_type == ActionType.SCROLL
    assert action.params["direction"] == "down"
    assert action.params["amount"] == 5
    assert action.params["x"] == 640
    assert action.params["y"] == 360


def test_scroll_without_coordinate_still_valid(brain: FaraBrain) -> None:
    action = _parse_tool(
        brain,
        _fara_tool_call("scroll", scroll_direction="up", scroll_amount=2),
    )
    assert action.action_type == ActionType.SCROLL
    assert "x" not in action.params and "y" not in action.params


def test_wait_uses_duration_field(brain: FaraBrain) -> None:
    action = _parse_tool(brain, _fara_tool_call("wait", duration=2.5))
    assert action.action_type == ActionType.WAIT
    assert action.params == {"seconds": 2.5}


def test_visit_url_routes_as_type_with_url_text(brain: FaraBrain) -> None:
    """visit_url piggybacks on the env's TYPE-of-URL auto-navigate path."""
    action = _parse_tool(
        brain, _fara_tool_call("visit_url", url="https://example.com/x"),
    )
    assert action.action_type == ActionType.TYPE
    assert action.params == {"text": "https://example.com/x"}
    assert "visit_url" in action.reasoning


def test_history_back_maps_to_alt_left(brain: FaraBrain) -> None:
    action = _parse_tool(brain, _fara_tool_call("history_back"))
    assert action.action_type == ActionType.KEY_PRESS
    assert action.params["keys"].lower() == "alt+left"


def test_web_search_becomes_google_url_type(brain: FaraBrain) -> None:
    action = _parse_tool(
        brain, _fara_tool_call("web_search", query="hello world"),
    )
    assert action.action_type == ActionType.TYPE
    expected = (
        f"https://www.google.com/search?q={quote_plus('hello world')}"
    )
    assert action.params["text"] == expected


def test_mouse_move_collapses_to_short_wait(brain: FaraBrain) -> None:
    action = _parse_tool(
        brain, _fara_tool_call("mouse_move", coordinate=[1, 1]),
    )
    assert action.action_type == ActionType.WAIT
    assert action.params["seconds"] <= 0.5


def test_pause_and_memorize_fact_collapses_with_note_in_reasoning(
    brain: FaraBrain,
) -> None:
    action = _parse_tool(
        brain, _fara_tool_call("pause_and_memorize_fact", text="phone=555"),
    )
    assert action.action_type == ActionType.WAIT
    assert "phone=555" in action.reasoning


def test_terminate_success_maps_to_done(brain: FaraBrain) -> None:
    action = _parse_tool(
        brain,
        _fara_tool_call("terminate", status="success", summary="ok"),
    )
    assert action.action_type == ActionType.DONE
    assert action.params["success"] is True
    assert action.params["summary"] == "ok"


def test_terminate_failure_maps_to_done_with_false(brain: FaraBrain) -> None:
    action = _parse_tool(
        brain,
        _fara_tool_call("terminate", status="failure", summary="stuck"),
    )
    assert action.action_type == ActionType.DONE
    assert action.params["success"] is False


# ── Regressions vs Holo3: double_click / right_click / drag ────────────


def test_double_click_non_native_still_maps_when_emitted(
    brain: FaraBrain,
) -> None:
    """If a planner adapter or fine-tune emits double_click anyway, do
    the obvious thing rather than dropping the action on the floor."""
    action = _parse_tool(
        brain, _fara_tool_call("double_click", coordinate=[714, 448]),
    )
    assert action.action_type == ActionType.DOUBLE_CLICK
    assert action.params["x"] == 640
    assert action.params["y"] == 360


def test_unknown_action_becomes_wait_with_reasoning(brain: FaraBrain) -> None:
    action = _parse_tool(brain, _fara_tool_call("right_click", coordinate=[1, 1]))
    assert action.action_type == ActionType.WAIT
    assert "right_click" in action.reasoning


# ── Text-fallback path (no tool_calls field) ───────────────────────────


def test_text_fallback_parses_computer_use_call(brain: FaraBrain) -> None:
    text = (
        "I'll click the search box.\n"
        "computer_use({'action': 'left_click', 'coordinate': [714, 448]})\n"
        "Predicted: field_focused"
    )
    result = brain._parse_response(
        _fake_response(content=text),
        screen_size=brain.default_screen_size,
    )
    assert result.action.action_type == ActionType.CLICK
    assert result.action.params["x"] == 640
    assert result.predicted_outcome == "field_focused"


def test_text_fallback_parses_bare_action_dict(brain: FaraBrain) -> None:
    text = '{"action": "type", "text": "hi"}'
    result = brain._parse_response(
        _fake_response(content=text),
        screen_size=brain.default_screen_size,
    )
    assert result.action.action_type == ActionType.TYPE
    assert result.action.params == {"text": "hi"}


def test_unparseable_response_produces_wait(brain: FaraBrain) -> None:
    result = brain._parse_response(
        _fake_response(content="just rambling, no action"),
        screen_size=brain.default_screen_size,
    )
    assert result.action.action_type == ActionType.WAIT


def test_terminate_keyword_in_text(brain: FaraBrain) -> None:
    result = brain._parse_response(
        _fake_response(content='all done\nterminate("success")'),
        screen_size=brain.default_screen_size,
    )
    assert result.action.action_type == ActionType.DONE
    assert result.action.params["success"] is True


# ── InferenceResult shape ──────────────────────────────────────────────


def test_inference_result_carries_predicted_outcome(brain: FaraBrain) -> None:
    result = brain._parse_response(
        _fake_response(
            content="reasoning\nPredicted: url_changed",
            tool_call=_fara_tool_call("left_click", coordinate=[10, 10]),
        ),
        screen_size=brain.default_screen_size,
    )
    assert result.predicted_outcome == "url_changed"


def test_brain_swappable_with_holo3_interface() -> None:
    """think() signature must match Holo3Brain so the runner stays
    brain-agnostic. We don't call it (no network) — just check the
    method exists with the same kwargs."""
    import inspect
    from mantis_agent.brain_holo3 import Holo3Brain

    fara_sig = inspect.signature(FaraBrain.think)
    holo3_sig = inspect.signature(Holo3Brain.think)
    assert list(fara_sig.parameters) == list(holo3_sig.parameters)
