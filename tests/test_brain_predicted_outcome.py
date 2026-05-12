"""Tests for #291 — brain_claude and brain_opencua emit predicted_outcome.

Mirrors the existing test_holo3_predicted_outcome.py shape so the three
brains (Holo3, Claude, OpenCUA) all have the same coverage contract.
"""

from __future__ import annotations

from mantis_agent.actions import Action, ActionType
from mantis_agent.brain_claude import InferenceResult as ClaudeIR
from mantis_agent.brain_opencua import InferenceResult as OpenCUAIR


# ── InferenceResult schema ──────────────────────────────────────────────


def test_claude_inference_result_defaults_predicted_outcome_empty() -> None:
    r = ClaudeIR(action=Action(ActionType.CLICK, {"x": 1, "y": 1}), raw_output="")
    assert r.predicted_outcome == ""


def test_claude_inference_result_carries_predicted_outcome() -> None:
    r = ClaudeIR(
        action=Action(ActionType.CLICK, {"x": 1, "y": 1}),
        raw_output="",
        predicted_outcome='{"expected": ["url_changed"]}',
    )
    assert r.predicted_outcome == '{"expected": ["url_changed"]}'


def test_opencua_inference_result_defaults_predicted_outcome_empty() -> None:
    r = OpenCUAIR(action=Action(ActionType.CLICK, {"x": 1, "y": 1}), raw_output="")
    assert r.predicted_outcome == ""


def test_opencua_inference_result_carries_predicted_outcome() -> None:
    r = OpenCUAIR(
        action=Action(ActionType.CLICK, {"x": 1, "y": 1}),
        raw_output="",
        predicted_outcome="Predicted: url_changed",
    )
    assert r.predicted_outcome == "Predicted: url_changed"


# ── System prompts mention the predicate contract ──────────────────────


def test_claude_system_prompt_documents_predicate_grammar() -> None:
    from mantis_agent.prompts import load_prompt
    p = load_prompt("claude_system")
    assert '"expected"' in p
    assert "url_contains" in p
    assert "url_changed" in p
    assert "field_focused" in p
    # Stay-silent rule prevents hallucinated predictions.
    assert "omit" in p.lower() or "skip" in p.lower()


def test_opencua_system_prompt_documents_predicate_grammar() -> None:
    from mantis_agent.prompts import load_prompt
    p = load_prompt("opencua_system")
    assert '"expected"' in p
    assert "url_contains" in p
    assert "field_focused" in p
    assert "omit" in p.lower() or "skip" in p.lower()


# ── extract_predicted_outcome shared helper ─────────────────────────────


def test_extract_predicted_outcome_prefers_json_block() -> None:
    from mantis_agent.gym.predicates import extract_predicted_outcome
    text = (
        "Clicking the button.\n"
        '{"expected": ["url_contains:/checkout"]}\n'
        "Predicted: page navigates"
    )
    out = extract_predicted_outcome(text)
    assert out.startswith('{"expected"')
    assert "url_contains" in out


def test_extract_predicted_outcome_falls_back_to_predicted_line() -> None:
    from mantis_agent.gym.predicates import extract_predicted_outcome
    text = "click(x=1, y=1)\nPredicted: url_changed, title_changed"
    out = extract_predicted_outcome(text)
    assert out == "url_changed, title_changed"


def test_extract_predicted_outcome_returns_empty_on_no_signal() -> None:
    from mantis_agent.gym.predicates import extract_predicted_outcome
    assert extract_predicted_outcome("just thinking") == ""
    assert extract_predicted_outcome("") == ""


def test_extract_predicted_outcome_caps_long_input() -> None:
    from mantis_agent.gym.predicates import (
        _PREDICTED_OUTCOME_MAX_CHARS,
        extract_predicted_outcome,
    )
    text = "Predicted: " + "x" * 5000
    assert len(extract_predicted_outcome(text)) <= _PREDICTED_OUTCOME_MAX_CHARS


# ── Claude _parse_response captures predicted_outcome ──────────────────


def test_claude_parse_response_captures_json_block() -> None:
    from mantis_agent.brain_claude import ClaudeBrain
    brain = ClaudeBrain(api_key="dummy")
    data = {
        "content": [
            {"type": "text", "text": (
                'I see the checkout link. {"expected": ["url_contains:/checkout"]}'
            )},
            {"type": "tool_use", "name": "computer", "input": {
                "action": "left_click", "coordinate": [100, 200],
            }},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    result = brain._parse_response(data)
    assert "url_contains" in result.predicted_outcome


def test_claude_parse_response_no_prediction_yields_empty() -> None:
    from mantis_agent.brain_claude import ClaudeBrain
    brain = ClaudeBrain(api_key="dummy")
    data = {
        "content": [
            {"type": "text", "text": "I'll click."},
            {"type": "tool_use", "name": "computer", "input": {
                "action": "left_click", "coordinate": [1, 2],
            }},
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    result = brain._parse_response(data)
    assert result.predicted_outcome == ""


# ── OpenCUA _parse_response captures predicted_outcome ─────────────────


def test_opencua_parse_response_captures_predicted_line() -> None:
    from mantis_agent.brain_opencua import OpenCUABrain
    brain = OpenCUABrain(base_url="http://localhost:8000/v1")
    data = {
        "choices": [{"message": {"content": (
            "I'll click.\n"
            "pyautogui.click(100, 200)\n"
            "Predicted: url_changed"
        )}}],
        "usage": {"total_tokens": 0},
    }
    result = brain._parse_response(data, screen_size=(1280, 720))
    assert result.predicted_outcome == "url_changed"


def test_opencua_parse_response_captures_json_block() -> None:
    from mantis_agent.brain_opencua import OpenCUABrain
    brain = OpenCUABrain(base_url="http://localhost:8000/v1")
    data = {
        "choices": [{"message": {"content": (
            'I will click. pyautogui.click(100, 200) {"expected": ["url_changed"]}'
        )}}],
        "usage": {"total_tokens": 0},
    }
    result = brain._parse_response(data, screen_size=(1280, 720))
    assert "url_changed" in result.predicted_outcome
    assert '"expected"' in result.predicted_outcome
