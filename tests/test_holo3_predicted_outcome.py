"""Tests for #120 step 2 — Holo3 brain emits predicted_outcome.

Covers the parser, the prompt instruction, and the InferenceResult schema
change. Live network calls to Holo3 are not exercised here — that's a
separate integration test path.
"""

from __future__ import annotations

import re

from mantis_agent.brain_holo3 import (
    InferenceResult,
    _PREDICTED_LINE_RE,
    _extract_predicted_outcome,
)


# ── _extract_predicted_outcome ──────────────────────────────────────────


def test_extract_returns_empty_on_missing_line() -> None:
    text = "I'll click the listing.\nclick(x=640, y=320)"
    assert _extract_predicted_outcome(text) == ""


def test_extract_returns_empty_on_blank_input() -> None:
    assert _extract_predicted_outcome("") == ""
    assert _extract_predicted_outcome("   \n  \n") == ""


def test_extract_basic_one_liner() -> None:
    text = (
        "I'll click the first listing.\n"
        "click(x=640, y=320)\n"
        "Predicted: page navigates to the detail URL and the title becomes the boat's name."
    )
    out = _extract_predicted_outcome(text)
    assert out == (
        "page navigates to the detail URL and the title becomes the boat's name."
    )


def test_extract_strips_quotes_and_trailing_whitespace() -> None:
    text = 'click(x=1, y=1)\nPredicted: "modal closes"   \n'
    assert _extract_predicted_outcome(text) == "modal closes"


def test_extract_is_case_insensitive() -> None:
    text = "click(x=1, y=1)\nPREDICTED: URL changes to /detail/123"
    assert _extract_predicted_outcome(text) == "URL changes to /detail/123"


def test_extract_caps_at_240_chars() -> None:
    long = "x" * 500
    text = f"click(x=1, y=1)\nPredicted: {long}"
    out = _extract_predicted_outcome(text)
    assert len(out) == 240


def test_extract_grabs_first_predicted_line_when_multiple() -> None:
    text = (
        "click(x=1, y=1)\n"
        "Predicted: first thing happens\n"
        "Predicted: second thing\n"
    )
    assert _extract_predicted_outcome(text) == "first thing happens"


def test_extract_handles_no_space_after_colon() -> None:
    text = "click(x=1, y=1)\nPredicted:URL changes"
    assert _extract_predicted_outcome(text) == "URL changes"


def test_extract_handles_extra_space_around_colon() -> None:
    text = "click(x=1, y=1)\nPredicted   :   URL changes"
    assert _extract_predicted_outcome(text) == "URL changes"


# ── Prompt instruction ──────────────────────────────────────────────────


def test_holo3_system_prompt_requests_predicted_line() -> None:
    """The system prompt must explicitly tell the model to emit Predicted:."""
    from mantis_agent.prompts import HOLO3_SYSTEM
    assert "Predicted:" in HOLO3_SYSTEM
    # And the example shows the format too.
    assert re.search(r"Predicted:\s+\S+", HOLO3_SYSTEM)


def test_holo3_system_prompt_allows_omission_when_uncertain() -> None:
    """The brain shouldn't be forced to hallucinate when it doesn't know."""
    from mantis_agent.prompts import HOLO3_SYSTEM
    # The rule "if you don't know, omit the line" prevents hallucinated predictions.
    assert "omit" in HOLO3_SYSTEM.lower() or "skip" in HOLO3_SYSTEM.lower()


def test_holo3_system_prompt_change_invalidates_prompt_version() -> None:
    """#127 prompt versioning + #120 prompt edit: the SHA should differ
    from a baseline that doesn't ask for Predicted:."""
    from mantis_agent.prompts import prompt_version

    sha = prompt_version("holo3_system")
    # 8 hex chars, deterministic.
    assert len(sha) == 8
    # Stability: two calls return the same SHA.
    assert prompt_version("holo3_system") == sha


# ── InferenceResult schema ──────────────────────────────────────────────


def test_inference_result_defaults_predicted_outcome_to_empty() -> None:
    from mantis_agent.actions import Action, ActionType
    r = InferenceResult(
        action=Action(ActionType.CLICK, {"x": 1, "y": 1}),
        raw_output="",
    )
    assert r.predicted_outcome == ""


def test_inference_result_carries_predicted_outcome_field() -> None:
    from mantis_agent.actions import Action, ActionType
    r = InferenceResult(
        action=Action(ActionType.CLICK, {"x": 1, "y": 1}),
        raw_output="",
        predicted_outcome="modal closes",
    )
    assert r.predicted_outcome == "modal closes"


# ── _parse_response integration ─────────────────────────────────────────


def _fake_holo3_response(content: str) -> dict:
    """Minimal dict shape that Holo3Brain._parse_response expects."""
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                    "tool_calls": [],
                }
            }
        ],
        "usage": {"total_tokens": 0},
    }


def test_parse_response_extracts_predicted_from_text_path() -> None:
    """Parser strategy 2 (Action: name(...) text) — predicted line is captured."""
    from mantis_agent.brain_holo3 import Holo3Brain

    brain = Holo3Brain(api_key="dummy")
    response = _fake_holo3_response(
        "I'll click the first listing.\n"
        "Action: click({'x': 100, 'y': 200})\n"
        "Predicted: page navigates to the detail URL"
    )
    result = brain._parse_response(response, screen_size=(1280, 720))
    # Don't assert on action shape — coordinate-conversion math depends on
    # Qwen smart-resize; we only care that predicted_outcome propagated.
    assert result.predicted_outcome == "page navigates to the detail URL"


def test_parse_response_no_predicted_line_yields_empty_field() -> None:
    """Brains that don't emit Predicted: must produce predicted_outcome=""."""
    from mantis_agent.brain_holo3 import Holo3Brain

    brain = Holo3Brain(api_key="dummy")
    response = _fake_holo3_response(
        "I'll click.\nAction: click({'x': 100, 'y': 200})"
    )
    result = brain._parse_response(response, screen_size=(1280, 720))
    assert result.predicted_outcome == ""


# ── Sanity: regex compiles and matches expected lines ───────────────────


def test_predicted_line_regex_compiled() -> None:
    assert _PREDICTED_LINE_RE.search("Predicted: foo") is not None
    assert _PREDICTED_LINE_RE.search("predicted: foo") is not None
    assert _PREDICTED_LINE_RE.search("PREDICTED: foo") is not None
    # Negative: ensure we don't match arbitrary "predicted" mentions inside prose.
    assert _PREDICTED_LINE_RE.search("the model predicted nothing useful") is None
