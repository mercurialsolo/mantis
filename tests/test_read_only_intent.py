"""Tests for read-only intent detection + validation (#831).

User feedback on HN extraction: "Even with instructions like 'do not
click story links / More', it still clicked/paginated and returned
URLs like ``news.ycombinator.com/?p=2``."

The fix: detect read-only intent in ``plan_text`` and (a) prepend a
hard constraint to the decomposer prompt, (b) reject decomposed plans
that still contain forbidden step types.
"""

from __future__ import annotations

import pytest

from mantis_agent.read_only_intent import (
    READ_ONLY_PROMPT_CONSTRAINT,
    is_read_only,
    validate_read_only_plan,
)


# ── Positive detection ────────────────────────────────────────────


@pytest.mark.parametrize("text", [
    "do not click any links",
    "Don't click on the More link",
    "Do not navigate away from the page",
    "Don't follow any links",
    "Read-only — just extract titles",
    "read only please",
    "stay on this page and read the titles",
    "Remain on the page",
    "Just read what's visible",
    "Read and report the headings",
    "Extract the visible content without clicking",
    "Pull the titles without navigating",
    "Show me the top stories visible on the current page",
    "List items shown on the page",
    # The exact HN feedback phrasing:
    "Open https://news.ycombinator.com/ and read the titles. Do not click any story links or More.",
])
def test_positive_read_only_phrasings(text):
    assert is_read_only(text), f"should detect read-only: {text!r}"


# ── Negative detection ────────────────────────────────────────────


@pytest.mark.parametrize("text", [
    "click into each story",
    "go to the next page and grab the titles",
    "navigate to /newest and paginate through 3 pages",
    "open each result in turn",
    "follow each link to read the full article",
    "",
])
def test_negative_phrasings_not_detected(text):
    assert not is_read_only(text), f"should not detect: {text!r}"


def test_empty_input_returns_false():
    assert is_read_only("") is False
    assert is_read_only("    ") is False


# ── Validation ───────────────────────────────────────────────────


def test_validate_passes_for_navigate_extract_plan():
    steps = [
        {"type": "navigate", "intent": "Open HN"},
        {"type": "extract_data", "intent": "Read titles", "claude_only": True},
    ]
    assert validate_read_only_plan(steps) == ""


def test_validate_passes_for_navigate_only():
    steps = [{"type": "navigate", "intent": "Open HN"}]
    assert validate_read_only_plan(steps) == ""


def test_validate_fails_when_plan_contains_click():
    steps = [
        {"type": "navigate"},
        {"type": "click", "intent": "Click first story"},
        {"type": "extract_data"},
    ]
    err = validate_read_only_plan(steps)
    assert err != ""
    assert "click" in err.lower()


def test_validate_fails_when_plan_contains_paginate():
    steps = [
        {"type": "navigate"},
        {"type": "paginate"},
        {"type": "extract_data"},
    ]
    err = validate_read_only_plan(steps)
    assert "paginate" in err.lower()


def test_validate_fails_when_plan_contains_loop():
    steps = [
        {"type": "navigate"},
        {"type": "extract_data"},
        {"type": "loop", "loop_target": 1, "loop_count": 5},
    ]
    err = validate_read_only_plan(steps)
    assert "loop" in err.lower()


def test_validate_lists_all_offending_steps():
    """When multiple forbidden steps slip through, the error must
    mention all of them so the operator can see the breadth of the
    leak."""
    steps = [
        {"type": "navigate"},
        {"type": "click"},
        {"type": "paginate"},
        {"type": "loop"},
    ]
    err = validate_read_only_plan(steps)
    assert "click" in err.lower()
    assert "paginate" in err.lower()
    assert "loop" in err.lower()


def test_validate_handles_microintent_objects():
    """Step shape may be a MicroIntent dataclass too, not just a dict."""
    from mantis_agent.plan_decomposer import MicroIntent

    steps = [
        MicroIntent(intent="open", type="navigate"),
        MicroIntent(intent="click x", type="click"),
    ]
    err = validate_read_only_plan(steps)
    assert "click" in err.lower()


def test_validate_handles_empty_steps():
    assert validate_read_only_plan([]) == ""


# ── Prompt constraint shape ──────────────────────────────────────


def test_read_only_constraint_documents_allowed_step_types():
    """The constraint string is sent to Claude; it must clearly enumerate
    what's allowed and what's forbidden so the model has actionable
    instructions, not just a vibe."""
    text = READ_ONLY_PROMPT_CONSTRAINT
    assert "navigate" in text
    assert "extract_data" in text
    assert "click" in text
    assert "paginate" in text
    assert "loop" in text
    assert "READ-ONLY" in text.upper()
