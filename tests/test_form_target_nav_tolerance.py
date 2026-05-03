"""Tests for the find_form_target prompt — LLM-driven semantic matching.

The earlier branch listed concrete examples ("LEADS (107)", "All Leads",
"⚡ Leads") in the prompt as a teaching aid for case/badge/icon
tolerance. Per the project's no-hardcoded-semantics directive, the
prompt is now principled: it tells the vision-language model to "match
by destination, not by exact text" and trusts the model's reasoning to
handle decorations on its own.

These tests verify the prompt carries the right principles without any
hardcoded example list.
"""

from __future__ import annotations

import inspect

from mantis_agent.extraction.extractor import ClaudeExtractor


def _grab_form_target_prompt() -> str:
    """Pull the find_form_target source text. The actual prompt is
    constructed at call time — inspecting source captures the templates."""
    return inspect.getsource(ClaudeExtractor.find_form_target)


# ── Principled instruction language ────────────────────────────────────


def test_prompt_says_match_by_destination_not_text() -> None:
    """The core principle: the LLM should reason about what a user is
    *trying to reach*, not parrot exact text matches."""
    src = _grab_form_target_prompt()
    text = src.lower()
    assert "destination" in text
    # Either phrasing makes the principle explicit.
    assert "exact text" in text or "not by exact" in text or "by destination" in text


def test_prompt_acknowledges_visual_decorations_generally() -> None:
    """The prompt should tell the LLM that on-screen renderings carry
    visual decorations the source plan didn't mention — without listing
    specific decorations as a substitute for general reasoning."""
    src = _grab_form_target_prompt()
    text = src.lower()
    # General categories the LLM should recognise on its own.
    assert "decoration" in text or "decorations" in text
    assert "visual" in text


def test_prompt_invokes_vision_language_capability() -> None:
    """A reminder that the model is multimodal — read visible text in
    context, decide if the underlying destination matches."""
    src = _grab_form_target_prompt()
    text = src.lower()
    assert "vision" in text or "visible text" in text


def test_prompt_uses_layout_for_disambiguation() -> None:
    """When multiple candidates match, layout (primary nav vs submenu,
    sidebar vs header) is the disambiguator — not a hardcoded preference
    list."""
    src = _grab_form_target_prompt()
    text = src.lower()
    assert "layout" in text or "primary" in text
    assert "submenu" in text or "sidebar" in text or "header" in text


# ── No hardcoded example lists ─────────────────────────────────────────


def test_prompt_does_not_list_specific_label_examples() -> None:
    """Regression guard: contributors might be tempted to add 'help'
    examples again. The principled instruction must stand on its own."""
    src = _grab_form_target_prompt()
    forbidden_decorations = [
        "LEADS (107)",  # the original count-badge example
        '"All Leads"',  # qualifier prefix example
        "⚡",          # leading-icon example with emoji
    ]
    for token in forbidden_decorations:
        assert token not in src, (
            f"Prompt should not hardcode the example {token!r} — let the "
            f"LLM reason about decorations from the principled instruction."
        )


def test_prompt_does_not_enumerate_qualifier_prefixes() -> None:
    """The earlier prompt listed 'All', 'My', 'Open', 'View' as known
    qualifier prefixes. That's the kind of hard-coded semantic list the
    no-regex directive forbids."""
    src = _grab_form_target_prompt()
    # The phrasing "All", "My", "Open", "View" appearing as a quoted
    # enumeration would indicate a regression.
    bad_pattern = '"All", "My"'
    assert bad_pattern not in src


# ── Backward compat: action enum + return shape preserved ──────────────


def test_prompt_still_documents_action_types() -> None:
    src = _grab_form_target_prompt()
    for action in ("click", "type", "select"):
        assert f'"{action}"' in src


def test_prompt_still_returns_not_found_payload() -> None:
    src = _grab_form_target_prompt()
    assert "not_found" in src


def test_prompt_still_requires_center_coordinates() -> None:
    src = _grab_form_target_prompt()
    assert "CENTER" in src or "center coordinates" in src.lower()


# ── Not-found behavior ─────────────────────────────────────────────────


def test_prompt_asks_for_descriptive_not_found_response() -> None:
    """When nothing matches, the LLM should describe what's actually on
    the page — that's how the runner learns whether to retry, scroll,
    or surface a planning error."""
    src = _grab_form_target_prompt()
    text = src.lower()
    assert "not_found" in text
    # A "describe what you see instead" instruction.
    assert "describe" in text or "instead" in text
