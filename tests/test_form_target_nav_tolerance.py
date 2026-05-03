"""Tests for `find_form_target` nav-label tolerance.

Surfaced by the staffcrm verify (run 20260503_113611_865a9fe4): after a
clean login the runner asked Claude to click a button labelled "Leads".
The CRM renders the link as "LEADS (107)" — uppercase + count badge.
The vision call returned not_found because the prompt only said "case,
punctuation, and small word variations are fine" without explicit
guidance on count badges, leading icons, or qualifier prefixes.

This module verifies the prompt now teaches Claude that a target_label
of "Leads" matches "LEADS (107)", "All Leads", "⚡ Leads", etc. The
test asserts on the prompt string content rather than a live API call —
behavior validation happens via Modal verify reruns.
"""

from __future__ import annotations

import inspect

from mantis_agent.extraction.extractor import ClaudeExtractor


def _grab_form_target_prompt() -> str:
    """Pull the find_form_target source text once. Same approach the runner's
    debug_stem dump uses to capture the actual prompt sent to Claude."""
    return inspect.getsource(ClaudeExtractor.find_form_target)


# ── Examples documented in the prompt ───────────────────────────────────


def test_prompt_mentions_count_badge_match() -> None:
    """The exact failure mode: 'LEADS (107)' must be documented as a
    valid match for 'Leads'."""
    src = _grab_form_target_prompt()
    assert "LEADS (107)" in src or "(107)" in src


def test_prompt_mentions_case_difference_example() -> None:
    src = _grab_form_target_prompt()
    # Source uses escaped quotes inside the f-string; check for either form.
    assert "LEADS" in src
    assert "case difference" in src.lower()


def test_prompt_mentions_qualifier_prefix() -> None:
    """Common prefixes that don't change the destination."""
    src = _grab_form_target_prompt()
    text = src.lower()
    # All four common qualifiers should appear as documented examples.
    for qualifier in ("all", "my", "open", "view"):
        assert qualifier in text


def test_prompt_mentions_leading_icon() -> None:
    src = _grab_form_target_prompt()
    text = src.lower()
    assert "icon" in text or "emoji" in text


def test_prompt_mentions_whitespace_tolerance() -> None:
    src = _grab_form_target_prompt()
    text = src.lower()
    assert "whitespace" in text


# ── Generalization: rule applies beyond "Leads" ─────────────────────────


def test_prompt_generalises_rule_beyond_specific_example() -> None:
    """The prompt must say the rule applies to any target_label, not just
    'Leads'. Otherwise Claude might treat the examples as the entire
    allow-list."""
    src = _grab_form_target_prompt()
    text = src.lower()
    assert "generalise" in text or "generalize" in text or "any target_label" in text


# ── Disambiguation: primary nav over secondary submenu ──────────────────


def test_prompt_prefers_primary_over_secondary_match() -> None:
    """When both 'LEADS (107)' (primary nav) and 'All Leads' (secondary
    submenu) appear on the page, prefer the canonical one. The staffcrm
    page literally has both elements; the one in the main nav is the
    correct destination."""
    src = _grab_form_target_prompt()
    text = src.lower()
    assert "primary" in text and "secondary" in text


# ── Backward compat: existing behaviors still documented ───────────────


def test_prompt_still_documents_action_types() -> None:
    """Prompt rewrite must preserve the click/type/select action enum."""
    src = _grab_form_target_prompt()
    for action in ("click", "type", "select"):
        assert f'"{action}"' in src


def test_prompt_still_returns_not_found_payload() -> None:
    src = _grab_form_target_prompt()
    assert "not_found" in src


def test_prompt_still_requires_center_coordinates() -> None:
    src = _grab_form_target_prompt()
    assert "CENTER" in src or "center coordinates" in src.lower()
