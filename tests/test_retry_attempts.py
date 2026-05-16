"""Shared retry-attempt rendering — roadmap #435 items 2 + 7.

The ``mantis_agent.gym.retry_attempts`` module centralises the line
format + window cap that the three brain adapters use when rendering
the outcome-tagged ``Recent attempts on this sub-goal`` block.

These tests pin the rendered shape so any future formatting change
ripples through one place rather than three.
"""

from __future__ import annotations

import pytest

from mantis_agent.gym.retry_attempts import (
    _PRIOR_ATTEMPTS_WINDOW,
    format_prior_attempt,
    render_attempts_block,
)


# ── format_prior_attempt — outcome verb map ─────────────────────────


def test_format_prior_attempt_known_kinds_get_canonical_verbs() -> None:
    """Each kind the runner stamps via ``_record_failure_for_retry``
    maps to a fixed human verb. Pin them — the brain has been seeing
    these strings since #440 and a phrasing change would alter the
    model's response patterns."""
    base = {"x": 10, "y": 20, "label": "X", "matched_label": "X"}
    out_nsc = format_prior_attempt({**base, "kind": "no_state_change"})
    assert "no observable state change" in out_nsc

    out_wt = format_prior_attempt({**base, "kind": "wrong_target"})
    assert "wrong target" in out_wt

    out_ble = format_prior_attempt({**base, "kind": "brain_loop_exhausted"})
    assert "brain ran out of moves" in out_ble

    out_sm = format_prior_attempt({**base, "kind": "selector_miss"})
    assert "didn't hit an interactive element" in out_sm


def test_format_prior_attempt_unknown_kind_falls_back() -> None:
    """A failure class we haven't enumerated still renders something
    legible — generic ``failed (<kind>)`` keeps the line useful and
    the model can still skip the listed coordinates."""
    out = format_prior_attempt({
        "x": 1, "y": 2, "label": "X",
        "kind": "novel_class_we_added_later",
    })
    assert "novel_class_we_added_later" in out
    assert "failed" in out


def test_format_prior_attempt_handles_missing_fields() -> None:
    """A partial record (no matched_label, no reason) still produces
    a sensible line rather than crashing or rendering ``None``."""
    out = format_prior_attempt({"x": 5, "y": 6, "kind": "no_state_change"})
    assert "(5, 6)" in out
    assert "no observable state change" in out
    # No stringified None / Mock / etc in the output.
    assert "None" not in out


def test_format_prior_attempt_includes_matched_label_when_present() -> None:
    """``matched_label`` (what vision actually targeted) is more
    informative than ``label`` (what the plan asked for) — prefer it
    in the rendered line."""
    out = format_prior_attempt({
        "x": 10, "y": 20,
        "label": "Apply",
        "matched_label": "Apply Filter",
        "kind": "wrong_target",
    })
    assert 'targeting "Apply Filter"' in out


def test_format_prior_attempt_falls_back_to_label_when_matched_missing() -> None:
    """If only ``label`` is set, render that — covers the early-failure
    case where the runner hasn't recorded what vision actually picked."""
    out = format_prior_attempt({
        "x": 10, "y": 20, "label": "Apply", "kind": "no_state_change",
    })
    assert 'targeting "Apply"' in out


def test_format_prior_attempt_clips_long_reason() -> None:
    """A 1000-char reason from agentic recovery would blow the prompt
    if rendered raw — the helper clips to 120 chars."""
    long_reason = "x" * 500
    out = format_prior_attempt({
        "x": 1, "y": 1, "kind": "no_state_change", "reason": long_reason,
    })
    # 120 chars + "; " prefix → at most ~125 chars of x's in the line.
    assert "x" * 121 not in out


# ── render_attempts_block — window cap + empty handling ─────────────


def test_render_attempts_block_returns_empty_for_none() -> None:
    """No prior failures → empty string. Caller's ``if block:`` splice
    stays one-line."""
    assert render_attempts_block(None) == ""
    assert render_attempts_block([]) == ""


def test_render_attempts_block_returns_empty_when_all_entries_malformed() -> None:
    """A list of non-dict entries (e.g. a MagicMock auto-attribute
    leaking in) renders empty rather than crashing."""
    assert render_attempts_block([None, "string", 42]) == ""  # type: ignore[list-item]


def test_render_attempts_block_caps_window() -> None:
    """Older entries get dropped — only the most-recent ``window``
    records are surfaced. The brain doesn't benefit from seeing every
    miss in a long retry chain."""
    attempts = [
        {"x": i, "y": i, "kind": "no_state_change"}
        for i in range(10)
    ]
    block = render_attempts_block(attempts, window=3)
    # Last three (i=7, 8, 9) appear; older ones don't.
    assert "(7, 7)" in block
    assert "(8, 8)" in block
    assert "(9, 9)" in block
    assert "(0, 0)" not in block
    assert "(5, 5)" not in block


def test_render_attempts_block_default_window_is_three() -> None:
    """Default window is 3 to keep prompts tight. Verify the constant
    matches and the rendered shape respects it without an explicit
    kwarg."""
    assert _PRIOR_ATTEMPTS_WINDOW == 3
    attempts = [
        {"x": i, "y": i, "kind": "no_state_change"}
        for i in range(5)
    ]
    block = render_attempts_block(attempts)
    assert "(2, 2)" in block  # last 3 = indices 2, 3, 4
    assert "(3, 3)" in block
    assert "(4, 4)" in block
    assert "(0, 0)" not in block
    assert "(1, 1)" not in block


def test_render_attempts_block_includes_header() -> None:
    """The block leads with the configurable header so the brain
    knows what to do with the lines that follow. Defaults to a
    'Recent attempts on this sub-goal' framing."""
    attempts = [{"x": 1, "y": 1, "kind": "no_state_change"}]
    block = render_attempts_block(attempts)
    assert "Recent attempts on this sub-goal" in block
    assert "do NOT repeat" in block


def test_render_attempts_block_indents_each_entry() -> None:
    """Lines are indented + numbered so the prompt reads naturally
    when the model glances at it — match the format the Holo3 step
    handler's scoped task uses elsewhere."""
    attempts = [
        {"x": 1, "y": 1, "kind": "no_state_change"},
        {"x": 2, "y": 2, "kind": "wrong_target"},
    ]
    block = render_attempts_block(attempts)
    assert "  1. " in block
    assert "  2. " in block


def test_render_attempts_block_filters_non_dict_then_caps() -> None:
    """Non-dict entries are stripped BEFORE the window cap kicks in —
    so two valid records survive even if the list also has trash."""
    attempts = [
        {"x": 1, "y": 1, "kind": "no_state_change"},
        "garbage",  # type: ignore[list-item]
        None,
        {"x": 2, "y": 2, "kind": "wrong_target"},
    ]
    block = render_attempts_block(attempts, window=3)
    assert "(1, 1)" in block
    assert "(2, 2)" in block
