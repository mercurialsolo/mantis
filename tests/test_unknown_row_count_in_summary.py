"""Tests for ``unknown_placeholder_row_count`` in run summary (#841).

User feedback: ``<UNKNOWN>``-filled rows can survive into
``extracted_rows.json``. Operators see them as "viable rows" until
they grep. Surface the count so the summary tells the truth about
extraction quality even when ``viable`` looks healthy.
"""

from __future__ import annotations

from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.server_utils import _collect_extracted_rows


def _make_step(rows: list[dict]) -> StepResult:
    return StepResult(
        step_index=0, intent="x", success=True, extracted_rows=rows,
    )


def _count_unknown_rows(step_results: list) -> int:
    """Replicate the counting logic from build_micro_result so the
    test can exercise it without booting the runner."""
    _UNKNOWN = {"", "unknown", "<unknown>", "none", "n/a", "na",
                "not visible", "not shown", "not available", "tbd"}
    rows_collected, _ = _collect_extracted_rows(step_results)
    unknown_count = 0
    for row in rows_collected:
        non_url = [
            str(v or "").strip().lower()
            for k, v in row.items() if k.lower() not in {"url", "story_url"}
        ]
        if non_url and all(v in _UNKNOWN for v in non_url):
            unknown_count += 1
    return unknown_count


def test_all_unknown_row_counted():
    step = _make_step([
        {"title": "<UNKNOWN>", "author": "none", "points": "", "url": "https://x.com"},
    ])
    assert _count_unknown_rows([step]) == 1


def test_partial_unknown_not_counted():
    """A row with some real data shouldn't count as UNKNOWN, even
    if other fields are empty."""
    step = _make_step([
        {"title": "Real Title", "author": "<UNKNOWN>", "points": "", "url": "https://x.com"},
    ])
    assert _count_unknown_rows([step]) == 0


def test_url_only_row_counted():
    """URL is excluded from the check (a real URL is the load-bearing
    identity). A row with URL set but all other fields placeholder
    is still extraction-thin."""
    step = _make_step([
        {"title": "<UNKNOWN>", "author": "none", "url": "https://example.com/x"},
    ])
    assert _count_unknown_rows([step]) == 1


def test_multiple_rows_mixed_quality():
    step = _make_step([
        {"title": "Good", "author": "alice", "url": "https://x.com/1"},     # real
        {"title": "<UNKNOWN>", "author": "none", "url": "https://x.com/2"}, # placeholder
        {"title": "Also Good", "author": "bob", "url": "https://x.com/3"},  # real
        {"title": "", "author": "", "url": ""},                              # all empty → counted
    ])
    assert _count_unknown_rows([step]) == 2


def test_empty_extracted_rows_returns_zero():
    step = _make_step([])
    assert _count_unknown_rows([step]) == 0


def test_only_url_field_in_row_not_counted():
    """A row that has ONLY the url field (no other columns at all)
    is degenerate — counts as 0 (nothing to compare against)."""
    step = _make_step([{"url": "https://example.com"}])
    assert _count_unknown_rows([step]) == 0


def test_unknown_placeholders_are_case_insensitive():
    step = _make_step([
        {"title": "UNKNOWN", "url": "x"},
        {"title": "Unknown", "url": "x"},
        {"title": "<unknown>", "url": "x"},
    ])
    assert _count_unknown_rows([step]) == 3


def test_whitespace_tolerated():
    step = _make_step([
        {"title": "  ", "author": "    ", "url": "x"},
    ])
    assert _count_unknown_rows([step]) == 1
