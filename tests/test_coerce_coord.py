"""Tests for the defensive coordinate coercion in find_form_target.

Surfaced by the staff-crm priority-field rerun on Modal — Claude's
``find_form_target`` tool_use occasionally returns coordinates as
strings with stray whitespace / trailing commas (``"x": "296, "``
observed) even though the input_schema requires
``"type": "integer"``. The previous code did ``int(parsed.get("x", 0))``
which raises ``ValueError`` and crashes the entire run. The fix
treats malformed coords as a not-found response so the runner falls
through to its retry / escalation / recovery path instead of halting
with an unhandled exception.
"""

from __future__ import annotations

from mantis_agent.extraction.extractor import _coerce_coord


def test_coerce_int_passes_through() -> None:
    assert _coerce_coord(540) == 540


def test_coerce_zero_returns_zero() -> None:
    """Zero is a meaningful coordinate (even if the caller treats it
    specially). Coerce as the int it is."""
    assert _coerce_coord(0) == 0


def test_coerce_negative_int() -> None:
    """Negative coordinates can happen on overlapping coordinate
    systems — return as-is, let the caller decide."""
    assert _coerce_coord(-5) == -5


def test_coerce_clean_string_int() -> None:
    assert _coerce_coord("540") == 540


def test_coerce_string_with_trailing_comma() -> None:
    """The canonical failure mode — Claude emitted ``"x": "296, "``
    on a long-prompt retry. Strip the trailing comma + whitespace."""
    assert _coerce_coord("296, ") == 296


def test_coerce_string_with_leading_whitespace() -> None:
    assert _coerce_coord("  540  ") == 540


def test_coerce_string_with_trailing_semicolon() -> None:
    """Defensive against several stray-punctuation modes."""
    assert _coerce_coord("100;") == 100


def test_coerce_string_with_trailing_bracket() -> None:
    """Sometimes the model copies a fragment of JSON syntax."""
    assert _coerce_coord("100]") == 100


def test_coerce_float_string_round_trips() -> None:
    """Some calls produce floats like ``"540.0"`` — accept and
    truncate to int rather than rejecting."""
    assert _coerce_coord("540.5") == 540


def test_coerce_float_value() -> None:
    assert _coerce_coord(540.5) == 540


def test_coerce_returns_none_for_non_numeric_string() -> None:
    """When the value is genuinely unparseable, return None so the
    caller can route to not-found rather than crash."""
    assert _coerce_coord("not a number") is None


def test_coerce_returns_none_for_empty_string() -> None:
    assert _coerce_coord("") is None
    assert _coerce_coord("   ") is None


def test_coerce_returns_none_for_none() -> None:
    assert _coerce_coord(None) is None


def test_coerce_returns_none_for_bool() -> None:
    """``bool`` is a subclass of ``int`` in Python — an LLM that emitted
    ``true`` / ``false`` for a coordinate is clearly malformed; reject
    explicitly so True doesn't smuggle in as 1."""
    assert _coerce_coord(True) is None
    assert _coerce_coord(False) is None


def test_coerce_returns_none_for_list() -> None:
    """If the LLM emitted ``[100, 200]`` as a coordinate, we don't
    have any reasonable interpretation — reject."""
    assert _coerce_coord([100, 200]) is None


def test_coerce_returns_none_for_dict() -> None:
    assert _coerce_coord({"x": 100}) is None
