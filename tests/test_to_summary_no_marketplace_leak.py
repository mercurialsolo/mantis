"""Tests for the BoatTrader summary leak fix (HN feedback).

User feedback: "For URL/title tasks, it sometimes returns wrong
lead-style output like ``VIABLE | Year | Make | Model...`` instead
of the requested fields."

Root cause: ``ExtractionResult.to_summary()`` had a "Legacy BoatTrader"
fallback that synthesised ``Year: ... | Make: ... | Model: ...``
even when the result was for a non-marketplace plan. After the #815
validator rip the schema-bound path was correct, but the fallback
kept printing marketplace-shape clauses.

Fix: the fallback now emits ONLY the populated fields and never
invents empty ``Year: <blank>`` clauses. An ExtractionResult with no
schema and no populated fields returns the empty string instead of
``VIABLE | Phone: none``.
"""

from __future__ import annotations

from mantis_agent.extraction import ExtractionResult, ExtractionSchema


# ── No schema + no fields → empty (not boattrader stub) ────────────


def test_empty_result_returns_empty_string():
    """A bare ExtractionResult with no schema and no fields must NOT
    produce ``VIABLE | Phone: none`` — that's the leak the user saw."""
    result = ExtractionResult()
    assert result.to_summary() == ""


def test_empty_result_does_not_say_year_make_model():
    """The textbook reproducer: a non-marketplace extraction that
    didn't bind a schema must not emit marketplace-shaped output."""
    result = ExtractionResult()
    summary = result.to_summary()
    assert "Year" not in summary
    assert "Make" not in summary
    assert "Model" not in summary
    assert "Phone" not in summary  # was leaking ``Phone: none``


# ── Populated raw fields, no schema → only populated emitted ──────


def test_url_only_extracts_url_only():
    """An HN-style extraction that captured just the URL: summary
    should show that URL, not invent Year/Make/Model placeholders."""
    result = ExtractionResult(url="https://news.ycombinator.com/item?id=12345")
    summary = result.to_summary()
    assert "URL:" in summary
    assert "news.ycombinator.com/item?id=12345" in summary
    # No marketplace shape.
    assert "Year" not in summary
    assert "Make" not in summary


def test_seller_only_extracts_seller_only():
    result = ExtractionResult(seller="Joe's Workshop")
    summary = result.to_summary()
    assert "Seller: Joe's Workshop" in summary
    assert "Year" not in summary
    assert "Make" not in summary


def test_year_make_legacy_still_works():
    """Backwards compat: a legacy marketplace caller that does
    populate Year/Make should still get those rendered. We dropped
    the invention of empty clauses, not the rendering of populated
    ones."""
    result = ExtractionResult(year="2020", make="Sea Ray", model="240")
    summary = result.to_summary()
    assert "Year: 2020" in summary
    assert "Make: Sea Ray" in summary
    assert "Model: 240" in summary


# ── Schema-bound path: unchanged contract ─────────────────────────


def test_schema_bound_url_title_emits_url_title():
    """With an HN-shape schema bound to the result, the summary uses
    the schema's field names — no marketplace baggage."""
    schema = ExtractionSchema(
        fields=[
            {"name": "title", "type": "str", "required": True},
            {"name": "story_url", "type": "str", "required": True},
        ],
    )
    result = ExtractionResult(
        _schema=schema,
        extracted_fields={
            "title": "Show HN: My new project",
            "story_url": "https://example.com/hn",
        },
    )
    summary = result.to_summary()
    assert "Title: Show HN: My new project" in summary
    assert "Story Url: https://example.com/hn" in summary
    # Schema-bound MUST NOT inject marketplace clauses either.
    assert "Year" not in summary
    assert "Make" not in summary


def test_phone_clause_still_shows_none_in_schema_path():
    """The ``Phone: none`` placeholder is a marketplace-recipe-only
    contract — the schema path still emits it when the schema
    declares a ``phone`` field, so existing boattrader callers don't
    regress."""
    schema = ExtractionSchema(
        fields=[
            {"name": "phone", "type": "str", "required": False},
            {"name": "url", "type": "str", "required": True},
        ],
    )
    result = ExtractionResult(
        _schema=schema,
        extracted_fields={"url": "https://example.com/listing"},  # no phone
    )
    summary = result.to_summary()
    assert "Phone: none" in summary
    assert "Url: https://example.com/listing" in summary
