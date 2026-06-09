"""Tests for the marketplace-boilerplate rip in extractor prompts (#807).

Pre-this PR, the schema-driven prompt path hardcoded "phone search
priority", "{spam_label} detection", and the "SKIP: sponsored,
advertisement, ..." filter around any schema's field list. For
non-marketplace plans (HN top-5, GitHub issues) this confused Claude
even though the inline `extract` block correctly enforced the right
fields.

Post-rip, marketplace blocks are gated on schema-declared signals:

- Phone block: schema has a field named "phone"
- Spam detection block: schema.spam_indicators non-empty
- SKIP-sponsored block in find-listings: same — spam_indicators
- year/make/title qualifier: schema declares both "year" and "make"
"""

from __future__ import annotations

from mantis_agent.extraction import ExtractionSchema
from mantis_agent.extraction.extractor import ClaudeExtractor


def _extractor_with_schema(schema: ExtractionSchema) -> ClaudeExtractor:
    """Build a ClaudeExtractor without touching the network client."""
    ext = ClaudeExtractor.__new__(ClaudeExtractor)
    ext.schema = schema
    return ext


# ── HN-style schema (no phone, no spam) ─────────────────────────────


def _hn_schema() -> ExtractionSchema:
    return ExtractionSchema(
        entity_name="hn_story",
        fields=[
            {"name": "rank",  "type": "int", "required": True},
            {"name": "title", "type": "str", "required": True},
            {"name": "story_url", "type": "str", "required": False},
            {"name": "points", "type": "int", "required": False},
        ],
        required_fields=["rank", "title"],
    )


def test_hn_schema_extract_prompt_has_no_phone_block():
    ext = _extractor_with_schema(_hn_schema())
    prompt = ext._get_extract_prompt()
    assert "phone" not in prompt.lower(), (
        f"phone boilerplate leaked into HN-shape prompt:\n{prompt}"
    )


def test_hn_schema_multi_extract_has_no_phone_or_spam_blocks():
    ext = _extractor_with_schema(_hn_schema())
    prompt = ext._get_multi_extract_prompt()
    assert "phone" not in prompt.lower()
    assert "spam" not in prompt.lower()
    assert "is_spam" not in prompt
    assert "detection" not in prompt.lower()


def test_hn_schema_find_listings_has_no_marketplace_sponsored_line():
    """The marketplace-specific "{spam_label} inventory" tail should NOT
    appear without spam_indicators. The generic "skip sponsored content"
    guidance is fine — sponsored cards are noise on every listing-style
    page."""
    ext = _extractor_with_schema(_hn_schema())
    prompt = ext._get_find_listings_prompt()
    # No marketplace-flavored tail.
    assert "inventory" not in prompt.lower()
    assert "dealer" not in prompt.lower()
    # The dealer-style spam-label injection should be absent.
    assert "spam" not in prompt.lower()


def test_hn_schema_find_listings_has_no_year_make_qualifier():
    ext = _extractor_with_schema(_hn_schema())
    prompt = ext._get_find_listings_prompt()
    assert "year/make" not in prompt
    # Should still ask for organic results, just not marketplace-shaped.
    assert "organic" in prompt.lower()


# ── Marketplace-style schema (phone + spam configured) ──────────────


def _marketplace_schema() -> ExtractionSchema:
    return ExtractionSchema(
        entity_name="boat listing",
        fields=[
            {"name": "year",  "type": "str", "required": True},
            {"name": "make",  "type": "str", "required": True},
            {"name": "model", "type": "str", "required": False},
            {"name": "price", "type": "str", "required": False},
            {"name": "phone", "type": "str", "required": False},
            {"name": "url",   "type": "str", "required": False},
        ],
        required_fields=["year", "make"],
        spam_indicators=["dealer website", "more from this dealer"],
        spam_label="dealer",
    )


def test_marketplace_schema_extract_keeps_phone_block():
    """Existing recipe behavior must be preserved end-to-end."""
    ext = _extractor_with_schema(_marketplace_schema())
    prompt = ext._get_extract_prompt()
    assert "phone" in prompt.lower()


def test_marketplace_schema_multi_extract_keeps_both_blocks():
    ext = _extractor_with_schema(_marketplace_schema())
    prompt = ext._get_multi_extract_prompt()
    assert "Phone search priority" in prompt
    assert "Dealer detection" in prompt
    assert "is_spam=true" in prompt


def test_marketplace_schema_find_listings_keeps_sponsored_skip():
    ext = _extractor_with_schema(_marketplace_schema())
    prompt = ext._get_find_listings_prompt()
    assert "sponsored" in prompt.lower()
    assert "dealer inventory" in prompt


def test_marketplace_schema_find_listings_has_year_make_qualifier():
    ext = _extractor_with_schema(_marketplace_schema())
    prompt = ext._get_find_listings_prompt()
    assert "year/make" in prompt


# ── Mixed schemas (partial marketplace shape) ───────────────────────


def test_phone_field_alone_enables_phone_block_without_spam():
    """Plan author who wants phone capture but no spam filtering —
    e.g. a contact-info scraper for a niche business directory."""
    schema = ExtractionSchema(
        entity_name="business",
        fields=[
            {"name": "name",  "type": "str", "required": True},
            {"name": "phone", "type": "str", "required": False},
        ],
        required_fields=["name"],
        # NO spam_indicators
    )
    ext = _extractor_with_schema(schema)
    prompt = ext._get_multi_extract_prompt()
    assert "Phone search priority" in prompt
    assert "is_spam" not in prompt


def test_spam_indicators_alone_enables_spam_block_without_phone():
    """Reverse: a plan that wants spam filtering but doesn't care
    about phones — e.g. a content-scraping plan that should skip
    sponsored articles."""
    schema = ExtractionSchema(
        entity_name="article",
        fields=[
            {"name": "headline", "type": "str", "required": True},
            {"name": "author", "type": "str", "required": False},
        ],
        required_fields=["headline"],
        spam_indicators=["sponsored", "promoted"],
        spam_label="sponsored",
    )
    ext = _extractor_with_schema(schema)
    prompt = ext._get_multi_extract_prompt()
    assert "is_spam=true" in prompt
    assert "Phone search priority" not in prompt


# ── Schema-driven invariants ────────────────────────────────────────


def test_extract_prompt_always_includes_entity_name():
    """Regardless of marketplace shape, the prompt should name the
    entity so Claude has context."""
    for schema in (_hn_schema(), _marketplace_schema()):
        ext = _extractor_with_schema(schema)
        assert schema.entity_name in ext._get_extract_prompt()
        assert schema.entity_name in ext._get_multi_extract_prompt()


def test_extract_prompt_always_includes_field_descriptions():
    """The field list is the load-bearing part — must always be
    present, never gated."""
    schema = _hn_schema()
    ext = _extractor_with_schema(schema)
    prompt = ext._get_multi_extract_prompt()
    for field in schema.fields:
        assert field["name"] in prompt


def test_no_schema_falls_through_to_legacy_prompts():
    """When no schema is bound, the prompts return the historical
    string verbatim. Tests guard against accidentally breaking the
    pre-#785 legacy path."""
    from mantis_agent.extraction.extractor import (
        EXTRACT_MULTI_SCREENSHOT_PROMPT,
        EXTRACT_PROMPT,
    )

    ext = ClaudeExtractor.__new__(ClaudeExtractor)
    ext.schema = None
    assert ext._get_extract_prompt() == EXTRACT_PROMPT
    assert ext._get_multi_extract_prompt() == EXTRACT_MULTI_SCREENSHOT_PROMPT


# ── _schema_has_field helper ────────────────────────────────────────


def test_schema_has_field_case_insensitive():
    ext = _extractor_with_schema(_marketplace_schema())
    assert ext._schema_has_field("phone")
    assert ext._schema_has_field("PHONE")
    assert ext._schema_has_field("Phone")


def test_schema_has_field_returns_false_on_unknown():
    ext = _extractor_with_schema(_hn_schema())
    assert not ext._schema_has_field("phone")
    assert not ext._schema_has_field("nonexistent")


def test_schema_has_field_handles_no_schema():
    ext = ClaudeExtractor.__new__(ClaudeExtractor)
    ext.schema = None
    assert not ext._schema_has_field("phone")
