"""Tests for #584 (find_listings exclusion plumbing) + #585 (extract_url
URL-pattern validation).

Both defend against the click-landed-on-marketing-CTA failure observed
on boattrader runs:

    [1] VIABLE | ... | URL: boattrader.com/boat-loans/

#584 prevents the wrong card from being picked AT find_all_listings.
#585 catches the click that got through anyway AT extract_url.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from mantis_agent.extraction.extractor import ClaudeExtractor
from mantis_agent.extraction.schema import ExtractionSchema
from mantis_agent.recipes.marketplace_listings.schema import SCHEMA


# ── #584: find_listings prompt enumerates exclusions ───────────────


def _extractor_with_schema(schema: ExtractionSchema) -> ClaudeExtractor:
    """Bypass __init__ — we only need the prompt builder + schema."""
    inst = ClaudeExtractor.__new__(ClaudeExtractor)
    inst.schema = schema
    return inst


def test_find_listings_prompt_enumerates_exclusions() -> None:
    prompt = _extractor_with_schema(SCHEMA)._get_find_listings_prompt()
    # All marketplace_listings exclusions must show up in the prompt.
    assert "EXCLUDE these specifically" in prompt
    assert "Get Pre-Qualified" in prompt
    assert "Boat Loans" in prompt
    assert "Insurance" in prompt
    assert "Live Video Tour" in prompt
    assert "Sponsored" in prompt
    assert "Newsletter" in prompt
    assert "Auction" in prompt


def test_find_listings_prompt_omits_block_when_no_exclusions() -> None:
    schema = ExtractionSchema(
        entity_name="boat",
        fields=[{"name": "year", "type": "str"}],
        required_fields=["year"],
        # no listing_card_exclusions
    )
    prompt = _extractor_with_schema(schema)._get_find_listings_prompt()
    assert "EXCLUDE these specifically" not in prompt
    # Generic line still present.
    assert "SKIP: sponsored" in prompt


def test_find_listings_prompt_tightens_organic_criteria() -> None:
    """The 'ONLY include' clause now requires both product identity
    AND price — catches CTAs that have a title-like header but no real
    listing price (e.g. financing card with 'Get Started')."""
    prompt = _extractor_with_schema(SCHEMA)._get_find_listings_prompt()
    assert "product identity" in prompt.lower()
    assert "price" in prompt.lower()


def test_schema_round_trips_listing_card_exclusions() -> None:
    payload = {
        "entity_name": "boat",
        "fields": [{"name": "year", "type": "str"}],
        "required_fields": ["year"],
        "listing_card_exclusions": ["financing CTA", "newsletter promo"],
    }
    schema = ExtractionSchema.from_dict(payload)
    assert schema.listing_card_exclusions == ["financing CTA", "newsletter promo"]


# ── #585: extract_url validates URL against detail_page_pattern ────


def _make_step(section: str = "extraction") -> MagicMock:
    step = MagicMock()
    step.type = "extract_url"
    step.section = section
    step.intent = "Read URL"
    step.gate = False
    step.claude_only = True
    return step


def _make_runner(detail_pattern: str = r"/boat/[\w-]+") -> MagicMock:
    """Build a runner double with just the attrs extract_url's wrong-page
    check reads."""
    from mantis_agent.site_config import SiteConfig
    runner = MagicMock()
    runner.site_config = SiteConfig(
        domain="boattrader.com",
        detail_page_pattern=detail_pattern,
    )
    runner.scanner = MagicMock()
    runner.scanner.is_duplicate = MagicMock(return_value=False)
    runner._current_page = 1
    runner._last_extracted = {}
    runner._last_known_url = ""
    runner.costs = {"claude_extract": 0}
    return runner


def _execute_extract_url(runner: MagicMock, extracted_url: str, section: str = "extraction"):
    """Invoke the extract_url path of ClaudeStepHandler with a mocked
    env + extractor that returns the given URL."""
    from mantis_agent.gym.step_handlers.claude_step import ClaudeStepHandler
    handler = ClaudeStepHandler.__new__(ClaudeStepHandler)
    handler.parent = runner

    env = MagicMock()
    env.screenshot = MagicMock(return_value=MagicMock())

    extractor = MagicMock()
    fake_data = MagicMock()
    fake_data.url = extracted_url
    fake_data.is_dealer = False
    extractor.extract = MagicMock(return_value=fake_data)

    ctx = MagicMock()
    ctx.env = env
    ctx.extractor = extractor
    ctx.dynamic_verifier = MagicMock()
    ctx.extraction_cache = None
    ctx.state = {"index": 3}

    step = _make_step(section=section)
    return handler.execute(step, ctx)


def test_extract_url_rejects_non_matching_url_in_extraction_section() -> None:
    runner = _make_runner()
    result = _execute_extract_url(
        runner, extracted_url="https://www.boattrader.com/boat-loans/",
    )
    assert result.success is False
    assert "WRONG_PAGE" in result.data
    assert result.failure_class == "wrong_target"


def test_extract_url_accepts_matching_url() -> None:
    runner = _make_runner()
    result = _execute_extract_url(
        runner, extracted_url="https://www.boattrader.com/boat/2018-sea-ray-240/",
    )
    assert result.success is True
    assert result.data.startswith("URL:")


def test_extract_url_skips_validation_outside_extraction_section() -> None:
    # A setup-section extract_url (e.g. initial URL read after navigate)
    # should NOT be rejected by the detail-page pattern check — different
    # phase, different expectation.
    runner = _make_runner()
    result = _execute_extract_url(
        runner,
        extracted_url="https://www.boattrader.com/boats/state-fl/",
        section="setup",
    )
    assert result.success is True


def test_extract_url_skips_validation_when_no_pattern_configured() -> None:
    # Plans / recipes without a detail_page_pattern (analysis-stage-only
    # configs) must not break.
    runner = _make_runner(detail_pattern="")
    result = _execute_extract_url(
        runner, extracted_url="https://www.boattrader.com/boat-loans/",
    )
    # Without a pattern, no validation — accepted.
    assert result.success is True


def test_extract_url_skips_validation_when_extracted_url_empty() -> None:
    # Empty URL is an extractor failure, not a wrong-page case. The
    # existing path returns success=False with empty data; the new check
    # must not change that.
    runner = _make_runner()
    result = _execute_extract_url(runner, extracted_url="")
    assert result.success is False
    # Not the wrong-page failure_class — empty url is a different shape.
    assert "WRONG_PAGE" not in (result.data or "")


def test_extract_url_duplicate_check_runs_after_pattern_validation() -> None:
    # Order matters: pattern check first, then dedup. A wrong-page URL
    # should be rejected as WRONG_PAGE even if it's also "seen" — fixing
    # the wrong-page bug is more diagnostic than the dedup signal.
    runner = _make_runner()
    runner.scanner.is_duplicate = MagicMock(return_value=True)  # would normally dedup
    result = _execute_extract_url(
        runner, extracted_url="https://www.boattrader.com/boat-loans/",
    )
    assert "WRONG_PAGE" in result.data
    # Dedup check should NOT have been reached when validation failed.
    runner.scanner.is_duplicate.assert_not_called()
