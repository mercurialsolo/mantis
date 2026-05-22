"""Tests for issue #597 — pre-filter scan output via prompt-injected
``already_clicked`` titles.

Without the parameter, ``find_all_listings`` returns every listing it
sees on the screenshot, and ``click.py`` filters them post-hoc by
title substring. That wastes Claude tokens (the brain re-classifies
known cards every iteration) and produces fuzzy false-positives on
shared titles.

With the parameter, the scan prompt explicitly tells Claude not to
return the already-clicked titles, shifting dedup to the model's
classification step. Caller passes the bounded tail
(``runner._extracted_titles[-12:]``) to keep the prompt token budget
small on long-running loops.

These tests pin: the parameter is accepted, an empty/None list leaves
the prompt unchanged, a populated list adds the explicit ALREADY
CLICKED block, the tail-12 cap is honored when the caller passes a
longer list, and falsy entries are filtered out of the prompt block.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from PIL import Image

from mantis_agent.extraction.extractor import ClaudeExtractor


def _img() -> Image.Image:
    return Image.new("RGB", (10, 10), color=(255, 255, 255))


def _captured_prompt(extractor: ClaudeExtractor) -> str:
    """Run find_all_listings with mocked ``_call`` and return the prompt
    the call site built (so tests can assert on its contents)."""
    call = MagicMock(return_value='{"status": "empty", "listings": []}')
    extractor._call = call  # type: ignore[method-assign]
    extractor.find_all_listings(_img(), already_clicked=getattr(
        extractor, "_test_already_clicked", None,
    ))
    return call.call_args.args[1]  # (image, prompt, ...)


def test_already_clicked_none_keeps_prompt_clean() -> None:
    """No already_clicked arg → no ALREADY CLICKED block appears."""
    extractor = ClaudeExtractor(api_key="dummy")
    prompt = _captured_prompt(extractor)
    assert "ALREADY CLICKED" not in prompt


def test_already_clicked_empty_list_keeps_prompt_clean() -> None:
    """Empty list → still no ALREADY CLICKED block (caller may pass
    an empty list on the very first iteration of a loop)."""
    extractor = ClaudeExtractor(api_key="dummy")
    extractor._test_already_clicked = []
    prompt = _captured_prompt(extractor)
    assert "ALREADY CLICKED" not in prompt


def test_already_clicked_populates_block_in_prompt() -> None:
    """A non-empty list inserts the ALREADY CLICKED block with each
    title quoted."""
    extractor = ClaudeExtractor(api_key="dummy")
    extractor._test_already_clicked = [
        "1997 Caroff CHATAM 52", "2015 Pioneer 197 Sportfish",
    ]
    prompt = _captured_prompt(extractor)
    assert "ALREADY CLICKED" in prompt
    assert '"1997 Caroff CHATAM 52"' in prompt
    assert '"2015 Pioneer 197 Sportfish"' in prompt


def test_already_clicked_caps_at_tail_12() -> None:
    """If the caller forgets to slice and passes a 20-entry list, the
    extractor's defensive tail-12 cap kicks in — older titles drop
    so the prompt token budget stays bounded on long-running loops."""
    extractor = ClaudeExtractor(api_key="dummy")
    titles = [f"Listing {i:02d}" for i in range(20)]  # 20 titles
    extractor._test_already_clicked = titles
    prompt = _captured_prompt(extractor)
    # First-8 should be excluded (out of the tail-12 window).
    for old in ("Listing 00", "Listing 01", "Listing 07"):
        assert f'"{old}"' not in prompt
    # Last-12 should all be present.
    for kept in ("Listing 08", "Listing 14", "Listing 19"):
        assert f'"{kept}"' in prompt


def test_already_clicked_drops_falsy_entries() -> None:
    """Empty / None entries in the list don't pollute the prompt."""
    extractor = ClaudeExtractor(api_key="dummy")
    extractor._test_already_clicked = ["1997 Caroff", "", None, "2020 Sea Ray"]
    prompt = _captured_prompt(extractor)
    assert '"1997 Caroff"' in prompt
    assert '"2020 Sea Ray"' in prompt
    # Empty quotes shouldn't appear from the empty-string filter.
    assert '""' not in prompt
    assert '"None"' not in prompt


def test_already_clicked_block_omitted_when_all_falsy() -> None:
    """All-falsy list → block doesn't appear at all (no `, , ,` mess)."""
    extractor = ClaudeExtractor(api_key="dummy")
    extractor._test_already_clicked = ["", None, ""]
    prompt = _captured_prompt(extractor)
    assert "ALREADY CLICKED" not in prompt
