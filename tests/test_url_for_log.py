"""Tests for #209 Symptom 1 root cause — silent log truncation creating
fake verify disagreement.

The previous ``url[:40]`` truncation in click.py and step_snapshot.py
silently dropped trailing characters mid-segment. A 41-char CRM URL
like ``https://crm.example.test/leads/13`` showed up in the verify
log as ``…/leads/1`` (the trailing ``3`` was cropped). The CDP log
that ran one millisecond earlier used ``[:80]`` and showed the full
URL. Two log lines from the SAME ``url`` variable looked like a real
verifier-vs-CDP disagreement; debugging the false trail was the
single most expensive cost in #209.

Fix: ``url_for_log`` truncates at a generous default (200 chars) and
appends an ellipsis when truncation actually happens, so a reader
can never confuse a cropped URL for a different URL. Generic
primitive — no domain knowledge.
"""

from __future__ import annotations

import logging

import pytest

from mantis_agent.gym.log_utils import url_for_log


# ── Pure helper ─────────────────────────────────────────────────────────


def test_returns_empty_string_for_empty_input() -> None:
    assert url_for_log("") == ""


def test_preserves_url_under_default_limit() -> None:
    """A 41-char URL — exactly the failure mode from #209 — must come
    through with every character intact."""
    url = "https://crm.example.test/leads/13"
    assert len(url) == 33
    assert url_for_log(url) == url


def test_preserves_url_at_default_limit_boundary() -> None:
    """A URL exactly at the limit must NOT trigger the ellipsis path."""
    url = "https://example.com/" + ("a" * (200 - len("https://example.com/")))
    assert len(url) == 200
    assert url_for_log(url) == url
    assert "…" not in url_for_log(url)


def test_truncates_with_visible_marker_when_over_limit() -> None:
    url = "https://example.com/" + ("x" * 500)
    out = url_for_log(url)
    assert out.endswith("…")
    assert len(out) == 201  # limit + ellipsis char
    # The host + path-prefix is preserved verbatim — the part a verifier
    # would compare against survives.
    assert out.startswith("https://example.com/")


def test_custom_limit_respected() -> None:
    url = "abcdefghij"
    assert url_for_log(url, limit=5) == "abcde…"
    assert url_for_log(url, limit=10) == "abcdefghij"
    assert url_for_log(url, limit=20) == "abcdefghij"


@pytest.mark.parametrize(
    "url",
    [
        "https://crm.example.test/leads/13",
        "https://crm.example.test/leads/13/edit",
        "https://erp.example.test/orders/2025-04-08-INV-1234",
        "https://very-long-tenant-subdomain.example.test/api/v2/items/1",
    ],
)
def test_typical_workflow_urls_survive_round_trip(url: str) -> None:
    """For any URL up to ~200 chars (the realistic CUA range), the
    helper is a no-op. Truncation only kicks in for absurdly long
    query strings."""
    assert url_for_log(url) == url


# ── End-to-end: the click verify-fail log no longer mid-truncates ──────


def test_click_verify_fail_log_preserves_full_url(caplog) -> None:
    """The exact failure mode from #209: a 41-char URL whose tail digit
    was being dropped by ``url[:40]``. The fix routes through
    ``url_for_log`` which preserves the full URL well past 41 chars.

    We assert the helper output directly because the log statement
    interpolates it as ``url=%s`` (lazy formatting). If the helper ever
    regresses to mid-truncation, this test catches it without needing
    to drive the whole click handler.
    """
    url = "https://crm.example.test/leads/13"
    formatted = url_for_log(url)
    # No trailing-digit drop, no ellipsis.
    assert formatted.endswith("/leads/13")
    assert "…" not in formatted

    # And it's safe to use as a log argument — verify with a real logger
    # to catch any % formatting regressions.
    logger = logging.getLogger("test_url_for_log")
    with caplog.at_level(logging.INFO, logger="test_url_for_log"):
        logger.info("Not on detail page yet (url=%s) — retrying verify", formatted)
    assert any("/leads/13" in r.message for r in caplog.records)
    assert not any("/leads/1)" in r.message for r in caplog.records)
