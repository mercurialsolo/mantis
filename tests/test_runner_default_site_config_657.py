"""#657 PR 3 — ``MicroPlanRunner`` defaults to ``SiteConfig.generic()``
instead of ``SiteConfig.default_boattrader()``. Stripped the
``{"boats"}`` literal from ``derive_filter_tokens`` while we're here.

Until this PR, every caller that didn't pass an explicit
``site_config=`` got the boattrader URL regex applied to its plan —
silently degrading any non-boattrader workflow's URL classification.

Contract pinned here:
  - A bare ``MicroPlanRunner(...)`` call without ``site_config=``
    lands with the neutral ``SiteConfig.generic()`` — empty URL
    patterns, no pagination synthesis, no boattrader-shaped gate
    prompt prefix.
  - Boattrader plans submitted through ``build_micro_suite`` still
    get the boattrader-shaped ``SiteConfig`` via the DomainProfile
    registry (#648 / #657 PR 1) — unchanged effective behaviour for
    the canonical production target.
  - ``derive_filter_tokens`` no longer special-cases the literal
    ``"boats"`` path segment; pagination segments (``page-N``) still
    skip as a domain-neutral convention.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from mantis_agent.gym.micro_runner import MicroPlanRunner
from mantis_agent.gym._runner_helpers import derive_filter_tokens
from mantis_agent.site_config import SiteConfig


# ── MicroPlanRunner default ───────────────────────────────────────


def test_micro_plan_runner_default_site_config_is_generic():
    """Bare runner construction lands with the neutral SiteConfig —
    no boattrader URL regex, no domain-specific gate prompt prefix."""
    r = MicroPlanRunner(brain=MagicMock(), env=MagicMock())
    assert r.site_config.detail_page_pattern == ""
    assert r.site_config.results_page_pattern == ""
    assert r.site_config.pagination_format == ""
    assert r.site_config.gate_verify_prompt == ""
    assert r.site_config.filtered_results_url == ""


def test_micro_plan_runner_explicit_site_config_wins():
    """Explicit ``site_config=`` arg still overrides the default —
    callers that thread a SiteConfig (HTTP path via #657 PR 2's
    ``_site_config`` plumbing, CLI path via the existing
    ``modal_cua_server.py:1278-1281`` resolution) are unaffected."""
    custom = SiteConfig(
        domain="example.com",
        detail_page_pattern=r"/item/\d+",
        results_page_pattern=r"/items/",
    )
    r = MicroPlanRunner(brain=MagicMock(), env=MagicMock(), site_config=custom)
    assert r.site_config.detail_page_pattern == r"/item/\d+"
    assert r.site_config.results_page_pattern == r"/items/"
    assert r.site_config.domain == "example.com"


def test_generic_default_is_truly_neutral():
    """The new default's ``is_detail_page`` falls back to the path-
    extension heuristic when a ``base_url`` is supplied — handles
    CRM-shape URLs (``/contacts`` → ``/contacts/123``) without needing
    a regex hardcoded."""
    r = MicroPlanRunner(brain=MagicMock(), env=MagicMock())
    # Path-extension heuristic via base_url.
    assert r.site_config.is_detail_page(
        "https://example.com/contacts/123",
        base_url="https://example.com/contacts",
    )
    # No regex, no base_url → returns False (no signal).
    assert not r.site_config.is_detail_page("https://example.com/anything")
    # ``is_results_page`` without a pattern → False.
    assert not r.site_config.is_results_page("https://example.com/contacts")


# ── derive_filter_tokens drops the "boats" literal exclusion ─────


def test_derive_filter_tokens_no_longer_special_cases_boats():
    """Boattrader URL ``/boats/state-fl/by-owner/`` now yields
    ``("boats", "state-fl", "by-owner")`` instead of
    ``("state-fl", "by-owner")``. Functionally equivalent downstream
    — ``url_has_required_filters`` checks the tokens are a subset of
    the URL, and ``boats`` is trivially in every boattrader URL."""
    tokens = derive_filter_tokens(
        "https://www.boattrader.com/boats/state-fl/by-owner/"
    )
    assert "boats" in tokens
    assert "state-fl" in tokens
    assert "by-owner" in tokens


def test_derive_filter_tokens_still_skips_pagination_segments():
    """``page-N`` segments aren't filters — that exclusion stays as
    a domain-neutral pagination convention."""
    tokens = derive_filter_tokens(
        "https://www.boattrader.com/boats/state-fl/by-owner/page-3/"
    )
    assert "page-3" not in tokens
    assert "page" not in tokens
    assert "state-fl" in tokens


def test_derive_filter_tokens_handles_non_boattrader_url():
    """Generic CRM-shape URL — no domain-specific magic, just path
    segments (minus pagination)."""
    tokens = derive_filter_tokens("https://crm.example/contacts/active/123")
    assert tokens == ("contacts", "active", "123")


def test_derive_filter_tokens_empty_for_bare_host():
    """No path → no tokens."""
    assert derive_filter_tokens("https://example.com") == ()
    assert derive_filter_tokens("https://example.com/") == ()
