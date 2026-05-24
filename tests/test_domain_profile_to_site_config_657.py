"""#657 PR 1 — DomainProfile.to_site_config() + SiteConfig.generic().

Foundational PR for the de-boattrader-ification work. Pins:

  - DomainProfile carries the URL-classification + pagination fields
    that today live only on ``SiteConfig.default_boattrader()``.
  - ``DomainProfile.to_site_config()`` renders a SiteConfig that's
    byte-equal (on the load-bearing fields) to the legacy default for
    boattrader.com — proves the registry can replace the hardcoded
    default with no regression.
  - ``SiteConfig.generic()`` is the neutral fallback for unknown
    domains: empty patterns, falls back to the path-extension
    heuristic when a base_url is available.
  - Profiles without URL fields produce a generic-shaped SiteConfig
    (lu.ma, staffai today — they only set nav_wait, not URL patterns).
"""

from __future__ import annotations

from mantis_agent.plan_tuning import DOMAIN_PROFILES, DomainProfile, resolve_domain_profile
from mantis_agent.site_config import SiteConfig


# ── SiteConfig.generic() ────────────────────────────────────────────


def test_generic_site_config_has_empty_url_patterns():
    s = SiteConfig.generic()
    assert s.detail_page_pattern == ""
    assert s.results_page_pattern == ""
    assert s.pagination_format == ""
    assert s.gate_verify_prompt == ""
    assert s.filtered_results_url == ""


def test_generic_is_detail_page_falls_back_to_path_extension():
    """Without a regex, ``is_detail_page`` returns True when child
    URL extends the base path with at least one new segment — the
    CRM-shape pattern (``/contacts`` → ``/contacts/123``)."""
    s = SiteConfig.generic()
    assert s.is_detail_page(
        "https://example.com/contacts/123",
        "https://example.com/contacts",
    )
    # Same URL → not a detail page.
    assert not s.is_detail_page(
        "https://example.com/contacts",
        "https://example.com/contacts",
    )
    # No base_url → returns False (no signal).
    assert not s.is_detail_page("https://example.com/anything")


def test_generic_is_results_page_returns_false_when_no_pattern():
    s = SiteConfig.generic()
    assert not s.is_results_page("https://example.com/listings")


def test_generic_paginated_url_returns_base_unchanged():
    s = SiteConfig.generic()
    assert s.paginated_url("https://example.com/listings", 2) == (
        "https://example.com/listings"
    )


# ── DomainProfile.to_site_config() ─────────────────────────────────


def _site_config_load_bearing_fields(s: SiteConfig) -> tuple:
    """The fields callers actually read at runtime (used by tests to
    compare without dragging in implementation-detail fields like
    ``prefer_som_grounding`` that aren't yet on DomainProfile)."""
    return (
        s.domain,
        s.detail_page_pattern,
        s.results_page_pattern,
        s.pagination_format,
        s.pagination_type,
        s.pagination_strip_pattern,
        s.gate_verify_prompt,
        s.filtered_results_url,
    )


def test_boattrader_profile_renders_to_default_boattrader_site_config():
    """The boattrader DomainProfile must produce a SiteConfig
    byte-equal (on load-bearing fields) to
    ``SiteConfig.default_boattrader()`` — proves the registry can
    drop-in replace the hardcoded default without behavior change."""
    profile = DOMAIN_PROFILES["boattrader.com"]
    rendered = profile.to_site_config()
    legacy = SiteConfig.default_boattrader()
    assert _site_config_load_bearing_fields(rendered) == _site_config_load_bearing_fields(legacy)


def test_profile_without_url_fields_renders_to_generic_shape():
    """A DomainProfile with only the plan-tuning knobs set (lu.ma,
    staffai) produces a SiteConfig whose URL patterns are all empty
    — equivalent to ``SiteConfig.generic()`` apart from the ``domain``
    label."""
    p = DOMAIN_PROFILES["lu.ma"]
    rendered = p.to_site_config()
    assert rendered.detail_page_pattern == ""
    assert rendered.results_page_pattern == ""
    assert rendered.pagination_format == ""
    assert rendered.filtered_results_url == ""
    assert rendered.gate_verify_prompt == ""
    # Domain label is preserved so observability surfaces can show
    # which domain the SiteConfig is for.
    assert rendered.domain == "lu.ma"


def test_to_site_config_pagination_type_defaults_to_path_suffix():
    """When the profile doesn't set ``pagination_type``, SiteConfig
    gets the legacy default ``"path_suffix"`` (matches
    ``SiteConfig.__dataclass_fields__`` default)."""
    p = DomainProfile(domain="test.com")
    s = p.to_site_config()
    assert s.pagination_type == "path_suffix"


def test_to_site_config_honors_query_param_pagination():
    """A profile that opts into query-param pagination (Zillow-shape)
    gets that flavor preserved end to end."""
    p = DomainProfile(
        domain="example.com",
        # ``pagination_format`` for query_param is just the
        # ``key=value`` body — ``paginated_url`` adds the leading
        # ``?`` (first param) or ``&`` (subsequent) separator.
        pagination_format="page={n}",
        pagination_type="query_param",
        pagination_strip_pattern=r"[?&]page=\d+",
    )
    s = p.to_site_config()
    assert s.pagination_format == "page={n}"
    assert s.pagination_type == "query_param"
    # Verify the actual URL synthesis matches the profile.
    assert s.paginated_url("https://example.com/listings", 3) == (
        "https://example.com/listings?page=3"
    )


def test_resolve_then_to_site_config_round_trip_for_subdomain():
    """A plan that lands on a subdomain still resolves to the
    parent's profile (suffix match), and the SiteConfig renders
    with the parent's URL knobs."""
    p = resolve_domain_profile("www.boattrader.com")
    assert p is not None
    s = p.to_site_config()
    assert s.detail_page_pattern == r"/boat/[\w-]+"
    assert s.is_detail_page("https://www.boattrader.com/boat/1986-marine-trader-europa-10167773/")


# ── Round-trip via to_dict / from_dict ─────────────────────────────


def test_site_config_round_trips_through_dict():
    """The SiteConfig produced by a DomainProfile must survive
    ``to_dict`` / ``from_dict`` — needed by #657 PR 2 which writes
    the SiteConfig into the suite payload as JSON."""
    profile = DOMAIN_PROFILES["boattrader.com"]
    rendered = profile.to_site_config()
    serialised = rendered.to_dict()
    revived = SiteConfig.from_dict(serialised)
    assert _site_config_load_bearing_fields(rendered) == _site_config_load_bearing_fields(revived)
