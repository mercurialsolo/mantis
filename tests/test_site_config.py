"""Tests for SiteConfig (issue #46)."""

from mantis_agent.site_config import SiteConfig, _path_extends


def test_default_boattrader():
    config = SiteConfig.default_boattrader()
    assert config.domain == "boattrader.com"
    assert config.is_detail_page("https://www.boattrader.com/boat/2020-sea-ray-240-123456/")
    assert not config.is_detail_page("https://www.boattrader.com/boats/by-owner/")
    assert config.is_results_page("https://www.boattrader.com/boats/by-owner/page-2/")
    assert not config.is_results_page("https://www.boattrader.com/boat/2020-sea-ray-240/")


def test_paginated_url_path_suffix():
    config = SiteConfig.default_boattrader()
    result = config.paginated_url(
        "https://www.boattrader.com/boats/by-owner/", 3
    )
    assert result == "https://www.boattrader.com/boats/by-owner/page-3/"


def test_paginated_url_strips_existing_page():
    config = SiteConfig.default_boattrader()
    result = config.paginated_url(
        "https://www.boattrader.com/boats/by-owner/page-2/", 5
    )
    assert result == "https://www.boattrader.com/boats/by-owner/page-5/"


def test_paginated_url_query_param():
    config = SiteConfig(
        pagination_format="page={n}",
        pagination_type="query_param",
        pagination_strip_pattern=r"[?&]page=\d+",
    )
    result = config.paginated_url("https://example.com/results", 2)
    assert result == "https://example.com/results?page=2"


def test_paginated_url_query_param_appends():
    config = SiteConfig(
        pagination_format="page={n}",
        pagination_type="query_param",
        pagination_strip_pattern=r"[?&]page=\d+",
    )
    result = config.paginated_url("https://example.com/results?q=test", 3)
    assert result == "https://example.com/results?q=test&page=3"


def test_zillow_config():
    config = SiteConfig(
        domain="zillow.com",
        detail_page_pattern=r"/homes/\d+_zpid",
        results_page_pattern=r"/homes/",
        pagination_format="{n}_p",
        pagination_type="path_suffix",
        pagination_strip_pattern=r"/\d+_p/?$",
    )
    assert config.is_detail_page("https://www.zillow.com/homes/12345678_zpid/")
    assert not config.is_detail_page("https://www.zillow.com/homes/miami-fl/")
    assert config.is_results_page("https://www.zillow.com/homes/miami-fl/")


def test_indeed_config():
    config = SiteConfig(
        domain="indeed.com",
        detail_page_pattern=r"/viewjob\?",
        results_page_pattern=r"/jobs\?",
        pagination_format="start={n}",
        pagination_type="query_param",
        pagination_strip_pattern=r"[?&]start=\d+",
    )
    assert config.is_detail_page("https://indeed.com/viewjob?jk=abc123")
    assert config.is_results_page("https://indeed.com/jobs?q=engineer&l=sf")


def test_serialization_roundtrip():
    config = SiteConfig.default_boattrader()
    data = config.to_dict()
    restored = SiteConfig.from_dict(data)
    assert restored.domain == config.domain
    assert restored.detail_page_pattern == config.detail_page_pattern
    assert restored.pagination_format == config.pagination_format
    assert restored.is_detail_page("https://boattrader.com/boat/test-123/")


def test_empty_patterns_safe():
    config = SiteConfig()
    assert not config.is_detail_page("https://example.com/anything")
    assert not config.is_results_page("https://example.com/anything")
    assert config.paginated_url("https://example.com", 2) == "https://example.com"


def test_from_probe_basic():
    from mantis_agent.graph.probe import ProbeResult

    probe = ProbeResult(
        url="https://example.com/search/results",
        domain="example.com",
        pagination_controls={"type": "numbered"},
        detail_page_pattern={"url_pattern": "/item/<slug>/"},
    )
    config = SiteConfig.from_probe(probe)
    assert config.domain == "example.com"
    assert config.is_detail_page("https://example.com/item/test-123/")


# ── Generic detail-page heuristic (#209 Symptom 1) ───────────────────────


def test_path_extends_basic_segment_addition():
    assert _path_extends("/leads/13", "/leads") is True
    assert _path_extends("/leads/13/edit", "/leads") is True
    assert _path_extends("/products/9", "/products") is True


def test_path_extends_rejects_same_path():
    assert _path_extends("/leads", "/leads") is False
    assert _path_extends("/leads/", "/leads") is False
    # Trailing-slash differences are normalised away by the segment split.


def test_path_extends_rejects_different_branch():
    assert _path_extends("/login", "/leads") is False
    assert _path_extends("/admin/users", "/leads") is False


def test_path_extends_handles_root_parent():
    assert _path_extends("/leads", "/") is True
    assert _path_extends("/leads", "") is True


def test_path_extends_rejects_shorter_or_equal():
    # Going UP the tree is not "deeper".
    assert _path_extends("/leads", "/leads/13") is False
    assert _path_extends("", "/leads") is False


# ── is_detail_page with base_url heuristic ───────────────────────────────


def test_is_detail_page_pattern_wins_over_base_url():
    """When detail_page_pattern is configured, base_url is ignored."""
    config = SiteConfig(detail_page_pattern=r"/boat/[\w-]+")
    # Pattern matches → True, regardless of base_url.
    assert config.is_detail_page(
        "https://x.com/boat/abc-123",
        base_url="https://other.com/listings",
    )
    # Pattern does not match → False, even if path-extends would say True.
    assert not config.is_detail_page(
        "https://x.com/something/else",
        base_url="https://x.com/something",
    )


def test_is_detail_page_falls_back_to_base_url_heuristic_when_no_pattern():
    """Empty SiteConfig — generic heuristic kicks in once base_url is passed.

    Covers the #209 Symptom 1 failure mode: a CRM with no pattern
    configured should still recognise /leads/13 as a detail page when
    the runner was on /leads.
    """
    config = SiteConfig()
    assert config.is_detail_page(
        "https://crm.example.test/leads/13",
        base_url="https://crm.example.test/leads",
    )
    assert config.is_detail_page(
        "https://crm.example.test/leads/13/edit",
        base_url="https://crm.example.test/leads",
    )


def test_is_detail_page_heuristic_rejects_same_path():
    """Same listings page after a filter change is NOT a detail page."""
    config = SiteConfig()
    assert not config.is_detail_page(
        "https://crm.example.test/leads?status=qualified",
        base_url="https://crm.example.test/leads",
    )


def test_is_detail_page_heuristic_rejects_login_redirect():
    """A click that redirected to /login is on the same host but a
    different branch — must not be classified as a detail page."""
    config = SiteConfig()
    assert not config.is_detail_page(
        "https://crm.example.test/login",
        base_url="https://crm.example.test/leads",
    )


def test_is_detail_page_heuristic_rejects_different_host():
    """A cross-host redirect (e.g. SSO provider) is never a detail page."""
    config = SiteConfig()
    assert not config.is_detail_page(
        "https://idp.example.test/sso/login",
        base_url="https://crm.example.test/leads",
    )


def test_is_detail_page_heuristic_returns_false_without_base_url():
    """Backwards-compatible: empty SiteConfig + no base_url → False
    (preserves the behaviour every existing call site relies on)."""
    config = SiteConfig()
    assert not config.is_detail_page("https://crm.example.test/leads/13")


def test_is_detail_page_handles_query_string_in_base():
    """Query strings on the base URL must not break the heuristic."""
    config = SiteConfig()
    assert config.is_detail_page(
        "https://crm.example.test/leads/13",
        base_url="https://crm.example.test/leads?status=qualified&page=2",
    )
