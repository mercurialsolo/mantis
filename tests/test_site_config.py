"""Tests for SiteConfig (issue #46)."""

from mantis_agent.site_config import SiteConfig


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
