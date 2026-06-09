"""Tests for the proxy-geo → (timezone, language) resolver (#825).

Resolves a ``diagnose_proxy_egress`` payload to the IANA timezone and
BCP-47 language tag that Chrome should report — so
``navigator.language`` / ``Intl.DateTimeFormat().resolvedOptions().timeZone``
agree with the IP geo a CF / DataDome scorer sees.
"""

from __future__ import annotations

import pytest

from mantis_agent.gym.geo_consistency import (
    lang_for_country,
    resolve_tz_and_lang,
    tz_for_country,
    tz_for_us_state,
)


# ── US state resolution ────────────────────────────────────────────


@pytest.mark.parametrize("region,expected_tz", [
    ("AZ", "America/Phoenix"),
    ("CA", "America/Los_Angeles"),
    ("NY", "America/New_York"),
    ("TX", "America/Chicago"),
    ("CO", "America/Denver"),
    ("HI", "Pacific/Honolulu"),
    ("AK", "America/Anchorage"),
])
def test_us_state_code_resolves(region, expected_tz):
    assert tz_for_us_state(region) == expected_tz


@pytest.mark.parametrize("region,expected_tz", [
    ("Arizona", "America/Phoenix"),
    ("California", "America/Los_Angeles"),
    ("New York", "America/New_York"),
    ("Texas", "America/Chicago"),
    ("Hawaii", "Pacific/Honolulu"),
])
def test_us_full_state_name_resolves(region, expected_tz):
    """ipinfo.io returns full state names, not codes — the helper
    must accept both."""
    assert tz_for_us_state(region) == expected_tz


def test_us_state_case_insensitive():
    assert tz_for_us_state("arizona") == "America/Phoenix"
    assert tz_for_us_state("ARIZONA") == "America/Phoenix"
    assert tz_for_us_state("aZ") == "America/Phoenix"


def test_us_unknown_state_falls_back_to_country_default():
    """Multi-tz states deliberately not in the table (FL, TN, KY)
    should fall back to the country default rather than guess wrong."""
    assert tz_for_us_state("Florida") == "America/New_York"
    assert tz_for_us_state("Tennessee") == "America/New_York"


def test_us_empty_state_returns_country_default():
    assert tz_for_us_state("") == "America/New_York"


# ── Country resolution ─────────────────────────────────────────────


@pytest.mark.parametrize("country,expected_tz", [
    ("US", "America/New_York"),
    ("CA", "America/Toronto"),
    ("MX", "America/Mexico_City"),
    ("GB", "Europe/London"),
    ("UK", "Europe/London"),  # legacy alias
    ("DE", "Europe/Berlin"),
    ("JP", "Asia/Tokyo"),
    ("AU", "Australia/Sydney"),
])
def test_country_tz_lookup(country, expected_tz):
    assert tz_for_country(country) == expected_tz


@pytest.mark.parametrize("country,expected_lang", [
    ("US", "en-US"),
    ("CA", "en-CA"),
    ("MX", "es-MX"),
    ("GB", "en-GB"),
    ("DE", "de-DE"),
    ("JP", "ja-JP"),
])
def test_country_lang_lookup(country, expected_lang):
    assert lang_for_country(country) == expected_lang


def test_country_case_insensitive():
    assert tz_for_country("us") == "America/New_York"
    assert lang_for_country("us") == "en-US"


def test_unknown_country_falls_back():
    """Never invent — fall back to the documented defaults."""
    assert tz_for_country("XX") == "America/New_York"
    assert lang_for_country("XX") == "en-US"


# ── End-to-end: full diag payload ──────────────────────────────────


def test_resolve_phoenix_az_proxy():
    """Canonical case: Oxylabs returns a Phoenix IP. Chrome should
    report Mountain Standard Time (no DST = Phoenix-specific)."""
    diag = {
        "ip": "208.58.174.149",
        "city": "Phoenix",
        "region": "Arizona",
        "country": "US",
        "org": "AS7029 Windstream Communications LLC",
    }
    tz, lang = resolve_tz_and_lang(diag)
    assert tz == "America/Phoenix"
    assert lang == "en-US"


def test_resolve_new_york_proxy():
    diag = {"ip": "1.2.3.4", "city": "Brooklyn", "region": "New York", "country": "US"}
    tz, lang = resolve_tz_and_lang(diag)
    assert tz == "America/New_York"
    assert lang == "en-US"


def test_resolve_canadian_proxy():
    diag = {"city": "Toronto", "region": "Ontario", "country": "CA"}
    tz, lang = resolve_tz_and_lang(diag)
    assert tz == "America/Toronto"
    assert lang == "en-CA"


def test_resolve_disabled_proxy_returns_fallback():
    """proxy_disabled=True flag → use the safe defaults so the env
    var write is still consistent."""
    tz, lang = resolve_tz_and_lang({"disabled": True})
    assert tz == "America/New_York"
    assert lang == "en-US"


def test_resolve_failed_probe_returns_fallback():
    """diagnose_proxy_egress errored out → don't guess."""
    tz, lang = resolve_tz_and_lang({"error": "HTTP 503"})
    assert tz == "America/New_York"
    assert lang == "en-US"


def test_resolve_none_returns_fallback():
    """Defensive — caller might forget to pass the diag through."""
    tz, lang = resolve_tz_and_lang(None)
    assert tz == "America/New_York"
    assert lang == "en-US"


def test_resolve_multi_tz_state_falls_back_to_country():
    """Florida is multi-tz — should land on country default rather
    than guess the wrong half."""
    diag = {"city": "Miami", "region": "Florida", "country": "US"}
    tz, _ = resolve_tz_and_lang(diag)
    assert tz == "America/New_York"  # country default


def test_resolve_unknown_country_uses_fallback():
    diag = {"city": "Atlantis", "region": "Mu", "country": "XX"}
    tz, lang = resolve_tz_and_lang(diag)
    assert tz == "America/New_York"
    assert lang == "en-US"


def test_resolve_country_only_no_region():
    """Region missing but country known → country default tz."""
    diag = {"city": "", "region": "", "country": "DE"}
    tz, lang = resolve_tz_and_lang(diag)
    assert tz == "Europe/Berlin"
    assert lang == "de-DE"
