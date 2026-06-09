"""Timezone + locale consistency with proxy exit geo (#825).

Modern bot-detection scoring cross-checks ``navigator.language`` /
``Intl.DateTimeFormat().resolvedOptions().timeZone`` /
``Date().getTimezoneOffset()`` against the IP geo. If the proxy
egresses through a Phoenix AZ residential IP but the browser reports
``America/New_York`` (because that's baked into the container image),
that's a detectable mismatch.

This module exposes pure-function helpers that turn the existing
``diagnose_proxy_egress`` payload into (timezone, language) hints. The
runner / setup_env writes ``TZ`` into the env *before* Chrome starts
(so Chrome inherits the right wall-clock) and issues the matching
``Emulation.setTimezoneOverride`` / ``setLocaleOverride`` CDP calls
after Chrome is up.

CUA-purity: this is a pure timestamp / language consistency check; no
DOM access, no behavioral feedback loop. Falls in the same provenance
class as the existing UA override.
"""

from __future__ import annotations

# US state code → primary IANA timezone. Only states with a single
# dominant timezone are listed; multi-tz states (TN, KY, IN, ND, SD,
# FL, MI) fall through to the country default. Names follow the
# IANA zoneinfo database.
_US_STATE_TZ: dict[str, str] = {
    "AL": "America/Chicago",
    "AK": "America/Anchorage",
    "AZ": "America/Phoenix",
    "AR": "America/Chicago",
    "CA": "America/Los_Angeles",
    "CO": "America/Denver",
    "CT": "America/New_York",
    "DE": "America/New_York",
    "DC": "America/New_York",
    "GA": "America/New_York",
    "HI": "Pacific/Honolulu",
    "ID": "America/Boise",
    "IL": "America/Chicago",
    "IA": "America/Chicago",
    "KS": "America/Chicago",
    "LA": "America/Chicago",
    "ME": "America/New_York",
    "MD": "America/New_York",
    "MA": "America/New_York",
    "MN": "America/Chicago",
    "MS": "America/Chicago",
    "MO": "America/Chicago",
    "MT": "America/Denver",
    "NE": "America/Chicago",
    "NV": "America/Los_Angeles",
    "NH": "America/New_York",
    "NJ": "America/New_York",
    "NM": "America/Denver",
    "NY": "America/New_York",
    "NC": "America/New_York",
    "OH": "America/New_York",
    "OK": "America/Chicago",
    "OR": "America/Los_Angeles",
    "PA": "America/New_York",
    "RI": "America/New_York",
    "SC": "America/New_York",
    "TX": "America/Chicago",
    "UT": "America/Denver",
    "VT": "America/New_York",
    "VA": "America/New_York",
    "WA": "America/Los_Angeles",
    "WV": "America/New_York",
    "WI": "America/Chicago",
    "WY": "America/Denver",
}

# ipinfo's ``region`` returns full state names ("Arizona") not codes;
# map both directions so callers don't have to pre-canonicalize.
_US_STATE_NAME_TO_CODE: dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT",
    "delaware": "DE", "district of columbia": "DC", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "iowa": "IA",
    "kansas": "KS", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "ohio": "OH",
    "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "texas": "TX",
    "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}

# Country code → primary IANA timezone (only the country defaults
# Chrome shows on a fresh install for common residential traffic). For
# multi-tz countries the runner SHOULD prefer state-level resolution
# above; this is the safety net.
_COUNTRY_TZ: dict[str, str] = {
    "US": "America/New_York",
    "CA": "America/Toronto",
    "MX": "America/Mexico_City",
    "GB": "Europe/London",
    "UK": "Europe/London",
    "DE": "Europe/Berlin",
    "FR": "Europe/Paris",
    "IT": "Europe/Rome",
    "ES": "Europe/Madrid",
    "NL": "Europe/Amsterdam",
    "BR": "America/Sao_Paulo",
    "AR": "America/Argentina/Buenos_Aires",
    "JP": "Asia/Tokyo",
    "KR": "Asia/Seoul",
    "CN": "Asia/Shanghai",
    "IN": "Asia/Kolkata",
    "AU": "Australia/Sydney",
    "NZ": "Pacific/Auckland",
    "ZA": "Africa/Johannesburg",
    "RU": "Europe/Moscow",
}

# Country code → BCP-47 primary language tag used in
# ``navigator.language`` / ``navigator.languages`` and the
# ``--lang`` Chrome flag.
_COUNTRY_LANG: dict[str, str] = {
    "US": "en-US",
    "CA": "en-CA",
    "MX": "es-MX",
    "GB": "en-GB",
    "UK": "en-GB",
    "DE": "de-DE",
    "FR": "fr-FR",
    "IT": "it-IT",
    "ES": "es-ES",
    "NL": "nl-NL",
    "BR": "pt-BR",
    "AR": "es-AR",
    "JP": "ja-JP",
    "KR": "ko-KR",
    "CN": "zh-CN",
    "IN": "en-IN",
    "AU": "en-AU",
    "NZ": "en-NZ",
    "ZA": "en-ZA",
    "RU": "ru-RU",
}

# Fallback when the country code isn't in either map. Empty string
# means "leave the existing Chrome default alone" — never invent.
_DEFAULT_TZ_FALLBACK = "America/New_York"
_DEFAULT_LANG_FALLBACK = "en-US"


def resolve_tz_and_lang(proxy_diag: dict | None) -> tuple[str, str]:
    """Return ``(tz, lang)`` for the proxy's egress geo.

    Returns the documented fallbacks (``America/New_York``, ``en-US``)
    when the geo is unknown — never empty strings — so the caller can
    set ``TZ`` and ``--lang`` without branching.

    Resolution order:

    1. US state → state-specific timezone (e.g. ``AZ`` → ``America/Phoenix``).
    2. Country code → country default timezone.
    3. Fallback (``America/New_York``, ``en-US``).

    The ``proxy_diag`` shape matches
    :func:`mantis_agent.task_loop.diagnose_proxy_egress` — keys
    ``ip``, ``city``, ``region``, ``country``, ``org`` on success;
    ``disabled``/``error`` flag dicts on failure. Failure returns
    the fallbacks unchanged.
    """
    if not isinstance(proxy_diag, dict) or proxy_diag.get("disabled") or proxy_diag.get("error"):
        return _DEFAULT_TZ_FALLBACK, _DEFAULT_LANG_FALLBACK

    country = str(proxy_diag.get("country", "") or "").upper().strip()
    region = str(proxy_diag.get("region", "") or "").strip()

    tz = ""
    if country == "US" and region:
        # ipinfo returns full state names. Tolerate codes too just in case.
        code = region.upper() if len(region) == 2 else _US_STATE_NAME_TO_CODE.get(region.lower(), "")
        if code:
            tz = _US_STATE_TZ.get(code, "")
    if not tz:
        tz = _COUNTRY_TZ.get(country, _DEFAULT_TZ_FALLBACK)

    lang = _COUNTRY_LANG.get(country, _DEFAULT_LANG_FALLBACK)
    return tz, lang


def tz_for_country(country: str) -> str:
    """Lookup helper exposed for tests and admin diagnostics."""
    return _COUNTRY_TZ.get(country.upper().strip(), _DEFAULT_TZ_FALLBACK)


def lang_for_country(country: str) -> str:
    """Lookup helper exposed for tests and admin diagnostics."""
    return _COUNTRY_LANG.get(country.upper().strip(), _DEFAULT_LANG_FALLBACK)


def tz_for_us_state(region: str) -> str:
    """Resolve a US state name or code to its primary timezone.

    Returns the country-default (``America/New_York``) when the
    region is unknown or empty.
    """
    region = region.strip()
    if not region:
        return _COUNTRY_TZ["US"]
    code = region.upper() if len(region) == 2 else _US_STATE_NAME_TO_CODE.get(region.lower(), "")
    return _US_STATE_TZ.get(code, _COUNTRY_TZ["US"])


__all__ = [
    "lang_for_country",
    "resolve_tz_and_lang",
    "tz_for_country",
    "tz_for_us_state",
]
