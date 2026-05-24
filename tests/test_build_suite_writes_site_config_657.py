"""#657 PR 2 — ``build_micro_suite`` writes ``_site_config`` from the
DomainProfile registry; the HTTP path's runner reads it.

Before this PR, every HTTP submission silently inherited
``SiteConfig.default_boattrader()`` via ``MicroPlanRunner.__init__``'s
fallback. lu.ma / mantis-crm / any non-boattrader plan got boattrader
URL regex applied to its URLs — ``is_detail_page`` / ``is_results_page``
returned False everywhere, silently degrading recovery + URL gates.

Contract pinned here:
  - ``build_micro_suite(domain="boattrader.com", ...)`` writes a
    ``_site_config`` dict in the suite that round-trips via
    ``SiteConfig.from_dict`` to a config with the boattrader URL
    patterns.
  - ``build_micro_suite(domain="lu.ma", ...)`` writes a generic-shaped
    SiteConfig (empty regex fields) — the lu.ma DomainProfile sets
    only nav_wait_seconds, not URL patterns.
  - ``build_micro_suite(domain="totally-unknown.tld", ...)`` does NOT
    write ``_site_config`` (no DomainProfile match) — the runner's
    fallback then takes over.
  - ``build_micro_suite(domain="", ...)`` does NOT write
    ``_site_config`` either (empty domain → no profile resolution).
  - Suite construction must never raise on DomainProfile lookup errors
    — best-effort and silent on resolution failure.
"""

from __future__ import annotations

from mantis_agent.server_utils import build_micro_suite
from mantis_agent.site_config import SiteConfig


def _empty_steps() -> list[dict]:
    return [{"intent": "navigate", "type": "navigate", "params": {}}]


def test_boattrader_domain_writes_site_config():
    suite = build_micro_suite(_empty_steps(), domain="boattrader.com")
    assert "_site_config" in suite
    sc = SiteConfig.from_dict(suite["_site_config"])
    assert sc.detail_page_pattern == r"/boat/[\w-]+"
    assert sc.results_page_pattern == r"/boats/"
    assert sc.pagination_format == "/page-{n}/"


def test_boattrader_subdomain_matches_via_suffix():
    """``www.boattrader.com`` resolves to the parent's profile."""
    suite = build_micro_suite(_empty_steps(), domain="www.boattrader.com")
    assert "_site_config" in suite
    sc = SiteConfig.from_dict(suite["_site_config"])
    assert sc.detail_page_pattern == r"/boat/[\w-]+"


def test_luma_domain_writes_generic_shaped_site_config():
    """lu.ma DomainProfile sets only nav_wait_seconds. The resulting
    SiteConfig has empty URL patterns — equivalent to generic shape,
    just with the domain label preserved."""
    suite = build_micro_suite(_empty_steps(), domain="lu.ma")
    assert "_site_config" in suite
    sc = SiteConfig.from_dict(suite["_site_config"])
    assert sc.domain == "lu.ma"
    assert sc.detail_page_pattern == ""
    assert sc.results_page_pattern == ""
    assert sc.pagination_format == ""


def test_unknown_domain_does_not_write_site_config():
    """Domain without a DomainProfile match → no ``_site_config`` key
    in the suite. The runner's fallback then takes over (today still
    boattrader; #657 PR 3 flips that to generic)."""
    suite = build_micro_suite(_empty_steps(), domain="totally-unknown-12345.tld")
    assert "_site_config" not in suite


def test_empty_domain_does_not_write_site_config():
    """Empty / missing domain string → no resolution attempt → no
    ``_site_config`` key. Same fallback behaviour as unknown domain."""
    suite = build_micro_suite(_empty_steps(), domain="")
    assert "_site_config" not in suite


def test_site_config_round_trips_to_runtime_shape():
    """End-to-end shape check: the suite's ``_site_config`` dict must
    deserialize back to a SiteConfig instance whose URL classifiers
    behave correctly on real URLs."""
    suite = build_micro_suite(_empty_steps(), domain="boattrader.com")
    sc = SiteConfig.from_dict(suite["_site_config"])
    assert sc.is_detail_page("https://www.boattrader.com/boat/1986-marine-trader-europa-10167773/")
    assert sc.is_results_page("https://www.boattrader.com/boats/state-fl/by-owner/")
    # Pagination synth: page 2 → /page-2/
    assert sc.paginated_url("https://www.boattrader.com/boats/state-fl/by-owner/", 2).endswith("/page-2/")


def test_site_config_resolution_is_best_effort(monkeypatch):
    """If DomainProfile resolution raises, suite construction still
    succeeds — observability never breaks the build path."""
    def _boom(domain):
        raise RuntimeError("simulated DomainProfile lookup failure")

    # Resolution is imported inside build_micro_suite — patch the
    # module path callers will see at runtime.
    monkeypatch.setattr(
        "mantis_agent.plan_tuning.resolve_domain_profile",
        _boom,
    )
    suite = build_micro_suite(_empty_steps(), domain="boattrader.com")
    # No crash, no _site_config (because resolution failed).
    assert "_site_config" not in suite
    # Other suite fields still landed.
    assert "_micro_plan" in suite
    assert suite["_micro_plan"] == _empty_steps()
