"""Tests for url_health — plan-evolution Phase 0 (#704).

Pure-Python classifier; no real browser / env needed. Covers each
documented subclass + the helper functions.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mantis_agent.gym.url_health import (
    classify,
    expand_expected_domains,
    read_page_signals,
)


# ── helpers ───────────────────────────────────────────────────────────


def test_expand_expected_domains_strips_www_and_adds_variants() -> None:
    domains = expand_expected_domains("https://www.boattrader.com/boats/state-fl/")
    assert "boattrader.com" in domains
    assert "www.boattrader.com" in domains
    assert "m.boattrader.com" in domains


def test_expand_expected_domains_handles_bare_domain() -> None:
    domains = expand_expected_domains("https://example.com/path")
    assert "example.com" in domains
    assert "www.example.com" in domains


def test_expand_expected_domains_empty_for_no_netloc() -> None:
    assert expand_expected_domains("about:blank") == set()
    assert expand_expected_domains("") == set()


def test_expand_expected_domains_includes_known_redirect_alias() -> None:
    # #931 follow-up: lu.ma 301s to luma.com — the redirect must not be
    # classified wrong_domain (it hard-halts multi-app chains). Both
    # directions resolve to the same equivalence group.
    from_lu_ma = expand_expected_domains("https://lu.ma/discover")
    assert "lu.ma" in from_lu_ma
    assert "luma.com" in from_lu_ma
    from_luma_com = expand_expected_domains("https://luma.com/discover")
    assert "lu.ma" in from_luma_com
    assert "luma.com" in from_luma_com


def test_redirect_alias_is_not_wrong_domain() -> None:
    # The end-to-end shape of the bug: requested lu.ma, landed luma.com.
    expected = expand_expected_domains("https://lu.ma/discover")
    assert classify(
        current_url="https://luma.com/discover",
        expected_domains=expected,
        page_title="Discover Events · Luma",
    ) == "ok"


def test_unrelated_domain_still_wrong_domain() -> None:
    # The alias map must not over-match: a genuine off-site redirect
    # (login wall on another domain) is still wrong_domain.
    expected = expand_expected_domains("https://lu.ma/discover")
    assert classify(
        current_url="https://accounts.example.com/login",
        expected_domains=expected,
        page_title="Sign in",
    ) == "wrong_domain"


# ── classifier: dns ───────────────────────────────────────────────────


def test_classify_dns_on_chrome_error() -> None:
    assert classify(
        current_url="chrome-error://chromewebdata/",
        expected_domains={"example.com"},
    ) == "dns"


def test_classify_dns_on_empty_url() -> None:
    """Some Chrome failure modes leave current_url empty."""
    assert classify(current_url="", expected_domains={"example.com"}) == "dns"


# ── classifier: blocked (CF / WAF) ────────────────────────────────────


def test_classify_blocked_on_cf_title() -> None:
    assert classify(
        current_url="https://boattrader.com/boats/",
        expected_domains={"boattrader.com"},
        page_title="Just a moment...",
    ) == "blocked"


def test_classify_blocked_precedes_wrong_domain() -> None:
    """CF interstitial under the requested domain is `blocked`, not
    `wrong_domain` — we did reach the right origin."""
    assert classify(
        current_url="https://challenges.cloudflare.com/...",
        expected_domains={"boattrader.com"},
        page_title="Just a moment...",
    ) == "blocked"


# ── classifier: wrong_domain ──────────────────────────────────────────


def test_classify_wrong_domain_on_external_redirect() -> None:
    assert classify(
        current_url="https://accounts.google.com/signin",
        expected_domains={"boattrader.com", "www.boattrader.com"},
        page_title="Sign in - Google Accounts",
    ) == "wrong_domain"


def test_classify_ok_when_landed_on_www_variant() -> None:
    """Plan starts at bare domain; landed on www. should be OK."""
    assert classify(
        current_url="https://www.boattrader.com/boats/state-fl/",
        expected_domains=expand_expected_domains("https://boattrader.com/boats/"),
        page_title="Boat Trader - FL",
    ) == "ok"


def test_classify_ok_when_landed_on_mobile_variant() -> None:
    assert classify(
        current_url="https://m.boattrader.com/boats/",
        expected_domains=expand_expected_domains("https://boattrader.com/"),
    ) == "ok"


# ── classifier: not_found ─────────────────────────────────────────────


def test_classify_not_found_on_404_title() -> None:
    assert classify(
        current_url="https://boattrader.com/boats/nonexistent/",
        expected_domains={"boattrader.com"},
        page_title="404 - Page Not Found - Boat Trader",
    ) == "not_found"


def test_classify_not_found_on_generic_not_found_title() -> None:
    assert classify(
        current_url="https://boattrader.com/boats/typo/",
        expected_domains={"boattrader.com"},
        page_title="Page not found",
    ) == "not_found"


# ── classifier: soft_404 ──────────────────────────────────────────────


def test_classify_soft_404_on_no_results_body() -> None:
    """200 OK, right domain, body says 'we couldn't find'."""
    assert classify(
        current_url="https://boattrader.com/boats/state-fl-typo/",
        expected_domains={"boattrader.com"},
        page_title="Boat Trader",
        page_body_text="We couldn't find any boats matching your search.",
    ) == "soft_404"


def test_classify_soft_404_on_no_listings() -> None:
    assert classify(
        current_url="https://boattrader.com/boat/expired-listing-id/",
        expected_domains={"boattrader.com"},
        page_title="Boat Trader",
        page_body_text="This listing is no longer available.",
    ) == "soft_404"


# ── classifier: ok ────────────────────────────────────────────────────


def test_classify_ok_on_healthy_page() -> None:
    assert classify(
        current_url="https://boattrader.com/boats/state-fl/by-owner/",
        expected_domains={"boattrader.com"},
        page_title="Boats by owner in Florida - Boat Trader",
        page_body_text="Showing 1-25 of 1247 boats",
    ) == "ok"


def test_classify_ok_when_no_expected_domains_provided() -> None:
    """When the caller can't provide expected_domains, we don't claim
    wrong_domain — only the title-based subclasses still apply."""
    assert classify(
        current_url="https://any-domain.com/page",
        expected_domains=set(),
        page_title="Welcome",
    ) == "ok"


# ── precedence ────────────────────────────────────────────────────────


def test_dns_beats_blocked() -> None:
    """An empty URL is dns even if title would match blocked markers."""
    assert classify(
        current_url="",
        expected_domains={"example.com"},
        page_title="Just a moment...",
    ) == "dns"


def test_wrong_domain_skips_not_found_check() -> None:
    """wrong_domain returns before not_found heuristics run."""
    assert classify(
        current_url="https://other-site.com/page-not-found",
        expected_domains={"boattrader.com"},
        page_title="404 Not Found",
    ) == "wrong_domain"


# ── read_page_signals helper ──────────────────────────────────────────


def test_read_page_signals_via_cdp_evaluate() -> None:
    env = MagicMock()
    env.current_url = "https://boattrader.com/boats/"
    env.cdp_evaluate.side_effect = ["My Title", "Body content here"]

    url, title, body = read_page_signals(env)
    assert url == "https://boattrader.com/boats/"
    assert title == "My Title"
    assert body == "Body content here"


def test_read_page_signals_returns_empty_when_cdp_missing() -> None:
    env = MagicMock(spec=["current_url"])
    env.current_url = "https://example.com/"
    url, title, body = read_page_signals(env)
    assert url == "https://example.com/"
    assert title == ""
    assert body == ""


def test_read_page_signals_falls_back_to_playwright_page() -> None:
    """Local-CLI Playwright env doesn't have cdp_evaluate but has page.title().

    Uses a plain class (not MagicMock) because `read_page_signals` checks
    `not callable(page)` to distinguish a Page instance from a bound
    method; MagicMock is always callable so it fails that gate.
    """

    class _FakePage:
        def title(self) -> str:
            return "Playwright Title"

    class _FakeEnv:
        current_url = "https://example.com/"
        _page = _FakePage()

    url, title, _body = read_page_signals(_FakeEnv())
    assert url == "https://example.com/"
    assert title == "Playwright Title"


def test_read_page_signals_swallows_env_exceptions() -> None:
    """Best-effort — any exception returns empty string for that field."""
    env = MagicMock()
    # current_url access raises
    type(env).current_url = property(
        lambda self: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    env.cdp_evaluate.side_effect = RuntimeError("dead pipe")
    url, title, body = read_page_signals(env)
    assert url == ""
    assert title == ""
    assert body == ""


# ── agentic_recovery short-circuit ────────────────────────────────────


def test_agentic_recovery_routes_bad_url_to_halt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 0 invariant: bad_url failures never hit the Claude tool —
    they short-circuit to halt with a structured reason."""
    from mantis_agent import agentic_recovery as ar

    # Ensure the function would otherwise try to make the API call.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-not-used")

    decision = ar.analyse_failure_and_recover(
        step=MagicMock(intent="navigate to boattrader", type="navigate", params={}),
        failure_data="bad_url:wrong_domain",
        screenshot=None,
        plan_context=[],
        attempts=1,
    )
    assert decision is not None
    assert decision.mode == "halt"
    assert "wrong_domain" in decision.reasoning


def test_agentic_recovery_short_circuit_each_subclass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mantis_agent import agentic_recovery as ar
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-not-used")

    for subclass in ("dns", "not_found", "wrong_domain", "soft_404"):
        decision = ar.analyse_failure_and_recover(
            step=MagicMock(intent="navigate", type="navigate", params={}),
            failure_data=f"bad_url:{subclass}",
            screenshot=None,
            plan_context=[],
            attempts=1,
        )
        assert decision is not None and decision.mode == "halt", (
            f"subclass={subclass} should halt"
        )
        assert subclass in decision.reasoning


# ── failure_class integration ─────────────────────────────────────────


def test_failure_class_recognises_bad_url_prefix() -> None:
    from mantis_agent.gym.failure_class import classify as fc_classify
    assert fc_classify("bad_url:wrong_domain") == "bad_url"
    assert fc_classify("bad_url:dns") == "bad_url"
    assert fc_classify("bad_url=not_found") == "bad_url"


def test_failure_class_legacy_classes_still_work() -> None:
    """Adding bad_url didn't disturb the existing rules."""
    from mantis_agent.gym.failure_class import classify as fc_classify
    assert fc_classify("no_state_change", page_title="") == "no_state_change"
    assert fc_classify("cloudflare interstitial detected") == "cf_challenge"
    assert fc_classify("", page_title="Just a moment...") == "cf_challenge"


# ── StepResult round-trip ─────────────────────────────────────────────


def test_step_result_persists_failure_subclass() -> None:
    """The new failure_subclass field survives to_dict/from_dict."""
    from mantis_agent.gym.checkpoint import StepResult

    r = StepResult(
        step_index=0,
        intent="navigate",
        success=False,
        failure_class="bad_url",
        failure_subclass="wrong_domain",
        final_url="https://other-site.com/",
        page_title="Sign in",
    )
    d = r.to_dict()
    assert d["failure_class"] == "bad_url"
    assert d["failure_subclass"] == "wrong_domain"

    r2 = StepResult.from_dict(d)
    assert r2.failure_class == "bad_url"
    assert r2.failure_subclass == "wrong_domain"
    assert r2.final_url == "https://other-site.com/"


def test_step_result_failure_subclass_defaults_empty() -> None:
    from mantis_agent.gym.checkpoint import StepResult
    r = StepResult(step_index=0, intent="x", success=True)
    assert r.failure_subclass == ""
