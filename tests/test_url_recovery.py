"""Tests for url_recovery — plan-evolution Phase 1 (#705).

Covers the two sources (pattern_transform + page_links) + the
orchestrator + the agentic_recovery integration that turns proposals
into ``edit_step`` decisions.

Web search is out of scope for Phase 1 per the implementation decision.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from mantis_agent.gym.url_recovery import (
    RewriteProposal,
    _coerce_links,
    _page_links,
    _pattern_transform,
    _tokenize,
    propose_url_rewrites,
)


# ── _tokenize ─────────────────────────────────────────────────────────


def test_tokenize_strips_stopwords_and_short_tokens() -> None:
    out = _tokenize("Navigate to the boats by-owner page")
    # 'navigate', 'to', 'the', 'page' are stopwords. 'by' is 2 chars
    # and not in the stopword list (intentional — too aggressive a
    # filter risks dropping real path tokens). 'boats' + 'by' + 'owner'
    # survive.
    assert out == {"boats", "by", "owner"}


def test_tokenize_drops_single_char_tokens() -> None:
    """A 1-char token from a hyphen split shouldn't survive."""
    out = _tokenize("a-good-test")
    assert "a" not in out
    assert "good" in out
    assert "test" in out


def test_tokenize_empty_returns_empty() -> None:
    assert _tokenize("") == set()
    assert _tokenize("the a to of") == set()


# ── _pattern_transform ────────────────────────────────────────────────


def test_pattern_transform_trailing_slash() -> None:
    out = _pattern_transform("https://boattrader.com/boats/by-owner")
    new_urls = {p.new_url for p in out}
    assert "https://boattrader.com/boats/by-owner/" in new_urls


def test_pattern_transform_strip_trailing_slash() -> None:
    out = _pattern_transform("https://boattrader.com/boats/by-owner/")
    new_urls = {p.new_url for p in out}
    assert "https://boattrader.com/boats/by-owner" in new_urls


def test_pattern_transform_slug_split_state() -> None:
    """`/boats/state-fl/` → `/boats/state/fl/` is the canonical rule."""
    out = _pattern_transform("https://boattrader.com/boats/state-fl/")
    new_urls = {p.new_url for p in out}
    assert "https://boattrader.com/boats/state/fl/" in new_urls


def test_pattern_transform_slug_merge() -> None:
    """Reverse: `/state/fl/` → `/state-fl/`."""
    out = _pattern_transform("https://boattrader.com/boats/state/fl/")
    new_urls = {p.new_url for p in out}
    assert "https://boattrader.com/boats/state-fl/" in new_urls


def test_pattern_transform_www_strip_and_add() -> None:
    out_no_www = _pattern_transform("https://boattrader.com/")
    assert any(p.new_url == "https://www.boattrader.com/" for p in out_no_www)

    out_www = _pattern_transform("https://www.boattrader.com/")
    assert any(p.new_url == "https://boattrader.com/" for p in out_www)


def test_pattern_transform_case_fold_query() -> None:
    out = _pattern_transform("https://example.com/page?Param=Value")
    new_urls = {p.new_url for p in out}
    assert "https://example.com/page?param=value" in new_urls


def test_pattern_transform_skips_own_url() -> None:
    """No proposal should equal the input URL."""
    failed = "https://boattrader.com/"
    out = _pattern_transform(failed)
    assert all(p.new_url != failed for p in out)


def test_pattern_transform_no_scheme_returns_empty() -> None:
    assert _pattern_transform("/no/scheme") == []
    assert _pattern_transform("") == []


def test_pattern_transform_confidence_in_range() -> None:
    out = _pattern_transform("https://boattrader.com/boats/state-fl/")
    for p in out:
        assert 0.0 <= p.confidence <= 1.0
        assert p.source == "pattern_transform"


def test_pattern_transform_dedupes_proposals() -> None:
    """Multiple rules converging on the same URL emit only once."""
    out = _pattern_transform("https://boattrader.com/x")
    new_urls = [p.new_url for p in out]
    assert len(new_urls) == len(set(new_urls))


# ── _page_links ───────────────────────────────────────────────────────


def test_page_links_skips_when_env_lacks_cdp() -> None:
    env = MagicMock(spec=[])  # no cdp_evaluate attribute
    assert _page_links(env, "https://example.com/typo", "navigate") == []


def test_page_links_skips_when_cdp_raises() -> None:
    env = MagicMock()
    env.cdp_evaluate.side_effect = RuntimeError("CDP unreachable")
    assert _page_links(env, "https://example.com/typo", "navigate") == []


def test_page_links_filters_off_domain() -> None:
    """Cross-site links don't get proposed."""
    env = MagicMock()
    env.cdp_evaluate.return_value = [
        {"text": "Sign in", "href": "https://accounts.google.com/signin"},
        {"text": "Boats by owner", "href": "https://boattrader.com/boats/by-owner/"},
    ]
    out = _page_links(env, "https://boattrader.com/x", "Find boats by owner")
    new_urls = {p.new_url for p in out}
    assert "https://accounts.google.com/signin" not in new_urls
    assert "https://boattrader.com/boats/by-owner/" in new_urls


def test_page_links_scores_by_token_overlap() -> None:
    env = MagicMock()
    env.cdp_evaluate.return_value = [
        {"text": "Unrelated link", "href": "https://boattrader.com/something-else/"},
        {"text": "Boats by owner FL", "href": "https://boattrader.com/boats/by-owner/"},
    ]
    out = _page_links(env, "https://boattrader.com/x", "boats owner florida")
    # The matching link should rank first.
    assert out
    assert out[0].new_url == "https://boattrader.com/boats/by-owner/"
    assert out[0].source == "page_links"
    assert "score=" in out[0].notes


def test_page_links_skips_self_referential_proposals() -> None:
    """A link pointing at the same failed URL isn't a useful proposal."""
    env = MagicMock()
    failed = "https://boattrader.com/boats/state-fl/"
    env.cdp_evaluate.return_value = [
        {"text": "Boats in FL", "href": failed},
    ]
    out = _page_links(env, failed, "boats florida")
    assert all(p.new_url != failed for p in out)


def test_page_links_returns_empty_on_malformed_cdp_payload() -> None:
    env = MagicMock()
    env.cdp_evaluate.return_value = "not a list"
    assert _page_links(env, "https://example.com/", "navigate") == []


def test_page_links_caps_to_top_3() -> None:
    env = MagicMock()
    # 5 matching links — should clip to 3 proposals
    env.cdp_evaluate.return_value = [
        {"text": f"Boats by owner {i}", "href": f"https://boattrader.com/boats/{i}"}
        for i in range(5)
    ]
    out = _page_links(env, "https://boattrader.com/x", "boats owner")
    assert len(out) <= 3


# ── _coerce_links ─────────────────────────────────────────────────────


def test_coerce_links_filters_non_dict_entries() -> None:
    raw = [{"text": "ok", "href": "https://x.com"}, "junk", None, {"href": ""}]
    out = _coerce_links(raw, max_links=10)
    assert len(out) == 1
    assert out[0].href == "https://x.com"


def test_coerce_links_caps_at_max() -> None:
    raw = [{"text": f"t{i}", "href": f"https://x.com/{i}"} for i in range(500)]
    out = _coerce_links(raw, max_links=10)
    assert len(out) == 10


# ── propose_url_rewrites orchestrator ─────────────────────────────────


def test_propose_dns_uses_pattern_transform_only() -> None:
    """DNS failure → no page loaded → page_links should be skipped."""
    env = MagicMock()
    env.cdp_evaluate.return_value = [{"text": "Home", "href": "https://x.com/"}]
    report = propose_url_rewrites(
        failed_url="https://boattrader.com/boats/state-fl/",
        failure_subclass="dns",
        intent_text="navigate to florida boats",
        env=env,
    )
    assert "pattern_transform" in report.sources_tried
    assert "page_links" not in report.sources_tried
    assert "dns_no_page_loaded" in report.sources_skipped.get("page_links", "")
    # CDP wasn't even consulted
    env.cdp_evaluate.assert_not_called()


def test_propose_not_found_runs_both_sources() -> None:
    env = MagicMock()
    env.cdp_evaluate.return_value = [
        {"text": "Boats by owner", "href": "https://boattrader.com/boats/by-owner/"},
    ]
    report = propose_url_rewrites(
        failed_url="https://boattrader.com/boats/typo/",
        failure_subclass="not_found",
        intent_text="boats by owner",
        env=env,
    )
    assert "pattern_transform" in report.sources_tried
    assert "page_links" in report.sources_tried


def test_propose_sorts_by_confidence_desc() -> None:
    report = propose_url_rewrites(
        failed_url="https://boattrader.com/boats/state-fl/",
        failure_subclass="not_found",
        intent_text="boats",
        env=None,  # page_links skipped silently
    )
    if len(report.proposals) >= 2:
        confidences = [p.confidence for p in report.proposals]
        assert confidences == sorted(confidences, reverse=True)


def test_propose_logs_env_lacking_cdp() -> None:
    """When env doesn't support CDP, page_links is marked skipped."""
    env_no_cdp = MagicMock(spec=[])
    report = propose_url_rewrites(
        failed_url="https://boattrader.com/boats/state-fl/",
        failure_subclass="wrong_domain",
        intent_text="boats",
        env=env_no_cdp,
    )
    assert "env_lacks_cdp_evaluate" in report.sources_skipped.get("page_links", "")


def test_propose_empty_when_no_sources_match() -> None:
    """Garbage URL + no env → pattern_transform empty + page_links skipped."""
    report = propose_url_rewrites(
        failed_url="garbage-not-a-url",
        failure_subclass="dns",
        intent_text="",
        env=None,
    )
    assert report.proposals == []


# ── agentic_recovery integration ──────────────────────────────────────


def _stub_step(intent: str = "", params: dict | None = None):
    s = MagicMock()
    s.intent = intent
    s.params = params or {}
    s.type = "navigate"
    return s


def test_agentic_recovery_emits_edit_step_for_bad_url_with_proposal() -> None:
    """End-to-end: bad_url failure with a pattern_transform-recoverable
    URL → recovery returns an `edit_step` decision rewriting params.url."""
    from mantis_agent import agentic_recovery as ar

    step = _stub_step(
        intent="Navigate to https://boattrader.com/boats/state-fl/ to find listings",
        params={"url": "https://boattrader.com/boats/state-fl/"},
    )
    decision = ar.analyse_failure_and_recover(
        step=step,
        failure_data="bad_url:not_found",
        screenshot=None,
        plan_context=[],
        attempts=1,
        env=None,
    )
    assert decision is not None
    assert decision.mode == "edit_step"
    assert "url" in decision.edited_step.get("params", {})
    new_url = decision.edited_step["params"]["url"]
    assert new_url != "https://boattrader.com/boats/state-fl/"
    assert "rewrite_url:pattern_transform" in decision.reasoning


def test_agentic_recovery_rewrites_intent_when_url_literal() -> None:
    """When the failed URL appears verbatim in intent prose, the rewrite
    replaces it so logs are coherent."""
    from mantis_agent import agentic_recovery as ar

    failed = "https://boattrader.com/boats/state-fl/"
    step = _stub_step(
        intent=f"Navigate to {failed} to find listings",
        params={"url": failed},
    )
    decision = ar.analyse_failure_and_recover(
        step=step,
        failure_data="bad_url:not_found",
        screenshot=None,
        plan_context=[],
        attempts=1,
        env=None,
    )
    assert decision is not None and decision.mode == "edit_step"
    new_url = decision.edited_step["params"]["url"]
    assert failed not in decision.edited_step["intent"]
    assert new_url in decision.edited_step["intent"]


def test_agentic_recovery_halts_when_no_proposal() -> None:
    """Garbage URL → no proposals → halt with structured reason."""
    from mantis_agent import agentic_recovery as ar

    step = _stub_step(
        intent="Navigate to this thing",
        params={"url": "garbage-not-a-url"},
    )
    decision = ar.analyse_failure_and_recover(
        step=step,
        failure_data="bad_url:dns",
        screenshot=None,
        plan_context=[],
        attempts=1,
        env=None,
    )
    assert decision is not None
    assert decision.mode == "halt"
    assert "rewrite_url found no candidates" in decision.reasoning


def test_agentic_recovery_uses_page_links_when_env_available() -> None:
    """When env has cdp_evaluate, page_links is consulted alongside pattern."""
    from mantis_agent import agentic_recovery as ar

    env = MagicMock()
    env.cdp_evaluate.return_value = [
        {"text": "Boats by owner Florida",
         "href": "https://boattrader.com/boats/state/fl/by-owner/"},
    ]
    step = _stub_step(
        intent="Navigate to https://boattrader.com/boats/state-fl/by-owner/",
        params={"url": "https://boattrader.com/boats/state-fl/by-owner/"},
    )
    decision = ar.analyse_failure_and_recover(
        step=step,
        failure_data="bad_url:wrong_domain",
        screenshot=None,
        plan_context=[],
        attempts=1,
        env=env,
    )
    assert decision is not None and decision.mode == "edit_step"
    # CDP was consulted at least once
    env.cdp_evaluate.assert_called()


def test_agentic_recovery_falls_through_for_non_bad_url() -> None:
    """Other failure_class blobs are NOT short-circuited; they go to
    Claude (or, without an API key, return None)."""
    import os
    from mantis_agent import agentic_recovery as ar

    # No API key set — should return None (legacy halt fallback)
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        step = _stub_step(intent="click the Save button", params={})
        decision = ar.analyse_failure_and_recover(
            step=step,
            failure_data="no_state_change:click",
            screenshot=None,
            plan_context=[],
            attempts=1,
            env=None,
        )
        assert decision is None  # would have hit Claude path
    finally:
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved


# ── RewriteProposal dataclass ────────────────────────────────────────


def test_rewrite_proposal_fields() -> None:
    p = RewriteProposal(
        new_url="https://x.com/",
        source="pattern_transform",
        confidence=0.6,
        notes="strip-www",
    )
    assert p.new_url == "https://x.com/"
    assert p.source == "pattern_transform"
    assert p.confidence == 0.6
    assert p.matched_link_text == ""


# ── helpers in agentic_recovery ──────────────────────────────────────


def test_extract_failed_url_from_step_prefers_params_url() -> None:
    from mantis_agent.agentic_recovery import _extract_failed_url_from_step
    step = _stub_step(
        intent="navigate to https://intent-url.com/",
        params={"url": "https://params-url.com/path"},
    )
    assert _extract_failed_url_from_step(step, {}) == "https://params-url.com/path"


def test_extract_failed_url_from_step_falls_back_to_intent() -> None:
    from mantis_agent.agentic_recovery import _extract_failed_url_from_step
    step = _stub_step(intent="Open https://intent-url.com/page", params={})
    assert _extract_failed_url_from_step(step, {}) == "https://intent-url.com/page"


def test_extract_failed_url_from_step_falls_back_to_page_context() -> None:
    from mantis_agent.agentic_recovery import _extract_failed_url_from_step
    step = _stub_step(intent="navigate somewhere", params={})
    assert _extract_failed_url_from_step(
        step, {"current_url": "https://ctx-url.com/"}
    ) == "https://ctx-url.com/"


def test_rewrite_intent_url_swaps_literal() -> None:
    from mantis_agent.agentic_recovery import _rewrite_intent_url
    out = _rewrite_intent_url(
        "Navigate to https://old.com/ and click", "https://old.com/", "https://new.com/"
    )
    assert out == "Navigate to https://new.com/ and click"


def test_rewrite_intent_url_unchanged_when_url_not_literal() -> None:
    from mantis_agent.agentic_recovery import _rewrite_intent_url
    intent = "Navigate to the listings page"
    out = _rewrite_intent_url(intent, "https://x.com/", "https://y.com/")
    assert out == intent
