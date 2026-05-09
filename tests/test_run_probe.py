"""Tests for the probe-emitter façade (#224 Phase 3).

Pin the wiring contract: ``run_probe(env, url, ...)`` returns a
:class:`ProbeResult` consumable by :meth:`SiteConfig.from_probe`,
falls back gracefully when the API key is missing or the analysis
call errors, and threads the optional :class:`ObjectiveSpec` into
the prober. The prober's heavy path (real navigation + multi-shot
screenshot) is exercised by integration tests against a real env;
these tests stub the env and Claude API.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from mantis_agent.graph.objective import ObjectiveSpec
from mantis_agent.probe import ProbeResult, run_probe
from mantis_agent.site_config import SiteConfig


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """SiteProber sleeps a total of 8s per probe to give a real browser
    time to settle. The tests don't have a real browser — patch sleep
    so the suite runs in <1s instead of ~80s with 12 cases."""
    import time
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)


def _stub_env(*, screenshot: Image.Image | None = None) -> MagicMock:
    """A minimal env that satisfies SiteProber's contract.

    SiteProber calls ``env.reset(task, start_url)``, ``env.screenshot()``
    (returns ``Image.Image``), and ``env.step(Action(...))``. The
    stub returns a tiny PIL image on each screenshot call so the
    prober doesn't bail out early on missing screenshots.
    """
    env = MagicMock()
    img = screenshot or Image.new("RGB", (10, 10), color="white")
    env.screenshot.return_value = img
    return env


def _claude_response(payload: dict[str, Any]) -> MagicMock:
    """Stubbed Anthropic /v1/messages response in legacy text shape.

    SiteProber currently parses prose + JSON, not tool_use; modernising
    it is a separate Phase. The tests target the run_probe wiring,
    not the prober internals, so the response shape mirrors the
    real API.
    """
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "content": [{"type": "text", "text": json.dumps(payload)}],
    }
    return resp


# ── Public surface ───────────────────────────────────────────────────────


def test_module_exports_run_probe_and_probe_result() -> None:
    """``from mantis_agent.probe import run_probe, ProbeResult`` is the
    canonical import path going forward — keep it stable."""
    from mantis_agent import probe as probe_mod

    assert probe_mod.run_probe is run_probe
    assert probe_mod.ProbeResult is ProbeResult


def test_run_probe_returns_probe_result_type() -> None:
    """Even when nothing more is configured (no API key, neutral
    objective), the helper returns a :class:`ProbeResult` with the
    URL populated. This is the contract ``SiteConfig.from_probe``
    relies on."""
    env = _stub_env()
    result = run_probe(env, "https://example.test/listings")

    assert isinstance(result, ProbeResult)
    assert result.url == "https://example.test/listings"


# ── Env interaction ──────────────────────────────────────────────────────


def test_run_probe_navigates_env_to_url() -> None:
    """The prober drives navigation via ``env.reset(start_url=url)``.
    Confirm the helper passes the URL through."""
    env = _stub_env()
    run_probe(env, "https://example.test/listings")

    env.reset.assert_called_once()
    kwargs = env.reset.call_args.kwargs
    assert kwargs["start_url"] == "https://example.test/listings"


def test_run_probe_captures_screenshots_at_multiple_scroll_positions() -> None:
    """SiteProber takes four screenshots (top, mid1, mid2, bottom).
    Confirm at least three calls land on the env so the prober
    actually inspects the page rather than bailing on first-paint."""
    env = _stub_env()
    run_probe(env, "https://example.test/listings")

    assert env.screenshot.call_count >= 3


# ── Objective threading ──────────────────────────────────────────────────


def test_run_probe_uses_provided_objective_for_domain() -> None:
    """When the caller passes an :class:`ObjectiveSpec` with domains,
    the resulting :class:`ProbeResult` carries that domain so
    ``SiteConfig.from_probe`` can pin it."""
    env = _stub_env()
    spec = ObjectiveSpec(
        raw_text="Extract listings",
        domains=["example.test"],
        target_entity="property",
    )
    result = run_probe(env, "https://example.test/listings", objective=spec)
    assert result.domain == "example.test"


def test_run_probe_constructs_default_objective_when_none() -> None:
    """A caller that doesn't pass an objective gets a neutral probe
    rather than a TypeError — the prober's contract is to accept any
    URL even without a structured spec."""
    env = _stub_env()
    result = run_probe(env, "https://example.test/listings")
    assert isinstance(result, ProbeResult)
    assert result.url == "https://example.test/listings"
    # No domain inferred from a missing objective.
    assert result.domain == ""


# ── Claude analysis (mocked) ─────────────────────────────────────────────


def test_run_probe_populates_analysis_fields_when_api_succeeds() -> None:
    """When ANTHROPIC_API_KEY is set and the Claude call returns a
    valid analysis JSON, ``ProbeResult`` carries the structured fields
    the SiteConfig overlay path needs (pagination_controls,
    listing_container, etc.)."""
    env = _stub_env()
    spec = ObjectiveSpec(
        raw_text="Extract listings", domains=["example.test"], target_entity="listing",
    )
    analysis_payload = {
        "page_type": "search_results",
        "filters_detected": [
            {"name": "Location", "options": ["New York"], "location": "left sidebar"},
        ],
        "listing_container": {
            "description": "card grid",
            "estimated_count": 24,
            "has_photos": True,
            "card_layout": "grid",
        },
        "pagination_controls": {
            "type": "numbered",
            "location": "bottom center",
            "next_text": "Next",
        },
        "dealer_signals": ["Sponsored"],
        "sponsored_signals": ["Promoted"],
        "estimated_listings_per_page": 24,
    }
    with patch("requests.post", return_value=_claude_response(analysis_payload)):
        result = run_probe(
            env,
            "https://example.test/listings",
            objective=spec,
            api_key="sk-test",
        )

    assert result.page_type == "search_results"
    assert result.estimated_listings_per_page == 24
    assert result.pagination_controls["type"] == "numbered"
    assert result.dealer_signals == ["Sponsored"]
    assert len(result.filters_detected) == 1
    assert result.filters_detected[0]["name"] == "Location"


def test_run_probe_returns_partial_result_without_api_key(monkeypatch) -> None:
    """No API key → no Claude call → analysis fields stay empty,
    but ``url`` / ``domain`` are still populated so ``from_probe``
    has something to work with."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = _stub_env()
    spec = ObjectiveSpec(raw_text="x", domains=["example.test"], target_entity="x")

    with patch("requests.post") as mock_post:
        result = run_probe(env, "https://example.test/listings", objective=spec)

    mock_post.assert_not_called()
    assert isinstance(result, ProbeResult)
    assert result.url == "https://example.test/listings"
    assert result.domain == "example.test"
    # Analysis fields stay at defaults.
    assert result.page_type == ""
    assert result.pagination_controls == {}


def test_run_probe_returns_partial_result_on_api_error() -> None:
    """API non-200 → fall back to a partial result rather than
    raising; ``ProbeResult`` carries url/domain but empty analysis."""
    env = _stub_env()
    spec = ObjectiveSpec(raw_text="x", domains=["example.test"], target_entity="x")
    err_resp = MagicMock(status_code=500, text="upstream timeout")

    with patch("requests.post", return_value=err_resp):
        result = run_probe(
            env, "https://example.test/listings", objective=spec, api_key="k",
        )

    assert isinstance(result, ProbeResult)
    assert result.url == "https://example.test/listings"
    assert result.page_type == ""


# ── End-to-end with SiteConfig.from_probe ───────────────────────────────


def test_probe_result_feeds_site_config_from_probe() -> None:
    """End-to-end: ``run_probe`` → ``SiteConfig.from_probe`` produces
    a usable ``SiteConfig`` with detected pagination + URL patterns
    populated. This is the canonical Phase 3 → Phase 4 pipeline
    callers will write."""
    env = _stub_env()
    spec = ObjectiveSpec(raw_text="x", domains=["example.test"], target_entity="x")
    analysis_payload = {
        "page_type": "search_results",
        "pagination_controls": {"type": "numbered", "location": "bottom"},
        "detail_page_pattern": {"url_pattern": "/items/<slug>"},
    }
    with patch("requests.post", return_value=_claude_response(analysis_payload)):
        probe_result = run_probe(
            env,
            "https://example.test/listings",
            objective=spec,
            api_key="k",
        )

    site = SiteConfig.from_probe(probe_result)
    assert site.domain == "example.test"
    # URL has no /page-N/ shape so from_probe falls back to query-param
    # pagination — but the wiring is what we're testing, not the
    # heuristic.
    assert site.pagination_format != ""


def test_probe_result_overlays_recipe_site_config() -> None:
    """End-to-end with Phase 1 overlay: probe-derived SiteConfig
    + recipe overlay produces the expected merged config."""
    from mantis_agent import recipes

    env = _stub_env()
    spec = ObjectiveSpec(
        raw_text="x", domains=["boattrader.com"], target_entity="x",
    )
    analysis_payload = {
        "page_type": "search_results",
        "pagination_controls": {"type": "numbered"},
    }
    with patch("requests.post", return_value=_claude_response(analysis_payload)):
        probe_result = run_probe(
            env, "https://www.boattrader.com/boats/", objective=spec, api_key="k",
        )

    base = SiteConfig.from_probe(probe_result)
    merged = base.overlay(recipes.load_site_config("marketplace_listings"))

    # Recipe fills in detail-page slug shape the probe couldn't see.
    assert merged.detail_page_pattern == r"/boat/[\w-]+"
    # Domain stays from probe (overlay's _str_pick keeps base when
    # recipe is also set — recipe's "boattrader.com" is identical).
    assert merged.domain == "boattrader.com"


# ── Defaults ─────────────────────────────────────────────────────────────


def test_run_probe_defaults_to_haiku_tier_model() -> None:
    """Phase 2/3 lift tasks use Haiku-tier — small structured-JSON
    extraction tasks don't need Opus, and probes can be batchy on
    first-paint of a new domain."""
    import inspect

    sig = inspect.signature(run_probe)
    default = sig.parameters["model"].default
    assert "haiku" in default.lower()
