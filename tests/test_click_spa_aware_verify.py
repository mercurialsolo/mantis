"""Tests for the SPA-aware post-click verifier.

Surfaced by the lu.ma smoke during PR #220's validation: clicks on
event cards open a detail panel without changing the URL. The URL-
based ``SiteConfig.is_detail_page`` check correctly handles full-
page-navigation sites (CRM row → ``/leads/13`` URL change) but
returns False for SPA modals where ``window.location`` stays put.

Generic fix: a new ``ClaudeExtractor.verify_post_click_navigation``
that compares before/after screenshots via tool_use schema and
decides whether the click landed on detail content (URL change OR
same-URL modal). Wired into the click handler's plain-click verify
gate as a fallback after the URL check returns False but before
middle-click escalation runs. No plan vocabulary; works for any SPA.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from PIL import Image

from mantis_agent.extraction.extractor import ClaudeExtractor


def _img(color: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    return Image.new("RGB", (10, 10), color=color)


# ── verify_post_click_navigation primitive ─────────────────────────────


def test_verify_post_click_routes_through_multi_tool_schema() -> None:
    """The verifier MUST use the schema-validated multi-image helper —
    NOT the legacy _call / _call_many / _parse_json path. Pin it so a
    refactor doesn't silently regress to prose-only."""
    extractor = ClaudeExtractor(api_key="dummy")
    multi = MagicMock()
    extractor._call_with_tool_schema_multi = multi  # type: ignore[method-assign]
    extractor._call = MagicMock(  # type: ignore[method-assign]
        side_effect=AssertionError("legacy _call must not be used"),
    )
    extractor._call_many = MagicMock(  # type: ignore[method-assign]
        side_effect=AssertionError("legacy _call_many must not be used"),
    )
    multi.return_value = {
        "navigated": True, "kind": "modal",
        "reason": "Event detail panel covering the page",
    }

    out = extractor.verify_post_click_navigation(
        _img(), _img((40, 50, 130)),
        intent="Click the first event card",
    )

    multi.assert_called_once()
    assert out == {
        "navigated": True, "kind": "modal",
        "reason": "Event detail panel covering the page",
    }


def test_verify_post_click_schema_locks_the_kind_enum() -> None:
    """The kind enum must include all four canonical outcomes so
    the click handler can reason about WHY the verifier said yes/no.
    'wrong_target' specifically lets us distinguish "click did
    something but not what we wanted" from "click did nothing"."""
    extractor = ClaudeExtractor(api_key="dummy")
    multi = MagicMock()
    extractor._call_with_tool_schema_multi = multi  # type: ignore[method-assign]
    multi.return_value = {"navigated": False, "kind": "no_change", "reason": "x"}

    extractor.verify_post_click_navigation(_img(), _img(), intent="x")

    schema = multi.call_args.kwargs["input_schema"]
    enum = schema["properties"]["kind"]["enum"]
    assert set(enum) == {"url_change", "modal", "no_change", "wrong_target"}
    assert set(schema["required"]) == {"navigated", "kind", "reason"}


def test_verify_post_click_passes_before_after_labels() -> None:
    """Each screenshot must be labelled so the model can refer to
    them in its decision. Without labels the model can't tell the
    images apart."""
    extractor = ClaudeExtractor(api_key="dummy")
    multi = MagicMock()
    extractor._call_with_tool_schema_multi = multi  # type: ignore[method-assign]
    multi.return_value = {"navigated": True, "kind": "modal", "reason": "x"}

    extractor.verify_post_click_navigation(_img(), _img(), intent="x")

    labels = multi.call_args.kwargs["labels"]
    assert labels[0] == "BEFORE click"
    assert labels[1] == "AFTER click"


def test_verify_post_click_returns_none_on_no_tool_use() -> None:
    """API regression / network failure → None. Caller treats as
    'couldn't determine; fall through to middle-click escalation'
    rather than guessing yes/no."""
    extractor = ClaudeExtractor(api_key="dummy")
    extractor._call_with_tool_schema_multi = MagicMock(return_value=None)  # type: ignore[method-assign]

    assert extractor.verify_post_click_navigation(_img(), _img(), "x") is None


# ── _call_with_tool_schema_multi helper itself ─────────────────────────


def test_multi_tool_schema_helper_passes_all_images_to_anthropic(monkeypatch) -> None:
    """The helper bundles N screenshots as base64 image content blocks
    in a SINGLE messages payload, alongside the prompt + per-image
    label texts. Lets Claude reason across frames."""
    captured = {}

    class _FakeResponse:
        status_code = 200

        @staticmethod
        def json() -> dict:
            return {
                "content": [
                    {
                        "type": "tool_use", "name": "t",
                        "input": {"navigated": True, "kind": "modal", "reason": "ok"},
                    },
                ],
            }

    def fake_post(url, headers, json, timeout):  # noqa: A002
        captured["body"] = json
        return _FakeResponse()

    import requests as real_requests
    monkeypatch.setattr(real_requests, "post", fake_post)

    extractor = ClaudeExtractor(api_key="dummy")
    out = extractor._call_with_tool_schema_multi(
        [_img(), _img((40, 50, 130))],
        "Compare these.",
        tool_name="t",
        tool_description="d",
        input_schema={
            "type": "object",
            "properties": {
                "navigated": {"type": "boolean"},
                "kind": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["navigated", "kind", "reason"],
        },
        labels=["A", "B"],
    )

    assert out == {"navigated": True, "kind": "modal", "reason": "ok"}
    # Tool choice forced; tool definition included with the schema.
    assert captured["body"]["tool_choice"] == {"type": "tool", "name": "t"}
    # Two image blocks + two label texts + one prompt text = 5 content
    # entries, both image blocks present.
    content = captured["body"]["messages"][0]["content"]
    image_blocks = [b for b in content if b.get("type") == "image"]
    assert len(image_blocks) == 2


# ── Click handler integration ──────────────────────────────────────────


def test_click_handler_falls_back_to_spa_verifier_when_url_check_fails(
    monkeypatch,
) -> None:
    """End-to-end: when the URL-based is_detail_page returns False
    after a click, the handler invokes the SPA verifier. If it
    returns navigated=True, the click is accepted as success and
    middle-click escalation is NOT run."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.click.time.sleep", lambda *_: None)
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.click.random.uniform", lambda *_: 0.0,
    )

    from mantis_agent.gym.listings_scanner import ListingsScanner
    from mantis_agent.gym.step_context import StepContext
    from mantis_agent.gym.step_handlers.click import ClaudeGuidedClickHandler
    from mantis_agent.plan_decomposer import MicroIntent

    class _FakeRunner:
        def __init__(self) -> None:
            self.scanner = ListingsScanner()
            self.costs: dict[str, float] = {
                "claude_extract": 0,
                "claude_grounding": 0,
                "gpu_steps": 0,
                "gpu_seconds": 0,
                "proxy_mb": 0.0,
            }
            self._current_page = 1
            self._last_known_url = ""
            self._last_extracted: dict = {}
            self._last_click_title = ""
            self._opened_detail_in_new_tab = False
            self._page_listing_count = 0
            self.scroll_state_calls: list[dict] = []

        @property
        def _page_listings(self): return self.scanner.page_listings
        @_page_listings.setter
        def _page_listings(self, v): self.scanner.page_listings = v
        @property
        def _page_listing_index(self): return self.scanner.page_listing_index
        @_page_listing_index.setter
        def _page_listing_index(self, v): self.scanner.page_listing_index = v
        @property
        def _viewport_stage(self): return self.scanner.viewport_stage
        @_viewport_stage.setter
        def _viewport_stage(self, v): self.scanner.viewport_stage = v
        @property
        def _max_viewport_stages(self): return self.scanner.max_viewport_stages
        @_max_viewport_stages.setter
        def _max_viewport_stages(self, v): self.scanner.max_viewport_stages = v
        @property
        def _results_base_url(self): return self.scanner.results_base_url
        @_results_base_url.setter
        def _results_base_url(self, v): self.scanner.results_base_url = v
        @property
        def _extracted_titles(self): return self.scanner.extracted_titles
        @_extracted_titles.setter
        def _extracted_titles(self, v): self.scanner.extracted_titles = v
        @property
        def _listings_on_page(self): return self._page_listing_count
        @_listings_on_page.setter
        def _listings_on_page(self, v): self._page_listing_count = v

        def _set_scroll_state(self, **kw): self.scroll_state_calls.append(kw)
        def _current_results_page_url(self): return self.scanner.results_base_url
        def _read_current_url(self, *args, **kw): return "https://lu.ma/discover"

    runner = _FakeRunner()
    runner._max_viewport_stages = 1
    runner._results_base_url = "https://lu.ma/discover"

    extractor = MagicMock()
    extractor.find_all_listings.return_value = [(440, 320, "Some Event")]
    extractor.verify_post_click_navigation.return_value = {
        "navigated": True, "kind": "modal",
        "reason": "Event detail panel opened over the discover grid",
    }

    env = MagicMock()
    env.screen_size = (1280, 800)
    env.screenshot.return_value = _img((40, 50, 130))

    ctx = StepContext(
        env=env, brain=None, extractor=extractor, grounding=None,
        cost_meter=None, dynamic_verifier=MagicMock(),
        scanner=runner.scanner, site_config=MagicMock(),
        tool_channel=None, extraction_cache=None,
        state={"index": 4},
    )
    # URL check returns False (lu.ma /discover stays in URL on click).
    ctx.site_config.is_detail_page.return_value = False

    step = MicroIntent(
        intent="Click the first event card",
        type="click",
        section="extraction",
    )
    result = ClaudeGuidedClickHandler(runner).execute(step, ctx)

    # SPA verifier fired exactly once.
    extractor.verify_post_click_navigation.assert_called_once()
    # And the click was accepted as success.
    assert result.success is True
    # No middle-click happened — the handler accepted on SPA verifier.
    middle_clicks = [
        c for c in env.step.call_args_list
        if getattr(c.args[0], "params", {}).get("button") == "middle"
    ]
    assert middle_clicks == []
    # Detail bookkeeping ran: scroll state advanced to detail_top.
    detail_top = [
        s for s in runner.scroll_state_calls if s.get("context") == "detail_top"
    ]
    assert len(detail_top) == 1


def test_click_handler_does_not_run_spa_verifier_when_url_check_passes(
    monkeypatch,
) -> None:
    """Common case: URL check passes → no SPA verifier call (saves a
    Claude call per successful detail-page navigation). The verifier
    is a fallback, not always-on."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.click.time.sleep", lambda *_: None)
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.click.random.uniform", lambda *_: 0.0,
    )

    from mantis_agent.gym.listings_scanner import ListingsScanner
    from mantis_agent.gym.step_context import StepContext
    from mantis_agent.gym.step_handlers.click import ClaudeGuidedClickHandler
    from mantis_agent.plan_decomposer import MicroIntent

    class _FakeRunner:
        def __init__(self) -> None:
            self.scanner = ListingsScanner()
            self.costs: dict[str, float] = {
                "claude_extract": 0,
                "claude_grounding": 0,
                "gpu_steps": 0,
                "gpu_seconds": 0,
                "proxy_mb": 0.0,
            }
            self._current_page = 1
            self._last_known_url = ""
            self._last_extracted: dict = {}
            self._last_click_title = ""
            self._opened_detail_in_new_tab = False
            self._page_listing_count = 0
            self.scroll_state_calls: list[dict] = []

        @property
        def _page_listings(self): return self.scanner.page_listings
        @_page_listings.setter
        def _page_listings(self, v): self.scanner.page_listings = v
        @property
        def _page_listing_index(self): return self.scanner.page_listing_index
        @_page_listing_index.setter
        def _page_listing_index(self, v): self.scanner.page_listing_index = v
        @property
        def _viewport_stage(self): return self.scanner.viewport_stage
        @_viewport_stage.setter
        def _viewport_stage(self, v): self.scanner.viewport_stage = v
        @property
        def _max_viewport_stages(self): return self.scanner.max_viewport_stages
        @_max_viewport_stages.setter
        def _max_viewport_stages(self, v): self.scanner.max_viewport_stages = v
        @property
        def _results_base_url(self): return self.scanner.results_base_url
        @_results_base_url.setter
        def _results_base_url(self, v): self.scanner.results_base_url = v
        @property
        def _extracted_titles(self): return self.scanner.extracted_titles
        @_extracted_titles.setter
        def _extracted_titles(self, v): self.scanner.extracted_titles = v
        @property
        def _listings_on_page(self): return self._page_listing_count
        @_listings_on_page.setter
        def _listings_on_page(self, v): self._page_listing_count = v

        def _set_scroll_state(self, **kw): self.scroll_state_calls.append(kw)
        def _current_results_page_url(self): return self.scanner.results_base_url
        def _read_current_url(self, *args, **kw): return "https://crm.example.test/leads/13"

    runner = _FakeRunner()
    runner._max_viewport_stages = 1
    runner._results_base_url = "https://crm.example.test/leads"

    extractor = MagicMock()
    extractor.find_all_listings.return_value = [(440, 320, "Lead row")]
    extractor.verify_post_click_navigation = MagicMock(
        side_effect=AssertionError("SPA verifier must not run on URL match"),
    )

    env = MagicMock()
    env.screen_size = (1280, 800)

    ctx = StepContext(
        env=env, brain=None, extractor=extractor, grounding=None,
        cost_meter=None, dynamic_verifier=MagicMock(),
        scanner=runner.scanner, site_config=MagicMock(),
        tool_channel=None, extraction_cache=None,
        state={"index": 4},
    )
    # URL check passes — full page navigation case.
    ctx.site_config.is_detail_page.return_value = True

    step = MicroIntent(
        intent="Click the first lead", type="click", section="extraction",
    )
    result = ClaudeGuidedClickHandler(runner).execute(step, ctx)

    assert result.success is True
    extractor.verify_post_click_navigation.assert_not_called()
