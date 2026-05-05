"""PaginateHandler unit tests — Phase 2 of EPIC #161.

Three-layer pagination: URL-based → Claude-guided → Holo3 fallback.
Tests pin the layer-selection logic and per-layer success bookkeeping.

- Layer 1 success: pagination_format set + env.reset succeeds → bypasses
  Layer 2/3, increments _current_page, records pagination(method="url")
- Layer 1 skip when pagination_format unset → falls through to Layer 2
- Layer 1 fail (env.reset raises) → falls through to Layer 2; failure
  recorded with reason
- Layer 2 Claude paginate: target found tuple → grounding refine →
  click → success; _listings_on_page reset to 0
- Layer 2 Claude paginate: 3 attempts of 'not_found' → returns failure,
  then Layer 3 Holo3 fallback runs
- Holo3 fallback success bookkeeping (record_pagination(method="holo3"))
- Holo3 fallback failure → final record_pagination(method="all_layers",
  reason="next_control_not_found")
- step_type property

No Xvfb, no GymRunner, no real ClaudeExtractor.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from mantis_agent.gym.step_context import StepContext
from mantis_agent.gym.step_handlers.paginate import PaginateHandler
from mantis_agent.plan_decomposer import MicroIntent


class _FakeRunner:
    def __init__(self) -> None:
        self.costs: dict[str, float] = {
            "claude_extract": 0,
            "claude_grounding": 0,
            "gpu_steps": 0,
            "gpu_seconds": 0,
            "proxy_mb": 0.0,
        }
        self._current_page = 1
        self._results_base_url = "https://example.com/cars"
        self._last_known_url = ""
        self._page_listing_count = 0
        self.scroll_state_calls: list[dict] = []
        self.holo_fallback_called = False
        self._holo_result_success = True

    @property
    def _listings_on_page(self) -> int:
        return self._page_listing_count

    @_listings_on_page.setter
    def _listings_on_page(self, value: int) -> None:
        self._page_listing_count = value

    def _set_scroll_state(self, **kwargs) -> None:
        self.scroll_state_calls.append(kwargs)

    def _current_results_page_url(self) -> str:
        return self._results_base_url

    def _execute_holo3_step(self, step, index):
        from mantis_agent.gym.checkpoint import StepResult
        self.holo_fallback_called = True
        return StepResult(step_index=index, intent=step.intent, success=self._holo_result_success)


def _ctx(runner, *, env=None, extractor=None, grounding=None, site_config=None) -> StepContext:
    return StepContext(
        env=env or MagicMock(),
        brain=None,
        extractor=extractor or MagicMock(),
        grounding=grounding,
        cost_meter=None,
        dynamic_verifier=MagicMock(),
        scanner=None,
        site_config=site_config or MagicMock(),
        tool_channel=None,
        extraction_cache=None,
        state={"index": 9},
    )


def _step() -> MicroIntent:
    return MicroIntent(intent="Go to next page", type="paginate")


# ── Layer 1: URL-based ──────────────────────────────────────────────


def test_layer1_url_pagination_success(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_handlers.paginate.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    site_config = MagicMock()
    site_config.pagination_format = "/page-{n}"
    site_config.paginated_url.return_value = "https://example.com/cars/page-2"
    env = MagicMock()
    ctx = _ctx(runner, env=env, site_config=site_config)

    result = PaginateHandler(runner).execute(_step(), ctx)

    assert result.success is True
    assert result.data == "url_paginate_page2"
    assert runner._current_page == 2
    assert runner._last_known_url == "https://example.com/cars/page-2"
    env.reset.assert_called_once_with(task="paginate_url", start_url="https://example.com/cars/page-2")
    # Layers 2/3 not invoked
    assert runner.holo_fallback_called is False
    record = ctx.dynamic_verifier.record_pagination.call_args
    assert record.kwargs["method"] == "url"
    assert record.kwargs["success"] is True


def test_layer1_skipped_when_pagination_format_unset_falls_to_layer2(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_handlers.paginate.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    site_config = MagicMock()
    site_config.pagination_format = ""  # disabled
    extractor = MagicMock()
    # Layer 2 finds Next button immediately
    extractor.find_paginate_target.return_value = (500, 600, "Next")
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor, site_config=site_config)

    result = PaginateHandler(runner).execute(_step(), ctx)

    assert result.success is True
    assert runner._current_page == 2
    # No env.reset call (Layer 1 skipped)
    env.reset.assert_not_called()


def test_layer1_failure_falls_through_to_layer2(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_handlers.paginate.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    site_config = MagicMock()
    site_config.pagination_format = "/page-{n}"
    site_config.paginated_url.return_value = "https://example.com/cars/page-2"
    env = MagicMock()
    env.reset.side_effect = RuntimeError("network blip")
    extractor = MagicMock()
    extractor.find_paginate_target.return_value = (500, 600, "Next")
    ctx = _ctx(runner, env=env, extractor=extractor, site_config=site_config)

    result = PaginateHandler(runner).execute(_step(), ctx)

    assert result.success is True  # Layer 2 saved us
    # Two pagination records: first url=fail, then claude_guided=success
    calls = ctx.dynamic_verifier.record_pagination.call_args_list
    assert len(calls) == 2
    assert calls[0].kwargs["method"] == "url"
    assert calls[0].kwargs["success"] is False
    assert calls[1].kwargs["method"] == "claude_guided"
    assert calls[1].kwargs["success"] is True


# ── Layer 2: Claude-guided ──────────────────────────────────────────


def test_layer2_claude_target_found_clicks_and_resets_listings_count(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_handlers.paginate.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    runner._page_listing_count = 7  # pre-set: should reset to 0
    site_config = MagicMock()
    site_config.pagination_format = ""
    extractor = MagicMock()
    extractor.find_paginate_target.return_value = (250, 700, "Next")
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor, site_config=site_config)

    result = PaginateHandler(runner).execute(_step(), ctx)

    assert result.success is True
    assert runner._listings_on_page == 0  # reset for new page
    assert runner.costs["gpu_steps"] == 1
    assert runner.costs["gpu_seconds"] == 4
    assert runner.costs["claude_extract"] == 1


def test_layer2_target_not_found_after_3_attempts_falls_to_layer3(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_handlers.paginate.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    runner._holo_result_success = True
    site_config = MagicMock()
    site_config.pagination_format = ""
    extractor = MagicMock()
    extractor.find_paginate_target.return_value = ("not_found",)
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor, site_config=site_config)

    result = PaginateHandler(runner).execute(_step(), ctx)

    assert result.success is True
    # 3 Layer 2 attempts billed + Holo3 fallback (which succeeded)
    assert runner.costs["claude_extract"] == 3
    assert runner.holo_fallback_called is True
    # Last record_pagination call: holo3 success
    last_call = ctx.dynamic_verifier.record_pagination.call_args_list[-1]
    assert last_call.kwargs["method"] == "holo3"
    assert last_call.kwargs["success"] is True


# ── Layer 3: Holo3 fallback ─────────────────────────────────────────


def test_layer3_failure_records_all_layers_failure(monkeypatch):
    monkeypatch.setattr("mantis_agent.gym.step_handlers.paginate.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    runner._holo_result_success = False  # last layer also fails
    site_config = MagicMock()
    site_config.pagination_format = ""
    extractor = MagicMock()
    extractor.find_paginate_target.return_value = ("not_found",)
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor, site_config=site_config)

    result = PaginateHandler(runner).execute(_step(), ctx)

    assert result.success is False
    assert runner.holo_fallback_called is True
    # Final pagination record: all_layers + reason
    last_call = ctx.dynamic_verifier.record_pagination.call_args_list[-1]
    assert last_call.kwargs["method"] == "all_layers"
    assert last_call.kwargs["reason"] == "next_control_not_found"


# ── Misc ────────────────────────────────────────────────────────────


def test_step_type_property():
    handler = PaginateHandler(_FakeRunner())
    assert handler.step_type == "paginate"
