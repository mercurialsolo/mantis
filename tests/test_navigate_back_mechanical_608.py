"""Tests for issue #608 — mechanical CDP-back primary path for the
``navigate_back`` step type.

Before the fix, ``navigate_back`` routed through ``Holo3StepHandler``
which gave the brain an open-ended N-step inner loop. The brain
typically burnt ~8 budget steps before declaring failure, then
``step_recovery`` fired up to 3 CDP-back/Alt+Left attempts with a
Claude extract per attempt. Live repro: boattrader run
20260523_041241_b12b4194 — 11 navigate_back failures, ~$0.83 +
~88 seconds wasted across the run.

These tests pin: the mechanical handler tries ``env.cdp_history_back``,
verifies the URL moved off the detail page, and returns success on
the happy path. They also pin the fall-through conditions (no CDP,
new-tab mode, no URL change, landed on another detail page) so the
brain still owns the cases the brain is needed for.

The dispatcher wiring (mechanical first, Holo3 on fall-through) is
exercised by the integration tests in ``test_step_registry.py`` —
unit tests here cover the handler in isolation.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from mantis_agent.gym.step_context import StepContext
from mantis_agent.gym.step_handlers.navigate_back import (
    MechanicalNavigateBackHandler,
)
from mantis_agent.plan_decomposer import MicroIntent


class _FakeRunner:
    """Minimal back-reference. Tests poke ``env``, ``site_config``,
    ``_best_effort_current_url``, ``_last_known_url``, and
    ``_opened_detail_in_new_tab``.
    """

    def __init__(self) -> None:
        self.env = MagicMock()
        self.site_config = MagicMock()
        self._opened_detail_in_new_tab = False
        self._last_known_url = ""
        self._url_seq: list[str] = []

    def _best_effort_current_url(self) -> str:
        if not self._url_seq:
            return ""
        if len(self._url_seq) == 1:
            return self._url_seq[0]
        return self._url_seq.pop(0)


def _ctx(runner: _FakeRunner) -> StepContext:
    return StepContext(
        env=runner.env,
        brain=None,
        extractor=None,
        grounding=None,
        cost_meter=None,
        dynamic_verifier=MagicMock(),
        scanner=None,
        site_config=runner.site_config,
        tool_channel=None,
        extraction_cache=None,
        state={"index": 6},
    )


def _step() -> MicroIntent:
    return MicroIntent(
        intent="Return to the search results page",
        type="navigate_back", section="extraction",
    )


# ── applies_to gates ───────────────────────────────────────────────


def test_applies_to_when_cdp_back_available_and_not_new_tab():
    runner = _FakeRunner()
    runner.env.cdp_history_back = lambda **_: True
    handler = MechanicalNavigateBackHandler(runner)
    assert handler.applies_to(_step()) is True


def test_does_not_apply_when_env_has_no_cdp_back():
    """Test stubs / older env mocks without cdp_history_back fall
    through to Holo3."""
    runner = _FakeRunner()
    # MagicMock-default env: every attribute is itself a MagicMock,
    # which IS callable — defeats the ``callable(...)`` check via
    # falsy intent. Explicitly delete the attr.
    del runner.env.cdp_history_back
    runner.env = type("E", (), {})()  # plain object, no cdp_history_back
    handler = MechanicalNavigateBackHandler(runner)
    assert handler.applies_to(_step()) is False


def test_does_not_apply_when_opened_detail_in_new_tab():
    """When the runner opened the detail in a new tab, the executor's
    inline ``execute_close_detail_tab`` path handles back navigation
    via ctrl+w. Mechanical CDP back would try to history.back() on the
    new tab which has no listings-page history entry."""
    runner = _FakeRunner()
    runner.env.cdp_history_back = lambda **_: True
    runner._opened_detail_in_new_tab = True
    handler = MechanicalNavigateBackHandler(runner)
    assert handler.applies_to(_step()) is False


def test_does_not_apply_when_params_brain_required():
    """Plans / runtime configs can opt out of mechanical back via
    ``params.brain_required=True`` for sites with intercepted history."""
    runner = _FakeRunner()
    runner.env.cdp_history_back = lambda **_: True
    handler = MechanicalNavigateBackHandler(runner)
    step = MicroIntent(
        intent="back", type="navigate_back",
        params={"brain_required": True},
    )
    assert handler.applies_to(step) is False


# ── execute happy path ─────────────────────────────────────────────


def test_cdp_back_to_results_page_returns_success():
    """CDP back changes URL from detail to results → handler returns
    success without invoking the brain."""
    runner = _FakeRunner()
    runner._url_seq = [
        "https://boattrader.com/boat/1997-caroff-chatam-52-10130796/",  # pre
        "https://boattrader.com/boats/state-fl/city-miami/by-owner/",   # post
    ]
    runner.env.cdp_history_back = MagicMock(return_value=True)
    runner.site_config.is_detail_page = lambda url: "/boat/" in url

    result = MechanicalNavigateBackHandler(runner).execute(_step(), _ctx(runner))

    assert result.success is True
    assert result.data.startswith("back_via_cdp:")
    assert "by-owner" in result.data
    # cdp_history_back called once with settle.
    assert runner.env.cdp_history_back.call_count == 1
    # Runner's last_known_url updated to the post URL.
    assert "by-owner" in runner._last_known_url


# ── execute fall-through cases ─────────────────────────────────────


def test_cdp_back_no_url_change_returns_failure():
    """CDP back didn't move the URL (empty history, intercepted call,
    SPA without history support) → return failure so dispatcher falls
    through to Holo3."""
    runner = _FakeRunner()
    runner._url_seq = ["https://boattrader.com/boat/1997-caroff-chatam-52-10130796/"]
    runner.env.cdp_history_back = MagicMock(return_value=False)
    runner.site_config.is_detail_page = lambda url: "/boat/" in url

    result = MechanicalNavigateBackHandler(runner).execute(_step(), _ctx(runner))

    assert result.success is False
    assert result.data == "cdp_back_no_url_change"
    assert result.failure_class == "cdp_back_no_change"


def test_cdp_back_to_another_detail_page_returns_failure():
    """CDP back moved the URL, but to another detail page (history had
    multiple detail entries). Brain should re-attempt or use a
    different strategy."""
    runner = _FakeRunner()
    runner._url_seq = [
        "https://boattrader.com/boat/1997-caroff-chatam-52-10130796/",  # pre
        "https://boattrader.com/boat/1987-beneteau-idylle-15-50-10139/",  # post — STILL detail
    ]
    runner.env.cdp_history_back = MagicMock(return_value=True)
    runner.site_config.is_detail_page = lambda url: "/boat/" in url

    result = MechanicalNavigateBackHandler(runner).execute(_step(), _ctx(runner))

    assert result.success is False
    assert result.data.startswith("back_to_detail_page:")
    assert result.failure_class == "back_to_detail_page"


def test_cdp_back_raises_exception_returns_failure():
    """If cdp_history_back raises (e.g. CDP socket closed mid-call),
    handler returns a failure with a labeled failure_class — caller
    falls through to Holo3."""
    runner = _FakeRunner()
    runner._url_seq = ["https://boattrader.com/boat/x/"]

    def _boom(**_kwargs: Any) -> bool:
        raise RuntimeError("websocket disconnected")

    runner.env.cdp_history_back = _boom
    runner.site_config.is_detail_page = lambda url: "/boat/" in url

    result = MechanicalNavigateBackHandler(runner).execute(_step(), _ctx(runner))

    assert result.success is False
    assert result.data.startswith("cdp_back_exception:")
    assert "RuntimeError" in result.data
    assert result.failure_class == "cdp_back_exception"


def test_step_type_property():
    handler = MechanicalNavigateBackHandler(_FakeRunner())
    assert handler.step_type == "navigate_back"


# ── dispatcher integration: registry routes navigate_back through
# the dispatcher; mechanical fires first on the happy path ─────────


def test_registry_routes_navigate_back_through_dispatcher_mechanical_wins(monkeypatch):
    """End-to-end through the registry: a runner whose env supports
    CDP back and whose URL moves off detail dispatches via the
    mechanical handler — the Holo3 brain handler is NOT invoked."""
    from mantis_agent.gym.step_handlers import default_registry

    runner = _FakeRunner()
    runner._url_seq = [
        "https://boattrader.com/boat/x/",
        "https://boattrader.com/boats/by-owner/",
    ]
    runner.env.cdp_history_back = MagicMock(return_value=True)
    runner.site_config.is_detail_page = lambda url: "/boat/" in url
    # Other runner attributes the registry constructor's handlers may
    # poke at construction time — stub generously.
    runner.scanner = MagicMock()
    runner.extractor = None
    runner.costs = {}
    runner.brain = MagicMock()
    runner.dynamic_verifier = MagicMock()
    runner.tool_channel = None
    runner.grounding = None

    reg = default_registry(runner)
    handler = reg.get("navigate_back")
    assert handler is not None
    assert handler.step_type == "navigate_back"
    result = handler.execute(_step(), _ctx(runner))
    assert result.success is True
    assert result.data.startswith("back_via_cdp:")
    # Brain was NOT consulted — mechanical took the request.
    runner.brain.assert_not_called()
