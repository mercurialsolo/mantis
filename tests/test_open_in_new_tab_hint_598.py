"""Tests for issue #598 — ``open_in_new_tab`` hint plumbing.

Two-part plumbing:

1. Plan decomposer post-process — when source plan prose contains
   "open in new tab" / "Ctrl+click" / "right-click" patterns, set
   ``hints.open_in_new_tab=True`` on every extraction-section click
   step. Idempotent (safe on cache reloads).

2. Click handler — when ``step.hints.open_in_new_tab`` is True,
   dispatch middle-click as the PRIMARY click. On success, mirrors
   the existing middle-click fallback contract (sets
   ``_opened_detail_in_new_tab=True`` so navigate_back routes via
   Ctrl+W, returns success with ``executor_backend="middle_primary"``).
   On failure, returns ``None`` so the caller falls through to the
   legacy plain-click chain (which still has its own middle-click
   fallback) — a missed middle-click can't drop a listing.

Screenshot correctness (the #598 ask): after middle-click +
``ctrl+Tab`` to switch focus to the new tab, ``env.screenshot()``
captures the active window — which is the new tab. The helper uses
the same ``ctrl+Tab`` + ``env.screenshot()`` sequence that the
existing FALLBACK path uses, so the screenshot routing contract is
unchanged. Pinned indirectly by asserting the helper sets
``_opened_detail_in_new_tab=True`` on success (the runner reads that
flag everywhere it needs to know the screenshot is from the new tab).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mantis_agent.actions import ActionType
from mantis_agent.gym.step_handlers.click import (
    ClaudeGuidedClickHandler,
    _try_open_in_new_tab_primary,
)
from mantis_agent.plan_decomposer import MicroIntent, MicroPlan, PlanDecomposer

from test_click_handler import _ctx, _FakeRunner  # type: ignore[import-not-found]


# ── plan_decomposer: hint application ───────────────────────────────


def _plan_with_click(intent: str, *, source_plan: str = "") -> MicroPlan:
    plan = MicroPlan(source_plan=source_plan)
    plan.steps.append(MicroIntent(intent="navigate", type="navigate", section="setup"))
    plan.steps.append(MicroIntent(intent=intent, type="click", section="extraction"))
    plan.steps.append(MicroIntent(intent="extract_data", type="extract_data", section="extraction"))
    for i, s in enumerate(plan.steps):
        s.index = i
    return plan


@pytest.mark.parametrize(
    "prose",
    [
        "Open the listing in a new tab.",
        "Right-click the listing title and select Open in new tab.",
        "Ctrl+click the title to open it in a new tab (Tab 2).",
        "Middle-click the next listing.",
    ],
)
def test_source_plan_prose_sets_hint_on_extraction_click(prose: str) -> None:
    plan = _plan_with_click("Click the next listing", source_plan=prose)
    PlanDecomposer._apply_open_in_new_tab_hint(plan)
    click_step = plan.steps[1]
    assert click_step.hints.get("open_in_new_tab") is True


def test_step_intent_prose_alone_also_sets_hint() -> None:
    """The intent text on the click step itself can match even when
    the source plan is empty — e.g. plans that compile straight into
    step intents without preserving a separate source narrative."""
    plan = _plan_with_click(
        "Right-click the listing title to open in a new tab",
        source_plan="",
    )
    PlanDecomposer._apply_open_in_new_tab_hint(plan)
    assert plan.steps[1].hints.get("open_in_new_tab") is True


def test_no_match_leaves_hint_unset() -> None:
    """Plans without any new-tab prose must NOT get the hint —
    generic 'click' steps still use the plain-click primary."""
    plan = _plan_with_click(
        "Click the listing title to open the detail page",
        source_plan="Click each listing in turn to extract its data.",
    )
    PlanDecomposer._apply_open_in_new_tab_hint(plan)
    assert plan.steps[1].hints.get("open_in_new_tab") is None


def test_setup_section_click_not_flagged() -> None:
    """Only extraction-section clicks qualify. A setup-section click
    (cookie banner, sign-in) must NOT get the hint even when the
    plan also has extraction-section open-in-new-tab prose."""
    plan = MicroPlan(source_plan="Open listings in a new tab.")
    plan.steps.append(MicroIntent(
        intent="Dismiss cookie banner", type="click", section="setup",
    ))
    plan.steps.append(MicroIntent(
        intent="Click listing", type="click", section="extraction",
    ))
    for i, s in enumerate(plan.steps):
        s.index = i
    PlanDecomposer._apply_open_in_new_tab_hint(plan)
    # Setup-section click: untouched.
    assert plan.steps[0].hints.get("open_in_new_tab") is None
    # Extraction-section click: flagged.
    assert plan.steps[1].hints.get("open_in_new_tab") is True


def test_no_extraction_click_is_a_no_op() -> None:
    """Plans without an extraction-click step shouldn't raise even
    when the source plan has the prose pattern (e.g. workflow plans
    where every action is fill_field / submit)."""
    plan = MicroPlan(source_plan="Open in new tab is mentioned but no click step exists.")
    plan.steps.append(MicroIntent(
        intent="fill", type="fill_field", section="setup",
    ))
    for i, s in enumerate(plan.steps):
        s.index = i
    PlanDecomposer._apply_open_in_new_tab_hint(plan)
    # Field step's hints stay empty.
    assert plan.steps[0].hints.get("open_in_new_tab") is None


def test_idempotent_on_repeat_application() -> None:
    """Cache load path runs the post-process again. Re-applying must
    not double-add or change the existing hint."""
    plan = _plan_with_click("click", source_plan="Open link in new tab.")
    PlanDecomposer._apply_open_in_new_tab_hint(plan)
    PlanDecomposer._apply_open_in_new_tab_hint(plan)  # second pass
    PlanDecomposer._apply_open_in_new_tab_hint(plan)  # third pass
    assert plan.steps[1].hints == {"open_in_new_tab": True}


def test_pattern_case_insensitive() -> None:
    """The prose pattern check is lower-cased so mixed-case
    capitalisation in real plans matches."""
    plan = _plan_with_click("click", source_plan="OPEN IN NEW TAB on each row.")
    PlanDecomposer._apply_open_in_new_tab_hint(plan)
    assert plan.steps[1].hints.get("open_in_new_tab") is True


def test_partial_word_does_not_match() -> None:
    """Loose patterns like the bare word 'tab' must NOT trigger the
    hint — the narrower phrases are intentional. 'click the filters
    tab' shouldn't compile to middle-click on the filter tab."""
    plan = _plan_with_click(
        "Click the filters tab to open the dropdown",
        source_plan="Open the filters tab to refine results.",
    )
    PlanDecomposer._apply_open_in_new_tab_hint(plan)
    assert plan.steps[1].hints.get("open_in_new_tab") is None


# ── click handler: hint-driven primary middle-click ─────────────────


def _step(*, hint: bool = False) -> MicroIntent:
    hints = {"open_in_new_tab": True} if hint else {}
    return MicroIntent(
        intent="Click listing", type="click", section="extraction",
        hints=hints,
    )


def _middle_click_count(env: MagicMock) -> int:
    return sum(
        1
        for c in env.step.call_args_list
        if getattr(c.args[0], "action_type", None) == ActionType.CLICK
        and (c.args[0].params or {}).get("button") == "middle"
    )


def _plain_click_count(env: MagicMock) -> int:
    return sum(
        1
        for c in env.step.call_args_list
        if getattr(c.args[0], "action_type", None) == ActionType.CLICK
        and (c.args[0].params or {}).get("button") in (None, "left")
    )


def test_helper_no_op_when_hint_unset() -> None:
    """No hint → helper returns None so the caller proceeds with
    the legacy plain-click chain. No middle-click dispatched."""
    runner = _FakeRunner()
    env = MagicMock()
    env.screen_size = (1280, 800)
    ctx = _ctx(runner, env=env)
    ctx.state["_executor_backend"] = "vision"
    result = _try_open_in_new_tab_primary(
        handler=None, runner=runner, env=env, ctx=ctx,
        step=_step(hint=False), index=4, x=100, y=200,
        title="Card A", site_config=ctx.site_config,
        dynamic_verifier=ctx.dynamic_verifier,
    )
    assert result is None
    assert _middle_click_count(env) == 0


def test_helper_middle_clicks_when_hint_set_and_landed_on_detail(monkeypatch) -> None:
    """Hint set + middle-click opens a detail page on the SAME tab
    (URL becomes detail-page on the first verify attempt). Helper
    returns StepResult success with executor_backend=middle_primary,
    sets _opened_detail_in_new_tab=True."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.click.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    # The production caller sets this at line 252 of click.py before
    # the helper runs; mirror that so the helper's extracted_titles
    # accumulator (which uses _last_click_title, same as the existing
    # fallback success path) has a value to append.
    runner._last_click_title = "1997 Caroff CHATAM 52"
    env = MagicMock()
    env.screen_size = (1280, 800)
    # Tabs go 1 → 2 (real new tab opened) so the flag is set.
    env.cdp_count_pages = MagicMock(side_effect=[1, 2])
    ctx = _ctx(runner, env=env)
    ctx.site_config.is_detail_page = lambda url, base_url=None: "/boat/" in url
    runner._read_current_url = MagicMock(return_value="https://boattrader.com/boat/1997-caroff/")

    result = _try_open_in_new_tab_primary(
        handler=None, runner=runner, env=env, ctx=ctx,
        step=_step(hint=True), index=4, x=499, y=187,
        title="1997 Caroff CHATAM 52", site_config=ctx.site_config,
        dynamic_verifier=ctx.dynamic_verifier,
    )

    assert result is not None
    assert result.success is True
    assert _middle_click_count(env) == 1
    # No plain-click was attempted — middle-click was PRIMARY.
    assert _plain_click_count(env) == 0
    # Critical (#598 screenshot routing): flag is set so navigate_back
    # routes via execute_close_detail_tab (ctrl+w on the new tab) and
    # subsequent screenshots come from the active (new) tab.
    assert runner._opened_detail_in_new_tab is True
    assert ctx.state["_executor_backend"] == "middle_primary"
    # Title accumulated for next-iteration scan prefilter (#597).
    assert "1997 Caroff CHATAM 52" in runner._extracted_titles


def test_helper_in_place_navigation_does_not_set_new_tab_flag(monkeypatch) -> None:
    """Issue #598 follow-up: when middle-click NAVIGATES IN-PLACE
    (some sites preventDefault on auxclick), tab count stays the same.
    Helper must NOT set ``_opened_detail_in_new_tab=True`` — otherwise
    the subsequent navigate_back would route via Ctrl+W and close the
    SOURCE tab, breaking the session.

    Observed in production run 20260523_055833_572c0b0e — 2 of 7 leads
    had results-page URLs because the flag was set when middle-click
    actually navigated in-place, and subsequent extract ran on the
    wrong tab. The tab-count diff gates the flag now."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.click.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    runner._last_click_title = "Caroff"
    env = MagicMock()
    env.screen_size = (1280, 800)
    # Tabs stay at 1 — middle-click navigated in-place.
    env.cdp_count_pages = MagicMock(side_effect=[1, 1])
    ctx = _ctx(runner, env=env)
    ctx.site_config.is_detail_page = lambda url, base_url=None: "/boat/" in url
    runner._read_current_url = MagicMock(return_value="https://boattrader.com/boat/1997-caroff/")

    result = _try_open_in_new_tab_primary(
        handler=None, runner=runner, env=env, ctx=ctx,
        step=_step(hint=True), index=4, x=499, y=187,
        title="Caroff", site_config=ctx.site_config,
        dynamic_verifier=ctx.dynamic_verifier,
    )

    assert result is not None
    assert result.success is True
    # Critical (#598 in-place fix): NO new tab → flag stays False so
    # navigate_back routes via mechanical CDP back (#609) instead of
    # Ctrl+W (which would close the source tab).
    assert runner._opened_detail_in_new_tab is False
    # Last-known URL still gets set (the runner needs it for dedup
    # / mark-seen plumbing); only the new-tab routing flag is gated.
    assert runner._last_known_url == "https://boattrader.com/boat/1997-caroff/"


def test_helper_switches_to_new_tab_then_finds_detail(monkeypatch) -> None:
    """Middle-click opened a new tab; first URL read on the SOURCE
    tab is still the results page, so helper sends ctrl+Tab + retries
    — second URL read on the NEW tab is the detail page. Verifies
    the screenshot will come from the right tab after the switch."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.click.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    env = MagicMock()
    env.screen_size = (1280, 800)
    # Tabs 1 → 2: a real new tab opened.
    env.cdp_count_pages = MagicMock(side_effect=[1, 2])
    ctx = _ctx(runner, env=env)
    ctx.site_config.is_detail_page = lambda url, base_url=None: "/boat/" in url
    # First call (source tab): results page; second call (after
    # ctrl+Tab to new tab): detail page.
    runner._read_current_url = MagicMock(side_effect=[
        "https://boattrader.com/boats/by-owner/",
        "https://boattrader.com/boat/1997-caroff/",
    ])

    result = _try_open_in_new_tab_primary(
        handler=None, runner=runner, env=env, ctx=ctx,
        step=_step(hint=True), index=4, x=499, y=187,
        title="Caroff", site_config=ctx.site_config,
        dynamic_verifier=ctx.dynamic_verifier,
    )

    assert result is not None
    assert result.success is True
    # ctrl+Tab dispatched between the two URL checks.
    keypresses = [
        c.args[0].params.get("keys")
        for c in env.step.call_args_list
        if getattr(c.args[0], "action_type", None) == ActionType.KEY_PRESS
    ]
    assert "ctrl+Tab" in keypresses
    assert runner._opened_detail_in_new_tab is True


def test_helper_blank_newtab_closes_and_fails(monkeypatch) -> None:
    """Middle-click landed on chrome://newtab/ (e.g. clicked card
    whitespace not a link). Helper closes the tab via ctrl+w and
    returns failure — does NOT fall through (a plain-click on the
    same coords wouldn't navigate either)."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.click.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    env = MagicMock()
    env.screen_size = (1280, 800)
    ctx = _ctx(runner, env=env)
    ctx.site_config.is_detail_page = lambda url, base_url=None: "/boat/" in url
    runner._read_current_url = MagicMock(return_value="chrome://newtab/")

    result = _try_open_in_new_tab_primary(
        handler=None, runner=runner, env=env, ctx=ctx,
        step=_step(hint=True), index=4, x=100, y=200,
        title="Empty card", site_config=ctx.site_config,
        dynamic_verifier=ctx.dynamic_verifier,
    )

    assert result is not None
    assert result.success is False
    assert result.data == "newtab_blank"
    # ctrl+w sent to clean up the empty tab.
    keypresses = [
        c.args[0].params.get("keys")
        for c in env.step.call_args_list
        if getattr(c.args[0], "action_type", None) == ActionType.KEY_PRESS
    ]
    assert "ctrl+w" in keypresses


def test_helper_no_new_tab_opened_falls_through(monkeypatch) -> None:
    """Middle-click fired but neither the source nor the new tab
    reports a detail-page URL after 2 verify attempts. Helper
    returns None so the caller falls through to the legacy plain-
    click chain — middle-click may have been off-target and a
    refined-grounding plain-click could still land."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.click.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    env = MagicMock()
    env.screen_size = (1280, 800)
    ctx = _ctx(runner, env=env)
    ctx.site_config.is_detail_page = lambda url, base_url=None: "/boat/" in url
    # Both tabs report results page → no detail landing.
    runner._read_current_url = MagicMock(return_value="https://boattrader.com/boats/by-owner/")

    result = _try_open_in_new_tab_primary(
        handler=None, runner=runner, env=env, ctx=ctx,
        step=_step(hint=True), index=4, x=100, y=200,
        title="Card B", site_config=ctx.site_config,
        dynamic_verifier=ctx.dynamic_verifier,
    )

    assert result is None
    # _opened_detail_in_new_tab NOT set — the caller's chain runs.
    assert runner._opened_detail_in_new_tab is False
    # Title NOT yet accumulated — that happens on the caller's
    # eventual success.
    assert "Card B" not in runner._extracted_titles


def test_helper_dispatch_exception_falls_through(monkeypatch) -> None:
    """env.step() raising on the middle-click dispatch must let the
    helper fall through to the legacy chain — a flaky CDP socket
    shouldn't drop the listing."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.click.time.sleep", lambda *_: None)

    runner = _FakeRunner()
    env = MagicMock()
    env.screen_size = (1280, 800)
    env.step = MagicMock(side_effect=RuntimeError("websocket disconnected"))
    ctx = _ctx(runner, env=env)
    ctx.site_config.is_detail_page = lambda url, base_url=None: True

    result = _try_open_in_new_tab_primary(
        handler=None, runner=runner, env=env, ctx=ctx,
        step=_step(hint=True), index=4, x=100, y=200,
        title="Card C", site_config=ctx.site_config,
        dynamic_verifier=ctx.dynamic_verifier,
    )

    assert result is None
    assert runner._opened_detail_in_new_tab is False


# ── execute() routes through helper when hint set ──────────────────


def test_execute_dispatches_middle_primary_when_hint_set(monkeypatch) -> None:
    """End-to-end through ClaudeGuidedClickHandler.execute: hint set,
    a card is found, middle-click opens detail → no plain-click is
    fired, _opened_detail_in_new_tab=True on the runner."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.click.time.sleep", lambda *_: None)
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.click.random.uniform", lambda *_: 0.0,
    )

    runner = _FakeRunner()
    runner._max_viewport_stages = 1
    extractor = MagicMock()
    extractor.find_all_listings.return_value = [(499, 187, "Caroff")]
    env = MagicMock()
    env.screen_size = (1280, 800)
    # End-to-end through execute(): a real new tab is opened so the
    # tab-count diff check passes and the routing flag is set.
    env.cdp_count_pages = MagicMock(side_effect=[1, 1, 2])
    ctx = _ctx(runner, env=env, extractor=extractor)
    ctx.site_config.is_detail_page = lambda url, base_url=None: "/boat/" in url
    runner._read_current_url = MagicMock(return_value="https://boattrader.com/boat/caroff/")

    result = ClaudeGuidedClickHandler(runner).execute(_step(hint=True), ctx)

    assert result.success is True
    assert _middle_click_count(env) == 1
    assert _plain_click_count(env) == 0
    assert runner._opened_detail_in_new_tab is True
