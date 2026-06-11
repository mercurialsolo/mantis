"""Tests for ``step_handlers.default_registry`` — Phase 2 cleanup of EPIC #161.

Pins which step types the default registry covers and which handler
class each maps to. Adding a new step type or unbinding an existing
one is a downstream-visible change (``MicroPlanRunner._execute_step``
dispatch routes through this registry); these tests catch a careless
rename or deletion.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from mantis_agent.gym.step_handlers import (
    ClaudeGuidedClickHandler,
    ClaudeGuidedFilterHandler,
    ClaudeGuidedFormHandler,
    ClaudeStepHandler,
    Holo3StepHandler,
    MechanicalNavigateBackHandler,
    MechanicalScrollHandler,
    NavigateHandler,
    PaginateHandler,
    default_registry,
)
from mantis_agent.plan_decomposer import MicroIntent


def _registry():
    """Build a registry against a bare runner stub.

    Handlers only consult the runner via parent back-reference at
    ``execute()`` time; construction is back-reference-stash-only, so a
    ``MagicMock`` is a sufficient parent for registry-shape tests.
    """
    return default_registry(MagicMock())


def test_registry_covers_every_step_type_phase2_lifted():
    """Every step type whose handler has been extracted under #161 Phase 2
    must be present in the default registry."""
    reg = _registry()
    expected = {
        "navigate", "click", "paginate", "filter",
        "fill_field", "submit", "select_option", "right_click",
        "extract_url", "extract_data",
        # #785 follow-up: multi-row extraction in one Claude call (HN
        # top-N pattern). Routes through the same ClaudeStepHandler
        # as extract_url / extract_data via the multi-row branch.
        "extract_rows",
        "scroll", "navigate_back",
        # #615: collect_urls primitive — single-pass listing URL harvest
        # for the fan-out runner (#616, #617).
        "collect_urls",
        # #643 stage 2: vision-only yes/no element existence check —
        # binds boolean to runner._state_vars for ``step.guard`` to
        # consume on subsequent conditional steps.
        "detect_visible",
        # User-bug fix: plan-text accessible bridge to the runner's
        # ``request_user_input`` host tool — emits the deterministic
        # pause/resume hook from a decomposed plan_text plan.
        "request_user_input",
    }
    assert set(reg.types()) == expected


def test_navigate_binds_to_NavigateHandler():
    assert isinstance(_registry().get("navigate"), NavigateHandler)


def test_click_binds_to_ClaudeGuidedClickHandler():
    assert isinstance(_registry().get("click"), ClaudeGuidedClickHandler)


def test_paginate_binds_to_PaginateHandler():
    assert isinstance(_registry().get("paginate"), PaginateHandler)


def test_filter_binds_to_ClaudeGuidedFilterHandler():
    assert isinstance(_registry().get("filter"), ClaudeGuidedFilterHandler)


def test_form_types_share_one_handler_instance():
    """fill_field / submit / select_option all bind to the SAME handler
    instance — that's the multi-binding contract."""
    reg = _registry()
    fill = reg.get("fill_field")
    submit = reg.get("submit")
    select = reg.get("select_option")
    assert isinstance(fill, ClaudeGuidedFormHandler)
    assert fill is submit is select


def test_claude_step_types_share_one_handler_instance():
    """extract_url / extract_data / extract_rows all bind to the SAME
    ClaudeStepHandler instance — Claude-only steps share state via
    the handler's parent-runner back-reference."""
    reg = _registry()
    url = reg.get("extract_url")
    data = reg.get("extract_data")
    rows = reg.get("extract_rows")
    assert isinstance(url, ClaudeStepHandler)
    assert url is data is rows


def test_navigate_back_uses_dispatcher_routing_to_mechanical_or_holo3():
    """``navigate_back`` was promoted from a Holo3-only step type to
    a dispatcher (issue #608). Mechanical CDP-back fires first when
    the env supports it; Holo3 owns the fall-through (no CDP, new-tab
    mode, no URL change, landed on another detail page).

    Previously the brain ran an N-step loop on every navigate_back
    step, declared failure, and step_recovery did 3× CDP+Alt+Left
    attempts. The mechanical primary skips ~$0.04 + ~8 sec / iter.
    """
    reg = _registry()
    dispatcher = reg.get("navigate_back")
    assert dispatcher is not None
    # Dispatcher is NOT the Holo3 instance directly.
    assert not isinstance(dispatcher, Holo3StepHandler)
    # Dispatcher exposes the right step_type marker for registry binding.
    assert dispatcher.step_type == "navigate_back"


def test_mechanical_navigate_back_handler_applies_to_predicate():
    """The mechanical handler claims a navigate_back step when the
    env exposes ``cdp_history_back`` AND the runner isn't in new-tab
    mode AND the plan didn't opt out via ``params.brain_required``."""
    runner = MagicMock()
    runner._opened_detail_in_new_tab = False
    runner.env = MagicMock()
    runner.env.cdp_history_back = lambda **_: True
    mech = MechanicalNavigateBackHandler(runner)

    assert mech.applies_to(MicroIntent(intent="x", type="navigate_back"))
    # New-tab mode → existing close-tab path owns it.
    runner._opened_detail_in_new_tab = True
    assert not mech.applies_to(MicroIntent(intent="x", type="navigate_back"))
    runner._opened_detail_in_new_tab = False
    # Opt-out via params.brain_required.
    assert not mech.applies_to(
        MicroIntent(
            intent="x", type="navigate_back",
            params={"brain_required": True},
        )
    )
    # Env without cdp_history_back → fall through to Holo3.
    runner.env = type("E", (), {})()  # plain object, no cdp_history_back
    assert not mech.applies_to(MicroIntent(intent="x", type="navigate_back"))


def test_scroll_uses_dispatcher_routing_to_mechanical_or_holo3():
    """``scroll`` routes through a dispatcher that prefers the
    deterministic mechanical handler when ``params.count`` is set,
    falling through to Holo3 for goal-shaped scroll intents.

    This is the (a) Layer-1 fix: a ``scroll`` step with explicit
    count never goes through the brain, so the brain can't fall
    back to clicking visible elements (the "misclick on ad-link"
    pattern that drove every recent boattrader halt).
    """
    reg = _registry()
    dispatcher = reg.get("scroll")
    # Dispatcher is NOT the Holo3 instance directly; it's a wrapper.
    assert dispatcher is not None
    assert not isinstance(dispatcher, Holo3StepHandler)

    # Build a fake StepContext + env that records env.step dispatches.
    env = MagicMock()
    # Simulate scrollY going from 0 → 600 so the verification gate passes.
    env.cdp_evaluate = MagicMock(side_effect=[0.0, 600.0])
    ctx = MagicMock()
    ctx.env = env
    ctx.state = {"index": 0}

    # With params.count = 1 → mechanical path fires (env.step called).
    mech_step = MicroIntent(
        intent="scroll down once",
        type="scroll",
        params={"count": 1, "direction": "down"},
    )
    dispatcher.execute(mech_step, ctx)
    assert env.step.called, (
        "scroll with params.count should dispatch via mechanical "
        "handler (env.step), not the brain"
    )


def test_mechanical_scroll_handler_applies_to_predicate_gated_on_count():
    """The mechanical handler claims a step only when ``params.count``
    is present + positive. Steps without count fall through to Holo3
    so vision-mediated scrolling ("scroll until X visible") still
    works for plans that need it."""
    runner = MagicMock()
    mech = MechanicalScrollHandler(runner)
    assert mech.applies_to(
        MicroIntent(intent="x", type="scroll", params={"count": 1})
    )
    assert mech.applies_to(
        MicroIntent(intent="x", type="scroll", params={"count": 3, "direction": "up"})
    )
    # No count → Holo3 should own this step.
    assert not mech.applies_to(
        MicroIntent(intent="scroll until target visible", type="scroll", params={})
    )
    # Count of 0 or negative → not valid; defer to Holo3.
    assert not mech.applies_to(
        MicroIntent(intent="x", type="scroll", params={"count": 0})
    )
    # Non-numeric count → defer to Holo3.
    assert not mech.applies_to(
        MicroIntent(intent="x", type="scroll", params={"count": "many"})
    )


def test_unregistered_step_types_return_none():
    """Step types we deliberately don't claim (gate is a flag, not a
    type; loop is handled by the executor; unknown types fall back to
    Holo3 inline) return None from registry.get."""
    reg = _registry()
    assert reg.get("loop") is None
    assert reg.get("nonexistent_type") is None
