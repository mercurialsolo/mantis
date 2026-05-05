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
    NavigateHandler,
    PaginateHandler,
    default_registry,
)


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
        "fill_field", "submit", "select_option",
        "extract_url", "extract_data",
        "scroll", "navigate_back",
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
    """extract_url / extract_data both bind to the SAME ClaudeStepHandler
    instance — Claude-only steps share state via the handler's
    parent-runner back-reference."""
    reg = _registry()
    url = reg.get("extract_url")
    data = reg.get("extract_data")
    assert isinstance(url, ClaudeStepHandler)
    assert url is data


def test_holo3_types_share_one_handler_instance():
    """scroll / navigate_back fall through to Holo3 in legacy dispatch;
    after cleanup they share one Holo3StepHandler in the registry."""
    reg = _registry()
    scroll = reg.get("scroll")
    nav_back = reg.get("navigate_back")
    assert isinstance(scroll, Holo3StepHandler)
    assert scroll is nav_back


def test_unregistered_step_types_return_none():
    """Step types we deliberately don't claim (gate is a flag, not a
    type; loop is handled by the executor; unknown types fall back to
    Holo3 inline) return None from registry.get."""
    reg = _registry()
    assert reg.get("loop") is None
    assert reg.get("nonexistent_type") is None
