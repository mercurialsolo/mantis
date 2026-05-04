"""Tests for StepHandler protocol + StepContext + HandlerRegistry.

Phase 2 types-only commit (EPIC #161). The protocol is defined; no
production handler has been extracted yet, so these tests exercise the
contract via a minimal fake handler. The dispatch lift in a follow-up
commit will populate the registry with real handlers.
"""

from __future__ import annotations

from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.gym.listings_scanner import ListingsScanner
from mantis_agent.gym.step_context import (
    HandlerRegistry,
    StepContext,
    StepHandler,
)
from mantis_agent.plan_decomposer import MicroIntent


class _FakeHandler:
    """Minimal handler used only to exercise the protocol + registry."""

    def __init__(self, step_type: str, return_success: bool = True) -> None:
        self._type = step_type
        self._return_success = return_success
        self.calls: list[tuple[MicroIntent, StepContext]] = []

    @property
    def step_type(self) -> str:
        return self._type

    def execute(self, step: MicroIntent, ctx: StepContext) -> StepResult:
        self.calls.append((step, ctx))
        return StepResult(
            step_index=0,
            intent=step.intent,
            success=self._return_success,
            data="fake",
        )


def test_protocol_runtime_check():
    """``isinstance(h, StepHandler)`` works on duck-typed implementations."""
    h = _FakeHandler("click")
    assert isinstance(h, StepHandler)


def test_handler_registry_register_and_get():
    reg = HandlerRegistry()
    nav = _FakeHandler("navigate")
    click = _FakeHandler("click")
    reg.register(nav)
    reg.register(click)

    assert reg.get("navigate") is nav
    assert reg.get("click") is click
    assert reg.get("paginate") is None
    assert "navigate" in reg
    assert "paginate" not in reg


def test_handler_registry_register_for_types_aliases_one_handler():
    """Form handler today serves submit/fill_field/select_option — same code path."""
    reg = HandlerRegistry()
    form = _FakeHandler("submit")
    reg.register_for_types(form, ("submit", "fill_field", "select_option"))

    assert reg.get("submit") is form
    assert reg.get("fill_field") is form
    assert reg.get("select_option") is form
    # Latest registration overrides; alias does not duplicate
    assert set(reg.types()) == {"submit", "fill_field", "select_option"}


def test_step_context_carries_all_collaborators():
    """Pin the field set so downstream handlers know what they get."""
    scanner = ListingsScanner()
    ctx = StepContext(
        env=object(),
        brain=object(),
        extractor=None,
        grounding=None,
        cost_meter=None,
        dynamic_verifier=None,
        scanner=scanner,
        site_config=None,
        tool_channel=None,
        extraction_cache=None,
    )
    assert ctx.scanner is scanner
    assert ctx.state == {}  # default-factory dict, instance-private


def test_step_context_state_is_handler_private_scratch():
    """Distinct contexts must not share state dicts (default_factory check)."""
    a = StepContext(env=object(), brain=object())
    b = StepContext(env=object(), brain=object())
    a.state["seen_screenshots"] = 3
    assert b.state == {}


def test_handler_executes_via_protocol():
    """Smoke: drive a handler through the protocol-typed call site."""
    h = _FakeHandler("click")
    ctx = StepContext(env=object(), brain=object())
    step = MicroIntent(intent="Click", type="click")
    result = h.execute(step, ctx)
    assert result.success is True
    assert h.calls == [(step, ctx)]
