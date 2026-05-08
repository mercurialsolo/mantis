"""Tests for #209 Symptom 4 — submit-step visual-affordance hints.

The previous unconditional ``Click the '{label}' button to submit the
form`` wording biased ``find_form_target`` toward button-shaped
candidates even when the actual target was a sidebar navigation link
or a tab. The decomposer now classifies each submit step via a small
``params.kind`` enum (``button`` / ``nav_link`` / ``tab`` /
``menu_item``); the form handler builds the search prompt from a
template table keyed on that kind.

Layer 1 (this file): the runtime form handler dispatches on the kind.
Layer 2 (test_decomposer_submit_kind.py): the decomposer prompt
teaches Claude to emit the kind. Both layers are domain-neutral.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from mantis_agent.gym.step_context import StepContext
from mantis_agent.gym.step_handlers.form import (
    _SUBMIT_KIND_DEFAULT,
    _SUBMIT_KIND_INTENT_TEMPLATES,
    ClaudeGuidedFormHandler,
    _build_submit_search_intent,
)
from mantis_agent.plan_decomposer import MicroIntent


# ── Pure helper ──────────────────────────────────────────────────────────


def test_build_intent_returns_fallback_when_label_empty() -> None:
    """No label means we cannot template a kind-aware prompt — defer
    to the step's natural-language intent."""
    out = _build_submit_search_intent("", "nav_link", fallback="step prose here")
    assert out == "step prose here"


@pytest.mark.parametrize(
    "kind,expected_token",
    [
        ("nav_link", "navigation link"),
        ("tab", "tab"),
        ("menu_item", "menu item"),
        ("row_link", "table row"),
        ("cell_link", "table cell"),
        ("button", "button to submit the form"),
    ],
)
def test_build_intent_picks_template_per_kind(kind: str, expected_token: str) -> None:
    out = _build_submit_search_intent("Leads", kind, fallback="ignored")
    assert "Leads" in out
    assert expected_token in out


def test_build_intent_unknown_kind_falls_back_to_button_template() -> None:
    """Forward-compat: a future ``kind`` the runtime doesn't yet know
    must NOT crash and must NOT use an empty prompt — it falls back to
    the default (button) framing."""
    out = _build_submit_search_intent("Save", "frobinator_widget", fallback="x")
    button_template = _SUBMIT_KIND_INTENT_TEMPLATES[_SUBMIT_KIND_DEFAULT]
    assert out == button_template.format(label="Save")


def test_default_kind_is_button() -> None:
    """Backward compat — submit steps without ``kind`` predate this PR."""
    assert _SUBMIT_KIND_DEFAULT == "button"
    assert "button" in _SUBMIT_KIND_INTENT_TEMPLATES


def test_all_known_kinds_have_templates() -> None:
    """Every kind documented in the decomposer prompt must have a
    runtime template — otherwise the decomposer can emit a kind the
    runtime silently downgrades to ``button``."""
    expected = {"button", "nav_link", "tab", "menu_item", "row_link", "cell_link"}
    assert expected.issubset(_SUBMIT_KIND_INTENT_TEMPLATES.keys())


def test_row_link_template_disambiguates_from_headers_and_badges() -> None:
    """row_link clicks are the new staff-crm pain point: Holo3 mistargeted
    column headers / status badges / sort controls when asked to "click
    the lead with Status=Qualified". The template must explicitly steer
    away from those non-row-body candidates so find_form_target picks
    the cell-text link instead."""
    out = _build_submit_search_intent("Optimus Prime", "row_link", fallback="x")
    assert "Optimus Prime" in out
    # Disambiguation cues that materially narrow the target set.
    assert "row" in out.lower()
    assert "header" in out.lower()
    assert "badge" in out.lower() or "status" in out.lower()


# ── Handler dispatch — search_intent observed via the extractor mock ─────


class _FakeRunner:
    def __init__(self) -> None:
        self.costs: dict[str, float] = {
            "claude_extract": 0,
            "gpu_steps": 0,
            "gpu_seconds": 0,
        }
        self._url_history: list[str] = ["https://app.example/x", "https://app.example/x/next"]
        self.dump_calls: list[tuple[str, Any]] = []

    def _best_effort_current_url(self) -> str:
        return self._url_history.pop(0) if self._url_history else ""

    def _adaptive_submit_settle(self, *, url_before: str) -> float:
        return 0.5

    def _safe_screenshot(self) -> Any:
        return MagicMock()

    def _dump_debug_screenshot(self, name_stem: str, screenshot: Any) -> None:
        self.dump_calls.append((name_stem, screenshot))


def _ctx(runner: _FakeRunner, *, env=None, extractor=None) -> StepContext:
    return StepContext(
        env=env or MagicMock(),
        brain=None,
        extractor=extractor or MagicMock(),
        grounding=None,
        cost_meter=None,
        dynamic_verifier=None,
        scanner=None,
        site_config=None,
        tool_channel=None,
        extraction_cache=None,
        state={"index": 1},
    )


def _run_submit_with_kind(monkeypatch, *, label: str, kind: str | None) -> str:
    """Drive the handler and return the search_intent it sent to find_form_target."""
    monkeypatch.setattr("mantis_agent.gym.step_handlers.form.time.sleep", lambda *_: None)
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.form.random.uniform", lambda a, b: a,
    )

    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.find_form_target.return_value = {"x": 100, "y": 200}
    env = MagicMock()
    ctx = _ctx(runner, env=env, extractor=extractor)

    params: dict[str, Any] = {"label": label}
    if kind is not None:
        params["kind"] = kind
    step = MicroIntent(intent="step prose", type="submit", params=params)
    ClaudeGuidedFormHandler(runner).execute(step, ctx)

    # First call to find_form_target carries the framed search_intent.
    call_args = extractor.find_form_target.call_args_list[0]
    return call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs["intent"]


def test_handler_uses_nav_link_template_when_kind_nav_link(monkeypatch) -> None:
    intent = _run_submit_with_kind(monkeypatch, label="Leads", kind="nav_link")
    assert "navigation link" in intent
    assert "Leads" in intent
    assert "submit the form" not in intent  # the wrong-bias wording is gone


def test_handler_uses_tab_template_when_kind_tab(monkeypatch) -> None:
    intent = _run_submit_with_kind(monkeypatch, label="Reports", kind="tab")
    assert "tab" in intent
    assert "Reports" in intent
    assert "submit the form" not in intent


def test_handler_uses_menu_item_template_when_kind_menu_item(monkeypatch) -> None:
    intent = _run_submit_with_kind(monkeypatch, label="Archive", kind="menu_item")
    assert "menu item" in intent
    assert "Archive" in intent


def test_handler_uses_button_template_when_kind_button(monkeypatch) -> None:
    intent = _run_submit_with_kind(monkeypatch, label="Save", kind="button")
    assert "button to submit the form" in intent
    assert "Save" in intent


def test_handler_defaults_to_button_when_kind_missing(monkeypatch) -> None:
    """Backward compatibility: cached plans from before this PR have no
    kind. The handler treats them as buttons (the prior wording)."""
    intent = _run_submit_with_kind(monkeypatch, label="Continue", kind=None)
    assert "button to submit the form" in intent
    assert "Continue" in intent


def test_handler_normalises_kind_case_and_whitespace(monkeypatch) -> None:
    """Defensive: Claude sometimes returns ``Nav_Link`` or `` nav_link ``;
    runtime should match by lowercasing + stripping."""
    intent = _run_submit_with_kind(monkeypatch, label="Settings", kind=" Nav_Link ")
    assert "navigation link" in intent
