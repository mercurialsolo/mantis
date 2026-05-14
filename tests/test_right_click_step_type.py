"""``right_click`` step type — three-layer pin (#373).

The original gap stacked across three layers, all in one PR:

1. ``MicroIntent.type`` accepts ``right_click`` and ``_build_intent``
   gives it extraction-friendly defaults (section=extraction, not
   required, no Holo3 grounding).
2. ``ClaudeExtractor.find_form_target`` / ``find_filter_target`` /
   ``find_target_by_affordance`` accept ``"right_click"`` in the
   ``action`` enum so Claude can suggest the verb adaptively even
   on a plain ``click`` step.
3. The :class:`~mantis_agent.gym.step_handlers.form.ClaudeGuidedFormHandler`
   is wired to handle ``right_click`` and dispatches an
   ``Action(CLICK, button="right")`` at the resolved target. The
   handler-level dispatch + ``form_target_not_found`` envelope are
   pinned in ``tests/test_form_handler.py``; this file holds the
   cross-layer pins that the issue called out.

These tests are deliberately schema- / config-level. The runtime
behaviour is exercised in ``test_form_handler.py``.
"""

from __future__ import annotations

import inspect

from mantis_agent.gym.step_handlers import default_registry
from mantis_agent.gym.step_handlers.form import ClaudeGuidedFormHandler
from mantis_agent.plan_decomposer import DECOMPOSE_PROMPT, PlanDecomposer


# ── Layer 1: MicroIntent / decomposer ──────────────────────────────────


def test_micro_intent_docstring_lists_right_click() -> None:
    """The :class:`MicroIntent` ``type`` field docstring is the
    canonical enum surface — newcomers reading the class learn the
    allowed values from there. ``right_click`` MUST appear so any
    plan author can find it without spelunking the dispatch code."""
    from mantis_agent import plan_decomposer

    src = inspect.getsource(plan_decomposer)
    # The comment on the ``type`` field lists every allowed step
    # type — grep for the canonical right-click token.
    assert "right_click" in src


def test_build_intent_routes_right_click_to_extraction_section() -> None:
    """A ``right_click`` step with no explicit ``section`` must
    default to ``"extraction"`` — right-clicks open context menus
    mid-flow, not during setup. Required must stay False (a missed
    context menu shouldn't be fatal). Grounding must stay False
    (find_form_target is Claude-only, not Holo3)."""
    intent = PlanDecomposer._build_intent({
        "intent": "Right-click on Save to open hidden options",
        "type": "right_click",
        "params": {"label": "Save"},
    })
    assert intent.type == "right_click"
    assert intent.section == "extraction"
    assert intent.required is False
    assert intent.grounding is False
    # claude_only stays False — find_form_target is Claude-only but
    # the runner doesn't flag it via claude_only (that gate is for
    # extract_url / extract_data which have no executor action).
    assert intent.claude_only is False
    # ``params`` is preserved verbatim — the handler reads label/
    # aliases from there.
    assert intent.params == {"label": "Save"}


def test_build_intent_respects_explicit_overrides_on_right_click() -> None:
    """Plan author can override the defaults — e.g. a setup-flow
    right-click on a SSO popup, or a critical right-click that must
    succeed (required=True) for the rest of the plan to be coherent."""
    intent = PlanDecomposer._build_intent({
        "intent": "Right-click on SSO context icon",
        "type": "right_click",
        "section": "setup",
        "required": True,
        "params": {"label": "SSO"},
    })
    assert intent.section == "setup"
    assert intent.required is True


def test_decomposer_prompt_documents_right_click_verb_mapping() -> None:
    """``DECOMPOSE_PROMPT`` must teach Claude when to emit
    ``right_click`` vs ``click``. Without an explicit verb mapping
    the decomposer would always pick ``click`` and the new primitive
    would be unreachable from text plans."""
    text = DECOMPOSE_PROMPT
    assert "right_click" in text
    # The verb-mapping section should describe the context-menu
    # semantics so Claude doesn't mis-route a plain "click" verb.
    assert "context menu" in text.lower()


# ── Layer 2: extractor schema ──────────────────────────────────────────


def test_find_form_target_schema_includes_right_click() -> None:
    """``find_form_target``'s tool schema must list ``right_click``
    so Claude can return the verb when the page genuinely needs the
    native context menu — adaptive routing on plain ``click`` steps."""
    src = inspect.getsource(
        __import__("mantis_agent.extraction.extractor", fromlist=["*"])
    )
    # The action-enum line MUST list right_click. There are three
    # copies of the same enum (find_filter_target, find_form_target,
    # find_target_by_affordance); pin that every occurrence carries
    # right_click — otherwise the gap re-opens on a partial fix.
    enum_count = src.count('"click", "right_click", "type", "select", "not_found"')
    assert enum_count == 3, (
        f"expected 3 schema copies to include right_click, found {enum_count}"
    )


def test_find_form_target_prompt_documents_right_click() -> None:
    """The prompt file teaches Claude when to pick right_click vs
    click on a form-shaped page. Without this, the schema enum
    accepts the verb but the model would never emit it."""
    from importlib.resources import files

    prompt = files("mantis_agent.prompts.files").joinpath(
        "find_form_target.txt"
    ).read_text(encoding="utf-8")
    assert "right_click" in prompt
    assert "context menu" in prompt.lower()


# ── Layer 3: handler registry wiring ───────────────────────────────────


def test_right_click_is_registered_to_form_handler() -> None:
    """``default_registry`` MUST bind ``right_click`` to the form
    handler so the runner's fall-through dispatch (``registry.get
    (step.type)``) picks it up. Without this, every right_click
    step would crash through to ``_execute_holo3_step`` which has
    no understanding of the verb."""
    from unittest.mock import MagicMock

    runner = MagicMock()
    reg = default_registry(runner)
    handler = reg.get("right_click")
    assert handler is not None
    assert isinstance(handler, ClaudeGuidedFormHandler)
    # The form handler is also bound to the existing three form types
    # — make sure right_click didn't accidentally replace one.
    for legacy_type in ("fill_field", "submit", "select_option"):
        assert isinstance(
            reg.get(legacy_type), ClaudeGuidedFormHandler
        ), f"existing binding for {legacy_type!r} regressed"
