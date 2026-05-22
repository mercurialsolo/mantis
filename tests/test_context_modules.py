"""Tests for context-aware prompt composition + hint injection.

Pin the ContextModule registry's MVP behaviour so a future tweak to a
predicate or module section doesn't silently change what the brain
sees for a given step type. Each test mounts a context, asks for the
composed prompt / hints, and asserts on the surface that callers
actually consume.
"""

from __future__ import annotations

from mantis_agent.context_modules import (
    BLOCKING_OVERLAY,
    ContextModule,
    KEYBOARD_SCROLL,
    MODULES,
    applicable_handler_hints,
    applicable_modules,
    applicable_step_hints,
    compose_system_prompt,
    current_context,
    push_step_context,
)


# ── Context-var plumbing ──────────────────────────────────────────────


def test_current_context_defaults_to_empty_dict() -> None:
    """No push → empty dict (NOT None) so modules can ``.get`` safely."""
    assert current_context() == {}


def test_push_step_context_publishes_and_restores() -> None:
    """Inside the ``with`` block ``current_context`` returns the pushed
    dict; outside it restores the prior value (empty here)."""
    assert current_context() == {}
    with push_step_context({"step_type": "scroll"}):
        assert current_context() == {"step_type": "scroll"}
    assert current_context() == {}


def test_push_step_context_supports_nesting() -> None:
    """Nested pushes restore the outer value on exit."""
    with push_step_context({"step_type": "click"}):
        with push_step_context({"step_type": "scroll"}):
            assert current_context()["step_type"] == "scroll"
        assert current_context()["step_type"] == "click"


def test_current_context_returns_a_copy() -> None:
    """Caller mutations on the returned dict must not affect future
    reads — protects against accidental coupling between modules."""
    with push_step_context({"step_type": "click"}):
        copy = current_context()
        copy["step_type"] = "mutated"
        assert current_context()["step_type"] == "click"


# ── Predicate firing ──────────────────────────────────────────────────


def test_blocking_overlay_only_on_extract_primitives() -> None:
    """BLOCKING_OVERLAY fires ONLY on actual extract_data /
    extract_url steps — NOT on scroll-in-extraction (KEYBOARD_SCROLL
    already covers image-misclick / carousel-trigger defense on
    scroll steps). Stacking both prompt sections on scroll
    paralyzed the brain (brain_loop_exhausted on scroll step
    despite cap=8, run 20260522_200135_af9e031a)."""
    assert BLOCKING_OVERLAY in applicable_modules({"step_type": "extract_data"})
    assert BLOCKING_OVERLAY in applicable_modules({"step_type": "extract_url"})
    # Inactive on scroll, regardless of section.
    assert BLOCKING_OVERLAY not in applicable_modules(
        {"step_type": "scroll", "step_section": "extraction"},
    )
    assert BLOCKING_OVERLAY not in applicable_modules(
        {"step_type": "scroll", "step_section": "setup"},
    )
    # Inactive on navigation / setup / pagination.
    assert BLOCKING_OVERLAY not in applicable_modules({})
    assert BLOCKING_OVERLAY not in applicable_modules({"step_type": "navigate"})
    assert BLOCKING_OVERLAY not in applicable_modules(
        {"step_type": "paginate"},
    )


def test_keyboard_scroll_only_on_scroll_steps() -> None:
    assert KEYBOARD_SCROLL in applicable_modules({"step_type": "scroll"})
    assert KEYBOARD_SCROLL not in applicable_modules({"step_type": "click"})
    assert KEYBOARD_SCROLL not in applicable_modules({"step_type": "fill_field"})


def test_registry_does_not_modulerize_base_prompt_defaults() -> None:
    """FORM FILLING and NAVIGATION guidance stays in the base
    holo3_system.txt prompt (always-on, battle-tested defaults the
    brain relies on for click / form interactions). The module
    registry only contains NEW behaviour added in this session
    (BLOCKING_OVERLAY, KEYBOARD_SCROLL) — moving the existing
    defaults into per-step modules regressed throughput (see runs
    20260522_160850_4484731f and _162958_370b9a1a). Pin the
    boundary: the registry's job is augmentation, not replacement."""
    names = {m.name for m in MODULES}
    assert "form_filling" not in names
    assert "navigate" not in names


# ── Prompt composition ───────────────────────────────────────────────


def test_compose_system_prompt_lean_base_for_empty_context() -> None:
    """Empty context → no module sections spliced. The base prompt
    stays lean for non-step contexts (e.g. eager module import)."""
    out = compose_system_prompt("BASE.", {})
    assert out.strip() == "BASE."  # nothing extra


def test_compose_system_prompt_extraction_scroll_keyboard_only() -> None:
    """A scroll step (regardless of section) gets KEYBOARD_SCROLL
    but NOT BLOCKING_OVERLAY. KEYBOARD_SCROLL already prohibits
    image clicks (the carousel trigger); stacking BLOCKING_OVERLAY's
    section on top doubles prompt overhead and paralyzed the brain."""
    out = compose_system_prompt(
        "BASE.", {"step_type": "scroll", "step_section": "extraction"},
    )
    assert "SCROLLING" in out
    assert "BLOCKING UI" not in out
    assert "FORM FILLING" not in out
    assert "NAVIGATION" not in out


def test_compose_system_prompt_setup_scroll_just_keyboard() -> None:
    """A setup-section scroll gets KEYBOARD_SCROLL only —
    BLOCKING_OVERLAY remains scoped out for scroll regardless of
    section."""
    out = compose_system_prompt(
        "BASE.", {"step_type": "scroll", "step_section": "setup"},
    )
    assert "SCROLLING" in out
    assert "BLOCKING UI" not in out


def test_compose_system_prompt_form_step_gets_no_extra_modules() -> None:
    """A form step (fill_field / submit / select_option) gets only
    the base prompt — no module sections. FORM FILLING guidance is
    in the base prompt, not the module registry."""
    out = compose_system_prompt("BASE.", {"step_type": "fill_field"})
    assert out.strip() == "BASE."


def test_compose_system_prompt_navigate_step_gets_no_extra_modules() -> None:
    """A navigate step gets only the base prompt. NAVIGATION guidance
    is in the base, not the registry."""
    out = compose_system_prompt("BASE.", {"step_type": "navigate"})
    assert out.strip() == "BASE."


def test_compose_system_prompt_extract_data_gets_blocking_overlay() -> None:
    """Extract_data steps DO get BLOCKING_OVERLAY — overlays on
    detail pages are the exact failure class to defend against."""
    out = compose_system_prompt("BASE.", {"step_type": "extract_data"})
    assert "BLOCKING UI" in out


def test_compose_system_prompt_reads_active_context_when_no_arg() -> None:
    """When ``ctx`` is not passed, composer reads ``current_context``."""
    with push_step_context({"step_type": "scroll"}):
        out = compose_system_prompt("BASE.")
    assert "SCROLLING" in out
    assert "FORM FILLING" not in out


def test_compose_system_prompt_predicate_error_doesnt_break_compose() -> None:
    """A module whose predicate raises is skipped; other modules still
    contribute. Pinning this so a buggy custom module can't disable
    the whole composition path."""
    def _boom(_ctx):
        raise RuntimeError("nope")
    broken = ContextModule(
        name="broken", applies_when=_boom, prompt_section="SHOULD NOT APPEAR",
    )
    out = compose_system_prompt(
        "BASE.", {"step_type": "scroll"},
        modules=[broken, KEYBOARD_SCROLL],
    )
    assert "SHOULD NOT APPEAR" not in out
    assert "SCROLLING" in out


# ── Handler hints ────────────────────────────────────────────────────


def test_handler_hints_for_extraction_scroll_only_avoid_image() -> None:
    """A scroll-in-extraction step gets avoid_image_click ONLY (from
    KEYBOARD_SCROLL). BLOCKING_OVERLAY's dismiss_overlay_first does
    NOT fire on scroll — it's reserved for extract_data /
    extract_url."""
    hints = applicable_handler_hints(
        {"step_type": "scroll", "step_section": "extraction"},
    )
    assert hints.get("avoid_image_click") is True
    assert "dismiss_overlay_first" not in hints


def test_handler_hints_for_extract_data_step_carries_overlay_signal() -> None:
    """The extract_data primitive itself gets the overlay-dismiss
    signal — that's where the brain is about to read structured
    fields and an overlay-blocked viewport is the actual risk."""
    hints = applicable_handler_hints({"step_type": "extract_data"})
    assert hints.get("dismiss_overlay_first") is True


def test_handler_hints_for_setup_scroll_only_avoid_image() -> None:
    """A setup-section scroll gets avoid_image_click only — no
    overlay signal because overlays don't matter on setup pages."""
    hints = applicable_handler_hints(
        {"step_type": "scroll", "step_section": "setup"},
    )
    assert hints.get("avoid_image_click") is True
    assert "dismiss_overlay_first" not in hints


def test_handler_hints_for_form_step_empty() -> None:
    """Form steps get no module hints; FORM FILLING lives in base."""
    hints = applicable_handler_hints({"step_type": "fill_field"})
    assert hints == {}


# ── Step-level hints (text appended to brain task) ───────────────────


def test_applicable_step_hints_empty_for_starter_modules() -> None:
    """Starter modules carry section text, not step_hint text. New
    modules can populate step_hint when they want a short note rather
    than a section."""
    assert applicable_step_hints({}) == []
    assert applicable_step_hints({"step_type": "scroll"}) == []


def test_applicable_step_hints_returns_added_module_text() -> None:
    """Demonstrate the registry surface: a module with a step_hint
    contributes its text when applicable."""
    custom = ContextModule(
        name="custom_hint", applies_when=lambda c: c.get("trigger") == "yes",
        step_hint="extra hint text",
    )
    hits = applicable_step_hints({"trigger": "yes"}, modules=[custom])
    assert hits == ["extra hint text"]
    miss = applicable_step_hints({"trigger": "no"}, modules=[custom])
    assert miss == []


# ── Registry sanity ──────────────────────────────────────────────────


def test_module_registry_unique_names() -> None:
    """Catch accidental duplicate registrations."""
    names = [m.name for m in MODULES]
    assert len(names) == len(set(names)), f"duplicate module names: {names}"


def test_module_registry_has_starter_set() -> None:
    """The MVP registry includes the two NEW-behaviour starter
    modules. FORM FILLING / NAVIGATION stay in the base prompt."""
    names = {m.name for m in MODULES}
    assert {"blocking_overlay", "keyboard_scroll"} <= names
