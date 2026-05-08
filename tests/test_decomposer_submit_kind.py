"""Tests for #209 Symptom 4 — decomposer emits ``params.kind`` on submit.

Layer 2 of the kind-aware submit dispatch. The runtime form-handler
side is covered in ``test_form_submit_kind.py``; here we assert the
prompt teaches Claude to classify each submit step into the small,
domain-neutral set of visual affordances the runtime understands.
"""

from __future__ import annotations

import inspect

from mantis_agent.plan_decomposer import DECOMPOSE_PROMPT, PlanDecomposer


# ── Prompt teaches the four kinds ────────────────────────────────────────


def test_prompt_names_kind_param_for_submit() -> None:
    """The submit verb mapping must require a ``kind`` param so the
    runtime form-handler can frame its search prompt correctly."""
    text = DECOMPOSE_PROMPT
    assert "params.kind" in text or '"kind"' in text


def test_prompt_documents_each_runtime_kind() -> None:
    """Every kind the runtime templates must appear as a worked
    example in the prompt — otherwise the runtime would silently
    downgrade unknown kinds to button."""
    text = DECOMPOSE_PROMPT
    for kind in ("button", "nav_link", "tab", "menu_item"):
        assert kind in text, f"prompt is missing kind={kind!r}"


def test_prompt_documents_default_kind_for_backward_compat() -> None:
    """When the source plan is ambiguous, the prompt must steer Claude
    to the ``button`` default rather than guessing."""
    text_lower = DECOMPOSE_PROMPT.lower()
    assert "default" in text_lower
    assert "button" in text_lower


def test_prompt_separates_nav_link_from_button_examples() -> None:
    """The previous prompt collapsed nav-link and button into a single
    submit example. The new prompt must show them as distinct cases."""
    text = DECOMPOSE_PROMPT
    # nav_link is associated with navigation phrasing
    assert "navigation" in text.lower()
    # The "go to the X page" in-app phrase is now a nav_link example,
    # not a generic submit. Keeping both phrases nearby.
    assert "go to the" in text.lower() or "open the" in text.lower()


# ── Cache key bumped so v20 caches re-decompose with kind ───────────────


def test_prompt_version_bumped_for_submit_kind() -> None:
    """A prompt change requires a cache-version bump so previously-
    decomposed plans get re-decomposed with the new rule."""
    src = inspect.getsource(PlanDecomposer.decompose_text)
    assert "v21_submit_kind" in src
    assert "v20_url_mirror" not in src
