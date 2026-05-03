"""Tests for literal-value preservation in the decomposer prompt.

A real CRM-workflow run revealed the runner was emitting
``[claude-form] fill_field '' = ''`` for both credential fields. The
decomposer was producing fill_field steps with empty `params.value`
even though the source plan named the username and password.

The fix tightens the prompt with a CRITICAL rule + worked examples
showing exactly how literal values must round-trip into
`params.value` / `params.option_label`. These tests assert the prompt
content; behavior validation happens via Modal verify reruns.

Tests use neutral placeholders (alice / hunter2 / acme corp) — never
real customer credentials. The repo guards against customer-token
leaks via tests/test_docs_client_isolation.py.
"""

from __future__ import annotations

from mantis_agent.plan_decomposer import DECOMPOSE_PROMPT


# ── The CRITICAL rule is present and prominent ─────────────────────────


def test_prompt_has_critical_literal_value_rule() -> None:
    text_lower = DECOMPOSE_PROMPT.lower()
    # The rule should be flagged with attention markers ("CRITICAL", "MUST")
    # so Claude doesn't gloss over it.
    assert "critical" in text_lower
    assert "preserve literal values" in text_lower or "literal value" in text_lower


def test_prompt_warns_against_empty_fill_field_value() -> None:
    """Explicit anti-example: do not emit fill_field with empty value
    when the source provided one."""
    text_lower = DECOMPOSE_PROMPT.lower()
    assert "empty" in text_lower
    assert (
        "type \"\"" in DECOMPOSE_PROMPT
        or 'value of ""' in text_lower
        or "empty `value`" in text_lower
    )


# ── Worked example: login credentials ─────────────────────────────────


def test_prompt_includes_login_credentials_worked_example() -> None:
    """The login pattern is shown with neutral placeholder credentials so
    Claude has a template to follow."""
    text = DECOMPOSE_PROMPT
    # Neutral placeholders — must NOT be real customer credentials.
    assert '"value": "alice"' in text
    assert '"value": "hunter2"' in text


def test_prompt_login_example_includes_submit_step() -> None:
    """A complete login sequence is fill_field × 2 + submit. The example
    must show all three so Claude sees the whole pattern."""
    text = DECOMPOSE_PROMPT
    assert '"submit"' in text
    assert "Sign In" in text


# ── Worked example: free-text input (search box etc.) ──────────────────


def test_prompt_includes_search_box_worked_example() -> None:
    """Generic literal-value preservation for any free-text input."""
    text = DECOMPOSE_PROMPT
    text_lower = text.lower()
    assert "search box" in text_lower
    assert '"value": "acme corp"' in text


# ── Worked example: dropdown / select_option ───────────────────────────


def test_prompt_includes_dropdown_worked_example() -> None:
    """Preservation rule applies to select_option's option_label too."""
    text = DECOMPOSE_PROMPT
    # Neutral domain-agnostic choice example.
    assert "Space Exploration" in text
    assert '"option_label"' in text


# ── Self-check rule ────────────────────────────────────────────────────


def test_prompt_includes_self_check_rule_for_literal_values() -> None:
    """Claude should verify its own output: every fill_field generated
    from a value-bearing source phrase must carry that value."""
    text_lower = DECOMPOSE_PROMPT.lower()
    assert "self-check" in text_lower
    assert "literal value" in text_lower or "must appear" in text_lower


# ── Backward compat — existing rules still in the prompt ───────────────


def test_prompt_still_documents_fill_field_params_shape() -> None:
    text = DECOMPOSE_PROMPT
    assert "fill_field" in text
    assert "label" in text
    assert "value" in text


def test_prompt_still_links_log_in_to_fill_field_then_submit() -> None:
    """The verb mapping for 'log in' must still point at fill_field +
    submit (not deprecated)."""
    text_lower = DECOMPOSE_PROMPT.lower()
    assert "log in" in text_lower or "sign in" in text_lower
    assert "fill_field" in text_lower
    assert "submit" in text_lower


# ── Cache-key invalidation ─────────────────────────────────────────────


def test_prompt_version_was_bumped() -> None:
    """A prompt change requires a cache-version bump so previously-
    decomposed plans get re-decomposed with the new rule."""
    import inspect
    from mantis_agent.plan_decomposer import PlanDecomposer

    src = inspect.getsource(PlanDecomposer.decompose_text)
    assert "v19_literal_values" in src
    assert "v18_pure_llm" not in src
