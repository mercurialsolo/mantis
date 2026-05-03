"""Tests for the in-app-navigation contract — LLM-driven only.

The earlier branch of this work used regex post-processing to rewrite
urlless ``navigate`` steps into ``submit`` steps. That regex pass was
removed: per the project's no-regex-semantic-matching directive, only
the LLM decides which step type fits a given source phrase.

These tests now verify the **prompt** carries enough context for the LLM
to follow the rule:

  • The source plan must contain an http(s):// URL for a step to be
    classified as ``navigate``.
  • In-app phrases ("Go to the Leads page", "Open Settings") must be
    classified as ``submit`` with params={"label": "<page name>"}.

Behavior validation happens via Modal verify reruns, not via static
regex tests.
"""

from __future__ import annotations

from mantis_agent.plan_decomposer import DECOMPOSE_PROMPT, PlanDecomposer


# ── Prompt teaches the navigate-requires-URL contract ─────────────────


def test_prompt_states_navigate_requires_url() -> None:
    """The CRITICAL RULE block names http:// and https:// explicitly."""
    text = DECOMPOSE_PROMPT
    assert "http://" in text and "https://" in text
    # The phrase "REQUIRES" or "no exceptions" should appear to give the
    # rule weight against Claude's tendency to be flexible.
    text_lower = text.lower()
    assert "requires" in text_lower or "no exceptions" in text_lower or "must" in text_lower


def test_prompt_provides_in_app_to_submit_worked_examples() -> None:
    """Worked examples show Claude exactly what to emit for in-app nav."""
    text = DECOMPOSE_PROMPT
    # The "Go the Leads Page" example should map to a submit with label
    # "Leads" — that's the staffcrm shape.
    assert "Leads" in text
    assert '"submit"' in text
    assert '"label"' in text


def test_prompt_includes_self_check_invariant() -> None:
    """A self-check rule lets Claude verify its own output without us
    enforcing it via regex post-process."""
    text_lower = DECOMPOSE_PROMPT.lower()
    # Either phrasing works — what matters is the verifiable invariant
    # ("every navigate step's intent contains http://").
    assert "http://" in DECOMPOSE_PROMPT and (
        "verify" in text_lower or "verifiable" in text_lower or "must contain" in text_lower
    )


# ── No regex post-process exists ────────────────────────────────────────


def test_no_regex_in_app_nav_patterns() -> None:
    """The regex-based post-process was removed. If a contributor adds it
    back without converting to LLM, this catches it."""
    assert not hasattr(PlanDecomposer, "_IN_APP_NAV_PATTERNS")
    assert not hasattr(PlanDecomposer, "_extract_in_app_page_label")
    assert not hasattr(PlanDecomposer, "_rewrite_urlless_navigates")


def test_decomposer_module_documents_no_semantic_regex() -> None:
    import mantis_agent.plan_decomposer as mod
    src = mod.__file__
    with open(src) as f:
        text = f.read()
    # The deprecation comment should be present so future readers
    # understand why the regex helpers are gone.
    assert "no regex" in text.lower() or "no semantic regex" in text.lower() \
        or "llm-only generalization" in text.lower()


# ── Backward compat: existing decomposer dispatch still runs ────────────


def test_decomposer_dispatch_still_runs_loop_target_fix() -> None:
    """Dispatch chain order changed (lost the urlless-nav rewrite) but
    loop target validation must still run."""
    import inspect
    src = inspect.getsource(PlanDecomposer.decompose_text)
    assert "_fix_loop_targets" in src
