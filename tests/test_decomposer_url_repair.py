"""Tests for #209 Symptom 4 — decomposer URL preservation hardening.

Three layers of defense against the v19 failure mode where Claude
paraphrases a navigate step's intent and drops the literal URL:

1. The v20 ``DECOMPOSE_PROMPT`` mirrors the URL into ``params.url`` and
   adds a "Navigate to <URL> for <purpose>" worked example.
2. ``PlanDecomposer._repair_navigate_urls`` re-scans the source plan
   for URLs and repairs urlless navigate steps post-decompose.
3. ``NavigateHandler`` falls back to ``params.url`` when the intent
   string lacks an ``https://`` substring.

These tests cover layers 1 and 2. Layer 3 is covered in
``test_navigate_handler.py``.
"""

from __future__ import annotations

import inspect

from mantis_agent.plan_decomposer import (
    DECOMPOSE_PROMPT,
    MicroIntent,
    MicroPlan,
    PlanDecomposer,
)


# ── Layer 1: prompt teaches URL mirroring ────────────────────────────────


def test_prompt_teaches_url_mirror_into_params_url() -> None:
    """The CRITICAL block must instruct Claude to put the URL in BOTH
    intent and params.url so the runtime fallback has something to read."""
    text = DECOMPOSE_PROMPT
    assert "params.url" in text or '"url"' in text
    # Worked example for the paraphrase failure mode from #209.
    assert "Navigate to https://app.example.com" in text


def test_prompt_warns_against_paraphrasing_the_url() -> None:
    """Sonnet's tendency: rewrite 'Navigate to <URL> for <purpose>' as
    'Navigate to the X system'. The prompt should call this out."""
    text_lower = DECOMPOSE_PROMPT.lower()
    assert "do not paraphrase" in text_lower or "keep the literal" in text_lower


def test_prompt_version_bumped_for_url_mirror() -> None:
    """Bumping the prompt-version cache key forces re-decomposition.
    If the cache key still says v19, callers will load stale results."""
    src = inspect.getsource(PlanDecomposer.decompose_text)
    assert "v20" in src
    assert "v19_literal_values" not in src


# ── Layer 2: post-decompose repair pass ──────────────────────────────────


def _make_plan(source: str, steps: list[MicroIntent]) -> MicroPlan:
    plan = MicroPlan(source_plan=source, domain="example.com")
    plan.steps = steps
    return plan


def test_repair_injects_source_url_into_urlless_navigate() -> None:
    """A navigate step that lost its URL gets the source URL written
    back into params.url so the navigate handler can recover it."""
    plan = _make_plan(
        source="1. Navigate to https://crm.example.test/leads for triage\n2. Open Settings",
        steps=[
            MicroIntent(intent="Navigate to the leads management system", type="navigate"),
            MicroIntent(intent="Open the Settings section", type="submit", params={"label": "Settings"}),
        ],
    )

    PlanDecomposer._repair_navigate_urls(plan)

    assert plan.steps[0].params == {"url": "https://crm.example.test/leads"}
    # Submit step is unchanged.
    assert plan.steps[1].params == {"label": "Settings"}


def test_repair_leaves_navigate_alone_when_intent_has_url() -> None:
    """The repair pass must not over-write a healthy navigate step."""
    plan = _make_plan(
        source="1. Go to https://example.com/login",
        steps=[
            MicroIntent(intent="Navigate to https://example.com/login", type="navigate"),
        ],
    )

    PlanDecomposer._repair_navigate_urls(plan)

    # No params.url injection — intent already has the URL.
    assert plan.steps[0].params in (None, {})


def test_repair_leaves_navigate_alone_when_params_url_already_set() -> None:
    """If the decomposer wrote params.url already, the repair pass
    is a no-op (don't double-pop a source URL)."""
    plan = _make_plan(
        source="1. Go to https://example.com/login\n2. Visit https://other.example.com",
        steps=[
            MicroIntent(
                intent="Navigate to login",
                type="navigate",
                params={"url": "https://example.com/login"},
            ),
            MicroIntent(intent="Open the data dashboard", type="navigate"),
        ],
    )

    PlanDecomposer._repair_navigate_urls(plan)

    # First step untouched (params.url already healthy).
    assert plan.steps[0].params == {"url": "https://example.com/login"}
    # Second step repaired with the SECOND source URL — cursor must skip
    # the URL that was already covered.
    assert plan.steps[1].params == {"url": "https://other.example.com"}


def test_repair_logs_error_when_more_urlless_navigates_than_source_urls(caplog) -> None:
    """If Claude paraphrases more navigate URLs away than the source
    actually had, the unmatched step is left broken and an ERROR is
    emitted so operators can see the prompt regression."""
    plan = _make_plan(
        source="1. Go to https://example.com",
        steps=[
            MicroIntent(intent="Navigate to system A", type="navigate"),
            MicroIntent(intent="Navigate to system B", type="navigate"),
        ],
    )

    with caplog.at_level("ERROR", logger="mantis_agent.plan_decomposer"):
        PlanDecomposer._repair_navigate_urls(plan)

    # First step repaired.
    assert plan.steps[0].params == {"url": "https://example.com"}
    # Second step unrepaired; an error log records it.
    assert plan.steps[1].params in (None, {})
    assert any("no source URL remains" in r.message for r in caplog.records)


def test_repair_is_called_after_fix_loop_targets_in_dispatch() -> None:
    """The repair pass must run as part of the decompose pipeline,
    not as an opt-in helper a caller has to invoke separately."""
    src = inspect.getsource(PlanDecomposer.decompose_text)
    assert "_repair_navigate_urls" in src
    fix_pos = src.index("_fix_loop_targets")
    repair_pos = src.index("_repair_navigate_urls(plan)")
    assert fix_pos < repair_pos, "Repair must run after loop-target fix"


def test_repair_is_noop_when_source_plan_has_no_urls() -> None:
    """An in-app-only plan (login form + page transitions) has no
    https URLs; repair must do nothing rather than crash."""
    plan = _make_plan(
        source="1. Log in with admin/admin\n2. Open the Reports tab",
        steps=[
            MicroIntent(intent="Open the Reports section", type="submit", params={"label": "Reports"}),
        ],
    )

    PlanDecomposer._repair_navigate_urls(plan)
    assert plan.steps[0].params == {"label": "Reports"}
