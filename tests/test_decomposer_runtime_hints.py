"""Tests for the decomposer's RUNTIME HINTS section (do-not-decompose).

Surfaced by the boattrader bench-plan smoke through ``mantis plan run``:
the bench plan starts with ``FIRST ACTION RULE: your very first action
MUST be wait for 15 seconds`` plus a ``BENCHMARK PREAMBLE`` block. The
decomposer's step-type vocabulary has no ``wait``, so it picked the
closest type — ``extract_data`` (claude_only) — which the runner then
ran through the marketplace-listings extractor on a freshly-loaded
search-results page. With no listing yet, the extractor rejected the
step as ``REJECTED_INCOMPLETE`` and the entire plan halted on step 0.

Generic fix: the prompt now teaches Claude to recognise runtime hints
(wait instructions, benchmark preambles, capability constraints) as
prompt metadata rather than steps. They get omitted; the runner's
existing settle / first-paint / render-wait primitives handle the
underlying timing concern.
"""

from __future__ import annotations

import inspect

from mantis_agent.plan_decomposer import DECOMPOSE_PROMPT, PlanDecomposer


# ── Section is present + names the failure pattern ────────────────────


def test_prompt_has_runtime_hints_do_not_decompose_section() -> None:
    """The section must be present with attention-grabbing wording so
    Claude sees it before reaching the verb mappings below."""
    text = DECOMPOSE_PROMPT
    assert "RUNTIME HINTS" in text
    text_lower = text.lower()
    assert "do not decompose" in text_lower or "do not emit" in text_lower


def test_prompt_calls_out_wait_for_seconds_pattern() -> None:
    """The exact phrasing from the boattrader bench plan
    (``Wait N seconds for the page to load``) should be a worked
    example so Claude sees the canonical pattern, not just the
    abstract rule."""
    text_lower = DECOMPOSE_PROMPT.lower()
    assert "wait" in text_lower
    assert "seconds" in text_lower
    # Either the hydration phrasing or the page-load phrasing.
    assert "hydration" in text_lower or "fully render" in text_lower or "page to load" in text_lower


def test_prompt_calls_out_benchmark_preamble_and_first_action_rule() -> None:
    """The bench plan's literal section headers (``BENCHMARK PREAMBLE``,
    ``FIRST ACTION RULE``) should be named so Claude pattern-matches
    against them rather than guessing."""
    text = DECOMPOSE_PROMPT
    assert "FIRST ACTION RULE" in text or "BENCHMARK PREAMBLE" in text


def test_prompt_explains_why_decomposing_wait_is_wrong() -> None:
    """The rule should explain WHY (``REJECTED_INCOMPLETE`` cascade)
    rather than just saying ``don't``. Future model rewrites of the
    prompt that drop the rule but keep the surrounding structure
    re-introduce the same bug; the rationale text is the guard."""
    text = DECOMPOSE_PROMPT
    assert "REJECTED_INCOMPLETE" in text


def test_prompt_lists_runner_settle_primitives_that_replace_wait() -> None:
    """The rule names the runner-side primitives that already cover
    the timing concern — so a reader (or model) can verify the omit
    decision is safe rather than wondering whether the page actually
    needs that wait."""
    text = DECOMPOSE_PROMPT
    text_lower = text.lower()
    # First-paint wait + form settle + render-wait are the three
    # primitives that absorb "wait N seconds" semantics already.
    assert "navigate handler" in text_lower or "18s" in text or "first paint" in text_lower
    assert "form handler" in text_lower or "settle" in text_lower


def test_prompt_omit_rule_extends_to_capability_constraints() -> None:
    """Constraints like ``Do NOT use Developer Tools`` are operator
    notes, not steps. The brain's action vocabulary doesn't include
    those primitives in the first place — emitting a step for them
    would add no-op clutter at best, confusion at worst."""
    text = DECOMPOSE_PROMPT
    text_lower = text.lower()
    assert "developer tools" in text_lower or "execute javascript" in text_lower or "capability constraint" in text_lower


# ── Cache key bumped so v22 caches re-decompose under v23 ─────────────


def test_cache_version_bumped_past_v23() -> None:
    """A prompt change requires a cache-version bump so previously-
    decomposed plans get re-decomposed under the new rule. v23 was
    the runtime-hints rule; v24 (issue #244) supersedes it by also
    routing verification extract_data through the gate path. The
    runtime-hints rule itself is preserved in the v24 prompt — this
    test just guards against stale-cache regression by asserting we
    moved past v23."""
    src = inspect.getsource(PlanDecomposer.decompose_text)
    assert "v23_skip_runtime_hints" not in src
    assert "v22_row_link" not in src


# Stay-generic guard for customer-token leaks is delegated to
# tests/test_docs_client_isolation.py, which scans every tracked
# file in the tree (including DECOMPOSE_PROMPT). Adding a redundant
# check here would itself leak the tokens this file is tracked for.
