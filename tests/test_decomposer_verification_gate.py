"""Tests for issue #244 — verification ``extract_data`` must route
through the gate path, not authoritative-extraction.

The decomposer routinely auto-inserts an ``extract_data`` micro-step
at the end of multi-step plans (e.g. *"Verify the page loaded"*,
*"Read the URL in the address bar and confirm..."*). Until this fix,
those steps lacked ``gate=True`` and fell through to the deep-extract
code path that applies the recipe schema. On a search/listing URL
that has no ``year``/``make`` (or whatever the recipe requires) the
extractor rejects the row and the entire plan halts. Three rounds
of host-side prompt patches (host-integration template)
couldn't reliably stop the orchestrator from emitting verification
sub-goals in natural English.

Two-layer fix:

1. **Prompt** — DECOMPOSE_PROMPT now carries an explicit
   "VERIFICATION VS AUTHORITATIVE EXTRACTION" section. Any
   ``extract_data`` step whose intent reads as verification
   ("verify", "confirm", "check that", "wait for", etc.) MUST set
   ``gate=True``. Authoritative extraction (read structured data,
   scrape, harvest fields) keeps today's behavior.

2. **Runtime safety net** in ``PlanDecomposer._build_intent`` — if
   the LLM emits an ``extract_data`` step with verification-language
   intent and no explicit ``gate`` field, auto-promote ``gate=True``.
   The prompt is the primary signal; the heuristic is a cheap
   second defence so a single missed prompt cue can't halt a plan.

The runner-side dispatch in ``_runner_helpers.py:run_dispatch``
already routes ``step.gate=True`` to ``extractor.verify_gate``
(boolean+reason, no schema) before the extract_data deep-extract
path — so flipping ``gate=True`` is enough to bypass schema
rejection without touching the runner.
"""

from __future__ import annotations

import inspect

from mantis_agent.plan_decomposer import (
    DECOMPOSE_PROMPT,
    MicroPlan,
    PlanDecomposer,
)


# ── Prompt-layer guards ───────────────────────────────────────────


def test_prompt_distinguishes_verification_from_authoritative_extraction() -> None:
    """DECOMPOSE_PROMPT must carry an explicit section spelling out
    the two modes — without it Claude routinely conflates them."""
    text = DECOMPOSE_PROMPT
    text_lower = text.lower()
    # Either the section header or the principle name.
    assert (
        "verification" in text_lower
        and ("authoritative" in text_lower or "schema" in text_lower)
    )


def test_prompt_says_verification_extract_data_must_set_gate_true() -> None:
    """The prompt must state the rule plainly so Claude doesn't
    have to infer it from end-of-setup wording (which is where the
    rule lived prior to this fix and got missed for mid-plan
    verifications)."""
    text = DECOMPOSE_PROMPT
    text_lower = text.lower()
    # Look for the rule near the verification language.
    assert "gate=true" in text_lower
    # The rule is not just for end-of-setup gates — must apply to
    # any verification-flavored extract_data step.
    assert "verify" in text_lower or "confirm" in text_lower


def test_prompt_lists_verification_trigger_phrases() -> None:
    """The prompt enumerates the canonical trigger phrases so Claude
    has concrete pattern-match anchors. The list is the same
    heuristic the runtime safety net applies (see test_build_intent
    cases below)."""
    text_lower = DECOMPOSE_PROMPT.lower()
    # At least three of the canonical triggers must appear.
    triggers = ["verify", "confirm", "check that", "make sure", "wait for", "read the url"]
    hits = sum(1 for t in triggers if t in text_lower)
    assert hits >= 3, f"only {hits} verification triggers in prompt"


def test_prompt_calls_out_recipe_schema_rejection_failure_mode() -> None:
    """The rule should explain WHY emitting verification as plain
    extract_data breaks — the recipe-schema rejection cascade.
    Future prompt rewrites that drop the rule but keep the
    surrounding structure re-introduce the same bug; the rationale
    text is the guard."""
    text = DECOMPOSE_PROMPT
    # Either "schema" or the canonical reject keyword from the
    # extractor must appear so the reasoning is preserved.
    assert "schema" in text.lower() or "REJECTED_INCOMPLETE" in text


def test_cache_version_bumped() -> None:
    """A prompt change requires a cache-version bump so previously-
    decomposed plans get re-decomposed under the new rule. The
    durable assertion is "the superseded version must not reappear"
    — pinning a specific current version forces a churn-edit on
    every legitimate downstream bump (e.g. #373's v25_right_click)."""
    src = inspect.getsource(PlanDecomposer.decompose_text)
    assert "v23_skip_runtime_hints" not in src
    # The verification-gate bump itself was v24; it must not silently
    # reappear and resurrect pre-#373 cached plans.
    assert "v24_verification_gate" not in src


# ── Runtime safety net (PlanDecomposer._build_intent) ────────────


def _step(intent: str, type_: str = "extract_data", **extra) -> dict:
    base = {"intent": intent, "type": type_}
    base.update(extra)
    return base


def test_build_intent_promotes_verify_intent_to_gate() -> None:
    """An ``extract_data`` step whose intent starts with *Verify*
    must come out with ``gate=True`` even if the LLM forgot to set
    it. The runner dispatches ``gate=True`` to the verify-gate
    path, bypassing recipe-schema rejection."""
    s = PlanDecomposer._build_intent(
        _step("Verify the page heading shows the expected filters")
    )
    assert s.type == "extract_data"
    assert s.gate is True
    # claude_only stays True so the verify_gate dispatch (which does
    # require a screenshot) still fires.
    assert s.claude_only is True


def test_build_intent_promotes_confirm_intent_to_gate() -> None:
    s = PlanDecomposer._build_intent(
        _step("Confirm the lead detail page shows Priority High")
    )
    assert s.gate is True


def test_build_intent_promotes_check_that_intent_to_gate() -> None:
    s = PlanDecomposer._build_intent(
        _step("Check that the URL contains /leads/")
    )
    assert s.gate is True


def test_build_intent_promotes_make_sure_intent_to_gate() -> None:
    s = PlanDecomposer._build_intent(
        _step("Make sure the form saved successfully")
    )
    assert s.gate is True


def test_build_intent_promotes_read_url_intent_to_gate() -> None:
    """The exact phrasing from issue #244's BoatTrader reproducer:
    ``Read the URL in the address bar and confirm the page shows
    boat listing cards``. This intent decomposed into an
    extract_data without gate, ran the marketplace_listings
    extractor on the search URL, and got rejected."""
    s = PlanDecomposer._build_intent(
        _step("Read the URL in the address bar and confirm boat listings show")
    )
    assert s.gate is True


def test_build_intent_promotes_wait_for_intent_to_gate() -> None:
    s = PlanDecomposer._build_intent(
        _step("Wait for the search results page to fully load")
    )
    assert s.gate is True


def test_build_intent_keeps_authoritative_extract_data_ungated() -> None:
    """An ``extract_data`` whose intent reads as authoritative
    extraction (read fields, scrape, harvest) must NOT be promoted
    — that's the existing deep-extract path the recipe schema is
    designed for."""
    s = PlanDecomposer._build_intent(
        _step("Read structured listing fields (year, make, price) from this card")
    )
    assert s.gate is False

    s2 = PlanDecomposer._build_intent(
        _step("Extract the boat detail page including title, price, and seller info")
    )
    assert s2.gate is False

    s3 = PlanDecomposer._build_intent(
        _step("Scrape the visible job listings into the result set")
    )
    assert s3.gate is False


def test_build_intent_respects_explicit_gate_false_override() -> None:
    """If the source plan / cached JSON explicitly sets
    ``gate=False``, the heuristic must NOT silently flip it. An
    operator override is the most reliable signal — defending
    against autonomous flipping prevents the heuristic from
    becoming unrideable."""
    s = PlanDecomposer._build_intent({
        "intent": "Verify the page loaded",
        "type": "extract_data",
        "gate": False,
    })
    # Explicitly set False stays False — no auto-promotion.
    assert s.gate is False


def test_build_intent_respects_explicit_gate_true() -> None:
    s = PlanDecomposer._build_intent({
        "intent": "Verify the page loaded",
        "type": "extract_data",
        "gate": True,
    })
    assert s.gate is True


def test_build_intent_does_not_promote_non_extract_data_step_types() -> None:
    """The auto-promotion is scoped to ``extract_data`` — the only
    step type that fans into the schema-reject cascade. A submit/
    fill_field/select_option step with ``Verify`` in its intent
    should keep its declared gate (default False) — those types
    don't reach the deep-extract path anyway."""
    for step_type in ("submit", "fill_field", "select_option", "navigate", "click"):
        s = PlanDecomposer._build_intent(
            _step("Verify by clicking Save", type_=step_type)
        )
        assert s.gate is False, f"{step_type!r} got auto-promoted but shouldn't"


def test_build_intent_case_insensitive_trigger_match() -> None:
    """The verification triggers shouldn't be brittle to casing —
    Claude sometimes lowercases instructions."""
    for intent in (
        "VERIFY the page rendered",
        "verify the page rendered",
        "Confirm The page rendered",
    ):
        s = PlanDecomposer._build_intent(_step(intent))
        assert s.gate is True


# ── End-to-end: from_dict surface ─────────────────────────────────


def test_micro_plan_from_dict_applies_promotion_through_build_intent() -> None:
    """``MicroPlan.from_dict`` runs every step through ``_build_intent``
    so cached JSON plans (which is how this hits production via
    the host integration's plan_json shortcut) also benefit
    from the safety net."""
    payload = {
        "steps": [
            {"intent": "Navigate to https://www.example.com",
             "type": "navigate",
             "params": {"url": "https://www.example.com"}},
            {"intent": "Verify the page rendered the expected header",
             "type": "extract_data"},
        ],
    }
    plan = MicroPlan.from_dict(payload)
    assert plan.steps[0].type == "navigate"
    assert plan.steps[1].type == "extract_data"
    assert plan.steps[1].gate is True
