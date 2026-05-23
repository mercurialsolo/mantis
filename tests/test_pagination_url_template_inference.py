"""Pagination URL template inference tests (#629).

The decomposer (LLM) can now emit a plan-level ``pagination_url_template``
when the source plan contains URL hints. The orchestrator consults this
field BEFORE falling back to ``paginate.params['url_template']`` (back-
compat) and the global default.

These tests pin:

  - MicroPlan.pagination_url_template round-trips through to_dict /
    from_dict.
  - LLM response parser accepts the field when present + well-formed,
    rejects malformed templates (missing placeholders).
  - build_micro_suite / _build_plan_from_suite carry the field through.
  - prepare_modal_partitions resolves in the documented order:
    plan-level → paginate.params → default.
  - Prompt content carries the guidance Claude needs.
"""

from __future__ import annotations

from mantis_agent.gym.fanout_runner import (
    DEFAULT_PAGINATION_URL_TEMPLATE,
    prepare_modal_partitions,
)
from mantis_agent.plan_decomposer import (
    DECOMPOSE_PROMPT,
    MicroIntent,
    MicroPlan,
    PlanDecomposer,
)
from mantis_agent.server_utils import build_micro_suite


# ── MicroPlan field round-trip ─────────────────────────────────────────


def test_micro_plan_pagination_template_default_empty() -> None:
    plan = MicroPlan(domain="x.com")
    assert plan.pagination_url_template == ""


def test_micro_plan_pagination_template_round_trip() -> None:
    plan = MicroPlan(domain="x.com")
    plan.pagination_url_template = "{base}?page={n}"
    restored = MicroPlan.from_dict(plan.to_dict())
    assert restored.pagination_url_template == "{base}?page={n}"


def test_micro_plan_template_absent_in_legacy_payload() -> None:
    """Legacy payloads (cached before #629) had no field — restored
    plan defaults to empty string, not None."""
    payload = {
        "steps": [{"intent": "x", "type": "navigate"}],
        "shapes": ["listings"],
        # No pagination_url_template.
    }
    restored = MicroPlan.from_dict(payload)
    assert restored.pagination_url_template == ""


# ── Decomposer prompt content ──────────────────────────────────────────


def test_prompt_mentions_pagination_url_template_field() -> None:
    assert "pagination_url_template" in DECOMPOSE_PROMPT


def test_prompt_documents_placeholder_syntax() -> None:
    """Claude must know the literal placeholders to emit — pinning
    them so a future prompt edit can't silently drop the contract."""
    assert "{base}" in DECOMPOSE_PROMPT
    assert "{n}" in DECOMPOSE_PROMPT


def test_prompt_documents_common_patterns() -> None:
    """At least the BoatTrader and Zillow shapes should be in the
    prompt examples so Claude recognises them in source plans."""
    assert "{base}/page-{n}/" in DECOMPOSE_PROMPT
    assert "{base}?page={n}" in DECOMPOSE_PROMPT


def test_prompt_says_omit_when_no_evidence() -> None:
    """Critical contract: never invent a template based on domain alone.
    Without this, Claude would emit a guess for every plan and the
    orchestrator would synthesize broken URLs."""
    assert "OMIT" in DECOMPOSE_PROMPT or "omit" in DECOMPOSE_PROMPT
    assert "Never invent" in DECOMPOSE_PROMPT or "never invent" in DECOMPOSE_PROMPT


# ── build_micro_suite / orchestrator integration ───────────────────────


def test_build_micro_suite_carries_template_when_set() -> None:
    suite = build_micro_suite(
        [{"intent": "navigate", "type": "navigate", "params": {"url": "https://x.com/"}}],
        domain="x.com",
        pagination_url_template="{base}?page={n}",
    )
    assert suite["_pagination_url_template"] == "{base}?page={n}"


def test_build_micro_suite_omits_template_when_empty() -> None:
    """Empty template → field absent from suite. Keeps the payload
    clean for the common case where no template is inferred."""
    suite = build_micro_suite(
        [{"intent": "navigate", "type": "navigate"}],
        domain="x.com",
    )
    assert "_pagination_url_template" not in suite


def _pagination_suite_with_template(template: str) -> dict:
    """Build a suite_dict the orchestrator can consume — pagination
    plan with the plan-level template field set."""
    plan = MicroPlan(domain="x.com")
    plan.steps = [
        MicroIntent(
            intent="Navigate", type="navigate", section="setup",
            params={"url": "https://x.com/listings"},
        ),
        MicroIntent(intent="Click", type="click", section="extraction"),
        MicroIntent(intent="URL", type="extract_url", section="extraction"),
        MicroIntent(intent="Scroll", type="scroll", section="extraction"),
        MicroIntent(intent="Extract", type="extract_data", section="extraction"),
        MicroIntent(intent="Back", type="navigate_back", section="extraction"),
        MicroIntent(
            intent="Inner loop", type="loop", section="extraction",
            loop_target=1, loop_count=25,
        ),
        MicroIntent(intent="Paginate", type="paginate", section="pagination"),
        MicroIntent(
            intent="Outer loop", type="loop", section="pagination",
            loop_target=1, loop_count=3,
        ),
    ]
    PlanDecomposer._classify_loop_groups(plan)
    suite = {
        "session_name": "x",
        "_micro_plan": [
            {
                "intent": s.intent, "type": s.type, "section": s.section,
                "params": dict(s.params or {}),
                "loop_target": s.loop_target, "loop_count": s.loop_count,
                "claude_only": s.claude_only, "gate": s.gate,
                "required": s.required, "grounding": s.grounding,
                "verify": s.verify, "reverse": s.reverse,
                "budget": s.budget, "hints": dict(s.hints or {}),
            }
            for s in plan.steps
        ],
        "_loop_groups": [
            {
                "loop_step_idx": g.loop_step_idx,
                "body_range": list(g.body_range),
                "shape": g.shape,
            }
            for g in plan.loop_groups
        ],
    }
    if template:
        suite["_pagination_url_template"] = template
    return suite


def test_orchestrator_uses_plan_level_template() -> None:
    """Plan-level template wins over the default; partition URLs use
    the operator-supplied shape."""
    suite = _pagination_suite_with_template("{base}?page={n}")
    partitions = prepare_modal_partitions(suite, workers=2)
    nav_urls = [
        next(s["params"]["url"] for s in p["_micro_plan"] if s["type"] == "navigate")
        for p in partitions
    ]
    # Page 1 = bare base; pages 2-3 = templated.
    assert nav_urls[1] == "https://x.com/listings?page=2"
    assert nav_urls[2] == "https://x.com/listings?page=3"


def test_orchestrator_falls_back_to_default_when_template_empty() -> None:
    suite = _pagination_suite_with_template("")
    partitions = prepare_modal_partitions(suite, workers=2)
    nav_urls = [
        next(s["params"]["url"] for s in p["_micro_plan"] if s["type"] == "navigate")
        for p in partitions
    ]
    # Default = ``{base}/page-{n}/``.
    assert nav_urls[1] == "https://x.com/listings/page-2/"


def test_orchestrator_plan_level_wins_over_paginate_param() -> None:
    """When BOTH are set, plan-level wins (the documented resolution
    order). Paginate-step is back-compat for plans authored before
    the plan-level field existed."""
    suite = _pagination_suite_with_template("{base}?p={n}")
    # Also inject paginate-step level template — should be ignored.
    for step in suite["_micro_plan"]:
        if step["type"] == "paginate":
            step["params"]["url_template"] = "{base}?DIFFERENT={n}"
    partitions = prepare_modal_partitions(suite, workers=2)
    nav_urls = [
        next(s["params"]["url"] for s in p["_micro_plan"] if s["type"] == "navigate")
        for p in partitions
    ]
    assert nav_urls[1] == "https://x.com/listings?p=2"


def test_orchestrator_uses_paginate_step_template_when_no_plan_level() -> None:
    """Without plan-level field, fall back to paginate-step
    ``url_template`` (back-compat with #617's original surface)."""
    suite = _pagination_suite_with_template("")
    for step in suite["_micro_plan"]:
        if step["type"] == "paginate":
            step["params"]["url_template"] = "{base}/p/{n}"
    partitions = prepare_modal_partitions(suite, workers=2)
    nav_urls = [
        next(s["params"]["url"] for s in p["_micro_plan"] if s["type"] == "navigate")
        for p in partitions
    ]
    assert nav_urls[1] == "https://x.com/listings/p/2"


def test_orchestrator_rejects_malformed_plan_level_template() -> None:
    """A template missing ``{base}`` or ``{n}`` is malformed — fall
    through to paginate-step / default rather than synthesising
    broken URLs."""
    suite = _pagination_suite_with_template("just-a-string-no-placeholders")
    partitions = prepare_modal_partitions(suite, workers=2)
    nav_urls = [
        next(s["params"]["url"] for s in p["_micro_plan"] if s["type"] == "navigate")
        for p in partitions
    ]
    # Falls through to default.
    assert nav_urls[1] == "https://x.com/listings/page-2/"


# ── Default constant pinning ───────────────────────────────────────────


def test_default_pagination_template_unchanged() -> None:
    """The default stays ``{base}/page-{n}/`` (BoatTrader-style) per
    #617. #629 adds a per-plan override, not a default change."""
    assert DEFAULT_PAGINATION_URL_TEMPLATE == "{base}/page-{n}/"
