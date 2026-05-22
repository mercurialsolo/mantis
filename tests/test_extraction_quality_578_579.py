r"""Tests for #578 (decomposer emits expect_url_contains) + #579
(extractor rejects UNKNOWN-only leads).

These two bugs surfaced together on boattrader run
``20260521_234022_65ddaaff`` (2026-05-21 PM): the agent navigated
off the listings flow to a marketing CTA (``/boat-loans/``), Claude
extracted all-``<UNKNOWN>`` fields, the viability check accepted the
junk row as a "1 lead" output, and there was no URL-hint short-circuit
on the gate to catch the wrong-page situation upstream.

Both fixes ship in one PR because they're complementary defences for
the same failure mode — one catches the wrong page (#578 gate hints +
PR #577 short-circuit fall-through to vision), the other catches the
junk extract that gets past (#579 viability check).
"""

from __future__ import annotations

from mantis_agent.extraction.result import ExtractionResult
from mantis_agent.extraction.schema import ExtractionSchema
from mantis_agent.plan_decomposer import MicroIntent, MicroPlan, PlanDecomposer


# ── #579: ExtractionResult.is_viable rejects UNKNOWN placeholders ──


def _result_legacy(**fields) -> ExtractionResult:
    return ExtractionResult(**fields)


def _schema() -> ExtractionSchema:
    return ExtractionSchema(
        entity_name="boat",
        fields=[{"name": "year", "type": "str"}, {"name": "make", "type": "str"}],
        required_fields=["year", "make"],
    )


def _result_schema(year: str = "", make: str = "") -> ExtractionResult:
    r = ExtractionResult(_schema=_schema())
    r.extracted_fields = {"year": year, "make": make}
    return r


def test_legacy_all_empty_not_viable() -> None:
    assert _result_legacy(year="", make="").is_viable() is False


def test_legacy_unknown_placeholder_not_viable() -> None:
    # The bug from boattrader run 20260521_234022_65ddaaff: model
    # returned "<UNKNOWN>" for every field, dedup accepted it.
    r = _result_legacy(year="<UNKNOWN>", make="<UNKNOWN>", seller="private seller")
    assert r.is_viable() is False
    assert "year" in r.missing_required_reason()
    assert "make" in r.missing_required_reason()


def test_legacy_none_placeholder_not_viable() -> None:
    r = _result_legacy(year="none", make="N/A", seller="private seller")
    assert r.is_viable() is False


def test_legacy_real_values_viable() -> None:
    r = _result_legacy(
        year="2020", make="Sea Ray", seller="John Smith (private)",
    )
    assert r.is_viable() is True


def test_legacy_year_real_make_unknown_not_viable() -> None:
    # Partial-UNKNOWN still fails the required check.
    r = _result_legacy(year="2020", make="<UNKNOWN>", seller="private")
    assert r.is_viable() is False


def test_schema_all_unknown_not_viable() -> None:
    r = _result_schema(year="<UNKNOWN>", make="<UNKNOWN>")
    r.is_dealer = False  # ensure not classified as dealer
    assert r.is_viable() is False
    reason = r.missing_required_reason()
    assert "year" in reason and "make" in reason


def test_schema_real_values_viable() -> None:
    r = _result_schema(year="2020", make="Sea Ray")
    r.is_dealer = False
    assert r.is_viable() is True


def test_unknown_check_handles_whitespace_and_case() -> None:
    # Mixed case + whitespace must still be treated as unknown.
    assert ExtractionResult._is_unknown("  Unknown  ")
    assert ExtractionResult._is_unknown("<UNKNOWN>")
    assert ExtractionResult._is_unknown("None")
    assert ExtractionResult._is_unknown("N/A")
    assert ExtractionResult._is_unknown("")
    assert ExtractionResult._is_unknown(None)


def test_unknown_check_does_not_strip_real_values() -> None:
    assert not ExtractionResult._is_unknown("2020")
    assert not ExtractionResult._is_unknown("Sea Ray")
    assert not ExtractionResult._is_unknown("0")  # numeric "0" is a real value


# ── #578: decomposer injects expect_url_contains on gate steps ─────


def _plan_with_navigate_and_gate(nav_url: str = "") -> MicroPlan:
    return MicroPlan(steps=[
        MicroIntent(
            intent=f"Navigate to {nav_url}" if nav_url else "Navigate",
            type="navigate",
            params={"url": nav_url} if nav_url else {},
            section="setup", required=True,
        ),
        MicroIntent(
            intent="Verify page heading shows boats for sale",
            type="extract_data",
            params={}, hints={},
            section="setup", required=True,
            gate=True, claude_only=True,
        ),
    ])


def test_url_hint_segments_picks_distinctive_only() -> None:
    url = "https://www.boattrader.com/boats/state-fl/city-miami/zip-33101/by-owner/radius-25/"
    segments = PlanDecomposer._extract_url_hint_segments(url)
    # All distinctive (digit or hyphen). "boats" excluded — too generic.
    assert "state-fl" in segments
    assert "city-miami" in segments
    assert "zip-33101" in segments
    assert "by-owner" in segments
    assert "radius-25" in segments
    assert "boats" not in segments


def test_url_hint_segments_excludes_query_and_fragment() -> None:
    url = "https://x.com/a-b/c-1?page=2#anchor"
    segments = PlanDecomposer._extract_url_hint_segments(url)
    assert "a-b" in segments
    assert "c-1" in segments
    # Query string and fragment must not leak into the hints — they
    # change across same-page navigations.
    assert all("page=" not in s for s in segments)
    assert all("anchor" not in s for s in segments)


def test_url_hint_segments_empty_for_bare_host() -> None:
    assert PlanDecomposer._extract_url_hint_segments("https://x.com/") == []
    assert PlanDecomposer._extract_url_hint_segments("") == []


def test_url_hint_segments_skips_generic_resource() -> None:
    # "discover" / "boats" — no digit, no hyphen — too generic.
    assert PlanDecomposer._extract_url_hint_segments("https://lu.ma/discover") == []
    assert PlanDecomposer._extract_url_hint_segments("https://x.com/boats/") == []


def test_inject_gate_url_hints_from_params_url() -> None:
    plan = _plan_with_navigate_and_gate(
        "https://www.boattrader.com/boats/state-fl/zip-33101/by-owner/",
    )
    PlanDecomposer._inject_gate_url_hints(plan)
    gate = plan.steps[1]
    assert gate.hints["expect_url_contains"] == ["state-fl", "zip-33101", "by-owner"]


def test_inject_gate_url_hints_from_intent_when_no_params() -> None:
    plan = MicroPlan(steps=[
        MicroIntent(
            intent="Navigate to https://www.boattrader.com/zip-33101/",
            type="navigate",
            params={},  # no URL in params — only in intent
            section="setup", required=True,
        ),
        MicroIntent(
            intent="Verify on listings page",
            type="extract_data", params={}, hints={},
            section="setup", required=True,
            gate=True, claude_only=True,
        ),
    ])
    PlanDecomposer._inject_gate_url_hints(plan)
    assert plan.steps[1].hints["expect_url_contains"] == ["zip-33101"]


def test_inject_gate_url_hints_preserves_author_hints() -> None:
    plan = _plan_with_navigate_and_gate(
        "https://www.boattrader.com/zip-33101/",
    )
    plan.steps[1].hints = {"expect_url_contains": ["custom-hint"]}
    PlanDecomposer._inject_gate_url_hints(plan)
    # Author-supplied hints win — never overridden.
    assert plan.steps[1].hints["expect_url_contains"] == ["custom-hint"]


def test_inject_gate_url_hints_skips_when_no_navigate() -> None:
    # Gate without a preceding navigate — nothing to derive from.
    plan = MicroPlan(steps=[
        MicroIntent(
            intent="Verify state", type="extract_data",
            params={}, hints={}, gate=True, claude_only=True,
        ),
    ])
    PlanDecomposer._inject_gate_url_hints(plan)
    assert plan.steps[0].hints == {}


def test_inject_gate_url_hints_skips_when_url_has_no_distinctive_segments() -> None:
    # /discover has no digit or hyphen — no hints to inject.
    plan = _plan_with_navigate_and_gate("https://lu.ma/discover")
    PlanDecomposer._inject_gate_url_hints(plan)
    assert "expect_url_contains" not in plan.steps[1].hints


def test_inject_gate_url_hints_skips_non_gate_extract_data() -> None:
    # Deep-extract (gate=False, claude_only=True) is NOT the short-
    # circuit target — only gate=True. Hints would do nothing here.
    plan = _plan_with_navigate_and_gate(
        "https://www.boattrader.com/zip-33101/",
    )
    plan.steps[1].gate = False
    PlanDecomposer._inject_gate_url_hints(plan)
    assert "expect_url_contains" not in plan.steps[1].hints
