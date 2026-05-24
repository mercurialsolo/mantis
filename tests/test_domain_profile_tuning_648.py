"""#648 — DomainProfile autonomous plan tuning.

Pins the contract for per-domain decomposer defaults:

  - Profile resolution: exact match wins, suffix match falls back.
  - Apply pass: stamps scroll backend / count, first-navigate wait,
    gate ``expect_url_contains``, and plan.pagination_url_template
    when the source plan didn't already pin them.
  - Idempotent: running twice yields the same plan.
  - Plan-author override: explicit values in the source plan / a prior
    pass survive — the profile never overwrites them.
  - Unknown domain: silent no-op.
"""

from __future__ import annotations

from mantis_agent.plan_decomposer import MicroIntent, MicroPlan, PlanDecomposer
from mantis_agent.plan_tuning import (
    DOMAIN_PROFILES,
    DomainProfile,
    apply_domain_profile,
    resolve_domain_profile,
)


# ── Resolution ────────────────────────────────────────────────────────


def test_resolve_returns_none_for_empty_or_unknown_domain():
    assert resolve_domain_profile("") is None
    assert resolve_domain_profile("totally-unknown-12345.tld") is None


def test_resolve_exact_match():
    p = resolve_domain_profile("boattrader.com")
    assert p is not None
    assert p.domain == "boattrader.com"


def test_resolve_suffix_match_for_subdomain():
    """``plan.domain="www.boattrader.com"`` matches the registry's
    ``"boattrader.com"`` via suffix matching."""
    p = resolve_domain_profile("www.boattrader.com")
    assert p is not None
    assert p.domain == "boattrader.com"


def test_resolve_is_case_insensitive():
    p = resolve_domain_profile("BoatTrader.COM")
    assert p is not None
    assert p.domain == "boattrader.com"


def test_resolve_strips_whitespace():
    assert resolve_domain_profile("  boattrader.com  ") is not None


# ── Apply: scroll knobs ───────────────────────────────────────────────


def _plan_with(steps: list[MicroIntent], domain: str = "boattrader.com") -> MicroPlan:
    return MicroPlan(steps=steps, domain=domain)


def test_apply_stamps_scroll_backend_when_unset():
    plan = _plan_with([MicroIntent(intent="x", type="scroll", params={})])
    profile = DomainProfile(domain="boattrader.com", scroll_backend="cdp")
    apply_domain_profile(plan, profile)
    assert plan.steps[0].params["backend"] == "cdp"


def test_apply_respects_plan_author_scroll_backend():
    plan = _plan_with(
        [MicroIntent(intent="x", type="scroll", params={"backend": "xdotool"})],
    )
    profile = DomainProfile(domain="boattrader.com", scroll_backend="cdp")
    apply_domain_profile(plan, profile)
    assert plan.steps[0].params["backend"] == "xdotool"


def test_apply_stamps_scroll_count_when_unset():
    plan = _plan_with([MicroIntent(intent="x", type="scroll", params={})])
    profile = DomainProfile(domain="boattrader.com", scroll_count=2)
    apply_domain_profile(plan, profile)
    assert plan.steps[0].params["count"] == 2


def test_apply_respects_plan_author_scroll_count():
    plan = _plan_with(
        [MicroIntent(intent="x", type="scroll", params={"count": 5})],
    )
    profile = DomainProfile(domain="boattrader.com", scroll_count=2)
    apply_domain_profile(plan, profile)
    assert plan.steps[0].params["count"] == 5


def test_apply_skips_scroll_count_when_profile_unset():
    """profile.scroll_count == 0 → "no opinion" → don't stamp anything."""
    plan = _plan_with([MicroIntent(intent="x", type="scroll", params={})])
    profile = DomainProfile(domain="x.com")  # scroll_count=0 default
    apply_domain_profile(plan, profile)
    assert "count" not in plan.steps[0].params


# ── Apply: first-navigate wait ────────────────────────────────────────


def test_apply_stamps_first_navigate_wait_seconds_when_unset():
    plan = _plan_with([
        MicroIntent(intent="go", type="navigate", params={"url": "https://x.com"}),
        MicroIntent(intent="scroll", type="scroll"),
    ])
    profile = DomainProfile(domain="x.com", nav_wait_seconds=8)
    apply_domain_profile(plan, profile)
    assert plan.steps[0].params["wait_after_load_seconds"] == 8


def test_apply_only_stamps_FIRST_navigate_wait():
    """Subsequent navigate steps don't get the cold-mount wait —
    that's specifically for the initial proxy/SPA paint."""
    plan = _plan_with([
        MicroIntent(intent="go 1", type="navigate"),
        MicroIntent(intent="scroll", type="scroll"),
        MicroIntent(intent="go 2", type="navigate"),
    ])
    profile = DomainProfile(domain="x.com", nav_wait_seconds=8)
    apply_domain_profile(plan, profile)
    assert plan.steps[0].params.get("wait_after_load_seconds") == 8
    assert "wait_after_load_seconds" not in plan.steps[2].params


def test_apply_respects_plan_author_first_navigate_wait():
    plan = _plan_with([
        MicroIntent(
            intent="go", type="navigate",
            params={"wait_after_load_seconds": 2},
        ),
    ])
    profile = DomainProfile(domain="x.com", nav_wait_seconds=8)
    apply_domain_profile(plan, profile)
    assert plan.steps[0].params["wait_after_load_seconds"] == 2


# ── Apply: gate expect_url_contains ───────────────────────────────────


def test_apply_stamps_expect_url_contains_on_gate_steps():
    plan = _plan_with([
        MicroIntent(intent="verify", type="extract_data", gate=True),
    ])
    profile = DomainProfile(
        domain="boattrader.com",
        expect_url_contains=("/boat/", "boat-"),
    )
    apply_domain_profile(plan, profile)
    assert plan.steps[0].hints["expect_url_contains"] == ["/boat/", "boat-"]


def test_apply_skips_expect_url_contains_on_non_gate_steps():
    plan = _plan_with([
        MicroIntent(intent="extract", type="extract_data", gate=False),
    ])
    profile = DomainProfile(
        domain="boattrader.com",
        expect_url_contains=("/boat/",),
    )
    apply_domain_profile(plan, profile)
    assert "expect_url_contains" not in plan.steps[0].hints


def test_apply_respects_plan_author_expect_url_contains():
    plan = _plan_with([
        MicroIntent(
            intent="verify", type="extract_data", gate=True,
            hints={"expect_url_contains": ["custom-path"]},
        ),
    ])
    profile = DomainProfile(
        domain="boattrader.com",
        expect_url_contains=("/boat/",),
    )
    apply_domain_profile(plan, profile)
    assert plan.steps[0].hints["expect_url_contains"] == ["custom-path"]


# ── Apply: plan-level pagination_url_template ─────────────────────────


def test_apply_stamps_pagination_url_template_when_plan_unset():
    plan = _plan_with([])
    profile = DomainProfile(
        domain="boattrader.com",
        pagination_url_template="{base}/page-{n}/",
    )
    apply_domain_profile(plan, profile)
    assert plan.pagination_url_template == "{base}/page-{n}/"


def test_apply_respects_plan_pagination_url_template():
    plan = _plan_with([])
    plan.pagination_url_template = "{base}/listing?p={n}"
    profile = DomainProfile(
        domain="boattrader.com",
        pagination_url_template="{base}/page-{n}/",
    )
    apply_domain_profile(plan, profile)
    assert plan.pagination_url_template == "{base}/listing?p={n}"


def test_apply_rejects_malformed_pagination_template():
    """Missing ``{base}`` or ``{n}`` placeholders → don't stamp."""
    plan = _plan_with([])
    profile = DomainProfile(
        domain="x.com",
        pagination_url_template="https://x.com/?page=N",  # no placeholders
    )
    apply_domain_profile(plan, profile)
    assert plan.pagination_url_template == ""


# ── Idempotency ───────────────────────────────────────────────────────


def test_apply_is_idempotent():
    """Running the pass twice yields the same plan as running it once."""
    plan = _plan_with([
        MicroIntent(intent="go", type="navigate", params={}),
        MicroIntent(intent="scroll", type="scroll", params={}),
        MicroIntent(intent="verify", type="extract_data", gate=True, hints={}),
    ])
    profile = DomainProfile(
        domain="boattrader.com",
        scroll_backend="cdp",
        scroll_count=2,
        nav_wait_seconds=8,
        pagination_url_template="{base}/page-{n}/",
        expect_url_contains=("/boat/",),
    )
    apply_domain_profile(plan, profile)
    snapshot = (
        plan.steps[0].params.copy(),
        plan.steps[1].params.copy(),
        plan.steps[2].hints.copy(),
        plan.pagination_url_template,
    )
    apply_domain_profile(plan, profile)
    assert (
        plan.steps[0].params,
        plan.steps[1].params,
        plan.steps[2].hints,
        plan.pagination_url_template,
    ) == snapshot


# ── Registry sanity ──────────────────────────────────────────────────


def test_boattrader_profile_known_knobs():
    p = DOMAIN_PROFILES["boattrader.com"]
    assert p.scroll_backend == "cdp"
    assert p.scroll_count == 2
    assert p.nav_wait_seconds == 8
    assert "{base}" in p.pagination_url_template
    assert "{n}" in p.pagination_url_template


def test_luma_profile_nav_wait_only():
    p = DOMAIN_PROFILES["lu.ma"]
    assert p.nav_wait_seconds == 8
    # No scroll opinions on lu.ma (short pages, vision scroll is fine).
    assert p.scroll_backend == ""
    assert p.scroll_count == 0


# ── Decomposer wiring ────────────────────────────────────────────────


def test_decomposer_apply_domain_tuning_pass_on_known_domain():
    """``_apply_domain_tuning`` stamps profile defaults on a plan whose
    ``domain`` resolves to a known profile."""
    plan = MicroPlan(domain="boattrader.com")
    plan.steps = [
        MicroIntent(intent="go", type="navigate"),
        MicroIntent(intent="scroll", type="scroll"),
    ]
    PlanDecomposer._apply_domain_tuning(plan)
    # First-navigate wait stamped from profile.
    assert plan.steps[0].params.get("wait_after_load_seconds") == 8
    # Scroll knobs stamped from profile.
    assert plan.steps[1].params.get("backend") == "cdp"
    assert plan.steps[1].params.get("count") == 2
    # Plan-level template stamped from profile.
    assert plan.pagination_url_template == "{base}/page-{n}/"


def test_decomposer_apply_domain_tuning_is_silent_on_unknown_domain():
    plan = MicroPlan(domain="totally-unknown-12345.tld")
    plan.steps = [MicroIntent(intent="x", type="scroll")]
    PlanDecomposer._apply_domain_tuning(plan)
    assert plan.steps[0].params == {}


def test_decomposer_apply_domain_tuning_silent_when_no_domain():
    """``MicroPlan.domain=""`` (regex didn't find a URL) → no-op."""
    plan = MicroPlan(domain="")
    plan.steps = [MicroIntent(intent="x", type="scroll")]
    PlanDecomposer._apply_domain_tuning(plan)
    assert plan.steps[0].params == {}
