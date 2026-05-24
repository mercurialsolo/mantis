"""DomainProfile registry — per-domain decomposer knobs (#648).

A :class:`DomainProfile` records the safe defaults the decomposer
should stamp on steps when a plan targets a known domain — values that
operators repeatedly discovered by trial and error in past runs.
Today: scroll count / backend, first-navigate wait, pagination URL
template. Tomorrow (#643 hint memory): the same registry becomes the
write target for the auto-learner.

Authoring rules
---------------

1. **Plan-author override wins.** ``apply_domain_profile`` only stamps
   a field when the source plan didn't set it. Operators retain full
   control by writing the knob explicitly.
2. **Profile defaults are opinions, not facts.** An entry exists when
   we've observed a recurring failure pattern across multiple runs of
   the same domain. Speculative entries belong on a feature branch,
   not in ``DOMAIN_PROFILES``.
3. **Domain matching is suffix-based.** ``plan.domain="www.boattrader.com"``
   matches a registry entry ``"boattrader.com"``. Exact match wins
   over suffix match.

The decomposer wires the pass in
:meth:`mantis_agent.plan_decomposer.PlanDecomposer.decompose_text`
after the fresh / cache paths and before the loop / URL / gate
normalization passes — see the call site for the integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..plan_decomposer import MicroPlan


@dataclass(frozen=True)
class DomainProfile:
    """Per-domain knobs that the decomposer applies after step
    generation. Frozen so operators can hand instances around without
    worrying about accidental mutation.

    Fields with the sentinel ``0`` (ints) or ``""`` (strings) signal
    "no opinion" — the apply pass skips them. Use small explicit values
    (``count=2``, ``nav_wait_seconds=8``) rather than re-encoding "no
    opinion" with negative numbers.

    Attributes
    ----------
    domain
        Suffix-matched against ``MicroPlan.domain``. Typically the
        registrable hostname (``boattrader.com``, ``lu.ma``).
    scroll_backend
        Default ``params.backend`` for ``scroll`` steps. ``"cdp"`` opts
        into the deterministic CDP path that bypasses Chrome wheel
        handlers — set on domains where vision scroll repeatedly hits
        ``brain_loop_exhausted`` (sticky overlays, inner-element
        scrollers). Empty ``""`` means "no opinion".
    scroll_count
        Default ``params.count`` for ``scroll`` steps. Positive int
        replaces decomposer-emitted counts when the plan didn't pin
        one; ``0`` means no opinion. Used to stop overshoot on short
        detail pages.
    nav_wait_seconds
        Default ``params.wait_after_load_seconds`` for the FIRST
        ``navigate`` step in the plan. Set on JS-heavy SPAs / CF-gated
        domains where the 4-second adaptive settle expires before
        first paint. ``0`` means no opinion.
    pagination_url_template
        Sets ``MicroPlan.pagination_url_template`` when the LLM didn't
        infer one. Must contain ``{base}`` and ``{n}`` placeholders.
        Empty string means no opinion.
    """

    domain: str
    scroll_backend: str = ""
    scroll_count: int = 0
    nav_wait_seconds: int = 0
    pagination_url_template: str = ""
    # Free-form note explaining WHY the entry exists — read by future
    # operators / the hint-memory writer (#643). Not consumed by the
    # apply pass.
    note: str = ""
    # Optional URL substring hints that any ``gate`` step on this
    # domain should expect (e.g. ``("/boat/", "boat-")`` so the
    # extract_data gate short-circuits via #563 URL match). Empty
    # tuple means no opinion.
    expect_url_contains: tuple[str, ...] = field(default_factory=tuple)


# ── Registry ─────────────────────────────────────────────────────────
#
# Add an entry when you've observed a recurring failure pattern on a
# domain that the decomposer's generic defaults didn't catch. Keep the
# entry small and motivated by concrete run data.
DOMAIN_PROFILES: dict[str, DomainProfile] = {
    # boattrader.com — v7/v8 runs:
    #   * count=4 overshoots short detail pages (description in
    #     footer) → count=2 lands the Description heading mid-viewport.
    #   * vision scroll hits ``brain_loop_exhausted`` 173x per run on
    #     listing pages → CDP backend skips Chrome wheel handlers.
    #   * CF-protected proxy fetch sometimes hasn't rendered by the 4s
    #     adaptive settle → 8s on the first navigate.
    #   * /page-{n}/ pagination shape observed across the by-owner /
    #     by-make / state-wide variants.
    "boattrader.com": DomainProfile(
        domain="boattrader.com",
        scroll_backend="cdp",
        scroll_count=2,
        nav_wait_seconds=8,
        pagination_url_template="{base}/page-{n}/",
        note=(
            "v7/v8: CDP scroll skips brain_loop_exhausted; "
            "count=2 lands Description mid-viewport on short pages; "
            "8s nav wait covers CF-proxy first paint."
        ),
        expect_url_contains=("/boat/",),
    ),
    # lu.ma — SPA cold-mount: Vite/React app hasn't mounted in 4s on
    # a fresh profile. ``feedback_spa_cold_mount_wait.md`` documents
    # the pattern. Mechanical scroll knobs unset — lu.ma pages are
    # generally short and vision scroll is fine there.
    "lu.ma": DomainProfile(
        domain="lu.ma",
        nav_wait_seconds=8,
        note=(
            "SPA cold-mount: Vite/React app needs 8s before first "
            "interactive element renders on a fresh Chrome profile."
        ),
    ),
    # staff-crm-long (synthetic test app served by sim_envs) — same
    # SPA cold-mount story as lu.ma, plus tab-walk grounding is
    # finicky enough that operators have repeatedly added an 8s wait
    # to the first navigate. See ``project_staff_crm_long_*`` memory
    # files for the multi-day debug history.
    "staffai.com": DomainProfile(
        domain="staffai.com",
        nav_wait_seconds=8,
        note=(
            "Multi-day stress test surfaced session layout drift + "
            "trust-gated dispatch (PR #447). 8s nav wait makes the "
            "first interactive paint reliable."
        ),
    ),
}


def resolve_domain_profile(domain: str) -> DomainProfile | None:
    """Look up a :class:`DomainProfile` for ``domain``.

    Exact match wins over suffix match. ``""`` / unknown domains
    return ``None``. Case-insensitive: the registry is canonicalised
    on lookup so plan authors don't have to lowercase their URLs.
    """
    if not domain:
        return None
    needle = domain.strip().lower()
    if not needle:
        return None
    # Exact match — direct hit.
    hit = DOMAIN_PROFILES.get(needle)
    if hit is not None:
        return hit
    # Suffix match — registry "boattrader.com" matches plan
    # "www.boattrader.com" / "boats.boattrader.com" / etc.
    # We prefer the longest registry key that suffix-matches so a
    # specific subdomain entry (if ever added) wins over a generic
    # parent entry.
    best: DomainProfile | None = None
    best_len = -1
    for key, profile in DOMAIN_PROFILES.items():
        if needle.endswith("." + key) or needle == key:
            if len(key) > best_len:
                best, best_len = profile, len(key)
    return best


def apply_domain_profile(
    plan: "MicroPlan",
    profile: DomainProfile,
) -> "MicroPlan":
    """Stamp profile defaults on ``plan`` in place; return ``plan``.

    Idempotent — running twice yields the same plan because every
    decision checks "did the source plan / a prior pass already set
    this?" before stamping.

    Decisions
    ---------

    * For every ``scroll`` step:
      - if profile.scroll_backend and not step.params["backend"]:
            step.params["backend"] = profile.scroll_backend
      - if profile.scroll_count > 0 and not step.params["count"]:
            step.params["count"] = profile.scroll_count
    * For the FIRST ``navigate`` step in the plan:
      - if profile.nav_wait_seconds > 0 and
            not step.params["wait_after_load_seconds"]:
            step.params["wait_after_load_seconds"] = profile.nav_wait_seconds
    * For every ``gate`` step:
      - if profile.expect_url_contains and
            not step.hints["expect_url_contains"]:
            step.hints["expect_url_contains"] = list(profile.expect_url_contains)
    * Plan-level:
      - if profile.pagination_url_template and not plan.pagination_url_template:
            plan.pagination_url_template = profile.pagination_url_template
    """
    saw_navigate = False
    for step in plan.steps:
        stype = step.type
        params = step.params if step.params is not None else {}
        hints = step.hints if step.hints is not None else {}

        if stype == "scroll":
            if profile.scroll_backend and not params.get("backend"):
                params["backend"] = profile.scroll_backend
            if profile.scroll_count > 0 and not params.get("count"):
                params["count"] = profile.scroll_count

        if stype == "navigate" and not saw_navigate:
            saw_navigate = True
            if (
                profile.nav_wait_seconds > 0
                and not params.get("wait_after_load_seconds")
            ):
                params["wait_after_load_seconds"] = profile.nav_wait_seconds

        if step.gate and profile.expect_url_contains:
            if not hints.get("expect_url_contains"):
                hints["expect_url_contains"] = list(profile.expect_url_contains)

        # Re-assign so dataclass instances always carry concrete dicts
        # (some MicroIntent fields default to a fresh dict; mutation
        # is observable on the original, but explicit assignment is
        # safer for future field-shape changes).
        step.params = params
        step.hints = hints

    if profile.pagination_url_template and not plan.pagination_url_template:
        if "{base}" in profile.pagination_url_template and "{n}" in profile.pagination_url_template:
            plan.pagination_url_template = profile.pagination_url_template

    return plan
