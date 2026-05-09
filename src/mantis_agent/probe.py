"""Probe-emitter façade — first-paint inspection of a site (issue #224 Phase 3).

Pairs with the Phase 1 overlay primitives (``SiteConfig.overlay``) and
the Phase 2 :func:`mantis_agent.objective.derive_from_plan` producer.
Phase 3 ships :func:`run_probe`, a thin function-style entry point
that takes a browser env + URL and returns a :class:`ProbeResult`
that ``SiteConfig.from_probe`` consumes:

    from mantis_agent import objective, probe, recipes
    from mantis_agent.extraction import ExtractionSchema
    from mantis_agent.site_config import SiteConfig

    spec   = objective.derive_from_plan(plan_text)
    schema = ExtractionSchema.from_objective(spec).overlay(
        recipes.load_schema(name) if name else None
    )

    p      = probe.run_probe(env, start_url, objective=spec)
    site   = SiteConfig.from_probe(p).overlay(
        recipes.load_site_config(name) if name else None
    )

The heavy lifting (Xvfb-backed Chromium, scroll-and-screenshot,
Claude vision analysis) lives on :class:`SiteProber` — :func:`run_probe`
is a 3-line wrapper that hides the prober's class shape from callers
who just want the result.

Top-level re-export of :class:`ProbeResult` so
``from mantis_agent.probe import ProbeResult, run_probe`` is the
canonical import path.
"""

from __future__ import annotations

import logging
from typing import Any

from .graph.objective import ObjectiveSpec
from .graph.probe import ProbeResult, SiteProber

__all__ = ["ProbeResult", "run_probe"]

logger = logging.getLogger(__name__)


def run_probe(
    env: Any,
    url: str,
    *,
    objective: ObjectiveSpec | None = None,
    api_key: str = "",
    model: str = "claude-haiku-4-5-20251001",
) -> ProbeResult:
    """Probe a site at first-paint and return a structured result.

    Issue #224 Phase 3: thin façade over :class:`SiteProber.probe`
    so callers don't have to know the prober's class API. The probe
    navigates the env to ``url``, captures four scroll-position
    screenshots, and asks Claude to identify page type, filters,
    listing container shape, pagination controls, and dealer / sponsored
    signals. The returned :class:`ProbeResult` is the input shape
    :meth:`SiteConfig.from_probe` consumes.

    Args:
        env: Browser env supporting ``reset(task, start_url)``,
            ``screenshot()``, and ``step(Action(...))``. Both
            ``XdotoolGymEnv`` (Modal-side, Xvfb + Chromium) and
            ``PlaywrightGymEnv`` (local) satisfy this contract.
        url: URL to probe. The env navigates to this URL on
            ``reset(start_url=url)``.
        objective: Optional :class:`ObjectiveSpec` giving the prober
            ``target_entity`` and ``raw_text`` to anchor the
            screenshot-analysis prompt. When omitted, an empty spec
            is constructed and the prober runs in domain-neutral mode.
        api_key: Anthropic API key. Falls back to the
            ``ANTHROPIC_API_KEY`` env var. When neither is set, the
            probe still navigates and captures screenshots — it just
            returns a partial :class:`ProbeResult` (URL + domain only,
            no analysis fields).
        model: Anthropic model for screenshot analysis. Defaults to
            a Haiku-tier model — the probe analysis is a structured
            JSON lift, not deep reasoning, and probes can be batchy
            on first-paint of a new domain.

    Returns:
        :class:`ProbeResult` with ``url`` and ``domain`` always
        populated. Other fields (page_type, filters_detected,
        listing_container, pagination_controls, dealer_signals,
        sponsored_signals, estimated_listings_per_page) populated
        when the Claude analysis call succeeds; left at their
        dataclass defaults otherwise. Callers can chain into
        :meth:`SiteConfig.from_probe` regardless — the from_probe
        path tolerates partial results.
    """
    if objective is None:
        objective = ObjectiveSpec(raw_text="")
    prober = SiteProber(env=env, api_key=api_key, model=model)
    return prober.probe(url=url, objective=objective)
