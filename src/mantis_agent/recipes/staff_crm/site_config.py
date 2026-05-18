"""SiteConfig overlay for the staff_crm recipe.

Carries URL/pagination patterns and the URL-filter encoding rules for
in-app staff/admin CRM workflows where the lead list supports
``?status=<value>&priority=<value>`` query-string filters but the
``LEAD VIEWS`` sidebar links are click-driven and may be visually
hard to ground (or have broken onclick interceptors on some
fixtures). Exposing the URL encoding here lets ``PlanEnhancer``
emit direct navigates for filter-change steps via the
``filter_url_strategies`` mechanism added in #464.

Same shape as ``recipes/marketplace_listings/site_config.py`` —
declare the SiteConfig overlay; the recipes loader picks it up via
:func:`mantis_agent.recipes.load_site_config`.
"""

from __future__ import annotations

from ...site_config import SiteConfig


# Maps lowercased objective / plan-text keywords to URL path segments
# (or templated segments with a ``{value}`` placeholder). Boolean
# keywords (no placeholder) toggle a fixed query-string fragment;
# templated keywords get a runtime-extracted value substituted in.
_FILTER_URL_STRATEGIES: dict[str, str] = {
    # ── Status filters — each "click 'Foo' in LEAD VIEWS" maps to a
    # query-string filter on /leads.
    "new leads": "status=New",
    "contacted": "status=Contacted",
    "qualified": "status=Qualified",
    "proposal": "status=Proposal",
    "negotiation": "status=Negotiation",
    "closed won": "status=Closed Won",
    "closed lost": "status=Closed Lost",
    # ── Priority filters — same shape on the BY PRIORITY sidebar /
    # the dropdown next to Status.
    "critical": "priority=Critical",
    "high": "priority=High",
    "medium": "priority=Medium",
    "low": "priority=Low",
}


SITE_CONFIG: SiteConfig = SiteConfig(
    # URL patterns for the lead/record surface — used by ``is_detail_page``
    # / ``is_results_page`` / pagination URL builders.
    detail_page_pattern=r"/leads/\d+",
    results_page_pattern=r"/leads(?:\?|$)",
    pagination_format="?page={n}",
    pagination_type="query_param",
    pagination_strip_pattern=r"[?&]page=\d+",
    filter_url_strategies=_FILTER_URL_STRATEGIES,
)
