"""staff_crm recipe — SiteConfig overlay for in-app CRM (Staff CRM-style).

This recipe documents the structural shape of a staff/internal CRM
admin console (lead list with sidebar filter views + Priority dropdown
+ row-detail edit). Its primary purpose today is supplying
``SiteConfig.filter_url_strategies`` so the graph enhancer (and any
caller that resolves a recipe name) can emit direct URL navigations
for filter-change steps instead of relying on sidebar-link clicks
that may be broken or visually hard to ground.

Per the recipes contract the core never imports from a specific
recipe. Resolution goes through :func:`mantis_agent.recipes.load_site_config`
with the recipe name.
"""

from __future__ import annotations
