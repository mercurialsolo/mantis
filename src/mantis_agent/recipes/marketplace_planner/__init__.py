"""marketplace_planner recipe — legacy text-plan → micro-plan optimizer
scoped to marketplace listings (BoatTrader-style).

This package holds the CLI tool previously living at
``src/mantis_agent/opus_planner.py``. The tool was never imported by any
core code path — only invoked via its own ``__main__`` block. Per issue
#462 it now lives under the marketplace_listings recipe family so the
marketplace-specific system prompts and planning rules are clearly
recipe-scoped rather than masquerading as generic CUA infrastructure.

It is *not* on the default planning path. The generic objective →
micro-plan path goes through :mod:`mantis_agent.plan_decomposer`
(``--micro``) and the recipe-aware graph stack
(``--graph-learn``).

Invoke the CLI with::

    python -m mantis_agent.recipes.marketplace_planner.planner <plan_file>
    python -m mantis_agent.recipes.marketplace_planner.planner <plan_file> --browse <screenshots_dir>
"""

from __future__ import annotations
