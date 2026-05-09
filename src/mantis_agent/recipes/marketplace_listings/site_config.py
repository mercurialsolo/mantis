"""SiteConfig overlay for the marketplace_listings recipe.

Carries URL/pagination patterns and gate-prompt scaffolding for boat-
trader-style listing sites. Sibling to ``schema.py``: where the schema
exports ``SCHEMA: ExtractionSchema``, this module exports
``SITE_CONFIG: SiteConfig``. Both are picked up by the symmetric
loaders in ``recipes/__init__.py``.

Per issue #224 this is an overlay, not a primary config — it
extends the probe-derived default with patterns the probe can't
infer from a single landing screenshot (the detail-URL slug shape is
the canonical example).
"""

from __future__ import annotations

from ...site_config import SiteConfig


SITE_CONFIG: SiteConfig = SiteConfig.default_boattrader()
