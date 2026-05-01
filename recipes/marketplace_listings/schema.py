"""Schema for the marketplace_listings recipe.

Currently re-exports ``ExtractionSchema.default_boattrader()`` so existing
run paths are unaffected. The follow-up PR will invert the dependency: the
schema constants live here, and the core's ``default_boattrader()`` becomes
a thin loader that imports from this module.

Public surface:

- ``SCHEMA``  — the ``ExtractionSchema`` instance to feed ``ClaudeExtractor``
- ``FIELDS``  — the field definitions, kept separate for callers that build
  their own schema variants

Importing this module is cheap and side-effect free. It does not pull
torch / transformers / Anthropic; only the core's ``extraction`` module.
"""

from __future__ import annotations

from mantis_agent.extraction import ExtractionSchema

SCHEMA: ExtractionSchema = ExtractionSchema.default_boattrader()
FIELDS: list[dict] = SCHEMA.fields
