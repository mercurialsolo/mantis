"""Schema for the marketplace_listings recipe.

Owns the constants directly (see ``_data.py``). The core's deprecated
``ExtractionSchema.default_boattrader()`` now imports from here, not the
other way round.

Public surface:

- ``SCHEMA``  — the ``ExtractionSchema`` instance to feed ``ClaudeExtractor``
- ``FIELDS``  — the field definitions
"""

from __future__ import annotations

from ...extraction import ExtractionSchema
from . import _data


def _build_schema() -> ExtractionSchema:
    return ExtractionSchema(
        entity_name=_data.ENTITY_NAME,
        fields=_data.fields(),
        required_fields=list(_data.REQUIRED_FIELDS),
        spam_indicators=list(_data.DEALER_TEXT_INDICATORS),
        spam_seller_indicators=list(_data.DEALER_SELLER_INDICATORS),
        spam_label=_data.SPAM_LABEL,
        forbidden_controls=list(_data.FORBIDDEN_CONTROLS),
        allowed_controls=list(_data.ALLOWED_CONTROLS),
    )


SCHEMA: ExtractionSchema = _build_schema()
FIELDS: list[dict] = SCHEMA.fields
