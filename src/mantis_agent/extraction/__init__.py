"""Claude-based data extraction.

Public surface (re-exported here for backwards compatibility — the
single-file ``mantis_agent.extraction`` was split into a package in
PR #105 to keep each module under 800 lines):

- :class:`ExtractionSchema` — domain-agnostic schema dataclass
- :class:`ExtractionResult` — structured row returned by ``extract()``
- :class:`ClaudeExtractor` — runtime that calls Claude and parses output
- :func:`parse_bool`, :func:`contains_dealer_text`,
  :func:`seller_looks_like_dealer` — generic helpers

Legacy underscore-prefixed names (``_parse_bool``,
``_contains_dealer_text``, ``_seller_looks_like_dealer``) are aliased
for one minor release.

``DEALER_TEXT_INDICATORS`` and ``DEALER_SELLER_INDICATORS`` were moved
under ``mantis_agent.recipes.marketplace_listings._data`` in PR #102.
Module-level access via PEP 562 ``__getattr__`` is preserved here with a
``DeprecationWarning``.
"""

from __future__ import annotations

from typing import Any

from .extractor import (
    EXTRACT_MULTI_SCREENSHOT_PROMPT,
    EXTRACT_PROMPT,
    EXTRACT_SCROLLED_PROMPT,
    FIND_LISTING_CONTENT_CONTROL_PROMPT,
    ClaudeExtractor,
)
from .result import ExtractionResult
from .schema import ExtractionSchema
from .spam import (
    _contains_dealer_text,
    _parse_bool,
    _seller_looks_like_dealer,
    contains_dealer_text,
    parse_bool,
    seller_looks_like_dealer,
)


def __getattr__(name: str) -> Any:  # PEP 562
    """Lazy-load ``DEALER_TEXT_INDICATORS`` / ``DEALER_SELLER_INDICATORS``
    from the marketplace_listings recipe with a ``DeprecationWarning``.

    Kept on the package root because that's where callers previously
    imported them from (``from mantis_agent.extraction import
    DEALER_TEXT_INDICATORS``).
    """
    if name in ("DEALER_TEXT_INDICATORS", "DEALER_SELLER_INDICATORS"):
        import warnings

        warnings.warn(
            f"mantis_agent.extraction.{name} has moved to "
            f"mantis_agent.recipes.marketplace_listings._data.{name}",
            DeprecationWarning,
            stacklevel=2,
        )
        from ..recipes.marketplace_listings import _data as _md

        return getattr(_md, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ExtractionSchema",
    "ExtractionResult",
    "ClaudeExtractor",
    # Generic helpers
    "parse_bool",
    "contains_dealer_text",
    "seller_looks_like_dealer",
    # Prompt constants used by callers that build their own pipelines
    "EXTRACT_PROMPT",
    "EXTRACT_SCROLLED_PROMPT",
    "EXTRACT_MULTI_SCREENSHOT_PROMPT",
    "FIND_LISTING_CONTENT_CONTROL_PROMPT",
    # Legacy underscore aliases — one minor release
    "_parse_bool",
    "_contains_dealer_text",
    "_seller_looks_like_dealer",
]
