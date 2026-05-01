"""Deprecated alias — moved to mantis_agent.recipes.marketplace_listings.rewards.

The terminal-grader for marketplace-listing extraction lives under the
recipe now. This module re-exports the new symbols for one minor release
so existing imports keep working.

Importing this module is silent. ``BoatTraderReward()`` instantiation
emits a ``DeprecationWarning``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

from ..recipes.marketplace_listings.rewards import (
    MarketplaceListingReward as _MarketplaceListingReward,
)
from ..recipes.marketplace_listings.rewards import _parse_summary as _parse_summary_impl


@dataclass
class BoatTraderReward(_MarketplaceListingReward):
    """Deprecated alias of :class:`MarketplaceListingReward`.

    Pre-sets ``allowed_domains=("boattrader.com",)`` so legacy callers
    that did ``BoatTraderReward()`` keep grading against the same domain.
    """

    allowed_domains: tuple[str, ...] = field(default=("boattrader.com",))

    def __post_init__(self) -> None:  # dataclass post-init
        warnings.warn(
            "mantis_agent.rewards.boattrader.BoatTraderReward is deprecated; "
            "use mantis_agent.recipes.marketplace_listings.rewards."
            "MarketplaceListingReward instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        parent_post = getattr(super(), "__post_init__", None)
        if callable(parent_post):
            parent_post()


def _parse_summary(summary: str) -> dict[str, Any]:
    """Deprecated alias for ``recipes.marketplace_listings.rewards._parse_summary``."""
    return _parse_summary_impl(summary)


__all__ = ["BoatTraderReward", "_parse_summary"]
