"""Listings-scan state owner — per-page card cache, seen-URL dedup, viewport.

Phase 1.2 of EPIC #161 (refactor MicroPlanRunner into composable
modules). Lifts 8 listings-specific attributes off the runner into a
single dataclass so they can be tested in isolation, persisted /
restored together, and eventually have their mutation logic pulled
out of ``run()`` (Phase 4).

This phase is **state-only**, no behavior change. The mutation logic
that uses these fields still lives on ``MicroPlanRunner.run()`` — it
just reads/writes through property delegates that point here. Phase 4
("de-leak listings") will move ``DUPLICATE`` semantics, the
``_listings_on_page`` injection, and the paginate-resets-everything
block into methods on ``ListingsScanner``.

State persisted by :class:`~.checkpoint_manager.CheckpointManager`:

- ``seen_urls``               — set of already-extracted detail URLs
- ``extracted_titles``        — exact card titles Claude returned
- ``page_listings``           — cached (x, y, title) coords for current viewport
- ``page_listing_index``      — next card to click from the cache
- ``viewport_stage``          — Home / PageDown / PageDown×2 / …
- ``results_base_url``        — search-results URL anchor (for paginate)
- ``required_filter_tokens``  — URL path tokens that must remain present
- ``max_viewport_stages``     — config: hard cap on viewport advancement

External readers go through property delegates on the runner; rename of
any of these fields is a coordinated rename across
:mod:`.checkpoint_manager`, :mod:`.browser_state`, and the test fixtures
in ``test_plan_aware_reverse.py`` / ``test_private_seller_filter.py`` /
``test_checkpoint_manager.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ListingsScanner:
    """Per-run state for results-page scanning.

    Initial values match the pre-refactor literals in
    ``MicroPlanRunner.__init__`` exactly so a checkpoint round-trip
    behaves identically.
    """

    seen_urls: set[str] = field(default_factory=set)
    extracted_titles: list[str] = field(default_factory=list)
    page_listings: list[tuple[int, int, str]] = field(default_factory=list)
    page_listing_index: int = 0
    viewport_stage: int = 0
    max_viewport_stages: int = 6
    results_base_url: str = ""
    required_filter_tokens: tuple[str, ...] = ()

    # Phase 4 will add behavior methods here (is_duplicate, on_page_change,
    # next_card, advance_viewport, etc.). Phase 1 is state-only — keep this
    # class boring on purpose so the leaf extraction doesn't sneak in any
    # behavior change.
