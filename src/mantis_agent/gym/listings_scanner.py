"""Listings-scan state + behavior — per-page card cache, seen-URL dedup, viewport.

Originally landed in EPIC #161 Phase 1.2 as a state-only container;
Phase 4 ("de-leak listings") added the behavior methods that used to
live in :meth:`MicroPlanRunner.run` and the registered handlers.

Behavior methods now own:

- :meth:`is_duplicate`       — DUPLICATE check the runner used to do
                               via ``url in self._seen_urls``
- :meth:`mark_seen`          — adds a URL to the dedup set after a
                               successful extraction
- :meth:`on_page_change`     — paginate-success reset (clears card
                               cache, viewport, extracted titles)
- :meth:`scroll_directive`   — listings_on_page → click-step intent
                               injection (the "Scroll down past the
                               first N listings…" hint that the
                               executor used to build inline)
- :meth:`advance_attempted`  — increment the attempted-listings counter
                               (DUPLICATE handler, deep-extract success)

State fields stay the canonical names so :mod:`.checkpoint_manager`
and the existing test fixtures continue to round-trip them unchanged.

State persisted by :class:`~.checkpoint_manager.CheckpointManager`:

- ``seen_urls``               — set of already-extracted detail URLs
- ``extracted_titles``        — exact card titles Claude returned
- ``page_listings``           — cached (x, y, title) coords for current viewport
- ``page_listing_index``      — next card to click from the cache
- ``viewport_stage``          — Home / PageDown / PageDown×2 / …
- ``results_base_url``        — search-results URL anchor (for paginate)
- ``required_filter_tokens``  — URL path tokens that must remain present
- ``max_viewport_stages``     — config: hard cap on viewport advancement
- ``listings_attempted``      — count of listings the runner has clicked
                                into on this page (used by click intent
                                hint and by the DUPLICATE skip-counter)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ListingsScanner:
    """Per-run state + behavior for results-page scanning.

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
    # Listings-attempted counter (was ``MicroPlanRunner._listings_on_page``
    # / ``_page_listing_count``). Lives here so the DUPLICATE handler and
    # the click-intent injection can read it without crossing the
    # parent-back-reference boundary.
    listings_attempted: int = 0

    # ── Behavior — was inline in run() / _execute_step / handlers ──────

    def is_duplicate(self, url: str) -> bool:
        """Return True if ``url`` was already extracted this run.

        The pre-Phase-4 version of this check appeared in three places:
        ``ClaudeStepHandler`` (extract_url and extract_data branches)
        and the executor's `_handle_duplicate`. Each computed
        ``url and url in self._seen_urls`` inline; this method is the
        canonical implementation.
        """
        return bool(url) and url in self.seen_urls

    def mark_seen(self, url: str) -> None:
        """Record ``url`` as extracted. No-op for empty strings."""
        if url:
            self.seen_urls.add(url)

    def on_page_change(self) -> None:
        """Reset all per-page state — called when paginate succeeds.

        Lifts the 7-line block out of ``RunExecutor._handle_success``:
        clears extracted titles + page_listings cache + listing index,
        resets viewport_stage to 0 (fresh Home), zeros listings_attempted.
        Loop counters stay on the executor (they're plan-graph state,
        not listings-specific).
        """
        self.listings_attempted = 0
        self.extracted_titles = []
        self.page_listings = []
        self.page_listing_index = 0
        self.viewport_stage = 0

    @staticmethod
    def scroll_directive_for(listings_count: int) -> str | None:
        """Return the click-intent hint when ``listings_count > 0``.

        Stateless because the executor tracks the "attempted on this
        page" counter separately from ``listings_attempted`` (the
        executor's count survives DUPLICATE skips; the scanner's
        counter advances on successful clicks). Same wording as the
        pre-Phase-4 inline directive so plan-decomposer caches and
        downstream prompts stay byte-identical.
        """
        if listings_count <= 0:
            return None
        return (
            f"Scroll down past the first {listings_count} listings. "
            f"Then click the next listing title text below a photo."
        )

    def advance_attempted(self) -> None:
        """Bump the attempted counter — called after a click attempt
        succeeds (deep-extract path) or after DUPLICATE skip-to-loop.
        """
        self.listings_attempted += 1
