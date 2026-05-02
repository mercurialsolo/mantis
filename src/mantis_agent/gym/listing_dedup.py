"""Lead/listing deduplication helpers — extracted from micro_runner.py
(#115, step 5).

Owns the pure parsing helpers (lead-key extraction, phone validation,
counts) that classify and deduplicate VIABLE extraction rows. These are
``@staticmethod`` / ``@classmethod`` on :class:`ListingDedup` so callers
that don't want a runner instance (tests, host adapters) can use them
directly:

    >>> from mantis_agent.gym.listing_dedup import ListingDedup
    >>> ListingDedup.lead_key("VIABLE | Year: 2024 | URL: https://x.test/a")
    'https://x.test/a'

The runner's per-page memory (``_seen_urls``, ``_extracted_titles``,
``_page_listings``, ``_page_listing_index``, ``_last_extracted``) stays
on the runner for now — there are 53 scattered access sites and
migrating them all in one PR would be unreviewable. Method bodies that
operate on those attributes can be added here in subsequent steps.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .checkpoint import StepResult


_VIABLE_PREFIX = "VIABLE"
_PHONE_NULL_VALUES: frozenset[str] = frozenset(
    {"", "none", "n/a", "na", "unknown", "not visible", "not shown"}
)


class ListingDedup:
    """Pure helpers for classifying / deduplicating VIABLE lead rows.

    The static methods are the canonical implementation; the runner's
    same-named ``_static_helpers`` stay as 1-line shims for backward compat.
    """

    # ── Filtering / extraction ──────────────────────────────────────────

    @staticmethod
    def successful_lead_data(results: list["StepResult"]) -> list[str]:
        """Return the ``data`` field of every ``success`` step that yielded
        a ``VIABLE`` extraction row."""
        return [
            r.data for r in results
            if r.success and (r.data or "").startswith(_VIABLE_PREFIX)
        ]

    @staticmethod
    def lead_key(data: str) -> str:
        """Stable dedup key. Prefers the URL inside the row; falls back to
        the first 100 chars when no URL is present (defensive — VIABLE rows
        always carry a URL in practice)."""
        url_match = re.search(r"URL:\s*([^|]+)", data)
        if url_match:
            return url_match.group(1).strip()
        return data[:100]

    @staticmethod
    def lead_has_phone(data: str) -> bool:
        """Heuristic: a VIABLE row counts as having a phone if it carries a
        ``Phone:`` field whose value is neither a null sentinel nor short
        enough to be a price misread (≥10 digits)."""
        phone_match = re.search(r"Phone:\s*([^|]+)", data, flags=re.IGNORECASE)
        if not phone_match:
            return False
        phone = phone_match.group(1).strip().lower()
        if phone in _PHONE_NULL_VALUES:
            return False
        return len(re.sub(r"\D", "", phone)) >= 10

    # ── Aggregations ────────────────────────────────────────────────────

    @classmethod
    def unique_leads_from_results(cls, results: list["StepResult"]) -> list[str]:
        """Deduplicated ``data`` strings for VIABLE rows, keyed by URL."""
        unique: dict[str, str] = {}
        for lead in cls.successful_lead_data(results):
            unique[cls.lead_key(lead)] = lead
        return list(unique.values())

    @classmethod
    def lead_counts(cls, results: list["StepResult"]) -> tuple[int, int]:
        """Return ``(unique_leads, phone_leads)`` over ``results``."""
        leads_by_key: dict[str, str] = {}
        for data in cls.successful_lead_data(results):
            leads_by_key[cls.lead_key(data)] = data
        total = len(leads_by_key)
        with_phone = sum(
            1 for data in leads_by_key.values() if cls.lead_has_phone(data)
        )
        return total, with_phone


__all__ = ["ListingDedup"]
