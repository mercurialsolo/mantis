"""Per-step state snapshot + diff helpers for plan-aware reverse (#121, step 1).

The current ``MicroPlanRunner._reverse_step`` blindly fires keystrokes
from a static ``REVERSE_ACTIONS`` map regardless of what the failed step
*actually* accomplished. For form steps that partially succeeded (typed
3 of 5 fields), Escape + Alt+Left destroys the work and the next attempt
re-types everything.

This module lands the **observability** half of the fix:

* :class:`StepStateSnapshot` captures the pre-step state in a small,
  hashable shape.
* :func:`capture` reads the live runner state into a snapshot.
* :func:`diff` computes the observed delta between two snapshots.

A follow-on PR uses the diff to decide reverse actions; this PR just
makes the signal visible in logs so we can validate the diff is correct
on real traces before changing recovery behavior.

State fields tracked:

* ``url``                 — last known URL (from ``runner._last_known_url``)
* ``current_page``        — pagination cursor
* ``viewport_stage``      — scroll viewport stage (0=top, 1=Page_Down, ...)
* ``focused_input``       — placeholder/name/value of the currently focused field
* ``scroll_signature``    — hash of the runner's scroll_state dict
* ``last_extracted_url``  — URL from the most recent extraction record

The snapshot is intentionally a *thin* view of runner state. We don't
copy the full ``_scroll_state`` / ``_last_extracted`` dicts — that would
inflate trajectory storage and complicate equality checks. The signature
hashes are sufficient for diff detection.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .micro_runner import MicroPlanRunner

logger = logging.getLogger(__name__)


def _hash_dict(d: dict[str, Any] | None) -> str:
    """Stable hex digest of a dict — used for scroll_state / focused_input
    equality. Empty/None inputs produce a constant sentinel so two no-state
    snapshots compare equal."""
    if not d:
        return "empty"
    try:
        payload = json.dumps(d, sort_keys=True, default=str)
    except (TypeError, ValueError):
        # Defensive: anything non-JSON-able falls through to a string repr.
        payload = repr(d)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class StepStateSnapshot:
    """Frozen view of runner state at a single point in time.

    Frozen so accidental mutation can't break diff comparisons. All
    fields default to neutral values so partial snapshots (e.g. taken
    before the runner has fully initialized) still compare cleanly.
    """

    url: str = ""
    current_page: int = 1
    viewport_stage: int = 0
    focused_input_signature: str = "empty"
    scroll_signature: str = "empty"
    last_extracted_url: str = ""
    extracted_titles_count: int = 0
    seen_urls_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serializable form for logs / trajectory persistence."""
        return {
            "url": self.url,
            "current_page": self.current_page,
            "viewport_stage": self.viewport_stage,
            "focused_input_sig": self.focused_input_signature,
            "scroll_sig": self.scroll_signature,
            "last_extracted_url": self.last_extracted_url,
            "extracted_titles_count": self.extracted_titles_count,
            "seen_urls_count": self.seen_urls_count,
        }


@dataclass
class StepDiff:
    """Observed delta between two snapshots — input to reverse decisions.

    Each boolean is set when the corresponding aspect of state changed.
    The :attr:`changed_fields` list gives a human-readable summary for
    logs.
    """

    url_changed: bool = False
    page_changed: bool = False
    viewport_changed: bool = False
    focus_changed: bool = False
    scroll_changed: bool = False
    extraction_added: bool = False
    new_urls_seen: bool = False
    changed_fields: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.changed_fields)

    def summary(self) -> str:
        return ", ".join(self.changed_fields) if self.changed_fields else "no change"


# ── Capture ──────────────────────────────────────────────────────────────


def capture(runner: "MicroPlanRunner") -> StepStateSnapshot:
    """Read runner state into a frozen snapshot.

    Runs synchronously and is cheap (a few attribute reads + 2 SHA1s),
    so it's safe to call before every step.
    """
    focused_input = None
    # Some env adapters expose a current focused-input record; others don't.
    try:
        focused_input = runner.env.last_focused_input  # type: ignore[attr-defined]
    except AttributeError:
        focused_input = None

    last_extracted = getattr(runner, "_last_extracted", {}) or {}
    return StepStateSnapshot(
        url=getattr(runner, "_last_known_url", "") or "",
        current_page=int(getattr(runner, "_current_page", 1) or 1),
        viewport_stage=int(getattr(runner, "_viewport_stage", 0) or 0),
        focused_input_signature=_hash_dict(focused_input),
        scroll_signature=_hash_dict(getattr(runner, "_scroll_state", {})),
        last_extracted_url=str(last_extracted.get("last_completed_url", "") or ""),
        extracted_titles_count=len(getattr(runner, "_extracted_titles", []) or []),
        seen_urls_count=len(getattr(runner, "_seen_urls", set()) or set()),
    )


# ── Diff ─────────────────────────────────────────────────────────────────


def diff(before: StepStateSnapshot, after: StepStateSnapshot) -> StepDiff:
    """Compute the observed delta between two snapshots.

    Pure function — order matters (before, after) but neither argument
    is mutated. Returns a :class:`StepDiff` with one bool per dimension
    that changed plus a human-readable summary list for logs.
    """
    out = StepDiff()
    if before.url != after.url:
        out.url_changed = True
        out.changed_fields.append(f"url: {before.url[:40]} → {after.url[:40]}")
    if before.current_page != after.current_page:
        out.page_changed = True
        out.changed_fields.append(
            f"page: {before.current_page} → {after.current_page}"
        )
    if before.viewport_stage != after.viewport_stage:
        out.viewport_changed = True
        out.changed_fields.append(
            f"viewport_stage: {before.viewport_stage} → {after.viewport_stage}"
        )
    if before.focused_input_signature != after.focused_input_signature:
        out.focus_changed = True
        out.changed_fields.append("focused_input")
    if before.scroll_signature != after.scroll_signature:
        out.scroll_changed = True
        out.changed_fields.append("scroll_state")
    if (
        after.last_extracted_url
        and after.last_extracted_url != before.last_extracted_url
    ):
        out.extraction_added = True
        out.changed_fields.append(
            f"last_extracted: {after.last_extracted_url[:40]}"
        )
    if after.seen_urls_count > before.seen_urls_count:
        out.new_urls_seen = True
        out.changed_fields.append(
            f"seen_urls +{after.seen_urls_count - before.seen_urls_count}"
        )
    return out


__all__ = ["StepStateSnapshot", "StepDiff", "capture", "diff"]
