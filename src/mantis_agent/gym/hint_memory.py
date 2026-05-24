"""Trajectory hint memory (#643) — Phase 1: store + types.

Records successful grounding anchors per
``(plan_signature, intent_hash, url_pattern)`` so subsequent runs of
the same plan can replay them as ``hints.preferred_target_description``
on grounding-heavy steps (``click(Show More)``, ad-hoc reveal toggles,
custom DOM widgets).

This Phase ships the foundation only: store protocol, in-memory + null
backends, key-derivation helpers, and the :class:`HintRecord`
schema. The recording hook (read anchor metadata from Holo3, write to
the store on step success) and the injection hook (read hints from the
store, stamp ``preferred_target_description`` on the step) land in
Phase 1b.

Why a foundation-first split: the anchor metadata that flows out of
Holo3 today varies by step type (click → `elv_text` + screen xy;
detect_visible → reason text; extract_data → schema-field anchors).
Phase 1b will introduce a small adapter per step type so the
``HintStore`` can stay schema-stable across the lifetime of #643's
follow-ups (URL-pattern key inference, confidence decay, persistent
disk backend).

CUA purity (``feedback_cua_no_dom_access.md``): anchors recorded here
are always visual — screen coordinates relative to a text anchor or
``elv_text`` strings Holo3 read from the screenshot. No CSS selectors,
no DOM probes. The store records what the brain SAW, not what the
DOM CONTAINS.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, Protocol

logger = logging.getLogger(__name__)


# ── Schema ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HintRecord:
    """One observed anchor — what Holo3 used to ground a step, plus
    enough metadata for downstream injection / decay.

    Fields
    ------
    anchor_text
        Visual text label the brain anchored on (e.g. ``"Show More"``,
        ``"Description"``, the ``elv_text`` field Holo3 returned).
        Always derived from the screenshot, never from the DOM.
    anchor_xy_offset
        ``(dx, dy)`` pixel offset from the nearest text anchor in the
        viewport at click / scroll time. Optional — empty tuple when
        the recording side doesn't know the offset (e.g. when the
        anchor IS the click target). Used by the injection side to
        bias Holo3's grounding window.
    viewport_stage
        Scroll stage (0 = above-the-fold, 1 = first scroll, …) the
        anchor was found at. Helps the injection side warm Holo3 to
        scroll before grounding rather than searching the wrong slice.
    confidence
        How confident the recording call was that this anchor worked.
        Range ``[0.0, 1.0]``. The recording side picks the value
        (typically derived from the grounding pass's own confidence
        signal or 1.0 on plain success); the injection side uses it
        to skip low-quality records.
    recorded_at
        Unix timestamp. Used by ``InMemoryHintStore``'s LRU eviction
        (oldest record at cap drops out) and by future confidence-
        decay layers (#643 Phase 2+).
    source_url
        URL the anchor was recorded on. Diagnostic — surfaced in the
        store's iteration interface for operator queries; not used
        for matching (that's the ``url_pattern`` key).
    """

    anchor_text: str
    anchor_xy_offset: tuple[int, int] = (0, 0)
    viewport_stage: int = 0
    confidence: float = 1.0
    recorded_at: float = field(default_factory=time.time)
    source_url: str = ""


@dataclass(frozen=True)
class HintKey:
    """Compound key for the hint store.

    Three axes establish "this anchor will probably work HERE":

    * ``plan_signature`` — the 12-char hash of the source plan
      (already populated everywhere). Same plan ⇒ same flow shape.
    * ``intent_hash`` — short hash of the step's intent text + step
      type. Same intent on the same plan ⇒ same target. Decoupled
      from ``step_index`` because plan refactors (insert / reorder
      / split) shift indices but rarely rewrite intent prose.
    * ``url_pattern`` — regex / substring that the recorded source URL
      matched. Phase 1 uses the registrable domain + first path
      segment (e.g. ``boattrader.com/boat``); Phase 2 / future work
      can specialize per recipe.
    """

    plan_signature: str
    intent_hash: str
    url_pattern: str

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.plan_signature, self.intent_hash, self.url_pattern)


# ── Store protocol + backends ──────────────────────────────────────


class HintStore(Protocol):
    """Per-deploy backend for recorded hints. Implementations:

    * :class:`NullHintStore` — every method no-ops; default when the
      runner isn't wired to a backend. Lets the recording hook fire
      unconditionally without a feature flag.
    * :class:`InMemoryHintStore` — single-process backend used by
      tests and the local fan-out runner; bounded LRU per key.
    * ``ModalDictHintStore`` (Phase 1b) — modal.Dict backend that
      survives across fan-out worker boundaries within one deploy.
    * Disk store (Phase 2) — persists across deploys on the
      ``/data/hints/`` volume.
    """

    def add(self, key: HintKey, record: HintRecord) -> None: ...
    def get(self, key: HintKey) -> list[HintRecord]: ...
    def size(self) -> int: ...


class NullHintStore:
    """No-op backend. ``add`` swallows, ``get`` returns an empty
    list, ``size`` is always 0. Lets the recording hook fire on every
    eligible step without a per-step ``if store is None`` guard.
    """

    def add(self, key: HintKey, record: HintRecord) -> None:  # noqa: ARG002
        return None

    def get(self, key: HintKey) -> list[HintRecord]:  # noqa: ARG002
        return []

    def size(self) -> int:
        return 0


class InMemoryHintStore:
    """Single-process backend with LRU eviction.

    Keeps up to :attr:`max_per_key` records per key (default 10) —
    the recording side appends; once full the oldest record drops.
    Different keys accumulate independently (no global cap in
    Phase 1; a global ceiling can land in Phase 2 once we observe
    real memory pressure).

    Records are kept in insertion order; ``get`` returns the list in
    newest-first order to make injection trivial (pick the top N).
    """

    def __init__(self, *, max_per_key: int = 10) -> None:
        # OrderedDict[HintKey.as_tuple(), list[HintRecord]] —
        # the inner list is mutable; eviction drops the oldest entry
        # at index 0 when the list reaches ``max_per_key + 1``.
        self._buckets: OrderedDict[
            tuple[str, str, str], list[HintRecord]
        ] = OrderedDict()
        self.max_per_key = max(1, int(max_per_key))

    def add(self, key: HintKey, record: HintRecord) -> None:
        bucket = self._buckets.setdefault(key.as_tuple(), [])
        bucket.append(record)
        if len(bucket) > self.max_per_key:
            # LRU eviction — drop the oldest (front of list).
            bucket.pop(0)

    def get(self, key: HintKey) -> list[HintRecord]:
        bucket = self._buckets.get(key.as_tuple(), [])
        # Newest-first for the injection side's "pick top N by
        # confidence × recency" lookup.
        return list(reversed(bucket))

    def size(self) -> int:
        return sum(len(b) for b in self._buckets.values())

    # ── Operator surface ──

    def iter_records(self) -> Iterable[tuple[HintKey, HintRecord]]:
        """Yield every stored (key, record) pair — diagnostic only.
        Order isn't guaranteed across implementations; tests should
        sort or collect into sets before asserting."""
        for raw_key, bucket in self._buckets.items():
            plan_sig, intent_hash, url_pattern = raw_key
            key = HintKey(
                plan_signature=plan_sig,
                intent_hash=intent_hash,
                url_pattern=url_pattern,
            )
            for rec in bucket:
                yield (key, rec)


# ── Key derivation helpers ──────────────────────────────────────────


def intent_hash_for(step: object) -> str:
    """Stable 12-hex-char hash of ``step.intent`` + ``step.type``.

    Decoupled from ``step_index`` so plan refactors (insert / reorder
    / split) that shift indices but preserve intent prose keep the
    same key. Empty intent / type yield a non-empty hash too (the
    hashing pipeline never returns "") — defensive against callers
    that hand us partially-filled step dicts.
    """
    intent = str(getattr(step, "intent", "") or "")
    step_type = str(getattr(step, "type", "") or "")
    raw = f"{step_type}|{intent}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


_URL_HOST_SEG_RE = re.compile(
    r"^https?://(?:www\.)?([^/]+)(?:/([^/?#]+))?",
    re.IGNORECASE,
)


def url_pattern_for(url: str) -> str:
    """Phase 1 URL pattern: ``<registrable-host>/<first-path-segment>``.

    Examples:
        ``https://www.boattrader.com/boat/1986-marine-trader-europa-10/``
            → ``boattrader.com/boat``
        ``https://lu.ma/event/abc-123``
            → ``lu.ma/event``
        ``https://example.com/`` → ``example.com``
        ``""`` → ``""``

    Phase 2 / future work (#643 follow-ups): per-recipe URL pattern
    inference. The marketplace_listings recipe can specialize to
    include the make/model token so "Show More click anchor for
    Cruisers Yachts" is keyed separately from generic boat detail
    pages. Phase 1 keeps the pattern broad (one bucket per host +
    section) since the in-memory store's per-key LRU cap of 10
    handles the spread.
    """
    if not url:
        return ""
    m = _URL_HOST_SEG_RE.match(url.strip())
    if not m:
        return ""
    host = (m.group(1) or "").lower()
    first_seg = (m.group(2) or "").lower()
    if not host:
        return ""
    if not first_seg:
        return host
    return f"{host}/{first_seg}"


def hint_key_for(
    plan_signature: str, step: object, url: str,
) -> HintKey:
    """One-shot constructor — derives the per-axis keys and bundles
    them into a :class:`HintKey`. Phase 1b's recording / injection
    hooks call this once at their entry points."""
    return HintKey(
        plan_signature=str(plan_signature or ""),
        intent_hash=intent_hash_for(step),
        url_pattern=url_pattern_for(url),
    )
