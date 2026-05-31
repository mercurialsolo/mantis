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
import json
import logging
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol

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


# ── Record (de)serialization (shared by the persistent backends) ────


def _record_to_dict(r: HintRecord) -> dict:
    """JSON-serializable form of a :class:`HintRecord`. Shared by the
    disk and modal.Dict backends so their on-the-wire shape can't drift."""
    return {
        "anchor_text": r.anchor_text,
        "anchor_xy_offset": list(r.anchor_xy_offset),
        "viewport_stage": r.viewport_stage,
        "confidence": r.confidence,
        "recorded_at": r.recorded_at,
        "source_url": r.source_url,
    }


def _record_from_dict(r: object) -> HintRecord | None:
    """Inverse of :func:`_record_to_dict`. Returns ``None`` for anything
    malformed so callers can drop it without aborting the whole load."""
    if not isinstance(r, dict):
        return None
    try:
        return HintRecord(
            anchor_text=str(r.get("anchor_text", "") or ""),
            anchor_xy_offset=tuple(r.get("anchor_xy_offset") or (0, 0))[:2],
            viewport_stage=int(r.get("viewport_stage", 0) or 0),
            confidence=float(r.get("confidence", 0.0) or 0.0),
            recorded_at=float(r.get("recorded_at", 0.0) or 0.0),
            source_url=str(r.get("source_url", "") or ""),
        )
    except Exception:  # noqa: BLE001 — drop malformed record
        return None


# ── DiskHintStore (Phase 1b) — tenant-scoped persistence ────────────


def _hint_store_root() -> str:
    """Where DiskHintStore persists. Env-overridable for tests."""
    return os.environ.get(
        "MANTIS_HINT_MEMORY_DIR",
        os.path.join(
            os.environ.get("MANTIS_DATA_DIR", "/data"), "hints",
        ),
    )


def _sanitize_scope(s: str) -> str:
    """Filesystem-safe id. Mirrors the convention used by
    ``server_utils.safe_state_key`` so tenant + plan paths align."""
    return "".join(c if c.isalnum() or c in "_-." else "_" for c in s)[:120] or "_"


class DiskHintStore:
    """JSON-on-volume `HintStore` with **per-tenant isolation**.

    Layout::

        /data/hints/<tenant_id>/<plan_signature>.json

    Tenant isolation is structural — different tenants can't see each
    other's anchors even when running the same plan. Same plan on the
    same tenant accumulates across deploys.

    The in-memory bucket semantics (LRU at ``max_per_key`` records) are
    preserved per (intent_hash, url_pattern) pair inside each plan file.

    Concurrency: atomic write via tmpfile + ``os.replace``. Concurrent
    workers race on the write — last-writer-wins. Acceptable for Phase 1b
    because:

    * The recording side appends one record per step success, and the
      record's value is the same shape across workers for the same
      (key) — so a lost write at most delays accumulation by one run.
    * The injection side reads at run start (single load); subsequent
      writes during the run don't affect that snapshot.

    Phase 2 may upgrade to SQLite if multi-region writes start producing
    visible drift. Phase 2 may also add a Modal Dict overlay so workers
    in the SAME run share writes immediately (today they only see each
    other's writes on the NEXT run).
    """

    def __init__(self, *, tenant_id: str, max_per_key: int = 10) -> None:
        if not tenant_id:
            raise ValueError("DiskHintStore requires a non-empty tenant_id")
        self._tenant_id = tenant_id
        self.max_per_key = max(1, int(max_per_key))

    # ── path helpers ──

    def _tenant_dir(self) -> str:
        return os.path.join(_hint_store_root(), _sanitize_scope(self._tenant_id))

    def _file_for(self, plan_signature: str) -> str:
        return os.path.join(
            self._tenant_dir(), f"{_sanitize_scope(plan_signature)}.json",
        )

    # ── I/O ──

    def _load(self, plan_signature: str) -> dict[str, list[HintRecord]]:
        path = self._file_for(plan_signature)
        try:
            with open(path) as f:
                raw = json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as exc:  # noqa: BLE001 — corrupt store → start fresh
            logger.warning("DiskHintStore: load failed %s (%s)", path, exc)
            return {}
        buckets: dict[str, list[HintRecord]] = {}
        for bucket_key, records in (raw or {}).items():
            if not isinstance(records, list):
                continue
            parsed = [
                rec for rec in (_record_from_dict(r) for r in records)
                if rec is not None
            ]
            if parsed:
                buckets[str(bucket_key)] = parsed
        return buckets

    def _save(self, plan_signature: str, buckets: dict[str, list[HintRecord]]) -> None:
        path = self._file_for(plan_signature)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        serializable: dict[str, list[dict]] = {
            bucket_key: [_record_to_dict(r) for r in records]
            for bucket_key, records in buckets.items()
        }
        tmp_path = f"{path}.tmp.{os.getpid()}.{int(time.time() * 1000)}"
        with open(tmp_path, "w") as f:
            json.dump(serializable, f, indent=2)
        os.replace(tmp_path, path)

    @staticmethod
    def _bucket_key(key: HintKey) -> str:
        """Compound key within a single plan file. Plan_signature lives
        in the filename; bucket key is intent_hash + url_pattern."""
        return f"{key.intent_hash}|{key.url_pattern}"

    # ── HintStore protocol ──

    def add(self, key: HintKey, record: HintRecord) -> None:
        if not key.plan_signature:
            return
        buckets = self._load(key.plan_signature)
        bk = self._bucket_key(key)
        bucket = buckets.setdefault(bk, [])
        bucket.append(record)
        if len(bucket) > self.max_per_key:
            bucket.pop(0)
        try:
            self._save(key.plan_signature, buckets)
        except Exception as exc:  # noqa: BLE001 — store failure must
            # never break a run. Silent on the hot path; the recording
            # side calls this best-effort.
            logger.warning(
                "DiskHintStore: save failed for tenant=%s plan=%s (%s)",
                self._tenant_id, key.plan_signature, exc,
            )

    def get(self, key: HintKey) -> list[HintRecord]:
        if not key.plan_signature:
            return []
        buckets = self._load(key.plan_signature)
        bucket = buckets.get(self._bucket_key(key), [])
        # Newest-first for the injection side's pick-top-N lookup.
        return list(reversed(bucket))

    def size(self) -> int:
        """Total record count across this tenant's plans."""
        td = self._tenant_dir()
        if not os.path.isdir(td):
            return 0
        total = 0
        for name in os.listdir(td):
            if not name.endswith(".json"):
                continue
            plan_sig = name[:-5]
            buckets = self._load(plan_sig)
            for bucket in buckets.values():
                total += len(bucket)
        return total

    # ── inspector surface ──

    def list_plan_signatures(self) -> list[str]:
        """Plan signatures with stored hints for this tenant."""
        td = self._tenant_dir()
        if not os.path.isdir(td):
            return []
        return sorted(f[:-5] for f in os.listdir(td) if f.endswith(".json"))

    def iter_records(self) -> Iterable[tuple[HintKey, HintRecord]]:
        for plan_sig in self.list_plan_signatures():
            buckets = self._load(plan_sig)
            for bk, records in buckets.items():
                intent_hash, _, url_pattern = bk.partition("|")
                key = HintKey(
                    plan_signature=plan_sig,
                    intent_hash=intent_hash,
                    url_pattern=url_pattern,
                )
                for r in records:
                    yield (key, r)


# ── ModalDictHintStore (Phase 1b) — cross-worker shared backend ─────


class ModalDictHintStore:
    """``modal.Dict``-backed `HintStore` shared across fan-out workers.

    Where :class:`DiskHintStore` persists across *deploys* on a volume,
    this backend is shared across the *workers of one deploy* through a
    ``modal.Dict``: a recording made by the Phase-1 fan-out worker is
    visible to the Phase-2 worker (and to the next run) immediately,
    without waiting for a volume write/read cycle. This is the backend
    the Learning Allocator's S0 retrieval rung reads/writes in prod so
    its effect actually reaches the remote run.

    Layout — one entry per ``(tenant_id, plan_signature)``::

        "<tenant>/<plan_signature>"  →  {"<intent>|<url_pattern>": [record-dict, …]}

    Tenant isolation is by key prefix, so one ``modal.Dict`` per deploy
    serves every tenant without cross-talk. Bucket / LRU semantics match
    :class:`DiskHintStore` (cap ``max_per_key``, newest-first ``get``).

    The backing object is dependency-injected (anything with
    ``get`` / ``__setitem__`` / ``keys``) so tests pass a plain ``dict``
    and prod passes a ``modal.Dict`` via :meth:`from_name`. Every access
    is best-effort: a store fault logs a warning and degrades to "no
    hints", never breaking a run. We never call ``len()`` on the backing
    object — ``modal.Dict`` raises ``TypeError`` on ``len()``
    (see ``feedback_modal_dict_no_len``); :meth:`size` iterates ``keys()``.
    """

    def __init__(
        self, backing: object, *, tenant_id: str, max_per_key: int = 10,
    ) -> None:
        if not tenant_id:
            raise ValueError("ModalDictHintStore requires a non-empty tenant_id")
        self._d = backing
        self._tenant_id = tenant_id
        self.max_per_key = max(1, int(max_per_key))

    @classmethod
    def from_name(
        cls, dict_name: str, *, tenant_id: str,
        max_per_key: int = 10, create_if_missing: bool = True,
    ) -> "ModalDictHintStore":
        """Build against a named ``modal.Dict``. ``modal`` is imported
        lazily so this module stays importable off-Modal (tests use the
        plain-``dict`` constructor instead)."""
        import modal  # noqa: PLC0415 — lazy so the module imports without modal

        backing = modal.Dict.from_name(dict_name, create_if_missing=create_if_missing)
        return cls(backing, tenant_id=tenant_id, max_per_key=max_per_key)

    # ── key + I/O ──

    def _entry_key(self, plan_signature: str) -> str:
        return (
            f"{_sanitize_scope(self._tenant_id)}/"
            f"{_sanitize_scope(plan_signature)}"
        )

    def _load(self, plan_signature: str) -> dict[str, list[HintRecord]]:
        try:
            raw = self._d.get(self._entry_key(plan_signature))  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 — store fault ≠ run fault
            logger.warning(
                "ModalDictHintStore: read failed tenant=%s plan=%s (%s)",
                self._tenant_id, plan_signature, exc,
            )
            return {}
        if not isinstance(raw, dict):
            return {}
        buckets: dict[str, list[HintRecord]] = {}
        for bucket_key, records in raw.items():
            if not isinstance(records, list):
                continue
            parsed = [
                rec for rec in (_record_from_dict(r) for r in records)
                if rec is not None
            ]
            if parsed:
                buckets[str(bucket_key)] = parsed
        return buckets

    def _save(self, plan_signature: str, buckets: dict[str, list[HintRecord]]) -> None:
        serializable = {
            bucket_key: [_record_to_dict(r) for r in records]
            for bucket_key, records in buckets.items()
        }
        try:
            self._d[self._entry_key(plan_signature)] = serializable  # type: ignore[index]
        except Exception as exc:  # noqa: BLE001 — store fault ≠ run fault
            logger.warning(
                "ModalDictHintStore: write failed tenant=%s plan=%s (%s)",
                self._tenant_id, plan_signature, exc,
            )

    @staticmethod
    def _bucket_key(key: HintKey) -> str:
        return f"{key.intent_hash}|{key.url_pattern}"

    # ── HintStore protocol ──

    def add(self, key: HintKey, record: HintRecord) -> None:
        if not key.plan_signature:
            return
        buckets = self._load(key.plan_signature)
        bucket = buckets.setdefault(self._bucket_key(key), [])
        bucket.append(record)
        if len(bucket) > self.max_per_key:
            bucket.pop(0)
        self._save(key.plan_signature, buckets)

    def get(self, key: HintKey) -> list[HintRecord]:
        if not key.plan_signature:
            return []
        buckets = self._load(key.plan_signature)
        # Newest-first for the injection side's pick-top-N lookup.
        return list(reversed(buckets.get(self._bucket_key(key), [])))

    def size(self) -> int:
        """Total record count for this tenant. Iterates ``keys()`` rather
        than ``len()`` — ``modal.Dict`` has no ``__len__``."""
        prefix = f"{_sanitize_scope(self._tenant_id)}/"
        total = 0
        for k in self._tenant_entry_keys(prefix):
            try:
                raw = self._d.get(k)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(raw, dict):
                continue
            total += sum(len(v) for v in raw.values() if isinstance(v, list))
        return total

    # ── inspector surface ──

    def _tenant_entry_keys(self, prefix: str) -> list[str]:
        try:
            return [str(k) for k in self._d.keys() if str(k).startswith(prefix)]  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            logger.warning("ModalDictHintStore: keys() failed (%s)", exc)
            return []

    def iter_records(self) -> Iterable[tuple[HintKey, HintRecord]]:
        prefix = f"{_sanitize_scope(self._tenant_id)}/"
        for k in self._tenant_entry_keys(prefix):
            plan_sig = k[len(prefix):]
            for bk, records in self._load(plan_sig).items():
                intent_hash, _, url_pattern = bk.partition("|")
                key = HintKey(
                    plan_signature=plan_sig,
                    intent_hash=intent_hash,
                    url_pattern=url_pattern,
                )
                for r in records:
                    yield (key, r)


# ── Recording-side adapter ──────────────────────────────────────────


# Step types that exercise grounding heavily — recording fires only for
# these. Other types (navigate, navigate_back, wait, done) don't ground
# from screenshot pixels, so there's nothing useful to record.
GROUNDING_STEP_TYPES: frozenset[str] = frozenset({
    "click", "claude_click", "detect_visible",
})


def extract_anchor_from_env(env: object) -> tuple[str, tuple[int, int]] | None:
    """Read the most recent SoM-click diagnostic off the env.

    XdotoolGymEnv stashes ``_last_som_diag`` after each ``cdp_click_at_point``
    call (see ``feedback_cua_cdp_post_action_verify.md`` for the
    provenance contract). The diagnostic carries ``elv_text`` (the
    visible text label Holo3 anchored to) and the click coordinates.

    Returns ``(anchor_text, (x, y))`` when a usable diagnostic exists,
    ``None`` when there's nothing to record. Callers treat ``None`` as
    "skip recording" — typical when the click went through xdotool only
    (no CDP verification path) or when the env doesn't expose
    ``_last_som_diag`` (test mocks).
    """
    diag = getattr(env, "_last_som_diag", None)
    if not isinstance(diag, dict):
        return None
    anchor = str(diag.get("elv_text") or "").strip()
    if not anchor:
        return None
    try:
        x = int(diag.get("x", 0))
        y = int(diag.get("y", 0))
    except (TypeError, ValueError):
        x, y = 0, 0
    return (anchor[:200], (x, y))


def record_hint_if_eligible(
    *,
    store: HintStore,
    plan_signature: str,
    step: object,
    step_type: str,
    success: bool,
    env: object,
    confidence: float = 1.0,
) -> HintRecord | None:
    """Producer-side gate + record.

    Fires only when:
      * The step type is in :data:`GROUNDING_STEP_TYPES`
      * The step succeeded
      * ``store`` isn't the :class:`NullHintStore`
      * The env exposed a usable SoM anchor

    Returns the recorded :class:`HintRecord` on success, ``None`` when
    any gate rejected. Never raises — best-effort throughout.
    """
    if isinstance(store, NullHintStore):
        return None
    if not success or step_type not in GROUNDING_STEP_TYPES:
        return None
    if not plan_signature:
        return None
    anchor_pair = extract_anchor_from_env(env)
    if anchor_pair is None:
        return None
    anchor_text, anchor_xy = anchor_pair

    url = ""
    try:
        url = str(getattr(env, "current_url", "") or "")
    except Exception:  # noqa: BLE001
        url = ""

    try:
        key = hint_key_for(plan_signature, step, url)
        record = HintRecord(
            anchor_text=anchor_text,
            anchor_xy_offset=anchor_xy,
            viewport_stage=int(getattr(env, "_viewport_stage", 0) or 0),
            confidence=float(confidence),
            source_url=url[:200],
        )
        store.add(key, record)
        logger.warning(
            "  [hint-memory] recorded anchor=%r step_type=%s plan=%s url_pattern=%s",
            anchor_text[:60], step_type, plan_signature[:8], key.url_pattern,
        )
        return record
    except Exception as exc:  # noqa: BLE001 — store failure ≠ run failure
        logger.debug("record_hint_if_eligible raised: %s", exc)
        return None


# ── Injection-side function ─────────────────────────────────────────


def apply_hint_overlay(
    plan: object, *,
    store: HintStore,
    plan_signature: str,
    start_url: str = "",
) -> int:
    """Pre-flight: stamp ``preferred_target_description`` on every
    grounding-bearing step that has a stored anchor.

    Returns the count of steps that received a hint. ``0`` when the
    store is empty or no eligible step matched.

    Called by the dispatcher BEFORE ``MicroPlanRunner.run()`` so the
    brain prompt sees the hint on the first attempt instead of
    re-discovering the anchor from pixels.

    Mutates ``plan.steps[i].hints`` in place — preserves existing hint
    keys, only sets ``preferred_target_description`` (and
    ``preferred_target_viewport_stage`` when known) when absent. We
    don't overwrite operator-authored hints; the store is a fallback,
    not an override.
    """
    if isinstance(store, NullHintStore) or not plan_signature:
        return 0

    applied = 0
    for step in getattr(plan, "steps", []) or []:
        step_type = str(getattr(step, "type", "") or "")
        if step_type not in GROUNDING_STEP_TYPES:
            continue
        key = hint_key_for(plan_signature, step, start_url)
        records = store.get(key)
        if not records:
            continue
        # Top record by confidence × recency (store returns newest-first).
        best = max(records, key=lambda r: (r.confidence, r.recorded_at))
        if best.confidence < 0.3:
            continue
        hints = dict(getattr(step, "hints", None) or {})
        # Don't overwrite operator-authored hints.
        if "preferred_target_description" not in hints:
            hints["preferred_target_description"] = best.anchor_text
        if "preferred_target_viewport_stage" not in hints and best.viewport_stage:
            hints["preferred_target_viewport_stage"] = best.viewport_stage
        step.hints = hints
        applied += 1
        logger.warning(
            "  [hint-overlay] step type=%s intent=%r ← anchor=%r (conf=%.2f)",
            step_type,
            str(getattr(step, "intent", "") or "")[:60],
            best.anchor_text[:60],
            best.confidence,
        )
    return applied


# ── Modal-path store factory ────────────────────────────────────────


def build_hint_store(
    task_suite: object, *,
    tenant_id: str,
    modal_dict_factory: Callable[[str], object] | None = None,
) -> HintStore:
    """Pick the `HintStore` backend for a Modal micro-run from suite metadata.

    The Modal dispatcher (``modal_cua_server``) calls this once before
    ``MicroPlanRunner.run`` to choose the backend the run overlays from
    and records into. Selection (first match wins):

    * ``_hint_store_disabled`` truthy → :class:`NullHintStore`. The run
      neither overlays nor records — the Learning Allocator's *frozen*
      policy (cold-start behaviour, no improvement applied).
    * ``_hint_store_dict_name`` non-empty → :class:`ModalDictHintStore`
      against that named ``modal.Dict``: a cross-worker shared store the
      Learning Allocator's *S0 retrieval* rung accumulates anchors in,
      so its effect reaches sibling workers and later runs immediately.
    * otherwise → :class:`DiskHintStore` keyed by ``tenant_id`` — today's
      production default (per-tenant JSON on the ``/data`` volume).

    ``modal_dict_factory`` is a test seam: when given, it's called with
    the dict name to build the backing object (tests pass one returning a
    plain dict), bypassing the lazy ``modal`` import inside
    :meth:`ModalDictHintStore.from_name`.

    Never raises. A ``modal.Dict`` open failure degrades to the
    production :class:`DiskHintStore`; an empty ``tenant_id`` (shouldn't
    happen on the Modal path, but DiskHintStore forbids it) degrades to
    :class:`NullHintStore`. A malformed suite can never break a run.
    """
    suite = task_suite if isinstance(task_suite, dict) else {}

    if bool(suite.get("_hint_store_disabled")):
        return NullHintStore()

    dict_name = str(suite.get("_hint_store_dict_name", "") or "").strip()
    if dict_name:
        try:
            if modal_dict_factory is not None:
                return ModalDictHintStore(
                    modal_dict_factory(dict_name), tenant_id=tenant_id,
                )
            return ModalDictHintStore.from_name(dict_name, tenant_id=tenant_id)
        except Exception as exc:  # noqa: BLE001 — never break a run
            logger.warning(
                "build_hint_store: ModalDictHintStore(%s) open failed (%s); "
                "falling back to DiskHintStore", dict_name, exc,
            )

    if not tenant_id:
        return NullHintStore()
    return DiskHintStore(tenant_id=tenant_id)
