"""Ref-based observation store with redaction-before-persist (#485).

Screenshots and other observations exist today inside the runner's
``StepResult.screenshot_png`` field (inline bytes), in the trace
exporter's PNG-sibling output, and in modal debug dirs. The CUA
design requires:

* Observations are persisted **before** the action dispatches —
  so the post-mortem of a failed action has the same screenshot
  the brain saw, not whatever the page has drifted to by the
  time the runner serializes a StepResult.
* Trajectory events reference observations **by ref** — no
  inline base64 (the validator in
  :func:`~.validation._validate_observation` already rejects
  inline blobs).
* A **redaction hook** runs before persistence so sensitive
  content (PII, credentials surfaced by the brain on a screen)
  is sanitized at write time. Redacting on read is too late —
  the original blob is already on disk by then.

This module owns the protocol + a default in-memory implementation.
The runner integration (replacing the ``placeholder://`` ref the
emitter currently stamps with real refs from a configured store)
lands in a follow-up PR — same substrate-first pattern as #476,
#487 and the rest of the contract stack.

Public surface:

* :class:`Observation` (re-exported via :mod:`.types`).
* :class:`RedactionPolicy` — callable applied to image bytes
  before they hit the store. Default is a passthrough.
* :class:`ObservationStore` — Protocol with put / get / exists.
* :class:`InMemoryObservationStore` — default implementation for
  tests + local runs. Production wiring uses a disk- or
  object-store-backed variant.
* :class:`StoredObservation` — what the store returns from get():
  bytes + the ref string + the redaction provenance.

Refs are stable, content-addressed strings — the writer can rely
on "same image bytes → same ref" so duplicate writes are free.
The default scheme is ``sha256:<hex>`` but the protocol doesn't
pin a format; storage backends can use any string they want as
long as it round-trips.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Protocol, runtime_checkable


logger = logging.getLogger(__name__)


# Type alias for the redaction hook. Bytes-in / bytes-out so it
# composes with any image processing pipeline (PIL / pure-PNG /
# cv2). The default identity policy is a no-op — operators wire
# a real redaction policy via the store constructor.
RedactionPolicy = Callable[[bytes], bytes]


def identity_redaction(image_bytes: bytes) -> bytes:
    """No-op redaction policy — passes bytes through unchanged.

    The default for v1 + tests. Production wiring will replace it
    with a real PII / credential scrubber once one exists in the
    repo. Kept as a named function (not a lambda) so log output
    + introspection identify the policy in use.
    """
    return image_bytes


@dataclass(frozen=True)
class StoredObservation:
    """What :meth:`ObservationStore.get` returns — the bytes that
    were persisted plus the ref string the writer used and the
    redaction-policy provenance.

    ``redaction_policy_name`` is the name of the redaction callable
    that ran at write time (``"identity"`` for the default no-op).
    Lets a reader detect "this image was redacted with policy X"
    vs "this image hit the store before redaction was configured"
    without consulting the writer's deploy history.
    """

    ref: str
    image_bytes: bytes
    redaction_policy_name: str = "identity"
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ObservationStore(Protocol):
    """The narrow contract any observation backend must satisfy.

    Implementations:

    * persist image bytes before returning a ref;
    * apply the configured :data:`RedactionPolicy` BEFORE
      persistence — redact-on-read is too late;
    * return refs that are stable (content-addressed) so duplicate
      writes are idempotent.

    The runner stashes the store on
    ``runner.observation_store``; the emit hook reads it before
    building the canonical :class:`Observation` and passes the
    real ref instead of the ``placeholder://`` default.

    ``runtime_checkable`` so handlers can ``isinstance`` a wired
    store for assertion clarity in tests.
    """

    def put(
        self,
        image_bytes: bytes,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Persist ``image_bytes`` (after redaction) and return a
        stable ref. The ref MUST be a string reference (no inline
        base64 — the :func:`~.validation._validate_observation`
        check rejects those at emit time)."""
        ...

    def get(self, ref: str) -> StoredObservation | None:
        """Look up a previously-stored observation by ref. Returns
        ``None`` when the ref doesn't exist (no exception — the
        store's job is to be unambiguous about missing refs)."""
        ...

    def exists(self, ref: str) -> bool:
        """True iff the ref has been written. Cheap predicate so
        callers can branch without paying the get() cost."""
        ...


class InMemoryObservationStore:
    """Default :class:`ObservationStore` — keeps bytes in a process-
    local dict (#485).

    Useful for:

    * tests + local runs;
    * bring-up before a real persistent backend is wired;
    * shadow-routing comparisons within a single process.

    Refs are ``sha256:<hex>`` strings derived from the
    POST-redaction bytes, so duplicate writes collapse to one
    entry and the ref is reproducible across processes (same
    redaction policy + same input → same ref).

    Production wiring uses a disk- or object-store-backed
    implementation that satisfies the same protocol; this class
    stays the test default + the explicit bring-up surface.
    """

    def __init__(
        self,
        *,
        redaction_policy: RedactionPolicy = identity_redaction,
        redaction_policy_name: str = "",
    ) -> None:
        self._store: dict[str, bytes] = {}
        self._redaction_policy: RedactionPolicy = redaction_policy
        # Default to the policy callable's name when no explicit
        # tag is supplied — keeps the StoredObservation's
        # provenance grep-able without ceremony.
        self._policy_name: str = (
            redaction_policy_name or getattr(redaction_policy, "__name__", "redaction")
        )

    @property
    def redaction_policy_name(self) -> str:
        """Public for tests + introspection."""
        return self._policy_name

    def put(
        self,
        image_bytes: bytes,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if not isinstance(image_bytes, (bytes, bytearray)):
            raise TypeError(
                f"ObservationStore.put requires bytes; got {type(image_bytes).__name__}"
            )
        if not image_bytes:
            raise ValueError("ObservationStore.put refuses empty bytes")
        # Redact BEFORE hashing + storing so the ref is keyed on
        # the post-redaction content. Two callers with the same
        # raw image will produce the same ref iff the redaction
        # policy is deterministic.
        redacted = self._redaction_policy(bytes(image_bytes))
        if not isinstance(redacted, (bytes, bytearray)) or not redacted:
            raise ValueError(
                "redaction policy returned empty / non-bytes — refusing to store"
            )
        ref = f"sha256:{hashlib.sha256(redacted).hexdigest()}"
        self._store[ref] = bytes(redacted)
        return ref

    def get(self, ref: str) -> StoredObservation | None:
        blob = self._store.get(ref)
        if blob is None:
            return None
        return StoredObservation(
            ref=ref,
            image_bytes=blob,
            redaction_policy_name=self._policy_name,
        )

    def exists(self, ref: str) -> bool:
        return ref in self._store

    def __len__(self) -> int:
        """How many distinct refs the store holds — useful for
        tests verifying dedup behaviour."""
        return len(self._store)


class DiskObservationStore:
    """Disk-backed :class:`ObservationStore` — same protocol as
    :class:`InMemoryObservationStore` but persists bytes to disk
    so refs survive process restart and an operator can fetch
    any screenshot referenced by trajectory.jsonl after the run
    has terminated.

    Storage layout — ``<root>/<sha[:2]>/<sha>.bin``:

    * Two-level fanout (first two hex chars as a sub-directory)
      keeps any one directory under ~256 entries even for runs
      with thousands of screenshots — a flat directory of 5k
      files makes ``ls`` painful on the operator path.
    * ``.bin`` suffix instead of ``.png`` so the store is honest
      about being content-typed only by what the redaction
      policy produces — a future redactor that recompresses or
      crops may not emit valid PNG bytes.

    Same content-addressed dedup semantics as in-memory: refs are
    ``sha256:<hex>`` derived from POST-redaction bytes. Two
    callers persisting identical inputs share a ref AND share
    the on-disk file (no duplicate write — :meth:`put` returns
    early when the target file already exists).

    **Bounded by ``max_bytes``** with LRU eviction by mtime
    (default 1 GiB, configurable via ctor arg or
    ``MANTIS_OBS_STORE_MAX_BYTES`` env). Without a cap, a long-
    running Modal volume would grow unbounded across runs —
    1280×720 PNGs at ~100 KB × 30 steps × 100s of runs/day
    saturates a typical mount in days. The LRU policy uses file
    mtime; since the dedup path doesn't touch mtime, "oldest" is
    "least recently first-written".

    Suitable as the default store for Modal / Baseten runs where
    the canonical events directory is already on a persistent
    volume. The in-memory variant stays the test default + the
    explicit "I don't care about persistence" surface.
    """

    _SUFFIX: str = ".bin"
    _MAX_BYTES_ENV: ClassVar[str] = "MANTIS_OBS_STORE_MAX_BYTES"
    _DEFAULT_MAX_BYTES: ClassVar[int] = 1 << 30  # 1 GiB

    def __init__(
        self,
        root_dir: str,
        *,
        max_bytes: int | None = None,
        redaction_policy: RedactionPolicy = identity_redaction,
        redaction_policy_name: str = "",
    ) -> None:
        if not root_dir:
            raise ValueError("DiskObservationStore requires a non-empty root_dir")
        import os
        self._root: str = os.fspath(root_dir)
        self._redaction_policy: RedactionPolicy = redaction_policy
        self._policy_name: str = (
            redaction_policy_name or getattr(redaction_policy, "__name__", "redaction")
        )
        # Cap resolution order: explicit ctor arg → env var →
        # 1 GiB default. ``max_bytes <= 0`` disables the cap
        # (only respected when set explicitly — accidental zero
        # in the env doesn't disable; falls back to default).
        if max_bytes is not None:
            self._max_bytes: int = int(max_bytes)
        else:
            env_value = os.environ.get(self._MAX_BYTES_ENV, "").strip()
            if env_value:
                try:
                    parsed = int(env_value)
                    self._max_bytes = parsed if parsed > 0 else self._DEFAULT_MAX_BYTES
                except ValueError:
                    self._max_bytes = self._DEFAULT_MAX_BYTES
            else:
                self._max_bytes = self._DEFAULT_MAX_BYTES
        # Create the root eagerly so a misconfigured path (read-
        # only mount, wrong tenant scope) fails at construction
        # time rather than first put().
        os.makedirs(self._root, exist_ok=True)
        # Scan existing content so a resumed run with prior
        # observations on disk counts toward the cap. O(N) on
        # init — a 10k-file store walks in milliseconds.
        self._current_bytes: int = self._scan_total_size()

    @property
    def root_dir(self) -> str:
        return self._root

    @property
    def redaction_policy_name(self) -> str:
        return self._policy_name

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def current_bytes(self) -> int:
        """Best-effort current footprint. Reflects the in-memory
        tally maintained across put / evict; rescan after external
        deletion via :meth:`rescan_size`."""
        return self._current_bytes

    def rescan_size(self) -> int:
        """Re-walk the store to reconcile ``current_bytes`` with the
        on-disk reality. Useful if an external process pruned files
        out from under us — without this the cap math gets stale."""
        self._current_bytes = self._scan_total_size()
        return self._current_bytes

    def _scan_total_size(self) -> int:
        """Walk the store, sum file sizes. O(N) — only called at
        init and on explicit rescan; the per-put path uses the
        in-memory tally instead."""
        import os
        total = 0
        for dirpath, _, filenames in os.walk(self._root):
            for fn in filenames:
                if not fn.endswith(self._SUFFIX):
                    continue
                try:
                    total += os.path.getsize(os.path.join(dirpath, fn))
                except OSError:
                    # File vanished between scandir + getsize —
                    # ignore. The next put / rescan converges.
                    pass
        return total

    def _evict_until_under(self, target_bytes: int) -> None:
        """Delete oldest files (by mtime) until ``current_bytes <=
        target_bytes``. No-op when already under target.

        LRU-ish — relies on filesystem mtime, which is preserved
        across dedup writes (the existing dedup path skips the
        write entirely when the file exists, so mtime stays
        first-write time). A more sophisticated policy could read
        atime, but most production mounts use noatime; first-write
        time is a reasonable proxy.

        Best-effort: ``OSError`` on individual deletes (concurrent
        readers, permission flips) gets logged + skipped — we keep
        evicting other candidates rather than abort.
        """
        if self._current_bytes <= target_bytes:
            return
        import os
        # Collect (mtime, path, size) for every file in the store.
        # 10k entries → ~5 ms on a typical SSD; an order of
        # magnitude before this becomes worth caching.
        candidates: list[tuple[float, str, int]] = []
        for dirpath, _, filenames in os.walk(self._root):
            for fn in filenames:
                if not fn.endswith(self._SUFFIX):
                    continue
                path = os.path.join(dirpath, fn)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                candidates.append((st.st_mtime, path, st.st_size))
        # Sort ascending by mtime → oldest first.
        candidates.sort(key=lambda t: t[0])
        for _mtime, path, size in candidates:
            if self._current_bytes <= target_bytes:
                break
            try:
                os.remove(path)
            except OSError as exc:
                logger.debug(
                    "DiskObservationStore: evict failed for %s: %s", path, exc,
                )
                continue
            self._current_bytes -= size
            if self._current_bytes < 0:
                self._current_bytes = 0

    def _path_for(self, ref: str) -> str:
        # ref is "sha256:<hex>"; strip the scheme prefix for the
        # filename so it round-trips through any URL-encoding
        # consumer that might choke on the colon.
        import os
        hex_part = ref.split(":", 1)[-1]
        if len(hex_part) < 2:
            # Defensive — a malformed ref produces a sentinel
            # filename rather than an exception. ``get`` /
            # ``exists`` will then deterministically miss.
            return os.path.join(self._root, "__malformed__", hex_part + self._SUFFIX)
        return os.path.join(self._root, hex_part[:2], hex_part + self._SUFFIX)

    def put(
        self,
        image_bytes: bytes,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if not isinstance(image_bytes, (bytes, bytearray)):
            raise TypeError(
                f"ObservationStore.put requires bytes; got {type(image_bytes).__name__}"
            )
        if not image_bytes:
            raise ValueError("ObservationStore.put refuses empty bytes")
        # Redact BEFORE hashing + storing — same semantic as
        # InMemoryObservationStore: refs key on cleaned content.
        redacted = self._redaction_policy(bytes(image_bytes))
        if not isinstance(redacted, (bytes, bytearray)) or not redacted:
            raise ValueError(
                "redaction policy returned empty / non-bytes — refusing to store"
            )
        ref = f"sha256:{hashlib.sha256(redacted).hexdigest()}"
        import os
        path = self._path_for(ref)
        # Idempotent dedup — skip the write when the file already
        # exists (filesystem is the truth; no need to track refs
        # in memory).
        if not os.path.exists(path):
            size = len(redacted)
            # Bounded store: evict oldest files until the new
            # write fits under the cap. The eviction is best-
            # effort — if we still can't fit (cap smaller than a
            # single observation, all files newer than this one,
            # external concurrent writes), we proceed with the
            # write anyway and let the next put's eviction catch
            # up. Logging the over-cap state at WARNING so
            # operators see it.
            if size <= self._max_bytes:
                if self._current_bytes + size > self._max_bytes:
                    self._evict_until_under(self._max_bytes - size)
                if self._current_bytes + size > self._max_bytes:
                    logger.warning(
                        "DiskObservationStore: eviction insufficient "
                        "(current=%d, would_add=%d, cap=%d) — writing anyway",
                        self._current_bytes, size, self._max_bytes,
                    )
            else:
                # Single observation bigger than the whole cap —
                # writing it overflows by definition. Log and
                # write; the next put will evict aggressively.
                logger.warning(
                    "DiskObservationStore: observation size %d exceeds "
                    "cap %d — writing without eviction",
                    size, self._max_bytes,
                )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Write to a temp file + atomic rename so a crash
            # mid-write can't leave a half-written file at the
            # canonical path (the next call computes the same
            # hash, sees no file, and retries cleanly).
            tmp_path = f"{path}.tmp.{os.getpid()}"
            with open(tmp_path, "wb") as f:
                f.write(bytes(redacted))
            os.replace(tmp_path, path)
            self._current_bytes += size
        return ref

    def get(self, ref: str) -> StoredObservation | None:
        import os
        path = self._path_for(ref)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            blob = f.read()
        return StoredObservation(
            ref=ref,
            image_bytes=blob,
            redaction_policy_name=self._policy_name,
        )

    def exists(self, ref: str) -> bool:
        import os
        return os.path.exists(self._path_for(ref))
