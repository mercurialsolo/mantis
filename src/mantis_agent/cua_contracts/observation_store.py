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
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


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
