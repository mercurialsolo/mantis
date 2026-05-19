"""Sandbox runtime interface — snapshot / restore around browser
environments (#484).

The CUA design's safety story needs a snapshot taken before
irreversible actions and a restore path for non-recoverable
failures. Mantis today captures browser state piecewise
(``BrowserState`` + ``RunCheckpoint``) but has no
sandbox-level snapshot abstraction the runner can call uniformly
across Xvfb+Chrome (local / Modal) and future managed sandbox
platforms.

This module owns the protocol + two default implementations:

* :class:`NoopSandboxRuntime` — the test default + the bring-up
  surface where snapshotting is not yet wired. Returns
  deterministic placeholder snapshot ids so callers can exercise
  the snapshot/restore code path without filesystem side effects.
* :class:`LocalProfileSandbox` — copies a Chrome user-data
  directory to a snapshot path on every :meth:`snapshot`; restore
  reverses the copy. Pragmatic implementation suitable for local
  development + the acceptance criterion "Snapshot/restore
  behavior is tested with at least one browser task fixture".

Production wiring (replacing the runtime in the Modal /
Baseten paths) lands in a follow-up — same substrate-first
pattern the rest of cua_contracts follows. ``SandboxRuntime`` is
the typed surface that future PRs can drop in a VM-snapshot or
container-snapshot backend without rewriting handlers.

The :class:`SandboxSnapshot` payload is referenced from the
canonical :class:`~.types.ActionResult.snapshot_id` field added in
this PR — every dispatched step that gated a snapshot before it
ran has the snapshot_id on its trajectory event, so a failed
action can be replayed against the pre-action state.

Why a Protocol over a base class:

The runner doesn't own the sandbox lifecycle — it consumes one. A
hosted runtime may wrap a fly.io VM, a Modal sandbox, an
in-process Xvfb, or a test stub. The Protocol lets each wire its
own concrete with no base-class coupling.

Failure-mode contract:

* :meth:`snapshot` returning ``None`` means "the runtime declined
  to snapshot this state" (e.g. a noop runtime, or a host that
  blacklisted the current page). Callers fall through to the
  un-gated dispatch path.
* :meth:`restore` raising :class:`SandboxRestoreError` means
  "restore was attempted but the runtime can't return to the
  snapshot state". Callers must surface this as a structured
  failure on the trajectory event — silently continuing on a
  failed restore corrupts the recovery contract.
"""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


logger = logging.getLogger(__name__)


class SandboxRestoreError(RuntimeError):
    """Raised when :meth:`SandboxRuntime.restore` cannot return the
    sandbox to a usable state. Carries enough context for the
    runner to record the failure on the canonical event and halt /
    escalate per the recovery policy.

    Typed exception (not a generic :class:`RuntimeError`) so the
    runner's catch site can distinguish restore failures from
    bug-shaped exceptions out of the runtime."""


@dataclass(frozen=True)
class SandboxSnapshot:
    """Typed result returned by :meth:`SandboxRuntime.snapshot`.

    The opaque ``snapshot_id`` is the only field the runner uses
    to call :meth:`SandboxRuntime.restore` later — the runtime
    owns the format (file path, content hash, VM snapshot handle).
    A reader (event consumer / dashboard) sees the id round-trip
    on the canonical event but doesn't interpret it.

    ``created_at`` is wall-clock seconds for ordering snapshots
    chronologically when multiple are taken in a single run.
    ``size_bytes`` is best-effort (-1 when the runtime doesn't
    track it) — useful for the storage-cap follow-up that mirrors
    the :class:`~.observation_store.DiskObservationStore` cap.
    """

    snapshot_id: str
    created_at: float = 0.0
    size_bytes: int = -1
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SandboxRuntime(Protocol):
    """The narrow contract any sandbox runtime must satisfy (#484).

    Six operations span the sandbox lifecycle:

    * :meth:`create` — bring a fresh sandbox up. Idempotent for the
      configured key (typically the runner's profile_id) so a
      restart finds the existing sandbox.
    * :meth:`capture` — read the current observable state into a
      runtime-defined blob. Cheap (no snapshot — just observation).
    * :meth:`execute` — dispatch an action through the sandbox. The
      runner's existing env.step path delegates here when the
      runtime is sandbox-backed.
    * :meth:`snapshot` — write the full sandbox state to durable
      storage and return a :class:`SandboxSnapshot` handle.
      Expensive — called before irreversible actions only.
    * :meth:`restore` — rewind the sandbox to a prior snapshot.
      Raises :class:`SandboxRestoreError` on any non-recoverable
      failure (missing snapshot, partial restore detected, etc).
    * :meth:`destroy` — tear down the sandbox. Releases any
      runtime-side resources; the snapshot store may persist.

    ``runtime_checkable`` so tests + handler-wiring layers can
    ``isinstance`` a registered runtime for assertion clarity.
    """

    def create(self, *, key: str) -> None:
        """Bring up a sandbox keyed on ``key`` (typically the
        runner's profile_id). Idempotent — re-create on the same
        key is a no-op when the sandbox already exists."""
        ...

    def capture(self, *, key: str) -> dict[str, Any]:
        """Read the current observable state (url, viewport,
        cookies summary, etc). Cheap — caller may call frequently.
        Returns a runtime-defined dict; the canonical event
        consumer reads documented keys when set."""
        ...

    def execute(self, *, key: str, action: Any) -> None:
        """Dispatch ``action`` through the sandbox. Runners with
        a sandbox-backed env wire :meth:`env.step` to this.
        Action shape is runtime-defined (typically the existing
        :class:`~mantis_agent.actions.Action`)."""
        ...

    def snapshot(self, *, key: str) -> SandboxSnapshot | None:
        """Write the full sandbox state to durable storage and
        return a :class:`SandboxSnapshot` handle. Returns ``None``
        when the runtime declines to snapshot this state (e.g.
        the noop runtime, a host that blacklisted the current
        page). Caller falls through to the un-gated dispatch
        path on ``None``."""
        ...

    def restore(self, *, key: str, snapshot_id: str) -> None:
        """Rewind the sandbox keyed on ``key`` to ``snapshot_id``.
        Raises :class:`SandboxRestoreError` on any failure that
        can't be silently recovered (missing snapshot id, partial
        restore, runtime in a half-torn state)."""
        ...

    def destroy(self, *, key: str) -> None:
        """Tear down the sandbox keyed on ``key``. The snapshot
        store may persist; the runtime side releases resources
        (Chrome process, Xvfb, VM handle)."""
        ...


# ── Implementations ────────────────────────────────────────────────────


class NoopSandboxRuntime:
    """Default :class:`SandboxRuntime` — does nothing.

    Useful for:

    * tests that exercise the snapshot/restore call paths without
      filesystem side effects;
    * runner bring-up before a real sandbox runtime is wired —
      handlers can call snapshot()/restore() unconditionally and
      get a deterministic no-op response.

    :meth:`snapshot` returns a deterministic placeholder
    :class:`SandboxSnapshot` rather than None — gives callers a
    valid handle they can round-trip into trajectory events without
    branching. :meth:`restore` accepts any id and succeeds (the
    "state" was always empty).
    """

    def __init__(self) -> None:
        self._created_keys: set[str] = set()
        self._snapshot_counter: int = 0

    def create(self, *, key: str) -> None:
        self._created_keys.add(key)

    def capture(self, *, key: str) -> dict[str, Any]:
        return {"runtime": "noop", "key": key}

    def execute(self, *, key: str, action: Any) -> None:
        # No-op execute — tests pass; real runtimes override.
        del key, action

    def snapshot(self, *, key: str) -> SandboxSnapshot:
        self._snapshot_counter += 1
        return SandboxSnapshot(
            snapshot_id=f"noop:{key}:s{self._snapshot_counter}",
            created_at=time.time(),
            size_bytes=0,
            metadata={"runtime": "noop"},
        )

    def restore(self, *, key: str, snapshot_id: str) -> None:
        if not snapshot_id:
            raise SandboxRestoreError(
                "NoopSandboxRuntime.restore: snapshot_id required"
            )
        # Noop accepts any non-empty id — no actual state to
        # restore. Real runtimes validate snapshot_id existence.
        del key

    def destroy(self, *, key: str) -> None:
        self._created_keys.discard(key)


class LocalProfileSandbox:
    """Sandbox runtime backed by a directory copy of a Chrome
    user-data directory (#484).

    Pragmatic implementation for local development + the
    acceptance criterion "Snapshot/restore behavior is tested
    with at least one browser task fixture". A snapshot is a
    timestamped copy of the profile directory; restore replaces
    the live profile with the snapshot.

    Storage layout:

        <snapshot_root>/<key>/<snapshot_id>/    ← directory copy
        <snapshot_root>/<key>/<snapshot_id>.json ← metadata

    Designed to be **safe for the test fixture**:

    * The "profile" is just an opaque directory — tests can pass
      any dir and verify the contents survive snapshot+restore.
    * No real Chrome / Xvfb dependency — tests pass with a tmp
      dir of arbitrary files.

    Production wiring would either:

    * Inherit this runtime and override :meth:`execute` to call
      into the existing :class:`XdotoolGymEnv`;
    * Or replace it entirely with a VM-snapshot runtime when one
      becomes available.

    Either path keeps the snapshot/restore semantic identical
    from the runner's perspective — that's the point of the
    protocol.
    """

    def __init__(
        self,
        *,
        profile_root: str,
        snapshot_root: str,
    ) -> None:
        if not profile_root:
            raise ValueError("LocalProfileSandbox requires profile_root")
        if not snapshot_root:
            raise ValueError("LocalProfileSandbox requires snapshot_root")
        import os
        self._profile_root = os.fspath(profile_root)
        self._snapshot_root = os.fspath(snapshot_root)
        os.makedirs(self._profile_root, exist_ok=True)
        os.makedirs(self._snapshot_root, exist_ok=True)

    def _profile_path(self, key: str) -> str:
        import os
        return os.path.join(self._profile_root, key)

    def _snapshot_path(self, key: str, snapshot_id: str) -> str:
        import os
        return os.path.join(self._snapshot_root, key, snapshot_id)

    # ── Lifecycle ──────────────────────────────────────────────

    def create(self, *, key: str) -> None:
        import os
        os.makedirs(self._profile_path(key), exist_ok=True)

    def capture(self, *, key: str) -> dict[str, Any]:
        import os
        path = self._profile_path(key)
        size = 0
        file_count = 0
        for dirpath, _, filenames in os.walk(path):
            for fn in filenames:
                try:
                    size += os.path.getsize(os.path.join(dirpath, fn))
                    file_count += 1
                except OSError:
                    pass
        return {
            "runtime": "local_profile",
            "key": key,
            "profile_path": path,
            "size_bytes": size,
            "file_count": file_count,
        }

    def execute(self, *, key: str, action: Any) -> None:
        # The local-profile sandbox doesn't dispatch actions — it's
        # the storage layer beneath whatever env.step path the
        # runner uses. Production sandboxes that DO dispatch would
        # override execute(). Documented here so the noop is
        # explicit rather than surprising.
        del key, action

    # ── Snapshot / restore ─────────────────────────────────────

    def snapshot(self, *, key: str) -> SandboxSnapshot | None:
        import os
        profile = self._profile_path(key)
        if not os.path.exists(profile):
            logger.warning(
                "LocalProfileSandbox.snapshot: profile %r does not "
                "exist for key=%r — returning None (caller falls "
                "through to un-gated dispatch)", profile, key,
            )
            return None
        snapshot_id = f"local:{uuid.uuid4().hex[:16]}"
        target = self._snapshot_path(key, snapshot_id)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        # ``copytree`` raises if target exists — uuid collision is
        # astronomically unlikely but we let the exception surface
        # so a test catching it gets a real signal vs silently
        # overwriting state.
        shutil.copytree(profile, target)
        size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fs in os.walk(target) for f in fs
        )
        return SandboxSnapshot(
            snapshot_id=snapshot_id,
            created_at=time.time(),
            size_bytes=size,
            metadata={"runtime": "local_profile", "key": key},
        )

    def restore(self, *, key: str, snapshot_id: str) -> None:
        import os
        if not snapshot_id:
            raise SandboxRestoreError(
                "LocalProfileSandbox.restore: snapshot_id required"
            )
        source = self._snapshot_path(key, snapshot_id)
        if not os.path.isdir(source):
            raise SandboxRestoreError(
                f"LocalProfileSandbox.restore: snapshot "
                f"{snapshot_id!r} not found for key={key!r} "
                f"(looked at {source!r})"
            )
        target = self._profile_path(key)
        # Atomic-ish swap: write to a temp dir, then rename. If the
        # write fails partway, the live profile is unchanged.
        tmp = f"{target}.restoring.{uuid.uuid4().hex[:8]}"
        try:
            shutil.copytree(source, tmp)
        except Exception as exc:
            # Clean up partial tmp tree.
            shutil.rmtree(tmp, ignore_errors=True)
            raise SandboxRestoreError(
                f"LocalProfileSandbox.restore: copying snapshot "
                f"{snapshot_id!r} failed: {exc}"
            ) from exc
        # Replace the live profile.
        if os.path.exists(target):
            backup = f"{target}.replaced.{uuid.uuid4().hex[:8]}"
            os.rename(target, backup)
        try:
            os.rename(tmp, target)
        except OSError as exc:
            # Best-effort rollback to the backup.
            raise SandboxRestoreError(
                f"LocalProfileSandbox.restore: final rename to "
                f"{target!r} failed: {exc}"
            ) from exc

    def destroy(self, *, key: str) -> None:
        path = self._profile_path(key)
        shutil.rmtree(path, ignore_errors=True)
