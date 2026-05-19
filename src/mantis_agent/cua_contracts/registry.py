"""Model registry with eval-report-gated promotion (#490).

A typed record per (role, version) that tracks rollout state +
the eval report that justified each promotion. Without this,
promotions are unauditable — the only signal that a model went
to prod is a deploy log + a Slack note.

The registry's job is narrow and substrate-only in v1:

* Hold records: role / version / artifact / rollout_state /
  eval_report_ref / prev_prod_version.
* Enforce the **eval-report-gated promotion** invariant — a model
  cannot move to SHADOW / CANARY / PROD without an eval report
  reference.
* Track the prev_prod_version per role so rollback is a one-call
  metadata flip (no need to re-deploy a build the rollback target
  artifact is already pinned).

What this PR doesn't do (intentionally):

* Persistence — the default :class:`InMemoryModelRegistry` is the
  test default + bring-up surface; a persistent backend (disk /
  database) is a follow-up.
* Integration with the existing training/promotion_scorecard.py
  scripts — they keep producing scorecards; the registry will
  consume those references once the scorecards emit a stable
  artifact path.
* Active-version lookup hooks into :class:`SplitFacade` /
  PassthroughFacade — those still take an explicit version_pin;
  the registry just holds the data the picker will consult.

The acceptance criteria from #490:

* Candidate models can be registered with role/version metadata. ✓
* Promotion to shadow/canary/prod records the eval report used. ✓
* Rollback target is visible and actionable. ✓
* Tests cover register, promote, reject missing eval, and
  rollback metadata. ✓ (see test_model_registry_and_shadow.py)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from .serving import Role


class RolloutState(str, Enum):
    """Where a (role, version) record sits in the rollout pipeline.

    State transitions (typical happy path):

        CANDIDATE → SHADOW → CANARY → PROD
                              ↓
                          ROLLED_BACK

    * **CANDIDATE** — registered but not yet exposed to any
      production traffic. Default for fresh registrations.
    * **SHADOW** — invoked alongside prod (via SplitFacade) to
      produce non-committing TrajectoryEvents for comparison. No
      side-effectful dispatch. Requires eval report at promotion.
    * **CANARY** — handles a slice of prod traffic (side-effectful)
      to grade real-world impact before full rollout. Requires
      eval report.
    * **PROD** — current production version for the role. Exactly
      one PROD record per role is the intended invariant (the
      registry's :meth:`promote` enforces this — promoting a new
      version to PROD demotes the prior PROD to ROLLED_BACK and
      stashes its version as ``prev_prod_version`` for fast
      rollback).
    * **ROLLED_BACK** — previously PROD, now displaced. The
      record stays around so an operator can re-promote it via
      :meth:`rollback`.
    """

    CANDIDATE = "candidate"
    SHADOW = "shadow"
    CANARY = "canary"
    PROD = "prod"
    ROLLED_BACK = "rolled_back"


# States that constitute exposure to production traffic. Promoting
# into any of these requires an eval report reference (per #490
# acceptance: "Promotion to shadow/canary/prod records the eval
# report used"). CANDIDATE is registered-but-inert; ROLLED_BACK is
# explicitly the "stop exposing" state — neither requires a fresh
# eval since they don't take new traffic.
_EVAL_GATED_STATES: frozenset[RolloutState] = frozenset({
    RolloutState.SHADOW,
    RolloutState.CANARY,
    RolloutState.PROD,
})


class ModelRegistryError(RuntimeError):
    """Raised when a registry operation violates an invariant."""


@dataclass(frozen=True)
class ModelRegistryRecord:
    """One row in the registry — keyed by ``(role, version)``.

    Frozen so a record returned from the registry can't be mutated
    in place; state changes go through the registry's typed
    methods which return a new record.

    Fields:

    * ``role`` — which agent role this version serves
      (:class:`Role` from #487).
    * ``version`` — string identifier the facade reads back from
      the underlying client (matches ``ModelCallResult.model_version``).
      Free-form; pinning per-role.
    * ``artifact`` — pointer to the actual model weights / config
      (deploy image tag, HuggingFace ref, S3 path). Opaque to the
      registry; consulted by the deploy / serving infra.
    * ``rollout_state`` — current position in the rollout pipeline.
    * ``eval_report_ref`` — required for SHADOW / CANARY / PROD;
      empty otherwise.
    * ``prev_prod_version`` — set when this record displaces an
      earlier PROD. Stores the prior version so rollback restores
      it without consulting external state.
    * ``metadata`` — free-form for future stamps (compiler
      version, hardware target, etc).
    """

    role: Role
    version: str
    artifact: str = ""
    rollout_state: RolloutState = RolloutState.CANDIDATE
    eval_report_ref: str = ""
    prev_prod_version: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class ModelRegistry(Protocol):
    """The narrow contract any registry backend must satisfy.

    Implementations:

    * persist records keyed on ``(role, version)``;
    * enforce the eval-report-gated promotion invariant;
    * maintain at most one ``PROD`` record per role;
    * surface ``prev_prod_version`` for fast rollback.

    ``runtime_checkable`` so tests can ``isinstance`` a wired
    registry for assertion clarity.
    """

    def register(
        self,
        *,
        role: Role,
        version: str,
        artifact: str = "",
        metadata: dict[str, str] | None = None,
    ) -> ModelRegistryRecord:
        """Add a CANDIDATE record for ``(role, version)``.

        Reregistering the same (role, version) is a no-op that
        returns the existing record — registration is idempotent
        so the deploy pipeline can call register on every push
        without checking first.

        Raises :class:`ModelRegistryError` only when the
        ``version`` is empty (registries must be addressable by
        version string)."""
        ...

    def promote(
        self,
        *,
        role: Role,
        version: str,
        to_state: RolloutState,
        eval_report_ref: str,
    ) -> ModelRegistryRecord:
        """Move ``(role, version)`` to ``to_state``.

        ``eval_report_ref`` is required when ``to_state`` is
        SHADOW / CANARY / PROD. Promotion to PROD demotes the
        prior PROD record (if any) to ROLLED_BACK and stashes
        its version on the new record's ``prev_prod_version``.

        Raises :class:`ModelRegistryError` on:
        * unknown (role, version) (must register first);
        * promotion to an eval-gated state without a report ref;
        * promotion to CANDIDATE (use register for fresh records);
        * promotion to ROLLED_BACK (use :meth:`rollback` instead)."""
        ...

    def rollback(self, *, role: Role) -> ModelRegistryRecord:
        """Restore the prior PROD version for ``role``.

        Reads the current PROD record's ``prev_prod_version``,
        demotes it to ROLLED_BACK, and promotes the prior version
        back to PROD. Eval report ref of the prior version is
        preserved (it's the same report that justified the
        original promotion).

        Raises :class:`ModelRegistryError` when:
        * no PROD record exists for the role;
        * the current PROD has no ``prev_prod_version`` (first
          ever prod has nothing to roll back to)."""
        ...

    def get(
        self, *, role: Role, version: str,
    ) -> ModelRegistryRecord | None:
        """Look up the record for ``(role, version)``. Returns
        ``None`` for unknown — no exception."""
        ...

    def list_role(self, *, role: Role) -> list[ModelRegistryRecord]:
        """All records for the role, in registration order. Caller
        filters by rollout_state as needed."""
        ...

    def current_prod(self, *, role: Role) -> ModelRegistryRecord | None:
        """Convenience: return the (single) PROD record for the
        role, or None when nothing is in prod yet. The
        SplitFacade's prod-side picker will call this to resolve
        the active version when no explicit version_pin is set."""
        ...


class InMemoryModelRegistry:
    """Default :class:`ModelRegistry` — keeps records in a process-
    local dict (#490).

    Useful for:

    * tests;
    * single-process deploys where registry state is rebuilt on
      restart from a config-as-code source;
    * bring-up before a persistent backend is wired.

    Production wiring uses a disk- or database-backed
    implementation that satisfies the same Protocol; this class
    stays the test default + the explicit "I don't need
    persistence" surface.
    """

    def __init__(self) -> None:
        # Keyed on (role, version) tuple. Order preserved by
        # insertion so ``list_role`` returns in registration order.
        self._records: dict[tuple[Role, str], ModelRegistryRecord] = {}

    # ── Register ────────────────────────────────────────────────

    def register(
        self,
        *,
        role: Role,
        version: str,
        artifact: str = "",
        metadata: dict[str, str] | None = None,
    ) -> ModelRegistryRecord:
        if not version:
            raise ModelRegistryError(
                "ModelRegistry.register: version must be non-empty"
            )
        key = (role, version)
        if key in self._records:
            # Idempotent — fresh registration of an existing
            # record returns the existing one (matches the
            # contract documented on the Protocol).
            return self._records[key]
        record = ModelRegistryRecord(
            role=role, version=version, artifact=artifact,
            rollout_state=RolloutState.CANDIDATE,
            eval_report_ref="",
            prev_prod_version="",
            metadata=dict(metadata or {}),
        )
        self._records[key] = record
        return record

    # ── Promote ─────────────────────────────────────────────────

    def promote(
        self,
        *,
        role: Role,
        version: str,
        to_state: RolloutState,
        eval_report_ref: str,
    ) -> ModelRegistryRecord:
        if to_state is RolloutState.CANDIDATE:
            raise ModelRegistryError(
                "ModelRegistry.promote: cannot promote to CANDIDATE "
                "— use register() for fresh records"
            )
        if to_state is RolloutState.ROLLED_BACK:
            raise ModelRegistryError(
                "ModelRegistry.promote: cannot promote to ROLLED_BACK "
                "— use rollback() to displace prod"
            )
        if to_state in _EVAL_GATED_STATES and not eval_report_ref:
            raise ModelRegistryError(
                f"ModelRegistry.promote: eval_report_ref is required "
                f"when promoting to {to_state.value} — every "
                f"production-exposing promotion must reference the "
                f"eval report that justified it (#490)"
            )
        key = (role, version)
        record = self._records.get(key)
        if record is None:
            raise ModelRegistryError(
                f"ModelRegistry.promote: no record for "
                f"({role.value}, {version!r}) — register() first"
            )
        prev_prod_version = record.prev_prod_version
        if to_state is RolloutState.PROD:
            # Demote the existing PROD (if any) to ROLLED_BACK and
            # stash its version on the new record for fast
            # rollback.
            current = self.current_prod(role=role)
            if current is not None and current.version != version:
                self._records[(role, current.version)] = ModelRegistryRecord(
                    role=current.role,
                    version=current.version,
                    artifact=current.artifact,
                    rollout_state=RolloutState.ROLLED_BACK,
                    eval_report_ref=current.eval_report_ref,
                    prev_prod_version=current.prev_prod_version,
                    metadata=dict(current.metadata),
                )
                prev_prod_version = current.version
        new_record = ModelRegistryRecord(
            role=record.role,
            version=record.version,
            artifact=record.artifact,
            rollout_state=to_state,
            eval_report_ref=eval_report_ref,
            prev_prod_version=prev_prod_version,
            metadata=dict(record.metadata),
        )
        self._records[key] = new_record
        return new_record

    # ── Rollback ────────────────────────────────────────────────

    def rollback(self, *, role: Role) -> ModelRegistryRecord:
        current = self.current_prod(role=role)
        if current is None:
            raise ModelRegistryError(
                f"ModelRegistry.rollback: no PROD record for "
                f"role={role.value} — nothing to roll back from"
            )
        if not current.prev_prod_version:
            raise ModelRegistryError(
                f"ModelRegistry.rollback: PROD record "
                f"({role.value}, {current.version!r}) has no "
                f"prev_prod_version — first-ever prod has nothing "
                f"to fall back to"
            )
        prev_key = (role, current.prev_prod_version)
        prev_record = self._records.get(prev_key)
        if prev_record is None:
            raise ModelRegistryError(
                f"ModelRegistry.rollback: prev_prod_version "
                f"{current.prev_prod_version!r} not in registry — "
                f"state corrupted"
            )
        # Demote current PROD to ROLLED_BACK.
        self._records[(role, current.version)] = ModelRegistryRecord(
            role=current.role,
            version=current.version,
            artifact=current.artifact,
            rollout_state=RolloutState.ROLLED_BACK,
            eval_report_ref=current.eval_report_ref,
            prev_prod_version=current.prev_prod_version,
            metadata=dict(current.metadata),
        )
        # Promote prev back to PROD. prev_prod_version on the
        # restored record is BLANK — the next promotion from a
        # candidate will repopulate it.
        restored = ModelRegistryRecord(
            role=prev_record.role,
            version=prev_record.version,
            artifact=prev_record.artifact,
            rollout_state=RolloutState.PROD,
            eval_report_ref=prev_record.eval_report_ref,
            prev_prod_version="",
            metadata=dict(prev_record.metadata),
        )
        self._records[prev_key] = restored
        return restored

    # ── Lookups ─────────────────────────────────────────────────

    def get(
        self, *, role: Role, version: str,
    ) -> ModelRegistryRecord | None:
        return self._records.get((role, version))

    def list_role(self, *, role: Role) -> list[ModelRegistryRecord]:
        return [r for (r_role, _), r in self._records.items() if r_role == role]

    def current_prod(self, *, role: Role) -> ModelRegistryRecord | None:
        for record in self.list_role(role=role):
            if record.rollout_state is RolloutState.PROD:
                return record
        return None
