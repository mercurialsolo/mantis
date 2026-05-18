"""Canonical per-step trajectory event emitter (#478).

The emitter is the writer half of the contract scaffolding from
:mod:`mantis_agent.cua_contracts`. It takes legacy ``StepResult`` +
``MicroIntent`` + dispatcher state and produces a validated
:class:`TrajectoryEvent`, then appends it to a JSONL store keyed by
``run_id``.

Idempotency:

* Each ``(run_id, step_index)`` pair emits exactly once.
* Resumed runs read the existing JSONL on startup and skip indices
  that were already written — no duplicate-on-retry, no
  duplicate-on-resume.
* The emit path is best-effort wrt the runner: a validation /
  filesystem failure logs and returns ``False`` but does NOT raise.
  The runner advances regardless — the canonical event stream is a
  side channel, not a step-blocker. (Future:#480 makes the verdict
  itself mandatory; that's a separate enforcement.)

Storage shape:

* ``<store_dir>/trajectory.jsonl`` — one JSON-encoded
  :class:`TrajectoryEvent` per line. Append-only so a partial write
  is recoverable. The directory is created on first emit.
* Same shape across local / Modal / Baseten paths so downstream
  consumers (shadow router, eval, registry) have one reader.

This module deliberately doesn't touch the existing
:mod:`mantis_agent.gym.trace_exporter` — that file's output stays as
compatibility plumbing for the trace UI and labelled-trace
converter. The canonical event stream is the *source of truth* the
new design will build on; trace_exporter is a view.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from .adapters import (
    action_result_from_action,
    observation_from_screenshot_ref,
    step_from_micro_intent,
    verdict_from_step_result,
)
from .types import ActionResult, SCHEMA_VERSION, TrajectoryEvent
from .validation import ContractValidationError, validate_trajectory_event

if TYPE_CHECKING:
    from ..actions import Action
    from ..gym.checkpoint import StepResult
    from ..plan_decomposer import MicroIntent

logger = logging.getLogger(__name__)


JSONL_FILENAME: str = "trajectory.jsonl"


def _dataclass_to_jsonable(obj: Any) -> Any:
    """Recursive dataclass → plain-dict conversion that also collapses
    string enums to their ``.value``.

    Stdlib's :func:`dataclasses.asdict` already recurses into nested
    dataclasses but leaves :class:`Enum` instances as-is, which
    ``json.dumps`` chokes on without ``default=`` plumbing. We collapse
    enums up front so callers see plain JSON-friendly dicts (helpful
    for tests, golden files, and any consumer that round-trips through
    ``json.loads``).
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _dataclass_to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _dataclass_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_jsonable(v) for v in obj]
    return obj


class TrajectoryEmitter:
    """Append-only, idempotent writer of :class:`TrajectoryEvent` records.

    Construct one per run. The emitter:

    * Validates each event against :func:`validate_trajectory_event`.
    * Refuses duplicate ``step_index`` writes for the same ``run_id``
      (in-memory dedup populated from the existing JSONL on init).
    * Appends one JSON-encoded event per line under
      ``<store_dir>/trajectory.jsonl``.
    * Logs + returns ``False`` on validation / IO failure; the runner
      keeps going.

    The emitter is intentionally narrow: callers pass the runtime
    artifacts (``MicroIntent``, ``StepResult``, optional ``Action`` /
    grounding trace / screenshot ref) and it does the projection +
    persistence. Future wiring (#479 grounding, #480 typed verdict,
    #487 / #488 version stamps) plumbs richer inputs through these
    same kwargs without changing the public surface.
    """

    def __init__(
        self,
        run_id: str,
        store_dir: str,
        *,
        versions: dict[str, str] | None = None,
    ) -> None:
        if not run_id:
            raise ValueError("TrajectoryEmitter requires a non-empty run_id")
        if not store_dir:
            raise ValueError("TrajectoryEmitter requires a non-empty store_dir")
        self.run_id = run_id
        self.store_dir = store_dir
        # ``versions`` is the slot for #487 / #488 stamps (planner /
        # grounding / browser / sandbox). Empty dict is fine in v1 —
        # the validator only requires presence, not contents.
        self.versions: dict[str, str] = dict(versions or {})
        self._jsonl_path: str = os.path.join(store_dir, JSONL_FILENAME)
        self._emitted_indices: set[int] = set()
        self._load_existing()

    # ── Public API ─────────────────────────────────────────────────

    def emit(
        self,
        step: "MicroIntent",
        result: "StepResult",
        *,
        action: "Action | None" = None,
        dispatched: bool = True,
        dispatch_error: str = "",
        grounding_trace: dict[str, Any] | None = None,
        screenshot_ref: str = "",
        url: str = "",
        viewport: tuple[int, int] = (0, 0),
        cost_usd: float = 0.0,
    ) -> bool:
        """Build + validate + persist a canonical event.

        Returns ``True`` on a fresh emit, ``False`` on duplicate or
        validation / IO failure. The runner treats the boolean as
        diagnostic — emit is a side channel.

        Idempotency is keyed on ``(run_id, step.step_index)``; a
        second call with the same index is a silent no-op so handlers
        that retry / replan / demote a step don't double-record.

        ``screenshot_ref`` defaults to a synthetic
        ``step_<index>_<run_id>`` placeholder when the caller doesn't
        have a real reference. The placeholder is short + non-base64,
        so the validator accepts it; the placeholder makes the JSONL
        self-contained pending the #479 grounding-trace integration
        that will plumb real screenshot refs.
        """
        step_index = int(getattr(result, "step_index", -1))
        if step_index in self._emitted_indices:
            return False

        try:
            event = self._build_event(
                step, result,
                action=action, dispatched=dispatched,
                dispatch_error=dispatch_error,
                grounding_trace=grounding_trace,
                screenshot_ref=screenshot_ref,
                url=url, viewport=viewport,
                cost_usd=cost_usd,
            )
            validate_trajectory_event(event)
        except ContractValidationError as exc:
            logger.warning(
                "trajectory emit step %d: validation failed: %s",
                step_index, exc,
            )
            return False
        except Exception as exc:  # noqa: BLE001 — never break a run
            logger.warning(
                "trajectory emit step %d: build raised: %s",
                step_index, exc,
            )
            return False

        try:
            self._append(event)
        except OSError as exc:
            logger.warning(
                "trajectory emit step %d: append failed (%s): %s",
                step_index, self._jsonl_path, exc,
            )
            return False

        self._emitted_indices.add(step_index)
        return True

    @property
    def jsonl_path(self) -> str:
        """Path to the canonical event stream — exposed for tests +
        downstream consumers that want to mmap / tail the file."""
        return self._jsonl_path

    def emitted_indices(self) -> set[int]:
        """Snapshot of step indices already persisted for this run.
        Useful for assertions in tests + for the runner to skip
        re-emitting on resume."""
        return set(self._emitted_indices)

    # ── Internal helpers ───────────────────────────────────────────

    def _build_event(
        self,
        step: "MicroIntent",
        result: "StepResult",
        *,
        action: "Action | None",
        dispatched: bool,
        dispatch_error: str,
        grounding_trace: dict[str, Any] | None,
        screenshot_ref: str,
        url: str,
        viewport: tuple[int, int],
        cost_usd: float,
    ) -> TrajectoryEvent:
        step_index = int(getattr(result, "step_index", -1))
        ref = screenshot_ref or self._placeholder_screenshot_ref(step_index)
        # captured_at defaults to "now" because legacy StepResult
        # doesn't carry the pre-action screenshot timestamp. #479's
        # grounding-trace integration will plumb the real timestamp;
        # for v1, ``now`` is close enough to bound the observation in
        # time and the validator accepts any non-negative float.
        observation = observation_from_screenshot_ref(
            ref, url=url, viewport=viewport, captured_at=time.time(),
        )
        # ``result.last_action`` is the runner's preferred source for
        # the action that drove this step; fall back to the explicit
        # ``action`` kwarg when the caller has a better signal.
        legacy_action = action if action is not None else getattr(result, "last_action", None)
        action_result = action_result_from_action(
            legacy_action,
            dispatched=dispatched,
            dispatch_error=dispatch_error,
            grounding_trace=grounding_trace,
        )
        # Deterministic handlers (navigate / paginate / gate /
        # fill_field) don't synthesise an ``Action`` — they execute
        # the step directly and leave ``StepResult.last_action`` at
        # None. The validator requires a non-empty ``action_type``,
        # so we fall back to the legacy ``MicroIntent.type`` (the
        # canonical name for what the handler dispatched). The
        # frozen ActionResult is replaced rather than mutated.
        if not action_result.action_type:
            step_type = str(getattr(step, "type", "") or "")
            if step_type:
                action_result = ActionResult(
                    schema_version=action_result.schema_version,
                    action_type=step_type,
                    params=action_result.params,
                    grounding_trace=action_result.grounding_trace,
                    dispatched=action_result.dispatched,
                    dispatch_error=action_result.dispatch_error,
                )
        return TrajectoryEvent(
            schema_version=SCHEMA_VERSION,
            run_id=self.run_id,
            step_index=step_index,
            step=step_from_micro_intent(step),
            observation=observation,
            action_result=action_result,
            verdict=verdict_from_step_result(result),
            versions=dict(self.versions),
            latency_seconds=float(getattr(result, "duration", 0.0) or 0.0),
            cost_usd=float(cost_usd),
            committed=True,
            emitted_at=time.time(),
        )

    def _placeholder_screenshot_ref(self, step_index: int) -> str:
        # Short, distinctive, deterministic — keeps the validator
        # happy (well under the inline-blob length threshold) and
        # makes a JSONL audit grep-able.
        return f"placeholder://{self.run_id}/step_{step_index}.png"

    def _append(self, event: TrajectoryEvent) -> None:
        os.makedirs(self.store_dir, exist_ok=True)
        line = json.dumps(_dataclass_to_jsonable(event), separators=(",", ":"))
        # newline-delimited JSON — append in a single write so a
        # partial write of one record can't corrupt a sibling. The
        # filesystem guarantees atomic single-syscall writes for
        # bytes under PIPE_BUF (4KB on Linux/macOS); events well
        # exceed that, so a crash mid-line is recoverable by
        # truncating the trailing partial line on next reader open.
        with open(self._jsonl_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _load_existing(self) -> None:
        """Populate ``_emitted_indices`` from a prior run's JSONL.

        Resume case (#478 acceptance: idempotent across restarts).
        A trailing partial line — possible if the writer crashed
        mid-flush — is tolerated: we read what JSON we can and ignore
        the rest. Subsequent emits will append clean records past
        the partial line, which downstream readers should handle the
        same way.
        """
        try:
            f = open(self._jsonl_path, encoding="utf-8")
        except FileNotFoundError:
            return
        with f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # Partial trailing line from a crashed writer —
                    # stop scanning; next emit will append past it.
                    break
                idx = record.get("step_index")
                if isinstance(idx, int) and idx >= 0:
                    self._emitted_indices.add(idx)
