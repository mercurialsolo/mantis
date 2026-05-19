"""Pre-dispatch preview / cross-check gate for high-risk actions (#481).

The CUA design's safety contract: before dispatching an
:class:`~mantis_agent.cua_contracts.ReversibilityClass.IRREVERSIBLE`
action (form submission, file upload, app launch, ToS / purchase /
send / delete confirmations dressed as a submit), highlight the
proposed target on a fresh screenshot and ask an independent
verifier "does this match the planner's intent?". Block dispatch
when the verifier says no or confidence is low.

This module owns the primitive — a pure function callers invoke
right before they hit ``env.step(...)``. It does NOT wire itself
into the existing form / click handlers; that integration is
intentionally a separate PR so the behaviour change can be
reviewed independently of the contract.

Why a separate verifier path:

The grounding model that picked the coordinates has already
committed to its answer. Asking the same model to re-check is a
near-no-op (it will agree with itself). A useful preview cross-
checks against either:

* a different model — e.g. the planner uses Claude grounding, the
  preview asks Holo3 (or vice versa);
* the same model with a different prompt framing — e.g. "is the
  highlighted region a Login button?" instead of "where's the
  Login button?".

For v1 the preview takes a generic ``verifier_fn`` callable so
callers can plug in whichever cross-check they prefer; the
infrastructure stays neutral.

Failure mode contract:

* preview PASS → caller dispatches as planned.
* preview FAIL → caller halts the step with a structured recovery
  hint; the runner's recovery layer can prompt-for-human or replan.
* preview ERROR (verifier raised / returned no usable signal) →
  caller treats as FAIL by default — safety contract is fail-closed
  on irreversible actions.

Gated by env var ``MANTIS_PREVIEW_GATE`` so production roll-out
goes: ship code → enable on staging → enable on prod tenant-by-
tenant. Default off keeps the existing production behaviour
unchanged.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from ..cua_contracts import ReversibilityClass, classify_action
from ..cua_contracts.validation import ContractValidationError

logger = logging.getLogger(__name__)


_PREVIEW_GATE_ENV: str = "MANTIS_PREVIEW_GATE"
_PREVIEW_CONFIDENCE_THRESHOLD_ENV: str = "MANTIS_PREVIEW_CONFIDENCE_THRESHOLD"
_DEFAULT_CONFIDENCE_THRESHOLD: float = 0.6


@dataclass(frozen=True)
class PreviewResult:
    """Outcome of a single preview cross-check.

    Surfaced into the runner as structured recovery metadata when
    the gate blocks dispatch — populates
    :class:`~mantis_agent.cua_contracts.ActionResult.grounding_trace`'s
    ``confirmation_evidence`` field so post-mortems see exactly
    why an irreversible step was held.
    """

    passed: bool
    confidence: float
    reason: str          # short machine code: "verifier_rejected", "low_confidence", "verifier_error", "skipped"
    evidence: str = ""   # human-readable verifier prose


class _VerifierCallable(Protocol):
    """Shape callers must satisfy when wiring a preview verifier.

    Given a labelled screenshot + the intent prose, return a
    ``(passed, confidence, evidence)`` triple. Implementations can
    wrap Claude / Holo3 / a heuristic — the gate primitive doesn't
    care which.
    """

    def __call__(
        self,
        *,
        screenshot: Any,
        intent: str,
        target_label: str,
        target_coordinates: tuple[int, int],
    ) -> tuple[bool, float, str]:
        ...


def is_enabled() -> bool:
    """True when the preview gate is opted in via env.

    Off by default. Operators flip the env per tenant / per deploy
    when rolling out the safety gate. Tests monkeypatch
    ``MANTIS_PREVIEW_GATE=on``.
    """
    return os.environ.get(_PREVIEW_GATE_ENV, "").strip().lower() in {"on", "1", "true", "yes"}


def confidence_threshold() -> float:
    """Configured minimum confidence; below this the gate blocks even
    when ``passed=True`` (defensive against verifiers that report
    weak signals as positive)."""
    raw = os.environ.get(_PREVIEW_CONFIDENCE_THRESHOLD_ENV, "").strip()
    if not raw:
        return _DEFAULT_CONFIDENCE_THRESHOLD
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "preview gate: invalid %s=%r; falling back to default %.2f",
            _PREVIEW_CONFIDENCE_THRESHOLD_ENV, raw, _DEFAULT_CONFIDENCE_THRESHOLD,
        )
        return _DEFAULT_CONFIDENCE_THRESHOLD
    # Clamp to a sensible range — values outside [0, 1] are config
    # bugs, not intent.
    return max(0.0, min(1.0, value))


def evaluate(
    *,
    action_type: str,
    intent: str,
    target_label: str,
    target_coordinates: tuple[int, int],
    screenshot: Any,
    verifier: _VerifierCallable | Callable[..., tuple[bool, float, str]],
) -> PreviewResult:
    """Run the pre-dispatch preview cross-check.

    Returns a :class:`PreviewResult` the caller branches on. Never
    raises — verifier errors are caught and downgraded to
    ``passed=False, reason="verifier_error"`` so the safety contract
    holds even when the cross-check infrastructure is broken.

    Skip conditions (return ``passed=True, reason="skipped"``):

    * preview gate disabled via env (production default);
    * ``action_type`` is not :class:`ReversibilityClass.IRREVERSIBLE`
      (no need to gate reversible / read-only actions).

    Fail conditions (return ``passed=False`` with a reason code):

    * ``verifier_rejected`` — verifier returned ``passed=False``.
    * ``low_confidence`` — verifier said ``passed=True`` but
      confidence < :func:`confidence_threshold`.
    * ``verifier_error`` — verifier raised, returned a non-
      conforming value, or the action_type itself was rejected by
      the ontology.

    The caller is responsible for stamping the result into the
    StepResult / grounding trace and for emitting a HALT / retry
    hint accordingly.
    """
    if not is_enabled():
        return PreviewResult(passed=True, confidence=1.0, reason="skipped")

    # Fail-closed on unknown action types — the safety contract is
    # "no irreversible action passes through ungated"; we can't
    # honour that if we can't classify.
    try:
        cls = classify_action(action_type)
    except ContractValidationError as exc:
        logger.warning("preview gate: unknown action_type %r — %s", action_type, exc)
        return PreviewResult(
            passed=False, confidence=0.0,
            reason="verifier_error",
            evidence=f"unknown action_type: {exc}",
        )

    if cls is not ReversibilityClass.IRREVERSIBLE:
        return PreviewResult(passed=True, confidence=1.0, reason="skipped")

    try:
        passed, confidence, evidence = verifier(
            screenshot=screenshot,
            intent=intent,
            target_label=target_label,
            target_coordinates=target_coordinates,
        )
    except Exception as exc:  # noqa: BLE001 — verifier errors must not crash the runner
        logger.warning(
            "preview gate: verifier raised on action_type=%s label=%r — %s",
            action_type, target_label, exc,
        )
        return PreviewResult(
            passed=False, confidence=0.0,
            reason="verifier_error",
            evidence=f"verifier raised: {exc.__class__.__name__}: {exc}",
        )

    # Coerce types defensively — verifiers are often hand-rolled.
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        logger.warning(
            "preview gate: verifier returned non-float confidence=%r; failing closed",
            confidence,
        )
        return PreviewResult(
            passed=False, confidence=0.0,
            reason="verifier_error",
            evidence=f"non-float confidence: {confidence!r}",
        )

    if not passed:
        return PreviewResult(
            passed=False, confidence=confidence,
            reason="verifier_rejected",
            evidence=str(evidence)[:500],
        )

    threshold = confidence_threshold()
    if confidence < threshold:
        return PreviewResult(
            passed=False, confidence=confidence,
            reason="low_confidence",
            evidence=(
                f"verifier passed but confidence={confidence:.2f} < "
                f"threshold={threshold:.2f}: {str(evidence)[:300]}"
            ),
        )

    return PreviewResult(
        passed=True, confidence=confidence,
        reason="verifier_accepted",
        evidence=str(evidence)[:500],
    )
