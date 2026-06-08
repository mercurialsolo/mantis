"""Resolve `compute_backend` from layered sources (#785, PR 2).

Precedence — highest wins:

1. **Plan-level** — `plan["runtime"]["compute_backend"]` (the plan
   author's choice; most local).
2. **Submission-level** — HTTP body / CLI flag (operator override —
   e.g. A/B-running the same plan on the other plane).
3. **Global default** — `ComputeBackend.COMPUTER_PLANE` (Xvfb + xdotool
   stays the safe default; pure-CUA is what production has been on).

The resolver is intentionally pure — no env-var reads, no I/O — so the
precedence is observable and testable in isolation.
"""

from __future__ import annotations

from typing import Any

from .compute_contract import ComputeBackend


def _coerce(value: Any) -> ComputeBackend | None:
    if value is None:
        return None
    if isinstance(value, ComputeBackend):
        return value
    if isinstance(value, str):
        try:
            return ComputeBackend(value)
        except ValueError:
            return None
    return None


def resolve_compute_backend(
    *,
    plan: dict[str, Any] | None = None,
    submission_value: Any = None,
    default: ComputeBackend = ComputeBackend.COMPUTER_PLANE,
) -> ComputeBackend:
    """Return the effective `ComputeBackend` for this run.

    `plan` is the raw micro-plan dict (after JSON parse). Looks at
    `plan["runtime"]["compute_backend"]`; tolerates the legacy
    flat-array plan shape (no runtime block) by treating it as absent.

    `submission_value` may be a `ComputeBackend`, a string, or None.
    Unrecognized strings fall through to the next layer (do NOT raise —
    forward-compat with future backend names).
    """
    if isinstance(plan, dict):
        runtime = plan.get("runtime")
        if isinstance(runtime, dict):
            plan_value = _coerce(runtime.get("compute_backend"))
            if plan_value is not None:
                return plan_value

    sub_value = _coerce(submission_value)
    if sub_value is not None:
        return sub_value

    return default
