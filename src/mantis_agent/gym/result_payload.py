"""Shared StepResult → result.json serialization.

Both ``mantis plan run`` (local) and ``mantis plan run-modal`` (the
Modal entrypoint) write a ``result.json`` with a ``steps`` array.
Keeping the packaging in one place ensures the two paths can't drift —
a failed step looks the same whether the browser ran on the laptop or
inside Modal.

Successful steps stay slim (index / intent / success / data / duration
/ steps_used). Failed steps carry the diagnostics added by the
executor in :func:`~.run_executor._stamp_failure_context`:

* ``failure_class`` — one of the categories from :mod:`~.failure_class`
* ``final_url`` — browser URL at the moment of failure
* ``page_title`` — page title at the moment of failure
* ``last_action`` — the final :class:`~mantis_agent.actions.Action`
  dispatched before the step recorded failure
* ``screenshot_b64`` — base64-encoded PNG of the post-failure
  viewport, when available

The screenshot is only included on failure to keep success payloads
small — successful step screenshots already get the per-run keep cap
enforced by :meth:`MicroPlanRunner._enforce_screenshot_cap`.
"""

from __future__ import annotations

import base64
from typing import Any


def _as_str(value: Any, default: str = "") -> str:
    """Coerce optional StepResult fields to a JSON-safe string.

    A real ``StepResult`` populates these fields with strings or empties.
    Some tests / hosts hand the executor a ``MagicMock`` or a near-duck;
    coerce defensively so the result payload always round-trips through
    ``json.dumps``.
    """
    return value if isinstance(value, str) else default


def pack_step(r: Any, *, time_breakdown: dict[str, float] | None = None) -> dict:
    payload = {
        "index": r.step_index,
        "intent": r.intent,
        "success": bool(r.success),
        "data": _as_str(getattr(r, "data", "")),
        "duration": float(getattr(r, "duration", 0.0)),
        "steps_used": int(getattr(r, "steps_used", 0)),
    }

    # Epic #362 Phase B: per-step wall-time bucket breakdown. Always
    # present when provided so consumers can iterate buckets without
    # branching on key existence; callers that don't have a TimeMeter
    # (legacy hosts, tests) omit the arg and the key stays out.
    if time_breakdown is not None:
        payload["time_breakdown"] = {k: round(v, 3) for k, v in time_breakdown.items()}

    # #419: brain reasoning lands on EVERY step (not just failures) so
    # the audit triple — reasoning / predicted / observed — is complete
    # for post-mortem and SFT pipelines. Handlers that don't drive a
    # brain leave ``reasoning`` empty; the key is omitted in that case
    # to keep the success-step payload slim.
    reasoning = _as_str(getattr(r, "reasoning", ""))
    if reasoning:
        payload["reasoning"] = reasoning

    # #480 mandatory verdict: surface the typed verdict on every step
    # (success or failure) so result.json carries the structured
    # outcome alongside the legacy ``success`` / ``failure_class``
    # fields. The executor stamps ``r.verdict`` before the cursor
    # advances; only ad-hoc callers that bypass the executor (legacy
    # hosts, sim-env halts) leave it None — those keep the legacy
    # shape unchanged. ``isinstance`` check matches the established
    # pattern (cf. ``last_action`` above) — defensive against
    # MagicMock duck-types from test hosts that auto-create
    # attributes that satisfy ``is not None``.
    from ..cua_contracts.types import Verdict
    verdict = getattr(r, "verdict", None)
    if isinstance(verdict, Verdict):
        kind = getattr(verdict.kind, "value", str(verdict.kind))
        payload["verdict"] = {
            "kind": kind,
            "reason": getattr(verdict, "reason", "") or "",
            "evidence": getattr(verdict, "evidence", "") or "",
            "confidence": float(getattr(verdict, "confidence", 0.0) or 0.0),
        }

    if payload["success"]:
        return payload

    payload["final_url"] = _as_str(getattr(r, "final_url", ""))
    payload["page_title"] = _as_str(getattr(r, "page_title", ""))

    # Honor a runner-stamped class if present; otherwise classify here
    # as a fallback. This keeps the schema self-describing even when
    # the StepResult bypassed the executor (host integrations, sim-env
    # halts, legacy resume paths).
    failure_class = _as_str(getattr(r, "failure_class", ""))
    if not failure_class:
        from .failure_class import classify
        failure_class = classify(payload["data"], payload["page_title"])
    payload["failure_class"] = failure_class or "unknown"

    from ..actions import Action

    last_action = getattr(r, "last_action", None)
    if isinstance(last_action, Action):
        at = last_action.action_type
        payload["last_action"] = {
            "type": getattr(at, "value", str(at)),
            "params": dict(last_action.params or {}),
            "reasoning": last_action.reasoning or "",
        }

    screenshot_png = getattr(r, "screenshot_png", None)
    if isinstance(screenshot_png, (bytes, bytearray)) and screenshot_png:
        payload["screenshot_b64"] = base64.b64encode(bytes(screenshot_png)).decode("ascii")

    return payload
