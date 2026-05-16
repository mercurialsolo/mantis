"""Run-time accounting of what the self-healing framework did.

Records every framework-initiated correction (intent rewrite, success
demotion, handler escalation, critic-emitted step insertion) into a
unified per-run list. Surfaces in ``result.json`` as
``healing_events: list[dict]`` so operators can audit what the
framework changed and why — without spelunking Modal logs.

Producer side: helpers below are called from the scattered correction
sites (``intent_rewriter`` invocation, ``_maybe_demote_*``,
``_maybe_set_handler_override``, ``ExecutionCritic`` directives).
Consumer side: ``build_micro_result`` reads ``runner._healing_events``
and emits it in the result envelope.

Schema is permissive (each event is a free-form dict); the helpers
enforce a canonical shape:

* ``kind`` — one of ``rewrite`` / ``demotion`` / ``handler_escalation``
  / ``insert_step``
* ``step_index`` — int
* ``source`` — which subsystem emitted it (``intent_rewriter`` /
  ``agentic_recovery`` / ``demote_form`` / ``demote_click`` /
  ``handler_override`` / ``critic``)
* ``at`` — UTC ISO timestamp
* kind-specific extras (rewrite has ``from_intent`` / ``to_intent`` /
  ``failure_class``; demotion has ``step_type`` / ``reason``; etc.)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append(runner: Any, event: dict[str, Any]) -> None:
    """Best-effort append. Healing events are observability; they
    must never break a run."""
    try:
        log = getattr(runner, "_healing_events", None)
        if log is None:
            runner._healing_events = []
            log = runner._healing_events
        if isinstance(log, list):
            log.append(event)
    except Exception as exc:  # noqa: BLE001
        logger.debug("healing_events append failed: %s", exc)


def record_rewrite(
    runner: Any,
    *,
    step_index: int,
    from_intent: str,
    to_intent: str,
    source: str,
    failure_class: str = "",
) -> None:
    """An intent was rewritten before the next retry. ``source`` is
    ``intent_rewriter`` (Phase B) or ``agentic_recovery`` (older
    edit_step path) or ``critic`` (Phase C-emitted directive)."""
    _append(runner, {
        "kind": "rewrite",
        "step_index": int(step_index),
        "source": str(source),
        "from_intent": str(from_intent)[:200],
        "to_intent": str(to_intent)[:200],
        "failure_class": str(failure_class),
        "at": _now_iso(),
    })


def record_demotion(
    runner: Any,
    *,
    step_index: int,
    step_type: str,
    reason: str,
    source: str = "executor",
) -> None:
    """A handler-reported success was demoted to failure. ``source``
    is ``demote_form`` / ``demote_click`` / ``critic``."""
    _append(runner, {
        "kind": "demotion",
        "step_index": int(step_index),
        "source": str(source),
        "step_type": str(step_type),
        "reason": str(reason)[:200],
        "at": _now_iso(),
    })


def record_handler_escalation(
    runner: Any,
    *,
    step_index: int,
    from_handler: str,
    to_handler: str,
    trigger: str,
) -> None:
    """The step's handler was overridden after N same-class failures
    (the Phase A escalation: 2× no_state_change → Holo3StepHandler)."""
    _append(runner, {
        "kind": "handler_escalation",
        "step_index": int(step_index),
        "source": "handler_override",
        "from_handler": str(from_handler),
        "to_handler": str(to_handler),
        "trigger": str(trigger),
        "at": _now_iso(),
    })


def record_insert_step(
    runner: Any,
    *,
    after_step_index: int,
    inserted_intent: str,
    inserted_type: str,
    reason: str,
) -> None:
    """The ExecutionCritic emitted an ``insert_step`` directive — a
    new step was spliced into the plan to handle a runtime condition
    (obstacle dismissal, recovery navigate, …)."""
    _append(runner, {
        "kind": "insert_step",
        "step_index": int(after_step_index),
        "source": "critic",
        "inserted_intent": str(inserted_intent)[:200],
        "inserted_type": str(inserted_type),
        "reason": str(reason)[:200],
        "at": _now_iso(),
    })


def record_replace_step(
    runner: Any,
    *,
    step_index: int,
    original_type: str,
    new_intent: str,
    new_type: str,
    reason: str,
    source: str = "critic",
) -> None:
    """The critic (typically a frontier-model observer) replaced the
    step at ``step_index`` in place — same plan slot, different
    action. Distinct from ``insert_step`` (which splices a NEW step)
    and from the ``rewrite`` event (which only edits the intent
    prose, keeping the same step type / params).

    Common trigger: a ``wrong_target`` failure cascade where vision
    keeps clicking adjacent items. The frontier model proposes a
    structurally different action (e.g. direct navigate instead of
    sidebar link click) that bypasses the grounding miss.
    """
    _append(runner, {
        "kind": "replace_step",
        "step_index": int(step_index),
        "source": str(source),
        "original_type": str(original_type),
        "new_intent": str(new_intent)[:200],
        "new_type": str(new_type),
        "reason": str(reason)[:200],
        "at": _now_iso(),
    })


def snapshot(runner: Any) -> list[dict[str, Any]]:
    """Plain list copy for the result envelope. Empty list when no
    events recorded (preserves ``json.dumps`` compatibility)."""
    log = getattr(runner, "_healing_events", None)
    if not isinstance(log, list):
        return []
    return [dict(e) for e in log if isinstance(e, dict)]


__all__ = [
    "record_demotion",
    "record_handler_escalation",
    "record_insert_step",
    "record_rewrite",
    "snapshot",
]
