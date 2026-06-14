"""Gap 2 — bridge runner logs into the Augur bundle's ``logs/`` panel.

:meth:`AugurAdapter.append_log` (augur.py:1107) is a clean no-op when
the session is inactive, but nothing ever called it — so every bundle
shipped ``logs:false`` and any diagnostics rule that reads the run log
(e.g. the ``no_state_change`` SoM root-cause) had nothing to read.

Two complementary feeds, wired together by this module:

1. :class:`AugurLogHandler` — a ``logging.Handler`` attached to the
   ``mantis_agent`` logger tree for the run's duration. Forwards
   ``WARNING``+ records to ``append_log``, so the diagnostics already
   emitted by ``step_recovery`` / ``critic`` / the runner (recovery
   fire, demotions, gate misses) land in the bundle without touching
   each call site. This is the "breadth" feed.

2. :func:`log_step_start` / :func:`log_step_outcome` — structured
   per-step lifecycle lines the executor emits explicitly at step
   start and step completion. This is the "structured detail" feed.

Reentrancy + loop safety:

* ``append_log`` and the adapter both ``logger.warning`` on failure.
  Without a guard, a failing POST would log a warning that re-enters
  the handler and recurses. A thread-local flag short-circuits the
  nested emit.
* Records emitted by the observability package itself
  (``augur`` / ``log_bridge`` / ``modelio``) are skipped outright —
  they're adapter-internal bookkeeping the ``append_log`` docstring
  explicitly says not to double-emit.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

#: Root of the logger tree the bridge attaches to.
_MANTIS_LOGGER = "mantis_agent"

#: Records from these loggers are adapter-internal — never forward them
#: (and never risk a feedback loop through the augur adapter's own
#: failure warnings).
_SKIP_LOGGER_PREFIXES = (
    "mantis_agent.observability.augur",
    "mantis_agent.observability.log_bridge",
    "mantis_agent.observability.modelio",
)


class AugurLogHandler(logging.Handler):
    """Forward ``WARNING``+ records to an open :class:`AugurAdapter`.

    Best-effort and self-isolating: any failure in ``emit`` is
    swallowed so logging never breaks a run, and a thread-local guard
    prevents the adapter's own failure-warnings from recursing back
    into the handler.
    """

    def __init__(self, augur: Any, *, level: int = logging.WARNING) -> None:
        super().__init__(level=level)
        self._augur = augur
        self._guard = threading.local()

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        if record.name.startswith(_SKIP_LOGGER_PREFIXES):
            return
        if getattr(self._guard, "active", False):
            return
        self._guard.active = True
        try:
            msg = self.format(record)
            # Per-step routing when the caller stamped an index on the
            # record (``logger.warning(..., extra={"augur_step_index": i})``);
            # otherwise the line lands in the run-level log.
            step_index = getattr(record, "augur_step_index", None)
            self._augur.append_log(msg, step_index=step_index, name="run")
        except Exception:  # noqa: BLE001 — logging must never break a run
            pass
        finally:
            self._guard.active = False


def attach_augur_log_bridge(
    runner: Any, augur: Any, *, level: int = logging.WARNING,
) -> AugurLogHandler | None:
    """Attach an :class:`AugurLogHandler` to the mantis logger tree and
    stash it on ``runner._augur_log_handler``.

    No-op (returns ``None``) when ``augur`` is None / inactive or a
    handler is already attached to this runner — so a resume or a
    double-open can't stack handlers.
    """
    if augur is None or not getattr(augur, "active", False):
        return None
    if getattr(runner, "_augur_log_handler", None) is not None:
        return getattr(runner, "_augur_log_handler")
    handler = AugurLogHandler(augur, level=level)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logging.getLogger(_MANTIS_LOGGER).addHandler(handler)
    runner._augur_log_handler = handler
    return handler


def detach_augur_log_bridge(runner: Any) -> None:
    """Remove + close the runner's :class:`AugurLogHandler`. Idempotent.

    Must run *before* ``augur.close()`` so the handler never appends to
    a finalized session.
    """
    handler = getattr(runner, "_augur_log_handler", None)
    if handler is None:
        return
    try:
        logging.getLogger(_MANTIS_LOGGER).removeHandler(handler)
        handler.close()
    except Exception as exc:  # noqa: BLE001 — never break cleanup
        logger.debug("detach_augur_log_bridge failed: %s", exc)
    finally:
        runner._augur_log_handler = None


# ── Strategic per-step lifecycle lines (the structured feed) ─────────


def log_step_start(augur: Any, step: Any, step_index: int) -> None:
    """Emit a ``step <i>: start`` line carrying the intent + target."""
    if augur is None or not getattr(augur, "active", False):
        return
    step_type = str(getattr(step, "type", "") or "")
    intent = str(getattr(step, "intent", "") or "")
    target = str(getattr(step, "target", "") or "")
    line = f"step {step_index}: start [{step_type}] {intent} {target}".rstrip()
    try:
        augur.append_log(line[:1000], step_index=step_index, name="run")
    except Exception:  # noqa: BLE001
        pass


def log_step_outcome(augur: Any, step_result: Any, step_index: int) -> None:
    """Emit a ``step <i>: ok|FAILED`` line. On failure the line carries
    ``failure_class`` + ``failure_subclass`` + the typed verdict reason
    + the runner's ``data`` note — the signal a ``no_state_change``
    diagnostics rule needs to fire off the bundle log."""
    if augur is None or not getattr(augur, "active", False):
        return
    data = str(getattr(step_result, "data", "") or "")
    if getattr(step_result, "success", False):
        line = f"step {step_index}: ok — {data}".rstrip()
    else:
        fc = str(getattr(step_result, "failure_class", "") or "unknown")
        sub = str(getattr(step_result, "failure_subclass", "") or "")
        verdict = getattr(step_result, "verdict", None)
        reason = str(getattr(verdict, "reason", "") or "") if verdict is not None else ""
        cls = f"{fc}/{sub}" if sub else fc
        line = (
            f"step {step_index}: FAILED [{cls}] verdict_reason={reason or '-'} — {data}"
        ).rstrip()
    try:
        augur.append_log(line[:1000], step_index=step_index, name="run")
    except Exception:  # noqa: BLE001
        pass
