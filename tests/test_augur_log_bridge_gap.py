"""Gap 2 — bridge runner logs into the Augur bundle's ``logs/`` panel.

Before this fix nothing called ``append_log`` so every bundle shipped
``logs:false`` and diagnostics rules reading the run log had nothing
to read. This pins both feeds:

* :class:`AugurLogHandler` forwards ``WARNING``+ records, skips the
  observability package's own records (no feedback loop), and is
  isolating (a failing ``append_log`` never raises through logging).
* attach / detach are idempotent and gated on adapter activity.
* :func:`log_step_start` / :func:`log_step_outcome` emit structured
  per-step lines; the failure line carries failure_class + verdict
  reason.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from mantis_agent.observability.log_bridge import (
    AugurLogHandler,
    attach_augur_log_bridge,
    detach_augur_log_bridge,
    log_step_outcome,
    log_step_start,
)


@pytest.fixture
def fake_augur():
    augur = MagicMock()
    augur.active = True
    augur.append_log = MagicMock()
    return augur


@pytest.fixture
def runner():
    r = MagicMock()
    # Start with no handler attached.
    if hasattr(r, "_augur_log_handler"):
        del r._augur_log_handler
    r._augur_log_handler = None
    return r


# ── AugurLogHandler.emit ────────────────────────────────────────────


def test_handler_forwards_warning_to_append_log(fake_augur):
    handler = AugurLogHandler(fake_augur)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    rec = logging.LogRecord(
        "mantis_agent.gym.step_recovery", logging.WARNING, __file__, 1,
        "recovery fired: navigate-back", None, None,
    )
    handler.emit(rec)
    fake_augur.append_log.assert_called_once()
    sent = fake_augur.append_log.call_args.args[0]
    assert "recovery fired" in sent
    assert "WARNING" in sent


def test_handler_skips_observability_own_records(fake_augur):
    """Records from the augur / modelio / log_bridge loggers are never
    forwarded — they'd loop and double-emit adapter bookkeeping."""
    handler = AugurLogHandler(fake_augur)
    handler.setFormatter(logging.Formatter("%(message)s"))
    for name in (
        "mantis_agent.observability.augur",
        "mantis_agent.observability.modelio",
        "mantis_agent.observability.log_bridge",
    ):
        handler.emit(logging.LogRecord(name, logging.WARNING, __file__, 1, "x", None, None))
    fake_augur.append_log.assert_not_called()


def test_handler_emit_never_raises_through_logging(fake_augur):
    """A failing append_log must be swallowed — logging cannot break
    the run, and the failure-warning must not recurse."""
    fake_augur.append_log.side_effect = RuntimeError("POST failed")
    handler = AugurLogHandler(fake_augur)
    handler.setFormatter(logging.Formatter("%(message)s"))
    rec = logging.LogRecord(
        "mantis_agent.gym.runner", logging.ERROR, __file__, 1, "boom", None, None,
    )
    # Must not raise.
    handler.emit(rec)


def test_handler_routes_step_index_from_record_extra(fake_augur):
    handler = AugurLogHandler(fake_augur)
    handler.setFormatter(logging.Formatter("%(message)s"))
    rec = logging.LogRecord(
        "mantis_agent.gym.runner", logging.WARNING, __file__, 1, "x", None, None,
    )
    rec.augur_step_index = 4
    handler.emit(rec)
    assert fake_augur.append_log.call_args.kwargs["step_index"] == 4


# ── attach / detach lifecycle ───────────────────────────────────────


def test_attach_is_noop_when_inactive(runner):
    inactive = MagicMock()
    inactive.active = False
    assert attach_augur_log_bridge(runner, inactive) is None
    assert runner._augur_log_handler is None


def test_attach_then_detach_round_trip(runner, fake_augur):
    tree = logging.getLogger("mantis_agent")
    before = list(tree.handlers)
    handler = attach_augur_log_bridge(runner, fake_augur)
    assert handler is not None
    assert handler in tree.handlers
    assert runner._augur_log_handler is handler

    detach_augur_log_bridge(runner)
    assert runner._augur_log_handler is None
    assert handler not in tree.handlers
    assert list(tree.handlers) == before


def test_attach_is_idempotent(runner, fake_augur):
    """A double-open (resume) must not stack handlers."""
    tree = logging.getLogger("mantis_agent")
    h1 = attach_augur_log_bridge(runner, fake_augur)
    h2 = attach_augur_log_bridge(runner, fake_augur)
    assert h1 is h2
    assert tree.handlers.count(h1) == 1
    detach_augur_log_bridge(runner)


def test_detach_is_idempotent(runner):
    # No handler attached — detach must be a clean no-op.
    detach_augur_log_bridge(runner)
    assert runner._augur_log_handler is None


def test_end_to_end_warning_reaches_append_log(runner, fake_augur):
    """A real logger.warning under the mantis tree reaches append_log
    while attached, and stops after detach."""
    attach_augur_log_bridge(runner, fake_augur)
    logging.getLogger("mantis_agent.gym.critic").warning("critic demoted target")
    assert fake_augur.append_log.call_count == 1

    detach_augur_log_bridge(runner)
    logging.getLogger("mantis_agent.gym.critic").warning("after detach")
    assert fake_augur.append_log.call_count == 1  # unchanged


# ── strategic per-step lines ────────────────────────────────────────


def test_log_step_start_emits_intent_and_target(fake_augur):
    step = MagicMock()
    step.type = "click"
    step.intent = "Click Sign In"
    step.target = "button.login"
    log_step_start(fake_augur, step, 2)
    line = fake_augur.append_log.call_args.args[0]
    assert "step 2: start [click]" in line
    assert "Click Sign In" in line
    assert fake_augur.append_log.call_args.kwargs["step_index"] == 2


def test_log_step_outcome_failure_carries_class_and_verdict(fake_augur):
    sr = MagicMock()
    sr.success = False
    sr.failure_class = "no_state_change"
    sr.failure_subclass = ""
    sr.data = "click ok, page unchanged"
    sr.verdict = MagicMock(reason="no_state_change")
    log_step_outcome(fake_augur, sr, 5)
    line = fake_augur.append_log.call_args.args[0]
    assert "step 5: FAILED [no_state_change]" in line
    assert "verdict_reason=no_state_change" in line
    assert "page unchanged" in line


def test_log_step_outcome_success_is_compact(fake_augur):
    sr = MagicMock()
    sr.success = True
    sr.data = "extracted 7 leads"
    log_step_outcome(fake_augur, sr, 1)
    line = fake_augur.append_log.call_args.args[0]
    assert line.startswith("step 1: ok")
    assert "extracted 7 leads" in line


def test_strategic_calls_noop_when_inactive():
    inactive = MagicMock()
    inactive.active = False
    inactive.append_log = MagicMock()
    log_step_start(inactive, MagicMock(), 0)
    log_step_outcome(inactive, MagicMock(success=True, data=""), 0)
    inactive.append_log.assert_not_called()
