"""#689 — runner-level ``publish_modelio_context`` default.

Coverage of ``session.record_modelio()`` was sparse (1.3% on the
staffai tenant) because most LLM call sites didn't have an outer
``publish_modelio_context`` wrapping them. Step handlers and helpers
that already published contexts (form-targeting → grounding,
step_recovery → step_recovery, runner verify → verifier) were the
exceptions; the bulk of model-decision LLM calls (extractor,
brain's action call) fell through with no context published, so the
client-side capture hook in ``_anthropic/client._record_modelio_if_active``
short-circuited on ``current_modelio_context() is None``.

The fix: ``RunExecutor._dispatch_step`` now wraps every
``_execute_step`` invocation in ``publish_modelio_context(layer="model",
step_index=...)`` as the DEFAULT layer for the step's duration. Step
handlers that have a more specific layer (form-targeting →
``grounding``, verifier path → ``verifier``) override via contextvar
nesting; the outer ``model`` reverts on the way out.

Contract pinned here:

* ``_dispatch_step`` publishes ``model`` modelio context around
  ``_execute_step``.
* The publish is a no-op when ``runner._augur`` is None / inactive
  (Augur disabled / SDK missing).
* Inner-scope overrides win — a step handler that publishes
  ``grounding`` sees ``grounding`` for its inner Anthropic calls; the
  outer ``model`` reasserts after the handler returns.
* Exception in the step body still resets the contextvar (the
  context manager's ``finally`` honors the contract).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import mantis_agent.observability.modelio as modelio_mod
from mantis_agent.observability.modelio import (
    current_modelio_context,
    publish_modelio_context,
)


@pytest.fixture
def fake_augur():
    """Mock AugurAdapter that the helper accepts (active=True)."""
    augur = MagicMock()
    augur.active = True
    return augur


# ── publish_modelio_context nesting semantics ──────────────────────


def test_publish_default_then_inner_override_then_revert(fake_augur):
    """Outer ``model`` + inner ``grounding`` → reads ``grounding``
    inside the inner block, reverts to ``model`` after."""
    with publish_modelio_context(fake_augur, layer="model", step_index=2):
        ctx = current_modelio_context()
        assert ctx is not None
        assert ctx.layer == "model"
        assert ctx.step_index == 2
        with publish_modelio_context(fake_augur, layer="grounding", step_index=2):
            inner = current_modelio_context()
            assert inner.layer == "grounding"
        # After inner exits, outer context restores.
        again = current_modelio_context()
        assert again.layer == "model"


def test_publish_resets_on_exception(fake_augur):
    """Exception inside the ``with`` block still resets the
    contextvar — no leak between steps."""
    assert current_modelio_context() is None
    try:
        with publish_modelio_context(fake_augur, layer="model", step_index=0):
            raise RuntimeError("step blew up")
    except RuntimeError:
        pass
    # Contextvar must have reset even though the body raised.
    assert current_modelio_context() is None


def test_publish_unknown_layer_logs_and_skips(monkeypatch, caplog, fake_augur):
    """Unknown layer string → WARN logged, yield without setting
    context. Catches typos in call sites at dev time."""
    import logging
    caplog.set_level(logging.WARNING, logger=modelio_mod.logger.name)
    with publish_modelio_context(fake_augur, layer="not-a-layer", step_index=0):
        # No context published since layer was rejected.
        assert current_modelio_context() is None
    assert any("unknown layer" in r.message for r in caplog.records)


def test_publish_inactive_augur_yields_without_context():
    """``augur=None`` or inactive → yield without setting context
    (telemetry never breaks the run)."""
    with publish_modelio_context(None, layer="model", step_index=0):
        assert current_modelio_context() is None
    inactive = MagicMock()
    inactive.active = False
    with publish_modelio_context(inactive, layer="model", step_index=0):
        assert current_modelio_context() is None


# ── RunExecutor wraps _execute_step with layer="model" ─────────────


def test_dispatch_step_publishes_model_context(fake_augur):
    """``RunExecutor._dispatch_step`` wraps ``_execute_step`` in a
    ``publish_modelio_context(layer="model")`` block so any deep
    Anthropic call inside lands as ``model`` modelio.

    Asserted by patching ``publish_modelio_context`` at the import
    site (run_executor.py imports it locally inside the function)
    and confirming it's invoked with the expected layer + step_index.
    """
    # Import sites in run_executor: ``from ..observability.modelio
    # import publish_modelio_context`` is local-in-function, so the
    # patch target is ``modelio_mod.publish_modelio_context`` (the
    # source-of-truth module) since the local rebind picks it up at
    # call time, not module load.
    seen: list[tuple[Any, str, int | None]] = []

    real_publish = publish_modelio_context

    def _spy(augur, *, layer, step_index=None):
        seen.append((augur, layer, step_index))
        return real_publish(augur, layer=layer, step_index=step_index)

    # Build a minimal RunExecutor enough to exercise the publish path.
    from mantis_agent.gym.run_executor import RunExecutor, RunState

    runner = MagicMock()
    runner._augur = fake_augur
    runner._active_checkpoint_context = None
    runner.pending_form_labels = []
    runner._latest_preview_result = None
    runner.env = MagicMock()
    runner._last_known_url = ""
    runner._pre_step_snapshot = None
    runner.time_meter = None
    # ``_execute_step`` returns a stub StepResult.
    fake_result = MagicMock()
    fake_result.screenshot_png = b"png"
    fake_result.success = True
    runner._execute_step = MagicMock(return_value=fake_result)

    plan = MagicMock()
    plan.steps = []
    state = RunState(checkpoint=MagicMock(), results=[], step_index=2,
                     loop_counters={}, listings_on_page=set())
    step = MagicMock()
    step.type = "click"
    step.verify = ""

    executor = RunExecutor.__new__(RunExecutor)
    executor.parent = runner
    # Patch dependencies that _dispatch_step calls.
    with patch.object(modelio_mod, "publish_modelio_context", side_effect=_spy), \
         patch("mantis_agent.gym.run_executor._pending_form_labels", return_value=[]), \
         patch("mantis_agent.gym.run_executor.reset_grounding_trace_stashes"), \
         patch("mantis_agent.gym.run_executor._maybe_run_reversibility_gate"), \
         patch("mantis_agent.gym.run_executor.step_snapshot.capture", return_value=MagicMock()), \
         patch("mantis_agent.gym.run_executor._read_env_url", return_value=""):
        # Suppress the inner time_meter.publish_dispatch since it's
        # not the contract under test.
        import mantis_agent.gym.time_meter as _tm
        with patch.object(_tm, "publish_dispatch", side_effect=publish_modelio_context):
            executor._dispatch_step(plan, state, step)

    # At least one call with layer="model" and step_index=2
    model_calls = [
        (a, layer, s) for (a, layer, s) in seen if layer == "model"
    ]
    assert any(s == 2 for (_, _, s) in model_calls), (
        f"Expected a model-layer publish at step_index=2 in {seen!r}"
    )


def test_dispatch_step_passes_runner_augur_to_publish(fake_augur):
    """The publish call forwards ``runner._augur`` so the contextvar
    knows which adapter to attach LLM records to. Critical because
    fan-out workers each have their own AugurAdapter — wrong augur =
    records land on the wrong session."""
    seen: list[Any] = []

    def _spy(augur, *, layer, step_index=None):
        seen.append(augur)
        return publish_modelio_context(augur, layer=layer, step_index=step_index)

    from mantis_agent.gym.run_executor import RunExecutor, RunState

    runner = MagicMock()
    runner._augur = fake_augur
    runner._active_checkpoint_context = None
    runner.pending_form_labels = []
    runner._latest_preview_result = None
    runner.env = MagicMock()
    runner._last_known_url = ""
    runner._pre_step_snapshot = None
    runner.time_meter = None
    fake_result = MagicMock()
    fake_result.screenshot_png = b"png"
    fake_result.success = True
    runner._execute_step = MagicMock(return_value=fake_result)

    plan = MagicMock()
    plan.steps = []
    state = RunState(checkpoint=MagicMock(), results=[], step_index=0,
                     loop_counters={}, listings_on_page=set())
    step = MagicMock()
    step.type = "click"
    step.verify = ""

    executor = RunExecutor.__new__(RunExecutor)
    executor.parent = runner
    with patch.object(modelio_mod, "publish_modelio_context", side_effect=_spy), \
         patch("mantis_agent.gym.run_executor._pending_form_labels", return_value=[]), \
         patch("mantis_agent.gym.run_executor.reset_grounding_trace_stashes"), \
         patch("mantis_agent.gym.run_executor._maybe_run_reversibility_gate"), \
         patch("mantis_agent.gym.run_executor.step_snapshot.capture", return_value=MagicMock()), \
         patch("mantis_agent.gym.run_executor._read_env_url", return_value=""):
        import mantis_agent.gym.time_meter as _tm
        with patch.object(_tm, "publish_dispatch", side_effect=publish_modelio_context):
            executor._dispatch_step(plan, state, step)

    assert fake_augur in seen, (
        f"Expected runner._augur ({fake_augur!r}) to be passed to "
        f"publish_modelio_context; saw {seen!r}"
    )


# Imports needed for the spy assertions above to read 'Any' type-tag.
from typing import Any  # noqa: E402
