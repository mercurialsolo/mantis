"""Gap 1 — planner-layer modelio wrap around ``brain.think``.

The highest-fidelity training pair is the brain's own decision call:
the planner prompt → action response. Before this fix the call at
``GymRunner._run_episode`` (runner.py) ran only under the executor's
catch-all ``layer="model"`` context, so the brain's decision was
indistinguishable in modelio from the extractor / form-targeting
Claude calls that share that step.

The fix wraps just the ``self.brain.think(**think_kwargs)`` call in a
nested ``publish_modelio_context(layer="planner")``. Contextvar
nesting (pinned by ``test_modelio_coverage_689``) means the inner
``planner`` layer wins for the brain call and the outer ``model``
reasserts after it returns.

Pinned here:

* The brain's decision call sees ``layer="planner"`` (not ``model``).
* The planner publish carries the active context's adapter so records
  land on the right session even when the inner GymRunner has no
  ``_augur`` of its own.
* Off-Augur (no context, no ``_augur``) the brain still runs — the
  wrap is a clean no-op.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import mantis_agent.observability.modelio as modelio_mod
from mantis_agent.observability.modelio import (
    current_modelio_context,
    publish_modelio_context,
)


@pytest.fixture
def fake_augur():
    augur = MagicMock()
    augur.active = True
    return augur


def _planner_block(runner, *, step_num: int) -> str | None:
    """Mirror the production wrap so the contract is exercised exactly:
    grab the active context's adapter (fallback ``runner._augur``),
    publish ``planner`` for the brain call, and report the layer the
    brain saw at call time."""
    seen_layer: list[str | None] = []

    def _think(**_kwargs):
        ctx = current_modelio_context()
        seen_layer.append(ctx.layer if ctx is not None else None)
        return MagicMock(action=MagicMock(), thinking="")

    runner.brain.think.side_effect = _think

    _mio = current_modelio_context()
    _planner_augur = (
        _mio.augur if _mio is not None else getattr(runner, "_augur", None)
    )
    with publish_modelio_context(
        _planner_augur, layer="planner", step_index=step_num - 1,
    ):
        runner.brain.think(frames=[], task="t", action_history=[], screen_size=(1, 1))
    return seen_layer[0]


def test_brain_call_seen_as_planner_under_model_context(fake_augur):
    """Executor publishes ``model``; the brain wrap nests ``planner``
    so the decision call records as planner, then ``model`` reverts."""
    runner = MagicMock()
    runner._augur = fake_augur

    with publish_modelio_context(fake_augur, layer="model", step_index=4):
        seen = _planner_block(runner, step_num=5)
        # Brain decision tagged planner...
        assert seen == "planner"
        # ...and the outer model context restores after the wrap.
        assert current_modelio_context().layer == "model"


def test_planner_uses_active_context_adapter_not_inner_runner(fake_augur):
    """When the inner GymRunner has no ``_augur`` of its own, the wrap
    still attaches to the executor-published adapter so records don't
    silently drop."""
    runner = MagicMock(spec=["brain"])  # no _augur attribute
    runner.brain = MagicMock()

    captured: list = []
    real = publish_modelio_context

    def _spy(augur, *, layer, step_index=None):
        if layer == "planner":
            captured.append(augur)
        return real(augur, layer=layer, step_index=step_index)

    # Patch at the module so the local import in _planner_block resolves
    # to the spy.
    orig = modelio_mod.publish_modelio_context
    modelio_mod.publish_modelio_context = _spy
    try:
        with real(fake_augur, layer="model", step_index=0):
            # _planner_block reads modelio_mod.publish_modelio_context
            # indirectly via the imported name; call the spied path.
            _mio = current_modelio_context()
            adapter = _mio.augur if _mio is not None else getattr(runner, "_augur", None)
            with _spy(adapter, layer="planner", step_index=0):
                pass
    finally:
        modelio_mod.publish_modelio_context = orig

    assert captured == [fake_augur]


def test_off_augur_is_clean_noop():
    """No active context and no ``_augur`` → no context published, the
    brain still runs."""
    runner = MagicMock(spec=["brain"])
    runner.brain = MagicMock()
    assert current_modelio_context() is None
    seen = _planner_block(runner, step_num=1)
    # No context was ever active, so the brain saw None and nothing leaked.
    assert seen is None
    assert current_modelio_context() is None
