"""Tests for the MockBrain plan-authoring stub (#274).

The mock brain is meant to be cheap (no GPU, no API), deterministic
(every think() returns DONE), and resolvable through the public
brain-registry surface so plan authors can flip ``MANTIS_BRAIN=mock``
and run their plans against a real :class:`MicroPlanRunner` without
paying for inference.

These tests pin the contract:

- ``MockBrain()`` constructs without side effects.
- ``load()`` is a no-op (safe to call repeatedly).
- ``think()`` always returns a DONE action.
- ``MockBrain`` satisfies the runtime-checked :class:`Brain` protocol.
- The registry resolves ``MANTIS_BRAIN=mock`` and
  ``MANTIS_MODEL=mock`` to the same factory.
- Importing the module pulls no GPU / network deps.
"""

from __future__ import annotations

import sys

import pytest

from mantis_agent.actions import ActionType
from mantis_agent.brain_mock import MockBrain
from mantis_agent.brain_protocol import (
    Brain,
    list_brains,
    resolve_brain,
    resolve_from_env,
)


def test_mock_brain_constructs_cheaply() -> None:
    brain = MockBrain()
    assert brain.model == "mock"
    assert brain.model_name == "mock"
    # Counter starts at zero — every think() bumps it for trace visibility.
    assert brain._think_count == 0


def test_mock_brain_load_is_noop() -> None:
    brain = MockBrain()
    brain.load()
    brain.load()  # idempotent
    assert brain._think_count == 0


def test_mock_brain_think_returns_done() -> None:
    brain = MockBrain()
    brain.load()
    result = brain.think(task="click the next listing")
    assert result.action.action_type is ActionType.DONE
    assert result.action.params == {}
    assert "mock brain" in result.action.reasoning
    assert "task='click the next listing'" in result.raw_output


def test_mock_brain_think_count_increments() -> None:
    brain = MockBrain()
    brain.think(task="t1")
    brain.think(task="t2")
    brain.think(task="t3")
    assert brain._think_count == 3


def test_mock_brain_accepts_full_kwarg_surface() -> None:
    """Match the Brain protocol — runner passes frames + history + screen_size."""
    brain = MockBrain()
    result = brain.think(
        frames=[],
        task="anything",
        action_history=[],
        screen_size=(1280, 720),
    )
    assert result.action.action_type is ActionType.DONE


def test_mock_brain_satisfies_brain_protocol() -> None:
    assert isinstance(MockBrain(), Brain)


# ── Registry integration ──────────────────────────────────────────────────


def test_mock_brain_registered_under_mock_name() -> None:
    assert "mock" in list_brains()


def test_resolve_brain_mock_returns_mock_instance() -> None:
    brain = resolve_brain("mock")
    assert isinstance(brain, MockBrain)


def test_resolve_from_env_picks_mock(monkeypatch) -> None:
    monkeypatch.setenv("MANTIS_BRAIN", "mock")
    monkeypatch.delenv("MANTIS_MODEL", raising=False)
    brain = resolve_from_env()
    assert isinstance(brain, MockBrain)


def test_resolve_from_env_legacy_model_var(monkeypatch) -> None:
    """``MANTIS_MODEL`` is the legacy alias; both should map to mock."""
    monkeypatch.delenv("MANTIS_BRAIN", raising=False)
    monkeypatch.setenv("MANTIS_MODEL", "mock")
    brain = resolve_from_env()
    assert isinstance(brain, MockBrain)


# ── Import-cost guard ─────────────────────────────────────────────────────


def test_mock_module_has_no_heavyweight_imports() -> None:
    """Pulling brain_mock should not transitively load torch / transformers
    / requests / anthropic. The slim install ships without those — if the
    mock brain accidentally references one, ``MANTIS_BRAIN=mock`` fails
    inside ``[orchestrator]`` deployments and the mock loses its point.

    We can't fully sandbox the import in a single process (other modules
    may have already loaded those packages), so the test only fails if
    a *fresh* import of brain_mock would pull them in. We check by
    re-importing the module under sys.modules introspection.
    """
    # Drop the mock module from sys.modules so reimporting it surfaces any
    # top-level imports it introduces.
    sys.modules.pop("mantis_agent.brain_mock", None)
    forbidden = {"torch", "transformers", "anthropic", "requests"}
    before = {m for m in forbidden if m in sys.modules}
    import mantis_agent.brain_mock  # noqa: F401 — side-effect: imports
    after = {m for m in forbidden if m in sys.modules}
    introduced = after - before
    assert not introduced, (
        f"brain_mock imported forbidden heavy deps: {sorted(introduced)}"
    )


# ── Friendly error when an unknown brain is selected ──────────────────────


def test_unknown_brain_name_lists_mock_in_the_available_list() -> None:
    """The mock brain only helps plan authors if they can discover it.
    Resolving an unknown name should error with a list that includes
    'mock' so the user sees it as an option."""
    with pytest.raises(KeyError) as ei:
        resolve_brain("does-not-exist")
    assert "mock" in str(ei.value)
