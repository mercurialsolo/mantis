"""Smoke tests for the Brain protocol + registry."""

from __future__ import annotations

import pytest

from mantis_agent.brain_protocol import (
    Brain,
    list_brains,
    register_brain,
    resolve_brain,
)


class _Stub:
    def load(self) -> None:
        self.loaded = True

    def think(self, frames, task, action_history=None, screen_size=(1920, 1080)):
        return {"action": None, "raw_output": ""}


def test_register_and_resolve_roundtrip(monkeypatch):
    # Snapshot + restore the registry to keep tests isolated.
    from mantis_agent import brain_protocol as bp

    monkeypatch.setattr(bp, "_REGISTRY", {})

    register_brain("stub", lambda: _Stub())
    assert "stub" in list_brains()

    brain = resolve_brain("stub")
    assert isinstance(brain, Brain)  # runtime_checkable Protocol

    brain.load()
    assert getattr(brain, "loaded", False) is True


def test_resolve_unknown_raises_with_available_list(monkeypatch):
    from mantis_agent import brain_protocol as bp

    monkeypatch.setattr(bp, "_REGISTRY", {})
    register_brain("a", lambda: _Stub())
    register_brain("b", lambda: _Stub())

    with pytest.raises(KeyError) as excinfo:
        resolve_brain("missing")
    msg = str(excinfo.value)
    assert "missing" in msg
    assert "a" in msg and "b" in msg


def test_register_rejects_empty_name(monkeypatch):
    from mantis_agent import brain_protocol as bp

    monkeypatch.setattr(bp, "_REGISTRY", {})
    with pytest.raises(ValueError):
        register_brain("", lambda: _Stub())
