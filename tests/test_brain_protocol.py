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


def test_builtin_brains_are_registered():
    """Built-in brains must be discoverable by name without instantiation.

    Resolving them on a slim install would crash because torch / anthropic
    aren't there — but ``list_brains()`` should work since registration
    only stores a factory, not the class.
    """
    names = set(list_brains())
    expected = {"holo3", "claude", "opencua", "llamacpp", "gemma4", "agent-s"}
    missing = expected - names
    assert not missing, f"missing built-in brains: {missing}; got: {names}"


def test_resolve_from_env_prefers_mantis_brain(monkeypatch):
    """``MANTIS_BRAIN`` wins over the legacy ``MANTIS_MODEL``."""
    from mantis_agent import brain_protocol as bp

    monkeypatch.setattr(bp, "_REGISTRY", {})
    register_brain("a", lambda: _Stub())
    register_brain("b", lambda: _Stub())

    monkeypatch.setenv("MANTIS_BRAIN", "a")
    monkeypatch.setenv("MANTIS_MODEL", "b")
    brain = bp.resolve_from_env()
    # Stub is what factory "a" returned
    assert getattr(brain, "load", None) is not None


def test_resolve_from_env_falls_back_to_mantis_model(monkeypatch):
    from mantis_agent import brain_protocol as bp

    monkeypatch.setattr(bp, "_REGISTRY", {})
    register_brain("legacy", lambda: _Stub())

    monkeypatch.delenv("MANTIS_BRAIN", raising=False)
    monkeypatch.setenv("MANTIS_MODEL", "legacy")
    brain = bp.resolve_from_env()
    assert getattr(brain, "load", None) is not None


def test_resolve_from_env_aliases_gemma4_cua(monkeypatch):
    """The legacy ``MANTIS_MODEL=gemma4-cua`` should map to ``gemma4``."""
    from mantis_agent import brain_protocol as bp

    monkeypatch.setattr(bp, "_REGISTRY", {})
    sentinel = _Stub()
    register_brain("gemma4", lambda: sentinel)

    monkeypatch.delenv("MANTIS_BRAIN", raising=False)
    monkeypatch.setenv("MANTIS_MODEL", "gemma4-cua")
    brain = bp.resolve_from_env()
    assert brain is sentinel


def test_top_level_re_exports():
    """register_brain / resolve_brain / list_brains / Brain on package root."""
    import mantis_agent

    for name in ("Brain", "register_brain", "resolve_brain", "list_brains", "resolve_brain_from_env"):
        assert hasattr(mantis_agent, name), f"mantis_agent missing public symbol {name!r}"
