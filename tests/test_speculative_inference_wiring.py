"""Tests for #118 — SpeculativeBrain wiring into BasetenCUARuntime.load().

Covers:

* Default behaviour wraps the inner brain in SpeculativeBrain.
* ``MANTIS_SPECULATIVE_INFERENCE=disabled`` keeps the bare brain (ablation
  toggle).
* The wrapping is idempotent across the brain protocol — load() / think()
  signatures pass through.

The end-to-end speculation logic itself (hit / miss / cancel) is already
covered by ``tests/test_speculative_brain.py`` from the original #118
step 2 ship. This file only pins down the runtime wiring.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest


def _runtime_module() -> Any:
    """Resolve the actual submodule (not the singleton instance that
    ``mantis_agent/baseten_server/__init__.py`` re-exports under the same
    name)."""
    __import__("mantis_agent.baseten_server.runtime")
    return sys.modules["mantis_agent.baseten_server.runtime"]


def _fresh_runtime() -> Any:
    """Construct an isolated runtime per test so module-level state from
    sibling tests doesn't leak in."""
    return _runtime_module().BasetenCUARuntime()


class _StubBrain:
    """Bare-minimum brain implementing the protocol the wiring expects."""

    def __init__(self) -> None:
        self.load_calls = 0

    def load(self) -> None:
        self.load_calls += 1

    def think(self, **kwargs: Any) -> Any:
        return type("R", (), {"action": None, "thinking": ""})()


def test_default_keeps_bare_brain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Default config leaves the inner brain bare — E2E ablation showed
    speculation regresses wall time on single-llama.cpp deployments
    (GPU contention between speculative + sync requests). Opt-in only."""
    monkeypatch.delenv("MANTIS_SPECULATIVE_INFERENCE", raising=False)
    monkeypatch.setattr(
        _runtime_module(), "_load_secret_environment", lambda: None,
    )
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))

    runtime = _fresh_runtime()
    runtime.model_kind = "holo3"
    inner = _StubBrain()
    runtime._load_holo3 = lambda: inner  # type: ignore[method-assign]

    runtime.load()
    assert runtime.brain is inner


def test_opt_in_wraps_brain_in_speculative_brain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """MANTIS_SPECULATIVE_INFERENCE=enabled wraps the inner brain."""
    monkeypatch.setenv("MANTIS_SPECULATIVE_INFERENCE", "enabled")
    monkeypatch.setattr(
        _runtime_module(), "_load_secret_environment", lambda: None,
    )
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))

    runtime = _fresh_runtime()
    runtime.model_kind = "holo3"
    inner = _StubBrain()
    runtime._load_holo3 = lambda: inner  # type: ignore[method-assign]

    runtime.load()

    from mantis_agent.speculative_brain import SpeculativeBrain
    assert isinstance(runtime.brain, SpeculativeBrain)
    assert runtime.brain.inner is inner
    assert runtime.loaded is True


def test_unknown_toggle_value_treated_as_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Any value other than the explicit 'enabled' keeps speculation off
    (safe default for the wall-time regression on single-GPU backends)."""
    monkeypatch.setenv("MANTIS_SPECULATIVE_INFERENCE", "yes")
    monkeypatch.setattr(
        _runtime_module(), "_load_secret_environment", lambda: None,
    )
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))

    runtime = _fresh_runtime()
    runtime.model_kind = "holo3"
    inner = _StubBrain()
    runtime._load_holo3 = lambda: inner  # type: ignore[method-assign]

    runtime.load()
    assert runtime.brain is inner


def test_speculative_brain_resets_counters_per_episode() -> None:
    """The runtime calls brain.reset() at the top of each /v1/cua run so
    per-run counters surface cleanly. Verify the reset hook is intact on
    SpeculativeBrain."""
    from mantis_agent.speculative_brain import SpeculativeBrain
    sb = SpeculativeBrain(_StubBrain())
    sb.hits = 7
    sb.misses = 2
    sb.synchronous_starts = 1
    sb.reset()
    assert sb.hits == 0
    assert sb.misses == 0
    assert sb.synchronous_starts == 0


def test_speculative_brain_hit_rate_safe_on_empty() -> None:
    """hit_rate() never divides by zero when no calls have happened."""
    from mantis_agent.speculative_brain import SpeculativeBrain
    sb = SpeculativeBrain(_StubBrain())
    assert sb.hit_rate() == 0.0


def test_speculative_brain_hit_rate_computation() -> None:
    from mantis_agent.speculative_brain import SpeculativeBrain
    sb = SpeculativeBrain(_StubBrain())
    sb.hits = 6
    sb.misses = 3
    sb.synchronous_starts = 1
    # 6 / 10 = 0.6
    assert sb.hit_rate() == pytest.approx(0.6)
