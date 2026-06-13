"""GymRunner wiring of the speculative-think wrapper (#848).

The wrapper itself is covered by ``test_speculative_brain.py``. These
tests pin the GymRunner-side opt-in contract: when
``MANTIS_SPECULATIVE_THINK`` is unset, ``runner.brain`` is the bare
inner brain; when set, it's wrapped; and per-task ``reset()`` is called
at the start of every ``run()`` so the wrapper's hit-rate counters
start clean.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from mantis_agent.gym.runner import GymRunner
from mantis_agent.speculative_brain import SpeculativeBrain


def _bare_brain() -> Any:
    b = MagicMock()
    b.think = MagicMock()
    b.load = MagicMock()
    # The form_controller setup at the top of ``run()`` calls
    # ``brain.query(prompt)`` and parses the response as JSON. Return
    # an empty list so the parse short-circuits before the env stub
    # raises.
    b.query = MagicMock(return_value="[]")
    return b


def _env() -> Any:
    env = MagicMock()
    env.screen_size = (1920, 1080)
    return env


def test_default_wrapping_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the env flag, ``runner.brain`` is the original brain."""
    monkeypatch.delenv("MANTIS_SPECULATIVE_THINK", raising=False)
    brain = _bare_brain()
    runner = GymRunner(brain=brain, env=_env())
    assert runner.brain is brain
    assert not isinstance(runner.brain, SpeculativeBrain)


@pytest.mark.parametrize("flag", ["1", "true", "TRUE", "yes", "on"])
def test_flag_enables_wrapping(monkeypatch: pytest.MonkeyPatch, flag: str) -> None:
    monkeypatch.setenv("MANTIS_SPECULATIVE_THINK", flag)
    brain = _bare_brain()
    runner = GymRunner(brain=brain, env=_env())
    assert isinstance(runner.brain, SpeculativeBrain)
    assert runner.brain.inner is brain


def test_flag_falsey_keeps_bare(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANTIS_SPECULATIVE_THINK", "0")
    brain = _bare_brain()
    runner = GymRunner(brain=brain, env=_env())
    assert runner.brain is brain


def test_run_resets_speculative_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    """``run()`` must clear the wrapper's pending future and counters
    at the top so per-task hit-rates are clean. We don't drive a full
    episode — short-circuit by raising inside the first env call so
    the call site is exercised then bails fast.
    """
    monkeypatch.setenv("MANTIS_SPECULATIVE_THINK", "1")
    brain = _bare_brain()
    env = _env()
    # Make env.reset raise so we exit ``run`` early — that's enough to
    # reach the ``reset_brain`` call but skip the rest of the loop.
    env.reset.side_effect = RuntimeError("test stub: short-circuit")

    runner = GymRunner(brain=brain, env=env)
    # Pre-set counters as if a previous task accumulated stats.
    runner.brain.hits = 7
    runner.brain.misses = 3
    runner.brain.synchronous_starts = 1

    with pytest.raises(RuntimeError):
        runner.run(task="noop")

    # The reset() on SpeculativeBrain zeroes the three counters.
    assert runner.brain.hits == 0
    assert runner.brain.misses == 0
    assert runner.brain.synchronous_starts == 0


def test_run_reset_swallows_brain_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    """A misbehaving custom brain whose ``reset`` raises must not break
    the runner — the reset is best-effort observability hygiene.
    """
    monkeypatch.delenv("MANTIS_SPECULATIVE_THINK", raising=False)
    brain = _bare_brain()
    brain.reset = MagicMock(side_effect=RuntimeError("no!"))
    env = _env()
    env.reset.side_effect = RuntimeError("test stub: short-circuit")

    runner = GymRunner(brain=brain, env=env)
    # The reset call should be attempted but its exception swallowed —
    # we only see the env.reset RuntimeError, not the brain.reset one.
    with pytest.raises(RuntimeError, match="test stub"):
        runner.run(task="noop")
    brain.reset.assert_called_once()
