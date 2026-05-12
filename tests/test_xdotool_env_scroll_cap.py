"""Regression tests for #320 — SCROLL.amount must be capped.

The pre-#320 bug: ``GymRunner._maybe_redirect_repeated_top_click``
substituted ``scroll(direction='down', amount=350)`` (intent: 350 px,
contract: wheel notches). ``XdotoolGymEnv._execute_action`` looped that
many times calling ``self._xdotool("click", "5")`` — each subprocess
fork ~100 ms — so the substituted action hung the runner for ~40 s
before producing the next observation.

Two safety nets, both tested here:

1. The substituted amount itself is now ``5``, not ``350``.
2. The env caps SCROLL.amount at ``_MAX_SCROLL_NOTCHES`` so any future
   caller that passes a pixel-unit value is bounded to ~4 s instead of
   minutes.
"""

from __future__ import annotations

import pytest

from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.xdotool_env import XdotoolGymEnv, _MAX_SCROLL_NOTCHES


@pytest.fixture
def env_with_recorder(monkeypatch: pytest.MonkeyPatch) -> tuple[XdotoolGymEnv, list[tuple[str, ...]]]:
    """Build an env that records every ``_xdotool`` call instead of forking."""
    env = XdotoolGymEnv.__new__(XdotoolGymEnv)
    env._viewport = (1280, 800)
    env._human_speed = False
    env._env = {}

    calls: list[tuple[str, ...]] = []

    def _record(self: XdotoolGymEnv, *args: str) -> None:
        calls.append(args)

    monkeypatch.setattr(XdotoolGymEnv, "_xdotool", _record)
    return env, calls


def test_scroll_clamped_to_max_notches(env_with_recorder) -> None:
    """A pixel-unit caller (amount=350) must not produce 350 subprocesses."""
    env, calls = env_with_recorder
    env._execute_action(
        Action(ActionType.SCROLL, {"direction": "down", "amount": 350})
    )
    # mousemove + (capped) clicks
    click_calls = [c for c in calls if c and c[0] == "click"]
    assert len(click_calls) == _MAX_SCROLL_NOTCHES
    assert all(c[1] == "5" for c in click_calls)


def test_scroll_within_cap_passes_through(env_with_recorder) -> None:
    """A reasonable caller (amount=5) must still execute exactly 5 notches."""
    env, calls = env_with_recorder
    env._execute_action(
        Action(ActionType.SCROLL, {"direction": "down", "amount": 5})
    )
    click_calls = [c for c in calls if c and c[0] == "click"]
    assert len(click_calls) == 5


def test_scroll_up_uses_button_4(env_with_recorder) -> None:
    env, calls = env_with_recorder
    env._execute_action(
        Action(ActionType.SCROLL, {"direction": "up", "amount": 3})
    )
    click_calls = [c for c in calls if c and c[0] == "click"]
    assert len(click_calls) == 3
    assert all(c[1] == "4" for c in click_calls)


def test_scroll_negative_amount_is_zero(env_with_recorder) -> None:
    """Defensive: a negative or None amount shouldn't loop or raise."""
    env, calls = env_with_recorder
    env._execute_action(
        Action(ActionType.SCROLL, {"direction": "down", "amount": -10})
    )
    click_calls = [c for c in calls if c and c[0] == "click"]
    assert len(click_calls) == 0


def test_max_scroll_notches_constant_is_sane() -> None:
    """If someone bumps the cap above ~50, revisit whether the iteration
    cost is still acceptable (~100 ms per notch)."""
    assert 1 < _MAX_SCROLL_NOTCHES <= 50
