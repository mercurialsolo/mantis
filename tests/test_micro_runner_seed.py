"""``MicroPlanRunner(seed=...)`` reseeds the global ``random`` module so
``human_speed`` per-action delays (random.uniform / random.randint calls
in playwright_env, xdotool_env, step_handlers/*) are reproducible across
runs of the same plan.

The wiring is one line in the constructor; the contract is that *every*
``random`` consumer downstream observes the same draw sequence when two
runners are built with the same seed.
"""

from __future__ import annotations

import random

from mantis_agent.gym.micro_runner import MicroPlanRunner


def _build(seed):
    return MicroPlanRunner(
        brain=None,
        env=None,
        grounding=None,
        extractor=None,
        run_key="seed-test",
        session_name="seed-test",
        max_cost=999.0,
        max_time_minutes=999,
        seed=seed,
    )


def test_seed_makes_random_reproducible() -> None:
    _build(seed=42)
    draws_first = [random.random() for _ in range(8)]

    _build(seed=42)
    draws_second = [random.random() for _ in range(8)]

    assert draws_first == draws_second


def test_different_seeds_yield_different_draws() -> None:
    _build(seed=42)
    a = [random.random() for _ in range(8)]

    _build(seed=1337)
    b = [random.random() for _ in range(8)]

    assert a != b


def test_seed_none_does_not_touch_global_rng() -> None:
    """``seed=None`` preserves the previous non-deterministic behavior:
    two consecutive ``seed=None`` runners must NOT collapse to the same
    draws (state continues from wherever the process left it)."""
    random.seed(0)
    _ = random.random()  # advance state past a known starting point

    _build(seed=None)
    a = random.random()

    _build(seed=None)
    b = random.random()

    assert a != b
    assert isinstance(_build(seed=None).seed, type(None))


def test_seed_attribute_exposed() -> None:
    runner = _build(seed=7)
    assert runner.seed == 7
