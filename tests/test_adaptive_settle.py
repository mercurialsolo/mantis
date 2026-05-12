"""Tests for #294 — adaptive settle helper.

Deterministic via injected ``capture``, ``sleep_fn``, and ``time_fn`` so the
tests don't rely on wall clock or hash-of-real-pixels behaviour.
"""

from __future__ import annotations

from typing import Callable

import pytest
from PIL import Image

from mantis_agent.gym.adaptive_settle import (
    is_enabled,
    wait_until_stable,
)


def _fake_clock(start: float = 0.0) -> tuple[Callable[[], float], Callable[[float], None]]:
    """Returns (time_fn, sleep_fn) sharing a mutable clock cell."""
    cell = [start]

    def time_fn() -> float:
        return cell[0]

    def sleep_fn(secs: float) -> None:
        cell[0] += secs

    return time_fn, sleep_fn


def _img(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (32, 32), color=color)


def _noisy_img(seed: int) -> Image.Image:
    """Image whose pHash actually differs per seed.

    ``phash_64`` is a dHash over a brightness gradient — solid-colour
    frames all share the same hash regardless of colour. Inject structured
    asymmetry by drawing a seed-dependent shape so each frame's hash is
    distinct.
    """
    import random
    rng = random.Random(seed)
    img = Image.new("RGB", (32, 32), color=(0, 0, 0))
    pixels = img.load()
    for _ in range(64):
        x, y = rng.randrange(32), rng.randrange(32)
        pixels[x, y] = (255, 255, 255)
    return img


# ── is_enabled toggle ──────────────────────────────────────────────────


def test_is_enabled_defaults_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MANTIS_ADAPTIVE_SETTLE", raising=False)
    assert is_enabled() is True


def test_is_enabled_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANTIS_ADAPTIVE_SETTLE", "disabled")
    assert is_enabled() is False


def test_is_enabled_unknown_value_treated_as_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any value other than the explicit 'disabled' keeps the gate on."""
    monkeypatch.setenv("MANTIS_ADAPTIVE_SETTLE", "yes")
    assert is_enabled() is True


# ── wait_until_stable: early-exit on stable frames ─────────────────────


def test_returns_early_when_frames_match_immediately() -> None:
    """Two consecutive identical captures = stable → return ~0 elapsed."""
    time_fn, sleep_fn = _fake_clock()
    stable = _img((128, 128, 128))
    captures = iter([stable, stable, stable])

    elapsed = wait_until_stable(
        capture=lambda: next(captures),
        max_seconds=3.0,
        poll_interval=0.1,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
    )
    # First capture sets baseline; second match returns. With 0.1 poll and
    # 1 inter-poll sleep, elapsed should be at most 0.2.
    assert elapsed <= 0.3


def test_caps_at_max_seconds_when_frames_keep_changing() -> None:
    """Frames change every poll → no stability → cap reached."""
    time_fn, sleep_fn = _fake_clock()
    counter = [0]

    def capture() -> Image.Image:
        counter[0] += 1
        return _noisy_img(counter[0])

    elapsed = wait_until_stable(
        capture=capture,
        max_seconds=1.0,
        poll_interval=0.1,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
    )
    assert elapsed >= 1.0
    assert elapsed < 1.3  # should not overshoot meaningfully


def test_zero_max_seconds_returns_immediately() -> None:
    time_fn, sleep_fn = _fake_clock()
    assert wait_until_stable(
        capture=lambda: _img((0, 0, 0)),
        max_seconds=0.0,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
    ) == 0.0


# ── wait_until_stable: missing-frame handling ──────────────────────────


def test_none_capture_skipped_does_not_declare_stable() -> None:
    """A capture returning None must not be hashed; stability is never
    declared on missing signal."""
    time_fn, sleep_fn = _fake_clock()
    # Return None forever — should hit the cap.
    elapsed = wait_until_stable(
        capture=lambda: None,
        max_seconds=0.5,
        poll_interval=0.1,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
    )
    assert elapsed >= 0.5


def test_capture_exception_treated_as_missing() -> None:
    time_fn, sleep_fn = _fake_clock()

    def boom() -> Image.Image:
        raise RuntimeError("camera died")

    elapsed = wait_until_stable(
        capture=boom,
        max_seconds=0.5,
        poll_interval=0.1,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
    )
    assert elapsed >= 0.5


# ── wait_until_stable: min_seconds floor ───────────────────────────────


def test_min_seconds_enforced_even_when_already_stable() -> None:
    """Callers can require at least one paint cycle before measuring."""
    time_fn, sleep_fn = _fake_clock()
    stable = _img((42, 42, 42))
    elapsed = wait_until_stable(
        capture=lambda: stable,
        max_seconds=2.0,
        poll_interval=0.05,
        min_seconds=0.3,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
    )
    assert elapsed >= 0.3


def test_min_seconds_clamped_to_max() -> None:
    """min_seconds > max_seconds is invalid; clamp rather than block."""
    time_fn, sleep_fn = _fake_clock()
    stable = _img((10, 10, 10))
    elapsed = wait_until_stable(
        capture=lambda: stable,
        max_seconds=0.2,
        poll_interval=0.05,
        min_seconds=1.0,  # exceeds max
        sleep_fn=sleep_fn,
        time_fn=time_fn,
    )
    assert elapsed <= 0.4


# ── wait_until_stable: require_consecutive parameter ───────────────────


def test_require_consecutive_higher_value() -> None:
    """Need N matches in a row — with N=3, two-then-different fails."""
    time_fn, sleep_fn = _fake_clock()
    a = _noisy_img(1)
    b = _noisy_img(2)
    captures = iter([a, a, b, a, a, a])

    elapsed = wait_until_stable(
        capture=lambda: next(captures, a),
        max_seconds=5.0,
        poll_interval=0.1,
        require_consecutive=3,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
    )
    # Eventually stabilises on `a`; should be well under cap.
    assert elapsed < 5.0


# ── Public API surface ─────────────────────────────────────────────────


def test_module_exports() -> None:
    """Locks the public surface other modules import."""
    from mantis_agent.gym import adaptive_settle as m
    assert callable(m.wait_until_stable)
    assert callable(m.wait_for_networkidle)
    assert callable(m.settle_after_action)
    assert callable(m.is_enabled)


# ── settle_after_action: step-handler shorthand ────────────────────────


class _StableEnv:
    """Env stub whose ``_screenshot`` always returns the same image."""

    def __init__(self) -> None:
        self._frame = _img((50, 50, 50))

    def _screenshot(self) -> Image.Image:
        return self._frame


class _NoCaptureEnv:
    """Env stub with no screenshot attribute — exercises defensive fallback."""


def test_settle_after_action_returns_quickly_on_stable_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MANTIS_ADAPTIVE_SETTLE", raising=False)
    from mantis_agent.gym.adaptive_settle import settle_after_action

    elapsed = settle_after_action(_StableEnv(), max_seconds=3.0)
    # Two consecutive captures of the constant frame match immediately,
    # so we land well under the 3s cap.
    assert elapsed < 1.0


def test_settle_after_action_falls_back_to_sleep_without_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env without ``_screenshot`` or ``screenshot`` must still settle —
    just with the fixed sleep, never skip."""
    monkeypatch.delenv("MANTIS_ADAPTIVE_SETTLE", raising=False)
    from mantis_agent.gym import adaptive_settle

    sleeps: list[float] = []
    monkeypatch.setattr(adaptive_settle.time, "sleep", lambda s: sleeps.append(s))

    elapsed = adaptive_settle.settle_after_action(
        _NoCaptureEnv(), max_seconds=1.5,
    )
    assert sleeps == [1.5]
    assert elapsed == 1.5


def test_settle_after_action_respects_ablation_toggle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MANTIS_ADAPTIVE_SETTLE", "disabled")
    from mantis_agent.gym import adaptive_settle

    sleeps: list[float] = []
    monkeypatch.setattr(adaptive_settle.time, "sleep", lambda s: sleeps.append(s))

    adaptive_settle.settle_after_action(_StableEnv(), max_seconds=2.0)
    # Ablation: fall through to the fixed sleep, ignore env entirely.
    assert sleeps == [2.0]


def test_settle_after_action_handles_zero_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MANTIS_ADAPTIVE_SETTLE", raising=False)
    from mantis_agent.gym.adaptive_settle import settle_after_action

    assert settle_after_action(_StableEnv(), max_seconds=0.0) == 0.0
    assert settle_after_action(_StableEnv(), max_seconds=-1.0) == 0.0
