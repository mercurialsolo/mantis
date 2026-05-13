"""``_wait_for_cf_challenge_clear`` polls until the Cloudflare interstitial
clears, replacing the old hardcoded 15s sleep in ``run_plan``.

Loaded by path because ``deploy/modal/modal_plan_runner.py`` isn't a
package and imports ``modal`` at top level. We skip when ``modal`` is
not installed (same pattern as ``test_oxylabs_username_format.py``).
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip(
    "modal",
    reason="modal package not installed; deploy module can't be loaded",
)


def _load_module():
    path = Path(__file__).parent.parent / "deploy" / "modal" / "modal_plan_runner.py"
    spec = importlib.util.spec_from_file_location("_test_modal_plan_runner_cf", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_test_modal_plan_runner_cf"] = mod
    spec.loader.exec_module(mod)
    return mod


def _env_with_cdp_sequence(values):
    """Build a fake env whose cdp_evaluate yields ``values`` in order.

    Once the sequence is exhausted, the last value repeats forever.
    """
    iterator = iter(values)
    last = [values[-1]] if values else [False]

    def _eval(_expr):
        try:
            v = next(iterator)
        except StopIteration:
            v = last[0]
        last[0] = v
        return v

    env = MagicMock()
    env.cdp_evaluate.side_effect = _eval
    return env


def test_returns_quickly_when_no_cf_marker(monkeypatch) -> None:
    """CF marker absent on the first poll → returns after at most
    ``min_seconds`` of floor + one poll."""
    mod = _load_module()
    env = _env_with_cdp_sequence([False])

    start = time.monotonic()
    elapsed = mod._wait_for_cf_challenge_clear(
        env, max_seconds=10.0, poll_interval=0.05, min_seconds=0.01,
    )
    wall = time.monotonic() - start

    assert elapsed <= 1.0
    assert wall <= 1.0
    env.cdp_evaluate.assert_called()


def test_caps_at_max_seconds_when_challenge_never_clears(monkeypatch) -> None:
    """CF marker stays present → poll loop is bounded by ``max_seconds``,
    not by an infinite wait."""
    mod = _load_module()
    env = _env_with_cdp_sequence([True])

    start = time.monotonic()
    elapsed = mod._wait_for_cf_challenge_clear(
        env, max_seconds=0.5, poll_interval=0.05, min_seconds=0.0,
    )
    wall = time.monotonic() - start

    assert 0.4 <= elapsed <= 1.2
    assert wall <= 1.5


def test_returns_when_challenge_clears_mid_poll() -> None:
    """CF marker present on first 3 polls, then clears → return shortly
    after the clear, well before ``max_seconds``."""
    mod = _load_module()
    env = _env_with_cdp_sequence([True, True, True, False])

    elapsed = mod._wait_for_cf_challenge_clear(
        env, max_seconds=5.0, poll_interval=0.05, min_seconds=0.0,
    )

    assert elapsed < 1.0
    assert env.cdp_evaluate.call_count >= 4


def test_treats_none_as_keep_polling() -> None:
    """``cdp_evaluate`` returning ``None`` (CDP unreachable, JS threw,
    page not ready) must NOT be interpreted as "challenge cleared" —
    keep polling until a real ``False`` or the cap."""
    mod = _load_module()
    env = _env_with_cdp_sequence([None, None, False])

    elapsed = mod._wait_for_cf_challenge_clear(
        env, max_seconds=5.0, poll_interval=0.05, min_seconds=0.0,
    )

    assert elapsed < 1.0
    assert env.cdp_evaluate.call_count >= 3


def test_env_override_for_max_seconds(monkeypatch) -> None:
    """``MANTIS_CF_PREWARM_MAX_SECONDS`` overrides the 45s default."""
    mod = _load_module()
    monkeypatch.setenv("MANTIS_CF_PREWARM_MAX_SECONDS", "0.3")
    env = _env_with_cdp_sequence([True])

    elapsed = mod._wait_for_cf_challenge_clear(
        env, poll_interval=0.05, min_seconds=0.0,
    )

    assert elapsed <= 1.0


def test_exception_from_cdp_treated_as_unknown() -> None:
    """``cdp_evaluate`` raising must not crash the poll — treat as
    "not cleared" and keep going."""
    mod = _load_module()
    env = MagicMock()
    calls = {"n": 0}

    def _eval(_expr):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("CDP unreachable")
        return False

    env.cdp_evaluate.side_effect = _eval

    elapsed = mod._wait_for_cf_challenge_clear(
        env, max_seconds=5.0, poll_interval=0.05, min_seconds=0.0,
    )

    assert elapsed < 1.0
    assert calls["n"] >= 3
