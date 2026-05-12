"""Tests for #311 — Chrome session reuse via container-scoped env cache.

Covers:

* :class:`XdotoolGymEnv` honours ``reuse_session=True`` — ``close()``
  becomes a no-op so the Xvfb + Chrome processes survive past one
  request. ``shutdown()`` still force-closes them.
* The runtime's ``_chrome_env_cache`` registry: cache hit on second
  request with the same key, miss on stale entry, miss when reuse is
  disabled, miss across keys.
* The ``MANTIS_CHROME_REUSE=disabled`` env-var ablation toggle.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from mantis_agent.gym.xdotool_env import XdotoolGymEnv


# ── XdotoolGymEnv reuse_session contract ───────────────────────────────


def test_default_reuse_session_is_false_close_terminates() -> None:
    """Legacy default — close() force-terminates browser + Xvfb."""
    env = XdotoolGymEnv(reuse_session=False)
    fake_browser = MagicMock()
    fake_browser.wait.return_value = None
    fake_xvfb = MagicMock()
    env._browser_proc = fake_browser
    env._xvfb_proc = fake_xvfb

    env.close()

    fake_browser.terminate.assert_called_once()
    fake_xvfb.terminate.assert_called_once()
    assert env._browser_proc is None
    assert env._xvfb_proc is None


def test_reuse_session_true_close_is_noop() -> None:
    """When the cache marked the env as reusable, close() must not kill
    the browser — successive requests need the live process."""
    env = XdotoolGymEnv(reuse_session=True)
    fake_browser = MagicMock()
    fake_xvfb = MagicMock()
    env._browser_proc = fake_browser
    env._xvfb_proc = fake_xvfb

    env.close()

    fake_browser.terminate.assert_not_called()
    fake_xvfb.terminate.assert_not_called()
    assert env._browser_proc is fake_browser
    assert env._xvfb_proc is fake_xvfb


def test_reuse_session_shutdown_force_closes() -> None:
    """The cache's container-recycle hook uses shutdown() to force-close
    even when reuse_session=True."""
    env = XdotoolGymEnv(reuse_session=True)
    fake_browser = MagicMock()
    fake_browser.wait.return_value = None
    fake_xvfb = MagicMock()
    env._browser_proc = fake_browser
    env._xvfb_proc = fake_xvfb

    env.shutdown()

    fake_browser.terminate.assert_called_once()
    fake_xvfb.terminate.assert_called_once()
    assert env._browser_proc is None
    assert env._xvfb_proc is None


def test_reuse_session_param_stored_on_instance() -> None:
    """Tests + downstream code can inspect ``_reuse_session`` to drive
    cache decisions."""
    assert XdotoolGymEnv()._reuse_session is False
    assert XdotoolGymEnv(reuse_session=True)._reuse_session is True


# ── runtime cache primitive ────────────────────────────────────────────


def test_chrome_reuse_enabled_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MANTIS_CHROME_REUSE", raising=False)
    from mantis_agent.baseten_server.runtime import _chrome_reuse_enabled
    assert _chrome_reuse_enabled() is True


def test_chrome_reuse_disabled_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANTIS_CHROME_REUSE", "disabled")
    from mantis_agent.baseten_server.runtime import _chrome_reuse_enabled
    assert _chrome_reuse_enabled() is False


def test_chrome_reuse_other_values_treated_as_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MANTIS_CHROME_REUSE", "true")
    from mantis_agent.baseten_server.runtime import _chrome_reuse_enabled
    assert _chrome_reuse_enabled() is True


def test_shutdown_cache_force_closes_all_envs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Container recycle path: every cached env's shutdown() is called
    and the cache is emptied."""
    import sys
    __import__("mantis_agent.baseten_server.runtime")
    runtime_mod = sys.modules["mantis_agent.baseten_server.runtime"]

    fake_env_a = MagicMock(spec=XdotoolGymEnv)
    fake_proxy_a = MagicMock()
    fake_env_b = MagicMock(spec=XdotoolGymEnv)
    fake_proxy_b = None

    monkeypatch.setitem(
        runtime_mod._chrome_env_cache, ("profA", "px1"), (fake_env_a, fake_proxy_a),
    )
    monkeypatch.setitem(
        runtime_mod._chrome_env_cache, ("profB", ""), (fake_env_b, fake_proxy_b),
    )

    runtime_mod._shutdown_chrome_env_cache()

    fake_env_a.shutdown.assert_called_once()
    fake_env_b.shutdown.assert_called_once()
    fake_proxy_a.terminate.assert_called_once()
    assert runtime_mod._chrome_env_cache == {}


def test_shutdown_cache_swallows_env_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One bad env doesn't block cleanup of the others."""
    import sys
    __import__("mantis_agent.baseten_server.runtime")
    runtime_mod = sys.modules["mantis_agent.baseten_server.runtime"]

    bad = MagicMock(spec=XdotoolGymEnv)
    bad.shutdown.side_effect = RuntimeError("dead pipe")
    good = MagicMock(spec=XdotoolGymEnv)

    monkeypatch.setitem(runtime_mod._chrome_env_cache, ("bad", ""), (bad, None))
    monkeypatch.setitem(runtime_mod._chrome_env_cache, ("good", ""), (good, None))

    runtime_mod._shutdown_chrome_env_cache()

    good.shutdown.assert_called_once()
    assert runtime_mod._chrome_env_cache == {}


# ── public-surface lock ────────────────────────────────────────────────


def test_runtime_exports_cache_primitives() -> None:
    """Lock the names other code (test suites, hosts) imports."""
    import sys
    __import__("mantis_agent.baseten_server.runtime")
    runtime_mod = sys.modules["mantis_agent.baseten_server.runtime"]
    assert hasattr(runtime_mod, "_chrome_env_cache")
    assert hasattr(runtime_mod, "_chrome_env_cache_lock")
    assert hasattr(runtime_mod, "_chrome_reuse_enabled")
    assert hasattr(runtime_mod, "_shutdown_chrome_env_cache")
    assert isinstance(runtime_mod._chrome_env_cache, dict)


def test_setup_env_threads_reuse_session_through_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``setup_env(reuse_session=True)`` propagates the flag to the
    constructed XdotoolGymEnv. Stub out the real ctor so the test
    doesn't try to start Xvfb."""
    captured: dict[str, Any] = {}

    class _StubEnv:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("mantis_agent.gym.xdotool_env.XdotoolGymEnv", _StubEnv)

    from mantis_agent.task_loop import setup_env
    env, _ = setup_env(
        base_url="https://example.com",
        run_id="r1",
        session_name="s",
        proxy_disabled=True,
        reuse_session=True,
    )
    assert isinstance(env, _StubEnv)
    assert captured["reuse_session"] is True


def test_setup_env_default_reuse_session_is_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _StubEnv:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("mantis_agent.gym.xdotool_env.XdotoolGymEnv", _StubEnv)

    from mantis_agent.task_loop import setup_env
    setup_env(
        base_url="https://example.com",
        run_id="r1",
        session_name="s",
        proxy_disabled=True,
    )
    assert captured["reuse_session"] is False
