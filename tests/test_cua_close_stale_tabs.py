"""tab-drift guard: reset() closes accumulated stale tabs on a fresh navigation.

Running many tasks on one warm profile piles up tabs; current_url and the CDP
navigate/read target both pick "the first page tab", so stale tabs cause
wrong-page landings. _close_stale_tabs keeps the active tab and closes the rest.
"""

from __future__ import annotations

import io
import json


from mantis_agent.gym.xdotool_env import XdotoolGymEnv


def _install_cdp(monkeypatch, tabs, closed_ids):
    """Fake /json/list (returns `tabs`) + /json/close/<id> (records the id)."""
    def _fake_urlopen(url, timeout=None):  # noqa: ANN001
        if url.endswith("/json/list"):
            return io.BytesIO(json.dumps(tabs).encode())
        if "/json/close/" in url:
            closed_ids.append(url.rsplit("/", 1)[-1])
            return io.BytesIO(b"Target is closing")
        raise AssertionError(f"unexpected url {url}")
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)


def _env() -> XdotoolGymEnv:
    env = XdotoolGymEnv.__new__(XdotoolGymEnv)
    env._cdp_port = 9222
    return env


def test_closes_all_but_active_tab(monkeypatch) -> None:
    tabs = [
        {"type": "page", "id": "A", "url": "https://active"},
        {"type": "page", "id": "B", "url": "https://stale1"},
        {"type": "page", "id": "C", "url": "https://stale2"},
        {"type": "background_page", "id": "X"},  # non-page → ignored
    ]
    closed: list[str] = []
    _install_cdp(monkeypatch, tabs, closed)
    _env()._close_stale_tabs()
    assert closed == ["B", "C"]  # kept the first page tab, closed the rest


def test_noop_when_single_tab(monkeypatch) -> None:
    closed: list[str] = []
    _install_cdp(monkeypatch, [{"type": "page", "id": "A", "url": "https://x"}], closed)
    _env()._close_stale_tabs()
    assert closed == []


def test_best_effort_on_cdp_failure(monkeypatch) -> None:
    def _boom(url, timeout=None):  # noqa: ANN001
        raise ConnectionError("cdp down")
    monkeypatch.setattr("urllib.request.urlopen", _boom)
    _env()._close_stale_tabs()  # must not raise
