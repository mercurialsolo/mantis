"""Tests for #582 — CDP tab-count diff in the click handler.

Before this fix, ``_opened_detail_in_new_tab`` was set only by the
middle-click fallback (``step_handlers/click.py:526``). When a plain
click triggered a site-side ``window.open()`` (boattrader card cards,
modifier-clicks bypassing middle-click, etc.) a new tab opened but
the runner never knew → next ``navigate_back`` dispatched ``Alt+Left``
against the wrong tab → ``navigate_back_recovered`` halt-reason cycle.

The fix: snapshot ``cdp_count_pages()`` before + after the click; if
count went up AND we landed on a detail page, set the flag.
"""

from __future__ import annotations

import json
from unittest.mock import patch
from urllib.error import URLError

from mantis_agent.gym.xdotool_env import XdotoolGymEnv


def _make_env() -> XdotoolGymEnv:
    """Construct an env shell with just the attrs ``cdp_count_pages`` reads."""
    env = XdotoolGymEnv.__new__(XdotoolGymEnv)
    env._cdp_port = 9222
    return env


def _mock_json_list(payload):
    """Wrap a payload as a urlopen-compatible response context manager."""
    class _Resp:
        def __init__(self, body): self._body = body
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read(self): return self._body.encode()
    return _Resp(json.dumps(payload))


# ── cdp_count_pages helper ─────────────────────────────────────────


def test_counts_page_type_tabs() -> None:
    env = _make_env()
    payload = [
        {"type": "page", "url": "https://x.com/a"},
        {"type": "page", "url": "https://y.com/b"},
        {"type": "worker", "url": "https://x.com/sw.js"},  # excluded
        {"type": "background_page", "url": "chrome-extension://abc/"},  # excluded
    ]
    with patch("urllib.request.urlopen", return_value=_mock_json_list(payload)):
        assert env.cdp_count_pages() == 2


def test_excludes_chrome_internal_pages() -> None:
    env = _make_env()
    payload = [
        {"type": "page", "url": "https://x.com/a"},
        {"type": "page", "url": "chrome://newtab/"},  # excluded
        {"type": "page", "url": "about:blank"},  # excluded
    ]
    with patch("urllib.request.urlopen", return_value=_mock_json_list(payload)):
        assert env.cdp_count_pages() == 1


def test_excludes_empty_url_pages() -> None:
    env = _make_env()
    payload = [
        {"type": "page", "url": "https://x.com/a"},
        {"type": "page", "url": ""},
        {"type": "page"},  # url key missing entirely
    ]
    with patch("urllib.request.urlopen", return_value=_mock_json_list(payload)):
        assert env.cdp_count_pages() == 1


def test_returns_zero_on_cdp_unreachable() -> None:
    env = _make_env()
    with patch("urllib.request.urlopen", side_effect=URLError("connection refused")):
        # Failure must not raise — caller treats 0 as "couldn't check".
        assert env.cdp_count_pages() == 0


def test_returns_zero_on_json_decode_failure() -> None:
    env = _make_env()

    class _BadResp:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read(self): return b"not-json{{"

    with patch("urllib.request.urlopen", return_value=_BadResp()):
        assert env.cdp_count_pages() == 0


def test_returns_zero_when_payload_not_list() -> None:
    env = _make_env()
    with patch("urllib.request.urlopen", return_value=_mock_json_list({"err": "x"})):
        assert env.cdp_count_pages() == 0


def test_handles_non_dict_entries_gracefully() -> None:
    env = _make_env()
    payload = [
        {"type": "page", "url": "https://x.com/a"},
        "garbage",
        None,
        42,
    ]
    with patch("urllib.request.urlopen", return_value=_mock_json_list(payload)):
        assert env.cdp_count_pages() == 1


def test_no_tabs_returns_zero() -> None:
    env = _make_env()
    with patch("urllib.request.urlopen", return_value=_mock_json_list([])):
        assert env.cdp_count_pages() == 0
