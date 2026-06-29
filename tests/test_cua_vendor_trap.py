"""cua-issues 2026-06-29 (S01) — browser-vendor "Reinstall Chrome" trap.

In run S01 a stray Return on Chrome's "Can't update Chrome → Reinstall
Chrome" bubble navigated to google.com/chrome and dead-ended behind the
proxy; the run then looped to max_steps and reported success. These tests
pin the predicate and the env-level navigate refusal that guard it.
"""

from __future__ import annotations

import pytest

from mantis_agent.gym.vendor_trap import is_browser_vendor_url
from mantis_agent.gym.xdotool_env import XdotoolGymEnv


@pytest.mark.parametrize(
    "url",
    [
        "https://www.google.com/chrome/",
        "https://www.google.com/chrome",
        "https://google.com/chrome/?brand=foo",
        "https://chrome.google.com/",
        "https://chrome.google.com/webstore",
        "HTTPS://WWW.GOOGLE.COM/Chrome/",  # case-insensitive
        "  https://www.google.com/chrome/  ",  # surrounding whitespace
    ],
)
def test_vendor_urls_flagged(url: str) -> None:
    assert is_browser_vendor_url(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "",
        None,
        "https://www.linkedin.com/feed/",
        "https://www.google.com/search?q=chrome",  # search, NOT the download page
        "https://www.google.com/",
        "https://mail.google.com/",
        "https://www.reddit.com/r/MachineLearning/",
        "not a url at all",
    ],
)
def test_non_vendor_urls_not_flagged(url) -> None:
    assert is_browser_vendor_url(url) is False


def test_navigate_refuses_vendor_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """_navigate_running_browser must NOT issue any CDP call for a vendor URL."""
    env = XdotoolGymEnv.__new__(XdotoolGymEnv)
    env._settle_time = 0.0

    cdp_calls: list[tuple[str, dict]] = []
    seeded: list[str] = []

    def _record_cdp(self, method, params=None, *, timeout: float = 3.0):
        cdp_calls.append((method, params or {}))
        return True, {}

    monkeypatch.setattr(XdotoolGymEnv, "_cdp_call", _record_cdp)
    monkeypatch.setattr(
        XdotoolGymEnv, "_seed_request_cookies",
        lambda self, url: seeded.append(url),
    )
    monkeypatch.setattr("mantis_agent.gym.xdotool_env.time.sleep", lambda *_: None)

    env._navigate_running_browser("https://www.google.com/chrome/")

    assert cdp_calls == []      # no Page.navigate attempted
    assert seeded == []         # bailed before cookie seeding


def test_navigate_allows_normal_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """A legitimate task URL still navigates via CDP Page.navigate."""
    env = XdotoolGymEnv.__new__(XdotoolGymEnv)
    env._settle_time = 0.0

    cdp_calls: list[tuple[str, dict]] = []

    def _record_cdp(self, method, params=None, *, timeout: float = 3.0):
        cdp_calls.append((method, params or {}))
        return True, {}

    monkeypatch.setattr(XdotoolGymEnv, "_cdp_call", _record_cdp)
    monkeypatch.setattr(XdotoolGymEnv, "_seed_request_cookies", lambda self, url: None)
    monkeypatch.setattr("mantis_agent.gym.xdotool_env.time.sleep", lambda *_: None)

    env._navigate_running_browser("https://www.linkedin.com/feed/")

    assert cdp_calls and cdp_calls[0][0] == "Page.navigate"
    assert cdp_calls[0][1]["url"] == "https://www.linkedin.com/feed/"
