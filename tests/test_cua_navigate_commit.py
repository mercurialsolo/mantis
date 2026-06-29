"""cua-issues 2026-06-29 — navigate-repaint / stale-session guard.

~14 /v1/cua runs issued a CDP Page.navigate (to Reddit/jobs/login) yet the
cached tab kept rendering linkedin.com/in/akhil08 — the screen never moved.
Page.navigate returning ok means the command was accepted, not that the
page reached the target. These pin the new commit-verification helpers.
"""

from __future__ import annotations

from mantis_agent.gym.xdotool_env import XdotoolGymEnv


def _env():
    env = XdotoolGymEnv.__new__(XdotoolGymEnv)
    env._settle_time = 0.0
    return env


# ── _url_host ───────────────────────────────────────────────────────────


def test_url_host_strips_www():
    assert XdotoolGymEnv._url_host("https://www.reddit.com/r/x") == "reddit.com"


def test_url_host_plain():
    assert XdotoolGymEnv._url_host("https://linkedin.com/in/akhil08") == "linkedin.com"


def test_url_host_empty_and_garbage():
    assert XdotoolGymEnv._url_host("") == ""
    assert XdotoolGymEnv._url_host("not a url") == ""


# ── _await_navigation_commit ────────────────────────────────────────────


def test_commit_true_when_host_matches(monkeypatch):
    env = _env()
    monkeypatch.setattr(
        type(env), "current_url",
        property(lambda self: "https://www.reddit.com/r/programming"),
    )
    assert env._await_navigation_commit("reddit.com", max_seconds=1.0) is True


def test_commit_false_when_tab_stays_stale(monkeypatch):
    """The bug: navigated to reddit but the live URL is still LinkedIn."""
    env = _env()
    monkeypatch.setattr(
        type(env), "current_url",
        property(lambda self: "https://linkedin.com/in/akhil08"),
    )
    assert env._await_navigation_commit("reddit.com", max_seconds=0.4) is False


def test_commit_true_when_target_host_empty(monkeypatch):
    # Can't verify (no host parsed) → don't block; preserve old behaviour.
    env = _env()
    assert env._await_navigation_commit("", max_seconds=0.0) is True
