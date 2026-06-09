"""Tests for the navigate URL-drift gate (#835).

When a navigate step lands successfully against a literal URL, the
runner stamps a hint (hostname + first-path-segment substring) on
``runner._post_navigate_url_hint``. Before the *next* step's
handler runs, the runner compares ``env.current_url`` against the
hint and short-circuits with ``failure_class="navigation_drift"``
when they diverge.

Pure-function helpers tested directly here; the dispatcher integration
is tested via the existing handler + run-executor test surface.
"""

from __future__ import annotations

import pytest

from mantis_agent.gym.navigate_gate import (
    check_drift,
    derive_expected_substring,
    is_enabled,
)


# ── is_enabled env gating ─────────────────────────────────────────


def test_default_enabled(monkeypatch):
    monkeypatch.delenv("MANTIS_AUTO_URL_GATE", raising=False)
    assert is_enabled() is True


@pytest.mark.parametrize("v", ["0", "false", "no", "off", "FALSE"])
def test_falsy_disables(monkeypatch, v):
    monkeypatch.setenv("MANTIS_AUTO_URL_GATE", v)
    assert is_enabled() is False


@pytest.mark.parametrize("v", ["1", "true", "anything"])
def test_truthy_enables(monkeypatch, v):
    monkeypatch.setenv("MANTIS_AUTO_URL_GATE", v)
    assert is_enabled() is True


# ── derive_expected_substring ─────────────────────────────────────


@pytest.mark.parametrize("url,expected", [
    ("https://news.ycombinator.com/newest", "news.ycombinator.com/newest"),
    ("https://news.ycombinator.com/", "news.ycombinator.com"),
    ("https://news.ycombinator.com", "news.ycombinator.com"),
    ("https://github.com/owner/repo/issues", "github.com/owner"),
    ("https://example.com/path", "example.com/path"),
    ("http://example.com", "example.com"),
    ("https://Example.COM/Foo", "example.com/Foo"),  # host lowercased
])
def test_derive_substring_canonical(url, expected):
    assert derive_expected_substring(url) == expected


@pytest.mark.parametrize("url", ["", None, "not-a-url", "/just/a/path"])
def test_derive_substring_empty_on_garbage(url):
    assert derive_expected_substring(url or "") == ""


# ── check_drift ───────────────────────────────────────────────────


def test_match_returns_empty():
    """No drift when the actual URL contains the expected substring."""
    assert check_drift(
        "https://news.ycombinator.com/newest?next=12345",
        "news.ycombinator.com/newest",
    ) == ""


def test_mismatch_returns_reason():
    reason = check_drift(
        "https://news.ycombinator.com/login?next=...",
        "news.ycombinator.com/newest",
    )
    assert reason
    assert "expected=" in reason
    assert "got=" in reason


def test_fragment_change_still_matches():
    """A pure fragment change after a navigate is not drift."""
    assert check_drift(
        "https://example.com/foo#section",
        "example.com/foo",
    ) == ""


def test_case_insensitive():
    """Host case differences shouldn't trigger drift — the helper
    expects lowercased expected_substring; URLs lowercased on compare."""
    assert check_drift(
        "https://NEWS.YCOMBINATOR.COM/newest",
        "news.ycombinator.com/newest",
    ) == ""


def test_empty_expected_returns_empty():
    """Nothing to gate on → no drift, even if actual is suspicious."""
    assert check_drift(
        "https://example.com/anything",
        "",
    ) == ""


def test_empty_actual_returns_empty():
    """Env hasn't reported a URL yet → don't fail on absence."""
    assert check_drift("", "example.com/path") == ""


# ── HN drift scenario from user feedback ──────────────────────────


def test_hn_newest_to_login_drift_detected():
    """The exact failure shape from user feedback: navigate to
    /newest succeeds, but the proxy / CF / something redirects to
    /login. The next step should not run vision against /login."""
    expected = derive_expected_substring("https://news.ycombinator.com/newest")
    drift = check_drift(
        "https://news.ycombinator.com/login?next=/newest",
        expected,
    )
    assert drift
    assert "newest" in drift  # the expected substring is in the reason


def test_hn_newest_to_top_story_open_detected():
    """Decomposer-emitted click on the first story landed on the
    story URL — different host entirely. Drift detected."""
    expected = derive_expected_substring("https://news.ycombinator.com/newest")
    drift = check_drift("https://anthropic.com/news/claude-fable-5", expected)
    assert drift
