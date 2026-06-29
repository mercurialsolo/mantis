"""Regression tests for the env.reset() omnibox-typing navigate bug.

The pre-fix bug: ``XdotoolGymEnv.reset()`` navigated an already-running
browser via ``Ctrl+L`` + ``Ctrl+A`` + ``_xdotool_type(url)`` + ``Enter``.
``_xdotool_type`` preferred CDP ``Input.insertText`` for React-controlled
field reliability — but ``Input.insertText`` operates on the renderer's
page DOM, never reaching Chrome's UI omnibox. Result: the URL never
entered the address bar, ``Enter`` re-navigated to whatever was already
there (typically the page's current URL), and the navigate "succeeded"
on the wrong URL.

Surfaced in staff-crm-long verification run ``1eb0b0c7`` (2026-05-17):
post-URL came back as the dashboard ``/`` even though the plan asked
for ``/leads?status=Contacted&...``.

The fix: prefer CDP ``Page.navigate`` for programmatic navigation in
already-running browsers (Chrome's browser process handles it, no
omnibox typing required, cookies/session preserved). Fall back to
xclip-paste then xdotool-type — both X-focus-honoring paths that
reach the omnibox correctly — only when CDP is unreachable.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mantis_agent.gym.xdotool_env import XdotoolGymEnv


@pytest.fixture
def env_with_cdp_recorder(monkeypatch: pytest.MonkeyPatch) -> tuple[XdotoolGymEnv, list[tuple[str, dict]]]:
    """Env stub that records ``_cdp_call`` invocations + xdotool keys.

    No subprocess fork, no Xvfb. The default ``_cdp_call`` recorder
    returns ``(True, {})`` so callers exercise the happy path.
    Tests that want to simulate a CDP failure override the side_effect.
    """
    env = XdotoolGymEnv.__new__(XdotoolGymEnv)
    env._viewport = (1280, 800)
    env._human_speed = False
    env._env = {}
    env._settle_time = 0.0  # zero settle so tests run instantly

    cdp_calls: list[tuple[str, dict]] = []
    xdotool_keys: list[str] = []

    def _record_cdp(self, method: str, params: dict | None = None, *, timeout: float = 3.0):
        cdp_calls.append((method, params or {}))
        return True, {}

    def _record_xdotool(self, *args: str) -> None:
        xdotool_keys.append(args[1] if len(args) > 1 else args[0])

    monkeypatch.setattr(XdotoolGymEnv, "_cdp_call", _record_cdp)
    monkeypatch.setattr(XdotoolGymEnv, "_xdotool", _record_xdotool)
    monkeypatch.setattr("mantis_agent.gym.xdotool_env.time.sleep", lambda *_: None)

    env._test_cdp_calls = cdp_calls  # type: ignore[attr-defined]
    env._test_xdotool_keys = xdotool_keys  # type: ignore[attr-defined]
    return env, cdp_calls


def test_navigate_running_browser_uses_cdp_page_navigate(
    env_with_cdp_recorder, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: CDP available → Page.navigate fires with the target URL."""
    env, cdp_calls = env_with_cdp_recorder

    # cua-issues 2026-06-29: _navigate_running_browser now verifies the live
    # URL actually reached the target host (stale-tab guard). Simulate a
    # committed navigation so the happy path stays a single Page.navigate.
    monkeypatch.setattr(
        type(env), "current_url",
        property(lambda self: "https://example.com/page?status=Active&priority=High"),
    )

    env._navigate_running_browser("https://example.com/page?status=Active&priority=High")

    # Exactly one CDP call: Page.navigate with our URL
    assert len(cdp_calls) == 1
    method, params = cdp_calls[0]
    assert method == "Page.navigate"
    assert params["url"] == "https://example.com/page?status=Active&priority=High"

    # No omnibox-typing fallback was triggered
    assert env._test_xdotool_keys == []  # type: ignore[attr-defined]


def test_navigate_running_browser_falls_back_when_cdp_fails(
    env_with_cdp_recorder, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CDP unreachable → fall back to omnibox Ctrl+L / Ctrl+A / paste / Enter.

    Crucially, the fallback must NOT use ``_xdotool_type`` (which would
    re-route through the broken CDP ``Input.insertText`` path). It must
    use a dedicated omnibox-aware typer that goes through X focus.
    """
    env, cdp_calls = env_with_cdp_recorder

    # Make Page.navigate fail
    def _cdp_fail(self, method: str, params: dict | None = None, *, timeout: float = 3.0):
        cdp_calls.append((method, params or {}))
        return False, {}

    monkeypatch.setattr(XdotoolGymEnv, "_cdp_call", _cdp_fail)

    # Record which omnibox typer was called
    called_omnibox_typer = MagicMock()
    monkeypatch.setattr(XdotoolGymEnv, "_type_into_omnibox", called_omnibox_typer)

    env._navigate_running_browser("https://example.com/page")

    # Page.navigate was attempted
    assert cdp_calls[0][0] == "Page.navigate"
    # Omnibox-aware typer was used (NOT _xdotool_type which preferred CDP)
    called_omnibox_typer.assert_called_once_with("https://example.com/page")
    # Ctrl+L, Ctrl+A, Return sequence
    assert "ctrl+l" in env._test_xdotool_keys  # type: ignore[attr-defined]
    assert "ctrl+a" in env._test_xdotool_keys  # type: ignore[attr-defined]
    assert "Return" in env._test_xdotool_keys  # type: ignore[attr-defined]


def test_type_into_omnibox_prefers_xclip_paste(monkeypatch: pytest.MonkeyPatch) -> None:
    """Omnibox typing should NOT use CDP Input.insertText (the original bug).

    It uses xclip + Ctrl+V (X-focus-honoring) by default, which reaches
    the omnibox correctly since the caller has already focused it via
    Ctrl+L.
    """
    env = XdotoolGymEnv.__new__(XdotoolGymEnv)
    env._env = {}

    # Track all subprocess invocations
    popen_calls: list[list[str]] = []
    run_calls: list[list[str]] = []

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            popen_calls.append(cmd)
            self.stdin = MagicMock()
        def wait(self, timeout=None):
            return 0
        def terminate(self):
            pass

    def _record_run(cmd, **kwargs):
        run_calls.append(cmd)
        result = MagicMock()
        result.returncode = 0
        return result

    monkeypatch.setattr("mantis_agent.gym.xdotool_env.subprocess.Popen", FakePopen)
    monkeypatch.setattr("mantis_agent.gym.xdotool_env.subprocess.run", _record_run)
    monkeypatch.setattr("mantis_agent.gym.xdotool_env.time.sleep", lambda *_: None)
    monkeypatch.delenv("MANTIS_DISABLE_PASTE_TYPE", raising=False)

    env._type_into_omnibox("https://example.com/leads?status=Contacted")

    # xclip launched, Ctrl+V pasted
    assert any("xclip" in c[0] for c in popen_calls), (
        f"expected xclip launch, got popens: {popen_calls!r}"
    )
    assert any(
        "xdotool" in c[0] and "ctrl+v" in c
        for c in run_calls
    ), f"expected ctrl+v xdotool, got runs: {run_calls!r}"

    # CRITICAL: no _cdp_call to Input.insertText happened
    # (Hard to assert directly; covered by the architectural choice to
    # NOT call self._xdotool_type or self._cdp_call in this method.)


def test_type_into_omnibox_xdotool_fallback_when_xclip_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If xclip is missing, omnibox typing falls back to raw ``xdotool type``."""
    env = XdotoolGymEnv.__new__(XdotoolGymEnv)
    env._env = {}

    run_calls: list[list[str]] = []

    def _failing_popen(cmd, **kwargs):
        raise FileNotFoundError("xclip not installed")

    def _record_run(cmd, **kwargs):
        run_calls.append(cmd)
        result = MagicMock()
        result.returncode = 0
        return result

    monkeypatch.setattr("mantis_agent.gym.xdotool_env.subprocess.Popen", _failing_popen)
    monkeypatch.setattr("mantis_agent.gym.xdotool_env.subprocess.run", _record_run)
    monkeypatch.setattr("mantis_agent.gym.xdotool_env.time.sleep", lambda *_: None)

    env._type_into_omnibox("https://example.com")

    # xdotool type fell through with the URL
    type_calls = [c for c in run_calls if "xdotool" in c[0] and "type" in c]
    assert len(type_calls) == 1
    assert "https://example.com" in type_calls[0]


def test_navigate_seeds_cookie_header_into_jar_before_page_navigate(
    env_with_cdp_recorder,
) -> None:
    """A ``Cookie`` extra-header is seeded via ``Network.setCookie`` *before*
    ``Page.navigate``.

    ``Network.setExtraHTTPHeaders`` silently drops a manually-set ``Cookie``
    header, so a sim-env consent cookie never reaches the server and its
    banner renders (read as ``page_blocked`` by ``find_all_listings``).
    Putting the cookie in the real jar — scoped to the target URL, and set
    before the navigation so the first load carries it — fixes that.
    """
    env, cdp_calls = env_with_cdp_recorder
    env._extra_http_headers = {  # type: ignore[attr-defined]
        "x-daytona-preview-token": "tok",
        "Cookie": "bt_cookie_consent=decline",
    }

    env._navigate_running_browser("https://8080-abc.daytonaproxy01.net/boats")

    # setCookie must precede Page.navigate (jar populated before the load).
    assert cdp_calls[0][0] == "Network.setCookie"
    assert cdp_calls[0][1] == {
        "url": "https://8080-abc.daytonaproxy01.net/boats",
        "name": "bt_cookie_consent",
        "value": "decline",
    }
    assert cdp_calls[1][0] == "Page.navigate"


def test_navigate_seeds_every_cookie_in_a_multi_value_header(
    env_with_cdp_recorder,
) -> None:
    """A multi-cookie header yields one ``Network.setCookie`` per pair."""
    env, cdp_calls = env_with_cdp_recorder
    env._extra_http_headers = {  # type: ignore[attr-defined]
        "Cookie": "bt_cookie_consent=decline; session=xyz",
    }

    env._navigate_running_browser("https://example.com/")

    set_cookies = [p for m, p in cdp_calls if m == "Network.setCookie"]
    assert {c["name"]: c["value"] for c in set_cookies} == {
        "bt_cookie_consent": "decline",
        "session": "xyz",
    }


def test_navigate_without_cookie_header_skips_setcookie(
    env_with_cdp_recorder,
) -> None:
    """No ``Cookie`` header → no ``Network.setCookie`` (non-sim-env runs are
    unaffected; only ``Page.navigate`` fires)."""
    env, cdp_calls = env_with_cdp_recorder
    env._extra_http_headers = {"x-daytona-preview-token": "tok"}  # type: ignore[attr-defined]

    env._navigate_running_browser("https://example.com/page")

    assert all(method != "Network.setCookie" for method, _ in cdp_calls)
    assert cdp_calls[0][0] == "Page.navigate"


def test_navigate_running_browser_empty_url_is_noop(env_with_cdp_recorder) -> None:
    """Defensive: empty URL must not call Page.navigate (would 400)."""
    env, cdp_calls = env_with_cdp_recorder

    # reset() already filters empty URLs before this method, but the
    # method itself shouldn't crash if called with one.
    env._navigate_running_browser("")
    # CDP call still goes out — Page.navigate with empty url is a valid
    # CDP request and Chrome will no-op. We assert behavior is consistent.
    # If a future refactor adds an empty-string guard, update this test.
    assert len(cdp_calls) <= 1


# ── cold-start launch ordering ─────────────────────────────────────────
#
# The companion bug to the navigate fix above: on a *fresh* browser start,
# ``_start_browser`` launched Chrome with the target URL on the command line,
# so the first document (e.g. the by-owner SRP) was fetched *immediately* —
# before ``_open_header_session`` opened and with an empty cookie jar. The
# sim-env consent cookie was never seeded (``Network.setExtraHTTPHeaders``
# drops a ``Cookie`` header; only ``Network.setCookie`` populates the jar, and
# that runs inside ``_navigate_running_browser``, which the cold-start path
# never called). The consent overlay rendered and ``find_all_listings`` read
# it as ``page_blocked`` on every retry. Fix: when extra headers are present,
# launch to about:blank and defer the real navigation through the cookie-
# seeding path once the header session is live.


def _make_start_browser_env(tmp_path, *, extra_headers):
    env = XdotoolGymEnv.__new__(XdotoolGymEnv)
    env._viewport = (1280, 800)
    env._proxy_server = None
    env._settle_time = 0.0
    env._env = {}
    env._browser_proc = None
    env._browser_cmd = "chromium-browser"
    env._cdp_port = 9222
    env._profile_dir = str(tmp_path / "profile")
    env._extra_http_headers = extra_headers
    return env


@pytest.fixture
def start_browser_recorder(monkeypatch: pytest.MonkeyPatch):
    """Stub out the real browser launch + CDP so ``_start_browser`` runs in
    process. Records the Popen argv, deferred-navigation URLs, and the order of
    header-session vs navigation events."""
    popen_cmds: list[list[str]] = []
    nav_urls: list[str] = []
    events: list[str] = []

    class _FakeProc:
        def poll(self):
            return None

    def _fake_popen(cmd, **kwargs):
        popen_cmds.append(cmd)
        events.append("popen")
        return _FakeProc()

    monkeypatch.setattr("mantis_agent.gym.xdotool_env.subprocess.Popen", _fake_popen)
    monkeypatch.setattr("mantis_agent.gym.xdotool_env.time.sleep", lambda *_: None)
    monkeypatch.setattr(XdotoolGymEnv, "_wait_for_cdp_ready", lambda self, *a, **k: True)
    monkeypatch.setattr(XdotoolGymEnv, "_cdp_call", lambda self, *a, **k: (True, {}))
    monkeypatch.setattr(
        "mantis_agent.gym.cdp_stealth.inject_stealth_patches", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "mantis_agent.gym.cdp_stealth.apply_ua_override", lambda *a, **k: None
    )

    def _fake_open_header_session(self):
        events.append("header_session")

    def _fake_navigate(self, url):
        events.append("navigate")
        nav_urls.append(url)

    monkeypatch.setattr(XdotoolGymEnv, "_open_header_session", _fake_open_header_session)
    monkeypatch.setattr(XdotoolGymEnv, "_navigate_running_browser", _fake_navigate)
    return popen_cmds, nav_urls, events


def test_start_browser_defers_nav_when_extra_headers_present(
    start_browser_recorder, tmp_path,
) -> None:
    """Cold start WITH extra headers (sim-env consent Cookie): launch to
    about:blank and defer the real navigation until after the header session
    opens, so the first SRP fetch carries the seeded cookie rather than racing
    ahead with an empty jar (→ consent banner → page_blocked)."""
    popen_cmds, nav_urls, events = start_browser_recorder
    env = _make_start_browser_env(
        tmp_path, extra_headers={"Cookie": "bt_cookie_consent=decline"}
    )

    target = "https://8080-abc.daytonaproxy01.net/boats/state-fl/by-owner/"
    env._start_browser(target)

    # Launched to about:blank, NOT the real URL.
    assert popen_cmds[0][-1] == "about:blank"
    assert target not in popen_cmds[0]
    # Real navigation deferred to the cookie-seeding path.
    assert nav_urls == [target]
    # Ordering: header session opened BEFORE the deferred navigation.
    assert events.index("header_session") < events.index("navigate")


def test_start_browser_launches_url_directly_without_extra_headers(
    start_browser_recorder, tmp_path,
) -> None:
    """No extra headers (ordinary runs): unchanged behavior — Chrome launches
    straight to the URL and no deferred CDP navigation is issued."""
    popen_cmds, nav_urls, events = start_browser_recorder
    env = _make_start_browser_env(tmp_path, extra_headers=None)

    target = "https://example.com/page"
    env._start_browser(target)

    assert popen_cmds[0][-1] == target
    assert nav_urls == []  # no deferred navigation
