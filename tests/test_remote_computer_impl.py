"""Tests for `RemoteComputerImpl` — the Phase 1 HTTPS client (#698).

Mock-server style: a `FakeServer` callable replaces `requests.post` and
verifies retry / dedup / timeout behavior without standing up a real
FastAPI server. End-to-end tests against the real server live in
`tests/test_computer_agent_server.py`.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest
import requests
from PIL import Image

from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.computer_client import (
    ComputerPlaneConfig,
    make_computer_client,
)
from mantis_agent.gym.remote_computer_impl import RemoteComputerImpl


@dataclass
class _Resp:
    status_code: int
    payload: dict[str, Any] = field(default_factory=dict)
    text: str = ""

    def json(self) -> dict[str, Any]:
        return self.payload


class _FakeServer:
    """Replacement for `requests.post` — records calls + returns canned
    responses keyed by path."""

    def __init__(self, *, screenshot_image_b64: str | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_next_n_500: dict[str, int] = {}
        self.timeout_next_n: dict[str, int] = {}
        self.screenshot_image_b64 = screenshot_image_b64 or _png_b64()
        self.session_token = "tok-abc"
        self.dedup_returncode_overrides: dict[str, int] = {}
        # Proxied-env state — flipped by individual tests as needed.
        self.current_url = "https://example.com/start"
        self.page_count = 1
        self.cdp_click_ok = True
        self.cdp_result_json = '{"result": {"value": 42}}'

    def __call__(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> _Resp:
        path = "/" + url.split("/", 3)[-1] if url.count("/") >= 3 else url
        self.calls.append({"path": path, "json": json, "headers": headers or {}})

        # Forced failures (e.g. simulated 502s) for retry tests.
        if self.fail_next_n_500.get(path, 0) > 0:
            self.fail_next_n_500[path] -= 1
            return _Resp(status_code=502, text="bad gateway")

        if self.timeout_next_n.get(path, 0) > 0:
            self.timeout_next_n[path] -= 1
            raise requests.ConnectTimeout("simulated timeout")

        if path == "/session/init":
            return _Resp(
                200,
                {
                    "session_token": self.session_token,
                    "chrome_pid": 1234,
                    "xvfb_display": ":99",
                },
            )
        if path == "/session/close":
            return _Resp(200, {"closed": True})
        if path == "/screenshot":
            return _Resp(
                200,
                {
                    "image_b64": self.screenshot_image_b64,
                    "width": 16,
                    "height": 16,
                    "scroll_y": 42,
                    "captured_at_ms": 1700000000000,
                },
            )
        if path == "/xdotool":
            step_id = (json or {}).get("step_id", "")
            return _Resp(
                200,
                {
                    "stdout": "",
                    "stderr": "",
                    "returncode": self.dedup_returncode_overrides.get(step_id, 0),
                    "deduplicated": False,
                },
            )
        if path == "/cdp":
            return _Resp(200, {"result_json": self.cdp_result_json, "returncode": 0})
        if path == "/cdp_click_at_point":
            return _Resp(200, {"ok": self.cdp_click_ok, "error": None})
        return _Resp(404, {}, text=f"no handler for {path}")

    def get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> _Resp:
        """Sibling of `__call__` for GET endpoints (`/current_url`,
        `/cdp_count_pages`). Recorded with the same `self.calls` shape
        as POSTs so assertions can grep paths uniformly.
        """
        path = "/" + url.split("/", 3)[-1] if url.count("/") >= 3 else url
        self.calls.append({"path": path, "json": None, "headers": headers or {}})
        if path == "/current_url":
            return _Resp(200, {"url": self.current_url})
        if path == "/cdp_count_pages":
            return _Resp(200, {"count": self.page_count})
        return _Resp(404, {}, text=f"no GET handler for {path}")


def _png_b64(size: tuple[int, int] = (16, 16)) -> str:
    img = Image.new("RGB", size, "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ── factory + construction ────────────────────────────────────────────


def test_factory_modal_backend_dispatches_to_remote() -> None:
    cfg = ComputerPlaneConfig(
        backend="modal",
        remote_base_url="https://example.invalid",
    )
    env = make_computer_client(cfg)
    assert isinstance(env, RemoteComputerImpl)
    assert env._base_url == "https://example.invalid"


def test_init_lazy_session_token_is_unset() -> None:
    env = RemoteComputerImpl(base_url="https://x")
    assert env._session_token is None


# ── session lifecycle ─────────────────────────────────────────────────


def test_reset_calls_session_init_then_screenshot() -> None:
    fake = _FakeServer()
    env = RemoteComputerImpl(base_url="https://x")
    with patch("requests.post", side_effect=fake):
        obs = env.reset(task="t")
    paths = [c["path"] for c in fake.calls]
    assert paths == ["/session/init", "/screenshot"]
    assert env._session_token == fake.session_token
    assert obs.screenshot.size == (16, 16)
    # Session header was attached to the screenshot request.
    assert fake.calls[1]["headers"].get("X-Mantis-Session") == fake.session_token


def test_close_sends_session_close_and_clears_token() -> None:
    fake = _FakeServer()
    env = RemoteComputerImpl(base_url="https://x")
    with patch("requests.post", side_effect=fake):
        env.reset(task="t")
        env.close()
    assert env._session_token is None
    paths = [c["path"] for c in fake.calls]
    assert "/session/close" in paths


def test_close_without_init_is_noop() -> None:
    env = RemoteComputerImpl(base_url="https://x")
    # No mock — must not call requests.post at all.
    env.close()
    assert env._session_token is None


# ── xdotool dispatch + retry ──────────────────────────────────────────


def test_step_click_translates_to_mousemove_then_click() -> None:
    fake = _FakeServer()
    env = RemoteComputerImpl(base_url="https://x", viewport=(1280, 720))
    action = Action(action_type=ActionType.CLICK, params={"x": 100, "y": 200})

    with patch("requests.post", side_effect=fake):
        env.reset(task="t")
        env.step(action)

    xdotool_calls = [c for c in fake.calls if c["path"] == "/xdotool"]
    assert [c["json"]["argv"] for c in xdotool_calls] == [
        ["mousemove", "100", "200"],
        ["click", "1"],
    ]
    # Each gets a distinct step_id.
    step_ids = [c["json"]["step_id"] for c in xdotool_calls]
    assert len(step_ids) == 2 and step_ids[0] != step_ids[1]


def test_step_scroll_emits_mousemove_plus_click_repeat() -> None:
    fake = _FakeServer()
    env = RemoteComputerImpl(base_url="https://x", viewport=(800, 600))
    action = Action(action_type=ActionType.SCROLL, params={"direction": "down", "amount": 5})

    with patch("requests.post", side_effect=fake):
        env.reset(task="t")
        env.step(action)

    argvs = [c["json"]["argv"] for c in fake.calls if c["path"] == "/xdotool"]
    assert argvs == [
        ["mousemove", "400", "300"],
        ["click", "--repeat", "5", "5"],
    ]


def test_step_type_text_emits_single_xdotool() -> None:
    fake = _FakeServer()
    env = RemoteComputerImpl(base_url="https://x")
    action = Action(action_type=ActionType.TYPE, params={"text": "hello"})

    with patch("requests.post", side_effect=fake):
        env.reset(task="t")
        env.step(action)

    argvs = [c["json"]["argv"] for c in fake.calls if c["path"] == "/xdotool"]
    assert argvs == [["type", "--delay", "0", "hello"]]


def test_step_wait_emits_no_xdotool_but_returns_screenshot() -> None:
    fake = _FakeServer()
    env = RemoteComputerImpl(base_url="https://x")

    with patch("requests.post", side_effect=fake), patch(
        "mantis_agent.gym.remote_computer_impl.time.sleep"
    ) as fake_sleep:
        env.reset(task="t")
        result = env.step(Action(action_type=ActionType.WAIT, params={"seconds": 2.0}))

    assert not any(c["path"] == "/xdotool" for c in fake.calls)
    fake_sleep.assert_called_once_with(2.0)
    assert result.observation.screenshot.size == (16, 16)


def test_done_action_emits_screenshot_only() -> None:
    fake = _FakeServer()
    env = RemoteComputerImpl(base_url="https://x")

    with patch("requests.post", side_effect=fake):
        env.reset(task="t")
        env.step(Action(action_type=ActionType.DONE))

    assert not any(c["path"] == "/xdotool" for c in fake.calls)


# ── retry on 5xx ──────────────────────────────────────────────────────


def test_xdotool_retries_with_same_step_id_on_transient_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Critical for idempotency: a retried `xdotool` must reuse the same
    `step_id` so the server's LRU dedup turns the retry into a no-op."""
    fake = _FakeServer()
    fake.fail_next_n_500["/xdotool"] = 1  # 1 transient failure
    env = RemoteComputerImpl(base_url="https://x")
    monkeypatch.setattr(
        "mantis_agent.gym.remote_computer_impl.time.sleep", lambda *_: None
    )

    with patch("requests.post", side_effect=fake):
        env.reset(task="t")
        env.remote_xdotool("key", "Return", step_id="step-fixed-id")

    xdotool_calls = [c for c in fake.calls if c["path"] == "/xdotool"]
    assert len(xdotool_calls) == 2
    assert {c["json"]["step_id"] for c in xdotool_calls} == {"step-fixed-id"}


def test_xdotool_gives_up_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeServer()
    fake.fail_next_n_500["/xdotool"] = 99  # always fail
    env = RemoteComputerImpl(base_url="https://x")
    monkeypatch.setattr(
        "mantis_agent.gym.remote_computer_impl.time.sleep", lambda *_: None
    )

    with patch("requests.post", side_effect=fake):
        env.reset(task="t")
        with pytest.raises(RuntimeError, match="502"):
            env.remote_xdotool("key", "Return", step_id="step-x")


def test_4xx_does_not_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Client errors must surface immediately — retrying a 401 helps nobody."""
    call_count = 0

    def _post(*_args: Any, **_kwargs: Any) -> _Resp:
        nonlocal call_count
        call_count += 1
        return _Resp(401, text="unauthorized")

    env = RemoteComputerImpl(base_url="https://x")
    env._session_token = "tok"
    monkeypatch.setattr(
        "mantis_agent.gym.remote_computer_impl.time.sleep", lambda *_: None
    )

    with patch("requests.post", side_effect=_post):
        with pytest.raises(RuntimeError, match="401"):
            env.remote_screenshot()
    assert call_count == 1


# ── headers ───────────────────────────────────────────────────────────


def test_auth_token_added_as_bearer() -> None:
    fake = _FakeServer()
    env = RemoteComputerImpl(base_url="https://x", auth_token="secret-abc")

    with patch("requests.post", side_effect=fake):
        env.reset(task="t")

    for call in fake.calls:
        assert call["headers"].get("Authorization") == "Bearer secret-abc"


def test_no_auth_token_means_no_authorization_header() -> None:
    fake = _FakeServer()
    env = RemoteComputerImpl(base_url="https://x")

    with patch("requests.post", side_effect=fake):
        env.reset(task="t")

    for call in fake.calls:
        assert "Authorization" not in call["headers"]


# ── CDP gating ────────────────────────────────────────────────────────


def test_cdp_disabled_by_default_raises_on_remote_cdp() -> None:
    env = RemoteComputerImpl(base_url="https://x")
    with pytest.raises(RuntimeError, match="enable_cdp=False"):
        env.remote_cdp("window.scrollY")


def test_cdp_enabled_dispatches() -> None:
    fake = _FakeServer()
    env = RemoteComputerImpl(base_url="https://x", enable_cdp=True)

    with patch("requests.post", side_effect=fake):
        env.reset(task="t")
        env.remote_cdp("window.scrollY")

    cdp_calls = [c for c in fake.calls if c["path"] == "/cdp"]
    assert len(cdp_calls) == 1
    assert cdp_calls[0]["json"]["expression"] == "window.scrollY"


# ── latency tracking ──────────────────────────────────────────────────


def test_latency_report_aggregates_screenshot_and_xdotool() -> None:
    fake = _FakeServer()
    env = RemoteComputerImpl(base_url="https://x")

    with patch("requests.post", side_effect=fake):
        env.reset(task="t")
        env.step(Action(action_type=ActionType.CLICK, params={"x": 1, "y": 2}))
        env.step(Action(action_type=ActionType.CLICK, params={"x": 3, "y": 4}))

    report = env.latency_report()
    # reset() takes 1 screenshot; each step also takes 1 screenshot.
    assert report["screenshot"]["count"] == 3
    # Each CLICK is 2 xdotool calls → 4 across the two steps.
    assert report["xdotool"]["count"] == 4


# ── XdotoolGymEnv-compatibility surface ──────────────────────────────


def test_session_init_forwards_start_url_profile_dir_extra_headers() -> None:
    fake = _FakeServer()
    env = RemoteComputerImpl(
        base_url="https://x",
        start_url="https://target.example/landing",
        profile_dir="/data/chrome-profile/t__p",
        extra_http_headers={"X-Daytona-Skip-Preview-Warning": "true"},
    )
    with patch("requests.post", side_effect=fake):
        env.session_init()
    init_call = next(c for c in fake.calls if c["path"] == "/session/init")
    assert init_call["json"]["start_url"] == "https://target.example/landing"
    assert init_call["json"]["profile_dir"] == "/data/chrome-profile/t__p"
    assert init_call["json"]["extra_http_headers"] == {
        "X-Daytona-Skip-Preview-Warning": "true",
    }


def test_screenshot_returns_pil_image_with_same_bytes_as_observation() -> None:
    fake = _FakeServer()
    env = RemoteComputerImpl(base_url="https://x")
    with patch("requests.post", side_effect=fake):
        env.reset(task="t")
        img_direct = env.screenshot()
        obs = env._capture()
    assert img_direct.size == (16, 16)
    assert obs.screenshot.size == (16, 16)


def test_shutdown_calls_session_close() -> None:
    fake = _FakeServer()
    env = RemoteComputerImpl(base_url="https://x")
    with patch("requests.post", side_effect=fake):
        env.reset(task="t")
        env.shutdown()
    assert env._session_token is None
    assert any(c["path"] == "/session/close" for c in fake.calls)


def test_capture_browser_state_is_empty_stub() -> None:
    env = RemoteComputerImpl(base_url="https://x")
    assert env.capture_browser_state() == {}


def test_current_url_proxies_get_current_url() -> None:
    fake = _FakeServer()
    fake.current_url = "https://example.com/leads?owner=true"
    env = RemoteComputerImpl(base_url="https://x")
    with patch("requests.post", side_effect=fake), patch(
        "requests.get", side_effect=fake.get
    ):
        env.reset(task="t")
        url = env.current_url
    assert url == "https://example.com/leads?owner=true"


def test_current_url_swallows_failure_and_returns_empty_string() -> None:
    def _failing_get(*_a: Any, **_kw: Any) -> _Resp:
        raise requests.ConnectTimeout("nope")

    env = RemoteComputerImpl(base_url="https://x")
    env._session_token = "tok"  # skip init
    with patch("requests.get", side_effect=_failing_get), patch(
        "mantis_agent.gym.remote_computer_impl.time.sleep", lambda *_: None
    ):
        assert env.current_url == ""


def test_cdp_evaluate_disabled_raises() -> None:
    env = RemoteComputerImpl(base_url="https://x")
    with pytest.raises(RuntimeError, match="enable_cdp=False"):
        env.cdp_evaluate("window.scrollY")


def test_cdp_evaluate_unwraps_runtime_evaluate_result_value() -> None:
    fake = _FakeServer()
    fake.cdp_result_json = '{"result": {"type": "number", "value": 1234}}'
    env = RemoteComputerImpl(base_url="https://x", enable_cdp=True)
    with patch("requests.post", side_effect=fake):
        env.reset(task="t")
        out = env.cdp_evaluate("window.scrollY")
    assert out == 1234


def test_cdp_evaluate_returns_none_on_undefined_result() -> None:
    fake = _FakeServer()
    fake.cdp_result_json = "{}"  # mirrors JS `undefined`
    env = RemoteComputerImpl(base_url="https://x", enable_cdp=True)
    with patch("requests.post", side_effect=fake):
        env.reset(task="t")
        assert env.cdp_evaluate("nothing") is None


def test_chrome_offset_px_uses_cdp_evaluate() -> None:
    fake = _FakeServer()
    fake.cdp_result_json = '{"result": {"value": 95}}'
    env = RemoteComputerImpl(base_url="https://x", enable_cdp=True)
    with patch("requests.post", side_effect=fake):
        env.reset(task="t")
        assert env._chrome_offset_px() == 95


def test_cdp_click_at_point_proxies_endpoint() -> None:
    fake = _FakeServer()
    fake.cdp_click_ok = True
    env = RemoteComputerImpl(base_url="https://x", enable_cdp=True)
    with patch("requests.post", side_effect=fake):
        env.reset(task="t")
        assert env.cdp_click_at_point(100, 200) is True
    click_calls = [c for c in fake.calls if c["path"] == "/cdp_click_at_point"]
    assert len(click_calls) == 1
    payload = click_calls[0]["json"]
    assert payload["x"] == 100 and payload["y"] == 200
    assert payload["via_pointer"] is False


def test_cdp_click_via_pointer_sets_via_pointer_flag() -> None:
    fake = _FakeServer()
    env = RemoteComputerImpl(base_url="https://x", enable_cdp=True)
    with patch("requests.post", side_effect=fake):
        env.reset(task="t")
        env.cdp_click_via_pointer(300, 400)
    click_calls = [c for c in fake.calls if c["path"] == "/cdp_click_at_point"]
    assert click_calls[0]["json"]["via_pointer"] is True


def test_cdp_click_requires_enable_cdp() -> None:
    env = RemoteComputerImpl(base_url="https://x")
    with pytest.raises(RuntimeError, match="enable_cdp=True"):
        env.cdp_click_at_point(1, 2)


def test_cdp_count_pages_proxies_get_endpoint() -> None:
    fake = _FakeServer()
    fake.page_count = 3
    env = RemoteComputerImpl(base_url="https://x")
    with patch("requests.post", side_effect=fake), patch(
        "requests.get", side_effect=fake.get
    ):
        env.reset(task="t")
        assert env.cdp_count_pages() == 3


def test_cdp_count_pages_returns_zero_on_failure() -> None:
    def _bad_get(*_a: Any, **_kw: Any) -> _Resp:
        raise requests.ConnectionError("nope")

    env = RemoteComputerImpl(base_url="https://x")
    env._session_token = "tok"
    with patch("requests.get", side_effect=_bad_get), patch(
        "mantis_agent.gym.remote_computer_impl.time.sleep", lambda *_: None
    ):
        assert env.cdp_count_pages() == 0


def test_cdp_history_back_returns_true_on_url_change() -> None:
    fake = _FakeServer()
    fake.cdp_result_json = "{}"  # `history.back()` returns undefined
    urls = iter(["https://example.com/before", "https://example.com/after"])

    def _get(url: str, **_kw: Any) -> _Resp:
        path = "/" + url.split("/", 3)[-1] if url.count("/") >= 3 else url
        fake.calls.append({"path": path, "json": None, "headers": {}})
        if path == "/current_url":
            return _Resp(200, {"url": next(urls)})
        return _Resp(404, {})

    env = RemoteComputerImpl(base_url="https://x", enable_cdp=True)
    with patch("requests.post", side_effect=fake), patch(
        "requests.get", side_effect=_get
    ), patch("mantis_agent.gym.remote_computer_impl.time.sleep", lambda *_: None):
        env.reset(task="t")
        assert env.cdp_history_back(settle_seconds=0.5) is True


def test_cdp_history_back_returns_false_when_url_unchanged() -> None:
    fake = _FakeServer()
    env = RemoteComputerImpl(base_url="https://x", enable_cdp=True)
    with patch("requests.post", side_effect=fake), patch(
        "requests.get", side_effect=fake.get
    ), patch("mantis_agent.gym.remote_computer_impl.time.sleep", lambda *_: None):
        env.reset(task="t")
        # `fake.current_url` is constant → no change → returns False.
        assert env.cdp_history_back(settle_seconds=0.3) is False


def test_cdp_history_back_returns_false_when_cdp_disabled() -> None:
    env = RemoteComputerImpl(base_url="https://x")
    assert env.cdp_history_back() is False
