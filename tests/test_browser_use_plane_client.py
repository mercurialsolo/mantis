"""Tests for `BrowserUsePlaneClient` (#785, PR 2).

Mock-server style: monkeypatch `requests.Session.post`/`.get` to verify
the wire shape without standing up a real FastAPI server. Mirrors the
pattern in `tests/test_remote_computer_impl.py`.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Any
import pytest
from PIL import Image

from mantis_agent.gym.browser_use_plane_client import BrowserUsePlaneClient
from mantis_agent.gym.browser_use_wire import (
    BrowserUseScreenshotResponse,
    BrowserUseSessionInitResponse,
    DispatchActionResponse,
)
from mantis_agent.gym.compute_contract import Capabilities, ComputeBackend


@dataclass
class _Resp:
    """Minimal `requests.Response` stand-in."""

    status_code: int = 200
    _json: dict[str, Any] = field(default_factory=dict)

    def json(self) -> dict[str, Any]:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _png_b64(w: int = 8, h: int = 8) -> str:
    img = Image.new("RGB", (w, h), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class _FakeServer:
    """Routes requests by URL suffix; records call counts."""

    def __init__(self) -> None:
        self.posts: list[tuple[str, dict[str, Any]]] = []
        self.gets: list[str] = []

    def post(self, url: str, **kw: Any) -> _Resp:
        self.posts.append((url, kw.get("json", {})))
        if url.endswith("/session/init"):
            return _Resp(
                _json=BrowserUseSessionInitResponse(
                    session_token="tok-abc",
                    browser_pid=12345,
                    capabilities=Capabilities.for_browser_use_plane().as_dict(),
                ).model_dump()
            )
        if url.endswith("/session/close"):
            return _Resp(_json={"closed": True})
        if url.endswith("/screenshot"):
            return _Resp(
                _json=BrowserUseScreenshotResponse(
                    image_b64=_png_b64(),
                    width=8,
                    height=8,
                    scroll_y=0,
                    captured_at_ms=1,
                ).model_dump()
            )
        if url.endswith("/dispatch"):
            return _Resp(_json=DispatchActionResponse(ok=True).model_dump())
        return _Resp(status_code=404)

    def get(self, url: str, **kw: Any) -> _Resp:
        self.gets.append(url)
        if url.endswith("/health"):
            return _Resp(_json={"ok": True, "last_action_ms": 100, "session_token": "tok-abc"})
        return _Resp(status_code=404)


def _make_client(server: _FakeServer) -> BrowserUsePlaneClient:
    client = BrowserUsePlaneClient(
        base_url="https://browser-use.test",
        tenant_id="t1",
        profile_id="p1",
        run_id="r1",
    )
    # Replace the session's HTTP methods with the fake server's.
    client._http.post = server.post  # type: ignore[method-assign]
    client._http.get = server.get  # type: ignore[method-assign]
    return client


def test_reset_initializes_session_and_returns_observation():
    server = _FakeServer()
    client = _make_client(server)
    obs = client.reset("noop")
    # session_init came first, then screenshot.
    urls = [p[0] for p in server.posts]
    assert urls[0].endswith("/session/init")
    assert urls[1].endswith("/screenshot")
    assert obs.screenshot.size == (8, 8)
    caps = client.capabilities()
    assert caps.dom_aware is True
    assert caps.backend is ComputeBackend.BROWSER_USE_PLANE


def test_default_capabilities_when_session_not_initialized():
    server = _FakeServer()
    client = _make_client(server)
    caps = client.capabilities()
    # Before session_init, client reports Browser-Use Plane defaults (NOT
    # Computer-Plane defaults — would be a wire mismatch).
    assert caps.dom_aware is True
    assert caps.backend is ComputeBackend.BROWSER_USE_PLANE


def test_close_releases_session():
    server = _FakeServer()
    client = _make_client(server)
    client.reset("noop")
    client.close()
    urls = [p[0] for p in server.posts]
    assert any(u.endswith("/session/close") for u in urls)


def test_step_dispatches_click_action():
    server = _FakeServer()
    client = _make_client(server)
    client.reset("noop")

    # Synthetic action shape matching the translator's expectations.
    class _A:
        type = "click"
        x = 100
        y = 200
        button = "left"
        click_count = 1

    # Bypass the `isinstance(action, Action)` check in the translator —
    # this test pins the dispatch wire shape, not the Action-translation
    # logic. Translator coverage lives in a separate test below.
    client._translate_action = lambda action: ("click", {  # type: ignore[method-assign]
        "x": action.x,
        "y": action.y,
        "button": action.button,
        "click_count": action.click_count,
    })
    client.step(_A())  # type: ignore[arg-type]
    urls = [p[0] for p in server.posts]
    assert any(u.endswith("/dispatch") for u in urls)
    # Check the dispatch payload had the expected click params.
    dispatch_payload = next(p for u, p in server.posts if u.endswith("/dispatch"))
    assert dispatch_payload["kind"] == "click"
    assert dispatch_payload["params"]["x"] == 100
    assert dispatch_payload["params"]["y"] == 200


def test_unconfigured_url_raises_executor_construction_error(monkeypatch):
    """When MANTIS_BROWSER_USE_URL is unset AND Modal SDK resolution
    fails, the executor must raise rather than silently return a
    client with no base_url.

    The Modal SDK lookup is patched to raise so the test runs
    deterministically — on a dev machine where ``mantis-browser-use``
    is deployed, the live ``modal.Function.from_name(...)`` call
    would succeed and silently mask the no-URL case the test is
    pinning down.
    """
    import mantis_agent.run_browser_use as run_browser_use_mod
    from mantis_agent.run_browser_use import make_browser_use_client

    monkeypatch.delenv("MANTIS_BROWSER_USE_URL", raising=False)

    # Force the Modal SDK lookup to fail — production sometimes can't
    # reach Modal (network blip, missing auth) and the executor must
    # surface the misconfig clearly in that case.
    def _fail_resolve() -> str:
        return ""

    monkeypatch.setattr(
        run_browser_use_mod, "resolve_browser_use_base_url", _fail_resolve
    )

    with pytest.raises(RuntimeError, match="Browser-Use Plane URL is not configured"):
        make_browser_use_client(base_url=None)


def test_screen_size_returns_viewport():
    server = _FakeServer()
    client = _make_client(server)
    assert client.screen_size == (1280, 720)


def test_health_returns_session_token_after_init():
    server = _FakeServer()
    client = _make_client(server)
    client.reset("noop")
    h = client.health()
    assert h.ok is True
    assert h.session_token == "tok-abc"
