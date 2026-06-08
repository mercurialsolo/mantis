"""Tests for `state.*` extensions on the Browser-Use Plane (#778, PR 3).

Covers:
- Wire-shape round trip for each `state.*` endpoint via mock server.
- `SupportsBrowserState` Protocol satisfaction on `BrowserUsePlaneClient`.
- Capability allowlist gating — pure-CUA executors raise
  `CapabilityNotAllowed` when handlers attempt to consume these.
- `safe_back` overshoot guard semantics.
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
    BrowserUseSessionInitResponse,
    FocusedElementSummary,
    StateClipboardResponse,
    StateCurrentUrlResponse,
    StateFocusedElementResponse,
    StatePageLoadResponse,
    StateSafeBackResponse,
    StateTabsResponse,
    TabSummary,
)
from mantis_agent.gym.compute_contract import (
    Capabilities,
    CapabilityAllowlist,
    CapabilityNotAllowed,
    SupportsBrowserState,
)
from mantis_agent.server.browser_use_agent import _matches_pin


@dataclass
class _Resp:
    status_code: int = 200
    _json: dict[str, Any] = field(default_factory=dict)

    def json(self) -> dict[str, Any]:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _png_b64() -> str:
    img = Image.new("RGB", (8, 8), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class _FakeServer:
    """Routes state.* + base-surface requests for unit tests."""

    def __init__(self) -> None:
        self.posts: list[tuple[str, dict[str, Any]]] = []
        self.gets: list[str] = []
        self.current_url = "https://news.ycombinator.com/"
        self.focused: FocusedElementSummary | None = None
        self.tabs: list[TabSummary] = []
        self.clipboard_text = ""
        self.ready_state: str = "complete"
        self.safe_back_response: StateSafeBackResponse | None = None

    def post(self, url: str, **kw: Any) -> _Resp:
        self.posts.append((url, kw.get("json", {})))
        if url.endswith("/session/init"):
            return _Resp(
                _json=BrowserUseSessionInitResponse(
                    session_token="tok",
                    capabilities=Capabilities.for_browser_use_plane().as_dict(),
                ).model_dump()
            )
        if url.endswith("/state/safe_back"):
            sb = self.safe_back_response or StateSafeBackResponse(
                popped=True, new_url=self.current_url
            )
            return _Resp(_json=sb.model_dump())
        return _Resp(status_code=404)

    def get(self, url: str, **kw: Any) -> _Resp:
        self.gets.append(url)
        if url.endswith("/state/current_url"):
            return _Resp(_json=StateCurrentUrlResponse(url=self.current_url).model_dump())
        if url.endswith("/state/tabs"):
            return _Resp(_json=StateTabsResponse(tabs=self.tabs).model_dump())
        if url.endswith("/state/focused_element"):
            return _Resp(_json=StateFocusedElementResponse(element=self.focused).model_dump())
        if url.endswith("/state/clipboard"):
            return _Resp(_json=StateClipboardResponse(text=self.clipboard_text).model_dump())
        if url.endswith("/state/page_load"):
            return _Resp(
                _json=StatePageLoadResponse(
                    ready_state=self.ready_state,  # type: ignore[arg-type]
                    last_resource_ms=None,
                ).model_dump()
            )
        return _Resp(status_code=404)


def _make_client(server: _FakeServer) -> BrowserUsePlaneClient:
    client = BrowserUsePlaneClient(
        base_url="https://browser-use.test",
        tenant_id="t1",
        profile_id="p1",
        run_id="r1",
    )
    client._http.post = server.post  # type: ignore[method-assign]
    client._http.get = server.get  # type: ignore[method-assign]
    return client


# ── Protocol satisfaction ─────────────────────────────────────────


def test_browser_use_client_satisfies_supports_browser_state():
    server = _FakeServer()
    client = _make_client(server)
    # runtime_checkable Protocol — handlers dispatch on this isinstance
    # check before calling state.* methods.
    assert isinstance(client, SupportsBrowserState)


# ── Capability gating ─────────────────────────────────────────────


def test_pure_cua_allowlist_blocks_dom_aware_capability():
    allowlist = CapabilityAllowlist.pure_cua(executor="run_holo3")
    with pytest.raises(CapabilityNotAllowed):
        allowlist.enforce("dom_aware")


def test_browser_use_allowlist_admits_dom_aware_capability():
    allowlist = CapabilityAllowlist.browser_use(executor="run_browser_use")
    # Should not raise.
    allowlist.enforce("dom_aware")


# ── state.* method round-trips ────────────────────────────────────


def test_state_current_url_returns_server_value():
    server = _FakeServer()
    server.current_url = "https://news.ycombinator.com/news"
    client = _make_client(server)
    assert client.state_current_url() == "https://news.ycombinator.com/news"
    # Subsequent reads must hit /state/current_url (no caching at
    # client level — server is source of truth).
    client.state_current_url()
    assert sum(1 for g in server.gets if g.endswith("/state/current_url")) == 2


def test_state_tabs_returns_list_of_tab_summaries():
    server = _FakeServer()
    server.tabs = [
        TabSummary(id="tab-0", title="HN", url="https://news.ycombinator.com/", is_active=True),
        TabSummary(id="tab-1", title="Article", url="https://example.com/a"),
    ]
    client = _make_client(server)
    tabs = client.state_tabs()
    assert len(tabs) == 2
    assert tabs[0]["is_active"] is True
    assert tabs[1]["url"] == "https://example.com/a"


def test_state_focused_element_projects_to_dict():
    server = _FakeServer()
    server.focused = FocusedElementSummary(
        tag="a",
        role="link",
        aria_label="Story title",
        text="Why we switched",
        href="https://example.com/article",
    )
    client = _make_client(server)
    el = client.state_focused_element()
    assert el is not None
    assert el["tag"] == "a"
    assert el["href"] == "https://example.com/article"


def test_state_focused_element_returns_none_when_nothing_focused():
    server = _FakeServer()
    server.focused = None
    client = _make_client(server)
    assert client.state_focused_element() is None


def test_state_clipboard_returns_text():
    server = _FakeServer()
    server.clipboard_text = "copied snippet"
    client = _make_client(server)
    assert client.state_clipboard() == "copied snippet"


def test_state_page_load_returns_ready_state():
    server = _FakeServer()
    server.ready_state = "interactive"
    client = _make_client(server)
    pl = client.state_page_load()
    assert pl["ready_state"] == "interactive"


def test_state_safe_back_records_pinned_origin_in_request():
    server = _FakeServer()
    server.safe_back_response = StateSafeBackResponse(
        popped=True, new_url="https://news.ycombinator.com/"
    )
    client = _make_client(server)
    result = client.state_safe_back(pinned_origin="https://news.ycombinator.com/*")
    assert result["popped"] is True
    posts_to_safe_back = [p for u, p in server.posts if u.endswith("/state/safe_back")]
    assert posts_to_safe_back[0]["pinned_origin"] == "https://news.ycombinator.com/*"


def test_state_safe_back_overshoot_returns_popped_false():
    server = _FakeServer()
    server.safe_back_response = StateSafeBackResponse(
        popped=False,
        new_url="https://news.ycombinator.com/",
        reason="overshoot_pinned_origin",
    )
    client = _make_client(server)
    result = client.state_safe_back(pinned_origin="https://news.ycombinator.com/*")
    assert result["popped"] is False
    assert result["reason"] == "overshoot_pinned_origin"


def test_state_methods_lazy_init_session():
    server = _FakeServer()
    client = _make_client(server)
    # No reset() / explicit init — calling state_current_url first must
    # implicitly run session/init.
    client.state_current_url()
    urls = [u for u, _ in server.posts]
    assert any(u.endswith("/session/init") for u in urls)


# ── _matches_pin helper ──────────────────────────────────────────


def test_matches_pin_glob_prefix():
    assert _matches_pin("https://news.ycombinator.com/news", "https://news.ycombinator.com/*")
    assert _matches_pin("https://news.ycombinator.com/", "https://news.ycombinator.com/*")
    assert not _matches_pin("https://example.com/", "https://news.ycombinator.com/*")


def test_matches_pin_exact_origin():
    assert _matches_pin("https://news.ycombinator.com/", "https://news.ycombinator.com")
    assert _matches_pin(
        "https://news.ycombinator.com/news?p=1", "https://news.ycombinator.com"
    )
    assert not _matches_pin("https://news.ycombinator.co/", "https://news.ycombinator.com")


def test_matches_pin_empty_always_matches():
    assert _matches_pin("https://anywhere.com/", "")
    assert _matches_pin("", "")
