"""Tests for `tabs.*` + `links.peek_target` extensions (#779, #780, PR 4).

Same mock-server pattern as `tests/test_browser_use_state.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from mantis_agent.gym.browser_use_plane_client import BrowserUsePlaneClient
from mantis_agent.gym.browser_use_wire import (
    BrowserUseSessionInitResponse,
    LinksPeekTargetResponse,
    TabsActivateResponse,
    TabsCloseResponse,
    TabsOpenInNewResponse,
)
from mantis_agent.gym.compute_contract import (
    Capabilities,
    SupportsLinkPeek,
    SupportsTabs,
)


@dataclass
class _Resp:
    status_code: int = 200
    _json: dict[str, Any] = field(default_factory=dict)

    def json(self) -> dict[str, Any]:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeServer:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict[str, Any]]] = []
        self.next_tab_id = "tab-1"
        self.opened_url = "https://example.com/article"
        self.opened_title = "Example Article"
        self.peek_response: LinksPeekTargetResponse | None = None

    def post(self, url: str, **kw: Any) -> _Resp:
        body = kw.get("json", {})
        self.posts.append((url, body))
        if url.endswith("/session/init"):
            return _Resp(
                _json=BrowserUseSessionInitResponse(
                    session_token="tok",
                    capabilities=Capabilities.for_browser_use_plane().as_dict(),
                ).model_dump()
            )
        if url.endswith("/tabs/open_in_new"):
            return _Resp(
                _json=TabsOpenInNewResponse(
                    tab_id=self.next_tab_id,
                    url=self.opened_url,
                    title=self.opened_title,
                ).model_dump()
            )
        if url.endswith("/tabs/close"):
            return _Resp(_json=TabsCloseResponse(closed=True).model_dump())
        if url.endswith("/tabs/activate"):
            return _Resp(
                _json=TabsActivateResponse(
                    activated=True, url=self.opened_url
                ).model_dump()
            )
        if url.endswith("/links/peek_target"):
            pr = self.peek_response or LinksPeekTargetResponse(
                href="https://example.com/article", target=None, tag="a"
            )
            return _Resp(_json=pr.model_dump())
        return _Resp(status_code=404)

    def get(self, url: str, **kw: Any) -> _Resp:
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


def test_browser_use_client_satisfies_supports_tabs():
    server = _FakeServer()
    client = _make_client(server)
    assert isinstance(client, SupportsTabs)


def test_browser_use_client_satisfies_supports_link_peek():
    server = _FakeServer()
    client = _make_client(server)
    assert isinstance(client, SupportsLinkPeek)


# ── tabs.* methods ────────────────────────────────────────────────


def test_tabs_open_in_new_with_url():
    server = _FakeServer()
    client = _make_client(server)
    tab_id = client.tabs_open_in_new(url="https://example.com/")
    assert tab_id == "tab-1"
    posts = [p for u, p in server.posts if u.endswith("/tabs/open_in_new")]
    assert posts[0]["url"] == "https://example.com/"
    assert posts[0]["via_selector"] is None
    assert "step_id" in posts[0] and len(posts[0]["step_id"]) > 0


def test_tabs_open_in_new_via_selector():
    server = _FakeServer()
    client = _make_client(server)
    client.tabs_open_in_new(via_selector="a.titleline")
    posts = [p for u, p in server.posts if u.endswith("/tabs/open_in_new")]
    assert posts[0]["via_selector"] == "a.titleline"
    assert posts[0]["url"] is None


def test_tabs_close_sends_tab_id():
    server = _FakeServer()
    client = _make_client(server)
    client.tabs_close("tab-3")
    posts = [p for u, p in server.posts if u.endswith("/tabs/close")]
    assert posts[0]["tab_id"] == "tab-3"


def test_tabs_activate_updates_last_url():
    server = _FakeServer()
    server.opened_url = "https://different.com/page"
    client = _make_client(server)
    client.tabs_activate("tab-2")
    posts = [p for u, p in server.posts if u.endswith("/tabs/activate")]
    assert posts[0]["tab_id"] == "tab-2"
    # last_url is internal but should reflect the activate response.
    assert client._state.last_url == "https://different.com/page"


# ── links.peek_target ─────────────────────────────────────────────


def test_links_peek_target_with_selector_returns_href():
    server = _FakeServer()
    server.peek_response = LinksPeekTargetResponse(
        href="https://example.com/x", target=None, tag="a"
    )
    client = _make_client(server)
    href = client.links_peek_target("a.titleline")
    assert href == "https://example.com/x"
    posts = [p for u, p in server.posts if u.endswith("/links/peek_target")]
    assert posts[0]["selector"] == "a.titleline"
    assert posts[0]["bbox"] is None


def test_links_peek_target_with_bbox_returns_href():
    server = _FakeServer()
    server.peek_response = LinksPeekTargetResponse(
        href="https://bbox-target.com/", target="_blank", tag="a"
    )
    client = _make_client(server)
    href = client.links_peek_target((100, 200, 250, 220))
    assert href == "https://bbox-target.com/"
    posts = [p for u, p in server.posts if u.endswith("/links/peek_target")]
    assert posts[0]["bbox"] == [100, 200, 250, 220]
    assert posts[0]["selector"] is None


def test_links_peek_target_returns_none_when_not_anchor():
    server = _FakeServer()
    server.peek_response = LinksPeekTargetResponse(href=None, target=None, tag="div")
    client = _make_client(server)
    assert client.links_peek_target("div.not-a-link") is None


def test_links_peek_target_full_returns_target_attribute():
    server = _FakeServer()
    server.peek_response = LinksPeekTargetResponse(
        href="https://x.com/", target="_blank", tag="a"
    )
    client = _make_client(server)
    full = client.links_peek_target_full("a.external")
    assert full["target"] == "_blank"
    assert full["href"] == "https://x.com/"


def test_links_peek_target_rejects_bad_type():
    server = _FakeServer()
    client = _make_client(server)
    with pytest.raises(TypeError, match="expected str"):
        client.links_peek_target(42)  # type: ignore[arg-type]


# ── plan.schema.json wiring ───────────────────────────────────────


class TestPlanSchemaTargetRole:
    def test_schema_declares_capture_link_step_type(self):
        schema_path = (
            Path(__file__).parent.parent
            / "docs"
            / "reference"
            / "plan.schema.json"
        )
        schema = json.loads(schema_path.read_text())
        step_props = schema["$defs"]["Step"]["properties"]
        types = step_props["type"]["enum"]
        assert "capture_link_in_new_tab" in types

    def test_schema_declares_target_role_field(self):
        schema_path = (
            Path(__file__).parent.parent
            / "docs"
            / "reference"
            / "plan.schema.json"
        )
        schema = json.loads(schema_path.read_text())
        step_props = schema["$defs"]["Step"]["properties"]
        assert "target_role" in step_props
        assert "source_selector" in step_props
        assert step_props["emit"]["enum"] == ["current_url", "current_url_title"]

    def test_schema_declares_on_no_navigation_default(self):
        schema_path = (
            Path(__file__).parent.parent
            / "docs"
            / "reference"
            / "plan.schema.json"
        )
        schema = json.loads(schema_path.read_text())
        step_props = schema["$defs"]["Step"]["properties"]
        assert step_props["on_no_navigation"]["default"] == "skip"
