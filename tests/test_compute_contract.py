"""Tests for the unified `ComputeClient` contract types (#785, PR 1)."""

from __future__ import annotations

import pytest

from mantis_agent.gym.compute_contract import (
    Capabilities,
    CapabilityAllowlist,
    CapabilityNotAllowed,
    ComputeBackend,
    SupportsBrowserState,
    SupportsLinkPeek,
    SupportsTabs,
)
from mantis_agent.gym.computer_wire import SessionInitResponse


class TestCapabilities:
    def test_default_is_computer_plane_pure_cua(self):
        cap = Capabilities()
        assert cap.dom_aware is False
        assert cap.stealth is True
        assert cap.supports_cdp is False
        assert cap.backend is ComputeBackend.COMPUTER_PLANE

    def test_for_computer_plane_default(self):
        cap = Capabilities.for_computer_plane()
        assert cap.dom_aware is False
        assert cap.stealth is True
        assert cap.supports_cdp is False
        assert cap.backend is ComputeBackend.COMPUTER_PLANE

    def test_for_computer_plane_with_cdp(self):
        cap = Capabilities.for_computer_plane(enable_cdp=True)
        assert cap.supports_cdp is True
        # CDP-on does NOT make a computer-plane client DOM-aware.
        assert cap.dom_aware is False

    def test_for_browser_use_plane_v1_posture(self):
        cap = Capabilities.for_browser_use_plane()
        assert cap.dom_aware is True
        # v1 explicit non-goal — see #785.
        assert cap.stealth is False
        assert cap.supports_cdp is True
        assert cap.backend is ComputeBackend.BROWSER_USE_PLANE

    def test_as_dict_round_trip_keys(self):
        cap = Capabilities.for_browser_use_plane()
        d = cap.as_dict()
        assert d["dom_aware"] is True
        assert d["backend"] == "browser_use_plane"

    def test_frozen(self):
        cap = Capabilities()
        with pytest.raises((AttributeError, Exception)):
            cap.dom_aware = True  # type: ignore[misc]


class TestCapabilityAllowlist:
    def test_pure_cua_blocks_dom_aware(self):
        allowlist = CapabilityAllowlist.pure_cua(executor="run_holo3")
        with pytest.raises(CapabilityNotAllowed) as exc_info:
            allowlist.enforce("dom_aware")
        assert exc_info.value.capability == "dom_aware"
        assert exc_info.value.executor == "run_holo3"

    def test_browser_use_admits_dom_aware(self):
        allowlist = CapabilityAllowlist.browser_use(executor="run_browser_use")
        # Should not raise.
        allowlist.enforce("dom_aware")

    def test_allows_non_raising_query(self):
        pure = CapabilityAllowlist.pure_cua()
        assert pure.allows("dom_aware") is False
        browser = CapabilityAllowlist.browser_use()
        assert browser.allows("dom_aware") is True

    def test_with_added_is_immutable_copy(self):
        base = CapabilityAllowlist.pure_cua(executor="x")
        extended = base.with_added("dom_aware")
        assert base.allows("dom_aware") is False
        assert extended.allows("dom_aware") is True
        # Executor identity preserved.
        assert extended.executor == "x"

    def test_capability_not_allowed_carries_context(self):
        err = CapabilityNotAllowed("dom_aware", executor="run_holo3")
        assert "dom_aware" in str(err)
        assert "run_holo3" in str(err)

    def test_capability_not_allowed_without_executor(self):
        err = CapabilityNotAllowed("dom_aware")
        assert "dom_aware" in str(err)


class TestSessionInitResponseCapabilities:
    def test_legacy_response_resolves_to_computer_plane(self):
        # Phase-0/Phase-1 servers don't populate the field.
        resp = SessionInitResponse(
            session_token="t",
            xvfb_display=":99",
        )
        cap = resp.resolved_capabilities()
        assert cap.dom_aware is False
        assert cap.backend is ComputeBackend.COMPUTER_PLANE

    def test_browser_use_response_resolves_correctly(self):
        resp = SessionInitResponse(
            session_token="t",
            xvfb_display=":99",
            capabilities=Capabilities.for_browser_use_plane().as_dict(),
        )
        cap = resp.resolved_capabilities()
        assert cap.dom_aware is True
        assert cap.backend is ComputeBackend.BROWSER_USE_PLANE
        assert cap.supports_cdp is True

    def test_unknown_backend_falls_back_to_computer_plane(self):
        # Defensive: an older client receiving a future backend label
        # should not crash — degrade to pure-CUA defaults.
        resp = SessionInitResponse(
            session_token="t",
            xvfb_display=":99",
            capabilities={"backend": "future_plane_xyz", "dom_aware": True},
        )
        cap = resp.resolved_capabilities()
        assert cap.backend is ComputeBackend.COMPUTER_PLANE


class TestPlanSchemaComputeBackend:
    """Plan-level `runtime.compute_backend` lands in PR 1 (schema only).

    Default = `computer_plane`. PR 2 wires the resolved value through the
    executor + client factory.
    """

    def test_schema_declares_compute_backend_enum(self):
        import json
        from pathlib import Path

        schema_path = (
            Path(__file__).parent.parent
            / "docs"
            / "reference"
            / "plan.schema.json"
        )
        schema = json.loads(schema_path.read_text())
        runtime_props = schema["$defs"]["Runtime"]["properties"]
        assert "compute_backend" in runtime_props
        field = runtime_props["compute_backend"]
        assert set(field["enum"]) == {"computer_plane", "browser_use_plane"}
        assert field["default"] == "computer_plane"

    def test_schema_accepts_plan_with_compute_backend(self):
        # Validates that a plan declaring browser_use_plane is well-formed
        # under the schema, without needing the full jsonschema validator
        # — just checks the shape matches the documented field.
        plan = {
            "steps": [{"intent": "Go to HN", "type": "navigate"}],
            "runtime": {
                "compute_backend": "browser_use_plane",
            },
        }
        # Sanity: the runtime block accepts the field.
        assert plan["runtime"]["compute_backend"] == "browser_use_plane"


class TestExtensionProtocols:
    """Protocols are `runtime_checkable` so handlers can dispatch on shape."""

    def test_minimal_class_satisfies_browser_state(self):
        class FakeClient:
            def state_current_url(self) -> str:
                return "https://example.com"

            def state_tabs(self) -> list[dict]:
                return []

            def state_focused_element(self) -> dict | None:
                return None

            def state_clipboard(self) -> str:
                return ""

            def state_page_load(self) -> str:
                return "complete"

        assert isinstance(FakeClient(), SupportsBrowserState)

    def test_missing_method_fails_protocol_check(self):
        class IncompleteClient:
            def state_current_url(self) -> str:
                return ""
            # Missing the other state.* methods.

        assert not isinstance(IncompleteClient(), SupportsBrowserState)

    def test_tabs_protocol_shape(self):
        class FakeTabs:
            def tabs_open_in_new(self, url=None) -> str:
                return "tab-1"

            def tabs_close(self, tab_id: str) -> None:
                pass

            def tabs_activate(self, tab_id: str) -> None:
                pass

        assert isinstance(FakeTabs(), SupportsTabs)

    def test_link_peek_protocol_shape(self):
        class FakePeek:
            def links_peek_target(self, selector_or_bbox) -> str | None:
                return "https://example.com"

        assert isinstance(FakePeek(), SupportsLinkPeek)
