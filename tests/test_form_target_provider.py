"""Direct coverage for :class:`ClaudeFormTargetProvider` (#406).

The end-to-end behavior is already covered by:

- ``tests/test_extractor_tool_use_migration.py`` — the find_form_target
  / find_target_by_affordance routing
- ``tests/test_select_option_verify.py`` — verify_dropdown_value
- ``tests/test_primary_action_fallback.py`` — affordance fallback path
- ``tests/test_form_handler.py`` — the step handler that consumes the
  provider results

This file pins what the legacy tests can't see: the provider can be
constructed and exercised *without* a ClaudeExtractor instance —
proves the methods are genuinely independent of the extractor module
they used to live in. If someone re-couples the provider to the
extractor by accident, these tests fail.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PIL import Image

from mantis_agent._anthropic.client import AnthropicToolUseClient
from mantis_agent.form_targeting import (
    ClaudeFormTargetProvider,
    FormTargetProvider,
)


def _img() -> Image.Image:
    return Image.new("RGB", (32, 32), color=(255, 255, 255))


def _client_with_response(payload: dict | None) -> AnthropicToolUseClient:
    """Build a client whose call_with_tool_schema returns `payload`."""
    client = AnthropicToolUseClient(api_key="k", model="m", log_prefix="ClaudeFormTarget")
    client.call_with_tool_schema = MagicMock(return_value=payload)  # type: ignore[method-assign]
    return client


# ── Protocol identity ──────────────────────────────────────────────


def test_claude_provider_satisfies_form_target_protocol() -> None:
    """Provider satisfies the runtime-checkable Protocol. The form
    handler (and any future host integration) can declare the
    ``FormTargetProvider`` type and accept either implementation."""
    provider = ClaudeFormTargetProvider(_client_with_response(None))
    assert isinstance(provider, FormTargetProvider)


def test_provider_constructible_without_an_extractor() -> None:
    """The whole point of the #406 split: a caller can build a
    provider from a raw AnthropicToolUseClient — no ClaudeExtractor
    instance required. Locks in the decoupling so the methods don't
    drift back into the extractor."""
    client = AnthropicToolUseClient(api_key="k", model="m")
    prov = ClaudeFormTargetProvider(client)
    assert prov is not None


# ── find_form_target ───────────────────────────────────────────────


def test_find_form_target_returns_validated_dict() -> None:
    """Happy path: tool_use returns a click target with action=type;
    the provider passes it through with the documented shape."""
    payload = {
        "x": 100, "y": 200, "action": "type",
        "value": "alice", "label": "User ID",
    }
    provider = ClaudeFormTargetProvider(_client_with_response(payload))
    out = provider.find_form_target(
        _img(), "Click the user id field",
        target_label="user id", target_value="alice",
    )
    assert out == payload


def test_find_form_target_returns_none_when_not_found() -> None:
    """``action=not_found`` → ``None`` (caller treats as miss)."""
    payload = {
        "x": 0, "y": 0, "action": "not_found",
        "value": "", "label": "Cloudflare challenge",
    }
    provider = ClaudeFormTargetProvider(_client_with_response(payload))
    assert provider.find_form_target(_img(), "x") is None


def test_find_form_target_returns_none_on_zero_coords() -> None:
    """(0, 0) is the "no coordinates" sentinel — refuse it."""
    payload = {
        "x": 0, "y": 0, "action": "click",
        "value": "", "label": "noop",
    }
    provider = ClaudeFormTargetProvider(_client_with_response(payload))
    assert provider.find_form_target(_img(), "x") is None


def test_find_form_target_coerces_string_coords() -> None:
    """The model sometimes emits coords as strings with whitespace /
    trailing commas (canonical: ``"x": "296, "``). The provider must
    accept and coerce them rather than crashing the run."""
    payload = {
        "x": "296, ", "y": "412", "action": "click",
        "value": "", "label": "Sign in",
    }
    provider = ClaudeFormTargetProvider(_client_with_response(payload))
    out = provider.find_form_target(_img(), "Click Sign in")
    assert out is not None
    assert out["x"] == 296 and out["y"] == 412


# ── find_target_by_affordance ──────────────────────────────────────


def test_find_target_by_affordance_returns_validated_dict() -> None:
    payload = {
        "x": 50, "y": 60, "action": "type", "label": "Le titre",
    }
    provider = ClaudeFormTargetProvider(_client_with_response(payload))
    out = provider.find_target_by_affordance(_img(), "Fill the title")
    assert out is not None
    assert out["x"] == 50 and out["y"] == 60
    assert out["action"] == "type"
    assert out["value"] == ""  # affordance results don't carry a canonical value


def test_find_target_by_affordance_not_found_returns_none() -> None:
    payload = {"x": 0, "y": 0, "action": "not_found", "label": "blank"}
    provider = ClaudeFormTargetProvider(_client_with_response(payload))
    assert provider.find_target_by_affordance(_img(), "Fill the title") is None


# ── verify_dropdown_value ──────────────────────────────────────────


def test_verify_dropdown_value_matches() -> None:
    """Local matcher decides — LLM only reports observed."""
    payload = {"observed": "High"}
    provider = ClaudeFormTargetProvider(_client_with_response(payload))
    out = provider.verify_dropdown_value(_img(), "Priority", "High")
    assert out == {"matches": True, "observed": "High"}


def test_verify_dropdown_value_mismatch() -> None:
    payload = {"observed": "Critical"}
    provider = ClaudeFormTargetProvider(_client_with_response(payload))
    out = provider.verify_dropdown_value(_img(), "Priority", "High")
    assert out == {"matches": False, "observed": "Critical"}


def test_verify_dropdown_value_returns_none_on_api_failure() -> None:
    """Client returned None → provider returns None so the caller can
    treat it as 'could not verify; trust the click'."""
    provider = ClaudeFormTargetProvider(_client_with_response(None))
    assert provider.verify_dropdown_value(_img(), "Priority", "High") is None


# ── from_extractor factory ─────────────────────────────────────────


def test_from_extractor_shares_api_key_and_model() -> None:
    """The factory pulls api_key + model off the extractor's client and
    creates a new client with the form-target log prefix — so a single
    runner has one canonical Anthropic config rather than two
    independently-configured clients drifting apart."""
    from mantis_agent.extraction.extractor import ClaudeExtractor
    extractor = ClaudeExtractor(api_key="my-key", model="claude-opus-4-7")
    provider = ClaudeFormTargetProvider.from_extractor(extractor)
    assert provider._client.api_key == "my-key"
    assert provider._client.model == "claude-opus-4-7"
    # Different log prefix so transient-error log lines disambiguate
    # source (extraction vs grounding).
    assert provider._client._log_prefix == "ClaudeFormTarget"


def test_from_extractor_rejects_object_without_client() -> None:
    """Defensive guard — passing some arbitrary object that doesn't
    expose ``_client: AnthropicToolUseClient`` raises rather than
    silently constructing a broken provider."""
    class Faux:
        pass

    with pytest.raises(TypeError, match="must expose a _client"):
        ClaudeFormTargetProvider.from_extractor(Faux())
