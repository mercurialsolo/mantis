"""Tests for the derive-first ObjectiveSpec producer (#224 Phase 2).

Pin the wiring contract: tool_use call shape, response coercion,
caching by plan text + prompt version, fallback to the heuristic
parser when the API key is missing or the call errors. The actual
LLM lift is exercised by integration tests against a real key —
these tests stub ``requests.post`` so they run offline.
"""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mantis_agent import objective as objective_mod
from mantis_agent.objective import (
    DERIVE_PROMPT_VERSION,
    ObjectiveSpec,
    OutputField,
    derive_from_plan,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    """Every test gets a fresh per-process cache."""
    objective_mod._reset_cache_for_tests()
    yield
    objective_mod._reset_cache_for_tests()


def _tool_response(input_payload: dict[str, Any]) -> MagicMock:
    """Build a stubbed Anthropic /v1/messages response in tool_use shape."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "content": [
            {
                "type": "tool_use",
                "name": "record_objective",
                "input": input_payload,
            }
        ]
    }
    return resp


# ── Module surface ───────────────────────────────────────────────────────


def test_module_re_exports_objective_dataclasses() -> None:
    """``from mantis_agent.objective import ObjectiveSpec`` must work
    so callers don't need to know the dataclasses physically live in
    the ``graph`` submodule."""
    from mantis_agent.objective import (
        CompletionCondition,
        ObjectiveSpec as ReExported,
        OutputField as ReExportedField,
    )
    from mantis_agent.graph.objective import ObjectiveSpec as Canonical

    assert ReExported is Canonical
    assert ReExportedField.__name__ == "OutputField"
    assert CompletionCondition.__name__ == "CompletionCondition"


# ── Successful tool_use lift ─────────────────────────────────────────────


def test_derive_lifts_target_entity_and_required_fields() -> None:
    """The plan calls phone REQUIRED for viability — derived schema
    must mark phone required and leave year non-required."""
    payload = {
        "target_entity": "boat listing",
        "domains": ["example.com"],
        "output_fields": [
            {"name": "phone", "type": "str", "required": True, "example": "(305) 555-1234"},
            {"name": "year", "type": "int", "required": False, "example": "2018"},
        ],
        "forbidden_actions": ["Contact Seller"],
        "allowed_reveal_actions": ["Show phone"],
        "viability_definition": "VIABLE only if a phone number is in the description.",
        "spam_text_indicators": [],
        "spam_seller_indicators": [],
        "spam_label": "",
    }
    with patch("requests.post", return_value=_tool_response(payload)):
        spec = derive_from_plan("plan body", api_key="k")

    assert isinstance(spec, ObjectiveSpec)
    assert spec.target_entity == "boat listing"
    assert spec.domains == ["example.com"]
    assert [(f.name, f.required) for f in spec.output_schema] == [
        ("phone", True),
        ("year", False),
    ]
    assert spec.forbidden_actions == ["Contact Seller"]
    assert spec.allowed_reveal_actions == ["Show phone"]
    assert "phone number" in spec.viability_definition
    # Plan didn't enumerate spam tokens — both lists must stay empty.
    assert spec.spam_text_indicators == []
    assert spec.spam_seller_indicators == []


def test_derive_drops_fields_with_empty_name() -> None:
    """Defensive: tool_use schema requires ``name`` but a malformed
    Anthropic response could still produce a stray entry. Coercion
    silently drops it rather than letting it pollute the schema."""
    payload = {
        "target_entity": "x",
        "output_fields": [
            {"name": "", "type": "str", "required": True},
            {"name": "title", "type": "str", "required": True},
        ],
    }
    with patch("requests.post", return_value=_tool_response(payload)):
        spec = derive_from_plan("plan", api_key="k")

    assert [f.name for f in spec.output_schema] == ["title"]


def test_derive_passes_tool_schema_to_anthropic() -> None:
    """Tool definition must include all derive-time fields and
    ``tool_choice`` must force the ``record_objective`` tool — that's
    what makes the response server-side-validated as JSON."""
    captured: dict[str, Any] = {}

    def _capture(url, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        return _tool_response({"target_entity": "x", "output_fields": []})

    with patch("requests.post", side_effect=_capture):
        derive_from_plan("plan", api_key="k")

    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    body = captured["json"]
    assert body["tool_choice"] == {"type": "tool", "name": "record_objective"}

    schema = body["tools"][0]["input_schema"]
    expected_keys = {
        "target_entity",
        "domains",
        "output_fields",
        "forbidden_actions",
        "allowed_reveal_actions",
        "viability_definition",
        "spam_text_indicators",
        "spam_seller_indicators",
        "spam_label",
    }
    assert set(schema["properties"].keys()) == expected_keys
    assert "target_entity" in schema["required"]
    assert "output_fields" in schema["required"]


# ── Caching ──────────────────────────────────────────────────────────────


def test_derive_caches_by_plan_text() -> None:
    """Same plan text → one Anthropic call. Repeated runs in the same
    process are free."""
    payload = {"target_entity": "x", "output_fields": []}
    mock = MagicMock(return_value=_tool_response(payload))
    with patch("requests.post", side_effect=mock):
        spec1 = derive_from_plan("identical plan body", api_key="k")
        spec2 = derive_from_plan("identical plan body", api_key="k")

    assert mock.call_count == 1
    assert spec1 is spec2


def test_derive_cache_distinguishes_different_plans() -> None:
    """Distinct plan texts → distinct cache entries."""
    payload1 = {"target_entity": "boat listing", "output_fields": []}
    payload2 = {"target_entity": "job posting", "output_fields": []}

    responses = [_tool_response(payload1), _tool_response(payload2)]
    mock = MagicMock(side_effect=responses)
    with patch("requests.post", side_effect=mock):
        spec1 = derive_from_plan("plan A", api_key="k")
        spec2 = derive_from_plan("plan B", api_key="k")

    assert mock.call_count == 2
    assert spec1.target_entity == "boat listing"
    assert spec2.target_entity == "job posting"


def test_derive_cache_key_uses_prompt_version() -> None:
    """Cache key must include ``DERIVE_PROMPT_VERSION`` so bumping
    the version invalidates every entry uniformly."""
    plan = "plan body"
    expected = hashlib.sha256(
        (DERIVE_PROMPT_VERSION + "\n" + plan).encode("utf-8")
    ).hexdigest()
    actual = objective_mod._cache_key(plan)
    assert actual == expected


# ── Fallback paths ───────────────────────────────────────────────────────


def test_derive_falls_back_when_api_key_missing(monkeypatch) -> None:
    """No API key → return heuristic ObjectiveSpec.parse output.
    Must NOT make any HTTP call."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with patch("requests.post") as mock:
        spec = derive_from_plan("Extract listings from https://example.com/things")

    mock.assert_not_called()
    assert isinstance(spec, ObjectiveSpec)
    # Heuristic picks up the URL.
    assert "example.com" in spec.domains


def test_derive_falls_back_on_api_error() -> None:
    """API non-200 → fall back to heuristic parser, don't raise."""
    err_resp = MagicMock(status_code=500, text="upstream timeout")
    with patch("requests.post", return_value=err_resp):
        spec = derive_from_plan("plan", api_key="k")

    assert isinstance(spec, ObjectiveSpec)
    # Heuristic returns a minimal spec.
    assert spec.raw_text == "plan"


def test_derive_falls_back_on_missing_tool_block() -> None:
    """Response is 200 but contains no tool_use block (model returned
    prose) — fall back rather than crashing."""
    resp = MagicMock(status_code=200)
    resp.json.return_value = {
        "content": [{"type": "text", "text": "I refuse to call the tool"}]
    }
    with patch("requests.post", return_value=resp):
        spec = derive_from_plan("plan", api_key="k")

    assert isinstance(spec, ObjectiveSpec)
    assert spec.raw_text == "plan"


def test_derive_falls_back_on_request_exception() -> None:
    """Network error mid-call → fall back, don't propagate."""
    with patch("requests.post", side_effect=ConnectionError("net down")):
        spec = derive_from_plan("plan", api_key="k")

    assert isinstance(spec, ObjectiveSpec)


# ── Integration with downstream consumers ───────────────────────────────


def test_derived_spec_feeds_extraction_schema() -> None:
    """End-to-end: ``derive_from_plan`` → ``ExtractionSchema.from_objective``
    must produce a usable schema. This is the canonical Phase 2/3 pipeline
    callers will write."""
    from mantis_agent.extraction import ExtractionSchema

    payload = {
        "target_entity": "property listing",
        "output_fields": [
            {"name": "address", "type": "str", "required": True, "example": "123 Main St"},
            {"name": "price", "type": "str", "required": True, "example": "$450,000"},
            {"name": "beds", "type": "int", "required": False, "example": "3"},
        ],
        "forbidden_actions": ["Contact Agent"],
        "allowed_reveal_actions": ["Show more"],
        "viability_definition": "Viable if address and price are visible",
        "spam_text_indicators": [],
        "spam_seller_indicators": [],
        "spam_label": "",
    }
    with patch("requests.post", return_value=_tool_response(payload)):
        spec = derive_from_plan("Find houses for sale on Zillow", api_key="k")

    schema = ExtractionSchema.from_objective(spec)

    assert schema.entity_name == "property listing"
    assert schema.required_fields == ["address", "price"]
    assert "Contact Agent" in schema.forbidden_controls
    assert "Show more" in schema.allowed_controls
    # Spam fields populated empty — recipe overlay is where production
    # tokens accumulate, not the derived spec.
    assert schema.spam_indicators == []


def test_derived_spec_overlays_with_marketplace_recipe() -> None:
    """Phase 1 + Phase 2 wired together: derive a property-flavoured
    spec, then overlay the marketplace_listings recipe — the recipe's
    spam vocabulary extends the (empty) derived list, but the derived
    schema body wins."""
    from mantis_agent import recipes
    from mantis_agent.extraction import ExtractionSchema

    payload = {
        "target_entity": "property listing",
        "output_fields": [
            {"name": "address", "type": "str", "required": True, "example": ""},
            {"name": "price", "type": "str", "required": True, "example": ""},
        ],
        "forbidden_actions": [],
        "allowed_reveal_actions": [],
        "viability_definition": "",
        "spam_text_indicators": [],
        "spam_seller_indicators": [],
        "spam_label": "",
    }
    with patch("requests.post", return_value=_tool_response(payload)):
        spec = derive_from_plan("plan", api_key="k")

    derived = ExtractionSchema.from_objective(spec)
    merged = derived.overlay(recipes.load_schema("marketplace_listings"))

    # Schema body from derive (NOT boat-listing's year/make).
    assert merged.entity_name == "property listing"
    assert merged.required_fields == ["address", "price"]
    # Recipe vocabulary extends the empty derived lists.
    assert "marinemax" in merged.spam_indicators
    assert "brokerage" in merged.spam_seller_indicators


# ── ObjectiveSpec round-trip ────────────────────────────────────────────


def test_objective_spec_round_trips_new_fields() -> None:
    """The new derive-time fields must round-trip through to_dict /
    from_dict so saved specs survive process boundaries."""
    spec = ObjectiveSpec(
        raw_text="plan",
        target_entity="boat listing",
        output_schema=[OutputField(name="phone", required=True)],
        viability_definition="phone present",
        spam_text_indicators=["dealer"],
        spam_seller_indicators=["llc"],
        spam_label="dealer",
    )
    restored = ObjectiveSpec.from_dict(spec.to_dict())
    assert restored.viability_definition == "phone present"
    assert restored.spam_text_indicators == ["dealer"]
    assert restored.spam_seller_indicators == ["llc"]
    assert restored.spam_label == "dealer"
