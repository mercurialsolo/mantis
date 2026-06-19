"""#931 P2 — opt-in screenshot grounding for /v1/cua click precision.

Pure CUA is brain-only by default; ``ground_clicks: true`` refines the
brain's click coords with the screenshot grounding model (the CUA-clean fix
for small/ambiguous targets, vs DOM SoM). It is a no-op without an Anthropic
key — the run falls back to brain-only rather than hard-failing.
"""

from __future__ import annotations

from mantis_agent.api_schemas import PureCUARequest
from mantis_agent.baseten_server.runtime import _should_ground_cua_clicks


# ── Request schema ──────────────────────────────────────────────────────


def test_ground_clicks_defaults_false():
    req = PureCUARequest.model_validate({"instruction": "find the export button"})
    assert req.ground_clicks is False
    assert req.model_dump(exclude_none=True).get("ground_clicks") is False


def test_ground_clicks_accepted_and_forwarded():
    req = PureCUARequest.model_validate(
        {"instruction": "x", "ground_clicks": True}
    )
    assert req.ground_clicks is True
    assert req.model_dump(exclude_none=True)["ground_clicks"] is True


# ── Runtime grounding-selection resolution ──────────────────────────────


def test_grounds_when_opted_in_and_key_present():
    assert _should_ground_cua_clicks({"ground_clicks": True}, has_anthropic_key=True) is True


def test_no_ground_when_opted_in_but_no_key():
    # Fall back to brain-only rather than fail the run.
    assert _should_ground_cua_clicks({"ground_clicks": True}, has_anthropic_key=False) is False


def test_no_ground_by_default_even_with_key():
    assert _should_ground_cua_clicks({}, has_anthropic_key=True) is False
    assert _should_ground_cua_clicks({"ground_clicks": False}, has_anthropic_key=True) is False
