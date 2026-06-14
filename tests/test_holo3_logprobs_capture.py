"""RL-prep — Holo3 logprobs capture (request shaping + schema mapping).

vLLM only returns per-token logprobs when asked; the Augur capture
hook + session flag existed, but the Holo3 request never set
``logprobs:true``, so ``response.logprobs`` was always empty — blocking
PPO/GRPO (no behaviour-policy logprob → no importance ratio / KL).

Pins:
* The planner payload requests logprobs ONLY when MANTIS_CAPTURE_LOGPROBS
  is set, with top_logprobs gated on MANTIS_LOGPROBS_TOP_K.
* ``_extract_openai_logprobs`` maps the vendor block onto
  modelio.schema.json's ``response.logprobs`` shape (drops ``bytes``,
  renames ``top_logprobs`` → ``top_alternatives``) and the result
  validates against the in-repo schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mantis_agent.observability.modelio import (
    _extract_openai_logprobs,
    publish_modelio_context,
    record_openai_modelio,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "data/augur/test/schema/modelio.schema.json"


def _vendor_logprobs_response():
    """An OpenAI / vLLM chat-completions response carrying logprobs in
    the vendor shape (bytes + top_logprobs)."""
    return {
        "choices": [{
            "message": {"content": "click"},
            "finish_reason": "stop",
            "logprobs": {"content": [
                {
                    "token": "click", "logprob": -0.05, "bytes": [99, 108],
                    "top_logprobs": [
                        {"token": "click", "logprob": -0.05, "bytes": [99]},
                        {"token": "tap", "logprob": -3.1, "bytes": [116]},
                    ],
                },
                {"token": "()", "logprob": -0.5, "bytes": [40, 41], "top_logprobs": []},
            ]},
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }


# ── mapper shape ────────────────────────────────────────────────────


def test_extract_maps_vendor_block_to_schema_shape():
    out = _extract_openai_logprobs(_vendor_logprobs_response())
    assert out is not None and len(out) == 2
    first = out[0]
    # bytes dropped; logprob required; token kept.
    assert set(first) <= {"token", "token_id", "logprob", "top_alternatives"}
    assert "bytes" not in first
    assert first["logprob"] == -0.05
    assert first["token"] == "click"
    # top_logprobs → top_alternatives, each mapped (bytes dropped).
    assert first["top_alternatives"][1] == {"token": "tap", "logprob": -3.1}


def test_extract_returns_none_when_absent():
    assert _extract_openai_logprobs({"choices": [{"message": {"content": "x"}}]}) is None
    assert _extract_openai_logprobs({"choices": []}) is None
    assert _extract_openai_logprobs({}) is None


def test_mapped_logprobs_validate_against_inrepo_schema():
    """The mapped record must satisfy modelio.schema.json — guards the
    additionalProperties:false / top_alternatives rename."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text())

    fake = MagicMock()
    fake.active = True
    captured = {}
    fake.record_modelio = lambda rec, **kw: captured.update(rec)

    payload = {"model": "Holo3-35B-A3B", "messages": [], "max_tokens": 256}
    with publish_modelio_context(fake, layer="planner", step_index=0):
        record_openai_modelio(
            request_payload=payload,
            response_json=_vendor_logprobs_response(),
            duration_ms=12,
        )
    assert captured["response"]["logprobs"][0]["logprob"] == -0.05
    # Full record validates.
    jsonschema.validate(instance=captured, schema=schema)


# ── request shaping (gated on env) ──────────────────────────────────


def _make_brain():
    from mantis_agent.brain_holo3 import Holo3Brain
    b = Holo3Brain.__new__(Holo3Brain)
    b.base_url = "http://x/v1"
    b.model = "Holo3-35B-A3B"
    b.model_name = "Holo3-35B-A3B"
    b.max_tokens = 256
    b.temperature = 0.0
    b.use_tool_calling = False
    b.enable_thinking = False
    b.api_key = ""          # ``_headers`` is a computed property
    b.extra_headers = {}
    return b


def _drive_think(brain):
    """Run think() with the network + parsing stubbed, return the POSTed
    payload."""
    seen = {}

    class _Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": {"content": "wait"}}]}

    def _post(url, json=None, **kw):
        seen["payload"] = json
        return _Resp()

    with patch("mantis_agent.brain_holo3.requests.post", side_effect=_post), \
         patch.object(brain, "_build_messages", return_value=[]), \
         patch.object(brain, "_parse_response", return_value=MagicMock()), \
         patch.object(brain, "_record_modelio_if_active"):
        brain.think(frames=[], task="t", action_history=[], screen_size=(1, 1))
    return seen["payload"]


def test_payload_omits_logprobs_by_default(monkeypatch):
    monkeypatch.delenv("MANTIS_CAPTURE_LOGPROBS", raising=False)
    payload = _drive_think(_make_brain())
    assert "logprobs" not in payload
    assert "top_logprobs" not in payload


def test_payload_requests_logprobs_when_enabled(monkeypatch):
    monkeypatch.setenv("MANTIS_CAPTURE_LOGPROBS", "1")
    monkeypatch.delenv("MANTIS_LOGPROBS_TOP_K", raising=False)
    payload = _drive_think(_make_brain())
    assert payload["logprobs"] is True
    # top_logprobs omitted when k=0 (sampled-token logprob only).
    assert "top_logprobs" not in payload


def test_payload_includes_top_logprobs_when_k_set(monkeypatch):
    monkeypatch.setenv("MANTIS_CAPTURE_LOGPROBS", "1")
    monkeypatch.setenv("MANTIS_LOGPROBS_TOP_K", "5")
    payload = _drive_think(_make_brain())
    assert payload["logprobs"] is True
    assert payload["top_logprobs"] == 5
