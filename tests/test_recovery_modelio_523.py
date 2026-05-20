"""PR B-5 of #523: step_recovery-layer modelio capture.

Two-piece wire because ``agentic_recovery`` uses raw ``requests.post``
(not the shared AnthropicToolUseClient that auto-captures):

1. ``agentic_recovery._call_recovery_tool`` — adds an inline
   ``record_anthropic_modelio`` call after a successful response.
   Fires only when a caller has published a layer context.
2. ``gym/step_recovery._try_agentic_recovery`` — wraps the
   ``analyse_failure_and_recover`` call in
   ``publish_modelio_context(layer="step_recovery", step_index=...)``
   so the inline wire at (1) sees an active context.

Judge layer (B-6 in the original campaign sketch) is N/A in Mantis —
no dedicated judge LLM call site exists. This PR closes the campaign.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("augur_sdk")

from mantis_agent.observability.augur import AugurAdapter
from mantis_agent.observability.modelio import publish_modelio_context


_FAKE_RECOVERY_RESPONSE: dict = {
    "id": "msg_recovery_001",
    "model": "claude-haiku-4-5-20251001",
    "role": "assistant",
    "stop_reason": "tool_use",
    "type": "message",
    "content": [
        {
            "type": "tool_use",
            "id": "tu_rec_001",
            "name": "record_recovery",
            "input": {
                "action": "retry",
                "rationale": "transient failure",
                "confidence": 0.7,
            },
        },
    ],
    "usage": {"input_tokens": 1_200, "output_tokens": 95},
}


# ── agentic_recovery._call_recovery_tool wire (inline capture) ──────────


def test_call_recovery_tool_emits_modelio_when_context_published(
    tmp_path: Path, monkeypatch,
):
    """End-to-end: with augur open + step_recovery context published,
    _call_recovery_tool writes a validated modelio record under
    modelio/<step>-step_recovery-<seq>.json."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    import mantis_agent.agentic_recovery as ar
    import requests as _requests

    def _fake_post(url, **kw):
        r = MagicMock()
        r.status_code = 200
        r.json.return_value = _FAKE_RECOVERY_RESPONSE
        return r

    monkeypatch.setattr(_requests, "post", _fake_post)

    a = AugurAdapter(
        run_id="recovery_e2e", tenant_id="t", session_name="s", out_dir=tmp_path,
    )
    step = MagicMock(intent="click submit", type="submit")
    with publish_modelio_context(a, layer="step_recovery", step_index=4):
        out = ar._call_recovery_tool(
            step=step,
            failure_data="no_state_change after click",
            screenshot=None,
            plan_context=["fill_field", "click submit"],
            attempts=2,
            api_key="test-key",
            model="claude-haiku-4-5-20251001",
            prior_hints=[],
        )
    a.close(status="completed")

    assert out == {"action": "retry", "rationale": "transient failure", "confidence": 0.7}

    files = sorted((tmp_path / "modelio").glob("*step_recovery*.json"))
    assert files, "_call_recovery_tool did not write a modelio record"
    record = json.loads(files[0].read_text())
    assert record["layer"] == "step_recovery"
    assert record["step_index"] == 4
    assert record["request"]["model"] == "claude-haiku-4-5-20251001"
    assert record["response"]["usage"]["prompt_tokens"] == 1_200
    assert record["response"]["usage"]["completion_tokens"] == 95
    # The tool_use block is captured verbatim under tool_calls.
    assert record["response"]["tool_calls"][0]["name"] == "record_recovery"


def test_call_recovery_tool_noop_without_context(tmp_path: Path, monkeypatch):
    """Default path — no caller publishes a context — must be a clean
    no-op. Bundle/modelio dir untouched."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    import mantis_agent.agentic_recovery as ar  # noqa: F401 — ensures module is loaded
    import requests as _requests
    monkeypatch.setattr(
        _requests, "post",
        lambda *a, **kw: MagicMock(
            status_code=200, json=lambda: _FAKE_RECOVERY_RESPONSE,
        ),
    )

    a = AugurAdapter(
        run_id="recovery_noop", tenant_id="t", session_name="s", out_dir=tmp_path,
    )
    step = MagicMock(intent="click submit", type="submit")
    # No publish_modelio_context wrap.
    ar._call_recovery_tool(
        step=step, failure_data="x", screenshot=None,
        plan_context=[], attempts=1,
        api_key="test-key", model="claude-haiku-4-5-20251001",
    )
    a.close(status="completed")
    modelio_dir = tmp_path / "modelio"
    if modelio_dir.exists():
        assert not list(modelio_dir.glob("*.json"))


def test_call_recovery_tool_capture_failure_does_not_break_recovery(
    tmp_path: Path, monkeypatch,
):
    """If record_anthropic_modelio raises (e.g. SDK schema bump),
    _call_recovery_tool must still return its parsed decision —
    telemetry never breaks recovery."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    import mantis_agent.agentic_recovery as ar  # noqa: F401 — ensures module is loaded
    import requests as _requests
    monkeypatch.setattr(
        _requests, "post",
        lambda *a, **kw: MagicMock(
            status_code=200, json=lambda: _FAKE_RECOVERY_RESPONSE,
        ),
    )
    import mantis_agent.observability.modelio as modelio_mod
    def _boom(**kwargs):
        raise RuntimeError("simulated SDK schema mismatch")
    monkeypatch.setattr(modelio_mod, "record_anthropic_modelio", _boom)

    a = AugurAdapter(
        run_id="recovery_err", tenant_id="t", session_name="s", out_dir=tmp_path,
    )
    step = MagicMock(intent="click submit", type="submit")
    with publish_modelio_context(a, layer="step_recovery", step_index=0):
        out = ar._call_recovery_tool(
            step=step, failure_data="x", screenshot=None,
            plan_context=[], attempts=1,
            api_key="test-key", model="claude-haiku-4-5-20251001",
        )
    a.close(status="completed")
    # Recovery decision still came through.
    assert out is not None
    assert out["action"] == "retry"


# ── step_recovery._try_agentic_recovery caller wrap (source-level) ──────


def test_step_recovery_wraps_analyse_call_in_publish_modelio_context():
    """``_try_agentic_recovery`` must wrap its
    ``analyse_failure_and_recover`` call in
    ``publish_modelio_context(layer="step_recovery", step_index=...)``
    so the inline wire in _call_recovery_tool fires."""
    from mantis_agent.gym.step_recovery import StepRecoveryPolicy
    src = inspect.getsource(StepRecoveryPolicy._try_agentic_recovery)
    assert "publish_modelio_context" in src, (
        "_try_agentic_recovery must wrap analyse_failure_and_recover in "
        "publish_modelio_context"
    )
    assert 'layer="step_recovery"' in src or "layer='step_recovery'" in src, (
        "The wrap must use layer='step_recovery' (one of the SDK enum literals)"
    )
    assert "step_index=step_index" in src, (
        "The wrap must thread step_index through"
    )
    # The publish_modelio_context must appear before the analyse call.
    pub_idx = src.index("publish_modelio_context")
    call_idx = src.index("analyse_failure_and_recover(")
    assert pub_idx < call_idx, (
        "publish_modelio_context must precede the analyse_failure_and_recover call"
    )
