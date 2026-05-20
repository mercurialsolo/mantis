"""PR B-2 of #523: planner-layer modelio capture in plan_decomposer.

Same shape contract as PR B-1's client wrapper:

* When a caller has published a layer context, decompose_text emits
  one modelio/<step>-planner-<seq>.json record.
* When no context has been published (the default), the call is a
  silent no-op — bundle shape is unchanged from main.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("augur_sdk")

from mantis_agent.observability.augur import AugurAdapter
from mantis_agent.observability.modelio import publish_modelio_context


_FAKE_PLANNER_RESPONSE: dict = {
    "id": "msg_planner_001",
    "model": "claude-opus-4-7",
    "role": "assistant",
    "stop_reason": "end_turn",
    "type": "message",
    "content": [
        {
            "type": "text",
            "text": '{"shapes": ["form"], "steps": ['
                    '{"intent": "Click Login", "type": "click", "budget": 3, "verify": "Login form visible"}'
                    ']}',
        },
    ],
    "usage": {"input_tokens": 800, "output_tokens": 120},
}


def _make_fake_response(payload: dict) -> MagicMock:
    """Build a fake requests.Response-like object."""
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = payload
    return r


def test_plan_decomposer_emits_modelio_when_context_published(
    tmp_path: Path, monkeypatch,
):
    """End-to-end: with augur open + a planner context published,
    decompose_text writes a validated modelio record under
    modelio/<step>-planner-<seq>.json. Confirms the planner-layer wire
    fires through the SDK's validate=True default."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    a = AugurAdapter(
        run_id="planner_e2e", tenant_id="t", session_name="s", out_dir=tmp_path,
    )

    # Patch out the network call.
    import mantis_agent.plan_decomposer as pd
    import requests as _requests
    monkeypatch.setattr(
        _requests, "post", lambda *a, **kw: _make_fake_response(_FAKE_PLANNER_RESPONSE),
    )

    decomposer = pd.PlanDecomposer(api_key="test-key", model="claude-opus-4-7")
    with publish_modelio_context(a, layer="planner", step_index=None):
        # Returns a MicroPlan; we only care about side-effects on the
        # bundle, not the parsed plan shape.
        decomposer.decompose_text("Click Login")
    a.close(status="completed")

    files = sorted((tmp_path / "modelio").glob("*planner*.json"))
    assert files, "plan_decomposer did not write a modelio record"
    record = json.loads(files[0].read_text())
    assert record["layer"] == "planner"
    # Run-scoped: step_index null OR omitted both pass the schema.
    assert record.get("step_index") in (None, 0)
    assert record["request"]["model"] == "claude-opus-4-7"
    # OpenAI usage shape (not Anthropic input_tokens / output_tokens)
    assert record["response"]["usage"]["prompt_tokens"] == 800
    assert record["response"]["usage"]["completion_tokens"] == 120
    # The text response is captured verbatim under response.text.
    assert "shapes" in record["response"]["text"]


def test_plan_decomposer_no_capture_without_context(tmp_path: Path, monkeypatch):
    """Default path — no caller publishes a context — must be a clean
    no-op. Bundle/modelio dir untouched. Critical: PR B-2 must NOT
    change today's on-disk shape for any existing CLI or server caller."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    a = AugurAdapter(
        run_id="planner_noop", tenant_id="t", session_name="s", out_dir=tmp_path,
    )

    import mantis_agent.plan_decomposer as pd
    import requests as _requests
    monkeypatch.setattr(
        _requests, "post", lambda *a, **kw: _make_fake_response(_FAKE_PLANNER_RESPONSE),
    )

    decomposer = pd.PlanDecomposer(api_key="test-key", model="claude-opus-4-7")
    # NO publish_modelio_context wrap.
    decomposer.decompose_text("Click Login")
    a.close(status="completed")

    modelio_dir = tmp_path / "modelio"
    if modelio_dir.exists():
        assert not list(modelio_dir.glob("*.json")), (
            "PR B-2 should not capture when no context is published — "
            "found leftover files in modelio/ on a no-context run"
        )


def test_plan_decomposer_capture_failure_does_not_break_decompose(
    tmp_path: Path, monkeypatch,
):
    """If the modelio capture raises (e.g. SDK schema bump), the
    decomposer must still return its parsed plan — per Augur spec §4.3,
    telemetry failures are non-fatal."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    a = AugurAdapter(
        run_id="planner_capture_err", tenant_id="t", session_name="s", out_dir=tmp_path,
    )

    import mantis_agent.plan_decomposer as pd
    import requests as _requests
    monkeypatch.setattr(
        _requests, "post", lambda *a, **kw: _make_fake_response(_FAKE_PLANNER_RESPONSE),
    )

    # Inject a broken record_anthropic_modelio that raises.
    import mantis_agent.observability.modelio as modelio_mod
    def _boom(**kwargs):
        raise RuntimeError("simulated SDK schema mismatch")
    monkeypatch.setattr(modelio_mod, "record_anthropic_modelio", _boom)

    decomposer = pd.PlanDecomposer(api_key="test-key", model="claude-opus-4-7")
    with publish_modelio_context(a, layer="planner", step_index=None):
        # MUST NOT raise — telemetry never breaks the decompose path.
        plan = decomposer.decompose_text("Click Login")
    a.close(status="completed")
    assert plan is not None
    assert plan.steps  # parsed successfully
