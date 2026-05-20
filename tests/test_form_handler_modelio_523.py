"""PR B-3 of #523: grounder-layer modelio capture in form handler.

`ClaudeGuidedFormHandler.execute()` wraps its dispatch in
`publish_modelio_context(augur, layer="grounding", step_index=...)`
so every `target_provider.find_form_target` / `find_target_by_affordance`
/ `verify_dropdown_value` call this handler issues lands a modelio
record under `modelio/<step>-grounding-<seq>.json`.

These tests exercise the wrap WITHOUT spinning up the whole runner —
asserting the contextvar state during the wrapped call is enough to
prove the wire fires correctly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("augur_sdk")

from mantis_agent.gym.step_handlers.form import ClaudeGuidedFormHandler
from mantis_agent.observability.augur import AugurAdapter
from mantis_agent.observability.modelio import current_modelio_context


def _make_step():
    from mantis_agent.plan_decomposer import MicroIntent
    return MicroIntent(
        intent="x", type="unsupported_type",  # forces the early "unknown" return path
        budget=1,
    )


def _make_ctx(env_stub: Any, extractor_stub: Any, index: int = 4):
    """Minimal StepContext with the fields execute() reads."""
    ctx = MagicMock()
    ctx.env = env_stub
    ctx.extractor = extractor_stub
    ctx.form_target_provider = extractor_stub
    ctx.state = {"index": index}
    return ctx


def test_execute_publishes_grounding_context_during_dispatch(
    monkeypatch, tmp_path: Path,
):
    """When execute() is called with an active augur, the grounding
    layer context is published for the duration of _dispatch — any
    LLM call inside picks it up."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    runner = MagicMock()
    runner._augur = AugurAdapter(
        run_id="grounding_v1", tenant_id="t", session_name="s", out_dir=tmp_path,
    )

    handler = ClaudeGuidedFormHandler(runner)
    captured: dict = {}

    def _spy_dispatch(self, step, ctx):
        captured["ctx"] = current_modelio_context()
        # Return early — we only care about the contextvar state.
        from mantis_agent.gym.checkpoint import StepResult
        return StepResult(step_index=int(ctx.state["index"]), intent="x", success=False)

    monkeypatch.setattr(ClaudeGuidedFormHandler, "_dispatch", _spy_dispatch)

    env = MagicMock()
    extractor = MagicMock()
    ctx = _make_ctx(env, extractor, index=4)
    handler.execute(_make_step(), ctx)

    runner._augur.close(status="completed")
    captured_ctx = captured["ctx"]
    assert captured_ctx is not None, (
        "Expected a published modelio context inside _dispatch, got None"
    )
    assert captured_ctx.layer == "grounding"
    assert captured_ctx.step_index == 4
    assert captured_ctx.augur is runner._augur


def test_execute_resets_context_after_dispatch(monkeypatch, tmp_path: Path):
    """Once _dispatch returns, the contextvar must be reset to its
    prior value — otherwise the next step's grounding calls (or
    cross-handler calls) would leak the previous step_index."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    runner = MagicMock()
    runner._augur = AugurAdapter(
        run_id="grounding_reset", tenant_id="t", session_name="s", out_dir=tmp_path,
    )
    handler = ClaudeGuidedFormHandler(runner)

    def _spy_dispatch(self, step, ctx):
        from mantis_agent.gym.checkpoint import StepResult
        return StepResult(step_index=int(ctx.state["index"]), intent="x", success=False)

    monkeypatch.setattr(ClaudeGuidedFormHandler, "_dispatch", _spy_dispatch)
    handler.execute(_make_step(), _make_ctx(MagicMock(), MagicMock()))
    runner._augur.close(status="completed")
    # After execute returns, no context is published.
    assert current_modelio_context() is None


def test_execute_noop_when_augur_inactive(monkeypatch, tmp_path: Path):
    """When the runner has no augur (None or disabled), execute()
    must still complete and _dispatch must still run — the
    contextmanager treats inactive augur as a clean no-op."""
    monkeypatch.setenv("MANTIS_AUGUR_DISABLED", "1")
    runner = MagicMock()
    runner._augur = AugurAdapter(
        run_id="grounding_noop", tenant_id="t", session_name="s", out_dir=tmp_path,
    )
    assert not runner._augur.active

    handler = ClaudeGuidedFormHandler(runner)
    captured: dict = {}

    def _spy_dispatch(self, step, ctx):
        captured["ctx"] = current_modelio_context()
        from mantis_agent.gym.checkpoint import StepResult
        return StepResult(step_index=int(ctx.state["index"]), intent="x", success=False)

    monkeypatch.setattr(ClaudeGuidedFormHandler, "_dispatch", _spy_dispatch)
    handler.execute(_make_step(), _make_ctx(MagicMock(), MagicMock()))
    # No context published because augur is inactive — the wrap is a no-op.
    assert captured["ctx"] is None


def test_execute_handles_runner_without_augur_attr(monkeypatch, tmp_path: Path):
    """Some MicroPlanRunner test fixtures don't set _augur at all.
    ``getattr(runner, '_augur', None)`` should return None and the
    wrap must still complete the dispatch."""
    runner = MagicMock(spec=[])  # No attributes
    handler = ClaudeGuidedFormHandler(runner)

    def _spy_dispatch(self, step, ctx):
        from mantis_agent.gym.checkpoint import StepResult
        return StepResult(step_index=int(ctx.state["index"]), intent="x", success=True)

    monkeypatch.setattr(ClaudeGuidedFormHandler, "_dispatch", _spy_dispatch)
    result = handler.execute(_make_step(), _make_ctx(MagicMock(), MagicMock()))
    assert result.success is True
