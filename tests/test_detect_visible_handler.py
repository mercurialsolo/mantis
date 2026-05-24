"""DetectVisibleHandler unit tests (#643 stage 2).

Pins the vision-only conditional-step contract:

- detect_visible binds a boolean to ``runner._state_vars[out_var]``
  based on the extractor's verify_gate yes/no answer
- absent out_var → handler returns a skip envelope (no vision call)
- absent extractor → defaults the var to False so dependent steps
  skip safely (don't strand callers on a missing extractor)
- verify_gate exceptions → default to False + log warning (never
  propagate; the step itself is non-fatal)
- claude_extract cost counter incremented on each successful call
"""

from __future__ import annotations

from unittest.mock import MagicMock

from mantis_agent.gym.step_context import StepContext
from mantis_agent.gym.step_handlers.detect_visible import DetectVisibleHandler
from mantis_agent.plan_decomposer import MicroIntent


class _FakeRunner:
    def __init__(self) -> None:
        self.costs: dict[str, float] = {"claude_extract": 0}
        self._state_vars: dict[str, object] = {}


def _ctx(env, extractor) -> StepContext:
    return StepContext(
        env=env, brain=None, extractor=extractor,
        grounding=None, cost_meter=None,
        dynamic_verifier=MagicMock(), scanner=MagicMock(),
        site_config=MagicMock(), tool_channel=None,
        extraction_cache=None, state={"index": 5},
    )


def test_detects_visible_binds_true_when_verify_gate_passes():
    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.verify_gate.return_value = (True, "Show More toggle clearly visible below the description")
    env = MagicMock()
    env.screenshot.return_value = MagicMock()

    step = MicroIntent(
        intent="Is a 'Show More' toggle visible inside the Description block?",
        type="detect_visible", out_var="has_show_more",
    )
    result = DetectVisibleHandler(runner).execute(step, _ctx(env, extractor))

    assert result.success is True
    assert runner._state_vars == {"has_show_more": True}
    assert result.data == "detect_visible:has_show_more=True"
    assert runner.costs["claude_extract"] == 1


def test_detects_visible_binds_false_when_verify_gate_fails():
    """verify_gate returning (False, reason) is the canonical 'element
    not visible' signal — handler still returns success=True (the
    detection itself completed); the BOOLEAN is what dependent steps
    consume via their ``guard``."""
    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.verify_gate.return_value = (False, "No Show More toggle in viewport")
    env = MagicMock()
    env.screenshot.return_value = MagicMock()

    step = MicroIntent(
        intent="Is a 'Show More' toggle visible?",
        type="detect_visible", out_var="has_show_more",
    )
    result = DetectVisibleHandler(runner).execute(step, _ctx(env, extractor))

    # Detection itself succeeded — the result is "False", that's a
    # valid finding not a failure.
    assert result.success is True
    assert runner._state_vars == {"has_show_more": False}
    assert result.data == "detect_visible:has_show_more=False"


def test_missing_out_var_returns_skip_envelope_no_vision_call():
    """A detect_visible step with no out_var is malformed — the
    detected value would be unreachable. Skip without burning the
    vision call so the operator sees the misconfiguration cleanly."""
    runner = _FakeRunner()
    extractor = MagicMock()
    env = MagicMock()

    step = MicroIntent(intent="Look for X", type="detect_visible")
    result = DetectVisibleHandler(runner).execute(step, _ctx(env, extractor))

    assert result.success is False
    assert result.skip is True
    assert result.skip_reason == "detect_visible_no_out_var"
    # No vision call attempted.
    extractor.verify_gate.assert_not_called()
    env.screenshot.assert_not_called()


def test_no_extractor_defaults_var_to_false_safely():
    """When the extractor isn't wired (test stubs, headless paths),
    default the variable to False so dependent guarded steps skip
    cleanly rather than crash on a missing key."""
    runner = _FakeRunner()
    env = MagicMock()
    env.screenshot.return_value = MagicMock()

    step = MicroIntent(
        intent="x", type="detect_visible", out_var="has_show_more",
    )
    result = DetectVisibleHandler(runner).execute(step, _ctx(env, extractor=None))

    assert result.success is False
    assert runner._state_vars == {"has_show_more": False}
    assert "no_extractor" in result.data


def test_verify_gate_exception_defaults_to_false_no_raise():
    """If verify_gate raises (network blip, schema parse error), the
    handler swallows + defaults False. The runner never sees the
    exception — the step is non-fatal by design."""
    runner = _FakeRunner()
    extractor = MagicMock()
    extractor.verify_gate.side_effect = RuntimeError("haiku quota exceeded")
    env = MagicMock()
    env.screenshot.return_value = MagicMock()

    step = MicroIntent(
        intent="x", type="detect_visible", out_var="has_show_more",
    )
    result = DetectVisibleHandler(runner).execute(step, _ctx(env, extractor))

    assert result.success is False
    assert runner._state_vars == {"has_show_more": False}
    assert "exception" in result.data
