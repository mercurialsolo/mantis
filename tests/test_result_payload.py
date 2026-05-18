"""StepResult → result.json packaging — shared between local CLI and
the Modal entrypoint.

Successful steps stay slim; failed steps carry diagnostics. The
contract this pins is the failure shape — dashboards and post-mortem
tools branch on the presence of these fields."""

from __future__ import annotations

import base64

from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.gym.result_payload import pack_step


def _success(**overrides) -> StepResult:
    return StepResult(
        step_index=overrides.pop("step_index", 0),
        intent=overrides.pop("intent", "navigate"),
        success=True,
        data=overrides.pop("data", ""),
        duration=overrides.pop("duration", 1.23),
        steps_used=overrides.pop("steps_used", 1),
        **overrides,
    )


def _failure(**overrides) -> StepResult:
    return StepResult(
        step_index=overrides.pop("step_index", 3),
        intent=overrides.pop("intent", "fill"),
        success=False,
        data=overrides.pop("data", "fill_error: not found"),
        duration=overrides.pop("duration", 2.5),
        steps_used=overrides.pop("steps_used", 4),
        failure_class=overrides.pop("failure_class", "selector_miss"),
        final_url=overrides.pop("final_url", "https://example.com/form"),
        page_title=overrides.pop("page_title", "Sign up"),
        **overrides,
    )


def test_success_payload_stays_slim() -> None:
    out = pack_step(_success())
    assert set(out) == {"index", "intent", "success", "data", "duration", "steps_used"}
    assert out["success"] is True


def test_failure_payload_includes_diagnostics() -> None:
    out = pack_step(_failure())
    assert out["success"] is False
    assert out["failure_class"] == "selector_miss"
    assert out["final_url"] == "https://example.com/form"
    assert out["page_title"] == "Sign up"


def test_failure_payload_fallback_classifies_from_data() -> None:
    """An old StepResult that never got stamped by the executor should
    still surface a useful class — pack_step classifies from data +
    page_title as a fallback so the schema is self-describing."""
    out = pack_step(_failure(failure_class="", data="gate:FAIL:Error 403"))
    assert out["failure_class"] == "cf_challenge"


def test_failure_payload_unclassifiable_data_lands_in_unknown() -> None:
    out = pack_step(_failure(failure_class="", data="??", page_title=""))
    assert out["failure_class"] == "unknown"


def test_failure_payload_serializes_last_action() -> None:
    action = Action(
        action_type=ActionType.CLICK,
        params={"x": 100, "y": 200, "button": "left"},
        reasoning="click the 'Sign up' button",
    )
    out = pack_step(_failure(last_action=action))
    assert out["last_action"]["type"] == ActionType.CLICK.value
    assert out["last_action"]["params"] == {"x": 100, "y": 200, "button": "left"}
    assert out["last_action"]["reasoning"] == "click the 'Sign up' button"


def test_failure_payload_omits_last_action_when_none() -> None:
    out = pack_step(_failure(last_action=None))
    assert "last_action" not in out


# ── #419: brain reasoning lands on every step ──────────────────────────


def test_success_payload_carries_reasoning_when_set() -> None:
    """#419: the audit triple needs reasoning on success steps too,
    not only failures. Otherwise post-mortem can't see the chain of
    thought that LED to the failed step — just the final one."""
    out = pack_step(_success(reasoning="I clicked the 'Continue' button because..."))
    assert out["reasoning"] == "I clicked the 'Continue' button because..."


def test_success_payload_omits_reasoning_when_empty() -> None:
    """Handlers that don't drive a brain (navigate / paginate / form
    fill / gate) leave reasoning empty — the key must stay out so
    success steps stay slim."""
    out = pack_step(_success())
    assert "reasoning" not in out


def test_failure_payload_carries_reasoning_alongside_last_action() -> None:
    """Reasoning is the chain-of-thought for this step (every brain
    iteration); last_action.reasoning is the FINAL action's reasoning
    only. Both can be present and they don't redundant — they cover
    different audit axes."""
    action = Action(
        action_type=ActionType.CLICK,
        params={"x": 100, "y": 200, "button": "left"},
        reasoning="final action: click submit",
    )
    out = pack_step(_failure(
        last_action=action,
        reasoning="step-level chain: tried scroll, then refocused, then clicked",
    ))
    assert out["reasoning"] == (
        "step-level chain: tried scroll, then refocused, then clicked"
    )
    assert out["last_action"]["reasoning"] == "final action: click submit"


def test_failure_payload_base64_encodes_screenshot() -> None:
    png_bytes = b"\x89PNG\r\n\x1a\nfake-png-bytes"
    out = pack_step(_failure(screenshot_png=png_bytes))
    assert "screenshot_b64" in out
    assert base64.b64decode(out["screenshot_b64"]) == png_bytes


def test_failure_payload_omits_screenshot_when_absent() -> None:
    out = pack_step(_failure(screenshot_png=None))
    assert "screenshot_b64" not in out


def test_success_payload_omits_screenshot_even_if_present() -> None:
    """Successful steps already get capped by
    ``_enforce_screenshot_cap``; result.json shouldn't carry them."""
    out = pack_step(_success(screenshot_png=b"big-success-png"))
    assert "screenshot_b64" not in out
    assert "failure_class" not in out


# ── Epic #362 Phase B: per-step time_breakdown ──────────────────────────


def test_time_breakdown_landed_when_provided() -> None:
    """Successful step + a TimeMeter breakdown dict → the payload
    surfaces ``time_breakdown`` rounded to 3dp."""
    bd = {"act": 1.2345, "think": 2.5}
    out = pack_step(_success(), time_breakdown=bd)
    assert out["time_breakdown"] == {"act": 1.234, "think": 2.5}


def test_time_breakdown_omitted_when_caller_passes_none() -> None:
    """Legacy callers that don't have a TimeMeter must not see a
    ``time_breakdown`` key — keeps pack_step backward-compatible."""
    out = pack_step(_success())
    assert "time_breakdown" not in out


def test_time_breakdown_present_on_failed_payloads() -> None:
    bd = {"act": 0.5, "settle": 1.0}
    out = pack_step(_failure(), time_breakdown=bd)
    assert out["time_breakdown"] == {"act": 0.5, "settle": 1.0}
    # Failure diagnostics still land alongside.
    assert out["failure_class"] == "selector_miss"
