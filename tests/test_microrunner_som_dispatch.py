"""SoM-anchored click dispatch in MicroPlanRunner step handlers (#300 follow-up).

Both ``GymRunner`` (/v1/cua) and ``MicroPlanRunner`` (/v1/predict)
share a single SoM-anchor primitive — :func:`gym.som_dispatch.try_som_click`.
This module covers:

1. The shared helper in isolation (policy / capability / CDP-call
   gating).
2. The step-handler integration: when ``StepContext.routing_policy``
   has ``som_for_unstructured_clicks=True`` AND the env exposes
   ``cdp_click_at_point``, click handlers SoM-anchor their primary
   click and tag the StepResult with ``executor_backend="som"``.
3. The dispatch boundary in ``_runner_helpers.execute_step``:
   ``_stamp_backend`` reads ``ctx.state["_executor_backend"]`` and
   pins it onto whatever StepResult the handler returns, so the
   handler doesn't have to thread the tag through 10+ return
   points.
4. The result aggregate: :func:`mantis_agent.server_utils.build_micro_result`
   surfaces ``executor_backend_counts`` on /v1/predict, mirroring
   :class:`RunResult.executor_backend_counts` on /v1/cua.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from mantis_agent.gym._runner_helpers import _stamp_backend
from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.gym.runner import RoutingPolicy
from mantis_agent.gym.som_dispatch import (
    INPUT_LIKE_TAGS,
    is_input_like,
    probe_element_tag_at,
    try_som_click,
)


# ── try_som_click ──────────────────────────────────────────────────────


class _CdpEnv:
    """Minimal env exposing only ``cdp_click_at_point``."""

    def __init__(self, returns: list[bool] | None = None, raises: bool = False) -> None:
        self.calls: list[tuple[int, int]] = []
        self._returns = list(returns or [True])
        self._raises = raises

    def cdp_click_at_point(self, x: int, y: int) -> bool:
        self.calls.append((x, y))
        if self._raises:
            raise RuntimeError("boom")
        if self._returns:
            return self._returns.pop(0)
        return True


def test_try_som_click_policy_off_returns_false() -> None:
    """Default policy (``som_for_unstructured_clicks=False``) skips."""
    env = _CdpEnv()
    assert try_som_click(env, 10, 20, RoutingPolicy()) is False
    assert env.calls == [], "CDP must not be called when policy is off"


def test_try_som_click_no_policy_returns_false() -> None:
    """Caller passing ``policy=None`` skips — defensive against unwired
    step contexts where the runner forgot to set routing_policy."""
    env = _CdpEnv()
    assert try_som_click(env, 1, 1, None) is False
    assert env.calls == []


def test_try_som_click_env_without_cdp_returns_false() -> None:
    """No ``cdp_click_at_point`` on the env ⇒ capability gate kicks in."""
    env = SimpleNamespace()  # no cdp_click_at_point
    policy = RoutingPolicy(som_for_unstructured_clicks=True)
    assert try_som_click(env, 1, 1, policy) is False


def test_try_som_click_success() -> None:
    """Policy on + CDP capable + CDP returns True ⇒ helper returns True
    and the env was called with the requested point."""
    env = _CdpEnv(returns=[True])
    policy = RoutingPolicy(som_for_unstructured_clicks=True)
    assert try_som_click(env, 42, 7, policy) is True
    assert env.calls == [(42, 7)]


def test_try_som_click_cdp_returns_false() -> None:
    """CDP returns False (no element at point / JS threw) ⇒ helper False.
    Caller is expected to fall back to xdotool."""
    env = _CdpEnv(returns=[False])
    policy = RoutingPolicy(som_for_unstructured_clicks=True)
    assert try_som_click(env, 0, 0, policy) is False
    assert env.calls == [(0, 0)]


def test_try_som_click_swallows_exceptions() -> None:
    """CDP raising mustn't propagate — the caller's fallback runs."""
    env = _CdpEnv(raises=True)
    policy = RoutingPolicy(som_for_unstructured_clicks=True)
    assert try_som_click(env, 5, 5, policy) is False


# ── probe_element_tag_at / is_input_like ───────────────────────────────


class _CdpEvalEnv:
    """Env exposing only ``cdp_evaluate``."""

    def __init__(self, result: Any = None, raises: bool = False) -> None:
        self.calls: list[str] = []
        self._result = result
        self._raises = raises

    def cdp_evaluate(self, expression: str) -> Any:
        self.calls.append(expression)
        if self._raises:
            raise RuntimeError("boom")
        return self._result


def test_probe_element_tag_at_returns_none_when_env_lacks_cdp() -> None:
    """No ``cdp_evaluate`` attribute → can't probe; caller falls
    through to legacy behaviour. Keeps the Playwright path and tests
    that don't wire CDP from being blocked by the new guard."""
    env = object()
    assert probe_element_tag_at(env, 10, 20) is None


def test_probe_element_tag_at_returns_none_on_cdp_exception() -> None:
    """CDP raising must not propagate — guard returns None and the
    caller proceeds as if CDP weren't available."""
    env = _CdpEvalEnv(raises=True)
    assert probe_element_tag_at(env, 10, 20) is None


def test_probe_element_tag_at_returns_none_when_no_element_at_point() -> None:
    """``elementFromPoint`` returns null for off-viewport points →
    the JS expression returns None; guard surfaces that as None."""
    env = _CdpEvalEnv(result=None)
    assert probe_element_tag_at(env, 9999, 9999) is None


def test_probe_element_tag_at_normalises_tag_and_contenteditable() -> None:
    """Tag is upper-cased; contentEditable is coerced to bool. The
    contract callers depend on: a stable dict shape regardless of
    what Chrome's CDP returns under the hood."""
    env = _CdpEvalEnv(result={"tag": "input", "contentEditable": 0})
    out = probe_element_tag_at(env, 100, 200)
    assert out == {"tag": "INPUT", "contentEditable": False}

    env2 = _CdpEvalEnv(result={"tag": "div", "contentEditable": 1})
    out2 = probe_element_tag_at(env2, 100, 200)
    assert out2 == {"tag": "DIV", "contentEditable": True}


def test_probe_element_tag_at_subtracts_chrome_offset_in_js(monkeypatch) -> None:
    """#413: caller passes screen coordinates (matching what xdotool
    will click), but ``document.elementFromPoint`` uses CSS-viewport
    coordinates. The JS expression sent through CDP must subtract the
    chrome offset (``outerHeight - innerHeight``) so the tag-guard
    reads the same element xdotool will hit.

    Lock the JS shape literally — a future refactor that drops the
    offset would silently bring back the lu.ma tag-guard false-
    positive (#413 reproducer)."""
    env = _CdpEvalEnv(result={"tag": "INPUT", "contentEditable": False})
    probe_element_tag_at(env, 618, 588)
    assert len(env.calls) == 1
    js = env.calls[0]
    # Both the chrome-offset computation and the elementFromPoint
    # call with the corrected viewport y must be in the script.
    assert "window.outerHeight - window.innerHeight" in js
    assert "document.elementFromPoint" in js
    # Y is subtracted by chromeH (the variable name we used in
    # the JS); literal subtraction would also be acceptable but
    # using the variable keeps the math overflow-safe.
    assert "- chromeH" in js


def test_cdp_click_at_point_subtracts_chrome_offset_in_js() -> None:
    """#413 sibling: the SoM ``el.click()`` dispatcher in
    ``xdotool_env.cdp_click_at_point`` shares the same screen-vs-
    viewport bug. Today SoM defaults to off so the bug hasn't bitten
    production traffic, but enabling SoM with the old JS would have
    clicked the wrong element (typically the button 85 px below the
    intended target). Pin the chromeH-subtraction logic literally —
    the JS now also returns a diagnostic dict (outerHeight,
    innerHeight, elementFromPoint at both screen and viewport coords)
    so the SoM click path can be debugged from Modal logs without
    a separate diagnostic build."""
    from mantis_agent.gym.xdotool_env import XdotoolGymEnv
    instance = XdotoolGymEnv.__new__(XdotoolGymEnv)

    captured: list[str] = []
    instance.cdp_evaluate = lambda expr: captured.append(expr) or True  # type: ignore[assignment]
    instance.cdp_click_at_point(618, 588)

    assert len(captured) == 1
    js = captured[0]
    # chromeH = outerHeight - innerHeight (the rule that maps
    # screen-y → viewport-y) must still be in the script.
    assert "window.outerHeight" in js
    assert "window.innerHeight" in js
    assert "chromeH = Math.max(0, oh - ih)" in js
    assert "document.elementFromPoint" in js
    # Y is subtracted by chromeH before the elementFromPoint call.
    assert "vy = sy - chromeH" in js


def test_cdp_click_at_point_returns_ok_from_diagnostic_dict() -> None:
    """The JS now returns a dict ``{ok, outerHeight, innerHeight, ...}``;
    the method must surface ``ok`` as the bool return so callers and
    the existing ``try_som_click`` policy gate are unaffected."""
    from mantis_agent.gym.xdotool_env import XdotoolGymEnv
    instance = XdotoolGymEnv.__new__(XdotoolGymEnv)

    instance.cdp_evaluate = lambda _expr: {  # type: ignore[assignment]
        "ok": True, "outerHeight": 720, "innerHeight": 635, "chromeH": 85,
        "vx": 44, "vy": 305,
        "elv_tag": "A", "elv_text": "Contacted (7)",
        "els_tag": "A", "els_text": "Contacted (7)",
    }
    assert instance.cdp_click_at_point(44, 390) is True

    instance.cdp_evaluate = lambda _expr: {  # type: ignore[assignment]
        "ok": False, "outerHeight": 0, "innerHeight": 0, "chromeH": 0,
        "vx": 44, "vy": 390,
        "elv_tag": None, "elv_text": "",
        "els_tag": None, "els_text": "",
    }
    assert instance.cdp_click_at_point(44, 390) is False


def test_cdp_click_at_point_back_compat_scalar_result() -> None:
    """Pre-diagnostic deploys / mocks that return a scalar boolean
    still get the same bool surface (back-compat for callers /
    tests that don't yet return the rich dict)."""
    from mantis_agent.gym.xdotool_env import XdotoolGymEnv
    instance = XdotoolGymEnv.__new__(XdotoolGymEnv)

    instance.cdp_evaluate = lambda _expr: True  # type: ignore[assignment]
    assert instance.cdp_click_at_point(10, 20) is True
    instance.cdp_evaluate = lambda _expr: False  # type: ignore[assignment]
    assert instance.cdp_click_at_point(10, 20) is False
    instance.cdp_evaluate = lambda _expr: None  # type: ignore[assignment]
    assert instance.cdp_click_at_point(10, 20) is False


def test_cdp_click_at_point_swallows_cdp_exceptions() -> None:
    """CDP failures must not propagate — caller's xdotool fallback
    runs on ``False``."""
    from mantis_agent.gym.xdotool_env import XdotoolGymEnv
    instance = XdotoolGymEnv.__new__(XdotoolGymEnv)

    def _raise(_expr):
        raise RuntimeError("CDP unreachable")

    instance.cdp_evaluate = _raise  # type: ignore[assignment]
    assert instance.cdp_click_at_point(10, 20) is False


def test_is_input_like_allows_unknown_when_cdp_unavailable() -> None:
    """``None`` tag_info means CDP couldn't probe — don't block on
    inconclusive evidence. Preserves legacy behaviour where the guard
    can't run."""
    assert is_input_like(None) is True


def test_is_input_like_accepts_canonical_input_tags() -> None:
    for tag in INPUT_LIKE_TAGS:
        assert is_input_like({"tag": tag, "contentEditable": False}) is True


def test_is_input_like_accepts_contenteditable_div() -> None:
    """Rich-text editors (Quill, ProseMirror) are DIVs with
    ``contentEditable=true`` — they accept typed input legitimately."""
    assert is_input_like({"tag": "DIV", "contentEditable": True}) is True


def test_is_input_like_rejects_button() -> None:
    """The canonical lu.ma failure: the chosen point overlaps the
    form's submit button. Refuse so the caller can report
    ``form_target_not_input`` rather than dismissing the form."""
    assert is_input_like({"tag": "BUTTON", "contentEditable": False}) is False


def test_is_input_like_rejects_anchor_and_div() -> None:
    """Anchor tags (Lu.ma's 'Sign In' link in the corner of the
    registration modal) and plain DIVs (modal backdrop / overlay) both
    fail the allow-list. Anything outside INPUT/TEXTAREA/LABEL/CE goes
    through the form_target_not_input failure path."""
    assert is_input_like({"tag": "A", "contentEditable": False}) is False
    assert is_input_like({"tag": "DIV", "contentEditable": False}) is False


# ── _stamp_backend dispatch boundary ───────────────────────────────────


def test_stamp_backend_promotes_ctx_scratch() -> None:
    """``ctx.state['_executor_backend']`` lands on the StepResult."""
    ctx = SimpleNamespace(state={"_executor_backend": "som"})
    result = StepResult(step_index=0, intent="click", success=True)
    stamped = _stamp_backend(result, ctx)
    assert stamped.executor_backend == "som"
    # Same object — ``_stamp_backend`` mutates in place and returns it
    # so the handler-call expression remains a one-liner.
    assert stamped is result


def test_stamp_backend_preserves_handler_set_backend() -> None:
    """Handlers that already set ``executor_backend`` win over the scratch
    (defensive — a handler may want a different label than the helper's
    auto-set)."""
    ctx = SimpleNamespace(state={"_executor_backend": "vision"})
    result = StepResult(step_index=0, intent="x", success=True, executor_backend="som")
    _stamp_backend(result, ctx)
    assert result.executor_backend == "som"


def test_stamp_backend_no_scratch_no_change() -> None:
    """Step types that don't dispatch a routable click leave the scratch
    unset; the StepResult keeps the default empty backend."""
    ctx = SimpleNamespace(state={})
    result = StepResult(step_index=0, intent="navigate", success=True)
    _stamp_backend(result, ctx)
    assert result.executor_backend == ""


def test_stamp_backend_handles_none_ctx() -> None:
    """Defensive: a None ctx (legacy callers) doesn't crash."""
    result = StepResult(step_index=0, intent="x", success=True)
    _stamp_backend(result, None)
    assert result.executor_backend == ""


# ── StepResult round-trip preserves executor_backend ────────────────────


def test_step_result_persists_executor_backend() -> None:
    """``StepResult.to_dict`` → ``from_dict`` round-trips the new field
    so checkpoint persistence sees it across resume."""
    r = StepResult(
        step_index=3, intent="submit Login", success=True,
        executor_backend="som",
    )
    payload = r.to_dict()
    assert payload.get("executor_backend") == "som"
    restored = StepResult.from_dict(payload)
    assert restored.executor_backend == "som"


def test_step_result_default_backend_empty() -> None:
    """Legacy checkpoints without the field decode without crashing."""
    payload = {
        "step_index": 1, "intent": "old plan", "success": True,
        "data": "", "steps_used": 0, "duration": 0.0,
        "reversed": False, "skip": False, "skip_reason": None,
    }
    r = StepResult.from_dict(payload)
    assert r.executor_backend == ""


# ── build_micro_result aggregation ──────────────────────────────────────


def test_build_micro_result_aggregates_executor_backend_counts() -> None:
    """The per-step ``executor_backend`` tags roll up into
    ``executor_backend_counts`` on the run aggregate."""
    from mantis_agent.server_utils import build_micro_result

    @dataclass
    class _FakeRunner:
        _final_costs: dict = None
        _final_status: str = "success"

        def __post_init__(self) -> None:
            self._final_costs = {"status": "success"}

        def _successful_lead_data(self, results: list) -> list:
            return []

        def _lead_key(self, lead: Any) -> str:
            return ""

        def _lead_has_phone(self, lead: Any) -> bool:
            return False

        def dynamic_verification_report(self, *, status: str) -> dict:
            return {"status": status, "verdict": "ok", "totals": {}, "checks": []}

    results = [
        StepResult(step_index=0, intent="navigate", success=True, executor_backend=""),
        StepResult(step_index=1, intent="fill_field", success=True, executor_backend="som"),
        StepResult(step_index=2, intent="submit", success=True, executor_backend="som"),
        StepResult(step_index=3, intent="extract_data", success=True, executor_backend="vision"),
    ]
    aggregate = build_micro_result(
        runner=_FakeRunner(), step_results=results,
        run_id="r", provider="p", session_name="s", model_name="holo3",
        elapsed_seconds=1.0,
    )
    assert aggregate["executor_backend_counts"] == {"som": 2, "vision": 1}
    # step_details carries per-step tag too, for offline ablation analysis.
    assert aggregate["step_details"][1]["executor_backend"] == "som"
    assert aggregate["step_details"][3]["executor_backend"] == "vision"


# ── RoutingPolicy default plumb into MicroPlanRunner ───────────────────


def test_microplan_runner_picks_up_routing_policy_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``MicroPlanRunner.__init__`` resolves ``routing_policy=None``
    via :meth:`RoutingPolicy.from_env` so a deploy can flip
    ``MANTIS_ROUTE_SOM_CLICKS=enabled`` without code changes."""
    monkeypatch.setenv("MANTIS_ROUTE_SOM_CLICKS", "enabled")
    from mantis_agent.gym.micro_runner import MicroPlanRunner

    runner = MicroPlanRunner.__new__(MicroPlanRunner)
    # Build the minimum attribute surface the policy plumb-in needs;
    # we're not exercising any handler, just the wiring.
    runner.routing_policy = RoutingPolicy.from_env()
    assert runner.routing_policy.som_for_unstructured_clicks is True
