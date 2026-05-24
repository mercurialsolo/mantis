"""MechanicalScrollHandler unit tests — (a) Layer-1 fix.

Pins the deterministic-dispatch contract for ``scroll`` steps that
specify an explicit ``params.count``. Routes around the brain so
Holo3 can't fall back to clicking visible elements when its inner
scroll loop doesn't observe page change (the failure mode behind
boattrader run 20260521_042509_b358f06f).

Coverage:
  - applies_to gating (count present + positive → True; missing,
    zero, negative, non-numeric → False)
  - dispatch count = N produces N env.step(SCROLL) calls
  - direction param honored (down/up; bogus values normalised to down)
  - scrollY readback verifies motion (delta >= 50px → success)
  - scrollY readback flags no-movement deterministically
    (failure_class='scroll_no_movement' instead of brain_loop_exhausted)
  - missing cdp_evaluate (no CDP) → success defers to env.step not raising
"""

from __future__ import annotations

from unittest.mock import MagicMock

from mantis_agent.actions import ActionType
from mantis_agent.gym.step_handlers.scroll import MechanicalScrollHandler
from mantis_agent.plan_decomposer import MicroIntent


def _ctx(env: MagicMock, index: int = 0):
    ctx = MagicMock()
    ctx.env = env
    ctx.state = {"index": index}
    return ctx


def _env_with_scroll(pre: float = 0.0, post: float = 600.0) -> MagicMock:
    env = MagicMock()
    # CDP read returns pre on first call, post on second (verification).
    env.cdp_evaluate = MagicMock(side_effect=[pre, post])
    return env


def test_applies_to_requires_positive_count():
    h = MechanicalScrollHandler(MagicMock())
    yes = MicroIntent(intent="x", type="scroll", params={"count": 1})
    assert h.applies_to(yes) is True


def test_applies_to_rejects_missing_count():
    h = MechanicalScrollHandler(MagicMock())
    no = MicroIntent(intent="scroll until X visible", type="scroll", params={})
    assert h.applies_to(no) is False


def test_applies_to_rejects_zero_count():
    h = MechanicalScrollHandler(MagicMock())
    no = MicroIntent(intent="x", type="scroll", params={"count": 0})
    assert h.applies_to(no) is False


def test_applies_to_rejects_non_numeric_count():
    h = MechanicalScrollHandler(MagicMock())
    no = MicroIntent(intent="x", type="scroll", params={"count": "many"})
    assert h.applies_to(no) is False


def test_dispatch_count_n_makes_n_env_step_calls():
    """count=3 → 3 env.step(SCROLL) dispatches."""
    env = _env_with_scroll(pre=0.0, post=2000.0)
    h = MechanicalScrollHandler(MagicMock())
    h.execute(
        MicroIntent(intent="x", type="scroll", params={"count": 3}),
        _ctx(env),
    )
    assert env.step.call_count == 3
    # Each call dispatched a SCROLL action.
    for call in env.step.call_args_list:
        action = call.args[0]
        assert action.action_type == ActionType.SCROLL


def test_direction_down_default():
    env = _env_with_scroll(pre=0.0, post=600.0)
    h = MechanicalScrollHandler(MagicMock())
    h.execute(
        MicroIntent(intent="x", type="scroll", params={"count": 1}),
        _ctx(env),
    )
    action = env.step.call_args.args[0]
    assert action.params["direction"] == "down"


def test_direction_up_honored():
    # Going up: pre=1000, post=400 → delta_up = 600
    env = MagicMock()
    env.cdp_evaluate = MagicMock(side_effect=[1000.0, 400.0])
    h = MechanicalScrollHandler(MagicMock())
    res = h.execute(
        MicroIntent(intent="x", type="scroll", params={"count": 1, "direction": "up"}),
        _ctx(env),
    )
    action = env.step.call_args.args[0]
    assert action.params["direction"] == "up"
    assert res.success is True


def test_direction_bogus_normalised_to_down():
    env = _env_with_scroll(pre=0.0, post=600.0)
    h = MechanicalScrollHandler(MagicMock())
    h.execute(
        MicroIntent(
            intent="x", type="scroll",
            params={"count": 1, "direction": "sideways"},
        ),
        _ctx(env),
    )
    assert env.step.call_args.args[0].params["direction"] == "down"


def test_scroll_motion_verified_via_cdp_readback():
    """When CDP confirms scrollY moved by >= 50px in the requested
    direction, success=True with deterministic data."""
    env = _env_with_scroll(pre=100.0, post=900.0)  # delta=800 >> 50
    h = MechanicalScrollHandler(MagicMock())
    res = h.execute(
        MicroIntent(intent="x", type="scroll", params={"count": 1}),
        _ctx(env),
    )
    assert res.success is True
    assert "scroll:down" in res.data


def test_no_movement_fails_with_scroll_no_movement_class():
    """When the page doesn't move (overflow:hidden body, sub-element
    scroller swallowing wheel, page already at top/bottom), the
    handler fails deterministically with
    failure_class='scroll_no_movement'. This is more actionable to
    the recovery layer than the previous brain_loop_exhausted."""
    env = _env_with_scroll(pre=500.0, post=500.0)  # no motion
    h = MechanicalScrollHandler(MagicMock())
    res = h.execute(
        MicroIntent(intent="x", type="scroll", params={"count": 1}),
        _ctx(env),
    )
    assert res.success is False
    assert res.failure_class == "scroll_no_movement"
    assert "pre=500" in res.data
    assert "post=500" in res.data


def test_below_50px_threshold_counts_as_no_movement():
    """Edge case: a 30-pixel scroll delta is below the meaningful-
    motion threshold (the page may have shifted by a CSS animation,
    not by our scroll dispatch). Treated as no movement."""
    env = _env_with_scroll(pre=100.0, post=130.0)  # delta=30 < 50
    h = MechanicalScrollHandler(MagicMock())
    res = h.execute(
        MicroIntent(intent="x", type="scroll", params={"count": 1}),
        _ctx(env),
    )
    assert res.success is False
    assert res.failure_class == "scroll_no_movement"


def test_missing_cdp_evaluate_skips_verification():
    """Test stub env without ``cdp_evaluate``: success defers to
    env.step() not raising. Keeps the handler testable in environments
    that don't mount Chrome (CI / unit tests)."""
    env = MagicMock(spec=["step"])  # no cdp_evaluate attribute
    h = MechanicalScrollHandler(MagicMock())
    res = h.execute(
        MicroIntent(intent="x", type="scroll", params={"count": 2}),
        _ctx(env),
    )
    assert res.success is True
    assert env.step.call_count == 2


def test_env_step_failure_returns_dispatch_error():
    """If env.step raises (xdotool crashed, container shutting
    down, etc.), the handler returns a deterministic failure with
    failure_class='scroll_dispatch_error' rather than propagating."""
    env = MagicMock()
    env.cdp_evaluate = MagicMock(return_value=0.0)
    env.step.side_effect = OSError("xdotool not found")
    h = MechanicalScrollHandler(MagicMock())
    res = h.execute(
        MicroIntent(intent="x", type="scroll", params={"count": 1}),
        _ctx(env),
    )
    assert res.success is False
    assert res.failure_class == "scroll_dispatch_error"


def test_notches_per_count_param_passes_through_to_env():
    """The plan can tune wheel notches per count via
    ``params.notches_per_count``. Default is 3 (env-level default)."""
    env = _env_with_scroll(pre=0.0, post=600.0)
    h = MechanicalScrollHandler(MagicMock())
    h.execute(
        MicroIntent(
            intent="x", type="scroll",
            params={"count": 1, "notches_per_count": 5},
        ),
        _ctx(env),
    )
    assert env.step.call_args.args[0].params["amount"] == 5


# ── #643 follow-up: CDP backend (skips Chrome wheel handlers entirely) ──


def test_applies_to_when_backend_is_cdp_even_without_count():
    """``params.backend == "cdp"`` activates the mechanical handler
    even when no count is specified — operator pinning the CDP backend
    is enough to opt out of vision scroll."""
    h = MechanicalScrollHandler(MagicMock())
    yes = MicroIntent(
        intent="scroll", type="scroll", params={"backend": "cdp"},
    )
    assert h.applies_to(yes) is True


def test_applies_to_when_hint_prefer_cdp_scroll_set():
    """``hints.prefer_cdp_scroll == True`` is the plan-author / hint-
    memory channel for opting into CDP scroll. Same activation as the
    explicit ``params.backend``."""
    h = MechanicalScrollHandler(MagicMock())
    yes = MicroIntent(
        intent="scroll", type="scroll", hints={"prefer_cdp_scroll": True},
    )
    assert h.applies_to(yes) is True


def test_cdp_backend_dispatches_scrollBy_js_per_count():
    """``params.backend=cdp count=3`` → 3 cdp_evaluate calls that
    each carry the scrollBy + scrollingElement.scrollBy + KeyboardEvent
    payload (the same triple-prong dispatch used by step_recovery.py).
    No env.step (xdotool) calls fire."""
    env = MagicMock()
    # 1 pre-scroll readback + 3 dispatch calls + 1 post-scroll readback = 5
    env.cdp_evaluate = MagicMock(side_effect=[0.0, None, None, None, 2000.0])
    h = MechanicalScrollHandler(MagicMock())
    res = h.execute(
        MicroIntent(
            intent="x", type="scroll",
            params={"count": 3, "backend": "cdp"},
        ),
        _ctx(env),
    )
    assert res.success is True
    # xdotool path is dormant.
    assert env.step.call_count == 0
    # 1 pre-readback + 3 dispatches + 1 post-readback.
    assert env.cdp_evaluate.call_count == 5
    # Inspect one of the dispatch calls to confirm the JS payload.
    dispatch_js = env.cdp_evaluate.call_args_list[1].args[0]
    assert "window.scrollBy" in dispatch_js
    assert "scrollingElement" in dispatch_js
    assert "PageDown" in dispatch_js


def test_cdp_backend_direction_up_uses_pageup_and_negative_height():
    env = MagicMock()
    env.cdp_evaluate = MagicMock(side_effect=[1000.0, None, 400.0])
    h = MechanicalScrollHandler(MagicMock())
    res = h.execute(
        MicroIntent(
            intent="x", type="scroll",
            params={"count": 1, "backend": "cdp", "direction": "up"},
        ),
        _ctx(env),
    )
    assert res.success is True
    dispatch_js = env.cdp_evaluate.call_args_list[1].args[0]
    assert "-window.innerHeight" in dispatch_js
    assert "PageUp" in dispatch_js


def test_cdp_backend_falls_back_to_xdotool_when_no_cdp_evaluate():
    """If the env doesn't expose ``cdp_evaluate`` (test stubs, etc.),
    the operator's hint was optimistic — fall through to xdotool
    rather than crash. The handler still produces a valid result so
    the runner doesn't halt."""
    env = MagicMock(spec=["step"])  # no cdp_evaluate
    h = MechanicalScrollHandler(MagicMock())
    res = h.execute(
        MicroIntent(
            intent="x", type="scroll",
            params={"count": 2, "backend": "cdp"},
        ),
        _ctx(env),
    )
    assert res.success is True
    # Fell through to xdotool because env had no CDP.
    assert env.step.call_count == 2


def test_cdp_backend_dispatch_error_returned_with_failure_class():
    """When CDP dispatch raises (Chrome devtools disconnected etc.),
    the handler returns ``scroll_dispatch_error`` rather than
    propagating."""
    env = MagicMock()
    env.cdp_evaluate = MagicMock(
        side_effect=[0.0, RuntimeError("devtools disconnected")],
    )
    h = MechanicalScrollHandler(MagicMock())
    res = h.execute(
        MicroIntent(
            intent="x", type="scroll",
            params={"count": 1, "backend": "cdp"},
        ),
        _ctx(env),
    )
    assert res.success is False
    assert res.failure_class == "scroll_dispatch_error"
