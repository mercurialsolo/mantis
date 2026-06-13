"""MicroPlanRunner hooks: step_callback (#74), cancel_event (#76),
register_tool (#71), launch_app (#72), pause/resume (#73).

These tests use a minimal fake env so the runner can be exercised without a
real Xvfb/Holo3 stack.
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image

from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.base import GymEnvironment, GymObservation, GymResult
from mantis_agent.gym.micro_runner import (
    MicroPlanRunner,
    PauseRequested,
    PauseState,
    StepResult,
)
from mantis_agent.plan_decomposer import MicroIntent, MicroPlan


# ── Fake env / brain ────────────────────────────────────────────────────


class _FakeEnv(GymEnvironment):
    """Minimal GymEnvironment that records actions and serves a tiny screenshot."""

    def __init__(self, viewport: tuple[int, int] = (320, 200)):
        self._viewport = viewport
        self.actions: list[Action] = []
        self.launch_calls: list[dict[str, Any]] = []
        self.reset_calls: list[dict[str, Any]] = []

    def reset(self, task: str, **kwargs: Any) -> GymObservation:
        self.reset_calls.append(dict(kwargs))
        return GymObservation(screenshot=self._image())

    def step(self, action: Action) -> GymResult:
        self.actions.append(action)
        if action.action_type == ActionType.LAUNCH_APP:
            self.launch_calls.append(dict(action.params or {}))
        return GymResult(GymObservation(screenshot=self._image()), 0.0, False, {})

    def screenshot(self) -> Image.Image:
        return self._image()

    def close(self) -> None:
        pass

    @property
    def screen_size(self) -> tuple[int, int]:
        return self._viewport

    def _image(self) -> Image.Image:
        return Image.new("RGB", self._viewport, "white")


def _runner(env: _FakeEnv, **kwargs: Any) -> MicroPlanRunner:
    """Build a runner with extractor+grounding stubbed out."""
    return MicroPlanRunner(
        brain=None,
        env=env,
        grounding=None,
        extractor=None,
        checkpoint_path="/tmp/_micro_runner_test_ckpt.json",
        run_key="test",
        session_name="test",
        max_cost=999.0,
        max_time_minutes=999,
        **kwargs,
    )


def _trivial_plan() -> MicroPlan:
    """Three navigate steps — fully synchronous, no Holo3/Claude needed."""
    plan = MicroPlan(domain="test")
    for i in range(3):
        plan.steps.append(
            MicroIntent(
                intent=f"Step {i}: navigate to https://example.test/{i}",
                type="navigate", budget=3, section="setup", required=False,
            )
        )
    return plan


# ── #74 step_callback + screenshot retention ────────────────────────────


def test_step_callback_fires_per_step_with_signature():
    env = _FakeEnv()
    seen: list[tuple[int, str, Any, bool]] = []

    def cb(idx: int, intent: str, action: Any, ok: bool) -> None:
        seen.append((idx, intent, action, ok))

    r = _runner(env, step_callback=cb)
    r.run(_trivial_plan())

    assert len(seen) == 3
    indices = [s[0] for s in seen]
    assert indices == [0, 1, 2]
    intents = [s[1] for s in seen]
    assert all("navigate" in s for s in intents)
    # action arg may be None for some step types — that's fine for the contract.


def test_step_callback_failure_does_not_break_run():
    env = _FakeEnv()

    def cb(*_args: Any, **_kw: Any) -> None:
        raise RuntimeError("observability bug")

    r = _runner(env, step_callback=cb)
    # Run must complete despite the broken callback.
    steps = r.run(_trivial_plan())
    assert len(steps) == 3


def test_step_results_carry_screenshot_bytes_by_default():
    env = _FakeEnv()
    r = _runner(env)
    steps = r.run(_trivial_plan())
    assert len(steps) == 3
    for s in steps:
        # PNG header check — shows the helper actually encoded something.
        assert s.screenshot_png is not None and s.screenshot_png[:8] == b"\x89PNG\r\n\x1a\n"


def test_time_meter_records_load_via_navigate_handler():
    """Phase A wire-ins (epic #362): the navigate step handler wraps
    ``env.reset`` in the ``load`` bucket — credited via the dispatch
    context published by the executor.

    ``settle`` is intentionally NOT asserted here: the broad test suite
    stubs ``adaptive_settle.*`` to instant-return via the autouse
    fixture in ``conftest.py`` for speed. The settle → dispatch credit
    path is verified directly in ``test_time_meter.py``."""
    env = _FakeEnv()
    r = _runner(env)
    r.run(_trivial_plan())

    meter = r.time_meter
    assert meter.totals["load"] > 0.0, (
        "navigate handler should record env.reset time to the load bucket"
    )
    # The dispatch publisher routes deep-helper records to the active
    # step_idx — verify per-step rather than just the aggregate.
    assert any(rec["load"] > 0.0 for rec in meter.per_step)


def test_time_meter_breakdown_fills_overhead_residual():
    """``breakdown()`` should account for time not wrapped in any
    ``measure()`` block as ``overhead`` so the dict sums close to
    ``elapsed_seconds()``."""
    env = _FakeEnv()
    r = _runner(env)
    r.run(_trivial_plan())

    meter = r.time_meter
    bd = meter.breakdown()
    # All buckets present, including overhead.
    assert set(bd) == set(meter.totals)
    # Overhead is non-negative — orchestration time the runner spent
    # outside the wrapped blocks.
    assert bd["overhead"] >= 0.0


def test_keep_screenshots_caps_retained_bytes():
    env = _FakeEnv()
    r = _runner(env, keep_screenshots=1)
    steps = r.run(_trivial_plan())
    # With cap=1 and 3 steps, only the last result keeps its PNG.
    has_png = [s.screenshot_png is not None for s in steps]
    assert has_png == [False, False, True]


def test_to_dict_excludes_screenshot_and_action():
    s = StepResult(step_index=0, intent="x", success=True, screenshot_png=b"fake-bytes")
    d = s.to_dict()
    assert "screenshot_png" not in d
    assert "last_action" not in d
    # Round-trip ignores ignored fields.
    s2 = StepResult.from_dict(d)
    assert s2.screenshot_png is None
    assert s2.last_action is None


def test_run_with_status_returns_completed_runner_result():
    env = _FakeEnv()
    r = _runner(env)
    result = r.run_with_status(_trivial_plan())
    assert result.status == "completed"
    assert len(result.steps) == 3


# ── #76 cancel_event ────────────────────────────────────────────────────


def test_cancel_event_stops_run_at_step_boundary():
    env = _FakeEnv()
    cancel = threading.Event()
    cb_calls: list[int] = []

    def cb(idx: int, *_args: Any) -> None:
        cb_calls.append(idx)
        # Trip cancel after the first step boundary completes.
        if idx == 0:
            cancel.set()

    r = _runner(env, step_callback=cb, cancel_event=cancel)
    result = r.run_with_status(_trivial_plan())

    assert result.cancelled is True
    assert result.status == "cancelled"
    # First step ran, second was preempted at the boundary.
    assert len(result.steps) == 1


def test_cancel_event_callable_is_supported():
    env = _FakeEnv()
    flips = {"hit": False}

    def cancel_fn() -> bool:
        return flips["hit"]

    def cb(idx: int, *_args: Any) -> None:
        if idx == 1:
            flips["hit"] = True

    r = _runner(env, step_callback=cb, cancel_event=cancel_fn)
    result = r.run_with_status(_trivial_plan())
    assert result.cancelled is True
    assert len(result.steps) == 2


def test_no_cancel_runs_all_steps():
    env = _FakeEnv()
    r = _runner(env)
    result = r.run_with_status(_trivial_plan())
    assert result.cancelled is False
    assert result.status == "completed"
    assert len(result.steps) == 3


# ── #71 register_tool ───────────────────────────────────────────────────


def test_register_tool_and_call_tool():
    env = _FakeEnv()
    r = _runner(env)
    captured: dict[str, Any] = {}

    def handler(args: dict[str, Any]) -> str:
        captured.update(args)
        return f"hi {args.get('name', 'anon')}"

    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    r.register_tool(name="greet", schema=schema, handler=handler)

    listed = r.list_tools()
    assert listed == [{"name": "greet", "schema": schema}]

    out = r.call_tool("greet", {"name": "Mantis"})
    assert out == "hi Mantis"
    assert captured == {"name": "Mantis"}


def test_register_tool_rejects_non_callable():
    env = _FakeEnv()
    r = _runner(env)
    with pytest.raises(TypeError):
        r.register_tool("bad", schema={}, handler="not-callable")  # type: ignore[arg-type]


def test_call_tool_unknown_raises_keyerror():
    env = _FakeEnv()
    r = _runner(env)
    with pytest.raises(KeyError):
        r.call_tool("missing")


def test_invoke_tool_surfaces_handler_errors_as_data():
    env = _FakeEnv()
    r = _runner(env)

    def handler(_args: dict[str, Any]) -> None:
        raise RuntimeError("nope")

    r.register_tool("kapow", {}, handler)
    ok, data = r._invoke_tool("kapow", {"x": 1})
    assert ok is False
    assert "RuntimeError" in data and "nope" in data


# ── #72 launch_app dispatch ─────────────────────────────────────────────


def test_launch_app_action_routes_through_env_step():
    env = _FakeEnv()
    env.step(Action(ActionType.LAUNCH_APP, {"name": "chromium", "args": ["--proxy-server=x"]}))
    assert env.launch_calls == [{"name": "chromium", "args": ["--proxy-server=x"]}]


def test_launch_app_in_xdotool_env_handles_missing_binary_gracefully():
    """The env shouldn't crash when launch target doesn't exist."""
    from mantis_agent.gym.xdotool_env import XdotoolGymEnv

    env = XdotoolGymEnv(display=":dummy", viewport=(800, 600))
    env._env = {"DISPLAY": ":dummy"}
    # Binary that definitely doesn't exist; helper logs and returns.
    env._launch_app({"name": "/nonexistent/mantis-test-binary-xyz"})


# ── #73 pause / resume ──────────────────────────────────────────────────


def test_pause_requested_yields_pause_state_via_tool_invocation():
    env = _FakeEnv()
    r = _runner(env)

    def request_user_input(args: dict[str, Any]) -> Any:
        # First call asks for input, second call (after resume) reads it.
        staged = r.consume_pause_input(default=None)
        if staged is None:
            raise PauseRequested(reason="user_input", prompt=args.get("prompt", ""))
        return staged

    r.register_tool(
        "request_user_input",
        {"type": "object", "properties": {"prompt": {"type": "string"}}},
        request_user_input,
    )

    # First invocation triggers a pause — `_invoke_tool` returns success but
    # records the pending pause for the run loop to surface. Pause state lives
    # on the ToolChannel after #115 step 2; the ``_invoke_tool`` shim still
    # delegates correctly.
    ok, data = r._invoke_tool("request_user_input", {"prompt": "MFA code?"})
    assert ok is True
    assert data.endswith(":pause")
    assert r.tool_channel.is_paused()
    assert r.tool_channel.pending_pause["tool"] == "request_user_input"

    # On resume, consume_pause_input returns the staged value.
    r._pause_input = "123456"
    second = r.call_tool("request_user_input", {"prompt": "MFA code?"})
    assert second == "123456"


def test_pause_state_round_trips_through_dict():
    state = PauseState(
        run_key="abc", plan_signature="xyz", session_name="test",
        step_index=4, pending_tool="request_user_input",
        pending_arguments={"prompt": "MFA?"}, prompt="MFA?",
        step_results=[], loop_counters={"3": 1}, listings_on_page=2,
        checkpoint_path="/tmp/x.json", timestamp=1234.5,
    )
    d = state.to_dict()
    restored = PauseState.from_dict(d)
    assert restored == state


def test_resume_requires_plan_argument():
    env = _FakeEnv()
    r = _runner(env)
    state = PauseState(plan_signature="abc")
    with pytest.raises(ValueError):
        r.resume(state, user_input="x")


def test_resume_rejects_mismatched_plan_signature():
    env = _FakeEnv()
    r = _runner(env)
    plan = _trivial_plan()
    state = PauseState(plan_signature="not-the-real-sig")
    with pytest.raises(ValueError):
        r.resume(state, user_input="x", plan=plan)


def test_resume_accepts_embedded_signature_when_recompute_differs(monkeypatch):
    """#882-followup regression — the live-discovered resume blocker.

    The PauseState stores the EMBEDDED ``_plan_signature``
    (``plan_signature_from_steps`` over full step dicts). resume() used
    to recompute via ``_compute_plan_signature`` (CheckpointManager — a
    different 14-field subset over MicroIntent objects), which never
    matched, so EVERY Modal micro-path resume raised a spurious
    mismatch. resume() must compare against the embedded
    ``self.plan_signature`` when the runner has one."""
    env = _FakeEnv()
    r = _runner(env)
    plan = _trivial_plan()
    recompute_sig = r._compute_plan_signature(plan)
    embedded_sig = "embedded-" + recompute_sig  # deliberately DIFFERENT
    assert embedded_sig != recompute_sig
    # The runner was constructed from a suite whose _plan_signature is
    # the embedded value — same as what the PauseState stored at pause.
    r.plan_signature = embedded_sig
    state = PauseState(plan_signature=embedded_sig)
    monkeypatch.setattr(r, "run", lambda *a, **kw: [])
    # Must NOT raise — the embedded signatures match even though the
    # CheckpointManager recompute would not.
    r.resume(state, user_input="x", plan=plan)


def test_resume_still_rejects_genuine_mismatch_with_embedded_sig(monkeypatch):
    """The embedded-signature preference must NOT defeat the real guard:
    a PauseState from a genuinely different plan still fails."""
    env = _FakeEnv()
    r = _runner(env)
    plan = _trivial_plan()
    r.plan_signature = "sig-for-the-plan-actually-running"
    state = PauseState(plan_signature="sig-from-a-totally-different-plan")
    monkeypatch.setattr(r, "run", lambda *a, **kw: [])
    with pytest.raises(ValueError):
        r.resume(state, user_input="x", plan=plan)


def test_pause_state_is_json_serializable():
    """PauseState lives in a Postgres JSONB column on host — keep it pure."""
    import json

    state = PauseState(
        run_key="abc", plan_signature="xyz", session_name="test",
        step_index=4, pending_tool="request_user_input",
        pending_arguments={"prompt": "MFA?"}, prompt="MFA?",
        step_results=[StepResult(step_index=0, intent="x", success=True).to_dict()],
        loop_counters={"3": 1}, listings_on_page=2,
        checkpoint_path="/tmp/x.json", timestamp=1234.5,
    )
    encoded = json.dumps(state.to_dict())
    decoded = json.loads(encoded)
    assert PauseState.from_dict(decoded) == state


# ── Epic #358 Phase A: browser-state restore on resume ─────────────────


def test_resume_calls_restore_browser_state(monkeypatch):
    """On resume(), the runner must invoke ``env.restore_browser_state``
    with the captured BrowserState. Verifies the wire-in without
    actually running a plan."""
    from mantis_agent.gym.checkpoint import BrowserState

    env = _FakeEnv()
    env.restore_browser_state = MagicMock()  # type: ignore[attr-defined]
    r = _runner(env)

    plan = _trivial_plan()
    sig = r._compute_plan_signature(plan)
    state = PauseState(
        plan_signature=sig,
        browser_state=BrowserState(
            url="https://example.test/2", scroll_x=0, scroll_y=1800,
        ),
    )

    # Stub run() so we don't drive the whole step loop.
    monkeypatch.setattr(r, "run", lambda *a, **kw: [])

    r.resume(state, user_input=None, plan=plan)
    env.restore_browser_state.assert_called_once()
    bs = env.restore_browser_state.call_args.args[0]
    assert bs.url == "https://example.test/2"
    assert bs.scroll_y == 1800


def test_resume_stages_user_input_for_substitution(monkeypatch):
    """#882-followup: on resume the run starts AFTER the paused
    request_user_input step, so its handler never re-runs to stage the
    value. resume() must stage it itself onto ``_staged_user_input`` so
    ``_build_effective_step`` can substitute ``{{user_input}}`` into
    downstream fill_field steps (else the literal placeholder is typed)."""
    env = _FakeEnv()
    r = _runner(env)
    plan = _trivial_plan()
    sig = r._compute_plan_signature(plan)
    state = PauseState(plan_signature=sig)
    monkeypatch.setattr(r, "run", lambda *a, **kw: [])

    r.resume(state, user_input="tomsmith", plan=plan)
    assert getattr(r, "_staged_user_input", None) == "tomsmith"


def test_resume_without_user_input_leaves_substitution_noop(monkeypatch):
    """A resume with no value must not stage a junk substitution source."""
    env = _FakeEnv()
    r = _runner(env)
    plan = _trivial_plan()
    sig = r._compute_plan_signature(plan)
    state = PauseState(plan_signature=sig)
    monkeypatch.setattr(r, "run", lambda *a, **kw: [])

    r.resume(state, user_input=None, plan=plan)
    assert getattr(r, "_staged_user_input", None) is None


def test_resume_works_when_env_lacks_restore_method(monkeypatch):
    """Legacy envs without ``restore_browser_state`` must not crash —
    resume should proceed and run the step loop as before."""
    env = _FakeEnv()  # no restore_browser_state method on _FakeEnv
    r = _runner(env)
    plan = _trivial_plan()
    sig = r._compute_plan_signature(plan)
    state = PauseState(plan_signature=sig)  # default empty browser_state

    monkeypatch.setattr(r, "run", lambda *a, **kw: [])
    # Should not raise.
    r.resume(state, user_input=None, plan=plan)


def test_resume_swallows_restore_exception(monkeypatch):
    """A restore_browser_state that raises (CDP unreachable mid-restore)
    must not break the resume — the run loop still starts."""
    env = _FakeEnv()
    env.restore_browser_state = MagicMock(  # type: ignore[attr-defined]
        side_effect=RuntimeError("CDP unreachable"),
    )
    r = _runner(env)
    plan = _trivial_plan()
    sig = r._compute_plan_signature(plan)
    state = PauseState(plan_signature=sig)

    run_called = {"yes": False}
    def _run_stub(*a, **kw):
        run_called["yes"] = True
        return []
    monkeypatch.setattr(r, "run", _run_stub)

    r.resume(state, user_input=None, plan=plan)
    assert run_called["yes"]
