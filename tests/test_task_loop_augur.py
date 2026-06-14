"""Augur instrumentation for the task_loop path (Claude / EvoCUA / OpenCUA /
Gemma4-CUA). The Holo3 micro path emits bundles via RunExecutor; these tests
cover the task_loop wedge that closes the gap for non-Holo3 brains."""

from __future__ import annotations

from types import SimpleNamespace

from mantis_agent import task_loop
from mantis_agent.actions import Action, ActionType
from mantis_agent.observability import augur as augur_mod


def _ts(step, action_type, reward=0.0, thinking=""):
    return SimpleNamespace(
        step=step,
        action=Action(action_type, {}),
        reward=reward,
        thinking=thinking,
        executor_backend="vision",
    )


# ── pure shim mapping ─────────────────────────────────────────────


def test_shim_maps_step_index_and_action():
    traj = [_ts(0, ActionType.CLICK), _ts(1, ActionType.TYPE)]
    shims = task_loop._gym_trajectory_to_step_shims(traj, run_success=True)
    assert [s.step_index for s in shims] == [0, 1]
    assert shims[0].last_action.action_type == ActionType.CLICK
    assert shims[1].last_action.action_type == ActionType.TYPE


def test_shim_success_positive_reward_passes():
    traj = [_ts(0, ActionType.CLICK, reward=1.0)]
    shims = task_loop._gym_trajectory_to_step_shims(traj, run_success=False)
    assert shims[0].success is True


def test_shim_final_step_reflects_run_success():
    traj = [_ts(0, ActionType.CLICK), _ts(1, ActionType.CLICK)]
    ok = task_loop._gym_trajectory_to_step_shims(traj, run_success=True)
    assert ok[-1].success is True and ok[0].success is False
    bad = task_loop._gym_trajectory_to_step_shims(traj, run_success=False)
    assert all(s.success is False for s in bad)


def test_shim_final_step_passes_on_done_termination():
    traj = [_ts(0, ActionType.CLICK), _ts(1, ActionType.CLICK)]
    # run_success False but the agent terminated via a deliberate `done`
    shims = task_loop._gym_trajectory_to_step_shims(
        traj, run_success=False, termination_reason="done",
    )
    assert shims[-1].success is True


# ── run-status mapping (issue #892 item 3) ─────────────────────────


def test_run_status_paused():
    assert task_loop._run_status_from_result(
        SimpleNamespace(paused=True, success=False, termination_reason="")
    ) == "paused"


def test_run_status_done_termination_is_succeeded():
    assert task_loop._run_status_from_result(
        SimpleNamespace(paused=False, success=False, termination_reason="done")
    ) == "succeeded"


def test_run_status_max_steps_is_halted():
    assert task_loop._run_status_from_result(
        SimpleNamespace(paused=False, success=False, termination_reason="max_steps")
    ) == "halted"


# ── emit wiring ───────────────────────────────────────────────────


class _FakeAdapter:
    last = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.active = True
        self.recorded: list[tuple] = []
        self.observed: list[tuple] = []
        self.closed: str | None = None
        _FakeAdapter.last = self

    def attach_observation(self, *, step_index, kind, png):
        self.observed.append((step_index, len(png) if png else 0))
        return f"shot/{step_index}.png" if png else None

    def record_step(self, *, step_result, observation_post=None, step_type="", **kw):
        self.recorded.append((step_result.step_index, step_type, step_result.success))

    def close(self, status=None):
        self.closed = status


def _config():
    return task_loop.TaskLoopConfig(
        run_id="run-xyz", session_name="sess", model_name="Claude (test)",
        results_prefix="claude", brain=SimpleNamespace(model_name="claude-test"),
        env=object(),
    )


def _handle(monkeypatch, adapter_cls=_FakeAdapter):
    monkeypatch.setattr(augur_mod, "AugurAdapter", adapter_cls)
    return task_loop.open_augur_handle(_config(), "task-1")


def test_open_handle_returns_none_when_inactive(monkeypatch):
    class Inactive(_FakeAdapter):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.active = False

    assert _handle(monkeypatch, Inactive) is None


def test_emit_records_each_step_and_closes_succeeded(monkeypatch):
    handle = _handle(monkeypatch)
    assert handle.adapter.kwargs["run_id"] == "run-xyz"
    assert handle.adapter.kwargs["brain_model_name"] == "claude-test"
    result = SimpleNamespace(
        success=True, paused=False, termination_reason="done", fallback_used=None,
        trajectory=[_ts(0, ActionType.CLICK), _ts(1, ActionType.TYPE)],
    )
    task_loop.emit_augur_run(_config(), "task-1", "do a thing", result, handle=handle)
    a = handle.adapter
    assert [r[0] for r in a.recorded] == [0, 1]
    assert a.recorded[0][1] == "click"
    assert a.recorded[1][1] == "type_text"
    assert a.closed == "succeeded"


def test_emit_status_halted_when_failed(monkeypatch):
    handle = _handle(monkeypatch)
    result = SimpleNamespace(
        success=False, paused=False, termination_reason="max_steps",
        fallback_used=None, trajectory=[_ts(0, ActionType.CLICK)],
    )
    task_loop.emit_augur_run(_config(), "t", "i", result, handle=handle)
    assert handle.adapter.closed == "halted"


def test_emit_noop_when_handle_none():
    # handle is None (Augur unconfigured) → nothing to do, no raise
    task_loop.emit_augur_run(
        _config(), "t", "i", SimpleNamespace(success=True, trajectory=[]), handle=None,
    )


def test_emit_attaches_screenshots_from_capture_dir(monkeypatch, tmp_path):
    handle = _handle(monkeypatch)
    handle.capture_dir = str(tmp_path)
    # GymRunner writes <NNNN>.png; provide one for step 0 only.
    (tmp_path / "0000.png").write_bytes(b"\x89PNG\r\n\x1a\nFAKE")
    result = SimpleNamespace(
        success=True, paused=False, termination_reason="done", fallback_used=None,
        trajectory=[_ts(0, ActionType.CLICK), _ts(1, ActionType.CLICK)],
    )
    task_loop.emit_augur_run(_config(), "task-1", "i", result, handle=handle)
    obs = dict(handle.adapter.observed)
    assert obs[0] > 0   # step 0 PNG bytes attached
    assert obs[1] == 0  # step 1 has no frame on disk


def test_emit_skips_screenshots_on_fallback(monkeypatch, tmp_path):
    handle = _handle(monkeypatch)
    handle.capture_dir = str(tmp_path)
    (tmp_path / "0000.png").write_bytes(b"\x89PNGFAKE")
    result = SimpleNamespace(
        success=False, paused=False, termination_reason="max_steps",
        fallback_used="claude", trajectory=[_ts(0, ActionType.CLICK)],
    )
    task_loop.emit_augur_run(_config(), "t", "i", result, handle=handle)
    # fallback result → frames belong to the standard run, skip them
    assert handle.adapter.observed == [(0, 0)]


def test_emit_never_raises(monkeypatch):
    handle = _handle(monkeypatch)

    class Boom:
        def __getattr__(self, _):
            raise RuntimeError("trajectory blew up")

    task_loop.emit_augur_run(_config(), "t", "i", Boom(), handle=handle)
