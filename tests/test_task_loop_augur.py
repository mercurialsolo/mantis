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


# ── emit wiring ───────────────────────────────────────────────────


class _FakeAdapter:
    last = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.active = True
        self.recorded: list[tuple] = []
        self.closed: str | None = None
        _FakeAdapter.last = self

    def attach_observation(self, *, step_index, kind, png):
        return None

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


def test_emit_records_each_step_and_closes_succeeded(monkeypatch):
    monkeypatch.setattr(augur_mod, "AugurAdapter", _FakeAdapter)
    result = SimpleNamespace(
        success=True, paused=False,
        trajectory=[_ts(0, ActionType.CLICK), _ts(1, ActionType.TYPE)],
    )
    task_loop.emit_augur_run(_config(), "task-1", "do a thing", result)
    a = _FakeAdapter.last
    assert [r[0] for r in a.recorded] == [0, 1]
    assert a.recorded[0][1] == "click"  # step_type from Action
    assert a.recorded[1][1] == "type_text"
    assert a.closed == "succeeded"
    assert a.kwargs["run_id"] == "run-xyz"
    assert a.kwargs["brain_model_name"] == "claude-test"


def test_emit_status_halted_when_failed(monkeypatch):
    monkeypatch.setattr(augur_mod, "AugurAdapter", _FakeAdapter)
    result = SimpleNamespace(
        success=False, paused=False, trajectory=[_ts(0, ActionType.CLICK)],
    )
    task_loop.emit_augur_run(_config(), "t", "i", result)
    assert _FakeAdapter.last.closed == "halted"


def test_emit_noop_when_adapter_inactive(monkeypatch):
    class Inactive(_FakeAdapter):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.active = False

    monkeypatch.setattr(augur_mod, "AugurAdapter", Inactive)
    result = SimpleNamespace(success=True, paused=False, trajectory=[_ts(0, ActionType.CLICK)])
    task_loop.emit_augur_run(_config(), "t", "i", result)
    # inactive → returned before recording / closing
    assert _FakeAdapter.last.recorded == []
    assert _FakeAdapter.last.closed is None


def test_emit_never_raises(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("adapter blew up")

    monkeypatch.setattr(augur_mod, "AugurAdapter", boom)
    # must swallow — telemetry never breaks a run
    task_loop.emit_augur_run(_config(), "t", "i", SimpleNamespace(success=True, trajectory=[]))
