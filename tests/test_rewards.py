"""Tests for the rewards/ package."""

from __future__ import annotations

from typing import Any

import pytest

from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.base import GymObservation, GymResult
from mantis_agent.gym.runner import RunResult, TrajectoryStep
from mantis_agent.rewards import (
    BoatTraderReward,
    EpisodeState,
    PlanAdherenceReward,
    RewardSignal,
)
from mantis_agent.rewards.boattrader import _parse_summary
from mantis_agent.rewards.components import (
    format_reward,
    loop_penalty,
    off_site_penalty,
    task_success_reward,
)


# ── helpers ─────────────────────────────────────────────────────────────


def _gr(info: dict[str, Any] | None = None, reward: float = 0.0) -> GymResult:
    """Build a GymResult with a stub observation."""
    return GymResult(
        observation=GymObservation(screenshot=None),  # type: ignore[arg-type]
        reward=reward,
        done=False,
        info=info or {},
    )


def _click(x: int = 100, y: int = 200) -> Action:
    return Action(ActionType.CLICK, {"x": x, "y": y})


def _done(success: bool, summary: str = "") -> Action:
    return Action(ActionType.DONE, {"success": success, "summary": summary})


def _trajectory(*actions: Action) -> list[TrajectoryStep]:
    return [
        TrajectoryStep(step=i + 1, action=a, thinking="", reward=0.0,
                       done=False, inference_time=0.0)
        for i, a in enumerate(actions)
    ]


def _run(trajectory: list[TrajectoryStep], **overrides: Any) -> RunResult:
    defaults: dict[str, Any] = dict(
        task="t", task_id="t", success=True, total_reward=0.0,
        total_steps=len(trajectory), total_time=1.0,
        trajectory=trajectory, termination_reason="done",
    )
    defaults.update(overrides)
    return RunResult(**defaults)


# ── components ──────────────────────────────────────────────────────────


def test_format_reward_well_formed_click() -> None:
    assert format_reward(_click()) == 0.1


def test_format_reward_missing_required_param() -> None:
    bad = Action(ActionType.CLICK, {"x": 100})  # missing y
    assert format_reward(bad) == 0.0


def test_format_reward_done_always_valid() -> None:
    assert format_reward(_done(True, "ok")) == 0.1


def test_off_site_penalty_backtracked() -> None:
    assert off_site_penalty({"backtracked": True}) == -0.5


def test_off_site_penalty_url_not_in_allowlist() -> None:
    info = {"url": "https://facebook.com/page"}
    assert off_site_penalty(info, allowed_domains=("boattrader.com",)) == -0.5


def test_off_site_penalty_url_on_allowlist() -> None:
    info = {"url": "https://www.boattrader.com/boat/123/"}
    assert off_site_penalty(info, allowed_domains=("boattrader.com",)) == 0.0


def test_loop_penalty_three_identical_actions() -> None:
    history = [_click(50, 50), _click(50, 50), _click(50, 50)]
    assert loop_penalty(history, window=3) == -0.2


def test_loop_penalty_breaks_on_different_action() -> None:
    history = [_click(50, 50), _click(60, 60), _click(50, 50)]
    assert loop_penalty(history, window=3) == 0.0


def test_task_success_reward() -> None:
    assert task_success_reward("done", success=True) == 1.0
    assert task_success_reward("nope", success=False) == 0.0


# ── PlanAdherenceReward ─────────────────────────────────────────────────


def test_plan_adherence_step_well_formed_click_on_site() -> None:
    r = PlanAdherenceReward(allowed_domains=("boattrader.com",))
    state = EpisodeState()
    sig = r.step(
        action=_click(),
        gym_result=_gr({"url": "https://www.boattrader.com/x"}),
        state=state,
    )
    assert sig.components == {"format": 0.1}
    assert pytest.approx(float(sig)) == 0.1


def test_plan_adherence_step_off_site_penalised() -> None:
    r = PlanAdherenceReward(allowed_domains=("boattrader.com",))
    state = EpisodeState()
    sig = r.step(
        action=_click(),
        gym_result=_gr({"url": "https://facebook.com"}),
        state=state,
    )
    assert sig.components["off_site"] == -0.5
    assert state.off_site_visits == 1


def test_plan_adherence_step_loop_penalised() -> None:
    r = PlanAdherenceReward()
    state = EpisodeState()
    state.action_history = [_click(50, 50), _click(50, 50)]
    sig = r.step(action=_click(50, 50), gym_result=_gr(), state=state)
    assert sig.components.get("loop") == -0.2
    assert state.loop_runs == 1


def test_plan_adherence_episode_success() -> None:
    r = PlanAdherenceReward(success_weight=1.0, plan_progress_weight=0.0)
    traj = _trajectory(_click(), _done(True, "all good"))
    sig = r.episode(run_result=_run(traj), state=EpisodeState(), ground_truth=None)
    assert sig.components["task_success"] == 1.0


def test_plan_adherence_episode_failure() -> None:
    r = PlanAdherenceReward(success_weight=1.0, plan_progress_weight=0.0)
    traj = _trajectory(_click(), _done(False, "stuck"))
    sig = r.episode(run_result=_run(traj), state=EpisodeState(), ground_truth=None)
    assert sig.components["task_success"] == 0.0


def test_plan_adherence_episode_includes_plan_progress() -> None:
    r = PlanAdherenceReward(success_weight=1.0, plan_progress_weight=0.3)
    state = EpisodeState(plan_step_idx=2, plan_steps_total=4)
    traj = _trajectory(_done(True, "ok"))
    sig = r.episode(run_result=_run(traj), state=state)
    assert sig.components["task_success"] == 1.0
    assert sig.components["plan_progress"] == pytest.approx(0.15)
    assert pytest.approx(float(sig)) == 1.15


# ── BoatTraderReward ────────────────────────────────────────────────────


def test_parse_summary_dollar_price_beats_url_slug() -> None:
    s = "2018 Sea Ray 240 Sundeck $42,500 https://www.boattrader.com/boat/2018-sea-ray-9876543/"
    rec = _parse_summary(s)
    assert rec["year"] == 2018
    assert rec["price"] == 42500  # not 9876543 from the URL slug
    assert "boattrader.com" in rec["url"]


def test_parse_summary_labeled_format() -> None:
    s = "Year: 2015, Make: Bayliner, Model: 175, Price: $9,500, URL: https://www.boattrader.com/boat/x/"
    rec = _parse_summary(s)
    assert rec == {
        "url": "https://www.boattrader.com/boat/x/",
        "year": 2015,
        "make": "Bayliner",
        "model": "175",
        "price": 9500,
    }


def test_parse_summary_empty_when_unparseable() -> None:
    assert _parse_summary("couldn't find listing") == {}


def test_boattrader_reward_gate_passes() -> None:
    r = BoatTraderReward()
    summary = "2018 Sea Ray 240 $42,500 https://www.boattrader.com/boat/2018-x/"
    traj = _trajectory(_click(), _done(True, summary))
    sig = r.episode(run_result=_run(traj), state=EpisodeState(), ground_truth=None)
    assert sig.components.get("gate_passed") == 1.0


def test_boattrader_reward_gate_fails_off_site_url() -> None:
    r = BoatTraderReward()
    summary = "2018 Sea Ray 240 $42,500 https://www.facebook.com/boat/123/"
    traj = _trajectory(_done(True, summary))
    sig = r.episode(run_result=_run(traj), state=EpisodeState(), ground_truth=None)
    # url_ok = False → no gate_passed
    assert sig.components.get("gate_passed") is None
    assert float(sig) == 0.0


def test_boattrader_reward_gate_fails_missing_field() -> None:
    r = BoatTraderReward()
    # No price.
    summary = "2018 Sea Ray 240 https://www.boattrader.com/boat/x/"
    traj = _trajectory(_done(True, summary))
    sig = r.episode(run_result=_run(traj), state=EpisodeState())
    assert sig.components.get("gate_passed") is None


def test_boattrader_reward_constraint_violation() -> None:
    r = BoatTraderReward()
    summary = "2018 Sea Ray 240 $9,000 https://www.boattrader.com/boat/x/"
    traj = _trajectory(_done(True, summary))
    sig = r.episode(
        run_result=_run(traj),
        state=EpisodeState(),
        ground_truth={"min_price": 35000},
    )
    assert sig.components.get("gate_passed") is None


def test_boattrader_reward_partial_credit_disabled_by_default() -> None:
    r = BoatTraderReward()
    traj = _trajectory(_done(False, "couldn't find"))
    sig = r.episode(run_result=_run(traj), state=EpisodeState())
    assert float(sig) == 0.0


def test_boattrader_reward_partial_credit_enabled() -> None:
    r = BoatTraderReward(field_partial_credit=0.4)
    summary = "Year: 2018, Make: Sea, Model: Ray"  # no price/url
    traj = _trajectory(_done(True, summary))
    sig = r.episode(run_result=_run(traj), state=EpisodeState())
    # 3 of 5 fields present: 0.4 * 3/5 = 0.24, minus url offsite penalty 0.4
    assert "gate_partial" in sig.components


# ── runner integration ─────────────────────────────────────────────────


def test_reward_signal_addition() -> None:
    a = RewardSignal(value=0.5, components={"x": 0.3, "y": 0.2})
    b = RewardSignal(value=1.0, components={"x": 0.5, "z": 0.5})
    c = a + b
    assert c.value == 1.5
    assert c.components == {"x": 0.8, "y": 0.2, "z": 0.5}
