"""Generic plan-adherence reward.

Composite signal targeting the failure modes the codebase already documents:
  - off-site navigation (gallery traps, social-icon clicks)
  - action loops (clicking the same element 5 times in a row)
  - malformed tool calls

Plus a terminal contribution for plan progress and explicit task success.

Weights below are starting points calibrated for browser CUA where:
  - Episodes are 20–50 steps long
  - Per-step rewards should sum to a small positive baseline if the agent
    is making any progress (so accumulated step reward ≪ terminal reward)
  - Negatives must be heavy enough that 2–3 violations cancel a successful
    episode — otherwise the policy reward-hacks via off-site escapes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .base import EpisodeState, RewardSignal
from .components import (
    format_reward,
    loop_penalty,
    off_site_penalty,
    task_success_reward,
)

if TYPE_CHECKING:
    from ..actions import Action
    from ..gym.base import GymResult
    from ..gym.runner import RunResult


@dataclass
class PlanAdherenceReward:
    """Per-step shaping + terminal success/plan-progress signal.

    Args:
        allowed_domains: tuple of host suffixes considered "on-site". When
            non-empty, off-site visits (per `info["url"]` host) trigger the
            off_site_penalty even if the adapter didn't backtrack.
        format_weight: per-step reward for a well-formed tool call.
        off_site_weight: per-step penalty for off-site navigation.
        loop_weight: per-step penalty for repeating the same action.
        loop_window: how many identical actions in a row trip loop_penalty.
        success_weight: terminal reward for done(success=true).
        plan_progress_weight: terminal reward = weight * fraction_passed.
    """

    allowed_domains: tuple[str, ...] = ()
    format_weight: float = 0.1
    off_site_weight: float = -0.5
    loop_weight: float = -0.2
    loop_window: int = 3
    success_weight: float = 1.0
    plan_progress_weight: float = 0.3

    def step(
        self,
        *,
        action: "Action",
        gym_result: "GymResult",
        state: EpisodeState,
    ) -> RewardSignal:
        components: dict[str, float] = {}

        components["format"] = format_reward(action, value=self.format_weight)

        off = off_site_penalty(
            gym_result.info,
            allowed_domains=self.allowed_domains,
            penalty=self.off_site_weight,
        )
        if off != 0.0:
            components["off_site"] = off
            state.off_site_visits += 1

        loop = loop_penalty(
            state.action_history + [action],
            window=self.loop_window,
            penalty=self.loop_weight,
        )
        if loop != 0.0:
            components["loop"] = loop
            state.loop_runs += 1

        return RewardSignal(value=sum(components.values()), components=components)

    def episode(
        self,
        *,
        run_result: "RunResult",
        state: EpisodeState,
        ground_truth: dict[str, Any] | None = None,
    ) -> RewardSignal:
        components: dict[str, float] = {}

        # Terminal success — read from the trajectory's last DONE action.
        success = False
        summary = ""
        for tstep in reversed(run_result.trajectory):
            if tstep.action.action_type.value == "done":
                success = bool(tstep.action.params.get("success", False))
                summary = str(tstep.action.params.get("summary", ""))
                break
        components["task_success"] = task_success_reward(
            summary=summary, success=success, value=self.success_weight,
        )

        # Plan progress — only meaningful when runner populated plan_steps_total.
        if state.plan_steps_total > 0:
            frac = state.plan_step_idx / state.plan_steps_total
            components["plan_progress"] = self.plan_progress_weight * frac

        return RewardSignal(value=sum(components.values()), components=components)
