"""Base types for reward functions.

A reward function turns Mantis trajectories into a scalar signal that an RL
trainer (or rejection sampler) can optimise. Each function exposes two hooks:

  step(action, gym_result, state)  → per-step reward
  episode(run_result, state, ...)  → terminal reward (task success, etc.)

`RewardSignal` carries both the scalar `value` (what the trainer consumes) and
a `components` breakdown (what humans read in logs). `EpisodeState` accumulates
context — action history, off-site visits, plan progress — so stateful
terms (loop penalty, plan adherence) can be expressed as pure functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ..actions import Action
    from ..gym.base import GymResult
    from ..gym.runner import RunResult


@dataclass
class RewardSignal:
    """Scalar reward + per-component breakdown for logging."""

    value: float = 0.0
    components: dict[str, float] = field(default_factory=dict)

    def __float__(self) -> float:
        return self.value

    def __add__(self, other: "RewardSignal") -> "RewardSignal":
        merged = {**self.components}
        for k, v in other.components.items():
            merged[k] = merged.get(k, 0.0) + v
        return RewardSignal(value=self.value + other.value, components=merged)


@dataclass
class EpisodeState:
    """Accumulated context across an episode.

    The runner passes this into every `step()` and `episode()` call. Reward
    components mutate it (e.g. bumping `off_site_visits`) so subsequent steps
    can see the running counts.
    """

    action_history: list["Action"] = field(default_factory=list)
    info_history: list[dict[str, Any]] = field(default_factory=list)

    # Stateful counters touched by reward components
    off_site_visits: int = 0
    loop_runs: int = 0

    # Plan tracking (populated by runner if a structured Plan is in use)
    plan_step_idx: int = 0
    plan_steps_passed: int = 0
    plan_steps_total: int = 0

    # Free-form scratch space for env-specific rewards (e.g. extracted records)
    extras: dict[str, Any] = field(default_factory=dict)


class RewardFn(Protocol):
    """A reward function for any GymEnvironment trajectory.

    Implementations should be cheap by default — derive signal from
    `gym_result.info` and `state` rather than calling external APIs.
    Use `episode()` for any expensive end-of-episode grading.
    """

    def step(
        self,
        *,
        action: "Action",
        gym_result: "GymResult",
        state: EpisodeState,
    ) -> RewardSignal:
        """Per-step reward. Mutate `state` to accumulate context."""
        ...

    def episode(
        self,
        *,
        run_result: "RunResult",
        state: EpisodeState,
        ground_truth: dict[str, Any] | None = None,
    ) -> RewardSignal:
        """Terminal reward. Called once at episode end."""
        ...
