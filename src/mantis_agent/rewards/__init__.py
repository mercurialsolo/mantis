"""Reward functions for Mantis trajectories.

The package is split into:
  base.py        — RewardSignal, EpisodeState, RewardFn protocol
  components.py  — reusable primitives (format_reward, off_site_penalty, ...)
  plan_adherence — generic per-step plan verifier reward
  boattrader     — terminal gate predicate for BoatTrader extraction

A reward function plugs into GymRunner via the `reward_fn=` kwarg. Each
TrajectoryStep gets `step()`'s value; RunResult's terminal_reward gets
`episode()`'s value.
"""

from .base import EpisodeState, RewardFn, RewardSignal
from .components import (
    format_reward,
    loop_penalty,
    off_site_penalty,
    task_success_reward,
    type_verified_reward,
    url_progress_reward,
)
from .plan_adherence import PlanAdherenceReward
from .boattrader import BoatTraderReward

__all__ = [
    "EpisodeState",
    "RewardFn",
    "RewardSignal",
    # Components
    "format_reward",
    "loop_penalty",
    "off_site_penalty",
    "task_success_reward",
    "type_verified_reward",
    "url_progress_reward",
    # Reward fns
    "PlanAdherenceReward",
    "BoatTraderReward",
]
