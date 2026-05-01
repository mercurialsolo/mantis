"""Reward functions for Mantis trajectories.

The package is split into:
  base.py        — RewardSignal, EpisodeState, RewardFn protocol
  components.py  — reusable primitives (format_reward, off_site_penalty, ...)
  plan_adherence — generic per-step plan verifier reward

Vertical-specific terminal gates live under ``mantis_agent.recipes.<name>.rewards``.
``BoatTraderReward`` (deprecated alias) is still re-exported here for one
minor release; new callers should import
``mantis_agent.recipes.marketplace_listings.rewards.MarketplaceListingReward``.

A reward function plugs into GymRunner via the ``reward_fn=`` kwarg. Each
TrajectoryStep gets ``step()``'s value; RunResult's terminal_reward gets
``episode()``'s value.
"""

from .base import EpisodeState, RewardFn, RewardSignal
from .boattrader import BoatTraderReward  # deprecated alias
from .components import (
    format_reward,
    loop_penalty,
    off_site_penalty,
    task_success_reward,
    type_verified_reward,
    url_progress_reward,
)
from .plan_adherence import PlanAdherenceReward

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
    # Deprecated — see module docstring
    "BoatTraderReward",
]
