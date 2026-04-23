"""Gym environment abstractions for Mantis CUA.

This package provides a modular interface for plugging Mantis's brain
(Gemma4Brain / LlamaCppBrain) into any gym-like environment — gym-anything,
OSWorld, HUD, or custom environments.

Architecture:
    GymEnvironment (ABC)     ← abstract reset/step/close + action/obs translation
        ├── GymAnythingEnv   ← gym-anything (Docker/QEMU/Remote)
        └── (future envs)    ← easy to extend

    GymRunner                ← drives any GymEnvironment with a Mantis brain
"""

from .base import GymEnvironment, GymObservation, GymResult
from .plans import Plan, load_plan, load_text_plan, save_plan, get_missing_inputs, text_to_yaml
from .playwright_env import PlaywrightGymEnv
from .runner import GymRunner
from .workflow_runner import WorkflowRunner, LoopConfig, IterationResult

__all__ = [
    "GymEnvironment",
    "GymObservation",
    "GymResult",
    "GymRunner",
    "Plan",
    "PlaywrightGymEnv",
    "get_missing_inputs",
    "load_plan",
    "load_text_plan",
    "LoopConfig",
    "save_plan",
    "text_to_yaml",
    "WorkflowRunner",
    "IterationResult",
]
