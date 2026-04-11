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
from .runner import GymRunner

__all__ = [
    "GymEnvironment",
    "GymObservation",
    "GymResult",
    "GymRunner",
]
