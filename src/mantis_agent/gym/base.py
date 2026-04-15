"""Abstract base class for gym-like environments.

Any environment that provides screenshots and accepts mouse/keyboard actions
can implement this interface to work with Mantis's brain and runner.

Design principles:
- Minimal interface: reset(), step(), close()
- Translation hooks: subclasses override action/observation conversion
- No dependency on any specific gym framework
- Compatible with both local and remote environments
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from ..actions import Action


@dataclass
class GymObservation:
    """Standardized observation from any gym environment.

    The runner only needs `screenshot` (PIL Image). Extra fields are
    environment-specific and passed through for verification/logging.
    """

    screenshot: Image.Image
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class GymResult:
    """Result of a single env.step() call."""

    observation: GymObservation
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class GymEnvironment(ABC):
    """Abstract gym environment that Mantis can drive.

    Subclass this to integrate any gym-like environment. The three methods
    to implement are reset(), step(), and close(). Override translate_action()
    and translate_observation() to handle format conversion.

    Example:
        class MyEnv(GymEnvironment):
            def reset(self, task, **kw):
                raw_obs = self._env.reset()
                return self.translate_observation(raw_obs)

            def step(self, action):
                gym_action = self.translate_action(action)
                raw_obs, reward, done, info = self._env.step(gym_action)
                return GymResult(self.translate_observation(raw_obs), reward, done, info)

            def close(self):
                self._env.close()
    """

    @abstractmethod
    def reset(self, task: str, **kwargs: Any) -> GymObservation:
        """Reset the environment for a new task.

        Args:
            task: Natural language task description.
            **kwargs: Environment-specific options (seed, task_id, etc.)

        Returns:
            Initial observation after reset.
        """

    @abstractmethod
    def step(self, action: Action) -> GymResult:
        """Execute an action and return the next observation.

        Args:
            action: Mantis Action to execute.

        Returns:
            GymResult with observation, reward, done flag, and info.
        """

    @abstractmethod
    def close(self) -> None:
        """Clean up environment resources."""

    @property
    @abstractmethod
    def screen_size(self) -> tuple[int, int]:
        """Return (width, height) of the environment screen in pixels."""

    def translate_action(self, action: Action) -> Any:
        """Convert a Mantis Action to the environment's native format.

        Default implementation returns the Action as-is. Override in
        subclasses that need a different action format.
        """
        return action

    def translate_observation(self, raw_observation: Any) -> GymObservation:
        """Convert an environment's raw observation to GymObservation.

        Default implementation assumes raw_observation is already a PIL Image.
        Override in subclasses that return different observation formats.
        """
        if isinstance(raw_observation, Image.Image):
            return GymObservation(screenshot=raw_observation)
        raise NotImplementedError(
            f"Cannot translate observation of type {type(raw_observation)}. "
            "Override translate_observation() in your environment subclass."
        )
