"""gym-anything adapter for Mantis CUA.

Bridges Mantis's brain to CMU L3's gym-anything framework, supporting both
local (GymAnythingEnv) and remote (RemoteGymEnv) environments.

gym-anything provides Docker/QEMU/AVF environments with browser, desktop,
and mobile applications. This adapter translates between:
  - Mantis Action → gym-anything action dicts (mouse/keyboard/control)
  - gym-anything rgb_screen observations → PIL Images

Reference: https://github.com/cmu-l3/gym-anything
"""

from __future__ import annotations

import logging
from typing import Any

from PIL import Image

from ..actions import Action, ActionType
from .base import GymEnvironment, GymObservation, GymResult

logger = logging.getLogger(__name__)


def _translate_mantis_action(action: Action) -> list[dict]:
    """Convert a Mantis Action to gym-anything action format.

    gym-anything expects a list of action dicts per step. Each dict uses
    one of: mouse, key, wait, screenshot, terminate.
    """
    match action.action_type:
        case ActionType.CLICK:
            x, y = action.params["x"], action.params["y"]
            button = action.params.get("button", "left")
            click_type = {
                "left": "left_click",
                "right": "right_click",
                "middle": "middle_click",
            }.get(button, "left_click")
            return [{"mouse": {click_type: [x, y]}}]

        case ActionType.DOUBLE_CLICK:
            x, y = action.params["x"], action.params["y"]
            return [{"mouse": {"double_click": [x, y]}}]

        case ActionType.TYPE:
            text = action.params["text"]
            return [{"key": {"type": text}}]

        case ActionType.KEY_PRESS:
            keys = action.params["keys"]
            return [{"key": {"key": keys}}]

        case ActionType.SCROLL:
            direction = action.params["direction"]
            amount = action.params.get("amount", 3)
            x = action.params.get("x", 0)
            y = action.params.get("y", 0)
            dx, dy = 0, 0
            if direction == "up":
                dy = amount
            elif direction == "down":
                dy = -amount
            elif direction == "left":
                dx = -amount
            elif direction == "right":
                dx = amount
            return [{"mouse": {"scroll": [x, y, dx, dy]}}]

        case ActionType.DRAG:
            sx = action.params["start_x"]
            sy = action.params["start_y"]
            ex = action.params["end_x"]
            ey = action.params["end_y"]
            return [{"mouse": {"left_click_drag": [sx, sy, ex, ey]}}]

        case ActionType.WAIT:
            seconds = action.params.get("seconds", 1.0)
            return [{"wait": seconds}]

        case ActionType.DONE:
            return [{"terminate": True}]

    return [{"wait": 1.0}]


def _obs_to_pil(obs: dict) -> GymObservation:
    """Convert gym-anything observation dict to GymObservation.

    gym-anything observations include:
      - rgb_screen: numpy array (H, W, 3) uint8
      - ui_tree: optional structured UI representation
      - cli_stdout: optional terminal output
    """
    rgb = obs.get("rgb_screen")
    if rgb is None:
        raise ValueError("Observation missing 'rgb_screen' — check env configuration")

    try:
        import numpy as np
        is_ndarray = isinstance(rgb, np.ndarray)
    except ImportError:
        is_ndarray = False

    if is_ndarray:
        screenshot = Image.fromarray(rgb)
    elif isinstance(rgb, Image.Image):
        screenshot = rgb
    else:
        raise TypeError(f"Unexpected rgb_screen type: {type(rgb)}")

    extras = {k: v for k, v in obs.items() if k != "rgb_screen"}
    return GymObservation(screenshot=screenshot, extras=extras)


class GymAnythingAdapter(GymEnvironment):
    """Adapter connecting Mantis to a gym-anything environment.

    Wraps either a local GymAnythingEnv or remote RemoteGymEnv from the
    gym-anything package, translating actions and observations.

    Args:
        env_dir: Path to the gym-anything environment directory.
        remote_url: If set, use RemoteGymEnv to connect to a master server.
            If None, use local GymAnythingEnv.
        runner: Runner type override (e.g. "docker", "qemu", "avf").
            If None, gym-anything auto-selects based on platform.
        resolution: Screen resolution as (width, height).
    """

    def __init__(
        self,
        env_dir: str,
        remote_url: str | None = None,
        runner: str | None = None,
        resolution: tuple[int, int] = (1920, 1080),
    ):
        self._env_dir = env_dir
        self._remote_url = remote_url
        self._runner = runner
        self._resolution = resolution
        self._env: Any = None
        self._task_id: str | None = None

    def _create_env(self, task_id: str) -> Any:
        """Lazily create the gym-anything environment."""
        if self._remote_url:
            from gym_anything import RemoteGymEnv
            return RemoteGymEnv.from_config(
                remote_url=self._remote_url,
                env_dir=self._env_dir,
                task_id=task_id,
            )
        else:
            from gym_anything import GymAnythingEnv
            kwargs: dict[str, Any] = {"env_dir": self._env_dir, "task_id": task_id}
            if self._runner:
                kwargs["runner"] = self._runner
            return GymAnythingEnv.from_config(**kwargs)

    def reset(self, task: str, **kwargs: Any) -> GymObservation:
        """Reset the gym-anything environment for a new task.

        Args:
            task: Task description (used by the runner, not the env directly).
            **kwargs: Must include 'task_id' for gym-anything task selection.
                Optionally 'seed' for reproducibility.
        """
        task_id = kwargs.get("task_id", "default")
        seed = kwargs.get("seed")

        if self._env is not None:
            self.close()

        self._env = self._create_env(task_id)
        self._task_id = task_id

        reset_kwargs = {}
        if seed is not None:
            reset_kwargs["seed"] = seed

        raw_obs = self._env.reset(**reset_kwargs)
        return _obs_to_pil(raw_obs)

    def step(self, action: Action) -> GymResult:
        """Execute a Mantis action in the gym-anything environment."""
        if self._env is None:
            raise RuntimeError("Environment not initialized — call reset() first")

        gym_actions = self.translate_action(action)
        raw_obs, reward, done, info = self._env.step(gym_actions)
        observation = _obs_to_pil(raw_obs)

        return GymResult(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
        )

    def close(self) -> None:
        """Release the gym-anything environment."""
        if self._env is not None:
            self._env.close()
            self._env = None
            self._task_id = None

    @property
    def screen_size(self) -> tuple[int, int]:
        return self._resolution

    def translate_action(self, action: Action) -> list[dict]:
        """Convert Mantis Action to gym-anything action list."""
        return _translate_mantis_action(action)

    def translate_observation(self, raw_observation: Any) -> GymObservation:
        """Convert gym-anything observation dict to GymObservation."""
        return _obs_to_pil(raw_observation)

    def __repr__(self) -> str:
        mode = "remote" if self._remote_url else "local"
        return f"GymAnythingAdapter(env_dir={self._env_dir!r}, mode={mode}, task={self._task_id})"
