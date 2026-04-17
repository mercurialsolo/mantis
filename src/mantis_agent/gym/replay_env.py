"""ReplayGymEnv — replay cached screenshots for fast prompt iteration.

Instead of running a real browser (30 min, $5 GPU), replay saved
screenshots from prior runs to test prompt/extraction changes in seconds.

Usage:
    # Capture screenshots during a real run:
    env = XdotoolGymEnv(..., save_screenshots="/data/screenshots/run_001")

    # Replay locally for prompt testing:
    env = ReplayGymEnv("/data/screenshots/run_001")
    runner = GymRunner(brain=brain, env=env, max_steps=40)
    result = runner.run(task=new_prompt)
    # Tests new prompt against same visual sequence — no browser needed

    # Or test a single screenshot directly:
    from replay_env import test_prompt
    test_prompt(brain, "path/to/screenshot.png", "Click the first listing card")
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from PIL import Image

from ..actions import Action, ActionType
from .base import GymEnvironment, GymObservation, GymResult

logger = logging.getLogger(__name__)


class ReplayGymEnv(GymEnvironment):
    """Replays cached screenshots for prompt testing.

    Loads a sequence of screenshots from disk. Each step() returns the
    next screenshot regardless of what action was taken. This lets you
    test different prompts against the same visual sequence.

    For action-dependent replay (branching), use ReplayGymEnv with
    action_map=True which matches actions to cached (action, screenshot) pairs.

    Args:
        screenshot_dir: Directory with numbered screenshots (000.png, 001.png, ...).
        loop: If True, loop back to start when screenshots exhausted.
        viewport: Reported screen size (must match screenshots).
    """

    def __init__(
        self,
        screenshot_dir: str,
        loop: bool = True,
        viewport: tuple[int, int] = (1280, 720),
    ):
        self._dir = Path(screenshot_dir)
        self._loop = loop
        self._viewport = viewport
        self._screenshots: list[Image.Image] = []
        self._index = 0
        self._metadata: dict = {}

        self._load_screenshots()

    def _load_screenshots(self) -> None:
        """Load all screenshots from directory."""
        if not self._dir.exists():
            logger.warning(f"Screenshot dir not found: {self._dir}")
            return

        # Load numbered PNGs
        files = sorted(self._dir.glob("*.png"))
        for f in files:
            try:
                self._screenshots.append(Image.open(f))
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")

        # Load metadata if exists
        meta_path = self._dir / "metadata.json"
        if meta_path.exists():
            self._metadata = json.loads(meta_path.read_text())

        logger.info(f"Loaded {len(self._screenshots)} screenshots from {self._dir}")

    def reset(self, task: str, **kwargs: Any) -> GymObservation:
        self._index = 0
        if not self._screenshots:
            return GymObservation(
                screenshot=Image.new("RGB", self._viewport, "gray"),
                extras={"replay": True, "error": "no screenshots"},
            )
        return GymObservation(
            screenshot=self._screenshots[0],
            extras={"replay": True, "index": 0},
        )

    def step(self, action: Action) -> GymResult:
        self._index += 1
        if self._index >= len(self._screenshots):
            if self._loop:
                self._index = 0
            else:
                # Return last screenshot
                self._index = len(self._screenshots) - 1

        screenshot = self._screenshots[self._index] if self._screenshots else \
            Image.new("RGB", self._viewport, "gray")

        return GymResult(
            observation=GymObservation(
                screenshot=screenshot,
                extras={"replay": True, "index": self._index},
            ),
            reward=0.0,
            done=self._index >= len(self._screenshots) - 1 and not self._loop,
            info={"replay": True},
        )

    def close(self) -> None:
        self._screenshots.clear()

    @property
    def screen_size(self) -> tuple[int, int]:
        return self._viewport

    @property
    def current_url(self) -> str:
        return ""

    def has_session(self, name: str) -> bool:
        return True

    def save_session(self, name: str) -> None:
        pass

    def load_session(self, name: str) -> None:
        pass


def save_screenshot(screenshot: Image.Image, save_dir: str, step: int,
                    action: Action | None = None, thinking: str = "") -> None:
    """Save a screenshot with metadata during a live run.

    Call this from XdotoolGymEnv._capture() or GymRunner step loop
    to build a screenshot cache for replay testing.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{step:04d}.png")
    screenshot.save(path)

    # Append metadata
    meta_path = os.path.join(save_dir, "metadata.json")
    meta = {}
    if os.path.exists(meta_path):
        meta = json.loads(open(meta_path).read())

    if "steps" not in meta:
        meta["steps"] = []

    meta["steps"].append({
        "step": step,
        "file": f"{step:04d}.png",
        "action": str(action) if action else None,
        "thinking": thinking[:200] if thinking else "",
    })

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def test_prompt(brain, screenshot_path: str, prompt: str,
                screen_size: tuple[int, int] = (1280, 720)) -> dict:
    """Test a single prompt against a single screenshot.

    Returns the brain's action and thinking — no env needed.

    Usage:
        result = test_prompt(brain, "screenshots/search_results.png",
                            "Click the first boat listing card")
        print(result["action"], result["thinking"])
    """
    img = Image.open(screenshot_path)
    result = brain.think(
        frames=[img],
        task=prompt,
        action_history=None,
        screen_size=screen_size,
    )
    return {
        "action": str(result.action),
        "action_type": result.action.action_type.value,
        "params": result.action.params,
        "thinking": result.thinking,
        "tokens": result.tokens_used,
    }


def test_prompt_batch(brain, screenshot_dir: str, prompt: str,
                      screen_size: tuple[int, int] = (1280, 720),
                      max_screenshots: int = 10) -> list[dict]:
    """Test a prompt against multiple screenshots.

    Returns results for each screenshot — useful for measuring
    how often a prompt produces the correct action across varied pages.
    """
    results = []
    screenshot_dir = Path(screenshot_dir)
    files = sorted(screenshot_dir.glob("*.png"))[:max_screenshots]

    for f in files:
        result = test_prompt(brain, str(f), prompt, screen_size)
        result["file"] = f.name
        results.append(result)

    return results
