"""Tests for gallery/lightbox trap detection and recovery."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from PIL import Image, ImageDraw

from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.base import GymObservation, GymResult
from mantis_agent.gym.gallery_trap import detect_gallery_trap, gallery_recovery_actions
from mantis_agent.gym.runner import GymRunner


def _normal_page() -> Image.Image:
    img = Image.new("RGB", (320, 180), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 30, 300, 70), fill=(235, 235, 235))
    draw.text((30, 42), "2026 Tracker Grizzly 12 Jon", fill=(20, 20, 20))
    draw.rectangle((20, 90, 120, 150), fill=(40, 100, 180))
    draw.rectangle((140, 95, 300, 120), fill=(245, 245, 245))
    return img


def _gallery_page() -> Image.Image:
    img = Image.new("RGB", (320, 180), (8, 8, 8))
    draw = ImageDraw.Draw(img)
    draw.rectangle((60, 35, 260, 145), fill=(150, 175, 190))
    draw.rectangle((70, 45, 250, 135), outline=(220, 230, 235), width=3)
    draw.text((145, 10), "1 of 68", fill=(240, 240, 240))
    draw.text((270, 10), "Close X", fill=(240, 240, 240))
    return img


def test_detect_gallery_trap_from_visual_overlay() -> None:
    detection = detect_gallery_trap(_gallery_page())

    assert detection.detected
    assert detection.confidence >= 0.65
    assert "dark" in detection.reason or "central" in detection.reason


def test_detect_gallery_trap_from_text_signal() -> None:
    detection = detect_gallery_trap(_normal_page(), text="The page shows a photo gallery 1 of 85")

    assert detection.detected
    assert detection.signals["text_gallery_signal"] is True


def test_uniform_dark_page_is_not_enough_without_text() -> None:
    detection = detect_gallery_trap(Image.new("RGB", (320, 180), (2, 2, 2)))

    assert not detection.detected


def test_normal_listing_page_is_not_gallery() -> None:
    detection = detect_gallery_trap(_normal_page())

    assert not detection.detected


def test_gallery_recovery_actions_are_bounded() -> None:
    actions = gallery_recovery_actions()

    assert [a.action_type for a in actions] == [ActionType.KEY_PRESS, ActionType.KEY_PRESS]
    assert [a.params["keys"] for a in actions] == ["escape", "alt+left"]


class _ClickThenGalleryBrain:
    def __init__(self) -> None:
        self.calls = 0

    def think(self, **_: Any) -> Any:
        self.calls += 1
        return SimpleNamespace(
            action=Action(ActionType.CLICK, {"x": 80, "y": 110}, reasoning="click listing title"),
            thinking="Click the listing title text.",
            predicted_outcome="detail page opens",
        )


class _GalleryRecoveryEnv:
    def __init__(self) -> None:
        self.actions: list[Action] = []
        self._screen = _normal_page()

    def reset(self, task: str, **kwargs: Any) -> GymObservation:
        return GymObservation(screenshot=_normal_page(), extras={"url": "https://example.test/list"})

    def step(self, action: Action) -> GymResult:
        self.actions.append(action)
        if action.action_type == ActionType.CLICK:
            self._screen = _gallery_page()
            return GymResult(
                observation=GymObservation(screenshot=self._screen),
                reward=0.0,
                done=False,
                info={"url": "https://example.test/list", "title": "Gallery"},
            )
        if action.action_type == ActionType.KEY_PRESS and action.params.get("keys") == "escape":
            self._screen = _normal_page()
            return GymResult(
                observation=GymObservation(screenshot=self._screen),
                reward=0.0,
                done=False,
                info={"url": "https://example.test/detail", "title": "Detail"},
            )
        return GymResult(
            observation=GymObservation(screenshot=self._screen),
            reward=0.0,
            done=False,
            info={"url": "https://example.test/detail", "title": "Detail"},
        )

    def close(self) -> None:
        pass

    @property
    def screen_size(self) -> tuple[int, int]:
        return (320, 180)


def test_runner_recovers_gallery_trap_after_click() -> None:
    env = _GalleryRecoveryEnv()
    runner = GymRunner(
        brain=_ClickThenGalleryBrain(),
        env=env,
        max_steps=1,
        frames_per_inference=1,
    )

    result = runner.run(task="Click listing title", task_id="gallery")

    assert [a.action_type for a in env.actions] == [ActionType.CLICK, ActionType.KEY_PRESS]
    assert env.actions[1].params["keys"] == "escape"
    assert result.trajectory[0].observed_state["gallery_trap_detected"] is True
    assert result.trajectory[0].observed_state["gallery_recovery_actions"] == ["escape"]
    assert result.trajectory[0].observed_state["gallery_recovery_success"] is True
    assert "gallery trap detected" in result.trajectory[0].feedback
