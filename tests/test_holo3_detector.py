"""Tests for Holo3 vision-detector helper parsing."""

from __future__ import annotations

from PIL import Image

from mantis_agent.gym.holo3_detector import find_submit_button


class _VisionBrain:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def detect_with_image(self, prompt: str, image: Image.Image) -> str:
        self.prompts.append(prompt)
        return self.response


def test_find_submit_button_parses_visible_button() -> None:
    brain = _VisionBrain(
        'analysis ignored {"found": true, "x": 42, "y": 99, "label": "Sign in"}'
    )
    image = Image.new("RGB", (100, 100))

    out = find_submit_button(brain, image, plan_intent="Log in to the dashboard")

    assert out == {"x": 42, "y": 99, "label": "Sign in"}
    assert "Log in to the dashboard" in brain.prompts[0]


def test_find_submit_button_returns_none_when_not_found() -> None:
    brain = _VisionBrain('{"found": false, "x": 0, "y": 0, "label": ""}')
    image = Image.new("RGB", (100, 100))

    assert find_submit_button(brain, image) is None
