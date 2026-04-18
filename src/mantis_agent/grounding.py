"""Grounded click targeting — refine approximate coordinates to precise UI elements.

Separates "what to click" (brain reasoning) from "where to click" (visual grounding).

The brain outputs approximate coordinates or a text description ("click the first listing").
The grounding model takes the screenshot + description and returns precise pixel coordinates
of the target UI element.

Supports multiple backends:
- OS-Atlas (HuggingFace, specialized for UI grounding)
- Region-based heuristic (no model needed, uses image analysis)
- Passthrough (no grounding, uses brain's coordinates as-is)

Architecture:
    Brain → Action(CLICK, x=500, y=300, reasoning="click listing card")
      ↓
    GroundingModel.ground(screenshot, description, x, y)
      ↓
    Refined Action(CLICK, x=487, y=312)  ← precise element center
      ↓
    xdotool executes
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class GroundingResult:
    """Result of grounding a click target."""
    x: int
    y: int
    confidence: float = 1.0
    description: str = ""


class GroundingModel(ABC):
    """Abstract grounding model interface."""

    @abstractmethod
    def ground(
        self,
        screenshot: Image.Image,
        description: str,
        initial_x: int | None = None,
        initial_y: int | None = None,
    ) -> GroundingResult:
        """Refine a click target from description + approximate coords.

        Args:
            screenshot: Current screen image.
            description: What to click (from brain's reasoning).
            initial_x: Brain's approximate X coordinate (may be None).
            initial_y: Brain's approximate Y coordinate (may be None).

        Returns:
            GroundingResult with refined coordinates.
        """
        ...


class PassthroughGrounding(GroundingModel):
    """No grounding — use brain's coordinates as-is. For testing."""

    def ground(self, screenshot, description, initial_x=None, initial_y=None):
        return GroundingResult(
            x=initial_x or screenshot.width // 2,
            y=initial_y or screenshot.height // 2,
            confidence=0.5,
            description="passthrough",
        )


class RegionGrounding(GroundingModel):
    """Heuristic grounding — avoid known bad regions, snap to content area.

    Uses spatial heuristics to avoid footer (social icons), header (menus),
    and sidebar (ads). Nudges clicks toward the main content area.

    This is a zero-model fallback that prevents the worst misclicks.
    """

    def __init__(self, viewport: tuple[int, int] = (1280, 720)):
        self.w, self.h = viewport
        # Define safe content region (excludes header, footer, sidebar)
        self.safe_top = 80       # Below header/nav bar
        self.safe_bottom = self.h - 60  # Above footer with social icons
        self.safe_left = 20
        self.safe_right = self.w - 20

    def ground(self, screenshot, description, initial_x=None, initial_y=None):
        x = initial_x or self.w // 2
        y = initial_y or self.h // 2

        # Clamp to safe region
        clamped = False
        if y > self.safe_bottom:
            y = self.safe_bottom - 20
            clamped = True
        if y < self.safe_top:
            y = self.safe_top + 20
            clamped = True
        if x < self.safe_left:
            x = self.safe_left + 20
            clamped = True
        if x > self.safe_right:
            x = self.safe_right - 20
            clamped = True

        if clamped:
            logger.info(f"RegionGrounding: clamped ({initial_x},{initial_y}) → ({x},{y})")

        return GroundingResult(
            x=x, y=y,
            confidence=0.7 if not clamped else 0.4,
            description=f"region-clamped" if clamped else "in-bounds",
        )


class ClaudeGrounding(GroundingModel):
    """Claude-based grounding — uses Anthropic API for precise click targeting.

    A DIFFERENT model from the executor (Gemma4/Holo3) to avoid same-model bias.
    The executor model tends to click photos instead of text — Claude doesn't
    have this bias and can accurately locate text elements.

    Cost: ~$0.005-0.01 per grounding call (1 screenshot + short prompt).
    Only called for CLICK actions, not every step.
    """

    GROUNDING_PROMPT = """\
Look at this screenshot ({width}x{height} pixels).

The user wants to click near ({init_x}, {init_y}). But that coordinate might be on a PHOTO which would open a gallery trap.

Find the nearest CLICKABLE TEXT near ({init_x}, {init_y}) that matches this intent:
{description}

RULES:
- Look NEAR the original coordinates ({init_x}, {init_y}) — within ~200 pixels
- Find TEXT that is clickable (title, link, button) — NOT a photo/image
- On boat listing sites, the clickable title text (Year Make Model) is DIRECTLY BELOW each photo
- The title text is typically at the BOTTOM EDGE of each listing card
- Do NOT jump to a completely different part of the page (like header, sidebar, or sort buttons)

Output ONLY two numbers: x y
Nothing else."""

    def __init__(self, api_key: str = "", model: str = "claude-sonnet-4-20250514"):
        import os
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model

    def ground(self, screenshot, description, initial_x=None, initial_y=None):
        import base64
        import re
        from io import BytesIO

        import requests

        if not description or not self.api_key:
            return GroundingResult(
                x=initial_x or screenshot.width // 2,
                y=initial_y or screenshot.height // 2,
                confidence=0.3,
                description="no api key or description",
            )

        buf = BytesIO()
        screenshot.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        prompt = self.GROUNDING_PROMPT.format(
            description=description,
            width=screenshot.width,
            height=screenshot.height,
            init_x=initial_x or screenshot.width // 2,
            init_y=initial_y or screenshot.height // 2,
        )

        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": 30,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                            {"type": "text", "text": prompt},
                        ],
                    }],
                },
                timeout=15,
            )

            if resp.status_code != 200:
                logger.warning(f"Claude grounding API error: {resp.status_code}")
                return GroundingResult(
                    x=initial_x or screenshot.width // 2,
                    y=initial_y or screenshot.height // 2,
                    confidence=0.2,
                    description="api error",
                )

            text = ""
            for block in resp.json().get("content", []):
                if block.get("type") == "text":
                    text = block["text"].strip()
                    break

            # Parse "x y" or "x=N y=N"
            nums = re.findall(r'\d+', text)
            if len(nums) >= 2:
                gx, gy = int(nums[0]), int(nums[1])
                gx = max(0, min(gx, screenshot.width - 1))
                gy = max(0, min(gy, screenshot.height - 1))
                logger.info(f"ClaudeGrounding: '{description[:40]}' → ({gx},{gy})")
                return GroundingResult(x=gx, y=gy, confidence=0.9, description=description)

        except Exception as e:
            logger.warning(f"Claude grounding failed: {e}")

        return GroundingResult(
            x=initial_x or screenshot.width // 2,
            y=initial_y or screenshot.height // 2,
            confidence=0.2,
            description="grounding failed",
        )


class LLMGrounding(GroundingModel):
    """LLM-based grounding via OpenAI-compatible API (llama.cpp, vLLM, etc).

    NOTE: Using the same model for grounding as for action selection
    doesn't fix visual biases. Use ClaudeGrounding for a different model.
    """

    GROUNDING_PROMPT = """\
Look at this screenshot ({width}x{height} pixels). I need to click on a specific TEXT element.

TARGET: {description}

RULES:
- Find the TEXT/LINK element, NOT any image or photo
- If the target is near a large photo, find the TEXT that is BELOW or BESIDE the photo
- NEVER return coordinates inside a large rectangular photo area
- Return the CENTER of the text element

Output ONLY: x=NUMBER y=NUMBER
Nothing else."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        model: str = "gemma-4",
        max_retries: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries

    def ground(self, screenshot, description, initial_x=None, initial_y=None):
        import base64
        import json
        from io import BytesIO

        import requests

        if not description:
            return GroundingResult(
                x=initial_x or screenshot.width // 2,
                y=initial_y or screenshot.height // 2,
                confidence=0.3,
                description="no description for grounding",
            )

        # Encode screenshot
        buf = BytesIO()
        screenshot.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        prompt = self.GROUNDING_PROMPT.format(
            description=description,
            width=screenshot.width,
            height=screenshot.height,
            init_x=initial_x or screenshot.width // 2,
            init_y=initial_y or screenshot.height // 2,
        )

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.model,
                        "max_tokens": 50,
                        "temperature": 0.0,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                                {"type": "text", "text": prompt},
                            ],
                        }],
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"]

                # Parse "x=NUMBER y=NUMBER"
                x_match = re.search(r'x\s*=\s*(\d+)', text)
                y_match = re.search(r'y\s*=\s*(\d+)', text)
                if x_match and y_match:
                    gx = int(x_match.group(1))
                    gy = int(y_match.group(1))
                    # Sanity check bounds
                    gx = max(0, min(gx, screenshot.width - 1))
                    gy = max(0, min(gy, screenshot.height - 1))
                    logger.info(f"LLMGrounding: '{description[:40]}' → ({gx},{gy})")
                    return GroundingResult(x=gx, y=gy, confidence=0.8, description=description)

            except Exception as e:
                logger.warning(f"LLMGrounding attempt {attempt+1} failed: {e}")

        # Fallback to initial coords
        return GroundingResult(
            x=initial_x or screenshot.width // 2,
            y=initial_y or screenshot.height // 2,
            confidence=0.2,
            description=f"grounding failed, using fallback",
        )
