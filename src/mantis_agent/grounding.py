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
from typing import TYPE_CHECKING

from PIL import Image

from ._anthropic.client import encode_screenshot_for_claude

if TYPE_CHECKING:
    from .grounding_cache import GroundingCache

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
        *,
        force_compute: bool = False,
    ) -> GroundingResult:
        """Refine a click target from description + approximate coords.

        Args:
            screenshot: Current screen image.
            description: What to click (from brain's reasoning).
            initial_x: Brain's approximate X coordinate (may be None).
            initial_y: Brain's approximate Y coordinate (may be None).
            force_compute: When True, bypass any attached
                :class:`~.grounding_cache.GroundingCache` and force a
                fresh remote call. High-risk click paths (#181) opt in
                so a stale cached coordinate doesn't pin a regression.
                Default False — caches remain hot for routine clicks.

        Returns:
            GroundingResult with refined coordinates.
        """
        ...


class PassthroughGrounding(GroundingModel):
    """No grounding — use brain's coordinates as-is. For testing."""

    def ground(self, screenshot, description, initial_x=None, initial_y=None, *, force_compute=False):
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

    def ground(self, screenshot, description, initial_x=None, initial_y=None, *, force_compute=False):
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
            description="region-clamped" if clamped else "in-bounds",
        )


class ClaudeGrounding(GroundingModel):
    """Claude-based grounding — uses Anthropic API for precise click targeting.

    A DIFFERENT model from the executor (Gemma4/Holo3) to avoid same-model bias.
    The executor model tends to click photos instead of text — Claude doesn't
    have this bias and can accurately locate text elements.

    Cost: ~$0.005-0.01 per grounding call (1 screenshot + short prompt).
    Only called for CLICK actions, not every step.
    """

    # #715 — split into a static SYSTEM prompt (cacheable) + a
    # per-call USER template (the dynamic geometry + description).
    # The system prompt is the same for every grounding call in a run,
    # so caching it saves ~150 input tokens / call after the first.
    SYSTEM_PROMPT = """\
Find the closest CLICKABLE TEXT element to the requested position.

RULES:
- Return the center coordinates of the nearest TEXT element (not an image)
- If the coordinates are on an image/photo, find the nearest text ADJACENT to that image
- Stay close to the original position — find the NEAREST text, not any text on the page

Output ONLY two numbers: x y
Nothing else."""

    GROUNDING_PROMPT = """\
Screenshot is {width}x{height} pixels. The user wants to click near \
({init_x}, {init_y}) to: {description}"""

    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-sonnet-4-6",
        cache: "GroundingCache | None" = None,
    ):
        import os
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        # #117 step 2: optional coordinate cache. When set, identical
        # (frame-region + description) pairs short-circuit the API call.
        self.cache = cache

    def ground(self, screenshot, description, initial_x=None, initial_y=None, *, force_compute=False):
        # #117 step 2: cache wrapper. Skip on empty description (caches
        # nothing meaningful) and on no-API-key (would just return the
        # fallback every time and pollute the cache).
        # #181: ``force_compute=True`` bypasses the cache so high-risk
        # clicks never inherit a stale coordinate.
        # Epic #362: time the whole call (cache hit or API miss) under
        # the ``claude_ground`` bucket. Cache hits are near-zero but
        # still credited so per-step breakdowns sum cleanly.
        import time as _time_mod
        t0 = _time_mod.monotonic()
        try:
            if not force_compute and self.cache is not None and description and self.api_key:
                return self.cache.lookup_or_compute(
                    screenshot, description,
                    lambda: self._ground_remote(screenshot, description, initial_x, initial_y),
                    initial_x=initial_x, initial_y=initial_y,
                )
            return self._ground_remote(screenshot, description, initial_x, initial_y)
        finally:
            try:
                from .gym.time_meter import record_to_current
                record_to_current("claude_ground", _time_mod.monotonic() - t0)
            except Exception as exc:  # noqa: BLE001 — observability, never fatal
                logger.debug("claude_ground time_meter credit failed: %s", exc)

    def _ground_remote(self, screenshot, description, initial_x=None, initial_y=None):
        import re

        if not description or not self.api_key:
            return GroundingResult(
                x=initial_x or screenshot.width // 2,
                y=initial_y or screenshot.height // 2,
                confidence=0.3,
                description="no api key or description",
            )

        # #518 — encode at the configured JPEG/PNG/WEBP setting; same
        # dimensions as the source so the coordinates Claude returns
        # are still in the screenshot's pixel space.
        b64, media_type = encode_screenshot_for_claude(screenshot)

        ix = initial_x or screenshot.width // 2
        iy = initial_y or screenshot.height // 2

        prompt = self.GROUNDING_PROMPT.format(
            description=description,
            width=screenshot.width,
            height=screenshot.height,
            init_x=ix,
            init_y=iy,
        )

        # #715 — partition into cached system + per-call user.
        from ._anthropic.cache import as_cached_system, extract_cache_telemetry

        try:
            # #836: route through the shared retry client. Grounding
            # is on the click hot path so per-call latency matters —
            # cap retries at 2 (one extra attempt on transient).
            from ._anthropic.client import AnthropicToolUseClient
            _g_client = AnthropicToolUseClient(
                api_key=self.api_key, model=self.model,
                log_prefix="[grounding]",
            )
            resp = _g_client.post_messages_with_retry(
                {
                    "model": self.model,
                    "max_tokens": 30,
                    "system": as_cached_system(self.SYSTEM_PROMPT),
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                            {"type": "text", "text": prompt},
                        ],
                    }],
                },
                timeout=15, max_attempts=2,
            )

            if resp is None:
                logger.warning("Claude grounding: Anthropic network exhaustion")
                return GroundingResult(
                    x=initial_x or screenshot.width // 2,
                    y=initial_y or screenshot.height // 2,
                    confidence=0.2,
                    description="network exhausted",
                )
            if resp.status_code != 200:
                logger.warning(f"Claude grounding API error: {resp.status_code}")
                return GroundingResult(
                    x=initial_x or screenshot.width // 2,
                    y=initial_y or screenshot.height // 2,
                    confidence=0.2,
                    description="api error",
                )
            # Cache telemetry — Modal suppresses INFO; WARNING-level so
            # operators can audit hit rate without an env-var dance.
            resp_json = resp.json()
            tele = extract_cache_telemetry(resp_json)
            if tele.get("cache_read_input_tokens", 0) > 0:
                logger.warning(
                    "  [cache] grounding: read=%d created=%d input=%d",
                    tele.get("cache_read_input_tokens", 0),
                    tele.get("cache_creation_input_tokens", 0),
                    tele.get("input_tokens", 0),
                )
            # Cost meter (#675 A/B follow-up).
            from .observability.claude_cost_meter import record_from_response
            record_from_response(
                source="grounding", model=self.model, response_json=resp_json,
            )

            text = ""
            for block in resp_json.get("content", []):
                if block.get("type") == "text":
                    text = block["text"].strip()
                    break

            # Parse "x y" or "x=N y=N"
            nums = re.findall(r'\d+', text)
            if len(nums) >= 2:
                gx, gy = int(nums[0]), int(nums[1])
                gx = max(0, min(gx, screenshot.width - 1))
                gy = max(0, min(gy, screenshot.height - 1))
                # Confidence scales with proximity to initial coords
                # Near refinements (< 100px) = high confidence
                # Far jumps (> 300px) = low confidence
                dx = abs(gx - (initial_x or screenshot.width // 2))
                dy = abs(gy - (initial_y or screenshot.height // 2))
                dist = (dx**2 + dy**2) ** 0.5
                conf = max(0.3, min(0.95, 1.0 - dist / 500))
                logger.info(f"ClaudeGrounding: '{description[:40]}' → ({gx},{gy}) conf={conf:.2f} dist={dist:.0f}")
                return GroundingResult(x=gx, y=gy, confidence=conf, description=description)

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
        cache: "GroundingCache | None" = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.cache = cache

    def ground(self, screenshot, description, initial_x=None, initial_y=None, *, force_compute=False):
        # #181: ``force_compute=True`` bypasses the cache for high-risk
        # clicks so a stale cached coordinate cannot pin a regression.
        if not force_compute and self.cache is not None and description:
            return self.cache.lookup_or_compute(
                screenshot, description,
                lambda: self._ground_remote(screenshot, description, initial_x, initial_y),
                initial_x=initial_x, initial_y=initial_y,
            )
        return self._ground_remote(screenshot, description, initial_x, initial_y)

    def _ground_remote(self, screenshot, description, initial_x=None, initial_y=None):
        import requests

        if not description:
            return GroundingResult(
                x=initial_x or screenshot.width // 2,
                y=initial_y or screenshot.height // 2,
                confidence=0.3,
                description="no description for grounding",
            )

        # #518 — encode at the configured format/quality. This path
        # talks to a vLLM /chat/completions endpoint (OpenAI-compat,
        # NOT Anthropic) so the env knobs still apply for network-
        # payload reduction even though the per-token cost isn't
        # billed against an Anthropic invoice.
        b64, media_type = encode_screenshot_for_claude(screenshot)

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
                                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
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
            description="grounding failed, using fallback",
        )
