"""SiteProber — screenshot-based site analysis without a brain model.

Navigates to a URL, captures screenshots at multiple scroll positions,
and uses Claude Sonnet to identify site structure: filters, listing
containers, pagination controls, detail-page patterns, and signals for
dealers/sponsored content.

No brain model (Holo3/Gemma4) required — uses direct navigation only.
Cost: ~$0.02 per probe (4-6 screenshots + analysis prompt).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

from PIL import Image

from .objective import ObjectiveSpec

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """What we learned about a site from screenshot analysis."""

    url: str = ""
    domain: str = ""
    page_type: str = ""  # "search_results", "detail", "login", "error"
    filters_detected: list[dict[str, Any]] = field(default_factory=list)
    listing_container: dict[str, Any] = field(default_factory=dict)
    pagination_controls: dict[str, Any] = field(default_factory=dict)
    detail_page_pattern: dict[str, Any] = field(default_factory=dict)
    dealer_signals: list[str] = field(default_factory=list)
    sponsored_signals: list[str] = field(default_factory=list)
    estimated_listings_per_page: int = 0
    screenshots: list[tuple[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "domain": self.domain,
            "page_type": self.page_type,
            "filters_detected": self.filters_detected,
            "listing_container": self.listing_container,
            "pagination_controls": self.pagination_controls,
            "detail_page_pattern": self.detail_page_pattern,
            "dealer_signals": self.dealer_signals,
            "sponsored_signals": self.sponsored_signals,
            "estimated_listings_per_page": self.estimated_listings_per_page,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProbeResult:
        return cls(
            url=data.get("url", ""),
            domain=data.get("domain", ""),
            page_type=data.get("page_type", ""),
            filters_detected=data.get("filters_detected", []),
            listing_container=data.get("listing_container", {}),
            pagination_controls=data.get("pagination_controls", {}),
            detail_page_pattern=data.get("detail_page_pattern", {}),
            dealer_signals=data.get("dealer_signals", []),
            sponsored_signals=data.get("sponsored_signals", []),
            estimated_listings_per_page=data.get("estimated_listings_per_page", 0),
        )


PROBE_RESULTS_PROMPT = """\
You are analyzing a search results page for a CUA (Computer Use Agent).

OBJECTIVE: {objective}
TARGET ENTITY: {target_entity}

I'm showing you {n_screenshots} screenshots of the page at different scroll positions.

Analyze the page and output ONLY valid JSON:
{{
  "page_type": "search_results or detail or login or error or other",
  "filters_detected": [
    {{"name": "filter name", "options": ["option1", "option2"], "location": "left sidebar or top bar or ..."}}
  ],
  "listing_container": {{
    "description": "how listings appear (cards with photo+title+price, table rows, etc.)",
    "estimated_count": 25,
    "has_photos": true,
    "card_layout": "horizontal or vertical or grid"
  }},
  "pagination_controls": {{
    "type": "numbered or next_button or infinite_scroll or load_more or none",
    "location": "bottom center or bottom right or ...",
    "next_text": "Next or > or ..."
  }},
  "dealer_signals": ["text patterns that indicate dealer/company listings, e.g. More From This Dealer"],
  "sponsored_signals": ["text patterns that indicate sponsored/ad content, e.g. Sponsored, Advertisement"],
  "estimated_listings_per_page": 25
}}

Focus on:
- What UI elements are filter controls (dropdowns, checkboxes, links)?
- How are search results displayed (cards, rows, tiles)?
- Where is the pagination (page numbers, Next button, infinite scroll)?
- What distinguishes organic results from sponsored/dealer content?
"""


PROBE_DETAIL_PROMPT = """\
You are analyzing a detail/item page for a CUA (Computer Use Agent).

OBJECTIVE: {objective}
TARGET ENTITY: {target_entity}

I'm showing you {n_screenshots} screenshots of a detail page at different scroll positions.

Analyze the page and output ONLY valid JSON:
{{
  "url_pattern": "pattern in the URL that identifies detail pages, e.g. /boat/<slug>/",
  "has_phone_section": true,
  "phone_location": "where phone numbers appear, e.g. right sidebar contact section",
  "has_expandable_sections": true,
  "expandable_sections": ["section names that can be expanded, e.g. Description, More Details"],
  "expand_controls": ["text of expand buttons, e.g. Show more, Read more, See full description"],
  "seller_info_location": "where seller info appears",
  "key_fields_visible": ["fields visible without scrolling, e.g. title, price, year"],
  "key_fields_hidden": ["fields that require scrolling or expanding, e.g. phone, description"]
}}
"""


class SiteProber:
    """Probe a site via navigate + screenshot + Claude analysis.

    No brain model required — uses env for navigation and Claude Sonnet
    for screenshot analysis.
    """

    def __init__(
        self,
        env: Any,
        api_key: str = "",
        model: str = "claude-sonnet-4-20250514",
    ):
        self.env = env
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model

    def probe(self, url: str, objective: ObjectiveSpec) -> ProbeResult:
        """Navigate to URL, capture multi-viewport screenshots, analyze with Claude."""
        from ..actions import Action, ActionType

        logger.info("Probing %s", url)
        result = ProbeResult(url=url, domain=objective.domains[0] if objective.domains else "")

        # Navigate
        self.env.reset(task="probe", start_url=url)
        time.sleep(4)

        # Capture screenshots at multiple scroll positions
        screenshots: list[tuple[str, Image.Image]] = []
        labels = ["top", "mid1", "mid2", "bottom"]
        for i, label in enumerate(labels):
            img = self.env.screenshot()
            if img:
                screenshots.append((label, img))
            if i < len(labels) - 1:
                self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Page_Down"}))
                time.sleep(1)

        if not screenshots:
            logger.warning("No screenshots captured during probe")
            return result

        result.screenshots = screenshots

        # Analyze with Claude
        analysis = self._analyze_results_page(screenshots, objective)
        if analysis:
            result.page_type = analysis.get("page_type", "")
            result.filters_detected = analysis.get("filters_detected", [])
            result.listing_container = analysis.get("listing_container", {})
            result.pagination_controls = analysis.get("pagination_controls", {})
            result.dealer_signals = analysis.get("dealer_signals", [])
            result.sponsored_signals = analysis.get("sponsored_signals", [])
            result.estimated_listings_per_page = analysis.get("estimated_listings_per_page", 0)

        # Scroll back to top
        self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
        time.sleep(1)

        logger.info(
            "Probe complete: %s (%s, %d filters, %d listings/page)",
            url,
            result.page_type,
            len(result.filters_detected),
            result.estimated_listings_per_page,
        )
        return result

    def probe_detail_page(self, detail_url: str, objective: ObjectiveSpec) -> dict[str, Any]:
        """Probe a single detail page to learn extraction patterns."""
        from ..actions import Action, ActionType

        logger.info("Probing detail page: %s", detail_url)
        self.env.reset(task="probe_detail", start_url=detail_url)
        time.sleep(4)

        screenshots: list[tuple[str, Image.Image]] = []
        for i, label in enumerate(["detail_top", "detail_mid", "detail_bottom"]):
            img = self.env.screenshot()
            if img:
                screenshots.append((label, img))
            if i < 2:
                self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Page_Down"}))
                time.sleep(1)

        if not screenshots:
            return {}

        return self._analyze_detail_page(screenshots, objective)

    def _analyze_results_page(
        self,
        screenshots: list[tuple[str, Image.Image]],
        objective: ObjectiveSpec,
    ) -> dict[str, Any]:
        prompt = PROBE_RESULTS_PROMPT.format(
            objective=objective.raw_text[:500],
            target_entity=objective.target_entity,
            n_screenshots=len(screenshots),
        )
        return self._call_claude_vision(prompt, screenshots)

    def _analyze_detail_page(
        self,
        screenshots: list[tuple[str, Image.Image]],
        objective: ObjectiveSpec,
    ) -> dict[str, Any]:
        prompt = PROBE_DETAIL_PROMPT.format(
            objective=objective.raw_text[:500],
            target_entity=objective.target_entity,
            n_screenshots=len(screenshots),
        )
        return self._call_claude_vision(prompt, screenshots)

    def _call_claude_vision(
        self,
        text_prompt: str,
        screenshots: list[tuple[str, Image.Image]],
    ) -> dict[str, Any]:
        """Send screenshots + text prompt to Claude and parse JSON response."""
        import requests

        if not self.api_key:
            logger.warning("SiteProber: no API key")
            return {}

        content: list[dict[str, Any]] = [{"type": "text", "text": text_prompt}]
        for label, img in screenshots:
            buf = BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": b64},
                }
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
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": content}],
                },
                timeout=30,
            )
            if resp.status_code != 200:
                logger.warning("SiteProber API error: %s", resp.status_code)
                return {}

            response_text = ""
            for block in resp.json().get("content", []):
                if block.get("type") == "text":
                    response_text = block["text"].strip()
                    break

            return self._parse_json(response_text)
        except Exception as e:
            logger.warning("SiteProber call failed: %s", e)
            return {}

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("SiteProber: failed to parse JSON response")
            return {}
