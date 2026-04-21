"""Claude-based data extraction — read structured data from screenshots.

Uses Claude Sonnet to extract boat listing data from a single screenshot.
Same API pattern as ClaudeGrounding but for data extraction instead of
click targeting.

Architecture:
    Holo3 navigates → clicks listing → screenshot captured
      ↓
    ClaudeExtractor.extract(screenshot) → structured data
      ↓
    Holo3 navigates back

Cost: ~$0.003-0.005 per extraction call (1 screenshot + short prompt).
Called once or twice per listing (top of page + after scrolling).

Usage:
    extractor = ClaudeExtractor()
    data = extractor.extract(screenshot)
    # data = {"year": "2018", "make": "Sea Ray", "model": "240 Sundeck",
    #         "price": "$42,500", "phone": "305-555-1234", "url": "boattrader.com/..."}
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import dataclass, field
from io import BytesIO

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Structured data extracted from a listing screenshot."""
    year: str = ""
    make: str = ""
    model: str = ""
    price: str = ""
    phone: str = ""
    url: str = ""
    seller: str = ""
    is_dealer: bool = False
    raw_response: str = ""
    confidence: float = 0.0

    def to_summary(self) -> str:
        """Format as the VIABLE summary string."""
        parts = []
        if self.year: parts.append(f"Year: {self.year}")
        if self.make: parts.append(f"Make: {self.make}")
        if self.model: parts.append(f"Model: {self.model}")
        if self.price: parts.append(f"Price: {self.price}")
        if self.phone: parts.append(f"Phone: {self.phone}")
        else: parts.append("Phone: none")
        if self.url: parts.append(f"URL: {self.url}")
        if self.seller: parts.append(f"Seller: {self.seller}")
        return "VIABLE | " + " | ".join(parts) if parts else ""

    def is_viable(self) -> bool:
        """Has enough data to be a useful lead."""
        return bool(self.year and self.make)


EXTRACT_PROMPT = """\
Look at this screenshot of a boat listing page.

Extract ALL of the following data visible on the page:

1. URL: Read the browser address bar at the TOP of the screen
2. Year: The model year (4-digit number like 2018)
3. Make: The manufacturer (e.g. Sea Ray, Grady-White, Boston Whaler)
4. Model: The model name (e.g. 240 Sundeck, Freedom 235)
5. Price: The asking price (e.g. $42,500)
6. Phone: Any phone number visible (10+ digits, format like 305-555-1234)
7. Seller: The seller name if shown
8. Is this a dealer listing? (look for "More From This Dealer" or "View Dealer Website")

For phone numbers: look in Description, Seller Notes, or contact sections.
NOT phone numbers: prices, years, zip codes, model numbers, HP ratings.

Output ONLY valid JSON:
{"year": "", "make": "", "model": "", "price": "", "phone": "", "url": "", "seller": "", "is_dealer": false}
"""

EXTRACT_SCROLLED_PROMPT = """\
Look at this screenshot. You have scrolled down on a boat listing page.

Look for:
1. Phone number in the Description or Seller Notes section (format: 305-555-1234, 10+ digits)
2. Seller name
3. Any additional details not captured from the top of the page

NOT phone numbers: prices ($45,000), years (2020), zip codes (33101), HP ratings.

Output ONLY valid JSON:
{"phone": "", "seller": "", "additional_info": ""}
"""


class ClaudeExtractor:
    """Extract structured data from listing screenshots using Claude Sonnet.

    Same API pattern as ClaudeGrounding — cheap, fast, vision-capable.
    Called 1-2 times per listing (top screenshot + scrolled screenshot).
    """

    def __init__(self, api_key: str = "", model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model

    def _call(self, screenshot: Image.Image, prompt: str) -> str:
        """Call Claude API with screenshot + prompt."""
        import requests

        if not self.api_key:
            logger.warning("ClaudeExtractor: no API key")
            return ""

        buf = BytesIO()
        screenshot.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

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
                    "max_tokens": 200,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                            {"type": "text", "text": prompt},
                        ],
                    }],
                },
                timeout=20,
            )
            if resp.status_code != 200:
                logger.warning(f"ClaudeExtractor API error: {resp.status_code}")
                return ""
            for block in resp.json().get("content", []):
                if block.get("type") == "text":
                    return block["text"].strip()
        except Exception as e:
            logger.warning(f"ClaudeExtractor failed: {e}")
        return ""

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Parse JSON from response (handles markdown fences)."""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {}

    def extract(self, screenshot: Image.Image) -> ExtractionResult:
        """Extract listing data from a detail page screenshot.

        Call this right after navigating to a listing page.
        Returns structured data with year, make, model, price, phone, URL.
        """
        text = self._call(screenshot, EXTRACT_PROMPT)
        parsed = self._parse_json(text)

        if not parsed:
            return ExtractionResult(raw_response=text, confidence=0.1)

        return ExtractionResult(
            year=str(parsed.get("year", "")),
            make=str(parsed.get("make", "")),
            model=str(parsed.get("model", "")),
            price=str(parsed.get("price", "")),
            phone=str(parsed.get("phone", "")),
            url=str(parsed.get("url", "")),
            seller=str(parsed.get("seller", "")),
            is_dealer=bool(parsed.get("is_dealer", False)),
            raw_response=text,
            confidence=0.9,
        )

    def extract_scrolled(self, screenshot: Image.Image) -> dict:
        """Extract additional data from a scrolled-down screenshot.

        Call this after scrolling past photos to the Description section.
        Primarily looking for phone numbers and seller info.
        """
        text = self._call(screenshot, EXTRACT_SCROLLED_PROMPT)
        parsed = self._parse_json(text)
        return parsed or {}

    def extract_full(self, top_screenshot: Image.Image,
                     scrolled_screenshot: Image.Image | None = None) -> ExtractionResult:
        """Full extraction: top of page + scrolled section.

        Combines data from both screenshots for maximum coverage.
        """
        result = self.extract(top_screenshot)

        if scrolled_screenshot and not result.phone:
            extra = self.extract_scrolled(scrolled_screenshot)
            if extra.get("phone"):
                result.phone = extra["phone"]
            if extra.get("seller") and not result.seller:
                result.seller = extra["seller"]

        return result

    def find_click_target(
        self,
        screenshot: Image.Image,
        skip_count: int = 0,
    ) -> tuple[int, int, str] | None:
        """Find the next listing to click on a search results page.

        Claude identifies the Nth unprocessed listing and returns its
        title text coordinates for Holo3 to click.

        Args:
            screenshot: Current search results page.
            skip_count: Number of listings to skip (already processed).

        Returns:
            (x, y, title_text) or None if no more listings found.
        """
        ordinal = {0: "first", 1: "second", 2: "third", 3: "fourth",
                   4: "fifth", 5: "sixth", 6: "seventh", 7: "eighth"}.get(
            skip_count, f"#{skip_count + 1}"
        )

        prompt = (
            f"Look at this search results page ({screenshot.width}x{screenshot.height} pixels).\n\n"
            f"Find the {ordinal} boat listing card on this page. "
            f"Each listing has a large photo on top and title text below it.\n\n"
            f"Return the CENTER coordinates of the listing TITLE TEXT "
            f"(the blue/clickable text showing Year Make Model, NOT the photo).\n\n"
            f"Output ONLY valid JSON: {{\"x\": N, \"y\": N, \"title\": \"the title text\"}}\n"
            f"If no {ordinal} listing exists, output: {{\"x\": 0, \"y\": 0, \"title\": \"none\"}}"
        )

        text = self._call(screenshot, prompt)
        parsed = self._parse_json(text)

        if not parsed or parsed.get("title") == "none":
            return None

        x = int(parsed.get("x", 0))
        y = int(parsed.get("y", 0))
        title = str(parsed.get("title", ""))

        if x == 0 and y == 0:
            return None

        logger.info(f"  [claude-target] '{title[:40]}' at ({x}, {y})")
        return (x, y, title)

    def find_paginate_target(self, screenshot: Image.Image) -> tuple[int, int] | None:
        """Find the Next page button or next page number on a search results page.

        Returns (x, y) coordinates of the Next button, or None if not found.
        """
        prompt = (
            f"Look at this page ({screenshot.width}x{screenshot.height} pixels).\n\n"
            f"Find the NEXT PAGE button or the next page number link. "
            f"It is usually at the bottom of the page. Look for:\n"
            f"- Text that says 'Next' or '>' or '>>'\n"
            f"- Page numbers like 1, 2, 3 where the next number is clickable\n\n"
            f"Return the CENTER coordinates of the Next button.\n"
            f"Output ONLY: {{\"x\": N, \"y\": N}}\n"
            f"If no Next button exists: {{\"x\": 0, \"y\": 0}}"
        )

        text = self._call(screenshot, prompt)
        parsed = self._parse_json(text)

        x = int(parsed.get("x", 0))
        y = int(parsed.get("y", 0))

        if x == 0 and y == 0:
            return None

        logger.info(f"  [claude-paginate] Next button at ({x}, {y})")
        return (x, y)
