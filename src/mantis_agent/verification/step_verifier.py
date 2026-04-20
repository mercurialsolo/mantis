"""Step-level verification for CUA actions.

Uses Claude Sonnet to compare before/after screenshots and verify
that a CUA action achieved its intent. Same API pattern as
ClaudeGrounding but for verification instead of click targeting.

Cost: ~$0.003 per verification call (1-2 screenshots + short prompt).
Only called for critical steps, not every mouse move.

Usage:
    verifier = StepVerifier()
    result = verifier.verify_step(before_img, after_img,
                                   intent="Click Private Seller filter",
                                   action="click(120, 450)")
    if not result.verified:
        print(f"Step failed: {result.issue} — {result.suggestion}")
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
class VerificationResult:
    """Result of verifying a CUA step."""
    verified: bool              # Did the step achieve its intent?
    page_changed: bool          # Did the screenshot visually change?
    issue: str = ""             # "" if OK, else: "off_site", "popup", "filter_lost", "no_change"
    suggestion: str = ""        # Recovery action if failed
    confidence: float = 0.5
    details: str = ""           # Raw verifier output for debugging


VERIFY_STEP_PROMPT = """\
You are verifying whether a CUA (Computer Use Agent) action succeeded.

INTENT: {intent}
ACTION TAKEN: {action}

I'm showing you TWO screenshots:
1. BEFORE the action was taken
2. AFTER the action was taken

Analyze what changed between the screenshots and determine:
1. Did the page visually change? (yes/no)
2. Did the action achieve the stated intent? (yes/no)
3. What issue occurred, if any? Choose one:
   - "none" — action succeeded
   - "no_change" — page didn't change (click missed or element unresponsive)
   - "popup" — a popup/modal appeared blocking the page
   - "off_site" — navigated to a different website
   - "filter_lost" — page changed but expected filter/state was lost
   - "wrong_page" — landed on unexpected page (404, about page, dealer site)
   - "gallery" — entered a photo gallery/lightbox
4. What recovery action should be taken? (or "none" if succeeded)

Output ONLY valid JSON:
{{"verified": true/false, "page_changed": true/false, "issue": "...", "suggestion": "..."}}
"""

VERIFY_FILTER_PROMPT = """\
Look at this screenshot ({width}x{height} pixels).

Check if these filters are currently ACTIVE on the page:
{filters}

Look for:
- Page heading or title mentioning the filter (e.g. "by owner", "private seller")
- Active filter pills/tags showing applied filters
- Result count (should be less than {max_results} if filtered)
- URL in address bar containing filter segments

Output ONLY valid JSON:
{{"verified": true/false, "filters_found": ["list of filters visible"], "result_count": "number or unknown", "issue": "none or description"}}
"""

VERIFY_PAGE_PROMPT = """\
Look at this screenshot ({width}x{height} pixels).

Check if the current page matches this expected state:
{expected}

Look for these signals: {signals}

Output ONLY valid JSON:
{{"verified": true/false, "page_matches": true/false, "issue": "none or description", "suggestion": "recovery action if needed"}}
"""


class StepVerifier:
    """Verifies CUA step outcomes using before/after screenshots.

    Uses Claude Sonnet API — same pattern as ClaudeGrounding.
    Cheap (~$0.003/call), fast, vision-capable.
    """

    def __init__(self, api_key: str = "", model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model

    def _call_claude(self, messages_content: list, max_tokens: int = 200) -> str:
        """Call Claude API with vision content. Returns response text."""
        import requests

        if not self.api_key:
            logger.warning("StepVerifier: no API key")
            return ""

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
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": messages_content}],
                },
                timeout=20,
            )
            if resp.status_code != 200:
                logger.warning(f"StepVerifier API error: {resp.status_code}")
                return ""
            for block in resp.json().get("content", []):
                if block.get("type") == "text":
                    return block["text"].strip()
        except Exception as e:
            logger.warning(f"StepVerifier call failed: {e}")
        return ""

    @staticmethod
    def _img_to_b64(img: Image.Image) -> str:
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Extract JSON from response text (handles markdown fences)."""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in text
            match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {}

    def verify_step(
        self,
        before: Image.Image,
        after: Image.Image,
        intent: str,
        action: str,
    ) -> VerificationResult:
        """Verify a step by comparing before/after screenshots.

        Args:
            before: Screenshot before the action.
            after: Screenshot after the action.
            intent: What the step was trying to accomplish.
            action: The action that was taken (e.g. "click(120, 450)").

        Returns:
            VerificationResult with verified status and any issues.
        """
        prompt = VERIFY_STEP_PROMPT.format(intent=intent, action=action)

        content = [
            {"type": "text", "text": "BEFORE screenshot:"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                          "data": self._img_to_b64(before)}},
            {"type": "text", "text": "AFTER screenshot:"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                          "data": self._img_to_b64(after)}},
            {"type": "text", "text": prompt},
        ]

        text = self._call_claude(content)
        parsed = self._parse_json(text)

        if not parsed:
            return VerificationResult(
                verified=True, page_changed=True,
                confidence=0.3, details="Could not parse verifier response",
            )

        return VerificationResult(
            verified=parsed.get("verified", True),
            page_changed=parsed.get("page_changed", True),
            issue=parsed.get("issue", ""),
            suggestion=parsed.get("suggestion", ""),
            confidence=0.85,
            details=text,
        )

    def verify_filter(
        self,
        screenshot: Image.Image,
        expected_filters: list[str],
        max_results: int = 50000,
    ) -> VerificationResult:
        """Verify that expected filters are visible on the page.

        Args:
            screenshot: Current page screenshot.
            expected_filters: List of filters that should be active
                (e.g. ["private seller", "zip 33101", "min price $35,000"]).
            max_results: Expected max result count if filters are applied.
        """
        prompt = VERIFY_FILTER_PROMPT.format(
            width=screenshot.width, height=screenshot.height,
            filters="\n".join(f"- {f}" for f in expected_filters),
            max_results=max_results,
        )

        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                          "data": self._img_to_b64(screenshot)}},
            {"type": "text", "text": prompt},
        ]

        text = self._call_claude(content)
        parsed = self._parse_json(text)

        if not parsed:
            return VerificationResult(
                verified=False, page_changed=True,
                issue="parse_error", confidence=0.3,
                details="Could not parse filter verification response",
            )

        verified = parsed.get("verified", False)
        filters_found = parsed.get("filters_found", [])
        issue = parsed.get("issue", "")

        return VerificationResult(
            verified=verified,
            page_changed=True,
            issue=issue if not verified else "",
            suggestion=f"Re-apply missing filters: {set(expected_filters) - set(filters_found)}" if not verified else "",
            confidence=0.85 if verified else 0.6,
            details=text,
        )

    def verify_on_correct_page(
        self,
        screenshot: Image.Image,
        expected_description: str,
        expected_signals: list[str],
    ) -> VerificationResult:
        """Check if current page matches expected state.

        Used before each iteration to ensure we haven't drifted
        to a dealer page, about page, or off-site.

        Args:
            screenshot: Current page screenshot.
            expected_description: What the page should look like.
            expected_signals: Text/elements that should be visible.
        """
        prompt = VERIFY_PAGE_PROMPT.format(
            width=screenshot.width, height=screenshot.height,
            expected=expected_description,
            signals=", ".join(expected_signals),
        )

        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                          "data": self._img_to_b64(screenshot)}},
            {"type": "text", "text": prompt},
        ]

        text = self._call_claude(content)
        parsed = self._parse_json(text)

        if not parsed:
            return VerificationResult(
                verified=True, page_changed=True,
                confidence=0.3, details="Could not parse page verification",
            )

        return VerificationResult(
            verified=parsed.get("verified", True),
            page_changed=True,
            issue=parsed.get("issue", ""),
            suggestion=parsed.get("suggestion", ""),
            confidence=0.8,
            details=text,
        )
