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
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)


# ── ExtractionSchema — domain-agnostic extraction configuration ───


@dataclass
class ExtractionSchema:
    """Describes what to extract, how to detect spam, and what viability means.

    When passed to ClaudeExtractor, overrides the hardcoded BoatTrader prompts
    with dynamic prompts generated from these fields.

    Use ExtractionSchema.from_objective(spec) to build from an ObjectiveSpec,
    or ExtractionSchema.default_boattrader() for backward compatibility.
    """

    entity_name: str = "listing"  # "boat listing", "job posting", "property"
    fields: list[dict[str, Any]] = field(default_factory=list)  # OutputField-like dicts
    required_fields: list[str] = field(default_factory=list)  # field names for viability
    spam_indicators: list[str] = field(default_factory=list)  # replaces DEALER_TEXT_INDICATORS
    spam_seller_indicators: list[str] = field(default_factory=list)  # replaces DEALER_SELLER_INDICATORS
    spam_label: str = "dealer/spam"  # what to call spam (e.g. "dealer", "recruiter")
    forbidden_controls: list[str] = field(default_factory=list)  # "Contact Seller", etc.
    allowed_controls: list[str] = field(default_factory=list)  # "Show more", "Show phone"

    @classmethod
    def from_objective(cls, objective: Any) -> ExtractionSchema:
        """Build from an ObjectiveSpec.

        All domain-specific signals (spam indicators, allowed reveal
        controls, forbidden lead-form labels) come from the objective. No
        hardcoded application-specific defaults are injected — callers that
        want them must specify them on the ObjectiveSpec.
        """
        fields = [
            {"name": f.name, "type": f.type, "required": f.required, "example": f.example}
            for f in getattr(objective, "output_schema", [])
        ]
        required = [f["name"] for f in fields if f.get("required", True)]
        forbidden = list(getattr(objective, "forbidden_actions", []))
        allowed = list(getattr(objective, "allowed_reveal_actions", []))
        spam_text = list(getattr(objective, "spam_text_indicators", []))
        spam_seller = list(getattr(objective, "spam_seller_indicators", []))
        spam_label = str(getattr(objective, "spam_label", "") or "non-organic")

        return cls(
            entity_name=getattr(objective, "target_entity", "item") or "item",
            fields=fields or cls._default_fields(),
            required_fields=required or ["url"],
            spam_indicators=spam_text,
            spam_seller_indicators=spam_seller,
            spam_label=spam_label,
            forbidden_controls=forbidden,
            allowed_controls=allowed,
        )

    @classmethod
    def default_boattrader(cls) -> ExtractionSchema:
        """The current hardcoded BoatTrader schema for backward compatibility."""
        return cls(
            entity_name="boat listing",
            fields=cls._boattrader_fields(),
            required_fields=["year", "make"],
            spam_indicators=list(DEALER_TEXT_INDICATORS),
            spam_seller_indicators=list(DEALER_SELLER_INDICATORS),
            spam_label="dealer",
            forbidden_controls=[
                "Contact Seller", "Request Info", "Email Seller",
                "Get Pre-Qualified", "loan", "financing",
            ],
            allowed_controls=[
                "Show more", "Read more", "See more", "Show phone",
                "View phone", "Call",
            ],
        )

    @staticmethod
    def _boattrader_fields() -> list[dict[str, Any]]:
        return [
            {"name": "year", "type": "str", "required": True, "example": "2018"},
            {"name": "make", "type": "str", "required": True, "example": "Sea Ray"},
            {"name": "model", "type": "str", "required": False, "example": "240 Sundeck"},
            {"name": "price", "type": "str", "required": False, "example": "$42,500"},
            {"name": "phone", "type": "str", "required": False, "example": "305-555-1234"},
            {"name": "url", "type": "str", "required": False, "example": "boattrader.com/boat/..."},
            {"name": "seller", "type": "str", "required": False, "example": "John Smith"},
        ]

    @staticmethod
    def _default_fields() -> list[dict[str, Any]]:
        return [
            {"name": "url", "type": "str", "required": True, "example": ""},
            {"name": "title", "type": "str", "required": False, "example": ""},
            {"name": "price", "type": "str", "required": False, "example": ""},
            {"name": "phone", "type": "str", "required": False, "example": ""},
            {"name": "seller", "type": "str", "required": False, "example": ""},
        ]

    def field_names(self) -> list[str]:
        return [f["name"] for f in self.fields]

    def json_template(self) -> str:
        """JSON template string for the extraction prompt."""
        obj = {}
        for f in self.fields:
            obj[f["name"]] = ""
        obj["is_spam"] = False
        return json.dumps(obj)

    def field_descriptions(self) -> str:
        """Numbered field list for extraction prompts."""
        lines = []
        for i, f in enumerate(self.fields, 1):
            example = f" (e.g. {f['example']})" if f.get("example") else ""
            required = " [REQUIRED]" if f.get("required") else ""
            lines.append(f"{i}. {f['name']}: {f.get('type', 'str')}{example}{required}")
        lines.append(f"{len(self.fields) + 1}. is_spam: Is this a {self.spam_label} listing? (true/false)")
        return "\n".join(lines)

    def contains_spam_text(self, text: str) -> bool:
        text_lower = text.lower()
        return any(ind in text_lower for ind in self.spam_indicators)

    def seller_looks_like_spam(self, seller: str) -> bool:
        seller_lower = seller.lower()
        return any(ind in seller_lower for ind in self.spam_seller_indicators)


# Spam indicator constants. Kept for ExtractionSchema.default_boattrader()
# which is now the only opt-in pathway that wires them in. Generic callers
# get an empty spam list and rely on their own ExtractionSchema to inject
# domain-specific indicators (recruiter spam for jobs, broker spam for
# real estate, etc.). Nothing here is referenced as an implicit default
# anywhere else in the file.
_LEGACY_BOATTRADER_TEXT_INDICATORS = (
    "dealername-",
    "dealer website",
    "view dealer website",
    "more from this dealer",
    "request a price",
    "condition-new",
    "certified dealer",
    "sponsored",
    "advertisement",
    "boatsgroup",
    "marinemax",
)

_LEGACY_BOATTRADER_SELLER_INDICATORS = (
    "marine",
    "marinemax",
    "yacht",
    "boats",
    "brokerage",
    "sales",
    "dealer",
    "center",
    "inc",
    "llc",
)

# Deprecated public aliases — read by ExtractionSchema.default_boattrader()
# only. New callers should populate ExtractionSchema fields explicitly.
DEALER_TEXT_INDICATORS = _LEGACY_BOATTRADER_TEXT_INDICATORS
DEALER_SELLER_INDICATORS = _LEGACY_BOATTRADER_SELLER_INDICATORS


def _parse_bool(value: object) -> bool:
    """Parse API booleans that may arrive as strings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def _contains_dealer_text(text: str) -> bool:
    text_lower = text.lower()
    return any(indicator in text_lower for indicator in DEALER_TEXT_INDICATORS)


def _seller_looks_like_dealer(seller: str) -> bool:
    seller_lower = seller.lower()
    return any(indicator in seller_lower for indicator in DEALER_SELLER_INDICATORS)


@dataclass
class ExtractionResult:
    """Structured data extracted from a listing screenshot.

    Named fields (year, make, model, etc.) are kept for backward compatibility.
    When an ExtractionSchema is set, the generic ``fields`` dict is the primary
    data store and all viability/spam checks use the schema configuration.
    """

    # Existing named fields (backward compat with BoatTrader)
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

    # Generic field storage — populated when schema is set
    extracted_fields: dict[str, str] = field(default_factory=dict)
    _schema: ExtractionSchema | None = field(default=None, repr=False)

    def dealer_reason(self) -> str:
        """Return a reason if this looks like spam/dealer inventory."""
        if self._schema:
            if self.is_dealer:
                return f"extractor marked as {self._schema.spam_label}"
            if self._schema.seller_looks_like_spam(self.seller or self.extracted_fields.get("seller", "")):
                seller = self.seller or self.extracted_fields.get("seller", "")
                return f"seller looks like {self._schema.spam_label}: {seller}"
            text = f"{self.url} {self.raw_response} " + " ".join(self.extracted_fields.values())
            if self._schema.contains_spam_text(text):
                return f"{self._schema.spam_label} indicator in listing text"
            return ""
        # Legacy BoatTrader path
        if self.is_dealer:
            return "extractor marked listing as dealer"
        if _seller_looks_like_dealer(self.seller):
            return f"seller looks like dealer: {self.seller}"
        if _contains_dealer_text(f"{self.url} {self.price} {self.raw_response}"):
            return "dealer/sponsored indicator in listing text"
        return ""

    def is_private_seller(self) -> bool:
        """Not spam/dealer."""
        return not self.dealer_reason()

    def has_phone(self) -> bool:
        """Require an actually visible phone number."""
        phone_val = self.phone or self.extracted_fields.get("phone", "")
        phone = phone_val.strip().lower()
        if phone in {"", "none", "n/a", "na", "unknown", "not visible", "not shown"}:
            return False
        digits = re.sub(r"\D", "", phone)
        return len(digits) >= 10

    def missing_required_reason(self) -> str:
        """Return why this extraction is not a usable lead."""
        if self._schema:
            missing = [
                name for name in self._schema.required_fields
                if not self.extracted_fields.get(name)
            ]
            return f"missing required field(s): {', '.join(missing)}" if missing else ""
        # Legacy
        missing = []
        if not self.year:
            missing.append("year")
        if not self.make:
            missing.append("make")
        return f"missing required field(s): {', '.join(missing)}" if missing else ""

    def to_summary(self) -> str:
        """Format as the VIABLE summary string."""
        if self._schema and self.extracted_fields:
            parts = []
            for f in self._schema.fields:
                name = f["name"]
                val = self.extracted_fields.get(name, "")
                if val:
                    parts.append(f"{name.replace('_', ' ').title()}: {val}")
                elif name == "phone":
                    parts.append("Phone: none")
            return "VIABLE | " + " | ".join(parts) if parts else ""
        # Legacy BoatTrader
        parts = []
        if self.year:
            parts.append(f"Year: {self.year}")
        if self.make:
            parts.append(f"Make: {self.make}")
        if self.model:
            parts.append(f"Model: {self.model}")
        if self.price:
            parts.append(f"Price: {self.price}")
        if self.phone:
            parts.append(f"Phone: {self.phone}")
        else:
            parts.append("Phone: none")
        if self.url:
            parts.append(f"URL: {self.url}")
        if self.seller:
            parts.append(f"Seller: {self.seller}")
        return "VIABLE | " + " | ".join(parts) if parts else ""

    def is_viable(self) -> bool:
        """Has enough data to be a useful lead (not spam, required fields present)."""
        if self._schema:
            has_required = all(
                self.extracted_fields.get(name)
                for name in self._schema.required_fields
            )
            return has_required and self.is_private_seller()
        # Legacy
        return bool(self.year and self.make and self.is_private_seller())


# Generic fallback prompts used when ClaudeExtractor is constructed without
# a schema. They describe the extractor's job in entity-neutral language and
# rely on the caller's plan/intent to provide context. Application-specific
# behaviour (boat listings, job postings, real-estate) MUST come through an
# explicit ExtractionSchema — the prompts below contain no hardcoded labels,
# field names, or industry verbs.

EXTRACT_PROMPT = """\
Look at this screenshot of a detail page.

Extract the structured data the page exposes. Read the browser URL bar,
the page heading, and the most prominent labelled fields. Where you see
clear key-value pairs (label : value, label \u2014 value, or stacked
label/value rows), return them.

Output ONLY valid JSON. The shape is open \u2014 use the field names
you see on the page. Always include "url" with the address-bar value:

{"url": "", "extracted": {"<field-name-as-shown>": "<value>", ...}}
"""

EXTRACT_SCROLLED_PROMPT = """\
Look at this screenshot. You have scrolled down on a detail page.

Read any newly-visible labelled fields, free-text description content,
and contact information. Don't repeat what was clearly visible at the
top of the page \u2014 focus on what's revealed by the scroll.

Output ONLY valid JSON:
{"extracted": {"<field-name>": "<value>", ...}, "additional_info": ""}
"""

EXTRACT_MULTI_SCREENSHOT_PROMPT = """\
You are looking at multiple screenshots from the SAME detail page,
captured at different scroll positions.

Extract every labelled field visible across all screenshots. Combine
them into one record. Always include "url" with the address-bar value
from whichever screenshot shows it.

Output ONLY valid JSON:
{"url": "", "extracted": {"<field-name>": "<value>", ...}}
"""

FIND_LISTING_CONTENT_CONTROL_PROMPT = """\
Look at this detail-page screenshot.

Find ONE visible control that should be clicked to reveal more
content the page is hiding behind a collapse/expand toggle. Typical
targets are "Show more", "Read more", "See more", "Expand", or any
visible chevron next to a collapsed section.

Avoid controls that submit forms, send messages, navigate away, or
trigger modals \u2014 only pick a control that expands content in place.

Return the center of the best target. Output ONLY valid JSON:
{"x": N, "y": N, "action": "expand|none", "label": "visible text", "reason": "brief reason"}

If no expand control is visible, output:
{"x": 0, "y": 0, "action": "none", "label": "", "reason": "none visible"}
"""


class ClaudeExtractor:
    """Extract structured data from listing screenshots using Claude Sonnet.

    Same API pattern as ClaudeGrounding — cheap, fast, vision-capable.
    Called 1-2 times per listing (top screenshot + scrolled screenshot).

    When ``schema`` is provided, prompts are generated dynamically from the
    schema fields instead of using the hardcoded BoatTrader constants.
    Callers that construct ``ClaudeExtractor()`` with no args get the existing
    BoatTrader behavior unchanged.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-sonnet-4-20250514",
        schema: ExtractionSchema | None = None,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.schema = schema
        self.debug_dir = os.environ.get("MANTIS_DEBUG_DIR", "/data/screenshots/claude_debug")

    # ── Dynamic prompt generation from schema ─────────────────────

    def _get_extract_prompt(self) -> str:
        """Return extraction prompt — dynamic from schema or legacy hardcoded."""
        if not self.schema:
            return EXTRACT_PROMPT
        s = self.schema
        return (
            f"Look at this screenshot of a {s.entity_name} page.\n\n"
            f"Extract ALL of the following data visible on the page:\n\n"
            f"{s.field_descriptions()}\n\n"
            f"For phone numbers: look in description, contact, or seller sections.\n"
            f"If no phone is visible, return phone as \"\".\n\n"
            f"Output ONLY valid JSON:\n{s.json_template()}"
        )

    def _get_multi_extract_prompt(self) -> str:
        """Return multi-screenshot extraction prompt."""
        if not self.schema:
            return EXTRACT_MULTI_SCREENSHOT_PROMPT
        s = self.schema
        spam_rules = ", ".join(f'"{ind}"' for ind in s.spam_indicators[:6])
        return (
            f"You are looking at multiple screenshots from the SAME {s.entity_name} page.\n"
            f"They were captured at different scroll positions.\n\n"
            f"Extract ALL fields visible across ALL screenshots:\n\n"
            f"{s.field_descriptions()}\n\n"
            f"Phone search priority:\n"
            f"- Description text, contact sections, detail areas\n"
            f"- Phone reveal buttons if visible\n"
            f"- International numbers are valid\n\n"
            f"{s.spam_label.title()} detection:\n"
            f"- is_spam=true for {s.spam_label} listings containing: {spam_rules}\n\n"
            f"If no phone is visible, use \"\".\n\n"
            f"Output ONLY valid JSON:\n{s.json_template()}"
        )

    def _get_find_listings_prompt(self, skip_titles: list[str] | None = None) -> str:
        """Return find-all-listings prompt."""
        if not self.schema:
            return ""  # Legacy path uses hardcoded prompt inline
        s = self.schema
        skip = ""
        if skip_titles:
            skip = "\n\nSKIP these already-processed items:\n" + "\n".join(f"- {t}" for t in skip_titles[:20])
        return (
            f"Look at this screenshot of a search results page.\n\n"
            f"Find ALL visible {s.entity_name} cards/items on this page.\n"
            f"For each, report the center coordinates and title text.\n\n"
            f"SKIP: sponsored, advertisement, {s.spam_label} inventory.\n"
            f"ONLY include organic {s.entity_name} results."
            f"{skip}\n\n"
            f"Output ONLY valid JSON:\n"
            f"{{\"listings\": [[x, y, \"title text\"], ...], \"pagination_y\": null_or_number}}"
        )

    def _get_content_control_prompt(self) -> str:
        """Return find-content-control prompt."""
        if not self.schema:
            return FIND_LISTING_CONTENT_CONTROL_PROMPT
        s = self.schema
        allowed = ", ".join(f'"{c}"' for c in s.allowed_controls)
        forbidden = ", ".join(f'"{c}"' for c in s.forbidden_controls)
        return (
            f"Look at this {s.entity_name} page screenshot.\n\n"
            f"Find ONE visible control that should be clicked to reveal more\n"
            f"seller-supplied text or contact information.\n\n"
            f"Prefer these safe targets:\n- {allowed}\n\n"
            f"Do NOT choose: {forbidden}\n\n"
            f"Return the center of the best target. Output ONLY valid JSON:\n"
            f"{{\"x\": N, \"y\": N, \"action\": \"expand_description|show_phone|none\", "
            f"\"label\": \"visible text\", \"reason\": \"brief reason\"}}\n\n"
            f"If no safe control is visible, output:\n"
            f"{{\"x\": 0, \"y\": 0, \"action\": \"none\", \"label\": \"\", \"reason\": \"none visible\"}}"
        )

    def _parse_schema_result(self, data: dict[str, Any]) -> ExtractionResult:
        """Parse Claude response dict into ExtractionResult with schema fields."""
        # Populate both named fields (backward compat) and generic dict
        result = ExtractionResult(
            year=str(data.get("year", "")),
            make=str(data.get("make", "")),
            model=str(data.get("model", "")),
            price=str(data.get("price", "")),
            phone=str(data.get("phone", "")),
            url=str(data.get("url", "")),
            seller=str(data.get("seller", "")),
            is_dealer=_parse_bool(data.get("is_dealer") or data.get("is_spam", False)),
            raw_response=json.dumps(data),
            _schema=self.schema,
        )
        # Fill generic fields from schema
        if self.schema:
            for f in self.schema.fields:
                name = f["name"]
                result.extracted_fields[name] = str(data.get(name, ""))
        return result

    def _debug_path(self, stem: str, suffix: str) -> str:
        """Build a writable debug artifact path."""
        candidates = [self.debug_dir, "/tmp/mantis_debug"]
        for base_dir in candidates:
            try:
                os.makedirs(base_dir, exist_ok=True)
                return os.path.join(base_dir, f"{stem}_{int(time.time())}{suffix}")
            except OSError:
                continue
        return os.path.join("/tmp", f"{stem}_{int(time.time())}{suffix}")

    def _call(self, screenshot: Image.Image, prompt: str, max_tokens: int = 200) -> str:
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
                    "max_tokens": max_tokens,
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
                logger.warning(
                    "ClaudeExtractor API error %s: %s",
                    resp.status_code,
                    resp.text[:500],
                )
                return ""
            for block in resp.json().get("content", []):
                if block.get("type") == "text":
                    return block["text"].strip()
        except Exception as e:
            logger.warning(f"ClaudeExtractor failed: {e}")
        return ""

    def _call_many(
        self,
        screenshots: list[Image.Image],
        prompt: str,
        labels: list[str] | None = None,
        max_tokens: int = 350,
    ) -> str:
        """Call Claude API with multiple screenshots and one prompt."""
        import requests

        if not self.api_key:
            logger.warning("ClaudeExtractor: no API key")
            return ""

        labels = labels or []
        content: list[dict] = [{"type": "text", "text": prompt}]
        for i, screenshot in enumerate(screenshots, 1):
            label = labels[i - 1] if i - 1 < len(labels) else f"screenshot {i}"
            buf = BytesIO()
            screenshot.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            content.append({"type": "text", "text": f"Screenshot {i}: {label}"})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": b64,
                },
            })

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
                    "messages": [{
                        "role": "user",
                        "content": content,
                    }],
                },
                timeout=30,
            )
            if resp.status_code != 200:
                logger.warning(
                    "ClaudeExtractor multi API error %s: %s",
                    resp.status_code,
                    resp.text[:500],
                )
                return ""
            for block in resp.json().get("content", []):
                if block.get("type") == "text":
                    return block["text"].strip()
        except Exception as e:
            logger.warning(f"ClaudeExtractor multi failed: {e}")
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

        When schema is set, uses dynamic prompts and populates generic fields.
        Otherwise uses legacy BoatTrader hardcoded prompt.
        """
        prompt = self._get_extract_prompt()
        text = self._call(screenshot, prompt)
        parsed = self._parse_json(text)

        if not parsed:
            return ExtractionResult(raw_response=text, confidence=0.1, _schema=self.schema)

        if self.schema:
            result = self._parse_schema_result(parsed)
            result.confidence = 0.9
            return result

        return ExtractionResult(
            year=str(parsed.get("year", "")),
            make=str(parsed.get("make", "")),
            model=str(parsed.get("model", "")),
            price=str(parsed.get("price", "")),
            phone=str(parsed.get("phone", "")),
            url=str(parsed.get("url", "")),
            seller=str(parsed.get("seller", "")),
            is_dealer=_parse_bool(parsed.get("is_dealer", False)),
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

    def extract_multi(
        self,
        screenshots: list[Image.Image],
        labels: list[str] | None = None,
    ) -> ExtractionResult:
        """Extract listing data from multiple screenshots of one detail page."""
        if not screenshots:
            return ExtractionResult(confidence=0.0, _schema=self.schema)

        prompt = self._get_multi_extract_prompt()
        text = self._call_many(
            screenshots,
            prompt,
            labels=labels,
            max_tokens=450,
        )
        parsed = self._parse_json(text)

        if not parsed:
            return ExtractionResult(raw_response=text, confidence=0.1, _schema=self.schema)

        if self.schema:
            result = self._parse_schema_result(parsed)
            result.confidence = 0.9
            return result

        return ExtractionResult(
            year=str(parsed.get("year", "")),
            make=str(parsed.get("make", "")),
            model=str(parsed.get("model", "")),
            price=str(parsed.get("price", "")),
            phone=str(parsed.get("phone", "")),
            url=str(parsed.get("url", "")),
            seller=str(parsed.get("seller", "")),
            is_dealer=_parse_bool(parsed.get("is_dealer", False)),
            raw_response=text,
            confidence=0.9,
        )

    def find_listing_content_control(self, screenshot: Image.Image) -> dict | None:
        """Find a safe expand or phone reveal control on a listing page.

        Returns a dict with x/y/action/label/reason, or None if no safe target
        is visible. This intentionally avoids generic Contact Seller forms.
        """
        debug_stem = "claude_listing_content_control"
        try:
            screenshot.save(self._debug_path(debug_stem, ".png"))
        except Exception:
            pass
        prompt = self._get_content_control_prompt()
        try:
            with open(self._debug_path(debug_stem, "_prompt.txt"), "w") as f:
                f.write(prompt)
        except Exception:
            pass

        text = self._call(screenshot, prompt)

        try:
            with open(self._debug_path(debug_stem, "_response.txt"), "w") as f:
                f.write(text)
        except Exception:
            pass

        parsed = self._parse_json(text)
        if not parsed:
            logger.warning(f"  [content-control] parse failed: {text[:200]}")
            return None

        action = str(parsed.get("action", "none"))
        x = int(parsed.get("x", 0))
        y = int(parsed.get("y", 0))
        if action == "none" or x == 0 or y == 0:
            return None

        label = str(parsed.get("label", ""))
        reason = str(parsed.get("reason", ""))
        forbidden = (
            "contact seller", "request info", "email seller",
            "get pre-qualified", "pre-qualified", "loan", "financing",
        )
        target_text = f"{label} {reason}".lower()
        if any(term in target_text for term in forbidden):
            logger.info(f"  [content-control] rejected unsafe target: {label[:60]}")
            return None

        return {
            "x": x,
            "y": y,
            "action": action,
            "label": label,
            "reason": reason,
        }

    def verify_gate(self, screenshot: Image.Image, expected: str) -> tuple[bool, str]:
        """Verify a gate condition from a screenshot.

        Used after setup to check if filters were actually applied.
        Returns (passed, reason).
        """
        prompt = (
            f"Look at this screenshot ({screenshot.width}x{screenshot.height} pixels).\n\n"
            f"Check this condition: {expected}\n\n"
            f"Look at the page heading, URL bar, result count, and any active filter tags.\n\n"
            f"Output ONLY valid JSON:\n"
            f"{{\"passed\": true/false, \"reason\": \"what you see that confirms or denies the condition\"}}"
        )

        text = self._call(screenshot, prompt)
        parsed = self._parse_json(text)

        if not parsed:
            return False, f"Could not parse verifier response: {text[:100]}"

        passed = bool(parsed.get("passed", False))
        reason = str(parsed.get("reason", ""))
        logger.info(f"  [gate] {'PASS' if passed else 'FAIL'}: {reason[:80]}")
        return passed, reason

    def find_click_target(
        self,
        screenshot: Image.Image,
        skip_count: int = 0,
        skip_urls: list[str] | None = None,
    ) -> tuple[int, int, str] | tuple[str] | None:
        """Find the next listing to click on a search results page.

        Args:
            screenshot: Current page screenshot.
            skip_count: Unused (kept for compatibility).
            skip_urls: Human-readable titles/slugs of already-extracted listings to skip.

        Returns:
            (x, y, title) — target found
            ("not_found",) — Claude confirmed no more listings
            ("error",) — API/parse failure (should retry, not treat as exhausted)
            None — empty API response (should retry)
        """
        skip_section = ""
        if skip_urls:
            skip_section = (
                "\n\nSKIP these listings (already extracted): "
                + ", ".join(skip_urls[:6])
                + "\nFind a DIFFERENT listing that is NOT in the skip list."
            )

        entity = self.schema.entity_name if self.schema else "item"
        prompt = (
            f"Look at this search-results screenshot ({screenshot.width}x{screenshot.height} pixels).\n\n"
            f"The top of the screenshot may show page header, search controls, "
            f"and filters. Result entries may start only in the LOWER part of "
            f"the screenshot; the bottom-most entry may be partially visible.\n\n"
            f"Find the first unprocessed {entity} entry — could be a card, "
            f"table row, list item, or any repeated clickable UI element. "
            f"Ignore page headers, filters, sort controls, ads, and footer links."
            f"{skip_section}\n\n"
            f"Return the CENTER coordinates of the entry's primary clickable "
            f"area (title text or main link, NOT any image).\n"
            f"If the exact title is hard to read, return approximate coordinates "
            f"and use \"unknown\" for the title.\n\n"
            f"Output ONLY valid JSON: {{\"x\": N, \"y\": N, \"title\": \"the title text or unknown\"}}\n"
            f"If no {entity} entry is visible anywhere in the screenshot, output: {{\"x\": 0, \"y\": 0, \"title\": \"none\"}}"
        )

        debug_stem = f"claude_click_skip{skip_count}"

        # Save screenshot Claude will see (for debugging)
        try:
            screenshot.save(self._debug_path(debug_stem, ".png"))
        except Exception as e:
            logger.debug(f"[claude-target] failed to save screenshot: {e}")

        try:
            with open(self._debug_path(debug_stem, "_prompt.txt"), "w") as f:
                f.write(prompt)
        except Exception as e:
            logger.debug(f"[claude-target] failed to save prompt: {e}")

        text = self._call(screenshot, prompt)

        # Save Claude's response
        try:
            with open(self._debug_path(debug_stem, "_response.txt"), "w") as f:
                f.write(text)
        except Exception as e:
            logger.debug(f"[claude-target] failed to save response: {e}")

        parsed = self._parse_json(text)

        if not parsed:
            logger.warning(f"  [claude-target] parse failed raw={text[:300]!r}")
            return ("error",)  # Parse failure — retry, don't treat as exhausted

        if parsed.get("title") == "none":
            logger.info(f"  [claude-target] Claude reported no listing for skip={skip_count}")
            return ("not_found",)  # Genuine exhaustion

        x = int(parsed.get("x", 0))
        y = int(parsed.get("y", 0))
        title = str(parsed.get("title", ""))

        if x == 0 and y == 0:
            logger.warning(f"  [claude-target] zero coordinates raw={text[:300]!r}")
            return ("error",)  # Bad coordinates — retry

        logger.info(f"  [claude-target] '{title[:40]}' at ({x}, {y})")
        return (x, y, title)

    def find_all_listings(
        self,
        screenshot: Image.Image,
    ) -> list[tuple[int, int, str]] | tuple[str]:
        """Find ALL listing cards on the page in ONE Claude call.

        Returns:
            list[(x, y, title)] — visible listings found
            ("empty",) — screenshot looks like a normal page but no listings visible
            ("blocked",) — screenshot appears to be an error/block/rate-limit page
            ("error",) — API/parse failure

        Cost: ~$0.003-0.005 (one call regardless of how many cards).
        The caller clicks each sequentially without calling Claude again.
        """
        debug_stem = "claude_find_all"
        # All entity / spam / layout context comes from the schema. When no
        # schema is provided, the prompt stays entity-neutral — it asks for
        # "the items the user wants to click" without naming a domain.
        entity = self.schema.entity_name if self.schema else "item"
        spam_label = self.schema.spam_label if self.schema else "non-organic"
        spam_examples_clause = ""
        if self.schema and self.schema.spam_indicators:
            examples = ", ".join(f'"{s}"' for s in self.schema.spam_indicators[:6])
            spam_examples_clause = (
                f"\nSpam signals to filter (caller-provided): {examples}."
            )
        prompt = (
            f"Look at this screenshot ({screenshot.width}x{screenshot.height} pixels).\n\n"
            f"Find ALL eligible {entity} entries visible in this screenshot. "
            f"Entries may be cards-with-photos, table rows, list items, profile "
            f"rows, panel sections, or any repeated UI element on a results "
            f"page. Include the bottom-most entry even if partially visible.\n\n"
            f"STRICT FILTER: only return organic entries the user can click. "
            f"Skip {spam_label} content, sponsored ads, navigation tabs, "
            f"sort/filter controls, headers, footers, and pagination bars."
            f"{spam_examples_clause}\n\n"
            f"If the screenshot shows an error page, rate limit, bot check, "
            f"CAPTCHA, or sign-in wall, mark it as blocked.\n\n"
            f"Return the CENTER coordinates of each entry's primary clickable "
            f"area (the title text or the row's main link, not any image). "
            f"If a title is hard to read, use \"unknown\".\n\n"
            f"Also check: is there a pagination control (page numbers or a "
            f"Next button) visible at the bottom? Note its Y coordinate.\n\n"
            f"Output ONLY valid JSON:\n"
            f"{{\"status\": \"ok\", \"listings\": [{{\"x\": N, \"y\": N, \"title\": \"text or unknown\", "
            f"\"is_organic\": true/false, \"reason\": \"brief\"}}, ...], "
            f"\"pagination_y\": N_or_null}}\n"
            f"If this looks like a normal page but no entries are visible, output:\n"
            f"{{\"status\": \"empty\", \"listings\": []}}\n"
            f"If the screenshot appears blocked or errored, output:\n"
            f"{{\"status\": \"blocked\", \"listings\": []}}"
        )

        try:
            screenshot.save(self._debug_path(debug_stem, ".png"))
        except Exception as e:
            logger.debug(f"[find_all] failed to save screenshot: {e}")

        try:
            with open(self._debug_path(debug_stem, "_prompt.txt"), "w") as f:
                f.write(prompt)
        except Exception as e:
            logger.debug(f"[find_all] failed to save prompt: {e}")

        text = self._call(screenshot, prompt, max_tokens=1500)

        try:
            with open(self._debug_path(debug_stem, "_response.txt"), "w") as f:
                f.write(text)
        except Exception as e:
            logger.debug(f"[find_all] failed to save response: {e}")

        # Parse JSON payload
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        parsed: dict | list | None = None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            pass

        if parsed is None:
            import re as _re
            match = _re.search(r'\{.*\}', text, _re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    parsed = None
            else:
                match = _re.search(r'\[.*\]', text, _re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group())
                    except json.JSONDecodeError:
                        parsed = None

        if parsed is None:
            logger.warning(f"  [find_all] No parseable JSON in response: {text[:200]}")
            return ("error",)

        status = "ok"
        items: list = []
        if isinstance(parsed, dict):
            status = str(parsed.get("status", "ok")).lower()
            raw_items = parsed.get("listings", [])
            if isinstance(raw_items, list):
                items = raw_items
        elif isinstance(parsed, list):
            # Backward compatibility with the old prompt.
            items = parsed
        else:
            logger.warning(f"  [find_all] Unexpected payload type: {type(parsed).__name__}")
            return ("error",)

        if status == "blocked":
            logger.warning("  [find_all] Claude classified screenshot as blocked/error page")
            return ("blocked",)
        if status == "empty":
            logger.info("  [find_all] Claude reported no listings in this viewport")
            return ("empty",)

        results = []
        for item in items:
            x = int(item.get("x", 0))
            y = int(item.get("y", 0))
            title = str(item.get("title", "unknown"))
            seller = str(item.get("seller", ""))
            reason = str(item.get("reason", ""))
            is_private_seller = item.get("is_private_seller")
            is_dealer = _parse_bool(item.get("is_dealer", False))
            is_sponsored = _parse_bool(item.get("is_sponsored", False))
            card_text = f"{title} {seller} {reason}"
            if (
                is_private_seller is False
                or is_dealer
                or is_sponsored
                or _contains_dealer_text(card_text)
                or _seller_looks_like_dealer(seller)
            ):
                logger.info(
                    "  [find_all] Skipping non-private card: title='%s' seller='%s' reason='%s'",
                    title[:50],
                    seller[:50],
                    reason[:80],
                )
                continue
            if x > 0 and y > 0:
                results.append((x, y, title))

        logger.info(f"  [find_all] Found {len(results)} listings in one call")
        return results

    def find_paginate_target(self, screenshot: Image.Image) -> tuple[int, int, str] | tuple[str] | None:
        """Find the Next page button or next page number on a search results page.

        Returns:
            (x, y, label) — pagination target found
            ("not_found",) — Claude confirmed no next-page control is visible
            ("error",) — API/parse failure or unusable coordinates
            None — empty response
        """
        prompt = (
            f"Look at this results-page screenshot ({screenshot.width}x{screenshot.height} pixels).\n\n"
            f"You are near the bottom of the results. The pagination bar is usually BELOW the last result cards "
            f"and ABOVE the footer/social icons. Find the control that goes to the NEXT results page.\n\n"
            f"Valid targets:\n"
            f"- A link that says 'Next'\n"
            f"- A right-arrow or chevron like '>'\n"
            f"- The next page number to the right of the current page number\n\n"
            f"Do NOT return footer links, social icons, newsletter/signup buttons, or ad links.\n\n"
            f"Return the CENTER coordinates of the next-page control.\n"
            f"Output ONLY valid JSON: {{\"x\": N, \"y\": N, \"label\": \"Next or 2 or >\"}}\n"
            f"If no next-page control is visible anywhere in the screenshot: {{\"x\": 0, \"y\": 0, \"label\": \"none\"}}"
        )

        debug_stem = "claude_paginate"

        try:
            screenshot.save(self._debug_path(debug_stem, ".png"))
        except Exception as e:
            logger.debug(f"[claude-paginate] failed to save screenshot: {e}")

        try:
            with open(self._debug_path(debug_stem, "_prompt.txt"), "w") as f:
                f.write(prompt)
        except Exception as e:
            logger.debug(f"[claude-paginate] failed to save prompt: {e}")

        text = self._call(screenshot, prompt)

        try:
            with open(self._debug_path(debug_stem, "_response.txt"), "w") as f:
                f.write(text)
        except Exception as e:
            logger.debug(f"[claude-paginate] failed to save response: {e}")

        parsed = self._parse_json(text)

        if not parsed:
            logger.warning(f"  [claude-paginate] parse failed raw={text[:300]!r}")
            return ("error",)

        if parsed.get("label") == "none":
            logger.info("  [claude-paginate] Claude reported no next-page control visible")
            return ("not_found",)

        x = int(parsed.get("x", 0))
        y = int(parsed.get("y", 0))
        label = str(parsed.get("label", ""))

        if x == 0 and y == 0:
            logger.warning(f"  [claude-paginate] zero coordinates raw={text[:300]!r}")
            return ("error",)

        logger.info(f"  [claude-paginate] '{label[:20]}' at ({x}, {y})")
        return (x, y, label)

    def find_filter_target(
        self,
        screenshot: Image.Image,
        filter_intent: str,
    ) -> dict | None:
        """Find a filter element on the page and determine how to interact with it.

        Args:
            screenshot: Current page screenshot.
            filter_intent: Description like "Click Private Seller option in seller type filter"
                          or "Enter zip code 33101 in location search field".

        Returns:
            dict with keys: x, y, action ("click"|"type"|"select"), value (for type/select),
                           label (what was found)
            None on failure.
        """
        prompt = (
            f"Look at this screenshot ({screenshot.width}x{screenshot.height} pixels).\n\n"
            f"TASK: {filter_intent}\n\n"
            f"This is a search/results page. Look for the matching filter "
            f"control wherever it lives — left sidebar, right sidebar, top "
            f"filter bar, modal, or inline above the results.\n\n"
            f"Common control shapes:\n"
            f"- Checkboxes or radio buttons next to a text label\n"
            f"- Text input fields with a label nearby or placeholder text\n"
            f"- Dropdown menus (current selection visible with a chevron)\n"
            f"- Clickable text links / pills for filter options\n"
            f"- Sliders or range inputs\n\n"
            f"The exact label in the TASK may differ from what's on screen — "
            f"prefer semantic matches over exact strings (e.g. a 'Sort by' "
            f"control could be labelled 'Order by' on this site). Match by "
            f"the FUNCTION the TASK describes, not literal wording.\n\n"
            f"Determine the interaction:\n"
            f"- \"click\" for checkboxes, radio buttons, links, pills, toggle buttons\n"
            f"- \"type\" for text input fields (click field, clear, type value)\n"
            f"- \"select\" for dropdown menus (click to open, then pick option)\n\n"
            f"Return the CENTER coordinates of the interactive element.\n"
            f"Output ONLY valid JSON:\n"
            f"{{\"x\": N, \"y\": N, \"action\": \"click|type|select\", "
            f"\"value\": \"text to type or option to select or empty\", "
            f"\"label\": \"what element you found\"}}\n"
            f"If you cannot find any matching filter element anywhere on the page:\n"
            f"{{\"x\": 0, \"y\": 0, \"action\": \"not_found\", "
            f"\"value\": \"\", \"label\": \"describe what you see instead\"}}"
        )

        debug_stem = "claude_filter"
        try:
            screenshot.save(self._debug_path(debug_stem, ".png"))
        except Exception:
            pass
        try:
            with open(self._debug_path(debug_stem, "_prompt.txt"), "w") as f:
                f.write(prompt)
        except Exception:
            pass

        text = self._call(screenshot, prompt)

        try:
            with open(self._debug_path(debug_stem, "_response.txt"), "w") as f:
                f.write(text)
        except Exception:
            pass

        parsed = self._parse_json(text)
        if not parsed:
            logger.warning(f"  [claude-filter] parse failed: {text[:200]}")
            return None

        if parsed.get("action") == "not_found":
            label = parsed.get("label", "unknown")
            logger.info(f"  [claude-filter] Element not visible: {filter_intent[:50]}")
            logger.info(f"  [claude-filter] What Claude sees: {label[:100]}")
            print(f"  [claude-filter] NOT FOUND: {label[:100]}")
            return None

        x = int(parsed.get("x", 0))
        y = int(parsed.get("y", 0))
        if x == 0 and y == 0:
            logger.warning("  [claude-filter] zero coordinates")
            return None

        result = {
            "x": x,
            "y": y,
            "action": str(parsed.get("action", "click")),
            "value": str(parsed.get("value", "")),
            "label": str(parsed.get("label", "")),
        }
        logger.info(f"  [claude-filter] '{result['label'][:40]}' at ({x},{y}) action={result['action']}")
        return result

    def find_form_target(
        self,
        screenshot: Image.Image,
        intent: str,
        *,
        target_label: str = "",
        target_value: str = "",
        target_aliases: list[str] | None = None,
    ) -> dict | None:
        """Find a labelled form element (input / button / dropdown / option) on any page.

        The non-listings counterpart to ``find_filter_target``. Used by the
        runner's ``fill_field`` / ``submit`` / ``select_option`` step types
        for login forms, edit forms, settings panels — anywhere the page has
        named fields-and-buttons rather than a listings grid.

        Args:
            screenshot: Current page screenshot.
            intent: Free-text description: "Click the user ID input field and
                enter alice", "Click the Login button", "Click the
                Industry Vertical dropdown".
            target_label: Optional structured label from
                ``MicroIntent.params["label"]`` (preferred — more reliable
                than parsing free text).
            target_value: Optional value to type / option to select. The
                runner re-reads this from ``params`` for the actual typing,
                but providing it here helps Claude disambiguate.
            target_aliases: Optional alternate labels — e.g.
                ``["Update", "Save", "Save Changes"]`` for an "Update Lead"
                submit button. Claude treats any alias as an acceptable
                visual match, so a plan written for one product can survive
                a copy-tweak in another. Issue #89 §2.

        Returns:
            dict with keys: ``x``, ``y``, ``action`` ("click" | "type" |
            "select"), ``value`` (text to type / option to pick), ``label``
            (what was found). None on failure.
        """
        target_clause = (
            f"\nThe target element label/text is: \"{target_label}\""
            if target_label else ""
        )
        value_clause = (
            f"\nThe value to type or option to select is: \"{target_value}\""
            if target_value else ""
        )
        aliases = [a for a in (target_aliases or []) if a]
        alias_clause = (
            "\nAcceptable equivalent labels (any of these is a valid match): "
            + ", ".join(f'"{a}"' for a in aliases)
            if aliases else ""
        )
        prompt = (
            f"Look at this screenshot ({screenshot.width}x{screenshot.height} pixels).\n\n"
            f"TASK: {intent}"
            f"{target_clause}{value_clause}{alias_clause}\n\n"
            f"This page is NOT a listings/search-results grid. It is a form, "
            f"login/edit page, settings panel, dialog, or similar. Exactly ONE "
            f"element on screen matches the task — find it.\n\n"
            f"Match by SHAPE first, then by visible label:\n"
            f"- Text input: a labelled box you can type into (label is above, "
            f"  to the left, or rendered as placeholder text inside).\n"
            f"- Button: a clickable element with visible text — match the button\n"
            f"  whose text matches (or semantically matches) the task target_label.\n"
            f"- Dropdown / <select>: shows the current value with a chevron.\n"
            f"- Option inside an opened dropdown: an overlay menu above the page content.\n"
            f"- Checkbox / radio: a small toggle with a label adjacent.\n\n"
            f"The TASK target_label may not match the on-screen text exactly — "
            f"prefer semantic matches (case, punctuation, and small word "
            f"variations are fine).\n\n"
            f"Determine the interaction:\n"
            f"- \"click\" for buttons, links, checkbox/radio/toggle, or to OPEN a dropdown.\n"
            f"- \"type\" for text inputs (the runner will click the field, clear, then type).\n"
            f"- \"select\" for picking an option from a dropdown that is already open "
            f"(if the dropdown is closed, return action=click on the dropdown control first).\n\n"
            f"Return the CENTER coordinates of the target element. Output ONLY valid JSON:\n"
            f"{{\"x\": N, \"y\": N, \"action\": \"click|type|select\", "
            f"\"value\": \"text to type or option to select or empty\", "
            f"\"label\": \"what element you found\"}}\n"
            f"If the target is not visible anywhere on the page:\n"
            f"{{\"x\": 0, \"y\": 0, \"action\": \"not_found\", "
            f"\"value\": \"\", \"label\": \"describe what you see instead\"}}"
        )

        debug_stem = "claude_form"
        try:
            screenshot.save(self._debug_path(debug_stem, ".png"))
        except Exception:
            pass
        try:
            with open(self._debug_path(debug_stem, "_prompt.txt"), "w") as f:
                f.write(prompt)
        except Exception:
            pass

        text = self._call(screenshot, prompt)

        try:
            with open(self._debug_path(debug_stem, "_response.txt"), "w") as f:
                f.write(text)
        except Exception:
            pass

        parsed = self._parse_json(text)
        if not parsed:
            logger.warning(f"  [claude-form] parse failed: {text[:200]}")
            return None

        if parsed.get("action") == "not_found":
            label = parsed.get("label", "unknown")
            logger.info(f"  [claude-form] target not visible: {intent[:60]}")
            logger.info(f"  [claude-form] What Claude sees: {label[:120]}")
            return None

        x = int(parsed.get("x", 0))
        y = int(parsed.get("y", 0))
        if x == 0 and y == 0:
            logger.warning("  [claude-form] zero coordinates")
            return None

        result = {
            "x": x,
            "y": y,
            "action": str(parsed.get("action", "click")),
            "value": str(parsed.get("value", target_value or "")),
            "label": str(parsed.get("label", target_label or "")),
        }
        logger.info(f"  [claude-form] '{result['label'][:40]}' at ({x},{y}) action={result['action']}")
        return result
