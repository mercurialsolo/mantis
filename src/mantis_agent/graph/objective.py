"""ObjectiveSpec — structured description of what the user wants.

Parses plain text objectives into a structured spec:
  domains, target entity, required filters, output schema, completion condition.

Separate from MicroPlan (execution artifact) — ObjectiveSpec is semantic.
It feeds into graph generation which then compiles to MicroPlan.

Usage:
    spec = ObjectiveSpec.parse("Search BoatTrader for private seller boats...")
    print(spec.target_entity)  # "private seller boat listing"
    print(spec.required_filters)  # ["Private Seller", "zip 33101"]
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class OutputField:
    """A field in the required output schema."""

    name: str  # "year", "make", "phone"
    type: str = "str"  # "str", "int", "bool"
    required: bool = True
    example: str = ""  # "2018", "Sea Ray", "(305) 555-1234"


@dataclass
class CompletionCondition:
    """When the objective is considered complete."""

    type: str = "page_exhaustion"  # "page_exhaustion", "count", "all_pages"
    max_items: int = 0  # 0 = unlimited
    max_pages: int = 0  # 0 = unlimited


PARSE_OBJECTIVE_PROMPT = """\
You are parsing a CUA (Computer Use Agent) objective into a structured specification.

OBJECTIVE TEXT:
{text}

Extract the following fields. Output ONLY valid JSON:
{{
  "domains": ["list of target website domains, e.g. boattrader.com"],
  "start_url": "the URL to start from, or empty string if not specified",
  "target_entity": "what kind of item to find, e.g. private seller boat listing",
  "required_filters": ["list of filters to apply, e.g. Private Seller, zip 33101"],
  "forbidden_actions": ["actions to avoid, e.g. Contact Seller button, Request Info form"],
  "allowed_reveal_actions": ["safe actions to reveal hidden info, e.g. Show more, Show phone"],
  "output_fields": [
    {{"name": "field_name", "type": "str", "required": true, "example": "example value"}}
  ],
  "completion_type": "page_exhaustion or count or all_pages",
  "max_items": 0,
  "max_pages": 0
}}

Rules:
- Extract domains from any URLs mentioned in the text
- If no URL is given, infer start_url from the domain and filters
- output_fields should match what the user wants extracted (year, make, model, price, phone, etc.)
- If the text mentions "all listings" or "every page", use completion_type="all_pages"
- If the text mentions a count limit, use completion_type="count" with max_items
- Default to completion_type="page_exhaustion" (exhaust visible items, then paginate)
- forbidden_actions: things like "Contact Seller", "Request Info", "Apply for Loan"
- allowed_reveal_actions: things like "Show more", "Read more", "Show phone", "Call"
"""


@dataclass
class ObjectiveSpec:
    """Structured description of what the user wants to accomplish."""

    raw_text: str
    domains: list[str] = field(default_factory=list)
    start_url: str = ""
    target_entity: str = ""
    required_filters: list[str] = field(default_factory=list)
    forbidden_actions: list[str] = field(default_factory=list)
    allowed_reveal_actions: list[str] = field(default_factory=list)
    output_schema: list[OutputField] = field(default_factory=list)
    completion: CompletionCondition = field(default_factory=CompletionCondition)
    objective_hash: str = ""

    def __post_init__(self):
        if not self.objective_hash:
            self.objective_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        normalized = re.sub(r"\s+", " ", self.raw_text.strip().lower())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def cache_key(self) -> str:
        """Domain + objective hash for graph cache lookup."""
        domain = self.domains[0] if self.domains else "unknown"
        return f"{domain}_{self.objective_hash[:12]}"

    @classmethod
    def parse(cls, text: str, api_key: str = "") -> ObjectiveSpec:
        """Parse plain text objective into structured spec using Claude Sonnet."""
        import requests

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            logger.warning("ObjectiveSpec.parse: no API key, falling back to heuristic")
            return cls._parse_heuristic(text)

        prompt = PARSE_OBJECTIVE_PROMPT.format(text=text)
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )
            if resp.status_code != 200:
                logger.warning("ObjectiveSpec.parse API error: %s", resp.status_code)
                return cls._parse_heuristic(text)

            response_text = ""
            for block in resp.json().get("content", []):
                if block.get("type") == "text":
                    response_text = block["text"].strip()
                    break

            return cls._from_claude_response(text, response_text)
        except Exception as e:
            logger.warning("ObjectiveSpec.parse failed: %s", e)
            return cls._parse_heuristic(text)

    @classmethod
    def _from_claude_response(cls, raw_text: str, response: str) -> ObjectiveSpec:
        """Parse Claude's JSON response into ObjectiveSpec."""
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1]
        if response.endswith("```"):
            response = response.rsplit("```", 1)[0]
        response = response.strip()

        data = json.loads(response)
        output_fields = [
            OutputField(
                name=f.get("name", ""),
                type=f.get("type", "str"),
                required=f.get("required", True),
                example=f.get("example", ""),
            )
            for f in data.get("output_fields", [])
        ]
        completion = CompletionCondition(
            type=data.get("completion_type", "page_exhaustion"),
            max_items=int(data.get("max_items", 0)),
            max_pages=int(data.get("max_pages", 0)),
        )
        return cls(
            raw_text=raw_text,
            domains=data.get("domains", []),
            start_url=data.get("start_url", ""),
            target_entity=data.get("target_entity", ""),
            required_filters=data.get("required_filters", []),
            forbidden_actions=data.get("forbidden_actions", []),
            allowed_reveal_actions=data.get("allowed_reveal_actions", []),
            output_schema=output_fields,
            completion=completion,
        )

    @classmethod
    def _parse_heuristic(cls, text: str) -> ObjectiveSpec:
        """Fallback parser — extract what we can without API."""
        domains: list[str] = []
        start_url = ""
        # Extract full URLs with paths first
        for url_match in re.finditer(r"https?://[^\s,)]+", text):
            url = url_match.group(0).rstrip("/.,;:)")
            if not start_url:
                start_url = url
            # Extract domain from URL
            domain_match = re.search(r"(?:www\.)?([\w\-]+\.[\w]+)", url)
            if domain_match:
                domain = domain_match.group(1)
                if domain not in domains:
                    domains.append(domain)
        # Also extract bare domain mentions
        for match in re.finditer(
            r"(?:www\.)?([\w\-]+\.(?:com|org|net|io|co))\b", text
        ):
            domain = match.group(1)
            if domain not in domains:
                domains.append(domain)

        filters: list[str] = []
        for keyword in [
            "private seller",
            "by owner",
            "for sale by owner",
            "zip",
            "price",
            "location",
            "city",
            "state",
        ]:
            if keyword in text.lower():
                filters.append(keyword)

        return cls(
            raw_text=text,
            domains=domains,
            start_url=start_url,
            target_entity="listing",
            required_filters=filters,
        )

    def to_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "domains": self.domains,
            "start_url": self.start_url,
            "target_entity": self.target_entity,
            "required_filters": self.required_filters,
            "forbidden_actions": self.forbidden_actions,
            "allowed_reveal_actions": self.allowed_reveal_actions,
            "output_schema": [
                {"name": f.name, "type": f.type, "required": f.required, "example": f.example}
                for f in self.output_schema
            ],
            "completion": {
                "type": self.completion.type,
                "max_items": self.completion.max_items,
                "max_pages": self.completion.max_pages,
            },
            "objective_hash": self.objective_hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ObjectiveSpec:
        output_schema = [
            OutputField(**f) for f in data.get("output_schema", [])
        ]
        completion_data = data.get("completion", {})
        completion = CompletionCondition(
            type=completion_data.get("type", "page_exhaustion"),
            max_items=completion_data.get("max_items", 0),
            max_pages=completion_data.get("max_pages", 0),
        )
        return cls(
            raw_text=data.get("raw_text", ""),
            domains=data.get("domains", []),
            start_url=data.get("start_url", ""),
            target_entity=data.get("target_entity", ""),
            required_filters=data.get("required_filters", []),
            forbidden_actions=data.get("forbidden_actions", []),
            allowed_reveal_actions=data.get("allowed_reveal_actions", []),
            output_schema=output_schema,
            completion=completion,
            objective_hash=data.get("objective_hash", ""),
        )
