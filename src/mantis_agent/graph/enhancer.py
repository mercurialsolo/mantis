"""PlanEnhancer — fill gaps in vague plans using site probe results.

Takes an ObjectiveSpec + ProbeResult and produces concrete, executable
PhaseNodes. Resolves ambiguity:
  - "Apply filter: private seller" → detected as URL-based → "Navigate to .../by-owner/"
  - "Scroll to description" → probe found 3 viewports → "Scroll 3 times past gallery"
  - "Extract phone" → probe found collapsed sections → "Click 'Show more' then read"

The enhancer uses Claude Sonnet to reason about the best approach given
what the probe discovered. No brain model needed.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from .graph import (
    PhaseEdge,
    PhaseNode,
    PhaseRole,
    Postcondition,
    Precondition,
    RepeatMode,
)
from .objective import ObjectiveSpec
from .probe import ProbeResult

logger = logging.getLogger(__name__)


_KEYWORD_VALUE_PATTERNS: dict[str, re.Pattern[str]] = {
    "zip": re.compile(r"(?:zip\s*(?:code)?\s*)(\d{5})"),
    "zip code": re.compile(r"(?:zip\s*(?:code)?\s*)(\d{5})"),
    "price": re.compile(r"\$\s*([\d,]+)"),
    "min price": re.compile(r"\$\s*([\d,]+)"),
    "minimum price": re.compile(r"\$\s*([\d,]+)"),
    "city": re.compile(r"\b(?:near|in)\s+([a-z][a-z\-]+)"),
    "location": re.compile(r"\b(?:near|in)\s+([a-z][a-z\-]+)"),
    "state": re.compile(r"\b(florida|california|texas|new york)\b"),
}

_STATE_ABBREVIATIONS: dict[str, str] = {
    "florida": "fl", "california": "ca", "texas": "tx", "new york": "ny",
}


def _extract_filter_value(keyword: str, objective_text: str) -> str:
    """Extract the value for a templated filter keyword from objective text.

    Driven by ``_KEYWORD_VALUE_PATTERNS``; returns the empty string when
    no match is found. State names are normalized to two-letter codes
    so the URL segment matches the canonical site convention (``fl``
    not ``florida``).
    """
    pattern = _KEYWORD_VALUE_PATTERNS.get(keyword.lower())
    if not pattern:
        return ""
    match = pattern.search(objective_text)
    if not match:
        return ""
    value = match.group(1).lower()
    if keyword.lower() == "state":
        return _STATE_ABBREVIATIONS.get(value, value)
    if keyword.lower() in ("price", "min price", "minimum price"):
        return value.replace(",", "")
    if value in ("the", "a", "an", "all"):
        return ""
    return value


ENHANCE_PROMPT = """\
You are enhancing a CUA (Computer Use Agent) plan with concrete site knowledge.

OBJECTIVE:
{objective_text}

SITE PROBE RESULTS (what we found on the page):
{probe_json}

REQUIRED FILTERS: {filters}
TARGET ENTITY: {entity}

Based on the probe results, determine the BEST approach for each step:

1. FILTER STRATEGY: How should each filter be applied?
   Options for each filter:
   - "url": Filter can be encoded in the URL path/params (most reliable)
   - "sidebar": Filter is a clickable option in a sidebar/panel
   - "dropdown": Filter is in a select/dropdown control
   - "search": Filter requires typing in a search box
   - "unknown": Not detected in probe

2. NAVIGATION URL: What is the best starting URL?
   If filters can be URL-encoded, build the full filtered URL.
   If not, use the base results page URL.

3. ITEM CLICK TARGET: How should each item / row / card be clicked?
   Describe the item layout based on probe results.

4. DETAIL PAGE ACTIONS: What needs to happen on the detail page?
   - How many scrolls to reach key content?
   - Are there expandable sections? Which ones?
   - Where is the relevant detail / contact information located?

5. PAGINATION: How does pagination work?
   - URL-based (/page-N/), query param (?page=N), or button click?

Output ONLY valid JSON:
{{
  "navigation_url": "full URL with filters if possible",
  "filter_strategy": [
    {{"filter": "filter name", "method": "url|sidebar|dropdown|search", "detail": "how to apply"}}
  ],
  "card_description": "how listing cards look on this site",
  "detail_scrolls": 3,
  "expandable_sections": ["section names"],
  "expand_controls": ["Show more", "Read more"],
  "phone_location": "where phone numbers appear",
  "pagination_method": "url_path|url_query|button_click",
  "pagination_detail": "e.g. /page-N/ appended to path"
}}
"""


class PlanEnhancer:
    """Fill gaps in vague plans using site probe knowledge.

    ``filter_url_strategies`` (issue #464) maps lowercased objective
    keywords to URL path-segment templates. When non-empty, the
    enhancer attempts to encode required filters as URL segments
    (the BoatTrader / marketplace_listings pattern). When empty
    (the default), the URL builder is a no-op — generic CRM / admin
    flows don't carry predictable URL encoding rules and shouldn't
    inherit boattrader-shaped URL guesses.

    Recipes populate this via ``SiteConfig.filter_url_strategies``;
    see ``recipes/marketplace_listings/site_config.py`` for the
    reference set.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-opus-4-7",
        filter_url_strategies: dict[str, str] | None = None,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.filter_url_strategies: dict[str, str] = dict(filter_url_strategies or {})

    def enhance(
        self,
        objective: ObjectiveSpec,
        probe: ProbeResult,
    ) -> dict[str, Any]:
        """Analyze objective + probe results and return enhancement data.

        Returns a dict with concrete details for plan generation:
          navigation_url, filter_strategy, card_description,
          detail_scrolls, expandable_sections, pagination_method, etc.
        """
        if not self.api_key:
            logger.warning("PlanEnhancer: no API key, using heuristic enhancement")
            return self._enhance_heuristic(objective, probe)

        # Try URL-based filter encoding as a hint for Claude
        candidate_url = self._try_build_filtered_url(
            objective.start_url or "", objective
        )
        url_hint = ""
        if candidate_url != (objective.start_url or ""):
            url_hint = (
                f"\n\nURL FILTER HINT: I attempted to build a filtered URL: {candidate_url}\n"
                f"If this site encodes filters as URL path segments, use this or correct the format.\n"
                f"If the site does NOT use URL-based filters, ignore this hint and use sidebar/dropdown methods.\n"
                f"IMPORTANT: Get the URL segment order correct for this specific site."
            )

        prompt = ENHANCE_PROMPT.format(
            objective_text=objective.raw_text[:1500] + url_hint,
            probe_json=json.dumps(probe.to_dict(), indent=2)[:2000],
            filters=", ".join(objective.required_filters) or "none specified",
            entity=objective.target_entity or "listing",
        )

        try:
            import requests

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
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )
            if resp.status_code == 200:
                text = ""
                for block in resp.json().get("content", []):
                    if block.get("type") == "text":
                        text = block["text"].strip()
                        break
                return self._parse_response(text, objective, probe)
        except Exception as e:
            logger.warning("PlanEnhancer API call failed: %s", e)

        return self._enhance_heuristic(objective, probe)

    def _parse_response(self, text: str, objective: ObjectiveSpec, probe: ProbeResult) -> dict[str, Any]:
        """Parse Claude's enhancement response."""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
        try:
            data = json.loads(text)
            # Validate and fill defaults
            # Try URL-based filter encoding
            nav_url = data.get("navigation_url") or objective.start_url or ""
            enhanced_url = self._try_build_filtered_url(nav_url, objective)
            data["navigation_url"] = enhanced_url

            # Mark filters as "url" if they appear encoded in the nav URL
            url_lower = enhanced_url.lower()
            for fs in data.get("filter_strategy", []):
                filt_lower = fs.get("filter", "").lower()
                if any(seg in url_lower for seg in [
                    "by-owner", "by_owner",
                ] if filt_lower in ("private seller", "by owner")) or (
                    "zip-" in url_lower and filt_lower in ("zip", "location", "zip code")
                ) or (
                    "price-" in url_lower and filt_lower in ("price", "min price", "minimum price")
                ):
                    fs["method"] = "url"
                    fs["detail"] = "Encoded in URL"
            data.setdefault("filter_strategy", [])
            data.setdefault("card_description", "")
            data.setdefault("detail_scrolls", 5)
            data.setdefault("expandable_sections", [])
            data.setdefault("expand_controls", ["Show more", "Read more"])
            data.setdefault("phone_location", "")
            data.setdefault("pagination_method", "button_click")
            data.setdefault("pagination_detail", "")
            return data
        except json.JSONDecodeError:
            logger.warning("PlanEnhancer: failed to parse response")
            return self._enhance_heuristic(objective, probe)

    def _enhance_heuristic(self, objective: ObjectiveSpec, probe: ProbeResult) -> dict[str, Any]:
        """Heuristic enhancement when API is unavailable."""
        url = objective.start_url or ""

        # Try to build a filtered URL from the objective text
        # Many sites encode filters as URL path segments or query params.
        # Look for clues in the objective text and probe data.
        url = self._try_build_filtered_url(url, objective)

        # Determine filter strategy per filter
        # If the URL was enhanced with filter segments, mark those as "url" method.
        # The keyword→segment-template mapping comes from the active recipe via
        # ``self.filter_url_strategies``; when empty, no filter is classified as
        # url-encoded (generic flow).
        url_lower = url.lower()
        filter_strategy = []
        for filt in objective.required_filters:
            filt_lower = filt.lower()
            method = "unknown"

            # Check if this filter got encoded into the URL by looking at the
            # recipe's templates. ``segment_prefix`` strips ``{value}`` so we
            # can match segments like ``zip-33101`` against template ``zip-{value}``.
            url_encoded = False
            for keyword, template in self.filter_url_strategies.items():
                if keyword.lower() != filt_lower:
                    continue
                segment_prefix = template.split("{value}")[0]
                if segment_prefix and segment_prefix in url_lower:
                    url_encoded = True
                    break

            if url_encoded:
                method = "url"
            else:
                # Check probe-detected filters
                for detected in probe.filters_detected:
                    options = [o.lower() for o in detected.get("options", [])]
                    if filt_lower in " ".join(options):
                        location = detected.get("location", "")
                        if "sidebar" in location.lower():
                            method = "sidebar"
                        elif "dropdown" in location.lower() or "select" in location.lower():
                            method = "dropdown"
                        else:
                            method = "sidebar"
                        break

            filter_strategy.append({
                "filter": filt,
                "method": method,
                "detail": "Encoded in URL" if method == "url" else f"Apply {filt} via {method}",
            })

        # Pagination detection
        pagination = probe.pagination_controls or {}
        pag_type = pagination.get("type", "next_button")
        pag_method = "button_click"
        pag_detail = ""
        if pag_type == "numbered":
            # Try to infer from URL
            if "/page-" in url:
                pag_method = "url_path"
                pag_detail = "/page-{n}/"
            else:
                pag_method = "url_query"
                pag_detail = "page={n}"

        # Detail page info
        detail_info = probe.detail_page_pattern or {}
        expandable = detail_info.get("expandable_sections", [])
        expand_controls = detail_info.get("expand_controls", ["Show more", "Read more"])
        phone_location = detail_info.get("phone_location", "")

        return {
            "navigation_url": url,
            "filter_strategy": filter_strategy,
            "card_description": probe.listing_container.get("description", ""),
            "detail_scrolls": 5,
            "expandable_sections": expandable,
            "expand_controls": expand_controls,
            "phone_location": phone_location,
            "pagination_method": pag_method,
            "pagination_detail": pag_detail,
        }

    def _try_build_filtered_url(self, base_url: str, objective: ObjectiveSpec) -> str:
        """Try to build a URL with filters encoded in the path.

        Drives URL-segment encoding off ``self.filter_url_strategies``
        (issue #464). When that mapping is empty the function is a no-op
        and returns ``base_url`` unchanged — the pre-#464 hardcoded
        BoatTrader patterns now live in the marketplace_listings recipe's
        ``site_config.py`` and only apply when that recipe is selected.

        Keyword entries with no ``{value}`` placeholder are boolean
        filters (toggle the segment when the keyword appears in the
        objective text). Entries with ``{value}`` are extracted-value
        filters — the function scans the objective text for the
        keyword's expected value pattern (zip = 5 digits, price = $N,
        city/state = first word after "in"/"near") and substitutes it.

        Empty mapping → no encoding. Empty objective text → returns
        ``base_url`` unchanged. Generic CRM / admin / inspect-only
        plans never picked up URL guesses, even before.
        """
        if not base_url or not self.filter_url_strategies:
            return base_url

        raw = objective.raw_text.lower()
        url = base_url.rstrip("/")
        segments: list[str] = []

        # Group strategies into boolean toggles vs templated. We iterate
        # in insertion order so recipes can document the canonical
        # segment order by listing them in that order.
        seen_segments: set[str] = set()
        for keyword, template in self.filter_url_strategies.items():
            kw_lower = keyword.lower()
            if "{value}" not in template:
                # Boolean toggle — encode if the keyword appears in
                # the objective and the segment isn't already on the URL.
                if kw_lower in raw and template not in url and template not in seen_segments:
                    segments.append(template)
                    seen_segments.add(template)
                continue
            # Templated — extract a value for the keyword.
            value = _extract_filter_value(keyword, raw)
            if not value:
                continue
            seg = template.format(value=value)
            if seg not in url and seg not in seen_segments:
                segments.append(seg)
                seen_segments.add(seg)

        if not segments:
            return base_url

        enhanced = url
        for seg in segments:
            enhanced = f"{enhanced}/{seg}"
        enhanced += "/"

        logger.info("PlanEnhancer: built filtered URL: %s", enhanced)
        return enhanced

    def build_enhanced_phases(
        self,
        objective: ObjectiveSpec,
        probe: ProbeResult,
        enhancement: dict[str, Any],
    ) -> tuple[dict[str, PhaseNode], list[PhaseEdge]]:
        """Build concrete PhaseNodes from enhancement data.

        Returns (phases_dict, edges_list) ready for WorkflowGraph.
        """
        nav_url = enhancement.get("navigation_url") or objective.start_url or ""
        entity = objective.target_entity or "listing"
        filter_strategy = enhancement.get("filter_strategy", [])

        phases: dict[str, PhaseNode] = {}

        # ── Navigate ──
        # If filters are URL-encoded, use the filtered URL directly
        url_filters = [f for f in filter_strategy if f.get("method") == "url"]
        if url_filters and nav_url:
            phases["navigate"] = PhaseNode(
                id="navigate",
                role=PhaseRole.SETUP,
                intent_template=f"Navigate to filtered results at {nav_url}",
                budget=3,
                required=True,
                postconditions=[Postcondition(description="Filtered results page loaded")],
            )
        else:
            phases["navigate"] = PhaseNode(
                id="navigate",
                role=PhaseRole.SETUP,
                intent_template=f"Navigate to {nav_url}",
                budget=3,
                required=True,
                postconditions=[Postcondition(description="Page loaded")],
            )

        # ── Filter steps (only for non-URL filters) ──
        filter_ids: list[str] = []
        non_url_filters = [f for f in filter_strategy if f.get("method") != "url"]
        for i, filt in enumerate(non_url_filters):
            fid = f"filter_{i}"
            filter_ids.append(fid)
            method = filt.get("method", "unknown")
            filter_name = filt.get("filter", "")

            if method == "sidebar":
                intent = f"Click {filter_name} option in the sidebar filter panel"
            elif method == "dropdown":
                intent = f"Select {filter_name} from the dropdown menu"
            elif method == "search":
                intent = f"Type {filter_name} in the search box and press Enter"
            else:
                intent = f"Apply filter: {filter_name}"

            phases[fid] = PhaseNode(
                id=fid,
                role=PhaseRole.SETUP,
                intent_template=intent,
                budget=8,
                grounding=True,
                required=True,
            )

        # ── Gate ──
        all_filters = ", ".join(f.get("filter", "") for f in filter_strategy) or "required filters"
        phases["verify_scope"] = PhaseNode(
            id="verify_scope",
            role=PhaseRole.GATE,
            intent_template=f"Verify page shows {entity} results with {all_filters} applied",
            claude_only=True,
            budget=0,
            gate=True,
            preconditions=[Precondition(description=f"Filters applied: {all_filters}")],
            postconditions=[Postcondition(
                description=f"Page shows filtered {entity} results",
                verify_prompt=f"Page shows {entity} results with these filters active: {all_filters}. Result count should be reasonable (not unfiltered).",
            )],
        )

        # ── Extraction phases ──
        # Click intent: bias toward organic results when the objective
        # supplies forbidden_actions ("Contact Seller", "Sponsored", etc.);
        # stay neutral otherwise so non-marketplace flows don't get
        # listings-shaped guidance.
        card_desc = enhancement.get("card_description", "")
        if objective.forbidden_actions:
            skip_hint = ", ".join(objective.forbidden_actions[:3])
            click_intent = f"Click an organic {entity} title; skip items matching: {skip_hint}"
            if card_desc:
                click_intent = f"Click the title text of an organic {entity} card ({card_desc}); skip items matching: {skip_hint}"
        else:
            click_intent = f"Click the {entity} title text on the results page"
            if card_desc:
                click_intent = f"Click the title text of a {entity} card ({card_desc})"

        phases["admit_candidate"] = PhaseNode(
            id="admit_candidate",
            role=PhaseRole.ADMISSION,
            intent_template=click_intent,
            budget=8,
            grounding=True,
            repeat=RepeatMode.FOR_EACH,
            source_phase="discover_candidates",
        )
        phases["extract_url"] = PhaseNode(
            id="extract_url",
            role=PhaseRole.EXTRACTION,
            intent_template="Read the URL from browser address bar",
            claude_only=True,
            budget=0,
            repeat=RepeatMode.FOR_EACH,
            source_phase="discover_candidates",
        )

        # Scroll with concrete count from probe
        scroll_count = enhancement.get("detail_scrolls", 5)
        phases["scroll_to_details"] = PhaseNode(
            id="scroll_to_details",
            role=PhaseRole.EXTRACTION,
            intent_template=f"Scroll down {scroll_count} times toward the description and detail sections",
            budget=max(scroll_count + 2, 10),
            repeat=RepeatMode.FOR_EACH,
            source_phase="discover_candidates",
        )

        # Extraction with expand info
        expand_sections = enhancement.get("expandable_sections", [])
        phone_location = enhancement.get("phone_location", "")
        extract_detail = "inspect contact area, expand collapsed sections, then extract structured data"
        if expand_sections:
            sections_str = ", ".join(expand_sections[:3])
            extract_detail = f"expand {sections_str} if collapsed, inspect contact area, then extract structured data"
        if phone_location:
            extract_detail += f". Phone numbers may be in {phone_location}"

        phases["extract_fields"] = PhaseNode(
            id="extract_fields",
            role=PhaseRole.EXTRACTION,
            intent_template=f"Reject spam, {extract_detail}",
            claude_only=True,
            budget=0,
            repeat=RepeatMode.FOR_EACH,
            source_phase="discover_candidates",
        )
        phases["return_to_results"] = PhaseNode(
            id="return_to_results",
            role=PhaseRole.RETURN,
            intent_template="Go back to search results page",
            budget=3,
            repeat=RepeatMode.FOR_EACH,
            source_phase="discover_candidates",
        )

        # ── Pagination ──
        pag_method = enhancement.get("pagination_method", "button_click")
        if pag_method == "button_click":
            pag_intent = "Click Next page button to continue to next results page"
        elif pag_method == "url_path":
            pag_intent = "Click Next page link or page number to load the next results page"
        else:
            pag_intent = "Click the next page control to load more results"

        phases["paginate"] = PhaseNode(
            id="paginate",
            role=PhaseRole.PAGINATION,
            intent_template=pag_intent,
            budget=10,
            grounding=True,
            repeat=RepeatMode.UNTIL_EXHAUSTED,
        )

        # ── Edges ──
        edges: list[PhaseEdge] = []
        prev = "navigate"
        for fid in filter_ids:
            edges.append(PhaseEdge(source=prev, target=fid))
            prev = fid
        edges.append(PhaseEdge(source=prev, target="verify_scope"))
        edges.extend([
            PhaseEdge(source="verify_scope", target="admit_candidate"),
            PhaseEdge(source="admit_candidate", target="extract_url"),
            PhaseEdge(source="extract_url", target="scroll_to_details"),
            PhaseEdge(source="scroll_to_details", target="extract_fields"),
            PhaseEdge(source="extract_fields", target="return_to_results"),
            PhaseEdge(source="return_to_results", target="paginate", condition="exhausted"),
            PhaseEdge(source="paginate", target="admit_candidate"),
        ])

        return phases, edges
