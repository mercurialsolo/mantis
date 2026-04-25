"""SiteConfig — URL patterns and page structure for a target site.

Replaces hardcoded BoatTrader patterns (/boats/, /boat/, /page-N/) with
configurable patterns that work for any site. Populated by SiteProber
or constructed manually.

Usage:
    # From a ProbeResult
    config = SiteConfig.from_probe(probe_result)

    # BoatTrader default (backward compat)
    config = SiteConfig.default_boattrader()

    # Check if URL is a detail page
    config.is_detail_page("https://boattrader.com/boat/2020-sea-ray-123/")  # True
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SiteConfig:
    """URL patterns and page structure for a target site."""

    domain: str = ""

    # URL patterns (regex)
    detail_page_pattern: str = ""  # e.g. r"/boat/[\w-]+" or r"/homes/\d+"
    results_page_pattern: str = ""  # e.g. r"/boats/" or r"/homes/"

    # Pagination
    pagination_format: str = ""  # template, e.g. "/page-{n}/" or "?page={n}" or "&start={n}"
    pagination_type: str = "path_suffix"  # "path_suffix", "query_param", "next_button"
    pagination_strip_pattern: str = ""  # regex to strip existing page param from URL

    # Gate verification
    gate_verify_prompt: str = ""  # Custom prompt for gate verification, or empty for generic

    # Filter recovery
    filtered_results_url: str = ""  # URL to navigate to if filters are lost

    def is_detail_page(self, url: str) -> bool:
        """Check if URL matches the detail page pattern."""
        if not self.detail_page_pattern:
            return False
        return bool(re.search(self.detail_page_pattern, url, re.IGNORECASE))

    def is_results_page(self, url: str) -> bool:
        """Check if URL matches the results page pattern."""
        if not self.results_page_pattern:
            return False
        return bool(re.search(self.results_page_pattern, url, re.IGNORECASE))

    def paginated_url(self, base_url: str, page: int) -> str:
        """Build a paginated URL for the given page number."""
        if not self.pagination_format:
            return base_url

        base_clean = base_url.rstrip("/")
        if self.pagination_strip_pattern:
            base_clean = re.sub(self.pagination_strip_pattern, "", base_clean)

        fmt = self.pagination_format
        if self.pagination_type == "path_suffix":
            return f"{base_clean}{fmt.format(n=page)}"
        elif self.pagination_type == "query_param":
            sep = "&" if "?" in base_clean else "?"
            return f"{base_clean}{sep}{fmt.format(n=page)}"
        return f"{base_clean}{fmt.format(n=page)}"

    @classmethod
    def default_boattrader(cls) -> SiteConfig:
        """The current hardcoded BoatTrader patterns for backward compatibility."""
        return cls(
            domain="boattrader.com",
            detail_page_pattern=r"/boat/[\w-]+",
            results_page_pattern=r"/boats/",
            pagination_format="/page-{n}/",
            pagination_type="path_suffix",
            pagination_strip_pattern=r"/page-\d+/?$",
            gate_verify_prompt=(
                "Page is a filtered results page with these active filters: "
            ),
            filtered_results_url="https://www.boattrader.com/boats/by-owner/",
        )

    @classmethod
    def from_probe(cls, probe_result: Any) -> SiteConfig:
        """Build SiteConfig from a ProbeResult."""
        domain = getattr(probe_result, "domain", "") or ""
        url = getattr(probe_result, "url", "") or ""

        # Detect pagination from probe
        pagination = getattr(probe_result, "pagination_controls", {}) or {}
        pagination_type_str = pagination.get("type", "next_button")
        pagination_format = ""
        pagination_strip = ""
        if pagination_type_str == "numbered" or pagination_type_str == "next_button":
            # Try to infer from URL structure
            if "/page-" in url:
                pagination_format = "/page-{n}/"
                pagination_strip = r"/page-\d+/?$"
                pagination_type_str = "path_suffix"
            else:
                pagination_format = "page={n}"
                pagination_strip = r"[?&]page=\d+"
                pagination_type_str = "query_param"

        # Detect detail page pattern from probe
        detail_pattern = ""
        detail_info = getattr(probe_result, "detail_page_pattern", {}) or {}
        if isinstance(detail_info, dict):
            url_pattern = detail_info.get("url_pattern", "")
            if url_pattern:
                # Convert user-friendly pattern to regex
                detail_pattern = url_pattern.replace("<slug>", r"[\w-]+").replace("<id>", r"\d+")

        # Detect results page pattern from URL
        results_pattern = ""
        if url:
            # Use the URL path up to the first query param as the results pattern
            from urllib.parse import urlparse
            parsed = urlparse(url)
            path = parsed.path.rstrip("/")
            if path:
                # Escape the path for regex use
                results_pattern = re.escape(path)

        return cls(
            domain=domain,
            detail_page_pattern=detail_pattern,
            results_page_pattern=results_pattern,
            pagination_format=pagination_format,
            pagination_type=pagination_type_str,
            pagination_strip_pattern=pagination_strip,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "domain": self.domain,
            "detail_page_pattern": self.detail_page_pattern,
            "results_page_pattern": self.results_page_pattern,
            "pagination_format": self.pagination_format,
            "pagination_type": self.pagination_type,
            "pagination_strip_pattern": self.pagination_strip_pattern,
            "gate_verify_prompt": self.gate_verify_prompt,
            "filtered_results_url": self.filtered_results_url,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> SiteConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
