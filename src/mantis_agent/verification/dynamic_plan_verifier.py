"""Dynamic execution coverage checks for looped browser plans.

The verifier records what the runtime discovered on each results page, which
items it attempted, which items opened successfully, and whether pagination was
attempted. It is intentionally independent of the CUA model so it can be used
both during extraction runs and during lightweight page-analysis runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _title_key(title: str) -> str:
    key = re.sub(r"\s+", " ", (title or "").strip().lower())
    return key or "unknown"


def _title_value(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("title") or item.get("name") or "unknown")
    if isinstance(item, (list, tuple)) and len(item) >= 3:
        return str(item[2] or "unknown")
    return str(item or "unknown")


@dataclass
class ViewportScan:
    page: int
    viewport_stage: int
    status: str
    cards_found: int
    new_cards: int
    titles: list[str] = field(default_factory=list)
    new_titles: list[str] = field(default_factory=list)
    url: str = ""
    pagination_y: int | None = None
    ts: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "viewport_stage": self.viewport_stage,
            "status": self.status,
            "cards_found": self.cards_found,
            "new_cards": self.new_cards,
            "titles": self.titles,
            "new_titles": self.new_titles,
            "url": self.url,
            "pagination_y": self.pagination_y,
            "ts": self.ts,
        }


@dataclass
class PageCoverage:
    page: int
    url: str = ""
    found_items: dict[str, str] = field(default_factory=dict)
    attempted_items: dict[str, str] = field(default_factory=dict)
    opened_items: dict[str, str] = field(default_factory=dict)
    completed_items: dict[str, str] = field(default_factory=dict)
    successful_items: dict[str, str] = field(default_factory=dict)
    failed_items: dict[str, str] = field(default_factory=dict)
    opened_urls: dict[str, str] = field(default_factory=dict)
    scans: list[ViewportScan] = field(default_factory=list)
    filter_passed: bool = False
    filter_failures: list[str] = field(default_factory=list)
    page_exhausted: bool = False
    pagination_attempted: bool = False
    pagination_succeeded: bool = False
    pagination_method: str = ""
    next_url: str = ""
    stop_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        found = list(self.found_items.values())
        attempted = list(self.attempted_items.values())
        opened = list(self.opened_items.values())
        completed = list(self.completed_items.values())
        successful = list(self.successful_items.values())
        failed = list(self.failed_items.values())
        missing = [
            title
            for key, title in self.found_items.items()
            if key not in self.attempted_items
        ]
        incomplete = [
            title
            for key, title in self.attempted_items.items()
            if key not in self.completed_items
        ]
        unopened = [
            title
            for key, title in self.attempted_items.items()
            if key not in self.opened_items
        ]
        return {
            "page": self.page,
            "url": self.url,
            "found_count": len(found),
            "attempted_count": len(attempted),
            "opened_count": len(opened),
            "completed_count": len(completed),
            "found_items_count": len(found),
            "attempted_items_count": len(attempted),
            "opened_items_count": len(opened),
            "completed_items_count": len(completed),
            "successful_items_count": len(successful),
            "failed_items_count": len(failed),
            "missing_attempts_count": len(missing),
            "incomplete_attempts_count": len(incomplete),
            "unopened_attempts_count": len(unopened),
            "found_items": found,
            "attempted_items": attempted,
            "opened_items": opened,
            "completed_items": completed,
            "successful_items": successful,
            "failed_items": failed,
            "found_titles": found,
            "attempted_titles": attempted,
            "opened_titles": opened,
            "completed_titles": completed,
            "opened_urls": list(self.opened_urls.values()),
            "missing_attempts": missing,
            "incomplete_attempts": incomplete,
            "unopened_attempts": unopened,
            "filter_passed": self.filter_passed,
            "filter_failures": self.filter_failures,
            "page_exhausted": self.page_exhausted,
            "pagination_attempted": self.pagination_attempted,
            "pagination_succeeded": self.pagination_succeeded,
            "pagination_method": self.pagination_method,
            "next_url": self.next_url,
            "stop_reason": self.stop_reason,
            "scans": [scan.to_dict() for scan in self.scans],
        }


class DynamicPlanVerifier:
    """Records and evaluates coverage for dynamic item-browsing plans."""

    def __init__(
        self,
        *,
        required_filter_tokens: tuple[str, ...] = (),
        plan_name: str = "",
    ) -> None:
        self.required_filter_tokens = tuple(required_filter_tokens)
        self.plan_name = plan_name
        self.pages: dict[int, PageCoverage] = {}
        self.events: list[dict[str, Any]] = []

    def set_required_filter_tokens(self, tokens: tuple[str, ...]) -> None:
        self.required_filter_tokens = tuple(tokens)

    def load_report(self, report: dict[str, Any]) -> None:
        """Restore verifier state from a previously persisted report."""
        if not isinstance(report, dict):
            return
        self.plan_name = str(report.get("plan_name") or self.plan_name)
        self.required_filter_tokens = tuple(report.get("required_filter_tokens") or ())
        self.events = list(report.get("events") or [])
        self.pages = {}
        for page_payload in report.get("pages") or []:
            try:
                page_num = int(page_payload.get("page") or 1)
            except Exception:
                page_num = 1
            page_cov = PageCoverage(
                page=page_num,
                url=str(page_payload.get("url") or ""),
                filter_passed=bool(page_payload.get("filter_passed")),
                filter_failures=list(page_payload.get("filter_failures") or []),
                page_exhausted=bool(page_payload.get("page_exhausted")),
                pagination_attempted=bool(page_payload.get("pagination_attempted")),
                pagination_succeeded=bool(page_payload.get("pagination_succeeded")),
                pagination_method=str(page_payload.get("pagination_method") or ""),
                next_url=str(page_payload.get("next_url") or ""),
                stop_reason=str(page_payload.get("stop_reason") or ""),
            )
            for attr, keys in (
                ("found_items", ("found_items", "found_titles")),
                ("attempted_items", ("attempted_items", "attempted_titles")),
                ("opened_items", ("opened_items", "opened_titles")),
                ("completed_items", ("completed_items", "completed_titles")),
                ("successful_items", ("successful_items",)),
                ("failed_items", ("failed_items",)),
            ):
                values: list[str] = []
                for key in keys:
                    values = list(page_payload.get(key) or [])
                    if values:
                        break
                target = getattr(page_cov, attr)
                for value in values:
                    target.setdefault(_title_key(str(value)), str(value))
            for url in page_payload.get("opened_urls") or []:
                page_cov.opened_urls.setdefault(str(url), str(url))
            for scan_payload in page_payload.get("scans") or []:
                try:
                    page_cov.scans.append(
                        ViewportScan(
                            page=int(scan_payload.get("page") or page_num),
                            viewport_stage=int(scan_payload.get("viewport_stage") or 0),
                            status=str(scan_payload.get("status") or ""),
                            cards_found=int(scan_payload.get("cards_found") or 0),
                            new_cards=int(scan_payload.get("new_cards") or 0),
                            titles=list(scan_payload.get("titles") or []),
                            new_titles=list(scan_payload.get("new_titles") or []),
                            url=str(scan_payload.get("url") or ""),
                            pagination_y=scan_payload.get("pagination_y"),
                            ts=str(scan_payload.get("ts") or _utc_now()),
                        )
                    )
                except Exception:
                    continue
            self.pages[page_num] = page_cov

    def _page(self, page: int, url: str = "") -> PageCoverage:
        page_cov = self.pages.setdefault(page, PageCoverage(page=page))
        if url:
            page_cov.url = url
        return page_cov

    def record_page_start(self, *, page: int, url: str = "") -> None:
        self._page(page, url)
        self._event("page_start", page=page, url=url)

    def record_filter_check(self, *, page: int, url: str, passed: bool, reason: str = "") -> None:
        page_cov = self._page(page, url)
        if passed:
            page_cov.filter_passed = True
        else:
            page_cov.filter_failures.append(reason or "filter_check_failed")
        self._event("filter_check", page=page, url=url, passed=passed, reason=reason)

    def record_viewport_scan(
        self,
        *,
        page: int,
        viewport_stage: int,
        cards: list[Any],
        new_cards: list[Any] | None = None,
        status: str = "ok",
        url: str = "",
        pagination_y: int | None = None,
    ) -> None:
        page_cov = self._page(page, url)
        titles = [_title_value(item) for item in cards]
        new_titles = [_title_value(item) for item in (new_cards if new_cards is not None else cards)]
        for title in titles:
            page_cov.found_items.setdefault(_title_key(title), title)
        scan = ViewportScan(
            page=page,
            viewport_stage=viewport_stage,
            status=status,
            cards_found=len(cards),
            new_cards=len(new_titles),
            titles=titles,
            new_titles=new_titles,
            url=url,
            pagination_y=pagination_y,
        )
        page_cov.scans.append(scan)
        self._event(
            "viewport_scan",
            page=page,
            viewport_stage=viewport_stage,
            status=status,
            cards_found=len(cards),
            new_cards=len(new_titles),
            titles=titles,
            url=url,
        )

    def record_item_attempt(
        self,
        *,
        page: int,
        item: str,
        viewport_stage: int | None = None,
    ) -> None:
        page_cov = self._page(page)
        page_cov.attempted_items.setdefault(_title_key(item), item)
        self._event("item_attempt", page=page, item=item, viewport_stage=viewport_stage)

    def record_item_opened(
        self,
        *,
        page: int,
        item: str,
        url: str,
    ) -> None:
        page_cov = self._page(page)
        key = _title_key(item)
        page_cov.opened_items.setdefault(key, item)
        if url:
            page_cov.opened_urls.setdefault(url, url)
        self._event("item_opened", page=page, item=item, url=url)

    def record_item_completed(
        self,
        *,
        page: int,
        item: str,
        url: str = "",
        success: bool = True,
        reason: str = "",
    ) -> None:
        page_cov = self._page(page)
        key = _title_key(item)
        page_cov.completed_items.setdefault(key, item)
        if success:
            page_cov.successful_items.setdefault(key, item)
        else:
            page_cov.failed_items.setdefault(key, item)
        self._event(
            "item_completed",
            page=page,
            item=item,
            url=url,
            success=success,
            reason=reason,
        )

    def record_listing_attempt(self, *, page: int, title: str, viewport_stage: int | None = None) -> None:
        self.record_item_attempt(page=page, item=title, viewport_stage=viewport_stage)

    def record_listing_opened(self, *, page: int, title: str, url: str) -> None:
        self.record_item_opened(page=page, item=title, url=url)

    def record_listing_completed(
        self,
        *,
        page: int,
        title: str,
        url: str = "",
        success: bool = True,
        reason: str = "",
    ) -> None:
        self.record_item_completed(page=page, item=title, url=url, success=success, reason=reason)

    def record_page_exhausted(self, *, page: int, reason: str = "") -> None:
        page_cov = self._page(page)
        page_cov.page_exhausted = True
        page_cov.stop_reason = reason or "page_exhausted"
        self._event("page_exhausted", page=page, reason=page_cov.stop_reason)

    def record_pagination(
        self,
        *,
        page: int,
        success: bool,
        method: str = "",
        next_url: str = "",
        reason: str = "",
    ) -> None:
        page_cov = self._page(page)
        page_cov.pagination_attempted = True
        page_cov.pagination_succeeded = success
        page_cov.pagination_method = method
        page_cov.next_url = next_url
        if reason:
            page_cov.stop_reason = reason
        self._event(
            "pagination",
            page=page,
            success=success,
            method=method,
            next_url=next_url,
            reason=reason,
        )

    def report(self, *, status: str = "running") -> dict[str, Any]:
        pages = [self.pages[key].to_dict() for key in sorted(self.pages)]
        found = sum(page["found_count"] for page in pages)
        attempted = sum(page["attempted_count"] for page in pages)
        opened = sum(page["opened_count"] for page in pages)
        completed = sum(page["completed_count"] for page in pages)
        successful = sum(page["successful_items_count"] for page in pages)
        failed_items = sum(page["failed_items_count"] for page in pages)
        missing = sum(page["missing_attempts_count"] for page in pages)
        incomplete = sum(page["incomplete_attempts_count"] for page in pages)
        unopened = sum(page["unopened_attempts_count"] for page in pages)
        checks = self._checks(status, pages)
        verdict = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
        if status in {"running", "queued", "analysis_only"}:
            verdict = status
        elif any(check["status"] == "pending" for check in checks):
            verdict = "partial"
        return {
            "plan_name": self.plan_name,
            "status": status,
            "verdict": verdict,
            "required_filter_tokens": list(self.required_filter_tokens),
            "totals": {
                "pages_seen": len(pages),
                "found_items": found,
                "attempted_items": attempted,
                "opened_items": opened,
                "completed_items": completed,
                "successful_items": successful,
                "failed_items": failed_items,
                "found_private_cards": found,
                "attempted_cards": attempted,
                "opened_cards": opened,
                "completed_cards": completed,
                "missing_attempts": missing,
                "incomplete_attempts": incomplete,
                "unopened_attempts": unopened,
            },
            "checks": checks,
            "pages": pages,
            "events": self.events[-500:],
        }

    def _checks(self, status: str, pages: list[dict[str, Any]]) -> list[dict[str, str]]:
        checks: list[dict[str, str]] = []
        if not pages:
            checks.append({
                "name": "pages_discovered",
                "status": "pending" if status in {"running", "queued", "analysis_only"} else "fail",
                "detail": "no result pages were observed",
            })
            return checks

        for page in pages:
            if not self.required_filter_tokens or page["filter_passed"]:
                checks.append({"name": f"page_{page['page']}_required_filters_present", "status": "pass"})
            elif status in {"running", "queued", "analysis_only"}:
                checks.append({
                    "name": f"page_{page['page']}_required_filters_present",
                    "status": "pending",
                    "detail": "required filter state has not been verified yet",
                })
            else:
                checks.append({
                    "name": f"page_{page['page']}_required_filters_present",
                    "status": "fail",
                    "detail": "required filter state was not verified",
                })

            if page["missing_attempts_count"] == 0:
                checks.append({"name": f"page_{page['page']}_found_items_attempted", "status": "pass"})
            elif status in {"running", "queued", "analysis_only"} and not page["page_exhausted"]:
                checks.append({
                    "name": f"page_{page['page']}_found_items_attempted",
                    "status": "pending",
                    "detail": f"{page['missing_attempts_count']} discovered cards not attempted yet",
                })
            else:
                checks.append({
                    "name": f"page_{page['page']}_found_items_attempted",
                    "status": "fail",
                    "detail": f"{page['missing_attempts_count']} discovered cards were not attempted",
                })

            if page["incomplete_attempts_count"] == 0:
                checks.append({"name": f"page_{page['page']}_attempted_items_completed", "status": "pass"})
            elif status in {"running", "queued", "analysis_only"}:
                checks.append({
                    "name": f"page_{page['page']}_attempted_items_completed",
                    "status": "pending",
                    "detail": f"{page['incomplete_attempts_count']} attempted items have no terminal extraction result yet",
                })
            else:
                checks.append({
                    "name": f"page_{page['page']}_attempted_items_completed",
                    "status": "fail",
                    "detail": f"{page['incomplete_attempts_count']} attempted items were not completed",
                })

            if page["unopened_attempts_count"] == 0:
                checks.append({"name": f"page_{page['page']}_attempted_items_opened", "status": "pass"})
            elif status in {"running", "queued", "analysis_only"}:
                checks.append({
                    "name": f"page_{page['page']}_attempted_items_opened",
                    "status": "pending",
                    "detail": f"{page['unopened_attempts_count']} attempted items have not opened yet",
                })
            else:
                checks.append({
                    "name": f"page_{page['page']}_attempted_items_opened",
                    "status": "fail",
                    "detail": f"{page['unopened_attempts_count']} attempted items never opened",
                })

            if page["failed_items_count"] == 0:
                checks.append({"name": f"page_{page['page']}_completed_without_item_failures", "status": "pass"})
            elif status in {"running", "queued", "analysis_only"}:
                checks.append({
                    "name": f"page_{page['page']}_completed_without_item_failures",
                    "status": "pending",
                    "detail": f"{page['failed_items_count']} items have failed terminal outcomes so far",
                })
            else:
                checks.append({
                    "name": f"page_{page['page']}_completed_without_item_failures",
                    "status": "fail",
                    "detail": f"{page['failed_items_count']} items had failed terminal outcomes",
                })

            if not page["pagination_attempted"] or page["page_exhausted"]:
                checks.append({"name": f"page_{page['page']}_page_exhausted_before_pagination", "status": "pass"})
            elif status in {"running", "queued", "analysis_only"}:
                checks.append({
                    "name": f"page_{page['page']}_page_exhausted_before_pagination",
                    "status": "pending",
                    "detail": "pagination was attempted before page exhaustion was recorded",
                })
            else:
                checks.append({
                    "name": f"page_{page['page']}_page_exhausted_before_pagination",
                    "status": "fail",
                    "detail": "pagination was attempted before visible items were exhausted",
                })

            if page["page_exhausted"] or page["pagination_attempted"] or status in {"running", "queued", "analysis_only"}:
                checks.append({"name": f"page_{page['page']}_has_terminal_state", "status": "pass"})
            else:
                checks.append({
                    "name": f"page_{page['page']}_has_terminal_state",
                    "status": "fail",
                    "detail": "page was left without exhaustion or pagination",
                })
        return checks

    def _event(self, kind: str, **payload: Any) -> None:
        self.events.append({"ts": _utc_now(), "kind": kind, **payload})
