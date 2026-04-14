"""PlanExecutor — directly execute plan steps via Playwright DOM queries.

For deterministic plan steps (navigate to URL, type text into a field,
click a button by label), bypass the vision model entirely and use
Playwright selectors. For ambiguous steps, fall back to the brain.

This is the hybrid approach: structured steps execute reliably via DOM,
while the brain handles visual reasoning for steps that need it.

Strategy per action:
  navigate → page.goto()
  type     → find input by placeholder/name/label, fill()
  click    → find element by text/role/selector, click()
  key      → page.keyboard.press()
  scroll   → page.mouse.wheel()
  wait     → time.sleep()
  verify   → DOM query for expected state
  (other)  → fall back to brain
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result of executing a single plan step."""
    success: bool
    method: str  # "direct" or "brain_fallback"
    detail: str = ""
    url_after: str = ""


class PlanExecutor:
    """Execute plan steps directly via Playwright, falling back to brain.

    Args:
        page: Playwright Page object, or an env with a .page property.
            If an env is passed, the page is resolved lazily (after reset).
        settle_time: Seconds to wait after each action.
    """

    def __init__(self, page=None, settle_time: float = 1.5, env=None):
        self._page_ref = page
        self._env = env
        self._settle_time = settle_time

    @property
    def _page(self):
        """Resolve the page lazily — env.page is only available after reset."""
        if self._page_ref is not None:
            return self._page_ref
        if self._env is not None and hasattr(self._env, "page"):
            return self._env.page
        return None

    def can_execute(self, step) -> bool:
        """Check if a plan step can be executed directly (no brain needed)."""
        return step.action in ("navigate", "type", "click", "key", "scroll", "wait", "verify")

    def execute(self, step, resolved_params: dict[str, str] | None = None) -> StepResult:
        """Execute a plan step directly via Playwright.

        Args:
            step: PlanStep to execute.
            resolved_params: Dict of resolved {{variable}} → value.

        Returns:
            StepResult with success/failure and detail.
        """
        if self._page is None:
            return StepResult(success=False, method="direct", detail="no page available (env not reset?)")

        resolved_params = resolved_params or {}
        action = step.action

        try:
            if action == "navigate":
                return self._navigate(step, resolved_params)
            elif action == "type":
                return self._type(step, resolved_params)
            elif action == "click":
                return self._click(step, resolved_params)
            elif action == "key":
                return self._key(step, resolved_params)
            elif action == "scroll":
                return self._scroll(step)
            elif action == "wait":
                return self._wait(step)
            elif action == "verify":
                return self._verify(step, resolved_params)
            else:
                return StepResult(success=False, method="unsupported", detail=f"Unknown action: {action}")
        except Exception as e:
            logger.warning(f"Direct execution failed for {action}: {e}")
            return StepResult(success=False, method="direct", detail=f"Error: {e}")

    def _navigate(self, step, params: dict) -> StepResult:
        url = step.params.get("url", step.target)
        # Resolve {{variables}}
        for key, val in params.items():
            url = url.replace(f"{{{{{key}}}}}", val)
        self._page.goto(url, wait_until="domcontentloaded")
        time.sleep(self._settle_time)
        return StepResult(success=True, method="direct", detail=f"navigated to {url}", url_after=self._page.url)

    def _type(self, step, params: dict) -> StepResult:
        text = step.params.get("text", "")
        for key, val in params.items():
            text = text.replace(f"{{{{{key}}}}}", val)

        target = step.target

        # Strategy 1: if target is specified, find by description
        if target:
            el = self._find_input(target)
            if el:
                el.click()
                time.sleep(0.3)
                try:
                    el.evaluate("el => { if (el.select) el.select(); el.value = ''; }")
                except Exception:
                    pass
                el.type(text)
                time.sleep(self._settle_time)
                actual = ""
                try:
                    actual = el.evaluate("el => el.value") or ""
                except Exception:
                    pass
                if text in actual:
                    return StepResult(success=True, method="direct", detail=f"typed '{text}' into '{target}' (verified)")
                else:
                    return StepResult(success=True, method="direct", detail=f"typed '{text}' into '{target}'")

        # Strategy 2: type into the currently focused element (covers select, input, textarea, contenteditable)
        try:
            focused_info = self._page.evaluate("""() => {
                const el = document.activeElement;
                if (!el) return null;
                const tag = el.tagName.toLowerCase();
                return {tag, editable: tag === 'input' || tag === 'textarea' || tag === 'select' || el.isContentEditable};
            }""")
            if focused_info and focused_info.get("editable"):
                # For select elements, try to find option with matching text
                if focused_info["tag"] == "select":
                    try:
                        self._page.select_option("select:focus", label=text)
                        time.sleep(self._settle_time)
                        return StepResult(success=True, method="direct", detail=f"selected '{text}' from dropdown")
                    except Exception:
                        pass
                # For input/textarea/contenteditable: keyboard type
                self._page.keyboard.type(text)
                time.sleep(self._settle_time)
                return StepResult(success=True, method="direct", detail=f"typed '{text}' into focused {focused_info['tag']}")
        except Exception:
            pass

        # Strategy 3: just type via keyboard (last resort — something might receive it)
        self._page.keyboard.type(text)
        time.sleep(self._settle_time)
        return StepResult(success=True, method="direct", detail=f"typed '{text}' via keyboard (no target found)")

    def _click(self, step, params: dict) -> StepResult:
        target = step.target
        for key, val in params.items():
            target = target.replace(f"{{{{{key}}}}}", val)

        # Strategy 1: find by exact text match (buttons, links)
        el = self._find_clickable(target)
        if el:
            el.click()
            time.sleep(self._settle_time)
            return StepResult(success=True, method="direct", detail=f"clicked '{target}'", url_after=self._page.url)

        # Strategy 2: find by role + name
        el = self._find_by_role(target)
        if el:
            el.click()
            time.sleep(self._settle_time)
            return StepResult(success=True, method="direct", detail=f"clicked role element '{target}'", url_after=self._page.url)

        # Strategy 3: target might be an input field (e.g. "User ID input field")
        el = self._find_input(target)
        if el:
            el.click()
            time.sleep(self._settle_time)
            return StepResult(success=True, method="direct", detail=f"clicked input '{target}'", url_after=self._page.url)

        return StepResult(success=False, method="direct", detail=f"could not find clickable element for '{target}'")

    def _key(self, step, params: dict) -> StepResult:
        keys = step.params.get("keys", "")
        self._page.keyboard.press(keys)
        time.sleep(self._settle_time)
        return StepResult(success=True, method="direct", detail=f"pressed {keys}")

    def _scroll(self, step) -> StepResult:
        direction = step.params.get("direction", "down")
        amount = step.params.get("amount", 3)
        vw, vh = 1280, 720  # Default viewport
        x, y = vw // 2, vh // 2
        dx, dy = 0, 0
        if direction == "up":
            dy = -amount * 100
        elif direction == "down":
            dy = amount * 100
        elif direction == "left":
            dx = -amount * 100
        elif direction == "right":
            dx = amount * 100
        self._page.mouse.move(x, y)
        self._page.mouse.wheel(dx, dy)
        time.sleep(self._settle_time)
        return StepResult(success=True, method="direct", detail=f"scrolled {direction} {amount}x")

    def _wait(self, step) -> StepResult:
        seconds = step.params.get("seconds", 2.0)
        time.sleep(min(float(seconds), 10.0))
        return StepResult(success=True, method="direct", detail=f"waited {seconds}s")

    def _verify(self, step, params: dict) -> StepResult:
        check = step.params.get("check", "")
        value = step.params.get("value", "")
        for key, val in params.items():
            value = value.replace(f"{{{{{key}}}}}", val)

        if check == "url_contains":
            if value.lower() in self._page.url.lower():
                return StepResult(success=True, method="direct", detail=f"URL contains '{value}'")
            return StepResult(success=False, method="direct", detail=f"URL '{self._page.url}' does not contain '{value}'")

        elif check == "url_not_contains":
            if value.lower() not in self._page.url.lower():
                return StepResult(success=True, method="direct", detail=f"URL does not contain '{value}' (good)")
            return StepResult(success=False, method="direct", detail=f"URL '{self._page.url}' still contains '{value}'")

        elif check == "page_contains_text":
            try:
                body_text = self._page.inner_text("body")
                if value.lower() in body_text.lower():
                    return StepResult(success=True, method="direct", detail=f"page contains '{value}'")
                return StepResult(success=False, method="direct", detail=f"page does not contain '{value}'")
            except Exception as e:
                return StepResult(success=False, method="direct", detail=f"verify error: {e}")

        return StepResult(success=False, method="direct", detail=f"unknown check type: {check}")

    # ── Element finding strategies ───────────────────────────────────────

    def _find_input(self, target: str):
        """Find an input element matching a natural language target.

        Uses Playwright's native locators (most reliable in headless):
        1. Type-based: password, email → input[type=X]
        2. get_by_placeholder (regex, handles case)
        3. get_by_label (for label-associated inputs)
        4. CSS id/name selectors built from keywords
        """
        target_lower = target.lower()

        # Strategy 1: type-based for well-known field types
        type_map = {
            "password": 'input[type="password"]',
            "email": 'input[type="email"]',
            "search": 'input[type="search"]',
        }
        for keyword, selector in type_map.items():
            if keyword in target_lower:
                try:
                    el = self._page.query_selector(selector)
                    if el and el.is_visible():
                        return el
                except Exception:
                    pass

        # Extract meaningful keywords from target
        keywords = [w for w in target_lower.split() if len(w) > 2 and w not in (
            "the", "input", "field", "for", "this", "into", "text", "enter", "your",
        )]

        # Strategy 2: Playwright get_by_placeholder (most reliable native locator)
        for kw in keywords:
            try:
                el = self._page.get_by_placeholder(re.compile(kw, re.IGNORECASE)).first
                if el.is_visible():
                    return el
            except Exception:
                pass

        # Strategy 3: Playwright get_by_label
        for kw in keywords:
            try:
                el = self._page.get_by_label(re.compile(kw, re.IGNORECASE)).first
                if el.is_visible():
                    return el
            except Exception:
                pass

        # Strategy 4: CSS selectors by id/name (case-sensitive but common IDs are lowercase)
        for kw in keywords:
            for selector in [f'input#{kw}', f'input[name="{kw}"]', f'textarea#{kw}']:
                try:
                    el = self._page.query_selector(selector)
                    if el and el.is_visible():
                        return el
                except Exception:
                    pass

        # Strategy 5: label for= association
        try:
            labels = self._page.query_selector_all("label")
            for label in labels:
                label_text = label.inner_text().lower()
                if any(word in label_text for word in keywords):
                    for_id = label.get_attribute("for")
                    if for_id:
                        el = self._page.query_selector(f"#{for_id}")
                        if el and el.is_visible():
                            return el
        except Exception:
            pass

        return None

    def _find_clickable(self, target: str):
        """Find a clickable element (button, link, etc.) by text content.

        Handles patterns like:
          "LEADS" → matches "LEADS (67)" via substring
          "Sign In button" → matches button with "Sign In" text
          "Update Lead" → matches button with exact text
          "first lead row in the table" → falls back to table row
          "Edit" → matches link/button containing "Edit"
        """
        # Clean target: strip quotes, "button", "link" suffixes
        clean = target.strip('"\'')
        clean = re.sub(r'\s+(button|link|tab|menu item)\s*$', '', clean, flags=re.IGNORECASE).strip()

        # Strategy 1: exact role match (button, link)
        for role in ["button", "link"]:
            try:
                el = self._page.get_by_role(role, name=re.compile(re.escape(clean), re.IGNORECASE)).first
                if el.is_visible():
                    return el
            except Exception:
                pass

        # Strategy 2: get_by_text (substring match)
        try:
            el = self._page.get_by_text(clean, exact=False).first
            if el.is_visible():
                return el
        except Exception:
            pass

        # Strategy 3: try each significant word individually (catches "LEADS" in "LEADS (67)")
        words = [w for w in clean.split() if len(w) > 2 and w.lower() not in (
            "the", "button", "link", "click", "on", "in", "for", "this", "first",
            "row", "table", "page", "field", "input", "menu", "navigation",
        )]
        for word in words:
            try:
                el = self._page.get_by_text(word, exact=False).first
                if el.is_visible():
                    return el
            except Exception:
                pass

        # Strategy 4: CSS href contains keyword (for navigation links)
        for word in words:
            try:
                el = self._page.query_selector(f'a[href*="{word.lower()}"]')
                if el and el.is_visible():
                    return el
            except Exception:
                pass

        # Strategy 5: table rows (for "first lead row" type targets)
        if any(w in target.lower() for w in ("row", "first", "item", "entry")):
            try:
                rows = self._page.query_selector_all("table tbody tr, [role='row'], .lead-row, .list-item")
                if rows:
                    for row in rows:
                        if row.is_visible():
                            return row
            except Exception:
                pass

        return None

    def _find_by_role(self, target: str):
        """Find element by ARIA role and name."""
        target_lower = target.lower()

        # Map target descriptions to ARIA roles
        role_hints = [
            ("button", ["button", "submit", "save", "update", "sign in", "log in", "export", "download"]),
            ("link", ["page", "tab", "menu", "nav", "leads", "dashboard", "home"]),
            ("combobox", ["dropdown", "select", "filter"]),
            ("checkbox", ["check", "enable", "toggle"]),
        ]

        for role, keywords in role_hints:
            if any(kw in target_lower for kw in keywords):
                try:
                    # Try to find by role with partial name
                    for kw in keywords:
                        if kw in target_lower:
                            el = self._page.get_by_role(role, name=re.compile(kw, re.IGNORECASE)).first
                            if el and el.is_visible():
                                return el
                except Exception:
                    pass

        return None
