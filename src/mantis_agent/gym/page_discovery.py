"""Page discovery — scan DOM and let the brain choose which element to interact with.

This is the bridge between the brain's reasoning and the executor's reliable
action. Instead of the brain guessing pixel coordinates from screenshots
(unreliable), or the executor pattern-matching natural language targets
(brittle), we:

1. Scan the DOM for all interactive elements (inputs, buttons, links, etc.)
2. Format them as a numbered list with text/type/placeholder
3. Ask the brain: "which element [N] should I interact with for this step?"
4. Execute the action on that specific element via its DOM reference

This is the SoM (Set-of-Mark) approach proven in VWA, adapted for plan execution.
The brain does what it's good at (reasoning about which element matches the intent),
the executor does what it's good at (reliably clicking/typing on DOM elements).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# JavaScript to discover all interactive elements on the page
DISCOVER_ELEMENTS_JS = """() => {
    const selectors = [
        'a[href]', 'button', 'input', 'select', 'textarea',
        '[role="button"]', '[role="link"]', '[role="menuitem"]',
        '[role="tab"]', '[role="checkbox"]', '[role="radio"]',
        '[role="option"]', '[role="switch"]', '[role="combobox"]',
        '[onclick]', '[tabindex]',
        'label', 'summary',
    ];
    const seen = new Set();
    const elements = [];
    for (const sel of selectors) {
        for (const el of document.querySelectorAll(sel)) {
            if (seen.has(el)) continue;
            seen.add(el);
            const rect = el.getBoundingClientRect();
            if (rect.width < 5 || rect.height < 5) continue;
            if (rect.top > window.innerHeight || rect.left > window.innerWidth) continue;
            if (rect.bottom < 0 || rect.right < 0) continue;
            const text = (el.textContent || '').trim().substring(0, 80);
            const value = el.value || '';
            const placeholder = el.placeholder || el.getAttribute('placeholder') || '';
            const ariaLabel = el.getAttribute('aria-label') || '';
            const name = el.getAttribute('name') || '';
            const id = el.id || '';
            const href = el.getAttribute('href') || '';
            elements.push({
                tag: el.tagName.toLowerCase(),
                text: text.replace(/\\s+/g, ' '),
                type: el.getAttribute('type') || '',
                role: el.getAttribute('role') || '',
                name: name,
                id: id,
                placeholder: placeholder,
                ariaLabel: ariaLabel,
                value: value.substring(0, 50),
                href: href.substring(0, 100),
                bbox: {
                    x: Math.round(rect.x),
                    y: Math.round(rect.y),
                    w: Math.round(rect.width),
                    h: Math.round(rect.height),
                },
            });
        }
    }
    return elements;
}"""


@dataclass
class PageElement:
    """A discovered interactive element on the page."""
    index: int
    tag: str
    text: str
    type: str = ""
    role: str = ""
    name: str = ""
    id: str = ""
    placeholder: str = ""
    aria_label: str = ""
    value: str = ""
    href: str = ""
    bbox: dict = field(default_factory=dict)

    def describe(self) -> str:
        """Human-readable one-line description for the brain."""
        parts = [f"[{self.index}]"]

        # Element type
        if self.tag == "input":
            input_type = self.type or "text"
            parts.append(f"<input type={input_type}>")
        elif self.tag == "button":
            parts.append("<button>")
        elif self.tag == "a":
            parts.append("<link>")
        elif self.tag == "select":
            parts.append("<dropdown>")
        elif self.tag == "textarea":
            parts.append("<textarea>")
        else:
            parts.append(f"<{self.tag}>")

        # Text/content
        if self.text and len(self.text) > 1:
            parts.append(f'"{self.text[:60]}"')
        if self.placeholder:
            parts.append(f'placeholder="{self.placeholder}"')
        if self.aria_label and self.aria_label != self.text:
            parts.append(f'label="{self.aria_label}"')
        if self.name:
            parts.append(f'name="{self.name}"')
        if self.id:
            parts.append(f'id="{self.id}"')
        if self.value:
            parts.append(f'value="{self.value}"')
        if self.href and self.tag == "a":
            parts.append(f'href="{self.href[:60]}"')

        return " ".join(parts)


class PageDiscovery:
    """Discover interactive elements on a page and let the brain choose.

    Args:
        page: Playwright Page object (or env with .page property).
        max_elements: Maximum elements to include in the brain prompt.
    """

    def __init__(self, page=None, env=None, max_elements: int = 50):
        self._page_ref = page
        self._env = env
        self._max_elements = max_elements
        self._last_elements: list[PageElement] = []

    @property
    def _page(self):
        if self._page_ref is not None:
            return self._page_ref
        if self._env and hasattr(self._env, "page"):
            return self._env.page
        return None

    def discover(self) -> list[PageElement]:
        """Scan the current page and return all interactive elements."""
        if self._page is None:
            return []

        try:
            raw_elements = self._page.evaluate(DISCOVER_ELEMENTS_JS)
        except Exception as e:
            logger.warning(f"Element discovery failed: {e}")
            return []

        elements = []
        for i, raw in enumerate(raw_elements[:self._max_elements]):
            elements.append(PageElement(
                index=i,
                tag=raw.get("tag", ""),
                text=raw.get("text", ""),
                type=raw.get("type", ""),
                role=raw.get("role", ""),
                name=raw.get("name", ""),
                id=raw.get("id", ""),
                placeholder=raw.get("placeholder", ""),
                aria_label=raw.get("ariaLabel", ""),
                value=raw.get("value", ""),
                href=raw.get("href", ""),
                bbox=raw.get("bbox", {}),
            ))

        self._last_elements = elements
        return elements

    def format_for_brain(self, elements: list[PageElement] | None = None) -> str:
        """Format discovered elements as a numbered list for the brain prompt.

        Returns a text block like:
            INTERACTIVE ELEMENTS ON PAGE:
            [0] <link> "HOME" href="/dashboard"
            [1] <link> "LEADS (67)" href="/leads"
            [2] <input type=text> placeholder="Search..." name="search"
            [3] <button> "Export CSV"
            ...
        """
        if elements is None:
            elements = self._last_elements

        if not elements:
            return "No interactive elements found on page."

        lines = ["INTERACTIVE ELEMENTS ON PAGE:"]
        for el in elements:
            lines.append(f"  {el.describe()}")

        if len(elements) >= self._max_elements:
            lines.append(f"  ... (showing first {self._max_elements})")

        return "\n".join(lines)

    def build_choice_prompt(
        self,
        task_step: str,
        elements: list[PageElement] | None = None,
        context: str = "",
    ) -> str:
        """Build a prompt asking the brain to choose an element.

        Args:
            task_step: What the agent needs to do (e.g., "Click on the Leads page")
            elements: Elements to choose from (uses last discovered if None)
            context: Additional context (what was done so far, etc.)

        Returns:
            Prompt string for the brain.
        """
        el_text = self.format_for_brain(elements)

        prompt = f"""You need to perform this action: {task_step}

{el_text}

{context}

Which element number [N] should I interact with to accomplish this action?
Reply with ONLY the element number. For example: 1

If the action requires typing text, reply with the element number followed by the text.
For example: 2 alice

If no element matches, reply: NONE"""
        return prompt

    def get_element_by_index(self, index: int) -> PageElement | None:
        """Get a previously discovered element by its index."""
        if 0 <= index < len(self._last_elements):
            return self._last_elements[index]
        return None

    def click_element(self, index: int) -> bool:
        """Click an element by its discovery index using Playwright."""
        el = self.get_element_by_index(index)
        if not el or not self._page:
            return False

        try:
            # Use bbox center for clicking (most reliable)
            cx = el.bbox.get("x", 0) + el.bbox.get("w", 0) // 2
            cy = el.bbox.get("y", 0) + el.bbox.get("h", 0) // 2
            self._page.mouse.click(cx, cy)
            return True
        except Exception as e:
            logger.warning(f"Click element [{index}] failed: {e}")
            return False

    def type_into_element(self, index: int, text: str) -> bool:
        """Click an element and type text into it."""
        el = self.get_element_by_index(index)
        if not el or not self._page:
            return False

        try:
            cx = el.bbox.get("x", 0) + el.bbox.get("w", 0) // 2
            cy = el.bbox.get("y", 0) + el.bbox.get("h", 0) // 2
            self._page.mouse.click(cx, cy)

            import time
            time.sleep(0.3)

            # Clear existing value and type new one
            self._page.keyboard.press("Control+a")
            self._page.keyboard.type(text)
            return True
        except Exception as e:
            logger.warning(f"Type into element [{index}] failed: {e}")
            return False

    def select_option(self, index: int, value: str) -> bool:
        """Select an option in a dropdown element."""
        el = self.get_element_by_index(index)
        if not el or not self._page:
            return False

        try:
            if el.tag == "select":
                selector = "select"
                if el.id:
                    selector = f"#{el.id}"
                elif el.name:
                    selector = f'select[name="{el.name}"]'
                self._page.select_option(selector, label=value)
                return True
            else:
                # Click to open dropdown, then type/click option
                self.click_element(index)
                import time
                time.sleep(0.5)
                self._page.keyboard.type(value)
                return True
        except Exception as e:
            logger.warning(f"Select option [{index}] failed: {e}")
            return False


def parse_brain_choice(response: str) -> tuple[int | None, str | None]:
    """Parse the brain's element choice response.

    Handles:
        "3"          → (3, None)
        "2 alice"  → (2, "alice")
        "NONE"       → (None, None)

    Returns:
        Tuple of (element_index, optional_text).
    """
    text = response.strip()

    if text.upper() == "NONE":
        return None, None

    # Extract number and optional text
    match = re.match(r"^\[?(\d+)\]?\s*(.*)", text)
    if match:
        idx = int(match.group(1))
        extra = match.group(2).strip() or None
        return idx, extra

    return None, None
