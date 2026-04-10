"""Chrome visual grounding: how to read web page screenshots."""

NAME = "Chrome visual grounding"
TAGS = [
    "chrome", "browser", "web", "click", "button", "link", "menu",
    "navigation", "layout", "screenshot", "visible", "element", "ui",
    "icon", "image", "scroll", "viewport",
]
TRIGGERS = [
    r"\bclick\b.*\b(button|link|menu|icon)\b",
    r"\bscroll\b", r"hamburger menu", r"\bsidebar\b",
    r"\bdropdown\b", r"\bbutton labeled\b",
    r"\bnav(igation)?\s*bar\b",
]
ALWAYS = False

CONTENT = """\
Chrome visual grounding techniques (how to read a web page screenshot):
- Layout convention: address bar at the very top, tabs above it, content area in the middle, browser status/dev tools at the bottom (if open).
- Common UI patterns to look for:
  - Hamburger menu: usually 3 horizontal lines, top-left or top-right corner
  - Search box: top-right or hero center, often has a magnifying-glass icon
  - Navigation bar: horizontal links across the top, just under the header
  - Sidebar: vertical column on left or right
  - Pagination: bottom of content area, ""Next"" / ""Previous"" / page numbers
  - Dropdown / hover menu: a small triangle or arrow next to the label
- Click on TEXT LABELS rather than icons when both are present — labels have larger hit areas and are less ambiguous.
- For ambiguous clickable regions, prefer the visual center of the element. Avoid clicking near edges or corners where you might miss.
- Pages often have invisible padding around clickable elements — if your click misses, try ±20 pixels away from the visible center, NOT closer to the text.
- Long pages need scrolling. Use `scroll(x, y, clicks=-3)` to scroll down 3 lines, or `key('Page_Down')` for a full screen. After scrolling, take a fresh screenshot before clicking.
- Modal dialogs (alerts, popups, cookie banners) often steal click focus — close them first via the X button or `key('Escape')` before continuing the actual task.\
"""
