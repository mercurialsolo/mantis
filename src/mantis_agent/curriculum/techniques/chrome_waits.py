"""Chrome waits: when and how long to pause for page loads and AJAX."""

NAME = "Chrome waits"
TAGS = [
    "chrome", "browser", "web", "wait", "load", "loading", "async",
    "ajax", "delay", "pause", "render", "page",
]
TRIGGERS = [
    r"\bload(ing)?\b", r"\bwait\b", r"\bdelay\b",
    r"page.*load", r"\bajax\b", r"\bspinner\b",
]
ALWAYS = False

CONTENT = """\
Chrome wait techniques (browser actions are async — patience prevents wasted steps):
- After ANY navigation (URL bar Enter, link click, form submit, back/forward), `wait(2)` then take a fresh screenshot before the next action. The DOM is rebuilding and clicking on stale elements is wasted steps.
- AJAX-heavy sites (Gmail, Facebook, modern SPAs) need longer waits — try `wait(3)` or `wait(5)` if the page looks the same after `wait(2)`.
- If you click and the screenshot doesn't change, the click probably missed OR the page is still loading. `wait(2)` and re-screenshot before retrying — don't immediately click again at the same coordinates.
- For tasks that need to detect a specific element appearing, take a screenshot, check if it's there, and if not, `wait(2)` and screenshot again. Don't poll faster than every 2 seconds.
- Search results, autocomplete suggestions, and infinite scroll all have ~500ms-1s delays. `wait(1)` is usually enough for these.
- Page-load spinners (the rotating wheel in the tab) indicate the page is still loading. Wait for them to disappear before reading the content area.
- DON'T spam screenshots in a tight loop — each screenshot consumes vision tokens and slows the run. One screenshot per `wait(2)` is the right cadence.\
"""
