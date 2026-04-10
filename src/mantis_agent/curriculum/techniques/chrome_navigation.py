"""Chrome navigation: address bar, tabs, history, find-on-page."""

NAME = "Chrome navigation"
TAGS = [
    "chrome", "browser", "url", "address", "tab", "tabs", "navigate",
    "back", "forward", "reload", "refresh", "find", "search", "page",
    "website", "site", "history", "bookmark",
]
TRIGGERS = [
    r"\bchrome\b", r"\bbrowser\b", r"\bbookmark\b", r"address bar",
    r"\bnew tab\b", r"\bclose tab\b", r"\btabs?\b",
    r"navigate to", r"go to.*\.com", r"\burl\b",
    r"find on page", r"search.*page",
]
ALWAYS = False

CONTENT = """\
Chrome navigation techniques:
- Focus the address bar with `key('ctrl+l')` then `type_text('https://example.com')` then `key('Return')`. Always check the resulting screenshot before next action — page loads are async.
- Open new tab: `key('ctrl+t')`. Close current tab: `key('ctrl+w')`. Cycle tabs: `key('ctrl+Tab')` (forward) or `key('ctrl+shift+Tab')` (back). Jump to tab N: `key('ctrl+1')` ... `key('ctrl+9')`.
- Back/forward in history: `key('alt+Left')` and `key('alt+Right')`.
- Reload page: `key('ctrl+r')`. Hard reload (bypass cache): `key('ctrl+shift+r')`.
- Find text on the current page: `key('ctrl+f')` opens the find toolbar, then `type_text('search term')` and `key('Return')` to jump to the next match.
- View page source: `key('ctrl+u')`.
- Zoom in/out for better readability: `key('ctrl+plus')` / `key('ctrl+minus')`. Reset: `key('ctrl+0')`.
- After ANY navigation action, `wait(2)` then take a fresh screenshot — pages do not load instantly and clicking on stale UI is wasted steps.\
"""
