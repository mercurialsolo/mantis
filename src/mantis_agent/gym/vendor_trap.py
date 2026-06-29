"""Recognise browser-vendor / update pages — the S01 "Reinstall Chrome" trap.

In the cua-issues 2026-06-29 audit, run **S01** dead-ended like this:

1. Chrome rendered its **"Can't update Chrome → Reinstall Chrome"** bubble
   (a vendor nag that appears on an outdated build with no auto-update
   infra — common in containers).
2. Loop-recovery pressed **Return** on a frozen frame; the keystroke
   activated the bubble's *Reinstall Chrome* button.
3. That navigated to **google.com/chrome**, which immediately failed with
   ``ERR_TUNNEL_CONNECTION_FAILED`` behind the proxy.
4. The run looped to ``max_steps`` and (pre-#940) reported ``succeeded`` —
   while having done the one thing the task prompt forbade.

A browser-vendor download/update page is **never** a legitimate task
destination, so reaching one is always a trap to abort, not progress. This
module is the high-precision predicate used by the env (refuse a navigate
*to* such a URL) and the runner (halt honestly if a stray keystroke lands
*on* one).

Kept deliberately narrow — only the Chrome download / reinstall surfaces
that S01 hit — so a real task that happens to touch ``google.com`` (search,
maps, mail) is never mis-flagged.
"""

from __future__ import annotations

from urllib.parse import urlparse

# (host, path-prefix) pairs that are vendor download/update destinations.
# An empty path-prefix means the whole host is a vendor surface.
_VENDOR_RULES: tuple[tuple[str, str], ...] = (
    ("google.com", "/chrome"),       # https://www.google.com/chrome (the S01 dead-end)
    ("www.google.com", "/chrome"),
    ("chrome.google.com", ""),       # vendor host in its entirety
)


def is_browser_vendor_url(url: str | None) -> bool:
    """True when *url* is a browser-vendor download / update page.

    >>> is_browser_vendor_url("https://www.google.com/chrome/")
    True
    >>> is_browser_vendor_url("https://chrome.google.com/")
    True
    >>> is_browser_vendor_url("https://www.google.com/search?q=chrome")
    False
    >>> is_browser_vendor_url("https://www.linkedin.com/feed/")
    False
    """
    if not url:
        return False
    try:
        parsed = urlparse(url.strip())
    except (ValueError, TypeError):
        return False

    host = (parsed.netloc or "").lower().split("@")[-1].split(":")[0]
    path = (parsed.path or "").lower()
    if not host:
        return False

    for vendor_host, path_prefix in _VENDOR_RULES:
        if host == vendor_host and path.startswith(path_prefix):
            return True
    return False
