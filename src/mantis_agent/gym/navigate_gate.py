"""Navigate-step URL-drift gate (#835).

When a ``navigate`` step has a literal URL in ``params.url``, the
runner stamps a derived ``expect_url_contains`` hint onto the next
step's pre-flight. Before the next step's vision call fires, the
runner compares the current page URL against the hint:

- Match → proceed.
- Mismatch → emit ``failure_class="navigation_drift"`` so step
  recovery can re-navigate instead of running the next step against
  the wrong page.

User feedback that surfaced this gap:

> HN navigation is not reliable. In one case it reached ``/newest``,
> but then drifted during the next step.

The hint is derived conservatively: hostname + first path segment.
``https://news.ycombinator.com/newest`` → ``news.ycombinator.com/newest``.
A bounce to ``news.ycombinator.com/login?next=...`` mismatches (extra
path); a soft fragment change ``/newest#top`` still matches.

CUA-purity: reading ``env.current_url`` to compare against an
expected value is *action-only* state verification (matches the
provenance of ``feedback_cua_cdp_post_action_verify.md``). We're not
deriving grounding from the DOM, just confirming the world looks the
way the navigate step claimed it would.

Gated by ``MANTIS_AUTO_URL_GATE`` (default ``"1"``); set to ``"0"`` to
disable system-wide.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse


def is_enabled() -> bool:
    """Whether the auto URL-drift gate is active.

    Default ``True``. Opt out with ``MANTIS_AUTO_URL_GATE=0``.
    Per-plan opt-out via ``step.expect_url_contains = False`` (the
    runner reads that on the next step).
    """
    raw = os.environ.get("MANTIS_AUTO_URL_GATE", "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    return True


def derive_expected_substring(url: str) -> str:
    """Return the substring the next step's URL should contain.

    Examples:
        ``https://news.ycombinator.com/newest`` →
            ``news.ycombinator.com/newest``
        ``https://github.com/owner/repo/issues`` →
            ``github.com/owner``
        ``https://example.com/`` → ``example.com``
        ``https://example.com`` → ``example.com``

    Returns the empty string when the URL is malformed or has no
    netloc — caller treats empty as "no gate".
    """
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
    except Exception:  # noqa: BLE001 — never propagate
        return ""
    host = (parsed.netloc or "").lower()
    if not host:
        return ""
    path = parsed.path or ""
    # Strip leading slash + trailing slash before splitting so a bare
    # ``/`` doesn't produce an empty segment.
    stripped = path.strip("/")
    if not stripped:
        return host
    first_segment = stripped.split("/", 1)[0]
    if not first_segment:
        return host
    return f"{host}/{first_segment}"


def check_drift(actual_url: str, expected_substring: str) -> str:
    """Compare ``actual_url`` against ``expected_substring``.

    Returns the empty string when there's no drift (match OR when
    the gate has nothing to compare against). Returns a non-empty
    ``expected=<exp>|got=<actual>`` drift reason when ``actual_url``
    doesn't contain ``expected_substring``.

    Case-insensitive on the URL but expects ``expected_substring``
    already lowercased by ``derive_expected_substring``.
    """
    if not expected_substring or not actual_url:
        # Empty expectation → nothing to gate on. Empty actual → the
        # env hasn't reported a URL yet; don't fail on absence.
        return ""
    if expected_substring in actual_url.lower():
        return ""
    return f"expected={expected_substring}|got={actual_url[:120]}"


__all__ = [
    "check_drift",
    "derive_expected_substring",
    "is_enabled",
]
