"""URL health classifier — Phase 0 of #703 plan-evolution.

Pure-functional classifier that maps `(current_url, expected_url,
expected_domains, page_title)` into a small fixed vocabulary so the
recovery loop sees a clean signal instead of inferring URL drift from
pixels.

The classifier is intentionally narrow — Phase 0 is observability only.
Phase 1 (#705) adds the matching `rewrite_url` recovery action that
consumes these subclasses; Phase 2 (#706) persists what works.

Subclasses (small on purpose — anything not matched is `ok`):

* `ok` — `current_url` is in `expected_domains` and isn't a known
  error template. The default; emitted when nothing went wrong.
* `dns` — Chrome's own error page. `current_url` starts with
  `chrome-error://` or is empty after a navigate. Means the request
  never reached an origin server (bad domain, no network).
* `wrong_domain` — navigation completed but landed on a domain that
  isn't in the plan's expected set. Catches redirects to login walls
  on different domains, account-suspension redirects, accidental
  cross-site redirects.
* `not_found` — origin returned a 404-shaped page. Detected via page
  title heuristics (no Chrome-side HTTP status is available without
  CDP `Network.responseReceived`; the heuristic catches the common
  cases without that plumbing).
* `soft_404` — 200 OK with `current_url` matching the expected
  domain, but page content matches "this page no longer exists" /
  "page not found" / "couldn't find what you were looking for"
  patterns. Harder than 404 because the URL looks fine.
* `blocked` — Cloudflare / WAF interstitial. Detected by the same
  title markers as `failure_class.cf_challenge` so the two paths
  don't fight. Phase 0 reports it but defers to the existing CF /
  external-pause handling; the rewrite layer doesn't try to "rewrite"
  around a CF block.

The classifier runs on the failure path so it can't pull heavy imports.
Pure Python, no I/O, microseconds.
"""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

UrlHealthSubclass = Literal[
    "ok",
    "dns",
    "not_found",
    "wrong_domain",
    "soft_404",
    "blocked",
]


# Same set as failure_class._CF_TITLE_MARKERS — kept in sync so
# url_health.classify and failure_class.classify agree about CF.
_BLOCKED_TITLE_MARKERS: tuple[str, ...] = (
    "just a moment",
    "performing security verification",
    "checking your browser",
    "verifying you are human",
    "verify you are human",
    "attention required",
)

# Title patterns that indicate the origin returned a "page not found"
# template. Lowercased substring match. Conservative — we'd rather
# under-classify than mis-classify a real page as 404.
_NOT_FOUND_TITLE_MARKERS: tuple[str, ...] = (
    "404",
    "page not found",
    "not found",
    "page doesn't exist",
    "page does not exist",
    "page no longer exists",
    "this page isn't available",
    "this page isn't here",
)

# Title patterns that indicate a soft-404: the URL resolved (200 OK),
# stayed on the expected domain, but the page content says "we don't
# have what you asked for". Some sites use this instead of a real 404
# for missing resources.
_SOFT_404_BODY_MARKERS: tuple[str, ...] = (
    "we couldn't find",
    "we could not find",
    "no results found",
    "nothing matches",
    "page is no longer available",
    "this listing is no longer available",
    "this item is no longer available",
)


def _normalize_netloc(netloc: str) -> str:
    """Strip leading `www.`, lowercase, drop port. Comparison-friendly."""
    if not netloc:
        return ""
    nl = netloc.lower().split(":", 1)[0]
    if nl.startswith("www."):
        nl = nl[4:]
    return nl


def expand_expected_domains(start_url: str) -> set[str]:
    """Derive the expected-domain set from a plan's start_url.

    Returns the normalised netloc + common subdomain variants the plan
    is likely to land on (e.g. `www.` ↔ bare, `m.` mobile). Callers can
    extend this with site-specific aliases via Phase 1's SiteConfig.
    """
    parsed = urlparse(start_url)
    netloc = _normalize_netloc(parsed.netloc)
    if not netloc:
        return set()
    return {netloc, f"www.{netloc}", f"m.{netloc}"}


def classify(
    *,
    current_url: str,
    expected_domains: set[str],
    page_title: str = "",
    page_body_text: str = "",
) -> UrlHealthSubclass:
    """Map browser state → URL-health subclass.

    Args:
        current_url: `env.current_url` after a navigate step.
        expected_domains: set of normalised netlocs the plan considers
            valid landings. Use :func:`expand_expected_domains` to
            derive from a single start_url, or pass an explicit set
            for multi-domain plans.
        page_title: `document.title` at failure time. Lowercased for
            matching.
        page_body_text: A short slice of visible body text for soft-404
            detection. Optional — many callers don't have it and the
            classifier degrades gracefully.

    Returns:
        One of `UrlHealthSubclass`. `ok` means no URL-health problem
        was detected; the caller may still find other failures via
        :func:`failure_class.classify`.
    """
    # DNS / Chrome-error / empty URL — the navigate didn't reach an
    # origin. Highest-priority signal.
    if not current_url or current_url.startswith("chrome-error://"):
        return "dns"

    title_lc = (page_title or "").lower()

    # Blocked (CF / WAF) takes precedence over wrong_domain because CF
    # serves its interstitial under the requested domain — we shouldn't
    # claim "wrong_domain" when the request actually reached the right
    # origin but got intercepted.
    if any(m in title_lc for m in _BLOCKED_TITLE_MARKERS):
        return "blocked"

    parsed = urlparse(current_url)
    current_netloc = _normalize_netloc(parsed.netloc)
    expected_norm = {_normalize_netloc(d) for d in expected_domains if d}

    # Wrong-domain: navigated outside the expected set. Skips the
    # not_found / soft_404 checks because those are domain-relative.
    if expected_norm and current_netloc and current_netloc not in expected_norm:
        return "wrong_domain"

    # 404-shaped title — the page itself self-identifies as missing.
    if any(m in title_lc for m in _NOT_FOUND_TITLE_MARKERS):
        return "not_found"

    # Soft-404 — title looks fine but body says "we don't have this".
    body_lc = (page_body_text or "").lower()
    if body_lc and any(m in body_lc for m in _SOFT_404_BODY_MARKERS):
        return "soft_404"

    return "ok"


def read_page_signals(env) -> tuple[str, str, str]:
    """Best-effort `(current_url, title, body_snippet)` for the classifier.

    Mirrors :func:`failure_class.read_failure_context` but also pulls a
    short body slice for soft-404 detection. Returns empty strings on
    any failure — the classifier degrades gracefully.

    The body snippet is the first 2 KB of visible text — enough for
    soft-404 marker matching without blowing prompt budgets if anything
    downstream logs it.
    """
    url = ""
    title = ""
    body = ""

    try:
        raw_url = getattr(env, "current_url", "") or ""
    except Exception:  # noqa: BLE001 — best-effort
        raw_url = ""
    # Defensive cast — test mocks return MagicMock for unconfigured
    # attributes. Treat anything non-string as "we don't know" rather
    # than coercing into a false-positive bad_url classification.
    url = raw_url if isinstance(raw_url, str) else ""

    cdp_eval = getattr(env, "cdp_evaluate", None)
    if callable(cdp_eval):
        try:
            t = cdp_eval("document.title || ''")
            if isinstance(t, str):
                title = t
        except Exception:  # noqa: BLE001
            title = ""
        try:
            b = cdp_eval(
                "(document.body && document.body.innerText || '').slice(0, 2048)"
            )
            if isinstance(b, str):
                body = b
        except Exception:  # noqa: BLE001
            body = ""

    # Fallback: Playwright `page.title()` for CLI runs that don't have
    # a `cdp_evaluate` method.
    if not title:
        page = getattr(env, "_page", None) or getattr(env, "page", None)
        if page is not None and not callable(page):
            try:
                title = page.title() or ""
            except Exception:  # noqa: BLE001
                title = ""

    return url, title, body
