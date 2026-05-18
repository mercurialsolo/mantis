"""Lightweight failure classifier for StepResults.

When a step fails, MicroPlanRunner records a short prose blob in
``StepResult.data`` (set by individual step handlers — ``gate:FAIL:...``,
``fill_error:...``, ``scan_error``, etc.). That string is the only
post-mortem signal that survives the Modal container teardown. This
module maps that prose + the page title at failure time into a small
fixed vocabulary of classes so result.json consumers can branch on
``failure_class`` instead of regex-ing prose.

Classes (stay small on purpose — anything not matched lands in
``unknown``):

* ``cf_challenge`` — Cloudflare / anti-bot interstitial. Recognized by
  HTTP 403 + the canonical CF interstitial titles.
* ``nav_timeout`` — page-load timeout from Playwright / CDP.
* ``http_4xx`` / ``http_5xx`` — origin returned an error status.
* ``selector_miss`` — click / fill / submit couldn't locate the target.
* ``no_state_change`` — action reported success but the runner-state
  snapshot saw no URL / page / scroll / focus change. Self-healing
  demotion signal (epic #377 Phase A) for click / submit /
  navigate_back where the handler over-reported success.
* ``brain_loop_exhausted`` — the inner GymRunner exited at the step
  budget OR with a loop-detector trip, without success. Surfaces
  steps whose intent is goal-shaped instead of mechanical (epic
  #377 Phase A.2). The next attempt should route through an intent
  rewriter (Phase B) rather than retrying the same intent.
* ``wrong_target`` — the SPA-aware ``verify_post_click_navigation``
  decided the click landed on the wrong destination (category card
  instead of an event detail, login wall, ad, …). Distinct from
  ``no_state_change`` (the click had no effect at all) — here the
  click DID navigate, but to the wrong page. Intent rewriting
  (Phase B) is the right response.
* ``extractor_error`` — Claude extractor failed or returned empty.
* ``budget_exceeded`` — cost / time / context budget tripped.
* ``unknown`` — no rule matched (caller should still surface ``data``).

The classifier is pure-functional and runs in microseconds — it lives
on the failure path so it can't pull in any heavy imports.
"""

from __future__ import annotations


_CF_TITLE_MARKERS: tuple[str, ...] = (
    "just a moment",
    "performing security verification",
    "checking your browser",
    "verifying you are human",
    "verify you are human",
    "attention required",
)


_DATA_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Order matters: more specific rules first. The demotion suffix
    # ``:no_state_change`` (epic #377) is appended to a step's existing
    # ``data`` when the runner downgrades success → fail — keep this
    # rule above ``selector_miss`` so its substring wins when the
    # pre-demotion ``data`` happened to contain a generic handler
    # marker.
    ("no_state_change", ("no_state_change",)),
    # Inner GymRunner ran to step-budget without success (epic #377
    # Phase A.2). The executor stamps this class directly on the
    # StepResult; this rule covers any fallback path that has only
    # ``data`` to read.
    ("brain_loop_exhausted", (
        "brain_loop_exhausted", "max_steps", "loop_terminated",
    )),
    # Click landed but on the wrong destination (epic #377 follow-up).
    # Click handler stamps this directly when
    # ``verify_post_click_navigation`` returns ``kind=wrong_target``;
    # rule below catches the substring on legacy result.json.
    ("wrong_target", ("wrong_target",)),
    ("cf_challenge", ("error 403", "403 forbidden", "cloudflare", "verify you are human")),
    ("http_4xx", ("error 404", "404", "error 401", "error 410", "error 4")),
    ("http_5xx", ("error 5", "502", "503", "504", "internal server error")),
    ("nav_timeout", ("timeout", "timed out", "navigation timeout")),
    ("selector_miss", (
        "fill_error", "submit_error", "click_error", "select_error",
        "not found", "no element", "element not visible",
        "filters_not_applied",
        # Holo3 / SoM grounding signals — when the brain returned
        # coordinates but the runner's SoM verification said the click
        # landed on the wrong element ("ok=False"), or when the click
        # dispatch couldn't pin a grounded target ("grounding=NO"), or
        # claude-director re-routed coordinates mid-attempt because the
        # original ones missed ("director: substituting"). All three
        # are evidence that the vision pipeline misidentified the
        # target — same family as a CSS selector miss.
        "som-click", "ok=false", "grounding=no",
        "director: substituting",
    )),
    ("extractor_error", ("scan_error", "extract_error", "extractor", "scrape")),
    ("budget_exceeded", (
        "budget_exceeded", "max_cost", "max_time", "listing_budget_exceeded",
    )),
)


def classify(data: str, page_title: str = "") -> str:
    """Return one of the documented class strings, or ``"unknown"``.

    Both inputs are lowercased and matched against substring rules. An
    empty ``data`` with a CF title still classifies as ``cf_challenge``
    so navigate-step halts that didn't write a verbose ``data`` still
    surface the right diagnosis.
    """
    title_lc = (page_title or "").lower()
    if any(m in title_lc for m in _CF_TITLE_MARKERS):
        return "cf_challenge"

    data_lc = (data or "").lower()
    if not data_lc:
        return "unknown"

    for klass, markers in _DATA_RULES:
        if any(m in data_lc for m in markers):
            return klass
    return "unknown"


def read_failure_context(env) -> tuple[str, str]:
    """Best-effort ``(url, title)`` lookup against any GymEnvironment.

    Tries Playwright's ``page`` first (most common in local CLI runs),
    then falls back to xdotool's CDP path. Returns empty strings on any
    failure — the caller stamps whatever it gets onto the StepResult so
    a missing title is still better than no diagnosis.
    """
    url = ""
    title = ""

    try:
        url = getattr(env, "current_url", "") or ""
    except Exception:
        url = ""

    page = getattr(env, "_page", None) or getattr(env, "page", None)
    if page is not None and not callable(page):
        try:
            title = page.title() or ""
        except Exception:
            title = ""

    if not title:
        cdp_eval = getattr(env, "cdp_evaluate", None)
        if callable(cdp_eval):
            try:
                t = cdp_eval("document.title || ''")
                if isinstance(t, str):
                    title = t
            except Exception:
                title = ""

    return url, title
