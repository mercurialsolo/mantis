"""Failure classifier — pin the documented vocabulary so result.json
consumers (dashboards, post-mortems, the upcoming retry policy) can
rely on a stable set of class strings."""

from __future__ import annotations

from mantis_agent.gym.failure_class import classify, read_failure_context


# ── classify(): data prose + title rules ─────────────────────────────────


def test_unknown_when_both_inputs_empty() -> None:
    assert classify("", "") == "unknown"


def test_cf_challenge_from_title_overrides_empty_data() -> None:
    """Navigate-step halts often record an empty ``data`` but the page
    title still tells the story."""
    assert classify("", "Just a moment...") == "cf_challenge"
    assert classify("", "Verify you are human | example.com") == "cf_challenge"


def test_cf_challenge_from_403() -> None:
    assert classify("gate:FAIL:Error 403 forbidden", "") == "cf_challenge"


def test_http_4xx_404() -> None:
    assert classify("gate:FAIL:Error 404", "") == "http_4xx"


def test_http_5xx() -> None:
    assert classify("gate:FAIL:Error 503", "") == "http_5xx"
    assert classify("Internal Server Error", "") == "http_5xx"


def test_nav_timeout() -> None:
    assert classify("navigate timeout exceeded 45000ms", "") == "nav_timeout"
    assert classify("Page.goto timed out", "") == "nav_timeout"


def test_selector_miss() -> None:
    assert classify("fill_error: input not found", "") == "selector_miss"
    assert classify("submit_error: no element matches", "") == "selector_miss"
    assert classify("click_error", "") == "selector_miss"
    assert classify("filters_not_applied", "") == "selector_miss"


def test_extractor_error() -> None:
    assert classify("scan_error", "") == "extractor_error"
    assert classify("extract_error: empty payload", "") == "extractor_error"


def test_no_state_change_demotion_signal() -> None:
    """Epic #377 Phase A: classifier picks up the ``:no_state_change``
    suffix that ``_maybe_demote_*_no_change`` appends to ``data``.
    Used as the fallback for legacy result.json without a stamped
    ``failure_class``; the executor stamps the class directly on
    fresh runs."""
    assert classify("click_error:no_state_change", "") == "no_state_change"
    assert classify("submit_ok:no_state_change", "") == "no_state_change"


def test_brain_loop_exhausted_signal() -> None:
    """Epic #377 Phase A.2: classifier picks up GymRunner's
    termination reasons that indicate budget burn without success.
    Holo3StepHandler stamps the class directly on fresh runs; this
    rule is the fallback path for legacy / external result.json."""
    assert classify("brain_loop_exhausted", "") == "brain_loop_exhausted"
    assert classify("max_steps reached after 10 attempts", "") == "brain_loop_exhausted"
    assert classify("loop_terminated by detector", "") == "brain_loop_exhausted"


def test_wrong_target_signal() -> None:
    """Epic #377 follow-up: classifier picks up the click handler's
    ``wrong_target`` data prose. The handler stamps the class
    directly on fresh runs; this rule covers fallback / legacy
    result.json that only carries the ``data`` blob."""
    assert classify("click_no_nav:wrong_target:landed on category page", "") == "wrong_target"
    assert classify("verify_kind=wrong_target", "") == "wrong_target"


def test_budget_exceeded() -> None:
    assert classify("listing_budget_exceeded:bound=per_url", "") == "budget_exceeded"
    assert classify("max_cost reached", "") == "budget_exceeded"


def test_title_classification_beats_data_classification() -> None:
    """If the page is a CF interstitial but the handler wrote a
    selector-miss style ``data`` (because the element wasn't on the CF
    page), prefer ``cf_challenge`` — that's the root cause."""
    assert (
        classify("fill_error: not found", "Just a moment...")
        == "cf_challenge"
    )


def test_unknown_falls_through_unrelated_data() -> None:
    assert classify("something_weird_happened", "") == "unknown"


# ── read_failure_context(): env duck-typing ──────────────────────────────


class _PlaywrightishEnv:
    """Mimics ``PlaywrightGymEnv``: ``current_url`` property + ``_page``
    attribute with a callable ``title()``."""

    def __init__(self, url: str, title: str):
        self._url = url
        self._page = type("_Page", (), {"title": lambda self_: title})()

    @property
    def current_url(self) -> str:
        return self._url


class _XdotoolishEnv:
    """Mimics ``XdotoolGymEnv``: ``current_url`` + ``cdp_evaluate``."""

    def __init__(self, url: str, title: str):
        self._url = url
        self._title = title

    @property
    def current_url(self) -> str:
        return self._url

    def cdp_evaluate(self, expr: str):
        if "document.title" in expr:
            return self._title
        return None


def test_read_context_playwright_path() -> None:
    env = _PlaywrightishEnv("https://example.com/x", "Example Page")
    assert read_failure_context(env) == ("https://example.com/x", "Example Page")


def test_read_context_xdotool_cdp_path() -> None:
    env = _XdotoolishEnv("https://cf.example/y", "Just a moment...")
    assert read_failure_context(env) == ("https://cf.example/y", "Just a moment...")


def test_read_context_swallows_exceptions() -> None:
    """A broken env must never crash the failure path."""
    class _Broken:
        @property
        def current_url(self):
            raise RuntimeError("env is half-torn-down")

        def cdp_evaluate(self, _expr):
            raise RuntimeError("CDP unreachable")

    url, title = read_failure_context(_Broken())
    assert url == ""
    assert title == ""


def test_read_context_returns_empty_strings_when_unsupported() -> None:
    """A plain object exposes neither attribute → return empties."""
    url, title = read_failure_context(object())
    assert url == ""
    assert title == ""


# ── classifier vocabulary stays small ────────────────────────────────────


def test_classifier_vocabulary_is_stable() -> None:
    """Anyone widening the vocabulary should bump this set deliberately —
    breaks force a docs / dashboard update."""
    allowed = {
        "cf_challenge", "http_4xx", "http_5xx", "nav_timeout",
        "selector_miss", "no_state_change", "brain_loop_exhausted",
        "wrong_target",
        "extractor_error", "budget_exceeded", "unknown",
    }
    samples = [
        ("", ""),
        ("", "Just a moment..."),
        ("gate:FAIL:Error 403", ""),
        ("gate:FAIL:Error 404", ""),
        ("Error 503", ""),
        ("navigation timeout", ""),
        ("fill_error: not found", ""),
        ("scan_error", ""),
        ("max_cost exceeded", ""),
        ("???", ""),
    ]
    for data, title in samples:
        assert classify(data, title) in allowed
