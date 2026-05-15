"""IntentRewriter — Phase B of epic #377.

Tests the pure-function contract: which failures trigger a rewrite,
how the prompt is built, how Claude's response is parsed, and the
guard rails (KEEP / empty / API failure / no-budget paths)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from mantis_agent.gym.intent_rewriter import (
    REWRITE_TRIGGERING_CLASSES,
    FailureContext,
    propose_rewrite,
    should_attempt_rewrite,
)


# ── should_attempt_rewrite — gating predicate ────────────────────────────


def test_attempt_only_for_triggering_classes() -> None:
    for klass in REWRITE_TRIGGERING_CLASSES:
        assert should_attempt_rewrite(klass, attempts_used=0) is True
    for klass in ("selector_miss", "cf_challenge", "extractor_error",
                  "http_4xx", "unknown", ""):
        assert should_attempt_rewrite(klass, attempts_used=0) is False, klass


def test_attempt_budget_caps_rewrites_per_step() -> None:
    klass = "brain_loop_exhausted"
    assert should_attempt_rewrite(klass, attempts_used=0, max_attempts=1) is True
    assert should_attempt_rewrite(klass, attempts_used=1, max_attempts=1) is False
    assert should_attempt_rewrite(klass, attempts_used=2, max_attempts=3) is True


# ── propose_rewrite — empty / missing-key paths ──────────────────────────


def test_returns_none_when_intent_empty() -> None:
    assert propose_rewrite("", [FailureContext("brain_loop_exhausted", "d")], api_key="k") is None


def test_returns_none_when_no_failures() -> None:
    assert propose_rewrite("Click X", [], api_key="k") is None


def test_returns_none_when_no_api_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    failures = [FailureContext("brain_loop_exhausted", "loop")]
    assert propose_rewrite("Click X", failures, api_key="") is None


# ── propose_rewrite — Claude response parsing ────────────────────────────


def _stub_claude_response(text: str):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"content": [{"type": "text", "text": text}]}
    return resp


def test_returns_rewritten_intent_on_success() -> None:
    failures = [FailureContext(
        "brain_loop_exhausted",
        "scroll_handler: max_steps after 10 attempts",
    )]
    with patch("requests.post", return_value=_stub_claude_response("Scroll down by one viewport")):
        out = propose_rewrite(
            "Scroll down to reveal title, date, location, host details",
            failures, api_key="k",
        )
    assert out == "Scroll down by one viewport"


def test_returns_none_on_keep_response() -> None:
    failures = [FailureContext("wrong_target", "click_no_nav:wrong_target")]
    with patch("requests.post", return_value=_stub_claude_response("KEEP")):
        out = propose_rewrite("Click submit", failures, api_key="k")
    assert out is None


def test_returns_none_on_keep_response_lowercase() -> None:
    """Be tolerant of casing — Claude sometimes echoes ``keep``."""
    failures = [FailureContext("brain_loop_exhausted", "loop")]
    with patch("requests.post", return_value=_stub_claude_response("keep")):
        out = propose_rewrite("Click X", failures, api_key="k")
    assert out is None


def test_strips_surrounding_quotes() -> None:
    """Claude sometimes wraps the rewrite in quotes — strip them."""
    failures = [FailureContext("brain_loop_exhausted", "loop")]
    quoted = '"Scroll down by one viewport"'
    with patch("requests.post", return_value=_stub_claude_response(quoted)):
        out = propose_rewrite("Scroll to reveal X", failures, api_key="k")
    assert out == "Scroll down by one viewport"


def test_returns_none_on_empty_response() -> None:
    failures = [FailureContext("brain_loop_exhausted", "loop")]
    with patch("requests.post", return_value=_stub_claude_response("")):
        out = propose_rewrite("Click X", failures, api_key="k")
    assert out is None


def test_returns_none_on_api_error() -> None:
    failures = [FailureContext("brain_loop_exhausted", "loop")]
    bad = MagicMock()
    bad.status_code = 500
    bad.text = "internal error"
    with patch("requests.post", return_value=bad):
        out = propose_rewrite("Click X", failures, api_key="k")
    assert out is None


def test_returns_none_on_network_exception() -> None:
    failures = [FailureContext("brain_loop_exhausted", "loop")]
    with patch("requests.post", side_effect=ConnectionError("network down")):
        out = propose_rewrite("Click X", failures, api_key="k")
    assert out is None


# ── propose_rewrite — prompt construction ───────────────────────────────


def test_prompt_includes_intent_and_failure_summary() -> None:
    failures = [
        FailureContext(
            "wrong_target",
            "click_no_nav:wrong_target:landed on /tech",
            page_title="Tech — Luma",
            final_url="https://luma.com/tech",
        ),
    ]
    captured: dict = {}

    def _capture(*args, **kwargs):
        captured["body"] = kwargs.get("json")
        return _stub_claude_response("Click the first event card under Browse by Category")

    with patch("requests.post", side_effect=_capture):
        propose_rewrite("Click the first event card", failures, api_key="k")

    body = captured["body"]
    user_content = body["messages"][0]["content"]
    text_blocks = [b for b in user_content if b.get("type") == "text"]
    text = "\n".join(b["text"] for b in text_blocks)
    assert "Click the first event card" in text
    assert "wrong_target" in text
    assert "/tech" in text


def test_prompt_attaches_latest_screenshot_when_present() -> None:
    failures = [FailureContext(
        "wrong_target", "x", screenshot_png=b"\x89PNG\r\n\x1a\nfake",
    )]
    captured: dict = {}

    def _capture(*args, **kwargs):
        captured["body"] = kwargs.get("json")
        return _stub_claude_response("Different intent")

    with patch("requests.post", side_effect=_capture):
        propose_rewrite("Click X", failures, api_key="k")

    image_blocks = [
        b for b in captured["body"]["messages"][0]["content"]
        if b.get("type") == "image"
    ]
    assert len(image_blocks) == 1
    assert image_blocks[0]["source"]["media_type"] == "image/png"


# ── propose_rewrite — pre-step rewrite handoff (issue #428 Part B) ─────


def test_drops_pre_step_rewrite_first_fix() -> None:
    """When Claude's rewrite describes a different verb to be executed
    BEFORE this step (e.g. "First fix the X field" for a submit step
    that's actually being blocked by client-side form validation),
    ``propose_rewrite`` returns None so the next retry uses the
    original intent. The downstream ``agentic_recovery`` loop then
    picks ``insert_steps`` with a normalize-field pre-step (issue #428).
    Without this, the next retry would try to submit a button labelled
    "First fix the …" — nonsensical, wastes a retry, then halts.
    """
    failures = [FailureContext("no_state_change", "submit_no_state_change")]
    with patch(
        "requests.post",
        return_value=_stub_claude_response(
            "First fix the invalid Estimated Deal Value field (currently 461927.81)"
        ),
    ):
        out = propose_rewrite("Click Update Lead", failures, api_key="k")
    assert out is None


def test_drops_pre_step_rewrite_before_clicking() -> None:
    failures = [FailureContext("no_state_change", "submit")]
    with patch(
        "requests.post",
        return_value=_stub_claude_response(
            "Before clicking, set Estimated Deal Value to a whole number"
        ),
    ):
        out = propose_rewrite("Click Update Lead", failures, api_key="k")
    assert out is None


def test_drops_pre_step_rewrite_dismiss_modal() -> None:
    failures = [FailureContext("wrong_target", "blocked_by_modal")]
    with patch(
        "requests.post",
        return_value=_stub_claude_response("Dismiss the modal then click Save"),
    ):
        out = propose_rewrite("Click Save", failures, api_key="k")
    assert out is None


def test_same_step_rewrite_not_dropped_by_pre_step_filter() -> None:
    """Regression guard: the pre-step filter must NOT match well-formed
    same-step rewrites that happen to start with a common word.
    """
    failures = [FailureContext("brain_loop_exhausted", "loop")]
    with patch(
        "requests.post",
        return_value=_stub_claude_response("Scroll down by one viewport"),
    ):
        out = propose_rewrite("Scroll to reveal X", failures, api_key="k")
    assert out == "Scroll down by one viewport"


def test_looks_like_pre_step_rewrite_unit() -> None:
    """Direct unit check on the predicate — independent of the
    Claude-call path so a regression in the filter is caught locally.
    """
    from mantis_agent.gym.intent_rewriter import _looks_like_pre_step_rewrite

    # Pre-step shapes — all flagged.
    assert _looks_like_pre_step_rewrite("First fix the X field")
    assert _looks_like_pre_step_rewrite("First, fix the X field")
    assert _looks_like_pre_step_rewrite("Before clicking, set Y to Z")
    assert _looks_like_pre_step_rewrite("Dismiss the modal then click")
    assert _looks_like_pre_step_rewrite("first scroll down then click")  # casing
    # Same-step shapes — passed through.
    assert not _looks_like_pre_step_rewrite("Click the first event card")
    assert not _looks_like_pre_step_rewrite("Scroll down by one viewport")
    assert not _looks_like_pre_step_rewrite("Pick Space Exploration from the dropdown")


def test_prompt_documents_pre_step_keep_directive() -> None:
    """The rewriter prompt must explicitly forbid pre-step-shaped
    rewrites and instruct Claude to respond ``KEEP`` so agentic_recovery
    can handle the precondition (issue #428 Part B).
    """
    from mantis_agent.gym import intent_rewriter

    prompt = intent_rewriter._REWRITE_PROMPT
    assert "PRE-STEP TERRITORY" in prompt or "pre-step" in prompt.lower()
    assert "KEEP" in prompt
    # The validation-blocked-submit case is named explicitly so Claude
    # has language to pattern-match on.
    assert "no_state_change" in prompt
    assert "validation" in prompt.lower()


def test_prompt_omits_image_when_no_screenshot() -> None:
    failures = [FailureContext("brain_loop_exhausted", "loop")]
    captured: dict = {}

    def _capture(*args, **kwargs):
        captured["body"] = kwargs.get("json")
        return _stub_claude_response("Scroll once")

    with patch("requests.post", side_effect=_capture):
        propose_rewrite("Scroll", failures, api_key="k")

    image_blocks = [
        b for b in captured["body"]["messages"][0]["content"]
        if b.get("type") == "image"
    ]
    assert image_blocks == []
