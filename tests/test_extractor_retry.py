"""Retry-with-backoff coverage for ClaudeExtractor's Anthropic POST helper (#403).

The lu.ma host-question registration plan halted at step 4 because
Anthropic returned 529 Overloaded on every find_form_target call —
``ClaudeExtractor._call_with_tool_schema`` previously returned ``None``
immediately on any non-200, so each scroll-probe + affordance fallback
inside ``fill_field`` burned its slot on what was a transient API
hiccup. The step then failed, agentic recovery fired twice more, the
run halted.

This module pins:

- transient HTTP errors (429 / 502 / 503 / 504 / 529) trigger retry
  with exponential backoff;
- ``Retry-After`` header is honoured when the response carries one;
- non-transient errors (4xx other than 429) return immediately without
  retry — no point burning the budget on a permanent 400 / 401;
- network exceptions retry up to the budget then return ``None``;
- ``time.sleep`` is monkeypatched so the suite stays fast (~10 ms per
  test) regardless of backoff math.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mantis_agent.extraction.extractor import (
    _TRANSIENT_STATUS_CODES,
    ClaudeExtractor,
    _retry_delay,
)


def _fake_response(status_code: int, headers: dict | None = None) -> MagicMock:
    """A requests.Response-ish stand-in."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.text = f"<body status={status_code}>"
    return resp


# ── _retry_delay ────────────────────────────────────────────────────


def test_retry_delay_honours_numeric_retry_after() -> None:
    """Server-supplied Retry-After (seconds) overrides exponential backoff."""
    assert _retry_delay(0, "3") == 3.0
    assert _retry_delay(5, "0.5") == 0.5
    # Empty string is falsy → fall through to backoff path.
    assert _retry_delay(0, "") >= 1.0


def test_retry_delay_ignores_non_numeric_retry_after() -> None:
    """HTTP-date Retry-After (RFC 7231 §7.1.3) is unsupported; fall back
    to the backoff path so we still wait something reasonable."""
    delay = _retry_delay(0, "Mon, 12 Jul 2026 12:00:00 GMT")
    assert delay >= 1.0  # exp(0) = 1 + jitter


def test_retry_delay_exponential_backoff_with_jitter() -> None:
    """Sequence: 1s, 2s, 4s, 8s (capped at 16s) plus up to 25% jitter."""
    for attempt, base in enumerate([1, 2, 4, 8]):
        d = _retry_delay(attempt, None)
        assert base <= d <= base * 1.25, f"attempt={attempt} got {d}"


def test_retry_delay_caps_at_16s() -> None:
    """Past attempt=4, the base stays at 16s — keep one retry from
    blowing past a half-minute budget."""
    d = _retry_delay(10, None)
    assert 16.0 <= d <= 20.0  # 16 + up to 25%


def test_retry_delay_negative_retry_after_clamped_to_zero() -> None:
    """A negative Retry-After (server bug) must not become a negative
    sleep — that crashes time.sleep on some platforms."""
    assert _retry_delay(0, "-5") == 0.0


# ── _TRANSIENT_STATUS_CODES sanity ───────────────────────────────────


def test_transient_set_includes_529_and_429() -> None:
    """529 (Anthropic overload) is the canonical retry case — without
    it we don't survive peak hours. 429 (rate limit) also retryable."""
    assert 529 in _TRANSIENT_STATUS_CODES
    assert 429 in _TRANSIENT_STATUS_CODES
    assert 502 in _TRANSIENT_STATUS_CODES
    assert 503 in _TRANSIENT_STATUS_CODES
    assert 504 in _TRANSIENT_STATUS_CODES


def test_transient_set_excludes_4xx_client_errors() -> None:
    """400 / 401 / 403 / 404 are caller bugs, not transient — burning
    the budget on retries doesn't help."""
    for code in (400, 401, 403, 404, 422):
        assert code not in _TRANSIENT_STATUS_CODES


# ── _post_anthropic_with_retry ───────────────────────────────────────


@pytest.fixture
def extractor() -> ClaudeExtractor:
    """Stub extractor with a dummy api_key so the early-return doesn't fire."""
    return ClaudeExtractor(api_key="test-key")


@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch) -> None:
    """Don't actually sleep during retry — keeps the suite snappy."""
    monkeypatch.setattr("mantis_agent.extraction.extractor.time.sleep", lambda *_: None)


def test_post_with_retry_returns_200_immediately(extractor: ClaudeExtractor) -> None:
    """Happy path — first attempt succeeds, no retry, no sleep."""
    ok = _fake_response(200)
    with patch("requests.post", return_value=ok) as post:
        result = extractor._post_anthropic_with_retry({}, timeout=30)
    assert result is ok
    assert post.call_count == 1


def test_post_with_retry_recovers_from_one_529(extractor: ClaudeExtractor) -> None:
    """The exact failure that motivated #403: one 529 then 200 — the
    plan run survives a brief Anthropic overload spike. Before this
    fix, the 529 propagated to the caller as ``None`` and the
    fill_field step halted."""
    responses = [_fake_response(529), _fake_response(200)]
    with patch("requests.post", side_effect=responses) as post:
        result = extractor._post_anthropic_with_retry({}, timeout=30)
    assert result.status_code == 200
    assert post.call_count == 2


def test_post_with_retry_recovers_from_429_with_retry_after(
    extractor: ClaudeExtractor,
) -> None:
    """Rate limit recovery: 429 + Retry-After honoured, then 200."""
    rate_limited = _fake_response(429, headers={"Retry-After": "2"})
    ok = _fake_response(200)
    with patch("requests.post", side_effect=[rate_limited, ok]) as post:
        result = extractor._post_anthropic_with_retry({}, timeout=30)
    assert result.status_code == 200
    assert post.call_count == 2


def test_post_with_retry_recovers_from_503(extractor: ClaudeExtractor) -> None:
    """Gateway hiccup (503) is also transient."""
    with patch("requests.post", side_effect=[_fake_response(503), _fake_response(200)]):
        result = extractor._post_anthropic_with_retry({}, timeout=30)
    assert result.status_code == 200


def test_post_with_retry_returns_400_immediately(extractor: ClaudeExtractor) -> None:
    """400 is a caller bug — no retry, return immediately. Retrying
    a malformed payload wastes budget AND looks like a transient
    issue in metrics."""
    bad = _fake_response(400)
    with patch("requests.post", return_value=bad) as post:
        result = extractor._post_anthropic_with_retry({}, timeout=30)
    assert result is bad
    assert post.call_count == 1


def test_post_with_retry_exhausts_budget_on_persistent_529(
    extractor: ClaudeExtractor,
) -> None:
    """If Anthropic stays down for >15s, we eventually give up — but
    we return the final Response so the caller can log the status."""
    persistent = [_fake_response(529) for _ in range(4)]
    with patch("requests.post", side_effect=persistent) as post:
        result = extractor._post_anthropic_with_retry({}, timeout=30, max_attempts=4)
    assert result.status_code == 529
    assert post.call_count == 4


def test_post_with_retry_returns_none_on_repeated_network_errors(
    extractor: ClaudeExtractor,
) -> None:
    """Connection-level failures across the budget yield ``None`` (no
    Response object to return). Caller treats as the existing
    not-found path."""
    import requests
    with patch(
        "requests.post",
        side_effect=requests.ConnectionError("dns lookup failed"),
    ) as post:
        result = extractor._post_anthropic_with_retry({}, timeout=30, max_attempts=3)
    assert result is None
    assert post.call_count == 3


def test_post_with_retry_recovers_from_transient_network_then_200(
    extractor: ClaudeExtractor,
) -> None:
    """One network blip then 200 — the connection-error retry path
    is symmetric to the transient-HTTP retry path."""
    import requests
    ok = _fake_response(200)
    with patch(
        "requests.post",
        side_effect=[requests.Timeout("read timeout"), ok],
    ) as post:
        result = extractor._post_anthropic_with_retry({}, timeout=30, max_attempts=3)
    assert result is ok
    assert post.call_count == 2


def test_call_with_tool_schema_recovers_through_retry(
    extractor: ClaudeExtractor,
) -> None:
    """End-to-end: ``_call_with_tool_schema`` survives a 529 because
    its underlying POST is now wrapped by the retry helper. This is
    the contract the lu.ma plan depends on."""
    from PIL import Image
    img = Image.new("RGB", (10, 10), color=(255, 255, 255))

    ok = _fake_response(200)
    ok.json = MagicMock(return_value={
        "content": [{
            "type": "tool_use",
            "name": "test_tool",
            "input": {"x": 100, "y": 200},
        }],
    })
    responses = [_fake_response(529), ok]
    with patch("requests.post", side_effect=responses) as post:
        out = extractor._call_with_tool_schema(
            img, "find the email field",
            tool_name="test_tool",
            tool_description="locate an element",
            input_schema={"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x", "y"]},
        )

    assert out == {"x": 100, "y": 200}
    assert post.call_count == 2
