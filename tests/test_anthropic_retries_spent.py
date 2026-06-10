"""Tests for ``last_retries_spent`` on ``AnthropicToolUseClient`` (#841).

Per-call retry count is now surfaced via a side-effect attribute on
the client so callers can feed it into ``failure_help.retries_spent``
or the run summary.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from mantis_agent._anthropic.client import AnthropicToolUseClient


def _make_client():
    return AnthropicToolUseClient(api_key="x", model="m", log_prefix="[test]")


def test_first_attempt_success_records_zero_retries():
    client = _make_client()
    fake_resp = MagicMock(status_code=200, headers={}, json=lambda: {})
    with patch("requests.post", return_value=fake_resp):
        client.post_messages_with_retry({}, timeout=1.0, max_attempts=4)
    assert client.last_retries_spent == 0


def test_one_retry_then_success_records_one():
    client = _make_client()
    transient = MagicMock(status_code=529, headers={}, json=lambda: {})
    final = MagicMock(status_code=200, headers={}, json=lambda: {})
    with patch("requests.post", side_effect=[transient, final]):
        with patch("time.sleep"):  # avoid real backoff
            client.post_messages_with_retry({}, timeout=1.0, max_attempts=4)
    assert client.last_retries_spent == 1


def test_exhausted_records_max_minus_one():
    """All 4 attempts return 529 — retries_spent should be 3 (we
    spent the retry budget without succeeding)."""
    client = _make_client()
    transient = MagicMock(status_code=529, headers={}, json=lambda: {})
    with patch("requests.post", return_value=transient):
        with patch("time.sleep"):
            client.post_messages_with_retry({}, timeout=1.0, max_attempts=4)
    assert client.last_retries_spent == 3


def test_non_transient_error_records_zero():
    """A 400 / 401 / etc. → no retry, records 0 (we didn't retry)."""
    client = _make_client()
    fake_400 = MagicMock(status_code=400, headers={}, json=lambda: {})
    with patch("requests.post", return_value=fake_400):
        client.post_messages_with_retry({}, timeout=1.0, max_attempts=4)
    assert client.last_retries_spent == 0


def test_network_exhaustion_records_max_minus_one():
    """All 4 attempts raise ConnectionError → last_retries_spent=3
    (we exhausted retries before raising)."""
    import requests
    client = _make_client()
    with patch("requests.post", side_effect=requests.ConnectionError("reset")):
        with patch("time.sleep"):
            result = client.post_messages_with_retry(
                {}, timeout=1.0, max_attempts=4,
            )
    assert result is None
    assert client.last_retries_spent == 3
