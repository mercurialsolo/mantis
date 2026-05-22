"""Unit tests for :mod:`mantis_agent.sim_envs.oracle_client`.

HTTP layer is exercised via ``unittest.mock`` — we don't boot a real
env, since the env-side tests already cover the route surface.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from mantis_agent.sim_envs.oracle_client import (
    fetch_mutations,
    last_mutation_id,
)


# ── fetch_mutations — happy paths ──────────────────────────────────────


def _mock_urlopen(payload: dict, *, status: int = 200):
    """Return a context-manager mock that yields ``payload`` as JSON."""
    body = json.dumps(payload).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False
    return mock_resp


def test_fetch_mutations_happy_path():
    payload = {
        "mutations": [
            {"id": 1, "operation": "consent_set", "target_type": "session", "target_id": "", "payload": {}},
            {"id": 2, "operation": "lead_submitted", "target_type": "boat", "target_id": "b1", "payload": {"email": "x@y.test"}},
        ],
    }
    with patch(
        "mantis_agent.sim_envs.oracle_client.urlopen",
        return_value=_mock_urlopen(payload),
    ):
        result = fetch_mutations("http://env.test", "tok")
    assert "error" not in result
    assert len(result["mutations"]) == 2
    assert result["mutations"][1]["operation"] == "lead_submitted"


def test_fetch_mutations_passes_since_query_param():
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        return _mock_urlopen({"mutations": []})

    with patch("mantis_agent.sim_envs.oracle_client.urlopen", side_effect=fake_urlopen):
        fetch_mutations("http://env.test/", "tok", since_id=42)

    assert captured["url"] == "http://env.test/__env__/mutations?since=42"
    # urllib lowercases header names on the Request object.
    assert captured["headers"].get("X-env-admin") == "tok"


def test_fetch_mutations_omits_since_when_zero():
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        return _mock_urlopen({"mutations": []})

    with patch("mantis_agent.sim_envs.oracle_client.urlopen", side_effect=fake_urlopen):
        fetch_mutations("http://env.test", "tok", since_id=0)

    assert captured["url"] == "http://env.test/__env__/mutations"


# ── fetch_mutations — failure paths ────────────────────────────────────


def test_fetch_mutations_empty_url_returns_error_no_call():
    with patch("mantis_agent.sim_envs.oracle_client.urlopen") as mock:
        result = fetch_mutations("", "tok")
        mock.assert_not_called()
    assert result["mutations"] == []
    assert "url is empty" in result["error"]


def test_fetch_mutations_empty_token_returns_error_no_call():
    with patch("mantis_agent.sim_envs.oracle_client.urlopen") as mock:
        result = fetch_mutations("http://env.test", "")
        mock.assert_not_called()
    assert result["mutations"] == []
    assert "admin_token is empty" in result["error"]


def test_fetch_mutations_http_error_returns_error_dict():
    from urllib.error import HTTPError

    def raise_http(*args, **kw):
        raise HTTPError("u", 403, "Forbidden", {}, None)  # type: ignore[arg-type]

    with patch("mantis_agent.sim_envs.oracle_client.urlopen", side_effect=raise_http):
        result = fetch_mutations("http://env.test", "wrong")
    assert result["mutations"] == []
    assert "HTTP 403" in result["error"]


def test_fetch_mutations_network_error_returns_error_dict():
    from urllib.error import URLError

    def raise_url(*args, **kw):
        raise URLError("connection refused")

    with patch("mantis_agent.sim_envs.oracle_client.urlopen", side_effect=raise_url):
        result = fetch_mutations("http://env.test", "tok")
    assert result["mutations"] == []
    assert "network" in result["error"]


def test_fetch_mutations_non_json_body_returns_error():
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"not json{"
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False

    with patch("mantis_agent.sim_envs.oracle_client.urlopen", return_value=mock_resp):
        result = fetch_mutations("http://env.test", "tok")
    assert result["mutations"] == []
    assert "non-JSON" in result["error"]


def test_fetch_mutations_non_dict_payload_returns_error():
    with patch(
        "mantis_agent.sim_envs.oracle_client.urlopen",
        return_value=_mock_urlopen(["not", "a", "dict"]),  # type: ignore[arg-type]
    ):
        result = fetch_mutations("http://env.test", "tok")
    assert result["mutations"] == []
    assert "non-dict payload" in result["error"]


def test_fetch_mutations_missing_mutations_list_returns_error():
    with patch(
        "mantis_agent.sim_envs.oracle_client.urlopen",
        return_value=_mock_urlopen({"other_key": []}),
    ):
        result = fetch_mutations("http://env.test", "tok")
    assert result["mutations"] == []
    assert "no mutations list" in result["error"]


# ── last_mutation_id ───────────────────────────────────────────────────


def test_last_mutation_id_empty_returns_zero():
    assert last_mutation_id([]) == 0


def test_last_mutation_id_picks_max():
    muts = [
        {"id": 5, "operation": "x"},
        {"id": 2, "operation": "y"},
        {"id": 9, "operation": "z"},
    ]
    assert last_mutation_id(muts) == 9


def test_last_mutation_id_handles_missing_id():
    muts = [
        {"operation": "x"},
        {"id": 3, "operation": "y"},
    ]
    assert last_mutation_id(muts) == 3


def test_last_mutation_id_returns_int_not_str():
    muts = [{"id": "7", "operation": "x"}]
    assert last_mutation_id(muts) == 7
