"""Tests for ``grade_run`` — the oracle-call wrapper.

Stand up a one-off HTTP server with a configurable oracle response and
assert that :func:`grade_run` parses it correctly. Failure modes covered:

* Passing / failing oracle responses round-trip into :class:`GradingResult`.
* 401 from a wrong admin token surfaces as ``error`` (not a raise).
* Network failure (closed port) surfaces as ``error`` (not a raise).
* Non-JSON body surfaces as ``error``.
* Missing ``task_id`` short-circuits with a clear ``error``.
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from mantis_agent.gym.grading import GradingResult, grade_run


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _OracleHandler(BaseHTTPRequestHandler):
    response: dict = {"passed": True, "score": 1.0, "reasons": ["ok"], "diff": {}}
    admin_token = "trusted"

    def log_message(self, fmt, *args):  # noqa: ANN001, A002
        return  # silence test log

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/__env__/oracle":
            self.send_response(404)
            self.end_headers()
            return
        if self.headers.get("X-Env-Admin") != self.admin_token:
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"admin"}')
            return
        qs = parse_qs(parsed.query)
        task_id = qs.get("task_id", [""])[0]
        body = dict(self.response)
        body["task_id"] = task_id
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def oracle_server():
    port = _free_port()
    server = HTTPServer(("127.0.0.1", port), _OracleHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", _OracleHandler.admin_token
    finally:
        server.shutdown()
        server.server_close()


def test_oracle_pass(oracle_server):
    url, token = oracle_server
    result = grade_run(url, token, "T01_demo")
    assert isinstance(result, GradingResult)
    assert result.passed is True
    assert result.score == 1.0
    assert result.task_id == "T01_demo"
    assert result.error is None
    assert "ok" in result.reasons


def test_oracle_fail(oracle_server):
    url, token = oracle_server
    _OracleHandler.response = {
        "passed": False, "score": 0.0,
        "reasons": ["wrong contact tagged"], "diff": {"unexpected": 1},
    }
    try:
        result = grade_run(url, token, "T02_demo")
        assert result.passed is False
        assert "wrong contact tagged" in result.reasons
        assert result.diff == {"unexpected": 1}
        assert result.error is None
    finally:
        _OracleHandler.response = {
            "passed": True, "score": 1.0, "reasons": ["ok"], "diff": {},
        }


def test_wrong_admin_token_yields_error_not_raise(oracle_server):
    url, _ = oracle_server
    result = grade_run(url, "WRONG_TOKEN", "T03_demo")
    assert result.passed is False
    assert result.error is not None
    assert "401" in result.error


def test_connection_failure_yields_error_not_raise():
    """Use a port that's definitely closed to provoke a connection refused."""
    port = _free_port()  # bind+release so OS knows nothing is listening
    result = grade_run(f"http://127.0.0.1:{port}", "any", "T04")
    assert result.passed is False
    assert result.error is not None


def test_grading_result_round_trips_to_dict():
    gr = GradingResult(
        task_id="T05", passed=True, score=0.42,
        reasons=["a", "b"], diff={"k": 1},
    )
    d = gr.to_dict()
    assert d["task_id"] == "T05"
    assert d["passed"] is True
    assert d["score"] == 0.42
    assert d["reasons"] == ["a", "b"]
    assert d["diff"] == {"k": 1}
    assert d["error"] is None


def test_empty_task_id_short_circuits():
    result = grade_run("http://does-not-matter", "tok", "")
    assert result.passed is False
    assert result.error is not None
    assert "task_id" in result.error.lower()


def test_empty_url_short_circuits():
    result = grade_run("", "tok", "T07")
    assert result.passed is False
    assert result.error is not None
    assert "env_url" in result.error.lower()
