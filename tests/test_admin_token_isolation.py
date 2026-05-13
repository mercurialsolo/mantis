"""Admin-token isolation tests for the stub env.

The harness contract (#336): the admin token used to call ``/__env__/*``
endpoints MUST NOT be reachable from the agent's browser context. The
agent navigates to ``/`` (a normal SPA), the harness signs admin calls
out-of-band, and the two surfaces share no auth material.

Tested invariants:

* ``/__env__/oracle`` returns 401 without ``X-Env-Admin``.
* ``/__env__/oracle`` returns 401 with the WRONG ``X-Env-Admin``.
* ``/__env__/reset`` / ``/__env__/seed`` / ``/__env__/clock`` /
  ``/__env__/state`` / ``/__env__/events`` all gate on the token.
* ``/__env__/health`` is intentionally OPEN (health checks can't
  authenticate). This is a contract assertion — surface in case anyone
  adds gating here later by accident.
* Agent-facing GET ``/`` doesn't echo the admin token anywhere in its
  body — sanity check on the response we'd render in a real env.
"""

from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from mantis_agent.sim_envs.local import LocalBackend


@pytest.fixture
def stub_env():
    backend = LocalBackend()
    handle = backend.start("stub-test", seed=42)
    try:
        backend.wait_healthy(handle, timeout_s=10.0)
        yield handle
    finally:
        backend.stop(handle)


def _get(url: str, *, headers: dict | None = None) -> tuple[int, str]:
    req = Request(url, headers=headers or {})
    try:
        with urlopen(req, timeout=2.0) as resp:  # noqa: S310
            return resp.status, resp.read().decode("utf-8")
    except HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


def _post(url: str, *, headers: dict | None = None, body: dict | None = None) -> tuple[int, str]:
    data = json.dumps(body or {}).encode("utf-8")
    req = Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urlopen(req, timeout=2.0) as resp:  # noqa: S310
            return resp.status, resp.read().decode("utf-8")
    except HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


@pytest.mark.parametrize("path", [
    "/__env__/oracle?task_id=T01",
    "/__env__/state",
    "/__env__/events",
])
def test_admin_endpoints_require_token_get(stub_env, path):
    """GET /__env__/* without the token returns 401."""
    status, body = _get(f"{stub_env.url}{path}")
    assert status == 401
    assert "admin token" in body.lower()


@pytest.mark.parametrize("path", [
    "/__env__/reset",
    "/__env__/seed",
    "/__env__/clock",
])
def test_admin_endpoints_require_token_post(stub_env, path):
    """POST /__env__/* without the token returns 401."""
    status, body = _post(f"{stub_env.url}{path}")
    assert status == 401
    assert "admin token" in body.lower()


def test_wrong_token_still_401(stub_env):
    status, _ = _get(
        f"{stub_env.url}/__env__/oracle?task_id=T",
        headers={"X-Env-Admin": "GUESSED_VALUE"},
    )
    assert status == 401


def test_health_is_open(stub_env):
    """Health endpoint MUST work without a token (health checks can't auth)."""
    status, body = _get(f"{stub_env.url}/__env__/health")
    assert status == 200
    assert json.loads(body)["ok"] is True


def test_correct_token_unlocks_oracle(stub_env):
    status, body = _get(
        f"{stub_env.url}/__env__/oracle?task_id=T01",
        headers={"X-Env-Admin": stub_env.admin_token},
    )
    assert status == 200
    payload = json.loads(body)
    assert payload["task_id"] == "T01"
    assert payload["passed"] is True


def test_agent_landing_page_does_not_leak_admin_token(stub_env):
    """The agent's view of the SPA must not contain the admin token."""
    status, body = _get(stub_env.url + "/")
    assert status == 200
    # The admin token is high-entropy random — assert it's absent in the
    # HTML we render to the agent.
    assert stub_env.admin_token not in body
    assert "stub env" in body.lower()
