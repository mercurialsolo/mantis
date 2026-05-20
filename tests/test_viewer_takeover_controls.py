"""Tests for the live-viewer takeover controls (#viewer-takeover).

Two features bundled:

1. **Proxy egress display** — ``task_loop.diagnose_proxy_egress``
   probes ipinfo through the proxy at startup; the viewer renders the
   result in the header so operators see the actual exit IP / geo
   instead of just the provider name.

2. **Take Over / Resume button** — viewer-side button that POSTs to
   ``/api/pause_run`` / ``/api/resume_run`` (local viewer endpoints
   that proxy to the cua-server API on behalf of the user). The
   real ``MANTIS_API_TOKEN`` never leaves the executor container.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch



# ── diagnose_proxy_egress ────────────────────────────────────────────────


def test_diagnose_proxy_egress_returns_disabled_for_empty_server():
    """No proxy configured → ``{"disabled": True}``. Viewer renders
    that as the ``DISABLED`` chip."""
    from mantis_agent.task_loop import diagnose_proxy_egress
    assert diagnose_proxy_egress("") == {"disabled": True}


def test_diagnose_proxy_egress_extracts_ip_city_region(monkeypatch):
    """ipinfo 200 → return the relevant fields. ``org`` truncated to
    80 chars (some ISPs return long strings)."""
    from mantis_agent.task_loop import diagnose_proxy_egress
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {
        "ip": "76.109.200.135",
        "city": "Miami",
        "region": "Florida",
        "country": "US",
        "org": "AS7922 Comcast Cable Communications, LLC " + "x" * 200,
    }
    with patch("requests.get", return_value=fake):
        d = diagnose_proxy_egress("http://127.0.0.1:3128")
    assert d["ip"] == "76.109.200.135"
    assert d["city"] == "Miami"
    assert d["region"] == "Florida"
    assert d["country"] == "US"
    assert d["org"].startswith("AS7922 Comcast")
    assert len(d["org"]) <= 80  # truncated


def test_diagnose_proxy_egress_returns_error_on_non_200(monkeypatch):
    """5xx / 403 / etc. → ``{"error": "HTTP 502"}`` (operator can see
    the upstream failure mode without a Modal log dive)."""
    from mantis_agent.task_loop import diagnose_proxy_egress
    fake = MagicMock()
    fake.status_code = 502
    fake.text = "Bad Gateway"
    with patch("requests.get", return_value=fake):
        d = diagnose_proxy_egress("http://127.0.0.1:3128")
    assert "error" in d
    assert "502" in d["error"]


def test_diagnose_proxy_egress_swallows_exception(monkeypatch):
    """Network failure / timeout / proxy auth error → return error
    dict; NEVER raise (telemetry must not break runs)."""
    from mantis_agent.task_loop import diagnose_proxy_egress
    with patch("requests.get", side_effect=ConnectionError("simulated tunnel down")):
        d = diagnose_proxy_egress("http://127.0.0.1:3128")
    assert "error" in d
    assert "ConnectionError" in d["error"]


# ── setup_env return-shape contract ──────────────────────────────────────


def test_setup_env_returns_three_tuple_with_proxy_diag(monkeypatch, tmp_path):
    """Signature change: ``setup_env`` now returns ``(env, proxy_proc,
    proxy_diag)``. All 4 cua-server callers were updated; this test
    locks the contract."""
    from mantis_agent import task_loop

    # Patch out the heavy machinery so the function returns quickly.
    monkeypatch.setattr(
        task_loop, "build_proxy_config",
        lambda **kw: {"server": "http://127.0.0.1:3128", "username": "u", "password": "p"},
    )
    monkeypatch.setattr(
        task_loop, "resolve_proxy_server",
        lambda proxy, **kw: ("http://127.0.0.1:3128", MagicMock()),
    )
    # Skip the real probe — patched above wouldn't reach here anyway.
    monkeypatch.setattr(
        task_loop, "diagnose_proxy_egress",
        lambda s, **kw: {"ip": "1.2.3.4", "city": "X"},
    )
    monkeypatch.setattr(
        "mantis_agent.gym.xdotool_env.XdotoolGymEnv",
        lambda **kw: MagicMock(),
    )

    result = task_loop.setup_env(
        base_url="http://test", run_id="r1", session_name="s1",
    )
    assert len(result) == 3, f"setup_env must return 3-tuple, got {len(result)}"
    env, proxy_proc, proxy_diag = result
    assert proxy_diag == {"ip": "1.2.3.4", "city": "X"}


def test_setup_env_returns_disabled_diag_when_proxy_off(monkeypatch):
    """When ``proxy_disabled=True``, the diag dict is the disabled
    sentinel (viewer renders ``DISABLED`` chip)."""
    from mantis_agent import task_loop
    monkeypatch.setattr(
        "mantis_agent.gym.xdotool_env.XdotoolGymEnv",
        lambda **kw: MagicMock(),
    )
    monkeypatch.setattr(
        task_loop, "diagnose_proxy_egress",
        lambda s, **kw: {"disabled": True} if not s else {"ip": "x"},
    )
    _, _, diag = task_loop.setup_env(
        base_url="http://test", run_id="r1", session_name="s1",
        proxy_disabled=True,
    )
    assert diag == {"disabled": True}


# ── setup_viewer signature ──────────────────────────────────────────────


def test_setup_viewer_accepts_new_kwargs():
    """``setup_viewer`` accepts ``proxy_diag`` + ``api_run_id`` +
    ``api_tenant_id`` so the takeover controls have what they need.
    When ``enabled=False`` it returns early — easy way to verify
    the signature without launching the modal tunnel."""
    from mantis_agent.task_loop import setup_viewer
    out = setup_viewer(
        False,
        proxy_diag={"ip": "1.2.3.4"},
        api_run_id="run_x",
        api_tenant_id="tenant_y",
    )
    assert out == (None, None, None)


# ── VIEWER_HTML wiring (source-level check) ──────────────────────────────


def test_viewer_html_has_take_over_button():
    """Header has a button id=takeover-btn with the JS handler wired
    to ``/api/pause_run`` and ``/api/resume_run``."""
    from mantis_agent.viewer import VIEWER_HTML
    assert 'id="takeover-btn"' in VIEWER_HTML
    assert "/api/pause_run" in VIEWER_HTML
    assert "/api/resume_run" in VIEWER_HTML


def test_viewer_html_has_proxy_pill():
    """Header has a proxy-stat element wired to ``/api/proxy_info``."""
    from mantis_agent.viewer import VIEWER_HTML
    assert 'id="proxy-stat"' in VIEWER_HTML
    assert 'id="proxy-info"' in VIEWER_HTML
    assert "/api/proxy_info" in VIEWER_HTML


def test_viewer_html_polls_run_state_for_button_sync():
    """The button must stay in sync with server-side auto-pause
    (cf_challenge) — periodic ``/api/run_state`` poll keeps the
    label correct without user click."""
    from mantis_agent.viewer import VIEWER_HTML
    assert "/api/run_state" in VIEWER_HTML
    # setInterval keeps the poll alive
    assert "setInterval" in VIEWER_HTML
