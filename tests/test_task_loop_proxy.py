"""Proxy setup helpers."""

from __future__ import annotations

from mantis_agent.task_loop import setup_env


def test_setup_env_proxy_disabled_ignores_proxy_env(monkeypatch) -> None:
    monkeypatch.setenv("PROXY_URL", "http://proxy.example:1234")
    monkeypatch.setenv("PROXY_USER", "user")
    monkeypatch.setenv("PROXY_PASS", "pass")

    env, proxy_proc, _proxy_diag = setup_env(
        base_url="https://news.ycombinator.com",
        run_id="20260507_000000",
        session_name="hn_smoke",
        proxy_disabled=True,
    )

    assert proxy_proc is None
    assert env._proxy_server == ""
