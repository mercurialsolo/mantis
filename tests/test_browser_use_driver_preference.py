"""Tests for the patchright preference in Browser-Use Plane (#826).

The Browser-Use Plane server defaults to importing ``patchright`` (a
patched-Playwright fork that strips automation tells at the binary
level) and falls back to vanilla ``playwright`` when patchright isn't
installed. Operators can force vanilla Playwright by setting
``MANTIS_BROWSER_USE_DRIVER=playwright``.
"""

from __future__ import annotations

import sys
import types

import pytest


@pytest.fixture
def fresh_env(monkeypatch):
    """Wipe both packages from sys.modules so the resolver's import
    attempt actually runs."""
    monkeypatch.delenv("MANTIS_BROWSER_USE_DRIVER", raising=False)
    for name in [
        "patchright", "patchright.sync_api",
        "playwright", "playwright.sync_api",
    ]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    return monkeypatch


def _install_stub_module(monkeypatch, name: str, sentinel) -> None:
    """Install a tiny stub module so the import succeeds and returns
    our sentinel via ``module.sync_playwright``."""
    parent_name, _, child_name = name.partition(".")
    parent = types.ModuleType(parent_name)
    child = types.ModuleType(name)
    child.sync_playwright = sentinel
    parent.sync_api = child
    monkeypatch.setitem(sys.modules, parent_name, parent)
    monkeypatch.setitem(sys.modules, name, child)


def test_patchright_preferred_when_both_installed(fresh_env):
    """Default behavior: patchright wins."""
    pr_sentinel = object()
    pw_sentinel = object()
    _install_stub_module(fresh_env, "patchright.sync_api", pr_sentinel)
    _install_stub_module(fresh_env, "playwright.sync_api", pw_sentinel)

    from mantis_agent.server.browser_use_agent import _import_sync_playwright
    chosen = _import_sync_playwright()
    assert chosen is pr_sentinel


def test_playwright_fallback_when_patchright_missing(fresh_env):
    """patchright not importable → fall back to playwright."""
    pw_sentinel = object()
    _install_stub_module(fresh_env, "playwright.sync_api", pw_sentinel)

    from mantis_agent.server.browser_use_agent import _import_sync_playwright
    chosen = _import_sync_playwright()
    assert chosen is pw_sentinel


def test_override_env_forces_vanilla_playwright(fresh_env):
    """``MANTIS_BROWSER_USE_DRIVER=playwright`` skips patchright even
    when it's importable. Escape hatch for the rare targets that
    detect patchright specifically."""
    pr_sentinel = object()
    pw_sentinel = object()
    _install_stub_module(fresh_env, "patchright.sync_api", pr_sentinel)
    _install_stub_module(fresh_env, "playwright.sync_api", pw_sentinel)
    fresh_env.setenv("MANTIS_BROWSER_USE_DRIVER", "playwright")

    from mantis_agent.server.browser_use_agent import _import_sync_playwright
    chosen = _import_sync_playwright()
    assert chosen is pw_sentinel


def test_override_case_insensitive(fresh_env):
    pr_sentinel = object()
    pw_sentinel = object()
    _install_stub_module(fresh_env, "patchright.sync_api", pr_sentinel)
    _install_stub_module(fresh_env, "playwright.sync_api", pw_sentinel)
    fresh_env.setenv("MANTIS_BROWSER_USE_DRIVER", "PLAYWRIGHT")

    from mantis_agent.server.browser_use_agent import _import_sync_playwright
    chosen = _import_sync_playwright()
    assert chosen is pw_sentinel
