"""mantis-boattrader env tests need ``deploy/sim_envs/mantis_boattrader`` on the path.

The boattrader env is a deploy artifact, not part of the ``mantis_agent``
package — it lives next to the Dockerfile under ``deploy/sim_envs/``.
Importing it for tests means putting that directory on ``sys.path`` so
``from app.main import create_app`` resolves the env's local package.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_ROOT = REPO_ROOT / "deploy" / "sim_envs" / "mantis_boattrader"


@pytest.fixture(autouse=True, scope="session")
def _env_on_syspath():
    """Add deploy/sim_envs/mantis_boattrader to sys.path for all tests in this dir."""
    if str(ENV_ROOT) not in sys.path:
        sys.path.insert(0, str(ENV_ROOT))
    yield


@pytest.fixture(autouse=True)
def _isolate_env_state(monkeypatch):
    """Reset module-level store + env vars between tests.

    The env's ``db`` module holds a singleton in-memory ``Store``. Tests
    that boot a fresh state must not see leads or mutations from earlier
    tests, so we re-import the ``app`` package each time.
    """
    monkeypatch.setenv("SEED", "42")
    monkeypatch.setenv("FAKE_NOW", "2026-01-15T09:00:00Z")
    monkeypatch.setenv("ENV_ADMIN_TOKEN", "test-admin-token")
    monkeypatch.setenv("LATENCY_MS_MIN", "0")
    monkeypatch.setenv("LATENCY_MS_MAX", "0")
    monkeypatch.setenv("LATENCY_FAILURE_RATE", "0")

    for mod_name in [m for m in list(sys.modules) if m.startswith("app")]:
        sys.modules.pop(mod_name, None)
    yield
    for mod_name in [m for m in list(sys.modules) if m.startswith("app")]:
        sys.modules.pop(mod_name, None)
