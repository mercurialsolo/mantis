"""mantis-crm env tests need ``deploy/sim_envs/mantis_crm`` on the path.

The CRM env is a deploy artifact, not part of the ``mantis_agent``
package — it lives next to the Dockerfile under ``deploy/sim_envs/``.
Importing it for tests means putting that directory on ``sys.path`` so
``from app.main import app`` resolves the env's local package.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_ROOT = REPO_ROOT / "deploy" / "sim_envs" / "mantis_crm"


@pytest.fixture(autouse=True, scope="session")
def _env_on_syspath():
    """Add deploy/sim_envs/mantis_crm to sys.path for all tests in this dir."""
    if str(ENV_ROOT) not in sys.path:
        sys.path.insert(0, str(ENV_ROOT))
    yield


@pytest.fixture(autouse=True)
def _isolate_env_state(monkeypatch):
    """Reset module-level DB connection + events between tests.

    The env's ``db`` module holds a singleton SQLite connection. Tests
    that boot a fresh in-memory DB must not see state from earlier tests.
    """
    monkeypatch.setenv("DB_PATH", ":memory:")
    monkeypatch.setenv("SEED", "42")
    monkeypatch.setenv("FAKE_NOW", "2026-01-15T09:00:00Z")
    monkeypatch.setenv("ENV_ADMIN_TOKEN", "test-admin-token")

    # Re-import the env packages cleanly so the module-level connection
    # is rebuilt against the fresh env vars.
    for mod_name in [m for m in list(sys.modules) if m.startswith("app")]:
        sys.modules.pop(mod_name, None)
    yield
    for mod_name in [m for m in list(sys.modules) if m.startswith("app")]:
        sys.modules.pop(mod_name, None)
