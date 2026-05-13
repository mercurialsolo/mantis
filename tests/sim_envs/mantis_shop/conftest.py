"""mantis-shop env tests need ``deploy/sim_envs/mantis_shop`` on the path.

Mirrors ``tests/sim_envs/mantis_crm/conftest.py``: the env lives next to
its Dockerfile under ``deploy/sim_envs/``, not inside the ``mantis_agent``
package, so we have to add it to ``sys.path`` for ``from app.main import app``
to resolve.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_ROOT = REPO_ROOT / "deploy" / "sim_envs" / "mantis_shop"


@pytest.fixture(autouse=True, scope="session")
def _env_on_syspath():
    if str(ENV_ROOT) not in sys.path:
        sys.path.insert(0, str(ENV_ROOT))
    yield


@pytest.fixture(autouse=True)
def _isolate_env_state(monkeypatch):
    """Reset module-level DB connection + events between tests."""
    monkeypatch.setenv("DB_PATH", ":memory:")
    monkeypatch.setenv("SEED", "42")
    monkeypatch.setenv("FAKE_NOW", "2026-01-15T09:00:00Z")
    monkeypatch.setenv("ENV_ADMIN_TOKEN", "test-admin-token")

    for mod_name in [m for m in list(sys.modules) if m.startswith("app")]:
        sys.modules.pop(mod_name, None)
    yield
    for mod_name in [m for m in list(sys.modules) if m.startswith("app")]:
        sys.modules.pop(mod_name, None)
