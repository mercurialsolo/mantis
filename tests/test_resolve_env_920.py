"""#920 — backend-agnostic env resolution (Daytona id vs direct Modal/URL).

Pins the routing in ``run_sealed_task._resolve_env`` without network: a direct
URL ref → used as-is with no preview token + admin from env; a Daytona id ref →
delegated to ``_daytona_env``; an empty ref → explicit error (never a silent
base-env run).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "holdout"))

import run_sealed_task as rst  # noqa: E402


def test_url_ref_used_directly_with_env_admin_token():
    env = {"ENV_ADMIN_TOKEN": "shared-admin"}
    info = rst._resolve_env("crm", "https://crm.example.modal.run/", env)
    assert info["url"] == "https://crm.example.modal.run"  # trailing slash trimmed
    assert info["preview_token"] == ""  # no Daytona token for a direct URL
    assert info["admin_token"] == "shared-admin"


def test_url_ref_prefers_per_env_admin_token():
    env = {"CRM_ADMIN_TOKEN": "crm-specific", "ENV_ADMIN_TOKEN": "shared"}
    info = rst._resolve_env("crm", "https://crm.example.modal.run", env)
    assert info["admin_token"] == "crm-specific"


def test_empty_ref_raises():
    with pytest.raises(RuntimeError, match="not wired"):
        rst._resolve_env("boattrader", "", {})


def test_daytona_id_ref_delegates(monkeypatch):
    calls = {}

    def _fake_daytona(sandbox_id, api_key):
        calls["sandbox_id"] = sandbox_id
        calls["api_key"] = api_key
        return {"url": "https://sb.daytonaproxy01.net", "preview_token": "pv", "admin_token": "a"}

    monkeypatch.setattr(rst, "_daytona_env", _fake_daytona)
    info = rst._resolve_env("indeed", "a72a3ffc-sandbox-id", {"DAYTONA_API_KEY": "dk"})
    assert calls == {"sandbox_id": "a72a3ffc-sandbox-id", "api_key": "dk"}
    assert info["preview_token"] == "pv"
