"""Tests for the Baseten per-tenant per-profile Chrome user-data-dir
isolation (parity with PR #426 on Modal).

The Baseten runtime previously passed ``data_root / "chrome-profile"``
as the profile_dir — a flat path shared across every request, despite
a surrounding comment that claimed tenant scoping. These tests pin the
new behaviour: ``_chrome_profile_dir_for_suite`` derives a per-profile
sub-path so cookies / localStorage / IndexedDB are isolated across
tenants and across ``profile_id``s on the same container.
"""

from __future__ import annotations

import os

import pytest

# The baseten_server.runtime module imports the Anthropic / mss /
# torch surface lazily, but the top-level import already pulls
# transitively-heavy modules. Skip the whole file when those aren't
# available so CI environments without the full extra-set can still
# collect the rest of the test suite.
pytest.importorskip("mantis_agent.baseten_server.runtime")
from mantis_agent.baseten_server.runtime import _chrome_profile_dir_for_suite


@pytest.fixture(autouse=True)
def _isolated_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path / "mantis-data"))


def test_per_profile_path_when_profile_id_present() -> None:
    """The most common shape: ``_profile_id`` is set by
    ``build_micro_suite`` to ``<tenant_id>__<caller_profile_id>``. The
    returned path embeds that segment under ``chrome-profile/``.
    """
    suite = {"_profile_id": "default__alice"}
    path = _chrome_profile_dir_for_suite(suite)
    assert path.endswith("/chrome-profile/default__alice")


def test_different_profiles_yield_different_paths() -> None:
    """The CORE isolation guarantee — two distinct ``profile_id``s
    on the same tenant + container must resolve to DIFFERENT paths.
    Without this, Chrome reuses the cached user-data-dir across runs
    and cookies / localStorage leak.
    """
    alice = _chrome_profile_dir_for_suite({"_profile_id": "default__alice"})
    bob = _chrome_profile_dir_for_suite({"_profile_id": "default__bob"})
    assert alice != bob


def test_different_tenants_yield_different_paths() -> None:
    """Cross-tenant isolation. ``build_micro_suite`` prefixes
    profile_id with tenant_id, so two tenants with the same caller-
    side profile_id ("default") still end up on distinct paths.
    """
    tenant_a = _chrome_profile_dir_for_suite({"_profile_id": "acme__default"})
    tenant_b = _chrome_profile_dir_for_suite({"_profile_id": "globex__default"})
    assert tenant_a != tenant_b


def test_falls_back_to_state_key_when_profile_id_missing() -> None:
    """Legacy callers that only set ``_state_key`` (pre-#341) still
    get a per-key dir rather than landing in the shared default.
    """
    suite = {"_state_key": "legacy-tenant__some-key"}
    path = _chrome_profile_dir_for_suite(suite)
    assert path.endswith("/chrome-profile/legacy-tenant__some-key")


def test_falls_back_to_default_when_neither_id_present() -> None:
    """No-id payloads end up under a ``default`` segment instead of
    sharing the previous ``data_root / "chrome-profile"`` flat path.
    The shared-default still represents a leak surface between
    unscoped callers, but it's strictly better than the pre-fix
    state (every caller shared one profile).
    """
    suite = {}
    path = _chrome_profile_dir_for_suite(suite)
    assert path.endswith("/chrome-profile/default")


def test_path_lives_under_data_root() -> None:
    """The path roots at ``MANTIS_DATA_DIR / "chrome-profile" / …``
    so the on-disk layout matches the doc's claim and is reachable
    by the session-reuse cache lookups.
    """
    expected_root = os.environ["MANTIS_DATA_DIR"]
    path = _chrome_profile_dir_for_suite({"_profile_id": "x__y"})
    assert path.startswith(expected_root + "/chrome-profile/")


def test_profile_id_sanitization_via_safe_state_key() -> None:
    """``safe_state_key`` is applied to the profile_id before it
    becomes a path segment, so a caller-supplied id with unsafe
    characters can't escape the chrome-profile directory.
    """
    # Caller-supplied ids that ``build_micro_suite`` should have
    # already sanitised — but we apply the helper defensively here
    # too. The final path segment must not contain a slash (which
    # would allow path-traversal); ``safe_state_key`` collapses
    # slashes into underscores. The ``..`` characters themselves
    # can survive in the middle of a filename — they're harmless
    # when there's no slash to start a traversal from.
    path = _chrome_profile_dir_for_suite({"_profile_id": "default__alice/../etc"})
    final_segment = path.rsplit("/chrome-profile/", 1)[-1]
    assert "/" not in final_segment
