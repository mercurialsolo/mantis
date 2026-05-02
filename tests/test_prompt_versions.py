"""Tests for #127 — prompt versioning metric + checkpoint persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from mantis_agent.prompts import (
    current_prompt_versions,
    list_prompts,
    prompt_version,
)


def test_prompt_version_is_8char_sha() -> None:
    sha = prompt_version("gemma4_system")
    assert len(sha) == 8
    assert all(c in "0123456789abcdef" for c in sha)


def test_prompt_version_stable_across_calls() -> None:
    a = prompt_version("holo3_system")
    b = prompt_version("holo3_system")
    assert a == b


def test_unknown_prompt_returns_unknown() -> None:
    assert prompt_version("does_not_exist") == "unknown"


def test_current_prompt_versions_covers_all_registered() -> None:
    versions = current_prompt_versions()
    assert set(versions) == set(list_prompts())
    for name, sha in versions.items():
        assert len(sha) == 8, name


def test_override_dir_changes_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An operator override file should produce a different SHA than the in-tree constant."""
    in_tree = prompt_version("claude_system")
    overlay = tmp_path / "claude_system.txt"
    overlay.write_text("override content body — different bytes than the in-tree constant")
    monkeypatch.setenv("MANTIS_PROMPTS_DIR", str(tmp_path))
    overridden = prompt_version("claude_system")
    assert overridden != in_tree
    assert overridden != "unknown"


def test_override_dir_unset_uses_in_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MANTIS_PROMPTS_DIR", raising=False)
    assert prompt_version("claude_system") != "unknown"


# ── Checkpoint persistence ───────────────────────────────────────────────


def test_run_checkpoint_round_trips_prompt_versions(tmp_path: Path) -> None:
    from mantis_agent.gym.micro_runner import RunCheckpoint

    cp = RunCheckpoint(
        run_key="abc",
        plan_signature="sig",
        prompt_versions={"gemma4_system": "deadbeef", "holo3_system": "cafef00d"},
    )
    target = tmp_path / "cp.json"
    cp.save(str(target))
    loaded = RunCheckpoint.load(str(target))
    assert loaded is not None
    assert loaded.prompt_versions == {
        "gemma4_system": "deadbeef",
        "holo3_system": "cafef00d",
    }


def test_run_checkpoint_default_prompt_versions_empty() -> None:
    from mantis_agent.gym.micro_runner import RunCheckpoint

    cp = RunCheckpoint()
    assert cp.prompt_versions == {}
