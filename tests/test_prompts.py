"""Tests for mantis_agent.prompts.load_prompt + MANTIS_PROMPTS_DIR override."""

from __future__ import annotations

import pytest

from mantis_agent.prompts import (
    CLAUDE_SYSTEM,
    GEMMA4_SYSTEM,
    HOLO3_SYSTEM,
    LLAMACPP_SYSTEM,
    OPENCUA_SYSTEM,
    list_prompts,
    load_prompt,
)


def test_list_prompts_contains_each_brain_system():
    expected = {
        "system_v1",
        "gemma4_system",
        "holo3_system",
        "claude_system",
        "opencua_system",
        "llamacpp_system",
    }
    assert expected.issubset(set(list_prompts()))


@pytest.mark.parametrize(
    "name,const",
    [
        ("gemma4_system", GEMMA4_SYSTEM),
        ("holo3_system", HOLO3_SYSTEM),
        ("claude_system", CLAUDE_SYSTEM),
        ("opencua_system", OPENCUA_SYSTEM),
        ("llamacpp_system", LLAMACPP_SYSTEM),
    ],
)
def test_brain_prompts_round_trip_via_loader(name: str, const: str):
    """``load_prompt(name)`` must equal the in-tree constant after .strip()."""
    assert load_prompt(name) == const.strip()


def test_load_prompt_substitutes_placeholders():
    out = load_prompt("system_v1", screen_width=1280, screen_height=720, password="hunter2")
    assert "1280x720" in out
    assert "hunter2" in out
    assert "__SCREEN_WIDTH__" not in out
    assert "__PASSWORD__" not in out


def test_load_prompt_unknown_name_raises():
    with pytest.raises(KeyError) as excinfo:
        load_prompt("nonexistent")
    assert "nonexistent" in str(excinfo.value)
    assert "Available" in str(excinfo.value)


def test_override_dir_takes_precedence(tmp_path, monkeypatch):
    """A file at ``$MANTIS_PROMPTS_DIR/<name>.txt`` overrides the constant."""
    override_dir = tmp_path / "prompts"
    override_dir.mkdir()
    (override_dir / "claude_system.txt").write_text(
        "CUSTOM CLAUDE PROMPT FOR TENANT XYZ", encoding="utf-8"
    )

    monkeypatch.setenv("MANTIS_PROMPTS_DIR", str(override_dir))
    assert load_prompt("claude_system") == "CUSTOM CLAUDE PROMPT FOR TENANT XYZ"


def test_override_dir_falls_back_when_file_missing(tmp_path, monkeypatch):
    """If the override dir lacks the requested name, fall through to constant."""
    override_dir = tmp_path / "prompts"
    override_dir.mkdir()
    # Only override 'holo3_system'; ask for 'claude_system'
    (override_dir / "holo3_system.txt").write_text("custom holo3", encoding="utf-8")

    monkeypatch.setenv("MANTIS_PROMPTS_DIR", str(override_dir))
    assert load_prompt("claude_system") == CLAUDE_SYSTEM.strip()
    assert load_prompt("holo3_system") == "custom holo3"


def test_override_dir_unset_uses_constants_only(monkeypatch):
    monkeypatch.delenv("MANTIS_PROMPTS_DIR", raising=False)
    assert load_prompt("opencua_system") == OPENCUA_SYSTEM.strip()


def test_override_dir_pointing_at_missing_path_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("MANTIS_PROMPTS_DIR", str(tmp_path / "does-not-exist"))
    # No file exists; loader falls through to the constant.
    assert load_prompt("opencua_system") == OPENCUA_SYSTEM.strip()


def test_override_dir_supports_substitutions(tmp_path, monkeypatch):
    """Substitutions still apply after the override loads from disk."""
    override_dir = tmp_path / "prompts"
    override_dir.mkdir()
    (override_dir / "system_v1.txt").write_text(
        "Screen: __SCREEN_WIDTH__x__SCREEN_HEIGHT__", encoding="utf-8"
    )

    monkeypatch.setenv("MANTIS_PROMPTS_DIR", str(override_dir))
    out = load_prompt("system_v1", screen_width=800, screen_height=600)
    assert out == "Screen: 800x600"
