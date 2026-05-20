"""Tests for the no-customer-info pre-commit lint.

Per the standing rule (auto-memory:
``feedback_no_customer_names_in_tracked_files``): scripts, plans,
prompts, and docs must not carry customer brand names / target
domains / seed lead IDs. This pre-commit gate stops violations
from landing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add scripts/ to path so we can import the lint module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import lint_no_customer_info as lint  # noqa: E402


# ── Banned-path detection ────────────────────────────────────────────────


def test_path_block_catches_customer_filename(tmp_path: Path, monkeypatch):
    """``scripts/run_<customer>_scraper.py``-shaped filename must be
    flagged immediately — these adhoc submit scripts are the canonical
    near-miss that motivated this lint."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "scripts").mkdir()
    bad = tmp_path / "scripts" / "run_boattrader_urlnav.py"
    bad.write_text("# harmless content\nx = 1\n")
    violations = lint._scan_file("scripts/run_boattrader_urlnav.py")
    assert violations, "filename containing 'boattrader' must be flagged"
    assert any("filename contains banned" in v for v in violations)


def test_path_block_allowed_for_tests(tmp_path: Path, monkeypatch):
    """tests/ is allowlisted — test files exercising customer
    integrations in isolation are permitted (per the memory's
    'tests-isolated allowlist' carve-out)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tests").mkdir()
    # Even with banned term in the path, tests/ is allowlisted.
    bad = tmp_path / "tests" / "test_boattrader_integration.py"
    bad.write_text("def test_x():\n    assert True\n")
    violations = lint._scan_file("tests/test_boattrader_integration.py")
    assert not violations, f"tests/ path must be allowlisted, got: {violations}"


def test_path_block_allowed_for_lint_script_itself():
    """The lint script + the .pre-commit-config.yaml + .gitignore
    contain the banned terms (as patterns). Must be allowlisted."""
    for p in [
        "scripts/lint_no_customer_info.py",
        ".pre-commit-config.yaml",
        ".gitignore",
    ]:
        assert lint._is_allowed_path(p), f"{p} must be allowlisted"


# ── Banned-term content detection ────────────────────────────────────────


def test_content_block_catches_customer_brand_in_docs(tmp_path: Path):
    """A doc file referencing a banned customer brand must be flagged
    with the line number + context."""
    bad = tmp_path / "docs" / "guide.md"
    bad.parent.mkdir()
    bad.write_text(
        "# Integration guide\n"
        "Connect via the StaffAI bridge.\n"  # banned: 'staffai'
        "Done.\n"
    )
    violations = lint._scan_file(str(bad))
    assert any("staffai" in v.lower() for v in violations), violations
    # Must report the line number (2nd line is the offender).
    assert any(":2:" in v for v in violations)


def test_content_block_catches_seed_lead_identifier(tmp_path: Path):
    """Customer-side seed lead identifiers (robot_name / contact
    email) burn into git history if adhoc submit scripts slip in.
    Must be caught at the term level."""
    bad = tmp_path / "scripts" / "harness.py"
    bad.parent.mkdir()
    bad.write_text(
        "SEED = {\n"
        "    'robot_name': 'Sentinel Prime',\n"  # banned
        "    'company': 'Pinnacle Robotics',\n"  # banned
        "}\n"
    )
    violations = lint._scan_file(str(bad))
    assert violations, "seed-lead identifiers must be flagged"
    # Both terms should be caught
    blob = "\n".join(violations).lower()
    assert "sentinel prime" in blob
    assert "pinnacle robotics" in blob


def test_content_block_clean_file_passes(tmp_path: Path):
    """Clean code must not trip the lint."""
    good = tmp_path / "src" / "module.py"
    good.parent.mkdir()
    good.write_text(
        "def add(a, b):\n"
        "    \"\"\"Sum two numbers.\"\"\"\n"
        "    return a + b\n"
    )
    assert lint._scan_file(str(good)) == []


def test_content_block_case_insensitive(tmp_path: Path):
    """Match must be case-insensitive — 'BoatTrader' and 'BOATTRADER'
    are both violations."""
    bad = tmp_path / "src" / "mixed.py"
    bad.parent.mkdir()
    bad.write_text("TARGET = 'BoatTrader'\nALT = 'BOATTRADER'\n")
    violations = lint._scan_file(str(bad))
    assert violations
    assert any("boattrader" in v.lower() for v in violations)


def test_content_block_tests_path_is_allowed(tmp_path: Path):
    """Files under tests/ skip CONTENT scanning too (not just path
    block) — test fixtures need to use real customer names."""
    test_file = tmp_path / "tests" / "test_integration.py"
    test_file.parent.mkdir()
    test_file.write_text(
        "def test_boattrader_smoke():\n"
        "    assert 'boattrader' in 'we test against boattrader.com'\n"
    )
    # Path starts with 'tests/' so it's allowlisted
    import os
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        violations = lint._scan_file("tests/test_integration.py")
        assert violations == [], f"tests/ must be allowlisted for content: {violations}"
    finally:
        os.chdir(cwd)


# ── End-to-end main() ────────────────────────────────────────────────────


def test_main_returns_zero_on_clean_files(tmp_path: Path, capsys):
    """No staged files / all clean → exit 0 (allows commit)."""
    good = tmp_path / "clean.py"
    good.write_text("x = 1\n")
    rc = lint.main([str(good)])
    assert rc == 0


def test_main_returns_one_on_violations(tmp_path: Path, capsys):
    """Any violation → exit 1 (blocks commit) + friendly stderr msg."""
    bad = tmp_path / "scripts" / "run_boattrader_probe.py"
    bad.parent.mkdir()
    bad.write_text("URL = 'https://boattrader.com'\n")
    rc = lint.main([str(bad)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "Customer-info pre-commit gate failed" in err
    assert "boattrader" in err.lower()


def test_main_no_args_uses_git_staged(monkeypatch):
    """Default behavior (no argv) reads staged files from git. We
    just verify the call site exists — actual git invocation is
    integration-tested by the pre-commit framework itself."""
    monkeypatch.setattr(lint, "_staged_files", lambda: [])
    rc = lint.main([])
    assert rc == 0  # no files → trivially clean


# ── Registry sanity ──────────────────────────────────────────────────────


def test_banned_terms_set_is_non_empty():
    """If someone shrinks this to zero, the lint becomes useless."""
    assert len(lint._BANNED_TERMS) >= 5


def test_banned_path_fragments_subset_of_banned_terms():
    """Path-block fragments should track the term registry —
    inconsistencies here mean a banned term could pass the path
    check but trip on content (or vice versa)."""
    term_lc = {t.lower() for t in lint._BANNED_TERMS}
    for frag in lint._BANNED_PATH_FRAGMENTS:
        # Path fragments may be normalized (underscores etc.), so we
        # only require substring overlap.
        assert any(frag in t or t in frag for t in term_lc), (
            f"path fragment {frag!r} has no related banned term"
        )
