"""Tests for the ``mantis plan validate`` CLI subcommand (#154).

Exercises the entry function directly (no subprocess spawn — too slow on
the per-test budget) and asserts the exit codes documented in the
module docstring: 0 = clean, 1 = error, 2 = warning-only.

The validator surface is tested separately in ``tests/test_plan_validator.py``;
these tests pin the CLI's input parsing, output formatting, and exit-code
contract.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from mantis_agent.cli import EXIT_ERROR, EXIT_OK, main


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def tmp_plan(tmp_path: Path):
    """Factory: write a JSON plan to a tmp file and return its path."""

    def _write(payload: dict | list) -> str:
        path = tmp_path / "plan.json"
        path.write_text(json.dumps(payload))
        return str(path)

    return _write


def _good_plan() -> dict:
    """A minimal plan PlanValidator considers clean."""
    return {
        "steps": [
            {
                "intent": "Navigate to https://example.com",
                "type": "navigate",
                "section": "setup",
                "required": True,
            },
            {
                "intent": "Verify page loaded",
                "type": "extract_data",
                "claude_only": True,
                "section": "setup",
                "gate": True,
                "verify": "page shows content",
            },
            {
                "intent": "Extract listings",
                "type": "extract_data",
                "claude_only": True,
                "section": "extraction",
            },
        ],
        "domain": "example.com",
    }


# ── Exit-code contract ─────────────────────────────────────────────────


def test_validate_clean_plan_exits_zero(tmp_plan, capsys):
    path = tmp_plan(_good_plan())
    assert main(["plan", "validate", path]) == EXIT_OK
    out = capsys.readouterr().out
    assert "3 steps" in out
    assert "clean" in out


def test_validate_empty_plan_exits_error(tmp_plan, capsys):
    path = tmp_plan({"steps": []})
    assert main(["plan", "validate", path]) == EXIT_ERROR
    out = capsys.readouterr().out
    assert "empty_plan" in out
    assert "ERROR" in out


def test_validate_missing_navigate_exits_error(tmp_plan, capsys):
    """PlanValidator flags any plan that doesn't start with a navigate
    step — direct from #110 (decomposer paraphrasing URLs out of plans)."""
    path = tmp_plan({
        "steps": [
            {
                "intent": "Click first listing",
                "type": "click",
                "section": "extraction",
            },
        ],
    })
    rc = main(["plan", "validate", path])
    assert rc == EXIT_ERROR
    out = capsys.readouterr().out
    # Must surface the missing-navigate code (or a related plan-level issue).
    assert "ERROR" in out


def test_validate_missing_path_exits_error(capsys):
    rc = main(["plan", "validate", "/does/not/exist.json"])
    assert rc == EXIT_ERROR
    err = capsys.readouterr().err
    assert "not found" in err


def test_validate_invalid_json_exits_error(tmp_path, capsys):
    bad = tmp_path / "broken.json"
    bad.write_text("{not valid")
    rc = main(["plan", "validate", str(bad)])
    assert rc == EXIT_ERROR
    err = capsys.readouterr().err
    assert "invalid JSON" in err


def test_validate_stdin_read(monkeypatch, capsys):
    """``mantis plan validate -`` reads JSON from stdin."""
    payload = json.dumps(_good_plan())
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))
    rc = main(["plan", "validate", "-"])
    assert rc == EXIT_OK
    out = capsys.readouterr().out
    assert "<stdin>" in out


def test_validate_bare_array_payload(tmp_plan, capsys):
    """The decomposer can emit a bare list at the top level. ``MicroPlan.from_dict``
    accepts both shapes; the CLI must too."""
    payload = _good_plan()["steps"]
    path = tmp_plan(payload)
    assert main(["plan", "validate", path]) == EXIT_OK


# ── --json output ──────────────────────────────────────────────────────


def test_validate_json_output_shape(tmp_plan, capsys):
    path = tmp_plan({"steps": []})
    rc = main(["plan", "validate", path, "--json"])
    assert rc == EXIT_ERROR
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["step_count"] == 0
    assert isinstance(parsed["errors"], list)
    assert any(e["code"] == "empty_plan" for e in parsed["errors"])


def test_validate_json_clean_plan(tmp_plan, capsys):
    path = tmp_plan(_good_plan())
    rc = main(["plan", "validate", path, "--json"])
    assert rc == EXIT_OK
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["errors"] == []


# ── Help / arg parsing ─────────────────────────────────────────────────


def test_no_args_prints_help_and_errors():
    """``mantis`` with no subcommand returns argparse error code (2 or 1)."""
    with pytest.raises(SystemExit):
        main([])


def test_plan_with_no_subcommand_errors():
    with pytest.raises(SystemExit):
        main(["plan"])
