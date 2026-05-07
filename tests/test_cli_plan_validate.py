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


# ── plan dry-run ───────────────────────────────────────────────────────


def test_dry_run_clean_plan_exits_zero(tmp_plan, capsys):
    path = tmp_plan(_good_plan())
    assert main(["plan", "dry-run", path]) == EXIT_OK
    out = capsys.readouterr().out
    assert "3 steps" in out
    # Header columns present
    assert "idx" in out and "type" in out
    # Each numbered step row appears
    assert "[00]" in out and "[01]" in out and "[02]" in out


def test_dry_run_annotates_required_and_gate(tmp_plan, capsys):
    path = tmp_plan(_good_plan())
    main(["plan", "dry-run", path])
    out = capsys.readouterr().out
    # required navigate: !req
    assert "!req" in out
    # gate verify on the second step
    assert "gate" in out


def test_dry_run_shows_loop_target_and_count(tmp_plan, capsys):
    payload = {
        "steps": [
            {"intent": "Navigate to https://x.test", "type": "navigate", "section": "setup"},
            {"intent": "Click an item", "type": "click", "section": "extraction"},
            {"intent": "Loop back to step 1", "type": "loop",
             "loop_target": 1, "loop_count": 10},
        ],
    }
    path = tmp_plan(payload)
    rc = main(["plan", "dry-run", path])
    assert rc == EXIT_OK
    out = capsys.readouterr().out
    assert "→ step [01]" in out
    assert "count=10" in out


def test_dry_run_warns_on_out_of_range_loop_target(tmp_plan, capsys):
    payload = {
        "steps": [
            {"intent": "Navigate to https://x.test", "type": "navigate", "section": "setup"},
            {"intent": "Loop", "type": "loop", "loop_target": 99, "loop_count": 1},
        ],
    }
    path = tmp_plan(payload)
    rc = main(["plan", "dry-run", path])
    # Out-of-range targets are a WARNING, not an exit-code failure — dry-run
    # is informational.
    assert rc == EXIT_OK
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "out of range" in out


def test_dry_run_json_output_shape(tmp_plan, capsys):
    path = tmp_plan(_good_plan())
    rc = main(["plan", "dry-run", path, "--json"])
    assert rc == EXIT_OK
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["step_count"] == 3
    assert parsed["domain"] == "example.com"
    assert isinstance(parsed["sections"], dict)
    # Each step has the structured fields the human view also surfaces.
    for step in parsed["steps"]:
        assert "type" in step and "section" in step and "required" in step


def test_dry_run_handles_missing_path(capsys):
    rc = main(["plan", "dry-run", "/does/not/exist.json"])
    assert rc == EXIT_ERROR
    err = capsys.readouterr().err
    assert "not found" in err


# ── plan init ──────────────────────────────────────────────────────────


def _stub_decompose_text(plan_text: str, *_, **__):
    """Stand in for PlanDecomposer.decompose_text — returns a clean MicroPlan
    that the validator accepts. Lets the init test cover the file-write +
    validate + dry-run pipeline without touching Anthropic.
    """
    from mantis_agent.plan_decomposer import MicroIntent, MicroPlan

    plan = MicroPlan(source_plan=plan_text, domain="example.com")
    plan.steps = [
        MicroIntent(
            intent="Navigate to https://example.com",
            type="navigate", section="setup", required=True,
        ),
        MicroIntent(
            intent="Verify page loaded",
            type="extract_data", section="setup",
            claude_only=True, gate=True, verify="page shows content",
        ),
        MicroIntent(
            intent="Extract data",
            type="extract_data", section="extraction", claude_only=True,
        ),
    ]
    return plan


def test_init_writes_plan_and_exits_zero(tmp_path, monkeypatch, capsys):
    """End-to-end happy path: init scaffolds a plan, writes it, validates clean."""
    monkeypatch.setattr(
        "mantis_agent.plan_decomposer.PlanDecomposer.decompose_text",
        lambda self, plan_text, **__: _stub_decompose_text(plan_text),
    )
    out = tmp_path / "out.json"
    rc = main([
        "plan", "init", "https://example.com",
        "--task", "Extract listings",
        "--output", str(out),
    ])
    assert rc == EXIT_OK
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["domain"] == "example.com"
    assert len(payload["steps"]) == 3
    out_text = capsys.readouterr().out
    assert "wrote" in out_text
    assert "validator clean" in out_text
    assert "Dry-run preview" in out_text


def test_init_default_output_path_derived_from_url(tmp_path, monkeypatch, capsys):
    """Without --output, init writes to <hostname-slug>_plan.json in cwd."""
    monkeypatch.setattr(
        "mantis_agent.plan_decomposer.PlanDecomposer.decompose_text",
        lambda self, plan_text, **__: _stub_decompose_text(plan_text),
    )
    monkeypatch.chdir(tmp_path)
    rc = main([
        "plan", "init", "https://news.ycombinator.com",
        "--task", "Extract stories",
    ])
    assert rc == EXIT_OK
    assert (tmp_path / "news_ycombinator_com_plan.json").exists()


def test_init_refuses_to_overwrite_without_flag(tmp_path, monkeypatch, capsys):
    target = tmp_path / "existing.json"
    target.write_text("{}")
    rc = main([
        "plan", "init", "https://example.com",
        "--task", "x", "--output", str(target),
    ])
    assert rc == EXIT_ERROR
    assert "already exists" in capsys.readouterr().err
    # File must be untouched.
    assert target.read_text() == "{}"


def test_init_no_validate_skips_validation(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "mantis_agent.plan_decomposer.PlanDecomposer.decompose_text",
        lambda self, plan_text, **__: _stub_decompose_text(plan_text),
    )
    out = tmp_path / "out.json"
    rc = main([
        "plan", "init", "https://example.com", "--task", "x",
        "--output", str(out), "--no-validate", "--no-dry-run",
    ])
    assert rc == EXIT_OK
    out_text = capsys.readouterr().out
    assert "validator clean" not in out_text
    assert "Dry-run preview" not in out_text


def test_init_returns_error_on_missing_api_key(tmp_path, monkeypatch, capsys):
    """The decomposer raises RuntimeError when ANTHROPIC_API_KEY is unset.
    The CLI must surface that as a clean exit-1 with the upstream error
    text — not a stack trace."""
    def _raise(self, plan_text, **_):
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    monkeypatch.setattr(
        "mantis_agent.plan_decomposer.PlanDecomposer.decompose_text", _raise,
    )
    rc = main([
        "plan", "init", "https://example.com", "--task", "x",
    ])
    assert rc == EXIT_ERROR
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in err


def test_init_requires_task_arg():
    with pytest.raises(SystemExit):
        main(["plan", "init", "https://example.com"])  # missing --task
