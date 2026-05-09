"""Tests for the ``mantis plan run-modal`` CLI subcommand.

The remote driver delegates execution to a deployed Modal function, so
we don't exercise the full end-to-end path here. Instead we pin the
CLI's wiring contract: argument validation, plan loading (text → submit
text, JSON → submit dict), header propagation, output-dir scaffolding,
exit codes, and the start-url requirement for text plans.

``modal.Function.lookup`` is patched via ``sys.modules`` because the
import lives inside ``cmd_plan_run_modal`` so the rest of the CLI stays
fast for users who don't have the modal SDK installed.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mantis_agent.cli import EXIT_ERROR, EXIT_OK, main


# ── Helpers ──────────────────────────────────────────────────────────────


def _write_json_plan(path: Path, *, with_url: bool = True) -> Path:
    payload = {
        "steps": [
            {
                "intent": (
                    "Navigate to https://crm.example.test/leads"
                    if with_url
                    else "Open the leads dashboard"
                ),
                "type": "navigate" if with_url else "submit",
                "section": "setup",
                "required": True,
            },
        ],
        "source_plan": "Open the leads dashboard",
        "domain": "crm.example.test",
    }
    path.write_text(json.dumps(payload))
    return path


def _make_modal_stub(remote_result: dict | None = None) -> tuple[types.ModuleType, MagicMock]:
    """Build a minimal ``modal`` module stub with ``Function.lookup``.

    Returns ``(stub_module, run_plan_mock)`` so tests can introspect
    what kwargs the CLI submitted to the remote function.
    """
    run_plan_mock = MagicMock()
    run_plan_mock.remote.return_value = remote_result or {
        "plan_signature": "abc12345",
        "session": "plan",
        "step_count": 1,
        "successes": 1,
        "failures": 0,
        "elapsed_seconds": 12.3,
        "final_url": "https://crm.example.test/leads",
        "costs": {"claude_extract": 0.0, "gpu_steps": 1},
        "steps": [
            {
                "index": 0,
                "intent": "Navigate to https://crm.example.test/leads",
                "success": True,
                "data": "",
                "duration": 1.0,
                "steps_used": 1,
            },
        ],
    }
    function_ns = types.SimpleNamespace(lookup=MagicMock(return_value=run_plan_mock))
    stub = types.ModuleType("modal")
    stub.Function = function_ns  # type: ignore[attr-defined]
    return stub, run_plan_mock


@pytest.fixture
def modal_stub(monkeypatch):
    stub, run_plan_mock = _make_modal_stub()
    monkeypatch.setitem(sys.modules, "modal", stub)
    yield {"stub": stub, "run_plan": run_plan_mock}


# ── Argument validation ──────────────────────────────────────────────────


def test_run_modal_fails_when_plan_missing(tmp_path: Path, capsys, modal_stub) -> None:
    code = main([
        "plan", "run-modal", str(tmp_path / "missing.json"),
        "--endpoint", "https://example/v1",
    ])
    assert code == EXIT_ERROR
    err = capsys.readouterr().err
    assert "plan file not found" in err
    modal_stub["run_plan"].remote.assert_not_called()


def test_run_modal_fails_when_modal_not_installed(tmp_path: Path, capsys, monkeypatch) -> None:
    """When the modal SDK isn't installed, surface a clear error rather
    than letting an ImportError bubble up unannotated."""
    plan_path = _write_json_plan(tmp_path / "plan.json")
    # Force ImportError by inserting a None entry — Python treats None
    # in sys.modules as "this import is blocked".
    monkeypatch.setitem(sys.modules, "modal", None)
    code = main([
        "plan", "run-modal", str(plan_path),
        "--endpoint", "https://example/v1",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert code == EXIT_ERROR
    err = capsys.readouterr().err
    assert "modal SDK not installed" in err


def test_run_modal_fails_when_lookup_raises(tmp_path: Path, capsys, monkeypatch) -> None:
    """A misspelled --app-name (or undeployed app) must produce a clear
    diagnostic instead of leaking the modal SDK's exception text alone."""
    stub, run_plan_mock = _make_modal_stub()
    stub.Function.lookup.side_effect = RuntimeError("app not found")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modal", stub)

    plan_path = _write_json_plan(tmp_path / "plan.json")
    code = main([
        "plan", "run-modal", str(plan_path),
        "--app-name", "wrong-name",
        "--endpoint", "https://example/v1",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert code == EXIT_ERROR
    err = capsys.readouterr().err
    assert "cannot resolve Modal function" in err
    assert "wrong-name/run_plan" in err


def test_run_modal_requires_start_url_for_text_plans(
    tmp_path: Path, capsys, modal_stub,
) -> None:
    """Text plans get decomposed inside Modal — the CLI can't introspect
    the navigate step locally, so --start-url must be supplied."""
    plan_path = tmp_path / "plan.txt"
    plan_path.write_text("Open the boattrader listings page and extract titles.")
    code = main([
        "plan", "run-modal", str(plan_path),
        "--endpoint", "https://example/v1",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert code == EXIT_ERROR
    err = capsys.readouterr().err
    assert "--start-url is required for text plans" in err


def test_run_modal_rejects_malformed_header(
    tmp_path: Path, capsys, modal_stub,
) -> None:
    plan_path = _write_json_plan(tmp_path / "plan.json")
    code = main([
        "plan", "run-modal", str(plan_path),
        "--endpoint", "https://example/v1",
        "--header", "MissingEqualsSign",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert code == EXIT_ERROR
    err = capsys.readouterr().err
    assert "KEY=VALUE" in err


# ── Wiring: kwargs passed to run_plan.remote ─────────────────────────────


def test_run_modal_submits_json_plan_as_dict(
    tmp_path: Path, modal_stub,
) -> None:
    plan_path = _write_json_plan(tmp_path / "plan.json")
    code = main([
        "plan", "run-modal", str(plan_path),
        "--endpoint", "https://example/v1",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert code == EXIT_OK

    kwargs = modal_stub["run_plan"].remote.call_args.kwargs
    assert isinstance(kwargs["plan_json"], dict)
    assert kwargs["plan_json"]["steps"][0]["type"] == "navigate"
    assert kwargs["plan_text"] is None
    # JSON plans get the start-url inferred from the navigate step.
    assert kwargs["start_url"] == "https://crm.example.test/leads"
    assert kwargs["brain_endpoint"] == "https://example/v1"


def test_run_modal_submits_text_plan_as_string(
    tmp_path: Path, modal_stub,
) -> None:
    plan_path = tmp_path / "plan.txt"
    plan_path.write_text("Extract titles from the boattrader listings page.")
    code = main([
        "plan", "run-modal", str(plan_path),
        "--endpoint", "https://example/v1",
        "--start-url", "https://www.boattrader.com/boats",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert code == EXIT_OK

    kwargs = modal_stub["run_plan"].remote.call_args.kwargs
    assert kwargs["plan_json"] is None
    assert "boattrader" in kwargs["plan_text"]
    assert kwargs["start_url"] == "https://www.boattrader.com/boats"


def test_run_modal_threads_extra_headers(
    tmp_path: Path, modal_stub,
) -> None:
    plan_path = _write_json_plan(tmp_path / "plan.json")
    code = main([
        "plan", "run-modal", str(plan_path),
        "--endpoint", "https://example/v1",
        "--header", "X-Mantis-Token=secret123",
        "--header", "X-Tenant=acme",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert code == EXIT_OK

    kwargs = modal_stub["run_plan"].remote.call_args.kwargs
    assert kwargs["brain_extra_headers"] == {
        "X-Mantis-Token": "secret123",
        "X-Tenant": "acme",
    }


def test_run_modal_threads_proxy_flags(
    tmp_path: Path, modal_stub,
) -> None:
    plan_path = _write_json_plan(tmp_path / "plan.json")
    code = main([
        "plan", "run-modal", str(plan_path),
        "--endpoint", "https://example/v1",
        "--use-proxy",
        "--proxy-session", "boattrader-1",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert code == EXIT_OK

    kwargs = modal_stub["run_plan"].remote.call_args.kwargs
    assert kwargs["use_proxy"] is True
    assert kwargs["proxy_session"] == "boattrader-1"


def test_run_modal_threads_cost_and_time_caps(
    tmp_path: Path, modal_stub,
) -> None:
    plan_path = _write_json_plan(tmp_path / "plan.json")
    code = main([
        "plan", "run-modal", str(plan_path),
        "--endpoint", "https://example/v1",
        "--max-cost", "2.5",
        "--max-time-minutes", "5",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert code == EXIT_OK

    kwargs = modal_stub["run_plan"].remote.call_args.kwargs
    assert kwargs["max_cost_usd"] == 2.5
    assert kwargs["max_time_minutes"] == 5


def test_run_modal_threads_detail_page_pattern(
    tmp_path: Path, modal_stub,
) -> None:
    plan_path = _write_json_plan(tmp_path / "plan.json")
    code = main([
        "plan", "run-modal", str(plan_path),
        "--endpoint", "https://example/v1",
        "--detail-page-pattern", r"/leads/\d+",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert code == EXIT_OK

    kwargs = modal_stub["run_plan"].remote.call_args.kwargs
    assert kwargs["detail_page_pattern"] == r"/leads/\d+"


def test_run_modal_uses_explicit_start_url_over_plan(
    tmp_path: Path, modal_stub,
) -> None:
    plan_path = _write_json_plan(tmp_path / "plan.json")
    code = main([
        "plan", "run-modal", str(plan_path),
        "--endpoint", "https://example/v1",
        "--start-url", "https://other.example.test/dashboard",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert code == EXIT_OK

    kwargs = modal_stub["run_plan"].remote.call_args.kwargs
    assert kwargs["start_url"] == "https://other.example.test/dashboard"


def test_run_modal_uses_custom_app_name(
    tmp_path: Path, modal_stub,
) -> None:
    plan_path = _write_json_plan(tmp_path / "plan.json")
    code = main([
        "plan", "run-modal", str(plan_path),
        "--app-name", "mantis-staging",
        "--endpoint", "https://example/v1",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert code == EXIT_OK

    lookup_args = modal_stub["stub"].Function.lookup.call_args.args
    assert lookup_args == ("mantis-staging", "run_plan")


# ── Output / exit codes ──────────────────────────────────────────────────


def test_run_modal_writes_result_json_for_json_plan(
    tmp_path: Path, modal_stub,
) -> None:
    plan_path = _write_json_plan(tmp_path / "plan.json")
    out_dir = tmp_path / "out"
    code = main([
        "plan", "run-modal", str(plan_path),
        "--endpoint", "https://example/v1",
        "--output-dir", str(out_dir),
    ])
    assert code == EXIT_OK

    plan_out = out_dir / "plan.json"
    result_out = out_dir / "result.json"
    assert plan_out.exists()
    assert result_out.exists()

    result = json.loads(result_out.read_text())
    assert result["successes"] == 1
    assert result["step_count"] == 1
    assert result["final_url"] == "https://crm.example.test/leads"


def test_run_modal_writes_plan_txt_for_text_plan(
    tmp_path: Path, modal_stub,
) -> None:
    plan_path = tmp_path / "plan.txt"
    plan_path.write_text("Hello plan body.")
    out_dir = tmp_path / "out"
    code = main([
        "plan", "run-modal", str(plan_path),
        "--endpoint", "https://example/v1",
        "--start-url", "https://example.test",
        "--output-dir", str(out_dir),
    ])
    assert code == EXIT_OK

    assert (out_dir / "plan.txt").read_text() == "Hello plan body."
    assert (out_dir / "result.json").exists()


def test_run_modal_returns_error_when_remote_reports_failures(
    tmp_path: Path, monkeypatch,
) -> None:
    stub, run_plan_mock = _make_modal_stub(remote_result={
        "plan_signature": "abc",
        "session": "plan",
        "step_count": 2,
        "successes": 1,
        "failures": 1,
        "elapsed_seconds": 5.0,
        "final_url": "",
        "costs": {},
        "steps": [],
    })
    monkeypatch.setitem(sys.modules, "modal", stub)

    plan_path = _write_json_plan(tmp_path / "plan.json")
    code = main([
        "plan", "run-modal", str(plan_path),
        "--endpoint", "https://example/v1",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert code == EXIT_ERROR


def test_run_modal_returns_error_when_remote_raises(
    tmp_path: Path, capsys, monkeypatch,
) -> None:
    stub, run_plan_mock = _make_modal_stub()
    run_plan_mock.remote.side_effect = RuntimeError("modal container OOM")
    monkeypatch.setitem(sys.modules, "modal", stub)

    plan_path = _write_json_plan(tmp_path / "plan.json")
    code = main([
        "plan", "run-modal", str(plan_path),
        "--endpoint", "https://example/v1",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert code == EXIT_ERROR
    err = capsys.readouterr().err
    assert "Modal run_plan raised" in err


def test_run_modal_endpoint_is_required_argparse_level(
    tmp_path: Path, capsys, modal_stub,
) -> None:
    """``--endpoint`` is marked ``required=True`` in argparse — omitting
    it should cause argparse to bail with exit-code 2 before our handler
    runs (no Modal call). Argparse writes its error to stderr; we only
    assert the handler didn't reach the SDK."""
    plan_path = _write_json_plan(tmp_path / "plan.json")
    with pytest.raises(SystemExit):
        main([
            "plan", "run-modal", str(plan_path),
            "--output-dir", str(tmp_path / "out"),
        ])
    modal_stub["run_plan"].remote.assert_not_called()
