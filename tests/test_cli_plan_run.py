"""Tests for the ``mantis plan run`` CLI subcommand.

The end-to-end runner needs a live remote brain + a real browser, so we
don't exercise the full path here. The tests pin the CLI's wiring
contract: argument validation, plan loading (text → decompose, JSON →
direct), output-dir scaffolding, error exit codes, and the
stay-generic discipline (neutral SiteConfig by default; per-plan
``--detail-page-pattern`` override hook).

Heavy dependencies (Holo3Brain, PlaywrightGymEnv, ClaudeGrounding,
ClaudeExtractor, MicroPlanRunner) are patched so the tests stay fast
and don't require playwright / Anthropic / a Modal endpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mantis_agent.cli import (
    EXIT_ERROR,
    EXIT_OK,
    _looks_like_text_plan,
    _parse_extra_headers,
    _platform_default_model,
    _resolve_first_navigate_url,
    main,
)


# ── Pure helpers ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,is_text",
    [
        ("plan.txt", True),
        ("plan.md", True),
        ("plan", True),  # no extension
        ("plan.json", False),
        ("plan.JSON", False),  # case-insensitive
    ],
)
def test_looks_like_text_plan(name: str, is_text: bool) -> None:
    assert _looks_like_text_plan(Path(name)) is is_text


def test_parse_extra_headers_basic() -> None:
    out = _parse_extra_headers(["X-Mantis-Token=abc", "X-Tenant=acme"])
    assert out == {"X-Mantis-Token": "abc", "X-Tenant": "acme"}


def test_parse_extra_headers_strips_whitespace() -> None:
    out = _parse_extra_headers(["  X-Foo = bar baz  "])
    assert out == {"X-Foo": "bar baz"}


def test_parse_extra_headers_rejects_missing_separator() -> None:
    with pytest.raises(ValueError, match="KEY=VALUE"):
        _parse_extra_headers(["NoEqualsHere"])


def test_parse_extra_headers_rejects_empty_key() -> None:
    with pytest.raises(ValueError, match="empty key"):
        _parse_extra_headers(["=value"])


def test_parse_extra_headers_handles_none() -> None:
    assert _parse_extra_headers(None) == {}


def test_platform_default_model_per_platform() -> None:
    assert "Holo3" in _platform_default_model("modal")
    assert _platform_default_model("baseten") == "holo3-35b-a3b"
    # Custom falls back to the generic name; specific user passes --model anyway.
    assert _platform_default_model("custom") == "holo3-35b-a3b"


def test_resolve_first_navigate_url_from_intent() -> None:
    from mantis_agent.plan_decomposer import MicroIntent, MicroPlan

    plan = MicroPlan(steps=[
        MicroIntent(intent="Open the leads dashboard", type="submit"),
        MicroIntent(
            intent="Navigate to https://crm.example.test/leads",
            type="navigate",
        ),
        MicroIntent(intent="Click the first lead", type="click"),
    ])
    assert _resolve_first_navigate_url(plan) == "https://crm.example.test/leads"


def test_resolve_first_navigate_url_from_params_url() -> None:
    """When the intent paraphrases the URL away (the #210 failure mode),
    the runner's params.url fallback applies — same logic here."""
    from mantis_agent.plan_decomposer import MicroIntent, MicroPlan

    plan = MicroPlan(steps=[
        MicroIntent(
            intent="Navigate to the leads management system",
            type="navigate",
            params={"url": "https://crm.example.test/leads"},
        ),
    ])
    assert _resolve_first_navigate_url(plan) == "https://crm.example.test/leads"


def test_resolve_first_navigate_url_returns_empty_when_none() -> None:
    from mantis_agent.plan_decomposer import MicroIntent, MicroPlan

    plan = MicroPlan(steps=[
        MicroIntent(intent="Open the leads section", type="submit"),
    ])
    assert _resolve_first_navigate_url(plan) == ""


# ── End-to-end argument validation (heavy deps patched) ──────────────────


def _write_json_plan(path: Path, *, with_url: bool = True) -> Path:
    """Write a minimal pre-decomposed plan; runner deps will be patched."""
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


def test_run_fails_when_plan_missing(tmp_path: Path, capsys) -> None:
    code = main([
        "plan", "run", str(tmp_path / "missing.json"),
        "--endpoint", "https://example/v1",
        "--anthropic-api-key", "k",
    ])
    assert code == EXIT_ERROR
    err = capsys.readouterr().err
    assert "plan file not found" in err


def test_run_fails_when_endpoint_missing(tmp_path: Path, capsys) -> None:
    plan_path = _write_json_plan(tmp_path / "plan.json")
    code = main([
        "plan", "run", str(plan_path),
        "--anthropic-api-key", "k",
    ])
    assert code == EXIT_ERROR
    err = capsys.readouterr().err
    assert "--endpoint is required" in err


def test_run_fails_when_anthropic_key_missing(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    plan_path = _write_json_plan(tmp_path / "plan.json")
    code = main([
        "plan", "run", str(plan_path),
        "--endpoint", "https://example/v1",
    ])
    assert code == EXIT_ERROR
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in err


def test_run_fails_when_no_start_url_resolvable(
    tmp_path: Path, capsys,
) -> None:
    """A plan with no navigate-with-URL and no --start-url: error,
    don't construct the browser."""
    plan_path = _write_json_plan(tmp_path / "plan.json", with_url=False)
    code = main([
        "plan", "run", str(plan_path),
        "--endpoint", "https://example/v1",
        "--anthropic-api-key", "k",
    ])
    assert code == EXIT_ERROR
    err = capsys.readouterr().err
    assert "start-url" in err


def test_run_fails_when_plan_has_no_steps(tmp_path: Path, capsys) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"steps": []}))
    code = main([
        "plan", "run", str(empty),
        "--endpoint", "https://example/v1",
        "--anthropic-api-key", "k",
    ])
    assert code == EXIT_ERROR
    err = capsys.readouterr().err
    assert "no steps" in err


def test_run_rejects_malformed_header(tmp_path: Path, capsys) -> None:
    plan_path = _write_json_plan(tmp_path / "plan.json")
    code = main([
        "plan", "run", str(plan_path),
        "--endpoint", "https://example/v1",
        "--anthropic-api-key", "k",
        "--header", "MissingEqualsSign",
    ])
    assert code == EXIT_ERROR
    err = capsys.readouterr().err
    assert "KEY=VALUE" in err


# ── Wiring contract — patched deps ───────────────────────────────────────


@pytest.fixture
def patched_deps():
    """Patch every heavy dep ``cmd_plan_run`` constructs.

    Returns a namespace with the mocks so the test can assert wiring.
    """
    with patch("mantis_agent.brain_holo3.Holo3Brain") as brain_cls, \
         patch("mantis_agent.gym.playwright_env.PlaywrightGymEnv") as env_cls, \
         patch("mantis_agent.grounding.ClaudeGrounding") as ground_cls, \
         patch("mantis_agent.extraction.ClaudeExtractor") as ext_cls, \
         patch("mantis_agent.gym.micro_runner.MicroPlanRunner") as runner_cls:
        brain_cls.return_value = MagicMock(model="holo3-35b-a3b")
        env_cls.return_value = MagicMock()
        ground_cls.return_value = MagicMock()
        ext_cls.return_value = MagicMock()
        runner_instance = MagicMock()
        runner_instance.run.return_value = [
            MagicMock(
                step_index=0, intent="Navigate to https://crm.example.test/leads",
                success=True, data="", duration=1.0, steps_used=1,
            ),
        ]
        runner_instance.plan_signature = "abc12345"
        runner_instance._last_known_url = "https://crm.example.test/leads"
        runner_instance.costs = {"claude_extract": 0, "gpu_steps": 1}
        runner_cls.return_value = runner_instance

        yield {
            "brain_cls": brain_cls,
            "env_cls": env_cls,
            "ground_cls": ground_cls,
            "ext_cls": ext_cls,
            "runner_cls": runner_cls,
            "runner_instance": runner_instance,
        }


def test_run_wires_runner_with_neutral_site_config_by_default(
    tmp_path: Path, patched_deps,
) -> None:
    """Stay-generic discipline: a neutral plan run must use SiteConfig()
    (empty pattern), not SiteConfig.default_boattrader(). The path-
    extension heuristic from #211 then applies."""
    plan_path = _write_json_plan(tmp_path / "plan.json")
    code = main([
        "plan", "run", str(plan_path),
        "--endpoint", "https://workspace--app-fn.modal.run/v1",
        "--anthropic-api-key", "k",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert code == EXIT_OK

    runner_kwargs = patched_deps["runner_cls"].call_args.kwargs
    sc = runner_kwargs["site_config"]
    assert sc.detail_page_pattern == ""
    assert sc.domain == ""


def test_run_threads_detail_page_pattern_into_site_config(
    tmp_path: Path, patched_deps,
) -> None:
    """``--detail-page-pattern`` is the per-plan injection point — when
    a recipe / decomposer infers a tighter pattern for the run, it goes
    here, not into the framework primitive."""
    plan_path = _write_json_plan(tmp_path / "plan.json")
    code = main([
        "plan", "run", str(plan_path),
        "--endpoint", "https://example/v1",
        "--anthropic-api-key", "k",
        "--detail-page-pattern", r"/leads/\d+",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert code == EXIT_OK

    sc = patched_deps["runner_cls"].call_args.kwargs["site_config"]
    assert sc.detail_page_pattern == r"/leads/\d+"


def test_run_writes_plan_and_result_json(tmp_path: Path, patched_deps) -> None:
    plan_path = _write_json_plan(tmp_path / "plan.json")
    out_dir = tmp_path / "out"
    code = main([
        "plan", "run", str(plan_path),
        "--endpoint", "https://example/v1",
        "--anthropic-api-key", "k",
        "--output-dir", str(out_dir),
    ])
    assert code == EXIT_OK

    plan_out = out_dir / "plan.json"
    result_out = out_dir / "result.json"
    assert plan_out.exists()
    assert result_out.exists()

    result = json.loads(result_out.read_text())
    assert result["successes"] == 1
    assert result["failures"] == 0
    assert result["step_count"] == 1
    assert result["final_url"] == "https://crm.example.test/leads"
    assert result["platform"] == "modal"
    assert result["endpoint"] == "https://example/v1"


def test_run_passes_platform_default_model_when_unspecified(
    tmp_path: Path, patched_deps,
) -> None:
    plan_path = _write_json_plan(tmp_path / "plan.json")
    main([
        "plan", "run", str(plan_path),
        "--platform", "baseten",
        "--endpoint", "https://model-x.api.baseten.co/production/sync/v1",
        "--anthropic-api-key", "k",
        "--output-dir", str(tmp_path / "out"),
    ])
    brain_kwargs = patched_deps["brain_cls"].call_args.kwargs
    assert brain_kwargs["model"] == "holo3-35b-a3b"


def test_run_explicit_model_overrides_platform_default(
    tmp_path: Path, patched_deps,
) -> None:
    plan_path = _write_json_plan(tmp_path / "plan.json")
    main([
        "plan", "run", str(plan_path),
        "--endpoint", "https://example/v1",
        "--anthropic-api-key", "k",
        "--model", "custom-checkpoint-v2",
        "--output-dir", str(tmp_path / "out"),
    ])
    brain_kwargs = patched_deps["brain_cls"].call_args.kwargs
    assert brain_kwargs["model"] == "custom-checkpoint-v2"


def test_run_threads_extra_headers_into_brain(
    tmp_path: Path, patched_deps,
) -> None:
    plan_path = _write_json_plan(tmp_path / "plan.json")
    main([
        "plan", "run", str(plan_path),
        "--endpoint", "https://example/v1",
        "--anthropic-api-key", "k",
        "--header", "X-Mantis-Token=secret123",
        "--header", "X-Tenant=acme",
        "--output-dir", str(tmp_path / "out"),
    ])
    brain_kwargs = patched_deps["brain_cls"].call_args.kwargs
    assert brain_kwargs["extra_headers"] == {
        "X-Mantis-Token": "secret123",
        "X-Tenant": "acme",
    }


def test_run_uses_explicit_start_url_over_plan_inference(
    tmp_path: Path, patched_deps,
) -> None:
    """When the caller passes --start-url, the browser opens there,
    even if the plan's first navigate step has a different URL."""
    plan_path = _write_json_plan(tmp_path / "plan.json")
    main([
        "plan", "run", str(plan_path),
        "--endpoint", "https://example/v1",
        "--anthropic-api-key", "k",
        "--start-url", "https://other.example.test/login",
        "--output-dir", str(tmp_path / "out"),
    ])
    env_kwargs = patched_deps["env_cls"].call_args.kwargs
    assert env_kwargs["start_url"] == "https://other.example.test/login"


def test_run_returns_error_when_runner_reports_failures(
    tmp_path: Path, patched_deps,
) -> None:
    """Exit code reflects step-level success/failure: 0 if every step
    succeeded, 1 otherwise."""
    patched_deps["runner_instance"].run.return_value = [
        MagicMock(
            step_index=0, intent="x", success=False, data="form_target_not_found",
            duration=1.0, steps_used=1,
        ),
    ]
    plan_path = _write_json_plan(tmp_path / "plan.json")
    code = main([
        "plan", "run", str(plan_path),
        "--endpoint", "https://example/v1",
        "--anthropic-api-key", "k",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert code == EXIT_ERROR


def test_run_propagates_max_cost_and_max_time(
    tmp_path: Path, patched_deps,
) -> None:
    plan_path = _write_json_plan(tmp_path / "plan.json")
    main([
        "plan", "run", str(plan_path),
        "--endpoint", "https://example/v1",
        "--anthropic-api-key", "k",
        "--max-cost", "2.5",
        "--max-time-minutes", "5",
        "--output-dir", str(tmp_path / "out"),
    ])
    runner_kwargs = patched_deps["runner_cls"].call_args.kwargs
    assert runner_kwargs["max_cost"] == 2.5
    assert runner_kwargs["max_time_minutes"] == 5
