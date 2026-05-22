"""Tests for the #518 cost-reduction sweep additions to the Augur wedge.

Kept in a separate file from ``test_observability_augur.py`` because that
file already runs ~22 cases and is the canonical home for #509's
acceptance criteria — keeping the sweep tests grouped makes it easy to
revert in isolation if a regression surfaces.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from random import Random

import pytest

pytest.importorskip("augur_sdk")

from PIL import Image

from mantis_agent._anthropic.client import (
    _resolve_image_quality,
    encode_screenshot_for_claude,
)
from mantis_agent.observability.augur import AugurAdapter


_PNG = (
    b"\x89PNG\r\n\x1a\n"
    + b"\x00\x00\x00\rIHDR"
    + b"\x00" * 13
    + b"\x00\x00\x00\x00IEND\xaeB\x60\x82"
)


class _FakeStepResult:
    def __init__(self, *, step_index: int = 0, intent: str = "x"):
        self.step_index = step_index
        self.intent = intent
        self.success = True
        self.skip = False
        self.reversed = False
        self.duration = 0.1
        self.failure_class = ""
        self.executor_backend = ""
        self.last_action = None
        self.verdict = None
        self.recovery_decision = None
        self.screenshot_png = _PNG


# ── Item 1: image encoder ───────────────────────────────────────────────


def test_encode_screenshot_jpeg_default_smaller_than_png(monkeypatch):
    """JPEG q=85 default should beat PNG by >50% on realistic
    screenshot data (UI noise + text). PNG wins on flat-color images,
    so we test on a noisy one closer to real browser output."""
    monkeypatch.delenv("MANTIS_CLAUDE_IMAGE_FORMAT", raising=False)
    monkeypatch.delenv("MANTIS_CLAUDE_IMAGE_QUALITY", raising=False)
    rnd = Random(0)
    img = Image.new("RGB", (1440, 900))
    px = img.load()
    for y in range(900):
        for x in range(1440):
            px[x, y] = (
                rnd.randint(200, 255),
                rnd.randint(200, 255),
                rnd.randint(200, 255),
            )
    jpeg_b64, jpeg_mt = encode_screenshot_for_claude(img)
    png_b64, png_mt = encode_screenshot_for_claude(img, format="png")
    assert jpeg_mt == "image/jpeg"
    assert png_mt == "image/png"
    jpeg_size = len(jpeg_b64) * 3 // 4
    png_size = len(png_b64) * 3 // 4
    assert jpeg_size < png_size * 0.5, (
        f"JPEG ({jpeg_size}B) should be <50% of PNG ({png_size}B) on noisy data"
    )


def test_encode_screenshot_preserves_dimensions_for_grounding(monkeypatch):
    """Critical: encoder must NOT downsample. Claude returns click
    coords in input-pixel space and the runner dispatches them against
    the original viewport — any dimension change silently mis-places
    every click."""
    monkeypatch.delenv("MANTIS_CLAUDE_IMAGE_FORMAT", raising=False)
    img = Image.new("RGB", (1440, 900), color="white")
    b64, _ = encode_screenshot_for_claude(img)
    decoded = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert decoded.size == (1440, 900), (
        f"Encoder must preserve dimensions; got {decoded.size}"
    )


def test_encode_screenshot_format_env_override(monkeypatch):
    """``MANTIS_CLAUDE_IMAGE_FORMAT`` opts back into PNG or WEBP when
    callers need different encoding."""
    img = Image.new("RGB", (100, 100), color="white")
    monkeypatch.setenv("MANTIS_CLAUDE_IMAGE_FORMAT", "png")
    _, mt = encode_screenshot_for_claude(img)
    assert mt == "image/png"
    monkeypatch.setenv("MANTIS_CLAUDE_IMAGE_FORMAT", "webp")
    _, mt = encode_screenshot_for_claude(img)
    assert mt == "image/webp"
    # Garbage value falls back to JPEG default
    monkeypatch.setenv("MANTIS_CLAUDE_IMAGE_FORMAT", "tiff")
    _, mt = encode_screenshot_for_claude(img)
    assert mt == "image/jpeg"


def test_encode_screenshot_quality_env_clamped(monkeypatch):
    """Quality is clamped to 1-95; garbage values fall back to 85."""
    monkeypatch.setenv("MANTIS_CLAUDE_IMAGE_QUALITY", "92")
    assert _resolve_image_quality() == 92
    monkeypatch.setenv("MANTIS_CLAUDE_IMAGE_QUALITY", "200")
    assert _resolve_image_quality() == 95
    monkeypatch.setenv("MANTIS_CLAUDE_IMAGE_QUALITY", "0")
    assert _resolve_image_quality() == 1
    monkeypatch.setenv("MANTIS_CLAUDE_IMAGE_QUALITY", "garbage")
    assert _resolve_image_quality() == 85


def test_encode_screenshot_rgba_converts_to_rgb_for_jpeg():
    """JPEG can't carry an alpha channel; the encoder must convert
    RGBA→RGB rather than raising. Xvfb captures land as RGBA on
    some paths."""
    img = Image.new("RGBA", (100, 100), color=(128, 64, 32, 200))
    b64, mt = encode_screenshot_for_claude(img, format="jpeg")
    assert mt == "image/jpeg"
    decoded = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert decoded.size == (100, 100)
    assert decoded.mode == "RGB"


# ── Item 2: per-step costs on StepTrace ────────────────────────────────


def test_step_trace_costs_field_lands_when_passed(monkeypatch, tmp_path: Path):
    """When ``record_step`` is called with a ``costs`` dict, the
    augur-sdk 0.1.6+ ``StepTrace.costs`` field lands on the trace so
    the workspace step-inspector renders per-step USD spend."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(run_id="costs_v1", tenant_id="t", session_name="s", out_dir=tmp_path)
    trace = a._build_step_trace(
        _FakeStepResult(step_index=0),
        "2026-05-19T10:00:00Z", "2026-05-19T10:00:01Z", None, None,
        costs={
            "total_usd": 0.123, "model_usd": 0.09, "gpu_usd": 0.03,
            "proxy_usd": 0.003,
            "tokens_in": 30_000, "tokens_out": 2_000, "cache_hit_tokens": 0,
        },
    )
    assert trace["costs"]["total_usd"] == 0.123
    assert trace["costs"]["tokens_in"] == 30_000
    # Zero-value keys stripped to keep the bundle compact.
    assert "cache_hit_tokens" not in trace["costs"]


def test_step_trace_latency_field_lands_when_passed(monkeypatch, tmp_path: Path):
    """augur-sdk 0.1.6 added ``StepTrace.latency`` alongside
    ``costs``. Same shape conventions — typed dict with per-layer ms,
    zero-value keys stripped."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(run_id="lat_v1", tenant_id="t", session_name="s", out_dir=tmp_path)
    trace = a._build_step_trace(
        _FakeStepResult(step_index=0),
        "2026-05-19T10:00:00Z", "2026-05-19T10:00:01Z", None, None,
        latency={
            "planner_ms": 1200,
            "grounding_ms": 800,
            "dispatch_ms": 350,
            "verifier_ms": 0,  # stripped
            "total_ms": 2350,
        },
    )
    assert trace["latency"]["planner_ms"] == 1200
    assert trace["latency"]["total_ms"] == 2350
    assert "verifier_ms" not in trace["latency"]


def test_step_trace_costs_omitted_for_all_zero_step(monkeypatch, tmp_path: Path):
    """Steps with no cost activity (deterministic navigate / verify)
    shouldn't carry a noisy zero-cost block on the trace."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(run_id="costs_v2", tenant_id="t", session_name="s", out_dir=tmp_path)
    trace = a._build_step_trace(
        _FakeStepResult(step_index=0),
        "2026-05-19T10:00:00Z", "2026-05-19T10:00:01Z", None, None,
        costs={"total_usd": 0.0, "tokens_in": 0, "tokens_out": 0},
    )
    assert "costs" not in trace
    trace2 = a._build_step_trace(
        _FakeStepResult(step_index=0),
        "2026-05-19T10:00:00Z", "2026-05-19T10:00:01Z", None, None,
        costs=None,
    )
    assert "costs" not in trace2


def test_record_cost_metric_supports_per_step_index(monkeypatch, tmp_path: Path):
    """``record_cost_metric`` now accepts a 0-based step_index and bumps
    to Augur's 1-based convention. Defaulting to None still parks the
    event at the run-level minimum (1)."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(run_id="cm_step_v1", tenant_id="t", session_name="s", out_dir=tmp_path)
    # Run-level (default)
    a.record_cost_metric(name="cost_total_usd", value=0.5)
    # Per-step (Mantis step 2 → Augur step 3)
    a.record_cost_metric(name="cost_step_delta_usd", value=0.06, step_index=2)
    a.close(status="completed")
    # Run-level events land in events/0001.jsonl
    run_evs = (tmp_path / "events" / "0001.jsonl").read_text().splitlines()
    assert any('"cost_total_usd"' in ln for ln in run_evs)
    # Per-step lands in events/0003.jsonl
    step_evs = (tmp_path / "events" / "0003.jsonl").read_text().splitlines()
    assert any('"cost_step_delta_usd"' in ln for ln in step_evs)


# ── #519: streaming path runs failure_class.classify() ───────────────────


def test_failure_class_classified_when_step_result_is_unknown(
    monkeypatch, tmp_path: Path,
):
    """#519: ``StepResult.failure_class=""`` or ``"unknown"`` must
    NOT silently ship as ``"unknown"``. Adapter runs the same
    ``failure_class.classify(data, page_title)`` the trace-exporter
    path uses — derives the canonical class (selector_miss /
    no_state_change / etc.) when the symptoms match a known rule.
    Leaving the field absent when classify also returns "unknown"
    lets the Augur viewer render a muted ``?`` chip instead of a
    misleading ``unknown`` literal."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(run_id="fc_v1", tenant_id="t", session_name="s", out_dir=tmp_path)

    # A step whose ``data`` blob matches the ``selector_miss`` rule
    # (substring "not found" / "click_error" / "som-click" etc.)
    # should derive that class even though StepResult left the field
    # blank. ``data`` shape mirrors what real handlers emit.
    sr = _FakeStepResult(step_index=0)
    sr.success = False
    sr.failure_class = ""
    sr.data = "click_error: target not found"
    sr.page_title = "Login | Example"
    trace = a._build_step_trace(
        sr, "2026-05-19T10:00:00Z", "2026-05-19T10:00:01Z", None, None,
    )
    assert trace.get("failure_class") == "selector_miss", (
        f"Expected classify() to derive selector_miss, got {trace.get('failure_class')!r}"
    )

    # And a Cloudflare-titled page should classify even with empty data —
    # documented behaviour of classify() when page_title is set.
    sr_cf = _FakeStepResult(step_index=0)
    sr_cf.success = False
    sr_cf.failure_class = "unknown"  # placeholder string, NOT a real class
    sr_cf.data = ""
    sr_cf.page_title = "Cloudflare | Verify you are human"
    trace_cf = a._build_step_trace(
        sr_cf, "2026-05-19T10:00:00Z", "2026-05-19T10:00:01Z", None, None,
    )
    assert trace_cf.get("failure_class") == "cf_challenge"

    # When the StepResult already has a class, the adapter MUST NOT
    # overwrite it — the producer is the source of truth.
    sr2 = _FakeStepResult(step_index=0)
    sr2.success = False
    sr2.failure_class = "cf_challenge"
    sr2.data = "selector not_found"  # would classify to something else
    sr2.page_title = ""
    trace2 = a._build_step_trace(
        sr2, "2026-05-19T10:00:00Z", "2026-05-19T10:00:01Z", None, None,
    )
    assert trace2["failure_class"] == "cf_challenge"

    # When BOTH StepResult AND classify return unknown, the field is
    # absent from the trace — Augur viewer renders a muted "?" chip
    # instead of a misleading "unknown" literal.
    sr3 = _FakeStepResult(step_index=0)
    sr3.success = False
    sr3.failure_class = "unknown"
    sr3.data = "ambiguous nothing useful here"
    sr3.page_title = ""
    trace3 = a._build_step_trace(
        sr3, "2026-05-19T10:00:00Z", "2026-05-19T10:00:01Z", None, None,
    )
    # Either absent OR a non-"unknown" class is acceptable
    assert trace3.get("failure_class") not in ("unknown",)


# ── Item 3: cache_tools on grounding ───────────────────────────────────


def test_grounding_call_sites_pass_cache_tools():
    """Light AST-ish grep: all three ``call_with_tool_schema`` invocations
    in ``form_targeting/claude.py`` should pass ``cache_tools=True``.
    Catches regressions where someone copies one of the calls without
    the prompt-cache flag."""
    import inspect
    import mantis_agent.form_targeting.claude as ftc
    src = inspect.getsource(ftc)
    # Each invocation block ends with a closing paren on its own line.
    # Easiest reliable check: count opens vs cache_tools=True occurrences.
    opens = src.count("call_with_tool_schema(")
    cache_tokens = src.count("cache_tools=True")
    assert opens >= 3, f"Expected ≥3 grounding call sites, found {opens}"
    assert cache_tokens >= opens, (
        f"Each call_with_tool_schema in form_targeting/claude.py must "
        f"pass cache_tools=True ({opens} calls, {cache_tokens} cache flags)"
    )


# ── #521 + #522: structured cost surfaces (augur-sdk 0.1.8) ─────────────


def _record_step_minimal(adapter: AugurAdapter, step_index: int = 0) -> None:
    """Test helper — drives ``record_step`` with just enough fields so the
    SDK accepts it. Tests below use this to give ``set_step_costs`` a
    target to patch."""
    sr = _FakeStepResult(step_index=step_index)
    sr.success = True
    sr.failure_class = ""
    sr.data = ""
    sr.page_title = ""
    adapter.record_step(
        step_result=sr,
        started_at="2026-05-20T10:00:00Z",
        ended_at="2026-05-20T10:00:01Z",
    )


def test_set_costs_lands_on_manifest_and_trace(monkeypatch, tmp_path: Path):
    """#521: ``set_costs`` writes the structured cost rollup to
    ``manifest.json.costs`` and mirrors it on ``trace.json.session.costs``
    — the canonical surface for Augur's Runs-list COST column as of
    SDK 0.1.8."""
    import json
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(
        run_id="set_costs_v1", tenant_id="t", session_name="s", out_dir=tmp_path,
    )
    a.set_costs(
        total_usd=0.92, model_usd=0.74, gpu_usd=0.15, proxy_usd=0.03,
        tokens_in=42_000, tokens_out=3_500, cache_hit_tokens=8_000,
    )
    a.close(status="completed")
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["costs"]["total_usd"] == 0.92
    assert manifest["costs"]["model_usd"] == 0.74
    assert manifest["costs"]["gpu_usd"] == 0.15
    assert manifest["costs"]["proxy_usd"] == 0.03
    assert manifest["costs"]["tokens_in"] == 42_000
    assert manifest["costs"]["tokens_out"] == 3_500
    assert manifest["costs"]["cache_hit_tokens"] == 8_000
    trace = json.loads((tmp_path / "trace.json").read_text())
    assert trace["session"]["costs"] == manifest["costs"]


def test_set_step_costs_patches_recorded_step_with_index_bump(
    monkeypatch, tmp_path: Path,
):
    """#522: ``set_step_costs`` patches an existing recorded step's
    ``costs`` block. Confirms the adapter bumps Mantis's 0-based step
    index to Augur's 1-based convention at the boundary — passing
    ``step_index=0`` must patch ``steps/0001.json``, not 0000."""
    import json
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(
        run_id="step_costs_v1", tenant_id="t", session_name="s", out_dir=tmp_path,
    )
    _record_step_minimal(a, step_index=0)
    a.set_step_costs(
        0,  # Mantis 0-based — adapter bumps to Augur 1
        total_usd=0.12, model_usd=0.10, tokens_in=2_400, tokens_out=180,
        cache_hit_tokens=600,
    )
    a.close(status="completed")
    step = json.loads((tmp_path / "steps" / "0001.json").read_text())
    assert step["costs"]["total_usd"] == 0.12
    assert step["costs"]["model_usd"] == 0.10
    assert step["costs"]["tokens_in"] == 2_400
    assert step["costs"]["cache_hit_tokens"] == 600


def test_set_costs_and_set_step_costs_noop_when_disabled(
    monkeypatch, tmp_path: Path,
):
    """Both wrappers must be no-ops when the adapter is disabled —
    telemetry never breaks a run. Idempotent across repeat calls."""
    monkeypatch.setenv("MANTIS_AUGUR_DISABLED", "1")
    a = AugurAdapter(
        run_id="noop_v1", tenant_id="t", session_name="s", out_dir=tmp_path,
    )
    assert not a.active
    # Neither call should raise even with adapter disabled.
    a.set_costs(total_usd=0.5, model_usd=0.4)
    a.set_step_costs(0, total_usd=0.1, model_usd=0.08)
    a.set_costs()  # all-None permitted
    # No bundle written.
    assert not (tmp_path / "manifest.json").exists()


def test_run_executor_emits_set_costs_with_token_counts(
    monkeypatch, tmp_path: Path,
):
    """End-to-end-ish: the executor's ``_emit_augur_aggregate_metrics``
    should call ``set_costs`` (not the legacy tag block) with the
    cost-meter snapshot's token counts. Catches regressions where the
    tag-write returns by accident."""
    from unittest.mock import MagicMock

    from mantis_agent.gym.run_executor import RunExecutor

    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    augur = AugurAdapter(
        run_id="exec_set_costs", tenant_id="t", session_name="s", out_dir=tmp_path,
    )
    augur_spy = MagicMock(wraps=augur)
    runner = MagicMock()
    runner._augur = augur_spy
    meter = MagicMock()
    meter.totals.return_value = (0.05, 0.40, 0.02, 0.47)
    meter.elapsed_seconds.return_value = 42.5
    meter.costs = {
        "claude_input_tokens": 12_000,
        "claude_output_tokens": 900,
        "claude_cached_input_tokens": 3_400,
    }
    runner.cost_meter = meter
    executor = RunExecutor.__new__(RunExecutor)
    executor.parent = runner
    executor._emit_augur_aggregate_metrics([])  # results unused for totals
    # Asserts: set_costs was called with the snapshot, NOT the old tag block.
    augur_spy.set_costs.assert_called_once()
    kwargs = augur_spy.set_costs.call_args.kwargs
    assert kwargs["total_usd"] == 0.47
    assert kwargs["model_usd"] == 0.40
    assert kwargs["gpu_usd"] == 0.05
    assert kwargs["proxy_usd"] == 0.02
    assert kwargs["tokens_in"] == 12_000
    assert kwargs["tokens_out"] == 900
    assert kwargs["cache_hit_tokens"] == 3_400
    # No legacy cost_* tags fired — pre-#521 code wrote cost_usd /
    # cost_gpu_usd / cost_claude_usd / cost_proxy_usd / claude_*_tokens.
    tag_keys = {c.args[0] for c in augur_spy.add_tag.call_args_list if c.args}
    legacy_keys = {
        "cost_usd", "cost_gpu_usd", "cost_claude_usd", "cost_proxy_usd",
        "claude_input_tokens", "claude_output_tokens",
        "claude_cached_input_tokens",
    }
    assert tag_keys.isdisjoint(legacy_keys), (
        f"Legacy cost-tag emission re-introduced: {tag_keys & legacy_keys}"
    )
    # elapsed_seconds is still a tag (no schema slot for wallclock).
    assert "elapsed_seconds" in tag_keys


def test_back_allocate_residual_patches_last_step_total(
    monkeypatch, tmp_path: Path,
):
    """When the finalize sync inflates the run total beyond the sum of
    per-step ``total_usd`` rows, the executor should patch the last
    emitted step so the per-step COST column visibly sums to the
    header. Residual lands on ``set_step_costs(last_index,
    total_usd=...)`` with the unchanged buckets preserved by the
    SDK's merge semantics."""
    from unittest.mock import MagicMock

    from mantis_agent.gym.run_executor import RunExecutor

    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    augur = AugurAdapter(
        run_id="back_alloc", tenant_id="t", session_name="s", out_dir=tmp_path,
    )
    augur_spy = MagicMock(wraps=augur)
    runner = MagicMock()
    runner._augur = augur_spy
    # Run total: $1.64 (mirrors the user-reported boattrader run).
    meter = MagicMock()
    meter.totals.return_value = (1.20, 0.40, 0.04, 1.64)
    meter.elapsed_seconds.return_value = 1938.0
    meter.costs = {
        "claude_input_tokens": 20_000,
        "claude_output_tokens": 1_500,
        "claude_cached_input_tokens": 0,
    }
    runner.cost_meter = meter
    # Three steps emitted, last is index=11 (the 12th step).
    runner._emitted_step_costs = [
        (9, {"total_usd": 0.05, "model_usd": 0.03, "gpu_usd": 0.02}),
        (10, {"total_usd": 0.10, "model_usd": 0.05, "gpu_usd": 0.05}),
        (11, {"total_usd": 0.13, "model_usd": 0.07, "gpu_usd": 0.06}),
    ]
    executor = RunExecutor.__new__(RunExecutor)
    executor.parent = runner
    executor._emit_augur_aggregate_metrics([])
    # Residual = 1.64 - (0.05 + 0.10 + 0.13) = 1.36; last step bumps
    # from 0.13 → 1.49 (0.13 + 1.36).
    augur_spy.set_step_costs.assert_called_once()
    call = augur_spy.set_step_costs.call_args
    assert call.args == (11,)  # last emitted step_index
    assert call.kwargs["total_usd"] == pytest.approx(1.49, abs=1e-6)
    # We DON'T touch gpu_usd / model_usd / proxy_usd — the merge
    # leaves them intact so the per-bucket numbers stay honest.
    assert "gpu_usd" not in call.kwargs
    assert "model_usd" not in call.kwargs


def test_back_allocate_residual_noop_when_below_noise_floor(
    monkeypatch, tmp_path: Path,
):
    """Skip the patch when residual < $0.001 — avoids churning the
    last step's row over rounding noise."""
    from unittest.mock import MagicMock

    from mantis_agent.gym.run_executor import RunExecutor

    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    augur = AugurAdapter(
        run_id="back_alloc_noop", tenant_id="t", session_name="s", out_dir=tmp_path,
    )
    augur_spy = MagicMock(wraps=augur)
    runner = MagicMock()
    runner._augur = augur_spy
    meter = MagicMock()
    meter.totals.return_value = (0.10, 0.05, 0.0, 0.1505)
    meter.elapsed_seconds.return_value = 60.0
    meter.costs = {
        "claude_input_tokens": 0, "claude_output_tokens": 0,
        "claude_cached_input_tokens": 0,
    }
    runner.cost_meter = meter
    runner._emitted_step_costs = [
        (0, {"total_usd": 0.10}),
        (1, {"total_usd": 0.05}),  # sum = 0.15; residual = 0.0005
    ]
    executor = RunExecutor.__new__(RunExecutor)
    executor.parent = runner
    executor._emit_augur_aggregate_metrics([])
    augur_spy.set_step_costs.assert_not_called()


def test_back_allocate_residual_noop_when_no_steps_emitted(
    monkeypatch, tmp_path: Path,
):
    """Zero-step halts (e.g. pre-loop failure) leave
    ``_emitted_step_costs`` empty / unset — nothing to patch."""
    from unittest.mock import MagicMock

    from mantis_agent.gym.run_executor import RunExecutor

    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    augur = AugurAdapter(
        run_id="back_alloc_zero", tenant_id="t", session_name="s", out_dir=tmp_path,
    )
    augur_spy = MagicMock(wraps=augur)
    runner = MagicMock()
    runner._augur = augur_spy
    # MagicMock auto-attrs would synthesize _emitted_step_costs as a
    # MagicMock — force it to absent-via-None so the getattr default
    # kicks in. Realistic: runner never reached _emit_augur_step.
    runner._emitted_step_costs = None
    meter = MagicMock()
    meter.totals.return_value = (0.50, 0.0, 0.0, 0.50)
    meter.elapsed_seconds.return_value = 10.0
    meter.costs = {
        "claude_input_tokens": 0, "claude_output_tokens": 0,
        "claude_cached_input_tokens": 0,
    }
    runner.cost_meter = meter
    executor = RunExecutor.__new__(RunExecutor)
    executor.parent = runner
    executor._emit_augur_aggregate_metrics([])
    augur_spy.set_step_costs.assert_not_called()
