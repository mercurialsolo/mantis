"""Tests for #480 (mandatory verdict) + #479 (grounding trace).

Covers:

* The ``StepResult.verdict`` field is populated by the executor's
  :func:`_stamp_verdict` hook from the legacy success/failure_class
  pair via the :func:`verdict_from_step_result` adapter.
* Missing verdicts fail closed: a downstream consumer that reads
  ``step_result.verdict`` must surface ``None`` (not silently
  fabricate one) so the contract violation is caught in tests.
* ``pack_step`` surfaces the typed verdict on every step.
* The canonical event emitter prefers the runner-stamped verdict
  over the inline adapter (so a handler that wires a richer verdict
  upstream wins).
* The ClaudeFormTargetProvider stashes a structured grounding trace
  on every call (success / not_found / bad-coords) and the executor
  emit hook forwards it through to the event.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image

from mantis_agent._anthropic.client import AnthropicToolUseClient
from mantis_agent.cua_contracts import (
    GroundingTrace,
    JSONL_FILENAME,
    SCHEMA_VERSION,
    TrajectoryEmitter,
    Verdict,
    VerdictKind,
    verdict_from_step_result,
)
from mantis_agent.form_targeting.claude import ClaudeFormTargetProvider
from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.gym.result_payload import pack_step
from mantis_agent.gym.run_executor import (
    _emit_canonical_trajectory_event,
    _stamp_verdict,
)
from mantis_agent.plan_decomposer import MicroIntent


# ── Fixtures ────────────────────────────────────────────────────────────


def _ok_result(index: int = 0, data: str = "extracted 7 leads") -> StepResult:
    return StepResult(
        step_index=index, intent="extract", success=True,
        data=data, duration=1.5,
    )


def _fail_result(
    *, index: int = 0, failure_class: str = "selector_miss",
    data: str = "not found",
) -> StepResult:
    return StepResult(
        step_index=index, intent="fill", success=False,
        data=data, duration=2.0, failure_class=failure_class,
    )


def _intent(step_type: str = "click") -> MicroIntent:
    return MicroIntent(intent="Click Sign Up", type=step_type, required=True)


# ── #480: verdict stamping ─────────────────────────────────────────────


def test_stamp_verdict_fills_field_when_unset() -> None:
    r = _ok_result()
    assert r.verdict is None
    _stamp_verdict(r)
    assert isinstance(r.verdict, Verdict)
    assert r.verdict.kind is VerdictKind.OK


def test_stamp_verdict_preserves_handler_stamped_verdict() -> None:
    """A handler that already stamped a richer verdict wins — the
    executor's adapter is a fallback, not an overwrite."""
    rich = Verdict(
        schema_version=SCHEMA_VERSION,
        kind=VerdictKind.RECOVERABLE,
        reason="verifier_disagreed",
        evidence="claude says no_change; xdotool says click ok",
        confidence=0.7,
    )
    r = _ok_result()
    r.verdict = rich
    _stamp_verdict(r)
    assert r.verdict is rich
    assert r.verdict.reason == "verifier_disagreed"


def test_stamp_verdict_failure_routes_to_recoverable_by_default() -> None:
    r = _fail_result(failure_class="selector_miss")
    _stamp_verdict(r)
    assert r.verdict.kind is VerdictKind.RECOVERABLE
    assert r.verdict.reason == "selector_miss"


def test_stamp_verdict_failure_routes_to_non_recoverable_for_known_set() -> None:
    r = _fail_result(failure_class="cf_challenge")
    _stamp_verdict(r)
    assert r.verdict.kind is VerdictKind.NON_RECOVERABLE


def test_stamp_verdict_uses_unknown_when_no_failure_class() -> None:
    """The validator requires a non-empty reason on failure
    verdicts; the adapter back-fills ``unknown`` so the typed verdict
    is always usable downstream."""
    r = _fail_result(failure_class="", data="something failed")
    _stamp_verdict(r)
    assert r.verdict.reason == "unknown"
    assert r.verdict.kind is VerdictKind.RECOVERABLE


# ── #480: result_payload surfaces verdict on every step ────────────────


def test_pack_step_includes_verdict_on_success() -> None:
    r = _ok_result()
    r.verdict = verdict_from_step_result(r)
    out = pack_step(r)
    assert out["verdict"] == {
        "kind": "ok", "reason": "", "evidence": "extracted 7 leads",
        "confidence": 1.0,
    }


def test_pack_step_includes_verdict_on_failure() -> None:
    r = _fail_result(failure_class="cf_challenge", data="403 Forbidden")
    r.verdict = verdict_from_step_result(r)
    out = pack_step(r)
    assert out["verdict"]["kind"] == "non_recoverable"
    assert out["verdict"]["reason"] == "cf_challenge"
    assert out["verdict"]["evidence"] == "403 Forbidden"


def test_pack_step_omits_verdict_when_none() -> None:
    """Ad-hoc callers that bypass the executor leave verdict=None;
    pack_step must round-trip that without raising. The legacy shape
    (success / failure_class / data) keeps working."""
    r = _ok_result()
    assert r.verdict is None
    out = pack_step(r)
    assert "verdict" not in out


# ── #480: missing verdict fails closed in the emit path ────────────────


def test_emit_with_runner_stamped_verdict_prefers_stamped(tmp_path: Path) -> None:
    """The emitter must use the runner-stamped verdict, not re-derive
    from success/failure_class. Confirms #480 acceptance: every
    committed step's typed verdict flows through the canonical event
    stream as the source of truth."""
    emitter = TrajectoryEmitter(run_id="r1", store_dir=str(tmp_path))
    r = _fail_result(failure_class="selector_miss", data="not found")
    # Simulate a handler that stamped a different verdict (e.g. an
    # explicit verifier that disagreed with the legacy projection).
    r.verdict = Verdict(
        schema_version=SCHEMA_VERSION,
        kind=VerdictKind.NON_RECOVERABLE,
        reason="explicit_terminate",
        evidence="verifier said retry would waste budget",
        confidence=0.95,
    )
    emitter.emit(_intent(), r)
    record = json.loads(
        (tmp_path / JSONL_FILENAME).read_text().strip().splitlines()[0],
    )
    # The richer verdict survived — not back-derived from
    # failure_class=selector_miss.
    assert record["verdict"]["kind"] == "non_recoverable"
    assert record["verdict"]["reason"] == "explicit_terminate"
    assert record["verdict"]["evidence"] == "verifier said retry would waste budget"


def test_emit_falls_back_to_adapter_when_runner_did_not_stamp(tmp_path: Path) -> None:
    """When the legacy code path didn't stamp a verdict (callers that
    bypass the executor's _stamp_verdict hook), the emitter MUST
    still produce a valid event — the adapter is the safety net."""
    emitter = TrajectoryEmitter(run_id="r1", store_dir=str(tmp_path))
    r = _ok_result()
    assert r.verdict is None  # confirm precondition
    emitter.emit(_intent(), r)
    record = json.loads(
        (tmp_path / JSONL_FILENAME).read_text().strip().splitlines()[0],
    )
    assert record["verdict"]["kind"] == "ok"


# ── #479: grounding trace from ClaudeFormTargetProvider ────────────────


def _make_provider() -> ClaudeFormTargetProvider:
    """Build a provider with a mock HTTP client so tests don't touch
    the network."""
    client = AnthropicToolUseClient(api_key="dummy", model="claude-haiku-4-5-20251001")
    return ClaudeFormTargetProvider(client)


def _img() -> Image.Image:
    return Image.new("RGB", (100, 100), "white")


def test_grounding_trace_stashed_on_successful_find() -> None:
    provider = _make_provider()
    provider._client.call_with_tool_schema = MagicMock(return_value={
        "x": 120, "y": 220, "action": "click", "value": "",
        "label": "Sign Up button",
    })

    result = provider.find_form_target(
        _img(), "Click Sign Up", target_label="Sign Up",
    )

    assert result is not None
    trace = provider.last_grounding_trace
    assert trace is not None
    assert trace["provider"] == "claude_form_target"
    assert trace["model_version"] == "claude-haiku-4-5-20251001"
    assert trace["target_label"] == "Sign Up button"
    assert trace["coordinates"] == (120, 220)
    assert trace["dispatch_strategy"] == "som_click"


def test_grounding_trace_stashed_on_not_found_with_evidence() -> None:
    """The failure paths are exactly the ones operators need a trace
    for. Confirm not_found surfaces the model's "what I saw" prose."""
    provider = _make_provider()
    provider._client.call_with_tool_schema = MagicMock(return_value={
        "x": 0, "y": 0, "action": "not_found", "value": "",
        "label": "Cloudflare challenge",
    })

    result = provider.find_form_target(_img(), "Click Sign Up")
    assert result is None
    trace = provider.last_grounding_trace
    assert trace is not None
    assert "not_found" in trace["confirmation_evidence"]
    assert "Cloudflare" in trace["confirmation_evidence"]


def test_grounding_trace_stashed_on_empty_tool_use() -> None:
    """API regression / overload returns no parsed payload — the
    trace still records the call happened + why it didn't land."""
    provider = _make_provider()
    provider._client.call_with_tool_schema = MagicMock(return_value=None)

    provider.find_form_target(_img(), "Click Sign Up")
    assert provider.last_grounding_trace["confirmation_evidence"] == "tool_use_empty"


def test_grounding_trace_stashed_on_zero_coordinates() -> None:
    provider = _make_provider()
    provider._client.call_with_tool_schema = MagicMock(return_value={
        "x": 0, "y": 0, "action": "click", "value": "", "label": "x",
    })
    provider.find_form_target(_img(), "Click")
    assert provider.last_grounding_trace["confirmation_evidence"] == "zero_coords"


def test_grounding_trace_type_documents_typed_dict_shape() -> None:
    """GroundingTrace is a TypedDict — total=False means every key is
    optional. Pin the documented key set so a stealth drop is caught."""
    typed_keys = set(GroundingTrace.__annotations__)
    assert typed_keys == {
        "provider", "model_version", "prompt_version", "confidence",
        "dispatch_strategy", "target_label", "coordinates",
        "fallback_chain", "confirmation_evidence",
    }


# ── #479: trace round-trips through the executor emit hook ─────────────


def test_executor_hook_forwards_runner_stashed_trace(
    monkeypatch, tmp_path: Path,
) -> None:
    """A handler that stashed a trace on ``runner._latest_grounding_trace``
    must see it appear in the canonical event's
    ``action_result.grounding_trace``. After emit, the runner's
    stash is cleared so the next step doesn't inherit it."""
    monkeypatch.setenv("MANTIS_CANONICAL_EVENTS_DIR", str(tmp_path))

    class _Runner:
        run_id = "trace_test"

    runner = _Runner()
    runner._latest_grounding_trace = {  # type: ignore[attr-defined]
        "provider": "claude_form_target",
        "model_version": "claude-haiku-4-5-20251001",
        "target_label": "Sign Up",
        "coordinates": (120, 220),
        "dispatch_strategy": "som_click",
    }
    r = _ok_result(index=0)
    _stamp_verdict(r)
    _emit_canonical_trajectory_event(runner, _intent(), r)

    # Trace cleared after emit.
    assert runner._latest_grounding_trace is None  # type: ignore[attr-defined]

    record = json.loads(
        (tmp_path / "trace_test" / JSONL_FILENAME).read_text().strip(),
    )
    trace = record["action_result"]["grounding_trace"]
    assert trace["provider"] == "claude_form_target"
    assert trace["coordinates"] == [120, 220]  # tuple → list via json
    assert trace["dispatch_strategy"] == "som_click"


def test_executor_hook_harvests_from_form_target_provider_stash(
    monkeypatch, tmp_path: Path,
) -> None:
    """Pilot path: when a handler called find_form_target, the
    provider stashes the trace on itself. The executor's emit hook
    auto-harvests from ``runner.form_target_provider.last_grounding_trace``
    even when no explicit ``runner._latest_grounding_trace`` is set.
    Confirms the wiring caught during PR #492's Modal verify (no
    handler-side changes needed in form.py)."""
    monkeypatch.setenv("MANTIS_CANONICAL_EVENTS_DIR", str(tmp_path))

    class _Provider:
        last_grounding_trace = {
            "provider": "claude_form_target",
            "model_version": "claude-haiku-4-5-20251001",
            "target_label": "Login",
            "coordinates": (753, 418),
            "dispatch_strategy": "som_click",
        }

    class _Runner:
        run_id = "provider_stash_test"
        form_target_provider = _Provider()

    runner = _Runner()
    r = _ok_result(index=0)
    _stamp_verdict(r)
    _emit_canonical_trajectory_event(runner, _intent("submit"), r)

    # Provider stash cleared after emit so step N+1 can't inherit it.
    assert runner.form_target_provider.last_grounding_trace is None

    record = json.loads(
        (tmp_path / "provider_stash_test" / JSONL_FILENAME).read_text().strip(),
    )
    trace = record["action_result"]["grounding_trace"]
    assert trace["provider"] == "claude_form_target"
    assert trace["coordinates"] == [753, 418]
    assert trace["target_label"] == "Login"


def test_reset_grounding_trace_stashes_clears_both_stashes() -> None:
    """The pre-step reset must clear both the explicit runner stash
    and the per-provider stash so a leftover trace from step N-1
    can't bleed into step N's canonical event."""
    from mantis_agent.gym.run_executor import reset_grounding_trace_stashes

    class _Provider:
        last_grounding_trace = {"provider": "stale", "target_label": "old"}

    class _Runner:
        _latest_grounding_trace = {"provider": "also_stale"}
        form_target_provider = _Provider()

    runner = _Runner()
    reset_grounding_trace_stashes(runner)
    assert runner._latest_grounding_trace is None
    assert runner.form_target_provider.last_grounding_trace is None


def test_executor_hook_empty_trace_when_runner_did_not_stash(
    monkeypatch, tmp_path: Path,
) -> None:
    """Handlers that don't run grounding (navigate / gate / paginate)
    leave the stash unset — the canonical event must still emit
    cleanly with an empty grounding_trace dict, not None."""
    monkeypatch.setenv("MANTIS_CANONICAL_EVENTS_DIR", str(tmp_path))

    class _Runner:
        run_id = "no_trace_test"

    runner = _Runner()
    r = _ok_result(index=0)
    _stamp_verdict(r)
    _emit_canonical_trajectory_event(runner, _intent("navigate"), r)

    record = json.loads(
        (tmp_path / "no_trace_test" / JSONL_FILENAME).read_text().strip(),
    )
    assert record["action_result"]["grounding_trace"] == {}
