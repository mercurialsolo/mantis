"""Tests for the Augur DebugSession adapter (#509).

Pinning the per-run behavior the issue's acceptance criteria call out:

* Without ``AUGUR_DSN`` set, the adapter still writes a path-stable
  bundle on disk that passes :func:`augur_sdk.validation.validate_bundle`.
* At least one DecisionEvent per Augur layer (``verifier`` /
  ``step_recovery`` / ``grounding``) lands when the corresponding
  Mantis ``_healing_events`` are forwarded.
* Disabling via ``MANTIS_AUGUR_DISABLED=1`` is a clean no-op.
* SDK-side emission failures don't propagate — adapter swallows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("augur_sdk")

from augur_sdk.validation import validate_bundle

from mantis_agent.observability.augur import (
    AugurAdapter,
    default_out_dir,
    is_enabled,
)


# Minimal-but-valid 1×1 PNG so attach_observation has bytes to copy.
_PNG = (
    b"\x89PNG\r\n\x1a\n"
    + b"\x00\x00\x00\rIHDR"
    + b"\x00" * 13
    + b"\x00\x00\x00\x00IEND\xaeB\x60\x82"
)


class _FakeAction:
    action_type = "click"
    text = ""
    selector = ""
    x = 100
    y = 200


class _FakeStepResult:
    """Stand-in for ``mantis_agent.gym.checkpoint.StepResult``."""

    def __init__(
        self,
        *,
        step_index: int = 1,
        intent: str = "Click submit",
        success: bool = True,
        failure_class: str = "",
        skip: bool = False,
        reversed_: bool = False,
        duration: float = 0.42,
    ) -> None:
        self.step_index = step_index
        self.intent = intent
        self.success = success
        self.skip = skip
        self.reversed = reversed_
        self.duration = duration
        self.failure_class = failure_class
        self.executor_backend = "som"
        self.last_action = _FakeAction()
        self.verdict = None
        self.recovery_decision = None
        self.screenshot_png = _PNG


def test_enabled_when_sdk_installed_and_no_opt_out(monkeypatch):
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    assert is_enabled() is True


def test_disabled_via_env(monkeypatch):
    monkeypatch.setenv("MANTIS_AUGUR_DISABLED", "1")
    assert is_enabled() is False
    a = AugurAdapter(run_id="r", tenant_id="t", session_name="s")
    assert a.active is False
    # All public methods are no-ops when disabled
    a.attach_observation(step_index=1, kind="post", png=_PNG)
    a.record_step(step_result=_FakeStepResult())
    a.drain_healing_events([{"layer": "critic-frontier", "step_index": 1}])
    assert a.close(status="completed") is None


def test_default_out_dir_uses_mantis_data_dir(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("MANTIS_AUGUR_DIR", raising=False)
    out = default_out_dir("run_xyz")
    assert out == tmp_path / "augur" / "run_xyz"


def test_default_out_dir_override(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MANTIS_AUGUR_DIR", str(tmp_path / "bundles"))
    out = default_out_dir("run_xyz")
    assert out == tmp_path / "bundles" / "run_xyz"


def test_bundle_writes_and_validates(monkeypatch, tmp_path: Path):
    """The full happy path: 1 step + 3 healing events → valid bundle."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    monkeypatch.delenv("AUGUR_DSN", raising=False)  # no streaming for unit test
    a = AugurAdapter(run_id="unit_v1", tenant_id="t", session_name="s", out_dir=tmp_path)
    assert a.active

    sr = _FakeStepResult(step_index=2)
    post = a.attach_observation(step_index=2, kind="post", png=_PNG)
    assert post and post.startswith("screenshots/")

    a.record_step(
        step_result=sr,
        started_at="2026-05-19T10:00:00Z",
        ended_at="2026-05-19T10:00:01Z",
        observation_post=post,
    )
    a.drain_healing_events([
        {"ts": "2026-05-19T10:00:00Z", "step_index": 2, "layer": "critic-frontier",
         "kind": "fire", "summary": "edit_step", "detail": {"op": "ReplaceStep"}},
        {"ts": "2026-05-19T10:00:00Z", "step_index": 2, "layer": "som-click",
         "kind": "result", "summary": "clicked", "detail": {"x": 100, "y": 200}},
        {"ts": "2026-05-19T10:00:00Z", "step_index": 2, "layer": "agentic-recovery",
         "kind": "decision", "summary": "retry", "detail": {}},
    ])
    manifest = a.close(status="completed")

    assert manifest is not None
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "trace.json").exists()
    # Mantis step_index=2 → Augur 1-based step_index=3 → screenshots/0003_post.png
    assert (tmp_path / "screenshots" / "0003_post.png").exists()

    issues = validate_bundle(tmp_path)
    assert issues == [], f"bundle failed validation: {issues}"


def test_three_augur_layers_land(monkeypatch, tmp_path: Path):
    """AC: at least one verifier / step_recovery / grounding event per bundle."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    monkeypatch.delenv("AUGUR_DSN", raising=False)
    a = AugurAdapter(run_id="layers_v1", tenant_id="t", session_name="s", out_dir=tmp_path)
    a.attach_observation(step_index=1, kind="post", png=_PNG)
    a.record_step(
        step_result=_FakeStepResult(step_index=1),
        started_at="2026-05-19T10:00:00Z",
        ended_at="2026-05-19T10:00:01Z",
    )
    a.drain_healing_events([
        {"step_index": 1, "layer": "critic-frontier", "kind": "fire", "summary": "v"},
        {"step_index": 1, "layer": "som-click", "kind": "result", "summary": "g"},
        {"step_index": 1, "layer": "agentic-recovery", "kind": "decision", "summary": "r"},
    ])
    a.close(status="completed")

    # Mantis step_index=1 → Augur 1-based → events/0002.jsonl
    events_file = tmp_path / "events" / "0002.jsonl"
    assert events_file.exists()
    layers = {
        json.loads(line)["layer"]
        for line in events_file.read_text().splitlines()
        if line.strip()
    }
    assert {"verifier", "step_recovery", "grounding"} <= layers


def test_unknown_layer_falls_back_to_runner(monkeypatch, tmp_path: Path):
    """Unmapped layer strings shouldn't crash — they map to the catch-all."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(run_id="fb_v1", tenant_id="t", session_name="s", out_dir=tmp_path)
    a.record_step(step_result=_FakeStepResult(step_index=1),
                  started_at="2026-05-19T10:00:00Z",
                  ended_at="2026-05-19T10:00:01Z")
    a.drain_healing_events([
        {"step_index": 1, "layer": "completely-made-up", "kind": "decision",
         "summary": "should not crash", "detail": {}},
    ])
    a.close(status="completed")
    # Mantis step_index=1 → Augur 1-based → events/0002.jsonl
    events_file = tmp_path / "events" / "0002.jsonl"
    layers = [json.loads(line)["layer"] for line in events_file.read_text().splitlines() if line.strip()]
    assert layers == ["runner"]


def test_cursor_only_emits_new_events(monkeypatch, tmp_path: Path):
    """Two drain calls on a growing list should emit each event exactly once."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(run_id="cur_v1", tenant_id="t", session_name="s", out_dir=tmp_path)
    healing: list[dict] = []
    a.record_step(step_result=_FakeStepResult(step_index=1),
                  started_at="2026-05-19T10:00:00Z",
                  ended_at="2026-05-19T10:00:01Z")
    healing.append({"step_index": 1, "layer": "critic-frontier", "kind": "fire", "summary": "first"})
    a.drain_healing_events(healing)
    a.record_step(step_result=_FakeStepResult(step_index=2),
                  started_at="2026-05-19T10:00:01Z",
                  ended_at="2026-05-19T10:00:02Z")
    healing.append({"step_index": 2, "layer": "som-click", "kind": "result", "summary": "second"})
    a.drain_healing_events(healing)
    a.close(status="completed")

    events_dir = tmp_path / "events"
    summaries: list[str] = []
    for jsonl in sorted(events_dir.glob("*.jsonl")):
        for line in jsonl.read_text().splitlines():
            if line.strip():
                summaries.append(json.loads(line)["summary"])
    # Each event emitted exactly once, no duplicate from the second drain
    assert summaries.count("first") == 1
    assert summaries.count("second") == 1


def test_non_fatal_on_emit_error(monkeypatch, tmp_path: Path):
    """SDK-side exceptions during emission must not propagate."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(run_id="err_v1", tenant_id="t", session_name="s", out_dir=tmp_path)
    assert a.active

    # Monkey-patch the underlying session so every emission method blows up.
    def boom(*_a, **_k):
        raise RuntimeError("simulated SDK failure")

    a._session.record_step = boom  # type: ignore[assignment]
    a._session.record_event = boom  # type: ignore[assignment]
    a._session.attach_observation = boom  # type: ignore[assignment]

    # None of these should raise.
    assert a.attach_observation(step_index=1, kind="post", png=_PNG) is None
    a.record_step(step_result=_FakeStepResult(step_index=1))
    a.drain_healing_events([{"step_index": 1, "layer": "critic-frontier",
                              "kind": "fire", "summary": "x"}])

    # Close itself should also tolerate a busted session.
    a._session.close = boom  # type: ignore[assignment]
    # set_status path swallowed too:
    a._session.set_status = boom  # type: ignore[assignment]
    assert a.close(status="halted") is None


def test_planner_reasoning_becomes_planner_decision_event(monkeypatch, tmp_path: Path):
    """``record_planner_reasoning`` emits a DecisionEvent with
    layer='planner' / kind='info' / summary=text-prefix / detail.text
    so the workspace's PLANNER REASONING panel populates instead of
    showing demo placeholder text."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(run_id="reasoning_v1", tenant_id="t", session_name="s", out_dir=tmp_path)
    a.record_step(
        step_result=_FakeStepResult(step_index=0),
        started_at="2026-05-20T10:00:00Z",
        ended_at="2026-05-20T10:00:01Z",
    )
    a.record_planner_reasoning(
        step_index=0,
        reasoning="I should click the Login button to authenticate.",
    )
    # Empty / whitespace reasoning is silently dropped
    a.record_planner_reasoning(step_index=0, reasoning="   ")
    a.close(status="completed")

    # Mantis step_index=0 → Augur 1-based → events/0001.jsonl
    events_file = tmp_path / "events" / "0001.jsonl"
    assert events_file.exists()
    lines = [json.loads(ln) for ln in events_file.read_text().splitlines() if ln.strip()]
    planner = [e for e in lines if e.get("layer") == "planner"]
    assert len(planner) == 1, "exactly one planner event from non-empty reasoning"
    assert planner[0]["kind"] == "info"
    assert planner[0]["step_index"] == 1
    assert "Login button" in planner[0]["summary"]
    assert planner[0]["detail"]["text"].startswith("I should")


def test_verbose_flag_gates_diagnostic_logging(monkeypatch, tmp_path: Path, caplog):
    """MANTIS_AUGUR_VERBOSE=1 elevates diagnostics to WARN; default-off
    keeps the log quiet so production runs stay un-noisy."""
    import logging
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)

    # Default: quiet
    monkeypatch.delenv("MANTIS_AUGUR_VERBOSE", raising=False)
    caplog.set_level(logging.WARNING, logger="mantis_agent.observability.augur")
    AugurAdapter(run_id="quiet_v1", tenant_id="t", session_name="s", out_dir=tmp_path / "quiet")
    init_warnings_quiet = [r for r in caplog.records if "AugurAdapter init" in r.message]
    assert init_warnings_quiet == [], "default-off → no AugurAdapter init WARN lines"

    # Verbose: chatty
    caplog.clear()
    monkeypatch.setenv("MANTIS_AUGUR_VERBOSE", "1")
    AugurAdapter(run_id="loud_v1", tenant_id="t", session_name="s", out_dir=tmp_path / "loud")
    init_warnings_loud = [r for r in caplog.records if "AugurAdapter init" in r.message]
    assert init_warnings_loud, "verbose=on → init WARN line emitted"


def test_grounding_emitted_for_click_with_coords(monkeypatch, tmp_path: Path):
    """Click action with x/y coords + executor_backend='som' lands a
    Grounding dict on the StepTrace so the workspace's per-step
    GROUNDING panel populates instead of saying 'No grounding'."""
    from types import SimpleNamespace
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(run_id="ground_v1", tenant_id="t", session_name="s", out_dir=tmp_path)
    sr = _FakeStepResult(step_index=0, intent="Click the Send button")
    sr.executor_backend = "som"
    sr.last_action = SimpleNamespace(action_type="click", x=300, y=420)
    trace = a._build_step_trace(sr, "2026-05-19T10:00:00Z", "2026-05-19T10:00:01Z", None, None)
    assert "grounding" in trace
    g = trace["grounding"]
    assert g["provider"] == "mantis-som-cdp"
    assert g["coordinates"] == {"x": 300.0, "y": 420.0}
    assert g["confidence"] == 0.99  # SoM = near-certain
    assert g["provenance"] == "screenshot"
    assert "Send button" in g["target_label"]


def test_grounding_omitted_for_nav_step(monkeypatch, tmp_path: Path):
    """Navigate / verify / extract steps shouldn't carry grounding —
    they're not coordinate-anchored. Omitting the field keeps the
    SDK's TypedDict ``total=False`` semantics happy."""
    from types import SimpleNamespace
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(run_id="ground_nav_v1", tenant_id="t", session_name="s", out_dir=tmp_path)
    sr = _FakeStepResult(step_index=0, intent="Navigate to https://example.com")
    sr.executor_backend = ""
    sr.last_action = SimpleNamespace(action_type="navigate", url="https://example.com")
    trace = a._build_step_trace(sr, "2026-05-19T10:00:00Z", "2026-05-19T10:00:01Z", None, None)
    assert "grounding" not in trace


def test_grounding_fires_when_backend_set_even_without_last_action(monkeypatch, tmp_path: Path):
    """The Mantis runner doesn't stamp ``StepResult.last_action`` on
    most code paths — only ``executor_backend`` is reliable. Adapter
    must emit Grounding whenever the backend is set; coordinates are
    optional (omitted cleanly when missing)."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(run_id="ground_no_last_v1", tenant_id="t", session_name="s", out_dir=tmp_path)
    sr = _FakeStepResult(step_index=0, intent="Click the Sign In button")
    sr.executor_backend = "som"
    sr.last_action = None  # Mantis's common case
    trace = a._build_step_trace(
        sr, "2026-05-19T10:00:00Z", "2026-05-19T10:00:01Z",
        None, None, step_type="click",
    )
    assert "grounding" in trace, "backend=som should emit grounding even without last_action"
    g = trace["grounding"]
    assert g["provider"] == "mantis-som-cdp"
    assert "Sign In" in g["target_label"]
    assert g["confidence"] == 0.99
    # Coordinates omitted when last_action isn't populated — schema is total=False
    assert "coordinates" not in g
    # action.type falls back to step_type when last_action.action_type isn't there
    assert trace["action"]["type"] == "click"
    assert trace["step_type"] == "click"


def test_step_type_fallback_when_no_last_action(monkeypatch, tmp_path: Path):
    """Without ``step_type`` AND without ``last_action.action_type``,
    action.type defaults to 'unknown'. That's the empty-state behaviour
    — workspace renders ``unknown`` for that, but we don't crash."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(run_id="step_type_v1", tenant_id="t", session_name="s", out_dir=tmp_path)
    sr = _FakeStepResult(step_index=0)
    sr.executor_backend = ""
    sr.last_action = None
    trace = a._build_step_trace(
        sr, "2026-05-19T10:00:00Z", "2026-05-19T10:00:01Z",
        None, None,  # no step_type kw
    )
    assert trace["action"]["type"] == "unknown"
    assert trace["step_type"] == "unknown"
    assert "grounding" not in trace  # no backend = no grounding


def test_grounding_reads_from_real_action_params_dict(monkeypatch, tmp_path: Path):
    """Mantis ``actions.Action`` is a dataclass with a ``params`` dict
    — x/y/text/etc. live INSIDE ``params``, not as top-level attrs.
    The wedge must read from there or grounding silently drops to
    None (the bug that hid behind the empty workspace GROUNDING
    panel through one verify cycle)."""
    from mantis_agent.actions import Action, ActionType
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(run_id="ground_real_v1", tenant_id="t", session_name="s", out_dir=tmp_path)
    sr = _FakeStepResult(step_index=0, intent="Click submit")
    sr.executor_backend = "som"
    # Real Mantis Action shape: coords go in ``params``, not on the
    # dataclass directly. The wedge MUST surface these for grounding
    # to fire on production click steps.
    sr.last_action = Action(action_type=ActionType.CLICK, params={"x": 512, "y": 320})
    trace = a._build_step_trace(sr, "2026-05-19T10:00:00Z", "2026-05-19T10:00:01Z", None, None)
    assert "grounding" in trace, "real Action.params dict must surface coords"
    assert trace["grounding"]["coordinates"] == {"x": 512.0, "y": 320.0}
    # action.params on the trace should also have the coords echoed
    assert trace["action"]["params"]["x"] == 512
    assert trace["action"]["params"]["y"] == 320


def test_grounding_vision_backend_marks_lower_confidence(monkeypatch, tmp_path: Path):
    """Vision-grounded clicks (brain's best guess) should report
    lower confidence than SoM-anchored (CDP-verified) ones."""
    from types import SimpleNamespace
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(run_id="ground_v2_v1", tenant_id="t", session_name="s", out_dir=tmp_path)
    sr = _FakeStepResult(step_index=0, intent="Click submit")
    sr.executor_backend = "vision"
    sr.last_action = SimpleNamespace(action_type="click", x=100, y=200)
    trace = a._build_step_trace(sr, "2026-05-19T10:00:00Z", "2026-05-19T10:00:01Z", None, None)
    assert trace["grounding"]["provider"] == "mantis-xdotool-vision"
    assert trace["grounding"]["confidence"] == 0.7


def test_add_tag_surfaces_on_session(monkeypatch, tmp_path: Path):
    """``add_tag`` writes to the session so the workspace can render
    chips like MODEL in the Runs list."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(run_id="tag_v1", tenant_id="t", session_name="s", out_dir=tmp_path)
    a.add_tag("model", "Holo3-35B-A3B")
    a.add_tag("failure_class", "selector_miss")
    # Empty key is silently dropped
    a.add_tag("", "ignored")
    a.close(status="completed")

    manifest = json.loads((tmp_path / "trace.json").read_text())
    tags = manifest.get("session", {}).get("tags", {})
    assert tags.get("model") == "Holo3-35B-A3B"
    assert tags.get("failure_class") == "selector_miss"


def test_append_log_forwards_to_sdk_when_available(monkeypatch, tmp_path: Path):
    """``append_log`` delegates to ``DebugSession.append_log`` when the
    SDK exposes it (0.1.3+). Empty text + no-active adapter are no-ops."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(run_id="log_v1", tenant_id="t", session_name="s", out_dir=tmp_path)
    assert a.active

    captured: list[tuple[str, dict]] = []

    def _spy(text, *, step_index=None, name="run"):
        captured.append((text, {"step_index": step_index, "name": name}))

    a._session.append_log = _spy  # type: ignore[assignment]
    a.append_log("runner-line 1")
    a.append_log("step 3 reasoning", step_index=3, name="reasoning")
    a.append_log("")  # empty → silent no-op
    a.close(status="completed")

    assert captured == [
        ("runner-line 1", {"step_index": None, "name": "run"}),
        ("step 3 reasoning", {"step_index": 3, "name": "reasoning"}),
    ]


def test_append_log_silent_no_op_when_sdk_lacks_method(monkeypatch, tmp_path: Path):
    """When the SDK install is older than 0.1.3 (no ``append_log`` on
    DebugSession), the adapter wrapper drops the call silently
    rather than raising AttributeError."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    from augur_sdk import DebugSession as _DS
    # Simulate the older SDK by removing the method from the class for
    # the duration of this test. ``monkeypatch.delattr`` restores on
    # teardown so the rest of the suite still sees the real method.
    monkeypatch.delattr(_DS, "append_log", raising=False)
    a = AugurAdapter(run_id="log_v2", tenant_id="t", session_name="s", out_dir=tmp_path)
    # Should not raise.
    a.append_log("anything")
    a.close(status="completed")


def test_cost_metric_emits_runner_metric_decision_event(monkeypatch, tmp_path: Path):
    """``record_cost_metric`` emits a layer='runner' / kind='metric'
    DecisionEvent so the workspace's COST column can derive totals
    from the standard SDK surface (no custom server-side parsing)."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(run_id="cost_v1", tenant_id="t", session_name="s", out_dir=tmp_path)
    a.record_cost_metric(
        name="cost_total_usd",
        value=2.345,
        detail={"gpu_usd": 0.5, "claude_usd": 1.8, "proxy_usd": 0.045,
                "elapsed_seconds": 120.5, "steps_executed": 9},
    )
    a.close(status="completed")

    # step_index=1 is the synthetic run-level slot
    events_file = tmp_path / "events" / "0001.jsonl"
    assert events_file.exists()
    lines = [json.loads(ln) for ln in events_file.read_text().splitlines() if ln.strip()]
    metrics = [e for e in lines if e.get("kind") == "metric"]
    assert len(metrics) == 1
    m = metrics[0]
    assert m["layer"] == "runner"
    assert m["detail"]["name"] == "cost_total_usd"
    assert m["detail"]["value"] == 2.345
    assert m["detail"]["claude_usd"] == 1.8
    assert m["detail"]["steps_executed"] == 9


def test_step_index_and_attempt_are_clamped_to_augur_minimums(monkeypatch, tmp_path: Path):
    """Augur's StepTrace schema requires step_index>=1 and
    recovery_decision.attempt>=1. Mantis is 0-based for both; the
    adapter must bump on the way out so per-step PUTs aren't silently
    rejected with HTTP 422 ``"step: 0 is less than the minimum of 1"``
    (the actual symptom from the #509 verification cycle).
    """
    from types import SimpleNamespace
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(run_id="bump_v1", tenant_id="t", session_name="s", out_dir=tmp_path)
    # 0-based attempt — typical Mantis RecoveryDecision shape
    sr = _FakeStepResult(step_index=0)
    sr.recovery_decision = SimpleNamespace(type="retry", reason="x", attempt=0)
    trace = a._build_step_trace(sr, "2026-05-19T10:00:00Z", "2026-05-19T10:00:01Z", None, None)
    assert trace["step_index"] == 1, "Mantis 0 → Augur 1"
    # #659: step_id carries the per-emission suffix so loop-iterated
    # dispatches don't collapse on the Augur server. First emission
    # for step_index=0 → ``step-0001-000``.
    assert trace["step_id"] == "step-0001-000"
    assert trace["recovery_decision"]["attempt"] == 1, "0 clamped to schema minimum"

    # Observation paths should also be 1-based (line up with step_id)
    post = a.attach_observation(step_index=0, kind="post", png=_PNG)
    assert post == "screenshots/0001_post.png"
