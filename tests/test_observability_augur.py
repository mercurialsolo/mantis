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
    assert (tmp_path / "screenshots" / "0002_post.png").exists()

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

    events_file = tmp_path / "events" / "0001.jsonl"
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
    events_file = tmp_path / "events" / "0001.jsonl"
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
