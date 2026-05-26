"""#683 + #684 — producer-side adoption of augur-sdk 0.6.0 per-step fields.

Three discrete additions on top of #680 / #681 (Phase A):

* ``AugurAdapter(task_spec=...)`` forwards to ``DebugSession(task_spec=...)``
  so child fan-out sessions carry the canonical task definition the
  trajectory buffer filters on (#683).
* ``AugurAdapter(brain_model_name=...)`` is stashed at adapter open and
  stamped on every step via ``session.set_step_versions(step_index,
  model=, grounder=, code_git_sha=)`` (#684). Without this, the policy
  registry collapses every run to one signature.
* ``run_fanout_dispatch`` stamps ``_fanout_task_spec`` on the suite so
  the Modal worker entrypoint can pull it back and pass to the child
  ``AugurAdapter``.

Contract pinned here:

* TypeError on the new kwargs strips them AND retries cleanly
  (best-effort observability — never break the run for missing
  metadata fields).
* ``set_step_versions`` is called when the SDK has the method AND we
  have at least one field beyond step_index. The empty-fields case
  skips the call to avoid landing empty captured_versions blocks.
* Children read ``_fanout_task_spec`` off the suite envelope same as
  ``_fanout_group_id``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import mantis_agent.observability.augur as augur_mod
from mantis_agent.gym.fanout_runner import run_fanout_dispatch
from mantis_agent.observability.augur import AugurAdapter
from mantis_agent.plan_decomposer import MicroIntent, MicroPlan, PlanDecomposer


@pytest.fixture
def force_augur_available(monkeypatch):
    """Same shape as the #680 fixture — flip ``_AUGUR_AVAILABLE``
    and stub ``CaptureMode`` so adapter init's pre-DebugSession
    helpers don't crash on CI runners that lack the SDK."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    monkeypatch.setattr(augur_mod, "_AUGUR_AVAILABLE", True)
    monkeypatch.setattr(
        augur_mod, "CaptureMode", lambda v=None: f"capture_mode:{v}",
    )


# ── AugurAdapter forwards task_spec to DebugSession (#683) ─────────


def test_augur_adapter_forwards_task_spec_to_debug_session(force_augur_available):
    """``AugurAdapter(task_spec=...)`` → ``DebugSession(task_spec=...)``."""
    fake_session = MagicMock()
    fake_session._stream = None
    debug_session_cls = MagicMock(return_value=fake_session)
    spec = {"task_spec_id": "boattrader.com.boattrader_scrape_urlnav.v1"}
    with patch.object(augur_mod, "DebugSession", debug_session_cls):
        AugurAdapter(
            run_id="child-rid", tenant_id="t", session_name="s",
            task_spec=spec,
        )
    debug_session_cls.assert_called_once()
    forwarded = debug_session_cls.call_args.kwargs["task_spec"]
    assert forwarded == spec
    # The adapter forwards a dict COPY (defensive) — mutating the
    # original after AugurAdapter() returns mustn't poison the bundle.
    assert forwarded is not spec


def test_augur_adapter_retries_without_task_spec_on_TypeError(force_augur_available):
    """Pre-0.6.0 SDKs reject ``task_spec=`` → retry strips it."""
    fake_session = MagicMock()
    fake_session._stream = None
    debug_session_cls = MagicMock(side_effect=[
        TypeError("got unexpected keyword 'task_spec'"),
        fake_session,
    ])
    with patch.object(augur_mod, "DebugSession", debug_session_cls):
        a = AugurAdapter(
            run_id="r", tenant_id="t", session_name="s",
            task_spec={"task_spec_id": "x.v1"},
        )
    assert debug_session_cls.call_count == 2
    assert "task_spec" in debug_session_cls.call_args_list[0].kwargs
    assert "task_spec" not in debug_session_cls.call_args_list[1].kwargs
    assert a._session is fake_session


def test_augur_adapter_retries_strips_both_kwargs_on_TypeError(force_augur_available):
    """When both ``group_id`` and ``task_spec`` are passed, a single
    TypeError on the new kwargs strips BOTH and retries once."""
    fake_session = MagicMock()
    fake_session._stream = None
    debug_session_cls = MagicMock(side_effect=[
        TypeError("got unexpected keyword 'task_spec'"),
        fake_session,
    ])
    with patch.object(augur_mod, "DebugSession", debug_session_cls):
        AugurAdapter(
            run_id="r", tenant_id="t", session_name="s",
            group_id="g", task_spec={"task_spec_id": "x.v1"},
        )
    assert debug_session_cls.call_count == 2
    first = debug_session_cls.call_args_list[0].kwargs
    assert "group_id" in first and "task_spec" in first
    retry = debug_session_cls.call_args_list[1].kwargs
    assert "group_id" not in retry and "task_spec" not in retry


def test_augur_adapter_omits_task_spec_when_caller_omits(force_augur_available):
    """No ``task_spec=`` arg → kwarg not forwarded to DebugSession.
    (Production single-runner sessions shouldn't carry a task_spec
    they didn't earn.)"""
    fake_session = MagicMock()
    fake_session._stream = None
    debug_session_cls = MagicMock(return_value=fake_session)
    with patch.object(augur_mod, "DebugSession", debug_session_cls):
        AugurAdapter(run_id="r", tenant_id="t", session_name="s")
    assert "task_spec" not in debug_session_cls.call_args.kwargs


# ── set_step_versions stamping (#684) ──────────────────────────────


def _stub_step_result(*, step_index: int, executor_backend: str = "som"):
    return SimpleNamespace(
        step_index=step_index, intent=f"step{step_index}",
        success=True, verdict=SimpleNamespace(kind="ok", reason="", confidence=1.0),
        data="", failure_class="", last_action=None, duration=0.0,
        page_title="", executor_backend=executor_backend, reasoning="",
    )


def test_record_step_stamps_model_grounder_git_sha(tmp_path, monkeypatch):
    """``record_step`` calls ``session.set_step_versions`` with model,
    grounder, code_git_sha — the policy registry's signature inputs."""
    monkeypatch.setenv("MANTIS_GIT_SHA", "abc1234")
    a = AugurAdapter(
        run_id="versions_test", tenant_id="t", session_name="s",
        out_dir=tmp_path, brain_model_name="Holo3-35B-A3B",
    )
    if not a.active:
        pytest.skip("AugurAdapter inactive")
    a._session = MagicMock()
    a._session.record_step = MagicMock()
    a._session.record_step_iteration = MagicMock()
    a._session.set_step_versions = MagicMock()
    a._session.set_loop_detected = MagicMock()
    a.record_step(step_result=_stub_step_result(step_index=2, executor_backend="som"))
    a._session.set_step_versions.assert_called_once_with(
        step_index=3, model="Holo3-35B-A3B", grounder="som", code_git_sha="abc1234",
    )


def test_record_step_omits_step_versions_when_no_signal(tmp_path, monkeypatch):
    """No brain_model_name + no executor_backend + no git_sha → skip
    the ``set_step_versions`` call (empty captured_versions on the
    trace is useless noise)."""
    monkeypatch.delenv("MANTIS_GIT_SHA", raising=False)
    a = AugurAdapter(
        run_id="versions_test", tenant_id="t", session_name="s",
        out_dir=tmp_path,  # no brain_model_name
    )
    if not a.active:
        pytest.skip("AugurAdapter inactive")
    a._session = MagicMock()
    a._session.record_step = MagicMock()
    a._session.set_step_versions = MagicMock()
    a.record_step(step_result=_stub_step_result(step_index=2, executor_backend=""))
    a._session.set_step_versions.assert_not_called()


def test_record_step_step_versions_no_op_on_pre_06_sdk(tmp_path):
    """SDKs without ``set_step_versions`` → method lookup misses,
    stamp is skipped, step trace still recorded."""
    a = AugurAdapter(
        run_id="versions_test", tenant_id="t", session_name="s",
        out_dir=tmp_path, brain_model_name="x",
    )
    if not a.active:
        pytest.skip("AugurAdapter inactive")
    # spec=[...] strips ``set_step_versions`` from the mock — matches
    # the pre-0.6.0 SDK surface.
    a._session = MagicMock(spec=["record_step"])
    a._session.record_step = MagicMock()
    a.record_step(step_result=_stub_step_result(step_index=2))
    a._session.record_step.assert_called_once()


def test_record_step_stamps_only_present_fields(tmp_path, monkeypatch):
    """Partial inputs (model set, grounder + git_sha absent) → kwargs
    forwarded omit the absent fields entirely instead of stamping
    empty strings."""
    monkeypatch.delenv("MANTIS_GIT_SHA", raising=False)
    a = AugurAdapter(
        run_id="versions_test", tenant_id="t", session_name="s",
        out_dir=tmp_path, brain_model_name="Holo3-35B-A3B",
    )
    if not a.active:
        pytest.skip("AugurAdapter inactive")
    a._session = MagicMock()
    a._session.record_step = MagicMock()
    a._session.set_step_versions = MagicMock()
    a.record_step(step_result=_stub_step_result(step_index=2, executor_backend=""))
    a._session.set_step_versions.assert_called_once_with(
        step_index=3, model="Holo3-35B-A3B",
    )


# ── run_fanout_dispatch stamps _fanout_task_spec on suite (#683) ───


def _make_eligible_suite() -> dict:
    plan = MicroPlan(domain="boattrader.com")
    plan.steps = [
        MicroIntent(
            intent="Navigate", type="navigate", section="setup",
            params={"url": "https://www.boattrader.com/boats/by-owner/"},
        ),
        MicroIntent(intent="Click card", type="click", section="extraction"),
        MicroIntent(intent="Read URL", type="extract_url", section="extraction"),
        MicroIntent(intent="Scroll", type="scroll", section="extraction"),
        MicroIntent(intent="Extract", type="extract_data", section="extraction"),
        MicroIntent(intent="Back", type="navigate_back", section="extraction"),
        MicroIntent(
            intent="Inner loop", type="loop", section="extraction",
            loop_target=1, loop_count=40,
        ),
    ]
    PlanDecomposer._classify_loop_groups(plan)
    return {
        "session_name": "x",
        "_plan_name": "boattrader_scrape_urlnav",
        "_plan_signature": "bt-test-sig",
        "_site_config": {"domain": "boattrader.com"},
        "_fanout_phase1_workers": 1,
        "_micro_plan": [
            {
                "intent": s.intent, "type": s.type, "section": s.section,
                "params": dict(s.params or {}),
                "loop_target": s.loop_target, "loop_count": s.loop_count,
                "claude_only": s.claude_only, "gate": s.gate,
                "required": s.required, "grounding": s.grounding,
                "verify": s.verify, "reverse": s.reverse,
                "budget": s.budget, "hints": dict(s.hints or {}),
            }
            for s in plan.steps
        ],
        "_loop_groups": [
            {
                "loop_step_idx": g.loop_step_idx,
                "body_range": list(g.body_range),
                "shape": g.shape,
            }
            for g in plan.loop_groups
        ],
    }


def _make_executor_stub():
    handle_p1 = MagicMock()
    handle_p1.get.return_value = {
        "viable": 0, "leads_with_phone": 0, "leads": [],
        "collected_urls": ["https://x/d/1"], "shared_seen_hits": 0,
    }
    handle_p2 = MagicMock()
    handle_p2.get.return_value = {
        "viable": 1, "leads_with_phone": 0,
        "leads": [{"url": "https://x/d/1"}], "shared_seen_hits": 0,
    }
    executor = MagicMock()
    executor.spawn.side_effect = [handle_p1, handle_p2]
    return executor


def test_run_fanout_dispatch_stamps_task_spec_on_suite(force_augur_available):
    """``_fanout_task_spec`` lands on the suite dict so the Modal
    worker entrypoint can plumb it onto ``MicroRunner._fanout_task_spec``."""
    fake_ctx = MagicMock()
    fake_ctx.__enter__ = MagicMock(return_value=MagicMock())
    fake_ctx.__exit__ = MagicMock(return_value=False)
    stub = MagicMock()
    stub.open_orchestrator = MagicMock(return_value=fake_ctx)
    suite = _make_eligible_suite()
    with patch.object(augur_mod, "DebugSession", stub):
        run_fanout_dispatch(
            suite,
            executor_fn=_make_executor_stub(),
            model="holo3", claude_model="",
            max_steps=10, workers=1,
            fanout_parent_run_id="fanout-683",
        )
    assert "_fanout_task_spec" in suite
    spec = suite["_fanout_task_spec"]
    assert spec["task_class"] == "boattrader.com"
    assert spec["task_spec_id"] == "boattrader.com.boattrader_scrape_urlnav.v1"
