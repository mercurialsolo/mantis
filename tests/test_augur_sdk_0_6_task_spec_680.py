"""#680 — augur-sdk 0.6.0 adoption: task_spec, group_id, loop_detected.

The orchestrator session opened by ``open_orchestrator_session`` now
forwards ``task_spec`` (composed from suite metadata) and ``group_id``
(== fan-out parent_run_id) as augur-sdk 0.6.0 kwargs. Children opened
via ``AugurAdapter(group_id=...)`` carry the matching group_id so the
viewer can correlate sibling rollouts (GRPO).

The adapter's ``record_step`` auto-stamps ``set_loop_detected`` when
the step's ``failure_class`` is loop-shaped — bridges the runner-side
detection into the SDK signal without per-call instrumentation.

Contract pinned here:

* ``_build_task_spec_from_suite`` produces a TaskSpec dict with
  ``task_spec_id`` / ``instruction`` / ``task_class`` / ``max_steps``
  / ``env_id`` keys from suite metadata; empty fields are dropped.
* ``open_orchestrator_session`` forwards ``task_spec=`` and
  ``group_id=`` through to the SDK opener via ``**session_kwargs``.
* Older SDKs that raise ``TypeError`` on the new kwargs trigger a
  retry without them (best-effort observability).
* ``AugurAdapter(group_id=...)`` forwards to
  ``DebugSession(group_id=...)``.
* ``AugurAdapter.record_step`` calls
  ``session.set_loop_detected(step_index=...)`` for the documented
  loop failure classes.
* SDK without ``set_loop_detected`` (pre-0.6.0) is a clean no-op.
* ``run_fanout_dispatch`` sets ``_fanout_group_id`` on the suite and
  forwards it as both the orchestrator session's ``group_id`` AND a
  child-readable suite envelope key.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import mantis_agent.observability.augur as augur_mod
from mantis_agent.gym.fanout_runner import (
    _build_task_spec_from_suite,
    run_fanout_dispatch,
)
from mantis_agent.observability.augur import (
    AugurAdapter,
    open_orchestrator_session,
)
from mantis_agent.plan_decomposer import MicroIntent, MicroPlan, PlanDecomposer


@pytest.fixture
def force_augur_available(monkeypatch):
    """CI runners don't install ``augur-sdk`` (it's an opt-in extra).
    Flip the module's SDK-available flag AND stub the imported names
    so the adapter init's pre-DebugSession helpers (notably
    ``_resolve_capture_mode``, which calls ``CaptureMode(...)``) don't
    crash before the test's ``patch.object`` on ``DebugSession`` even
    has a chance to land.
    """
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    monkeypatch.setattr(augur_mod, "_AUGUR_AVAILABLE", True)
    # CaptureMode is a real enum at runtime; on CI without augur-sdk
    # it's None at module load. Use a lambda that returns a sentinel
    # so ``_resolve_capture_mode`` succeeds — the DebugSession mock
    # in each test ignores the actual value.
    monkeypatch.setattr(
        augur_mod, "CaptureMode", lambda v=None: f"capture_mode:{v}",
    )


# ── _build_task_spec_from_suite ────────────────────────────────────


def test_task_spec_full_suite_yields_all_fields():
    """A suite with plan_name + site_config.domain produces a fully
    populated TaskSpec (id + instruction + class + max_steps + env_id)."""
    spec = _build_task_spec_from_suite(
        {
            "_plan_name": "boattrader_scrape_urlnav",
            "_site_config": {"domain": "boattrader.com"},
        },
        phase1_workers=4, phase1_max_pages=4, phase2_workers=4,
        max_steps=240,
    )
    assert spec["task_spec_id"] == "boattrader.com.boattrader_scrape_urlnav.v1"
    assert spec["instruction"] == "boattrader scrape urlnav"
    assert spec["task_class"] == "boattrader.com"
    assert spec["max_steps"] == 240
    assert "fanout(p1=4xp4,p2=4)" in spec["env_id"]
    assert spec["env_id"].startswith("modal:mantis-cua-server:")


def test_task_spec_without_plan_name_drops_id_and_instruction():
    """Missing ``_plan_name`` → task_spec_id / instruction omitted but
    task_class + env_id + max_steps still populated."""
    spec = _build_task_spec_from_suite(
        {"_site_config": {"domain": "example.com"}},
        phase1_workers=2, phase1_max_pages=2, phase2_workers=4,
        max_steps=100,
    )
    assert "task_spec_id" not in spec or spec["task_spec_id"] == "example.com.fanout.v1"
    assert "instruction" not in spec
    assert spec["task_class"] == "example.com"
    assert spec["max_steps"] == 100


def test_task_spec_without_domain_falls_back_to_unknown():
    """No site_config → task_class becomes ``unknown`` and we skip
    the task_spec_id (no useful canonical name to mint)."""
    spec = _build_task_spec_from_suite(
        {"_plan_name": "rogue_plan"},
        phase1_workers=1, phase1_max_pages=1, phase2_workers=1,
        max_steps=10,
    )
    assert "task_class" not in spec
    assert "task_spec_id" not in spec
    assert spec["instruction"] == "rogue plan"
    assert spec["max_steps"] == 10


def test_task_spec_fallback_domain_from_url_pattern():
    """Domain absent on _site_config but URL patterns present → derive
    from the first pattern's host."""
    spec = _build_task_spec_from_suite(
        {
            "_plan_name": "foo",
            "_site_config": {"url_patterns": ["https://mantis-crm/x"]},
        },
        phase1_workers=1, phase1_max_pages=1, phase2_workers=1,
        max_steps=10,
    )
    assert spec["task_class"] == "mantis-crm"
    assert spec["task_spec_id"] == "mantis-crm.foo.v1"


# ── open_orchestrator_session — passes task_spec + group_id ────────


def test_orchestrator_session_forwards_task_spec_and_group_id(force_augur_available):
    """The opener receives ``task_spec`` and ``group_id`` as kwargs."""
    fake_ctx = MagicMock()
    fake_ctx.__enter__ = MagicMock(return_value=MagicMock())
    fake_ctx.__exit__ = MagicMock(return_value=False)
    opener = MagicMock(return_value=fake_ctx)
    stub = MagicMock()
    stub.open_orchestrator = opener
    with patch.object(augur_mod, "DebugSession", stub):
        with open_orchestrator_session(
            run_id="fanout-rid", tags={"phase1_workers": "2"},
            task_spec={"task_spec_id": "x.y.v1", "instruction": "do it"},
            group_id="fanout-rid",
        ):
            pass
    opener.assert_called_once()
    call_kwargs = opener.call_args.kwargs
    assert call_kwargs["task_spec"] == {
        "task_spec_id": "x.y.v1", "instruction": "do it",
    }
    assert call_kwargs["group_id"] == "fanout-rid"


def test_orchestrator_session_retries_without_06_kwargs_on_TypeError(
    force_augur_available,
):
    """An SDK that predates 0.6.0 raises ``TypeError`` on the new
    kwargs. The helper retries WITHOUT them so the parent row still
    opens (server-side grouping via parent_run_id keeps working;
    only the RL-training metadata is dropped)."""
    fake_ctx = MagicMock()
    fake_ctx.__enter__ = MagicMock(return_value=MagicMock())
    fake_ctx.__exit__ = MagicMock(return_value=False)
    # First call (with task_spec/group_id) raises TypeError;
    # retry without them succeeds.
    opener = MagicMock(
        side_effect=[TypeError("got unexpected keyword 'task_spec'"), fake_ctx],
    )
    stub = MagicMock()
    stub.open_orchestrator = opener
    with patch.object(augur_mod, "DebugSession", stub):
        with open_orchestrator_session(
            run_id="r", task_spec={"a": "b"}, group_id="g", tags={},
        ) as s:
            assert s is not None
    assert opener.call_count == 2
    # First call carried the 0.6.0 kwargs.
    first = opener.call_args_list[0].kwargs
    assert "task_spec" in first and "group_id" in first
    # Retry stripped them.
    retry = opener.call_args_list[1].kwargs
    assert "task_spec" not in retry and "group_id" not in retry


def test_orchestrator_session_no_06_kwargs_when_caller_omits(force_augur_available):
    """Caller didn't pass task_spec / group_id → opener call has
    neither key. (Confirms the kwargs are conditional, not stamped
    with ``None`` defaults that the SDK might reject.)"""
    fake_ctx = MagicMock()
    fake_ctx.__enter__ = MagicMock(return_value=MagicMock())
    fake_ctx.__exit__ = MagicMock(return_value=False)
    opener = MagicMock(return_value=fake_ctx)
    stub = MagicMock()
    stub.open_orchestrator = opener
    with patch.object(augur_mod, "DebugSession", stub):
        with open_orchestrator_session(run_id="r", tags={}):
            pass
    kw = opener.call_args.kwargs
    assert "task_spec" not in kw
    assert "group_id" not in kw


# ── AugurAdapter forwards group_id to DebugSession ─────────────────


def test_augur_adapter_forwards_group_id_to_debug_session(force_augur_available):
    """When ``AugurAdapter(group_id=...)`` is passed, the SDK session
    is opened with ``group_id`` as a constructor kwarg."""
    fake_session = MagicMock()
    fake_session._stream = None
    debug_session_cls = MagicMock(return_value=fake_session)
    with patch.object(augur_mod, "DebugSession", debug_session_cls):
        AugurAdapter(
            run_id="child-rid", tenant_id="t", session_name="s",
            group_id="fanout-parent-rid",
        )
    debug_session_cls.assert_called_once()
    kwargs = debug_session_cls.call_args.kwargs
    assert kwargs["group_id"] == "fanout-parent-rid"


def test_augur_adapter_retries_without_group_id_on_TypeError(force_augur_available):
    """Pre-0.6.0 SDKs raise ``TypeError`` on ``group_id``. The adapter
    retries without it so the child session still opens (loses the
    GRPO correlation but keeps the rest of the bundle intact)."""
    fake_session = MagicMock()
    fake_session._stream = None
    debug_session_cls = MagicMock(
        side_effect=[TypeError("got unexpected keyword 'group_id'"), fake_session],
    )
    with patch.object(augur_mod, "DebugSession", debug_session_cls):
        a = AugurAdapter(
            run_id="child-rid", tenant_id="t", session_name="s",
            group_id="fanout-parent-rid",
        )
    assert debug_session_cls.call_count == 2
    # First call carried group_id; retry stripped it.
    assert "group_id" in debug_session_cls.call_args_list[0].kwargs
    assert "group_id" not in debug_session_cls.call_args_list[1].kwargs
    # Adapter still ended up with an active session.
    assert a._session is fake_session


def test_augur_adapter_omits_group_id_when_caller_omits(force_augur_available):
    """No ``group_id=`` on the adapter → the kwarg isn't forwarded to
    DebugSession (production-path preservation; today most runs are
    non-fanout single-runner)."""
    fake_session = MagicMock()
    fake_session._stream = None
    debug_session_cls = MagicMock(return_value=fake_session)
    with patch.object(augur_mod, "DebugSession", debug_session_cls):
        AugurAdapter(run_id="solo-rid", tenant_id="t", session_name="s")
    kw = debug_session_cls.call_args.kwargs
    assert "group_id" not in kw


# ── record_step auto-stamps set_loop_detected ──────────────────────


def _stub_step_result(*, step_index: int, failure_class: str = ""):
    return SimpleNamespace(
        step_index=step_index, intent=f"step{step_index}",
        success=False, verdict=SimpleNamespace(kind="failed", reason="", confidence=1.0),
        data="", failure_class=failure_class, last_action=None, duration=0.0,
        page_title="", executor_backend="", reasoning="",
    )


@pytest.mark.parametrize(
    "fc",
    [
        "no_state_change",
        "brain_loop_exhausted",
        "scroll_no_movement",
        "loop",
        "hard_loop",
        "soft_loop",
    ],
)
def test_record_step_stamps_loop_detected_on_loop_failure_class(tmp_path, fc):
    """For every documented loop failure class, ``record_step`` calls
    ``session.set_loop_detected(step_index=augur_index)`` exactly once
    per step emission."""
    a = AugurAdapter(
        run_id="loop_test", tenant_id="t", session_name="s",
        out_dir=tmp_path,
    )
    if not a.active:
        pytest.skip("AugurAdapter inactive — SDK not installed in this env")
    a._session = MagicMock()
    a._session.record_step = MagicMock()
    a._session.record_step_iteration = MagicMock()
    a._session.set_loop_detected = MagicMock()

    a.record_step(step_result=_stub_step_result(step_index=3, failure_class=fc))

    a._session.set_loop_detected.assert_called_once_with(step_index=4)


def test_record_step_does_not_stamp_loop_detected_on_success(tmp_path):
    """Successful steps (no failure_class) don't trip
    ``set_loop_detected``."""
    a = AugurAdapter(
        run_id="loop_test", tenant_id="t", session_name="s",
        out_dir=tmp_path,
    )
    if not a.active:
        pytest.skip("AugurAdapter inactive")
    a._session = MagicMock()
    a._session.record_step = MagicMock()
    a._session.set_loop_detected = MagicMock()
    a.record_step(step_result=_stub_step_result(step_index=3, failure_class=""))
    a._session.set_loop_detected.assert_not_called()


def test_record_step_skips_loop_detected_on_unrelated_failure_class(tmp_path):
    """Failure classes outside the loop set (selector_miss,
    wrong_target, ...) do NOT trip ``set_loop_detected``."""
    a = AugurAdapter(
        run_id="loop_test", tenant_id="t", session_name="s",
        out_dir=tmp_path,
    )
    if not a.active:
        pytest.skip("AugurAdapter inactive")
    a._session = MagicMock()
    a._session.record_step = MagicMock()
    a._session.set_loop_detected = MagicMock()
    a.record_step(step_result=_stub_step_result(
        step_index=3, failure_class="selector_miss",
    ))
    a._session.set_loop_detected.assert_not_called()


def test_record_step_no_op_when_sdk_lacks_set_loop_detected(tmp_path):
    """Pre-0.6.0 SDKs without ``set_loop_detected`` → stamp call is
    skipped cleanly (no exception, step trace still recorded)."""
    a = AugurAdapter(
        run_id="loop_test", tenant_id="t", session_name="s",
        out_dir=tmp_path,
    )
    if not a.active:
        pytest.skip("AugurAdapter inactive")
    # MagicMock(spec=[...]) strips attrs not in the spec list, so
    # ``set_loop_detected`` is absent — matches pre-0.6.0 SDK shape.
    a._session = MagicMock(spec=["record_step"])
    a._session.record_step = MagicMock()
    # No exception even though set_loop_detected isn't on the session.
    a.record_step(step_result=_stub_step_result(
        step_index=3, failure_class="no_state_change",
    ))
    a._session.record_step.assert_called_once()


# ── run_fanout_dispatch sets _fanout_group_id on the suite ─────────


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
        "_plan_signature": "boattrader-test-sig",
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


def _make_executor_stub(*, phase1_urls: list[str]):
    handle_p1 = MagicMock()
    handle_p1.get.return_value = {
        "viable": 0, "leads_with_phone": 0, "leads": [],
        "collected_urls": phase1_urls, "shared_seen_hits": 0,
    }
    handle_p2 = MagicMock()
    handle_p2.get.return_value = {
        "viable": 1, "leads_with_phone": 0,
        "leads": [{"url": phase1_urls[0]}] if phase1_urls else [],
        "shared_seen_hits": 0,
    }
    executor = MagicMock()
    executor.spawn.side_effect = [handle_p1, handle_p2]
    return executor


def test_run_fanout_dispatch_stamps_fanout_group_id_on_suite(force_augur_available):
    """``run_fanout_dispatch`` sets ``_fanout_group_id`` on the suite
    dict so the worker entrypoint (Modal HTTP path) can forward it to
    ``AugurAdapter.group_id``."""
    fake_ctx = MagicMock()
    fake_ctx.__enter__ = MagicMock(return_value=MagicMock())
    fake_ctx.__exit__ = MagicMock(return_value=False)
    stub = MagicMock()
    stub.open_orchestrator = MagicMock(return_value=fake_ctx)
    suite = _make_eligible_suite()
    with patch.object(augur_mod, "DebugSession", stub):
        run_fanout_dispatch(
            suite,
            executor_fn=_make_executor_stub(phase1_urls=["https://x/d/1"]),
            model="holo3", claude_model="",
            max_steps=10, workers=1,
            fanout_parent_run_id="fanout-678",
        )
    assert suite["_fanout_group_id"] == "fanout-678"


def test_run_fanout_dispatch_forwards_task_spec_and_group_id_to_opener(
    force_augur_available,
):
    """The orchestrator session opens with task_spec + group_id
    derived from the suite."""
    fake_ctx = MagicMock()
    fake_ctx.__enter__ = MagicMock(return_value=MagicMock())
    fake_ctx.__exit__ = MagicMock(return_value=False)
    opener = MagicMock(return_value=fake_ctx)
    stub = MagicMock()
    stub.open_orchestrator = opener
    with patch.object(augur_mod, "DebugSession", stub):
        run_fanout_dispatch(
            _make_eligible_suite(),
            executor_fn=_make_executor_stub(phase1_urls=["https://x/d/1"]),
            model="holo3", claude_model="",
            max_steps=240, workers=4,
            fanout_parent_run_id="fanout-678",
        )
    kw = opener.call_args.kwargs
    assert kw["group_id"] == "fanout-678"
    spec = kw["task_spec"]
    assert spec["task_class"] == "boattrader.com"
    assert spec["task_spec_id"] == "boattrader.com.boattrader_scrape_urlnav.v1"
    assert spec["max_steps"] == 240
