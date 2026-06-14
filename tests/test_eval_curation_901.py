"""#901 — producer-side eval curation: task_spec + success_conditions + mark_for_eval.

Pins the three producer gaps:
1. general/micro/prod runs compose a fallback TaskSpec (was: only fan-out).
2. TaskSpecs carry ``success_conditions`` (was: never populated).
3. ``AugurAdapter.mark_for_eval`` forwards to the SDK (was: never called).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import mantis_agent.observability.augur as augur_mod
from mantis_agent.gym.fanout_runner import _build_task_spec_from_suite
from mantis_agent.observability.augur import AugurAdapter
from mantis_agent.observability.eval_curation import (
    compose_task_spec,
    default_success_conditions,
    task_spec_from_runner,
)


@pytest.fixture
def force_augur_available(monkeypatch):
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    monkeypatch.setattr(augur_mod, "_AUGUR_AVAILABLE", True)
    monkeypatch.setattr(augur_mod, "CaptureMode", lambda v=None: f"capture_mode:{v}")


# ── success_conditions + compose ───────────────────────────────────


def test_default_success_conditions():
    assert default_success_conditions() == [{"type": "task_success"}]


def test_compose_drops_empty_keys_and_defaults_success():
    spec = compose_task_spec(task_spec_id="d.p.v1")
    assert spec == {
        "task_spec_id": "d.p.v1",
        "success_conditions": [{"type": "task_success"}],
    }


def test_compose_includes_provided_fields():
    spec = compose_task_spec(
        task_spec_id="d.p.v1", instruction="do it", task_class="d",
        env_id="mantis:s", max_steps=12,
        success_conditions=[{"type": "url_contains", "value": "/secure"}],
    )
    assert spec["instruction"] == "do it"
    assert spec["max_steps"] == 12
    assert spec["success_conditions"] == [{"type": "url_contains", "value": "/secure"}]


# ── task_spec_from_runner (gap 1) ──────────────────────────────────


def _runner(plan_name, session_name="sess"):
    return SimpleNamespace(plan_name=plan_name, session_name=session_name, plan_signature="sig")


def _plan(domain="", n_steps=3):
    steps = [SimpleNamespace(intent=f"step {i}") for i in range(n_steps)]
    return SimpleNamespace(domain=domain, steps=steps)


def test_runner_spec_with_domain_and_plan_name():
    spec = task_spec_from_runner(_runner("the_login_flow"), _plan(domain="herokuapp.com", n_steps=4))
    assert spec["task_spec_id"] == "herokuapp.com.the_login_flow.v1"
    assert spec["instruction"] == "the login flow"
    assert spec["task_class"] == "herokuapp.com"
    assert spec["env_id"] == "mantis:sess"
    assert spec["max_steps"] == 4
    assert spec["success_conditions"] == [{"type": "task_success"}]


def test_runner_spec_without_domain():
    spec = task_spec_from_runner(_runner("solo_plan"), _plan(domain=""))
    assert spec["task_spec_id"] == "solo_plan.v1"
    assert "task_class" not in spec


def test_runner_spec_none_without_plan_name():
    # health / trigger / ad-hoc runs → no canonical task → not eval-eligible
    assert task_spec_from_runner(_runner(""), _plan(domain="x.com")) is None


# ── fanout builder now carries success_conditions (gap 2) ──────────


def test_fanout_task_spec_includes_success_conditions():
    spec = _build_task_spec_from_suite(
        {"_plan_name": "p", "_site_config": {"domain": "x.com"}},
        phase1_workers=2, phase1_max_pages=2, phase2_workers=2, max_steps=50,
    )
    assert spec["success_conditions"] == [{"type": "task_success"}]


# ── AugurAdapter.mark_for_eval (gap 3) ─────────────────────────────


def _adapter_with_fake_session(force_augur_available, session):
    cls = MagicMock(return_value=session)
    with patch.object(augur_mod, "DebugSession", cls):
        return AugurAdapter(run_id="r", tenant_id="t", session_name="s")


def test_mark_for_eval_forwards_to_session(force_augur_available):
    session = MagicMock()
    session._stream = None
    a = _adapter_with_fake_session(force_augur_available, session)
    assert a.active
    a.mark_for_eval(step_index=4, reason="representative_success:x.p.v1")
    session.mark_for_eval.assert_called_once_with(
        step_index=4, reason="representative_success:x.p.v1",
    )


def test_mark_for_eval_noop_when_sdk_lacks_method(force_augur_available):
    # pre-API SDK: session has no mark_for_eval → clean no-op, no raise
    session = MagicMock(spec=["close", "set_status"])
    session._stream = None
    a = _adapter_with_fake_session(force_augur_available, session)
    a.mark_for_eval(reason="x")  # must not raise


def test_mark_for_eval_noop_when_inactive():
    # No SDK / disabled → inactive adapter → no-op (no session to call)
    a = AugurAdapter(run_id="r", tenant_id="t", session_name="s")
    if a.active:
        pytest.skip("adapter unexpectedly active")
    a.mark_for_eval(reason="x")  # must not raise
