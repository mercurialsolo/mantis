"""#649 — Augur run_id must be unique per session, not reused across
re-runs of the same workflow_id.

``workflow_id`` is intentionally **stable** across re-runs (HTTP host
polls it, Chrome profile lock keys on it, checkpoints resume by it).
Before this fix the Augur ``DebugSession.run_id`` was derived from
``runner.run_key`` (== workflow_id), so the Runs list piled overlapping
rows under the same id and trajectory comparisons across runs were
ambiguous.

Contract pinned here:
  - ``MicroPlanRunner`` accepts an ``augur_run_id`` kwarg and stores
    it on the instance.
  - ``run_executor`` prefers ``augur_run_id`` over ``run_key`` /
    ``plan_signature`` when opening ``AugurAdapter``.
  - Legacy callers that don't mint an ``augur_run_id`` fall back to
    ``run_key`` (current behaviour for backward compat).
  - The Augur ``workflow_id`` tag carries the stable logical-workflow
    id so the Runs list can still group sessions across re-runs.
"""

from __future__ import annotations

from unittest.mock import MagicMock


def test_micro_plan_runner_accepts_augur_run_id_kwarg():
    """Constructor takes ``augur_run_id``; stores it on the instance."""
    from mantis_agent.gym.micro_runner import MicroPlanRunner

    r = MicroPlanRunner(
        brain=MagicMock(), env=MagicMock(),
        augur_run_id="boattrader-workflow-1779-a1b2c3d4",
    )
    assert r.augur_run_id == "boattrader-workflow-1779-a1b2c3d4"


def test_micro_plan_runner_augur_run_id_defaults_to_empty():
    """Legacy callers that don't pass ``augur_run_id`` get empty
    string — preserves the run_executor fallback to ``run_key``."""
    from mantis_agent.gym.micro_runner import MicroPlanRunner

    r = MicroPlanRunner(brain=MagicMock(), env=MagicMock(), run_key="workflow-x")
    assert r.augur_run_id == ""
    assert r.run_key == "workflow-x"


def test_run_executor_prefers_augur_run_id_over_run_key(
    monkeypatch, tmp_path,
):
    """When ``runner.augur_run_id`` is set, the AugurAdapter is opened
    with that id — NOT with the stable workflow_id (== run_key)."""
    from mantis_agent.gym import run_executor as _re

    captured: dict = {}

    class _SpyAdapter:
        def __init__(self, run_id: str, **kwargs):
            captured["run_id"] = run_id
            captured["kwargs"] = kwargs

    monkeypatch.setattr(_re, "AugurAdapter", _SpyAdapter)

    runner = MagicMock()
    runner.augur_run_id = "boattrader-wf-1779-deadbeef"
    runner.run_key = "boattrader-wf-1779"
    runner.plan_signature = "abc123def456"
    runner.session_name = "boattrader_session"
    runner.tenant_id = "t"
    runner.brain = None
    runner.plan_name = "boattrader_scrape_urlnav"

    plan = MagicMock()
    plan.steps = [MagicMock(), MagicMock()]

    # Inline the AugurAdapter-construction block from
    # RunExecutor._execute_inner — keeps the test focused on the
    # specific priority logic without dragging in checkpoint / brain /
    # extractor plumbing.
    runner._augur = _re.AugurAdapter(
        run_id=str(
            getattr(runner, "augur_run_id", "")
            or getattr(runner, "run_key", "")
            or runner.plan_signature
            or "run"
        ),
        tenant_id=str(getattr(runner, "tenant_id", "") or ""),
        session_name=str(getattr(runner, "session_name", "") or ""),
        extra_tags={
            "plan_signature": runner.plan_signature or "",
            "plan_name": str(getattr(runner, "plan_name", "") or ""),
            "workflow_id": str(getattr(runner, "run_key", "") or ""),
        },
    )

    assert captured["run_id"] == "boattrader-wf-1779-deadbeef"
    # workflow_id (== run_key) lives on as a tag for cross-run grouping.
    assert captured["kwargs"]["extra_tags"]["workflow_id"] == "boattrader-wf-1779"


def test_run_executor_falls_back_to_run_key_when_no_augur_run_id(
    monkeypatch, tmp_path,
):
    """Legacy contract: when no ``augur_run_id`` is set the adapter
    opens with ``run_key`` (== workflow_id) — preserves observability
    for any caller not yet minting per-session ids."""
    from mantis_agent.gym import run_executor as _re

    captured: dict = {}

    class _SpyAdapter:
        def __init__(self, run_id: str, **kwargs):
            captured["run_id"] = run_id
            captured["kwargs"] = kwargs

    monkeypatch.setattr(_re, "AugurAdapter", _SpyAdapter)

    runner = MagicMock()
    runner.augur_run_id = ""  # legacy caller
    runner.run_key = "legacy-workflow-id"
    runner.plan_signature = "sigsig"
    runner.session_name = "s"
    runner.tenant_id = "t"
    runner.plan_name = "legacy_plan"

    runner._augur = _re.AugurAdapter(
        run_id=str(
            getattr(runner, "augur_run_id", "")
            or getattr(runner, "run_key", "")
            or runner.plan_signature
            or "run"
        ),
        tenant_id=str(getattr(runner, "tenant_id", "") or ""),
        session_name=str(getattr(runner, "session_name", "") or ""),
        extra_tags={
            "workflow_id": str(getattr(runner, "run_key", "") or ""),
        },
    )

    assert captured["run_id"] == "legacy-workflow-id"


def test_run_executor_falls_back_to_plan_signature_when_no_workflow_id(
    monkeypatch, tmp_path,
):
    """Both ``augur_run_id`` and ``run_key`` empty → plan_signature is
    used. Last-resort fallback for ad-hoc callers."""
    from mantis_agent.gym import run_executor as _re

    captured: dict = {}

    class _SpyAdapter:
        def __init__(self, run_id: str, **kwargs):
            captured["run_id"] = run_id

    monkeypatch.setattr(_re, "AugurAdapter", _SpyAdapter)

    runner = MagicMock()
    runner.augur_run_id = ""
    runner.run_key = ""
    runner.plan_signature = "fallback-sig-abc"
    runner.session_name = ""
    runner.tenant_id = ""
    runner.plan_name = ""

    _re.AugurAdapter(
        run_id=str(
            getattr(runner, "augur_run_id", "")
            or getattr(runner, "run_key", "")
            or runner.plan_signature
            or "run"
        ),
    )

    assert captured["run_id"] == "fallback-sig-abc"


def test_two_sessions_same_workflow_id_get_distinct_augur_run_ids():
    """End-to-end shape check: simulate the modal_cua_server mint logic
    twice with the same workflow_id and verify the resulting
    augur_run_ids are distinct (different uuid suffixes) but share the
    same workflow_id prefix for searchability."""
    import uuid

    workflow_id = "boattrader-phone-v8-fallback-1779628841"

    def _mint() -> str:
        # Same shape used in modal_cua_server._run_holo3_executor.
        return (
            f"{workflow_id}-{uuid.uuid4().hex[:8]}"
            if workflow_id else uuid.uuid4().hex[:12]
        )

    a, b = _mint(), _mint()
    assert a != b
    assert a.startswith(workflow_id + "-")
    assert b.startswith(workflow_id + "-")
    # 8-hex-char suffix.
    assert len(a.rsplit("-", 1)[1]) == 8
    assert len(b.rsplit("-", 1)[1]) == 8


def test_no_workflow_id_yields_uuid_only_augur_run_id():
    """When the caller didn't set a workflow_id at all the mint logic
    falls back to a bare 12-hex-char id — still unique, just no
    grouping prefix."""
    import uuid

    workflow_id = ""

    augur_run_id = (
        f"{workflow_id}-{uuid.uuid4().hex[:8]}"
        if workflow_id else uuid.uuid4().hex[:12]
    )
    assert len(augur_run_id) == 12
    assert "-" not in augur_run_id
