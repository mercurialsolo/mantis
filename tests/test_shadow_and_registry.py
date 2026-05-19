"""Tests for #489 (shadow routing) + #490 (model registry).

Covers:

* SplitFacade fans out reads (planner / grounding / verifier) to
  both prod and shadow facades; prod result drives the return;
  shadow result lands on the sink.
* ACTOR role is refused on the shadow path — side-effect
  suppression (shadow models can't double-dispatch).
* Shadow-side exceptions never propagate to prod.
* No shadow facade configured → SplitFacade is a passthrough.
* emit_shadow_disagreement writes a non-committing TrajectoryEvent
  (committed=False) carrying the candidate's model + prompt
  versions for downstream comparison.
* ModelRegistry: register / promote / rollback / current_prod
  contracts. Eval-report-gated promotions to SHADOW/CANARY/PROD.
  Rollback restores prev_prod_version. Idempotent register.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mantis_agent.cua_contracts import (
    InMemoryModelRegistry,
    JSONL_FILENAME,
    ModelCallResult,
    ModelRegistryError,
    ModelRegistryRecord,
    Role,
    RolloutState,
    RoutingMode,
    SplitFacade,
    TrajectoryEmitter,
    emit_shadow_disagreement,
)


# ── #489: SplitFacade fans out reads to both facades ───────────────────


def _result(role: Role, *, mode: RoutingMode, payload: str) -> ModelCallResult:
    return ModelCallResult(
        role=role, routing_mode=mode, payload=payload,
        model_version=f"model-{mode.value}",
        prompt_version="prompt_v1",
    )


def test_split_facade_returns_prod_result_to_caller() -> None:
    """The caller sees the prod facade's result. Shadow is a side
    channel — its return value never replaces prod."""
    prod = MagicMock()
    prod.invoke.return_value = _result(Role.PLANNER, mode=RoutingMode.PROD, payload="prod-out")
    shadow = MagicMock()
    shadow.invoke.return_value = _result(Role.PLANNER, mode=RoutingMode.SHADOW, payload="shadow-out")

    split = SplitFacade(prod=prod, shadow=shadow)
    result = split.invoke(role=Role.PLANNER, payload={"prompt": "x"})

    assert result.payload == "prod-out"
    prod.invoke.assert_called_once()
    shadow.invoke.assert_called_once()


def test_split_facade_passes_routing_mode_shadow_to_shadow_facade() -> None:
    """Whatever routing_mode the caller passes flows to PROD;
    SHADOW always invoked with routing_mode=SHADOW so downstream
    comparison reports group cleanly."""
    prod = MagicMock()
    prod.invoke.return_value = _result(Role.PLANNER, mode=RoutingMode.PROD, payload=None)
    shadow = MagicMock()
    shadow.invoke.return_value = _result(Role.PLANNER, mode=RoutingMode.SHADOW, payload=None)
    split = SplitFacade(prod=prod, shadow=shadow)

    split.invoke(role=Role.PLANNER, payload={}, routing_mode=RoutingMode.PROD)

    assert prod.invoke.call_args.kwargs["routing_mode"] is RoutingMode.PROD
    assert shadow.invoke.call_args.kwargs["routing_mode"] is RoutingMode.SHADOW


def test_split_facade_invokes_sink_with_both_results() -> None:
    prod = MagicMock()
    prod.invoke.return_value = _result(Role.GROUNDING, mode=RoutingMode.PROD, payload="p")
    shadow = MagicMock()
    shadow.invoke.return_value = _result(Role.GROUNDING, mode=RoutingMode.SHADOW, payload="s")
    sink = MagicMock()
    split = SplitFacade(prod=prod, shadow=shadow, shadow_event_sink=sink)

    split.invoke(role=Role.GROUNDING, payload={})

    sink.assert_called_once()
    call = sink.call_args.kwargs
    assert call["role"] is Role.GROUNDING
    assert call["prod_result"].payload == "p"
    assert call["shadow_result"].payload == "s"


# ── #489: ACTOR role refused on shadow path ───────────────────────────


def test_split_facade_refuses_shadow_for_actor_role() -> None:
    """ACTOR shadow would double-dispatch env.step → reject. The
    prod side still runs; the shadow side is silently skipped."""
    prod = MagicMock()
    prod.invoke.return_value = _result(Role.ACTOR, mode=RoutingMode.PROD, payload=None)
    shadow = MagicMock()
    sink = MagicMock()
    split = SplitFacade(prod=prod, shadow=shadow, shadow_event_sink=sink)

    result = split.invoke(role=Role.ACTOR, payload={})

    prod.invoke.assert_called_once()
    shadow.invoke.assert_not_called()
    sink.assert_not_called()
    assert result.routing_mode is RoutingMode.PROD


# ── #489: shadow errors don't propagate ───────────────────────────────


def test_split_facade_swallows_shadow_exception() -> None:
    """Shadow's whole purpose is to learn without affecting prod.
    A broken candidate cannot crash the production path."""
    prod = MagicMock()
    prod.invoke.return_value = _result(Role.PLANNER, mode=RoutingMode.PROD, payload="prod-ok")
    shadow = MagicMock()
    shadow.invoke.side_effect = RuntimeError("candidate crashed")
    sink = MagicMock()

    split = SplitFacade(prod=prod, shadow=shadow, shadow_event_sink=sink)
    result = split.invoke(role=Role.PLANNER, payload={})

    assert result.payload == "prod-ok"
    sink.assert_not_called()


def test_split_facade_swallows_sink_exception() -> None:
    """Sink errors are observability — must never crash prod return."""
    prod = MagicMock()
    prod.invoke.return_value = _result(Role.PLANNER, mode=RoutingMode.PROD, payload="prod-ok")
    shadow = MagicMock()
    shadow.invoke.return_value = _result(Role.PLANNER, mode=RoutingMode.SHADOW, payload=None)
    sink = MagicMock(side_effect=OSError("disk full"))

    split = SplitFacade(prod=prod, shadow=shadow, shadow_event_sink=sink)
    result = split.invoke(role=Role.PLANNER, payload={})

    assert result.payload == "prod-ok"


# ── #489: no shadow facade → passthrough ──────────────────────────────


def test_split_facade_passthrough_when_no_shadow() -> None:
    """A SplitFacade with shadow=None behaves like the prod facade
    alone — useful for environments where shadow isn't configured."""
    prod = MagicMock()
    prod.invoke.return_value = _result(Role.PLANNER, mode=RoutingMode.PROD, payload="passthrough")
    split = SplitFacade(prod=prod, shadow=None)

    result = split.invoke(role=Role.PLANNER, payload={})

    assert result.payload == "passthrough"
    prod.invoke.assert_called_once()


# ── #489: emit_shadow_disagreement writes committed=False event ───────


def test_emit_shadow_disagreement_writes_non_committing_event(tmp_path: Path) -> None:
    emitter = TrajectoryEmitter(run_id="r1", store_dir=str(tmp_path))
    prod_result = ModelCallResult(
        role=Role.PLANNER, routing_mode=RoutingMode.PROD, payload="prod",
        model_version="opus-prod",
    )
    shadow_result = ModelCallResult(
        role=Role.PLANNER, routing_mode=RoutingMode.SHADOW, payload="cand",
        model_version="opus-candidate-v2", prompt_version="prompt_v9",
    )

    ok = emit_shadow_disagreement(
        emitter, role=Role.PLANNER,
        prod_result=prod_result, shadow_result=shadow_result,
        run_id="r1", step_index=3,
    )
    assert ok is True

    record = json.loads((tmp_path / JSONL_FILENAME).read_text().strip())
    assert record["committed"] is False
    # Candidate model + prompt stamps land in versions, role-keyed.
    assert record["versions"]["planner_model"] == "opus-candidate-v2"
    assert record["versions"]["planner_prompt"] == "prompt_v9"
    # Step / action_result document the shadow nature so a reader
    # filtering by ``committed=False`` knows what they got.
    assert record["step"]["action_type"] == "shadow_invocation"
    assert record["action_result"]["dispatched"] is False
    assert "shadow" in record["action_result"]["dispatch_error"].lower()


# ── #490: ModelRegistry — register ────────────────────────────────────


def test_registry_register_creates_candidate_record() -> None:
    reg = InMemoryModelRegistry()
    rec = reg.register(role=Role.PLANNER, version="opus-4-7", artifact="modal/img:abc")
    assert isinstance(rec, ModelRegistryRecord)
    assert rec.role is Role.PLANNER
    assert rec.version == "opus-4-7"
    assert rec.rollout_state is RolloutState.CANDIDATE
    assert rec.eval_report_ref == ""
    assert rec.artifact == "modal/img:abc"


def test_registry_register_rejects_empty_version() -> None:
    reg = InMemoryModelRegistry()
    with pytest.raises(ModelRegistryError, match="version"):
        reg.register(role=Role.PLANNER, version="")


def test_registry_register_is_idempotent() -> None:
    """Deploy pipelines call register on every push — duplicate
    registrations of the same (role, version) must be a no-op."""
    reg = InMemoryModelRegistry()
    a = reg.register(role=Role.GROUNDING, version="haiku-4-5")
    b = reg.register(role=Role.GROUNDING, version="haiku-4-5")
    assert a is b  # same record returned


# ── #490: ModelRegistry — promote ─────────────────────────────────────


def test_registry_promote_to_shadow_requires_eval_report() -> None:
    reg = InMemoryModelRegistry()
    reg.register(role=Role.PLANNER, version="opus-4-7")
    with pytest.raises(ModelRegistryError, match="eval_report_ref"):
        reg.promote(
            role=Role.PLANNER, version="opus-4-7",
            to_state=RolloutState.SHADOW, eval_report_ref="",
        )


def test_registry_promote_to_canary_requires_eval_report() -> None:
    reg = InMemoryModelRegistry()
    reg.register(role=Role.GROUNDING, version="haiku-4-5")
    with pytest.raises(ModelRegistryError, match="eval_report_ref"):
        reg.promote(
            role=Role.GROUNDING, version="haiku-4-5",
            to_state=RolloutState.CANARY, eval_report_ref="",
        )


def test_registry_promote_to_prod_requires_eval_report() -> None:
    reg = InMemoryModelRegistry()
    reg.register(role=Role.VERIFIER, version="haiku-4-5")
    with pytest.raises(ModelRegistryError, match="eval_report_ref"):
        reg.promote(
            role=Role.VERIFIER, version="haiku-4-5",
            to_state=RolloutState.PROD, eval_report_ref="",
        )


def test_registry_promote_to_shadow_with_report() -> None:
    reg = InMemoryModelRegistry()
    reg.register(role=Role.PLANNER, version="opus-4-7")
    rec = reg.promote(
        role=Role.PLANNER, version="opus-4-7",
        to_state=RolloutState.SHADOW,
        eval_report_ref="reports/eval-2026-05-19.html",
    )
    assert rec.rollout_state is RolloutState.SHADOW
    assert rec.eval_report_ref == "reports/eval-2026-05-19.html"


def test_registry_promote_unknown_record_raises() -> None:
    reg = InMemoryModelRegistry()
    with pytest.raises(ModelRegistryError, match="no record"):
        reg.promote(
            role=Role.PLANNER, version="never-registered",
            to_state=RolloutState.SHADOW,
            eval_report_ref="reports/x.html",
        )


def test_registry_promote_to_candidate_rejected() -> None:
    reg = InMemoryModelRegistry()
    reg.register(role=Role.PLANNER, version="opus-4-7")
    with pytest.raises(ModelRegistryError, match="CANDIDATE"):
        reg.promote(
            role=Role.PLANNER, version="opus-4-7",
            to_state=RolloutState.CANDIDATE, eval_report_ref="r",
        )


def test_registry_promote_to_rolled_back_rejected() -> None:
    """Demoting via promote() bypasses the rollback prev-version
    bookkeeping — force the caller through rollback() instead."""
    reg = InMemoryModelRegistry()
    reg.register(role=Role.PLANNER, version="opus-4-7")
    with pytest.raises(ModelRegistryError, match="ROLLED_BACK"):
        reg.promote(
            role=Role.PLANNER, version="opus-4-7",
            to_state=RolloutState.ROLLED_BACK, eval_report_ref="r",
        )


def test_registry_promote_to_prod_displaces_prior_prod() -> None:
    """Promoting a new PROD demotes the old one to ROLLED_BACK and
    stashes its version as prev_prod_version for fast rollback."""
    reg = InMemoryModelRegistry()
    reg.register(role=Role.PLANNER, version="v1")
    reg.promote(
        role=Role.PLANNER, version="v1", to_state=RolloutState.PROD,
        eval_report_ref="report-v1",
    )
    reg.register(role=Role.PLANNER, version="v2")
    new_prod = reg.promote(
        role=Role.PLANNER, version="v2", to_state=RolloutState.PROD,
        eval_report_ref="report-v2",
    )
    assert new_prod.rollout_state is RolloutState.PROD
    assert new_prod.prev_prod_version == "v1"
    # Old prod demoted.
    old = reg.get(role=Role.PLANNER, version="v1")
    assert old is not None
    assert old.rollout_state is RolloutState.ROLLED_BACK


# ── #490: ModelRegistry — rollback ────────────────────────────────────


def test_registry_rollback_restores_prev_prod() -> None:
    reg = InMemoryModelRegistry()
    reg.register(role=Role.PLANNER, version="v1")
    reg.promote(role=Role.PLANNER, version="v1", to_state=RolloutState.PROD, eval_report_ref="r1")
    reg.register(role=Role.PLANNER, version="v2")
    reg.promote(role=Role.PLANNER, version="v2", to_state=RolloutState.PROD, eval_report_ref="r2")
    # Now v2 is prod, v1 is rolled_back, prev_prod_version=v1.

    restored = reg.rollback(role=Role.PLANNER)
    assert restored.version == "v1"
    assert restored.rollout_state is RolloutState.PROD
    # The displaced v2 is now ROLLED_BACK.
    v2 = reg.get(role=Role.PLANNER, version="v2")
    assert v2.rollout_state is RolloutState.ROLLED_BACK


def test_registry_rollback_with_no_prod_raises() -> None:
    reg = InMemoryModelRegistry()
    with pytest.raises(ModelRegistryError, match="no PROD"):
        reg.rollback(role=Role.PLANNER)


def test_registry_rollback_with_no_prev_prod_raises() -> None:
    """First-ever prod has no prev to fall back to — rollback raises
    with a structured error so the caller knows nothing to do."""
    reg = InMemoryModelRegistry()
    reg.register(role=Role.PLANNER, version="v1")
    reg.promote(role=Role.PLANNER, version="v1", to_state=RolloutState.PROD, eval_report_ref="r1")
    with pytest.raises(ModelRegistryError, match="prev_prod_version"):
        reg.rollback(role=Role.PLANNER)


# ── #490: ModelRegistry — listing + current_prod ──────────────────────


def test_registry_current_prod_returns_prod_record() -> None:
    reg = InMemoryModelRegistry()
    reg.register(role=Role.PLANNER, version="v1")
    reg.register(role=Role.PLANNER, version="v2")
    reg.promote(role=Role.PLANNER, version="v1", to_state=RolloutState.PROD, eval_report_ref="r")
    cur = reg.current_prod(role=Role.PLANNER)
    assert cur is not None
    assert cur.version == "v1"


def test_registry_current_prod_returns_none_when_nothing_prod() -> None:
    reg = InMemoryModelRegistry()
    reg.register(role=Role.PLANNER, version="v1")
    assert reg.current_prod(role=Role.PLANNER) is None


def test_registry_list_role_filters_by_role() -> None:
    reg = InMemoryModelRegistry()
    reg.register(role=Role.PLANNER, version="opus-4-7")
    reg.register(role=Role.GROUNDING, version="haiku-4-5")
    reg.register(role=Role.PLANNER, version="opus-4-8")
    planners = reg.list_role(role=Role.PLANNER)
    assert [r.version for r in planners] == ["opus-4-7", "opus-4-8"]


def test_registry_get_returns_none_for_unknown() -> None:
    reg = InMemoryModelRegistry()
    assert reg.get(role=Role.PLANNER, version="never-existed") is None
