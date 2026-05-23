"""augur-sdk 0.2.1 branch_context wiring for fan-out (#631).

Verifies the helper that builds the ``branch_context`` payload from
suite metadata. The downstream AugurAdapter.__init__ forwarding is
covered by inspection — the kwarg is plumbed but only fires when
augur-sdk is installed AND a DSN is configured.
"""

from __future__ import annotations

from mantis_agent.gym.fanout_runner import build_fanout_branch_context


def test_branch_context_returns_none_without_parent_run_id() -> None:
    """Single-worker / non-fanout runs have no ``_fanout_parent_run_id``
    → return None so AugurAdapter opens without a branch label
    (preserves today's session shape)."""
    assert build_fanout_branch_context({}) is None
    assert build_fanout_branch_context({"session_name": "test"}) is None


def test_branch_context_minimum_shape_with_parent_id() -> None:
    """Just a parent_run_id is enough to produce a valid context.
    branch_id auto-defaults to ``{parent}:{session_name}``."""
    ctx = build_fanout_branch_context({
        "_fanout_parent_run_id": "fanout-abc-123",
        "session_name": "worker_x",
    })
    assert ctx is not None
    assert ctx["parent_run_id"] == "fanout-abc-123"
    assert ctx["branch_point_step_index"] == 0
    assert ctx["mutated_axis"] == "action"
    assert ctx["branch_id"] == "fanout-abc-123:worker_x"


def test_branch_context_phase_lands_in_mutation() -> None:
    """``_fanout_phase`` (phase1_collect / phase2_extract / pagination_partition)
    surfaces in the mutation payload so Augur cohort filters can split
    Phase-1 vs Phase-2 vs pagination on the same parent."""
    ctx = build_fanout_branch_context({
        "_fanout_parent_run_id": "fanout-abc",
        "_fanout_phase": "phase2_extract",
        "session_name": "w1",
    })
    assert ctx["mutation"]["phase"] == "phase2_extract"


def test_branch_context_url_count_lands_in_mutation() -> None:
    """Phase-2 workers stamp how many URLs they own — useful when
    inspecting why one worker took 3× as long as siblings."""
    ctx = build_fanout_branch_context({
        "_fanout_parent_run_id": "fanout-abc",
        "_fanout_url_count": 5,
        "session_name": "w1",
    })
    assert ctx["mutation"]["url_count"] == 5


def test_branch_context_explicit_branch_id_wins() -> None:
    """Orchestrator-supplied ``_fanout_branch_id`` overrides the
    auto-default (which uses session_name) — gives the orchestrator
    control over the human-readable label."""
    ctx = build_fanout_branch_context({
        "_fanout_parent_run_id": "fanout-abc",
        "_fanout_branch_id": "fanout-abc:phase2_w3",
        "session_name": "ignored",
    })
    assert ctx["branch_id"] == "fanout-abc:phase2_w3"


def test_branch_context_mutated_axis_is_action_for_fanout() -> None:
    """Pin the contract: mantis fan-out is always action-axis (per
    augur-sdk 0.2.1 SPEC §10 — different URL = different action).
    The SDK auto-mode resolves this to ``sandbox`` (no replay prefix).
    Changing this would break the Augur platform's cohort semantics."""
    ctx = build_fanout_branch_context({
        "_fanout_parent_run_id": "fanout-abc",
    })
    assert ctx["mutated_axis"] == "action"


def test_branch_context_branch_point_is_zero_for_fanout() -> None:
    """Fan-out workers execute from step 0 — there's no parent prefix
    to replay (Phase 1 produced a URL list, not a step trajectory).
    Pin to 0 so the schema validator accepts the bundle."""
    ctx = build_fanout_branch_context({
        "_fanout_parent_run_id": "fanout-abc",
    })
    assert ctx["branch_point_step_index"] == 0
