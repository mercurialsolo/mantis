"""#659 — Augur ``step_id`` must be unique per emission so loop-iterated
plan steps don't collapse on the Augur server.

Live repro (boattrader rerun ``20260524_180432_d308f9de``,
workflow_id ``boattrader-verify-upgrades-rerun-1779645859``):

  Worker MICRO-PLAN COMPLETE: ``Steps: 291`` over 60 min
  Augur UI:                    ``10 steps``

The inner extraction loop iterated step 2 (click) 33 times. Each
emission wrote to ``step_id="step-0003"``. Augur server keyed on
``step_id`` and replaced in place — net 1 row visible per plan
position.

Contract pinned here:
  - First emission for ``step_index=N`` → ``step_id=step-{N+1:04d}-000``
  - Second emission for the same index → ``…-001``
  - Different indices accumulate counts independently.
  - The ``step_index`` field STAYS the plan position so Augur UI can
    group by it.
  - The adapter's ``_step_emission_counts`` resets across adapter
    instances (each run gets a fresh AugurAdapter, fresh counts).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Match the convention from ``tests/test_observability_augur.py``:
# skip the whole module when ``augur_sdk`` isn't importable so CI
# environments that don't bundle the SDK don't fail on attribute access
# against a ``None`` session.
pytest.importorskip("augur_sdk")

from mantis_agent.observability.augur import AugurAdapter  # noqa: E402


def _stub_step_result(step_index: int, *, success: bool = True) -> SimpleNamespace:
    """Minimal StepResult shape that ``_build_step_trace`` reads."""
    return SimpleNamespace(
        step_index=step_index,
        intent=f"step{step_index}",
        success=success,
        verdict=SimpleNamespace(kind="ok", reason="", confidence=1.0),
        data="",
        failure_class="",
        last_action=None,
        duration=0.0,
        page_title="",
        executor_backend="",
        reasoning="",
    )


def _open_adapter(tmp_path: Path) -> AugurAdapter:
    """AugurAdapter writing to ``tmp_path`` so we exercise the real
    streaming path without polluting global ``data/augur/`` dir."""
    return AugurAdapter(
        run_id="step_id_test", tenant_id="t", session_name="s",
        out_dir=tmp_path,
    )


def test_first_emission_for_index_carries_suffix_zero(tmp_path):
    a = _open_adapter(tmp_path)
    trace = a._build_step_trace(
        _stub_step_result(step_index=0),
        started_at="", ended_at="",
        observation_pre=None, observation_post=None,
    )
    assert trace["step_id"] == "step-0001-000"
    assert trace["step_index"] == 1


def test_second_emission_for_same_index_increments_suffix(tmp_path):
    """Loop iteration shape — same plan position, multiple emissions."""
    a = _open_adapter(tmp_path)
    a._build_step_trace(
        _stub_step_result(step_index=2),
        started_at="", ended_at="",
        observation_pre=None, observation_post=None,
    )
    second = a._build_step_trace(
        _stub_step_result(step_index=2),
        started_at="", ended_at="",
        observation_pre=None, observation_post=None,
    )
    assert second["step_id"] == "step-0003-001"
    # step_index still tracks plan position (so Augur UI can group).
    assert second["step_index"] == 3


def test_loop_iteration_produces_n_distinct_step_ids(tmp_path):
    """33 inner-loop iterations of step 2 → 33 distinct step_ids."""
    a = _open_adapter(tmp_path)
    seen_ids: set[str] = set()
    for _ in range(33):
        trace = a._build_step_trace(
            _stub_step_result(step_index=2),
            started_at="", ended_at="",
            observation_pre=None, observation_post=None,
        )
        seen_ids.add(trace["step_id"])
    assert len(seen_ids) == 33
    # Sanity: lowest and highest suffix bracket the range.
    assert "step-0003-000" in seen_ids
    assert "step-0003-032" in seen_ids


def test_emission_counts_isolated_per_step_index(tmp_path):
    """Different plan step_indices accumulate counts independently —
    re-emitting step 2 doesn't bump step 5's counter."""
    a = _open_adapter(tmp_path)
    a._build_step_trace(
        _stub_step_result(step_index=2),
        started_at="", ended_at="",
        observation_pre=None, observation_post=None,
    )
    a._build_step_trace(
        _stub_step_result(step_index=2),
        started_at="", ended_at="",
        observation_pre=None, observation_post=None,
    )
    # Step 5's first emission gets suffix 0 (not 2).
    trace5 = a._build_step_trace(
        _stub_step_result(step_index=5),
        started_at="", ended_at="",
        observation_pre=None, observation_post=None,
    )
    assert trace5["step_id"] == "step-0006-000"


def test_fresh_adapter_instance_starts_with_zero_counts(tmp_path):
    """Each run gets a fresh AugurAdapter → fresh emission counts.
    Pins the 'no cross-run leakage' invariant."""
    a1 = _open_adapter(tmp_path / "a1")
    a1._build_step_trace(
        _stub_step_result(step_index=0),
        started_at="", ended_at="",
        observation_pre=None, observation_post=None,
    )
    a2 = _open_adapter(tmp_path / "a2")
    trace = a2._build_step_trace(
        _stub_step_result(step_index=0),
        started_at="", ended_at="",
        observation_pre=None, observation_post=None,
    )
    assert trace["step_id"] == "step-0001-000"


def test_step_id_suffix_zero_padded_to_3_digits(tmp_path):
    """The suffix is 3-digit zero-padded so lexicographic sort matches
    numeric sort up through 999 iterations. (1000+ iterations is well
    beyond any realistic plan; if it ever matters, expand padding —
    until then 3 digits is the sweet spot for trace UIs.)"""
    a = _open_adapter(tmp_path)
    for i in range(15):
        trace = a._build_step_trace(
            _stub_step_result(step_index=4),
            started_at="", ended_at="",
            observation_pre=None, observation_post=None,
        )
    # 15th emission (0-indexed): suffix should be '014'
    assert trace["step_id"] == "step-0005-014"


def test_step_index_field_unchanged_across_emissions(tmp_path):
    """``step_index`` is the Augur grouping key — must NOT change
    across iterations of the same plan position."""
    a = _open_adapter(tmp_path)
    indices = []
    for _ in range(5):
        trace = a._build_step_trace(
            _stub_step_result(step_index=2),
            started_at="", ended_at="",
            observation_pre=None, observation_post=None,
        )
        indices.append(trace["step_index"])
    assert indices == [3, 3, 3, 3, 3]


# ── End-to-end shape: record_step → counter increments ──


def test_record_step_increments_counter_via_public_api(tmp_path, monkeypatch):
    """``record_step`` is the canonical entrypoint — verify the
    per-emission counter advances when callers use the documented
    surface (not just the internal ``_build_step_trace`` helper)."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = _open_adapter(tmp_path)
    # Guard against environments where ``augur-sdk`` is importable but
    # the SDK refuses to open a session (DSN policy / version skew).
    # Without this the spy on the next line dereferences ``None``.
    if not a.active:
        pytest.skip("AugurAdapter inactive — SDK opened no session")

    # Stash a spy on the session so we capture the trace dicts the
    # SDK would have received.
    captured: list[dict] = []
    real_record = a._session.record_step
    a._session.record_step = MagicMock(side_effect=lambda t: captured.append(t) or real_record(t))

    for _ in range(4):
        a.record_step(
            step_result=_stub_step_result(step_index=2),
            started_at="2026-05-24T00:00:00Z", ended_at="2026-05-24T00:00:01Z",
            step_type="click",
        )

    ids = [t["step_id"] for t in captured]
    assert ids == [
        "step-0003-000", "step-0003-001",
        "step-0003-002", "step-0003-003",
    ]
