"""augur-sdk 0.3.0 step-iteration semantics (mercurialsolo/augur-sdk#30, #31).

After the upstream SDK release, mantis routes the first emission for
each plan ``step_index`` through ``record_step`` (canonical) and every
subsequent emission through ``record_step_iteration`` (appends to the
canonical's iterations + bumps ``step_iterations``).

Before this change, mantis spammed ``record_step`` for every loop
iteration. The 0.3.0 SDK accepts that pattern but emits a
DeprecationWarning AND the Augur viewer can't render the proper
"N (M)" iteration count. Routing iterations through
``record_step_iteration`` is the clean migration.

Contract pinned here:
    - First emission for a given ``step_index`` → ``record_step``.
    - Subsequent emissions for that same index →
      ``record_step_iteration``.
    - Per-index isolation — emitting step 2 then step 5 should produce
      TWO canonical emissions (one each).
    - Fallback for pre-0.3.0 SDKs that don't have
      ``record_step_iteration`` — every call goes through
      ``record_step``, no behaviour change vs today.
    - Adapter-internal state (``_canonical_step_recorded``) resets
      across adapter instances.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("augur_sdk")

from mantis_agent.observability.augur import AugurAdapter  # noqa: E402


def _stub_step_result(step_index: int) -> SimpleNamespace:
    return SimpleNamespace(
        step_index=step_index,
        intent=f"step{step_index}",
        success=True,
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
    return AugurAdapter(
        run_id="iter_test", tenant_id="t", session_name="s",
        out_dir=tmp_path,
    )


# ── Routing: canonical vs iteration ─────────────────────────────────


def test_first_emission_per_index_routes_to_record_step(tmp_path, monkeypatch):
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = _open_adapter(tmp_path)
    if not a.active:
        pytest.skip("AugurAdapter inactive")
    a._session = MagicMock()
    a._session.record_step = MagicMock()
    a._session.record_step_iteration = MagicMock()

    a.record_step(step_result=_stub_step_result(step_index=2))

    a._session.record_step.assert_called_once()
    a._session.record_step_iteration.assert_not_called()
    # The set is keyed by 1-based augur_index, matching the SDK.
    assert 3 in a._canonical_step_recorded


def test_subsequent_emissions_route_to_record_step_iteration(tmp_path, monkeypatch):
    """Same step_index emitted twice → first canonical, second iteration."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = _open_adapter(tmp_path)
    if not a.active:
        pytest.skip("AugurAdapter inactive")
    a._session = MagicMock()
    a._session.record_step = MagicMock()
    a._session.record_step_iteration = MagicMock()

    a.record_step(step_result=_stub_step_result(step_index=2))
    a.record_step(step_result=_stub_step_result(step_index=2))

    assert a._session.record_step.call_count == 1
    assert a._session.record_step_iteration.call_count == 1
    # Iteration was passed (augur_index, trace) — the 1-based index.
    iter_args = a._session.record_step_iteration.call_args.args
    assert iter_args[0] == 3  # augur_index


def test_loop_body_30_iterations_yield_1_canonical_29_iterations(tmp_path, monkeypatch):
    """The boattrader inner-loop shape: same step_index dispatched
    repeatedly. After 30 emissions: 1 canonical + 29 iterations."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = _open_adapter(tmp_path)
    if not a.active:
        pytest.skip("AugurAdapter inactive")
    a._session = MagicMock()
    a._session.record_step = MagicMock()
    a._session.record_step_iteration = MagicMock()

    for _ in range(30):
        a.record_step(step_result=_stub_step_result(step_index=4))

    assert a._session.record_step.call_count == 1
    assert a._session.record_step_iteration.call_count == 29


def test_different_step_indices_each_get_their_own_canonical(tmp_path, monkeypatch):
    """Emitting steps 0, 1, 2 → 3 distinct canonical record_step calls,
    zero iterations. Routing is per-index isolated."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = _open_adapter(tmp_path)
    if not a.active:
        pytest.skip("AugurAdapter inactive")
    a._session = MagicMock()
    a._session.record_step = MagicMock()
    a._session.record_step_iteration = MagicMock()

    a.record_step(step_result=_stub_step_result(step_index=0))
    a.record_step(step_result=_stub_step_result(step_index=1))
    a.record_step(step_result=_stub_step_result(step_index=2))

    assert a._session.record_step.call_count == 3
    a._session.record_step_iteration.assert_not_called()


def test_fallback_when_sdk_lacks_record_step_iteration(tmp_path, monkeypatch):
    """Older SDKs (pre-0.3.0) don't have ``record_step_iteration``;
    every emission must keep going through ``record_step`` so the
    adapter stays compatible with the pin floor."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = _open_adapter(tmp_path)
    if not a.active:
        pytest.skip("AugurAdapter inactive")
    # Simulate a pre-0.3.0 session that has record_step but NO
    # record_step_iteration.
    a._session = MagicMock(spec=["record_step"])
    a._session.record_step = MagicMock()

    a.record_step(step_result=_stub_step_result(step_index=2))
    a.record_step(step_result=_stub_step_result(step_index=2))

    # Both calls land on record_step (the SDK then collapses on
    # step_index — pre-0.3.0 behaviour).
    assert a._session.record_step.call_count == 2


def test_canonical_set_resets_across_adapter_instances(tmp_path, monkeypatch):
    """Each AugurAdapter instance has its own ``_canonical_step_recorded``
    set; one run's history doesn't leak into the next."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a1 = _open_adapter(tmp_path / "a1")
    a2 = _open_adapter(tmp_path / "a2")
    if not (a1.active and a2.active):
        pytest.skip("AugurAdapter inactive")
    a1._session = MagicMock()
    a1._session.record_step = MagicMock()
    a1._session.record_step_iteration = MagicMock()
    a2._session = MagicMock()
    a2._session.record_step = MagicMock()
    a2._session.record_step_iteration = MagicMock()

    a1.record_step(step_result=_stub_step_result(step_index=2))
    # a2's first emission for step 2 is STILL canonical, even though
    # a1 already recorded one for the same index.
    a2.record_step(step_result=_stub_step_result(step_index=2))

    a1._session.record_step.assert_called_once()
    a2._session.record_step.assert_called_once()
    a1._session.record_step_iteration.assert_not_called()
    a2._session.record_step_iteration.assert_not_called()


def test_canonical_set_initialized_at_construction(tmp_path, monkeypatch):
    """Fresh adapter has an empty ``_canonical_step_recorded`` set
    even before any step emits."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = _open_adapter(tmp_path)
    assert a._canonical_step_recorded == set()
