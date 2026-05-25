"""#678 — orchestrator-session helper + ``run_fanout_dispatch`` wiring.

augur-sdk 0.4.0 (mercurialsolo/augur-sdk#38) added
``DebugSession.open_orchestrator(...)`` for opening parent-only
sessions that surface fan-out aggregate metadata (phase counts,
fan-out pattern) in the Augur viewer's runs-list. Children carry
``branch_context.parent_run_id`` pointing at this session's run_id;
the server / viewer (mercurialsolo/augur#138) groups by that field.

Contract pinned here:

* ``open_orchestrator_session`` is a no-op (yields ``None``) when
  ``MANTIS_AUGUR_DISABLED`` is set OR the SDK predates 0.4.0
  (``open_orchestrator`` attribute missing) — never raises out.
* When the SDK is available and active, the helper opens the session
  before the body runs and closes it on the way out, even when the
  body raises.
* The helper stamps the ``augur.session_type=orchestrator`` tag
  (``_ORCHESTRATOR_TAG_KEY`` / ``_VALUE``) so the viewer recognises
  the parent-only shape.
* ``run_fanout_dispatch`` opens the orchestrator session exactly once
  per dispatch call, before any worker spawn, with aggregate tags
  (``phase1_workers``, ``phase2_workers_configured``,
  ``fanout_pattern``, ``phase1_max_pages``, optional
  ``plan_signature``).
* Opener exceptions and ``__enter__`` / ``__exit__`` exceptions are
  swallowed (logged at WARN) — the dispatch body still runs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import mantis_agent.observability.augur as augur_mod
from mantis_agent.gym.fanout_runner import run_fanout_dispatch
from mantis_agent.observability.augur import (
    _ORCHESTRATOR_TAG_KEY,
    _ORCHESTRATOR_TAG_VALUE,
    open_orchestrator_session,
)
from mantis_agent.plan_decomposer import MicroIntent, MicroPlan, PlanDecomposer


@pytest.fixture
def force_augur_available(monkeypatch):
    """Flip ``_AUGUR_AVAILABLE=True`` for tests that exercise the
    opener path.

    Without this, CI runners that don't install ``augur-sdk`` (the
    package is an opt-in ``observability`` extra in pyproject.toml)
    short-circuit ``is_enabled() → False`` and the helper returns
    ``None`` before reaching the patched ``DebugSession``. The tests
    pin wiring behaviour, not the SDK install — patch the module's
    SDK-available flag and the opener path becomes reachable.
    """
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    monkeypatch.setattr(augur_mod, "_AUGUR_AVAILABLE", True)


# ── open_orchestrator_session — no-op paths ─────────────────────────


def test_yields_none_when_augur_disabled(monkeypatch):
    """``MANTIS_AUGUR_DISABLED=1`` → helper is a no-op."""
    monkeypatch.setenv("MANTIS_AUGUR_DISABLED", "1")
    with open_orchestrator_session(run_id="x", tags={}) as s:
        assert s is None


def test_yields_none_when_sdk_predates_040(force_augur_available):
    """SDK without ``open_orchestrator`` → no-op (server still
    synthesizes the parent row from children)."""
    # Force the SDK-shape probe to miss the attribute even if the
    # installed SDK has it. ``DebugSession`` itself stays truthy so we
    # don't trip the "module missing" branch.
    stub = MagicMock()
    # Stripping the attr completely so getattr returns the fallback.
    if hasattr(stub, "open_orchestrator"):
        del stub.open_orchestrator
    with patch.object(augur_mod, "DebugSession", stub):
        with open_orchestrator_session(run_id="x", tags={}) as s:
            assert s is None


def test_yields_none_when_debugsession_module_missing(force_augur_available):
    """``DebugSession is None`` (SDK not installed) → no-op."""
    with patch.object(augur_mod, "DebugSession", None):
        with open_orchestrator_session(run_id="x", tags={}) as s:
            assert s is None


def test_opener_exception_is_swallowed(force_augur_available):
    """If the SDK opener itself raises, fall through to yielding
    ``None`` (best-effort observability — never break the run)."""
    stub = MagicMock()
    stub.open_orchestrator = MagicMock(side_effect=RuntimeError("boom"))
    with patch.object(augur_mod, "DebugSession", stub):
        with open_orchestrator_session(run_id="x", tags={}) as s:
            assert s is None


def test_enter_exception_is_swallowed(force_augur_available):
    """``__enter__`` raising on the opener's return is also caught."""
    fake_ctx = MagicMock()
    fake_ctx.__enter__ = MagicMock(side_effect=RuntimeError("enter-boom"))
    stub = MagicMock()
    stub.open_orchestrator = MagicMock(return_value=fake_ctx)
    with patch.object(augur_mod, "DebugSession", stub):
        with open_orchestrator_session(run_id="x", tags={}) as s:
            assert s is None


def test_exit_exception_is_swallowed_after_normal_body(force_augur_available):
    """``__exit__`` raising after the body completed cleanly → caller
    still gets back control (no exception propagates)."""
    fake_ctx = MagicMock()
    fake_session = MagicMock(name="orchestrator_session")
    fake_ctx.__enter__ = MagicMock(return_value=fake_session)
    fake_ctx.__exit__ = MagicMock(side_effect=RuntimeError("exit-boom"))
    stub = MagicMock()
    stub.open_orchestrator = MagicMock(return_value=fake_ctx)
    with patch.object(augur_mod, "DebugSession", stub):
        with open_orchestrator_session(run_id="x", tags={}) as s:
            assert s is fake_session
    # Reaching here means the exit-exception was swallowed.
    fake_ctx.__exit__.assert_called_once()


# ── open_orchestrator_session — opener call shape ──────────────────


def test_opener_called_with_orchestrator_marker_tag(force_augur_available):
    """The helper stamps ``augur.session_type=orchestrator`` even when
    the caller passes ``tags`` without it (the SDK 0.4.0 helper also
    sets this internally; the eager stamp guarantees the session
    metadata flush at ``__enter__`` already carries it)."""
    fake_ctx = MagicMock()
    fake_ctx.__enter__ = MagicMock(return_value=MagicMock())
    fake_ctx.__exit__ = MagicMock(return_value=False)
    opener = MagicMock(return_value=fake_ctx)
    stub = MagicMock()
    stub.open_orchestrator = opener
    with patch.object(augur_mod, "DebugSession", stub):
        with open_orchestrator_session(
            run_id="r1", session_name="fanout-x", tags={"phase1_workers": "4"},
        ):
            pass
    opener.assert_called_once()
    call_tags = opener.call_args.kwargs["tags"]
    assert call_tags[_ORCHESTRATOR_TAG_KEY] == _ORCHESTRATOR_TAG_VALUE
    assert call_tags["phase1_workers"] == "4"
    assert opener.call_args.kwargs["run_id"] == "r1"
    assert opener.call_args.kwargs["session_name"] == "fanout-x"
    assert opener.call_args.kwargs["client_name"] == "mantis"


def test_opener_caller_tags_take_precedence_over_default_marker(force_augur_available):
    """A caller that explicitly passes a custom ``augur.session_type``
    tag wins over the default — the helper only ``setdefault``s the
    marker. (Edge-case escape hatch for a producer that wants to
    relabel; doesn't change the contract.)"""
    opener = MagicMock(return_value=MagicMock(
        __enter__=MagicMock(return_value=MagicMock()),
        __exit__=MagicMock(return_value=False),
    ))
    stub = MagicMock()
    stub.open_orchestrator = opener
    custom_tags = {_ORCHESTRATOR_TAG_KEY: "custom", "phase1_workers": "2"}
    with patch.object(augur_mod, "DebugSession", stub):
        with open_orchestrator_session(run_id="r1", tags=custom_tags):
            pass
    call_tags = opener.call_args.kwargs["tags"]
    assert call_tags[_ORCHESTRATOR_TAG_KEY] == "custom"
    assert call_tags["phase1_workers"] == "2"


# ── run_fanout_dispatch — orchestrator wiring ──────────────────────


def _make_eligible_suite() -> dict:
    """Boattrader-shaped suite with a ``parallelizable_url_collect``
    loop group — eligible for fan-out dispatch. Same shape as
    ``test_run_fanout_dispatch_673._make_url_collect_suite``; inlined
    here so the orchestrator-wiring tests stay self-contained."""
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
        "_plan_signature": "boattrader-orch-test-sig",
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


def _make_executor_stub(*, phase1_urls: list[str], phase2_viable: int = 1):
    """Mock executor that returns Phase-1 then Phase-2 results."""
    handle_p1 = MagicMock()
    handle_p1.get.return_value = {
        "viable": 0, "leads_with_phone": 0, "leads": [],
        "collected_urls": phase1_urls, "shared_seen_hits": 0,
    }
    handle_p2 = MagicMock()
    handle_p2.get.return_value = {
        "viable": phase2_viable, "leads_with_phone": 0,
        "leads": [{"url": u} for u in phase1_urls[:phase2_viable]],
        "shared_seen_hits": 0,
    }
    executor = MagicMock()
    executor.spawn.side_effect = [handle_p1, handle_p2]
    return executor


def test_dispatch_opens_orchestrator_with_aggregate_tags(force_augur_available):
    """When dispatch runs, the orchestrator session is opened exactly
    once with phase counts + fan-out pattern + plan signature tags."""
    opener = MagicMock(return_value=MagicMock(
        __enter__=MagicMock(return_value=MagicMock()),
        __exit__=MagicMock(return_value=False),
    ))
    stub = MagicMock()
    stub.open_orchestrator = opener
    with patch.object(augur_mod, "DebugSession", stub):
        result = run_fanout_dispatch(
            _make_eligible_suite(),
            executor_fn=_make_executor_stub(phase1_urls=["https://x/d/1"]),
            model="holo3", claude_model="",
            max_steps=10, workers=1,
            fanout_parent_run_id="fanout-test-678",
        )
    assert result is not None
    opener.assert_called_once()
    kwargs = opener.call_args.kwargs
    assert kwargs["run_id"] == "fanout-test-678"
    tags = kwargs["tags"]
    assert tags[_ORCHESTRATOR_TAG_KEY] == _ORCHESTRATOR_TAG_VALUE
    assert tags["fanout_pattern"] == "phase1_collect_phase2_extract"
    assert tags["phase1_workers"] == "1"
    assert tags["phase2_workers_configured"] == "1"
    assert tags["phase1_max_pages"] == "1"
    assert "plan_signature" in tags  # caller passed _plan_signature


def test_dispatch_skips_orchestrator_when_disabled(monkeypatch):
    """``MANTIS_AUGUR_DISABLED=1`` → orchestrator opener is never
    called; dispatch still returns the aggregate envelope."""
    monkeypatch.setenv("MANTIS_AUGUR_DISABLED", "1")
    opener = MagicMock()
    stub = MagicMock()
    stub.open_orchestrator = opener
    with patch.object(augur_mod, "DebugSession", stub):
        result = run_fanout_dispatch(
            _make_eligible_suite(),
            executor_fn=_make_executor_stub(phase1_urls=["https://x/d/1"]),
            model="holo3", claude_model="",
            max_steps=10, workers=1,
            fanout_parent_run_id="fanout-test-678",
        )
    assert result is not None
    opener.assert_not_called()


def test_dispatch_skips_orchestrator_when_no_url_collect_group(monkeypatch):
    """Non-eligible suites short-circuit BEFORE the orchestrator is
    opened — no point opening a parent row for a single-runner path."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    opener = MagicMock()
    stub = MagicMock()
    stub.open_orchestrator = opener
    with patch.object(augur_mod, "DebugSession", stub):
        result = run_fanout_dispatch(
            {"_micro_plan": []},  # no loop group
            executor_fn=MagicMock(),
            model="holo3", claude_model="",
            max_steps=10, workers=1,
            fanout_parent_run_id="ineligible",
        )
    assert result is None
    opener.assert_not_called()


def test_dispatch_passes_workers_count_through_to_tags(force_augur_available):
    """Aggregate ``phase2_workers_configured`` tag mirrors the
    ``workers`` argument — the caller-configured Phase-2 fan-out width
    (actual width post-dedup may be lower; reading both off the
    envelope is a server / viewer concern)."""
    opener = MagicMock(return_value=MagicMock(
        __enter__=MagicMock(return_value=MagicMock()),
        __exit__=MagicMock(return_value=False),
    ))
    stub = MagicMock()
    stub.open_orchestrator = opener
    with patch.object(augur_mod, "DebugSession", stub):
        run_fanout_dispatch(
            _make_eligible_suite(),
            executor_fn=_make_executor_stub(phase1_urls=["https://x/d/1"]),
            model="holo3", claude_model="",
            max_steps=10, workers=8,
            fanout_parent_run_id="fanout-test-678",
        )
    assert opener.call_args.kwargs["tags"]["phase2_workers_configured"] == "8"


def test_dispatch_session_closed_on_normal_exit(force_augur_available):
    """``__exit__`` is called on the way out, even on a successful
    dispatch — sessions don't leak."""
    fake_ctx = MagicMock()
    fake_ctx.__enter__ = MagicMock(return_value=MagicMock())
    fake_ctx.__exit__ = MagicMock(return_value=False)
    stub = MagicMock()
    stub.open_orchestrator = MagicMock(return_value=fake_ctx)
    with patch.object(augur_mod, "DebugSession", stub):
        run_fanout_dispatch(
            _make_eligible_suite(),
            executor_fn=_make_executor_stub(phase1_urls=["https://x/d/1"]),
            model="holo3", claude_model="",
            max_steps=10, workers=1,
            fanout_parent_run_id="fanout-test-678",
        )
    fake_ctx.__exit__.assert_called_once()


def test_dispatch_session_closed_when_phase1_returns_no_urls(force_augur_available):
    """Phase-1 returning zero URLs causes dispatch to return ``None``,
    but the orchestrator session was already open — it MUST close on
    the way out so the parent row finalizes."""
    fake_ctx = MagicMock()
    fake_ctx.__enter__ = MagicMock(return_value=MagicMock())
    fake_ctx.__exit__ = MagicMock(return_value=False)
    stub = MagicMock()
    stub.open_orchestrator = MagicMock(return_value=fake_ctx)
    with patch.object(augur_mod, "DebugSession", stub):
        result = run_fanout_dispatch(
            _make_eligible_suite(),
            executor_fn=_make_executor_stub(phase1_urls=[]),  # empty harvest
            model="holo3", claude_model="",
            max_steps=10, workers=1,
            fanout_parent_run_id="fanout-test-678",
        )
    assert result is None
    fake_ctx.__exit__.assert_called_once()


def test_dispatch_session_closed_when_opener_raises(force_augur_available):
    """Opener exception → orchestrator yields ``None`` AND dispatch
    body still runs (returns the normal aggregate envelope). The
    fan-out is best-effort observability — never break the run."""
    stub = MagicMock()
    stub.open_orchestrator = MagicMock(side_effect=RuntimeError("opener-boom"))
    with patch.object(augur_mod, "DebugSession", stub):
        result = run_fanout_dispatch(
            _make_eligible_suite(),
            executor_fn=_make_executor_stub(phase1_urls=["https://x/d/1"]),
            model="holo3", claude_model="",
            max_steps=10, workers=1,
            fanout_parent_run_id="fanout-test-678",
        )
    assert result is not None
    assert result["collected_urls"] == ["https://x/d/1"]
