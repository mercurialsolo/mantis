"""Audit batch — click-dispatch escalation.

Two CUA-conformant escapes for click failures that previously slid
into ``halt`` after wasting Claude calls:

1. **Plan-supplied ``hints.fallback_url``** — when a click / submit
   step accumulates 2+ rewrite-triggering failures, the critic
   replaces it with a direct ``navigate`` to the URL the plan
   author named. Deterministic — fires BEFORE the frontier-model
   capability so plans with known structural alternatives skip
   the Claude consultation entirely.

2. **SoM ok-but-no-state-change retry via Input.dispatchMouseEvent**
   — when ``el.click()`` returns ok=True but the URL didn't change
   AND the SoM path was used, the form handler retries with a
   real-pointer CDP event chain (``mouseMoved`` → ``mousePressed``
   → ``mouseReleased``). The protocol-level events are
   ``isTrusted=true``, so SPA frameworks that gate on trusted
   gestures accept them.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.gym.critic import (
    ExecutionCritic,
    ReplaceStep,
)
from mantis_agent.plan_decomposer import MicroIntent, MicroPlan


# ── hints.fallback_url — deterministic critic rule ──────────────────


def _runner_with_failure_history(step_index: int, n_failures: int):
    runner = SimpleNamespace(
        _results_base_url="https://example.com",
        _healing_events=[],
        _step_failure_history={
            step_index: [
                {"x": i, "y": i, "kind": "no_state_change"}
                for i in range(n_failures)
            ]
        },
        _recovery_attempts_per_step={},
        _total_recovery_attempts=0,
        _recovery_hints={},
        _critic_frontier_fired_steps=set(),
        env=None,
    )
    return runner


def _state(step_index: int):
    from mantis_agent.gym.checkpoint import RunCheckpoint
    from mantis_agent.gym.run_executor import RunState
    return RunState(
        checkpoint=RunCheckpoint(run_key="t", plan_signature="s", session_name="x"),
        step_index=step_index,
    )


def test_fallback_url_replaces_step_after_two_failures() -> None:
    """When a submit step has 2+ rewrite-triggering failures AND
    ``hints.fallback_url`` is set, the critic emits a ``ReplaceStep``
    directive with type=navigate and the fallback URL."""
    runner = _runner_with_failure_history(step_index=6, n_failures=2)
    critic = ExecutionCritic(runner)
    plan = MicroPlan(domain="x", steps=[
        MicroIntent(
            intent="Click Contacted", type="submit",
            params={"label": "Contacted"},
            hints={"fallback_url": "/leads?status=Contacted"},
        ),
    ])
    state = _state(0)  # only one step in the plan; step_index=0 reads steps[0]
    runner._step_failure_history = {
        0: [
            {"x": 1, "y": 1, "kind": "no_state_change"},
            {"x": 2, "y": 2, "kind": "wrong_target"},
        ]
    }
    result = StepResult(
        step_index=0, intent="x", success=False,
        failure_class="no_state_change",
    )
    out = critic.observe_step(
        plan, state, plan.steps[0], result, recovery_continued=True,
    )
    assert isinstance(out, ReplaceStep)
    assert out.step_type == "navigate"
    # Relative path resolved against the runner's _results_base_url
    # (the test fixture seeds it to "https://example.com").
    assert out.params == {"url": "https://example.com/leads?status=Contacted"}
    assert "fallback_url" in out.reason


def test_fallback_url_skipped_below_threshold() -> None:
    """Threshold = 2 prior failures. Only one failure on record →
    the deterministic rule stays silent; cheap retry path gets a
    chance first."""
    runner = _runner_with_failure_history(step_index=0, n_failures=1)
    critic = ExecutionCritic(runner)
    plan = MicroPlan(domain="x", steps=[
        MicroIntent(
            intent="x", type="submit",
            hints={"fallback_url": "/leads?status=Contacted"},
        ),
    ])
    result = StepResult(
        step_index=0, intent="x", success=False,
        failure_class="no_state_change",
    )
    out = critic.observe_step(
        plan, _state(0), plan.steps[0], result, recovery_continued=True,
    )
    assert out is None


def test_fallback_url_skipped_without_hint() -> None:
    """A plan without ``hints.fallback_url`` falls through to the
    frontier capability (or terminal recovery). The deterministic
    rule needs the plan author to name the alternative URL — it
    doesn't invent one."""
    runner = _runner_with_failure_history(step_index=0, n_failures=3)
    critic = ExecutionCritic(runner)
    plan = MicroPlan(domain="x", steps=[
        MicroIntent(intent="x", type="submit"),  # no hints
    ])
    result = StepResult(
        step_index=0, intent="x", success=False,
        failure_class="no_state_change",
    )
    out = critic.observe_step(
        plan, _state(0), plan.steps[0], result, recovery_continued=True,
    )
    assert out is None  # no fallback emitted; critic may try frontier


def test_fallback_url_fires_on_selector_miss() -> None:
    """Holo3 / SoM grounding misses classify as ``selector_miss``. The
    deterministic fallback_url rule is MORE permissive than the
    LLM-consulting frontier capability (no Claude cost; the URL is
    plan-author-supplied), so it accepts ``selector_miss`` as a
    triggering class too. Without this, a v30-shape plan with
    fallback_url emitted on a click step that Holo3 can't ground
    burns the retry budget on alternative-strategy substitutions
    before the deterministic navigate ever fires (live repro in
    run 20260518_054136_b72b7ab5)."""
    runner = _runner_with_failure_history(step_index=0, n_failures=2)
    critic = ExecutionCritic(runner)
    plan = MicroPlan(domain="x", steps=[
        MicroIntent(
            intent="Click Contacted in sidebar", type="submit",
            params={"label": "Contacted"},
            hints={"fallback_url": "https://example.com/leads?status=Contacted"},
        ),
    ])
    runner._step_failure_history = {
        0: [
            {"x": 1, "y": 1, "kind": "selector_miss"},
            {"x": 2, "y": 2, "kind": "selector_miss"},
        ]
    }
    result = StepResult(
        step_index=0, intent="x", success=False,
        failure_class="selector_miss",
    )
    out = critic.observe_step(
        plan, _state(0), plan.steps[0], result, recovery_continued=True,
    )
    assert isinstance(out, ReplaceStep)
    assert out.step_type == "navigate"
    assert out.params == {"url": "https://example.com/leads?status=Contacted"}


def test_fallback_url_relative_path_resolved_against_current_url() -> None:
    """Plan-author-supplied fallback_url is conventionally a path-
    relative URL (the plan doesn't know the origin at decompose time).
    The critic should resolve it against the browser's current page
    origin before emitting the navigate step — Modal's navigate
    dispatcher requires a full http(s)://... URL and fails on bare
    paths. Live repro: run 20260518_113708_25c9d5e8 — critic
    correctly promoted step 6 to fallback navigate but the navigate
    halted on the relative path."""
    runner = _runner_with_failure_history(step_index=0, n_failures=2)
    runner.env = SimpleNamespace(current_url="https://crm.example.com/dashboard")
    critic = ExecutionCritic(runner)
    plan = MicroPlan(domain="x", steps=[
        MicroIntent(
            intent="Click Contacted", type="submit",
            hints={"fallback_url": "/leads?status=Contacted"},
        ),
    ])
    runner._step_failure_history = {
        0: [
            {"x": 1, "y": 1, "kind": "selector_miss"},
            {"x": 2, "y": 2, "kind": "selector_miss"},
        ]
    }
    result = StepResult(
        step_index=0, intent="x", success=False,
        failure_class="selector_miss",
    )
    out = critic.observe_step(
        plan, _state(0), plan.steps[0], result, recovery_continued=True,
    )
    assert isinstance(out, ReplaceStep)
    assert out.params == {
        "url": "https://crm.example.com/leads?status=Contacted",
    }


def test_fallback_url_absolute_url_preserved() -> None:
    """When the plan author already supplied an absolute URL, the
    critic uses it verbatim — no rewrites, no origin guessing."""
    runner = _runner_with_failure_history(step_index=0, n_failures=2)
    runner.env = SimpleNamespace(current_url="https://example.com/page")
    critic = ExecutionCritic(runner)
    abs_url = "https://other-site.example.com/leads?status=Contacted"
    plan = MicroPlan(domain="x", steps=[
        MicroIntent(
            intent="Click Contacted", type="submit",
            hints={"fallback_url": abs_url},
        ),
    ])
    runner._step_failure_history = {
        0: [
            {"x": 1, "y": 1, "kind": "selector_miss"},
            {"x": 2, "y": 2, "kind": "selector_miss"},
        ]
    }
    result = StepResult(
        step_index=0, intent="x", success=False,
        failure_class="selector_miss",
    )
    out = critic.observe_step(
        plan, _state(0), plan.steps[0], result, recovery_continued=True,
    )
    assert isinstance(out, ReplaceStep)
    assert out.params == {"url": abs_url}


def test_row_link_dom_href_promotes_to_navigate() -> None:
    """When a click/submit step has ``params.kind="row_link"`` and
    ``hints.expect_url_contains``, and the env's ``cdp_evaluate``
    returns a matching href, the critic replaces the step with a
    direct navigate to that href. Closes the loop on the canonical
    Holo3 row-link grounding gap (live repro:
    20260518_161044_079742b8 step 8)."""
    runner = _runner_with_failure_history(step_index=0, n_failures=1)
    # Mock env with cdp_evaluate returning the DOM-derived href.
    def fake_cdp(js: str):
        # Validate the JS contains the expected pattern array.
        assert "/leads/" in js
        return "https://crm.example.com/leads/289"
    runner.env = SimpleNamespace(cdp_evaluate=fake_cdp)
    critic = ExecutionCritic(runner)
    plan = MicroPlan(domain="x", steps=[
        MicroIntent(
            intent="Open the first lead by clicking its Robot Name",
            type="submit",
            params={"label": "first lead row", "kind": "row_link"},
            hints={"expect_url_contains": ["/leads/"]},
        ),
    ])
    runner._step_failure_history = {
        0: [{"x": 0, "y": 0, "kind": "selector_miss"}],
    }
    result = StepResult(
        step_index=0, intent="x", success=False,
        failure_class="selector_miss",
    )
    out = critic.observe_step(
        plan, _state(0), plan.steps[0], result, recovery_continued=True,
    )
    assert isinstance(out, ReplaceStep)
    assert out.step_type == "navigate"
    assert out.params == {"url": "https://crm.example.com/leads/289"}
    assert "row" in out.reason


def test_row_link_dom_href_skips_when_kind_is_not_row_link() -> None:
    """The DOM-href rule is scoped to ``kind="row_link"`` clicks
    specifically — other click shapes (buttons, plain links, tabs)
    use the regular fallback_url path or no fallback at all."""
    runner = _runner_with_failure_history(step_index=0, n_failures=2)
    runner.env = SimpleNamespace(cdp_evaluate=lambda js: "https://x/y")
    critic = ExecutionCritic(runner)
    plan = MicroPlan(domain="x", steps=[
        MicroIntent(
            intent="Click Save", type="submit",
            params={"label": "Save", "kind": "button"},  # not row_link
            hints={"expect_url_contains": ["/saved"]},
        ),
    ])
    runner._step_failure_history = {
        0: [{"x": 1, "y": 1, "kind": "selector_miss"},
            {"x": 2, "y": 2, "kind": "selector_miss"}],
    }
    result = StepResult(
        step_index=0, intent="x", success=False,
        failure_class="selector_miss",
    )
    out = critic.observe_step(
        plan, _state(0), plan.steps[0], result, recovery_continued=True,
    )
    # Should fall through (no fallback_url either) — None.
    assert out is None


def test_row_link_dom_href_skips_when_cdp_unavailable() -> None:
    """When the env doesn't expose ``cdp_evaluate`` (e.g. some
    Playwright test envs), the rule no-ops. The critic falls through
    to whatever capability is next in the chain."""
    runner = _runner_with_failure_history(step_index=0, n_failures=1)
    runner.env = SimpleNamespace()  # no cdp_evaluate
    critic = ExecutionCritic(runner)
    plan = MicroPlan(domain="x", steps=[
        MicroIntent(
            intent="Open first row", type="submit",
            params={"label": "first row", "kind": "row_link"},
            hints={"expect_url_contains": ["/leads/"]},
        ),
    ])
    runner._step_failure_history = {
        0: [{"x": 0, "y": 0, "kind": "selector_miss"}],
    }
    result = StepResult(
        step_index=0, intent="x", success=False,
        failure_class="selector_miss",
    )
    out = critic.observe_step(
        plan, _state(0), plan.steps[0], result, recovery_continued=True,
    )
    assert out is None


def test_row_link_dom_href_skips_below_failure_threshold() -> None:
    """The deterministic rule waits for ≥1 prior failure so the
    visual grounding path gets a chance first."""
    runner = _runner_with_failure_history(step_index=0, n_failures=0)
    runner.env = SimpleNamespace(cdp_evaluate=lambda js: "https://x/y/1")
    critic = ExecutionCritic(runner)
    plan = MicroPlan(domain="x", steps=[
        MicroIntent(
            intent="Open first row", type="submit",
            params={"label": "first row", "kind": "row_link"},
            hints={"expect_url_contains": ["/leads/"]},
        ),
    ])
    runner._step_failure_history = {0: []}
    result = StepResult(
        step_index=0, intent="x", success=False,
        failure_class="selector_miss",
    )
    out = critic.observe_step(
        plan, _state(0), plan.steps[0], result, recovery_continued=True,
    )
    assert out is None


def test_fallback_url_fires_on_unknown_class() -> None:
    """When the failure classifier can't categorise the prose (returns
    ``unknown``), the deterministic rule still fires if the plan
    supplied a fallback_url. Frontier capability stays gated on the
    narrow class set; deterministic navigate is cheap."""
    runner = _runner_with_failure_history(step_index=0, n_failures=2)
    critic = ExecutionCritic(runner)
    plan = MicroPlan(domain="x", steps=[
        MicroIntent(
            intent="Click Contacted", type="submit",
            hints={"fallback_url": "https://example.com/leads?status=Contacted"},
        ),
    ])
    runner._step_failure_history = {
        0: [
            {"x": 1, "y": 1, "kind": "unknown"},
            {"x": 2, "y": 2, "kind": "unknown"},
        ]
    }
    result = StepResult(
        step_index=0, intent="x", success=False,
        failure_class="unknown",
    )
    out = critic.observe_step(
        plan, _state(0), plan.steps[0], result, recovery_continued=True,
    )
    assert isinstance(out, ReplaceStep)
    assert out.step_type == "navigate"


def test_fallback_url_skipped_for_non_click_step_types() -> None:
    """The rule is scoped to ``submit`` and ``click`` step types
    (the ones that have a meaningful structural alternative).
    ``fill_field`` / ``select_option`` / ``extract_data`` would
    have to fill / select / read — a navigate doesn't replace
    those semantics."""
    runner = _runner_with_failure_history(step_index=0, n_failures=3)
    critic = ExecutionCritic(runner)
    plan = MicroPlan(domain="x", steps=[
        MicroIntent(
            intent="x", type="fill_field",
            hints={"fallback_url": "/x"},
        ),
    ])
    result = StepResult(
        step_index=0, intent="x", success=False,
        failure_class="no_state_change",
    )
    out = critic.observe_step(
        plan, _state(0), plan.steps[0], result, recovery_continued=True,
    )
    assert out is None


@pytest.mark.parametrize(
    "non_triggering_class",
    ["cf_challenge", "http_4xx", "http_5xx", "nav_timeout",
     "extractor_error", "budget_exceeded"],
)
def test_fallback_url_skipped_for_unrelated_failure_classes(non_triggering_class: str) -> None:
    """Failures unrelated to target identification — Cloudflare, HTTP
    error pages, navigation timeouts, extractor errors, budget burns
    — won't be helped by navigating to the same site. The rule
    deliberately ignores them. Triggering classes are
    ``REWRITE_TRIGGERING_CLASSES | {selector_miss, unknown}`` (the
    target-identification failure family)."""
    runner = _runner_with_failure_history(step_index=0, n_failures=3)
    critic = ExecutionCritic(runner)
    plan = MicroPlan(domain="x", steps=[
        MicroIntent(
            intent="x", type="click",
            hints={"fallback_url": "/x"},
        ),
    ])
    result = StepResult(
        step_index=0, intent="x", success=False,
        failure_class=non_triggering_class,
    )
    out = critic.observe_step(
        plan, _state(0), plan.steps[0], result, recovery_continued=True,
    )
    assert out is None


def test_fallback_url_emits_reasoning_trace_event() -> None:
    """When the deterministic fallback_url rule fires, a structured
    event lands on ``runner._healing_events`` so the viewer overlay
    can render it on the timeline alongside the Claude-based critic
    events. Without this, the rule was invisible to the trace
    endpoint (count=0 events even when the rule fired)."""
    runner = _runner_with_failure_history(step_index=0, n_failures=2)
    runner._reasoning_jsonl_path = None  # don't write to disk; in-memory only
    critic = ExecutionCritic(runner)
    plan = MicroPlan(domain="x", steps=[
        MicroIntent(
            intent="Click Contacted", type="submit",
            hints={"fallback_url": "https://x.example/leads?status=Contacted"},
        ),
    ])
    result = StepResult(
        step_index=0, intent="x", success=False,
        failure_class="no_state_change",
    )
    out = critic.observe_step(
        plan, _state(0), plan.steps[0], result, recovery_continued=True,
    )
    assert isinstance(out, ReplaceStep)
    # A reasoning event landed.
    events = [
        e for e in runner._healing_events
        if isinstance(e, dict) and e.get("layer") == "critic-fallback-url"
    ]
    assert len(events) == 1
    event = events[0]
    assert event["kind"] == "fire"
    assert event["category"] == "reasoning"
    assert "leads?status=Contacted" in event["summary"]
    assert event["detail"]["failure_class"] == "no_state_change"
    assert event["detail"]["failure_count"] == 2


def test_navigate_back_recovery_emits_reasoning_trace_event() -> None:
    """Same trace plumbing for the older navigate_back rule. Without
    this the deterministic recovery slipped through the viewer's
    timeline."""
    runner = SimpleNamespace(
        _results_base_url="https://example.com/discover",
        _healing_events=[],
    )
    critic = ExecutionCritic(runner)
    plan = MicroPlan(domain="x", steps=[
        MicroIntent(intent="back", type="navigate_back"),
    ])
    result = StepResult(
        step_index=0, intent="back", success=False,
        failure_class="brain_loop_exhausted",
    )
    out = critic.observe_step(
        plan, _state(0), plan.steps[0], result, recovery_continued=True,
    )
    assert out is not None  # InsertStep emitted
    events = [
        e for e in runner._healing_events
        if isinstance(e, dict)
        and e.get("layer") == "critic-navigate-back-recovery"
    ]
    assert len(events) == 1
    assert events[0]["kind"] == "fire"
    assert "discover" in events[0]["summary"]


def test_fallback_url_fires_before_frontier_capability(monkeypatch) -> None:
    """The deterministic rule must run BEFORE the frontier-model
    capability. Otherwise a plan with a known fallback wastes the
    cost of a Claude call before the rule fires.

    Verify by: enable frontier, ensure analyse_failure_and_recover
    is NOT called when fallback_url is set."""
    monkeypatch.setenv("MANTIS_CRITIC_FRONTIER", "enabled")
    runner = _runner_with_failure_history(step_index=0, n_failures=3)
    from mantis_agent import agentic_recovery
    monkeypatch.setattr(
        agentic_recovery, "analyse_failure_and_recover",
        lambda **_: pytest.fail("Claude must not be called when fallback_url is set"),
    )

    critic = ExecutionCritic(runner)
    plan = MicroPlan(domain="x", steps=[
        MicroIntent(
            intent="x", type="submit",
            hints={"fallback_url": "/leads?status=Contacted"},
        ),
    ])
    result = StepResult(
        step_index=0, intent="x", success=False,
        failure_class="no_state_change",
    )
    out = critic.observe_step(
        plan, _state(0), plan.steps[0], result, recovery_continued=True,
    )
    assert isinstance(out, ReplaceStep)


# ── cdp_click_via_pointer — Input.dispatchMouseEvent ────────────────


def test_cdp_click_via_pointer_dispatches_three_events() -> None:
    """A real-pointer click dispatches mouseMoved → mousePressed →
    mouseReleased. Order matters: trust-gated frameworks observe
    the sequence."""
    from mantis_agent.gym.xdotool_env import XdotoolGymEnv

    instance = XdotoolGymEnv.__new__(XdotoolGymEnv)
    instance.cdp_evaluate = lambda _expr: 87  # chrome_h
    captured: list = []
    instance._cdp_call = lambda method, params: (  # type: ignore[assignment]
        captured.append((method, params)) or (True, {})
    )

    ok = instance.cdp_click_via_pointer(100, 200)
    assert ok is True
    assert len(captured) == 3
    methods = [m for m, _ in captured]
    assert methods == [
        "Input.dispatchMouseEvent",
        "Input.dispatchMouseEvent",
        "Input.dispatchMouseEvent",
    ]
    types = [p.get("type") for _, p in captured]
    assert types == ["mouseMoved", "mousePressed", "mouseReleased"]
    # Y is chrome-offset-corrected (screen 200 - 87 = 113).
    assert all(p.get("y") == 113 for _, p in captured)


def test_cdp_click_via_pointer_press_release_carry_click_count() -> None:
    """``Input.dispatchMouseEvent`` for mousePressed / mouseReleased
    must include ``clickCount: 1`` — otherwise Chrome treats the
    event as a move-equivalent and frameworks don't fire onclick."""
    from mantis_agent.gym.xdotool_env import XdotoolGymEnv
    instance = XdotoolGymEnv.__new__(XdotoolGymEnv)
    instance.cdp_evaluate = lambda _expr: 0
    captured: list = []
    instance._cdp_call = lambda method, params: (  # type: ignore[assignment]
        captured.append((method, params)) or (True, {})
    )

    instance.cdp_click_via_pointer(50, 60)
    moved, pressed, released = captured
    assert "clickCount" not in moved[1]
    assert pressed[1].get("clickCount") == 1
    assert released[1].get("clickCount") == 1
    # Pressed carries buttons=1 (left); released carries buttons=0.
    assert pressed[1].get("buttons") == 1
    assert released[1].get("buttons") == 0


def test_cdp_click_via_pointer_returns_false_when_any_dispatch_fails() -> None:
    """If any of the three CDP calls fails (network blip, target
    detached), the helper returns False so the caller falls back
    to xdotool / Enter-key / demote."""
    from mantis_agent.gym.xdotool_env import XdotoolGymEnv

    instance = XdotoolGymEnv.__new__(XdotoolGymEnv)
    instance.cdp_evaluate = lambda _expr: 0
    call_count = {"n": 0}

    def _fail_on_second(method, params):
        call_count["n"] += 1
        return (call_count["n"] != 2, {})  # 2nd call (mousePressed) fails

    instance._cdp_call = _fail_on_second  # type: ignore[assignment]
    assert instance.cdp_click_via_pointer(10, 20) is False


def test_cdp_click_via_pointer_chrome_offset_zero_when_eval_fails() -> None:
    """``_chrome_offset_px`` is a best-effort JS eval — if it raises,
    we default to chromeH=0 (the safe degenerate case where the
    screen-y equals viewport-y, matching headless modes that don't
    report chrome at all)."""
    from mantis_agent.gym.xdotool_env import XdotoolGymEnv

    instance = XdotoolGymEnv.__new__(XdotoolGymEnv)

    def _raise(_expr):
        raise RuntimeError("CDP unreachable")

    instance.cdp_evaluate = _raise  # type: ignore[assignment]
    assert instance._chrome_offset_px() == 0
