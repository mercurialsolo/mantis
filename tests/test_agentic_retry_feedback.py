"""Tests for the agentic retry feedback loop (#224 follow-up).

Surfaced by the staff-crm rerun on Modal post-PR-#229: step 5
("Click first Qualified lead row") failed three times in a row with
``submit:Qualified:no_state_change``. Each retry re-ran
``find_form_target`` with the same prompt and got the same broken
match — Claude kept picking the status pill text instead of the row
container, and the verifier kept correctly rejecting it.

Fix: when a submit step fails with ``no_state_change``, record the
clicked (x, y, label) on ``runner._step_failure_history[step_index]``.
On the next attempt at the same step, the form handler reads the
history and tells ``find_form_target`` "avoid these previous
targets." The LLM then has explicit feedback that its first guess
landed on a non-action element and can pick differently.

These tests pin the new ``_record_failure_for_retry`` helper, the
form handler's history-aware prompt augmentation, and the success-
path clearing of history.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from PIL import Image

from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.gym.run_executor import RunExecutor, RunState
from mantis_agent.gym.step_snapshot import StepStateSnapshot
from mantis_agent.plan_decomposer import MicroIntent


def _make_runner(*, last_submit_target: dict | None = None) -> MagicMock:
    """Stub runner with the failure-history surface populated."""
    runner = MagicMock()
    runner.extractor = None  # disable the visual fallback so the
    # demote path runs uncontested for these tests
    runner._last_submit_pre_screenshot = None
    runner._last_submit_target = last_submit_target
    runner._step_failure_history = {}
    runner.costs = {"claude_extract": 0}
    runner._last_known_url = "https://crm.test/"
    runner._current_page = 1
    runner._viewport_stage = 0
    runner._scroll_state = {}
    runner._last_extracted = {}
    runner._extracted_titles = []
    runner._seen_urls = set()
    runner.env = MagicMock()
    runner.env.last_focused_input = None
    return runner


def _state_with_submit_success(step_index: int = 5) -> RunState:
    state = RunState.fresh(
        run_key="t", session_name="t", plan_signature="sig",
    )
    state.step_index = step_index
    state.results.append(
        StepResult(
            step_index=step_index, intent="Click Qualified lead",
            success=True, data="submit:Qualified",
            duration=3.0, steps_used=1,
        )
    )
    return state


# ── _record_failure_for_retry ───────────────────────────────────────────


def test_record_failure_appends_to_history() -> None:
    """A demoted submit must drop a record into
    ``_step_failure_history[step_index]`` carrying the click target
    + failure mode for the next retry to see."""
    runner = _make_runner(
        last_submit_target={
            "x": 540, "y": 220,
            "label": "Qualified", "matched_label": "Qualified", "step_index": 5,
        },
    )
    RunExecutor._record_failure_for_retry(
        runner=runner, step_index=5,
        kind="no_state_change",
        reason="visual verifier saw no UI change",
    )
    history = runner._step_failure_history[5]
    assert len(history) == 1
    assert history[0]["x"] == 540
    assert history[0]["y"] == 220
    assert history[0]["label"] == "Qualified"
    assert history[0]["kind"] == "no_state_change"


def test_record_failure_skips_when_no_target_stash() -> None:
    """If the form handler didn't stash a click target (e.g. the step
    failed with ``form_target_not_found`` before reaching the click),
    there's nothing to feed back — record_failure is a no-op."""
    runner = _make_runner(last_submit_target=None)
    RunExecutor._record_failure_for_retry(
        runner=runner, step_index=5, kind="no_state_change", reason="",
    )
    # No history added.
    assert runner._step_failure_history.get(5, []) == []


def test_record_failure_accumulates_across_retries() -> None:
    """Each retry's failure is appended — the form handler then
    surfaces the last N records to find_form_target so the LLM
    sees the full failure trajectory, not just the most recent."""
    runner = _make_runner(
        last_submit_target={"x": 100, "y": 200, "label": "Save", "matched_label": "Save"},
    )
    for i in range(3):
        RunExecutor._record_failure_for_retry(
            runner=runner, step_index=7, kind="no_state_change",
            reason=f"attempt {i + 1}",
        )
    assert len(runner._step_failure_history[7]) == 3


def test_record_failure_is_per_step_index() -> None:
    """Records keyed by step_index — failures on step 5 don't leak
    into step 7's history."""
    runner = _make_runner(
        last_submit_target={"x": 1, "y": 1, "label": "A", "matched_label": "A"},
    )
    RunExecutor._record_failure_for_retry(
        runner=runner, step_index=5, kind="no_state_change", reason="",
    )
    runner._last_submit_target = {"x": 9, "y": 9, "label": "B", "matched_label": "B"}
    RunExecutor._record_failure_for_retry(
        runner=runner, step_index=7, kind="no_state_change", reason="",
    )
    assert len(runner._step_failure_history[5]) == 1
    assert len(runner._step_failure_history[7]) == 1
    assert runner._step_failure_history[5][0]["label"] == "A"
    assert runner._step_failure_history[7][0]["label"] == "B"


# ── _maybe_demote_form_no_change records on demote ──────────────────────


def test_demote_records_failure_when_demoting() -> None:
    """End-to-end: a submit gets demoted, the executor records the
    failure into ``_step_failure_history`` with the click target."""
    runner = _make_runner(
        last_submit_target={"x": 540, "y": 220, "label": "Qualified", "matched_label": "Qualified"},
    )
    executor = RunExecutor(parent=runner)
    state = _state_with_submit_success(step_index=5)
    pre = StepStateSnapshot(url="https://crm.test/")
    submit_step = MicroIntent(intent="Click Qualified lead", type="submit")

    executor._maybe_demote_form_no_change(state, submit_step, pre)

    # Demoted (no extractor, no visual rescue).
    assert state.results[-1].success is False
    # Failure recorded for the next retry to read.
    history = runner._step_failure_history[5]
    assert len(history) == 1
    assert history[0]["x"] == 540
    assert history[0]["label"] == "Qualified"
    # Stash cleared after the check.
    assert runner._last_submit_target is None


def test_demote_does_not_record_when_visually_changed() -> None:
    """When the visual verifier rescues a same-URL submit, the
    success is kept and NO failure record is added — the click was
    correct, the snapshot just couldn't see the delta."""
    extractor = MagicMock()
    extractor.verify_post_click_navigation.return_value = {
        "navigated": True, "kind": "modal", "reason": "dashboard appeared",
    }
    runner = _make_runner(
        last_submit_target={"x": 540, "y": 220, "label": "Login", "matched_label": "Login"},
    )
    runner.extractor = extractor
    runner._last_submit_pre_screenshot = "PIL_PRE"
    runner._safe_screenshot.return_value = "PIL_POST"

    executor = RunExecutor(parent=runner)
    state = _state_with_submit_success(step_index=3)
    pre = StepStateSnapshot(url="https://crm.test/")
    submit_step = MicroIntent(intent="Click Login", type="submit")

    executor._maybe_demote_form_no_change(state, submit_step, pre)

    # Success kept — no failure recorded.
    assert state.results[-1].success is True
    assert runner._step_failure_history.get(3, []) == []


# ── Form handler reads failure_history before searching ─────────────────


def test_form_handler_appends_failure_history_to_search_intent() -> None:
    """When the form handler runs a submit step that has prior failures,
    it must augment the find_form_target prompt with an "AVOID these
    previous targets" block. The LLM then has explicit feedback to
    pick a different element."""
    from mantis_agent.actions import Action, ActionType  # noqa: F401
    from mantis_agent.gym.step_handlers.form import (
        ClaudeGuidedFormHandler, StepContext,
    )

    env = MagicMock()
    env.screenshot.return_value = Image.new("RGB", (10, 10))

    extractor = MagicMock()
    # Found on first try — we want to inspect the prompt that was passed.
    extractor.find_form_target.return_value = {
        "x": 100, "y": 100, "action": "click", "value": "", "label": "Different Row",
    }

    runner = _make_runner()
    runner._step_failure_history = {
        5: [
            {
                "x": 540, "y": 220, "label": "Qualified",
                "matched_label": "Qualified",
                "kind": "no_state_change",
                "reason": "visual verifier saw no UI change",
            },
        ],
    }
    runner.extractor = extractor
    runner.env = env
    runner._best_effort_current_url = MagicMock(return_value="https://crm.test/")
    runner._adaptive_submit_settle = MagicMock(return_value=0.0)
    runner._safe_screenshot = MagicMock(return_value=None)
    runner._dump_debug_screenshot = MagicMock()
    runner._last_known_url = "https://crm.test/"
    runner.costs = {"claude_extract": 0, "gpu_steps": 0, "gpu_seconds": 0.0}

    step = MicroIntent(
        intent="Click first Qualified lead",
        type="submit",
        params={"label": "Qualified"},
    )

    # Override the rendered-screenshot wait so the test stays fast.
    import mantis_agent.gym.step_handlers.form as form_mod
    orig = form_mod._wait_for_rendered_screenshot
    form_mod._wait_for_rendered_screenshot = lambda *_a, **_kw: env.screenshot.return_value
    try:
        ctx = StepContext(env=env, brain=MagicMock(), extractor=extractor)
        ctx.state["index"] = 5  # match the failure-history key
        ClaudeGuidedFormHandler(runner).execute(step, ctx)
    finally:
        form_mod._wait_for_rendered_screenshot = orig

    # The first find_form_target call's intent argument must include
    # the avoid-previous-target block.
    first_call = extractor.find_form_target.call_args_list[0]
    intent_arg = first_call.args[1] if len(first_call.args) >= 2 else first_call.kwargs.get("intent")
    assert intent_arg is not None
    assert "previous attempts" in intent_arg.lower()
    assert "(540, 220)" in intent_arg
    assert "Qualified" in intent_arg
    assert "DIFFERENT" in intent_arg


def test_form_handler_search_intent_unchanged_when_no_history() -> None:
    """First attempt at a step (no prior failures) must keep the
    search_intent unchanged — no "previous attempts" block. This is
    the regression test for the no-history path."""
    from mantis_agent.gym.step_handlers.form import (
        ClaudeGuidedFormHandler, StepContext, _build_submit_search_intent,
    )

    env = MagicMock()
    env.screenshot.return_value = Image.new("RGB", (10, 10))
    extractor = MagicMock()
    extractor.find_form_target.return_value = {
        "x": 100, "y": 100, "action": "click", "value": "", "label": "Login",
    }

    runner = _make_runner()
    runner._step_failure_history = {}  # no history
    runner.extractor = extractor
    runner.env = env
    runner._best_effort_current_url = MagicMock(return_value="https://crm.test/")
    runner._adaptive_submit_settle = MagicMock(return_value=0.0)
    runner._safe_screenshot = MagicMock(return_value=None)
    runner._dump_debug_screenshot = MagicMock()
    runner._last_known_url = "https://crm.test/"
    runner.costs = {"claude_extract": 0, "gpu_steps": 0, "gpu_seconds": 0.0}

    step = MicroIntent(
        intent="Click Login button",
        type="submit",
        params={"label": "Login"},
    )

    import mantis_agent.gym.step_handlers.form as form_mod
    orig = form_mod._wait_for_rendered_screenshot
    form_mod._wait_for_rendered_screenshot = lambda *_a, **_kw: env.screenshot.return_value
    try:
        ctx = StepContext(env=env, brain=MagicMock(), extractor=extractor)
        ctx.state["index"] = 5  # match the failure-history key
        ClaudeGuidedFormHandler(runner).execute(step, ctx)
    finally:
        form_mod._wait_for_rendered_screenshot = orig

    intent_arg = extractor.find_form_target.call_args_list[0].args[1]
    assert "previous attempts" not in intent_arg.lower()
    # The plain build still produced the canonical submit prompt.
    expected = _build_submit_search_intent("Login", "button", step.intent)
    assert intent_arg == expected


# ── Success path clears failure history ─────────────────────────────────


def test_success_clears_failure_history_for_step() -> None:
    """When a step finally succeeds (e.g. retry 3 lands on the right
    target), the failure history for that step_index must clear so
    a future occurrence of the same step_index (loop iteration,
    resumed plan) starts with a clean slate."""
    runner = _make_runner()
    runner._step_failure_history = {
        5: [
            {"x": 1, "y": 1, "label": "Old", "matched_label": "Old",
             "kind": "no_state_change", "reason": ""},
        ],
    }
    runner.scanner = MagicMock()  # noqa: SLF001 — required by _handle_success

    executor = RunExecutor(parent=runner)
    state = _state_with_submit_success(step_index=5)
    success_step = MicroIntent(intent="Click X", type="submit")
    success_result = state.results[-1]

    # _handle_success path; it doesn't run the full success machinery
    # for non-paginate / non-loop step types but DOES run the cleanup
    # block we added.
    try:
        executor._handle_success(MagicMock(), state, success_step, success_result)
    except Exception:
        # _handle_success has more side-effects than we model here —
        # the failure-history clear runs early so any later exception
        # doesn't matter for this assertion.
        pass

    assert runner._step_failure_history.get(5, []) == []


# ── End-to-end: 3 retries get progressively richer feedback ─────────────


def test_three_retries_accumulate_full_failure_trajectory() -> None:
    """After 3 demotions on the same step, the history carries 3
    records; the form handler can prepend all of them so the LLM
    sees the full failure trajectory ('don't pick (a,b), (c,d), or
    (e,f) — they all do nothing')."""
    runner = _make_runner()
    runner._step_failure_history = {}

    targets = [
        {"x": 540, "y": 220, "label": "Qualified", "matched_label": "Qualified"},
        {"x": 540, "y": 250, "label": "Qualified", "matched_label": "Qualified"},
        {"x": 540, "y": 280, "label": "Qualified", "matched_label": "Qualified"},
    ]

    for i, target in enumerate(targets):
        runner._last_submit_target = target
        RunExecutor._record_failure_for_retry(
            runner=runner, step_index=5, kind="no_state_change",
            reason=f"retry {i}",
        )

    history = runner._step_failure_history[5]
    assert len(history) == 3
    ys = [r["y"] for r in history]
    assert ys == [220, 250, 280]
