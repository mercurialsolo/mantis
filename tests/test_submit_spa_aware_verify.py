"""Tests for the SPA-aware submit verifier.

Surfaced by the staff-crm rerun on Modal (this work, run completed
2026-05-09 ~03:55 PDT): step 3 (login click) succeeded — credentials
filled, button clicked, page UI changed from a login form to a
dashboard — but the URL stayed at the CRM root. The
runner-state snapshot couldn't see the UI delta (URL same, page
same, scroll same), so ``_maybe_demote_form_no_change`` falsely
demoted the success to failure.

Fix: before demoting a same-URL submit, run the same SPA-aware
visual diff PR #222 added for clicks. The submit handler stashes
the pre-click screenshot at ``runner._last_submit_pre_screenshot``;
the executor takes a post-screenshot, asks
``ClaudeExtractor.verify_post_click_navigation`` whether the page
UI actually changed, and if so keeps the success.

These tests pin the new ``_submit_visually_changed`` helper +
its integration with ``_maybe_demote_form_no_change`` against
five scenarios: visual change confirms keep-success, visual no-
change still demotes, missing pre-screenshot falls through to
existing demotion (no regression of #222 behavior on click-only
runs), missing extractor falls through, and the verifier is only
consulted on submit (not on other step types).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.gym.run_executor import RunExecutor, RunState
from mantis_agent.gym.step_snapshot import StepStateSnapshot
from mantis_agent.plan_decomposer import MicroIntent


def _make_runner(
    *,
    extractor: MagicMock | None,
    pre_screenshot: object | None = "PIL_PRE",
    post_screenshot: object | None = "PIL_POST",
) -> MagicMock:
    """Build a stub runner satisfying the demotion-check contract.

    Captures the minimal surface PlanExecutor reads from runner:
    ``extractor`` (None disables the visual fallback),
    ``_last_submit_pre_screenshot`` (the stash from the form
    handler), ``_safe_screenshot()`` (post screenshot capture),
    ``costs`` (claude_extract counter the verifier increments),
    ``_last_known_url`` / ``_current_page`` / ``_viewport_stage``
    / ``_scroll_state`` / ``_extracted_titles`` / ``_seen_urls``
    / ``_last_extracted`` (the inputs to step_snapshot.capture).
    """
    runner = MagicMock()
    runner.extractor = extractor
    runner._last_submit_pre_screenshot = pre_screenshot
    runner._safe_screenshot.return_value = post_screenshot
    runner.costs = {"claude_extract": 0}
    # step_snapshot.capture reads these — return values that produce
    # a stable, no-change diff so the demotion path triggers.
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


def _executor(runner: MagicMock) -> RunExecutor:
    return RunExecutor(parent=runner)


def _state_with_submit_success() -> RunState:
    """A RunState whose last result is a successful submit."""
    state = RunState.fresh(
        run_key="test_run", session_name="t", plan_signature="sig",
    )
    state.step_index = 3
    state.results.append(
        StepResult(
            step_index=3,
            intent="Click the Login button",
            success=True,
            data="submit:Login",
            duration=3.0,
            steps_used=1,
        )
    )
    return state


def _submit_step() -> MicroIntent:
    return MicroIntent(intent="Click the Login button", type="submit")


# ── _submit_visually_changed in isolation ───────────────────────────────


def test_visually_changed_returns_true_on_navigated_modal() -> None:
    extractor = MagicMock()
    extractor.verify_post_click_navigation.return_value = {
        "navigated": True, "kind": "modal", "reason": "dashboard panel appeared",
    }
    runner = _make_runner(extractor=extractor)
    executor = _executor(runner)

    result = executor._submit_visually_changed(runner, _submit_step())

    assert result is True
    extractor.verify_post_click_navigation.assert_called_once_with(
        "PIL_PRE", "PIL_POST", "Click the Login button",
    )
    # Verifier costs a Claude call.
    assert runner.costs["claude_extract"] == 1


def test_visually_changed_returns_true_on_url_change() -> None:
    """``url_change`` kind also counts — covers the case where the
    snapshot diff missed the URL transition (rare, but possible
    when ``_best_effort_current_url`` returned stale data)."""
    extractor = MagicMock()
    extractor.verify_post_click_navigation.return_value = {
        "navigated": True, "kind": "url_change", "reason": "now on /dashboard",
    }
    runner = _make_runner(extractor=extractor)
    executor = _executor(runner)
    assert executor._submit_visually_changed(runner, _submit_step()) is True


def test_visually_changed_returns_false_on_no_change() -> None:
    """``no_change`` kind = page genuinely didn't change → demote."""
    extractor = MagicMock()
    extractor.verify_post_click_navigation.return_value = {
        "navigated": False, "kind": "no_change", "reason": "still login form",
    }
    runner = _make_runner(extractor=extractor)
    executor = _executor(runner)
    assert executor._submit_visually_changed(runner, _submit_step()) is False


def test_visually_changed_returns_false_on_wrong_target() -> None:
    """``wrong_target`` = something changed but it wasn't the intended
    flow (e.g. a tooltip popped, not a form submission). Don't keep
    success — demote so the caller retries."""
    extractor = MagicMock()
    extractor.verify_post_click_navigation.return_value = {
        "navigated": True, "kind": "wrong_target",
        "reason": "tooltip appeared on hover",
    }
    runner = _make_runner(extractor=extractor)
    executor = _executor(runner)
    assert executor._submit_visually_changed(runner, _submit_step()) is False


def test_visually_changed_returns_false_when_no_pre_screenshot() -> None:
    """If the form handler didn't stash a pre-screenshot (e.g. the
    handler errored before reaching the click), fall through to the
    legacy demotion path. No regression for runs without the new
    stash."""
    runner = _make_runner(extractor=MagicMock(), pre_screenshot=None)
    executor = _executor(runner)
    assert executor._submit_visually_changed(runner, _submit_step()) is False


def test_visually_changed_returns_false_when_no_extractor() -> None:
    """Running without ClaudeExtractor (Holo3-only smoke modes) — fall
    through to legacy demotion."""
    runner = _make_runner(extractor=None)
    executor = _executor(runner)
    assert executor._submit_visually_changed(runner, _submit_step()) is False


def test_visually_changed_returns_false_when_extractor_lacks_method() -> None:
    """Some extractor stubs (older test fixtures) don't expose
    ``verify_post_click_navigation``. Use getattr-with-default and
    fall through gracefully."""
    extractor = MagicMock(spec=[])  # no verify_post_click_navigation attr
    runner = _make_runner(extractor=extractor)
    executor = _executor(runner)
    assert executor._submit_visually_changed(runner, _submit_step()) is False


def test_visually_changed_returns_false_on_post_screenshot_failure() -> None:
    """``_safe_screenshot`` raised — fall through, don't propagate."""
    extractor = MagicMock()
    runner = _make_runner(extractor=extractor)
    runner._safe_screenshot.side_effect = RuntimeError("display lost")
    executor = _executor(runner)
    assert executor._submit_visually_changed(runner, _submit_step()) is False


def test_visually_changed_returns_false_on_verifier_exception() -> None:
    """Verifier raised (network / API timeout) — fall through."""
    extractor = MagicMock()
    extractor.verify_post_click_navigation.side_effect = ConnectionError("API down")
    runner = _make_runner(extractor=extractor)
    executor = _executor(runner)
    assert executor._submit_visually_changed(runner, _submit_step()) is False


def test_visually_changed_returns_false_on_non_dict_response() -> None:
    """Verifier returned None / string — defensive against shape drift."""
    extractor = MagicMock()
    extractor.verify_post_click_navigation.return_value = None
    runner = _make_runner(extractor=extractor)
    executor = _executor(runner)
    assert executor._submit_visually_changed(runner, _submit_step()) is False


# ── _maybe_demote_form_no_change integration ────────────────────────────


def test_demote_keeps_success_when_visual_verifier_confirms_change(
    monkeypatch,
) -> None:
    """End-to-end: a successful submit with no snapshot delta but
    confirmed UI change must NOT be demoted to failure. This is the
    exact staff-crm login fix."""
    extractor = MagicMock()
    extractor.verify_post_click_navigation.return_value = {
        "navigated": True, "kind": "modal", "reason": "dashboard appeared",
    }
    runner = _make_runner(extractor=extractor)
    executor = _executor(runner)
    state = _state_with_submit_success()

    # pre_snapshot identical to post (capture returns the same shape)
    pre = StepStateSnapshot(url="https://crm.test/")
    executor._maybe_demote_form_no_change(state, _submit_step(), pre)

    result = state.results[-1]
    assert result.success is True
    assert "no_state_change" not in (result.data or "")
    # Pre-screenshot stash must be cleared so a later step doesn't
    # consume a stale screenshot.
    assert runner._last_submit_pre_screenshot is None


def test_demote_still_fires_when_visual_verifier_rejects(monkeypatch) -> None:
    """When the visual verifier confirms the page didn't change either,
    the legacy demotion still triggers."""
    extractor = MagicMock()
    extractor.verify_post_click_navigation.return_value = {
        "navigated": False, "kind": "no_change", "reason": "still on login",
    }
    runner = _make_runner(extractor=extractor)
    executor = _executor(runner)
    state = _state_with_submit_success()

    pre = StepStateSnapshot(url="https://crm.test/")
    executor._maybe_demote_form_no_change(state, _submit_step(), pre)

    result = state.results[-1]
    assert result.success is False
    assert "no_state_change" in (result.data or "")


def test_demote_still_fires_when_no_extractor_available() -> None:
    """No extractor wired (Holo3-only smoke runs) — legacy demotion
    must still trigger without the visual-verify fallback. This is
    the regression test for the existing #121 behavior."""
    runner = _make_runner(extractor=None)
    executor = _executor(runner)
    state = _state_with_submit_success()

    pre = StepStateSnapshot(url="https://crm.test/")
    executor._maybe_demote_form_no_change(state, _submit_step(), pre)

    result = state.results[-1]
    assert result.success is False
    assert "no_state_change" in (result.data or "")


def test_demote_skips_non_submit_step_types(monkeypatch) -> None:
    """The demotion is submit-only — click / fill_field / select_option
    must pass through untouched even when the visual verifier would
    say no-change."""
    extractor = MagicMock()
    extractor.verify_post_click_navigation.return_value = {
        "navigated": False, "kind": "no_change",
    }
    runner = _make_runner(extractor=extractor)
    executor = _executor(runner)
    state = _state_with_submit_success()

    pre = StepStateSnapshot(url="https://crm.test/")
    select_step = MicroIntent(intent="Pick option", type="select_option")
    executor._maybe_demote_form_no_change(state, select_step, pre)

    # Untouched.
    result = state.results[-1]
    assert result.success is True
    assert "no_state_change" not in (result.data or "")
    extractor.verify_post_click_navigation.assert_not_called()


def test_demote_clears_pre_screenshot_after_check(monkeypatch) -> None:
    """The stash must always clear after the check — keeps it from
    bleeding into the next step's verifier call. Tested in both
    code paths: keep-success AND demote-to-failure."""
    extractor = MagicMock()
    extractor.verify_post_click_navigation.return_value = {
        "navigated": True, "kind": "modal",
    }
    runner = _make_runner(extractor=extractor)
    executor = _executor(runner)
    state = _state_with_submit_success()

    pre = StepStateSnapshot(url="https://crm.test/")
    executor._maybe_demote_form_no_change(state, _submit_step(), pre)
    assert runner._last_submit_pre_screenshot is None
