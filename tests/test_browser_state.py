"""Tests for #115 step 4 — BrowserState extracted from MicroPlanRunner."""

from __future__ import annotations

from typing import Any

from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.browser_state import BrowserState


# ── Fake parent (mirrors MicroPlanRunner attribute surface) ─────────────


class _FakeSiteConfig:
    """Minimal SiteConfig stand-in for URL composition tests."""

    pagination_strip_pattern = r"/page-\d+/?$"

    def paginated_url(self, base: str, page: int) -> str:
        return f"{base.rstrip('/')}/page-{page}/"


class _FakeEnv:
    """Records env.reset / env.step calls so resume tests can assert order."""

    def __init__(self) -> None:
        self.reset_calls: list[dict] = []
        self.step_calls: list[Action] = []
        self.fail_reset = False

    def reset(self, **kwargs: Any) -> None:
        if self.fail_reset:
            raise RuntimeError("env.reset boom")
        self.reset_calls.append(kwargs)

    def step(self, action: Action) -> None:
        self.step_calls.append(action)


class _FakeRunner:
    """Just the attributes BrowserState reads/writes."""

    def __init__(self) -> None:
        self.site_config = _FakeSiteConfig()
        self.env = _FakeEnv()
        self._scroll_state: dict[str, Any] = {}
        self._viewport_stage: int = 0
        self._results_base_url: str = ""
        self._current_page: int = 1
        self._last_known_url: str = ""
        self.checkpoint_calls: list[str] = []

    def _checkpoint_active_progress(self, reason: str) -> None:
        self.checkpoint_calls.append(reason)


def _bs() -> tuple[BrowserState, _FakeRunner]:
    runner = _FakeRunner()
    return BrowserState(runner), runner


# ── current_results_page_url ────────────────────────────────────────────


def test_current_results_page_url_empty_when_no_base() -> None:
    bs, _ = _bs()
    assert bs.current_results_page_url() == ""


def test_current_results_page_url_strips_pagination_on_page_one() -> None:
    bs, r = _bs()
    r._results_base_url = "https://x.test/listings/page-3/"
    r._current_page = 1
    assert bs.current_results_page_url() == "https://x.test/listings/"


def test_current_results_page_url_uses_paginated_url_for_higher_pages() -> None:
    bs, r = _bs()
    r._results_base_url = "https://x.test/listings/"
    r._current_page = 4
    assert bs.current_results_page_url() == "https://x.test/listings/page-4/"


# ── reentry_url_for_step ────────────────────────────────────────────────


class _FakeStep:
    def __init__(self, type_: str) -> None:
        self.type = type_


class _FakePlan:
    def __init__(self, *types: str) -> None:
        self.steps = [_FakeStep(t) for t in types]


def test_reentry_url_results_for_click_step() -> None:
    bs, r = _bs()
    r._results_base_url = "https://x.test/r/"
    r._current_page = 2
    r._last_known_url = "https://x.test/detail/123"
    plan = _FakePlan("filter", "click", "scroll")
    # Click step → results URL.
    assert bs.reentry_url_for_step(plan, 1) == "https://x.test/r/page-2/"


def test_reentry_url_last_known_for_extract_step() -> None:
    bs, r = _bs()
    r._results_base_url = "https://x.test/r/"
    r._last_known_url = "https://x.test/detail/123"
    plan = _FakePlan("click", "extract_data")
    assert bs.reentry_url_for_step(plan, 1) == "https://x.test/detail/123"


def test_reentry_url_falls_back_when_no_step_at_index() -> None:
    bs, r = _bs()
    r._last_known_url = "https://x.test/last"
    plan = _FakePlan("click")
    # next_step_index out of range → fallback chain
    assert bs.reentry_url_for_step(plan, 99) == "https://x.test/last"


# ── set_scroll_state ────────────────────────────────────────────────────


def test_set_scroll_state_writes_full_state_dict() -> None:
    bs, r = _bs()
    r._last_known_url = "https://x.test/p1"
    bs.set_scroll_state(context="results", page_downs=2, wheel_downs=5, label="vp1")
    s = r._scroll_state
    assert s["context"] == "results"
    assert s["url"] == "https://x.test/p1"
    assert s["page_downs"] == 2
    assert s["wheel_downs"] == 5
    assert s["label"] == "vp1"
    assert "updated_at" in s


def test_set_scroll_state_clamps_negatives_to_zero() -> None:
    bs, r = _bs()
    bs.set_scroll_state(context="x", page_downs=-3, wheel_downs=-99, viewport_stage=-1)
    assert r._scroll_state["page_downs"] == 0
    assert r._scroll_state["wheel_downs"] == 0
    assert r._scroll_state["viewport_stage"] == 0


def test_set_scroll_state_with_flush_calls_checkpoint() -> None:
    bs, r = _bs()
    bs.set_scroll_state(context="results", flush=True)
    assert r.checkpoint_calls == ["scroll_state:results"]


# ── update_scroll_state_from_trajectory ─────────────────────────────────


class _FakeStepRecord:
    def __init__(self, action: Action) -> None:
        self.action = action


class _FakeResult:
    def __init__(self, actions: list[Action]) -> None:
        self.trajectory = [_FakeStepRecord(a) for a in actions]


def test_update_scroll_state_counts_page_down_and_home_reset() -> None:
    bs, r = _bs()
    actions = [
        Action(ActionType.KEY_PRESS, {"keys": "Page_Down"}),
        Action(ActionType.KEY_PRESS, {"keys": "Page_Down"}),
        Action(ActionType.KEY_PRESS, {"keys": "Home"}),  # resets to 0
        Action(ActionType.KEY_PRESS, {"keys": "Page_Down"}),
    ]
    bs.update_scroll_state_from_trajectory(_FakeResult(actions), context="x")
    assert r._scroll_state["page_downs"] == 1


def test_update_scroll_state_handles_scroll_down_amount() -> None:
    bs, r = _bs()
    actions = [
        Action(ActionType.SCROLL, {"direction": "down", "amount": 5}),
        Action(ActionType.SCROLL, {"direction": "down", "amount": 3}),
        Action(ActionType.SCROLL, {"direction": "up", "amount": 2}),
    ]
    bs.update_scroll_state_from_trajectory(_FakeResult(actions), context="x")
    assert r._scroll_state["wheel_downs"] == 6


def test_update_scroll_state_marks_end_reached_on_end_key() -> None:
    bs, r = _bs()
    actions = [Action(ActionType.KEY_PRESS, {"keys": "End"})]
    bs.update_scroll_state_from_trajectory(_FakeResult(actions), context="x")
    assert r._scroll_state.get("end_reached") is True


# ── restore_scroll_position ─────────────────────────────────────────────


def test_restore_scroll_position_no_state_is_noop() -> None:
    bs, r = _bs()
    bs.restore_scroll_position()
    assert r.env.step_calls == []


def test_restore_scroll_position_skips_when_url_differs() -> None:
    bs, r = _bs()
    r._scroll_state = {"url": "https://x.test/a/", "page_downs": 3}
    r._last_known_url = "https://x.test/b/"
    bs.restore_scroll_position()
    # Different URL — skip everything.
    assert r.env.step_calls == []


def test_restore_scroll_position_replays_page_downs() -> None:
    bs, r = _bs()
    r._scroll_state = {
        "url": "https://x.test/a/",
        "page_downs": 3,
        "wheel_downs": 0,
    }
    r._last_known_url = "https://x.test/a/"
    bs.restore_scroll_position()
    # Home + 3 Page_Down presses
    keys = [a.params.get("keys") for a in r.env.step_calls]
    assert keys == ["Home", "Page_Down", "Page_Down", "Page_Down"]


def test_restore_scroll_position_caps_page_downs_at_12() -> None:
    bs, r = _bs()
    r._scroll_state = {"url": "https://x.test/a/", "page_downs": 50}
    r._last_known_url = "https://x.test/a/"
    bs.restore_scroll_position()
    keys = [a.params.get("keys") for a in r.env.step_calls]
    assert keys.count("Page_Down") == 12


# ── resume_browser_state ────────────────────────────────────────────────


def test_resume_browser_state_empty_url_returns_false() -> None:
    bs, _ = _bs()
    assert bs.resume_browser_state("") is False


def test_resume_browser_state_returns_false_on_env_error(monkeypatch) -> None:
    bs, r = _bs()
    r.env.fail_reset = True
    # Speed up: avoid the 12s sleep in the success path; we don't reach it
    # because env.reset raises first.
    assert bs.resume_browser_state("https://x.test/x") is False


def test_resume_browser_state_updates_last_known_url(monkeypatch) -> None:
    bs, r = _bs()
    # Stub time.sleep to avoid the 12s wait in the resume path.
    import mantis_agent.gym.browser_state as bs_mod
    monkeypatch.setattr(bs_mod.time, "sleep", lambda _s: None)
    ok = bs.resume_browser_state("https://x.test/restored")
    assert ok is True
    assert r._last_known_url == "https://x.test/restored"
    assert r.env.reset_calls == [{"task": "resume", "start_url": "https://x.test/restored"}]


# ── Backward-compat: MicroPlanRunner shims still work ──────────────────


def test_runner_shims_delegate_to_browser_state() -> None:
    from mantis_agent.gym.micro_runner import MicroPlanRunner

    runner = MicroPlanRunner.__new__(MicroPlanRunner)
    runner.site_config = _FakeSiteConfig()
    runner.env = _FakeEnv()
    runner._scroll_state = {}
    runner._viewport_stage = 0
    runner._results_base_url = "https://x.test/r/"
    runner._current_page = 3
    runner._last_known_url = ""
    runner.browser_state = BrowserState(runner)

    assert runner._current_results_page_url() == "https://x.test/r/page-3/"
