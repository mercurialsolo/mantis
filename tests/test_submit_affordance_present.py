"""Plan-declared success affordance for in-place submits.

Mirror of the type-verification false-positive, in the other direction:
a LinkedIn connection request succeeds with NO navigation and nothing
*appearing* — the invitation modal CLOSES and the button flips to
"Pending". ``verify_post_click_navigation`` only recognises content
appearing, so such submits were falsely demoted to ``submit_failed``
(run report: "/predict logged submit_failed while the invite actually
sent").

These tests pin the new ``hints.expect_text_present`` escape hatch:
``RunExecutor._submit_affordance_visible`` keeps a no-delta submit's
success when a plan-declared confirmation phrase is visible (via the
cheap Haiku ``verify_gate``), and ``_normalize_expect_text`` accepts the
str / list / empty hint shapes.
"""

from __future__ import annotations

from mantis_agent.gym.run_executor import RunExecutor


def _exec() -> RunExecutor:
    return RunExecutor.__new__(RunExecutor)


class _FakeExtractor:
    def __init__(self, passed: bool, *, raises: bool = False):
        self._passed = passed
        self._raises = raises
        self.calls: list[str] = []

    def verify_gate(self, screenshot, condition):  # noqa: ANN001
        self.calls.append(condition)
        if self._raises:
            raise RuntimeError("api down")
        return self._passed, "reason"


class _FakeRunner:
    def __init__(self, extractor, *, screenshot=object()):
        self.extractor = extractor
        self._screenshot = screenshot
        self.costs = {"claude_extract": 0}

    def _safe_screenshot(self):
        return self._screenshot


# ── _normalize_expect_text ──────────────────────────────────────────────


def test_normalize_accepts_list():
    assert RunExecutor._normalize_expect_text(
        {"expect_text_present": ["Pending", "Invitation sent"]}
    ) == ["Pending", "Invitation sent"]


def test_normalize_accepts_str():
    assert RunExecutor._normalize_expect_text(
        {"expect_text_present": "Pending"}
    ) == ["Pending"]


def test_normalize_strips_and_drops_blanks():
    assert RunExecutor._normalize_expect_text(
        {"expect_text_present": ["  Pending ", "", "   "]}
    ) == ["Pending"]


def test_normalize_absent_hint_is_empty():
    assert RunExecutor._normalize_expect_text({}) == []
    assert RunExecutor._normalize_expect_text({"other": 1}) == []


# ── _submit_affordance_visible ──────────────────────────────────────────


def test_affordance_visible_keeps_success():
    ex = _FakeExtractor(passed=True)
    runner = _FakeRunner(ex)
    assert _exec()._submit_affordance_visible(runner, ["Pending"]) is True
    # The condition handed to verify_gate names the phrase.
    assert "Pending" in ex.calls[0]
    assert runner.costs["claude_extract"] == 1  # counted only when it matched


def test_affordance_absent_allows_demotion():
    ex = _FakeExtractor(passed=False)
    runner = _FakeRunner(ex)
    assert _exec()._submit_affordance_visible(runner, ["Pending"]) is False
    assert runner.costs["claude_extract"] == 0


def test_no_phrases_is_false():
    runner = _FakeRunner(_FakeExtractor(passed=True))
    assert _exec()._submit_affordance_visible(runner, []) is False


def test_missing_extractor_is_false():
    runner = _FakeRunner(None)
    assert _exec()._submit_affordance_visible(runner, ["Pending"]) is False


def test_screenshot_none_is_false():
    runner = _FakeRunner(_FakeExtractor(passed=True), screenshot=None)
    assert _exec()._submit_affordance_visible(runner, ["Pending"]) is False


def test_verify_gate_exception_is_false():
    """API error must never mask a real failure — fall through to demote."""
    runner = _FakeRunner(_FakeExtractor(passed=True, raises=True))
    assert _exec()._submit_affordance_visible(runner, ["Pending"]) is False
