"""Per-step recovery hints helper — epic #377 Phase A.3.

Pins the API contract used by every step handler that splices
hints into its Claude prompt. The producer side (when hints get
ADDED via the agentic recovery loop) is covered by the step-recovery
tests; this module tests the consumer side."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from mantis_agent.gym import recovery_hints


# ── get_hint_block ──────────────────────────────────────────────────────


def test_returns_empty_when_runner_has_no_hints_attr() -> None:
    runner = SimpleNamespace()
    assert recovery_hints.get_hint_block(runner, 0) == ""


def test_returns_empty_when_step_has_no_hints() -> None:
    runner = SimpleNamespace(_recovery_hints={2: ["earlier hint"]})
    assert recovery_hints.get_hint_block(runner, 0) == ""


def test_returns_empty_when_all_stored_hints_are_falsy() -> None:
    """Empty strings / None entries are dropped — the block stays
    out of the prompt unless there's actual content."""
    runner = SimpleNamespace(_recovery_hints={0: ["", None, ""]})
    assert recovery_hints.get_hint_block(runner, 0) == ""


def test_returns_block_with_one_hint() -> None:
    runner = SimpleNamespace(_recovery_hints={0: ["click row 3 not row 1"]})
    block = recovery_hints.get_hint_block(runner, 0)
    assert "RECOVERY HINTS" in block
    assert "click row 3 not row 1" in block
    assert block.startswith("\n\n")  # safe to concat onto a search_intent


def test_returns_block_with_multiple_hints_in_order() -> None:
    runner = SimpleNamespace(_recovery_hints={
        0: ["first hint", "second hint", "third hint"],
    })
    block = recovery_hints.get_hint_block(runner, 0)
    idx1 = block.find("first hint")
    idx2 = block.find("second hint")
    idx3 = block.find("third hint")
    assert 0 < idx1 < idx2 < idx3


def test_defensive_against_magicmock_runner() -> None:
    """MagicMock auto-creates _recovery_hints as a Mock, not a dict.
    The helper must return empty rather than splice ``Mock`` repr
    into a production prompt."""
    runner = MagicMock()
    assert recovery_hints.get_hint_block(runner, 0) == ""


def test_defensive_against_non_list_stored_value() -> None:
    """If something writes a non-list value (string / dict / int),
    don't crash — return empty."""
    runner = SimpleNamespace(_recovery_hints={0: "not a list"})
    assert recovery_hints.get_hint_block(runner, 0) == ""


# ── has_hints / count ────────────────────────────────────────────────────


def test_has_hints_false_for_empty() -> None:
    runner = SimpleNamespace()
    assert recovery_hints.has_hints(runner, 0) is False


def test_has_hints_true_with_real_content() -> None:
    runner = SimpleNamespace(_recovery_hints={0: ["", "real hint"]})
    assert recovery_hints.has_hints(runner, 0) is True


def test_count_returns_non_empty_only() -> None:
    runner = SimpleNamespace(_recovery_hints={0: ["", "a", None, "b"]})
    assert recovery_hints.count(runner, 0) == 2


def test_count_zero_on_missing_attr() -> None:
    assert recovery_hints.count(SimpleNamespace(), 5) == 0


# ── Wire-in: Holo3StepHandler splices hints into the inner task ──────────


def test_holo3_step_handler_splices_hints_into_inner_gym_runner_task() -> None:
    """When ``_recovery_hints[step_index]`` carries entries, the
    Holo3 handler appends them to ``step.intent`` and forwards the
    enriched string as the inner ``GymRunner.run(task=...)`` arg.
    Epic #377 Phase A.3 generalizes the form-only consumption path."""
    from unittest.mock import patch

    from mantis_agent.gym.step_context import StepContext
    from mantis_agent.gym.step_handlers.holo3 import Holo3StepHandler
    from mantis_agent.plan_decomposer import MicroIntent

    class _FakeRunner:
        def __init__(self) -> None:
            self.costs = {"claude_extract": 0}
            self._last_known_url = ""
            self.on_step = None
            self._recovery_hints = {7: ["avoid the photo, click the title"]}

        def _update_scroll_state_from_trajectory(self, *_a, **_kw) -> None:
            pass

    runner = _FakeRunner()
    ctx = StepContext(
        env=MagicMock(),
        brain=MagicMock(),
        extractor=None,
        grounding=None,
        cost_meter=None,
        dynamic_verifier=None,
        scanner=None,
        site_config=None,
        tool_channel=None,
        extraction_cache=None,
        state={"index": 7},
    )
    step = MicroIntent(intent="Click the first event card", type="click", budget=4)

    with patch("mantis_agent.gym.step_handlers.holo3.GymRunner") as mock_cls:
        mock_inner = MagicMock()
        mock_inner.run.return_value = MagicMock(
            success=True, total_steps=2, total_time=1.0,
            termination_reason="done",
        )
        mock_cls.return_value = mock_inner

        Holo3StepHandler(runner).execute(step, ctx)

    forwarded_task = mock_inner.run.call_args.kwargs["task"]
    assert forwarded_task.startswith("Click the first event card")
    assert "RECOVERY HINTS" in forwarded_task
    assert "avoid the photo, click the title" in forwarded_task


def test_holo3_step_handler_no_change_when_no_hints() -> None:
    """No hints stored → ``task`` is exactly ``step.intent`` (no
    trailing block, no whitespace drift)."""
    from unittest.mock import patch

    from mantis_agent.gym.step_context import StepContext
    from mantis_agent.gym.step_handlers.holo3 import Holo3StepHandler
    from mantis_agent.plan_decomposer import MicroIntent

    class _FakeRunner:
        def __init__(self) -> None:
            self.costs = {"claude_extract": 0}
            self._last_known_url = ""
            self.on_step = None
            self._recovery_hints = {}  # empty

        def _update_scroll_state_from_trajectory(self, *_a, **_kw) -> None:
            pass

    runner = _FakeRunner()
    ctx = StepContext(
        env=MagicMock(),
        brain=MagicMock(),
        extractor=None,
        grounding=None,
        cost_meter=None,
        dynamic_verifier=None,
        scanner=None,
        site_config=None,
        tool_channel=None,
        extraction_cache=None,
        state={"index": 3},
    )
    step = MicroIntent(intent="Click submit", type="click", budget=4)

    with patch("mantis_agent.gym.step_handlers.holo3.GymRunner") as mock_cls:
        mock_inner = MagicMock()
        mock_inner.run.return_value = MagicMock(
            success=True, total_steps=1, total_time=0.5,
            termination_reason="done",
        )
        mock_cls.return_value = mock_inner

        Holo3StepHandler(runner).execute(step, ctx)

    assert mock_inner.run.call_args.kwargs["task"] == "Click submit"
