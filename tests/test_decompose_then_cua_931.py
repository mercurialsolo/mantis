"""#931 P3 — decompose-then-cua + director off-switch.

Long multi-step instructions derail the small CUA model (loops, mis-clicks).
``decompose: true`` seeds an ordered sub-goal roadmap from the Claude
decomposer and drives the brain with the augmented task — the planning
bridge between /v1/predict and /v1/cua. The Claude director also gains a
clean off-switch for strictly brain-only runs.
"""

from __future__ import annotations

from mantis_agent.api_schemas import PureCUARequest
from mantis_agent.baseten_server.runtime import _build_decomposed_task


# ── request schema ──────────────────────────────────────────────────────


def test_decompose_defaults_false():
    req = PureCUARequest.model_validate({"instruction": "book a demo"})
    assert req.decompose is False


def test_decompose_accepted_and_forwarded():
    req = PureCUARequest.model_validate({"instruction": "x", "decompose": True})
    assert req.decompose is True
    assert req.model_dump(exclude_none=True)["decompose"] is True


# ── _build_decomposed_task ──────────────────────────────────────────────


def test_build_task_appends_ordered_subgoals():
    task = _build_decomposed_task(
        "Send a connection request to the first result",
        ["Open the search page", "Type the name", "Click the first result", "Click Connect"],
    )
    assert task.startswith("Send a connection request to the first result")
    assert "1. Open the search page" in task
    assert "4. Click Connect" in task
    # ordering preserved
    assert task.index("1. Open") < task.index("2. Type") < task.index("4. Click Connect")


def test_build_task_no_subgoals_returns_raw_instruction():
    assert _build_decomposed_task("just do it", []) == "just do it"


def test_build_task_skips_blank_subgoals():
    task = _build_decomposed_task("x", ["  ", "", "real goal"])
    assert "1. real goal" in task
    assert "2." not in task  # blanks dropped, not numbered


# ── director off-switch (env-gated enable logic) ────────────────────────


def _director_enabled(*, key: str, toggle: str | None, brain_name: str) -> bool:
    """Mirror of the runner's director-enable predicate (runner.py ~826)."""
    t = (toggle or "enabled").strip().lower()
    return bool(key) and t not in ("0", "off", "disabled", "false") and brain_name != "ClaudeBrain"


def test_director_enabled_by_default_with_key():
    assert _director_enabled(key="sk-x", toggle=None, brain_name="Holo3Brain") is True


def test_director_disabled_by_toggle():
    assert _director_enabled(key="sk-x", toggle="disabled", brain_name="Holo3Brain") is False
    assert _director_enabled(key="sk-x", toggle="off", brain_name="Holo3Brain") is False


def test_director_disabled_without_key():
    assert _director_enabled(key="", toggle="enabled", brain_name="Holo3Brain") is False


def test_director_disabled_for_claude_brain():
    assert _director_enabled(key="sk-x", toggle="enabled", brain_name="ClaudeBrain") is False
