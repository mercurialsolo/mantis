"""Lock the public orchestrator surface that vision_claude (and other hosts)
import from ``mantis_agent``.

These tests fail loudly if a refactor breaks the integration spec contract.
"""

from __future__ import annotations

import sys


def test_top_level_orchestrator_imports():
    """Names listed in `from mantis_agent import (...)` must resolve."""
    import mantis_agent

    for name in (
        "MicroPlanRunner",
        "RunnerResult",
        "PauseRequested",
        "PauseState",
        "StepResult",
        "MicroPlan",
        "MicroIntent",
        "PlanDecomposer",
        "scale_brain_to_display",
    ):
        assert hasattr(mantis_agent, name), f"mantis_agent.{name} missing"


def test_microplan_round_trip_through_dict():
    """Hosts hand-author micro_plans and pass dict-shaped payloads in."""
    from mantis_agent import MicroIntent, MicroPlan

    plan = MicroPlan(
        steps=[
            MicroIntent(
                intent="Navigate to https://example.com",
                type="navigate",
                budget=3,
                section="setup",
                required=True,
            ),
        ],
        domain="example.com",
    )
    encoded = plan.to_dict()
    restored = MicroPlan.from_dict(encoded)
    assert restored.domain == "example.com"
    assert len(restored.steps) == 1
    assert restored.steps[0].intent.startswith("Navigate to")
    assert restored.steps[0].type == "navigate"


def test_microplan_from_dict_accepts_bare_step_list():
    """Convenience: pass the steps array directly without the wrapper."""
    from mantis_agent import MicroPlan

    plan = MicroPlan.from_dict([
        {"intent": "Click", "type": "click", "budget": 8},
        {"intent": "Loop", "type": "loop", "loop_target": 0, "loop_count": 3},
    ])
    assert len(plan.steps) == 2
    assert plan.steps[1].type == "loop"
    assert plan.steps[1].loop_count == 3


def test_holo3_brain_accepts_extra_headers_and_overrides_authorization():
    """vision_claude ships X-Mantis-Token and (sometimes) Api-Key auth."""
    from mantis_agent.brain_holo3 import Holo3Brain

    brain = Holo3Brain(
        base_url="https://mantis.example/v1",
        api_key="ignored-bearer",
        extra_headers={
            "X-Mantis-Token": "token123",
            "Authorization": "Api-Key gateway-secret",
        },
    )
    headers = brain._headers
    assert headers["X-Mantis-Token"] == "token123"
    # extra_headers must win over the default Bearer token.
    assert headers["Authorization"] == "Api-Key gateway-secret"


def test_holo3_brain_default_headers_unchanged_without_extra():
    """Backwards-compat: existing callers without extra_headers see no change."""
    from mantis_agent.brain_holo3 import Holo3Brain

    brain = Holo3Brain(api_key="abc")
    headers = brain._headers
    assert headers["Authorization"] == "Bearer abc"
    assert "X-Mantis-Token" not in headers


def test_orchestrator_surface_imports_no_heavy_deps():
    """Importing `mantis_agent` must not pull torch / vllm / pyautogui.

    vision_claude installs ``mantis-agent[orchestrator]`` which lists only
    requests + pydantic. If a refactor sneaks a torch import into the
    transitive import chain, this test fails loudly so we catch it before
    a vision_claude pip install starts dragging GPU deps.
    """
    # Force a fresh import — drop anything we may have pulled earlier.
    for mod in list(sys.modules):
        if mod == "mantis_agent" or mod.startswith("mantis_agent."):
            del sys.modules[mod]

    import mantis_agent  # noqa: F401 — side-effect: populate sys.modules

    # Trigger lazy attribute loads — these populate the orchestrator surface.
    _ = mantis_agent.MicroPlanRunner
    _ = mantis_agent.MicroPlan
    _ = mantis_agent.scale_brain_to_display

    forbidden = {"torch", "vllm", "pyautogui", "transformers", "playwright"}
    leaked = forbidden & set(sys.modules)
    assert not leaked, (
        f"mantis_agent transitively imported heavy deps: {leaked}. "
        "Move the offending import behind a function-local import or extras gate."
    )
