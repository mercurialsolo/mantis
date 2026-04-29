"""Mantis agent — world-class vision, patient observation, precise action.

The orchestrator surface — what hosts (e.g. vision_claude) need to drive
``MicroPlanRunner`` against a remote Mantis service — is re-exported at the
top level. Heavyweight subpackages (gym envs, brains, extraction, grounding)
remain available via deep imports for callers that want them.

Public orchestrator surface:

    from mantis_agent import (
        MicroPlanRunner, RunnerResult, PauseRequested, PauseState,
        StepResult, MicroPlan, MicroIntent, PlanDecomposer,
        scale_brain_to_display,
    )

Top-level imports here only pull modules listed in the ``[orchestrator]``
extra (``requests``, ``pydantic``). They do not import torch / vLLM /
playwright / pyautogui — keep it that way so vision_claude can install
``mantis-agent[orchestrator]`` without dragging GPU deps.
"""

__version__ = "0.1.0"

# Lazy re-exports — defined on first attribute access so ``import mantis_agent``
# stays cheap and circular-import safe.
__all__ = [
    "MicroPlanRunner",
    "RunnerResult",
    "PauseRequested",
    "PauseState",
    "StepResult",
    "MicroPlan",
    "MicroIntent",
    "PlanDecomposer",
    "scale_brain_to_display",
]


def __getattr__(name: str):  # PEP 562
    if name in {
        "MicroPlanRunner",
        "RunnerResult",
        "PauseRequested",
        "PauseState",
        "StepResult",
    }:
        from .gym.micro_runner import (
            MicroPlanRunner,
            PauseRequested,
            PauseState,
            RunnerResult,
            StepResult,
        )
        globals().update(
            MicroPlanRunner=MicroPlanRunner,
            RunnerResult=RunnerResult,
            PauseRequested=PauseRequested,
            PauseState=PauseState,
            StepResult=StepResult,
        )
        return globals()[name]
    if name in {"MicroPlan", "MicroIntent", "PlanDecomposer"}:
        from .plan_decomposer import MicroIntent, MicroPlan, PlanDecomposer
        globals().update(
            MicroPlan=MicroPlan,
            MicroIntent=MicroIntent,
            PlanDecomposer=PlanDecomposer,
        )
        return globals()[name]
    if name == "scale_brain_to_display":
        from .gym.xdotool_env import scale_brain_to_display
        globals()["scale_brain_to_display"] = scale_brain_to_display
        return scale_brain_to_display
    raise AttributeError(f"module 'mantis_agent' has no attribute {name!r}")
